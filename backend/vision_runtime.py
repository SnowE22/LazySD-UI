"""Runtime adapter for local Vision-Language Models (VLM) and the official
FLUX 3 Image Klein Prompt Upsampler (``unsloth/flux-3-image-klein-pup-GGUF`` &
``tonera/flux-3-image-klein-pup-nunchaku``).

Handles:
- Checking Ollama availability and version (``/api/version``, ``/api/tags``).
- Running ``tonera/flux-3-image-klein-pup-nunchaku`` natively via ``envs/extras``
  (``NunchakuQwen3_5ForConditionalGeneration``) in an isolated subprocess that
  releases 100% of VRAM upon completion.
- Running ``unsloth/flux-3-image-klein-pup-GGUF`` and general VLMs via Ollama
  ``/api/chat`` with ``keep_alive="0s"``.
- Downloading models either from Hugging Face (for Nunchaku SVDQ & GGUF) or via
  Ollama ``/api/pull`` with real-time progress tracking.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt_fallback import DEFAULT_NEGATIVE_PROMPT
from .vision_models import (
    clear_download_state,
    get_catalog_entry,
    get_download_state,
    is_local_vision_model_installed,
    is_tag_matching,
    set_download_state,
)

logger = logging.getLogger("quantfunc.vision_runtime")

SYSTEM_PROMPT = (
    "You are a prompt engineer for Stable Diffusion / FLUX Inpainting.\n"
    "In both images, the BRIGHT RED highlighted shapes mark the EXACT masked pixels to be repainted.\n"
    "CRITICAL RULES:\n"
    "1. Describe ONLY what should appear INSIDE the bright red masked shapes AFTER the edit.\n"
    "2. NEVER describe characters, faces, hair, hats, or clothing that are OUTSIDE the red mask, "
    "even if a character stands next to or between red masked shapes!\n"
    "3. If the user asks to delete/remove/clean text, SFX, sound effects, speech bubbles, or objects: "
    "do NOT mention the removed text/object and do NOT describe nearby characters. Describe ONLY the "
    "underlying background surface, colors, lighting, gradients, and brush texture that should seamlessly "
    "fill the red shapes behind the removed element.\n"
    "4. negative_prompt MUST be a comma-separated list of forbidden artifacts (e.g. "
    '"japanese text, sfx, kanji, katakana, letters, speech bubble, watermark, blurry, seam"). '
    'NEVER write "no background", "no colors", "no details", "no style", "no characters", or words starting with "no ".\n'
    '5. Return ONLY valid JSON: {"prompt": "...", "negative_prompt": "..."} (prompt max 40 words).'
)

_LAST_ACTIVE_OLLAMA_MODEL: Optional[str] = None

_BAD_NEGATIVE_CONCEPTS = {
    "background",
    "backgrounds",
    "color",
    "colors",
    "colour",
    "colours",
    "detail",
    "details",
    "style",
    "styles",
    "character",
    "characters",
    "person",
    "people",
    "art",
    "artwork",
    "illustration",
    "image",
    "shading",
    "lighting",
    "texture",
    "textures",
    "quality",
    "high quality",
    "anime",
    "manga",
    "lineart",
    "lines",
    "scene",
}

_REMOVAL_KEYWORDS_RE = re.compile(
    r"\b(delete|remove|erase|clean|clear|strip|wi?thout|rid\s+of|sfx|text|kanji|katakana|hiragana|subtitle|bubble|watermark|"
    r"убрать|убери|удалить|удали|стереть|сотри|очистить|очисти|замазать|без\s+текста|текст|звук|баббл)\b",
    re.IGNORECASE,
)


def _is_removal_instruction(instruction: str) -> bool:
    return bool(_REMOVAL_KEYWORDS_RE.search(instruction or ""))


def _sanitize_negative_prompt(neg_raw: str, instruction: str = "") -> str:
    tokens: List[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", neg_raw or ""):
        t = part.strip().strip('."\'')
        if not t:
            continue
        t_clean = re.sub(r"^(?:no|without|not)\s+", "", t, flags=re.IGNORECASE).strip()
        if not t_clean:
            continue
        low = t_clean.lower()
        if low in _BAD_NEGATIVE_CONCEPTS or low in ("none", "n/a", "null", "nothing"):
            continue
        if low not in seen:
            seen.add(low)
            tokens.append(t_clean)

    if _is_removal_instruction(instruction):
        for req in ("text", "japanese sfx", "kanji", "katakana", "letters", "speech bubble", "watermark", "blurry", "visible seam"):
            if req.lower() not in seen:
                seen.add(req.lower())
                tokens.append(req)

    if not tokens:
        return DEFAULT_NEGATIVE_PROMPT
    return ", ".join(tokens)[:500]

_FLUX3_PUP_PROMPTS_CACHE: Optional[Dict[str, str]] = None


def get_flux3_pup_system_prompts() -> Dict[str, str]:
    """Load Black Forest Labs' official ``system_prompts.json`` (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`)."""
    global _FLUX3_PUP_PROMPTS_CACHE
    if _FLUX3_PUP_PROMPTS_CACHE is not None:
        return _FLUX3_PUP_PROMPTS_CACHE
    prompts_file = Path(__file__).resolve().parent / "registry" / "flux3_pup_system_prompts.json"
    try:
        if prompts_file.is_file():
            _FLUX3_PUP_PROMPTS_CACHE = json.loads(prompts_file.read_text(encoding="utf-8"))
            return _FLUX3_PUP_PROMPTS_CACHE
    except Exception as exc:
        logger.warning("[vision] Could not read flux3_pup_system_prompts.json: %s", exc)
    _FLUX3_PUP_PROMPTS_CACHE = {
        "t2i": (
            "You are the prompt upsampler of a text-to-image model. Expand the user's short "
            "request into ONE detailed, self-contained caption: the exact text the image model "
            "receives. Output only the caption."
        ),
        "i2i": (
            "You are the prompt upsampler of an image-editing model. You see the reference "
            "image(s) and the user's short request. Expand it into ONE detailed, self-contained "
            "instruction as one continuous English prose block. Output only the instruction."
        ),
    }
    return _FLUX3_PUP_PROMPTS_CACHE


def is_flux3_pup_model(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    m = model_id.lower()
    if "flux-3-image-klein-pup" in m or "flux3-pup" in m:
        return True
    entry = get_catalog_entry(model_id)
    return bool(entry and "flux3_pup" in (entry.get("capabilities") or []))


class VisionRuntimeError(RuntimeError):
    """Base error for vision runtime failures."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


class VisionTimeoutError(VisionRuntimeError):
    """Raised when VLM inference times out."""

    def __init__(self, message: str = "Vision model request timed out") -> None:
        super().__init__(message, status_code=504)


def validate_ollama_url(raw_url: str) -> str:
    """Ensure the Ollama base URL points to a valid http/https endpoint."""
    url = (raw_url or "http://127.0.0.1:11434").strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid Ollama URL: {raw_url!r}")
    return url


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
            if not raw:
                return {}
            return json.loads(raw)
    except (TimeoutError, socket.timeout) as exc:
        raise VisionTimeoutError(f"Ollama timed out after {timeout}s") from exc
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        if exc.code == 404:
            raise VisionRuntimeError(f"Ollama model or endpoint not found: {body}", status_code=409) from exc
        raise VisionRuntimeError(f"Ollama HTTP {exc.code}: {body or exc.reason}", status_code=503) from exc
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            raise VisionTimeoutError(f"Ollama timed out after {timeout}s") from exc
        raise VisionRuntimeError(f"Ollama unavailable at {url}: {exc.reason}", status_code=503) from exc
    except Exception as exc:
        raise VisionRuntimeError(f"Ollama error: {exc}", status_code=503) from exc


_VISION_TAG_HINTS = (
    "vl",
    "vision",
    "llava",
    "minicpm-v",
    "gemma3",
    "moondream",
    "pixtral",
)


def tag_supports_vision(tag: str, capabilities: Optional[List[str]] = None) -> bool:
    """Return True only for multimodal/VLM tags.

    Ollama reports a ``capabilities`` array on ``/api/tags`` in recent versions.
    Older versions omit it, so fall back to the local catalog and well-known
    vision-model name hints.  Text-only models such as ``qwen2.5-coder`` must
    never be selected as a VLM.
    """
    name = (tag or "").strip()
    if not name:
        return False
    if capabilities is not None:
        caps = [str(c).strip().lower() for c in capabilities]
        return "vision" in caps

    lower = name.lower()
    if "coder" in lower:
        return False
    entry = get_catalog_entry(name)
    if entry and "vision" in [str(c).lower() for c in (entry.get("capabilities") or [])]:
        return True
    return any(hint in lower for hint in _VISION_TAG_HINTS)


def pick_vision_tag(installed_tags: List[str], vision_tags: Optional[List[str]] = None) -> Optional[str]:
    """Choose the first installed tag that is actually a VLM."""
    candidates = list(vision_tags or [])
    if not candidates:
        candidates = [tag for tag in (installed_tags or []) if tag_supports_vision(tag)]
    return candidates[0] if candidates else None


def check_ollama_status(ollama_url: str, timeout: float = 2.5) -> Dict[str, Any]:
    """Query Ollama ``/api/tags`` (and ``/api/version``) to check availability."""
    base = validate_ollama_url(ollama_url)
    tags: List[str] = []
    vision_tags: List[str] = []
    version: Optional[str] = None
    try:
        tags_resp = _http_json(f"{base}/api/tags", method="GET", timeout=timeout)
        models = tags_resp.get("models") or []
        for m in models:
            name = m.get("name") or m.get("model")
            if not (isinstance(name, str) and name.strip()):
                continue
            clean_name = name.strip()
            tags.append(clean_name)
            caps = m.get("capabilities")
            if tag_supports_vision(clean_name, caps if isinstance(caps, list) else None):
                vision_tags.append(clean_name)
    except VisionRuntimeError:
        return {
            "ollama_available": False,
            "ollama_version": None,
            "installed_tags": [],
            "vision_tags": [],
        }

    try:
        ver_resp = _http_json(f"{base}/api/version", method="GET", timeout=timeout)
        version = ver_resp.get("version")
    except Exception:
        version = None

    return {
        "ollama_available": True,
        "ollama_version": version,
        "installed_tags": tags,
        "vision_tags": vision_tags,
    }


def _sanitize_positive_prompt(prompt_raw: str) -> str:
    p = (prompt_raw or "").strip()
    # Strip any CJK (Japanese/Chinese/Korean) characters so the diffusion model never renders text
    p = re.sub(r'["\'«„“”\(\[]*[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]+["\'»“”\)\]]*', "", p)
    # Remove any meta-references to the red mask overlay so the diffusion model doesn't paint red shapes
    p = re.sub(
        r"\s*(?:to\s+seamlessly\s+fill|filling|inside|within|in)\s+the\s+(?:bright\s+)?(?:red[-\s]+)?(?:masked|highlighted)\s+(?:shapes|area|region|pixels|zone)s?\.?$",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(r"\bbright\s+red\s+(?:masked|highlighted)\s+(?:shapes|area|region|pixels)\b", "surrounding area", p, flags=re.IGNORECASE)
    p = re.sub(r"\bred\s+(?:masked|highlighted)\s+(?:shapes|area|region|pixels)\b", "surrounding area", p, flags=re.IGNORECASE)
    p = re.sub(r"\s{2,}", " ", p)
    return p.strip().rstrip(",").strip()


def parse_vlm_response_text(raw_text: str, instruction: str = "") -> Tuple[str, str]:
    """Extract ``(prompt, negative_prompt)`` from VLM or FLUX 3 PUP text output."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from vision model")

    # Strip <think>...</think> blocks if any model emitted reasoning tags
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # Strip ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match and brace_match.group(0) != cleaned:
        candidates.append(brace_match.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                prompt = _sanitize_positive_prompt(str(obj.get("prompt") or ""))
                neg = str(obj.get("negative_prompt") or obj.get("negative") or "").strip()
                if prompt:
                    neg_clean = _sanitize_negative_prompt(neg, instruction=instruction)
                    return prompt[:1000], neg_clean[:500]
        except Exception:
            continue

    # Plain prose caption (e.g. from FLUX 3 Prompt Upsampler)
    prose_lines = [
        ln.strip()
        for ln in cleaned.splitlines()
        if ln.strip() and not ln.strip().startswith(("{", "}"))
    ]
    if prose_lines:
        combined = " ".join(prose_lines)
        combined = re.sub(r"^(prompt\s*:\s*)", "", combined, flags=re.IGNORECASE).strip().strip('"')
        if combined:
            return combined[:1500], DEFAULT_NEGATIVE_PROMPT[:500]

    raise ValueError("Could not extract prompt from vision model response")


def run_nunchaku_pup_inference(
    *,
    model_dir: Path,
    quant_file: str,
    prompt_text: str,
    images_b64: Optional[List[str]] = None,
    timeout_sec: int = 120,
    max_new_tokens: int = 180,
) -> str:
    """Run ``tonera/flux-3-image-klein-pup-nunchaku`` inside the ``envs/extras`` Python environment."""
    from .extras import extras_python

    py_exe = extras_python()
    if not py_exe or not Path(py_exe).is_file():
        raise VisionRuntimeError("Nunchaku Python environment (envs/extras) is not installed", status_code=503)

    quant_path = (model_dir / quant_file).resolve()
    if not quant_path.is_file():
        raise VisionRuntimeError(f"Nunchaku PUP weights not found: {quant_path}", status_code=409)

    # Ensure system_prompts.json exists in model_dir
    sys_prompts_path = model_dir / "system_prompts.json"
    if not sys_prompts_path.is_file():
        sys_prompts_path.write_text(
            json.dumps(get_flux3_pup_system_prompts(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    runner_code = r"""
import base64, io, json, sys
from pathlib import Path
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor
from nunchaku import NunchakuQwen3_5ForConditionalGeneration
import nunchaku.models.text_encoders.qwen3_5_pup as _q35_pup

try:
    import triton  # noqa: F401
    _has_triton = True
except ImportError:
    _has_triton = False
    # Windows envs/extras may not have Triton; use native PyTorch CUDA ops for RMSNorm & SiLU*Mul
    def _rms_gain_forward(self, x):
        out = self._norm(x.float()) * self.weight.float()
        return out.type_as(x)
    def _rms_gated_forward(self, hidden_states, gate=None):
        return super(_q35_pup.Qwen3_5RMSNormGatedFused, self).forward(hidden_states, gate)
    def _mlp_gate_up_forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)
    _q35_pup.Qwen3_5RMSNormGain.forward = _rms_gain_forward
    _q35_pup.Qwen3_5RMSNormGatedFused.forward = _rms_gated_forward
    _q35_pup.Qwen3_5MLPGateUp.forward = _mlp_gate_up_forward

req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_dir = req["model_dir"]
quant_path = req["quant_path"]
prompt = req["prompt"]
images_b64 = req.get("images_b64") or []
max_new_tokens = int(req.get("max_new_tokens") or 512)

sys_prompts = json.loads((Path(model_dir) / "system_prompts.json").read_text(encoding="utf-8"))
processor = AutoProcessor.from_pretrained(model_dir)
upsampler = NunchakuQwen3_5ForConditionalGeneration.from_pretrained(
    quant_path, device="cuda", dtype=torch.bfloat16, decode_cuda_graph=_has_triton
).eval()

pil_images = []
for b64_str in images_b64:
    raw = base64.b64decode(b64_str)
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    pil_images.append(im)

if pil_images:
    text = (
        "Reference images follow in order (the first image, the second image, ...).\n"
        f"User request: {prompt}\nOutput the caption."
    )
    content = [{"type": "image", "image": im} for im in pil_images] + [{"type": "text", "text": text}]
    messages = [{"role": "system", "content": sys_prompts["i2i"]}, {"role": "user", "content": content}]
else:
    messages = [{"role": "system", "content": sys_prompts["t2i"]}, {"role": "user", "content": prompt}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to("cuda")

with torch.no_grad():
    output = upsampler.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

prompt_tokens = int(inputs["input_ids"].shape[1])
new_tokens = output[0, prompt_tokens:]
caption = processor.decode(new_tokens, skip_special_tokens=True).strip()
print("PUP_RESULT_JSON:" + json.dumps({"caption": caption}, ensure_ascii=True), flush=True)
"""
    with tempfile.TemporaryDirectory(prefix="lazysd_pup_") as tmpdir:
        req_file = Path(tmpdir) / "request.json"
        script_file = Path(tmpdir) / "run_pup.py"
        req_file.write_text(
            json.dumps({
                "model_dir": str(model_dir.resolve()),
                "quant_path": str(quant_path),
                "prompt": prompt_text,
                "images_b64": images_b64 or [],
                "max_new_tokens": int(max_new_tokens or 180),
            }),
            encoding="utf-8",
        )
        script_file.write_text(runner_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [str(py_exe), str(script_file), str(req_file)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(15, int(timeout_sec)),
            )
        except subprocess.TimeoutExpired as exc:
            raise VisionTimeoutError(f"FLUX.3 Nunchaku Prompt Upsampler timed out after {timeout_sec}s") from exc

        if proc.returncode != 0:
            err_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise VisionRuntimeError(f"Nunchaku PUP failed: {err_tail}", status_code=503)

        for line in (proc.stdout or "").splitlines():
            if line.startswith("PUP_RESULT_JSON:"):
                data = json.loads(line[len("PUP_RESULT_JSON:"):])
                caption = str(data.get("caption") or "").strip()
                if caption:
                    return caption

        raise VisionRuntimeError("Nunchaku PUP did not return a caption", status_code=503)


def generate_vlm_prompt(
    *,
    ollama_url: str,
    model_id: str,
    instruction: str,
    overview_b64: str,
    crop_b64: str,
    analysis_hint: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 120,
    keep_alive: str = "0s",
    models_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Generate ``(prompt, negative_prompt)`` for Inpaint via Nunchaku PUP, FLUX 3 PUP GGUF, or Ollama VLM."""
    entry = get_catalog_entry(model_id)
    if entry and entry.get("provider") == "nunchaku":
        base_models_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
        model_dir = base_models_dir / entry["local_dir"]
        instr_clean = (instruction or "").strip()
        colors = ", ".join((analysis_hint or {}).get("dominant_colors") or [])
        if _is_removal_instruction(instr_clean):
            req_text = (
                f"In the bright red-highlighted mask region: {instr_clean}. "
                "Do NOT describe any unmasked character, face, hair, or outfit standing next to the red mask. "
                f"Describe ONLY the clean background surface, lighting, colors ({colors or 'matching surroundings'}), "
                "and manga/anime brush texture that should seamlessly fill the red masked area after removal (max 45 words)."
            )
        elif instr_clean:
            req_text = (
                f"In the bright red-highlighted mask region: {instr_clean}. "
                "Focus strictly on what should appear inside the red mask, matching surrounding art style (max 45 words)."
            )
        else:
            req_text = (
                "Seamlessly restore the background texture, lighting, and colors inside the bright red-highlighted mask region, "
                "ignoring unmasked characters outside the red mask (max 45 words)."
            )
        caption = run_nunchaku_pup_inference(
            model_dir=model_dir,
            quant_file=entry["primary_file"],
            prompt_text=req_text,
            images_b64=[overview_b64, crop_b64],
            timeout_sec=timeout_sec,
            max_new_tokens=140,
        )
        p_out, n_out = parse_vlm_response_text(caption, instruction=instruction)
        return _sanitize_positive_prompt(p_out), _sanitize_negative_prompt(n_out, instruction=instruction)

    global _LAST_ACTIVE_OLLAMA_MODEL
    base = validate_ollama_url(ollama_url)
    use_flux3_pup = is_flux3_pup_model(model_id)
    effective_keep_alive = keep_alive if (keep_alive and keep_alive != "0s") else "5m"
    _LAST_ACTIVE_OLLAMA_MODEL = model_id

    if use_flux3_pup:
        sys_prompts = get_flux3_pup_system_prompts()
        req_text = instruction.strip() if (instruction and instruction.strip()) else (
            "Seamlessly restore the area marked in red in the first image and shown close-up in the second image."
        )
        user_content = (
            "Reference images follow in order (the first image, the second image). "
            "Only the bright RED highlighted mask area is being repainted; ignore unmasked characters outside the red area.\n"
            f"User request for the red masked area: {req_text}\nOutput the caption."
        )
        payload = {
            "model": model_id,
            "stream": False,
            "keep_alive": effective_keep_alive,
            "think": False,
            "messages": [
                {"role": "system", "content": sys_prompts["i2i"]},
                {
                    "role": "user",
                    "content": user_content,
                    "images": [overview_b64, crop_b64],
                },
            ],
            "options": {
                "temperature": 0.7,
                "top_k": 1,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "num_predict": 220,
            },
        }
    else:
        user_parts = []
        instr_clean = (instruction or "").strip()[:500]
        is_removal = _is_removal_instruction(instr_clean)
        colors = ", ".join((analysis_hint or {}).get("dominant_colors") or [])
        rtype = (analysis_hint or {}).get("region_type") or "colored_art"

        if instr_clean:
            user_parts.append(f"User edit instruction for the BRIGHT RED masked pixels: {instr_clean}")
        else:
            user_parts.append("User edit instruction: seamlessly inpaint and restore the background inside the BRIGHT RED masked pixels.")

        if colors:
            user_parts.append(f"Measured boundary colors around the red mask: [{colors}] (region_type={rtype}).")

        if is_removal:
            user_parts.append(
                "IMPORTANT: This is an OBJECT/TEXT/SFX REMOVAL task! "
                "Do NOT describe any character, face, hair, hat, or outfit standing next to the red mask. "
                "Describe ONLY the clean background surface, lighting, colors, and art texture that should fill "
                f"the red masked shapes after the unwanted SFX/text is erased (matching surrounding {colors or 'colors'})."
            )
        else:
            user_parts.append(
                "IMPORTANT: Focus strictly on the BRIGHT RED highlighted mask shapes in Image 2. "
                "Do NOT caption unmasked characters outside the red highlight."
            )

        user_parts.append(
            'Respond ONLY with JSON: {"prompt": "...", "negative_prompt": "..."}.'
        )
        payload = {
            "model": model_id,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": effective_keep_alive,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n".join(user_parts),
                    "images": [overview_b64, crop_b64],
                },
            ],
            "options": {
                "temperature": 0.15,
                "num_predict": 130,
            },
        }

    resp = _http_json(
        f"{base}/api/chat",
        method="POST",
        payload=payload,
        timeout=max(5.0, float(timeout_sec)),
    )
    msg = resp.get("message") or {}
    content = msg.get("content") or resp.get("response") or ""
    return parse_vlm_response_text(str(content), instruction=instruction)


def upsample_flux3_prompt(
    *,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    images_b64: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    ollama_url: str = "http://127.0.0.1:11434",
    timeout_sec: int = 120,
    keep_alive: str = "0s",
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Upsample a short user prompt into a detailed FLUX 3 Image Klein caption (`t2i` or `i2i`)."""
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Prompt cannot be empty")
    if len(clean_prompt) > 2000:
        clean_prompt = clean_prompt[:2000]

    base_models_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
    sys_prompts = get_flux3_pup_system_prompts()
    has_images = bool(images_b64)
    target_prefix = f"TARGET CANVAS: {int(width or 1024)}:{int(height or 1024)}\n" if not has_images else ""
    formatted_request = f"{target_prefix}{clean_prompt}"

    # 1. Check if Nunchaku PUP is requested or installed locally
    nunchaku_entry = get_catalog_entry("flux3-pup-nunchaku")
    if (
        (model_id == "flux3-pup-nunchaku" or not model_id)
        and nunchaku_entry
        and is_local_vision_model_installed(nunchaku_entry, models_dir=base_models_dir)
    ):
        caption = run_nunchaku_pup_inference(
            model_dir=base_models_dir / nunchaku_entry["local_dir"],
            quant_file=nunchaku_entry["primary_file"],
            prompt_text=formatted_request,
            images_b64=images_b64,
            timeout_sec=timeout_sec,
        )
        return {
            "prompt": caption,
            "source": "nunchaku_pup",
            "model": "flux3-pup-nunchaku",
        }

    # 2. Try Ollama with the requested model or any installed vision model.
    # Never fall back to a text-only model (e.g. qwen2.5-coder).
    st = check_ollama_status(ollama_url)
    installed_tags = st.get("installed_tags") or []
    vision_tags = st.get("vision_tags")
    if vision_tags is None:
        vision_tags = [tag for tag in installed_tags if tag_supports_vision(tag)]
    if st["ollama_available"] and vision_tags:
        chosen_tag = None
        if model_id and is_tag_matching(model_id, vision_tags):
            chosen_tag = next(
                (tag for tag in vision_tags if is_tag_matching(model_id, [tag])),
                vision_tags[0],
            )
        else:
            chosen_tag = vision_tags[0]

        if chosen_tag:
            base = validate_ollama_url(ollama_url)
            if has_images:
                user_msg: Dict[str, Any] = {
                    "role": "user",
                    "content": (
                        "Reference images follow in order (the first image, the second image, ...).\n"
                        f"User request: {clean_prompt}\nOutput the caption."
                    ),
                    "images": images_b64,
                }
                sys_content = sys_prompts["i2i"]
                pres_pen = 0.0
            else:
                user_msg = {"role": "user", "content": formatted_request}
                sys_content = sys_prompts["t2i"]
                pres_pen = 1.5

            resp = _http_json(
                f"{base}/api/chat",
                method="POST",
                payload={
                    "model": chosen_tag,
                    "stream": False,
                    "keep_alive": keep_alive or "0s",
                    "think": False,
                    "messages": [
                        {"role": "system", "content": sys_content},
                        user_msg,
                    ],
                    "options": {
                        "temperature": 1.0,
                        "top_k": 1,
                        "top_p": 0.95,
                        "presence_penalty": pres_pen,
                        "repeat_penalty": 1.0,
                        "num_predict": 600,
                    },
                },
                timeout=max(5.0, float(timeout_sec)),
            )
            msg = resp.get("message") or {}
            content = str(msg.get("content") or resp.get("response") or "").strip()
            content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
            if content:
                return {
                    "prompt": content[:2000],
                    "source": "ollama_pup",
                    "model": chosen_tag,
                }

    # 3. Deterministic BFL-style local caption expansion fallback (works offline with zero models)
    aspect_hint = (
        "wide horizontal composition"
        if width > height * 1.15
        else ("tall vertical composition" if height > width * 1.15 else "balanced square composition")
    )
    expanded = (
        f"Full-color high-detail image, {aspect_hint}. {clean_prompt.rstrip('.')}. "
        "Crisp subject boundaries, natural coherent lighting, accurate anatomy and proportions, "
        "rich material textures, clean background separation."
    )
    return {
        "prompt": expanded,
        "source": "heuristic_pup",
        "model": None,
        "warnings": [
            "FLUX.3 Prompt Upsampler model is not installed yet — used local template expansion. "
            "You can download FLUX.3 Prompt Upsampler (Nunchaku SVDQ or Unsloth GGUF) in the Inpaint Assistant."
        ],
    }


def unload_vision_model(ollama_url: str, model_id: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Explicitly unload any active VLM / Prompt Upsampler from Ollama VRAM before starting diffusion."""
    global _LAST_ACTIVE_OLLAMA_MODEL
    targets = []
    for cand in (_LAST_ACTIVE_OLLAMA_MODEL, model_id):
        if cand and cand != "flux3-pup-nunchaku" and cand not in targets:
            targets.append(cand)
    unloaded = False
    try:
        base = validate_ollama_url(ollama_url)
        # Also query /api/ps so any running Ollama model (PUP GGUF, Qwen3-VL, Gemma3, etc.) is unloaded
        try:
            ps_resp = _http_json(f"{base}/api/ps", method="GET", timeout=1.5)
            for m in ps_resp.get("models") or []:
                m_name = m.get("name") or m.get("model")
                if isinstance(m_name, str) and m_name.strip() and m_name.strip() not in targets:
                    targets.append(m_name.strip())
        except Exception:
            pass
        for t_model in targets:
            try:
                _http_json(
                    f"{base}/api/generate",
                    method="POST",
                    payload={"model": t_model, "keep_alive": 0},
                    timeout=timeout,
                )
                unloaded = True
            except Exception:
                pass
        return unloaded
    except Exception as exc:
        logger.debug("[vision] Unload skipped or failed: %s", exc)
        return False


def _notify_progress(
    cb: Optional[Callable[..., None]],
    pct: float,
    detail: str,
    downloaded_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None,
    speed_bps: Optional[float] = None,
    status: str = "downloading",
    error: Optional[str] = None,
) -> None:
    if not cb:
        return
    try:
        cb(
            pct,
            detail,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
            status=status,
            error=error,
        )
    except TypeError:
        try:
            cb(pct, detail)
        except Exception:
            pass
    except Exception:
        pass


def _download_hf_vision_model(
    entry: Dict[str, Any],
    job_id: str,
    models_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> None:
    """Download a HuggingFace-hosted vision/PUP model (Nunchaku SVDQ or Unsloth GGUF) into ``models/cache/<local_dir>``."""
    from backend.sources import get_source

    model_id = entry["id"]
    repo_id = entry["repo_id"]
    base_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
    target_dir = base_dir / entry["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN") or None
    src = get_source("huggingface", token=token)

    files_to_fetch = list(entry.get("support_files") or []) + [entry["primary_file"]]
    total_files = len(files_to_fetch)

    for idx, fname in enumerate(files_to_fetch, 1):
        base_pct = (idx - 1) / max(1, total_files) * 100.0
        span_pct = 100.0 / max(1, total_files)
        detail = f"Downloading {fname} ({idx}/{total_files})..."
        set_download_state(
            model_id,
            status="downloading",
            job_id=job_id,
            progress=round(base_pct, 1),
            detail=detail,
        )
        _notify_progress(progress_callback, round(base_pct, 1), detail, status="downloading")

        def _file_progress(dl_bytes: int, tot_bytes: int, speed_bps: float, _fname: str = fname, _idx: int = idx, _base: float = base_pct, _span: float = span_pct) -> None:
            frac = (dl_bytes / float(tot_bytes)) if tot_bytes > 0 else 0.0
            cur_pct = min(99.9, round(_base + frac * _span, 1))
            dl_mb = dl_bytes / (1024 * 1024)
            tot_mb = tot_bytes / (1024 * 1024) if tot_bytes > 0 else 0.0
            spd_mb = (speed_bps or 0.0) / (1024 * 1024)
            spd_str = f" · {spd_mb:.1f} MB/s" if spd_mb >= 0.1 else ""
            msg = f"{_fname} ({_idx}/{total_files}): {dl_mb:.1f} / {tot_mb:.1f} MB{spd_str}" if tot_bytes > 0 else f"Downloading {_fname} ({_idx}/{total_files})..."
            set_download_state(
                model_id,
                status="downloading",
                job_id=job_id,
                progress=cur_pct,
                detail=msg,
            )
            _notify_progress(
                progress_callback,
                cur_pct,
                msg,
                downloaded_bytes=dl_bytes,
                total_bytes=tot_bytes,
                speed_bps=speed_bps,
                status="downloading",
            )

        src.download_file(
            repo_id=repo_id,
            file_path=fname,
            local_dir=target_dir,
            revision="main",
            progress_callback=_file_progress,
            job_id=job_id,
        )

    set_download_state(
        model_id,
        status="installed",
        job_id=job_id,
        progress=100.0,
        detail="Installed",
    )
    _notify_progress(progress_callback, 100.0, "Installed ✓", status="complete")


def import_local_vision_model_to_ollama(
    model_id: str,
    ollama_url: str = "http://127.0.0.1:11434",
    models_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    job_id: Optional[str] = None,
    timeout_sec: float = 600.0,
) -> Dict[str, Any]:
    """Create an Ollama model from already-downloaded GGUF + mmproj files.

    This avoids re-downloading Gemma/Qwen weights that the Hub already stored in
    ``models/cache/<local_dir>``.  Ollama copies the layers into its own blob
    store, but no network transfer is involved.
    """
    entry = get_catalog_entry(model_id)
    if not entry:
        raise VisionRuntimeError(f"Unknown vision model: {model_id}", status_code=404)
    primary_file = str(entry.get("primary_file") or "").strip()
    mmproj_file = str(entry.get("mmproj_file") or "").strip()
    local_dir = str(entry.get("local_dir") or "").strip()
    if not primary_file or not mmproj_file or not local_dir:
        raise VisionRuntimeError(
            f"Model '{model_id}' does not declare GGUF + mmproj files.",
            status_code=400,
        )

    base_dir = (Path(models_dir) if models_dir else (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
    model_dir = base_dir / local_dir
    primary_path = model_dir / primary_file
    mmproj_path = model_dir / mmproj_file
    if not primary_path.is_file() or primary_path.stat().st_size < 1024:
        raise VisionRuntimeError(
            f"Local GGUF not found: {primary_path}. Download the model first.",
            status_code=409,
        )
    if not mmproj_path.is_file() or mmproj_path.stat().st_size < 1024:
        raise VisionRuntimeError(
            f"Local vision projector (mmproj) not found: {mmproj_path}.",
            status_code=409,
        )

    from .ollama_engine import find_ollama_binary
    ollama_exe = find_ollama_binary()
    if not ollama_exe:
        raise VisionRuntimeError(
            "Ollama binary not found. Install Ollama or start it first.",
            status_code=503,
        )

    tag = entry["id"]
    detail = f"Importing local GGUF into Ollama as '{tag}'..."
    set_download_state(
        model_id,
        status="downloading",
        job_id=job_id,
        progress=5.0,
        detail=detail,
    )
    _notify_progress(progress_callback, 5.0, detail, status="downloading")

    # Point the CLI at the same daemon the app uses.
    parsed = urllib.parse.urlparse(validate_ollama_url(ollama_url))
    env = os.environ.copy()
    if parsed.hostname:
        env["OLLAMA_HOST"] = f"{parsed.hostname}:{parsed.port or 11434}"

    try:
        with tempfile.TemporaryDirectory(prefix="lazysd_ollama_import_") as tmp:
            modelfile = Path(tmp) / "Modelfile"
            # Two FROM lines: base LLM weights + vision projector (mmproj).
            modelfile.write_text(
                f'FROM "{primary_path.as_posix()}"\nFROM "{mmproj_path.as_posix()}"\n',
                encoding="utf-8",
            )
            proc = subprocess.run(
                [str(ollama_exe), "create", tag, "-f", str(modelfile)],
                capture_output=True,
                text=True,
                timeout=max(30.0, float(timeout_sec)),
                env=env,
            )
    except subprocess.TimeoutExpired as exc:
        raise VisionRuntimeError(
            f"Ollama import timed out after {timeout_sec}s.",
            status_code=504,
        ) from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise VisionRuntimeError(
            f"Ollama import failed: {err or 'unknown error'}",
            status_code=500,
        )

    st = check_ollama_status(ollama_url)
    if not is_tag_matching(tag, st.get("installed_tags") or []):
        raise VisionRuntimeError(
            f"Ollama import finished but '{tag}' is missing from /api/tags.",
            status_code=500,
        )

    set_download_state(
        model_id,
        status="installed",
        job_id=job_id,
        progress=100.0,
        detail="Imported into Ollama",
    )
    _notify_progress(progress_callback, 100.0, "Imported into Ollama ✓", status="complete")
    return {"model": tag, "status": "installed"}


def start_vision_model_import(
    ollama_url: str,
    model_id: str,
    progress_callback: Optional[Callable[..., None]] = None,
    models_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Start an asynchronous local-GGUF -> Ollama import job."""
    entry = get_catalog_entry(model_id)
    if not entry:
        raise VisionRuntimeError(f"Unknown vision model: {model_id}", status_code=404)
    existing = get_download_state(entry["id"])
    if existing and existing.get("status") == "downloading":
        return existing

    state = set_download_state(
        entry["id"],
        status="downloading",
        job_id=job_id,
        progress=0.0,
        detail="Preparing Ollama import...",
    )
    job_id = state["job_id"]

    def _worker() -> None:
        try:
            import_local_vision_model_to_ollama(
                entry["id"],
                ollama_url=ollama_url,
                models_dir=models_dir,
                progress_callback=progress_callback,
                job_id=job_id,
            )
        except Exception as exc:
            logger.warning("[vision] Ollama import failed for %s: %s", entry["id"], exc)
            set_download_state(
                entry["id"],
                status="error",
                job_id=job_id,
                progress=0.0,
                detail="Import failed",
                error=str(exc),
            )
            _notify_progress(
                progress_callback,
                0.0,
                f"Import failed: {exc}",
                status="error",
                error=str(exc),
            )

    thread = threading.Thread(target=_worker, name=f"vision-import-{entry['id']}", daemon=True)
    thread.start()
    return state


def start_vision_model_download(
    ollama_url: str,
    model_id: str,
    progress_callback: Optional[Callable[..., None]] = None,
    models_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Start (or attach to) an asynchronous download job for a VLM or FLUX.3 PUP model."""
    existing = get_download_state(model_id)
    if existing and existing.get("status") == "downloading":
        return existing

    entry = get_catalog_entry(model_id)
    eff_provider = (preferred_provider or (entry.get("provider") if entry else "ollama") or "ollama").strip().lower()
    if entry and entry.get("provider") == "nunchaku":
        eff_provider = "nunchaku"

    state = set_download_state(
        model_id,
        status="downloading",
        job_id=job_id,
        progress=0.0,
        detail=f"Starting download ({eff_provider})...",
    )
    job_id = state["job_id"]

    def _worker() -> None:
        try:
            # Download via HuggingFace DownloadManager when provider is 'nunchaku' or 'gguf' (local HF cache)
            if entry and eff_provider in ("nunchaku", "gguf", "llamacpp") and entry.get("repo_id") and entry.get("primary_file"):
                _download_hf_vision_model(
                    entry,
                    job_id=job_id,
                    models_dir=models_dir,
                    hf_token=hf_token,
                    progress_callback=progress_callback,
                )
                return

            base = validate_ollama_url(ollama_url)
            url = f"{base}/api/pull"
            data = json.dumps({"name": model_id, "stream": True}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    status_text = str(chunk.get("status") or "downloading")
                    total = float(chunk.get("total") or 0.0)
                    completed = float(chunk.get("completed") or 0.0)
                    pct = (completed / total * 100.0) if total > 0 else 0.0
                    set_download_state(
                        model_id,
                        status="downloading",
                        job_id=job_id,
                        progress=pct,
                        detail=status_text,
                    )
                    _notify_progress(
                        progress_callback,
                        pct,
                        status_text,
                        downloaded_bytes=int(completed) if completed > 0 else None,
                        total_bytes=int(total) if total > 0 else None,
                        status="downloading",
                    )

            set_download_state(
                model_id,
                status="installed",
                job_id=job_id,
                progress=100.0,
                detail="Installed",
            )
            _notify_progress(progress_callback, 100.0, "Installed ✓", status="complete")
        except Exception as exc:
            logger.warning("[vision] Download failed for %s: %s", model_id, exc)
            is_cancelled = "cancelled by user" in str(exc).lower()
            set_download_state(
                model_id,
                status="not_installed" if is_cancelled else "error",
                job_id=job_id,
                progress=0.0,
                detail="Cancelled" if is_cancelled else "Download failed",
                error=None if is_cancelled else str(exc),
            )
            _notify_progress(
                progress_callback,
                0.0,
                "Cancelled" if is_cancelled else f"Download failed: {exc}",
                status="cancelled" if is_cancelled else "error",
                error=str(exc),
            )

    t = threading.Thread(target=_worker, name=f"vision-dl-{model_id}", daemon=True)
    t.start()
    return state


def delete_vision_model(
    ollama_url: str,
    model_id: str,
    timeout: float = 10.0,
    models_dir: Optional[Path] = None,
) -> bool:
    """Delete a vision/PUP model from local HF cache and/or Ollama."""
    entry = get_catalog_entry(model_id)
    removed_local = False
    if entry and entry.get("local_dir") and entry.get("primary_file"):
        base_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
        primary_path = base_dir / entry["local_dir"] / entry["primary_file"]
        if primary_path.is_file():
            primary_path.unlink(missing_ok=True)
            removed_local = True

    if entry and entry.get("provider") == "nunchaku":
        clear_download_state(model_id)
        return True

    try:
        base = validate_ollama_url(ollama_url)
        _http_json(
            f"{base}/api/delete",
            method="DELETE",
            payload={"name": model_id},
            timeout=timeout,
        )
    except VisionRuntimeError:
        if not removed_local:
            raise
    clear_download_state(model_id)
    return True
