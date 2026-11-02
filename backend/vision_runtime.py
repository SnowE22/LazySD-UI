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
    set_download_state,
)

logger = logging.getLogger("quantfunc.vision_runtime")

SYSTEM_PROMPT = (
    "You are a prompt engineer for Stable Diffusion / Flux inpainting.\n"
    "You receive: the whole page with the edited region highlighted in red, "
    "and a close-up crop of that region.\n"
    "Return ONLY valid JSON:\n"
    '{"prompt": "...", "negative_prompt": "..."}\n'
    "The prompt must describe what should appear inside the masked area: "
    "colors, texture, style, continuity with surroundings.\n"
    "Do not transcribe or reproduce any text from the image.\n"
    'Do not include "no text" instructions as text to render; use the negative prompt.\n'
    "Max 60 words."
)

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


def check_ollama_status(ollama_url: str, timeout: float = 2.5) -> Dict[str, Any]:
    """Query Ollama ``/api/tags`` (and ``/api/version``) to check availability."""
    base = validate_ollama_url(ollama_url)
    tags: List[str] = []
    version: Optional[str] = None
    try:
        tags_resp = _http_json(f"{base}/api/tags", method="GET", timeout=timeout)
        models = tags_resp.get("models") or []
        for m in models:
            name = m.get("name") or m.get("model")
            if isinstance(name, str) and name.strip():
                tags.append(name.strip())
    except VisionRuntimeError:
        return {
            "ollama_available": False,
            "ollama_version": None,
            "installed_tags": [],
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
    }


def parse_vlm_response_text(raw_text: str) -> Tuple[str, str]:
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
                prompt = str(obj.get("prompt") or "").strip()
                neg = str(obj.get("negative_prompt") or obj.get("negative") or "").strip()
                if prompt:
                    if not neg:
                        neg = DEFAULT_NEGATIVE_PROMPT
                    return prompt[:1000], neg[:500]
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
) -> str:
    """Run ``tonera/flux-3-image-klein-pup-nunchaku`` inside the ``envs/extras`` Python environment."""
    from .extras import get_extras_python

    py_exe = get_extras_python()
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

req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_dir = req["model_dir"]
quant_path = req["quant_path"]
prompt = req["prompt"]
images_b64 = req.get("images_b64") or []
max_new_tokens = int(req.get("max_new_tokens") or 512)

sys_prompts = json.loads((Path(model_dir) / "system_prompts.json").read_text(encoding="utf-8"))
processor = AutoProcessor.from_pretrained(model_dir)
upsampler = NunchakuQwen3_5ForConditionalGeneration.from_pretrained(
    quant_path, dtype=torch.bfloat16
).eval().to("cuda")

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
print("PUP_RESULT_JSON:" + json.dumps({"caption": caption}, ensure_ascii=False), flush=True)
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
                "max_new_tokens": 512,
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
        req_text = instruction.strip() if (instruction and instruction.strip()) else (
            "Remove any unwanted text or artifacts in the highlighted region and seamlessly restore the background texture and colors."
        )
        caption = run_nunchaku_pup_inference(
            model_dir=model_dir,
            quant_file=entry["primary_file"],
            prompt_text=req_text,
            images_b64=[overview_b64, crop_b64],
            timeout_sec=timeout_sec,
        )
        return parse_vlm_response_text(caption)

    base = validate_ollama_url(ollama_url)
    use_flux3_pup = is_flux3_pup_model(model_id)

    if use_flux3_pup:
        sys_prompts = get_flux3_pup_system_prompts()
        req_text = instruction.strip() if (instruction and instruction.strip()) else (
            "Seamlessly restore the area marked in red in the first image and shown close-up in the second image."
        )
        user_content = (
            "Reference images follow in order (the first image, the second image).\n"
            f"User request: {req_text}\nOutput the caption."
        )
        payload = {
            "model": model_id,
            "stream": False,
            "keep_alive": keep_alive or "0s",
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
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "num_predict": 420,
            },
        }
    else:
        user_parts = []
        if instruction and instruction.strip():
            user_parts.append(f"User instruction: {instruction.strip()[:500]}")
        else:
            user_parts.append("User instruction: seamlessly inpaint and restore the masked area.")
        if analysis_hint:
            colors = ", ".join(analysis_hint.get("dominant_colors") or [])
            rtype = analysis_hint.get("region_type") or "colored_art"
            user_parts.append(f"Context hint: region_type={rtype}, surrounding_colors=[{colors}].")
        user_parts.append(
            "Image 1 shows the full page with the red mask overlay. "
            "Image 2 shows the close-up crop around the masked region. "
            'Respond ONLY with JSON: {"prompt": "...", "negative_prompt": "..."}.'
        )
        payload = {
            "model": model_id,
            "stream": False,
            "keep_alive": keep_alive or "0s",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n".join(user_parts),
                    "images": [overview_b64, crop_b64],
                },
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": 220,
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
    return parse_vlm_response_text(str(content))


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

    # 2. Try Ollama with the requested model or any installed PUP/vision model
    st = check_ollama_status(ollama_url)
    if st["ollama_available"] and st["installed_tags"]:
        chosen_tag = None
        if model_id and any(t.lower() == model_id.lower() for t in st["installed_tags"]):
            chosen_tag = model_id
        else:
            for tag in st["installed_tags"]:
                if "flux-3-image-klein-pup" in tag.lower() or "qwen" in tag.lower() or "llava" in tag.lower() or "minicpm" in tag.lower():
                    chosen_tag = tag
                    break
            if not chosen_tag:
                chosen_tag = st["installed_tags"][0]

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
    """Explicitly unload the active VLM from Ollama VRAM before starting diffusion."""
    if not model_id or model_id == "flux3-pup-nunchaku":
        return False
    try:
        base = validate_ollama_url(ollama_url)
        _http_json(
            f"{base}/api/generate",
            method="POST",
            payload={"model": model_id, "keep_alive": 0},
            timeout=timeout,
        )
        return True
    except Exception as exc:
        logger.debug("[vision] Unload skipped or failed for %s: %s", model_id, exc)
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


def start_vision_model_download(
    ollama_url: str,
    model_id: str,
    progress_callback: Optional[Callable[..., None]] = None,
    models_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Start (or attach to) an asynchronous download job for a VLM or FLUX.3 PUP model."""
    existing = get_download_state(model_id)
    if existing and existing.get("status") == "downloading":
        return existing

    entry = get_catalog_entry(model_id)
    state = set_download_state(
        model_id,
        status="downloading",
        job_id=job_id,
        progress=0.0,
        detail="Starting download...",
    )
    job_id = state["job_id"]

    def _worker() -> None:
        try:
            # HuggingFace-hosted PUP weights (Nunchaku SVDQ & Unsloth GGUF) download directly via Hub DownloadManager
            if entry and entry.get("provider") in ("nunchaku", "gguf") and entry.get("repo_id"):
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

    if entry and entry.get("provider") in ("nunchaku", "gguf"):
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
