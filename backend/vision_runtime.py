"""Ollama runtime adapter for local vision-language models (VLM).

Handles:
- Checking Ollama availability and version (``/api/version``, ``/api/tags``).
- Sending multimodal chat completions (``/api/chat``) with base64 images and
  ``keep_alive="0s"`` so the VLM frees VRAM immediately after replying.
- Explicit VRAM unloading before diffusion runs.
- Pulling (``/api/pull``) with streaming progress and deleting (``/api/delete``)
  vision models.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt_fallback import DEFAULT_NEGATIVE_PROMPT
from .vision_models import (
    clear_download_state,
    get_download_state,
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
    """Extract ``(prompt, negative_prompt)`` from VLM text output.

    Strips markdown code fences, attempts JSON parsing anywhere in the response,
    and falls back to the first non-empty line if the model did not emit valid JSON.
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from vision model")

    # Strip ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Try direct JSON or first {...} object
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

    # Fallback: take the first non-empty non-brace line
    for line in cleaned.splitlines():
        line_s = line.strip().strip('"').strip("'")
        if line_s and not line_s.startswith("{") and not line_s.startswith("}"):
            line_s = re.sub(r"^(prompt\s*:\s*)", "", line_s, flags=re.IGNORECASE).strip()
            if line_s:
                return line_s[:1000], DEFAULT_NEGATIVE_PROMPT[:500]

    raise ValueError("Could not extract prompt from vision model response")


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
) -> Tuple[str, str]:
    """Call Ollama ``/api/chat`` with overview + crop images and return ``(prompt, negative_prompt)``."""
    base = validate_ollama_url(ollama_url)
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


def unload_vision_model(ollama_url: str, model_id: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Explicitly unload the active VLM from Ollama VRAM before starting diffusion."""
    if not model_id:
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


def start_vision_model_download(
    ollama_url: str,
    model_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Start (or attach to) an asynchronous Ollama ``/api/pull`` job."""
    existing = get_download_state(model_id)
    if existing and existing.get("status") == "downloading":
        return existing

    state = set_download_state(
        model_id,
        status="downloading",
        progress=0.0,
        detail="Starting download...",
    )
    job_id = state["job_id"]
    base = validate_ollama_url(ollama_url)

    def _worker() -> None:
        url = f"{base}/api/pull"
        data = json.dumps({"name": model_id, "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
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
                    if progress_callback:
                        try:
                            progress_callback(pct, status_text)
                        except Exception:
                            pass

            set_download_state(
                model_id,
                status="installed",
                job_id=job_id,
                progress=100.0,
                detail="Installed",
            )
        except Exception as exc:
            logger.warning("[vision] Pull failed for %s: %s", model_id, exc)
            set_download_state(
                model_id,
                status="error",
                job_id=job_id,
                progress=0.0,
                detail="Download failed",
                error=str(exc),
            )

    t = threading.Thread(target=_worker, name=f"ollama-pull-{model_id}", daemon=True)
    t.start()
    return state


def delete_vision_model(ollama_url: str, model_id: str, timeout: float = 10.0) -> bool:
    """Delete a model from Ollama via ``DELETE /api/delete``."""
    base = validate_ollama_url(ollama_url)
    _http_json(
        f"{base}/api/delete",
        method="DELETE",
        payload={"name": model_id},
        timeout=timeout,
    )
    clear_download_state(model_id)
    return True
