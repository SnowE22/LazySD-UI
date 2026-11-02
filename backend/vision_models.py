"""Catalog and runtime status tracking for local Vision-Language Models (VLM).

Supports the Inpaint Prompt Generator (``docs/INPAINT_PROMPT_GENERATOR_TZ.md``).
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional


VISION_MODELS_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "qwen2.5vl:3b",
        "name": "Qwen2.5-VL 3B",
        "size_gb": 3.2,
        "vram_min_gb": 6,
        "recommended": True,
        "capabilities": ["vision"],
        "description": "Fast & compact vision model recommended for GPUs with 6-8 GB VRAM.",
    },
    {
        "id": "qwen2.5vl:7b",
        "name": "Qwen2.5-VL 7B",
        "size_gb": 6.0,
        "vram_min_gb": 12,
        "recommended": False,
        "capabilities": ["vision"],
        "description": "Higher detail manga/art comprehension for GPUs with 12+ GB VRAM.",
    },
    {
        "id": "minicpm-v:8b",
        "name": "MiniCPM-V 8B",
        "size_gb": 5.5,
        "vram_min_gb": 12,
        "recommended": False,
        "capabilities": ["vision"],
        "description": "Strong OCR and scene layout understanding for 12+ GB VRAM.",
    },
    {
        "id": "llava:7b",
        "name": "LLaVA 7B",
        "size_gb": 4.7,
        "vram_min_gb": 8,
        "recommended": False,
        "capabilities": ["vision"],
        "description": "Classic general-purpose vision-language model (fallback compatible).",
    },
]

_LOCK = threading.Lock()
# Maps model_id -> {"job_id", "status", "progress", "detail", "error", "updated_at"}
_DOWNLOAD_STATE: Dict[str, Dict[str, Any]] = {}


def get_catalog_entry(model_id: str) -> Optional[Dict[str, Any]]:
    norm = (model_id or "").strip().lower()
    for item in VISION_MODELS_CATALOG:
        if item["id"].lower() == norm:
            return dict(item)
    return None


def _normalize_tag(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t.endswith(":latest"):
        return t[:-7]
    return t


def is_tag_matching(catalog_id: str, installed_tags: List[str]) -> bool:
    cid = (catalog_id or "").strip().lower()
    cid_base = _normalize_tag(cid)
    for raw in installed_tags:
        r = (raw or "").strip().lower()
        if r == cid or _normalize_tag(r) == cid:
            return True
        if ":" not in cid and _normalize_tag(r) == cid_base:
            return True
    return False


def get_download_state(model_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        st = _DOWNLOAD_STATE.get(model_id)
        return dict(st) if st else None


def set_download_state(
    model_id: str,
    *,
    status: str,
    job_id: Optional[str] = None,
    progress: float = 0.0,
    detail: str = "",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    with _LOCK:
        prev = _DOWNLOAD_STATE.get(model_id, {})
        entry = {
            "job_id": job_id or prev.get("job_id") or f"vlm_{uuid.uuid4().hex[:8]}",
            "model_id": model_id,
            "status": status,
            "progress": round(max(0.0, min(100.0, float(progress))), 1),
            "detail": detail,
            "error": error,
            "updated_at": time.time(),
        }
        _DOWNLOAD_STATE[model_id] = entry
        return dict(entry)


def clear_download_state(model_id: str) -> None:
    with _LOCK:
        _DOWNLOAD_STATE.pop(model_id, None)


def build_vision_models_response(
    installed_tags: List[str],
    ollama_available: bool,
    provider: str = "ollama",
) -> Dict[str, Any]:
    """Build the payload for ``GET /api/vision/models``."""
    models_out: List[Dict[str, Any]] = []
    for item in VISION_MODELS_CATALOG:
        mid = item["id"]
        installed = is_tag_matching(mid, installed_tags)
        dl = get_download_state(mid)

        if dl and dl.get("status") == "downloading":
            status = "downloading"
        elif installed:
            status = "installed"
        elif dl and dl.get("status") == "error":
            status = "error"
        else:
            status = "not_installed"

        entry: Dict[str, Any] = {
            "id": mid,
            "name": item["name"],
            "size_gb": item["size_gb"],
            "vram_min_gb": item["vram_min_gb"],
            "recommended": item["recommended"],
            "installed": bool(installed),
            "capabilities": list(item.get("capabilities", ["vision"])),
            "status": status,
            "description": item.get("description", ""),
        }
        if dl:
            entry["progress"] = dl.get("progress", 0.0)
            entry["detail"] = dl.get("detail", "")
            entry["job_id"] = dl.get("job_id")
            if dl.get("error"):
                entry["error"] = dl["error"]
        models_out.append(entry)

    return {
        "provider": provider,
        "available": bool(ollama_available),
        "models": models_out,
    }
