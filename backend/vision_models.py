"""Catalog and runtime status tracking for local Vision-Language Models (VLM)
and the official FLUX 3 Image Klein Prompt Upsampler (PUP).

Supports:
- ``tonera/flux-3-image-klein-pup-nunchaku`` (Nunchaku SVDQ INT4 Qwen3.5-9B VL, runs natively via ``envs/extras`` without Ollama).
- ``unsloth/flux-3-image-klein-pup-GGUF`` (``UD-Q4_K_XL``, ``Q4_K_M``, ``Q8_0`` + ``mmproj-F16.gguf``, via Ollama or local GGUF cache).
- Standard Ollama vision models (``qwen2.5vl:3b``, ``qwen2.5vl:7b``, ``minicpm-v:8b``, ``llava:7b``).
"""
from __future__ import annotations

from pathlib import Path
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


NUNCHAKU_PUP_SUPPORT_FILES: List[str] = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "system_prompts.json",
]

VISION_MODELS_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "flux3-pup-nunchaku",
        "name": "FLUX.3 Prompt Upsampler (Nunchaku SVDQ INT4)",
        "provider": "nunchaku",
        "supported_providers": ["nunchaku"],
        "repo_id": "tonera/flux-3-image-klein-pup-nunchaku",
        "local_dir": "flux_3_pup_nunchaku",
        "primary_file": "svdq-int4-flux-3-image-klein-pup-nunchaku.safetensors",
        "support_files": NUNCHAKU_PUP_SUPPORT_FILES,
        "size_gb": 6.68,
        "vram_min_gb": 8,
        "recommended": True,
        "capabilities": ["vision", "flux3_pup", "t2i", "i2i"],
        "description": "Official FLUX.3 Klein Prompt Upsampler (Qwen3.5-9B VL) quantized with SVDQuant INT4 for Nunchaku. Runs natively without Ollama.",
    },
    {
        "id": "hf.co/unsloth/flux-3-image-klein-pup-GGUF:UD-Q4_K_XL",
        "name": "FLUX.3 Prompt Upsampler (Unsloth GGUF UD-Q4_K_XL)",
        "provider": "gguf",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/flux-3-image-klein-pup-GGUF",
        "local_dir": "flux_3_pup_gguf",
        "primary_file": "flux-3-image-klein-pup-UD-Q4_K_XL.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf", "system_prompts.json"],
        "size_gb": 6.41,
        "vram_min_gb": 8,
        "recommended": True,
        "capabilities": ["vision", "flux3_pup", "t2i", "i2i"],
        "description": "Unsloth Dynamic 4-bit (imatrix) GGUF + mmproj-F16 of FLUX 3 Image Klein Prompt Upsampler (~7.5 GB VRAM).",
    },
    {
        "id": "hf.co/unsloth/flux-3-image-klein-pup-GGUF:Q4_K_M",
        "name": "FLUX.3 Prompt Upsampler (Unsloth GGUF Q4_K_M)",
        "provider": "gguf",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/flux-3-image-klein-pup-GGUF",
        "local_dir": "flux_3_pup_gguf",
        "primary_file": "flux-3-image-klein-pup-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf", "system_prompts.json"],
        "size_gb": 6.10,
        "vram_min_gb": 8,
        "recommended": False,
        "capabilities": ["vision", "flux3_pup", "t2i", "i2i"],
        "description": "Compact Q4_K_M (imatrix) GGUF + mmproj-F16 of FLUX 3 Image Klein Prompt Upsampler.",
    },
    {
        "id": "hf.co/unsloth/flux-3-image-klein-pup-GGUF:Q8_0",
        "name": "FLUX.3 Prompt Upsampler (Unsloth GGUF Q8_0)",
        "provider": "gguf",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/flux-3-image-klein-pup-GGUF",
        "local_dir": "flux_3_pup_gguf",
        "primary_file": "flux-3-image-klein-pup-Q8_0.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf", "system_prompts.json"],
        "size_gb": 9.73,
        "vram_min_gb": 12,
        "recommended": False,
        "capabilities": ["vision", "flux3_pup", "t2i", "i2i"],
        "description": "High-precision Q8_0 GGUF + mmproj-F16 of FLUX 3 Image Klein Prompt Upsampler (near-lossless vs BF16).",
    },
    {
        "id": "qwen3-vl:4b",
        "name": "Qwen3-VL 4B (Новое поколение)",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "ollama_fallback_tag": "hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M",
        "repo_id": "unsloth/Qwen3-VL-4B-Instruct-GGUF",
        "local_dir": "qwen3_vl_gguf",
        "primary_file": "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf"],
        "size_gb": 3.1,
        "vram_min_gb": 6,
        "recommended": True,
        "capabilities": ["vision", "t2i", "i2i"],
        "description": "Новейшая мультимодальная модель Qwen3-VL 4B: точное понимание манги, SFX, анатомии и сцен при минимальном расходе VRAM (6-8 GB).",
    },
    {
        "id": "qwen3-vl:8b",
        "name": "Qwen3-VL 8B (Новое поколение)",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "ollama_fallback_tag": "hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
        "repo_id": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
        "local_dir": "qwen3_vl_gguf",
        "primary_file": "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf"],
        "size_gb": 5.2,
        "vram_min_gb": 8,
        "recommended": True,
        "capabilities": ["vision", "t2i", "i2i"],
        "description": "Флагманская компактная VLM Qwen3-VL 8B для глубокого анализа сложных фонов, ракурсов и инпейнта (8-12 GB VRAM).",
    },
    {
        "id": "gemma3:4b",
        "name": "Gemma 3 Vision 4B (Google)",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/gemma-3-4b-it-GGUF",
        "local_dir": "gemma3_vl_gguf",
        "primary_file": "gemma-3-4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf"],
        "size_gb": 3.3,
        "vram_min_gb": 6,
        "recommended": True,
        "capabilities": ["vision", "t2i", "i2i"],
        "description": "Современная мультимодальная модель Google Gemma 3 (4B) с высокой скоростью генерации промптов для FLUX / SD (6-8 GB VRAM).",
    },
    {
        "id": "gemma3:12b",
        "name": "Gemma 3 Vision 12B (Google)",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/gemma-3-12b-it-GGUF",
        "local_dir": "gemma3_vl_gguf",
        "primary_file": "gemma-3-12b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf"],
        "size_gb": 8.1,
        "vram_min_gb": 12,
        "recommended": False,
        "capabilities": ["vision", "t2i", "i2i"],
        "description": "Старшая версия Google Gemma 3 (12B) с детальным пониманием композиции, освещения и художественного стиля (12+ GB VRAM).",
    },
    {
        "id": "qwen2.5vl:3b",
        "name": "Qwen2.5-VL 3B",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
        "local_dir": "qwen2_5_vl_gguf",
        "primary_file": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf"],
        "size_gb": 3.2,
        "vram_min_gb": 6,
        "recommended": True,
        "capabilities": ["vision"],
        "description": "Быстрая и компактная VLM-модель для видеокарт с 6-8 GB VRAM.",
    },
    {
        "id": "qwen2.5vl:7b",
        "name": "Qwen2.5-VL 7B",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "local_dir": "qwen2_5_vl_gguf",
        "primary_file": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "support_files": ["mmproj-F16.gguf"],
        "size_gb": 6.0,
        "vram_min_gb": 12,
        "recommended": False,
        "capabilities": ["vision"],
        "description": "Higher detail manga/art comprehension for GPUs with 12+ GB VRAM.",
    },
    {
        "id": "minicpm-v:8b",
        "name": "MiniCPM-V 8B",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "openbmb/MiniCPM-V-2_6-gguf",
        "local_dir": "minicpm_v_gguf",
        "primary_file": "ggml-model-Q4_K_M.gguf",
        "mmproj_file": "mmproj-model-f16.gguf",
        "support_files": ["mmproj-model-f16.gguf"],
        "size_gb": 5.5,
        "vram_min_gb": 12,
        "recommended": False,
        "capabilities": ["vision"],
        "description": "Strong OCR and scene layout understanding for 12+ GB VRAM.",
    },
    {
        "id": "llava:7b",
        "name": "LLaVA 7B",
        "provider": "ollama",
        "supported_providers": ["ollama", "gguf"],
        "repo_id": "mys/ggml_llava-v1.5-7b",
        "local_dir": "llava_gguf",
        "primary_file": "ggml-model-q4_k.gguf",
        "mmproj_file": "mmproj-model-f16.gguf",
        "support_files": ["mmproj-model-f16.gguf"],
        "size_gb": 4.7,
        "vram_min_gb": 8,
        "recommended": False,
        "capabilities": ["vision"],
        "description": "Classic general-purpose vision-language model (fallback compatible).",
    },
]

_LOCK = threading.Lock()
_DOWNLOAD_STATE: Dict[str, Dict[str, Any]] = {}


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "cache"


def get_catalog_entry(model_id: str) -> Optional[Dict[str, Any]]:
    norm = (model_id or "").strip().lower()
    for item in VISION_MODELS_CATALOG:
        if item["id"].lower() == norm:
            return dict(item)
        if item.get("primary_file") and item["primary_file"].lower() == norm:
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
    entry = get_catalog_entry(catalog_id)
    alt_tag = str((entry or {}).get("ollama_fallback_tag") or "").strip().lower()
    for raw in installed_tags:
        r = (raw or "").strip().lower()
        if r == cid or _normalize_tag(r) == cid:
            return True
        if alt_tag and (r == alt_tag or _normalize_tag(r) == _normalize_tag(alt_tag)):
            return True
        if ":" not in cid and _normalize_tag(r) == cid_base:
            return True
    return False


def is_local_vision_model_installed(entry: Dict[str, Any], models_dir: Optional[Path] = None) -> bool:
    """Check whether a HuggingFace-backed vision/PUP model (Nunchaku or GGUF) is present on disk."""
    local_dir_name = entry.get("local_dir")
    primary_file = entry.get("primary_file")
    if not local_dir_name or not primary_file:
        return False
    base = (models_dir or _default_models_dir()).resolve()
    target_dir = base / local_dir_name
    primary_path = target_dir / primary_file
    if not primary_path.is_file() or primary_path.stat().st_size < 1024:
        return False
    mmproj = entry.get("mmproj_file")
    if mmproj and not (target_dir / mmproj).is_file():
        return False
    return True


def is_vision_model_installed(
    model_id: str,
    installed_tags: List[str],
    models_dir: Optional[Path] = None,
) -> bool:
    """True when ``model_id`` is installed either in Ollama or in the local HF model cache."""
    if is_tag_matching(model_id, installed_tags):
        return True
    entry = get_catalog_entry(model_id)
    if entry and is_local_vision_model_installed(entry, models_dir=models_dir):
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
    models_dir: Optional[Path] = None,
    nunchaku_available: bool = True,
) -> Dict[str, Any]:
    """Build the payload for ``GET /api/vision/models``."""
    models_out: List[Dict[str, Any]] = []
    for item in VISION_MODELS_CATALOG:
        mid = item["id"]
        installed_ollama = is_tag_matching(mid, installed_tags)
        installed_local = is_local_vision_model_installed(item, models_dir=models_dir)
        installed = bool(installed_ollama or installed_local)
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
            "provider": item.get("provider", "ollama"),
            "supported_providers": list(item.get("supported_providers") or [item.get("provider", "ollama")]),
            "repo_id": item.get("repo_id"),
            "local_dir": item.get("local_dir"),
            "primary_file": item.get("primary_file"),
            "size_gb": item["size_gb"],
            "vram_min_gb": item["vram_min_gb"],
            "recommended": item["recommended"],
            "installed": bool(installed),
            "installed_ollama": bool(installed_ollama),
            "installed_local": bool(installed_local),
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
        "available": bool(ollama_available or nunchaku_available),
        "ollama_available": bool(ollama_available),
        "nunchaku_available": bool(nunchaku_available),
        "models": models_out,
    }
