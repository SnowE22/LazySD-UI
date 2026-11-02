"""Orchestrator for Inpaint region prompt generation (``POST /api/inpaint/prompt``).

Combines:
1. Strict source-path validation (``uploads/``, ``outputs/``, ``cache/`` only) and
   mask decoding via ``backend/masking.py``.
2. Image preparation for VLM:
   - ``overview``: full page resized to ``max_image_dim`` (default 1024px) with a
     semi-transparent red mask overlay (alpha ~0.40).
   - ``crop``: region around the mask bbox (respecting ``crop_mode`` / ``crop_padding`` /
     ``crop_box``) resized to ``max_image_dim`` without overlay.
3. VLM call via ``backend/vision_runtime.py`` (Ollama) with automatic fallback to
   ``backend/prompt_fallback.py`` when VLM is disabled, uninstalled, or unreachable.
4. In-memory LRU cache (50 entries) keyed by SHA-1 of source path + mtime + mask hash +
   instruction + model + crop parameters.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import io
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .masking import (
    decode_mask_to_float,
    mask_bbox,
    normalize_crop_mode,
    prepare_mask_arrays,
    resolve_crop_box,
)
from .prompt_fallback import generate_heuristic_prompt
from .vision_models import get_catalog_entry, is_local_vision_model_installed, is_tag_matching
from .vision_runtime import (
    VisionRuntimeError,
    VisionTimeoutError,
    check_ollama_status,
    generate_vlm_prompt,
)

logger = logging.getLogger("quantfunc.prompt_gen")

MAX_CACHE_ENTRIES = 50
MAX_SOURCE_PIXELS = 64_000_000  # 64 MP safety limit (e.g. 8000x8000)
MAX_ENCODED_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB per image payload

_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


class PromptGenValidationError(ValueError):
    """Raised for HTTP 400 validation errors (empty mask, bad path, huge image)."""


def clear_prompt_cache() -> None:
    with _CACHE_LOCK:
        _PROMPT_CACHE.clear()


def resolve_safe_source_path(
    source_path: str,
    *,
    project_root: Optional[Path] = None,
    allowed_dirs: Optional[List[Path]] = None,
) -> Path:
    """Resolve ``source_path`` and verify it resides inside ``uploads/``, ``outputs/``, or ``cache/``."""
    raw = (source_path or "").strip()
    if not raw:
        raise PromptGenValidationError("source_path is required")

    # Strip leading URL prefix or query string if frontend passed "/outputs/foo.png?t=..."
    clean = raw.split("?", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if clean.startswith("/"):
        clean = clean.lstrip("/")

    if ".." in clean.split("/"):
        raise PromptGenValidationError("Invalid source_path: path traversal is not allowed")

    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    roots = allowed_dirs or [
        (root / "uploads").resolve(),
        (root / "outputs").resolve(),
        (root / "cache").resolve(),
    ]

    candidate = Path(clean)
    resolved_candidates: List[Path] = []
    if not candidate.is_absolute():
        resolved_candidates.append((root / candidate).resolve())
        for rdir in roots:
            # Uploads are stored under <output_dir>/uploads and referenced as "uploads/<name>".
            resolved_candidates.append((rdir / candidate).resolve())
            resolved_candidates.append((rdir / candidate.name).resolve())
            resolved_candidates.append((rdir / "uploads" / candidate.name).resolve())
    else:
        resolved_candidates.append(candidate.resolve())

    for cand in resolved_candidates:
        for allowed in roots:
            try:
                cand.relative_to(allowed)
            except ValueError:
                continue
            if cand.is_file():
                return cand

    raise PromptGenValidationError(
        f"Source image not found or outside allowed directories (uploads/, outputs/, cache/): {source_path}"
    )


def _resize_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    max_dim = max(128, int(max_dim or 1024))
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img.copy()
    scale = float(max_dim) / float(max(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _encode_image_b64(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=88)
    data = buf.getvalue()
    if len(data) > MAX_ENCODED_IMAGE_BYTES:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=75)
        data = buf.getvalue()
    if len(data) > MAX_ENCODED_IMAGE_BYTES:
        raise PromptGenValidationError("Encoded image exceeds 8 MB payload limit")
    return base64.b64encode(data).decode("ascii")


def _apply_red_mask_highlight(
    img: Image.Image,
    mask_pil_full: Image.Image,
    *,
    alpha_strength: float = 0.44,
    dim_outside: float = 0.0,
) -> Image.Image:
    """Overlay bright red highlight on masked pixels and optionally dim unmasked pixels."""
    mask_resized = mask_pil_full.resize(img.size, Image.Resampling.BILINEAR)
    mask_2d = np.asarray(mask_resized, dtype=np.float32) / 255.0
    mask_norm = mask_2d[:, :, None]

    arr = np.asarray(img, dtype=np.float32)
    if dim_outside > 0.0:
        outside_factor = 1.0 - dim_outside * (1.0 - mask_norm)
        arr = arr * outside_factor

    red_layer = np.zeros_like(arr)
    red_layer[:, :, 0] = 245.0
    red_layer[:, :, 1] = 25.0
    red_layer[:, :, 2] = 40.0
    alpha = mask_norm * alpha_strength
    blended = np.clip(arr * (1.0 - alpha) + red_layer * alpha, 0.0, 255.0)

    # Draw a crisp bright red/yellow edge contour around the mask so VLM distinguishes mask vs character
    binary = mask_2d >= 0.25
    if binary.any():
        padded = np.pad(binary, 1, mode="edge")
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        edge = binary & (~eroded)
        blended[edge, 0] = 255.0
        blended[edge, 1] = 60.0
        blended[edge, 2] = 20.0

    return Image.fromarray(blended.astype(np.uint8), mode="RGB")


def build_vlm_images(
    source_image: Image.Image,
    raw_mask: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    max_dim: int = 640,
) -> Tuple[str, str]:
    """Build ``(overview_b64, crop_b64)`` for VLM consumption (capped to 560px for fast ViT prefill)."""
    vlm_dim = min(max(256, int(max_dim or 560)), 560)
    rgb = source_image.convert("RGB")
    mask_u8 = (np.clip(raw_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_full_pil = Image.fromarray(mask_u8, mode="L")

    over_img = _resize_max_dim(rgb, vlm_dim)
    overview_composited = _apply_red_mask_highlight(
        over_img,
        mask_full_pil,
        alpha_strength=0.42,
        dim_outside=0.0,
    )

    x0, y0, x1, y1 = (int(v) for v in crop_box)
    crop_rgb = _resize_max_dim(rgb.crop((x0, y0, x1, y1)), vlm_dim)
    crop_mask_pil = mask_full_pil.crop((x0, y0, x1, y1))
    # Highlight the masked pixels in the close-up crop too so VLM never confuses an unmasked character
    # inside the crop bounding box with the masked SFX/background around them!
    crop_composited = _apply_red_mask_highlight(
        crop_rgb,
        crop_mask_pil,
        alpha_strength=0.45,
        dim_outside=0.18,
    )

    return _encode_image_b64(overview_composited), _encode_image_b64(crop_composited)


def _compute_cache_key(
    *,
    resolved_source: Path,
    mask_array: np.ndarray,
    instruction: str,
    vision_model: str,
    crop_mode: str,
    crop_box: Tuple[int, int, int, int],
    crop_padding: int,
    use_vlm: bool,
) -> str:
    try:
        mtime = f"{resolved_source.stat().st_mtime_ns}"
    except Exception:
        mtime = "0"
    # Downsample mask to 64x64 for fast deterministic hashing
    mask_u8 = (np.clip(mask_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    small_mask = Image.fromarray(mask_u8, mode="L").resize((64, 64), Image.Resampling.NEAREST)
    mask_hash = hashlib.sha1(np.asarray(small_mask, dtype=np.uint8).tobytes()).hexdigest()

    raw_key = "|".join([
        str(resolved_source),
        mtime,
        mask_hash,
        (instruction or "").strip(),
        (vision_model or "").strip(),
        crop_mode,
        f"{crop_box[0]},{crop_box[1]},{crop_box[2]},{crop_box[3]}",
        str(crop_padding),
        "vlm" if use_vlm else "heur",
    ])
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def generate_inpaint_prompt(
    *,
    source_path: str,
    mask: Any,
    instruction: str = "",
    crop_mode: str = "auto",
    crop_box: Any = None,
    crop_padding: int = 64,
    vision_model: Optional[str] = None,
    vision_provider: Optional[str] = None,
    use_vlm: bool = True,
    settings: Any = None,
    project_root: Optional[Path] = None,
    allowed_dirs: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Execute the full Inpaint prompt generation pipeline."""
    instruction_clean = (instruction or "").strip()
    if len(instruction_clean) > 500:
        raise PromptGenValidationError("instruction exceeds maximum length of 500 characters")

    if not mask or (isinstance(mask, str) and not mask.strip()):
        raise PromptGenValidationError("mask is required and cannot be empty")

    resolved_source = resolve_safe_source_path(
        source_path,
        project_root=project_root,
        allowed_dirs=allowed_dirs,
    )

    try:
        with Image.open(resolved_source) as src_file:
            width, height = src_file.size
            if width * height > MAX_SOURCE_PIXELS:
                raise PromptGenValidationError(
                    f"Source image is too large ({width}x{height} exceeds 64 MP limit)"
                )
            source_image = src_file.convert("RGB")
    except PromptGenValidationError:
        raise
    except Exception as exc:
        raise PromptGenValidationError(f"Could not open source image: {exc}") from exc

    try:
        raw_mask, engine_mask = prepare_mask_arrays(mask, target_size=(width, height))
    except Exception as exc:
        raise PromptGenValidationError(f"Invalid mask: {exc}") from exc

    bbox = mask_bbox(raw_mask)
    if bbox is None:
        raise PromptGenValidationError("Mask is empty — draw over the area you want to edit first")

    norm_mode = normalize_crop_mode(crop_mode)
    try:
        resolved_box = resolve_crop_box(
            engine_mask,
            mode=norm_mode,
            padding=int(crop_padding if crop_padding is not None else 64),
            align=16,
            min_size=256,
            manual_box=crop_box,
            coverage_mask=raw_mask,
        )
    except Exception as exc:
        raise PromptGenValidationError(str(exc)) from exc

    vision_enabled = bool(getattr(settings, "vision_enabled", True)) if settings is not None else True
    fallback_enabled = bool(getattr(settings, "vision_fallback_enabled", True)) if settings is not None else True
    ollama_url = str(getattr(settings, "vision_ollama_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434")
    keep_alive = str(getattr(settings, "vision_keep_alive", "0s") or "0s")
    timeout_sec = int(getattr(settings, "vision_timeout_sec", 120) or 120)
    max_image_dim = int(getattr(settings, "vision_max_image_dim", 1024) or 1024)
    selected_model = (
        (vision_model or "").strip()
        or str(getattr(settings, "vision_model", "qwen2.5vl:3b") or "qwen2.5vl:3b")
    )

    req_provider = (vision_provider or getattr(settings, "vision_provider", "auto") or "auto").strip().lower()
    effective_use_vlm = bool(use_vlm and vision_enabled and req_provider != "heuristic")
    cache_key = _compute_cache_key(
        resolved_source=resolved_source,
        mask_array=raw_mask,
        instruction=instruction_clean,
        vision_model=selected_model,
        crop_mode=norm_mode,
        crop_box=resolved_box,
        crop_padding=int(crop_padding or 64),
        use_vlm=effective_use_vlm,
    )

    with _CACHE_LOCK:
        cached_item = _PROMPT_CACHE.get(cache_key)
        if cached_item is not None:
            _PROMPT_CACHE.move_to_end(cache_key)
            out = dict(cached_item)
            out["cached"] = True
            return out

    # Always compute heuristic analysis (provides dominant_colors, region_type, mask_box, and fallback)
    heur = generate_heuristic_prompt(
        source_image=source_image,
        mask_array=raw_mask,
        mask_box=bbox,
        instruction=instruction_clean,
    )

    models_dir = getattr(settings, "models_dir", None) if settings is not None else None
    cat_entry = get_catalog_entry(selected_model)
    is_nunchaku_pup = bool(cat_entry and cat_entry.get("provider") == "nunchaku")

    warnings: List[str] = []
    if effective_use_vlm:
        can_run_vlm = False
        if is_nunchaku_pup:
            if is_local_vision_model_installed(cat_entry, models_dir=models_dir):
                can_run_vlm = True
            else:
                if not fallback_enabled:
                    raise VisionRuntimeError(
                        f"Vision model '{selected_model}' is not downloaded yet.",
                        status_code=409,
                    )
                warnings.append(f"Model '{selected_model}' is not downloaded yet; used local heuristic fallback.")
        else:
            status_info = check_ollama_status(ollama_url)
            if not status_info["ollama_available"]:
                if not fallback_enabled:
                    raise VisionRuntimeError(
                        f"Ollama is not reachable at {ollama_url}. Start Ollama or enable heuristic fallback.",
                        status_code=503,
                    )
                warnings.append("Ollama is not running; used local heuristic fallback.")
            elif is_tag_matching(selected_model, status_info["installed_tags"]):
                can_run_vlm = True
            elif cat_entry and is_local_vision_model_installed(cat_entry, models_dir=models_dir) and status_info["installed_tags"]:
                # Local GGUF is downloaded in cache and Ollama is running — use best available Ollama VLM tag
                fallback_tag = next(
                    (t for t in status_info["installed_tags"] if any(k in t.lower() for k in ("qwen", "flux-3", "minicpm", "llava"))),
                    status_info["installed_tags"][0],
                )
                warnings.append(
                    f"Local GGUF '{selected_model}' is in HF cache; executed via active Ollama VLM '{fallback_tag}'."
                )
                selected_model = fallback_tag
                can_run_vlm = True
            else:
                if not fallback_enabled:
                    raise VisionRuntimeError(
                        f"Vision model '{selected_model}' is not installed in Ollama.",
                        status_code=409,
                    )
                warnings.append(f"Vision model '{selected_model}' is not installed; used local heuristic fallback.")

        if can_run_vlm:
            try:
                overview_b64, crop_b64 = build_vlm_images(
                    source_image,
                    raw_mask,
                    resolved_box,
                    max_dim=max_image_dim,
                )
                vlm_prompt, vlm_neg = generate_vlm_prompt(
                    ollama_url=ollama_url,
                    model_id=selected_model,
                    instruction=instruction_clean,
                    overview_b64=overview_b64,
                    crop_b64=crop_b64,
                    analysis_hint=heur["analysis"],
                    timeout_sec=timeout_sec,
                    keep_alive=keep_alive,
                    models_dir=models_dir,
                )
                result = {
                    "prompt": vlm_prompt,
                    "negative_prompt": vlm_neg,
                    "source": "vlm",
                    "model": selected_model,
                    "warnings": warnings,
                    "analysis": heur["analysis"],
                    "cached": False,
                }
                with _CACHE_LOCK:
                    _PROMPT_CACHE[cache_key] = dict(result)
                    while len(_PROMPT_CACHE) > MAX_CACHE_ENTRIES:
                        _PROMPT_CACHE.popitem(last=False)
                return result
            except VisionTimeoutError as exc:
                if not fallback_enabled:
                    raise
                warnings.append(f"Vision model timed out ({exc}); used local heuristic fallback.")
            except VisionRuntimeError as exc:
                if not fallback_enabled:
                    raise
                warnings.append(f"Vision model error ({exc}); used local heuristic fallback.")
            except Exception as exc:
                if not fallback_enabled:
                    raise VisionRuntimeError(f"Vision model failed: {exc}", status_code=503) from exc
                warnings.append(f"Vision model response error ({exc}); used local heuristic fallback.")

    result = {
        "prompt": heur["prompt"],
        "negative_prompt": heur["negative_prompt"],
        "source": "heuristic",
        "model": None,
        "warnings": warnings,
        "analysis": heur["analysis"],
        "cached": False,
    }
    # Only cache heuristic results when heuristic mode was explicitly requested (not on VLM error fallback)
    if not effective_use_vlm:
        with _CACHE_LOCK:
            _PROMPT_CACHE[cache_key] = dict(result)
            while len(_PROMPT_CACHE) > MAX_CACHE_ENTRIES:
                _PROMPT_CACHE.popitem(last=False)
    return result
