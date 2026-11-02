"""Mask (inpaint) primitives shared by the Generate quick-mask and the Inpaint studio.

MASK SEMANTICS (single source of truth for the whole project)
-------------------------------------------------------------
    white  (1.0) = the region that MUST be regenerated (edited)
    black  (0.0) = the region that MUST stay exactly as in the original
    grey         = soft transition (alpha blend between the two)

The browser paints white strokes on a transparent RGBA canvas, so the exported
PNG has "alpha == coverage" semantics.  When decoding we therefore combine the
luminance and the alpha channel (``max(lum, alpha)``) — ``Image.convert("L")``
alone would silently drop the alpha channel and report an empty mask.

GUARANTEE
---------
Whatever the engine does, the final image is assembled on the backend:

    result = generated * mask + original * (1 - mask)

Pixels outside the mask are therefore bit-identical to the original image
(the mask is exactly 0 there), which is what the specification demands:
the guarantee must come from compositing, not from asking the model nicely.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

# Maximum size of a decoded mask payload (25 MB, same limit as /api/upload).
MAX_MASK_BYTES = 25 * 1024 * 1024

# Maximum accepted mask resolution (matches validate_resolution upper bound).
MAX_MASK_DIMENSION = 4096

MaskSource = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray, None]


class MaskError(ValueError):
    """Raised when a mask payload cannot be decoded."""


# ---------- Engine capability table ----------
# Every entry uses the SAME polarity (white = edit, black = preserve):
#
#   quantfunc  : --mask-image PATH
#                "Inpaint mask (PNG/JPG; alpha or grayscale; white=inpaint,
#                 black=preserve)"  -> polarity confirmed by the engine --help.
#   sd_cpp     : --mask PATH
#                Upstream src/pipeline/image.cpp builds the denoise mask from
#                the image (1 = denoise/regenerate, 0 = keep init latent), so
#                white = edit, black = preserve.  The mask is only honoured
#                together with -i/--init-img.
#   nunchaku   : Python engine (no CLI flag).  Flux2KleinInpaintPipeline
#                receives mask_image (white repainted, black preserved).
#   orbitquant : same Python contract as nunchaku.
ENGINE_MASK_PARAMS: dict[str, dict] = {
    "quantfunc": {"arg": "--mask-image", "white_is_edit": True, "supports": True},
    "sd_cpp": {"arg": "--mask", "white_is_edit": True, "supports": True},
    "nunchaku": {"arg": None, "white_is_edit": True, "supports": True},
    "orbitquant": {"arg": None, "white_is_edit": True, "supports": True},
}


def engine_mask_support(engine: Optional[str]) -> bool:
    """True when the given engine can consume an inpaint mask."""
    if not engine:
        return False
    params = ENGINE_MASK_PARAMS.get(str(engine))
    return bool(params and params.get("supports"))


def engine_mask_arg(engine: Optional[str]) -> Optional[str]:
    """CLI flag used to pass a mask to the given engine (None if unsupported).

    Python engines (nunchaku/orbitquant) support masks but have no CLI flag, so
    ``None`` is a valid answer even when :func:`engine_mask_support` is True.
    """
    if not engine:
        return None
    params = ENGINE_MASK_PARAMS.get(str(engine))
    if not params or not params.get("supports"):
        return None
    arg = params.get("arg")
    return str(arg) if arg else None


def _decode_data_url(data_url: str) -> bytes:
    """Extract the raw bytes of a ``data:image/...;base64,`` URL."""
    if not isinstance(data_url, str):
        raise MaskError("Mask must be a base64 data URL")
    header, separator, payload = data_url.partition(",")
    if (
        not separator
        or header not in {
            "data:image/png;base64",
            "data:image/jpeg;base64",
            "data:image/webp;base64",
        }
        or not payload
    ):
        raise MaskError("Mask must be a PNG, JPEG, or WebP base64 data URL")
    if len(payload) > MAX_MASK_BYTES * 4 // 3 + 8:
        raise MaskError("Mask payload exceeds the size limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MaskError(f"Mask payload is not valid base64: {exc}") from exc
    if len(raw) > MAX_MASK_BYTES:
        raise MaskError(
            f"Mask payload too large ({len(raw)} bytes, limit {MAX_MASK_BYTES})"
        )
    return raw


def _open_mask_image(mask_source: MaskSource) -> Image.Image:
    """Open any supported mask source into a PIL image (never closes the source)."""
    if mask_source is None:
        raise MaskError("Mask source is empty")

    if isinstance(mask_source, np.ndarray):
        arr = mask_source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise MaskError(f"Unsupported mask array shape: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        if arr.shape[2] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2
            )
        return Image.fromarray(arr, "RGBA")

    if isinstance(mask_source, Image.Image):
        return mask_source.copy()

    if isinstance(mask_source, (bytes, bytearray)):
        raw = bytes(mask_source)
        if len(raw) > MAX_MASK_BYTES:
            raise MaskError("Mask payload too large")
        try:
            return Image.open(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - surfaced as a user-facing error
            raise MaskError(f"Cannot read mask image: {exc}") from exc

    if isinstance(mask_source, (str, Path)):
        text = str(mask_source)
        if text.startswith("data:image"):
            raw = _decode_data_url(text)
            try:
                return Image.open(io.BytesIO(raw))
            except Exception as exc:  # noqa: BLE001
                raise MaskError(f"Cannot read mask image: {exc}") from exc
        path = Path(text)
        if not path.is_file():
            raise MaskError(f"Mask file not found: {path}")
        if path.stat().st_size > MAX_MASK_BYTES:
            raise MaskError("Mask file too large")
        try:
            return Image.open(path)
        except Exception as exc:  # noqa: BLE001
            raise MaskError(f"Cannot read mask image: {exc}") from exc

    raise MaskError(f"Unsupported mask type: {type(mask_source)}")


def decode_mask_to_float(
    mask_source: MaskSource,
    target_size: Optional[Tuple[int, int]] = None,
    invert_mask: bool = False,
) -> Optional[np.ndarray]:
    """Decode a mask into a float32 ``[H, W]`` array with values in ``[0..1]``.

    ``target_size`` is ``(width, height)``; when given, the mask is resampled
    to that size (bilinear, so soft edges stay soft).
    """
    if mask_source is None or (isinstance(mask_source, str) and not mask_source):
        return None

    image = _open_mask_image(mask_source)
    try:
        width, height = image.size
        if width > MAX_MASK_DIMENSION or height > MAX_MASK_DIMENSION:
            raise MaskError(
                f"Mask resolution {width}x{height} exceeds the limit "
                f"of {MAX_MASK_DIMENSION}px"
            )
        image.load()
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            arr = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
            rgb_lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            alpha_chan = arr[:, :, 3]
            m_f = np.maximum(rgb_lum, alpha_chan)
        else:
            m_f = np.array(image.convert("L"), dtype=np.float32) / 255.0

        if target_size is not None:
            target_w, target_h = int(target_size[0]), int(target_size[1])
            if m_f.shape[:2] != (target_h, target_w):
                resized = Image.fromarray(
                    np.clip(m_f * 255.0, 0, 255).astype(np.uint8), "L"
                ).resize((target_w, target_h), Image.BILINEAR)
                m_f = np.array(resized, dtype=np.float32) / 255.0

        m_f = np.clip(m_f, 0.0, 1.0)
        if invert_mask:
            m_f = 1.0 - m_f
        return m_f
    except (OSError, ValueError) as exc:
        if isinstance(exc, MaskError):
            raise
        raise MaskError(f"Cannot decode mask image: {exc}") from exc
    finally:
        image.close()


def feather_mask(
    mask: np.ndarray,
    grow: int = 0,
    blur: float = 0.0,
) -> np.ndarray:
    """Dilate and/or blur a float mask while keeping values in ``[0..1]``.

    ``grow`` is a radius in pixels (square dilation, same spirit as the
    quantfunc ``--mask-grow`` flag); ``blur`` is a Gaussian sigma in pixels.
    Engines without native grow/blur arguments (sd.cpp, diffusers inpaint)
    receive the result of this helper instead.
    """
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 2 or arr.size == 0:
        return arr

    grow = max(0, min(128, int(grow or 0)))
    blur = max(0.0, min(64.0, float(blur or 0.0)))

    if grow > 0:
        kernel = grow * 2 + 1
        image = Image.fromarray((arr * 255.0).astype(np.uint8), "L")
        image = image.filter(ImageFilter.MaxFilter(kernel))
        arr = np.asarray(image, dtype=np.float32) / 255.0

    if blur > 0.0:
        image = Image.fromarray((arr * 255.0).astype(np.uint8), "L")
        image = image.filter(ImageFilter.GaussianBlur(blur))
        arr = np.asarray(image, dtype=np.float32) / 255.0

    return np.clip(arr, 0.0, 1.0)


def prepare_mask_arrays(
    mask_source: MaskSource,
    target_size: Tuple[int, int],
    invert_mask: bool = False,
    grow: int = 0,
    blur: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a mask and return ``(raw, engine)`` arrays in source coordinates.

    ``raw`` is used for the backend compositing guarantee; ``engine`` is the
    dilated/blurred version handed to the engine.
    """
    raw = decode_mask_to_float(
        mask_source,
        target_size=target_size,
        invert_mask=invert_mask,
    )
    if raw is None:
        raise MaskError("Mask payload is empty")
    return raw, feather_mask(raw, grow=grow, blur=blur)


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Encode a float ``[H, W]`` mask (0..1) into PNG bytes (grayscale)."""
    arr = np.clip(np.asarray(mask, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, format="PNG")
    return buf.getvalue()


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame covered by the mask (0..1)."""
    arr = np.asarray(mask, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def is_mask_empty(mask: Optional[np.ndarray], threshold: float = 0.004) -> bool:
    """True when the mask paints (almost) nothing."""
    if mask is None:
        return True
    return mask_coverage(mask) <= threshold


def composite_with_mask(
    generated: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    """Compose ``generated`` over ``original`` using the mask.

    ``result = generated * mask + original * (1 - mask)``

    The composite is produced at the ORIGINAL image resolution so that every
    pixel outside the mask stays bit-identical to the source image.
    """
    orig_w, orig_h = original.size
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.shape[:2] != (orig_h, orig_w):
        mask_arr = decode_mask_to_float(mask_arr, target_size=(orig_w, orig_h))
        if mask_arr is None:  # pragma: no cover - defensive
            mask_arr = np.zeros((orig_h, orig_w), dtype=np.float32)

    gen_rgb = generated.convert("RGB").resize((orig_w, orig_h), Image.LANCZOS)
    orig_rgba = original.convert("RGBA")
    has_alpha = "A" in original.getbands() or "transparency" in original.info

    gen_arr = np.asarray(gen_rgb, dtype=np.float32)
    orig_arr = np.asarray(orig_rgba, dtype=np.float32)[:, :, :3]

    m3 = mask_arr[:, :, None]
    out = gen_arr * m3 + orig_arr * (1.0 - m3)
    out_rgb = np.clip(out, 0, 255).astype(np.uint8)
    if not has_alpha:
        return Image.fromarray(out_rgb, "RGB")
    out_rgba = np.dstack((out_rgb, np.asarray(orig_rgba)[:, :, 3]))
    return Image.fromarray(out_rgba, "RGBA")


# ---------- Crop-to-mask (real compute savings) ----------
# Engines that "support" masks often still denoise the FULL frame (SD.CPP and
# diffusers inpaint pipelines resample the whole latent), so a 1% mask costs as
# much as a 100% one.  To get actual savings the source and mask are cropped to
# the mask bounding box (plus padding for context), the engine runs on that
# small crop, and the result is pasted back into the original frame.
MASK_CROP_PADDING = 64
MASK_CROP_ALIGN = 16
MASK_CROP_MIN_SIZE = 256

# Crop modes exposed by the Inpaint studio:
#   auto       - mask bbox + default padding/min size (fully automatic)
#   parameters - mask bbox + user supplied padding/min size
#   manual     - exact rectangle selected by the user on the canvas
MASK_CROP_MODES = ("auto", "parameters", "manual")
MASK_CROP_MANUAL_MIN_SIZE = 64

# User editable ranges of the "parameters" mode (see the specification).
MASK_CROP_PADDING_RANGE = (0, 1024)
MASK_CROP_MIN_SIZE_RANGE = (64, 4096)

# Threshold that decides whether a pixel belongs to the painted mask.
MASK_CROP_COVERAGE_THRESHOLD = 0.004


def normalize_crop_mode(value: Optional[str]) -> str:
    """Validate/normalize a crop mode name."""
    mode = str(value or "auto").strip().lower()
    if mode not in MASK_CROP_MODES:
        raise ValueError(
            "mask_crop_mode must be one of: " + ", ".join(MASK_CROP_MODES)
        )
    return mode


def _coerce_manual_crop_box(box) -> Tuple[int, int, int, int]:
    """Accept ``[x0, y0, x1, y1]`` (or an equivalent mapping)."""
    if isinstance(box, dict):
        values = [box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1")]
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        values = list(box)
    else:
        raise ValueError("mask_crop_box must be [x0, y0, x1, y1]")
    if any(value is None for value in values):
        raise ValueError("mask_crop_box must contain four coordinates")
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("mask_crop_box coordinates must be numbers") from exc
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("mask_crop_box coordinates must be finite numbers")
    x0, y0, x1, y1 = (int(round(number)) for number in numbers)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def resolve_manual_crop_box(
    manual_box,
    width: int,
    height: int,
    align: int = MASK_CROP_ALIGN,
    min_size: int = MASK_CROP_MANUAL_MIN_SIZE,
) -> Tuple[int, int, int, int]:
    """Normalize a user drawn rectangle into a valid engine crop box.

    The rectangle is expressed in the coordinate system of the ORIGINAL image
    (never in CSS pixels of the browser).  It is clamped to the image bounds,
    aligned outward to the engine grid (16 px) and finally checked against the
    minimal side length (64 px).
    """
    width = max(1, int(width))
    height = max(1, int(height))
    x0, y0, x1, y1 = _coerce_manual_crop_box(manual_box)
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("mask_crop_box is outside the image")

    align = max(1, int(align))
    x0 = (x0 // align) * align
    y0 = (y0 // align) * align
    x1 = min(width, ((x1 + align - 1) // align) * align)
    y1 = min(height, ((y1 + align - 1) // align) * align)

    min_size = max(1, int(min_size))
    if (x1 - x0) < min_size or (y1 - y0) < min_size:
        raise ValueError(
            f"mask_crop_box is too small (minimum {min_size}px per side)"
        )
    return int(x0), int(y0), int(x1), int(y1)


def resolve_crop_box(
    mask: np.ndarray,
    mode: str = "auto",
    padding: int = MASK_CROP_PADDING,
    align: int = MASK_CROP_ALIGN,
    min_size: int = MASK_CROP_MIN_SIZE,
    manual_box=None,
    coverage_mask: Optional[np.ndarray] = None,
    max_aspect_ratio: float = 2.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Resolve the engine crop box for the requested crop mode.

    ``auto`` ignores the supplied padding/min size and uses the project
    defaults.  ``parameters`` honours the supplied values.  ``manual`` uses the
    user rectangle, aligned outward to the engine grid and clamped to the
    image.

    ``coverage_mask`` is the RAW (painted, not grown/blurred) mask.  It is the
    reference used by the manual mode to guarantee that the crop covers the
    whole painted mask - a manual rectangle that cuts the mask would silently
    regenerate only a part of it.  When it is omitted the ``mask`` argument is
    used instead.
    """
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None
    height, width = arr.shape[:2]
    mode = normalize_crop_mode(mode)

    if mode == "manual":
        if manual_box is None:
            raise ValueError(
                "mask_crop_box is required when mask_crop_mode is 'manual'"
            )
        x0, y0, x1, y1 = resolve_manual_crop_box(
            manual_box,
            width,
            height,
            align=align,
            min_size=MASK_CROP_MANUAL_MIN_SIZE,
        )
        coverage = (
            arr if coverage_mask is None
            else np.asarray(coverage_mask, dtype=np.float32)
        )
        if coverage.ndim != 2 or coverage.shape != arr.shape:
            raise ValueError(
                "Mask size does not match the source image "
                f"({coverage.shape} != {arr.shape})"
            )
        bbox = mask_bbox(coverage, threshold=MASK_CROP_COVERAGE_THRESHOLD)
        if bbox is None:
            raise ValueError("Mask is empty - paint the area to edit")
        bx0, by0, bx1, by1 = bbox
        if bx0 < x0 or by0 < y0 or bx1 > x1 or by1 > y1:
            raise ValueError(
                "The manual crop box must cover the whole mask "
                f"(mask x {bx0}-{bx1}, y {by0}-{by1}; "
                f"crop x {x0}-{x1}, y {y0}-{y1})"
            )
        return int(x0), int(y0), int(x1), int(y1)

    if mode == "auto":
        padding = MASK_CROP_PADDING
        min_size = MASK_CROP_MIN_SIZE
    else:
        padding = max(
            MASK_CROP_PADDING_RANGE[0],
            min(MASK_CROP_PADDING_RANGE[1], int(padding)),
        )
        min_size = max(
            MASK_CROP_MIN_SIZE_RANGE[0],
            min(MASK_CROP_MIN_SIZE_RANGE[1], int(min_size)),
        )

    return compute_crop_box(
        arr,
        padding=padding,
        align=align,
        min_size=min_size,
        max_aspect_ratio=max_aspect_ratio,
    )


def mask_bbox(
    mask: np.ndarray,
    threshold: float = 0.004,
) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` (exclusive) of the painted area, or None."""
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None
    ys, xs = np.nonzero(arr > float(threshold))
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def compute_crop_box(
    mask: np.ndarray,
    padding: int = MASK_CROP_PADDING,
    align: int = MASK_CROP_ALIGN,
    min_size: int = MASK_CROP_MIN_SIZE,
    max_aspect_ratio: float = 2.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute the engine crop box for a mask.

    The box is the mask bounding box expanded by ``padding``, rounded outward
    to the ``align`` grid (engines require multiples of 8/16), clamped to the
    image, and grown to at least ``min_size`` so tiny strokes do not produce
    degenerate latents. If ``max_aspect_ratio`` is set (> 0), extreme thin or
    tall boxes are expanded symmetrically outward to stay within the ratio
    bound. Returns ``None`` for an empty mask.
    """
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None
    height, width = arr.shape[:2]
    bbox = mask_bbox(arr)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    pad = max(0, int(padding))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)

    align = max(1, int(align))
    x0 = (x0 // align) * align
    y0 = (y0 // align) * align
    x1 = min(width, ((x1 + align - 1) // align) * align)
    y1 = min(height, ((y1 + align - 1) // align) * align)

    min_size = max(1, int(min_size))
    if x1 - x0 < min_size:
        extra = min_size - (x1 - x0)
        grow_left = min(x0, extra // 2)
        x0 -= grow_left
        x1 = min(width, x1 + (extra - grow_left))
        if x1 - x0 < min_size:
            x0 = max(0, x1 - min_size)
    if y1 - y0 < min_size:
        extra = min_size - (y1 - y0)
        grow_top = min(y0, extra // 2)
        y0 -= grow_top
        y1 = min(height, y1 + (extra - grow_top))
        if y1 - y0 < min_size:
            y0 = max(0, y1 - min_size)

    if max_aspect_ratio and max_aspect_ratio > 0:
        box_w = x1 - x0
        box_h = y1 - y0
        if box_w > 0 and box_h > 0:
            if box_w > box_h * max_aspect_ratio:
                target_h = min(height, int(math.ceil(box_w / float(max_aspect_ratio))))
                if target_h > box_h:
                    extra = target_h - box_h
                    grow_top = min(y0, extra // 2)
                    y0 -= grow_top
                    y1 = min(height, y1 + (extra - grow_top))
                    if y1 - y0 < target_h:
                        y0 = max(0, y1 - target_h)
            elif box_h > box_w * max_aspect_ratio:
                target_w = min(width, int(math.ceil(box_h / float(max_aspect_ratio))))
                if target_w > box_w:
                    extra = target_w - box_w
                    grow_left = min(x0, extra // 2)
                    x0 -= grow_left
                    x1 = min(width, x1 + (extra - grow_left))
                    if x1 - x0 < target_w:
                        x0 = max(0, x1 - target_w)

            # Re-align outward
            x0 = (x0 // align) * align
            y0 = (y0 // align) * align
            x1 = min(width, ((x1 + align - 1) // align) * align)
            y1 = min(height, ((y1 + align - 1) // align) * align)

    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def crop_array(mask: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop a 2D float mask to ``box`` (returns a contiguous copy)."""
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(np.asarray(mask, dtype=np.float32)[y0:y1, x0:x1])


def crop_image(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop a PIL image to ``box``."""
    return image.crop(tuple(int(v) for v in box))


def paste_crop_into_image(
    generated: Image.Image,
    original: Image.Image,
    box: Tuple[int, int, int, int],
) -> Image.Image:
    """Paste a generated crop back into a copy of the original frame."""
    x0, y0, x1, y1 = (int(v) for v in box)
    target = original.copy()
    region = generated.convert(target.mode)
    if region.size != (x1 - x0, y1 - y0):
        region = region.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    target.paste(region, (x0, y0))
    return target


MAX_CROP_INFERENCE_DIM = 1024


def scale_crop_for_safe_inference(
    crop_img: Image.Image,
    crop_mask: np.ndarray,
    max_dim: int = MAX_CROP_INFERENCE_DIM,
    align: int = 16,
) -> Tuple[Image.Image, np.ndarray, bool]:
    """Scale down crop image and mask if dimensions exceed max_dim.

    Prevents CUDA Out-Of-Memory errors on large crops (e.g. 4K images).
    Returns (safe_image, safe_mask, was_scaled).
    """
    w, h = crop_img.size
    if max(w, h) <= max_dim or max_dim <= 0:
        return crop_img, crop_mask, False

    scale = float(max_dim) / float(max(w, h))
    target_w = max(align, int(round((w * scale) / align)) * align)
    target_h = max(align, int(round((h * scale) / align)) * align)

    scaled_img = crop_img.resize((target_w, target_h), Image.LANCZOS)

    mask_arr = np.asarray(crop_mask, dtype=np.float32)
    mask_uint8 = (np.clip(mask_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    scaled_mask_pil = mask_pil.resize((target_w, target_h), Image.BILINEAR)
    scaled_mask = np.asarray(scaled_mask_pil, dtype=np.float32) / 255.0

    return scaled_img, scaled_mask, True


def composite_crop_into_image(
    generated: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Image.Image:
    """Composite a generated crop over the original inside ``box`` only.

    Outside the box the original is returned untouched (bit-identical), which
    keeps the global compositing guarantee while the engine only ran on the
    crop.  ``mask`` is the raw (not grown) mask in crop coordinates.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    target = original.copy()
    target_w = x1 - x0
    target_h = y1 - y0
    gen_region = generated.convert(target.mode)
    if gen_region.size != (target_w, target_h):
        gen_region = gen_region.resize((target_w, target_h), Image.LANCZOS)
    original_region = original.crop((x0, y0, x1, y1))
    composed = composite_with_mask(gen_region, original_region, mask)
    target.paste(composed, (x0, y0))
    return target


def save_image_file(image: Image.Image, directory: Path, name: str) -> Path:
    """Persist an arbitrary PIL image as PNG inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    image.save(path, format="PNG")
    return path


def save_mask_file(mask: np.ndarray, directory: Path, name: str) -> Path:
    """Persist a mask as PNG inside ``directory`` (created on demand)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(encode_mask_png(mask))
    return path


def mask_file_name(prefix: str = "mask", extension: str = ".png") -> str:
    """Build a safe file name for a mask artefact."""
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in ("-", "_")) or "mask"
    return f"{safe_prefix}_{os.getpid()}{extension}"


def split_mask_into_islands(
    mask: np.ndarray,
    base_box: Tuple[int, int, int, int],
    min_area: int = 256,
    merge_distance: int = 64,
) -> list:
    """Split a binary mask into connected components (islands).

    Returns a list of dicts, each containing:
        - ``box``: ``(x0, y0, x1, y1)`` crop box for the island
        - ``mask``: cropped binary array for the island
        - ``area``: number of non-zero pixels

    Islands closer than ``merge_distance`` pixels are merged into a single
    island to avoid overlapping crops.  Islands smaller than ``min_area``
    pixels are discarded.

    Uses ``scipy.ndimage.label`` when available (fast), falling back to a
    simple flood-fill implementation when scipy is not installed.
    """
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        return []

    # Try scipy first (much faster for large masks)
    try:
        from scipy.ndimage import label as _scipy_label
        component_mask, num_features = _scipy_label(arr > 0)
        if num_features == 0:
            return []
    except ImportError:
        # Fallback: simple flood-fill using BFS
        component_mask = np.zeros(arr.shape, dtype=np.int32)
        num_features = 0
        visited = np.zeros(arr.shape, dtype=bool)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if arr[y, x] > 0 and not visited[y, x]:
                    num_features += 1
                    # BFS flood-fill
                    queue = [(x, y)]
                    visited[y, x] = True
                    component_mask[y, x] = num_features
                    while queue:
                        cx, cy = queue.pop(0)
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 <= nx < arr.shape[1]
                                and 0 <= ny < arr.shape[0]
                                and arr[ny, nx] > 0
                                and not visited[ny, nx]
                            ):
                                visited[ny, nx] = True
                                component_mask[ny, nx] = num_features
                                queue.append((nx, ny))

    x0, y0, x1, y1 = (int(v) for v in base_box)
    crop_arr = arr[y0:y1, x0:x1]
    crop_comp = component_mask[y0:y1, x0:x1]

    islands = []
    for comp_id in range(1, num_features + 1):
        comp = (crop_comp == comp_id)
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(comp)
        if xs.size == 0:
            continue
        ix0 = int(xs.min())
        iy0 = int(ys.min())
        ix1 = int(xs.max()) + 1
        iy1 = int(ys.max()) + 1
        # Add padding to island box
        pad = max(16, min(64, (ix1 - ix0) // 4))
        ix0 = max(0, ix0 - pad)
        iy0 = max(0, iy0 - pad)
        ix1 = min(crop_arr.shape[1], ix1 + pad)
        iy1 = min(crop_arr.shape[0], iy1 + pad)
        islands.append({
            "box": (x0 + ix0, y0 + iy0, x0 + ix1, y0 + iy1),
            "mask": np.ascontiguousarray(crop_arr[iy0:iy1, ix0:ix1]),
            "area": area,
        })

    # Sort by area (largest first)
    islands.sort(key=lambda i: i["area"], reverse=True)

    # Merge nearby islands
    if merge_distance > 0 and len(islands) > 1:
        merged = [islands[0]]
        for island in islands[1:]:
            should_merge = False
            for i, existing in enumerate(merged):
                ex0, ey0, ex1, ey1 = existing["box"]
                ix0, iy0, ix1, iy1 = island["box"]
                # Check if boxes are close enough to merge
                dx = max(0, ex0 - ix1, ix0 - ex1)
                dy = max(0, ey0 - iy1, iy0 - ey1)
                if dx <= merge_distance and dy <= merge_distance:
                    # Merge: expand box to cover both
                    mx0, my0 = min(ex0, ix0), min(ey0, iy0)
                    mx1, my1 = max(ex1, ix1), max(ey1, iy1)
                    existing["box"] = (mx0, my0, mx1, my1)
                    existing["mask"] = np.ascontiguousarray(arr[my0:my1, mx0:mx1])
                    existing["area"] += island["area"]
                    should_merge = True
                    break
            if not should_merge:
                merged.append(island)
        islands = merged

    return islands
