# -*- coding: utf-8 -*-
"""
Inpaint Masked Window (Crop & Stitch) Pipeline for LazySD-UI.
Allows high-resolution images (manga, 4K, 2K) to be inpainted on 8GB GPUs:
1. Detects the mask bounding box with padding.
2. Crops & normalizes the sub-region to a safe inference window (e.g. 512, 768, 1024).
3. Executes diffusion exclusively on the cropped window (saving 70%+ VRAM and speeding up inference).
4. Smoothly stitches and feathers the result back into the full-resolution master canvas.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter

# Safe Pillow resampling enum fallback
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

@dataclass
class InpaintCropContext:
    box: Tuple[int, int, int, int]
    crop_img_path: Path
    crop_mask_path: Path
    w_out: int
    h_out: int
    orig_size: Tuple[int, int]
    orig_image_path: Path
    crop_mask_orig: Image.Image

def prepare_inpaint_crop(
    orig_image_path: Path,
    mask_image_path: Path,
    uploads_dir: Path,
    target_size: int = 768,
    padding_ratio: float = 0.20,
) -> Optional[InpaintCropContext]:
    """
    Finds mask bounding box, crops sub-region with context padding,
    scales to target_size (divisible by 64), and saves temporary files for sd-cli.
    Returns InpaintCropContext or None if mask is empty or whole-image is preferred.
    """
    if not orig_image_path.exists() or not mask_image_path.exists():
        return None

    uploads_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(orig_image_path) as full_orig, Image.open(mask_image_path) as full_mask:
        orig_img = full_orig.convert("RGB")
        mask_img = full_mask.convert("L")

        orig_w, orig_h = orig_img.size
        if mask_img.size != (orig_w, orig_h):
            mask_img = mask_img.resize((orig_w, orig_h), RESAMPLE_BILINEAR)

        mask_bbox = mask_img.getbbox()
        if not mask_bbox:
            return None

        x0, y0, x1, y1 = mask_bbox
        mask_w = max(1, x1 - x0)
        mask_h = max(1, y1 - y0)

        # Context padding around mask (at least 64px)
        pad = max(64, int(max(mask_w, mask_h) * padding_ratio))
        bx0 = max(0, x0 - pad)
        by0 = max(0, y0 - pad)
        bx1 = min(orig_w, x1 + pad)
        by1 = min(orig_h, y1 + pad)

        crop_w = bx1 - bx0
        crop_h = by1 - by0

        # Enforce reasonable aspect ratio (at most 2:1) by expanding smaller axis if possible
        if crop_w > crop_h * 2:
            extra = (crop_w // 2 - crop_h) // 2
            by0 = max(0, by0 - extra)
            by1 = min(orig_h, by1 + extra)
            crop_h = by1 - by0
        elif crop_h > crop_w * 2:
            extra = (crop_h // 2 - crop_w) // 2
            bx0 = max(0, bx0 - extra)
            bx1 = min(orig_w, bx1 + extra)
            crop_w = bx1 - bx0

        box = (bx0, by0, bx1, by1)

        # Extract cropped region
        crop_img = orig_img.crop(box)
        crop_mask_orig = mask_img.crop(box)

        # Calculate target scaled dimensions divisible by 64
        max_dim = max(crop_w, crop_h)
        scale = min(1.0, target_size / max_dim)
        w_out = max(256, int(round((crop_w * scale) / 64.0)) * 64)
        h_out = max(256, int(round((crop_h * scale) / 64.0)) * 64)

        # Rescale for diffusion model inference
        scaled_img = crop_img.resize((w_out, h_out), RESAMPLE_LANCZOS)
        scaled_mask = crop_mask_orig.resize((w_out, h_out), RESAMPLE_BILINEAR)

        uid = uuid.uuid4().hex
        crop_img_path = uploads_dir / f"crop_{uid}.png"
        crop_mask_path = uploads_dir / f"crop_mask_{uid}.png"

        scaled_img.save(crop_img_path)
        scaled_mask.save(crop_mask_path)

        return InpaintCropContext(
            box=box,
            crop_img_path=crop_img_path,
            crop_mask_path=crop_mask_path,
            w_out=w_out,
            h_out=h_out,
            orig_size=(orig_w, orig_h),
            orig_image_path=orig_image_path,
            crop_mask_orig=crop_mask_orig,
        )

def stitch_inpaint_result(
    crop_ctx: InpaintCropContext,
    generated_image_path: Path,
    mask_blur: int = 4,
) -> Path:
    """
    Takes generated crop from generated_image_path, rescales to original bounding box,
    feathers mask edges with GaussianBlur, smoothly composites onto original image,
    and updates generated_image_path with the full-resolution master result.
    Cleans up temporary crop files.
    """
    try:
        if not generated_image_path.exists():
            return generated_image_path

        bx0, by0, bx1, by1 = crop_ctx.box
        crop_w = bx1 - bx0
        crop_h = by1 - by0

        with Image.open(generated_image_path) as gen_file, Image.open(crop_ctx.orig_image_path) as master_file:
            gen_crop = gen_file.convert("RGB")
            master_img = master_file.convert("RGB")

            # Scale generated crop back to original cropped box
            resized_gen = gen_crop.resize((crop_w, crop_h), RESAMPLE_LANCZOS)
            orig_crop = master_img.crop(crop_ctx.box)

            # Feather mask edges for seamless transition
            blur_radius = max(1, mask_blur)
            feathered_mask = crop_ctx.crop_mask_orig.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Composite inpainted crop with original background
            blended_crop = Image.composite(resized_gen, orig_crop, feathered_mask)

            # Paste into master image
            master_img.paste(blended_crop, (bx0, by0))
            master_img.save(generated_image_path)

        return generated_image_path
    finally:
        # Cleanup temporary crop files
        crop_ctx.crop_img_path.unlink(missing_ok=True)
        crop_ctx.crop_mask_path.unlink(missing_ok=True)
