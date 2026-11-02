"""Heuristic fallback prompt generator for Inpaint Studio.

Operates purely locally with ``Pillow`` and ``numpy`` (no external ML or VLM
dependencies).  Analyses the masked region and its surrounding context ring
to classify the surface (``white_bubble``, ``flat_color``, ``colored_art``),
extract dominant palette colours, parse the user's instruction, and assemble
an English diffusion prompt + negative prompt.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


DEFAULT_NEGATIVE_PROMPT = (
    "text, japanese, kanji, kana, letters, sfx, speech bubble outline, "
    "watermark, signature, seams, blurry, low contrast, color shift"
)

# Bilingual keyword patterns for user intent classification
_RE_REMOVE = re.compile(
    r"\b(удал\w*|убер\w*|убрат\w*|сотр\w*|стерет\w*|очист\w*|чист\w*|"
    r"clean\w*|remov\w*|eras\w*|clear\w*|rid\b)\b",
    re.IGNORECASE,
)
_RE_REPLACE = re.compile(
    r"\b(замен\w*|вместо|сдела\w*|нарису\w*|добав\w*|"
    r"replac\w*|instead|draw\w*|add\w*|turn\s+into|make\s+it)\b",
    re.IGNORECASE,
)
_RE_BACKGROUND = re.compile(
    r"\b(фон\w*|зали\w*|восстанов\w*|продолж\w*|"
    r"background|fill\w*|restor\w*|continu\w*|blend\w*)\b",
    re.IGNORECASE,
)
_RE_TEXT_SFX = re.compile(
    r"\b(текст\w*|надпис\w*|букв\w*|иероглиф\w*|звук\w*|сфх|"
    r"text\w*|letter\w*|kanji|kana|hiragana|katakana|sfx|sound\s*effect\w*|caption\w*|subtitle\w*)\b",
    re.IGNORECASE,
)
_RE_BUBBLE = re.compile(
    r"\b(баббл\w*|бабл\w*|пузыр\w*|облачк\w*|рамочк\w*|"
    r"speech\s*bubble\w*|bubble\w*|balloon\w*|dialog\s*box\w*)\b",
    re.IGNORECASE,
)

# Common Russian -> English colour / texture hints inside user instructions
_RU_EN_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"красн\w*", re.IGNORECASE), "red"),
    (re.compile(r"сини\w*|сине\w*|голуб\w*", re.IGNORECASE), "blue"),
    (re.compile(r"зелен\w*|зелён\w*", re.IGNORECASE), "green"),
    (re.compile(r"желт\w*|жёлт\w*", re.IGNORECASE), "yellow"),
    (re.compile(r"оранжев\w*", re.IGNORECASE), "orange"),
    (re.compile(r"фиолетов\w*|пурпур\w*", re.IGNORECASE), "purple"),
    (re.compile(r"розов\w*", re.IGNORECASE), "pink"),
    (re.compile(r"коричнев\w*", re.IGNORECASE), "brown"),
    (re.compile(r"бел\w*", re.IGNORECASE), "white"),
    (re.compile(r"черн\w*|чёрн\w*", re.IGNORECASE), "black"),
    (re.compile(r"сер\w*", re.IGNORECASE), "grey"),
    (re.compile(r"темн\w*|тёмн\w*", re.IGNORECASE), "dark"),
    (re.compile(r"светл\w*", re.IGNORECASE), "light"),
    (re.compile(r"брызг\w*|всплеск\w*", re.IGNORECASE), "ink splash"),
    (re.compile(r"скринтон\w*|тон\w*", re.IGNORECASE), "manga screentone"),
    (re.compile(r"штрих\w*", re.IGNORECASE), "crosshatching"),
    (re.compile(r"градиент\w*", re.IGNORECASE), "smooth gradient"),
    (re.compile(r"небо|неба", re.IGNORECASE), "sky background"),
    (re.compile(r"волос\w*", re.IGNORECASE), "hair strands"),
    (re.compile(r"одежд\w*|плащ\w*|куртк\w*", re.IGNORECASE), "clothing fabric folds"),
    (re.compile(r"кожа|кожу|лиц\w*", re.IGNORECASE), "smooth skin shading"),
    (re.compile(r"дерев\w*|листв\w*", re.IGNORECASE), "foliage texture"),
    (re.compile(r"стен\w*", re.IGNORECASE), "wall surface texture"),
]


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """Convert an RGB tuple (0..255) into a descriptive English colour phrase."""
    r, g, b = (float(c) / 255.0 for c in rgb)
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    val = cmax
    sat = 0.0 if cmax <= 1e-5 else (delta / cmax)

    if val > 0.90 and sat < 0.12:
        return "white"
    if val < 0.14:
        return "black"
    if sat < 0.14:
        if val < 0.38:
            return "dark grey"
        if val > 0.72:
            return "light grey"
        return "neutral grey"

    # Compute hue in degrees [0, 360)
    if delta <= 1e-5:
        hue = 0.0
    elif cmax == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif cmax == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)

    if hue < 15 or hue >= 345:
        base = "red"
    elif hue < 42:
        base = "orange" if val > 0.55 else "brown"
    elif hue < 68:
        base = "yellow" if sat > 0.35 else "warm beige"
    elif hue < 160:
        base = "green"
    elif hue < 200:
        base = "teal"
    elif hue < 255:
        base = "blue"
    elif hue < 295:
        base = "purple"
    else:
        base = "crimson" if val < 0.55 else "pink"

    if val < 0.38 and base not in ("brown", "crimson"):
        return f"dark {base}"
    if val > 0.82 and sat < 0.45 and base not in ("warm beige",):
        return f"pale {base}"
    return base


def extract_dominant_colors(
    image: Image.Image,
    sample_mask: Optional[np.ndarray] = None,
    max_colors: int = 5,
) -> List[str]:
    """Return up to ``max_colors`` dominant hex colours sorted by area."""
    rgb_img = image.convert("RGB")
    small = rgb_img.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)

    if sample_mask is not None and sample_mask.ndim == 2 and sample_mask.size > 0:
        mask_pil = Image.fromarray((np.clip(sample_mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        mask_small = np.asarray(mask_pil.resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8)
        pixels = arr[mask_small > 32]
        if pixels.shape[0] < 16:
            pixels = arr.reshape(-1, 3)
    else:
        pixels = arr.reshape(-1, 3)

    if pixels.shape[0] == 0:
        return ["#ffffff"]

    # Quantise RGB into 16-level buckets (4 bits per channel) and average each cluster
    buckets = (pixels // 24).astype(np.int32)
    keys = (buckets[:, 0] << 8) | (buckets[:, 1] << 4) | buckets[:, 2]
    unique_keys, counts = np.unique(keys, return_counts=True)

    # Filter out tiny noise clusters (< 2% of pixels) unless only one cluster exists
    min_count = max(1, int(pixels.shape[0] * 0.02))
    order = np.argsort(-counts)
    hex_colors: List[str] = []
    for idx in order:
        if len(hex_colors) >= max_colors:
            break
        if counts[idx] < min_count and len(hex_colors) > 0:
            continue
        cluster_pixels = pixels[keys == unique_keys[idx]]
        mean_rgb = tuple(int(round(float(v))) for v in cluster_pixels.mean(axis=0))
        hex_val = _rgb_to_hex(mean_rgb)
        if hex_val not in hex_colors:
            hex_colors.append(hex_val)

    return hex_colors or ["#ffffff"]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.lstrip("#")
    if len(s) != 6:
        return (255, 255, 255)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def analyze_region(
    source_image: Image.Image,
    mask_array: np.ndarray,
    mask_box: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """Analyse the masked region and its surrounding context ring (10-25% expansion)."""
    rgb_img = source_image.convert("RGB")
    width, height = rgb_img.size
    x0, y0, x1, y1 = (int(v) for v in mask_box)
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))

    bw = x1 - x0
    bh = y1 - y0
    pad_x = max(8, int(round(bw * 0.20)))
    pad_y = max(8, int(round(bh * 0.20)))

    rx0 = max(0, x0 - pad_x)
    ry0 = max(0, y0 - pad_y)
    rx1 = min(width, x1 + pad_x)
    ry1 = min(height, y1 + pad_y)

    ring_crop = rgb_img.crop((rx0, ry0, rx1, ry1))
    mask_crop = np.asarray(mask_array, dtype=np.float32)[ry0:ry1, rx0:rx1]

    # Surrounding ring = outside the active mask (or whole box if mask covers 100%)
    ring_selector = (mask_crop <= 0.25).astype(np.float32)
    if float(ring_selector.sum()) < 24.0:
        ring_selector = np.ones_like(mask_crop, dtype=np.float32)

    dominant_colors = extract_dominant_colors(ring_crop, sample_mask=ring_selector, max_colors=5)

    small_rgb = np.asarray(
        ring_crop.resize((64, 64), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    ring_pil = Image.fromarray((np.clip(ring_selector, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    small_ring = np.asarray(ring_pil.resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8) > 32
    if int(small_ring.sum()) < 16:
        small_ring = np.ones((64, 64), dtype=bool)

    cmax = small_rgb.max(axis=2)
    cmin = small_rgb.min(axis=2)
    delta = cmax - cmin
    sat_map = np.where(cmax > 1e-5, delta / np.maximum(cmax, 1e-5), 0.0)
    luma_map = 0.299 * small_rgb[:, :, 0] + 0.587 * small_rgb[:, :, 1] + 0.114 * small_rgb[:, :, 2]

    # Sample brightness & saturation primarily on the context ring, with full-patch fallback
    ring_luma = luma_map[small_ring]
    ring_sat = sat_map[small_ring]
    mean_brightness = float(ring_luma.mean()) if ring_luma.size else float(luma_map.mean())
    mean_saturation = float(ring_sat.mean()) if ring_sat.size else float(sat_map.mean())
    brightness_std = float(ring_luma.std()) if ring_luma.size else float(luma_map.std())

    # Simple Sobel / finite-difference edge density on 64x64 luma
    gx = np.abs(np.diff(luma_map, axis=1, prepend=luma_map[:, :1]))
    gy = np.abs(np.diff(luma_map, axis=0, prepend=luma_map[:1, :]))
    grad = np.hypot(gx, gy)
    edge_density = float((grad[small_ring] > 0.12).mean()) if ring_luma.size else float((grad > 0.12).mean())

    # Region classification per TZ Section 9.5
    if mean_brightness > 0.82 and mean_saturation < 0.15 and brightness_std < 0.14:
        region_type = "white_bubble"
    elif brightness_std < 0.09 and edge_density < 0.12:
        region_type = "flat_color"
    else:
        region_type = "colored_art"

    return {
        "dominant_colors": dominant_colors,
        "region_type": region_type,
        "mask_box": [int(x0), int(y0), int(x1), int(y1)],
        "brightness": round(mean_brightness, 4),
        "saturation": round(mean_saturation, 4),
        "brightness_std": round(brightness_std, 4),
        "edge_density": round(edge_density, 4),
    }


def _extract_instruction_traits(instruction: str) -> Dict[str, Any]:
    text = (instruction or "").strip()
    is_remove = bool(_RE_REMOVE.search(text))
    is_replace = bool(_RE_REPLACE.search(text))
    is_background = bool(_RE_BACKGROUND.search(text))
    has_text_mention = bool(_RE_TEXT_SFX.search(text))
    has_bubble_mention = bool(_RE_BUBBLE.search(text))

    en_hints: List[str] = []
    for pattern, phrase in _RU_EN_HINTS:
        if pattern.search(text) and phrase not in en_hints:
            en_hints.append(phrase)

    # Keep plain ASCII English descriptors if user wrote English words directly
    ascii_words = re.findall(r"[a-zA-Z]{3,}", text)
    stop_words = {
        "the", "and", "with", "into", "from", "this", "that", "area",
        "remove", "clean", "erase", "clear", "replace", "instead", "text",
        "japanese", "kanji", "kana", "sfx", "make", "fill", "restore",
    }
    custom_en = [w.lower() for w in ascii_words if w.lower() not in stop_words]

    return {
        "raw": text,
        "is_remove": is_remove,
        "is_replace": is_replace and not is_remove,
        "is_background": is_background,
        "has_text_mention": has_text_mention,
        "has_bubble_mention": has_bubble_mention,
        "en_hints": en_hints,
        "custom_en": custom_en,
    }


def generate_heuristic_prompt(
    source_image: Image.Image,
    mask_array: np.ndarray,
    mask_box: Tuple[int, int, int, int],
    instruction: str = "",
) -> Dict[str, Any]:
    """Build a `{prompt, negative_prompt, analysis}` payload using local heuristics."""
    analysis = analyze_region(source_image, mask_array, mask_box)
    traits = _extract_instruction_traits(instruction)

    colors_hex = analysis["dominant_colors"]
    color_names: List[str] = []
    for h in colors_hex:
        cname = rgb_to_color_name(hex_to_rgb(h))
        if cname not in color_names:
            color_names.append(cname)
    primary_color = color_names[0] if color_names else "neutral"
    secondary_color = color_names[1] if len(color_names) > 1 else primary_color

    region_type = analysis["region_type"]
    if traits["has_bubble_mention"] and primary_color in ("white", "light grey"):
        region_type = "white_bubble"
        analysis["region_type"] = "white_bubble"

    hint_suffix = ""
    if traits["en_hints"]:
        hint_suffix = ", " + ", ".join(traits["en_hints"][:3])

    if traits["is_replace"] and (traits["custom_en"] or traits["en_hints"]):
        subject_parts = traits["en_hints"] + traits["custom_en"][:6]
        subject = " ".join(dict.fromkeys(subject_parts)).strip()
        style = (
            "clean manga lineart and paper texture"
            if analysis["saturation"] < 0.18
            else f"matching {primary_color} manga illustration style"
        )
        prompt = f"{subject}, {style}, seamless integration, clean edges, no text"
    elif region_type == "white_bubble":
        prompt = (
            "clean white speech bubble interior, smooth white fill, "
            f"subtle paper texture{hint_suffix}, seamless edges, no text"
        )
    elif region_type == "flat_color":
        prompt = (
            f"seamless {primary_color} fill, matching surrounding {primary_color}"
            f"{hint_suffix}, smooth paper texture, clean edges, no text"
        )
    else:
        if secondary_color != primary_color:
            color_phrase = f"{primary_color} and {secondary_color}"
        else:
            color_phrase = primary_color
        texture_phrase = (
            "manga screentone and ink lineart texture"
            if analysis["saturation"] < 0.18
            else "manga brush and illustration texture"
        )
        prompt = (
            f"seamless {color_phrase} background, matching surrounding {primary_color}"
            f"{hint_suffix}, {texture_phrase}, clean edges, no text"
        )

    neg_parts = [DEFAULT_NEGATIVE_PROMPT]
    if traits["has_text_mention"]:
        neg_parts.append("handwriting, typography, speech text, dialogue, font")

    prompt = prompt.strip()[:1000]
    negative_prompt = ", ".join(neg_parts).strip()[:500]

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "analysis": analysis,
    }
