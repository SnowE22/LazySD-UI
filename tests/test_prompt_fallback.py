"""Unit tests for the local heuristic Inpaint prompt generator (``backend/prompt_fallback.py``)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from backend.prompt_fallback import (
    DEFAULT_NEGATIVE_PROMPT,
    analyze_region,
    extract_dominant_colors,
    generate_heuristic_prompt,
    rgb_to_color_name,
)


def _make_mask(width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 1.0
    return mask


def test_white_speech_bubble_classification_and_prompt():
    img = Image.new("RGB", (256, 256), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    # Draw some dark text inside the center of the white bubble
    draw.rectangle((110, 110, 146, 146), fill=(20, 20, 20))

    box = (96, 96, 160, 160)
    mask = _make_mask(256, 256, box)
    res = generate_heuristic_prompt(img, mask, box, instruction="убрать японский текст")

    assert res["analysis"]["region_type"] == "white_bubble"
    assert "white speech bubble" in res["prompt"]
    assert "no text" in res["prompt"]
    assert "japanese" in res["negative_prompt"]
    assert "handwriting" in res["negative_prompt"]


def test_flat_color_region_classification():
    img = Image.new("RGB", (256, 256), (40, 90, 200))
    box = (80, 80, 160, 160)
    mask = _make_mask(256, 256, box)

    res = generate_heuristic_prompt(img, mask, box, instruction="восстановить фон")
    assert res["analysis"]["region_type"] == "flat_color"
    assert "blue" in res["prompt"]
    assert "seamless" in res["prompt"]


def test_colored_art_splash_classification_and_palette():
    img = Image.new("RGB", (256, 256), (185, 25, 35))
    draw = ImageDraw.Draw(img)
    # Add high-contrast black ink streaks around the region so std/edges are high
    for y in range(40, 220, 10):
        draw.line((40, y, 220, y + 5), fill=(15, 10, 12), width=4)

    box = (80, 80, 176, 176)
    mask = _make_mask(256, 256, box)
    res = generate_heuristic_prompt(img, mask, box, instruction="убрать текст, восстановить красные брызги")

    assert res["analysis"]["region_type"] == "colored_art"
    assert len(res["analysis"]["dominant_colors"]) >= 2
    assert "red" in res["prompt"]
    assert "ink splash" in res["prompt"]
    assert DEFAULT_NEGATIVE_PROMPT in res["negative_prompt"]


def test_explicit_replacement_instruction():
    img = Image.new("RGB", (256, 256), (220, 220, 220))
    box = (64, 64, 192, 192)
    mask = _make_mask(256, 256, box)

    res = generate_heuristic_prompt(img, mask, box, instruction="replace with glowing blue crystal")
    assert "glowing" in res["prompt"]
    assert "blue" in res["prompt"]
    assert "crystal" in res["prompt"]


def test_rgb_to_color_name_and_dominant_colors():
    assert rgb_to_color_name((250, 250, 250)) == "white"
    assert rgb_to_color_name((10, 10, 10)) == "black"
    assert "red" in rgb_to_color_name((200, 20, 25))

    img = Image.new("RGB", (128, 128), (200, 30, 30))
    colors = extract_dominant_colors(img)
    assert len(colors) >= 1
    assert colors[0].startswith("#")
