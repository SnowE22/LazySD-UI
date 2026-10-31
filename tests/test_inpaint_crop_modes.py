"""Tests for the Inpaint generation-area (crop) modes.

Covers the three modes required by the specification:

  auto       - mask bbox with the project defaults (padding 64, min size 256),
  parameters - mask bbox with a user supplied padding / min size,
  manual     - exact rectangle drawn by the user on the source image.

The manual mode must never cut a part of the painted mask away, the
coordinates are always interpreted in the pixel grid of the ORIGINAL image and
every backend has to apply exactly the same rules.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.masking import (  # noqa: E402
    MASK_CROP_ALIGN,
    MASK_CROP_MANUAL_MIN_SIZE,
    MASK_CROP_MIN_SIZE,
    MASK_CROP_MIN_SIZE_RANGE,
    MASK_CROP_MODES,
    MASK_CROP_PADDING,
    MASK_CROP_PADDING_RANGE,
    composite_crop_into_image,
    compute_crop_box,
    crop_array,
    feather_mask,
    normalize_crop_mode,
    resolve_crop_box,
    resolve_manual_crop_box,
)

FRONTEND = root_dir / "frontend"
INDEX_HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "js" / "app.js").read_text(encoding="utf-8")
INPAINT_JS = (FRONTEND / "js" / "modules" / "inpaint.js").read_text(encoding="utf-8")
APP_PY = (root_dir / "app.py").read_text(encoding="utf-8")
WORKER_PY = (root_dir / "backend" / "worker.py").read_text(encoding="utf-8")
EXTRAS_PY = (root_dir / "backend" / "extras.py").read_text(encoding="utf-8")
SD_CPP_PY = (root_dir / "backend" / "sd_cpp.py").read_text(encoding="utf-8")


def _mask_with_square(size=512, box=(120, 100, 260, 200)):
    """Raw (painted) mask with a single white rectangle."""
    mask = np.zeros((size, size), dtype=np.float32)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 1.0
    return mask


def _mask_data_url(size=64, box=(16, 16, 48, 48)) -> str:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            img.putpixel((x, y), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# backend: crop mode semantics
# ---------------------------------------------------------------------------

def test_crop_modes_are_exactly_the_three_specified_ones():
    assert MASK_CROP_MODES == ("auto", "parameters", "manual")
    assert normalize_crop_mode("MANUAL") == "manual"
    assert normalize_crop_mode(None) == "auto"
    try:
        normalize_crop_mode("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown crop mode must be rejected")


def test_auto_mode_keeps_the_default_behaviour():
    """auto must stay equivalent to the historical compute_crop_box()."""
    mask = _mask_with_square()
    expected = compute_crop_box(mask, padding=MASK_CROP_PADDING, min_size=MASK_CROP_MIN_SIZE)
    assert resolve_crop_box(mask, mode="auto") == expected
    # The user supplied values are ignored in auto mode.
    assert resolve_crop_box(mask, mode="auto", padding=0, min_size=64) == expected


def test_parameters_mode_honours_padding_and_min_size():
    mask = _mask_with_square()
    box = resolve_crop_box(mask, mode="parameters", padding=0, min_size=64)
    assert box is not None
    x0, y0, x1, y1 = box
    # The mask bbox is (120, 100, 260, 200); without padding the box hugs it
    # and is aligned outward to the 16px grid.
    assert (x0, y0, x1, y1) == (112, 96, 272, 208)
    assert x0 % MASK_CROP_ALIGN == 0 and y0 % MASK_CROP_ALIGN == 0

    wide = resolve_crop_box(mask, mode="parameters", padding=128, min_size=256)
    assert wide is not None
    # The padding grows the box around the mask (clamped to the image).
    assert wide[0] == max(0, 120 - 128)
    assert wide[1] == max(0, 100 - 128)
    assert wide[2] >= 260 + 128 and wide[3] >= 200 + 128
    assert wide[2] - wide[0] >= wide[2] - wide[0]
    assert (wide[2] - wide[0]) >= 256 and (wide[3] - wide[1]) >= 256


def test_parameters_mode_clamps_out_of_range_values():
    mask = _mask_with_square()
    low = resolve_crop_box(mask, mode="parameters", padding=-100, min_size=1)
    high = resolve_crop_box(mask, mode="parameters", padding=100000, min_size=100000)
    assert low is not None and high is not None
    # padding is clamped into [0, 1024] and min size into [64, 4096].
    assert low[0] >= 120 - MASK_CROP_PADDING_RANGE[1]
    assert high[2] - high[0] <= 512
    assert MASK_CROP_PADDING_RANGE == (0, 1024)
    assert MASK_CROP_MIN_SIZE_RANGE == (64, 4096)


def test_manual_mode_aligns_outward_and_clamps_to_the_image():
    mask = _mask_with_square()
    box = resolve_crop_box(mask, mode="manual", manual_box=[121, 101, 259, 199])
    assert box is not None
    x0, y0, x1, y1 = box
    # Aligned outward to the 16px grid.
    assert (x0, y0, x1, y1) == (112, 96, 272, 208)
    assert x0 % MASK_CROP_ALIGN == 0 and y0 % MASK_CROP_ALIGN == 0
    assert (x1 - x0) % MASK_CROP_ALIGN == 0 and (y1 - y0) % MASK_CROP_ALIGN == 0


def test_manual_mode_clamps_coordinates_outside_the_image():
    mask = _mask_with_square(size=512)
    box = resolve_crop_box(mask, mode="manual", manual_box=[-500, -500, 5000, 5000])
    assert box == (0, 0, 512, 512)


def test_manual_mode_requires_a_rectangle():
    mask = _mask_with_square()
    try:
        resolve_crop_box(mask, mode="manual", manual_box=None)
    except ValueError as exc:
        assert "mask_crop_box" in str(exc)
    else:
        raise AssertionError("The manual mode must refuse to work without a rectangle")


def test_manual_mode_rejects_non_finite_coordinates():
    mask = _mask_with_square()
    for bad in ([0, 0, float("inf"), 100], [0, 0, float("nan"), 100]):
        try:
            resolve_crop_box(mask, mode="manual", manual_box=bad)
        except ValueError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError("Non finite coordinates must be rejected")


def test_manual_mode_rejects_a_rectangle_smaller_than_64px():
    mask = _mask_with_square()
    try:
        resolve_crop_box(mask, mode="manual", manual_box=[100, 100, 130, 130])
    except ValueError as exc:
        assert "too small" in str(exc)
    else:
        raise AssertionError("A rectangle smaller than 64px per side must be rejected")


def test_manual_mode_must_cover_the_whole_painted_mask():
    """The core guarantee: a manual rectangle may not cut the mask."""
    mask = _mask_with_square()
    try:
        resolve_crop_box(mask, mode="manual", manual_box=[100, 80, 200, 220])
    except ValueError as exc:
        assert "whole mask" in str(exc)
    else:
        raise AssertionError("A rectangle that cuts the mask must be rejected")


def test_manual_mode_coverage_is_checked_against_the_raw_mask():
    """The grown/blurred engine mask must not widen the required rectangle.

    The user paints a small square; the engine mask is dilated by 6px.  A
    rectangle that covers the painted square is valid - the dilated border may
    simply be clipped by the crop.
    """
    raw = _mask_with_square(box=(200, 200, 240, 240))
    engine = feather_mask(raw, grow=6, blur=0.0)
    box = resolve_crop_box(
        engine,
        mode="manual",
        manual_box=[192, 192, 248, 248],
        coverage_mask=raw,
    )
    assert box is not None
    assert box == (192, 192, 256, 256)


def test_manual_mode_rejects_a_mask_size_mismatch():
    raw = _mask_with_square(size=512)
    engine = np.zeros((256, 256), dtype=np.float32)
    try:
        resolve_crop_box(engine, mode="manual", manual_box=[0, 0, 256, 256], coverage_mask=raw)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("A mask that does not match the image must be rejected")


def test_manual_mode_rejects_an_empty_mask():
    empty = np.zeros((512, 512), dtype=np.float32)
    try:
        resolve_crop_box(empty, mode="manual", manual_box=[0, 0, 512, 512])
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("An empty mask must be rejected in manual mode")


def test_resolve_manual_crop_box_normalises_reversed_coordinates():
    box = resolve_manual_crop_box([300, 200, 100, 50], width=512, height=512)
    assert box == (96, 48, 304, 208)


# ---------------------------------------------------------------------------
# backend: the result invariants
# ---------------------------------------------------------------------------

def test_manual_crop_keeps_the_original_size_and_the_pixels_outside_the_mask():
    """Integration: the crop is an inner working area, the result keeps W x H."""
    original = Image.new("RGB", (256, 256), (11, 22, 33))
    original.putpixel((5, 5), (200, 100, 50))
    generated = Image.new("RGB", (64, 64), (250, 240, 230))

    raw = np.zeros((256, 256), dtype=np.float32)
    raw[100:150, 100:150] = 1.0

    box = resolve_crop_box(raw, mode="manual", manual_box=[96, 96, 160, 160])
    assert box == (96, 96, 160, 160)

    cropped_mask = crop_array(raw, box)
    result = composite_crop_into_image(generated, original, cropped_mask, box)
    out = np.asarray(result)

    assert result.size == original.size, "the result must keep the original frame"
    assert tuple(out[5, 5]) == (200, 100, 50), "outside the mask must be untouched"
    assert tuple(out[10, 10]) == (11, 22, 33), "outside the mask must be untouched"
    assert tuple(out[120, 120]) == (250, 240, 230), "inside the mask -> generated"


# ---------------------------------------------------------------------------
# backend: every adapter applies the same rules
# ---------------------------------------------------------------------------

def test_all_backends_use_the_same_crop_resolver_and_coverage_mask():
    for source in (WORKER_PY, EXTRAS_PY, SD_CPP_PY):
        assert "resolve_crop_box" in source
        assert "coverage_mask=" in source, "the raw mask must be used for the coverage check"
        assert "mask_crop_mode" in source
        assert "mask_crop_padding" in source
        assert "mask_crop_min_size" in source
        assert "mask_crop_box" in source


def test_generate_request_declares_the_crop_fields():
    from backend.worker import GenerateRequest

    request = GenerateRequest(model_id="demo", prompt="demo")
    assert request.mask_crop_mode == "auto"
    assert request.mask_crop_padding == MASK_CROP_PADDING
    assert request.mask_crop_min_size == MASK_CROP_MIN_SIZE
    assert request.mask_crop_box is None

    request.mask_crop_mode = "manual"
    request.mask_crop_box = [0, 0, 64, 64]
    assert request.mask_crop_box == [0, 0, 64, 64]


# ---------------------------------------------------------------------------
# api: request validation
# ---------------------------------------------------------------------------

def _client():
    import app as app_module
    from fastapi.testclient import TestClient

    return TestClient(app_module.create_app())


def _post_generate(client, **extra):
    payload = {
        "model_id": "qwen_image_edit",
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "init_image_path": "source.png",
        "mask": _mask_data_url(),
    }
    payload.update(extra)
    return client.post("/api/generate", json=payload)


def test_generate_endpoint_rejects_an_unknown_crop_mode():
    client = _client()
    response = _post_generate(client, mask_crop_mode="bogus")
    assert response.status_code == 400, response.text
    assert "mask_crop_mode" in response.json()["detail"]


def test_generate_endpoint_rejects_out_of_range_crop_values():
    client = _client()
    response = _post_generate(client, mask_crop_padding=5000)
    assert response.status_code == 400, response.text
    assert "mask_crop_padding" in response.json()["detail"]

    response = _post_generate(client, mask_crop_min_size=8)
    assert response.status_code == 400, response.text
    assert "mask_crop_min_size" in response.json()["detail"]


def test_generate_endpoint_requires_a_rectangle_for_the_manual_mode():
    client = _client()
    response = _post_generate(client, mask_crop_mode="manual")
    assert response.status_code == 400, response.text
    assert "mask_crop_box" in response.json()["detail"]


def test_generate_endpoint_rejects_a_malformed_rectangle():
    client = _client()
    for bad in ([1, 2, 3], "nope", [1, 2, 3, "x"], [1, 2, 3, None]):
        response = _post_generate(client, mask_crop_mode="manual", mask_crop_box=bad)
        assert response.status_code == 400, response.text
        assert "mask_crop_box" in response.json()["detail"]


def test_generate_endpoint_rejects_non_finite_rectangle_coordinates():
    client = _client()
    # "Infinity" is not valid JSON, so the raw body is used on purpose.
    body = (
        '{"model_id": "qwen_image_edit", "prompt": "test", '
        '"width": 1024, "height": 1024, "steps": 4, '
        '"init_image_path": "source.png", '
        '"mask_crop_mode": "manual", '
        '"mask_crop_box": [0, 0, Infinity, 100]}'
    )
    response = client.post(
        "/api/generate",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert "finite" in response.json()["detail"]


def test_request_helpers_normalise_the_rectangle():
    import app as app_module

    assert app_module._parse_mask_crop_box(None) is None
    assert app_module._parse_mask_crop_box([10, 20, 30, 40]) == [10, 20, 30, 40]
    assert app_module._parse_mask_crop_box([30, 40, 10, 20]) == [10, 20, 30, 40]
    assert app_module._parse_mask_crop_box({"x0": 0, "y0": 0, "x1": 8, "y1": 8}) == [0, 0, 8, 8]
    assert app_module._parse_int_in_range("5", "value", (0, 10)) == 5
    for bad in ([1, 2, 3], [1, 2, 3, None]):
        try:
            app_module._parse_mask_crop_box(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must be rejected")
    try:
        app_module._parse_int_in_range(99, "value", (0, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("Out of range values must be rejected")


# ---------------------------------------------------------------------------
# frontend: markup, API and localisation
# ---------------------------------------------------------------------------

def test_index_html_wires_the_generation_area_ui():
    for marker in (
        'id="inpaint-crop-overlay"',
        'id="inpaint-crop-block"',
        'id="inpaint-crop-mode"',
        'id="inpaint-crop-params"',
        'id="inpaint-crop-manual"',
        'id="inpaint-crop-padding"',
        'id="inpaint-crop-min-size"',
        'id="inpaint-crop-select"',
        'id="inpaint-crop-reset"',
        'id="inpaint-crop-coords"',
        'id="inpaint-crop-warning"',
    ):
        assert marker in INDEX_HTML, f"index.html is missing {marker}"

    for marker in (
        'value="auto"',
        'value="parameters"',
        'value="manual"',
    ):
        assert marker in INDEX_HTML, f"index.html is missing the crop mode {marker}"


def test_frontend_exposes_the_crop_api_in_both_implementations():
    app_api = (
        "getInpaintCropState",
        "resolveInpaintManualCropBox",
        "getInpaintMaskBBox",
        "getInpaintCropProblem",
        "setInpaintCropMode",
        "setInpaintCropPadding",
        "setInpaintCropMinSize",
        "toggleInpaintCropSelection",
        "resetInpaintCropSelection",
        "renderInpaintCropUI",
        "renderInpaintCropOverlay",
        "updateInpaintCropStatus",
        "initInpaintCropEvents",
    )
    for name in app_api:
        assert name in APP_JS, f"app.js is missing {name}"
        assert f"export function {name}" in INPAINT_JS, f"modules/inpaint.js is missing {name}"


def test_frontend_sends_the_crop_parameters_to_the_backend():
    for source in (APP_JS, INPAINT_JS):
        assert "mask_crop_mode" in source
        assert "mask_crop_padding" in source
        assert "mask_crop_min_size" in source
        assert "mask_crop_box" in source


def test_frontend_uses_the_source_pixel_grid_for_the_selection():
    """The rectangle must be built from the canvas coordinates, not the screen."""
    for source in (APP_JS, INPAINT_JS):
        assert "pointerToCanvasPoint" in source
        assert "resolveInpaintManualCropBox" in source


def test_locales_contain_the_crop_keys():
    for name in ("ru.json", "en.json"):
        data = json.loads((FRONTEND / "locale" / name).read_text(encoding="utf-8"))
        for key in (
            "inpaint.crop.title",
            "inpaint.crop.mode.auto",
            "inpaint.crop.mode.parameters",
            "inpaint.crop.mode.manual",
            "inpaint.crop.desc.auto",
            "inpaint.crop.desc.parameters",
            "inpaint.crop.desc.manual",
            "inpaint.crop.padding",
            "inpaint.crop.min_size",
            "inpaint.crop.select",
            "inpaint.crop.selecting",
            "inpaint.crop.reset",
            "inpaint.crop.no_selection",
            "inpaint.crop.error.no_box",
            "inpaint.crop.error.too_small",
            "inpaint.crop.error.not_covered",
        ):
            assert key in data, f"{name} is missing {key}"

    ru = json.loads((FRONTEND / "locale" / "ru.json").read_text(encoding="utf-8"))
    en = json.loads((FRONTEND / "locale" / "en.json").read_text(encoding="utf-8"))
    assert set(ru) == set(en), "the locales must stay in sync"


def test_manual_crop_mode_blocks_the_generation_without_a_rectangle():
    """The frontend must not send a manual request without a selection."""
    for source in (APP_JS, INPAINT_JS):
        assert "getInpaintCropProblem" in source
        assert 'crop.mode === "manual"' in source
