"""Backend tests for the inpaint mask pipeline.

Covers the guarantees required by the specification:
  * one unambiguous mask semantic (white = edit, black = preserve),
  * the final image is composited as ``generated * mask + original * (1 - mask)``,
  * pixels outside the mask stay bit-identical to the original,
  * models without mask support are detected before a job is started.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.masking import (  # noqa: E402
    MaskError,
    composite_with_mask,
    decode_mask_to_float,
    encode_mask_png,
    engine_mask_arg,
    engine_mask_support,
    feather_mask,
    is_mask_empty,
    mask_coverage,
    prepare_mask_arrays,
)


def _png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _rgba_mask(painted: bool) -> Image.Image:
    """Transparent canvas with a white opaque square in the middle (or empty)."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    if painted:
        for x in range(16, 48):
            for y in range(16, 48):
                img.putpixel((x, y), (255, 255, 255, 255))
    return img


def test_decode_mask_uses_alpha_channel():
    """A transparent PNG must decode to 0 outside the painted strokes."""
    data_url = _png_data_url(_rgba_mask(painted=True))
    mask = decode_mask_to_float(data_url, target_size=(64, 64))

    assert mask is not None
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert mask.max() > 0.99, "painted white area must decode to 1.0"
    assert mask[0, 0] == 0.0, "unpainted (transparent) area must decode to 0.0"
    assert mask[32, 32] > 0.99
    assert mask_coverage(mask) < 0.7


def test_decode_mask_invert():
    mask = decode_mask_to_float(_png_data_url(_rgba_mask(painted=True)), target_size=(64, 64))
    inverted = decode_mask_to_float(
        _png_data_url(_rgba_mask(painted=True)), target_size=(64, 64), invert_mask=True
    )
    assert mask is not None and inverted is not None
    assert abs(mask[32, 32] - 1.0) < 1e-6
    assert abs(inverted[32, 32] - 0.0) < 1e-6
    assert abs(inverted[0, 0] - 1.0) < 1e-6


def test_decode_mask_rejects_garbage():
    try:
        decode_mask_to_float("data:image/png;base64,%%%not-base64%%%")
    except MaskError:
        pass
    else:
        raise AssertionError("Invalid mask payload must raise MaskError")


def test_decode_mask_rejects_noncanonical_data_urls():
    valid = _png_data_url(_rgba_mask(painted=True))
    for data_url in (
        valid.replace("data:image/png;base64,", "data:image/svg+xml;base64,"),
        valid.replace("base64,", "base64,%%%"),
    ):
        try:
            decode_mask_to_float(data_url)
        except MaskError:
            pass
        else:
            raise AssertionError("Noncanonical mask data URL must raise MaskError")


def test_decode_mask_rejects_oversized_dimensions_before_loading_pixels():
    oversized = Image.new("L", (4097, 1), 0)
    try:
        decode_mask_to_float(oversized)
    except MaskError:
        pass
    else:
        raise AssertionError("Oversized mask dimensions must raise MaskError")


def test_empty_mask_detection():
    empty = decode_mask_to_float(_png_data_url(_rgba_mask(painted=False)), target_size=(64, 64))
    assert empty is not None
    assert is_mask_empty(empty) is True

    painted = decode_mask_to_float(_png_data_url(_rgba_mask(painted=True)), target_size=(64, 64))
    assert painted is not None
    assert is_mask_empty(painted) is False


def test_composite_preserves_pixels_outside_the_mask():
    """The core guarantee: outside the mask the original survives untouched."""
    original = Image.new("RGB", (32, 32), (10, 20, 30))
    original.putpixel((20, 5), (200, 100, 50))
    generated = Image.new("RGB", (32, 32), (250, 240, 230))

    mask = np.zeros((32, 32), dtype=np.float32)
    mask[0:16, 0:16] = 1.0  # top-left quadrant is edited

    result = composite_with_mask(generated, original, mask)
    out = np.asarray(result.convert("RGB"))

    # Outside the mask -> identical to the original
    assert tuple(out[20, 20]) == (10, 20, 30)
    assert tuple(out[5, 20]) == (200, 100, 50)
    # Inside the mask -> the generated pixels
    assert tuple(out[2, 2]) == (250, 240, 230)


def test_composite_soft_edge_blends():
    original = Image.new("RGB", (8, 8), (0, 0, 0))
    generated = Image.new("RGB", (8, 8), (100, 200, 40))

    mask = np.full((8, 8), 0.5, dtype=np.float32)
    out = np.asarray(composite_with_mask(generated, original, mask).convert("RGB"))
    expected = (100 * 0.5 + 0 * 0.5, 200 * 0.5, 40 * 0.5)
    assert abs(int(out[0, 0][0]) - int(expected[0])) <= 1
    assert abs(int(out[0, 0][1]) - int(expected[1])) <= 1
    assert abs(int(out[0, 0][2]) - int(expected[2])) <= 1


def test_composite_preserves_source_alpha():
    original = Image.new("RGBA", (8, 8), (20, 40, 60, 0))
    original.putpixel((0, 0), (10, 30, 50, 64))
    original.putpixel((1, 0), (70, 90, 110, 255))
    generated = Image.new("RGB", (8, 8), (200, 210, 220))
    mask = np.ones((8, 8), dtype=np.float32)
    mask[0, :2] = 0.0

    result = composite_with_mask(generated, original, mask)
    out = np.asarray(result)
    assert result.mode == "RGBA"
    assert tuple(out[0, 0]) == (10, 30, 50, 64)
    assert tuple(out[0, 1]) == (70, 90, 110, 255)
    assert tuple(out[4, 4]) == (200, 210, 220, 0)


def test_composite_uses_original_resolution():
    original = Image.new("RGB", (100, 50), (1, 2, 3))
    generated = Image.new("RGB", (256, 256), (9, 9, 9))
    mask = np.zeros((50, 100), dtype=np.float32)

    result = composite_with_mask(generated, original, mask)
    assert result.size == (100, 50), "composite must keep the original frame size"
    assert tuple(np.asarray(result)[0, 0]) == (1, 2, 3)


def test_encode_mask_png_roundtrip():
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:8, 4:8] = 1.0
    png = encode_mask_png(mask)
    decoded = decode_mask_to_float(png, target_size=(16, 16))
    assert decoded is not None
    assert decoded[6, 6] > 0.99
    assert decoded[0, 0] == 0.0


def test_engine_mask_support_table():
    """Engines with a confirmed mask contract may be used."""
    assert engine_mask_support("quantfunc") is True
    assert engine_mask_arg("quantfunc") == "--mask-image"
    assert engine_mask_support("sd_cpp") is True
    assert engine_mask_arg("sd_cpp") == "--mask"
    # Python engines take mask_image directly and have no CLI flag.
    for engine in ("nunchaku", "orbitquant"):
        assert engine_mask_support(engine) is True
        assert engine_mask_arg(engine) is None
    for engine in ("sdnq", ""):
        assert engine_mask_support(engine) is False
        assert engine_mask_arg(engine) is None
    assert engine_mask_support(None) is False


def test_model_supports_mask():
    from backend.registry.loader import get_registry, model_supports_mask

    registry = get_registry()
    qwen_edit = registry.get("qwen_image_edit")
    assert model_supports_mask(qwen_edit) is True, "Qwen-Image-Edit must be mask capable"

    # QuantFunc FLUX cards stay untouched: QF masking is intentionally off.
    for model_id in ("flux_klein_4b", "flux_klein_9b"):
        spec = registry.get(model_id)
        assert spec is not None
        assert model_supports_mask(spec) is False, f"{model_id} must stay non-maskable"

    # FLUX.2 Klein Nunchaku / OrbitQuant and SD.CPP cards opt in explicitly.
    for model_id in (
        "flux_2_klein_4b_nunchaku",
        "flux_2_klein_4b_orbitquant",
        "flux_2_klein_9b_nunchaku",
        "flux_2_klein_9b_kv_nunchaku",
        "flux_2_klein_9b_orbitquant",
        "flux_2_klein_base_9b_gguf",
        "flux_3_image_klein_gguf",
        "flux_schnell_gguf",
    ):
        spec = registry.get(model_id)
        assert spec is not None, f"{model_id} must be registered"
        assert model_supports_mask(spec) is True, f"{model_id} must accept masks"

    # FLUX.3 Nunchaku uses a ModularPipeline without inpaint wiring.
    assert model_supports_mask(registry.get("flux_3_image_klein_nunchaku")) is False

    assert model_supports_mask(None) is False


def test_registry_mask_capable_listing():
    from backend.registry.loader import get_registry

    registry = get_registry()
    capable = registry.list_mask_capable()
    capable_ids = {spec.id for spec in capable}
    assert capable, "at least one mask capable model must be registered"
    assert "qwen_image_edit" in capable_ids
    for model_id in (
        "flux_2_klein_4b_nunchaku",
        "flux_2_klein_4b_orbitquant",
        "flux_2_klein_9b_nunchaku",
        "flux_2_klein_9b_kv_nunchaku",
        "flux_2_klein_9b_orbitquant",
        "flux_2_klein_base_9b_gguf",
        "flux_3_image_klein_gguf",
        "flux_schnell_gguf",
    ):
        assert model_id in capable_ids
    for model_id in ("flux_klein_4b", "flux_klein_9b", "flux_3_image_klein_nunchaku"):
        assert model_id not in capable_ids
    for spec in capable:
        assert spec.supports_mask is True
        assert engine_mask_support(spec.engine) is True


def test_feather_mask_dilates_and_keeps_background_zero():
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[14:18, 14:18] = 1.0

    grown = feather_mask(mask, grow=3)
    assert grown[12, 12] > 0.99, "dilation must expand the painted region"
    assert grown[0, 0] == 0.0, "background must stay untouched"

    blurred = feather_mask(mask, blur=2.0)
    assert blurred[16, 16] > 0.0
    assert blurred[0, 0] == 0.0
    assert float(blurred.max()) <= 1.0


def test_prepare_mask_arrays_returns_raw_and_engine_versions():
    raw, engine = prepare_mask_arrays(
        _png_data_url(_rgba_mask(painted=True)),
        target_size=(64, 64),
        grow=2,
        blur=1.0,
    )
    assert raw.shape == (64, 64)
    assert engine.shape == (64, 64)
    assert raw[32, 32] > 0.99
    assert engine[32, 32] > 0.99
    assert raw[0, 0] == 0.0
    assert engine[0, 0] == 0.0
    assert mask_coverage(engine) >= mask_coverage(raw), "grow must not shrink coverage"


def test_workers_wire_the_mask_into_the_engine():
    """The engine adapters must actually forward the prepared mask."""
    extras_src = (root_dir / "backend" / "extras.py").read_text(encoding="utf-8")
    sd_cpp_src = (root_dir / "backend" / "sd_cpp.py").read_text(encoding="utf-8")

    assert "Flux2KleinInpaintPipeline" in extras_src
    assert 'call_kwargs["mask_image"]' in extras_src
    assert '"mask_path"' in extras_src
    assert "prepare_mask_arrays" in extras_src
    # mask_strength == 0.0 must survive the payload round-trip.
    assert "mask_strength_value" in extras_src

    assert "prepare_mask_arrays" in sd_cpp_src
    assert 'cmd.extend(["--mask"' in sd_cpp_src
    assert 'cmd.extend(["--strength"' in sd_cpp_src
    assert "mask_strength_value" in sd_cpp_src


def test_generate_backend_filter_does_not_empty_engine_cards():
    """/api/generate used to filter transformers by the request backend
    (default "lighting"), which is a QuantFunc-only concept.  Nunchaku /
    OrbitQuant / SD.CPP cards expose the ENGINE name as the transformer
    backend, so the filter emptied the list and generation failed with
    "No compatible transformer found for backend=lighting".
    """
    from backend.registry.loader import get_registry

    registry = get_registry()
    for model_id in (
        "flux_2_klein_4b_nunchaku",
        "flux_2_klein_4b_orbitquant",
        "flux_2_klein_9b_nunchaku",
        "flux_2_klein_9b_kv_nunchaku",
        "flux_2_klein_9b_orbitquant",
        "flux_2_klein_base_9b_gguf",
        "flux_3_image_klein_gguf",
        "flux_schnell_gguf",
    ):
        spec = registry.get(model_id)
        assert spec is not None, model_id
        assert spec.get_transformers_for_tier(86, backend="lighting") == [], model_id
        assert spec.get_transformers_for_tier(86, backend=None), model_id

    # QuantFunc cards keep the lighting/svdq filtering behaviour.
    qwen = registry.get("qwen_image_edit")
    assert qwen.get_transformers_for_tier(86, backend="lighting")


def test_custom_model_card_can_declare_mask_support():
    """The card editor must be able to flip the mask capability of a model."""
    from backend.custom_models import _apply_card_updates

    manifest = {"id": "demo", "name": "Demo", "pipeline": "text_to_image", "engine": "quantfunc"}
    updated = _apply_card_updates(manifest, {"supports_mask": True})
    assert updated.get("supports_mask") is True

    updated = _apply_card_updates(updated, {"supports_mask": False})
    assert updated.get("supports_mask") is False
