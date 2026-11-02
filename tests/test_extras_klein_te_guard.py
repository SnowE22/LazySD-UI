import json
from pathlib import Path

from backend.extras import (
    _default_nunchaku_te_spec,
    _detect_klein_size_tag,
    _is_compatible_extras_text_encoder,
    _is_compatible_peer_base_dir,
)


def test_detect_klein_size_tag():
    assert _detect_klein_size_tag("flux_2_klein_9b_nunchaku", "tonera/FLUX.2-klein-9B-Nunchaku") == "9b"
    assert _detect_klein_size_tag("flux_2_klein_9b_kv_nunchaku") == "9b"
    assert _detect_klein_size_tag("flux_2_klein_4b_nunchaku", "tonera/FLUX.2-klein-4B-Nunchaku") == "4b"
    assert _detect_klein_size_tag("flux_3_image_klein_nunchaku", "tonera/flux-3-image-klein-nunchaku") == "flux3"


def test_is_compatible_extras_text_encoder_guards_4b_vs_9b():
    te_4b = {
        "id": "qwen3_4b_text_nunchaku",
        "name": "Qwen3-4B Text Encoder (Tonera SVDQ INT4)",
        "repo": "tonera/Qwen3-4B-text-Nunchaku",
        "file": "svdq-int4-Qwen3-4B-text-Nunchaku.safetensors",
        "source": {
            "repo_id": "tonera/Qwen3-4B-text-Nunchaku",
            "file_path": "svdq-int4-Qwen3-4B-text-Nunchaku.safetensors",
        },
    }
    te_8b = _default_nunchaku_te_spec("9b")
    te_gguf = {
        "id": "qwen3_8b_q8_0",
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q8_0.gguf",
    }

    # 9B model must accept 8B Nunchaku safetensors and reject 4B or GGUF
    assert _is_compatible_extras_text_encoder(te_8b, "9b") is True
    assert _is_compatible_extras_text_encoder(te_4b, "9b") is False
    assert _is_compatible_extras_text_encoder(te_gguf, "9b") is False
    assert _is_compatible_extras_text_encoder("svdq-int4-Qwen3-4B-text-Nunchaku.safetensors", "9b") is False
    assert _is_compatible_extras_text_encoder("svdq-int4-Qwen3-text-Nunchaku.safetensors", "9b") is True

    # 4B model must accept 4B Nunchaku safetensors and reject 8B or GGUF
    assert _is_compatible_extras_text_encoder(te_4b, "4b") is True
    assert _is_compatible_extras_text_encoder(te_8b, "4b") is False
    assert _is_compatible_extras_text_encoder(te_gguf, "4b") is False


def test_is_compatible_peer_base_dir_prevents_cross_size_borrowing():
    assert _is_compatible_peer_base_dir("FLUX.2-klein-9B-OrbitQuant-W4A4", "klein", "9b") is True
    assert _is_compatible_peer_base_dir("FLUX.2-klein-4B-OrbitQuant-W4A4", "klein", "9b") is False
    assert _is_compatible_peer_base_dir("flux-3-image-klein-nunchaku", "klein", "9b") is False
    assert _is_compatible_peer_base_dir("FLUX.2-klein-4B-Nunchaku", "klein", "4b") is True
    assert _is_compatible_peer_base_dir("FLUX.2-klein-9B-Nunchaku", "klein", "4b") is False


def test_klein_9b_model_cards_use_8b_text_encoder():
    models_dir = Path(__file__).resolve().parent.parent / "backend" / "registry" / "models"
    for card_name in (
        "flux_2_klein_9b_nunchaku.json",
        "flux_2_klein_9b_kv_nunchaku.json",
        "flux_2_klein_9b_orbitquant.json",
    ):
        card_path = models_dir / card_name
        data = json.loads(card_path.read_text(encoding="utf-8"))
        tes = data.get("text_encoders", [])
        assert len(tes) >= 1, f"{card_name} has no text_encoders"
        for te in tes:
            assert _is_compatible_extras_text_encoder(te, "9b"), f"Incompatible TE in {card_name}: {te}"
            assert te.get("file") == "svdq-int4-Qwen3-text-Nunchaku.safetensors"
            assert (te.get("source") or {}).get("repo_id") == "tonera/Qwen3-text-Nunchaku"
        ea = data.get("engine_args", {})
        assert ea.get("llm") == "svdq-int4-Qwen3-text-Nunchaku.safetensors"
        assert ea.get("llm_repo") == "tonera/Qwen3-text-Nunchaku"
