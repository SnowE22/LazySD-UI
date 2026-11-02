"""Tests for strict target_architecture compatibility guardrails across engines.

Verifies that LoRA models bound to an exact base architecture (e.g. flux_2_klein_9b)
are compatible across all engines supporting that architecture (Nunchaku, SD.CPP / GGUF, QuantFunc),
while being strictly rejected on different architectures (such as flux_2_klein_4b,
flux_3_image_klein, or qwen_image_2_1).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.registry.loader import ModelSpec
from backend.lora import (
    check_lora_compatibility,
    update_lora_metadata,
)


# ---------------------------------------------------------------------------
# 1. ModelSpec target_architecture inference and explicit override
# ---------------------------------------------------------------------------

def test_modelspec_target_architecture_derivation():
    # 9B Klein across engines
    m9b_nunchaku = ModelSpec(id="flux_2_klein_9b_nunchaku", name="Klein 9B Nunchaku", pipeline="text_to_image", family="klein", engine="nunchaku")
    assert m9b_nunchaku.get_target_architecture() == "flux_2_klein_9b"

    m9b_orbitquant = ModelSpec(id="flux_2_klein_9b_orbitquant", name="Klein 9B OrbitQuant", pipeline="text_to_image", family="klein", engine="orbitquant")
    assert m9b_orbitquant.get_target_architecture() == "flux_2_klein_9b"

    m9b_gguf = ModelSpec(id="flux_2_klein_base_9b_gguf", name="Klein 9B GGUF", pipeline="text_to_image", family="flux", engine="sd_cpp")
    assert m9b_gguf.get_target_architecture() == "flux_2_klein_9b"

    # 4B Klein
    m4b = ModelSpec(id="flux_2_klein_4b_nunchaku", name="Klein 4B Nunchaku", pipeline="text_to_image", family="klein", engine="nunchaku")
    assert m4b.get_target_architecture() == "flux_2_klein_4b"

    # FLUX.3 Image Klein
    mflux3 = ModelSpec(id="flux_3_image_klein_nunchaku", name="FLUX.3 Klein", pipeline="text_to_image", family="flux", engine="nunchaku")
    assert mflux3.get_target_architecture() == "flux_3_image_klein"

    # Qwen & SDXL
    mqwen = ModelSpec(id="qwen_image_2_1_nunchaku", name="Qwen Image 2.1", pipeline="text_to_image", family="qwen", engine="nunchaku")
    assert mqwen.get_target_architecture() == "qwen_image_2_1"

    msdxl = ModelSpec(id="sdxl_base", name="SDXL Base", pipeline="text_to_image", family="sdxl", engine="diffusers")
    assert msdxl.get_target_architecture() == "sdxl"

    # Explicit override takes precedence
    mcustom = ModelSpec(id="custom_model", name="Custom", pipeline="text_to_image", family="flux", target_architecture="custom_v1")
    assert mcustom.get_target_architecture() == "custom_v1"


# ---------------------------------------------------------------------------
# 2. Metadata persistence for target_architecture
# ---------------------------------------------------------------------------

def test_lora_metadata_update_target_architecture(tmp_path: Path):
    dummy_lora = tmp_path / "my_lora.safetensors"
    dummy_lora.write_bytes(b"dummy_weights")
    stored_meta = {}

    with patch("backend.lora.get_loras_dir", return_value=tmp_path), \
         patch("backend.lora.load_all_lora_meta", return_value=stored_meta), \
         patch("backend.lora.save_all_lora_meta") as mock_save:

        res = update_lora_metadata("my_lora.safetensors", {
            "target_architecture": "flux_2_klein_9b",
            "trigger": "klein_style",
            "scale_default": 0.85,
        })

        assert res["target_architecture"] == "flux_2_klein_9b"
        assert res["trigger"] == "klein_style"
        assert res["scale_default"] == 0.85
        assert mock_save.called
        saved_dict = mock_save.call_args[0][0]
        assert saved_dict["my_lora.safetensors"]["target_architecture"] == "flux_2_klein_9b"


# ---------------------------------------------------------------------------
# 3. LoRA Compatibility checks: match across engines
# ---------------------------------------------------------------------------

def test_check_lora_compatibility_cross_engine_same_architecture(tmp_path: Path):
    lora_file = tmp_path / "flux2_klein_9b_art.safetensors"
    lora_file.write_bytes(b"dummy")

    m9b_nunchaku = ModelSpec(id="flux_2_klein_9b_nunchaku", name="Klein 9B Nunchaku", pipeline="text_to_image", family="klein", engine="nunchaku")
    m9b_gguf = ModelSpec(id="flux_2_klein_base_9b_gguf", name="Klein 9B GGUF", pipeline="text_to_image", family="flux", engine="sd_cpp")
    m9b_quantfunc = ModelSpec(id="flux_2_klein_9b_quantfunc", name="Klein 9B QuantFunc", pipeline="text_to_image", family="klein", engine="quantfunc")

    fake_meta = {
        "flux2_klein_9b_art.safetensors": {
            "target_architecture": "flux_2_klein_9b",
            "family": "klein",
        }
    }

    with patch("backend.lora.get_loras_dir", return_value=tmp_path), \
         patch("backend.lora.load_all_lora_meta", return_value=fake_meta):
        # Engine 1: Nunchaku
        res_nunchaku = check_lora_compatibility("flux2_klein_9b_art.safetensors", m9b_nunchaku)
        assert res_nunchaku["compatible"] is True
        assert res_nunchaku["match_level"] == "target_architecture"
        assert "100% совместимо" in res_nunchaku["reason"]

        # Engine 2: GGUF / sd_cpp (different family tag in spec, but identical target_architecture)
        res_gguf = check_lora_compatibility("flux2_klein_9b_art.safetensors", m9b_gguf)
        assert res_gguf["compatible"] is True
        assert res_gguf["match_level"] == "target_architecture"
        assert "100% совместимо" in res_gguf["reason"]

        # Engine 3: QuantFunc
        res_qf = check_lora_compatibility("flux2_klein_9b_art.safetensors", m9b_quantfunc)
        assert res_qf["compatible"] is True
        assert res_qf["match_level"] == "target_architecture"


# ---------------------------------------------------------------------------
# 4. LoRA Compatibility checks: strict mismatch prevention
# ---------------------------------------------------------------------------

def test_check_lora_compatibility_mismatch_rejection(tmp_path: Path):
    lora_file = tmp_path / "flux2_klein_9b_art.safetensors"
    lora_file.write_bytes(b"dummy")

    m4b_nunchaku = ModelSpec(id="flux_2_klein_4b_nunchaku", name="Klein 4B Nunchaku", pipeline="text_to_image", family="klein", engine="nunchaku")
    mflux3 = ModelSpec(id="flux_3_image_klein_nunchaku", name="FLUX.3 Klein", pipeline="text_to_image", family="flux", engine="nunchaku")
    mqwen = ModelSpec(id="qwen_image_2_1_nunchaku", name="Qwen 2.1", pipeline="text_to_image", family="qwen", engine="nunchaku")

    fake_meta = {
        "flux2_klein_9b_art.safetensors": {
            "target_architecture": "flux_2_klein_9b",
            "family": "klein",
        }
    }

    with patch("backend.lora.get_loras_dir", return_value=tmp_path), \
         patch("backend.lora.load_all_lora_meta", return_value=fake_meta):
        # 9B LoRA on 4B model -> REJECTED even though both share broad 'klein' family
        res_4b = check_lora_compatibility("flux2_klein_9b_art.safetensors", m4b_nunchaku)
        assert res_4b["compatible"] is False
        assert "Несовместимая архитектура весов" in res_4b["reason"]
        assert "flux_2_klein_9b" in res_4b["reason"]
        assert "flux_2_klein_4b" in res_4b["reason"]

        # 9B LoRA on FLUX.3 -> REJECTED
        res_flux3 = check_lora_compatibility("flux2_klein_9b_art.safetensors", mflux3)
        assert res_flux3["compatible"] is False
        assert "Несовместимая архитектура весов" in res_flux3["reason"]

        # 9B LoRA on Qwen -> REJECTED
        res_qwen = check_lora_compatibility("flux2_klein_9b_art.safetensors", mqwen)
        assert res_qwen["compatible"] is False
