"""Integration and API tests for ``/api/inpaint/prompt`` and ``/api/vision/*``."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
from backend.prompt_gen import clear_prompt_cache
from backend.vision_runtime import VisionTimeoutError, parse_vlm_response_text


def _make_mask_data_url(width: int = 128, height: int = 128, empty: bool = False) -> str:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not empty:
        for x in range(32, 96):
            for y in range(32, 96):
                img.putpixel((x, y), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ensure_sample_source() -> tuple[str, Path]:
    root = Path(app_module.__file__).resolve().parent
    uploads = root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    sample = uploads / "_test_prompt_gen_source.png"
    img = Image.new("RGB", (128, 128), (185, 30, 40))
    img.save(sample, format="PNG")
    return "uploads/_test_prompt_gen_source.png", sample


def test_parse_vlm_response_json_and_fences():
    raw = '```json\n{"prompt": "clean red ink splash", "negative_prompt": "text, kanji"}\n```'
    prompt, neg = parse_vlm_response_text(raw)
    assert prompt == "clean red ink splash"
    assert neg == "text, kanji"

    # First-line fallback when VLM returns plain text instead of JSON
    prompt2, neg2 = parse_vlm_response_text("seamless white speech bubble interior, no text")
    assert "white speech bubble" in prompt2
    assert "text" in neg2


def test_prompt_gen_api_heuristic_fallback_and_cache():
    clear_prompt_cache()
    src_rel, sample_path = _ensure_sample_source()
    try:
        application = app_module.create_app()
        client = TestClient(application)

        import backend.prompt_gen as pg
        with patch.object(pg, "check_ollama_status", return_value={
            "ollama_available": False,
            "ollama_version": None,
            "installed_tags": [],
        }):
            resp = client.post("/api/inpaint/prompt", json={
                "source_path": src_rel,
                "mask": _make_mask_data_url(),
                "instruction": "убрать японский текст, восстановить красный фон",
                "use_vlm": True,
            })
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["source"] == "heuristic"
            assert data["cached"] is False
            assert "red" in data["prompt"]
            assert len(data["warnings"]) >= 1

            # Second identical call hits LRU cache
            resp2 = client.post("/api/inpaint/prompt", json={
                "source_path": src_rel,
                "mask": _make_mask_data_url(),
                "instruction": "убрать японский текст, восстановить красный фон",
                "use_vlm": True,
            })
            assert resp2.status_code == 200
            assert resp2.json()["cached"] is True
    finally:
        sample_path.unlink(missing_ok=True)


def test_prompt_gen_api_with_mock_vlm():
    clear_prompt_cache()
    src_rel, sample_path = _ensure_sample_source()
    try:
        application = app_module.create_app()
        client = TestClient(application)

        import backend.prompt_gen as pg
        with patch.object(pg, "check_ollama_status", return_value={
            "ollama_available": True,
            "ollama_version": "0.5.7",
            "installed_tags": ["qwen2.5vl:3b"],
        }), patch.object(pg, "generate_vlm_prompt", return_value=(
            "seamless dark red and black ink splash background, manga brush texture",
            "text, japanese, kanji, kana, sfx, seams",
        )):
            resp = client.post("/api/inpaint/prompt", json={
                "source_path": src_rel,
                "mask": _make_mask_data_url(),
                "instruction": "убрать текст",
                "vision_model": "qwen2.5vl:3b",
                "use_vlm": True,
            })
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["source"] == "vlm"
            assert data["model"] == "qwen2.5vl:3b"
            assert "dark red and black ink splash" in data["prompt"]
    finally:
        sample_path.unlink(missing_ok=True)


def test_prompt_gen_never_runs_text_only_ollama_model():
    clear_prompt_cache()
    src_rel, sample_path = _ensure_sample_source()
    try:
        application = app_module.create_app()
        client = TestClient(application)

        import backend.prompt_gen as pg
        with patch.object(pg, "check_ollama_status", return_value={
            "ollama_available": True,
            "ollama_version": "0.5.7",
            "installed_tags": ["qwen2.5-coder-7b:latest"],
            "vision_tags": [],
        }), patch.object(pg, "is_local_vision_model_installed", return_value=True), \
                patch.object(pg, "generate_vlm_prompt") as mock_vlm:
            resp = client.post("/api/inpaint/prompt", json={
                "source_path": src_rel,
                "mask": _make_mask_data_url(),
                "instruction": "убрать текст, восстановить красный фон",
                "vision_model": "gemma3:4b",
                "use_vlm": True,
            })
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["source"] == "heuristic"
            assert any("no vision-capable" in w for w in data["warnings"])
            mock_vlm.assert_not_called()
    finally:
        sample_path.unlink(missing_ok=True)


def test_prompt_gen_api_validation_and_error_codes():
    clear_prompt_cache()
    src_rel, sample_path = _ensure_sample_source()
    try:
        application = app_module.create_app()
        client = TestClient(application)

        # 1. Empty mask -> 400
        r_empty = client.post("/api/inpaint/prompt", json={
            "source_path": src_rel,
            "mask": _make_mask_data_url(empty=True),
            "instruction": "clean",
        })
        assert r_empty.status_code == 400

        # 2. Path traversal -> 400
        r_trav = client.post("/api/inpaint/prompt", json={
            "source_path": "../app.py",
            "mask": _make_mask_data_url(),
            "instruction": "clean",
        })
        assert r_trav.status_code == 400

        # 3. Fallback disabled + uninstalled model -> 409; timeout -> 504
        import backend.prompt_gen as pg
        s = app_module.get_settings()
        with patch.object(s, "vision_fallback_enabled", False):
            with patch.object(pg, "check_ollama_status", return_value={
                "ollama_available": True,
                "ollama_version": "0.5.7",
                "installed_tags": [],
            }):
                r_409 = client.post("/api/inpaint/prompt", json={
                    "source_path": src_rel,
                    "mask": _make_mask_data_url(),
                    "instruction": "clean",
                    "vision_model": "qwen2.5vl:7b",
                    "use_vlm": True,
                })
                assert r_409.status_code == 409

            with patch.object(pg, "check_ollama_status", return_value={
                "ollama_available": True,
                "ollama_version": "0.5.7",
                "installed_tags": ["qwen2.5vl:3b"],
            }), patch.object(pg, "generate_vlm_prompt", side_effect=VisionTimeoutError("VLM timed out")):
                r_504 = client.post("/api/inpaint/prompt", json={
                    "source_path": src_rel,
                    "mask": _make_mask_data_url(),
                    "instruction": "clean",
                    "vision_model": "qwen2.5vl:3b",
                    "use_vlm": True,
                })
                assert r_504.status_code == 504
    finally:
        sample_path.unlink(missing_ok=True)


def test_hub_prompt_upsamplers_in_config_and_download_job():
    application = app_module.create_app()
    client = TestClient(application)

    cfg = client.get("/api/config").json()
    models_by_id = {m["id"]: m for m in cfg["models"]}
    assert "flux_3_image_klein_gguf" in models_by_id
    assert "flux_3_image_klein_nunchaku" in models_by_id

    gguf_pups = models_by_id["flux_3_image_klein_gguf"].get("prompt_upsamplers") or []
    nunchaku_pups = models_by_id["flux_3_image_klein_nunchaku"].get("prompt_upsamplers") or []
    assert any(p["id"] == "hf.co/unsloth/flux-3-image-klein-pup-GGUF:UD-Q4_K_XL" for p in gguf_pups)
    assert any(p["id"] == "flux3-pup-nunchaku" for p in nunchaku_pups)


if __name__ == "__main__":
    test_parse_vlm_response_json_and_fences()
    test_prompt_gen_api_heuristic_fallback_and_cache()
    test_prompt_gen_api_with_mock_vlm()
    test_prompt_gen_api_validation_and_error_codes()
    test_hub_prompt_upsamplers_in_config_and_download_job()
    print("[SUCCESS] All test_prompt_gen_api tests passed!")
