"""Unit tests for vision model catalog and status tracking (``backend/vision_models.py``)."""
from __future__ import annotations

from backend.vision_models import (
    VISION_MODELS_CATALOG,
    build_vision_models_response,
    clear_download_state,
    get_catalog_entry,
    is_tag_matching,
    set_download_state,
)


def test_catalog_contains_required_vision_models():
    ids = {m["id"] for m in VISION_MODELS_CATALOG}
    assert {"qwen2.5vl:3b", "qwen2.5vl:7b", "minicpm-v:8b", "llava:7b"}.issubset(ids)
    qwen3b = get_catalog_entry("qwen2.5vl:3b")
    assert qwen3b is not None
    assert qwen3b["recommended"] is True
    assert "vision" in qwen3b["capabilities"]


def test_tag_matching_and_status_building():
    assert is_tag_matching("qwen2.5vl:3b", ["qwen2.5vl:3b", "llava:latest"])
    assert not is_tag_matching("qwen2.5vl:7b", ["qwen2.5vl:3b"])

    clear_download_state("minicpm-v:8b")
    resp = build_vision_models_response(
        installed_tags=["qwen2.5vl:3b"],
        ollama_available=True,
    )
    assert resp["available"] is True
    by_id = {m["id"]: m for m in resp["models"]}
    assert by_id["qwen2.5vl:3b"]["installed"] is True
    assert by_id["qwen2.5vl:3b"]["status"] == "installed"
    assert by_id["qwen2.5vl:7b"]["installed"] is False
    assert by_id["qwen2.5vl:7b"]["status"] == "not_installed"


def test_download_state_transitions():
    mid = "minicpm-v:8b"
    try:
        st = set_download_state(mid, status="downloading", progress=42.5, detail="pulling layer")
        assert st["status"] == "downloading"
        assert st["progress"] == 42.5

        resp = build_vision_models_response(installed_tags=[], ollama_available=True)
        by_id = {m["id"]: m for m in resp["models"]}
        assert by_id[mid]["status"] == "downloading"
        assert by_id[mid]["progress"] == 42.5

        set_download_state(mid, status="error", error="network failure")
        resp2 = build_vision_models_response(installed_tags=[], ollama_available=True)
        by_id2 = {m["id"]: m for m in resp2["models"]}
        assert by_id2[mid]["status"] == "error"
        assert by_id2[mid]["error"] == "network failure"
    finally:
        clear_download_state(mid)
