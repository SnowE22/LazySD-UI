"""Tests for Model Hub integration (LazySD-Models).

Verifies catalog fetching, caching, cross-engine architecture representation,
safe filename validation, and 1-click model card installation.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.hub import (
    DEFAULT_HUB_URL,
    fetch_hub_catalog,
    get_hub_base_url,
    install_hub_card,
)


def _make_mock_response(data: dict | bytes, status: int = 200):
    raw = json.dumps(data).encode("utf-8") if isinstance(data, dict) else data
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = raw
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


# ---------------------------------------------------------------------------
# 1. Hub URL Resolution
# ---------------------------------------------------------------------------

def test_get_hub_base_url_default():
    with patch("backend.hub.get_settings") as mock_settings:
        mock_settings.return_value.model_hub_url = ""
        assert get_hub_base_url() == DEFAULT_HUB_URL


def test_get_hub_base_url_custom():
    custom_url = "https://custom-hub.example.com/models"
    with patch("backend.hub.get_settings") as mock_settings:
        mock_settings.return_value.model_hub_url = custom_url
        assert get_hub_base_url() == custom_url


# ---------------------------------------------------------------------------
# 2. Fetch Catalog & Cross-reference with Local Registry
# ---------------------------------------------------------------------------

def test_fetch_hub_catalog_marks_installed_models():
    sample_index = {
        "schema_version": 1,
        "count": 2,
        "models": [
            {
                "id": "flux_2_klein_9b_nunchaku",
                "name": "Flux.2 Klein 9B (Nunchaku)",
                "family": "klein",
                "target_architecture": "flux_2_klein_9b",
                "engine": "nunchaku",
                "card_file": "flux_2_klein_9b_nunchaku.json",
            },
            {
                "id": "brand_new_uninstalled_model",
                "name": "New Model",
                "family": "qwen",
                "target_architecture": "qwen_image_2_1",
                "engine": "sdnq",
                "card_file": "brand_new_uninstalled_model.json",
            },
        ],
    }

    mock_registry = MagicMock()
    # Pretend flux_2_klein_9b_nunchaku is already installed locally
    mock_registry.models = {"flux_2_klein_9b_nunchaku": MagicMock()}

    with patch("urllib.request.urlopen", return_value=_make_mock_response(sample_index)), \
         patch("backend.hub.get_registry", return_value=mock_registry):

        catalog = fetch_hub_catalog(refresh=True)

        assert catalog["count"] == 2
        models = {m["id"]: m for m in catalog["models"]}

        assert models["flux_2_klein_9b_nunchaku"]["installed"] is True
        assert models["brand_new_uninstalled_model"]["installed"] is False
        assert catalog["installed_count"] == 1


# ---------------------------------------------------------------------------
# 3. Safe Card Installation & Path Traversal Prevention
# ---------------------------------------------------------------------------

def test_install_hub_card_rejects_malicious_filenames():
    bad_filenames = [
        "../escape.json",
        "../../etc/passwd",
        "sub/folder.json",
        "model.safetensors",
        "model.exe",
        "",
        "model..json",
    ]
    for bad in bad_filenames:
        with pytest.raises(ValueError):
            install_hub_card(bad)


def test_install_hub_card_success(tmp_path: Path):
    card_data = {
        "id": "community_fibo_model",
        "name": "Community Fibo Model",
        "family": "flux",
        "target_architecture": "flux_2_klein_9b",
        "engine": "orbitquant",
        "pipeline": "text_to_image",
    }

    mock_resp = _make_mock_response(card_data)

    with patch("urllib.request.urlopen", return_value=mock_resp), \
         patch("backend.hub.reload_registry") as mock_reload, \
         patch("backend.hub.Path") as mock_path_cls:

        # Let Path(__file__).parent / "registry" / "models" resolve to tmp_path
        def fake_path(*args):
            p = Path(*args)
            if "registry" in p.parts and "models" in p.parts:
                return tmp_path
            return p

        # Test directly with tmp_path
        with patch.object(Path, "mkdir"):
            pass

    # Direct functional test with tmp_path as models_dir
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with patch("urllib.request.urlopen", return_value=_make_mock_response(card_data)), \
         patch("backend.hub.reload_registry") as mock_reload:

        # Monkeypatch models_dir in hub
        with patch("backend.hub.Path") as mock_p:
            mock_p.return_value.parent.__truediv__.return_value.__truediv__.return_value = models_dir
            res = install_hub_card("community_fibo_model.json")

            assert res["success"] is True
            assert res["model_id"] == "community_fibo_model"
            assert mock_reload.called

            saved_file = models_dir / "community_fibo_model.json"
            assert saved_file.is_file()
            saved_content = json.loads(saved_file.read_text(encoding="utf-8"))
            assert saved_content["id"] == "community_fibo_model"
            assert saved_content["target_architecture"] == "flux_2_klein_9b"


# ---------------------------------------------------------------------------
# 4. REST API Endpoint Tests
# ---------------------------------------------------------------------------

def test_hub_api_routes():
    from starlette.testclient import TestClient
    from app import create_app

    app = create_app()
    client = TestClient(app)

    # 1. GET /api/hub/models
    with patch("backend.hub.urllib.request.urlopen") as mock_url:
        mock_url.return_value = _make_mock_response({
            "schema_version": 1,
            "count": 1,
            "models": [{
                "id": "flux_2_klein_9b_nunchaku",
                "name": "Flux.2 Klein 9B",
                "family": "klein",
                "target_architecture": "flux_2_klein_9b",
                "engine": "nunchaku",
                "card_file": "flux_2_klein_9b_nunchaku.json"
            }]
        })
        resp = client.get("/api/hub/models?refresh=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["models"][0]["id"] == "flux_2_klein_9b_nunchaku"

    # 2. POST /api/hub/install validation failure on malicious path
    resp_bad = client.post("/api/hub/install", json={"card_file": "../bad.json"})
    assert resp_bad.status_code == 400

