"""Tests for Gradio Live (Hugging Face) tunnel provider and share API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from starlette.testclient import TestClient

from backend.share import PROVIDERS, GradioProvider, CloudflaredProvider, get_share_manager
from backend.settings import get_settings


def test_gradio_provider_registered():
    assert "gradio" in PROVIDERS
    assert "cloudflared" in PROVIDERS
    assert PROVIDERS["gradio"] is GradioProvider
    assert GradioProvider.id == "gradio"
    assert "Gradio Live" in GradioProvider.label


def test_gradio_url_parser():
    provider = GradioProvider()
    log_line = "2026/11/02 15:49:37 [I] [proxy.go:123] start proxy success: https://e559ef6ae0c7b99a6a.gradio.live"
    url = provider.parse_url(log_line)
    assert url == "https://e559ef6ae0c7b99a6a.gradio.live"

    no_url = "2026/11/02 15:49:37 [W] [service.go:132] login to server failed"
    assert provider.parse_url(no_url) is None


def test_share_manager_status_has_providers():
    mgr = get_share_manager()
    status = mgr.status()
    assert "providers" in status
    provider_ids = [p["id"] for p in status["providers"]]
    assert "cloudflared" in provider_ids
    assert "gradio" in provider_ids


def test_share_options_api():
    from app import create_app
    client = TestClient(create_app())

    # Switch to gradio
    resp = client.post("/api/share/options", json={"share_tunnel_provider": "gradio"})
    assert resp.status_code == 200
    assert get_settings().share_tunnel_provider == "gradio"

    # Switch back to cloudflared
    resp = client.post("/api/share/options", json={"share_tunnel_provider": "cloudflared"})
    assert resp.status_code == 200
    assert get_settings().share_tunnel_provider == "cloudflared"
