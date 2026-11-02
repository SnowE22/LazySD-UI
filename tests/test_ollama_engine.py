"""Unit tests for Ollama engine detection and status."""
from __future__ import annotations

from backend.ollama_engine import (
    find_ollama_binary,
    get_ollama_status_dict,
    is_ollama_installed,
    is_ollama_running,
)


def test_ollama_detection_and_status():
    assert is_ollama_installed() is True
    binary = find_ollama_binary()
    assert binary is not None
    assert binary.is_file()

    st = get_ollama_status_dict()
    assert st["installed"] is True
    assert "version" in st
    assert "url" in st


if __name__ == "__main__":
    test_ollama_detection_and_status()
    print("[SUCCESS] Ollama engine tests passed!")
