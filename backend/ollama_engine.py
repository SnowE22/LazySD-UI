"""
Ollama Engine Manager for LazySD-UI.

Handles:
- Detecting Ollama installation on Windows (PATH, LocalAppData, ProgramFiles).
- Checking whether Ollama daemon (127.0.0.1:11434) is running.
- Starting and stopping ``ollama serve`` as a background service.
- Installing Ollama via winget or official installer.
- Pulling and managing Vision models (e.g. ``qwen2.5vl:3b``).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger("quantfunc.ollama_engine")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def find_ollama_binary() -> Optional[Path]:
    """Locate the ollama.exe binary on the host system."""
    # 0. Check bundled portable install in bin/windows/ollama
    bundled = Path(__file__).resolve().parent.parent / "bin" / "windows" / "ollama" / "ollama.exe"
    if bundled.is_file():
        return bundled

    # 1. Check PATH
    which_path = shutil.which("ollama")
    if which_path:
        p = Path(which_path)
        if p.is_file():
            return p

    # 2. Check standard Windows user install path
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        p = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if p.is_file():
            return p

    # 3. Check ProgramFiles
    prog_files = os.environ.get("ProgramFiles")
    if prog_files:
        p = Path(prog_files) / "Ollama" / "ollama.exe"
        if p.is_file():
            return p

    prog_files_x86 = os.environ.get("ProgramFiles(x86)")
    if prog_files_x86:
        p = Path(prog_files_x86) / "Ollama" / "ollama.exe"
        if p.is_file():
            return p

    return None


def is_ollama_installed() -> bool:
    """Return True if Ollama binary is found on disk."""
    return find_ollama_binary() is not None


def is_ollama_running(url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> bool:
    """Return True if the Ollama HTTP daemon responds."""
    try:
        base = url.rstrip("/")
        req = urllib.request.Request(
            f"{base}/api/version",
            headers={"User-Agent": "LazySD-UI"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_ollama_models(url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.5) -> List[str]:
    """List model tags available in Ollama."""
    try:
        base = url.rstrip("/")
        req = urllib.request.Request(
            f"{base}/api/tags",
            headers={"User-Agent": "LazySD-UI"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models") or []
            tags = []
            for m in models:
                tag = m.get("name") or m.get("model")
                if tag:
                    tags.append(str(tag).strip())
            return tags
    except Exception:
        return []


def get_ollama_status_dict(url: str = DEFAULT_OLLAMA_URL) -> Dict[str, Any]:
    """Return a unified status dictionary for the Settings → Engines tab."""
    binary = find_ollama_binary()
    running = is_ollama_running(url)
    models = get_installed_ollama_models(url) if running else []
    version = None

    if running:
        try:
            base = url.rstrip("/")
            req = urllib.request.Request(f"{base}/api/version", headers={"User-Agent": "LazySD-UI"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                version = data.get("version")
        except Exception:
            pass

    if not version and binary:
        try:
            res = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = (res.stdout or res.stderr or "").strip()
            # e.g. "ollama version is 0.4.0"
            version = out.replace("ollama version is", "").replace("ollama", "").strip() or out
        except Exception:
            pass

    return {
        "installed": binary is not None,
        "running": running,
        "binary_path": str(binary) if binary else None,
        "version": version,
        "url": url,
        "models": models,
        "has_qwen_vl": any("qwen2.5vl" in m.lower() or "qwen2.5-vl" in m.lower() for m in models),
    }


def start_ollama_server(url: str = DEFAULT_OLLAMA_URL, timeout: float = 15.0) -> bool:
    """Start ``ollama serve`` in the background if not already active."""
    if is_ollama_running(url):
        return True

    binary = find_ollama_binary()
    if not binary:
        raise RuntimeError("Ollama binary not found on disk. Please install Ollama first.")

    logger.info("Starting Ollama background server: %s serve", binary)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    # Spawn daemon
    subprocess.Popen(
        [str(binary), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=str(binary.parent),
    )

    # Wait for daemon to respond
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_ollama_running(url, timeout=1.0):
            logger.info("Ollama daemon is now responsive at %s", url)
            return True
        time.sleep(0.5)

    return is_ollama_running(url)


def stop_ollama_server() -> bool:
    """Stop running Ollama process on Windows."""
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], capture_output=True, timeout=5)
            subprocess.run(["taskkill", "/IM", "ollama_app.exe", "/F"], capture_output=True, timeout=5)
            return True
        except Exception as exc:
            logger.warning("Could not stop ollama: %s", exc)
            return False
    else:
        try:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False


def download_and_extract_portable_ollama(
    progress_callback: Optional[Callable[[float, str], None]] = None,
    force: bool = False,
) -> Path:
    """Download and unpack standalone portable Ollama into bin/windows/ollama."""
    target_dir = Path(__file__).resolve().parent.parent / "bin" / "windows" / "ollama"
    target_exe = target_dir / "ollama.exe"
    if target_exe.is_file() and not force:
        return target_exe

    target_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = Path(__file__).resolve().parent.parent / "bin" / "windows" / "ollama.zip"
    url = "https://github.com/ollama/ollama/releases/download/v0.40.13/ollama-windows-amd64.zip"

    from .downloader import get_download_manager, DownloadStatus
    mgr = get_download_manager()

    def _on_prog(p: dict):
        if progress_callback:
            tot = float(p.get("total_bytes") or 1.0)
            dl = float(p.get("downloaded_bytes") or 0.0)
            pct = min(99.0, (dl / max(1.0, tot)) * 100.0) if tot > 0 else 0.0
            spd = (float(p.get("speed_bps") or 0)) / (1024 * 1024)
            progress_callback(pct, f"Downloading Ollama ({pct:.1f}% · {spd:.1f} MB/s)...")

    task = mgr.submit(
        task_id="dl_ollama_portable",
        url=url,
        destination=dest_zip,
        num_connections=8,
        progress_callback=_on_prog,
    )
    if task._main_thread:
        task._main_thread.join()

    if task.status != DownloadStatus.COMPLETE or not dest_zip.is_file():
        raise RuntimeError(f"Ollama download failed: {task.error_message or task.status}")

    if progress_callback:
        progress_callback(99.0, "Extracting Ollama...")

    import zipfile
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(target_dir)

    dest_zip.unlink(missing_ok=True)
    if not target_exe.is_file():
        raise RuntimeError(f"Extracted archive did not contain {target_exe}")

    if progress_callback:
        progress_callback(100.0, "Ollama installed ✓")
    return target_exe
