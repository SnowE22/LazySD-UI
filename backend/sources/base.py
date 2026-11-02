"""
ModelSource — abstract interface for model repositories.

Two implementations: HuggingFaceSource and ModelScopeSource. Both follow
the same API so the rest of the backend can switch between them transparently.

Key design:
    - download_file / download_snapshot both support progress callbacks
    - resume is handled natively by the SDKs (via .incomplete files)
    - SHA-256 is exposed via RemoteFile.sha256 when available (LFS only on HF)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ---------- Progress callback type ----------
# Called with (downloaded_bytes: int, total_bytes: int, speed_bps: float)
ProgressCallback = Callable[[int, int, float], None]


# ---------- Data classes ----------
@dataclass
class RemoteFile:
    """A file in a remote model repository."""
    path: str                           # Relative path inside the repo (e.g. "transformer/klein-9b-50x-above-lighting-4steps.safetensors")
    size: int = 0                       # Bytes (0 if unknown)
    sha256: Optional[str] = None        # SHA-256 hash if known (LFS only on HF)
    is_lfs: bool = False                # LFS-tracked (large file)
    revision: str = "main"              # Git revision (main / master / specific commit)


@dataclass
class ModelMetadata:
    """Metadata about a remote model repository."""
    repo_id: str                        # "QuantFunc/Klein-9B-Series"
    description: str = ""
    license: str = ""
    files_count: int = 0
    size_total: int = 0                 # Bytes
    last_modified: Optional[str] = None
    tags: list[str] = field(default_factory=list)


# ---------- Abstract interface ----------
class ModelSource(ABC):
    """Abstract source of models — HuggingFace, ModelScope, or local dir."""

    source_type: str = "abstract"

    @abstractmethod
    def list_files(self, repo_id: str, revision: str = "main") -> list[RemoteFile]:
        """List all files in a remote repository."""
        ...

    @abstractmethod
    def file_exists(self, repo_id: str, file_path: str, revision: str = "main") -> bool:
        """Check if a specific file exists in the remote repository."""
        ...

    @abstractmethod
    def get_metadata(self, repo_id: str, revision: str = "main") -> ModelMetadata:
        """Fetch metadata about the repository."""
        ...

    @abstractmethod
    def download_file(
        self,
        repo_id: str,
        file_path: str,
        local_dir: Path,
        revision: str = "main",
        progress_callback: Optional[ProgressCallback] = None,
        job_id: Optional[str] = None,
    ) -> Path:
        """Download a single file. Returns the local path.

        Supports multi-connection chunked downloading with resume and pause.
        """
        ...

    @abstractmethod
    def download_snapshot(
        self,
        repo_id: str,
        local_dir: Path,
        allow_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        revision: str = "main",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download an entire repository (or filtered subset). Returns the local dir.

        allow_patterns: list of glob patterns like ["transformer/*", "*.json"]
        Resume is handled natively by the SDK.
        """
        ...

    # ---------- Marker file for "download complete" ----------
    # QuantFunc plugin uses .quantfunc_download_complete — we reuse the same convention.
    COMPLETE_MARKER = ".quantfunc_download_complete"

    def mark_complete(self, local_dir: Path, repo_id: str) -> None:
        """Write a marker file indicating a successful snapshot download."""
        marker = local_dir / self.COMPLETE_MARKER
        marker.write_text(f"repo_id={repo_id}\n", encoding="utf-8")

    def is_complete(self, local_dir: Path) -> bool:
        """Check if a snapshot was fully downloaded (marker file present)."""
        return (local_dir / self.COMPLETE_MARKER).exists()

    def clear_complete_marker(self, local_dir: Path) -> None:
        """Remove the marker file (e.g. after partial re-download)."""
        marker = local_dir / self.COMPLETE_MARKER
        if marker.exists():
            marker.unlink()


# ---------- Factory ----------
def get_source(source_type: str, token: Optional[str] = None) -> ModelSource:
    """Factory — get a ModelSource instance by type string.

    Args:
        source_type: "huggingface" or "modelscope"
        token: Optional auth token

    Returns:
        ModelSource instance

    Raises:
        ImportError: If the required SDK is not installed
        ValueError: If source_type is unknown
    """
    if source_type == "huggingface":
        from .huggingface import HuggingFaceSource
        return HuggingFaceSource(token=token)
    elif source_type == "modelscope":
        from .modelscope import ModelScopeSource
        return ModelScopeSource(token=token)
    else:
        raise ValueError(f"Unknown source type: {source_type!r}. Expected 'huggingface' or 'modelscope'.")
