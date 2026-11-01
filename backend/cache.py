"""
Cache manager — tracks local model files, computes SHA-256 checksums,
maintains the state machine: Installed / Incomplete / Missing / Corrupted.

Closes the gap in QuantFunc plugin (which only checksums engine binary,
not model weights).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .settings import get_settings


# ---------- File states ----------
class FileState(str, Enum):
    INSTALLED = "installed"          # File exists + checksum OK
    INCOMPLETE = "incomplete"        # .incomplete file exists (partial download)
    MISSING = "missing"              # File doesn't exist
    CORRUPTED = "corrupted"          # File exists but checksum mismatch
    DOWNLOADABLE = "downloadable"    # Not in cache, available on remote


@dataclass
class CacheEntry:
    """State of a single file in the cache."""
    path: Path
    expected_size: int
    expected_sha256: Optional[str]
    state: FileState
    actual_size: int = 0
    actual_sha256: Optional[str] = None

    @property
    def state_label(self) -> str:
        """Human-readable state for UI."""
        return {
            FileState.INSTALLED: "✓ installed",
            FileState.INCOMPLETE: "⚠ incomplete",
            FileState.MISSING: "✗ missing",
            FileState.CORRUPTED: "✗ corrupted",
            FileState.DOWNLOADABLE: "↓ downloadable",
        }[self.state]

    @property
    def state_color(self) -> str:
        """CSS color for state badge."""
        return {
            FileState.INSTALLED: "#10B981",     # green
            FileState.INCOMPLETE: "#F59E0B",    # yellow
            FileState.MISSING: "#71717A",       # gray
            FileState.CORRUPTED: "#EF4444",     # red
            FileState.DOWNLOADABLE: "#0EA5E9",  # blue
        }[self.state]


# ---------- Chunked SHA-256 ----------
def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file, reading in chunks (memory-efficient).

    For a 8 GB safetensors file this takes ~30 seconds on SSD.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------- Folder aliases (host-named folders for imported models) ----------
_REPO_FOLDER_ALIASES: dict[str, str] = {}


def set_repo_folder_alias(repo_id: str, folder: Optional[str]) -> None:
    """Map a repo_id to a host-chosen local folder name (single path segment)."""
    if not repo_id:
        return
    clean = (folder or "").strip().replace("\\", "/").strip("/")
    if not clean or "/" in clean or clean in (".", ".."):
        _REPO_FOLDER_ALIASES.pop(repo_id, None)
        return
    _REPO_FOLDER_ALIASES[repo_id] = clean


def _repo_dir_name(repo_id: str) -> str:
    return _REPO_FOLDER_ALIASES.get(repo_id) or repo_id.split("/")[-1]


# ---------- Cache manager ----------
class CacheManager:
    """Manages the local model cache directory + state tracking."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or get_settings().models_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Path helpers ----------
    def path_for(self, repo_id: str, file_path: str) -> Path:
        """Get the local path where a remote file should be stored."""
        # Sanitize repo_id → directory name (e.g. "QuantFunc/Klein-9B-Series" → "Klein-9B-Series")
        return self.cache_dir / _repo_dir_name(repo_id) / file_path

    def repo_dir_for(self, repo_id: str) -> Path:
        """Get the local directory for a repo's files."""
        return self.cache_dir / _repo_dir_name(repo_id)

    def find_existing_file(self, repo_id: str, file_path: str) -> Optional[Path]:
        """Find an existing file in the repo dir, checking canonical path and subdirectories."""
        if not repo_id or not file_path:
            return None
        repo_dir = self.repo_dir_for(repo_id)
        if not repo_dir.exists():
            return None
        canonical = repo_dir / file_path
        if canonical.is_file():
            return canonical
        name = Path(file_path).name
        # Direct common subfolder checks (fastest)
        for sub in ("transformer", "transformers", "text_encoder", "text_encoders", "vae", "tokenizer", "scheduler", ""):
            cand = (repo_dir / sub / name) if sub else (repo_dir / name)
            if cand.is_file():
                return cand
        # Fallback: search repo_dir (ignoring .cache and .incomplete)
        try:
            for cand in repo_dir.rglob(name):
                if cand.is_file() and not cand.name.endswith(".incomplete") and ".cache" not in cand.parts:
                    return cand
        except Exception:
            pass
        return None


    # ---------- State inspection ----------
    def check_file(
        self,
        repo_id: str,
        file_path: str,
        expected_size: int = 0,
        expected_sha256: Optional[str] = None,
        compute_checksum: bool = True,
    ) -> CacheEntry:
        """Check the state of a single file in cache.

        Args:
            compute_checksum: If True and file exists, compute SHA-256 to verify.
                             Set False for fast existence check only.
        """
        existing = self.find_existing_file(repo_id, file_path)
        local_path = existing or self.path_for(repo_id, file_path)
        incomplete_path = local_path.with_suffix(local_path.suffix + ".incomplete")

        # Case 1: File doesn't exist
        if not local_path.exists():
            if incomplete_path.exists():
                return CacheEntry(
                    path=local_path,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    state=FileState.INCOMPLETE,
                    actual_size=incomplete_path.stat().st_size,
                )
            return CacheEntry(
                path=local_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                state=FileState.DOWNLOADABLE,
            )

        # File exists — get size
        actual_size = local_path.stat().st_size
        actual_sha256 = None

        # Case 2: Size validation
        # Manifests often estimate size_bytes via decimal GB (up to 7.4% difference from binary GiB)
        # or upstream quant weights vary slightly.
        if expected_size > 0:
            is_suspiciously_small = actual_size < 1024 and expected_size > 1024 * 1024
            divergence = abs(actual_size - expected_size) / expected_size
            if is_suspiciously_small or (divergence > 0.15 and not expected_sha256):
                return CacheEntry(
                    path=local_path,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    state=FileState.CORRUPTED,
                    actual_size=actual_size,
                    actual_sha256=None,
                )

        # Case 3: Size OK, no expected sha256 → assume installed
        if not expected_sha256 or not compute_checksum:
            return CacheEntry(
                path=local_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                state=FileState.INSTALLED,
                actual_size=actual_size,
                actual_sha256=None,
            )

        # Case 4: Full checksum verification
        actual_sha256 = compute_sha256(local_path)
        if actual_sha256 == expected_sha256:
            return CacheEntry(
                path=local_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                state=FileState.INSTALLED,
                actual_size=actual_size,
                actual_sha256=actual_sha256,
            )

        # Checksum mismatch
        return CacheEntry(
            path=local_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            state=FileState.CORRUPTED,
            actual_size=actual_size,
            actual_sha256=actual_sha256,
        )

    def check_repo_files(
        self,
        repo_id: str,
        files: list[dict],  # [{"path": "transformer/...", "size": 8100000000, "sha256": "..."}]
        compute_checksum: bool = True,
    ) -> list[CacheEntry]:
        """Check state of multiple files. Returns list of CacheEntry."""
        return [
            self.check_file(
                repo_id=repo_id,
                file_path=f["path"],
                expected_size=f.get("size", 0),
                expected_sha256=f.get("sha256"),
                compute_checksum=compute_checksum,
            )
            for f in files
        ]

    # ---------- Mutations ----------
    def delete_file(self, repo_id: str, file_path: str) -> bool:
        """Delete a file from cache (e.g. corrupted)."""
        local_path = self.find_existing_file(repo_id, file_path) or self.path_for(repo_id, file_path)
        if local_path.exists():
            local_path.unlink()
            return True
        return False

    def delete_repo(self, repo_id: str) -> int:
        """Delete all files for a repo. Returns count of deleted files."""
        repo_dir = self.repo_dir_for(repo_id)
        if not repo_dir.exists():
            return 0
        count = 0
        for f in repo_dir.rglob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        # Remove empty dirs
        for d in sorted(repo_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if repo_dir.exists() and not any(repo_dir.iterdir()):
            repo_dir.rmdir()
        return count

    def get_cache_size(self) -> int:
        """Total size of cache directory in bytes."""
        if not self.cache_dir.exists():
            return 0
        total = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def list_cached_repos(self) -> list[str]:
        """List repo IDs that have at least one file in cache."""
        if not self.cache_dir.exists():
            return []
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]

    # ---------- Output directory ----------
    def save_output(self, image_path: Path, name: Optional[str] = None) -> Path:
        """Copy a generated image to the output directory.

        Args:
            image_path: Source image file (typically in temp dir)
            name: Optional output filename. If None, uses timestamp.

        Returns:
            Path to saved file in output directory.
        """
        import shutil
        from datetime import datetime

        output_dir = get_settings().output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"qf_{timestamp}.png"

        # Validate filename
        from .security import validate_filename
        validate_filename(name)

        dest = output_dir / name
        shutil.copy2(image_path, dest)
        return dest

    def list_outputs(self, limit: int | None = 50) -> list[Path]:
        """List recent output files, newest first.

        ``limit=None`` returns all outputs (used by the paginated History API).
        """
        output_dir = get_settings().output_dir
        if not output_dir.exists():
            return []
        files = [f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files if limit is None else files[:limit]
