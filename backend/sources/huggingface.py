"""
HuggingFaceSource — implementation using huggingface_hub SDK.

Native resume via .incomplete files.
SHA-256 available for LFS-tracked files only (lfs_sha256 field).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Disable Rust-based hf-xet and hf_transfer to prevent out-of-memory / allocation crashes on Windows
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
try:
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_DISABLE_XET = True
    _hfc.HF_HUB_ENABLE_HF_TRANSFER = False
except Exception:
    pass

from .base import ModelSource, RemoteFile, ModelMetadata, ProgressCallback


class HuggingFaceSource(ModelSource):
    source_type = "huggingface"

    def __init__(self, token: Optional[str] = None):
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            ) from e

        self.token = token
        self.api = HfApi(token=token)

    def list_files(self, repo_id: str, revision: str = "main") -> list[RemoteFile]:
        info = self.api.repo_info(repo_id, revision=revision, files_metadata=True)
        return [
            RemoteFile(
                path=sibling.rfilename,
                size=sibling.size or 0,
                sha256=getattr(sibling, "lfs_sha256", None),
                is_lfs=getattr(sibling, "lfs", False),
                revision=revision,
            )
            for sibling in info.siblings
        ]

    def file_exists(self, repo_id: str, file_path: str, revision: str = "main") -> bool:
        try:
            from huggingface_hub import hf_hub_url, get_hf_file_metadata
            url = hf_hub_url(repo_id=repo_id, filename=file_path, revision=revision)
            get_hf_file_metadata(url, token=self.token)
            return True
        except Exception:
            return False

    def get_metadata(self, repo_id: str, revision: str = "main") -> ModelMetadata:
        info = self.api.repo_info(repo_id, revision=revision, files_metadata=True)
        return ModelMetadata(
            repo_id=repo_id,
            description=info.description if hasattr(info, "description") else "",
            license=getattr(info, "license", "") or "",
            files_count=len(info.siblings),
            size_total=sum((s.size or 0) for s in info.siblings),
            last_modified=str(info.last_modified) if hasattr(info, "last_modified") else None,
            tags=list(getattr(info, "tags", []) or []),
        )

    def download_file(
        self,
        repo_id: str,
        file_path: str,
        local_dir: Path,
        revision: str = "main",
        progress_callback: Optional[ProgressCallback] = None,
        job_id: Optional[str] = None,
    ) -> Path:
        from pathlib import Path
        local_path = Path(local_dir) / file_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_file():
            return local_path

        # Use MultiPartDownloader with pause/resume and auto-retry
        try:
            from backend.downloader import get_download_manager, DownloadStatus

            direct_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}"
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            task_id = job_id or f"hf_{repo_id.replace('/', '_')}_{Path(file_path).name}"

            def _forward_progress(p: dict):
                if progress_callback:
                    progress_callback(
                        int(p.get("downloaded_bytes") or 0),
                        int(p.get("total_bytes") or 0),
                        float(p.get("speed_bps") or 0.0),
                    )

            mgr = get_download_manager()
            dl = mgr.submit(
                task_id=task_id,
                url=direct_url,
                destination=local_path,
                headers=headers,
                num_connections=4,
                progress_callback=_forward_progress,
                metadata={"repo_id": repo_id, "file_path": file_path, "source": "huggingface"},
            )

            if dl._main_thread:
                dl._main_thread.join()

            if dl.status == DownloadStatus.COMPLETE and local_path.is_file():
                return local_path
            if dl.status == DownloadStatus.CANCELLED:
                raise RuntimeError(f"Download cancelled by user: {file_path}")
            if dl.status == DownloadStatus.PAUSED:
                return local_path
            if dl.status == DownloadStatus.ERROR and dl.error_message:
                print(f"[hf] MultiPartDownloader error: {dl.error_message}. Falling back to hf_hub_download...")
        except Exception as exc:
            if "cancelled by user" in str(exc):
                raise
            print(f"[hf] MultiPartDownloader fallback due to: {exc}")

        # Fallback to standard huggingface_hub
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            revision=revision,
            local_dir=str(local_dir),
            token=self.token,
        )
        return Path(downloaded)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: Path,
        allow_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        revision: str = "main",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        # Ensure target directory exists
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        local_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
            local_dir=str(local_dir),
            token=self.token,
        )

        # Write completion marker
        self.mark_complete(Path(local_dir), repo_id)

        return Path(local_path)
