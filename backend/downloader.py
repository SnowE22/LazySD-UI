"""Multi-connection resumable downloader with pause, resume and auto-retry.

Supports:
- HTTP Range multi-part parallel connections (e.g. 4-8 threads per file)
- Pause and Resume with zero data loss (state persisted in .download_meta.json)
- Automatic retry of broken chunks with exponential backoff
- Single-stream fallback for non-Range servers
- Real-time progress, speed (bytes/sec), and ETA estimation
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import urllib3

logger = logging.getLogger("downloader")


class DownloadStatus:
    QUEUED = "queued"
    CONNECTING = "connecting"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class PartInfo:
    index: int
    start: int
    end: int
    downloaded: int = 0
    done: bool = False

    @property
    def total_bytes(self) -> int:
        return max(0, self.end - self.start + 1)


ProgressCallback = Callable[[Dict[str, Any]], None]


class MultiPartDownloader:
    """Manages multi-connection chunked download of a single file."""

    def __init__(
        self,
        url: str,
        destination: Path,
        task_id: str,
        headers: Optional[Dict[str, str]] = None,
        num_connections: int = 4,
        progress_callback: Optional[ProgressCallback] = None,
        max_retries: int = 10,
        chunk_size: int = 128 * 1024,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.destination = Path(destination)
        self.task_id = task_id
        self.headers = headers or {}
        self.num_connections = max(1, min(16, int(num_connections)))
        self.progress_callback = progress_callback
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.metadata = metadata or {}

        self.status = DownloadStatus.QUEUED
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.speed_bps = 0.0
        self.eta_seconds: Optional[float] = None
        self.error_message: Optional[str] = None

        self.parts: List[PartInfo] = []
        self.supports_range = False
        self.resolved_url: str = url

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._worker_threads: List[threading.Thread] = []
        self._main_thread: Optional[threading.Thread] = None

        # Speed measurement window
        self._speed_samples: List[Tuple[float, int]] = []
        self._last_emit_time = 0.0

        self.meta_file = self.destination.parent / f"{self.destination.name}.download_meta.json"

    def _part_file(self, part_idx: int) -> Path:
        return self.destination.parent / f"{self.destination.name}.part_{part_idx}"

    def start(self) -> None:
        """Start or resume the download in a background thread."""
        with self._lock:
            if self.status in (DownloadStatus.DOWNLOADING, DownloadStatus.COMPLETING, DownloadStatus.COMPLETE):
                return
            self._pause_event.clear()
            self._cancel_event.clear()
            self.status = DownloadStatus.CONNECTING
            self.error_message = None

        self._main_thread = threading.Thread(target=self._run, daemon=True)
        self._main_thread.start()

    def pause(self) -> bool:
        """Pause the active download."""
        with self._lock:
            if self.status not in (DownloadStatus.CONNECTING, DownloadStatus.DOWNLOADING):
                return False
            self.status = DownloadStatus.PAUSED
            self._pause_event.set()

        self._emit_progress()
        logger.info(f"[{self.task_id}] Download paused by user.")
        return True

    def resume(self) -> bool:
        """Resume a paused download."""
        with self._lock:
            if self.status != DownloadStatus.PAUSED:
                return False
            self.status = DownloadStatus.CONNECTING
            self._pause_event.clear()

        self._main_thread = threading.Thread(target=self._run, daemon=True)
        self._main_thread.start()
        logger.info(f"[{self.task_id}] Download resumed by user.")
        return True

    def cancel(self, delete_files: bool = True) -> bool:
        """Cancel the download and clean up temporary part files."""
        with self._lock:
            self.status = DownloadStatus.CANCELLED
            self._cancel_event.set()
            self._pause_event.set()

        if delete_files:
            self._cleanup_temp_files()

        self._emit_progress()
        logger.info(f"[{self.task_id}] Download cancelled.")
        return True

    def _cleanup_temp_files(self) -> None:
        """Remove partial files and metadata."""
        try:
            if self.meta_file.exists():
                self.meta_file.unlink()
        except OSError:
            pass

        for i in range(len(self.parts) or 16):
            part_p = self._part_file(i)
            try:
                if part_p.exists():
                    part_p.unlink()
            except OSError:
                pass

        tmp_dest = self.destination.parent / f"{self.destination.name}.tmp"
        try:
            if tmp_dest.exists():
                tmp_dest.unlink()
        except OSError:
            pass

    def _save_metadata(self) -> None:
        """Persist current download progress to disk."""
        data = {
            "task_id": self.task_id,
            "url": self.url,
            "resolved_url": self.resolved_url,
            "total_bytes": self.total_bytes,
            "supports_range": self.supports_range,
            "num_connections": self.num_connections,
            "parts": [
                {
                    "index": p.index,
                    "start": p.start,
                    "end": p.end,
                    "downloaded": p.downloaded,
                    "done": p.done,
                }
                for p in self.parts
            ],
            "metadata": self.metadata,
        }
        try:
            self.meta_file.parent.mkdir(parents=True, exist_ok=True)
            self.meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[{self.task_id}] Failed to save metadata: {exc}")

    def _load_metadata(self) -> bool:
        """Load progress from disk if metadata exists and matches destination."""
        if not self.meta_file.exists():
            return False
        try:
            data = json.loads(self.meta_file.read_text(encoding="utf-8"))
            if data.get("url") != self.url and data.get("resolved_url") != self.resolved_url:
                return False
            self.total_bytes = int(data.get("total_bytes") or 0)
            self.supports_range = bool(data.get("supports_range"))
            self.parts = [
                PartInfo(
                    index=int(p["index"]),
                    start=int(p["start"]),
                    end=int(p["end"]),
                    downloaded=int(p.get("downloaded", 0)),
                    done=bool(p.get("done", False)),
                )
                for p in data.get("parts", [])
            ]
            return len(self.parts) > 0
        except Exception as exc:
            logger.warning(f"[{self.task_id}] Could not parse metadata: {exc}")
            return False

    def _run(self) -> None:
        """Main execution loop."""
        try:
            self.destination.parent.mkdir(parents=True, exist_ok=True)

            # Step 1: Probe remote URL (HEAD / Range test)
            if not self._load_metadata() or self.total_bytes <= 0:
                self._probe_remote()

            if self._cancel_event.is_set():
                return
            if self._pause_event.is_set():
                self.status = DownloadStatus.PAUSED
                self._emit_progress()
                return

            with self._lock:
                self.status = DownloadStatus.DOWNLOADING

            # Step 2: Download according to Range capability
            if self.supports_range and self.total_bytes > 5 * 1024 * 1024:
                self._run_multipart()
            else:
                self._run_single()

            if self._cancel_event.is_set():
                return
            if self._pause_event.is_set():
                self.status = DownloadStatus.PAUSED
                self._emit_progress()
                return

            # Step 3: Complete & assemble
            with self._lock:
                self.status = DownloadStatus.COMPLETING
            self._emit_progress()

            if self.supports_range and len(self.parts) > 1:
                self._merge_parts()

            with self._lock:
                self.status = DownloadStatus.COMPLETE
                self.downloaded_bytes = self.total_bytes
                self.speed_bps = 0.0
                self.eta_seconds = 0.0

            self._cleanup_temp_files()
            self._emit_progress()
            logger.info(f"[{self.task_id}] Download finished successfully -> {self.destination}")

        except Exception as exc:
            if not self._cancel_event.is_set() and not self._pause_event.is_set():
                logger.error(f"[{self.task_id}] Download error: {exc}", exc_info=True)
                with self._lock:
                    self.status = DownloadStatus.ERROR
                    self.error_message = str(exc)
                self._emit_progress()

    def _probe_remote(self) -> None:
        """Probe the remote URL to check file size and Range header support."""
        session = requests.Session()
        resp = None
        try:
            resp = session.head(
                self.url,
                headers=self.headers,
                allow_redirects=True,
                timeout=self.timeout,
            )
            self.resolved_url = resp.url
        except Exception:
            # Fall back to GET with 1 byte Range
            pass

        if not resp or resp.status_code >= 400 or not resp.headers.get("Content-Length"):
            range_hdr = dict(self.headers)
            range_hdr["Range"] = "bytes=0-1"
            resp = session.get(
                self.url,
                headers=range_hdr,
                stream=True,
                allow_redirects=True,
                timeout=self.timeout,
            )
            self.resolved_url = resp.url

        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP probe failed with status {resp.status_code}")

        # Check total size
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total_str = content_range.split("/")[-1].strip()
            if total_str.isdigit():
                self.total_bytes = int(total_str)
        if not self.total_bytes:
            self.total_bytes = int(resp.headers.get("Content-Length", 0))

        # Check range support
        accept_ranges = resp.headers.get("Accept-Ranges", "").lower()
        if accept_ranges == "bytes" or resp.status_code == 206:
            self.supports_range = True
        else:
            # Verify range with a small test
            test_resp = session.get(
                self.resolved_url,
                headers={"Range": "bytes=0-0"},
                stream=True,
                timeout=10,
            )
            self.supports_range = test_resp.status_code == 206

        # Partition into parts
        if self.supports_range and self.total_bytes > 5 * 1024 * 1024:
            # Auto-scale connections based on size
            if self.total_bytes > 4 * 1024 * 1024 * 1024:
                num_parts = max(self.num_connections, 6)
            elif self.total_bytes > 500 * 1024 * 1024:
                num_parts = max(self.num_connections, 4)
            else:
                num_parts = min(self.num_connections, 4)

            self.parts = []
            part_size = self.total_bytes // num_parts
            for i in range(num_parts):
                start = i * part_size
                end = (i + 1) * part_size - 1 if i < num_parts - 1 else self.total_bytes - 1
                self.parts.append(PartInfo(index=i, start=start, end=end))
        else:
            self.parts = [PartInfo(index=0, start=0, end=max(0, self.total_bytes - 1))]

        self._save_metadata()

    def _run_multipart(self) -> None:
        """Launch parallel workers for each incomplete part."""
        threads = []
        for part in self.parts:
            part_path = self._part_file(part.index)
            if part_path.exists():
                part.downloaded = part_path.stat().st_size
                if part.downloaded >= part.total_bytes:
                    part.done = True

            if not part.done:
                t = threading.Thread(target=self._worker_download_part, args=(part,), daemon=True)
                threads.append(t)
                t.start()

        self._worker_threads = threads
        for t in threads:
            t.join()

        # Check if all completed
        all_done = all(p.done for p in self.parts)
        if not all_done and not self._pause_event.is_set() and not self._cancel_event.is_set():
            raise RuntimeError("One or more download parts failed after maximum retries.")

    def _worker_download_part(self, part: PartInfo) -> None:
        """Download a single range part with automatic reconnect and retries."""
        part_path = self._part_file(part.index)
        attempt = 0

        while attempt < self.max_retries and not self._pause_event.is_set() and not self._cancel_event.is_set():
            try:
                current_size = part_path.stat().st_size if part_path.exists() else 0
                part.downloaded = current_size

                if current_size >= part.total_bytes:
                    part.done = True
                    self._save_metadata()
                    return

                req_headers = dict(self.headers)
                range_start = part.start + current_size
                range_end = part.end
                req_headers["Range"] = f"bytes={range_start}-{range_end}"

                session = requests.Session()
                with session.get(
                    self.resolved_url,
                    headers=req_headers,
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"Part {part.index} server returned {resp.status_code}")

                    with open(part_path, "ab") as f:
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
                            if self._pause_event.is_set() or self._cancel_event.is_set():
                                return
                            if chunk:
                                f.write(chunk)
                                written = len(chunk)
                                with self._lock:
                                    part.downloaded += written
                                    self._record_speed(written)
                                self._emit_progress_throttled()

                # Verify part size
                final_size = part_path.stat().st_size if part_path.exists() else 0
                if final_size >= part.total_bytes:
                    part.done = True
                    part.downloaded = part.total_bytes
                    self._save_metadata()
                    return
                else:
                    # Premature close without error -> retry remaining
                    attempt += 1
                    time.sleep(min(2 ** attempt, 10))

            except (requests.RequestException, socket.error, urllib3.exceptions.HTTPError, OSError) as exc:
                attempt += 1
                if self._pause_event.is_set() or self._cancel_event.is_set():
                    return
                logger.warning(
                    f"[{self.task_id}] Part {part.index} error (attempt {attempt}/{self.max_retries}): {exc}. Retrying..."
                )
                time.sleep(min(2 ** attempt, 15))

    def _run_single(self) -> None:
        """Fallback single-stream download with resume if Range is supported."""
        tmp_path = self.destination.parent / f"{self.destination.name}.tmp"
        attempt = 0

        while attempt < self.max_retries and not self._pause_event.is_set() and not self._cancel_event.is_set():
            try:
                current_size = tmp_path.stat().st_size if tmp_path.exists() else 0
                req_headers = dict(self.headers)
                if self.supports_range and current_size > 0:
                    req_headers["Range"] = f"bytes={current_size}-"
                    mode = "ab"
                else:
                    current_size = 0
                    mode = "wb"

                session = requests.Session()
                with session.get(
                    self.resolved_url,
                    headers=req_headers,
                    stream=True,
                    timeout=self.timeout,
                ) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"Server returned status {resp.status_code}")

                    with open(tmp_path, mode) as f:
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
                            if self._pause_event.is_set() or self._cancel_event.is_set():
                                return
                            if chunk:
                                f.write(chunk)
                                written = len(chunk)
                                with self._lock:
                                    current_size += written
                                    self.downloaded_bytes = current_size
                                    self._record_speed(written)
                                self._emit_progress_throttled()

                if self.total_bytes > 0 and current_size >= self.total_bytes:
                    shutil.move(str(tmp_path), str(self.destination))
                    return
                elif self.total_bytes == 0 and current_size > 0:
                    self.total_bytes = current_size
                    shutil.move(str(tmp_path), str(self.destination))
                    return
                else:
                    attempt += 1
                    time.sleep(min(2 ** attempt, 10))

            except (requests.RequestException, socket.error, urllib3.exceptions.HTTPError, OSError) as exc:
                attempt += 1
                if self._pause_event.is_set() or self._cancel_event.is_set():
                    return
                logger.warning(
                    f"[{self.task_id}] Single-stream error (attempt {attempt}/{self.max_retries}): {exc}. Retrying..."
                )
                time.sleep(min(2 ** attempt, 15))

    def _merge_parts(self) -> None:
        """Merge all downloaded part files into final destination file."""
        tmp_dest = self.destination.parent / f"{self.destination.name}.tmp"
        with open(tmp_dest, "wb") as outfile:
            for part in self.parts:
                part_p = self._part_file(part.index)
                if not part_p.exists():
                    raise RuntimeError(f"Missing part file {part_p}")
                with open(part_p, "rb") as infile:
                    shutil.copyfileobj(infile, outfile, length=1024 * 1024)

        # Atomic replacement
        if self.destination.exists():
            self.destination.unlink()
        shutil.move(str(tmp_dest), str(self.destination))

    def _record_speed(self, byte_count: int) -> None:
        now = time.time()
        self._speed_samples.append((now, byte_count))
        # Keep only samples from last 3.0 seconds for smooth rolling average
        cutoff = now - 3.0
        self._speed_samples = [s for s in self._speed_samples if s[0] >= cutoff]

        if len(self._speed_samples) > 1:
            total_window_bytes = sum(s[1] for s in self._speed_samples)
            window_time = max(0.1, self._speed_samples[-1][0] - self._speed_samples[0][0])
            self.speed_bps = total_window_bytes / window_time
            remaining_bytes = max(0, self.total_bytes - self.get_total_downloaded())
            if self.speed_bps > 1024:
                self.eta_seconds = remaining_bytes / self.speed_bps
            else:
                self.eta_seconds = None

    def get_total_downloaded(self) -> int:
        if self.parts:
            return sum(p.downloaded for p in self.parts)
        return self.downloaded_bytes

    def _emit_progress_throttled(self) -> None:
        now = time.time()
        if now - self._last_emit_time >= 0.25:  # 4 times per second
            self._last_emit_time = now
            self._emit_progress()

    def _emit_progress(self) -> None:
        if not self.progress_callback:
            return
        downloaded = self.get_total_downloaded()
        total = self.total_bytes or 0
        pct = (downloaded / float(total)) if total > 0 else 0.0

        payload = {
            "task_id": self.task_id,
            "status": self.status,
            "progress": min(1.0, max(0.0, pct)),
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "speed_bps": self.speed_bps,
            "eta_seconds": self.eta_seconds,
            "message": self._format_status_message(),
            "error": self.error_message,
            "destination": str(self.destination),
            "metadata": self.metadata,
        }
        try:
            self.progress_callback(payload)
        except Exception as exc:
            logger.debug(f"[{self.task_id}] Error in progress_callback: {exc}")

    def _format_status_message(self) -> str:
        if self.status == DownloadStatus.COMPLETE:
            return "Загрузка завершена ✓"
        if self.status == DownloadStatus.PAUSED:
            return "Загрузка приостановлена"
        if self.status == DownloadStatus.CANCELLED:
            return "Загрузка отменена"
        if self.status == DownloadStatus.ERROR:
            return f"Ошибка: {self.error_message or 'сбой сети'}"
        if self.status == DownloadStatus.COMPLETING:
            return "Сборка скачанных сегментов..."
        if self.status == DownloadStatus.CONNECTING:
            return "Подключение к серверу..."

        dl_gb = self.get_total_downloaded() / (1024 * 1024 * 1024)
        total_gb = (self.total_bytes or 0) / (1024 * 1024 * 1024)
        speed_mb = (self.speed_bps or 0.0) / (1024 * 1024)

        speed_str = f"{speed_mb:.1f} MB/s" if speed_mb >= 0.1 else ""
        eta_str = ""
        if self.eta_seconds is not None and self.eta_seconds > 0:
            eta_m = int(self.eta_seconds // 60)
            eta_s = int(self.eta_seconds % 60)
            eta_str = f"ETA: {eta_m}m {eta_s}s" if eta_m > 0 else f"ETA: {eta_s}s"

        info = [s for s in (speed_str, eta_str) if s]
        info_joined = f" ({', '.join(info)})" if info else ""
        return f"{dl_gb:.2f} GB / {total_gb:.2f} GB{info_joined}"

    def to_dict(self) -> Dict[str, Any]:
        downloaded = self.get_total_downloaded()
        total = self.total_bytes or 0
        pct = (downloaded / float(total)) if total > 0 else 0.0
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": min(1.0, max(0.0, pct)),
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "speed_bps": self.speed_bps,
            "eta_seconds": self.eta_seconds,
            "message": self._format_status_message(),
            "destination": str(self.destination),
            "parts_count": len(self.parts),
            "metadata": self.metadata,
            "error": self.error_message,
        }


class DownloadManager:
    """Singleton download manager coordinating multi-threaded downloads."""

    def __init__(self):
        self._tasks: Dict[str, MultiPartDownloader] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        task_id: str,
        url: str,
        destination: Path,
        headers: Optional[Dict[str, str]] = None,
        num_connections: int = 4,
        progress_callback: Optional[ProgressCallback] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultiPartDownloader:
        """Register and start a multi-part download task."""
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing and existing.status in (DownloadStatus.CONNECTING, DownloadStatus.DOWNLOADING):
                return existing

            downloader = MultiPartDownloader(
                url=url,
                destination=destination,
                task_id=task_id,
                headers=headers,
                num_connections=num_connections,
                progress_callback=progress_callback,
                metadata=metadata,
            )
            self._tasks[task_id] = downloader

        downloader.start()
        return downloader

    def pause(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return False
        return task.pause()

    def resume(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return False
        return task.resume()

    def cancel(self, task_id: str, delete_files: bool = True) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return False
        return task.cancel(delete_files=delete_files)

    def get_task(self, task_id: str) -> Optional[MultiPartDownloader]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            tasks = list(self._tasks.values())
        return [t.to_dict() for t in tasks]


_GLOBAL_MANAGER: Optional[DownloadManager] = None


def get_download_manager() -> DownloadManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = DownloadManager()
    return _GLOBAL_MANAGER
