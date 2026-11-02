import json
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import pytest

from backend.downloader import (
    DownloadManager,
    DownloadStatus,
    MultiPartDownloader,
    PartInfo,
    get_download_manager,
)


@pytest.fixture
def tmp_download_dir(tmp_path):
    d = tmp_path / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_part_info_total_bytes():
    part = PartInfo(index=0, start=0, end=1023)
    assert part.total_bytes == 1024
    assert part.downloaded == 0
    assert not part.done


def test_downloader_initialization(tmp_download_dir):
    dest = tmp_download_dir / "model.safetensors"
    dl = MultiPartDownloader(
        url="https://example.com/model.safetensors",
        destination=dest,
        task_id="task_123",
        num_connections=4,
    )
    assert dl.task_id == "task_123"
    assert dl.num_connections == 4
    assert dl.status == DownloadStatus.QUEUED
    assert dl.destination == dest
    assert dl.meta_file == tmp_download_dir / "model.safetensors.download_meta.json"


def test_downloader_pause_and_resume_state(tmp_download_dir):
    dest = tmp_download_dir / "test_file.bin"
    dl = MultiPartDownloader(
        url="https://example.com/test_file.bin",
        destination=dest,
        task_id="test_pause_resume",
    )
    dl.status = DownloadStatus.DOWNLOADING
    assert dl.pause() is True
    assert dl.status == DownloadStatus.PAUSED
    assert dl._pause_event.is_set()

    # Pause again returns False
    assert dl.pause() is False

    # Mock _run to prevent actual HTTP call
    with patch.object(dl, "_run", return_value=None):
        assert dl.resume() is True
        assert dl.status == DownloadStatus.CONNECTING
        assert not dl._pause_event.is_set()


def test_downloader_cancel_and_cleanup(tmp_download_dir):
    dest = tmp_download_dir / "to_cancel.bin"
    meta = tmp_download_dir / "to_cancel.bin.download_meta.json"
    part0 = tmp_download_dir / "to_cancel.bin.part_0"
    part1 = tmp_download_dir / "to_cancel.bin.part_1"

    meta.write_text("{}", encoding="utf-8")
    part0.write_bytes(b"123")
    part1.write_bytes(b"456")

    dl = MultiPartDownloader(
        url="https://example.com/to_cancel.bin",
        destination=dest,
        task_id="test_cancel",
    )
    dl.parts = [
        PartInfo(index=0, start=0, end=2),
        PartInfo(index=1, start=3, end=5),
    ]

    dl.cancel(delete_files=True)
    assert dl.status == DownloadStatus.CANCELLED
    assert not meta.exists()
    assert not part0.exists()
    assert not part1.exists()


def test_downloader_merge_parts(tmp_download_dir):
    dest = tmp_download_dir / "merged.bin"
    part0 = tmp_download_dir / "merged.bin.part_0"
    part1 = tmp_download_dir / "merged.bin.part_1"
    part2 = tmp_download_dir / "merged.bin.part_2"

    part0.write_bytes(b"HELLO_")
    part1.write_bytes(b"PARALLEL_")
    part2.write_bytes(b"WORLD!")

    dl = MultiPartDownloader(
        url="https://example.com/merged.bin",
        destination=dest,
        task_id="test_merge",
    )
    dl.parts = [
        PartInfo(index=0, start=0, end=5, downloaded=6, done=True),
        PartInfo(index=1, start=6, end=14, downloaded=9, done=True),
        PartInfo(index=2, start=15, end=20, downloaded=6, done=True),
    ]
    dl._merge_parts()

    assert dest.exists()
    assert dest.read_bytes() == b"HELLO_PARALLEL_WORLD!"


def test_download_manager_lifecycle(tmp_download_dir):
    mgr = DownloadManager()
    dest = tmp_download_dir / "managed.bin"

    with patch.object(MultiPartDownloader, "start", return_value=None):
        task = mgr.submit(
            task_id="job_001",
            url="https://example.com/managed.bin",
            destination=dest,
            num_connections=4,
        )
        assert task.task_id == "job_001"
        assert mgr.get_task("job_001") is task

        tasks = mgr.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "job_001"

        task.status = DownloadStatus.DOWNLOADING
        assert mgr.pause("job_001") is True
        assert task.status == DownloadStatus.PAUSED

        assert mgr.cancel("job_001") is True
        assert task.status == DownloadStatus.CANCELLED


def test_api_download_endpoints(tmp_download_dir):
    from fastapi.testclient import TestClient
    from app import create_app
    from backend.downloader import get_download_manager, MultiPartDownloader

    client = TestClient(create_app())
    mgr = get_download_manager()
    dest = tmp_download_dir / "api_test.bin"

    with patch.object(MultiPartDownloader, "start", return_value=None):
        task = mgr.submit(
            task_id="api_dl_job_42",
            url="https://example.com/api_test.bin",
            destination=dest,
            num_connections=4,
        )

        # GET /api/downloads
        resp = client.get("/api/downloads")
        assert resp.status_code == 200
        items = resp.json()
        assert any(item["task_id"] == "api_dl_job_42" for item in items)

        # GET /api/downloads/{task_id}
        resp = client.get("/api/downloads/api_dl_job_42")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "api_dl_job_42"

        # POST /api/downloads/{task_id}/pause
        task.status = DownloadStatus.DOWNLOADING
        resp = client.post("/api/downloads/api_dl_job_42/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"
        assert task.status == DownloadStatus.PAUSED

        # POST /api/downloads/{task_id}/resume
        with patch.object(task, "_run", return_value=None):
            resp = client.post("/api/downloads/api_dl_job_42/resume")
            assert resp.status_code == 200
            assert resp.json()["status"] == "resumed"

        # POST /api/downloads/{task_id}/cancel
        resp = client.post("/api/downloads/api_dl_job_42/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        assert task.status == DownloadStatus.CANCELLED
