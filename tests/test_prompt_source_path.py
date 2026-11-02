"""Regression test: ``uploads/<name>`` must resolve to ``<output_dir>/uploads/<name>``."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from backend.prompt_gen import PromptGenValidationError, resolve_safe_source_path


def test_resolve_upload_relative_to_output_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "outputs"
        (out / "uploads").mkdir(parents=True)
        img = out / "uploads" / "ref_test.png"
        Image.new("RGB", (8, 8)).save(img)

        allowed = [(root / "uploads").resolve(), out.resolve(), (root / "cache").resolve()]
        got = resolve_safe_source_path("uploads/ref_test.png", project_root=root, allowed_dirs=allowed)
        assert got == img.resolve()

        got2 = resolve_safe_source_path("/outputs/uploads/ref_test.png?t=1", project_root=root, allowed_dirs=allowed)
        assert got2 == img.resolve()

        for bad in ("uploads/missing.png", "../secret.png"):
            try:
                resolve_safe_source_path(bad, project_root=root, allowed_dirs=allowed)
            except PromptGenValidationError:
                continue
            raise AssertionError(f"expected rejection for {bad}")


if __name__ == "__main__":
    test_resolve_upload_relative_to_output_dir()
    print("[SUCCESS] source path resolver test passed")
