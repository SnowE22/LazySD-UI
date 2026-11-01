"""Regression tests for safe weight deletion and folder-layout HF imports.

The deletion tests use a temporary cache and a fake registry, so they never touch
real model weights. The layout test mirrors the file list of
``WaveCut/Fibo-2.0-edit-turbo-OrbitQuant-W4A8`` without network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FIBO_FILES = [
    {"path": "manifest.json", "size": 1254},
    {"path": "fibo2_orbitquant.py", "size": 8467},
    {"path": "run_fibo2.py", "size": 3235},
    {"path": "fonts/DejaVuSans.ttf", "size": 757076},
    {"path": "processor/tokenizer.json", "size": 11422650},
    {"path": "scheduler/scheduler_config.json", "size": 486},
    {"path": "transformer/config.json", "size": 623},
    {"path": "transformer/gptq-info.json", "size": 24272},
    {"path": "transformer/model.safetensors", "size": 4329176674},
    {"path": "transformer/orbitquant_codebooks.safetensors", "size": 860},
    {"path": "transformer/orbitquant_manifest.json", "size": 93568},
    {"path": "transformer/orbitquant_rotations.safetensors", "size": 354796},
    {"path": "text_encoder/config.json", "size": 2834},
    {"path": "text_encoder/model.safetensors", "size": 3427991944},
    {"path": "vae/config.json", "size": 909},
    {"path": "vae/diffusion_pytorch_model.safetensors", "size": 336213556},
]


# ---------------------------------------------------------------------------
# Hugging Face folder layout (Fibo / OrbitQuant W4A8)
# ---------------------------------------------------------------------------

def test_fibo_folder_layout_is_self_contained():
    from backend.model_sniffer import sniff_files

    result = sniff_files(
        FIBO_FILES,
        repo_id="WaveCut/Fibo-2.0-edit-turbo-OrbitQuant-W4A8",
        tags=["image-to-image", "orbitquant", "base_model:quantized:briaai/Fibo-2.0-edit-turbo"],
    )

    assert result["layout"] == "folders"
    assert result["has_model_index"] is False
    assert result["has_component_configs"] is True
    assert result["suggested_engine"] == "orbitquant"
    assert result["suggested_family"] == "flux"
    assert result["missing"] == []
    assert result["needs_manual"] is False
    assert "fibo2_orbitquant.py" in result["pipeline_code"]
    assert any("model_index.json" in warning for warning in result["warnings"])

    components = result["components"]
    assert [f["path"] for f in components["diffusion_model"]] == ["transformer/model.safetensors"]
    assert [f["path"] for f in components["text_encoder"]] == ["text_encoder/model.safetensors"]
    assert [f["path"] for f in components["vae"]] == ["vae/diffusion_pytorch_model.safetensors"]
    assert {f["path"] for f in components["aux"]} == {
        "transformer/orbitquant_codebooks.safetensors",
        "transformer/orbitquant_rotations.safetensors",
    }
    assert set(result["defaults"]) == {"text_encoder", "vae"}


def test_folders_without_component_configs_still_requires_base():
    from backend.model_sniffer import sniff_files

    result = sniff_files(
        [
            {"path": "transformer/model.safetensors", "size": 4_000_000_000},
            {"path": "text_encoder/model.safetensors", "size": 3_000_000_000},
            {"path": "vae/diffusion_pytorch_model.safetensors", "size": 300_000_000},
        ],
        repo_id="someone/orbitquant-without-configs",
        tags=["orbitquant"],
    )

    assert result["suggested_engine"] == "orbitquant"
    assert result["has_component_configs"] is False
    assert "base" in result["missing"]
    assert result["needs_manual"] is True


# ---------------------------------------------------------------------------
# VAE cache detection and download targets
# ---------------------------------------------------------------------------

def _vae_model(repo_id: str, **engine_args):
    return SimpleNamespace(
        id="vae_model",
        sources=SimpleNamespace(primary=SimpleNamespace(repo_id=repo_id)),
        engine_args=engine_args,
        transformers=[],
        text_encoders=[],
    )


def test_vae_payload_finds_declared_vae_in_primary_repo(tmp_path):
    from backend.cache import CacheManager
    from backend.model_files import vae_download_refs, vae_payload

    cache = CacheManager(tmp_path / "cache")
    _write(cache.repo_dir_for("org/model") / "flux2-vae.safetensors")
    model = _vae_model(
        "org/model",
        vae="flux2-vae.safetensors",
        vae_repo="unsloth/FLUX.2-VAE",
        vae_remote="split_files/vae/flux2-vae.safetensors",
    )

    payload = vae_payload(model, "org/model", cache)
    assert payload["installed"] is True
    assert payload["repo_id"] == "unsloth/FLUX.2-VAE"
    assert payload["path"] == "split_files/vae/flux2-vae.safetensors"
    assert payload["local_repo_id"] == "org/model"
    assert payload["local_path"] == "flux2-vae.safetensors"
    assert payload["resolved_path"] == "flux2-vae.safetensors"

    refs = vae_download_refs(model, "org/model")
    assert refs == {
        "source_repo": "unsloth/FLUX.2-VAE",
        "source_path": "split_files/vae/flux2-vae.safetensors",
        "local_repo": "org/model",
        "local_path": "flux2-vae.safetensors",
    }


def test_vae_payload_finds_vae_in_separate_vae_repo(tmp_path):
    from backend.cache import CacheManager
    from backend.model_files import vae_payload

    cache = CacheManager(tmp_path / "cache")
    _write(
        cache.repo_dir_for("unsloth/FLUX.2-VAE")
        / "split_files" / "vae" / "flux2-vae.safetensors"
    )
    model = _vae_model(
        "org/model",
        vae="flux2-vae.safetensors",
        vae_repo="unsloth/FLUX.2-VAE",
        vae_remote="split_files/vae/flux2-vae.safetensors",
    )

    payload = vae_payload(model, "org/model", cache)
    assert payload["installed"] is True
    assert payload["resolved_repo_id"] == "unsloth/FLUX.2-VAE"


def test_vae_payload_infers_repo_local_vae_when_manifest_has_none(tmp_path):
    from backend.cache import CacheManager
    from backend.model_files import vae_payload

    cache = CacheManager(tmp_path / "cache")
    _write(
        cache.repo_dir_for("org/turbo")
        / "vae" / "diffusion_pytorch_model.safetensors"
    )
    model = _vae_model("org/turbo")

    payload = vae_payload(model, "org/turbo", cache)
    assert payload["installed"] is True
    assert payload["inferred"] is True
    assert payload["repo_id"] == "org/turbo"
    assert payload["path"] == "vae/diffusion_pytorch_model.safetensors"
    assert payload["local_path"] == payload["path"]

    assert vae_payload(_vae_model("org/empty"), "org/empty", cache) is None


def test_model_file_refs_include_vae_local_target(tmp_path):
    from backend.model_files import model_file_refs

    model = _vae_model(
        "org/model",
        vae="flux2-vae.safetensors",
        vae_repo="unsloth/FLUX.2-VAE",
        vae_remote="split_files/vae/flux2-vae.safetensors",
    )
    refs = model_file_refs(model)
    assert ("unsloth/FLUX.2-VAE", "split_files/vae/flux2-vae.safetensors") in refs
    assert ("org/model", "flux2-vae.safetensors") in refs


def test_vae_download_refs_rejects_absolute_paths():
    from backend.model_files import vae_download_refs

    model = _vae_model("org/model", vae="/etc/passwd")
    assert vae_download_refs(model, "org/model") is None


# ---------------------------------------------------------------------------
# Safe deletion endpoint
# ---------------------------------------------------------------------------

def _fake_model(model_id, repo_id, transformer_path, te_repo=None, te_path=None, vae_path=None):
    text_encoders = []
    if te_path:
        text_encoders.append({
            "file": te_path,
            "path": te_path,
            "source": {"repo_id": te_repo or repo_id, "file_path": te_path},
        })
    engine_args = {"diffusion_model": transformer_path}
    if vae_path:
        engine_args.update({
            "vae_repo": repo_id,
            "vae_remote": vae_path,
            "vae_file": vae_path,
        })
    return SimpleNamespace(
        id=model_id,
        sources=SimpleNamespace(primary=SimpleNamespace(repo_id=repo_id)),
        transformers=[SimpleNamespace(id=f"{model_id}_variant", path=transformer_path, repo_id=repo_id)],
        text_encoders=text_encoders,
        engine_args=engine_args,
        is_custom=True,
    )


def _write(path: Path, payload: bytes = b"dummy") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_delete_variant_is_exact_and_keeps_encoders(tmp_path, monkeypatch):
    import backend.cache as cache_module

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        cache_module,
        "get_settings",
        lambda: SimpleNamespace(models_dir=cache_root, output_dir=tmp_path),
    )

    repo = "WaveCut/Fibo-2.0-edit-turbo-OrbitQuant-W4A8"
    model = _fake_model(
        "fibo",
        repo,
        "transformer/model.safetensors",
        te_repo=repo,
        te_path="text_encoder/model.safetensors",
        vae_path="vae/diffusion_pytorch_model.safetensors",
    )
    registry = SimpleNamespace(get=lambda mid: model if mid == "fibo" else None,
                                list_all=lambda: [model])

    import app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "get_registry", lambda: registry)
    client = TestClient(app_module.create_app())

    variant = cache_root / "Fibo-2.0-edit-turbo-OrbitQuant-W4A8" / "transformer"
    _write(variant / "model.safetensors")
    _write(variant / "orbitquant_codebooks.safetensors")
    _write(variant / "orbitquant_rotations.safetensors")
    _write(variant / "model.safetensors.incomplete", b"partial")
    _write(cache_root / "Fibo-2.0-edit-turbo-OrbitQuant-W4A8" / "text_encoder" / "model.safetensors")
    _write(cache_root / "Fibo-2.0-edit-turbo-OrbitQuant-W4A8" / "vae" / "diffusion_pytorch_model.safetensors")

    response = client.delete("/api/models/fibo/files", params={"variant_id": "fibo_variant"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["deleted_files_count"] == 4
    assert not (variant / "model.safetensors").exists()
    assert not (variant / "orbitquant_codebooks.safetensors").exists()
    assert not (variant / "orbitquant_rotations.safetensors").exists()
    assert not (variant / "model.safetensors.incomplete").exists()
    assert not variant.exists()
    assert (cache_root / "Fibo-2.0-edit-turbo-OrbitQuant-W4A8" / "text_encoder" / "model.safetensors").is_file()
    assert (cache_root / "Fibo-2.0-edit-turbo-OrbitQuant-W4A8" / "vae" / "diffusion_pytorch_model.safetensors").is_file()

    missing = client.delete("/api/models/fibo/files", params={"variant_id": "nope"})
    assert missing.status_code == 404


def test_delete_protects_files_shared_with_another_model(tmp_path, monkeypatch):
    import backend.cache as cache_module

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        cache_module,
        "get_settings",
        lambda: SimpleNamespace(models_dir=cache_root, output_dir=tmp_path),
    )

    shared_repo = "org/model-a"
    shared_te = "text_encoder/model.safetensors"
    model_a = _fake_model(
        "model_a",
        "org/model-a",
        "transformer/model.safetensors",
        te_repo=shared_repo,
        te_path=shared_te,
        vae_path="vae/diffusion_pytorch_model.safetensors",
    )
    # Another model also uses model-a's repo-local encoder; deleting model-a must
    # report the shared file and leave it on disk.
    model_b = _fake_model(
        "model_b",
        "org/model-b",
        "transformer/model.safetensors",
        te_repo=shared_repo,
        te_path=shared_te,
    )
    registry = SimpleNamespace(
        get=lambda mid: {"model_a": model_a, "model_b": model_b}.get(mid),
        list_all=lambda: [model_a, model_b],
    )

    import app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "get_registry", lambda: registry)
    client = TestClient(app_module.create_app())

    shared_path = _write(cache_root / "model-a" / shared_te)
    vae_path = _write(cache_root / "model-a" / "vae" / "diffusion_pytorch_model.safetensors")
    _write(cache_root / "model-a" / "transformer" / "model.safetensors")

    response = client.delete("/api/models/model_a/files")
    assert response.status_code == 200, response.text
    body = response.json()

    assert shared_path.is_file(), "shared encoder must never be deleted"
    assert f"{shared_repo}/{shared_te}" in body["skipped_shared"]
    assert not vae_path.exists()
    assert not (cache_root / "model-a" / "transformer" / "model.safetensors").exists()


def test_delete_refuses_paths_outside_cache(tmp_path, monkeypatch):
    import backend.cache as cache_module

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        cache_module,
        "get_settings",
        lambda: SimpleNamespace(models_dir=cache_root, output_dir=tmp_path),
    )

    outside = _write(tmp_path / "outside.safetensors", b"do-not-delete")
    model = _fake_model("escape", "org/escape", "../../outside.safetensors")
    registry = SimpleNamespace(get=lambda mid: model if mid == "escape" else None,
                                list_all=lambda: [model])

    import app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "get_registry", lambda: registry)
    client = TestClient(app_module.create_app())

    response = client.delete("/api/models/escape/files")
    assert response.status_code == 200, response.text
    assert outside.is_file()
    assert response.json()["deleted_files_count"] == 0


def test_custom_model_delete_uses_safe_helpers():
    from backend.custom_models import delete_custom_model
    import inspect

    source = inspect.getsource(delete_custom_model)
    assert "collect_delete_targets" in source
    assert "delete_targets" in source
    assert "shutil.rmtree" not in source


def test_shared_helper_protects_files_and_expands_shards(tmp_path):
    from backend.cache import CacheManager
    from backend.model_files import collect_delete_targets, delete_targets

    cache = CacheManager(cache_dir=tmp_path / "cache")
    other = _fake_model(
        "other",
        "org/model-a",
        "transformer/other.safetensors",
        te_repo="org/model-a",
        te_path="text_encoder/model.safetensors",
    )
    model = _fake_model(
        "model-a",
        "org/model-a",
        "transformer/model-00001-of-00002.safetensors",
        te_repo="org/model-a",
        te_path="text_encoder/model.safetensors",
    )

    shard_one = _write(cache.path_for("org/model-a", "transformer/model-00001-of-00002.safetensors"))
    shard_two = _write(cache.path_for("org/model-a", "transformer/model-00002-of-00002.safetensors"))
    partial = _write(
        cache.path_for("org/model-a", "transformer/model-00001-of-00002.safetensors.incomplete")
    )
    shared_te = _write(cache.path_for("org/model-a", "text_encoder/model.safetensors"))

    targets, protected, skipped = collect_delete_targets(model, [model, other], cache)
    result = delete_targets(cache, targets)

    assert not shard_one.exists()
    assert not shard_two.exists()
    assert not partial.exists()
    assert shared_te.is_file()
    assert "org/model-a/text_encoder/model.safetensors" in skipped
    assert result["deleted_files_count"] == 3


# ---------------------------------------------------------------------------
# Frontend wiring guards
# ---------------------------------------------------------------------------

def test_models_hub_delete_and_downloads_panel_wiring():
    app_js = (ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    styles_css = (ROOT / "frontend" / "css" / "styles.css").read_text(encoding="utf-8")

    assert "async deleteModelFiles(modelId, variantId, btnEl)" in app_js
    assert "App.deleteModelFiles(this.dataset.modelId, this.dataset.variantId, this)" in app_js
    assert "variant-state installed" in app_js
    assert "registerDownload(jobId" in app_js
    assert "renderDownloads()" in app_js
    assert "clearFinishedDownloads()" in app_js
    assert 'id="downloads-panel"' in index_html
    assert "App.clearFinishedDownloads()" in index_html
    assert ".downloads-panel" in styles_css
    assert ".btn-delete-files" in styles_css


def test_subfolder_transformer_detection_and_size_tolerance(tmp_path):
    from backend.cache import CacheManager, FileState

    cache = CacheManager(cache_dir=tmp_path / "cache")
    repo_id = "test/subfolder-model"
    rd = cache.repo_dir_for(repo_id)

    # 1. Place file in transformer/ subfolder while querying by flat filename
    tr_file = rd / "transformer" / "model.safetensors"
    tr_file.parent.mkdir(parents=True, exist_ok=True)
    # 5.80 GB decimal vs 6.22 GB binary mismatch simulation (7% difference)
    # We write 5000 bytes with expected 5368 bytes (6.8% difference)
    tr_file.write_bytes(b"x" * 5000)

    found = cache.find_existing_file(repo_id, "model.safetensors")
    assert found == tr_file, f"Expected {tr_file}, got {found}"

    entry = cache.check_file(
        repo_id=repo_id,
        file_path="model.safetensors",
        expected_size=5368,
        compute_checksum=False,
    )
    assert entry.state == FileState.INSTALLED
    assert entry.actual_size == 5000


def test_te_missing_guard_in_app_js():
    app_js = (ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    assert "const teMissing = teList.length > 0 && !teList.some(te => te.installed);" in app_js
    assert "const installedVar = variants.find(v => v.state === \"installed\" || v.installed);" in app_js
