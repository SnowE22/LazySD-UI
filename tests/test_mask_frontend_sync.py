"""Synchronisation + integration tests for the mask / inpaint feature.

Checks that the active frontend (frontend/js/app.js) and the ES6 mirrors
(frontend/js/modules/*) expose the same mask API, that index.html wires the
new UI, and that the backend refuses a mask for a model that cannot use one
*before* the job is started.
"""
import base64
import io
import sys
from pathlib import Path

from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

FRONTEND = root_dir / "frontend"
INDEX_HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "js" / "app.js").read_text(encoding="utf-8")
MASK_CORE_JS = (FRONTEND / "js" / "modules" / "mask-core.js").read_text(encoding="utf-8")
GEN_MASK_JS = (FRONTEND / "js" / "modules" / "generate-mask.js").read_text(encoding="utf-8")
INPAINT_JS = (FRONTEND / "js" / "modules" / "inpaint.js").read_text(encoding="utf-8")
GENERATOR_JS = (FRONTEND / "js" / "modules" / "generator.js").read_text(encoding="utf-8")


def _mask_data_url() -> str:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_index_html_wires_mask_and_inpaint_ui():
    for marker in (
        'id="gen-mask-canvas"',
        'id="gen-mask-panel"',
        'id="gen-mask-toggle"',
        'id="gen-mask-tool-brush"',
        'id="gen-mask-tool-eraser"',
        'id="gen-mask-undo"',
        'id="gen-mask-redo"',
        'id="gen-mask-clear"',
        'id="gen-mask-invert"',
        'id="gen-mask-size"',
        'id="gen-mask-softness"',
        'id="tab-inpaint"',
        'id="inpaint-canvas-area"',
        'id="inpaint-canvas-wrap"',
        'id="inpaint-composite"',
        'id="inpaint-mask-stack"',
        'id="inpaint-layers-list"',
        'id="inpaint-layer-opacity"',
        'id="inpaint-prompt"',
        'id="inpaint-model-select"',
        'id="inpaint-generate-btn"',
    ):
        assert marker in INDEX_HTML, f"index.html is missing {marker}"


def test_inpaint_nav_item_is_enabled():
    assert 'data-tab="inpaint" disabled' not in INDEX_HTML, "Inpaint menu item must be enabled"
    assert 'data-tab="inpaint"' in INDEX_HTML


def test_app_js_and_modules_expose_the_same_mask_api():
    app_api = [
        "createMaskSurface",
        "pointerToCanvasPoint",
        "syncOverlayGeometry",
        "getGenMaskState",
        "toggleGenMask",
        "setGenMaskTool",
        "setGenMaskBrushSize",
        "setGenMaskSoftness",
        "undoGenMask",
        "redoGenMask",
        "clearGenMask",
        "invertGenMask",
        "applyGenMaskParams",
        "getMaskCapableModels",
        "initGenMaskUI",
        "getInpaintState",
        "loadInpaintSource",
        "renderInpaintLayersUI",
        "renderInpaintComposite",
        "addInpaintMaskLayer",
        "addInpaintResultLayer",
        "selectInpaintLayer",
        "removeInpaintLayer",
        "duplicateInpaintLayer",
        "moveInpaintLayer",
        "toggleInpaintLayerVisible",
        "renameInpaintLayer",
        "setInpaintLayerOpacity",
        "undoInpaint",
        "redoInpaint",
        "clearInpaintMask",
        "invertInpaintMask",
        "generateInpaint",
        "renderInpaintModels",
        "updateInpaintModelWarning",
        "initInpaintStudio",
    ]
    for name in app_api:
        assert name in APP_JS, f"app.js is missing {name}"

    module_api = {
        MASK_CORE_JS: [
            "export function createMaskSurface",
            "export function pointerToCanvasPoint",
            "export function syncOverlayGeometry",
        ],
        GEN_MASK_JS: [
            "export function getGenMaskState",
            "export function toggleGenMask",
            "export function applyGenMaskParams",
            "export function getMaskCapableModels",
            "export function initGenMaskUI",
        ],
        INPAINT_JS: [
            "export function getInpaintState",
            "export function renderInpaintLayersUI",
            "export function renderInpaintComposite",
            "export function addInpaintMaskLayer",
            "export function addInpaintResultLayer",
            "export function selectInpaintLayer",
            "export function removeInpaintLayer",
            "export function duplicateInpaintLayer",
            "export function moveInpaintLayer",
            "export function toggleInpaintLayerVisible",
            "export function renameInpaintLayer",
            "export function setInpaintLayerOpacity",
            "export function undoInpaint",
            "export function redoInpaint",
            "export function clearInpaintMask",
            "export function invertInpaintMask",
            "export async function generateInpaint",
            "export async function loadInpaintSource",
            "export function initInpaintStudio",
        ],
    }
    for source, markers in module_api.items():
        for marker in markers:
            assert marker in source, f"{source} is missing {marker}"


def test_generate_sends_the_mask_to_the_backend():
    assert "applyGenMaskParams" in GENERATOR_JS
    assert "params.mask =" in GEN_MASK_JS
    assert "params.mask_strength" in GEN_MASK_JS
    assert "supports_mask" in GENERATOR_JS, "Generate must pre-flight the mask capability"


def test_generate_mask_uses_the_active_reference_image():
    assert "this.state.img2imgRef" in APP_JS
    assert "reference?.url || this.state.currentPreviewUrl" in APP_JS
    assert "reference?.url || app.state.currentPreviewUrl" in GEN_MASK_JS
    assert "params.init_image_path = state.source.path" in GEN_MASK_JS
    assert "app.setGenMaskSource({ path, url, name: path })" in GENERATOR_JS


def test_inpaint_upload_drag_drop_and_clear_controls_are_wired():
    for marker in (
        'id="inpaint-canvas-actions"',
        'id="inpaint-clear-source"',
        'id="inpaint-drop-overlay"',
        "inpaint.action.upload",
        "inpaint.action.clear",
    ):
        assert marker in INDEX_HTML or marker in (FRONTEND / "locale" / "ru.json").read_text(encoding="utf-8")
    assert "async apiUpload(file)" in APP_JS
    assert "const res = await this._fetchWithRetry(\"/api/upload\"" in APP_JS
    assert 'addEventListener("drop"' in APP_JS
    assert 'addEventListener("dragover"' in APP_JS
    assert "clearInpaintSource()" in APP_JS
    assert "export function clearInpaintSource" in INPAINT_JS
    assert "apiUpload(file)" in INPAINT_JS


def test_locales_contain_the_mask_and_inpaint_keys():
    import json

    for name in ("ru.json", "en.json"):
        data = json.loads((FRONTEND / "locale" / name).read_text(encoding="utf-8"))
        for key in (
            "generate.mask.title",
            "generate.mask.hint_load",
            "generate.mask.brush",
            "generate.mask.eraser",
            "generate.mask.undo",
            "generate.mask.redo",
            "generate.mask.clear",
            "generate.mask.invert",
            "generate.mask.size",
            "generate.mask.softness",
            "generate.mask.model_unsupported",
            "inpaint.generate",
            "inpaint.layer.source",
            "inpaint.layer.mask",
            "inpaint.layer.result",
            "inpaint.layers.title",
            "inpaint.layers.opacity",
            "inpaint.hint.model_unsupported",
            "inpaint.action.upload",
            "inpaint.action.clear",
            "inpaint.action.clear_title",
            "inpaint.drop_here",
        ):
            assert key in data, f"{name} is missing {key}"


def test_cache_busting_for_app_js():
    assert 'src="/js/app.js?v=' in INDEX_HTML


def test_generate_endpoint_rejects_mask_for_unsupported_model(monkeypatch):
    """The user must get a clear error BEFORE the job starts."""
    import app as app_module
    from fastapi.testclient import TestClient

    application = app_module.create_app()
    client = TestClient(application)

    class _Settings:
        enable_quantfunc_models = True
        engine_version = "0.0.17"

    class _Gpu:
        sm = 86
        tier = "50x-above"
        vram_gb = 24.0
        name = "Test GPU"

    monkeypatch.setattr(app_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(app_module, "is_engine_installed", lambda *a, **k: True)
    monkeypatch.setattr(app_module.STATE, "gpu_info", _Gpu())

    response = client.post("/api/generate", json={
        "model_id": "flux_klein_4b",   # text_to_image -> no mask support
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "init_image_path": "some_image.png",
        "mask": _mask_data_url(),
    })

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, dict), "the error must carry structured details"
    assert detail["error"] == "mask_unsupported"
    assert "does not support inpaint masks" in detail["message"]
    compatible = detail["compatible_models"]
    assert compatible, "the error must list the mask capable models"
    assert any(item["id"] == "qwen_image_edit" for item in compatible)


def test_generate_endpoint_requires_a_source_image_for_a_mask(monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    application = app_module.create_app()
    client = TestClient(application)

    class _Settings:
        enable_quantfunc_models = True
        engine_version = "0.0.17"

    class _Gpu:
        sm = 86
        tier = "50x-above"
        vram_gb = 24.0
        name = "Test GPU"

    monkeypatch.setattr(app_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(app_module, "is_engine_installed", lambda *a, **k: True)
    monkeypatch.setattr(app_module.STATE, "gpu_info", _Gpu())

    response = client.post("/api/generate", json={
        "model_id": "qwen_image_edit",
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "mask": _mask_data_url(),
    })

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str) and "source image" in detail.lower()


def test_generate_endpoint_rejects_malformed_present_mask():
    import app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.create_app())
    response = client.post("/api/generate", json={
        "model_id": "flux_klein_4b",
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "mask": "not-an-image-data-url",
    })
    assert response.status_code == 400
    assert "mask must" in response.json()["detail"].lower()


def test_generate_endpoint_rejects_mask_path_traversal(monkeypatch, tmp_path):
    import app as app_module
    from fastapi.testclient import TestClient

    application = app_module.create_app()
    client = TestClient(application)

    class _Settings:
        enable_quantfunc_models = True
        engine_version = "0.0.17"
        output_dir = tmp_path

    class _Gpu:
        sm = 86
        tier = "50x-above"
        vram_gb = 24.0
        name = "Test GPU"

    monkeypatch.setattr(app_module, "get_settings", lambda: _Settings())
    monkeypatch.setattr(app_module, "is_engine_installed", lambda *a, **k: True)
    monkeypatch.setattr(app_module.STATE, "gpu_info", _Gpu())

    response = client.post("/api/generate", json={
        "model_id": "qwen_image_edit",
        "prompt": "test prompt",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "init_image_path": "source.png",
        "mask_path": "..\\..\\outside.png",
    })
    assert response.status_code == 400
    assert "path" in response.json()["detail"].lower()


def test_frontend_snapshots_strokes_and_jobs():
    for source in (APP_JS, INPAINT_JS):
        assert "strokeSnapshot" in source
        assert "maskDataUrl" in source
        assert "sourceToken" in source
        assert "sourceLoadSeq" in source
        assert "state.layers.push(layer)" in source
    for source in (APP_JS, GEN_MASK_JS):
        assert "sourceLoadSeq" in source
        assert "surface.reset(state.source.width, state.source.height)" in source


def test_models_endpoint_exposes_the_mask_capability():
    import app as app_module
    from fastapi.testclient import TestClient

    application = app_module.create_app()
    client = TestClient(application)

    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    models = payload["models"] if isinstance(payload, dict) else payload
    by_id = {m["id"]: m for m in models}
    assert "supports_mask" in by_id["qwen_image_edit"]
    assert by_id["qwen_image_edit"]["supports_mask"] is True
    # QF FLUX cards stay untouched.
    assert by_id["flux_klein_4b"]["supports_mask"] is False
    assert by_id["flux_klein_9b"]["supports_mask"] is False
    # Nunchaku / OrbitQuant / SD.CPP FLUX cards opt in.
    for model_id in (
        "flux_2_klein_4b_nunchaku",
        "flux_2_klein_4b_orbitquant",
        "flux_2_klein_9b_nunchaku",
        "flux_2_klein_9b_kv_nunchaku",
        "flux_2_klein_9b_orbitquant",
        "flux_2_klein_base_9b_gguf",
        "flux_3_image_klein_gguf",
        "flux_schnell_gguf",
    ):
        assert by_id[model_id]["supports_mask"] is True, model_id
    assert by_id["flux_3_image_klein_nunchaku"]["supports_mask"] is False
