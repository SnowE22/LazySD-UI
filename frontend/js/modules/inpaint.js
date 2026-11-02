/* ============================================================
   QuantFunc WebUI — Inpaint Studio (layered workspace)
   Layers: Исходник (locked base) / Маска (editable) / Результат (generated fragment).
   The result layer is clipped by the mask it was generated from, so the
   original layer is never modified.
   ============================================================ */

import { apiGet, apiPost, apiUpload, toast, escapeHtml } from "../api.js";
import {
  createMaskSurface,
  pointerToCanvasPoint,
  MASK_DEFAULT_BRUSH_SIZE,
  MASK_DEFAULT_SOFTNESS,
} from "./mask-core.js";

const INPAINT_MAX_HISTORY = 12;

let layerSeq = 0;
function nextLayerId(prefix) {
  layerSeq += 1;
  return `${prefix}_${Date.now().toString(36)}_${layerSeq.toString(36)}`;
}

// ---------- state ----------
export function getInpaintState(appRef = null) {
  const app = appRef || window.App;
  if (!app) return null;
  if (!app.state.inpaint) {
    app.state.inpaint = {
      source: null,
      layers: [],
      activeLayerId: null,
      history: [],
      future: [],
      brushSize: MASK_DEFAULT_BRUSH_SIZE,
      softness: MASK_DEFAULT_SOFTNESS,
      mode: "brush",
      generating: false,
      applyingResult: false,
      jobId: null,
      pollTimer: null,
      lastMaskSnapshot: null,
      sourceLoadSeq: 0,
      jobContext: null,
      // Generation area (crop) - see the "generation area" helpers below.
      crop: {
        mode: "auto",
        padding: 64,
        minSize: 256,
        box: null,
        selecting: false,
        dragStart: null,
        dragRect: null,
      },
    };
  }
  return app.state.inpaint;
}

function t(key, fallback, vars) {
  const fn = typeof window !== "undefined" ? window.t : null;
  if (typeof fn === "function") {
    const value = fn(key, vars || {});
    if (value && value !== key) return value;
  }
  return fallback || key;
}

// ---------- helpers ----------
function compositeCanvas(app) {
  return document.getElementById("inpaint-composite");
}

function maskStack(app) {
  return document.getElementById("inpaint-mask-stack");
}

function activeLayer(state) {
  if (!state) return null;
  return state.layers.find((l) => l.id === state.activeLayerId) || null;
}

function activeMaskLayer(state) {
  const layer = activeLayer(state);
  return layer && layer.type === "mask" ? layer : null;
}

function layerById(state, id) {
  return state.layers.find((l) => l.id === id) || null;
}

// ---------- mask layer plumbing ----------
function createMaskLayer(state, name) {
  const width = state.source?.width || 1024;
  const height = state.source?.height || 1024;

  const dataCanvas = document.createElement("canvas");
  dataCanvas.width = width;
  dataCanvas.height = height;

  const viewCanvas = document.createElement("canvas");
  viewCanvas.width = width;
  viewCanvas.height = height;
  viewCanvas.className = "inpaint-mask-canvas";
  viewCanvas.dataset.layerId = "";

  const surface = createMaskSurface(dataCanvas, { visualCanvas: viewCanvas });

  const layer = {
    id: nextLayerId("mask"),
    type: "mask",
    name: name || t("inpaint.layer.mask", "Маска"),
    visible: true,
    opacity: 1,
    locked: false,
    surface,
    dataCanvas,
    viewCanvas,
    // Bumped on every mask mutation so the cached mask bbox stays valid.
    revision: 0,
    maskRev: layerSeq,
  };
  viewCanvas.dataset.layerId = layer.id;
  return layer;
}

function attachMaskLayerView(app, layer) {
  const stack = maskStack(app);
  if (!stack || !layer) return;
  if (layer.viewCanvas && layer.viewCanvas.parentElement !== stack) {
    stack.appendChild(layer.viewCanvas);
  }
  layer.viewCanvas.style.display = "none";
  layer.viewCanvas.style.pointerEvents = "none";
}

function refreshMaskLayerViews(app) {
  const state = getInpaintState(app);
  if (!state) return;
  for (const layer of state.layers) {
    if (layer.type !== "mask") continue;
    if (!layer.viewCanvas) continue;
    const isActive = layer.id === state.activeLayerId;
    layer.viewCanvas.style.display = isActive && layer.visible ? "block" : "none";
    layer.viewCanvas.style.pointerEvents = isActive && layer.visible ? "auto" : "none";
    layer.viewCanvas.style.opacity = String(layer.opacity ?? 1);
  }
}

// ---------- rendering ----------
export function renderInpaintComposite(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  const canvas = compositeCanvas(app);
  if (!canvas || !state) return;

  const width = state.source?.width || canvas.width || 1024;
  const height = state.source?.height || canvas.height || 1024;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  for (const layer of state.layers) {
    if (!layer.visible) continue;
    if (layer.type === "source") {
      if (layer.image) {
        ctx.save();
        ctx.globalAlpha = Number.isFinite(layer.opacity) ? layer.opacity : 1;
        ctx.drawImage(layer.image, 0, 0, width, height);
        ctx.restore();
      }
    } else if (layer.type === "result") {
      drawResultLayer(ctx, layer, width, height);
    }
  }
}

function drawResultLayer(ctx, layer, width, height) {
  if (!layer.image) return;
  const tmp = getScratchCanvas(width, height);
  const tctx = tmp.getContext("2d");
  tctx.clearRect(0, 0, width, height);
  tctx.drawImage(layer.image, 0, 0, width, height);

  if (layer.maskImage) {
    tctx.globalCompositeOperation = "destination-in";
    tctx.drawImage(layer.maskImage, 0, 0, width, height);
    tctx.globalCompositeOperation = "source-over";
  }

  ctx.save();
  ctx.globalAlpha = Number.isFinite(layer.opacity) ? layer.opacity : 1;
  ctx.drawImage(tmp, 0, 0);
  ctx.restore();
}

let _scratch = null;
function getScratchCanvas(width, height) {
  if (!_scratch) _scratch = document.createElement("canvas");
  if (_scratch.width !== width || _scratch.height !== height) {
    _scratch.width = width;
    _scratch.height = height;
  }
  return _scratch;
}

// ---------- history (strokes + layer operations) ----------
function captureSnapshot(state) {
  return {
    activeLayerId: state.activeLayerId,
    layers: state.layers.map((layer) => ({
      id: layer.id,
      type: layer.type,
      name: layer.name,
      visible: layer.visible,
      opacity: layer.opacity,
      url: layer.url || null,
      maskDataUrl: layer.type === "mask" ? layer.surface.toDataURL() : null,
      maskImageSrc: layer.type === "result" ? layer.maskImageSrc || null : null,
    })),
  };
}

function pushHistory(state, snapshot = null) {
  state.history.push(snapshot || captureSnapshot(state));
  while (state.history.length > INPAINT_MAX_HISTORY) state.history.shift();
  state.future.length = 0;
}

async function restoreSnapshot(app, snapshot) {
  const state = getInpaintState(app);
  if (!state || !snapshot) return;

  const stack = maskStack(app);
  if (stack) stack.innerHTML = "";

  const restored = [];
  for (const saved of snapshot.layers) {
    if (saved.type === "source") {
      const source = state.source;
      if (!source) continue;
      restored.push({
        id: saved.id,
        type: "source",
        name: saved.name,
        visible: saved.visible,
        opacity: saved.opacity,
        locked: true,
        image: source.img,
        url: source.url,
      });
    } else if (saved.type === "mask") {
      const layer = createMaskLayer(state, saved.name);
      layer.id = saved.id;
      layer.visible = saved.visible;
      layer.opacity = saved.opacity;
      if (saved.maskDataUrl) await layer.surface.loadDataURL(saved.maskDataUrl);
      attachMaskLayerView(app, layer);
      restored.push(layer);
    } else if (saved.type === "result") {
      const layer = {
        id: saved.id,
        type: "result",
        name: saved.name,
        visible: saved.visible,
        opacity: saved.opacity,
        url: saved.url || null,
        image: null,
        maskImage: null,
        maskImageSrc: saved.maskImageSrc || null,
      };
      if (layer.url) {
        const img = new Image();
        await new Promise((resolve) => {
          img.onload = resolve;
          img.onerror = resolve;
          img.src = layer.url;
        });
        layer.image = img;
      }
      if (layer.maskImageSrc) {
        const mimg = new Image();
        await new Promise((resolve) => {
          mimg.onload = resolve;
          mimg.onerror = resolve;
          mimg.src = layer.maskImageSrc;
        });
        layer.maskImage = mimg;
      }
      restored.push(layer);
    }
  }

  state.layers = restored;
  state.activeLayerId = snapshot.activeLayerId;
  if (!layerById(state, state.activeLayerId)) {
    const fallback = state.layers.find((l) => l.type === "mask") || state.layers[0] || null;
    state.activeLayerId = fallback ? fallback.id : null;
  }
  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
}

export function undoInpaint(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state || !state.history.length) return;
  const current = captureSnapshot(state);
  const previous = state.history.pop();
  state.future.push(current);
  restoreSnapshot(app, previous);
}

export function redoInpaint(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state || !state.future.length) return;
  const current = captureSnapshot(state);
  const next = state.future.pop();
  state.history.push(current);
  restoreSnapshot(app, next);
}

// ---------- source ----------
export async function loadInpaintSource(url, path, name, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!app || !state) return false;
  if (state.generating || state.applyingResult) {
    toast("warning", t("inpaint.hint.generating", "Генерация уже выполняется"));
    return;
  }

  const loadSeq = (state.sourceLoadSeq || 0) + 1;
  state.sourceLoadSeq = loadSeq;
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("Не удалось загрузить изображение"));
    img.src = url;
  }).catch((err) => {
    toast("error", err.message || "Не удалось загрузить изображение");
  });

  if (state.sourceLoadSeq !== loadSeq || !img.complete || !img.naturalWidth) return false;

  state.source = {
    token: loadSeq,
    url,
    path: path || "",
    name: name || (url.split("/").pop() || "image.png"),
    width: img.naturalWidth,
    height: img.naturalHeight,
    img,
  };

  state.layers = [];
  state.history.length = 0;
  state.future.length = 0;

  const sourceLayer = {
    id: nextLayerId("source"),
    type: "source",
    name: t("inpaint.layer.source", "Исходник"),
    visible: true,
    opacity: 1,
    locked: true,
    image: img,
    url,
  };
  state.layers.push(sourceLayer);
  state.activeLayerId = sourceLayer.id;

  const stack = maskStack(app);
  if (stack) stack.innerHTML = "";

  const maskLayer = createMaskLayer(state);
  attachMaskLayerView(app, maskLayer);
  state.layers.push(maskLayer);
  state.activeLayerId = maskLayer.id;

  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
  // A new source invalidates the previously selected generation area.
  resetInpaintCropSelection(app);
  updateInpaintStatus(app);
  syncInpaintGeometry(app);
  return true;
}

export function clearInpaintSource(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!app || !state) return;
  state.sourceLoadSeq = (state.sourceLoadSeq || 0) + 1;
  state.source = null;
  state.layers = [];
  state.activeLayerId = null;
  state.history.length = 0;
  state.future.length = 0;
  state.jobContext = null;

  const stack = maskStack(app);
  if (stack) stack.replaceChildren();
  const canvas = compositeCanvas(app);
  if (canvas) {
    canvas.width = 1024;
    canvas.height = 1024;
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
  renderInpaintLayersUI(app);
  resetInpaintCropSelection(app);
  updateInpaintStatus(app);
  syncInpaintGeometry(app);
}

// ---------- geometry ----------
export function syncInpaintGeometry(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!app || !state) return;
  const canvas = compositeCanvas(app);
  if (!canvas || !state.source) return;
  if (canvas.width !== state.source.width) canvas.width = state.source.width;
  if (canvas.height !== state.source.height) canvas.height = state.source.height;
  const overlay = cropOverlay();
  if (overlay) {
    if (overlay.width !== state.source.width) overlay.width = state.source.width;
    if (overlay.height !== state.source.height) overlay.height = state.source.height;
  }
  for (const layer of state.layers) {
    if (layer.type !== "mask") continue;
    if (layer.viewCanvas.width !== state.source.width) layer.viewCanvas.width = state.source.width;
    if (layer.viewCanvas.height !== state.source.height) layer.viewCanvas.height = state.source.height;
  }
}

// ---------- layers panel ----------
export function renderInpaintLayersUI(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  const list = document.getElementById("inpaint-layers-list");
  if (!app || !state || !list) return;

  const layer = activeLayer(state);
  const opacityInput = document.getElementById("inpaint-layer-opacity");
  const opacityNum = document.getElementById("inpaint-layer-opacity-num");
  const opacity = layer ? Number(layer.opacity ?? 1) : 1;
  if (opacityInput) opacityInput.value = String(opacity);
  if (opacityNum) opacityNum.textContent = opacity.toFixed(2);

  const deleteBtn = document.getElementById("inpaint-layer-delete");
  if (deleteBtn) deleteBtn.disabled = !layer || layer.type === "source";

  // Top-most layer first (like the Halftone Studio panel)
  const ordered = [...state.layers].reverse();

  list.innerHTML = ordered.map((l) => {
    const isActive = l.id === state.activeLayerId;
    const badge =
      l.type === "source" ? "🔒" : l.type === "mask" ? "🎭" : "✨";
    const thumb = layerThumb(l);
    return `
      <div class="inpaint-layer-item${isActive ? " active" : ""}" data-layer-id="${l.id}" onclick="App.selectInpaintLayer('${l.id}')">
        <span class="inpaint-layer-badge">${badge}</span>
        <span class="inpaint-layer-thumb">${thumb}</span>
        <span class="inpaint-layer-name" title="${escapeHtml(l.name)}">${escapeHtml(l.name)}</span>
        <button class="inpaint-layer-eye" title="${l.visible ? "Скрыть" : "Показать"}" onclick="event.stopPropagation(); App.toggleInpaintLayerVisible('${l.id}')">${l.visible ? "👁" : "🚫"}</button>
      </div>`;
  }).join("");

  if (!state.layers.length) {
    list.innerHTML = `<div class="inpaint-layers-empty">${t("inpaint.layers.empty", "Слоёв пока нет — загрузите изображение")}</div>`;
  }
}

function layerThumb(layer) {
  if (layer.type === "source") {
    return layer.url ? `<img src="${escapeHtml(layer.url)}" alt="" />` : "";
  }
  if (layer.type === "result") {
    return layer.url ? `<img src="${escapeHtml(layer.url)}" alt="" />` : "";
  }
  return `<span class="inpaint-layer-thumb-mask">🎭</span>`;
}

export function selectInpaintLayer(id, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = layerById(state, id);
  if (!layer) return;
  state.activeLayerId = id;
  refreshMaskLayerViews(app);
  renderInpaintLayersUI(app);
  updateInpaintStatus(app);
}

export function addInpaintMaskLayer(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  if (!state.source) {
    toast("warning", t("inpaint.hint.no_source", "Сначала загрузите изображение"));
    return;
  }
  pushHistory(state);
  const layer = createMaskLayer(state, `${t("inpaint.layer.mask", "Маска")} ${state.layers.filter((l) => l.type === "mask").length + 1}`);
  attachMaskLayerView(app, layer);
  state.layers.push(layer);
  state.activeLayerId = layer.id;
  refreshMaskLayerViews(app);
  renderInpaintLayersUI(app);
  updateInpaintStatus(app);
}

export function addInpaintResultLayer(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  if (!state.source) {
    toast("warning", t("inpaint.hint.no_source", "Сначала загрузите изображение"));
    return;
  }
  pushHistory(state);
  const layer = {
    id: nextLayerId("result"),
    type: "result",
    name: `${t("inpaint.layer.result", "Результат")} ${state.layers.filter((l) => l.type === "result").length + 1}`,
    visible: true,
    opacity: 1,
    image: null,
    maskImage: null,
    maskImageSrc: null,
    url: null,
  };
  state.layers.push(layer);
  state.activeLayerId = layer.id;
  renderInpaintLayersUI(app);
  updateInpaintStatus(app);
}

export function removeInpaintLayer(id, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = layerById(state, id);
  if (!layer) return;
  if (layer.type === "source") {
    toast("warning", t("inpaint.hint.source_locked", "Слой «Исходник» удалить нельзя"));
    return;
  }
  pushHistory(state);
  state.layers = state.layers.filter((l) => l.id !== id);
  if (layer.viewCanvas && layer.viewCanvas.parentElement) layer.viewCanvas.parentElement.removeChild(layer.viewCanvas);
  if (state.activeLayerId === id) {
    const fallback = state.layers.find((l) => l.type === "mask") || state.layers[state.layers.length - 1] || null;
    state.activeLayerId = fallback ? fallback.id : null;
  }
  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
  updateInpaintStatus(app);
}

export function duplicateInpaintLayer(id, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = layerById(state, id);
  if (!layer) return;
  if (layer.type === "source") {
    toast("warning", t("inpaint.hint.source_locked", "Слой «Исходник» дублировать нельзя"));
    return;
  }
  pushHistory(state);
  const copy = { ...layer };
  copy.id = nextLayerId(layer.type);
  copy.name = `${layer.name} ${t("inpaint.layer.copy_suffix", "(копия)")}`;
  if (layer.type === "mask") {
    copy.dataCanvas = document.createElement("canvas");
    copy.dataCanvas.width = layer.dataCanvas.width;
    copy.dataCanvas.height = layer.dataCanvas.height;
    copy.dataCanvas.getContext("2d").drawImage(layer.dataCanvas, 0, 0);
    copy.viewCanvas = document.createElement("canvas");
    copy.viewCanvas.width = layer.viewCanvas.width;
    copy.viewCanvas.height = layer.viewCanvas.height;
    copy.viewCanvas.className = "inpaint-mask-canvas";
    copy.surface = createMaskSurface(copy.dataCanvas, { visualCanvas: copy.viewCanvas });
    copy.surface.loadDataURL(layer.surface.toDataURL());
    attachMaskLayerView(app, copy);
  }
  const index = state.layers.findIndex((l) => l.id === id);
  state.layers.splice(index + 1, 0, copy);
  state.activeLayerId = copy.id;
  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
}

export function moveInpaintLayer(id, direction, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const index = state.layers.findIndex((l) => l.id === id);
  if (index < 0) return;
  const target = direction === "up" ? index + 1 : index - 1;
  if (target < 0 || target >= state.layers.length) return;
  pushHistory(state);
  const [item] = state.layers.splice(index, 1);
  state.layers.splice(target, 0, item);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
}

export function toggleInpaintLayerVisible(id, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = layerById(state, id);
  if (!layer) return;
  pushHistory(state);
  layer.visible = !layer.visible;
  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
}

export function renameInpaintLayer(id, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = layerById(state, id);
  if (!layer) return;
  const name = window.prompt(t("inpaint.layers.rename_prompt", "Название слоя"), layer.name);
  if (name === null) return;
  pushHistory(state);
  layer.name = String(name).slice(0, 80) || layer.name;
  renderInpaintLayersUI(app);
}

export function setInpaintLayerOpacity(id, value, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = layerById(state, id);
  if (!layer) return;
  layer.opacity = Math.max(0, Math.min(1, Number(value)));
  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  const num = document.getElementById("inpaint-layer-opacity-num");
  if (num) num.textContent = layer.opacity.toFixed(2);
}

// ---------- mask tools ----------
export function setInpaintTool(mode, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  state.mode = mode === "eraser" ? "eraser" : "brush";
  const layer = activeMaskLayer(state);
  if (layer) layer.surface.mode = state.mode;
  document.getElementById("inpaint-tool-brush")?.classList.toggle("active", state.mode === "brush");
  document.getElementById("inpaint-tool-eraser")?.classList.toggle("active", state.mode === "eraser");
}

export function setInpaintBrushSize(value, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  state.brushSize = Math.max(1, Math.min(4096, Number(value) || MASK_DEFAULT_BRUSH_SIZE));
  const label = document.getElementById("inpaint-brush-size-val");
  if (label) label.textContent = String(Math.round(state.brushSize));
  const layer = activeMaskLayer(state);
  if (layer) layer.surface.brushSize = state.brushSize;
  updateBrushCursor(app);
}

export function setInpaintSoftness(value, appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  state.softness = Math.max(0, Math.min(1, Number(value)));
  const label = document.getElementById("inpaint-softness-val");
  if (label) label.textContent = `${Math.round(state.softness * 100)}%`;
  const layer = activeMaskLayer(state);
  if (layer) layer.surface.softness = state.softness;
  updateBrushCursor(app);
}

export function clearInpaintMask(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = activeMaskLayer(state);
  if (!layer) {
    toast("warning", t("inpaint.hint.select_mask", "Выберите слой маски"));
    return;
  }
  pushHistory(state);
  layer.surface.clear(false);
  layer.revision = (layer.revision || 0) + 1;
  updateInpaintStatus(app);
}

export function invertInpaintMask(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = activeMaskLayer(state);
  if (!layer) {
    toast("warning", t("inpaint.hint.select_mask", "Выберите слой маски"));
    return;
  }
  pushHistory(state);
  layer.surface.invert(false);
  layer.revision = (layer.revision || 0) + 1;
  updateInpaintStatus(app);
}

// ---------- generation area (crop modes) ----------
function cropOverlay() {
  return document.getElementById("inpaint-crop-overlay");
}

export function getInpaintCropState(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!state) return null;
  if (!state.crop) {
    state.crop = {
      mode: "auto",
      padding: 64,
      minSize: 256,
      box: null,
      selecting: false,
      dragStart: null,
      dragRect: null,
    };
  }
  return state.crop;
}

// Mirrors backend/masking.py :: resolve_manual_crop_box().
export function resolveInpaintManualCropBox(box, width, height) {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const align = 16;
  const minSize = 64;
  const w = Math.max(1, Math.round(Number(width) || 0));
  const h = Math.max(1, Math.round(Number(height) || 0));
  let x0 = Math.round(Number(box[0]) || 0);
  let y0 = Math.round(Number(box[1]) || 0);
  let x1 = Math.round(Number(box[2]) || 0);
  let y1 = Math.round(Number(box[3]) || 0);
  if (x0 > x1) { const tmp = x0; x0 = x1; x1 = tmp; }
  if (y0 > y1) { const tmp = y0; y0 = y1; y1 = tmp; }
  x0 = Math.max(0, Math.min(w, x0));
  y0 = Math.max(0, Math.min(h, y0));
  x1 = Math.max(0, Math.min(w, x1));
  y1 = Math.max(0, Math.min(h, y1));
  if (x1 <= x0 || y1 <= y0) return null;
  x0 = Math.floor(x0 / align) * align;
  y0 = Math.floor(y0 / align) * align;
  x1 = Math.min(w, Math.ceil(x1 / align) * align);
  y1 = Math.min(h, Math.ceil(y1 / align) * align);
  if (x1 - x0 < minSize || y1 - y0 < minSize) return null;
  return [x0, y0, x1, y1];
}

// Bounding box of the painted mask, in source pixels.
// The result is cached - the mask only changes when a stroke is finished.
export function getInpaintMaskBBox(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  const layer = activeMaskLayer(state);
  if (!layer || !layer.dataCanvas) return null;
  const crop = getInpaintCropState(app);
  const key = `${layer.id}:${layer.maskRev || 0}:${layer.revision || 0}`;
  if (crop && crop.maskBBoxKey === key) return crop.maskBBox;

  const canvas = layer.dataCanvas;
  const width = canvas.width || 0;
  const height = canvas.height || 0;
  if (!width || !height) return null;
  let data;
  try {
    data = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  } catch (err) {
    return null;
  }
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      // alpha / 255 > 0.004  <=>  alpha >= 2 (same threshold as the backend)
      if (data[(row + x) * 4 + 3] > 1) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  const box = maxX < 0 ? null : [minX, minY, maxX + 1, maxY + 1];
  if (crop) {
    crop.maskBBoxKey = key;
    crop.maskBBox = box;
  }
  return box;
}

// Human readable problem of the current manual selection, or null.
export function getInpaintCropProblem(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  const crop = getInpaintCropState(app);
  if (!state || !crop || crop.mode !== "manual") return null;
  if (!crop.box) {
    return t(
      "inpaint.crop.error.no_box",
      "Сначала выделите область генерации на изображении"
    );
  }
  const resolved = resolveInpaintManualCropBox(
    crop.box,
    state.source ? state.source.width : 0,
    state.source ? state.source.height : 0
  );
  if (!resolved) {
    return t(
      "inpaint.crop.error.too_small",
      "Область слишком маленькая (минимум 64 px по стороне)"
    );
  }
  const bbox = getInpaintMaskBBox(app);
  if (
    bbox &&
    (bbox[0] < resolved[0] || bbox[1] < resolved[1] ||
      bbox[2] > resolved[2] || bbox[3] > resolved[3])
  ) {
    return t(
      "inpaint.crop.error.not_covered",
      "Выделение не покрывает маску целиком — расширьте его"
    );
  }
  return null;
}

export function setInpaintCropMode(mode, appRef = null) {
  const app = appRef || window.App;
  const crop = getInpaintCropState(app);
  if (!crop) return;
  crop.mode = ["auto", "parameters", "manual"].includes(mode) ? mode : "auto";
  if (crop.mode !== "manual") {
    crop.selecting = false;
    crop.dragStart = null;
    crop.dragRect = null;
  }
  renderInpaintCropUI(app);
  renderInpaintCropOverlay(app);
  updateInpaintCropStatus(app);
}

export function setInpaintCropPadding(value, appRef = null) {
  const app = appRef || window.App;
  const crop = getInpaintCropState(app);
  if (!crop) return;
  const num = parseInt(value, 10);
  crop.padding = Number.isFinite(num) ? Math.max(0, Math.min(1024, num)) : 64;
  updateInpaintCropStatus(app);
}

export function setInpaintCropMinSize(value, appRef = null) {
  const app = appRef || window.App;
  const crop = getInpaintCropState(app);
  if (!crop) return;
  const num = parseInt(value, 10);
  crop.minSize = Number.isFinite(num) ? Math.max(64, Math.min(4096, num)) : 256;
  updateInpaintCropStatus(app);
}

export function toggleInpaintCropSelection(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  const crop = getInpaintCropState(app);
  if (!state || !crop) return;
  if (!state.source) {
    toast("warning", t("inpaint.hint.no_source", "Сначала загрузите изображение"));
    return;
  }
  if (crop.mode !== "manual") setInpaintCropMode("manual", app);
  crop.selecting = !crop.selecting;
  crop.dragStart = null;
  crop.dragRect = null;
  renderInpaintCropUI(app);
  renderInpaintCropOverlay(app);
  updateInpaintCropStatus(app);
}

export function resetInpaintCropSelection(appRef = null) {
  const app = appRef || window.App;
  const crop = getInpaintCropState(app);
  if (!crop) return;
  crop.box = null;
  crop.selecting = false;
  crop.dragStart = null;
  crop.dragRect = null;
  renderInpaintCropUI(app);
  renderInpaintCropOverlay(app);
  updateInpaintCropStatus(app);
}

function normalizeInpaintCropRect(a, b, width, height) {
  if (!a || !b) return null;
  const w = Math.max(1, Math.round(Number(width) || 0));
  const h = Math.max(1, Math.round(Number(height) || 0));
  const x0 = Math.max(0, Math.min(w, Math.min(a.x, b.x)));
  const y0 = Math.max(0, Math.min(h, Math.min(a.y, b.y)));
  const x1 = Math.max(0, Math.min(w, Math.max(a.x, b.x)));
  const y1 = Math.max(0, Math.min(h, Math.max(a.y, b.y)));
  return [x0, y0, x1, y1];
}

export function renderInpaintCropUI(appRef = null) {
  const app = appRef || window.App;
  const crop = getInpaintCropState(app);
  if (!crop) return;

  const modeSelect = document.getElementById("inpaint-crop-mode");
  if (modeSelect && modeSelect.value !== crop.mode) modeSelect.value = crop.mode;

  const params = document.getElementById("inpaint-crop-params");
  if (params) params.style.display = crop.mode === "parameters" ? "flex" : "none";

  const manual = document.getElementById("inpaint-crop-manual");
  if (manual) manual.style.display = crop.mode === "manual" ? "flex" : "none";

  const desc = document.getElementById("inpaint-crop-desc");
  if (desc) {
    const key = crop.mode === "auto"
      ? "inpaint.crop.desc.auto"
      : (crop.mode === "parameters"
        ? "inpaint.crop.desc.parameters"
        : "inpaint.crop.desc.manual");
    desc.textContent = t(key, "");
  }

  const selectBtn = document.getElementById("inpaint-crop-select");
  if (selectBtn) {
    selectBtn.classList.toggle("active", !!crop.selecting);
    selectBtn.textContent = crop.selecting
      ? t("inpaint.crop.selecting", "Отмена выделения")
      : t("inpaint.crop.select", "Выделить область");
  }

  const resetBtn = document.getElementById("inpaint-crop-reset");
  if (resetBtn) resetBtn.disabled = !crop.box;

  const paddingInput = document.getElementById("inpaint-crop-padding");
  if (paddingInput && document.activeElement !== paddingInput) {
    paddingInput.value = String(crop.padding);
  }
  const minSizeInput = document.getElementById("inpaint-crop-min-size");
  if (minSizeInput && document.activeElement !== minSizeInput) {
    minSizeInput.value = String(crop.minSize);
  }

  updateInpaintCropOverlayVisibility(app);
}

function updateInpaintCropOverlayVisibility(appRef = null) {
  const app = appRef || window.App;
  const overlay = cropOverlay();
  if (!overlay) return;
  const state = getInpaintState(app);
  const crop = getInpaintCropState(app);
  if (!state || !crop) return;
  const visible = crop.mode === "manual" && !!state.source;
  overlay.style.display = visible ? "block" : "none";
  overlay.style.pointerEvents = crop.selecting && visible ? "auto" : "none";
  overlay.classList.toggle("selecting", !!crop.selecting && visible);
}

export function renderInpaintCropOverlay(appRef = null) {
  const app = appRef || window.App;
  const overlay = cropOverlay();
  if (!overlay) return;
  const state = getInpaintState(app);
  const crop = getInpaintCropState(app);
  if (!state || !crop) return;

  const width = state.source ? state.source.width : overlay.width;
  const height = state.source ? state.source.height : overlay.height;
  if (overlay.width !== width) overlay.width = width;
  if (overlay.height !== height) overlay.height = height;

  const ctx = overlay.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);

  const rect = crop.dragRect || crop.box;
  if (!rect) return;
  const x0 = Math.max(0, Math.min(width, rect[0]));
  const y0 = Math.max(0, Math.min(height, rect[1]));
  const x1 = Math.max(0, Math.min(width, rect[2]));
  const y1 = Math.max(0, Math.min(height, rect[3]));
  if (x1 <= x0 || y1 <= y0) return;

  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.beginPath();
  ctx.rect(0, 0, width, height);
  ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.fill("evenodd");
  ctx.restore();

  const scale = Math.max(1, width / 1024);
  const problem = getInpaintCropProblem(app);
  ctx.strokeStyle = problem ? "#f59e0b" : (crop.selecting ? "#38bdf8" : "#a78bfa");
  ctx.lineWidth = Math.max(2, Math.round(3 * scale));
  ctx.strokeRect(
    x0 + 0.5,
    y0 + 0.5,
    Math.max(0, x1 - x0 - 1),
    Math.max(0, y1 - y0 - 1)
  );

  const handle = Math.max(5, Math.round(9 * scale));
  ctx.fillStyle = ctx.strokeStyle;
  const corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]];
  corners.forEach(([cx, cy]) => {
    ctx.fillRect(cx - handle / 2, cy - handle / 2, handle, handle);
  });
}

export function updateInpaintCropStatus(appRef = null) {
  const app = appRef || window.App;
  const crop = getInpaintCropState(app);
  if (!crop) return;

  const coords = document.getElementById("inpaint-crop-coords");
  if (coords) {
    if (crop.mode !== "manual") {
      coords.textContent = "";
    } else if (!crop.box) {
      coords.textContent = t("inpaint.crop.no_selection", "Область не выбрана");
    } else {
      const x0 = Math.round(crop.box[0]);
      const y0 = Math.round(crop.box[1]);
      const w = Math.round(crop.box[2] - crop.box[0]);
      const h = Math.round(crop.box[3] - crop.box[1]);
      coords.textContent = `X ${x0} - Y ${y0} - W ${w} - H ${h}`;
    }
  }

  const warning = document.getElementById("inpaint-crop-warning");
  if (warning) {
    const problem = getInpaintCropProblem(app);
    warning.style.display = problem ? "block" : "none";
    if (problem) warning.textContent = problem;
  }
}

export function initInpaintCropEvents(appRef = null) {
  const app = appRef || window.App;
  if (!app) return;
  const overlay = cropOverlay();
  if (!overlay) return;
  if (overlay.dataset.eventsBound === "1") return;
  overlay.dataset.eventsBound = "1";

  overlay.addEventListener("pointerdown", (event) => {
    const crop = getInpaintCropState(app);
    const state = getInpaintState(app);
    if (!crop || !state || !crop.selecting || event.button !== 0) return;
    if (!state.source) return;
    event.preventDefault();
    event.stopPropagation();
    const point = pointerToCanvasPoint(overlay, event);
    if (!point) return;
    crop.dragStart = point;
    crop.dragRect = null;
    try { overlay.setPointerCapture(event.pointerId); } catch (_) {}
    renderInpaintCropOverlay(app);
  });

  overlay.addEventListener("pointermove", (event) => {
    const crop = getInpaintCropState(app);
    const state = getInpaintState(app);
    if (!crop || !state || !crop.selecting || !crop.dragStart) return;
    const point = pointerToCanvasPoint(overlay, event);
    if (!point) return;
    crop.dragRect = normalizeInpaintCropRect(
      crop.dragStart,
      point,
      state.source ? state.source.width : 0,
      state.source ? state.source.height : 0
    );
    renderInpaintCropOverlay(app);
    updateInpaintCropStatus(app);
  });

  const finish = (event, discard) => {
    const crop = getInpaintCropState(app);
    const state = getInpaintState(app);
    if (!crop || !state || !crop.selecting || !crop.dragStart) return;
    const point = pointerToCanvasPoint(overlay, event) || crop.dragStart;
    const rect = normalizeInpaintCropRect(
      crop.dragStart,
      point,
      state.source ? state.source.width : 0,
      state.source ? state.source.height : 0
    );
    crop.dragStart = null;
    crop.dragRect = null;
    crop.selecting = false;
    try { overlay.releasePointerCapture(event.pointerId); } catch (_) {}
    if (!discard && rect && rect[2] - rect[0] >= 1 && rect[3] - rect[1] >= 1) {
      crop.box = rect;
    }
    renderInpaintCropUI(app);
    renderInpaintCropOverlay(app);
    updateInpaintCropStatus(app);
    const problem = getInpaintCropProblem(app);
    if (problem) toast("warning", problem);
  };

  overlay.addEventListener("pointerup", (event) => finish(event, false));
  overlay.addEventListener("pointercancel", (event) => finish(event, true));
}

// ---------- status ----------
export function updateInpaintStatus(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!app || !state) return;
  const statusEl = document.getElementById("inpaint-status");
  const hintEl = document.getElementById("inpaint-empty-hint");
  const clearBtn = document.getElementById("inpaint-clear-source");
  const layer = activeMaskLayer(state);

  if (hintEl) hintEl.style.display = state.source ? "none" : "flex";
  if (clearBtn) clearBtn.disabled = !state.source;

  if (!statusEl) return;
  if (!state.source) {
    statusEl.textContent = t("inpaint.hint.no_source", "Сначала загрузите изображение");
    return;
  }
  if (!layer) {
    statusEl.textContent = t("inpaint.hint.select_mask", "Выберите слой маски");
    return;
  }
  const coverage = Math.round(layer.surface.coverage() * 100);
  statusEl.textContent = `${t("inpaint.status.coverage", "Покрытие маски")}: ${coverage}%`;
}

// ---------- pointer painting ----------
function updateBrushCursor(app) {
  const cursor = document.getElementById("inpaint-brush-cursor");
  if (!cursor) return;
  const state = getInpaintState(app);
  if (!state) return;
  const layer = activeMaskLayer(state);
  if (!layer || !layer.viewCanvas) {
    cursor.style.display = "none";
    return;
  }
  const rect = layer.viewCanvas.getBoundingClientRect();
  if (!rect.width) return;
  const scale = rect.width / (layer.viewCanvas.width || 1);
  const diameter = Math.max(4, state.brushSize * scale);
  cursor.style.width = `${diameter}px`;
  cursor.style.height = `${diameter}px`;

  const inner = cursor.querySelector(".inpaint-brush-cursor-inner") || document.getElementById("inpaint-brush-cursor-inner");
  const softness = Number(state.softness) || 0;
  if (inner && softness > 0) {
    const innerRatio = Math.max(0.05, 1 - softness);
    const innerDiam = Math.max(2, diameter * innerRatio);
    inner.style.display = "block";
    inner.style.width = `${innerDiam}px`;
    inner.style.height = `${innerDiam}px`;
    cursor.style.borderStyle = "dashed";
  } else if (inner) {
    inner.style.display = "none";
    cursor.style.borderStyle = "solid";
  }
}

export function initInpaintMaskEvents(appRef = null) {
  const app = appRef || window.App;
  if (!app) return;
  const state = getInpaintState(app);
  const stack = maskStack(app);
  if (!stack) return;
  if (stack.dataset.eventsBound === "1") return;
  stack.dataset.eventsBound = "1";

  let painting = false;
  let strokeSnapshot = null;
  let strokeMaskData = null;

  const layerFromEvent = (event) => {
    const el = event.target.closest?.(".inpaint-mask-canvas");
    if (!el) return null;
    const id = el.dataset.layerId;
    return id ? layerById(state, id) : null;
  };

  stack.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || window.spacePressed) return;
    const layer = layerFromEvent(event);
    if (!layer || layer.type !== "mask") return;
    event.preventDefault();
    event.stopPropagation();
    const point = pointerToCanvasPoint(layer.viewCanvas, event);
    if (!point) return;
    painting = true;
    strokeSnapshot = captureSnapshot(state);
    strokeMaskData = layer.surface.toDataURL();
    layer.surface.mode = state.mode;
    layer.surface.beginStroke(point.x, point.y);
    try { layer.viewCanvas.setPointerCapture(event.pointerId); } catch (_) {}
  });

  stack.addEventListener("pointermove", (event) => {
    const layer = layerFromEvent(event);
    if (layer) {
      const point = pointerToCanvasPoint(layer.viewCanvas, event);
      if (point) {
        layer.surface.continueStroke(point.x, point.y);
      }
      updateBrushCursor(app);
      const cursor = document.getElementById("inpaint-brush-cursor");
      if (cursor) {
        cursor.style.display = "block";
        cursor.style.left = `${event.clientX}px`;
        cursor.style.top = `${event.clientY}px`;
      }
    }
  });

  const endStroke = (event) => {
    if (!painting) return;
    painting = false;
    const layer = activeMaskLayer(state);
    if (layer) layer.surface.endStroke();
    if (layer) layer.revision = (layer.revision || 0) + 1;
    if (layer && strokeMaskData !== layer.surface.toDataURL()) {
      pushHistory(state, strokeSnapshot);
    }
    strokeSnapshot = null;
    strokeMaskData = null;
    updateInpaintStatus(app);
  };
  stack.addEventListener("pointerup", endStroke);
  stack.addEventListener("pointercancel", endStroke);
  stack.addEventListener("pointerleave", () => {
    const cursor = document.getElementById("inpaint-brush-cursor");
    if (cursor) cursor.style.display = "none";
  });
}

// ---------- generation ----------
export async function generateInpaint(appRef = null) {
  const app = appRef || window.App;
  const state = getInpaintState(app);
  if (!app || !state) return;

  if (!state.source || !state.source.path) {
    toast("warning", t("inpaint.hint.no_source", "Сначала загрузите изображение"));
    return;
  }

  const layer = activeMaskLayer(state);
  if (!layer || layer.type !== "mask" || !layer.visible) {
    toast("warning", t("inpaint.hint.select_mask", "Выберите видимый слой маски"));
    return;
  }
  if (layer.surface.isEmpty()) {
    toast("warning", t("inpaint.hint.empty_mask", "Маска пуста — нарисуйте область для изменения"));
    return;
  }

  const prompt = document.getElementById("inpaint-prompt")?.value?.trim() || "";
  const negativePrompt = document.getElementById("inpaint-negative-prompt")?.value?.trim() || "";
  if (!prompt) {
    toast("warning", t("inpaint.hint.no_prompt", "Введите промпт"));
    return;
  }

  const modelId = document.getElementById("inpaint-model-select")?.value || app.state.selectedModel;
  const model = (app.state.models || []).find((m) => m.id === modelId);
  if (!model) {
    toast("warning", t("inpaint.hint.no_model", "Выберите модель"));
    return;
  }
  if (!model.supports_mask) {
    const compatible = (app.state.models || []).filter((m) => m.supports_mask);
    const names = compatible.map((m) => m.name).join(", ") || t("inpaint.hint.no_compatible", "нет в наличии");
    toast("error", t("inpaint.hint.model_unsupported", `Модель «${model.name}» не поддерживает маску. Поддерживают: ${names}`, { name: model.name, list: names }));
    return;
  }

  const steps = parseInt(document.getElementById("inpaint-steps")?.value || "4", 10);
  const cfg = parseFloat(document.getElementById("inpaint-cfg")?.value || "1.0");
  const seedRaw = document.getElementById("inpaint-seed")?.value?.trim();
  const seed = seedRaw === "" || seedRaw === "random" ? null : parseInt(seedRaw, 10);

  // Generation area (crop mode).
  const crop = getInpaintCropState(app) || { mode: "auto", padding: 64, minSize: 256, box: null };
  if (crop.mode === "manual") {
    const problem = getInpaintCropProblem(app);
    if (problem) {
      toast("warning", problem);
      return;
    }
  }
  const cropBox = crop.mode === "manual" && crop.box
    ? crop.box.map((value) => Math.round(value))
    : null;

  const maskDataUrl = layer.surface.toDataURL();
  const source = state.source;
  state.jobContext = {
    sourceToken: source.token,
    sourcePath: source.path,
    maskLayerId: layer.id,
    maskDataUrl,
  };

  const denoiseVal = parseFloat(document.getElementById("inpaint-strength")?.value || "1.0");
  const strength = Number.isFinite(denoiseVal) ? Math.max(0.05, Math.min(1.0, denoiseVal)) : 1.0;

  const params = {
    model_id: modelId,
    prompt,
    negative_prompt: negativePrompt,
    width: state.source.width,
    height: state.source.height,
    steps: Number.isFinite(steps) ? steps : 4,
    cfg: Number.isFinite(cfg) ? cfg : 1.0,
    seed: Number.isFinite(seed) ? seed : null,
    sampler: document.getElementById("inpaint-sampler")?.value || "euler",
    init_image_path: state.source.path,
    init_image_strength: strength,
    mask: maskDataUrl,
    mask_strength: strength,
    mask_grow: 6,
    mask_blur: 0.0,
    mask_crop_mode: crop.mode || "auto",
    mask_crop_padding: Math.round(Number(crop.padding) || 0),
    mask_crop_min_size: Math.round(Number(crop.minSize) || 256),
    mask_crop_box: cropBox,
    loras: (Array.isArray(app.state.loras) ? app.state.loras : []).map(l => ({
      name: l.name || l.file,
      strength: Number.isFinite(Number(l.strength)) ? Number(l.strength) : 1.0,
    })),
  };

  state.generating = true;
  const btn = document.getElementById("inpaint-generate-btn");
  if (btn) btn.disabled = true;

  try {
    const result = await apiPost("/api/generate", params);
    state.jobId = result.job_id;
    state.jobContext.jobId = result.job_id;
    startInpaintPolling(app, result.job_id);
  } catch (e) {
    state.generating = false;
    state.jobContext = null;
    if (btn) btn.disabled = false;
    const data = e?.data || {};
    const compatible = Array.isArray(data.compatible_models) ? data.compatible_models : [];
    if (compatible.length) {
      toast("error", `${e.message} ${t("inpaint.hint.switch_model", "Переключитесь на:")} ${compatible.map((m) => m.name).join(", ")}`);
    } else {      toast("error", e.message || "Ошибка генерации");
    }
  }
}

function startInpaintPolling(app, jobId) {
  const state = getInpaintState(app);
  if (!state) return;
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    try {
      const status = await apiGet(`/api/jobs/${jobId}`);
      const progressEl = document.getElementById("inpaint-progress");
      if (progressEl) {
        progressEl.style.display = status.status === "complete" ? "none" : "flex";
        progressEl.textContent = `${Math.round((status.progress || 0) * 100)}%`;
      }
      if (status.status === "complete") {
        stopInpaintPolling(app);
        await applyInpaintResult(app, status, layer => layer);
      } else if (status.status === "error") {
        stopInpaintPolling(app);
        state.jobContext = null;
        toast("error", status.error || status.message || t("inpaint.hint.generation_error", "Ошибка генерации"));
      } else if (status.status === "cancelled") {
        stopInpaintPolling(app);
        state.jobContext = null;
        toast("info", t("inpaint.hint.cancelled", "Генерация отменена"));
      }
    } catch (e) {
      stopInpaintPolling(app);
      state.jobContext = null;
      toast("error", e.message || "Ошибка опроса задания");
    }
  }, 400);
}

function stopInpaintPolling(app) {
  const state = getInpaintState(app);
  if (!state) return;
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  state.generating = false;
  const btn = document.getElementById("inpaint-generate-btn");
  if (btn) btn.disabled = false;
  const progressEl = document.getElementById("inpaint-progress");
  if (progressEl) progressEl.style.display = "none";
}

async function applyInpaintResult(app, status, _unused) {
  const state = getInpaintState(app);
  if (!state) return;
  const context = state.jobContext;
  if (
    !context
    || state.jobId !== context.jobId
    || state.source?.token !== context.sourceToken
    || state.sourceLoadSeq !== context.sourceToken
  ) {
    state.jobContext = null;
    toast("warning", t("inpaint.hint.source_changed", "Источник изменился; результат не добавлен"));
    return;
  }

  const raw = status.raw_image_path || null;
  const composite = status.image_path || "";
  const fragmentUrl = raw ? `/outputs/${raw}` : `/outputs/${String(composite).split(/[\\/]/).pop()}`;

  state.applyingResult = true;
  try {
  const resultImage = new Image();
  await new Promise((resolve, reject) => {
    resultImage.onload = resolve;
    resultImage.onerror = () => reject(new Error("Не удалось загрузить результат генерации"));
    resultImage.src = fragmentUrl;
  });

  const maskSnapshot = context.maskDataUrl;
  const maskImage = new Image();
  await new Promise((resolve, reject) => {
    maskImage.onload = resolve;
    maskImage.onerror = () => reject(new Error("Не удалось загрузить снимок маски"));
    maskImage.src = maskSnapshot;
  });
  if (
    state.source?.token !== context.sourceToken
    || state.sourceLoadSeq !== context.sourceToken
  ) {
    state.jobContext = null;
    toast("warning", t("inpaint.hint.source_changed", "Источник изменился; результат не добавлен"));
    return;
  }
  pushHistory(state);

  const layer = {
    id: nextLayerId("result"),
    type: "result",
    name: `${t("inpaint.layer.result", "Результат")} ${state.layers.filter((l) => l.type === "result").length + 1}`,
    visible: true,
    opacity: 1,
    image: resultImage,
    maskImage,
    maskImageSrc: maskSnapshot,
    url: fragmentUrl,
  };

  state.layers.push(layer);
  state.activeLayerId = layer.id;
  state.jobContext = null;

  refreshMaskLayerViews(app);
  renderInpaintComposite(app);
  renderInpaintLayersUI(app);
  updateInpaintStatus(app);
  toast("success", t("inpaint.hint.done", "Слой результата создан"));
  } finally {
    state.applyingResult = false;
  }
}

// ---------- model list ----------
export function renderInpaintModels(appRef = null) {
  const app = appRef || window.App;
  const select = document.getElementById("inpaint-model-select");
  if (!app || !select) return;

  const models = app.state.models || [];
  const previous = select.value;
  select.innerHTML = "";
  for (const model of models) {
    if (model.hidden_from_host) continue;
    const opt = document.createElement("option");
    opt.value = model.id;
    const mark = model.installed ? "✓" : "⬇";
    const maskMark = model.supports_mask ? "🎭" : "⚠";
    opt.textContent = `${mark}${maskMark} ${model.name}`;
    select.appendChild(opt);
  }
  if (previous && models.some((m) => m.id === previous)) {
    select.value = previous;
  } else if (app.state.selectedModel) {
    select.value = app.state.selectedModel;
  }
  updateInpaintModelWarning(app);
}

export function updateInpaintModelWarning(appRef = null) {
  const app = appRef || window.App;
  const select = document.getElementById("inpaint-model-select");
  const warnEl = document.getElementById("inpaint-model-warning");
  if (!app || !select || !warnEl) return;
  const model = (app.state.models || []).find((m) => m.id === select.value);
  if (!model) {
    warnEl.style.display = "none";
    return;
  }
  if (model.supports_mask) {
    warnEl.style.display = "none";
    return;
  }
  const compatible = (app.state.models || []).filter((m) => m.supports_mask);
  warnEl.style.display = "flex";
  warnEl.innerHTML = `⚠ ${t("inpaint.hint.model_unsupported_short", "Модель не поддерживает маску")} — ${compatible.map((m) => escapeHtml(m.name)).join(", ") || "—"}`;
}

// ---------- source pickers ----------
export function useCanvasImageInInpaint(appRef = null) {
  const app = appRef || window.App;
  if (!app) return;
  const url = app.state.currentPreviewUrl;
  const path = app.state.currentPreviewFilename;
  if (!url || !path) {
    toast("warning", t("inpaint.hint.no_source", "Сначала загрузите изображение"));
    return;
  }
  loadInpaintSource(url, path, path, app);
}

export async function uploadInpaintFile(file, appRef = null) {
  const app = appRef || window.App;
  if (!app || !file) return;
  const allowedTypes = ["image/png", "image/jpeg", "image/webp"];
  if (
    !allowedTypes.includes(file.type)
    && !/\.(png|jpe?g|webp)$/i.test(file.name || "")
  ) {
    toast("error", "Поддерживаются только PNG, JPEG и WebP");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    toast("error", "Размер файла не должен превышать 25 МБ");
    return;
  }
  try {
    const data = await apiUpload(file);
    const loaded = await loadInpaintSource(data.url, data.path, file.name || data.path, app);
    if (loaded) toast("success", t("inpaint.hint.loaded", "Изображение загружено"));
  } catch (e) {
    toast("error", e.message || "Ошибка загрузки");
  }
}

// ---------- init ----------
export function initInpaintStudio(appRef = null) {
  const app = appRef || window.App;
  if (!app) return;
  const state = getInpaintState(app);

  initInpaintMaskEvents(app);
  initInpaintCropEvents(app);
  renderInpaintModels(app);
  renderInpaintLayersUI(app);
  renderInpaintCropUI(app);
  updateInpaintCropStatus(app);
  updateInpaintStatus(app);

  const brushBtn = document.getElementById("inpaint-tool-brush");
  if (brushBtn && !brushBtn.dataset.bound) {
    brushBtn.dataset.bound = "1";
    brushBtn.addEventListener("click", () => setInpaintTool("brush", app));
  }
  const eraserBtn = document.getElementById("inpaint-tool-eraser");
  if (eraserBtn && !eraserBtn.dataset.bound) {
    eraserBtn.dataset.bound = "1";
    eraserBtn.addEventListener("click", () => setInpaintTool("eraser", app));
  }

  const fileInput = document.getElementById("inpaint-file-input");
  if (fileInput && !fileInput.dataset.bound) {
    fileInput.dataset.bound = "1";
    fileInput.addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) uploadInpaintFile(file, app);
      event.target.value = "";
    });
  }

  const canvasArea = document.getElementById("inpaint-canvas-area");
  if (canvasArea && !canvasArea.dataset.dropBound) {
    canvasArea.dataset.dropBound = "1";
    let dragDepth = 0;
    canvasArea.addEventListener("dragenter", (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      dragDepth += 1;
      canvasArea.classList.add("drag-over");
    });
    canvasArea.addEventListener("dragover", (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    });
    canvasArea.addEventListener("dragleave", (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) canvasArea.classList.remove("drag-over");
    });
    canvasArea.addEventListener("drop", (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
      dragDepth = 0;
      canvasArea.classList.remove("drag-over");
      const file = Array.from(event.dataTransfer.files || []).find((item) =>
        ["image/png", "image/jpeg", "image/webp"].includes(item.type)
        || /\.(png|jpe?g|webp)$/i.test(item.name || "")
      );
      if (file) uploadInpaintFile(file, app);
      else toast("error", "Перетащите изображение PNG, JPEG или WebP");
    });
  }


// ---------- LocalStorage persistence & crisp-zoom (ported from SnowE22) ----------
function setupInpaintPersistence(app) {
  try {
    const pPrompt = localStorage.getItem("lazysd_inpaint_prompt");
    if (pPrompt !== null) {
      const el = document.getElementById("inpaint-prompt");
      if (el) el.value = pPrompt;
    }
    const pNeg = localStorage.getItem("lazysd_inpaint_negative_prompt");
    if (pNeg !== null) {
      const el = document.getElementById("inpaint-negative-prompt");
      if (el) el.value = pNeg;
    }
    const pSteps = localStorage.getItem("lazysd_inpaint_steps");
    if (pSteps !== null) {
      const el = document.getElementById("inpaint-steps");
      if (el) el.value = pSteps;
    }
    const pCfg = localStorage.getItem("lazysd_inpaint_cfg");
    if (pCfg !== null) {
      const el = document.getElementById("inpaint-cfg");
      if (el) el.value = pCfg;
    }
    const pStrength = localStorage.getItem("lazysd_inpaint_strength");
    if (pStrength !== null) {
      const el = document.getElementById("inpaint-strength");
      if (el) el.value = pStrength;
    }
    const pSampler = localStorage.getItem("lazysd_inpaint_sampler");
    if (pSampler !== null) {
      const el = document.getElementById("inpaint-sampler");
      if (el) el.value = pSampler;
    }

    // Attach listeners
    const fields = [
      ["inpaint-prompt", "lazysd_inpaint_prompt"],
      ["inpaint-negative-prompt", "lazysd_inpaint_negative_prompt"],
      ["inpaint-steps", "lazysd_inpaint_steps"],
      ["inpaint-cfg", "lazysd_inpaint_cfg"],
      ["inpaint-strength", "lazysd_inpaint_strength"],
      ["inpaint-sampler", "lazysd_inpaint_sampler"],
    ];
    for (const [id, key] of fields) {
      const el = document.getElementById(id);
      if (el && !el.dataset.persistBound) {
        el.dataset.persistBound = "1";
        el.addEventListener("input", () => {
          try { localStorage.setItem(key, el.value); } catch (_) {}
        });
        el.addEventListener("change", () => {
          try { localStorage.setItem(key, el.value); } catch (_) {}
        });
      }
    }
  } catch (err) {
    console.warn("[inpaint] localStorage persistence unavailable", err);
  }
}

function setupInpaintCrispZoom() {
  if (window._inpaintCrispBound) return;
  window._inpaintCrispBound = true;
  window.addEventListener("viewport-transform", (e) => {
    if (e.detail?.mode === "inpaint") {
      const zoom = Number(e.detail.zoom) || 1;
      const isCrisp = zoom >= 1.25;
      const comp = document.getElementById("inpaint-composite");
      const overlay = document.getElementById("inpaint-crop-overlay");
      if (comp) comp.classList.toggle("crisp-zoom", isCrisp);
      if (overlay) overlay.classList.toggle("crisp-zoom", isCrisp);
      document.querySelectorAll(".inpaint-layer-canvas").forEach((c) => {
        c.classList.toggle("crisp-zoom", isCrisp);
      });
    }
  });
}

  if (!state._initialized) {
    state._initialized = true;
    setInpaintTool("brush", app);
    setInpaintBrushSize(state.brushSize, app);
    setInpaintSoftness(state.softness, app);
  }
  setupInpaintPersistence(app);
  setupInpaintCrispZoom();
}


export function applyInpaintPreset(key, appRef = null) {
  const app = appRef || window.App;
  const presets = {
    clean_text: {
      prompt: "clean empty dialog speech bubble, seamless white fill, clean background, monochrome manga",
      negative: "text, letters, words, japanese, kanji, sound effects, sfx, speech, artifacts, watermark",
    },
    screentone: {
      prompt: "uniform monochrome manga screentone texture, fine halftone dot pattern, seamless shading",
      negative: "text, letters, sfx, distorted pattern, moire, noisy lines, blur",
    },
    lineart: {
      prompt: "clean sharp manga anime lineart, black ink line drawing, detailed, crisp monochrome",
      negative: "blurry, smudged, low contrast, grayscale noise, text, sfx",
    },
    clean_bg: {
      prompt: "clean detailed manga background scenery, seamless perspective, monochrome screen tone",
      negative: "speech bubble, dialogue, text, character cutouts, watermark, signature",
    },
  };
  const p = presets[key];
  if (!p) return;
  const promptEl = document.getElementById("inpaint-prompt");
  const negEl = document.getElementById("inpaint-negative-prompt");
  if (promptEl) {
    promptEl.value = p.prompt;
    try { localStorage.setItem("lazysd_inpaint_prompt", p.prompt); } catch (_) {}
  }
  if (negEl) {
    negEl.value = p.negative;
    try { localStorage.setItem("lazysd_inpaint_negative_prompt", p.negative); } catch (_) {}
  }
  if (app?.toast) app.toast("info", `Пресет: ${p.prompt.slice(0, 32)}...`);
}
