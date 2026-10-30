/* ============================================================
   QuantFunc WebUI — Generate quick mask editing
   A single mask layer drawn on top of the image shown on the canvas.
   The mask is bound to IMAGE coordinates, therefore zoom / pan / window
   resizing never shifts it. Without a mask the Generate behaviour is
   exactly the same as before.
   ============================================================ */

import { apiUpload, toast, escapeHtml } from "../api.js";
import {
  createMaskSurface,
  pointerToCanvasPoint,
  syncOverlayGeometry,
  MASK_DEFAULT_BRUSH_SIZE,
  MASK_DEFAULT_SOFTNESS,
} from "./mask-core.js";

function t(key, fallback) {
  const fn = typeof window !== "undefined" ? window.t : null;
  if (typeof fn === "function") {
    const value = fn(key);
    if (value && value !== key) return value;
  }
  return fallback || key;
}

export function getGenMaskState(appRef = null) {
  const app = appRef || window.App;
  if (!app) return null;
  if (!app.state.genMask) {
    app.state.genMask = {
      enabled: false,
      surface: null,
      source: null,          // { url, path, name, width, height }
      brushSize: MASK_DEFAULT_BRUSH_SIZE,
      softness: MASK_DEFAULT_SOFTNESS,
      mode: "brush",
      bound: false,
      sourceLoadSeq: 0,
    };
  }
  return app.state.genMask;
}

function viewCanvas() {
  return document.getElementById("gen-mask-canvas");
}

function previewImage() {
  return document.getElementById("preview-img");
}

function panel() {
  return document.getElementById("gen-mask-panel");
}

// ---------- geometry ----------
export function syncGenMaskGeometry(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  const canvas = viewCanvas();
  const img = previewImage();
  if (!state || !canvas || !img) return;
  syncOverlayGeometry(canvas, img);
}

// ---------- UI ----------
export function updateGenMaskPanel(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;

  const panelEl = panel();
  const toggleBtn = document.getElementById("gen-mask-toggle");
  const hintEl = document.getElementById("gen-mask-hint");
  const sourceEl = document.getElementById("gen-mask-source-name");
  const statusEl = document.getElementById("gen-mask-status");
  const canvas = viewCanvas();

  const hasSource = Boolean(state.source);
  const canDraw = state.enabled && hasSource;

  if (toggleBtn) toggleBtn.classList.toggle("active", state.enabled);
  const fabBtn = document.getElementById("gen-mask-fab");
  if (fabBtn) fabBtn.classList.toggle("active", state.enabled);
  const inspectorBtn = document.getElementById("inspector-gen-mask-btn");
  if (inspectorBtn) inspectorBtn.classList.toggle("active", state.enabled);
  const inspectorHint = document.getElementById("inspector-gen-mask-hint");
  if (inspectorHint) inspectorHint.style.display = hasSource ? "none" : "block";
  if (panelEl) panelEl.style.display = state.enabled ? "flex" : "none";
  if (canvas) {
    canvas.style.display = canDraw ? "block" : "none";
    canvas.style.pointerEvents = canDraw ? "auto" : "none";
  }

  if (hintEl) hintEl.style.display = hasSource ? "none" : "flex";
  if (sourceEl) sourceEl.textContent = state.source?.name || t("generate.mask.no_source", "нет изображения");

  if (statusEl) {
    if (!hasSource) {
      statusEl.textContent = t("generate.mask.hint_load", "Загрузите изображение, чтобы рисовать маску");
    } else if (!state.surface || state.surface.isEmpty()) {
      statusEl.textContent = t("generate.mask.hint_empty", "Маска пуста — нарисуйте область для изменения");
    } else {
      statusEl.textContent = `${t("generate.mask.coverage", "Покрытие")}: ${Math.round(state.surface.coverage() * 100)}%`;
    }
  }

  const undoBtn = document.getElementById("gen-mask-undo");
  const redoBtn = document.getElementById("gen-mask-redo");
  if (undoBtn) undoBtn.disabled = !state.surface || !state.surface.canUndo;
  if (redoBtn) redoBtn.disabled = !state.surface || !state.surface.canRedo;
}

function ensureSurface(app) {
  const state = getGenMaskState(app);
  const canvas = viewCanvas();
  if (!state || !canvas) return null;
  if (!state.surface) {
    const dataCanvas = document.createElement("canvas");
    dataCanvas.width = 1024;
    dataCanvas.height = 1024;
    state.surface = createMaskSurface(dataCanvas, { visualCanvas: canvas });
    state.surface.brushSize = state.brushSize;
    state.surface.softness = state.softness;
    state.surface.mode = state.mode;
  }
  return state.surface;
}

// ---------- source ----------
export function setGenMaskSource(appRef, source) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state || !source) return;
  const sourceLoadSeq = (state.sourceLoadSeq || 0) + 1;
  state.sourceLoadSeq = sourceLoadSeq;

  const img = previewImage();
  const placeholder = document.getElementById("preview-placeholder");
  const wrap = document.getElementById("preview-image-wrap");

  if (placeholder) placeholder.style.display = "none";
  if (wrap) wrap.style.display = "flex";

  const applyLoadedSource = () => {
    if (state.sourceLoadSeq !== sourceLoadSeq) return;
    const changed = !state.source
      || state.source.url !== source.url
      || state.source.path !== source.path;
    state.source = {
      url: source.url,
      path: source.path,
      name: source.name || source.url.split("/").pop(),
      width: img?.naturalWidth || 1024,
      height: img?.naturalHeight || 1024,
    };
    const surface = ensureSurface(app);
    if (changed && surface) {
      surface.reset(state.source.width, state.source.height);
      syncGenMaskGeometry(app);
      updateGenMaskPanel(app);
    } else {
      resizeMaskToSource(app);
    }
  };

  if (img) {
    img.src = source.url;
    const expectedUrl = new URL(source.url, document.baseURI).href;
    if (img.complete && img.naturalWidth && img.currentSrc === expectedUrl) {
      applyLoadedSource();
    } else {
      img.addEventListener("load", applyLoadedSource, { once: true });
    }
  } else {
    applyLoadedSource();
  }

  updateGenMaskPanel(app);
  syncGenMaskGeometry(app);
}

export function clearGenMaskSource(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;
  state.sourceLoadSeq = (state.sourceLoadSeq || 0) + 1;
  state.source = null;
  state.surface?.reset(1024, 1024);
  const img = previewImage();
  const placeholder = document.getElementById("preview-placeholder");
  const wrap = document.getElementById("preview-image-wrap");
  if (img) img.removeAttribute("src");
  if (wrap) wrap.style.display = "none";
  if (placeholder) placeholder.style.display = "flex";
  updateGenMaskPanel(app);
  syncGenMaskGeometry(app);
}

function resizeMaskToSource(app) {
  const state = getGenMaskState(app);
  const surface = ensureSurface(app);
  const img = previewImage();
  if (!state || !surface || !img) return;
  const width = img.naturalWidth || state.source?.width || 1024;
  const height = img.naturalHeight || state.source?.height || 1024;
  surface.resize(width, height);
  if (state.source) {
    state.source.width = width;
    state.source.height = height;
  }
  syncGenMaskGeometry(app);
  updateGenMaskPanel(app);
}

// ---------- toggle ----------
export function toggleGenMask(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;

  if (state.enabled) {
    state.enabled = false;
    const img = previewImage();
    const placeholder = document.getElementById("preview-placeholder");
    const wrap = document.getElementById("preview-image-wrap");
    if (app.state.currentPreviewUrl) {
      if (img) img.src = app.state.currentPreviewUrl;
      if (wrap) wrap.style.display = "flex";
      if (placeholder) placeholder.style.display = "none";
    } else {
      if (img) img.removeAttribute("src");
      if (wrap) wrap.style.display = "none";
      if (placeholder) placeholder.style.display = "flex";
    }
    updateGenMaskPanel(app);
    return;
  }

  const reference = app.state.img2imgRef;
  const currentUrl = reference?.url || app.state.currentPreviewUrl;
  const currentName = reference?.path || app.state.currentPreviewFilename;
  if (!currentUrl || !currentName) {
    // No image yet: open the panel with the hint so the user can pick one.
    state.enabled = true;
    updateGenMaskPanel(app);
    toast("warning", t("generate.mask.hint_load", "Загрузите изображение, чтобы рисовать маску"));
    return;
  }

  setGenMaskSource(app, {
    url: currentUrl,
    path: currentName,
    name: reference?.filename || currentName,
  });
  state.enabled = true;
  updateGenMaskPanel(app);
  syncGenMaskGeometry(app);
}

/** Use an image as the mask source and switch the mask editor on. */
export function enableGenMaskWithSource(appRef, source) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;
  setGenMaskSource(app, source);
  if (source?.path && source?.url) {
    app.state.img2imgRef = {
      path: source.path,
      url: source.url,
      filename: source.name || source.path,
    };
    app.updateNewDock?.(app);
  }
  state.enabled = true;
  resizeMaskToSource(app);
  updateGenMaskPanel(app);
  syncGenMaskGeometry(app);
}

// ---------- tools ----------
export function setGenMaskTool(mode, appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;
  state.mode = mode === "eraser" ? "eraser" : "brush";
  if (state.surface) state.surface.mode = state.mode;
  document.getElementById("gen-mask-tool-brush")?.classList.toggle("active", state.mode === "brush");
  document.getElementById("gen-mask-tool-eraser")?.classList.toggle("active", state.mode === "eraser");
  updateGenMaskPanel(app);
}

export function setGenMaskBrushSize(value, appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;
  state.brushSize = Math.max(1, Math.min(4096, Number(value) || MASK_DEFAULT_BRUSH_SIZE));
  if (state.surface) state.surface.brushSize = state.brushSize;
  const label = document.getElementById("gen-mask-size-val");
  if (label) label.textContent = String(Math.round(state.brushSize));
}

export function setGenMaskSoftness(value, appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return;
  state.softness = Math.max(0, Math.min(1, Number(value)));
  if (state.surface) state.surface.softness = state.softness;
  const label = document.getElementById("gen-mask-softness-val");
  if (label) label.textContent = `${Math.round(state.softness * 100)}%`;
}

export function undoGenMask(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state || !state.surface) return;
  state.surface.undo().then(() => updateGenMaskPanel(app));
}

export function redoGenMask(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state || !state.surface) return;
  state.surface.redo().then(() => updateGenMaskPanel(app));
}

export function clearGenMask(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state || !state.surface) return;
  state.surface.clear();
  updateGenMaskPanel(app);
}

export function invertGenMask(appRef = null) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state || !state.surface) return;
  state.surface.invert();
  updateGenMaskPanel(app);
}

// ---------- source pickers ----------
export async function uploadGenMaskFile(file, appRef = null) {
  const app = appRef || window.App;
  if (!app || !file) return;
  try {
    const data = await apiUpload(file);
    setGenMaskSource(app, {
      url: data.url,
      path: data.path,
      name: file.name || data.path,
    });
    toast("success", t("generate.mask.loaded", "Изображение загружено"));
  } catch (e) {
    toast("error", e.message || "Ошибка загрузки");
  }
}

export function useCanvasImageForMask(appRef = null) {
  const app = appRef || window.App;
  if (!app) return;
  if (!app.state.currentPreviewUrl || !app.state.currentPreviewFilename) {
    toast("warning", t("generate.mask.hint_load", "Загрузите изображение, чтобы рисовать маску"));
    return;
  }
  setGenMaskSource(app, {
    url: app.state.currentPreviewUrl,
    path: app.state.currentPreviewFilename,
    name: app.state.currentPreviewFilename,
  });
}

// ---------- generate integration ----------
/**
 * Add the mask to the generate params.
 * Returns "none" | "ok" | "empty" so the caller can react.
 */
export function applyGenMaskParams(appRef, params) {
  const app = appRef || window.App;
  const state = getGenMaskState(app);
  if (!app || !state) return "none";
  if (!state.enabled || !state.surface || !state.source) return "none";
  if (state.surface.isEmpty()) return "empty";

  params.init_image_path = state.source.path;
  params.init_image_strength = 1.0;
  params.ref_mode = "img2img";
  // Generate in the source frame so that the mask needs no resampling.
  const sourceW = Math.round((state.source.width || 1024) / 8) * 8;
  const sourceH = Math.round((state.source.height || 1024) / 8) * 8;
  params.width = Math.max(256, Math.min(4096, sourceW));
  params.height = Math.max(256, Math.min(4096, sourceH));
  params.mask = state.surface.toDataURL();
  params.mask_strength = 1.0;
  params.mask_grow = 6;
  params.mask_blur = 0.0;
  return "ok";
}

/** List of models able to work with a mask (used for the pre-flight check). */
export function getMaskCapableModels(appRef = null) {
  const app = appRef || window.App;
  if (!app) return [];
  return (app.state.models || []).filter((m) => m.supports_mask);
}

// ---------- init ----------
export function initGenMaskUI(appRef = null) {
  const app = appRef || window.App;
  if (!app) return;
  const state = getGenMaskState(app);
  if (state.bound) {
    updateGenMaskPanel(app);
    return;
  }
  state.bound = true;

  const canvas = viewCanvas();
  if (canvas) {
    let painting = false;

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const st = getGenMaskState(app);
      if (!st?.enabled) return;
      const surface = ensureSurface(app);
      if (!surface) return;
      const point = pointerToCanvasPoint(canvas, event);
      if (!point) return;
      event.preventDefault();
      event.stopPropagation();
      painting = true;
      surface.mode = st.mode;
      surface.beginStroke(point.x, point.y);
      try { canvas.setPointerCapture(event.pointerId); } catch (_) {}
      updateGenMaskPanel(app);
    });

    canvas.addEventListener("pointermove", (event) => {
      const st = getGenMaskState(app);
      if (!st?.enabled) return;
      const point = pointerToCanvasPoint(canvas, event);
      if (painting && point && st.surface) {
        st.surface.continueStroke(point.x, point.y);
      }
    });

    const end = (event) => {
      if (!painting) return;
      painting = false;
      const st = getGenMaskState(app);
      if (st?.surface) st.surface.endStroke();
      updateGenMaskPanel(app);
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("pointerleave", end);
  }

  // Keep the overlay glued to the image when the layout changes
  const img = previewImage();
  if (img && typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(() => syncGenMaskGeometry(app));
    observer.observe(img);
    state.maskObserver = observer;
  }
  window.addEventListener("resize", () => syncGenMaskGeometry(app));

  const brushBtn = document.getElementById("gen-mask-tool-brush");
  brushBtn?.addEventListener("click", () => setGenMaskTool("brush", app));
  const eraserBtn = document.getElementById("gen-mask-tool-eraser");
  eraserBtn?.addEventListener("click", () => setGenMaskTool("eraser", app));

  const fileInput = document.getElementById("gen-mask-file-input");
  fileInput?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) uploadGenMaskFile(file, app);
    event.target.value = "";
  });

  updateGenMaskPanel(app);
}
