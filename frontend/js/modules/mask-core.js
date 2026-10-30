/* ============================================================
   QuantFunc WebUI — Shared Mask Engine
   Brush / eraser / softness / undo / redo / clear / invert.

   Mask semantics (identical to the backend, backend/masking.py):
       white  = region to regenerate
       black  = region to preserve
       grey   = soft transition

   The mask lives in IMAGE coordinates: the canvas resolution equals the
   natural resolution of the source image, therefore zooming, panning or
   resizing the window never shifts the mask. The pointer position is
   converted with `canvas.width / rect.width`, so the brush always stays
   exactly under the cursor.
   ============================================================ */

export const MASK_DEFAULT_BRUSH_SIZE = 64;      // in image pixels
export const MASK_DEFAULT_SOFTNESS = 0.5;       // 0 = hard edge, 1 = max falloff
export const MASK_MAX_HISTORY = 24;
export const MASK_OVERLAY_COLOR = "rgba(239, 68, 68, 0.45)";
export const MASK_ERASE_OVERLAY_COLOR = "rgba(56, 189, 248, 0.45)";

function clamp(value, min, max) {
  const v = Number(value);
  if (!Number.isFinite(v)) return min;
  return Math.min(max, Math.max(min, v));
}

/** Convert a pointer event into canvas (image) coordinates. */
export function pointerToCanvasPoint(canvas, event) {
  if (!canvas || !event) return null;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

/**
 * Align an overlay canvas with the element it decorates (an <img>).
 * Uses layout offsets (offsetLeft/offsetWidth) so viewport zoom/pan
 * transforms do not distort the overlay.
 */
export function syncOverlayGeometry(canvas, targetEl) {
  if (!canvas || !targetEl) return;
  const parent = canvas.offsetParent || canvas.parentElement;
  let left = targetEl.offsetLeft || 0;
  let top = targetEl.offsetTop || 0;

  if (parent && targetEl.offsetParent && targetEl.offsetParent !== parent) {
    const targetRect = targetEl.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const scale = parentRect.width ? (parent.offsetWidth || parentRect.width) / parentRect.width : 1;
    left = (targetRect.left - parentRect.left) * scale;
    top = (targetRect.top - parentRect.top) * scale;
  }

  canvas.style.left = `${left}px`;
  canvas.style.top = `${top}px`;
  canvas.style.width = `${targetEl.offsetWidth || targetEl.clientWidth || 0}px`;
  canvas.style.height = `${targetEl.offsetHeight || targetEl.clientHeight || 0}px`;
}

/**
 * Create a mask surface bound to a canvas.
 *
 * @param {HTMLCanvasElement} dataCanvas  canvas holding the mask (white strokes, transparent bg)
 * @param {object} options                 { visualCanvas, brushSize, softness, mode, color, maxHistory }
 */
export function createMaskSurface(dataCanvas, options = {}) {
  if (!dataCanvas) return null;

  const visualCanvas = options.visualCanvas || null;
  const ctx = dataCanvas.getContext("2d");
  const vctx = visualCanvas ? visualCanvas.getContext("2d") : null;

  const st = {
    brushSize: clamp(options.brushSize ?? MASK_DEFAULT_BRUSH_SIZE, 1, 4096),
    softness: clamp(options.softness ?? MASK_DEFAULT_SOFTNESS, 0, 1),
    mode: options.mode === "eraser" ? "eraser" : "brush",
    color: options.color || MASK_OVERLAY_COLOR,
    eraseColor: options.eraseColor || MASK_ERASE_OVERLAY_COLOR,
    maxHistory: clamp(options.maxHistory ?? MASK_MAX_HISTORY, 1, 100),
  };

  const undoStack = [];
  const redoStack = [];
  let drawing = false;
  let lastPoint = null;
  let dirty = false;
  let visualQueued = false;
  let scratch = null;

  // ---------- visual ----------
  function renderVisual() {
    if (!vctx || !visualCanvas) return;
    vctx.clearRect(0, 0, visualCanvas.width, visualCanvas.height);
    vctx.drawImage(dataCanvas, 0, 0, visualCanvas.width, visualCanvas.height);
    vctx.globalCompositeOperation = "source-in";
    vctx.fillStyle = st.mode === "eraser" ? st.eraseColor : st.color;
    vctx.fillRect(0, 0, visualCanvas.width, visualCanvas.height);
    vctx.globalCompositeOperation = "source-over";
  }

  function scheduleVisual() {
    if (!vctx || visualQueued) return;
    visualQueued = true;
    requestAnimationFrame(() => {
      visualQueued = false;
      renderVisual();
    });
  }

  // ---------- history ----------
  function snapshot() {
    try {
      return dataCanvas.toDataURL("image/png");
    } catch (_) {
      return "";
    }
  }

  function pushHistory() {
    const snap = snapshot();
    if (!snap) return;
    undoStack.push(snap);
    while (undoStack.length > st.maxHistory) undoStack.shift();
    redoStack.length = 0;
  }

  function restoreFromDataURL(url) {
    if (!url) return Promise.resolve(false);
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, dataCanvas.width, dataCanvas.height);
        ctx.drawImage(img, 0, 0, dataCanvas.width, dataCanvas.height);
        dirty = true;
        renderVisual();
        resolve(true);
      };
      img.onerror = () => resolve(false);
      img.src = url;
    });
  }

  function undo() {
    if (!undoStack.length) return Promise.resolve(false);
    redoStack.push(snapshot());
    const prev = undoStack.pop();
    return restoreFromDataURL(prev).then((ok) => {
      if (!ok) {
        // keep the stack consistent even if the blob could not be restored
        undoStack.push(prev);
      }
      return ok;
    });
  }

  function redo() {
    if (!redoStack.length) return Promise.resolve(false);
    undoStack.push(snapshot());
    const next = redoStack.pop();
    return restoreFromDataURL(next).then((ok) => {
      if (!ok) redoStack.push(next);
      return ok;
    });
  }

  // ---------- painting ----------
  function paintDot(x, y, erase) {
    const radius = Math.max(0.5, st.brushSize / 2);
    ctx.save();
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    if (st.softness <= 0.02) {
      ctx.fillStyle = erase ? "rgba(0,0,0,1)" : "rgba(255,255,255,1)";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const inner = Math.max(0, radius * (1 - st.softness));
      const grad = ctx.createRadialGradient(x, y, inner, x, y, radius);
      if (erase) {
        grad.addColorStop(0, "rgba(0,0,0,1)");
        grad.addColorStop(1, "rgba(0,0,0,0)");
      } else {
        grad.addColorStop(0, "rgba(255,255,255,1)");
        grad.addColorStop(1, "rgba(255,255,255,0)");
      }
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function strokeTo(x, y) {
    const erase = st.mode === "eraser";
    if (!lastPoint) {
      paintDot(x, y, erase);
      lastPoint = { x, y };
      return;
    }
    const dx = x - lastPoint.x;
    const dy = y - lastPoint.y;
    const dist = Math.hypot(dx, dy);
    const step = Math.max(1, (st.brushSize / 2) * 0.25);
    if (dist < 0.0001) return;
    const count = Math.max(1, Math.ceil(dist / step));
    for (let i = 1; i <= count; i++) {
      const t = i / count;
      paintDot(lastPoint.x + dx * t, lastPoint.y + dy * t, erase);
    }
    lastPoint = { x, y };
  }

  // ---------- public API ----------
  return {
    canvas: dataCanvas,
    visualCanvas,
    get brushSize() { return st.brushSize; },
    set brushSize(v) { st.brushSize = clamp(v, 1, 4096); },
    get softness() { return st.softness; },
    set softness(v) { st.softness = clamp(v, 0, 1); },
    get mode() { return st.mode; },
    set mode(v) { st.mode = v === "eraser" ? "eraser" : "brush"; scheduleVisual(); },
    get dirty() { return dirty; },
    get canUndo() { return undoStack.length > 0; },
    get canRedo() { return redoStack.length > 0; },
    get isDrawing() { return drawing; },

    /** Resize the mask canvas (image coordinates) and drop the history. */
    resize(width, height) {
      const w = Math.max(1, Math.round(width));
      const h = Math.max(1, Math.round(height));
      if (dataCanvas.width === w && dataCanvas.height === h) return;
      dataCanvas.width = w;
      dataCanvas.height = h;
      if (visualCanvas) {
        visualCanvas.width = w;
        visualCanvas.height = h;
      }
      ctx.clearRect(0, 0, w, h);
      undoStack.length = 0;
      redoStack.length = 0;
      dirty = false;
      renderVisual();
    },

    reset(width, height) {
      const w = Math.max(1, Math.round(width));
      const h = Math.max(1, Math.round(height));
      dataCanvas.width = w;
      dataCanvas.height = h;
      if (visualCanvas) {
        visualCanvas.width = w;
        visualCanvas.height = h;
      }
      ctx.clearRect(0, 0, w, h);
      undoStack.length = 0;
      redoStack.length = 0;
      dirty = false;
      renderVisual();
    },

    /** Remove everything (optionally recording an undo step). */
    clear(record = true) {
      if (record) pushHistory();
      ctx.clearRect(0, 0, dataCanvas.width, dataCanvas.height);
      dirty = true;
      renderVisual();
    },

    /** Invert the mask (white <-> black). */
    invert(record = true) {
      if (record) pushHistory();
      const w = dataCanvas.width;
      const h = dataCanvas.height;
      if (!scratch) {
        scratch = document.createElement("canvas");
      }
      scratch.width = w;
      scratch.height = h;
      const sctx = scratch.getContext("2d");
      sctx.clearRect(0, 0, w, h);
      sctx.fillStyle = "#000000";
      sctx.fillRect(0, 0, w, h);
      sctx.globalCompositeOperation = "destination-out";
      sctx.drawImage(dataCanvas, 0, 0);
      sctx.globalCompositeOperation = "source-over";

      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(scratch, 0, 0);
      dirty = true;
      renderVisual();
    },

    /** Start a stroke at the given canvas coordinates. */
    beginStroke(x, y, eraseOverride = null) {
      pushHistory();
      drawing = true;
      lastPoint = null;
      const prevMode = st.mode;
      if (eraseOverride === true) st.mode = "eraser";
      else if (eraseOverride === false) st.mode = "brush";
      strokeTo(x, y);
      if (eraseOverride !== null) st.mode = prevMode;
      dirty = true;
      scheduleVisual();
    },

    /** Continue the current stroke. */
    continueStroke(x, y) {
      if (!drawing) return;
      strokeTo(x, y);
      dirty = true;
      scheduleVisual();
    },

    /** Finish the current stroke. */
    endStroke() {
      if (!drawing) return;
      drawing = false;
      lastPoint = null;
      renderVisual();
    },

    undo,
    redo,

    /** PNG data URL of the mask (white strokes on a transparent background). */
    toDataURL() {
      return snapshot();
    },

    /** Restore the mask from a PNG data URL. */
    loadDataURL(url) {
      return restoreFromDataURL(url).then((ok) => {
        dirty = ok;
        renderVisual();
        return ok;
      });
    },

    /** Fraction of the frame covered by the mask (0..1). */
    coverage() {
      const w = 64;
      const h = 64;
      if (!scratch) scratch = document.createElement("canvas");
      scratch.width = w;
      scratch.height = h;
      const sctx = scratch.getContext("2d");
      sctx.clearRect(0, 0, w, h);
      sctx.drawImage(dataCanvas, 0, 0, w, h);
      const data = sctx.getImageData(0, 0, w, h).data;
      let sum = 0;
      for (let i = 3; i < data.length; i += 4) sum += data[i];
      return sum / (255 * w * h);
    },

    isEmpty() {
      return this.coverage() <= 0.002;
    },

    renderVisual,
    scheduleVisual,
  };
}
