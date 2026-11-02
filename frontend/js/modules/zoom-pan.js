/* ============================================================
   QuantFunc WebUI — Viewport Zoom & Pan Engine
   Smooth cursor-centered zoom, pan, floating HUD, and scrollbars.
   Shared across Generation, Halftone Studio GT, and Manga Studio.
   ============================================================ */

import { toast } from "../api.js";

export const viewports = {};
let viewportsInitialized = false;
let spacePressed = false;

export function initAllViewportZoomPan(appRef = null) {
  if (viewportsInitialized) return;
  viewportsInitialized = true;

  initViewportZoomPan("generate", {
    areaSelector: "#tab-generate .preview-area",
    wrapId: "preview-image-wrap",
    hudId: "gen-zoom-hud",
    labelId: "gen-zoom-label",
    scrollXId: "gen-scrollbar-x",
    thumbXId: "gen-scrollbar-thumb-x",
    scrollYId: "gen-scrollbar-y",
    thumbYId: "gen-scrollbar-thumb-y",
  }, appRef);

  initViewportZoomPan("halftone", {
    areaId: "ht-dropzone",
    wrapId: "ht-preview-wrap",
    hudId: "ht-zoom-hud",
    labelId: "ht-zoom-label",
    scrollXId: "ht-scrollbar-x",
    thumbXId: "ht-scrollbar-thumb-x",
    scrollYId: "ht-scrollbar-y",
    thumbYId: "ht-scrollbar-thumb-y",
  }, appRef);

  initViewportZoomPan("manga", {
    areaId: "manga-dropzone",
    wrapId: "manga-preview-wrap",
    hudId: "manga-zoom-hud",
    labelId: "manga-zoom-label",
    scrollXId: "manga-scrollbar-x",
    thumbXId: "manga-scrollbar-thumb-x",
    scrollYId: "manga-scrollbar-y",
    thumbYId: "manga-scrollbar-thumb-y",
  }, appRef);

  initViewportZoomPan("inpaint", {
    areaId: "inpaint-canvas-area",
    wrapId: "inpaint-canvas-wrap",
    hudId: "inpaint-zoom-hud",
    labelId: "inpaint-zoom-label",
  }, appRef);

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
      spacePressed = true;
      window.spacePressed = true;
      document.querySelectorAll(".preview-area, .studio-canvas-area").forEach(el => el.classList.add("space-held"));
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      spacePressed = false;
      window.spacePressed = false;
      document.querySelectorAll(".preview-area, .studio-canvas-area").forEach(el => el.classList.remove("space-held"));
    }
  });

  window.addEventListener("blur", () => {
    spacePressed = false;
    window.spacePressed = false;
    document.querySelectorAll(".preview-area, .studio-canvas-area").forEach(el => el.classList.remove("space-held"));
  });
}

export function initViewportZoomPan(mode, cfg, appRef = null) {
  const area = cfg.areaId ? document.getElementById(cfg.areaId) : document.querySelector(cfg.areaSelector);
  const wrap = document.getElementById(cfg.wrapId);
  if (!area || !wrap) return;

  const vp = {
    mode,
    area,
    wrap,
    hud: document.getElementById(cfg.hudId),
    label: document.getElementById(cfg.labelId),
    scrollX: document.getElementById(cfg.scrollXId),
    thumbX: document.getElementById(cfg.thumbXId),
    scrollY: document.getElementById(cfg.scrollYId),
    thumbY: document.getElementById(cfg.thumbYId),
    zoom: 1.0,
    minZoom: 0.5,
    maxZoom: 8.0,
    panX: 0,
    panY: 0,
    hideTimer: null,
  };

  viewports[mode] = vp;

  // 1. Mouse Wheel Zoom centered around cursor
  area.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.16 : 1 / 1.16;
    const nextZoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
    if (Math.abs(nextZoom - vp.zoom) < 0.001) {
      showViewportHudAndScrollbars(vp);
      return;
    }

    const areaRect = area.getBoundingClientRect();
    const cx = e.clientX - (areaRect.left + areaRect.width * 0.5);
    const cy = e.clientY - (areaRect.top + areaRect.height * 0.5);
    const ratio = nextZoom / vp.zoom;

    vp.panX = cx - (cx - vp.panX) * ratio;
    vp.panY = cy - (cy - vp.panY) * ratio;
    vp.zoom = nextZoom;

    applyViewportTransform(vp, true);
  }, { passive: false });

  // 2. Pointer Pan & Zoom-Tool Click/Drag
  let isPanning = false;
  let startX = 0;
  let startY = 0;
  let startPanX = 0;
  let startPanY = 0;
  let movedDist = 0;
  let isZoomToolClick = false;

  area.addEventListener("pointerdown", (e) => {
    // The inpaint mask canvas handles its own pointer events (painting).
    if (e.button === 0 && e.target.closest?.(".inpaint-mask-canvas")) {
      return;
    }
    if (
      e.target.closest(".viewport-zoom-hud") ||
      e.target.closest(".viewport-scrollbar-x") ||
      e.target.closest(".viewport-scrollbar-y") ||
      e.target.closest(".preview-toolbar") ||
      e.target.closest(".mask-fab") ||
      e.target.closest(".mask-panel") ||
      e.target.closest(".inpaint-canvas-actions") ||
      e.target.closest(".compare-slider-bar")
    ) {
      return;
    }

    const htTool = (appRef?.state?.halftone?.tool || window.App?.state?.halftone?.tool) || "view";
    const isMiddle = e.button === 1;
    const isRight = e.button === 2;
    const isLeft = e.button === 0;

    if (mode === "halftone" && htTool === "zoom" && (isLeft || isRight)) {
      isPanning = true;
      isZoomToolClick = true;
    } else if (isMiddle || (isLeft && spacePressed)) {
      isPanning = true;
      isZoomToolClick = false;
    } else if (isLeft && mode === "halftone" && htTool === "view") {
      isPanning = true;
      isZoomToolClick = false;
    } else if (isLeft && (mode === "manga" || mode === "generate")) {
      isPanning = true;
      isZoomToolClick = false;
    } else {
      return;
    }

    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    startPanX = vp.panX;
    startPanY = vp.panY;
    movedDist = 0;

    area.classList.add("panning-active");
    try { area.setPointerCapture(e.pointerId); } catch (_) {}
    showViewportHudAndScrollbars(vp);
  });

  area.addEventListener("pointermove", (e) => {
    if (!isPanning) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    movedDist = Math.hypot(dx, dy);

    vp.panX = startPanX + dx;
    vp.panY = startPanY + dy;
    applyViewportTransform(vp, true);
  });

  const endPan = (e) => {
    if (!isPanning) return;
    isPanning = false;
    area.classList.remove("panning-active");
    try { area.releasePointerCapture(e.pointerId); } catch (_) {}

    if (isZoomToolClick && movedDist < 6) {
      const zoomOut = e.altKey || e.button === 2;
      const factor = zoomOut ? 1 / 1.35 : 1.35;
      const nextZoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
      const areaRect = area.getBoundingClientRect();
      const cx = e.clientX - (areaRect.left + areaRect.width * 0.5);
      const cy = e.clientY - (areaRect.top + areaRect.height * 0.5);
      const ratio = nextZoom / vp.zoom;

      vp.panX = cx - (cx - vp.panX) * ratio;
      vp.panY = cy - (cy - vp.panY) * ratio;
      vp.zoom = nextZoom;

      applyViewportTransform(vp, true);
    }
    isZoomToolClick = false;
  };

  area.addEventListener("pointerup", endPan);
  area.addEventListener("pointercancel", endPan);
  area.addEventListener("contextmenu", (e) => {
    const htTool = (appRef?.state?.halftone?.tool || window.App?.state?.halftone?.tool);
    if (mode === "halftone" && htTool === "zoom") {
      e.preventDefault();
    }
  });

  // Double-click resets zoom to 100% Fit
  area.addEventListener("dblclick", (e) => {
    if (
      e.target.closest(".viewport-zoom-hud") ||
      e.target.closest(".preview-toolbar")
    ) {
      return;
    }
    const htTool = (appRef?.state?.halftone?.tool || window.App?.state?.halftone?.tool);
    if (mode !== "halftone" || htTool === "view") {
      resetViewportZoom(mode);
    }
  });

  // 3. Interactive Draggable Floating Scrollbars
  bindScrollbarDrag(vp, "x");
  bindScrollbarDrag(vp, "y");
}

function bindScrollbarDrag(vp, axis) {
  const track = axis === "x" ? vp.scrollX : vp.scrollY;
  const thumb = axis === "x" ? vp.thumbX : vp.thumbY;
  if (!track || !thumb) return;

  let dragging = false;
  let startPos = 0;
  let startPan = 0;

  thumb.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    dragging = true;
    startPos = axis === "x" ? e.clientX : e.clientY;
    startPan = axis === "x" ? vp.panX : vp.panY;

    try { thumb.setPointerCapture(e.pointerId); } catch (_) {}
    showViewportHudAndScrollbars(vp);
  });

  thumb.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const delta = (axis === "x" ? e.clientX : e.clientY) - startPos;
    const bounds = getViewportPanBounds(vp);
    const maxPan = axis === "x" ? bounds.maxPanX : bounds.maxPanY;
    const trackLen = axis === "x" ? track.clientWidth : track.clientHeight;
    const thumbLen = axis === "x" ? thumb.offsetWidth : thumb.offsetHeight;
    const avail = Math.max(1, trackLen - thumbLen);

    if (maxPan > 0) {
      const panDelta = -(delta / avail) * (maxPan * 2);
      if (axis === "x") vp.panX = startPan + panDelta;
      else vp.panY = startPan + panDelta;
      applyViewportTransform(vp, true);
    }
  });

  const stopThumb = (e) => {
    if (!dragging) return;
    dragging = false;
    try { thumb.releasePointerCapture(e.pointerId); } catch (_) {}
  };

  thumb.addEventListener("pointerup", stopThumb);
  thumb.addEventListener("pointercancel", stopThumb);
}

export function getViewportPanBounds(vp) {
  const vw = vp.area.clientWidth || 1;
  const vh = vp.area.clientHeight || 1;
  const baseW = vp.wrap.offsetWidth || vw * 0.8;
  const baseH = vp.wrap.offsetHeight || vh * 0.8;
  const cw = baseW * vp.zoom;
  const ch = baseH * vp.zoom;

  const marginX = Math.max(vw, cw) * 0.75;
  const marginY = Math.max(vh, ch) * 0.75;
  const maxPanX = Math.max(marginX * 0.5, (cw - vw) * 0.5 + marginX);
  const maxPanY = Math.max(marginY * 0.5, (ch - vh) * 0.5 + marginY);

  return { vw, vh, baseW, baseH, cw, ch, maxPanX, maxPanY };
}

export function applyViewportTransform(vp, showHud = true) {
  if (!vp || !vp.wrap) return;
  const bounds = getViewportPanBounds(vp);

  vp.panX = Math.max(-bounds.maxPanX, Math.min(bounds.maxPanX, vp.panX));
  vp.panY = Math.max(-bounds.maxPanY, Math.min(bounds.maxPanY, vp.panY));

  if (Math.abs(vp.zoom - 1.0) < 0.01 && Math.hypot(vp.panX, vp.panY) < 4) {
    vp.zoom = 1.0;
    vp.panX = 0;
    vp.panY = 0;
  }

  vp.wrap.style.transform = `translate3d(${vp.panX.toFixed(1)}px, ${vp.panY.toFixed(1)}px, 0) scale(${vp.zoom.toFixed(4)})`;

  if (vp.label) {
    vp.label.textContent = `🔍 ${Math.round(vp.zoom * 100)}%`;
  }

  const showScroll = vp.zoom > 1.03;
  if (vp.scrollX && vp.thumbX) {
    if (showScroll && bounds.maxPanX > 1) {
      const trackW = vp.scrollX.clientWidth || 200;
      const ratioW = Math.max(0.15, Math.min(0.85, bounds.vw / Math.max(bounds.cw, bounds.vw)));
      const thumbW = Math.max(28, Math.round(trackW * ratioW));
      const normX = Math.max(0, Math.min(1, (bounds.maxPanX - vp.panX) / (2 * bounds.maxPanX)));
      const leftPx = Math.round(normX * (trackW - thumbW));
      vp.thumbX.style.width = `${thumbW}px`;
      vp.thumbX.style.left = `${leftPx}px`;
    }
  }

  if (vp.scrollY && vp.thumbY) {
    if (showScroll && bounds.maxPanY > 1) {
      const trackH = vp.scrollY.clientHeight || 200;
      const ratioH = Math.max(0.15, Math.min(0.85, bounds.vh / Math.max(bounds.ch, bounds.vh)));
      const thumbH = Math.max(28, Math.round(trackH * ratioH));
      const normY = Math.max(0, Math.min(1, (bounds.maxPanY - vp.panY) / (2 * bounds.maxPanY)));
      const topPx = Math.round(normY * (trackH - thumbH));
      vp.thumbY.style.height = `${thumbH}px`;
      vp.thumbY.style.top = `${topPx}px`;
    }
  }

  if (showHud) {
    showViewportHudAndScrollbars(vp);
  }
}

export function showViewportHudAndScrollbars(vp) {
  if (!vp) return;
  if (vp.hud) vp.hud.classList.add("visible");

  const showScroll = vp.zoom > 1.03;
  if (vp.scrollX) vp.scrollX.classList.toggle("visible", showScroll);
  if (vp.scrollY) vp.scrollY.classList.toggle("visible", showScroll);

  if (vp.hideTimer) clearTimeout(vp.hideTimer);
  vp.hideTimer = setTimeout(() => {
    if (vp.hud) vp.hud.classList.remove("visible");
    if (vp.scrollX) vp.scrollX.classList.remove("visible");
    if (vp.scrollY) vp.scrollY.classList.remove("visible");
  }, 1450);
}

export function initViewportHudOpacity() {
  const saved = localStorage.getItem("lazysd_zoom_hud_opacity");
  if (saved) {
    document.documentElement.style.setProperty("--zoom-hud-opacity", saved);
  }
}

export function cycleViewportHudOpacity() {
  const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--zoom-hud-opacity")) || 0.92;
  let next = 0.92;
  if (cur > 0.85) next = 0.65;
  else if (cur > 0.55) next = 0.40;
  else if (cur > 0.30) next = 0.18;
  else next = 0.95;

  document.documentElement.style.setProperty("--zoom-hud-opacity", String(next));
  try {
    localStorage.setItem("lazysd_zoom_hud_opacity", String(next));
  } catch (_) {}
  toast("info", `Прозрачность панели зума: ${Math.round(next * 100)}%`);
}

export function stepViewportZoom(mode, dir) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  const factor = dir > 0 ? 1.25 : 1 / 1.25;
  vp.zoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
  applyViewportTransform(vp, true);
}

export function setViewportZoom(mode, targetZoom, showHud = true) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  vp.zoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, Number(targetZoom) || 1.0));
  applyViewportTransform(vp, showHud);
}

export function resetViewportZoom(mode) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  vp.zoom = 1.0;
  vp.panX = 0;
  vp.panY = 0;
  applyViewportTransform(vp, true);
}
