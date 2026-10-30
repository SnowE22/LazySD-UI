/* ============================================================
   QuantFunc WebUI — Native ES6 Main Entry Point
   Composes isolated modules into reactive App object.
   Zero bundlers, zero build step, pure vanilla JavaScript.
   ============================================================ */

import { state } from "./state.js";
import { escapeHtml, toast, apiGet, apiPost, apiPatch, apiDelete, apiUpload } from "./api.js";
import * as zoomPan from "./modules/zoom-pan.js";
import * as dock from "./modules/dock.js";
import * as newDock from "./modules/new-dock.js";
import * as modelsHub from "./modules/models-hub.js";
import * as loraHub from "./modules/lora-hub.js";
import * as settings from "./modules/settings.js";
import * as generator from "./modules/generator.js";
import * as halftone from "./modules/halftone.js";
import * as manga from "./modules/manga.js";
import * as maskCore from "./modules/mask-core.js";
import * as generateMask from "./modules/generate-mask.js";
import * as inpaint from "./modules/inpaint.js";

export const App = {
  state,

  // --- Network Layer ---
  apiGet,
  apiPost,
  apiPatch,
  apiDelete,
  apiUpload,
  toast,
  escapeHtml,

  // --- App Lifecycle ---
  async init() {
    console.log("[main] QuantFunc WebUI initializing (ES6 Native Modules)...");

    try { this.initSidebar(); } catch (e) { console.warn("[init] initSidebar error:", e); }
    try { this.initInspector(); } catch (e) { console.warn("[init] initInspector error:", e); }

    if (window.initI18n) {
      try { await window.initI18n(); } catch (e) { console.warn("[init] initI18n error:", e); }
    }

    try { await this.loadConfig(); } catch (e) { console.warn("[init] loadConfig error:", e); }
    try { await this.loadPresets(); } catch (e) { console.warn("[init] loadPresets error:", e); }

    try { this.renderGpuIndicator(); } catch (e) { console.warn("[init] renderGpuIndicator error:", e); }
    try { this.renderSystemStats(this.state.config?.system_stats); } catch (e) { console.warn("[init] renderSystemStats error:", e); }
    try { this.startStatsPolling(); } catch (e) { console.warn("[init] startStatsPolling error:", e); }

    try { this.renderModels(); } catch (e) { console.warn("[init] renderModels error:", e); }
    try { this.updateEngineStatusUI(); } catch (e) { console.warn("[init] updateEngineStatusUI error:", e); }
    try { this.updateDockPills(); } catch (e) { console.warn("[init] updateDockPills error:", e); }
    try { this.initDockSettings(); } catch (e) { console.warn("[init] initDockSettings error:", e); }

    try { this.loadLoras(); } catch (e) { console.warn("[init] loadLoras error:", e); }
    try { this.initCompareEvents(); } catch (e) { console.warn("[init] initCompareEvents error:", e); }
    try { this.loadShareStatus(); } catch (e) { console.warn("[init] loadShareStatus error:", e); }
    try { this.refreshExtrasStatus(); } catch (e) { console.warn("[init] refreshExtrasStatus error:", e); }

    try { this.bindEvents(); } catch (e) { console.warn("[init] bindEvents error:", e); }
    try { this.initAllViewportZoomPan(); } catch (e) { console.warn("[init] initAllViewportZoomPan error:", e); }
    try { this.initViewportHudOpacity(); } catch (e) { console.warn("[init] initViewportHudOpacity error:", e); }
    try { this.initGenMaskUI(); } catch (e) { console.warn("[init] initGenMaskUI error:", e); }
    try { this.initInpaintStudio(); } catch (e) { console.warn("[init] initInpaintStudio error:", e); }

    console.log("[main] Ready. GPU:", this.state.gpuInfo?.name || "not detected");
  },

  // --- Sidebar & Inspector ---
  toggleSidebar() {
    const shell = document.querySelector(".app-shell");
    if (!shell) return;
    const isCollapsed = shell.classList.toggle("sidebar-collapsed");
    try {
      localStorage.setItem("lazysd_sidebar_collapsed", isCollapsed ? "1" : "0");
    } catch (_) {}
  },

  initSidebar() {
    try {
      const saved = localStorage.getItem("lazysd_sidebar_collapsed");
      if (saved === "1") {
        document.querySelector(".app-shell")?.classList.add("sidebar-collapsed");
      }
    } catch (_) {}
  },

  toggleInspector() {
    const inspector = document.getElementById("topaz-inspector");
    const btn = document.getElementById("inspector-toggle-btn");
    if (!inspector) return;
    const isCollapsed = inspector.classList.toggle("collapsed");
    if (btn) btn.classList.toggle("active", !isCollapsed);
    try {
      localStorage.setItem("lazysd_inspector_collapsed", isCollapsed ? "1" : "0");
    } catch (_) {}
  },

  initInspector() {
    const inspector = document.getElementById("topaz-inspector");
    const btn = document.getElementById("inspector-toggle-btn");
    if (!inspector) return;
    try {
      const saved = localStorage.getItem("lazysd_inspector_collapsed");
      const isCollapsed = saved === "1";
      inspector.classList.toggle("collapsed", isCollapsed);
      if (btn) btn.classList.toggle("active", !isCollapsed);
    } catch (_) {}
  },

  // --- Config & System Stats ---
  async loadConfig() {
    try {
      const config = await this.apiGet("/api/config");
      this.state.config = config;
      this.state.account = config.account || { role: "host", id: "host", username: "Хост", is_guest: false };
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = this.sortModelsList(config.models || []);
      if (config.settings?.qwen_vae_type) {
        this.state.selectedVae = config.settings.qwen_vae_type;
      }
      window.QF_SETTINGS = config.settings;
      this.updateAccountRoleUI();
      const hideUninstalledChk = document.getElementById("setting-hide-uninstalled-gen");
      if (hideUninstalledChk && config.settings) {
        hideUninstalledChk.checked = Boolean(config.settings.hide_uninstalled_in_generator);
      }
    } catch (e) {
      console.error("[app] Config load failed:", e);
      this.toast("error", "Не удалось загрузить конфигурацию сервера");
    }
  },

  async refreshConfig() {
    try {
      const config = await this.apiGet("/api/config");
      this.state.config = config;
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = this.sortModelsList(config.models || []);
      this.renderGpuIndicator();
      this.renderSystemStats(config.system_stats);
      this.renderModels();
      this.updateEngineStatusUI();
    } catch (e) {
      console.warn("[app] Refresh config failed:", e);
    }
  },

  startStatsPolling() {
    if (this._statsInterval) clearInterval(this._statsInterval);
    this._statsInterval = setInterval(async () => {
      try {
        const stats = await this.apiGet("/api/system/stats");
        this.renderSystemStats(stats);
      } catch (_) {}
    }, 3500);
  },

  updateAccountRoleUI() {
    const acc = this.state.account || { role: "host", id: "host", username: "Хост", is_guest: false };
    const isGuest = Boolean(acc.is_guest);
    this.state.isGuest = isGuest;

    if (isGuest) {
      document.body.classList.add("role-guest");
    } else {
      document.body.classList.remove("role-guest");
      this.loadGuestAccounts();
    }
  },

  updateEngineStatusUI() {
    const badge = document.getElementById("engine-status-badge");
    if (!badge) return;
    const st = this.state.engineStatus;
    if (st && st.status === "ready") {
      badge.textContent = `✓ Engine v${st.version || '0.0.17'}`;
      badge.className = "engine-badge ready";
    } else {
      badge.textContent = "Engine Missing";
      badge.className = "engine-badge missing";
    }
  },

  // --- Router & Navigation ---
  switchTab(tabName) {
    this.state.currentTab = tabName;
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    document.querySelectorAll(".tab-panel").forEach(panel => {
      panel.style.display = "none";
    });

    const targetId = (tabName === "img2img" || tabName === "image_edit") ? "tab-generate" : `tab-${tabName}`;
    const activePanel = document.getElementById(targetId);
    if (activePanel) {
      const flexPanels = ["tab-generate", "tab-inpaint"];
      activePanel.style.display = flexPanels.includes(targetId) ? "flex" : "block";
    }

    if (tabName === "history") {
      this.loadHistory();
    } else if (tabName === "models") {
      this.renderModelsTab();
    } else if (tabName === "templates") {
      this.renderPresetsUI();
    } else if (tabName === "halftone") {
      this.initHalftoneStudio();
    } else if (tabName === "manga") {
      this.initMangaStudio();
    } else if (tabName === "inpaint") {
      this.initInpaintStudio();
    }
  },

  // --- Presets / Templates ---
  async loadPresets() {
    try {
      const res = await this.apiGet("/api/presets");
      this.state.presets = (res && res.presets) ? res.presets : (Array.isArray(res) ? res : []);
      this.renderPresetsUI();
    } catch (e) {
      console.warn("[main] Failed to load presets:", e);
      this.state.presets = [];
      this.renderPresetsUI();
    }
  },

  renderPresetsUI() {
    const grid = document.getElementById("dock-presets-grid") || document.getElementById("presets-grid");
    if (!grid) return;
    const presets = this.state.presets || [];
    if (!presets.length) {
      grid.innerHTML = '<div style="font-size: 11px; color: var(--text-dim); padding: 8px;">Нет сохранённых шаблонов</div>';
      return;
    }
    grid.innerHTML = presets.map((p, idx) => `
      <div class="dock-preset-chip" onclick="App.applyPreset(${idx})">
        <span>${escapeHtml(p.name || p.title || "Preset")}</span>
        <button class="preset-del-btn" onclick="App.deletePreset('${escapeHtml(p.id)}', event)" title="Удалить">×</button>
      </div>
    `).join("");
  },

  openSavePresetModal() {
    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "flex";
  },

  closeSavePresetModal() {
    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "none";
  },

  async saveCurrentAsPreset() {
    const nameInput = document.getElementById("preset-name-input");
    const name = nameInput ? nameInput.value.trim() : "";
    if (!name) {
      this.toast("warning", "Введите имя шаблона");
      return;
    }
    const promptInput = document.getElementById("prompt-input");
    const payload = {
      name,
      prompt: promptInput ? promptInput.value : "",
      model_id: this.state.selectedModel || "",
      variant_id: this.state.selectedVariant || "",
      cfg: parseFloat(document.getElementById("cfg-input")?.value || "1.0"),
      noise: parseFloat(document.getElementById("noise-input")?.value || "0.0"),
      steps: parseInt(document.getElementById("steps-input")?.value || "4", 10),
      seed: parseInt(document.getElementById("seed-input")?.value || "-1", 10),
      resolution: document.getElementById("resolution-select")?.value || "1024x1024",
    };
    try {
      await this.apiPost("/api/presets", payload);
      this.closeSavePresetModal();
      await this.loadPresets();
      this.toast("success", `Шаблон "${name}" сохранён`);
    } catch (e) {
      this.toast("error", e.message || "Ошибка сохранения шаблона");
    }
  },

  async deletePreset(presetId, event) {
    if (event) event.stopPropagation();
    if (!confirm("Удалить этот шаблон?")) return;
    try {
      await this.apiDelete(`/api/presets/${encodeURIComponent(presetId)}`);
      await this.loadPresets();
      this.toast("info", "Шаблон удалён");
    } catch (e) {
      this.toast("error", e.message || "Ошибка удаления шаблона");
    }
  },

  applyPreset(idx) {
    const p = this.state.presets?.[idx];
    if (!p) return;
    const promptInput = document.getElementById("prompt-input");
    if (promptInput && p.prompt) promptInput.value = p.prompt;
    if (p.model_id) {
      const sel = document.getElementById("model-select");
      if (sel) {
        sel.value = p.model_id;
        this.onModelChange(p.variant_id);
      }
    }
    this.closeAllDockPopovers();
    this.updateDockPills();
    this.toast("info", `Шаблон "${p.name}" применён`);
  },

  // --- Compare Curtain ---
  initCompareEvents() {
    const bar = document.getElementById("compare-slider-bar");
    const container = document.getElementById("compare-container");
    if (!bar || !container) return;

    let dragging = false;
    const onMove = (e) => {
      if (!dragging) return;
      const rect = container.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      const pct = (x / rect.width) * 100;
      this.updateCompareSlider(pct);
    };

    bar.addEventListener("pointerdown", (e) => {
      dragging = true;
      try { bar.setPointerCapture(e.pointerId); } catch (_) {}
    });
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", (e) => {
      if (dragging) {
        dragging = false;
        try { bar.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    });
  },

  toggleCompare() {
    const container = document.getElementById("compare-container");
    const mainImg = document.getElementById("preview-img");
    const btn = document.getElementById("btn-compare-toggle");
    if (!container || !mainImg) return;

    this.state.compareActive = !this.state.compareActive;
    container.style.display = this.state.compareActive ? "block" : "none";
    mainImg.style.display = this.state.compareActive ? "none" : "block";
    if (btn) btn.classList.toggle("active", this.state.compareActive);

    if (this.state.compareActive) {
      this.updateCompareImages();
      this.updateCompareSlider(this.state.comparePercentage || 50);
    }
  },

  updateCompareImages() {
    const beforeImg = document.getElementById("compare-before-img");
    const afterImg = document.getElementById("compare-after-img");
    if (beforeImg) beforeImg.src = this.state.previousPreviewUrl || this.state.currentPreviewUrl || "";
    if (afterImg) afterImg.src = this.state.currentPreviewUrl || "";
  },

  updateCompareSlider(pct) {
    this.state.comparePercentage = pct;
    const clip = document.getElementById("compare-after-clip");
    const bar = document.getElementById("compare-slider-bar");
    if (clip) clip.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    if (bar) bar.style.left = `${pct}%`;
  },

  // --- Model Selector & Variant sync ---
  renderModels() {
    const sel = document.getElementById("model-select");
    if (!sel) return;
    let models = this.state.models.filter(m => !m.hidden_from_host);
    if (this.state.settings?.hide_uninstalled_in_generator) {
      const installedOnly = models.filter(m => m.installed);
      if (installedOnly.length > 0) models = installedOnly;
    }
    sel.innerHTML = models.map(m => {
      const inst = m.installed ? "✓ " : "⬇ ";
      return `<option value="${escapeHtml(m.id)}">${inst}${escapeHtml(m.name)}</option>`;
    }).join("");

    if (!this.state.selectedModel && models.length > 0) {
      const firstInst = models.find(m => m.installed) || models[0];
      this.state.selectedModel = firstInst.id;
      sel.value = firstInst.id;
    } else if (this.state.selectedModel) {
      sel.value = this.state.selectedModel;
    }
    this.onModelChange();
  },

  async onModelChange(targetVariantId = null) {
    const sel = document.getElementById("model-select");
    const modelId = sel?.value || this.state.selectedModel;
    if (!modelId) return;

    this.state.selectedModel = modelId;
    try {
      const stateData = await this.apiGet(`/api/models/${modelId}/state`);
      this.state.modelState = stateData;
      this.renderVariants(stateData, targetVariantId);
    } catch (_) {
      this.renderVariants(null, targetVariantId);
    }
    this.updateModelTrigger();
    this.updateDockPills();
  },

  renderVariants(stateData, targetVariantId = null) {
    const varSel = document.getElementById("variant-select");
    if (!varSel) return;
    const transformers = stateData?.transformers || [];
    varSel.innerHTML = transformers.map(t => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.name || t.id)}</option>`).join("");

    if (targetVariantId && transformers.some(t => t.id === targetVariantId)) {
      varSel.value = targetVariantId;
    } else if (transformers.length > 0) {
      varSel.value = transformers[0].id;
    }
    this.state.selectedVariant = varSel.value;
  },

  updateModelTrigger() {
    generator.updateModelTrigger ? generator.updateModelTrigger(this) : null;
    const current = this.state.models.find(m => m.id === this.state.selectedModel);
    const nameEl = document.getElementById("model-trigger-name");
    const subEl = document.getElementById("model-trigger-sub");
    if (nameEl) nameEl.textContent = current?.name || "—";
    if (subEl) subEl.textContent = current ? `${this.state.selectedVariant || "default"}` : "—";
    this.updateDockPills();
  },

  // --- Bind Global DOM Events ---
  bindEvents() {
    if (this._eventsBound) return;
    this._eventsBound = true;

    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.addEventListener("click", () => {
        if (!btn.disabled) this.switchTab(btn.dataset.tab);
      });
    });

    const modelSelect = document.getElementById("model-select");
    if (modelSelect) modelSelect.addEventListener("change", () => this.onModelChange());

    const modelPickerTrigger = document.getElementById("model-picker-trigger");
    if (modelPickerTrigger) modelPickerTrigger.addEventListener("click", () => this.openModelPickerModal());

    const genBtn = document.getElementById("generate-btn");
    if (genBtn) genBtn.addEventListener("click", () => this.onGenerate());

    const dockGenBtn = document.getElementById("dock-generate-btn");
    if (dockGenBtn) dockGenBtn.addEventListener("click", () => this.onGenerate());

    // Sync sliders
    const syncPair = (sliderId, inputId) => {
      const s = document.getElementById(sliderId);
      const i = document.getElementById(inputId);
      if (s && i) {
        s.addEventListener("input", () => { i.value = s.value; this.updateDockPills(); });
        i.addEventListener("input", () => { s.value = i.value; this.updateDockPills(); });
      }
    };
    syncPair("cfg-slider", "cfg-input");
    syncPair("noise-slider", "noise-input");
    // Global dragover & drop prevention so dropped files NEVER open in the browser
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
    }, false);
    window.addEventListener("drop", (e) => {
      if (!e.target.closest("#tab-generate") && !e.target.closest("#ht-dropzone") && !e.target.closest("#manga-dropzone")) {
        e.preventDefault();
      }
    }, false);

    // Dropzone setup for Generator tab
    const previewArea = document.getElementById("tab-generate");
    if (previewArea && window.Dropzone) {
      this.dropzone = new window.Dropzone(previewArea, {
        onFile: (fileInfo) => {
          this.uploadImg2ImgFile(fileInfo.file);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 50,
      });
    }

    // Dropzone setup for Halftone Studio tab
    const htArea = document.getElementById("ht-dropzone");
    if (htArea && window.Dropzone) {
      this.dropzoneHt = new window.Dropzone(htArea, {
        onFile: (fileInfo) => {
          this.loadHalftoneSource(fileInfo.dataUrl, fileInfo.name);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 50,
      });
    }

    // Dropzone setup for Manga Screen AI tab
    const mangaArea = document.getElementById("manga-dropzone");
    if (mangaArea && window.Dropzone) {
      this.dropzoneManga = new window.Dropzone(mangaArea, {
        onFile: (fileInfo) => {
          this.loadMangaSource(fileInfo.dataUrl, fileInfo.name);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 50,
      });
    }


    // Close popovers on click outside
    document.addEventListener("click", (e) => {
      if (!this.state.activeDockPopover) return;
      const dockWrap = document.getElementById("gemini-dock-wrap");
      const newDockWrap = document.getElementById("new-dock-wrap");
      if (dockWrap && !dockWrap.contains(e.target) && !newDockWrap?.contains(e.target)) {
        this.closeAllDockPopovers();
      }
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        this.onGenerate();
      }
      if (e.key === "Escape") {
        if (this.state.activeDockPopover) this.closeAllDockPopovers();
        else if (this.state.isModelPickerOpen) this.closeModelPickerModal();
        else if (this.state.selectedHistoryItem) this.closeHistoryModal();
      }
    });
  },

  dismissUninstalledNotice() {
    const notice = document.getElementById("gen-uninstalled-notice");
    if (notice) notice.style.display = "none";
    const banner = document.getElementById("uninstalled-models-notice");
    if (banner) banner.style.display = "none";
  },

  async onSettingHideUninstalledChange(chk) {
    const checked = (typeof chk === "boolean") ? chk : Boolean(chk?.checked ?? chk);
    try {
      await this.apiPatch("/api/settings", { hide_uninstalled_in_generator: checked });
      if (this.state.settings) this.state.settings.hide_uninstalled_in_generator = checked;
      this.renderModels();
      this.updateDockPills();
      this.toast("info", checked ? "Нескачанные модели скрыты из Дока" : "Отображаются все модели");
    } catch (e) {
      this.toast("error", e.message || "Ошибка сохранения настроек");
    }
  },

  goToModelInHub(modelId) {
    const id = modelId || this.state.selectedModel || "";
    this.switchTab("models");
    this.state.modelsTabSearch = id;
    const searchInput = document.getElementById("models-tab-search");
    if (searchInput) {
      searchInput.value = id;
      if (this.onModelsTabSearch) this.onModelsTabSearch();
    }
    this.renderModelsTab();
  },

  // --- Delegate to Modules ---
  ...zoomPan,
  ...dock,
  ...newDock,
  ...modelsHub,
  ...loraHub,
  ...settings,
  ...generator,
  ...halftone,
  ...manga,
  ...maskCore,
  ...generateMask,
  ...inpaint,
};

// ============================================================
// Mount global window bridges for legacy HTML inline handlers
// ============================================================
window.App = App;
window.escapeHtml = escapeHtml;
window.toast = toast;

window.loadHistory = () => App.loadHistory();
window.closeHistoryModal = () => App.closeHistoryModal();
window.clearImg2ImgRef = () => App.clearImg2ImgRef();
window.updateStrengthLabel = (val) => App.updateStrengthLabel(val);
window.sendCurrentPreviewToImg2Img = () => App.sendCurrentPreviewToImg2Img();
window.verifyEngine = (v) => App.verifyEngine(v);
window.downloadEngine = (v, e) => App.downloadEngine(v, e);

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    App.init();
  });
} else {
  App.init();
}
