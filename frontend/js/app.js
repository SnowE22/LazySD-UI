/* ============================================================
   QuantFunc WebUI — main app JavaScript
   Bridges frontend HTML/CSS with Python backend via REST API.

   Backend endpoints (defined in app.py):
     GET  /api/config              → {settings, gpu, engine_status, models}
     POST /api/generate            → starts generation job, returns job_id
     GET  /api/jobs/<id>           → polls job status
     POST /api/jobs/<id>/cancel    → cancels running job
     GET  /api/models              → list available models
     GET  /api/models/<id>/state   → check cache state for a model
     POST /api/models/<id>/download → download missing files
     GET  /api/history             → list output files
     PATCH /api/settings           → update settings
     POST /api/engine/download     → download engine binary
     POST /api/engine/verify       → verify engine checksum
     GET  /locale/<lang>.json      → translations
   ============================================================ */

function localizedDisabledReason(reason, fallbackKey = "models.disabled_notice") {
  const raw = String(reason || "").trim();
  if (!raw) return t(fallbackKey);
  if (raw.startsWith("Временно не поддерживается") || raw.includes("сырые веса upstream")) {
    return t("models.disabled_reason_raw");
  }
  return raw;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const ICON_DOWNLOAD = `<svg class="btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>`;
const ICON_TRASH = `<svg class="btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="m19 6-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`;
const ICON_CHECK = `<svg class="btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>`;

const App = {
  // ---------- State ----------
  state: {
    settings: null,
    gpuInfo: null,
    engineStatus: null,
    models: [],
    presets: [],
    currentTab: "generate",
    currentJob: null,
    pollInterval: null,
    selectedModel: null,
    selectedVariant: null,
    modelChangeRequestId: 0,
    historyItems: [],
    selectedHistoryItem: null,
    historyPage: 1,
    historyPageSize: 24,
    historyTotal: 0,
    historyPages: 0,
    historyLoading: false,
    historyRequestSeq: 0,
    historySearchTimer: null,
    historyServerPaginated: false,
    presence: null,
    presenceTimer: null,
    queueJobId: null,
    img2imgRef: null,
    lastPreviewName: null,
    lastPreviewUrl: null,
    modelPickerSearch: "",
    modelPickerFilter: "all",
    isModelPickerOpen: false,
    modelsTabFilter: "all",
    modelsTabSearch: "",
    encoders: null,
    selectedVae: "texture_fix",
    activeDockPopover: null,
    downloads: {},
    hfInspectedRepo: null,
    activeLoras: [],
    loraLibrary: [],
    modelCardEditor: null,
    componentPicker: null,
    loraBinding: null,
    compareActive: false,
    comparePercentage: 50,
    previousPreviewUrl: null,
    currentPreviewUrl: null,
    currentPreviewFilename: null,
    dockPosition: "center",
    dockMinimized: false,
    dockOpacity: 0.88,
    dockBlur: 20,
    dockDragEnabled: false,
    dockCustomX: null,
    dockCustomY: null,
    dockVariant: "legacy",
    newDockFloat: false,
    newDockCustomX: null,
    newDockCustomY: null,
    shareStatus: null,
    extrasStatus: null,
    account: null,
    guestAccounts: [],
    historyFilter: "all",
    genMask: null,
    inpaint: null,
  },

  // ---------- 3-Tier Model Sorting ----------
  sortModelsList(models) {
    if (!Array.isArray(models)) return [];
    const tierScore = (m) => {
      if (m.disabled === true) return 2;
      if (m.installed === true) return 0;
      return 1;
    };
    return [...models].sort((a, b) => {
      const ta = tierScore(a);
      const tb = tierScore(b);
      if (ta !== tb) return ta - tb;
      return String(a.name || "").localeCompare(String(b.name || ""), undefined, { sensitivity: "base" });
    });
  },

  // ---------- Init ----------
  async init() {
    Object.assign(this, await import("./modules/new-dock.js"));
    console.log("[app] LazySD UI initializing...");

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
    try { this.initHistorySettings(); } catch (e) { console.warn("[init] initHistorySettings error:", e); }
    try { this.initPresence(); } catch (e) { console.warn("[init] initPresence error:", e); }
    try { this.loadHistory(); } catch (e) { console.warn("[init] loadHistory error:", e); }
    try { this.bindEvents(); } catch (e) { console.warn("[init] bindEvents error:", e); }
    try { this.initAllViewportZoomPan(); } catch (e) { console.warn("[init] initAllViewportZoomPan error:", e); }
    try { this.initViewportHudOpacity(); } catch (e) { console.warn("[init] initViewportHudOpacity error:", e); }
    try { this.initGenMaskUI(); } catch (e) { console.warn("[init] initGenMaskUI error:", e); }
    try { this.initInpaintStudio(); } catch (e) { console.warn("[init] initInpaintStudio error:", e); }

    console.log("[app] Ready. GPU:", this.state.gpuInfo?.name || "not detected");
  },

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

  async toggleQuantFuncModels(checked) {
    try {
      await this.apiPatch("/api/settings", { enable_quantfunc_models: checked });
      if (this.state.settings) {
        this.state.settings.enable_quantfunc_models = checked;
      }
      this.toast("info", checked ? "Модели QuantFunc включены" : "Модели QuantFunc отключены");
      await this.refreshConfig();
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      console.error("[app] Failed to toggle QuantFunc models:", err);
      this.toast("error", `Ошибка: ${err.message}`);
    }
  },

  // ---------- API helpers ----------
  async _fetchWithRetry(path, options = {}, retries = 0) {
    let lastError = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      let res;
      try {
        res = await fetch(path, options);
      } catch (error) {
        if (error?.name === "AbortError") throw error;
        lastError = error;
        if (attempt >= retries) throw error;
        await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)));
        continue;
      }

      if (res.ok) return res;

      const payload = await res.json().catch(() => ({}));
      const error = new Error(payload.detail || res.statusText || `HTTP ${res.status}`);
      error.status = res.status;
      if (![502, 503, 504].includes(res.status) || attempt >= retries) throw error;
      lastError = error;
      await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)));
    }
    throw lastError || new Error("Request failed");
  },

  async apiGet(path, retries = 2) {
    const res = await this._fetchWithRetry(path, {}, retries);
    return res.json();
  },

  async apiPost(path, body = null, retries = 0) {
    const opts = {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : {},
    };
    if (body) opts.body = JSON.stringify(body);
    const res = await this._fetchWithRetry(path, opts, retries);
    return res.json();
  },

  async apiUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await this._fetchWithRetry("/api/upload", {
      method: "POST",
      body: formData,
    }, 0);
    return res.json();
  },

  async apiPatch(path, body) {
    const res = await fetch(path, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async apiDelete(path) {
    const res = await fetch(path, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  // ---------- Config & Identity ----------
  updateAccountRoleUI() {
    const acc = this.state.account || { role: "host", id: "host", username: t("common.host"), is_guest: false };
    const isGuest = Boolean(acc.is_guest);

    if (isGuest) {
      document.body.classList.add("role-guest");
      const badge = document.getElementById("user-role-badge");
      if (badge) {
        badge.className = "user-role-badge guest";
        badge.title = t("settings.account.guest_mode_title");
      }
      const icon = document.getElementById("user-role-icon");
      if (icon) icon.textContent = "👤";
      const title = document.getElementById("user-role-title");
      if (title) title.textContent = t("common.guest") + ": " + (acc.username || t("settings.guest.default_name"));
      const bannerName = document.getElementById("guest-username-display");
      if (bannerName) bannerName.textContent = acc.username || t("common.guest");
      const guestProfileUser = document.getElementById("guest-profile-username");
      if (guestProfileUser) guestProfileUser.textContent = acc.username || t("common.guest");
    } else {
      document.body.classList.remove("role-guest");
      const badge = document.getElementById("user-role-badge");
      if (badge) {
        badge.className = "user-role-badge host";
        badge.title = t("settings.account.host_mode_title");
      }
      const icon = document.getElementById("user-role-icon");
      if (icon) icon.textContent = "👑";
      const title = document.getElementById("user-role-title");
      if (title) title.textContent = t("common.host");

      // Host can view and manage guest accounts
      this.loadGuestAccounts();

      // Populate HuggingFace token in Settings
      const hfInput = document.getElementById("setting-hf-token");
      if (hfInput && this.state.settings?.huggingface_token) {
        hfInput.value = this.state.settings.huggingface_token;
      }
    }
  },

  async saveGuestPassword() {
    const input = document.getElementById("guest-input-new-password");
    const pwd = (input?.value || "").trim();
    if (!pwd || pwd.length < 3) {
      this.toast("warning", t("settings.guest.password_short"));
      return;
    }
    try {
      await this.apiPost("/api/account/change-password", { new_password: pwd });
      if (input) input.value = "";
      this.toast("success", t("settings.guest.password_saved"));
    } catch (e) {
      this.toast("error", t("settings.guest.password_failed") + e.message);
    }
  },

  async loadConfig() {
    try {
      const config = await this.apiGet("/api/config");
      this.state.config = config;
      this.state.account = config.account || { role: "host", id: "host", username: t("common.host"), is_guest: false };
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = this.sortModelsList(config.models || []);
      this.renderPaths(config);
      if (config.settings?.qwen_vae_type) {
        this.state.selectedVae = config.settings.qwen_vae_type;
      }
      // Inject into window for i18n detection
      window.QF_INITIAL_CONFIG = config;
      window.QF_SETTINGS = config.settings;
      this.updateAccountRoleUI();
      const hideUninstalledChk = document.getElementById("setting-hide-uninstalled-gen");
      if (hideUninstalledChk && config.settings) {
        hideUninstalledChk.checked = Boolean(config.settings.hide_uninstalled_in_generator);
      }
      if (config.presence) this.applyPresence(config.presence);
    } catch (e) {
      console.error("[app] Failed to load config:", e);
      this.toast("error", t("errors.network_error") + ": " + e.message);
    }
  },

  async refreshConfig() {
    // Re-fetch config from backend after a state change (e.g. engine download, model card edit).
    // Each UI step is isolated so one failing renderer can never leave the Hub showing stale data.
    let config;
    try {
      const res = await this._fetchWithRetry("/api/config", { cache: "no-store" }, 2);
      config = await res.json();
    } catch (e) {
      console.error("[app] Failed to refresh config:", e);
      return;
    }
    const step = (name, fn) => {
      try { fn(); } catch (e) { console.error(`[app] refreshConfig step '${name}' failed:`, e); }
    };
    this.state.config = config;
    this.state.account = config.account || { role: "host", id: "host", username: t("common.host"), is_guest: false };
    this.state.settings = config.settings;
    this.state.gpuInfo = config.gpu;
    this.state.engineStatus = config.engine_status;
    this.state.models = this.sortModelsList(config.models || []);
    // Shared-encoder list is derived from model cards; force a refetch on next Encoders view.
    this.state.encoders = null;
    window.QF_INITIAL_CONFIG = config;
    window.QF_SETTINGS = config.settings;
    step("paths", () => this.renderPaths(config));
    step("vae", () => { if (config.settings?.qwen_vae_type) this.state.selectedVae = config.settings.qwen_vae_type; });
    step("account", () => this.updateAccountRoleUI());
    step("gpu", () => this.renderGpuIndicator());
    step("stats", () => this.renderSystemStats(config.system_stats));
    step("models", () => this.renderModels());
    step("engine", () => this.updateEngineStatusUI());
    step("dock", () => this.updateDockPills());
    step("presence", () => { if (config.presence) this.applyPresence(config.presence); });
  },

  // ---------- Connected users / presence ----------
  async initPresence() {
    if (this.state.config?.presence) this.applyPresence(this.state.config.presence);
    this.heartbeatPresence();
    if (!this.state.presenceTimer) {
      this.state.presenceTimer = setInterval(() => this.heartbeatPresence(), 20000);
    }
    if (!this._presenceVisibilityBound) {
      this._presenceVisibilityBound = true;
      document.addEventListener("visibilitychange", () => {
        if (!document.hidden) this.heartbeatPresence();
      });
    }
  },

  async heartbeatPresence() {
    try {
      const data = await this.apiPost("/api/presence/heartbeat");
      this.applyPresence(data);
    } catch (e) {
      const indicator = document.getElementById("presence-indicator");
      if (indicator) {
        indicator.classList.add("presence-offline");
        indicator.title = t("presence.offline") || "Мониторинг подключений недоступен";
      }
    }
  },

  applyPresence(data) {
    this.state.presence = data || null;
    const indicator = document.getElementById("presence-indicator");
    const countEl = document.getElementById("presence-count");
    const enabled = Boolean(data && data.enabled !== false);
    if (indicator) {
      indicator.style.display = enabled ? "inline-flex" : "none";
      indicator.classList.toggle("presence-offline", false);
      indicator.classList.toggle("presence-many", enabled && Number(data?.count || 0) > 1);
      if (enabled) {
        const users = Array.isArray(data?.users) ? data.users : [];
        const names = users.map(u => u.username).filter(Boolean).slice(0, 8).join(", ");
        indicator.title = names
          ? `${t("presence.connected", { count: data.count })} — ${names}`
          : t("presence.connected", { count: data?.count || 0 });
      }
    }
    if (countEl) countEl.textContent = String(enabled ? (data?.count || 0) : 0);
    this.renderPresenceSettings(data);
  },

  renderPresenceSettings(data) {
    const checkbox = document.getElementById("setting-monitor-connections");
    if (checkbox) checkbox.checked = data ? data.enabled !== false : Boolean(this.state.settings?.monitor_connections);
    const countEl = document.getElementById("presence-settings-count");
    const listEl = document.getElementById("presence-users-list");
    if (!listEl) return;
    const enabled = Boolean(data && data.enabled !== false);
    if (!enabled) {
      if (countEl) countEl.textContent = "0";
      listEl.innerHTML = `<span class="presence-settings-empty">${escapeHtml(t("presence.disabled"))}</span>`;
      return;
    }
    const users = Array.isArray(data?.users) ? data.users : [];
    if (countEl) countEl.textContent = String(data?.count || users.length || 0);
    if (!users.length) {
      listEl.innerHTML = `<span class="presence-settings-empty">${escapeHtml(t("presence.none"))}</span>`;
      return;
    }
    listEl.innerHTML = users.map(user => {
      const role = user.is_guest ? t("common.guest") : t("common.host");
      const displayName = !user.is_guest && user.username === "Хост" ? t("common.host") : user.username;
      const roleBadge = role && role !== displayName ? ` <small>${escapeHtml(role)}</small>` : "";
      const seen = Number(user.last_seen_ago || 0);
      const seenText = seen < 5 ? t("presence.now") : t("presence.seconds_ago", { n: Math.round(seen) });
      return `<span class="presence-user-chip" title="${escapeHtml(seenText)}">${user.is_guest ? "👤" : "👑"} ${escapeHtml(displayName)}${roleBadge}</span>`;
    }).join("");
  },

  async onSettingMonitorConnectionsChange(checked) {
    try {
      await this.apiPatch("/api/settings", { monitor_connections: Boolean(checked) });
      if (this.state.settings) this.state.settings.monitor_connections = Boolean(checked);
      if (checked) {
        await this.heartbeatPresence();
      } else {
        this.applyPresence({ enabled: false, count: 0, users: [] });
      }
      this.toast("info", checked ? t("presence.enabled_toast") : t("presence.disabled_toast"));
    } catch (e) {
      this.toast("error", e.message || t("presence.save_error"));
    }
  },

  updateEngineStatusUI() {
    // Update Settings tab engine status display after download.
    const config = this.state.config || window.QF_INITIAL_CONFIG || {};
    const engines = config.engines || {};

    const qf17El = document.getElementById("setting-engine-qf17-status");
    if (qf17El) {
      const qf17 = engines.quantfunc_0_17 || {};
      const isInst = qf17.installed ?? (this.state.engineStatus?.status === "ok");
      qf17El.innerHTML = isInst
        ? `<span class="state-badge installed">✓ ${escapeHtml(t("common.installed"))} (v0.0.17)</span>`
        : `<span class="state-badge missing">✗ ${escapeHtml(t("common.not_installed"))}</span>`;
    }

    const qf12El = document.getElementById("setting-engine-qf12-status");
    if (qf12El) {
      const qf12 = engines.quantfunc_0_12 || {};
      const isInst = qf12.installed ?? (config.installed_engines?.includes("0.0.12"));
      qf12El.innerHTML = isInst
        ? `<span class="state-badge installed">✓ ${escapeHtml(t("common.installed"))} (v0.0.12)</span>`
        : `<span class="state-badge missing">✗ ${escapeHtml(t("common.not_installed"))}</span>`;
    }

    const sdCppEl = document.getElementById("setting-engine-sdcpp-status");
    if (sdCppEl) {
      const sd = engines.sd_cpp || {};
      const isInst = sd.installed ?? config.sd_cpp_installed;
      const ver = sd.version || config.sd_cpp_version || 'ready';
      sdCppEl.innerHTML = isInst
        ? `<span class="state-badge installed">✓ ${escapeHtml(t("common.installed"))} (${escapeHtml(ver)})</span>`
        : `<span class="state-badge missing">✗ ${escapeHtml(t("common.not_installed"))}</span>`;
    }

    const sdnqEl = document.getElementById("setting-engine-sdnq-status");
    if (sdnqEl) {
      const sdnq = engines.sdnq || {};
      const isInst = sdnq.installed ?? config.sdnq_installed;
      const ver = sdnq.version || config.sdnq_version || 'ready';
      sdnqEl.innerHTML = isInst
        ? `<span class="state-badge installed">✓ ${escapeHtml(t("common.installed"))} (${escapeHtml(ver)})</span>`
        : `<span class="state-badge missing">✗ ${escapeHtml(t("common.not_installed"))}</span>`;
    }

    const qfToggleEl = document.getElementById("setting-enable-quantfunc");
    const qfBadgeEl = document.getElementById("setting-qf-toggle-badge");
    const qfEnabled = !!(this.state.settings?.enable_quantfunc_models);
    if (qfToggleEl) {
      qfToggleEl.checked = qfEnabled;
    }
    if (qfBadgeEl) {
      qfBadgeEl.innerHTML = qfEnabled
        ? `<span class="state-badge installed">✓ ${escapeHtml(t("common.enabled"))}</span>`
        : `<span class="state-badge missing">${escapeHtml(t("common.disabled"))}</span>`;
    }
  },

  // ---------- Renderers ----------
  renderGpuIndicator() {
    const nameEl = document.getElementById("gpu-name");
    const dotEl = document.getElementById("gpu-status");
    if (!nameEl || !dotEl) return;

    if (this.state.gpuInfo && this.state.gpuInfo.name) {
      const gpu = this.state.gpuInfo;
      const shortName = String(gpu.name).replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
      nameEl.textContent = `${shortName} · ${Math.round(gpu.vram_mb / 1024)} GB`;
      dotEl.classList.remove("offline");
      dotEl.title = `${gpu.tier} · SM ${gpu.sm} · Driver ${gpu.driver_version}`;
    } else {
      nameEl.textContent = t("gpu.not_detected");
      dotEl.classList.add("offline");
    }
  },

  renderPaths(config) {
    const st = config?.settings || {};
    const modelsEl = document.getElementById("setting-models-dir");
    const outputsEl = document.getElementById("setting-output-dir");
    if (modelsEl && st.models_dir) modelsEl.value = st.models_dir;
    if (outputsEl && st.output_dir) outputsEl.value = st.output_dir;
  },

  async savePaths() {
    const models = (document.getElementById("setting-models-dir")?.value || "").trim();
    const outputs = (document.getElementById("setting-output-dir")?.value || "").trim();
    if (!models || !outputs) {
      this.toast("warning", t("settings.paths.save_failed") + t("controls.custom_path"));
      return;
    }
    try {
      await this.apiPatch("/api/settings", { models_dir: models, output_dir: outputs });
      this.toast("success", t("settings.paths.saved"));
      await this.refreshConfig();
    } catch (e) {
      this.toast("error", t("settings.paths.save_failed") + e.message);
    }
  },

  renderSystemStats(stats) {
    if (!stats) return;
    const cpuEl = document.getElementById("cpu-stat-text");
    const gpuStatEl = document.getElementById("gpu-stat-text");
    const gpuSepEl = document.getElementById("gpu-stat-sep");
    const vramEl = document.getElementById("vram-stat-text");
    const ramEl = document.getElementById("ram-stat-text");

    if (cpuEl && stats.cpu_percent !== undefined) {
      const cpuPct = Math.round(stats.cpu_percent);
      const color = cpuPct > 90 ? "#f87171" : (cpuPct > 70 ? "#fbbf24" : "var(--text-primary)");
      const cores = stats.cpu_cores ? ` (${stats.cpu_cores}C)` : "";
      cpuEl.innerHTML = `CPU: <strong style="color:${color}">${cpuPct}%</strong>${cores}`;
    }

    if (gpuStatEl) {
      const parts = [];
      if (stats.gpu_temp_c != null) parts.push(`${stats.gpu_temp_c}°C`);
      if (stats.gpu_power_w != null) parts.push(`${stats.gpu_power_w}W`);
      if (stats.gpu_util != null) parts.push(`${stats.gpu_util}%`);
      if (parts.length > 0) {
        gpuStatEl.innerHTML = `GPU: <strong>${parts.join(" · ")}</strong>`;
        gpuStatEl.style.display = "";
        if (gpuSepEl) gpuSepEl.style.display = "";
      } else {
        gpuStatEl.style.display = "none";
        if (gpuSepEl) gpuSepEl.style.display = "none";
      }
    }

    if (vramEl) {
      if (stats.vram_total_gb > 0) {
        const vramPct = stats.vram_percent || 0;
        const color = vramPct > 90 ? "#f87171" : (vramPct > 75 ? "#fbbf24" : "var(--text-primary)");
        vramEl.innerHTML = `VRAM: <strong style="color:${color}">${stats.vram_used_gb.toFixed(1)}/${stats.vram_total_gb.toFixed(1)} GB</strong> (${Math.round(vramPct)}%)`;
      } else {
        vramEl.textContent = "VRAM: —";
      }
    }

    if (ramEl) {
      if (stats.ram_total_gb > 0) {
        const ramPct = stats.ram_percent || 0;
        const color = ramPct > 90 ? "#f87171" : (ramPct > 80 ? "#fbbf24" : "var(--text-primary)");
        ramEl.innerHTML = `RAM: <strong style="color:${color}">${stats.ram_used_gb.toFixed(1)}/${stats.ram_total_gb.toFixed(1)} GB</strong> (${Math.round(ramPct)}%)`;
      } else {
        ramEl.textContent = "RAM: —";
      }
    }
  },

  startStatsPolling() {
    if (this._statsInterval) clearInterval(this._statsInterval);
    this._statsInterval = setInterval(async () => {
      if (document.hidden) return;
      try {
        const stats = await this.apiGet("/api/system/stats");
        this.renderSystemStats(stats);
      } catch (_) {}
    }, 2000);
  },

  renderModels() {
    const select = document.getElementById("model-select");
    if (!select) return;

    this.state.models = this.sortModelsList(this.state.models || []);

    select.innerHTML = "";
    if (!this.state.models.length) {
      select.innerHTML = `<option value="">${t("common.loading")}</option>`;
      return;
    }

    // Filter models compatible with current GPU
    const gpuSm = this.state.gpuInfo?.sm || 0;
    let compatible = this.state.models.filter(m =>
      !m.hidden_from_host && m.requirements.gpu_min_sm <= gpuSm
    );
    if (this.state.settings?.hide_uninstalled_in_generator) {
      const installedOnly = compatible.filter(m => m.installed);
      if (installedOnly.length > 0) {
        compatible = installedOnly;
      }
    }

    // Graceful fallback: if GPU SM is not detected or no models match, show all models
    if (!compatible.length) {
      compatible = this.state.models;
    }

    compatible = this.sortModelsList(compatible);

    if (!compatible.length) {
      select.innerHTML = `<option value="">No models available</option>`;
      return;
    }

    for (const model of compatible) {
      const opt = document.createElement("option");
      opt.value = model.id;
      if (model.disabled) {
        opt.disabled = true;
        opt.textContent = `⛔ ${model.name} (${model.family}) [Отключено]`;
      } else {
        const instMark = model.installed ? "✓ " : "⬇ ";
        opt.textContent = `${instMark}${model.name} (${model.family})`;
      }
      select.appendChild(opt);
    }

    // Auto-select first non-disabled model (or preserve existing selection)
    const existing = this.state.selectedModel && compatible.find(m => m.id === this.state.selectedModel && !m.disabled);
    const firstActive = existing || compatible.find(m => !m.disabled) || compatible[0];
    if (firstActive) {
      select.value = firstActive.id;
      this.state.selectedModel = firstActive.id;
      this.onModelChange();
    }
    this.updateModelTrigger();
    this.renderModelsTab();
  },

  updateModelTrigger() {
    const nameEl = document.getElementById("model-trigger-name");
    const subEl = document.getElementById("model-trigger-sub");
    const badgeEl = document.getElementById("model-trigger-badge");

    const current = this.state.models.find(m => m.id === this.state.selectedModel);
    if (!current) {
      if (nameEl) nameEl.textContent = "—";
      if (subEl) subEl.textContent = "—";
      this.updateDockPills();
      return;
    }

    if (nameEl) nameEl.textContent = current.name;
    const variant = this.state.selectedVariant || current.transformers?.[0]?.id || "default";
    let engineVer = "";
    if (current.engine === "sd_cpp") {
      engineVer = "SD.CPP";
    } else if (current.engine === "sdnq") {
      engineVer = "SDNQ";
    } else if (current.engine_version) {
      engineVer = `v${current.engine_version}`;
    }
    const steps = document.getElementById("steps-input")?.value || current.transformers?.[0]?.steps || 4;
    
    if (subEl) {
      if (current.disabled) {
        subEl.innerHTML = `<span style="color: #f87171;">⚠️ Отключено (веса с дефектами)</span>`;
      } else {
        const parts = [];
        if (engineVer) parts.push(engineVer);
        parts.push(`${steps} steps`);
        if (variant) parts.push(variant.replace(/-(?:50x|40x|30x-below)(?:-base)?/g, ""));
        subEl.textContent = parts.join(" · ");
      }
    }

    if (badgeEl) {
      if (current.disabled) {
        badgeEl.textContent = "⛔ Отключено";
      } else if (current.engine === "sd_cpp") {
        badgeEl.textContent = "⚡ SD.CPP";
      } else if (current.engine === "sdnq") {
        badgeEl.textContent = "⚡ SDNQ";
      } else {
        badgeEl.textContent = current.engine_version ? `⚡ v${current.engine_version}` : "⚡";
      }
    }

    // Dynamic CFG & Noise recommendations
    let recCfg = 1.0;
    if (current.default_cfg !== undefined && current.default_cfg !== null) {
      recCfg = Number(current.default_cfg);
    } else if (current.id.includes("turbo") || current.engine === "sdnq") {
      recCfg = 1.0;
    } else if (current.id.startsWith("qwen_image_2_1") || current.family === "qwen") {
      recCfg = 3.0;
    } else if (current.family === "sdxl" || current.id.includes("sdxl")) {
      recCfg = 7.0;
    } else if (current.family === "flux" && current.id.includes("dev")) {
      recCfg = 3.5;
    } else {
      recCfg = 1.0;
    }

    const cfgBadge = document.getElementById("cfg-rec-badge");
    if (cfgBadge) {
      const recLabel = t("controls.rec") || "Реком.";
      cfgBadge.textContent = `(${recLabel}: ${recCfg.toFixed(1)})`;
    }

    const noiseBadge = document.getElementById("noise-rec-badge");
    if (noiseBadge) {
      const recLabel = t("controls.rec") || "Реком.";
      noiseBadge.textContent = `(${recLabel}: 0.0)`;
    }

    this.updateDockPills();
  },

  // ---------- Model Hub Modal ----------
  openModelPickerModal() {
    const modal = document.getElementById("model-picker-modal");
    if (!modal) return;
    this.state.isModelPickerOpen = true;
    modal.style.display = "flex";
    this.renderModelPickerModal();
    const searchInput = document.getElementById("model-picker-search");
    if (searchInput) {
      searchInput.value = this.state.modelPickerSearch || "";
      searchInput.focus();
    }
  },

  closeModelPickerModal() {
    const modal = document.getElementById("model-picker-modal");
    if (!modal) return;
    this.state.isModelPickerOpen = false;
    modal.style.display = "none";
  },

  onModelPickerSearch(query) {
    this.state.modelPickerSearch = (query || "").toLowerCase().trim();
    this.renderModelPickerModal();
  },

  setModelPickerFilter(filter, btnEl) {
    this.state.modelPickerFilter = filter;
    document.querySelectorAll("#model-picker-filters .filter-chip").forEach(el => {
      el.classList.toggle("active", el === btnEl);
    });
    this.renderModelPickerModal();
  },

  renderModelPickerModal() {
    const grid = document.getElementById("model-picker-grid");
    if (!grid) return;

    if (!this.state.models || !this.state.models.length) {
      grid.innerHTML = `<div class="placeholder-text">${t("common.loading")}</div>`;
      return;
    }

    const query = this.state.modelPickerSearch || "";
    const filter = this.state.modelPickerFilter || "all";

    const filtered = this.sortModelsList(this.state.models).filter(m => {
      // Search query
      if (query) {
        const matchesName = m.name?.toLowerCase().includes(query);
        const matchesFamily = m.family?.toLowerCase().includes(query);
        const matchesDesc = m.description?.toLowerCase().includes(query);
        if (!matchesName && !matchesFamily && !matchesDesc) return false;
      }
      // Visibility filter
      if (filter === "hidden") {
        return m.hidden_from_host === true || m.hidden_for_guests === true;
      } else if (m.hidden_from_host === true) {
        return false;
      }

      // Category filter
      if (filter === "installed") {
        return m.installed === true;
      } else if (filter === "custom") {
        return m.is_custom === true;
      } else if (filter === "klein") {
        return m.family === "klein" || m.id.includes("klein");
      } else if (filter === "qwen") {
        return m.family === "qwen" || m.id.includes("qwen");
      } else if (filter === "z-image") {
        return m.family === "z-image" || m.id.includes("z_image");
      } else if (filter === "sd_cpp") {
        return m.engine === "sd_cpp" || m.id.includes("gguf");
      } else if (filter === "sdnq") {
        return m.engine === "sdnq" || m.id.includes("sdnq");
      }
      return true;
    });

    if (!filtered.length) {
      grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px 0;">${t("models.no_results")}</div>`;
      return;
    }

    grid.innerHTML = "";
    for (const model of filtered) {
      const isSelected = model.id === this.state.selectedModel;
      const isInstalled = model.installed === true;
      const isDisabled = model.disabled === true;

      const card = document.createElement("div");
      card.className = `model-picker-card ${isSelected ? "active" : ""} ${isDisabled ? "model-card-qf-disabled" : ""}`;

      let engineBadge = "";
      if (model.is_custom) {
        engineBadge = `<span class="badge-custom">🎨 Custom</span>`;
      } else if (model.engine === "sd_cpp") {
        engineBadge = `<span class="badge-engine sd-cpp" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">⚡ Engine: SD.CPP</span>`;
      } else if (model.engine === "sdnq") {
        engineBadge = `<span class="badge-engine sdnq" style="background: rgba(139, 92, 246, 0.18); color: #c4b5fd; border-color: rgba(139, 92, 246, 0.35);">⚡ Engine: SDNQ</span>`;
      } else if (model.engine_version) {
        engineBadge = `<span class="badge-engine ${model.engine_version === '0.0.12' ? 'v12' : ''}">⚡ Engine v${escapeHtml(model.engine_version)}</span>`;
      }

      let statusBadge = "";
      if (isDisabled) {
        statusBadge = `<span class="badge-unsupported">${escapeHtml(t("models.unsupported"))}</span>`;
      } else if (isInstalled) {
        statusBadge = `<span class="badge-installed">✓ ${t("models.status_ready")}</span>`;
      } else {
        statusBadge = `<span class="badge-missing">⬇ ${t("models.status_download")}</span>`;
      }

      const keyBadge = model.key_protected
        ? `<span class="model-badge key-protected" style="font-size: 10px; padding: 2px 6px;">🔑 Key Protected</span>`
        : "";

      let visBadges = "";
      if (model.hidden_from_host) {
        visBadges += `<span class="model-badge" style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4);" title="${escapeHtml(t("models.vis_hidden"))}">🙈 ${escapeHtml(t("models.vis_hidden"))}</span>`;
      }
      if (model.hidden_for_guests) {
        visBadges += `<span class="model-badge" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4);" title="${escapeHtml(t("models.vis_host_only"))}">🔒 ${escapeHtml(t("models.vis_host_only"))}</span>`;
      }

      const vramMin = model.requirements?.vram_min_gb ? `${model.requirements.vram_min_gb} GB` : "8 GB";
      const defaultSteps = model.transformers?.[0]?.steps || 4;

      let actionBtnHtml = "";
      let deleteBtnHtml = "";
      let visBtnHtml = "";
      const canManageModels = !this.state.account?.is_guest;
      if (canManageModels) {
        visBtnHtml = `<button class="card-action-btn secondary" style="width: auto; padding: 0 10px; font-size: 11px; height: 32px;" onclick="App.openModelVisibilityModal('${escapeHtml(model.id)}', event)" title="${escapeHtml(t("models.visibility"))}">👁️ ${escapeHtml(t("models.visibility"))}</button>`;
      }
      if (canManageModels) {
        deleteBtnHtml = `<button class="delete-custom-btn" onclick="App.deleteCustomModel('${escapeHtml(model.id)}', event)" title="${escapeHtml(t("models.delete_hint") || "Удалить карточку и скачанные веса")}">🗑 ${escapeHtml(t("common.delete") || "Удалить")}</button>`;
      }

      if (isDisabled) {
        actionBtnHtml = `<button class="card-action-btn disabled" disabled title="${escapeHtml(localizedDisabledReason(model.disabled_reason, "models.disabled_title"))}">${escapeHtml(t("models.disabled_btn"))}</button>`;
      } else if (isSelected) {
        actionBtnHtml = `<button class="card-action-btn active" disabled>✓ ${t("models.active_model")}</button>`;
      } else if (isInstalled) {
        actionBtnHtml = `<button class="card-action-btn primary" onclick="App.selectModelFromModal('${escapeHtml(model.id)}')">${t("models.select_model")}</button>`;
      } else {
        actionBtnHtml = `<button class="card-action-btn download" onclick="App.downloadModelVariant('${escapeHtml(model.id)}', null, this)">⬇ ${t("models.download_and_select")}</button>`;
      }

      const varCount = model.transformers?.length || 1;
      const teCount = model.text_encoders?.length || 0;
      const varInfo = varCount > 1 ? `<span>⚡ ${varCount} ${t("models.variants_count") || "variants"}</span>` : "";
      const teInfo = teCount > 0 ? `<span>📝 ${teCount} ${t("models.encoders_count") || "encoders"}</span>` : "";

      const noticeHtml = isDisabled
        ? `<div style="margin-top: 8px; padding: 6px 10px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; font-size: 11px; color: #f87171;">⚠️ ${escapeHtml(localizedDisabledReason(model.disabled_reason))}</div>`
        : "";

      card.innerHTML = `
        <div class="card-top">
          <div class="card-header-row">
            <div class="card-title-group">
              <span class="card-model-name">${escapeHtml(model.name)}</span>
              <span class="card-family-tag">${escapeHtml(model.family || model.pipeline)}</span>
            </div>
            <div class="card-badges">
              ${engineBadge}
              ${statusBadge}
              ${keyBadge}
              ${visBadges}
            </div>
          </div>
          <p class="card-desc">${escapeHtml(model.description || "")}</p>
          ${noticeHtml}
          <div class="card-specs">
            <span>📼 ${t("models.vram_min")}: ${vramMin}</span>
            <span>⏱ ${t("models.steps")}: ${defaultSteps}</span>
            <span>💻 Min SM: ${model.requirements?.gpu_min_sm || 75}+</span>
            ${varInfo}
            ${teInfo}
          </div>
        </div>
        <div class="card-footer" style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
          ${actionBtnHtml}
          ${visBtnHtml}
          ${deleteBtnHtml}
        </div>
      `;

      // Allow clicking card to select if installed and not disabled
      if (!isSelected && isInstalled && !isDisabled) {
        card.style.cursor = "pointer";
        card.addEventListener("click", (e) => {
          if (e.target.tagName === "BUTTON") return;
          this.selectModelFromModal(model.id);
        });
      } else if (isDisabled) {
        card.style.cursor = "not-allowed";
        card.addEventListener("click", (e) => {
          if (e.target.tagName === "BUTTON") return;
          this.toast("warning", localizedDisabledReason(model.disabled_reason));
        });
      }

      grid.appendChild(card);
    }
  },

  async selectModelFromModal(modelId) {
    const model = this.state.models.find(m => m.id === modelId);
    if (model && model.disabled) {
      this.toast("warning", localizedDisabledReason(model.disabled_reason));
      return;
    }

    this.state.selectedModel = modelId;
    try { localStorage.setItem("lazysd_selected_model", modelId); } catch (_) {}

    const select = document.getElementById("model-select");
    if (select) {
      select.value = modelId;
    }

    if (this.state.currentTab !== "generate" && this.state.currentTab !== "inpaint") {
      this.switchTab("generate");
    }

    await this.onModelChange();
    this.updateModelTrigger();
    this.closeModelPickerModal();
    this.toast("info", `${t("models.active_model")}: ${modelId}`);
  },

  async renderModelsTab() {
    const listEl = document.getElementById("models-list");
    if (!listEl) return;

    if (!this.state.models.length) {
      listEl.innerHTML = `<div class="placeholder-text">${t("common.loading")}</div>`;
      return;
    }

    const filter = this.state.modelsTabFilter || "all";
    if (filter === "encoders") {
      await this.renderEncodersTab(listEl);
      return;
    }

    const query = (this.state.modelsTabSearch || "").trim().toLowerCase();
    const filtered = this.sortModelsList(this.state.models).filter(model => {
      if (query) {
        const haystack = [
          model.id,
          model.name,
          model.family,
          model.engine,
          model.description,
          ...(model.text_encoders || []).map(te => `${te.id || ""} ${te.name || ""} ${te.file || ""}`),
          ...(model.transformers || []).map(varItem => varItem.id || ""),
        ].filter(Boolean).join(" ").toLowerCase();
        if (!haystack.includes(query)) return false;
      }

      if (filter === "installed") return model.installed === true;
      if (filter === "custom") return model.is_custom === true;
      if (filter === "hidden") return model.hidden_from_host === true || model.hidden_for_guests === true;
      if (filter === "svdq") {
        return model.engine === "nunchaku"
          || (model.tags && model.tags.includes("svdq"))
          || (model.family && model.family.includes("svdq"))
          || (model.transformers && model.transformers.some(item => item.backend === "nunchaku" || item.id.includes("svdq")));
      }
      if (filter === "sd_cpp") return model.engine === "sd_cpp" || model.id.includes("gguf");
      if (filter === "sdnq") return model.engine === "sdnq" || model.id.includes("sdnq");
      if (filter === "klein") {
        return model.family === "klein" || model.family === "flux"
          || model.id.includes("flux") || model.id.includes("klein");
      }
      if (filter === "qwen") return model.family === "qwen" || model.id.includes("qwen");
      if (filter === "z-image") return model.family === "z-image" || model.id.includes("z_image");
      return true;
    });

    if (!filtered.length) {
      listEl.innerHTML = `<div class="placeholder-text" style="padding: 40px 0;">${t("models.no_results") || "Модели не найдены"}</div>`;
      return;
    }

    listEl.innerHTML = "";
    const grid = document.createElement("div");
    grid.className = "models-grid";

    for (const model of filtered) {
      const isDisabled = model.disabled === true;
      const card = document.createElement("div");
      card.className = `model-card ${isDisabled ? "model-card-qf-disabled" : ""}`;

      const keyBadge = model.key_protected
        ? `<span class="model-badge key-protected">🔑 Key Protected</span>`
        : `<span class="model-badge free">🔓 Free Access</span>`;
      
      const pipelineBadge = `<span class="model-badge pipeline">${escapeHtml(model.pipeline)}</span>`;
      const familyBadge = `<span class="model-badge family">${escapeHtml(model.family)}</span>`;
      let engineBadge = "";
      if (model.is_custom) {
        engineBadge = `<span class="badge-custom">🎨 Custom</span>`;
      } else if (model.engine === "sd_cpp") {
        engineBadge = `<span class="badge-engine sd-cpp" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">⚡ SD.CPP</span>`;
      } else if (model.engine === "sdnq") {
        engineBadge = `<span class="badge-engine sdnq" style="background: rgba(139, 92, 246, 0.18); color: #c4b5fd; border-color: rgba(139, 92, 246, 0.35);">⚡ SDNQ</span>`;
      } else if (model.engine_version) {
        engineBadge = `<span class="badge-engine ${model.engine_version === '0.0.12' ? 'v12' : ''}">⚡ v${escapeHtml(model.engine_version)}</span>`;
      }

      let installedBadge = "";
      if (isDisabled) {
        installedBadge = `<span class="badge-unsupported">⚠️ ${escapeHtml(t("models.unsupported"))}</span>`;
      } else if (model.installed) {
        installedBadge = `<span class="badge-installed">✓ ${t("models.status_ready")}</span>`;
      } else {
        installedBadge = `<span class="badge-missing">⬇ ${t("models.status_download")}</span>`;
      }

      const statusBadge = model.available === false || model.status === "coming_soon"
        ? `<span class="model-badge key-protected" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; border-color: rgba(245, 158, 11, 0.4);">⏳ Coming Soon</span>`
        : "";

      const vramStr = model.requirements?.vram_min_gb ? `${model.requirements.vram_min_gb} GB VRAM min` : "";

      let variantsHtml = "";
      if (model.transformers && model.transformers.length) {
        for (const v of model.transformers) {
          let variantAction = "";
          if (isDisabled) {
            variantAction = `<button type="button" class="btn-download" disabled title="${escapeHtml(localizedDisabledReason(model.disabled_reason, "common.disabled"))}">⛔ <span>${escapeHtml(t("common.disabled"))}</span></button>`;
          } else if (v.installed) {
            variantAction = `
              <div class="variant-actions">
                <span class="variant-state installed" title="${escapeHtml(t("models.component_on_disk"))}">${ICON_CHECK}<span>${escapeHtml(t("models.component_on_disk"))}</span></span>
                <button type="button" class="btn-delete-files" data-model-id="${escapeHtml(model.id)}" data-variant-id="${escapeHtml(v.id)}" onclick="App.deleteModelFiles(this.dataset.modelId, this.dataset.variantId, this)" title="${escapeHtml(t("models.component_delete"))}" aria-label="${escapeHtml(t("models.component_delete"))} ${escapeHtml(v.id)}">${ICON_TRASH}<span>${escapeHtml(t("models.component_delete"))}</span></button>
              </div>`;
          } else {
            variantAction = `<button type="button" class="btn-download" data-model-id="${escapeHtml(model.id)}" data-variant-id="${escapeHtml(v.id)}" onclick="App.downloadModelVariant(this.dataset.modelId, this.dataset.variantId, this)" title="${escapeHtml(t("models.component_download"))}" aria-label="${escapeHtml(t("models.component_download"))} ${escapeHtml(v.id)}">${ICON_DOWNLOAD}<span>${escapeHtml(t("models.component_download"))}</span></button>`;
          }
          variantsHtml += `
            <div class="variant-item">
              <div class="variant-info">
                <span class="variant-name">${escapeHtml(v.id)}</span>
                <span class="variant-sub">${escapeHtml(v.backend)} · ${v.steps} steps · ${(v.size_gb || 0).toFixed(1)} GB</span>
              </div>
              ${variantAction}
            </div>
          `;
        }
      } else {
        variantsHtml = `<div class="placeholder-text" style="padding: 8px 0; font-size: 13px; text-align: left;">Weights pending release on ModelScope</div>`;
      }

      const isSelected = model.id === this.state.selectedModel;
      let selectBtnHtml = "";
      if (isDisabled) {
        selectBtnHtml = `<button class="card-action-btn disabled" style="margin-top: 12px; width: auto;" disabled title="${escapeHtml(localizedDisabledReason(model.disabled_reason, "common.disabled"))}">⛔ ${escapeHtml(t("common.disabled"))}</button>`;
      } else if (isSelected) {
        selectBtnHtml = `<button class="card-action-btn active" style="margin-top: 12px; width: auto;" disabled>✓ ${t("models.active_model")}</button>`;
      } else {
        selectBtnHtml = `<button class="card-action-btn primary" style="margin-top: 12px; width: auto;" onclick="App.selectModelFromModal('${escapeHtml(model.id)}')">⚡ ${t("models.select_model")}</button>`;
      }

      const canEditCard = !this.state.account?.is_guest;
      const deleteModelBtn = canEditCard
        ? `<button class="delete-custom-btn" style="margin-top: 12px; margin-left: 8px;" onclick="App.deleteCustomModel('${escapeHtml(model.id)}', event)" title="${escapeHtml(t('models.delete_hint') || 'Удалить карточку и скачанные веса')}">🗑 ${escapeHtml(t("common.delete") || "Удалить")}</button>`
        : "";

      const editCardBtn = canEditCard
        ? `<button class="history-action-btn" style="width: auto; margin-top: 12px; margin-left: 8px;" onclick="App.openModelCardEditor('${escapeHtml(model.id)}')" title="${escapeHtml(t("models.edit_card_title"))}">✎ ${escapeHtml(t("models.edit_card"))}</button>`
        : "";

      const roleLabelFor = role => role === "text_encoder"
        ? t("models.component_role_te")
        : role === "vae" ? t("models.component_role_vae") : t("models.component_role_aux");
      const componentAction = (role, index, teId, installed, name) => {
        const actionLabel = name || roleLabelFor(role);
        return installed
          ? `<div class="variant-actions">
             <span class="variant-state installed" title="${escapeHtml(t("models.component_on_disk"))}">${ICON_CHECK}<span>${escapeHtml(t("models.component_on_disk"))}</span></span>
             <button type="button" class="btn-delete-files" onclick="App.deleteModelComponent('${escapeHtml(model.id)}', '${role}', ${index}, '${escapeHtml(teId || "")}', '${escapeHtml(actionLabel)}', this)" title="${escapeHtml(t("models.component_delete"))} ${escapeHtml(actionLabel)}" aria-label="${escapeHtml(t("models.component_delete"))} ${escapeHtml(actionLabel)}">${ICON_TRASH}<span>${escapeHtml(t("models.component_delete"))}</span></button>
           </div>`
          : `<button type="button" class="btn-download" data-model-id="${escapeHtml(model.id)}" data-role="${role}" data-index="${index}" data-te-id="${escapeHtml(teId || "")}" onclick="App.downloadModelComponent(this.dataset.modelId, this.dataset.role, this.dataset.index, this.dataset.teId, this)" title="${escapeHtml(t("models.component_download"))} ${escapeHtml(actionLabel)}" aria-label="${escapeHtml(t("models.component_download"))} ${escapeHtml(actionLabel)}">${ICON_DOWNLOAD}<span>${escapeHtml(t("models.component_download"))}</span></button>`;
      };

      const teList = Array.isArray(model.text_encoders) ? model.text_encoders : [];
      const teMissing = teList.length > 0 && !teList.some(te => te.installed);
      const teRows = teList.map((te, idx) => {
        const tech = [te.repo_id, te.path || te.file, te.precision, te.size_gb ? `${te.size_gb} GB` : ""]
          .filter(Boolean).join(" · ");
        return `
          <div class="component-item">
            <div class="component-info">
              <span class="component-name">${escapeHtml(te.name || te.file || te.id || t("controls.text_encoder"))}</span>
              <span class="component-sub" title="${escapeHtml(tech)}">${escapeHtml(tech)}</span>
            </div>
            <div class="component-actions">${componentAction("text_encoder", idx, te.id, te.installed, te.name || te.file || te.path || te.id)}</div>
          </div>`;
      }).join("");
      const teSectionTitle = t("models.te_section", { count: teList.length }) +
        (teMissing ? ` <span class="section-warning">${escapeHtml(t("models.te_missing"))}</span>` : "");
      const teSection = (teList.length || model.is_custom)
        ? `<details class="model-card-section">
             <summary class="model-card-section-title">${teSectionTitle}</summary>
             <div class="model-variants-list">${teRows || `<div class="component-empty">${escapeHtml(t("models.te_empty"))}</div>`}</div>
           </details>`
        : "";

      const vaeList = (Array.isArray(model.vae_files) && model.vae_files.length)
        ? model.vae_files
        : (model.vae ? [model.vae] : []);
      const vaeMissing = vaeList.length > 0 && !vaeList.some(vae => vae.installed);
      const vaeRows = vaeList.map((vae, idx) => {
        const tech = [vae.repo_id, vae.path].filter(Boolean).join(" · ");
        const fileName = String(vae.path || "").split("/").pop() || "VAE";
        const actions = (vae.installed && (vae.inferred || vae.shared))
          ? `<div class="variant-actions"><span class="variant-state installed" title="${escapeHtml(t("models.component_on_disk"))}">${ICON_CHECK}<span>${escapeHtml(t("models.component_on_disk"))}${vae.shared ? ` (${escapeHtml(vae.shared_from || t("common.shared"))})` : ""}</span></span></div>`
          : componentAction("vae", idx, "", vae.installed, "");
        return `
          <div class="component-item">
            <div class="component-info">
              <span class="component-name">${escapeHtml(idx === 0 ? "VAE" : fileName)}</span>
              <span class="component-sub" title="${escapeHtml(tech)}">${escapeHtml(tech)}</span>
            </div>
            <div class="component-actions">${actions}</div>
          </div>`;
      }).join("");
      const vaeRow = vaeList.length
        ? vaeRows
        : `<div class="component-empty">${escapeHtml(t("models.vae_empty"))}${canEditCard ? ` ${escapeHtml(t("models.edit_card"))}.` : ""}</div>`;
      const vaeSectionTitle = t("models.vae_section") +
        (vaeList.length > 1 ? ` (${vaeList.length})` : "") +
        (vaeMissing ? ` <span class="section-warning">${escapeHtml(t("models.te_missing"))}</span>` : "");
      const vaeSection = `<details class="model-card-section">
             <summary class="model-card-section-title">${vaeSectionTitle}</summary>
             <div class="model-card-section-hint">${escapeHtml(t("models.card_editor.vae_priority_hint"))}</div>
             <div class="model-variants-list">${vaeRow}</div>
           </details>`;

      const auxList = Array.isArray(model.aux_files) ? model.aux_files : [];
      const auxRows = auxList.map((aux, idx) => {
        const tech = [aux.repo_id, aux.path, aux.size_gb ? `${aux.size_gb} GB` : ""].filter(Boolean).join(" · ");
        return `
          <div class="component-item">
            <div class="component-info">
              <span class="component-name">${escapeHtml(String(aux.path || "").split("/").pop() || t("models.component_role_aux"))}</span>
              <span class="component-sub" title="${escapeHtml(tech)}">${escapeHtml(tech)}</span>
            </div>
            <div class="component-actions">${componentAction("aux", idx, "", aux.installed, String(aux.path || "").split("/").pop())}</div>
          </div>`;
      }).join("");
      const auxSection = auxList.length
        ? `<details class="model-card-section">
             <summary class="model-card-section-title">${escapeHtml(t("models.aux_section", { count: auxList.length }))}</summary>
             <div class="model-variants-list">${auxRows}</div>
           </details>`
        : "";

      const noticeHtml = isDisabled
        ? `<div style="margin-top: 10px; padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; font-size: 12px; color: #f87171;">⚠️ ${escapeHtml(localizedDisabledReason(model.disabled_reason, "models.disabled_upstream_hint"))}</div>`
        : "";

      card.innerHTML = `
        <div>
          <div class="model-card-header">
            <span class="model-card-title">${escapeHtml(model.name)}</span>
            <div class="model-card-badges">
              ${engineBadge}
              ${installedBadge}
              ${statusBadge}
              ${keyBadge}
              ${pipelineBadge}
              ${familyBadge}
            </div>
          </div>
          <p class="model-card-description" style="margin-top: 8px;">${escapeHtml(model.description || "")}</p>
          ${noticeHtml}
          <div class="model-card-reqs" style="margin-top: 10px;">
            <span>💻 Min SM: ${model.requirements?.gpu_min_sm || 75}+</span>
            ${vramStr ? `<span>📼 ${vramStr}</span>` : ""}
          </div>
          <details class="model-card-section">
            <summary class="model-card-section-title">Transformers (${model.transformers?.length || 0})</summary>
            <div class="model-variants-list">${variantsHtml}</div>
          </details>
          ${teSection}
          ${vaeSection}
          ${auxSection}
          <div style="margin-top: 8px; display: flex; align-items: center; flex-wrap: wrap; gap: 4px;">
            ${selectBtnHtml}
            ${editCardBtn}
            ${deleteModelBtn}
          </div>
        </div>
      `;

      grid.appendChild(card);
    }

    listEl.appendChild(grid);
  },

  async renderEncodersTab(listEl) {
    if (!listEl) return;
    listEl.innerHTML = `<div class="placeholder-text">${escapeHtml(t("models.encoders_loading"))}</div>`;

    try {
      if (!Array.isArray(this.state.encoders)) {
        this.state.encoders = await this.apiGet("/api/encoders");
      }

      const query = (this.state.modelsTabSearch || "").trim().toLowerCase();
      const encoders = (this.state.encoders || []).filter(enc => {
        if (!query) return true;
        return [enc.name, enc.file, enc.repo_id, enc.precision]
          .filter(Boolean)
          .some(value => String(value).toLowerCase().includes(query));
      });

      if (!encoders.length) {
        listEl.innerHTML = `<div class="placeholder-text" style="padding: 40px 0;">${escapeHtml(t("models.encoders_empty"))}</div>`;
        return;
      }

      listEl.innerHTML = "";
      const grid = document.createElement("div");
      grid.className = "encoders-grid";

      for (const enc of encoders) {
        const card = document.createElement("div");
        card.className = "encoder-card";

        const statusBadge = enc.installed
          ? `<span class="badge-installed">${escapeHtml(t("models.encoder_installed"))}</span>`
          : `<span class="badge-missing">${escapeHtml(t("models.encoder_missing"))}</span>`;
        const sizeText = Number.isFinite(Number(enc.size_gb)) && Number(enc.size_gb) > 0
          ? `${Number(enc.size_gb).toFixed(1)} GB`
          : "";
        const usedBy = Array.isArray(enc.used_by) && enc.used_by.length
          ? `<div class="encoder-used-by">${enc.used_by.map(model =>
              `<span class="encoder-used-tag">${escapeHtml(model.name || model.id || "")}</span>`
            ).join("")}</div>`
          : "";
        const encoderPayload = `data-repo-id="${escapeHtml(enc.repo_id)}" data-file-path="${escapeHtml(enc.file_path || enc.file)}" data-file="${escapeHtml(enc.file || "")}" data-te-id="${escapeHtml(enc.id)}" data-used-by="${escapeHtml(String((enc.used_by || []).length))}"`;
        const actionBtn = enc.installed
          ? `<div class="variant-actions">
               <span class="card-action-btn active" style="width: auto;" >${ICON_CHECK} ${escapeHtml(t("models.encoder_cached"))}</span>
               <button type="button" class="btn-delete-files encoder-delete-btn" ${encoderPayload} onclick="App.deleteSharedEncoder(this)" title="${escapeHtml(t("models.encoder_delete_title"))}">${ICON_TRASH}<span>${escapeHtml(t("models.component_delete"))}</span></button>
             </div>`
          : `<button type="button" class="btn-download encoder-download-btn" data-repo-id="${escapeHtml(enc.repo_id)}" data-file-path="${escapeHtml(enc.file_path || enc.file)}" data-te-id="${escapeHtml(enc.id)}" onclick="App.downloadSharedEncoder(this.dataset.repoId, this.dataset.filePath, this.dataset.teId, this)">${ICON_DOWNLOAD}<span>${escapeHtml(t("models.encoder_download"))}${sizeText ? ` (${sizeText})` : ""}</span></button>`;

        card.innerHTML = `
          <div>
            <div class="encoder-card-header">
              <span class="encoder-title">${escapeHtml(enc.name || enc.file || enc.id)}</span>
              ${statusBadge}
            </div>
            <div class="encoder-meta">
              <span>📁 ${escapeHtml(t("models.encoder_file"))}: <code>${escapeHtml(enc.file || "")}</code></span>
              <span>🌐 ${escapeHtml(t("models.encoder_repo"))}: <code>${escapeHtml(enc.repo_id || "")}</code></span>
              <span>⚙ ${escapeHtml(t("models.encoder_format"))}: <code>${escapeHtml(enc.precision || "int4")}</code>${sizeText ? ` · ${sizeText}` : ""}</span>
            </div>
            ${usedBy}
          </div>
          <div style="margin-top: 12px;">
            ${actionBtn}
          </div>
        `;
        grid.appendChild(card);
      }

      listEl.appendChild(grid);
    } catch (err) {
      listEl.innerHTML = `<div class="placeholder-text" style="color: #f87171;">${escapeHtml(t("status.error"))}: ${escapeHtml(err.message)}</div>`;
    }
  },

  async downloadSharedEncoder(repoId, filePath, teId, btnEl) {
    if (btnEl) {
      btnEl.disabled = true;
      btnEl.classList.add("is-loading");
    }
    try {
      const result = await this.apiPost("/api/encoders/download", {
        repo_id: repoId,
        file_path: filePath,
      });
      this.state.encoders = null;
      this.toast("info", "Скачивание энкодера начато...");
      if (result.job_id && this.startDownloadPolling) {
        this.startDownloadPolling(result.job_id, {
          label: `Энкодер: ${filePath}`,
          kind: "encoder",
          repoId,
        });
      }
    } catch (err) {
      this.toast("error", "Ошибка скачивания: " + err.message);
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.classList.remove("is-loading");
      }
    }
  },

  async deleteSharedEncoder(btnEl) {
    const repoId = btnEl?.dataset.repoId || "";
    const filePath = btnEl?.dataset.filePath || "";
    const fileName = btnEl?.dataset.file || "";
    const teId = btnEl?.dataset.teId || "";
    const usedBy = Number(btnEl?.dataset.usedBy || 0);
    if (!repoId || (!filePath && !fileName)) return;
    const confirmed = confirm(
      t("models.encoder_delete_confirm", { name: fileName || filePath }) +
      (usedBy > 1 ? `\n\n${t("models.encoder_used_by", { count: usedBy })}` : "")
    );
    if (!confirmed) return;

    const payload = { repo_id: repoId, file_path: filePath, file: fileName, id: teId };
    try {
      let result = await this.apiPost("/api/encoders/delete", payload);
      if (result.status === "shared") {
        const forced = confirm(
          t("models.encoder_shared_confirm", { files: (result.shared || []).join("\n") })
        );
        if (!forced) return;
        result = await this.apiPost("/api/encoders/delete", { ...payload, force: true });
      }
      if (result.status === "ok") {
        const freedMb = Number(result.freed_mb || 0);
        const freedText = freedMb >= 1024 ? `${(freedMb / 1024).toFixed(2)} GB` : `${freedMb.toFixed(1)} MB`;
        this.toast("success", `${t("models.encoder_deleted")} (${freedText})`);
      } else {
        this.toast("info", t("models.encoder_not_found"));
      }
      this.state.encoders = null;
      await this.renderModelsTab();
      await this.refreshConfig();
    } catch (err) {
      this.toast("error", t("models.encoder_delete_failed") + err.message);
    }
  },

  async deleteModelComponent(modelId, role, index, componentId, label, btnEl) {
    const roleLabel = role === "text_encoder"
      ? t("models.component_role_te")
      : role === "vae" ? "VAE" : t("models.component_role_aux");
    const cleanName = String(label || "").trim();
    const nameSuffix = cleanName && cleanName.toLowerCase() !== roleLabel.toLowerCase()
      ? ` «${cleanName}»`
      : "";
    const confirmed = confirm(
      t("models.component_delete_confirm", {
        role: roleLabel,
        name: nameSuffix,
      })
    );
    if (!confirmed) return;

    if (btnEl) {
      btnEl.disabled = true;
      btnEl.classList.add("is-loading");
    }
    const payload = { role, index: Number(index || 0), id: componentId || null };
    try {
      let result = await this.apiPost(`/api/models/${encodeURIComponent(modelId)}/component-delete`, payload);
      if (result.status === "shared") {
        const forced = confirm(
          t("models.component_shared_confirm", { files: (result.shared || []).join("\n") })
        );
        if (!forced) {
          if (btnEl) { btnEl.disabled = false; btnEl.classList.remove("is-loading"); }
          return;
        }
        result = await this.apiPost(`/api/models/${encodeURIComponent(modelId)}/component-delete`, { ...payload, force: true });
      }
      if (result.status === "ok") {
        const freedMb = Number(result.freed_mb || 0);
        const freedText = freedMb >= 1024 ? `${(freedMb / 1024).toFixed(2)} GB` : `${freedMb.toFixed(1)} MB`;
        this.toast("success", `${t("models.component_deleted")} (${freedText})`);
      } else {
        this.toast("info", t("models.component_not_found"));
      }
      await this.refreshConfig();
      if (this.state.selectedModel === modelId) {
        await this.onModelChange(this.state.selectedVariant || null);
      } else {
        await this.renderModelsTab();
      }
    } catch (err) {
      this.toast("error", t("models.component_delete_failed") + err.message);
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.classList.remove("is-loading");
      }
    }
  },

  async downloadModelComponent(modelId, role, index, teId, btnEl) {
    const model = this.state.models.find(m => m.id === modelId);
    if (btnEl) {
      btnEl.disabled = true;
      btnEl.classList.add("is-loading");
    }
    try {
      const result = await this.apiPost(`/api/models/${encodeURIComponent(modelId)}/component-download`, {
        role,
        index: parseInt(index || "0", 10),
        id: teId || null,
      });
      if (result.status === "already_installed") {
        this.toast("info", "Компонент уже скачан");
        await this.refreshConfig();
        return;
      }
      const roleLabel = role === "text_encoder" ? "Энкодер" : role === "vae" ? "VAE" : "Sidecar";
      this.toast("info", `${roleLabel}: скачивание начато...`);
      if (result.job_id && this.startDownloadPolling) {
        this.startDownloadPolling(result.job_id, {
          label: `${model?.name || modelId} · ${roleLabel}`,
          kind: role,
          modelId,
          repoId: result.repo_id,
        });
      }
    } catch (err) {
      this.toast("error", "Ошибка скачивания: " + err.message);
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.classList.remove("is-loading");
      }
    }
  },

  async downloadModelVariant(modelId, variantId, btnEl) {
    const model = this.state.models.find(m => m.id === modelId);
    if (model && model.disabled) {
      this.toast("warning", localizedDisabledReason(model.disabled_reason));
      return;
    }

    this.state.selectedModel = modelId;
    if (variantId) {
      this.state.selectedVariant = variantId;
    }
    
    // Sync model select dropdown in generate tab if present
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
      modelSelect.value = modelId;
      await this.onModelChange(variantId || null);
    }

    if (btnEl) {
      btnEl.disabled = true;
      btnEl.classList.add("is-loading");
    }
    const modelName = model?.name || modelId;
    const started = await this.downloadModel({
      label: variantId ? `${modelName} · ${variantId}` : modelName,
      kind: "model",
      modelId,
      variantId: variantId || null,
    });
    if (!started && btnEl) {
      btnEl.disabled = false;
      btnEl.classList.remove("is-loading");
    }
  },

  // ---------- Event handlers ----------
  bindEvents() {
    if (this._eventsBound) return;
    this._eventsBound = true;

    // Sidebar navigation
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        this.switchTab(btn.dataset.tab);
      });
    });

    // Model select
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
      modelSelect.addEventListener("change", () => this.onModelChange());
    }

    // Model picker modal trigger
    const modelPickerTrigger = document.getElementById("model-picker-trigger");
    if (modelPickerTrigger) {
      modelPickerTrigger.addEventListener("click", () => this.openModelPickerModal());
    }

    // Variant select
    const variantSelect = document.getElementById("variant-select");
    if (variantSelect) {
      variantSelect.addEventListener("change", () => {
        this.state.selectedVariant = variantSelect.value;
        const currentVariant = this.state.modelState?.transformers?.find(v => v.id === variantSelect.value);
        if (currentVariant) {
          const stepsInput = document.getElementById("steps-input");
          if (stepsInput && currentVariant.steps) {
            stepsInput.value = currentVariant.steps;
          }
        }
        this.updateModelTrigger();
        if (this.state.modelState) {
          const activeTe = this.state.modelState.text_encoders?.find(te => te.id === this.state.selectedTextEncoder);
          const activeVar = currentVariant || (this.state.modelState.transformers || [])[0];
          const activeVarMissing = activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted");
          const hasMissing = activeVarMissing
                          || !this.state.modelState.base_model_installed
                          || (activeTe && !activeTe.installed && this.state.selectedTextEncoder !== "custom");
          this.updateDownloadButton(hasMissing, this.state.modelState);
        }
      });
    }

    // Steps, Sampler, Resolution, Seed sync with Dock pills
    const stepsInput = document.getElementById("steps-input");
    if (stepsInput) {
      stepsInput.addEventListener("input", () => this.updateModelTrigger());
    }
    const samplerSelect = document.getElementById("sampler-select");
    if (samplerSelect) {
      samplerSelect.addEventListener("change", () => this.updateDockPills());
    }
    const resSelect = document.getElementById("resolution-select");
    if (resSelect) {
      resSelect.addEventListener("change", () => this.updateDockPills());
    }
    const seedInput = document.getElementById("seed-input");
    if (seedInput) {
      seedInput.addEventListener("input", () => this.updateDockPills());
    }

    // CFG slider & input sync
    const cfgSlider = document.getElementById("cfg-slider");
    const cfgInput = document.getElementById("cfg-input");
    if (cfgSlider && cfgInput) {
      cfgSlider.addEventListener("input", () => {
        cfgInput.value = cfgSlider.value;
        this.updateDockPills();
      });
      cfgInput.addEventListener("input", () => {
        cfgSlider.value = cfgInput.value;
        this.updateDockPills();
      });
    }

    // Noise slider & input sync (+ auto-enable stochastic sampler when noise > 0)
    const noiseSlider = document.getElementById("noise-slider");
    const noiseInput = document.getElementById("noise-input");
    const autoSyncStochasticSampler = (valStr) => {
      const n = parseFloat(valStr);
      const sSel = document.getElementById("sampler-select");
      if (!isNaN(n) && n > 0 && sSel && sSel.value === "euler") {
        sSel.value = "euler_a";
      }
      this.updateDockPills();
    };
    if (noiseSlider && noiseInput) {
      noiseSlider.addEventListener("input", () => {
        noiseInput.value = noiseSlider.value;
        autoSyncStochasticSampler(noiseSlider.value);
      });
      noiseInput.addEventListener("input", () => {
        noiseSlider.value = noiseInput.value;
        autoSyncStochasticSampler(noiseInput.value);
      });
    }

    // Negative prompt input dot indicator & dock pill highlight
    const negInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negInput) {
      negInput.addEventListener("input", () => {
        const hasNeg = negInput.value.trim().length > 0;
        if (negDot) negDot.style.display = hasNeg ? "inline-block" : "none";
        this.updateDockPills();
      });
    }

    // Generate button
    const genBtn = document.getElementById("generate-btn");
    if (genBtn) {
      genBtn.addEventListener("click", () => this.onGenerate());
    }

    // Close dock popovers on outside click
    document.addEventListener("click", (e) => {
      if (!this.state.activeDockPopover) return;
      const dockWrap = document.getElementById("gemini-dock-wrap");
      const newDockWrap = document.getElementById("new-dock-wrap");
      if (dockWrap && !dockWrap.contains(e.target) && !newDockWrap?.contains(e.target)) {
        this.closeAllDockPopovers();
      }
    });

    // Keyboard shortcut: Ctrl/Cmd+Enter
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        this.onGenerate();
      }
      // ESC: close popovers/modals first, then cancel generation
      if (e.key === "Escape") {
        if (this.state.activeDockPopover) {
          this.closeAllDockPopovers();
        } else if (this.state.isModelPickerOpen) {
          this.closeModelPickerModal();
        } else if (this.state.selectedHistoryItem) {
          this.closeHistoryModal();
        } else if (this.state.currentJob) {
          this.cancelJob();
        }
      }
    });

    // Cancel button
    const cancelBtn = document.getElementById("progress-cancel");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", () => this.cancelJob());
    }

    // History search filter
    const historySearch = document.getElementById("history-search");
    if (historySearch) {
      historySearch.addEventListener("input", () => {
        if (this.state.historySearchTimer) clearTimeout(this.state.historySearchTimer);
        this.state.historySearchTimer = setTimeout(() => {
          this.state.historyPage = 1;
          this.loadHistory(1);
        }, 300);
      });
    }

    // Global dragover & drop prevention so dropped files NEVER open in the browser
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
    }, false);
    window.addEventListener("drop", (e) => {
      if (!e.target.closest("#tab-generate") && !e.target.closest("#ht-dropzone") && !e.target.closest("#manga-dropzone") && !e.target.closest("#inpaint-canvas-area")) {
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

    // Modal background click to close
    const historyModal = document.getElementById("history-modal");
    if (historyModal) {
      historyModal.addEventListener("click", (e) => {
        if (e.target === historyModal) this.closeHistoryModal();
      });
    }

    const modelPickerModal = document.getElementById("model-picker-modal");
    if (modelPickerModal) {
      modelPickerModal.addEventListener("click", (e) => {
        if (e.target === modelPickerModal) this.closeModelPickerModal();
      });
    }

    // Language toggle
    document.querySelectorAll(".lang-btn").forEach(btn => {
      btn.addEventListener("click", () => window.setLocale(btn.dataset.lang));
    });

    // Collapsible downloads panel: starts collapsed; open state is not persisted.
    const downloadsPanel = document.getElementById("downloads-panel");
    if (downloadsPanel && downloadsPanel.tagName === "DETAILS") {
      downloadsPanel.open = false;
    }
    this.renderDownloads();

    // Custom downward prompt resize handler (prevents Flexbox upward expansion bug)
    this.initPromptResize();
  },

  initPromptResize() {
    const handle = document.getElementById("prompt-resize-handle");
    const textarea = document.getElementById("prompt-input");
    const previewArea = document.querySelector("#tab-generate .preview-area");
    if (!handle || !textarea) return;

    let isDragging = false;
    let startY = 0;
    let startHeight = 0;

    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      isDragging = true;
      startY = e.clientY;
      startHeight = textarea.offsetHeight;
      if (previewArea && !previewArea.dataset.lockedHeight) {
        const h = Math.max(180, previewArea.offsetHeight);
        previewArea.style.minHeight = `${h}px`;
        previewArea.style.flex = "0 0 auto";
        previewArea.dataset.lockedHeight = "1";
      }
      document.body.style.userSelect = "none";
      document.body.style.cursor = "ns-resize";
    });

    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dy = e.clientY - startY;
      const nextHeight = Math.max(76, Math.min(360, startHeight + dy));
      textarea.style.height = `${nextHeight}px`;
    });

    document.addEventListener("mouseup", () => {
      if (!isDragging) return;
      isDragging = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      if (textarea.offsetHeight <= 84 && previewArea) {
        previewArea.style.minHeight = "";
        previewArea.style.flex = "";
        delete previewArea.dataset.lockedHeight;
      }
    });
  },

  switchTab(tabName) {
    this.state.currentTab = tabName;

    // Update sidebar active state
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    // Hide all tab panels
    document.querySelectorAll(".tab-panel").forEach(panel => {
      panel.style.display = "none";
    });

    // Map tab aliases (e.g. img2img / image_edit -> tab-generate)
    const targetId = (tabName === "img2img" || tabName === "image_edit") ? "tab-generate" : `tab-${tabName}`;

    // Show selected panel
    const activePanel = document.getElementById(targetId);
    if (activePanel) {
      const flexPanels = ["tab-generate", "tab-inpaint"];
      activePanel.style.display = flexPanels.includes(targetId) ? "flex" : "block";
    }

    // Auto-load history, models, or templates tab
    if (tabName === "history") {
      this.loadHistory(this.state.historyPage);
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

  toggleNegativePrompt() {
    const content = document.getElementById("negative-prompt-content");
    const input = document.getElementById("negative-prompt-input");
    const pill = document.getElementById("dock-neg-pill");
    if (!content) return;
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    if (pill) pill.classList.toggle("active", isHidden);
    if (isHidden && input) {
      input.focus();
    }
  },

  async saveQwenVae(value) {
    this.state.selectedVae = value;
    const vaeSelect = document.getElementById("vae-select");
    if (vaeSelect) vaeSelect.value = value;
    try {
      await this.apiPatch("/api/settings", { qwen_vae_type: value });
      this.toast("success", `${t("settings.vae.qwen_title") || "Qwen VAE"}: ${value} ✓`);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async onModelChange(preferredVariantId = null) {
    const select = document.getElementById("model-select");
    if (!select || !select.value) return;

    const modelId = select.value;
    const requestId = ++this.state.modelChangeRequestId;
    this.state.selectedModel = modelId;

    // Fetch model state (cache check)
    try {
      const state = await this.apiGet(`/api/models/${modelId}/state`);
      if (requestId !== this.state.modelChangeRequestId || select.value !== modelId) return;
      this.renderVariants(state, preferredVariantId);
    } catch (e) {
      console.error("[app] Failed to fetch model state:", e);
    }
    if (requestId !== this.state.modelChangeRequestId || select.value !== modelId) return;

    // Dynamic CFG recommendation sync
    const activeModel = this.state.models.find(m => m.id === modelId);
    let recCfg = 1.0;
    if (activeModel && typeof activeModel.default_cfg === "number") {
      recCfg = activeModel.default_cfg;
    } else if (activeModel && (activeModel.id.includes("turbo") || activeModel.engine === "sdnq")) {
      recCfg = 1.0;
    } else if (activeModel && (activeModel.id.startsWith("qwen_image_2_1") || activeModel.family === "qwen")) {
      recCfg = 3.0;
    } else if (activeModel && (activeModel.family === "sdxl" || activeModel.id.includes("sdxl"))) {
      recCfg = 7.0;
    } else if (activeModel && activeModel.family === "flux" && activeModel.id.includes("dev")) {
      recCfg = 3.5;
    }

    const cfgInput = document.getElementById("cfg-input");
    const cfgSlider = document.getElementById("cfg-slider");
    if (cfgInput && cfgSlider) {
      cfgInput.value = recCfg.toFixed(1);
      cfgSlider.value = recCfg.toFixed(1);
    }

    if (requestId !== this.state.modelChangeRequestId || select.value !== modelId) return;
    await this.loadLoras(true);
    if (requestId !== this.state.modelChangeRequestId || select.value !== modelId) return;
    this.updateModelTrigger();
  },

  renderVariants(modelState, preferredVariantId = null) {
    this.state.modelState = modelState;
    const variantSelect = document.getElementById("variant-select");
    if (!variantSelect) return;

    variantSelect.innerHTML = "";

    const variants = modelState.transformers || [];
    if (!variants.length) {
      variantSelect.innerHTML = `<option>${t("common.loading")}</option>`;
      return;
    }

    // Text Encoder select & Guardrails
    const teGroup = document.getElementById("text-encoder-group");
    const teSelect = document.getElementById("text-encoder-select");
    const teHint = document.getElementById("text-encoder-hint");
    const customTeInput = document.getElementById("custom-text-encoder-input");
    const teLink = document.getElementById("toggle-custom-te-link");
    const backendLabel = document.getElementById("backend-label");
    const activeModelObj = this.state.models.find(m => m.id === modelState.model_id);
    const isSDCpp = activeModelObj && activeModelObj.engine === "sd_cpp";
    const isSDNQ = activeModelObj && activeModelObj.engine === "sdnq";
    const isQuantFuncQwen = activeModelObj && activeModelObj.engine === "quantfunc" && activeModelObj.id === "qwen_image_2_1";

    if (backendLabel) {
      backendLabel.textContent = (isSDCpp || isSDNQ) ? (t("controls.quantization") || "Quantization") : (t("controls.backend") || "Backend");
    }

    let teMissing = false;

    if (teGroup && teSelect) {
      if (modelState.text_encoders && modelState.text_encoders.length > 0) {
        teGroup.style.display = "flex";
        teSelect.innerHTML = "";
        
        if (isQuantFuncQwen || isSDNQ) {
          teSelect.disabled = true;
          teSelect.style.display = "block";
          if (customTeInput) customTeInput.style.display = "none";
          if (teLink) teLink.style.display = "none";
          teGroup.classList.add("locked");
        } else {
          teSelect.disabled = false;
          if (teLink) teLink.style.display = "inline";
          teGroup.classList.remove("locked");
        }
        if (teHint) teHint.style.display = "none";

        // Sort text encoders: installed first, then recommended, then others
        const sortedEncoders = [...modelState.text_encoders].sort((a, b) => {
          if (Boolean(a.installed) !== Boolean(b.installed)) {
            return a.installed ? -1 : 1;
          }
          if (Boolean(a.recommended) !== Boolean(b.recommended)) {
            return a.recommended ? -1 : 1;
          }
          return 0;
        });

        // Determine active encoder: pick first installed if available
        let activeTe = sortedEncoders.find(te => te.id === this.state.selectedTextEncoder);
        if (!activeTe || (!activeTe.installed && sortedEncoders.some(te => te.installed))) {
          activeTe = sortedEncoders.find(te => te.installed) || sortedEncoders.find(te => te.recommended) || sortedEncoders[0];
          if (activeTe) {
            this.state.selectedTextEncoder = activeTe.id;
          }
        }

        for (const te of sortedEncoders) {
          const opt = document.createElement("option");
          opt.value = te.id;
          opt.textContent = `${te.name} (${te.installed ? '✓' : '⬇'})`;
          if (this.state.selectedTextEncoder === te.id) {
            opt.selected = true;
          }
          teSelect.appendChild(opt);
        }

        if (!isQuantFuncQwen && !isSDNQ) {
          const customOpt = document.createElement("option");
          customOpt.value = "custom";
          customOpt.textContent = `Custom file / ComfyUI...`;
          if (this.state.selectedTextEncoder === "custom") {
            customOpt.selected = true;
            if (customTeInput) customTeInput.style.display = "block";
          } else {
            teSelect.style.display = "block";
            if (customTeInput) customTeInput.style.display = "none";
            if (teLink) teLink.textContent = t("controls.custom_path") || "Custom path";
          }
          teSelect.appendChild(customOpt);
        }

        if (!activeTe && modelState.text_encoders.length > 0) {
          activeTe = modelState.text_encoders[0];
          this.state.selectedTextEncoder = activeTe.id;
        }

        if (this.state.selectedTextEncoder !== "custom" && activeTe && !activeTe.installed) {
          teMissing = true;
        }
      } else {
        teGroup.style.display = "none";
        teGroup.classList.remove("locked");
        if (teHint) teHint.style.display = "none";
        if (customTeInput) customTeInput.style.display = "none";
      }
    }

    // VAE select
    const vaeGroup = document.getElementById("vae-group");
    const vaeSelect = document.getElementById("vae-select");
    const customVaeInput = document.getElementById("custom-vae-input");
    const vaeLink = document.getElementById("toggle-custom-vae-link");
    if (vaeGroup && vaeSelect) {
      const vaeList = (Array.isArray(activeModelObj?.vae_files) && activeModelObj.vae_files.length)
        ? activeModelObj.vae_files
        : (activeModelObj?.vae ? [activeModelObj.vae] : []);
      const isQwen = Boolean(activeModelObj && (activeModelObj.family === "qwen" || activeModelObj.id.includes("qwen")));
      const vaeFiles = vaeList.filter(vae => vae.installed && vae.local_file);
      if (isQwen || vaeList.length) {
        vaeGroup.style.display = "flex";
        vaeSelect.innerHTML = "";
        const autoOpt = document.createElement("option");
        autoOpt.value = "auto";
        autoOpt.textContent = t("controls.vae_auto") || "Auto (model default)";
        vaeSelect.appendChild(autoOpt);
        if (isQwen) {
          for (const value of ["texture_fix", "official"]) {
            const opt = document.createElement("option");
            opt.value = value;
            opt.textContent = t(`controls.vae_${value}`) || value;
            vaeSelect.appendChild(opt);
          }
        }
        for (const vae of vaeFiles) {
          const fileName = String(vae.path || "").split("/").pop() || "VAE";
          const opt = document.createElement("option");
          opt.value = `file:${vae.local_file}`;
          opt.textContent = `${fileName} · ${vae.repo_id || ""}`.trim();
          opt.title = vae.local_file;
          vaeSelect.appendChild(opt);
        }
        const desired = this.state.selectedVae
          || (isQwen ? (this.state.settings?.qwen_vae_type || "texture_fix") : "auto");
        vaeSelect.value = [...vaeSelect.options].some(opt => opt.value === desired) ? desired : "auto";
        if (!customVaeInput || !customVaeInput.value.trim()) {
          vaeSelect.style.display = "block";
          if (customVaeInput) customVaeInput.style.display = "none";
          if (vaeLink) vaeLink.textContent = t("controls.custom_path") || "Custom path";
        }
      } else {
        vaeGroup.style.display = "none";
        vaeSelect.style.display = "block";
        if (customVaeInput) customVaeInput.style.display = "none";
        if (vaeLink) vaeLink.textContent = t("controls.custom_path") || "Custom path";
      }
    }

    // Group: recommended first
    const recommended = variants.filter(v => v.recommended);
    const others = variants.filter(v => !v.recommended);

    if (recommended.length) {
      const grp = document.createElement("optgroup");
      grp.label = t("model.recommended");
      for (const v of recommended) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.id} (${v.backend}, ${v.steps} steps, ${(v.size_gb || 0).toFixed(1)} GB) · ${v.state_label}`;
        grp.appendChild(opt);
      }
      variantSelect.appendChild(grp);
    }

    if (others.length) {
      const grp = document.createElement("optgroup");
      grp.label = t("model.available");
      for (const v of others) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.id} (${v.backend}, ${v.steps} steps, ${(v.size_gb || 0).toFixed(1)} GB) · ${v.state_label}`;
        grp.appendChild(opt);
      }
      variantSelect.appendChild(grp);
    }

    // Select preferred variant if provided, otherwise first recommended
    const preferred = preferredVariantId ? variants.find(v => v.id === preferredVariantId) : null;
    const installedVar = variants.find(v => v.state === "installed" || v.installed);
    const activeVar = preferred || installedVar || recommended[0] || others[0] || variants[0];
    if (activeVar) {
      variantSelect.value = activeVar.id;
      this.state.selectedVariant = activeVar.id;
      const stepsInput = document.getElementById("steps-input");
      if (stepsInput && activeVar.steps) {
        stepsInput.value = activeVar.steps;
      }
    }

    // Check if selected variant / base model / text encoder are missing
    const activeVarMissing = activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted");
    const hasMissing = activeVarMissing
                    || !modelState.base_model_installed
                    || teMissing;
    this.updateDownloadButton(hasMissing, modelState);
    this.updateModelTrigger();
  },

  onTextEncoderChange() {
    const teSelect = document.getElementById("text-encoder-select");
    const customTeInput = document.getElementById("custom-text-encoder-input");
    const link = document.getElementById("toggle-custom-te-link");
    if (!teSelect) return;
    this.state.selectedTextEncoder = teSelect.value;
    if (teSelect.value === "custom") {
      if (customTeInput) {
        customTeInput.style.display = "block";
        customTeInput.focus();
      }
      if (link) link.textContent = t("controls.standard_select") || "List";
    } else {
      if (customTeInput) customTeInput.style.display = "none";
      if (link) link.textContent = t("controls.custom_path") || "Custom path";
    }
    if (this.state.modelState) {
      const activeTe = this.state.modelState.text_encoders?.find(te => te.id === teSelect.value);
      const activeVar = this.state.modelState.transformers?.find(v => v.id === this.state.selectedVariant) || (this.state.modelState.transformers || [])[0];
      const activeVarMissing = activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted");
      const hasMissing = activeVarMissing
                      || !this.state.modelState.base_model_installed
                      || (activeTe && !activeTe.installed && teSelect.value !== "custom");
      this.updateDownloadButton(hasMissing, this.state.modelState);
    }
    this.updateDockPills();
  },

  toggleCustomTeInput(e) {
    if (e) e.preventDefault();
    const customTeInput = document.getElementById("custom-text-encoder-input");
    const teSelect = document.getElementById("text-encoder-select");
    const link = document.getElementById("toggle-custom-te-link");
    if (!customTeInput || !teSelect) return;
    const isHidden = customTeInput.style.display === "none";
    customTeInput.style.display = isHidden ? "block" : "none";
    teSelect.style.display = isHidden ? "none" : "block";
    if (isHidden) {
      this.state.selectedTextEncoder = "custom";
      if (teSelect.querySelector('option[value="custom"]')) {
        teSelect.value = "custom";
      }
      customTeInput.focus();
    } else {
      const firstOpt = teSelect.querySelector('option:not([value="custom"])');
      if (firstOpt) {
        teSelect.value = firstOpt.value;
        this.state.selectedTextEncoder = firstOpt.value;
      }
    }
    if (link) link.textContent = isHidden ? (t("controls.standard_select") || "List") : (t("controls.custom_path") || "Custom path");
  },

  toggleCustomVaeInput(e) {
    if (e) e.preventDefault();
    const customVaeInput = document.getElementById("custom-vae-input");
    const vaeSelect = document.getElementById("vae-select");
    const link = document.getElementById("toggle-custom-vae-link");
    if (!customVaeInput || !vaeSelect) return;
    const isHidden = customVaeInput.style.display === "none";
    customVaeInput.style.display = isHidden ? "block" : "none";
    vaeSelect.style.display = isHidden ? "none" : "block";
    if (link) link.textContent = isHidden ? (t("controls.standard_select") || "List") : (t("controls.custom_path") || "Custom path");
    if (isHidden) customVaeInput.focus();
  },

  async onVaeChange() {
    const val = document.getElementById("vae-select")?.value || "auto";
    this.state.selectedVae = val === "auto" ? "" : val;
    if (val === "auto" || val.startsWith("file:")) return;
    try {
      await this.apiPatch("/api/settings", { qwen_vae_type: val });
    } catch (e) {
      console.warn("Failed to persist VAE setting:", e);
    }
  },

  updateDownloadButton(showButton, modelState) {
    // Hide intrusive separate button inside dock
    const dlBtn = document.getElementById("download-model-btn");
    if (dlBtn) dlBtn.style.display = "none";

    // Manage canvas floating notice (#gen-uninstalled-notice)
    const notice = document.getElementById("gen-uninstalled-notice");
    const noticeText = document.getElementById("gen-uninstalled-notice-text");
    if (notice) {
      const activeModelObj = this.state.models.find(m => m.id === (modelState?.model_id || this.state.selectedModel));
      const activeTe = modelState?.text_encoders?.find(t => t.id === this.state.selectedTextEncoder);
      if (showButton) {
        if (activeModelObj && !activeModelObj.installed) {
          if (noticeText) noticeText.textContent = `Модель «${activeModelObj.name}» не скачана. Скачайте её в Model Hub перед генерацией.`;
        } else if (activeTe && !activeTe.installed && this.state.selectedTextEncoder !== "custom") {
          if (noticeText) noticeText.textContent = `Текстовый энкодер «${activeTe.name}» не скачан. Скачайте его в Model Hub.`;
        } else if (modelState && !modelState.base_model_installed) {
          if (noticeText) noticeText.textContent = `Базовые веса модели не скачаны. Скачайте их в Model Hub.`;
        } else {
          if (noticeText) noticeText.textContent = `Компоненты модели не скачаны. Перейдите в Model Hub для загрузки.`;
        }
        notice.style.display = "flex";
      } else {
        notice.style.display = "none";
      }
    }
  },

  async downloadModel(meta = null) {
    if (!this.state.selectedModel) return false;
    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (activeModel && activeModel.disabled) {
      this.toast("error", localizedDisabledReason(activeModel.disabled_reason, "models.disabled_title"));
      return false;
    }
    try {
      const teSelect = document.getElementById("text-encoder-select");
      const teId = teSelect && teSelect.value && teSelect.value !== "custom" ? teSelect.value : (this.state.selectedTextEncoder || null);
      const result = await this.apiPost(
        `/api/models/${this.state.selectedModel}/download`,
        {
          variant_id: this.state.selectedVariant,
          text_encoder_id: teId,
        }
      );
      this.toast("info", "Скачивание файлов модели начато...");
      this.startDownloadPolling(result.job_id, meta || {
        label: activeModel?.name || this.state.selectedModel,
        kind: "model",
        modelId: this.state.selectedModel,
        variantId: this.state.selectedVariant || null,
      });
      return true;
    } catch (e) {
      this.toast("error", "Download failed: " + e.message);
      return false;
    }
  },

  registerDownload(jobId, meta = {}) {
    if (!jobId) return;
    this.state.downloads[jobId] = {
      jobId,
      label: meta.label || `Загрузка ${String(jobId).slice(0, 8)}`,
      kind: meta.kind || "model",
      modelId: meta.modelId || null,
      variantId: meta.variantId || null,
      repoId: meta.repoId || null,
      status: "queued",
      progress: 0,
      message: "В очереди...",
      startedAt: Date.now(),
    };
    const panel = document.getElementById("downloads-panel");
    if (panel && panel.tagName === "DETAILS") panel.open = true;
    this.renderDownloads();
  },

  updateDownload(jobId, status = {}) {
    const entry = this.state.downloads[jobId];
    if (!entry) return;
    if (status.status) entry.status = status.status;
    if (typeof status.progress === "number") entry.progress = Math.max(0, Math.min(1, status.progress));
    if (status.message) entry.message = status.message;
    if (status.error) entry.message = status.error;
    this.renderDownloads();
  },

  renderDownloads() {
    const list = document.getElementById("downloads-list");
    if (!list) return;
    const entries = Object.values(this.state.downloads).sort((a, b) => b.startedAt - a.startedAt);
    const activeStatuses = new Set(["queued", "initializing", "downloading", "loading", "paused"]);
    const activeCount = entries.filter(e => activeStatuses.has(e.status)).length;
    const countEl = document.getElementById("downloads-active-count");
    if (countEl) {
      countEl.textContent = String(activeCount);
      countEl.hidden = activeCount === 0;
    }
    if (!entries.length) {
      list.innerHTML = `<div class="downloads-empty">${escapeHtml(t("models.downloads.empty"))}</div>`;
      return;
    }
    const statusLabels = {
      queued: "В очереди",
      initializing: "Подготовка",
      downloading: "Скачивание",
      loading: "Загрузка",
      paused: "Пауза",
      complete: "Готово",
      error: "Ошибка",
      cancelled: "Отменено",
    };
    list.innerHTML = entries.map(entry => {
      const active = activeStatuses.has(entry.status);
      const pct = Math.round((entry.progress || 0) * 100);
      const statusClass = entry.status === "error" ? "error" : (entry.status === "complete" ? "done" : (entry.status === "cancelled" ? "cancelled" : (entry.status === "paused" ? "paused" : "active")));
      let action = "";
      if (entry.status === "downloading") {
        action = `
          <button type="button" class="download-pause-btn" onclick="event.stopPropagation(); App.pauseDownload('${escapeHtml(entry.jobId)}')">⏸ Пауза</button>
          <button type="button" class="download-cancel-btn" onclick="event.stopPropagation(); App.cancelDownload('${escapeHtml(entry.jobId)}')">✕</button>
        `;
      } else if (entry.status === "paused") {
        action = `
          <button type="button" class="download-resume-btn" onclick="event.stopPropagation(); App.resumeDownload('${escapeHtml(entry.jobId)}')">▶ Продолжить</button>
          <button type="button" class="download-cancel-btn" onclick="event.stopPropagation(); App.cancelDownload('${escapeHtml(entry.jobId)}')">✕</button>
        `;
      } else if (active) {
        action = `<button type="button" class="download-cancel-btn" onclick="event.stopPropagation(); App.cancelDownload('${escapeHtml(entry.jobId)}')">Отменить</button>`;
      } else {
        action = `<button type="button" class="download-dismiss-btn" onclick="event.stopPropagation(); App.dismissDownload('${escapeHtml(entry.jobId)}')">Убрать</button>`;
      }
      return `
        <div class="download-item ${statusClass}">
          <div class="download-item-head">
            <span class="download-item-title" title="${escapeHtml(entry.label)}">${escapeHtml(entry.label)}</span>
            <span class="download-item-status">${statusLabels[entry.status] || escapeHtml(entry.status)}</span>
          </div>
          <div class="download-progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
            <span style="width: ${pct}%"></span>
          </div>
          <div class="download-item-foot">
            <span class="download-item-message" title="${escapeHtml(entry.message || "")}">${escapeHtml(entry.message || "")}</span>
            ${action}
          </div>
        </div>
      `;
    }).join("");
  },

  async pauseDownload(jobId) {
    try {
      await this.apiPost(`/api/jobs/${encodeURIComponent(jobId)}/pause`);
      const entry = this.state.downloads[jobId];
      if (entry) {
        entry.status = "paused";
        entry.message = "Загрузка приостановлена";
      }
      this.renderDownloads();
      this.toast("info", "Загрузка приостановлена");
    } catch (e) {
      this.toast("error", "Не удалось приостановить загрузку: " + (e.message || ""));
    }
  },

  async resumeDownload(jobId) {
    try {
      await this.apiPost(`/api/jobs/${encodeURIComponent(jobId)}/resume`);
      const entry = this.state.downloads[jobId];
      if (entry) {
        entry.status = "downloading";
        entry.message = "Возобновление загрузки...";
      }
      this.renderDownloads();
      this.toast("info", "Загрузка возобновлена");
    } catch (e) {
      this.toast("error", "Не удалось возобновить загрузку: " + (e.message || ""));
    }
  },

  dismissDownload(jobId) {
    delete this.state.downloads[jobId];
    this.renderDownloads();
  },

  clearFinishedDownloads() {
    for (const [jobId, entry] of Object.entries(this.state.downloads)) {
      if (!["queued", "initializing", "downloading", "loading"].includes(entry.status)) {
        delete this.state.downloads[jobId];
      }
    }
    this.renderDownloads();
  },

  async cancelDownload(jobId) {
    try {
      await this.apiPost(`/api/jobs/${encodeURIComponent(jobId)}/cancel`);
      const entry = this.state.downloads[jobId];
      if (entry) {
        entry.status = "cancelled";
        entry.message = "Отменено пользователем";
      }
      this.renderDownloads();
    } catch (e) {
      this.toast("error", "Не удалось отменить загрузку: " + e.message);
    }
  },

  startDownloadPolling(jobId, meta = {}) {
    const overlay = document.getElementById("progress-overlay");
    if (overlay) overlay.style.display = "flex";
    this.registerDownload(jobId, meta);

    let pollCount = 0;
    const MAX_POLLS = 3600;  // 30 minutes at 500ms interval — fail-safe against stuck jobs

    const interval = setInterval(async () => {
      pollCount++;
      if (pollCount > MAX_POLLS) {
        clearInterval(interval);
        this.updateDownload(jobId, { status: "error", message: "Таймаут загрузки (30 минут)" });
        if (overlay) overlay.style.display = "none";
        this.toast("error", "Download timed out (30 min) — check server logs");
        return;
      }
      try {
        const status = await this.apiGet(`/api/jobs/${jobId}`);
        this.updateDownload(jobId, status);
        this.updateProgress(status);

        if (status.status === "complete") {
          clearInterval(interval);
          if (overlay) overlay.style.display = "none";
          this.toast("success", "Model files downloaded ✓");
          // Refresh model/encoder metadata so the Models Hub flips to "Installed".
          this.state.encoders = null;
          await this.refreshConfig();
        } else if (status.status === "error") {
          clearInterval(interval);
          if (overlay) overlay.style.display = "none";
          this.toast("error", "Download failed: " + (status.error || status.message || ""));
        } else if (status.status === "cancelled") {
          clearInterval(interval);
          if (overlay) overlay.style.display = "none";
          this.toast("info", "Download cancelled");
        }
      } catch (e) {
        console.error("[app] Download polling failed:", e);
        // If we get HTTP 404 (job not found), stop polling
        clearInterval(interval);
        this.updateDownload(jobId, { status: "error", message: e.message || "Задание не найдено" });
        if (overlay) overlay.style.display = "none";
        this.toast("error", "Download job not found — try again");
      }
    }, 500);
  },

  // ---------- Generation ----------
  async onGenerate() {
    const prompt = document.getElementById("prompt-input")?.value?.trim();
    if (!prompt) {
      this.toast("warning", t("prompt.placeholder"));
      return;
    }

    if (!this.state.selectedModel) {
      this.toast("warning", t("controls.model"));
      return;
    }

    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (activeModel && activeModel.disabled) {
      this.toast("error", localizedDisabledReason(activeModel.disabled_reason));
      return;
    }

    if (activeModel && activeModel.installed === false) {
      this.toast("warning", `Модель "${activeModel.name}" не скачана. Перейдите в Model Hub для загрузки.`);
      const noticeEl = document.getElementById("gen-uninstalled-notice");
      const noticeText = document.getElementById("gen-uninstalled-notice-text");
      if (noticeText) noticeText.textContent = `Модель «${activeModel.name}» не скачана. Перейдите в Model Hub для загрузки.`;
      if (noticeEl) noticeEl.style.display = "flex";
      return;
    }

    const teSelect = document.getElementById("text-encoder-select");
    const activeTe = this.state.modelState?.text_encoders?.find(t => t.id === teSelect?.value);
    if (activeTe && !activeTe.installed && teSelect?.value !== "custom") {
      this.toast("warning", `Текстовый энкодер "${activeTe.name}" не скачан. Скачайте его в Model Hub.`);
      const noticeEl = document.getElementById("gen-uninstalled-notice");
      const noticeText = document.getElementById("gen-uninstalled-notice-text");
      if (noticeText) noticeText.textContent = `Текстовый энкодер «${activeTe.name}» не скачан. Скачайте его в Model Hub.`;
      if (noticeEl) noticeEl.style.display = "flex";
      return;
    }

    if (activeModel?.family === "flux3") {
      const extrasEnv = this.state.extrasStatus?.env;
      if (extrasEnv?.nunchaku && extrasEnv.has_flux3 === false) {
        this.toast("error", "FLUX.3 SVDQuant требует Nunchaku >= 1.4.0. Обновите Nunchaku в Настройки → Движки или выберите 'Flux.2 Klein (Nunchaku)'.");
        return;
      }
      if (extrasEnv?.diffusers && extrasEnv.has_diffusers_flux3 === false) {
        this.toast("error", "FLUX.3 требует diffusers из GitHub main (стабильный 0.41.0 не поддерживает FLUX.3). Переустановите Nunchaku в Настройки → Движки — установщик поставит diffusers main.");
        return;
      }
    }
    const engineType = activeModel?.engine || "quantfunc";
    if (engineType === "quantfunc" && (!this.state.engineStatus || this.state.engineStatus.status === "missing")) {
      this.toast("error", t("errors.engine_not_installed"));
      this.switchTab("settings");
      return;
    }

    const res = this.parseResolution();

    const variantSelect = document.getElementById("variant-select");
    const selectedVariantId = variantSelect?.value || this.state.selectedVariant;
    const currentVariant = this.state.modelState?.transformers?.find(v => v.id === selectedVariantId);
    const backend = currentVariant ? currentVariant.backend : "lighting";

    // Collect params
    const cfgVal = parseFloat(document.getElementById("cfg-input")?.value);
    const noiseVal = parseFloat(document.getElementById("noise-input")?.value);
    const negPrompt = document.getElementById("negative-prompt-input")?.value?.trim() || "";
    const samplerVal = document.getElementById("sampler-select")?.value || "euler";

    const params = {
      model_id: this.state.selectedModel,
      variant_id: selectedVariantId,
      prompt: prompt,
      negative_prompt: negPrompt,
      seed: this.parseSeed(),
      width: res.width,
      height: res.height,
      steps: parseInt(document.getElementById("steps-input")?.value || (currentVariant ? String(currentVariant.steps) : "4"), 10),
      cfg: !isNaN(cfgVal) ? cfgVal : 1.0,
      noise: !isNaN(noiseVal) ? noiseVal : 0.0,
      sampler: samplerVal,
      backend: backend,
    };

    // Attach active LoRA stack
    if (this.state.activeLoras && this.state.activeLoras.length > 0) {
      params.loras = this.state.activeLoras.map(l => ({
        name: l.name,
        strength: l.strength !== undefined ? parseFloat(l.strength) : 1.0,
      }));
    }

    if (this.state.img2imgRef) {
      params.init_image_path = this.state.img2imgRef.path;
      params.init_image_strength = parseFloat(
        document.getElementById("img2img-strength")?.value || "0.85"
      );
      params.ref_mode = document.getElementById("img2img-mode")?.value || "control_ref";
      const cnInput = document.getElementById("controlnet-custom-path");
      const cnPath = cnInput && cnInput.style.display !== "none" ? cnInput.value.trim() : "";
      if (cnPath) {
        params.control_net_path = cnPath;
      }
    }

    const teGroup = document.getElementById("text-encoder-group");
    if (teGroup && teGroup.style.display !== "none") {
      const teSelect = document.getElementById("text-encoder-select");
      const customTeInput = document.getElementById("custom-text-encoder-input");
      const customTeVal = customTeInput && customTeInput.style.display !== "none" ? customTeInput.value.trim() : "";
      if (customTeVal) {
        params.custom_text_encoder_path = customTeVal;
      } else if (teSelect && teSelect.value && teSelect.value !== "custom") {
        params.text_encoder_id = teSelect.value;
      }
    }

    const vaeGroup = document.getElementById("vae-group");
    if (vaeGroup && vaeGroup.style.display !== "none") {
      const customVaeInput = document.getElementById("custom-vae-input");
      const customVaeVal = customVaeInput && customVaeInput.style.display !== "none" ? customVaeInput.value.trim() : "";
      if (customVaeVal) {
        params.custom_vae_path = customVaeVal;
      } else {
        const vaeSelect = document.getElementById("vae-select");
        if (vaeSelect && vaeSelect.value && vaeSelect.value !== "auto") {
          if (vaeSelect.value.startsWith("file:")) {
            params.custom_vae_path = vaeSelect.value.slice(5);
          } else {
            params.vae = vaeSelect.value;
          }
        }
      }
    }

    // Mask editing (Generate): the mask defines the edited region, everything
    // outside of it is restored by the backend compositor.
    const maskStatus = this.applyGenMaskParams(params);
    if (maskStatus === "empty") {
      this.toast("warning", this.maskT("generate.mask.empty_block", "Маска пуста — нарисуйте область или отключите режим «Маска»"));
      return;
    }
    if (maskStatus === "ok") {
      const maskModel = (this.state.models || []).find((m) => m.id === params.model_id);
      if (maskModel && !maskModel.supports_mask) {
        const capable = this.getMaskCapableModels();
        const names = capable.map((m) => m.name).join(", ") || this.maskT("generate.mask.no_capable", "нет в наличии");
        this.toast("error", this.maskT(
          "generate.mask.model_unsupported",
          `Модель «${maskModel.name}» не поддерживает маску. Поддерживают: ${names}`,
          { name: maskModel.name, list: names }
        ));
        return;
      }
      params.init_image_strength = 1.0;
    }

    // Disable button + show progress
    this.setGenerating(true);

    try {
      // Start generation
      const result = await this.apiPost("/api/generate", params);
      this.state.currentJob = result.job_id;

      // Poll for status. The queue overlay appears only if the first server
      // response is actually "queued"; a job that owns the GPU slot must not
      // flash a false "GPU busy" popup while the worker initializes.

      this.startPolling(result.job_id);

    } catch (e) {
      console.error("[app] Generate failed:", e);
      this.toast("error", t("errors.missing_files") + ": " + e.message);
      this.setGenerating(false);
    }
  },

  snapDimension(val, multiple = 64) {
    const v = parseInt(val, 10);
    if (isNaN(v) || v <= 0) return 1024;
    return Math.max(multiple, Math.round(v / multiple) * multiple);
  },

  parseResolution() {
    const val = document.getElementById("resolution-select")?.value?.trim() || "1024x1024";
    if (val.includes("x")) {
      const parts = val.split("x");
      const w = parseInt(parts[0], 10);
      const h = parseInt(parts[1], 10);
      return {
        width: this.snapDimension(!isNaN(w) && w > 0 ? w : 1024, 64),
        height: this.snapDimension(!isNaN(h) && h > 0 ? h : 1024, 64),
      };
    }
    const num = parseInt(val, 10);
    const sz = this.snapDimension(!isNaN(num) && num > 0 ? num : 1024, 64);
    return { width: sz, height: sz };
  },

  parseSeed() {
    const val = document.getElementById("seed-input")?.value?.trim();
    if (!val || val === "random" || val === t("controls.seed.random")) {
      return null;  // Backend will randomize
    }
    const n = parseInt(val, 10);
    return isNaN(n) ? null : n;
  },

  startPolling(jobId) {
    if (this.state.pollInterval) {
      clearInterval(this.state.pollInterval);
    }

    let pollCount = 0;
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 10;
    const MAX_POLLS = 9000;  // 30 min at 200ms — fail-safe against stuck jobs

    this.state.pollInterval = setInterval(async () => {
      pollCount++;
      if (pollCount > MAX_POLLS) {
        this.stopPolling();
        this.setGenerating(false);
        this.toast("error", "Generation timed out (30 min) — check server logs");
        return;
      }
      try {
        const status = await this.apiGet(`/api/jobs/${jobId}`);
        consecutiveErrors = 0;
        this.updateProgress(status, jobId);

        if (status.status === "complete") {
          this.stopPolling();
          this.setGenerating(false);
          this.showPreview(status.image_path);
          this.toast("success", t("toast.generation_complete"));
        } else if (status.status === "error") {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("error", t("toast.generation_error") + ": " + (status.error || status.message || ""));
        } else if (status.status === "cancelled") {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("info", t("toast.cancelled"));
        }
      } catch (e) {
        consecutiveErrors++;
        console.warn(`[app] Polling warning (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, e);
        const errMsg = String(e?.message || "");
        if (errMsg.includes("404") || errMsg.includes("Not Found")) {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("error", "Generation job lost (404) — try again");
        } else if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("error", "Network connection lost during generation — try again");
        }
      }
    }, 200);  // Poll every 200ms
  },

  stopPolling() {
    if (this.state.pollInterval) {
      clearInterval(this.state.pollInterval);
      this.state.pollInterval = null;
    }
  },

  async cancelJob() {
    const jobId = this.state.currentJob || this.state.queueJobId;
    if (!jobId) return;
    try {
      await this.apiPost(`/api/jobs/${jobId}/cancel`);
    } catch (e) {
      console.error("[app] Cancel failed:", e);
      this.toast("error", e.message || t("status.cancelled"));
    }
  },

  async cancelQueueJob() {
    const jobId = this.state.queueJobId;
    if (!jobId) {
      await this.cancelJob();
      return;
    }
    try {
      await this.apiPost(`/api/jobs/${jobId}/cancel`);
      this.hideQueueStatus();
      this.toast("info", t("queue.cancelled"));
    } catch (e) {
      this.toast("error", e.message || t("queue.cancel_failed"));
    }
  },

  showQueueStatus(status, jobId = null) {
    const overlay = document.getElementById("queue-overlay");
    if (!overlay) return;
    this.state.queueJobId = jobId || this.state.currentJob || this.state.queueJobId;
    const position = Math.max(1, Number(status.queue_position || 1));
    const total = Math.max(position, Number(status.queue_size || position));
    const posEl = document.getElementById("queue-position");
    const totalEl = document.getElementById("queue-total");
    const messageEl = document.getElementById("queue-message");
    if (posEl) posEl.textContent = String(position);
    if (totalEl) totalEl.textContent = String(total);
    if (messageEl) {
      messageEl.textContent = t("queue.initial") || "Ожидание свободного GPU...";
    }
    overlay.style.display = "flex";
    const statusPill = document.getElementById("status-pill");
    const statusMsg = document.getElementById("status-message");
    if (statusPill) {
      statusPill.className = "status-pill warning";
      statusPill.textContent = t("queue.short");
    }
    if (statusMsg) statusMsg.textContent = t("queue.waiting", { position, total });
  },

  hideQueueStatus() {
    const overlay = document.getElementById("queue-overlay");
    if (overlay) overlay.style.display = "none";
    this.state.queueJobId = null;
  },

  updateProgress(status, jobId = null) {
    const overlay = document.getElementById("progress-overlay");
    const fill = document.getElementById("progress-fill");
    const text = document.getElementById("progress-text");
    const statusPill = document.getElementById("status-pill");
    const statusMsg = document.getElementById("status-message");

    if (!overlay || !fill || !text) return;

    if (status.status === "queued") {
      overlay.style.display = "none";
      this.showQueueStatus(status, jobId);
      return;
    }
    this.hideQueueStatus();
    overlay.style.display = "flex";

    const pct = (status.progress || 0) * 100;
    fill.style.width = `${pct}%`;

    let msg = t(`status.${status.status}`) || status.status;
    if (status.step && status.total_steps) {
      msg += ` · ${status.step}/${status.total_steps}`;
    }
    if (status.message) {
      msg += ` · ${status.message}`;
    }
    text.textContent = msg;

    if (statusPill) {
      statusPill.className = "status-pill warning";
      statusPill.textContent = status.step && status.total_steps ? `${status.step}/${status.total_steps}` : "Busy";
    }
    if (statusMsg) {
      statusMsg.textContent = status.message || "";
    }
  },

  setGenerating(busy) {
    const btn = document.getElementById("generate-btn");
    const overlay = document.getElementById("progress-overlay");
    const statusPill = document.getElementById("status-pill");
    const statusMsg = document.getElementById("status-message");
    if (!btn) return;

    if (busy) {
      btn.classList.add("busy");
      btn.disabled = true;
      btn.textContent = t("controls.busy");
      if (overlay) overlay.style.display = "none";
      if (statusPill) {
        statusPill.className = "status-pill warning";
        statusPill.textContent = t("controls.busy") || "Generating...";
      }
    } else {
      btn.classList.remove("busy");
      btn.disabled = false;
      btn.textContent = t("controls.generate");
      if (overlay) overlay.style.display = "none";
      this.hideQueueStatus();
      if (statusPill) {
        statusPill.className = "status-pill success";
        statusPill.textContent = t("status.ready") || "Ready";
      }
      if (statusMsg) {
        statusMsg.textContent = "";
      }
    }
    this.updateNewDock(this);
  },

  showPreview(imagePath) {
    const placeholder = document.getElementById("preview-placeholder");
    const wrap = document.getElementById("preview-image-wrap");
    const img = document.getElementById("preview-img");

    if (placeholder) placeholder.style.display = "none";
    if (wrap) wrap.style.display = "flex";
    if (img) {
      if (this.state.currentPreviewUrl) {
        this.state.previousPreviewUrl = this.state.currentPreviewUrl;
      }
      const filename = imagePath.split(/[\/\\]/).pop();
      const url = `/outputs/${encodeURIComponent(filename)}?t=${Date.now()}`;
      if (!this.state.genMask?.enabled) img.src = url;
      this.state.currentPreviewFilename = filename;
      this.state.currentPreviewUrl = url;
      this.state.lastPreviewName = filename;
      this.state.lastPreviewUrl = url;

      if (this.state.compareActive) {
        this.updateCompareImages();
      }
    }
  },

  // ---------- History ----------
  initHistorySettings() {
    try {
      const saved = parseInt(localStorage.getItem("lazysd_history_page_size"), 10);
      if ([12, 24, 48].includes(saved)) this.state.historyPageSize = saved;
    } catch (_) {}
    const select = document.getElementById("history-page-size");
    if (select) select.value = String(this.state.historyPageSize);
  },

  async loadHistory(page = this.state.historyPage) {
    const grid = document.getElementById("history-grid");
    if (grid) grid.classList.add("history-loading");
    const seq = ++this.state.historyRequestSeq;
    this.state.historyLoading = true;
    try {
      const size = this.state.historyPageSize || 24;
      const filter = this.state.historyFilter || "all";
      const query = (document.getElementById("history-search")?.value || "").trim();
      const params = new URLSearchParams({
        page: String(page || 1),
        page_size: String(size),
        filter,
      });
      if (query) params.set("q", query);

      const data = await this.apiGet(`/api/history?${params.toString()}`);
      if (seq !== this.state.historyRequestSeq) return;

      if (Array.isArray(data)) {
        // Backward-compatible fallback for an older backend / cached response.
        let items = data;
        const myId = this.state.account?.id || "host";
        if (filter === "mine") items = items.filter(it => (it.owner_id || "host") === myId);
        else if (filter === "guests") items = items.filter(it => !["host", "host_remote"].includes(it.owner_id || "host"));
        else if (filter === "shared") items = items.filter(it => Boolean(it.shared));
        if (query) {
          const needle = query.toLowerCase();
          items = items.filter(it =>
            (it.prompt || "").toLowerCase().includes(needle) ||
            (it.model_id || "").toLowerCase().includes(needle) ||
            (it.name || "").toLowerCase().includes(needle) ||
            (it.owner_name || "").toLowerCase().includes(needle)
          );
        }
        const pages = items.length ? Math.ceil(items.length / size) : 0;
        const current = pages ? Math.max(1, Math.min(pages, page || 1)) : 1;
        this.state.historyServerPaginated = false;
        this.state.historyTotal = items.length;
        this.state.historyPages = pages;
        this.state.historyPage = current;
        this.state.historyItems = items.slice((current - 1) * size, current * size);
      } else {
        this.state.historyServerPaginated = true;
        this.state.historyItems = Array.isArray(data.items) ? data.items : [];
        this.state.historyTotal = Number(data.total || 0);
        this.state.historyPages = Number(data.pages || 0);
        this.state.historyPage = Number(data.page || 1);
        this.state.historyPageSize = Number(data.page_size || size);
      }

      const countEl = document.getElementById("history-count");
      if (countEl) countEl.textContent = String(this.state.historyTotal || 0);
      this.renderHistory();
    } catch (e) {
      if (seq !== this.state.historyRequestSeq) return;
      console.error("[app] Failed to load history:", e);
      this.toast("error", t("errors.history_load_failed") + ": " + e.message);
    } finally {
      if (seq === this.state.historyRequestSeq) {
        this.state.historyLoading = false;
        if (grid) grid.classList.remove("history-loading");
      }
    }
  },

  setHistoryFilter(filter) {
    this.state.historyFilter = filter || "all";
    this.state.historyPage = 1;
    document.querySelectorAll(".history-filter-chip").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.filter === this.state.historyFilter);
    });
    this.loadHistory(1);
  },

  setHistoryPage(page) {
    const target = parseInt(page, 10);
    if (!target || target === this.state.historyPage) return;
    if (target < 1 || (this.state.historyPages && target > this.state.historyPages)) return;
    this.state.historyPage = target;
    this.loadHistory(target);
    const panel = document.getElementById("tab-history");
    if (panel) panel.scrollTop = 0;
  },

  setHistoryPageSize(sz) {
    const size = parseInt(sz, 10);
    this.state.historyPageSize = [12, 24, 48].includes(size) ? size : 24;
    this.state.historyPage = 1;
    try {
      localStorage.setItem("lazysd_history_page_size", String(this.state.historyPageSize));
    } catch (_) {}
    this.loadHistory(1);
  },

  renderHistoryPagination() {
    const el = document.getElementById("history-pagination");
    if (!el) return;
    const pages = Number(this.state.historyPages || 0);
    const page = Number(this.state.historyPage || 1);
    if (pages <= 1) {
      el.innerHTML = "";
      el.style.display = "none";
      return;
    }

    const numbers = new Set([1, pages, page, page - 2, page - 1, page + 1, page + 2]);
    const sorted = [...numbers].filter(n => n >= 1 && n <= pages).sort((a, b) => a - b);
    let html = `<button type="button" class="history-page-btn" onclick="App.setHistoryPage(${page - 1})" ${page <= 1 ? "disabled" : ""} title="Назад">‹</button>`;
    let previous = 0;
    for (const number of sorted) {
      if (previous && number - previous > 1) html += `<span class="history-page-gap">…</span>`;
      html += `<button type="button" class="history-page-btn ${number === page ? "active" : ""}" onclick="App.setHistoryPage(${number})" ${number === page ? "aria-current=\"page\"" : ""}>${number}</button>`;
      previous = number;
    }
    html += `<button type="button" class="history-page-btn" onclick="App.setHistoryPage(${page + 1})" ${page >= pages ? "disabled" : ""} title="Вперёд">›</button>`;
    html += `<span class="history-page-info">${page} / ${pages} · всего ${this.state.historyTotal}</span>`;
    el.innerHTML = html;
    el.style.display = "flex";
  },

  renderHistory() {
    const grid = document.getElementById("history-grid");
    if (!grid) return;

    const items = this.state.historyItems || [];
    grid.innerHTML = "";

    if (!items.length) {
      grid.innerHTML = `
        <div class="preview-placeholder" style="grid-column: 1 / -1;">
          <div class="placeholder-text" data-i18n="history.empty">${t("history.empty")}</div>
        </div>
      `;
      this.renderHistoryPagination();
      return;
    }

    for (const item of items) {
      const card = document.createElement("div");
      card.className = "history-item";

      const timeStr = item.modified ? new Date(item.modified * 1000).toLocaleDateString(undefined, {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
      }) : "";

      const promptDisplay = item.prompt ? item.prompt : item.name;
      const isHostItem = ["host", "host_remote"].includes(item.owner_id || "host");
      const authorBadge = isHostItem ? `👑 ${t("common.host")}` : `👤 ${escapeHtml(item.owner_name || t("common.guest"))}`;
      const sharedBadge = item.shared ? `<span class="history-shared-badge" title="Видна всем пользователям">🌐 Общая</span>` : "";

      card.innerHTML = `
        <div class="history-item-thumb-wrapper">
          <div class="history-item-badges">
            <span class="history-author-badge">${authorBadge}</span>
            ${sharedBadge}
          </div>
          <img src="${escapeHtml(item.url)}" alt="${escapeHtml(promptDisplay)}" loading="lazy">
          <div class="history-item-overlay">
            <div class="history-item-prompt-preview">${escapeHtml(promptDisplay)}</div>
          </div>
        </div>
        <div class="history-item-info">
          <div class="history-item-title" title="${escapeHtml(promptDisplay)}">${escapeHtml(promptDisplay)}</div>
          <div class="history-item-meta">
            <span>${escapeHtml(item.model_id || "QuantFunc")}</span>
            <span>${timeStr}</span>
          </div>
        </div>
      `;

      card.addEventListener("click", () => this.openHistoryModal(item));
      grid.appendChild(card);
    }
    this.renderHistoryPagination();
  },

  openHistoryModal(item) {
    this.state.selectedHistoryItem = item;
    const modal = document.getElementById("history-modal");
    if (!modal) return;

    const img = document.getElementById("history-modal-img");
    const promptEl = document.getElementById("history-modal-prompt");
    const promptBox = document.getElementById("history-modal-prompt-box");
    const promptToggle = document.getElementById("history-prompt-toggle");
    const modelEl = document.getElementById("history-modal-model");
    const seedEl = document.getElementById("history-modal-seed");
    const resEl = document.getElementById("history-modal-res");
    const stepsEl = document.getElementById("history-modal-steps");
    const backendEl = document.getElementById("history-modal-backend");
    const timeEl = document.getElementById("history-modal-time");
    const authorEl = document.getElementById("history-modal-author");
    const sharedStatusEl = document.getElementById("history-modal-shared-status");
    const shareBtn = document.getElementById("history-modal-share-btn");
    const shareText = document.getElementById("history-modal-share-text");

    const promptText = item.prompt || t("history.no_prompt");
    if (img) img.src = item.url;
    if (promptEl) {
      promptEl.textContent = promptText;
      promptEl.scrollTop = 0;
    }
    if (promptBox) {
      promptBox.classList.remove("collapsed");
    }
    if (promptToggle) {
      promptToggle.style.display = promptText.length > 110 ? "inline-block" : "none";
      promptToggle.textContent = "Свернуть ▴";
    }
    if (modelEl) modelEl.textContent = item.model_id || "—";
    if (seedEl) seedEl.textContent = item.seed !== null && item.seed !== undefined ? item.seed : "—";
    if (resEl) resEl.textContent = item.width && item.height ? `${item.width} × ${item.height}` : "—";
    if (stepsEl) stepsEl.textContent = item.steps || "—";
    if (backendEl) backendEl.textContent = item.backend || "—";
    if (timeEl) timeEl.textContent = item.generation_time_sec ? `${item.generation_time_sec}s` : "—";

    const isHostItem = ["host", "host_remote"].includes(item.owner_id || "host");
    const authorName = isHostItem ? `👑 ${t("common.host")}` : `👤 ${item.owner_name || t("common.guest")}`;
    if (authorEl) authorEl.textContent = authorName;
    if (sharedStatusEl) {
      sharedStatusEl.textContent = item.shared ? "🌐 Доступна всем" : "🔒 Личная (только для автора)";
      sharedStatusEl.style.color = item.shared ? "#34d399" : "var(--text-dim)";
    }

    if (shareBtn) {
      const myId = this.state.account?.id || "host";
      const canShare = !this.state.account?.is_guest || (item.owner_id === myId);
      shareBtn.style.display = canShare ? "inline-flex" : "none";
      if (shareText) {
        shareText.textContent = item.shared ? "Скрыть из общей истории" : "Поделиться в общую историю";
      }
    }

    // Bind modal action buttons
    const reuseBtn = document.getElementById("history-modal-reuse-btn");
    const sendImg2ImgBtn = document.getElementById("history-modal-send-img2img-btn");
    const copyBtn = document.getElementById("history-modal-copy-btn");
    const deleteBtn = document.getElementById("history-modal-delete-btn");

    if (reuseBtn) {
      reuseBtn.onclick = () => {
        this.reuseHistoryItem(item);
        this.closeHistoryModal();
      };
    }

    if (sendImg2ImgBtn) {
      sendImg2ImgBtn.onclick = () => {
        this.setImg2ImgRef({ path: item.name, url: item.url });
        this.closeHistoryModal();
      };
    }

    if (copyBtn) {
      copyBtn.onclick = () => {
        if (item.prompt) {
          navigator.clipboard.writeText(item.prompt)
            .then(() => this.toast("success", t("history.copy_prompt") + " ✓"))
            .catch(() => this.toast("error", t("errors.clipboard_denied")));
        }
      };
    }

    if (deleteBtn) {
      deleteBtn.onclick = async () => {
        if (confirm(t("history.confirm_delete"))) {
          await this.deleteHistoryItem(item.name);
          this.closeHistoryModal();
        }
      };
    }

    const openFolderBtn = document.getElementById("history-modal-open-folder-btn");
    if (openFolderBtn) {
      openFolderBtn.onclick = () => {
        this.openFolder("outputs", item.name);
      };
    }

    modal.style.display = "flex";
  },

  async toggleShareCurrentHistoryItem() {
    const item = this.state.selectedHistoryItem;
    if (!item) return;
    try {
      const res = await this.apiPost(`/api/history/${encodeURIComponent(item.name)}/share`, {
        shared: !item.shared,
      });
      if (res && res.status === "ok") {
        item.shared = res.shared;
        this.toast("success", item.shared ? "Картинка добавлена в общую историю 🌐" : "Картинка скрыта из общей истории 🔒");
        this.openHistoryModal(item);
        this.renderHistory();
      }
    } catch (e) {
      this.toast("error", "Не удалось изменить статус доступа: " + e.message);
    }
  },

  toggleHistoryPrompt() {
    const box = document.getElementById("history-modal-prompt-box");
    const btn = document.getElementById("history-prompt-toggle");
    if (!box || !btn) return;
    const isCollapsed = box.classList.toggle("collapsed");
    btn.textContent = isCollapsed ? "Развернуть ▾" : "Свернуть ▴";
  },

  closeHistoryModal() {
    const modal = document.getElementById("history-modal");
    if (modal) modal.style.display = "none";
    this.state.selectedHistoryItem = null;
  },

  // ---------- Hugging Face Token & Guest Accounts ----------
  toggleTokenVisibility(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;
    input.type = input.type === "password" ? "text" : "password";
  },

  async saveHfToken() {
    const token = document.getElementById("setting-hf-token")?.value?.trim() || "";
    try {
      await this.apiPatch("/api/settings", { huggingface_token: token });
      if (this.state.settings) this.state.settings.huggingface_token = token;
      this.toast("success", t("settings.network.token_saved"));
    } catch (e) {
      this.toast("error", t("settings.network.token_error") + e.message);
    }
  },

  async testHfToken() {
    const token = document.getElementById("setting-hf-token")?.value?.trim();
    const btn = document.getElementById("btn-test-hf-token");
    if (btn) btn.textContent = t("settings.network.testing");
    try {
      const res = await this.apiPost("/api/hf/test-token", { token: token || undefined });
      if (res.valid) {
        this.toast("success", res.message || t("settings.network.token_valid"));
      } else {
        this.toast("error", res.message || t("settings.network.token_invalid"));
      }
    } catch (e) {
      this.toast("error", t("settings.network.test_error") + e.message);
    } finally {
      if (btn) btn.textContent = t("settings.network.test");
    }
  },

  async loadGuestAccounts() {
    if (this.state.account?.is_guest) return;
    try {
      const res = await this.apiGet("/api/accounts");
      if (res && res.accounts) {
        this.state.guestAccounts = res.accounts;
        this.renderGuestAccounts(res.accounts);
      }
    } catch (e) {
      console.warn("[accounts] Could not load guest accounts:", e);
    }
  },

  renderGuestAccounts(accounts) {
    const list = document.getElementById("guest-accounts-list");
    if (!list) return;

    if (!accounts || !accounts.length) {
      list.innerHTML = `
        <div style="color: var(--text-dim); font-size: 12.5px; padding: 12px; text-align: center; background: var(--bg-card); border-radius: 8px; border: 1px dashed var(--border);">
          ${escapeHtml(t("settings.accounts.empty"))}
        </div>
      `;
      return;
    }

    list.innerHTML = "";
    for (const acc of accounts) {
      const card = document.createElement("div");
      card.className = "guest-account-card";

      const createdDate = acc.created_at ? new Date(acc.created_at * 1000).toLocaleDateString() : "";
      const pwdDisplay = acc.initial_password
        ? `<span style="font-family: monospace; color: #a78bfa;">${escapeHtml(acc.initial_password)}</span>`
        : `<span style="color: var(--text-dim); font-size: 11px;">${escapeHtml(t("settings.accounts.password_changed"))}</span>`;

      card.innerHTML = `
        <div class="guest-account-info">
          <div class="guest-account-name">
            <span>👤 ${escapeHtml(acc.username)}</span>
            <small style="color: var(--text-dim); font-weight: normal; font-size: 11px;">(${createdDate})</small>
          </div>
          <div class="guest-account-token" style="display: flex; gap: 12px; font-size: 11px;">
            <span>${escapeHtml(t("settings.accounts.password_label"))} ${pwdDisplay}</span>
            <span style="color: var(--text-dim);">${escapeHtml(t("settings.accounts.token_label"))} ${escapeHtml((acc.token || '').slice(0, 10))}...</span>
          </div>
        </div>
        <div class="guest-account-actions">
          <button type="button" class="setting-btn" onclick="App.copyGuestCredentials('${escapeHtml(acc.username)}', '${escapeHtml(acc.initial_password || '')}')" title="${escapeHtml(t("settings.accounts.copy_data_title"))}">${escapeHtml(t("settings.accounts.copy_data"))}</button>
          <button type="button" class="setting-btn" onclick="App.showCreateGuestModal('${acc.id}', '${escapeHtml(acc.username)}', '${escapeHtml(acc.initial_password || '')}')" title="${escapeHtml(t("settings.accounts.edit_title"))}">✏️</button>
          <button type="button" class="setting-btn danger" onclick="App.deleteGuestAccount('${acc.id}')" title="${escapeHtml(t("settings.accounts.delete_title"))}">🗑️</button>
        </div>
      `;
      list.appendChild(card);
    }
  },

  generateRandomGuestPassword() {
    const chars = "abcdefghjkmnpqrstuvwxyz23456789";
    let pass = "guest_";
    for (let i = 0; i < 6; i++) {
      pass += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    const pwdInput = document.getElementById("guest-account-password-input");
    if (pwdInput) pwdInput.value = pass;
    return pass;
  },

  showCreateGuestModal(accountId = null, currentName = "", currentPassword = "") {
    const modal = document.getElementById("guest-account-modal");
    const title = document.getElementById("guest-modal-title");
    const idInput = document.getElementById("guest-account-edit-id");
    const nameInput = document.getElementById("guest-account-username-input");
    const pwdInput = document.getElementById("guest-account-password-input");
    if (!modal) return;

    if (accountId) {
      if (title) title.textContent = t("settings.accounts.edit_modal_title");
      if (idInput) idInput.value = accountId;
      if (nameInput) nameInput.value = currentName || "";
      if (pwdInput) pwdInput.value = currentPassword || "";
    } else {
      if (title) title.textContent = t("settings.accounts.new_modal_title");
      if (idInput) idInput.value = "";
      if (nameInput) nameInput.value = "";
      this.generateRandomGuestPassword();
    }
    modal.style.display = "flex";
    if (nameInput) nameInput.focus();
  },

  closeCreateGuestModal() {
    const modal = document.getElementById("guest-account-modal");
    if (modal) modal.style.display = "none";
  },

  async confirmCreateGuestAccount() {
    const idInput = document.getElementById("guest-account-edit-id");
    const nameInput = document.getElementById("guest-account-username-input");
    const pwdInput = document.getElementById("guest-account-password-input");
    const username = (nameInput?.value || "").trim();
    const password = (pwdInput?.value || "").trim();

    if (!username) {
      this.toast("error", t("settings.accounts.name_required"));
      return;
    }
    const editId = idInput?.value;
    try {
      if (editId) {
        await this.apiPatch(`/api/accounts/${encodeURIComponent(editId)}`, { username, password });
        this.toast("success", t("settings.accounts.updated"));
      } else {
        const res = await this.apiPost("/api/accounts", { username, password });
        this.toast("success", t("settings.accounts.created", { name: username }));
        if (res && res.account) {
          this.copyGuestCredentials(username, password || res.account.initial_password);
        }
      }
      this.closeCreateGuestModal();
      await this.loadGuestAccounts();
    } catch (e) {
      this.toast("error", "Ошибка: " + e.message);
    }
  },

  async deleteGuestAccount(accountId) {
    if (!confirm(t("settings.accounts.delete_confirm"))) return;
    try {
      await this.apiDelete(`/api/accounts/${encodeURIComponent(accountId)}`);
      this.toast("success", t("settings.accounts.deleted"));
      await this.loadGuestAccounts();
    } catch (e) {
      this.toast("error", t("settings.accounts.delete_failed") + e.message);
    }
  },

  copyGuestCredentials(username, password) {
    // Determine active tunnel url or current host
    const tunnelInput = document.getElementById("setting-share-url");
    let tunnelUrl = tunnelInput && tunnelInput.value && !tunnelInput.value.includes("...")
      ? tunnelInput.value.trim()
      : window.location.origin;

    // Strip any query parameters or trailing slashes to guarantee clean URL
    try {
      const parsed = new URL(tunnelUrl);
      tunnelUrl = `${parsed.protocol}//${parsed.host}`;
    } catch (_) {
      tunnelUrl = tunnelUrl.split("?")[0].replace(/\/+$/, "");
    }

    const pwdText = password ? password : t("settings.accounts.saved_password");
    const text = t("settings.accounts.credentials_text", { url: tunnelUrl, username, password: pwdText });

    navigator.clipboard.writeText(text)
      .then(() => this.toast("success", t("settings.accounts.credentials_copied")))
      .catch(() => {
        prompt(t("settings.accounts.credentials_prompt"), text);
      });
  },

  copyGuestLink() {
    const url = window.location.origin;
    navigator.clipboard.writeText(url)
      .then(() => this.toast("success", t("settings.accounts.link_copied")))
      .catch(() => this.toast("info", `Ссылка: ${url}`));
  },

  async reuseHistoryItem(item) {
    if (!item) return;

    // Set prompt
    const promptInput = document.getElementById("prompt-input");
    if (promptInput && item.prompt) {
      promptInput.value = item.prompt;
    }

    // Set model and await variant rendering first so steps/CFG aren't overwritten
    if (item.model_id) {
      const modelSelect = document.getElementById("model-select");
      if (modelSelect) {
        modelSelect.value = item.model_id;
        await this.onModelChange(item.variant_id || null);
      }
    }

    // Set resolution
    if (item.width) {
      const resSelect = document.getElementById("resolution-select");
      if (resSelect) {
        const combo = `${item.width}x${item.height || item.width}`;
        const square = item.width.toString();
        if (resSelect.querySelector(`option[value="${combo}"]`)) {
          resSelect.value = combo;
        } else if (resSelect.querySelector(`option[value="${square}"]`)) {
          resSelect.value = square;
        }
      }
    }

    // Set negative prompt
    const negPromptInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negPromptInput) {
      negPromptInput.value = item.negative_prompt || "";
      if (negDot) {
        negDot.style.display = (item.negative_prompt && item.negative_prompt.trim().length > 0) ? "inline-block" : "none";
      }
    }

    // Set steps
    if (item.steps) {
      const stepsInput = document.getElementById("steps-input");
      if (stepsInput) stepsInput.value = item.steps;
    }

    // Set CFG
    if (item.cfg !== undefined && item.cfg !== null) {
      const cfgInput = document.getElementById("cfg-input");
      const cfgSlider = document.getElementById("cfg-slider");
      if (cfgInput) cfgInput.value = item.cfg;
      if (cfgSlider) cfgSlider.value = item.cfg;
    }

    // Set Noise
    if (item.noise !== undefined && item.noise !== null) {
      const noiseInput = document.getElementById("noise-input");
      const noiseSlider = document.getElementById("noise-slider");
      if (noiseInput) noiseInput.value = item.noise;
      if (noiseSlider) noiseSlider.value = item.noise;
    }

    // Set Sampler
    if (item.sampler) {
      const samplerSelect = document.getElementById("sampler-select");
      if (samplerSelect && samplerSelect.querySelector(`option[value="${item.sampler}"]`)) {
        samplerSelect.value = item.sampler;
      }
    }

    // Set VAE
    if (item.vae) {
      const vaeSelect = document.getElementById("vae-select");
      if (vaeSelect && vaeSelect.querySelector(`option[value="${item.vae}"]`)) {
        vaeSelect.value = item.vae;
        this.state.selectedVae = item.vae;
      }
    }

    // Set seed
    if (item.seed !== null && item.seed !== undefined) {
      const seedInput = document.getElementById("seed-input");
      if (seedInput) seedInput.value = item.seed;
    }

    this.updateModelTrigger();

    // Show preview of selected image in preview area
    this.showPreview(item.name);

    // Switch to generate tab
    this.switchTab("generate");
    this.toast("info", t("history.reuse") + " ✓");
  },

  async deleteHistoryItem(filename) {
    try {
      const res = await fetch(`/api/history/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.toast("success", t("common.delete") + " ✓");
      await this.loadHistory();
    } catch (e) {
      console.error("[app] Delete history error:", e);
      this.toast("error", t("errors.history_delete_failed") + ": " + e.message);
    }
  },

  async openFolder(folderType = "outputs", filename = null) {
    if (!filename && folderType === "outputs") {
      filename = this.state.currentPreviewFilename || null;
      if (!filename) {
        try {
          const previewImg = document.getElementById("preview-img");
          if (previewImg && previewImg.src) {
            const u = new URL(previewImg.src, window.location.href);
            const parts = u.pathname.split("/");
            const last = parts[parts.length - 1];
            if (last && (last.endsWith(".png") || last.endsWith(".jpg") || last.endsWith(".webp"))) {
              filename = decodeURIComponent(last);
            }
          }
        } catch (e) {}
      }
    }

    try {
      const res = await this.apiPost("/api/system/open-folder", {
        folder_type: folderType,
        filename: filename,
      });
      if (res.status === "ok") {
        const label = filename
          ? t("models.file_opened", { name: filename })
          : t("models.folder_opened", { path: res.folder || res.path || "" });
        this.toast("success", "📂 " + label);
      } else if (res.status === "remote") {
        this.toast("info", t("models.remote_folder"));
        this.switchTab("history");
      } else {
        this.toast("error", res.message || "Не удалось открыть папку");
      }
    } catch (e) {
      console.warn("openFolder error:", e);
      this.toast("error", "Не удалось открыть папку: " + e.message);
    }
  },

  // ---------- Toast notifications ----------
  toast(kind, message) {
    let container = document.querySelector(".toast-container");
    if (!container) {
      container = document.createElement("div");
      container.className = "toast-container";
      document.body.appendChild(container);
    }
    const toast = document.createElement("div");
    toast.className = `toast ${kind}`;
    toast.textContent = message;
    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      toast.style.transition = "all 200ms";
      setTimeout(() => toast.remove(), 200);
    }, 4000);
  },

  // ---------- Img2Img / Native ControlNet Reference ----------
  onRefModeChange(preserveStrength = false) {
    const mode = document.getElementById("img2img-mode")?.value || "control_ref";
    const hintEl = document.getElementById("img2img-mode-hint");
    const statusDot = document.getElementById("img2img-status-dot");
    const strengthLabel = document.getElementById("img2img-strength-label");
    const strengthSlider = document.getElementById("img2img-strength");
    const strengthVal = document.getElementById("img2img-strength-val");

    if (mode === "control_ref") {
      const tipText = "✓ Изоляция шагов активна: --init-img и --strength отключены. Модель выполнит 100% шагов с чистого листа, перенося лайн и скринтоны через нативный Reference / ControlNet.";
      if (hintEl) {
        hintEl.style.color = "var(--success, #22c55e)";
        hintEl.innerHTML = "✓ Изоляция шагов активна: <code>--init-img</code> и <code>--strength</code> отключены. Модель выполнит 100% шагов с чистого листа, перенося лайн и скринтоны через нативный Reference / ControlNet.";
      }
      if (statusDot) {
        statusDot.style.background = "#10B981";
        statusDot.title = tipText;
      }
      if (strengthLabel) strengthLabel.textContent = "Сила:";
      if (strengthSlider) {
        strengthSlider.max = "1.00";
        if (!preserveStrength) strengthSlider.value = "0.85";
      }
      if (strengthVal && !preserveStrength) strengthVal.textContent = "0.85";
    } else if (mode === "controlnet_canny") {
      const tipText = "✒️ Canny / Lineart препроцессор: выделяет контуры туши и передает в ControlNet / Reference при 100% шагов с чистого белого шума.";
      if (hintEl) {
        hintEl.style.color = "var(--info, #38bdf8)";
        hintEl.innerHTML = tipText;
      }
      if (statusDot) {
        statusDot.style.background = "#38bdf8";
        statusDot.title = tipText;
      }
      if (strengthLabel) strengthLabel.textContent = "Контур:";
      if (strengthSlider) {
        strengthSlider.max = "1.00";
        if (!preserveStrength) strengthSlider.value = "0.90";
      }
      if (strengthVal && !preserveStrength) strengthVal.textContent = "0.90";
    } else {
      const tipText = "⚠️ Классический Img2Img (зашумление исходника через --init-img + --strength). Может размывать белый фон и тонкие скринтоны.";
      if (hintEl) {
        hintEl.style.color = "var(--warning, #fbbf24)";
        hintEl.innerHTML = "⚠️ Классический Img2Img (зашумление исходника через <code>--init-img</code> + <code>--strength</code>). Может размывать белый фон и тонкие скринтоны.";
      }
      if (statusDot) {
        statusDot.style.background = "#F59E0B";
        statusDot.title = tipText;
      }
      if (strengthLabel) strengthLabel.textContent = "Шум:";
      if (strengthSlider) {
        strengthSlider.max = "0.95";
        if (!preserveStrength) strengthSlider.value = "0.70";
      }
      if (strengthVal && !preserveStrength) strengthVal.textContent = "0.70";
    }
  },

  toggleControlNetInput(e) {
    if (e) e.preventDefault();
    const wrap = document.getElementById("controlnet-custom-wrap");
    const input = document.getElementById("controlnet-custom-path");
    if (!input) return;
    if (wrap) {
      const isHidden = wrap.style.display === "none";
      wrap.style.display = isHidden ? "block" : "none";
      input.style.display = isHidden ? "block" : "none";
      if (isHidden) input.focus();
    } else {
      const isHidden = input.style.display === "none";
      input.style.display = isHidden ? "block" : "none";
      if (isHidden) input.focus();
    }
  },

  onDockFileSelected(inputEl) {
    const file = inputEl?.files?.[0];
    if (file) {
      this.uploadImg2ImgFile(file);
      inputEl.value = "";
    }
  },

  async uploadImg2ImgFile(file) {
    if (!file) return;

    const ext = "." + (file.name ? file.name.split(".").pop().toLowerCase() : "");
    const validExts = [".png", ".jpg", ".jpeg", ".webp"];
    const validMimes = ["image/png", "image/jpeg", "image/webp"];
    if (!validMimes.includes(file.type) && !validExts.includes(ext)) {
      this.toast("error", "Unsupported file format. Allowed: PNG, JPEG, WEBP");
      return;
    }
    const sizeMb = file.size / 1024 / 1024;
    if (sizeMb > 25) {
      this.toast("error", `File too large (${sizeMb.toFixed(1)} MB). Maximum allowed size is 25 MB.`);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    this.toast("info", t("common.loading"));
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      this.setImg2ImgRef({
        path: data.path,
        url: data.url,
        filename: data.filename,
      });
      this.toast("success", t("img2img.title") + " ✓");
    } catch (e) {
      console.error("[app] Upload failed:", e);
      this.toast("error", t("errors.upload_failed") + ": " + e.message);
    }
  },

  setImg2ImgRef(refData) {
    this.state.img2imgRef = refData;
    const container = document.getElementById("img2img-container");
    const thumb = document.getElementById("img2img-thumb");

    if (refData && refData.url) {
      if (thumb) thumb.src = refData.url;
      if (container) container.style.display = "flex";
      this.onRefModeChange(true);
      this.switchTab("generate");
    } else {
      if (container) container.style.display = "none";
    }
    this.updateNewDock?.(this);
    if (this.state.genMask?.enabled && refData?.url) {
      this.setGenMaskSource({
        url: refData.url,
        path: refData.path,
        name: refData.filename || refData.path,
      });
    }
  },

  clearImg2ImgRef() {
    this.state.img2imgRef = null;
    const container = document.getElementById("img2img-container");
    if (container) container.style.display = "none";
    this.updateNewDock?.(this);
    if (this.state.genMask?.enabled) {
      if (this.state.currentPreviewUrl && this.state.currentPreviewFilename) {
        this.setGenMaskSource({
          url: this.state.currentPreviewUrl,
          path: this.state.currentPreviewFilename,
          name: this.state.currentPreviewFilename,
        });
      } else {
        this.clearGenMaskSource();
      }
    }
  },

  updateStrengthLabel(val) {
    const label = document.getElementById("img2img-strength-val");
    if (label) label.textContent = parseFloat(val).toFixed(2);
  },

  sendCurrentPreviewToImg2Img() {
    if (!this.state.lastPreviewName && !this.state.lastPreviewUrl) {
      const img = document.getElementById("preview-img");
      if (img && img.src && !img.src.endsWith("#")) {
        const src = img.src;
        const cleanSrc = src.split("?")[0];
        const filename = cleanSrc.split("/").pop();
        if (filename) {
          this.setImg2ImgRef({
            path: filename,
            url: src,
            filename: filename,
          });
          this.toast("success", t("img2img.send_to") + " ✓");
          return;
        }
      }
      this.toast("warning", t("history.empty"));
      return;
    }
    const cleanUrl = this.state.lastPreviewUrl ? this.state.lastPreviewUrl.split("?")[0] : "";
    const cleanFilename = this.state.lastPreviewName ? this.state.lastPreviewName.split("?")[0] : (cleanUrl ? cleanUrl.split("/").pop() : "");
    this.setImg2ImgRef({
      path: cleanFilename,
      url: this.state.lastPreviewUrl,
      filename: cleanFilename,
    });
    this.toast("success", t("img2img.send_to") + " ✓");
  },

  // ---------- Gemini Floating Dock & Popovers ----------
  autoResizePrompt(textarea) {
    if (!textarea) return;
    textarea.style.height = "auto";
    const nextH = Math.max(46, Math.min(150, textarea.scrollHeight));
    textarea.style.height = `${nextH}px`;
  },

  updateDockPills() {
    // 1. Model Pill Text
    const modelPillText = document.getElementById("dock-model-pill-text");
    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (modelPillText) {
      if (activeModel) {
        const shortVar = (this.state.selectedVariant || "")
          .replace(/^(flux2_klein_|flux3_klein_|qwen_image_2_1_)/i, "")
          .replace(/-(?:50x|40x|30x-below)(?:-base)?/g, "");
        modelPillText.textContent = shortVar ? `${activeModel.name} (${shortVar})` : activeModel.name;
      } else {
        modelPillText.textContent = "Модель";
      }
    }

    // 2. Resolution Pill Text & Chips
    const resVal = document.getElementById("resolution-select")?.value || "1024x1024";
    const resPillText = document.getElementById("dock-res-pill-text");
    if (resPillText) {
      resPillText.textContent = resVal.replace("x", "×");
    }
    document.querySelectorAll("#dock-res-grid .dock-res-chip").forEach(chip => {
      chip.classList.toggle("active", chip.dataset.res === resVal);
    });

    // 3. Params Pill Text
    const samplerVal = document.getElementById("sampler-select")?.value || "euler";
    const stepsVal = document.getElementById("steps-input")?.value || "4";
    const cfgVal = document.getElementById("cfg-input")?.value || "1.0";
    const noiseVal = document.getElementById("noise-input")?.value || "0.0";
    const seedVal = document.getElementById("seed-input")?.value || "";
    const paramsPillText = document.getElementById("dock-params-pill-text");
    if (paramsPillText) {
      const sLabel = samplerVal === "euler_a" ? "Euler A" : (samplerVal === "heun" ? "Heun" : "Euler");
      paramsPillText.textContent = `${sLabel} · ${stepsVal} · CFG ${parseFloat(cfgVal || 1).toFixed(1)}`;
    }

    // 4. Negative Prompt Pill State
    const negVal = document.getElementById("negative-prompt-input")?.value?.trim() || "";
    const negPill = document.getElementById("dock-neg-pill");
    if (negPill) {
      negPill.classList.toggle("has-neg", negVal.length > 0);
    }

    // 5. Sync Quick Popover Selects with Inspector Controls
    const mainModelSel = document.getElementById("model-select");
    const dockModelSel = document.getElementById("dock-model-quick-select");
    if (mainModelSel && dockModelSel) {
      dockModelSel.innerHTML = mainModelSel.innerHTML;
      dockModelSel.value = mainModelSel.value;
    }

    const mainVarSel = document.getElementById("variant-select");
    const dockVarSel = document.getElementById("dock-variant-quick-select");
    if (mainVarSel && dockVarSel) {
      dockVarSel.innerHTML = mainVarSel.innerHTML;
      dockVarSel.value = mainVarSel.value;
    }

    const mainTeGroup = document.getElementById("text-encoder-group");
    const mainTeSel = document.getElementById("text-encoder-select");
    const dockTeRow = document.getElementById("dock-te-quick-field") || document.getElementById("dock-te-quick-row");
    const dockTeSel = document.getElementById("dock-te-quick-select");
    if (mainTeGroup && mainTeSel && dockTeRow && dockTeSel) {
      const visible = mainTeGroup.style.display !== "none";
      dockTeRow.style.display = visible ? "flex" : "none";
      if (visible) {
        dockTeSel.innerHTML = mainTeSel.innerHTML;
        dockTeSel.value = mainTeSel.value;
        dockTeSel.disabled = mainTeSel.disabled;
      }
    }

    const mainVaeGroup = document.getElementById("vae-group");
    const mainVaeSel = document.getElementById("vae-select");
    const dockVaeRow = document.getElementById("dock-vae-quick-field");
    const dockVaeSel = document.getElementById("dock-vae-quick-select");
    if (mainVaeGroup && mainVaeSel && dockVaeRow && dockVaeSel) {
      const visible = mainVaeGroup.style.display !== "none";
      dockVaeRow.style.display = visible ? "flex" : "none";
      if (visible) {
        dockVaeSel.innerHTML = mainVaeSel.innerHTML;
        dockVaeSel.value = mainVaeSel.value;
        dockVaeSel.disabled = mainVaeSel.disabled;
      }
    }

    const dockSampler = document.getElementById("dock-sampler-quick");
    if (dockSampler) dockSampler.value = samplerVal;
    const dockSteps = document.getElementById("dock-steps-quick");
    if (dockSteps) dockSteps.value = stepsVal;
    const dockCfg = document.getElementById("dock-cfg-quick");
    if (dockCfg) dockCfg.value = cfgVal;
    const dockCfgLbl = document.getElementById("dock-cfg-val-label");
    if (dockCfgLbl) dockCfgLbl.textContent = cfgVal;
    const dockNoise = document.getElementById("dock-noise-quick");
    if (dockNoise) dockNoise.value = noiseVal;
    const dockNoiseLbl = document.getElementById("dock-noise-val-label");
    if (dockNoiseLbl) dockNoiseLbl.textContent = noiseVal;
    const dockSeed = document.getElementById("dock-seed-quick");
    if (dockSeed) dockSeed.value = seedVal;
    this.updateNewDock(this);
  },

  toggleDockPopover(name, event) {
    if (event) event.stopPropagation();
    const map = {
      model: "dock-popover-model",
      res: "dock-popover-res",
      params: "dock-popover-params",
      presets: "dock-popover-presets",
    };
    const targetId = map[name];
    if (!targetId) return;

    if (this.state.activeDockPopover === name) {
      this.closeAllDockPopovers();
      return;
    }

    this.closeAllDockPopovers();
    const el = document.getElementById(targetId);
    if (el) {
      this.updateDockPills();
      if (name === "presets") this.renderPresetsUI();
      el.classList.remove("dock-popover-below");
      el.style.display = "block";
      this.state.activeDockPopover = name;

      // Keep the popover inside the viewport without changing dock geometry.
      const wrap = document.getElementById("gemini-dock-wrap");
      const dock = document.getElementById("gemini-dock");
      if (wrap && dock) {
        const dockRect = dock.getBoundingClientRect();
        const spaceAbove = dockRect.top - 12;
        const spaceBelow = window.innerHeight - dockRect.bottom - 12;
        if (el.offsetHeight > spaceAbove && spaceBelow > spaceAbove) {
          el.classList.add("dock-popover-below");
        }
      }
    }
  },

  closeAllDockPopovers() {
    ["dock-popover-model", "dock-popover-res", "dock-popover-params", "dock-popover-presets"].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.style.display = "none";
        el.classList.remove("dock-popover-below");
      }
    });
    this.state.activeDockPopover = null;
  },

  async onDockModelSelect(modelId) {
    const mainModelSel = document.getElementById("model-select");
    if (mainModelSel) {
      mainModelSel.value = modelId;
      await this.onModelChange();
    }
  },

  onDockVariantSelect(variantId) {
    const mainVarSel = document.getElementById("variant-select");
    if (mainVarSel) {
      mainVarSel.value = variantId;
      mainVarSel.dispatchEvent(new Event("change"));
    }
  },

  onDockTeSelect(teId) {
    const mainTeSel = document.getElementById("text-encoder-select");
    if (mainTeSel) {
      mainTeSel.value = teId;
      this.onTextEncoderChange();
    }
  },

  onDockVaeSelect(vaeValue) {
    const mainVaeSel = document.getElementById("vae-select");
    if (mainVaeSel) {
      mainVaeSel.value = vaeValue;
      this.onVaeChange();
      this.updateDockPills();
    }
  },

  selectResolutionChip(resValue) {
    const resSel = document.getElementById("resolution-select");
    if (resSel) {
      resSel.value = resValue;
      this.updateDockPills();
    }
    this.closeAllDockPopovers();
  },

  syncFromDockParams(field, val) {
    if (field === "sampler") {
      const el = document.getElementById("sampler-select");
      if (el) el.value = val;
    } else if (field === "steps") {
      const el = document.getElementById("steps-input");
      if (el) el.value = val;
    } else if (field === "cfg") {
      const inp = document.getElementById("cfg-input");
      const sld = document.getElementById("cfg-slider");
      const lbl = document.getElementById("dock-cfg-val-label");
      if (inp) inp.value = val;
      if (sld) sld.value = val;
      if (lbl) lbl.textContent = val;
    } else if (field === "noise") {
      const inp = document.getElementById("noise-input");
      const sld = document.getElementById("noise-slider");
      const lbl = document.getElementById("dock-noise-val-label");
      if (inp) inp.value = val;
      if (sld) sld.value = val;
      if (lbl) lbl.textContent = val;
      const n = parseFloat(val);
      const sSel = document.getElementById("sampler-select");
      if (!isNaN(n) && n > 0 && sSel && sSel.value === "euler") {
        sSel.value = "euler_a";
      }
    } else if (field === "seed") {
      const el = document.getElementById("seed-input");
      if (el) el.value = val;
    }
    this.updateModelTrigger();
  },

  // ---------- Presets / Templates System ----------
  async loadPresets() {
    try {
      const res = await this.apiGet("/api/presets");
      this.state.presets = (res && res.presets) ? res.presets : (Array.isArray(res) ? res : []);
      this.renderPresetsUI();
    } catch (e) {
      console.warn("[app] Failed to load presets:", e);
      this.state.presets = [];
      this.renderPresetsUI();
    }
  },

  renderPresetsUI() {
    const presets = this.state.presets || [];

    // 1. Render inside Gemini Dock Popover (#dock-presets-list)
    const dockList = document.getElementById("dock-presets-list");
    if (dockList) {
      dockList.innerHTML = "";
      if (presets.length === 0) {
        dockList.innerHTML = `
          <div style="padding: 16px; text-align: center; color: var(--text-dim); font-size: 12px;">
            ${escapeHtml(t("templates.empty_dock"))}
            <div style="margin-top: 10px;">
              <button type="button" class="generate-btn" style="height: 30px; padding: 0 14px; font-size: 11.5px;" onclick="App.openSavePresetModal(); App.closeAllDockPopovers();">${escapeHtml(t("templates.save_current_dock"))}</button>
            </div>
          </div>
        `;
      }
      for (const p of presets) {
        const row = document.createElement("div");
        row.className = "dock-preset-item";
        const editBtn = p.can_edit
          ? `<button class="history-action-btn" style="padding: 2px 6px; font-size: 11px; height: 22px; width: auto;" onclick="App.openEditPresetModal('${escapeHtml(p.id)}', event)" title="${escapeHtml(t("templates.edit_title"))}">✏️</button>`
          : "";
        const delBtn = p.can_delete
          ? `<button class="delete-custom-btn" onclick="App.deletePreset('${escapeHtml(p.id)}', event)" title="${escapeHtml(t("templates.delete_title"))}">🗑</button>`
          : "";
        row.innerHTML = `
          <div class="dock-preset-info">
            <span class="dock-preset-icon">${escapeHtml(p.icon || "✨")}</span>
            <div style="min-width: 0;">
              <div class="dock-preset-title">${escapeHtml(p.title)}</div>
              <div class="dock-preset-sub">${escapeHtml(p.description || "")}</div>
            </div>
          </div>
          <div style="display: flex; gap: 4px; align-items: center;">
            ${editBtn}
            ${delBtn}
          </div>
        `;
        row.addEventListener("click", (e) => {
          if (e.target.tagName === "BUTTON") return;
          this.applyPreset(p.id);
          this.closeAllDockPopovers();
        });
        dockList.appendChild(row);
      }
    }

    // 2. Render inside Templates Tab (#templates-grid)
    const grid = document.getElementById("templates-grid");
    if (grid) {
      grid.innerHTML = "";
      for (const p of presets) {
        const card = document.createElement("div");
        card.className = "model-card";
        const ownerBadge = p.owner_id === "host"
          ? `<span class="badge-installed">${escapeHtml(t("templates.host_badge"))}</span>`
          : `<span class="badge-custom">👤 ${escapeHtml(p.owner_name || t("common.guest"))}</span>`;
        const privBadge = p.is_private
          ? `<span class="badge-custom" style="background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); color: #f87171;">${escapeHtml(t("templates.private_badge"))}</span>`
          : "";
        const editBtn = p.can_edit
          ? `<button class="card-action-btn" style="width: auto;" onclick="App.openEditPresetModal('${escapeHtml(p.id)}')">${escapeHtml(t("templates.edit"))}</button>`
          : "";
        const delBtn = p.can_delete
          ? `<button class="delete-custom-btn" onclick="App.deletePreset('${escapeHtml(p.id)}', event)">${escapeHtml(t("templates.delete_btn"))}</button>`
          : "";
        card.innerHTML = `
          <div>
            <div class="model-card-header">
              <span class="model-card-title">${escapeHtml(p.icon || "✨")} ${escapeHtml(p.title)}</span>
              <div class="model-card-badges" style="display: flex; gap: 6px; flex-wrap: wrap;">
                ${ownerBadge}
                ${privBadge}
              </div>
            </div>
            <p class="model-card-description" style="margin-top: 8px;">${escapeHtml(p.description || "")}</p>
            <div class="model-card-reqs" style="margin-top: 10px;">
              <span>📐 ${escapeHtml(p.resolution || "1024x1024")}</span>
              <span>⏱ ${escapeHtml(t("templates.steps", { count: p.steps || 20 }))}</span>
              <span>🎛️ CFG ${p.cfg ?? 3.5}</span>
              <span>🖼️ ${escapeHtml(p.ref_mode || "control_ref")}</span>
            </div>
            ${p.prompt ? `<div style="margin-top: 10px; padding: 8px 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; font-size: 11px; color: var(--text-muted); max-height: 64px; overflow: hidden;">${escapeHtml(p.prompt)}</div>` : ""}
            <div style="margin-top: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
              <button class="card-action-btn primary" style="width: auto;" onclick="App.applyPreset('${escapeHtml(p.id)}')">${escapeHtml(t("templates.apply_btn"))}</button>
              ${editBtn}
              ${delBtn}
            </div>
          </div>
        `;
        grid.appendChild(card);
      }
    }
  },

  async applyPreset(presetId) {
    const preset = (this.state.presets || []).find(p => p.id === presetId);
    if (!preset) return;

    // 1. Model & Variant (if specified and available)
    if (preset.model_id) {
      const targetModel = this.state.models.find(m => m.id === preset.model_id && !m.disabled);
      if (targetModel) {
        const modelSel = document.getElementById("model-select");
        if (modelSel) {
          modelSel.value = targetModel.id;
          await this.onModelChange(preset.variant_id || null);
        }
      }
    }

    // 2. Prompt & Negative Prompt
    const promptInput = document.getElementById("prompt-input");
    if (promptInput && preset.prompt) {
      const cur = promptInput.value.trim();
      if (preset.prompt_mode === "append" && cur) {
        promptInput.value = `${cur}, ${preset.prompt}`;
      } else {
        promptInput.value = preset.prompt;
      }
      this.autoResizePrompt(promptInput);
    }

    const negInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negInput && preset.negative_prompt !== undefined) {
      negInput.value = preset.negative_prompt || "";
      if (negDot) {
        negDot.style.display = negInput.value.trim().length > 0 ? "inline-block" : "none";
      }
    }

    // 3. Resolution, Steps, CFG, Sampler, Noise
    if (preset.resolution) {
      const resSel = document.getElementById("resolution-select");
      if (resSel && resSel.querySelector(`option[value="${preset.resolution}"]`)) {
        resSel.value = preset.resolution;
      }
    }
    if (preset.steps) {
      const stepsEl = document.getElementById("steps-input");
      if (stepsEl) stepsEl.value = preset.steps;
    }
    if (preset.cfg !== undefined && preset.cfg !== null) {
      const cfgInp = document.getElementById("cfg-input");
      const cfgSld = document.getElementById("cfg-slider");
      if (cfgInp) cfgInp.value = preset.cfg;
      if (cfgSld) cfgSld.value = preset.cfg;
    }
    if (preset.sampler) {
      const sampEl = document.getElementById("sampler-select");
      if (sampEl && sampEl.querySelector(`option[value="${preset.sampler}"]`)) {
        sampEl.value = preset.sampler;
      }
    }
    if (preset.noise !== undefined && preset.noise !== null) {
      const nInp = document.getElementById("noise-input");
      const nSld = document.getElementById("noise-slider");
      if (nInp) nInp.value = preset.noise;
      if (nSld) nSld.value = preset.noise;
    }

    // 4. Reference / ControlNet mode & strength
    if (preset.ref_mode) {
      const modeEl = document.getElementById("img2img-mode");
      if (modeEl && modeEl.querySelector(`option[value="${preset.ref_mode}"]`)) {
        modeEl.value = preset.ref_mode;
        this.onRefModeChange(true);
      }
    }
    if (preset.strength !== undefined && preset.strength !== null) {
      const strSld = document.getElementById("img2img-strength");
      const strVal = document.getElementById("img2img-strength-val");
      if (strSld) strSld.value = preset.strength;
      if (strVal) strVal.textContent = Number(preset.strength).toFixed(2);
    }
    this.updateNewDock?.(this);

    this.updateModelTrigger();
    if (this.state.currentTab !== "generate") {
      this.switchTab("generate");
    }
    this.toast("success", `Шаблон «${preset.title}» применён ✓`);
  },

  openSavePresetModal() {
    this.closeAllDockPopovers();
    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "flex";

    const idInput = document.getElementById("preset-save-id");
    if (idInput) idInput.value = "";
    const titleHeader = document.getElementById("preset-modal-title");
    if (titleHeader) titleHeader.textContent = "⭐ Сохранить шаблон (Preset)";
    const titleInput = document.getElementById("preset-save-name");
    if (titleInput) {
      titleInput.value = "";
      titleInput.focus();
    }
    const iconInput = document.getElementById("preset-save-icon");
    if (iconInput) iconInput.value = "⭐";
    const catInput = document.getElementById("preset-save-category");
    if (catInput) catInput.value = "Пользовательские";
    const descInput = document.getElementById("preset-save-desc");
    if (descInput) descInput.value = "";

    const promptPre = document.getElementById("preset-save-prompt");
    if (promptPre) promptPre.value = document.getElementById("prompt-input")?.value || "";
    const negPre = document.getElementById("preset-save-neg");
    if (negPre) negPre.value = document.getElementById("negative-prompt-input")?.value || "";

    const stepsPre = document.getElementById("preset-save-steps");
    if (stepsPre) stepsPre.value = document.getElementById("steps-input")?.value || "20";
    const cfgPre = document.getElementById("preset-save-cfg");
    if (cfgPre) cfgPre.value = document.getElementById("cfg-input")?.value || "3.5";
    const noisePre = document.getElementById("preset-save-noise");
    if (noisePre) noisePre.value = document.getElementById("noise-input")?.value || "0.0";
    const privPre = document.getElementById("preset-save-private");
    if (privPre) privPre.checked = false;
  },

  openEditPresetModal(presetId, event) {
    if (event) event.stopPropagation();
    this.closeAllDockPopovers();
    const preset = (this.state.presets || []).find(p => p.id === presetId);
    if (!preset) return;

    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "flex";

    const idInput = document.getElementById("preset-save-id");
    if (idInput) idInput.value = preset.id;
    const titleHeader = document.getElementById("preset-modal-title");
    if (titleHeader) titleHeader.textContent = `✏️ Редактировать шаблон: «${preset.title}»`;

    const titleInput = document.getElementById("preset-save-name");
    if (titleInput) {
      titleInput.value = preset.title || "";
      titleInput.focus();
    }
    const iconInput = document.getElementById("preset-save-icon");
    if (iconInput) iconInput.value = preset.icon || "⭐";
    const catInput = document.getElementById("preset-save-category");
    if (catInput) catInput.value = preset.category || "Пользовательские";
    const descInput = document.getElementById("preset-save-desc");
    if (descInput) descInput.value = preset.description || "";

    const promptPre = document.getElementById("preset-save-prompt");
    if (promptPre) promptPre.value = preset.prompt || preset.prompt_template || "";
    const negPre = document.getElementById("preset-save-neg");
    if (negPre) negPre.value = preset.negative_prompt || "";

    const stepsPre = document.getElementById("preset-save-steps");
    if (stepsPre) stepsPre.value = preset.steps || 20;
    const cfgPre = document.getElementById("preset-save-cfg");
    if (cfgPre) cfgPre.value = preset.cfg ?? 3.5;
    const noisePre = document.getElementById("preset-save-noise");
    if (noisePre) noisePre.value = preset.noise ?? 0.0;
    const privPre = document.getElementById("preset-save-private");
    if (privPre) privPre.checked = Boolean(preset.is_private);
  },

  closeSavePresetModal() {
    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "none";
  },

  async saveCurrentAsPreset() {
    const titleEl = document.getElementById("preset-save-name");
    const title = titleEl?.value?.trim();
    if (!title) {
      this.toast("warning", "Введите название шаблона");
      return;
    }
    const editId = document.getElementById("preset-save-id")?.value?.trim() || null;
    const promptVal = document.getElementById("preset-save-prompt")?.value?.trim()
      ?? document.getElementById("prompt-input")?.value?.trim()
      ?? "";
    const negVal = document.getElementById("preset-save-neg")?.value?.trim()
      ?? document.getElementById("negative-prompt-input")?.value?.trim()
      ?? "";

    const stepsVal = parseInt(document.getElementById("preset-save-steps")?.value || "20", 10);
    const cfgVal = parseFloat(document.getElementById("preset-save-cfg")?.value || "3.5");
    const noiseVal = parseFloat(document.getElementById("preset-save-noise")?.value || "0.0");
    const isPrivate = Boolean(document.getElementById("preset-save-private")?.checked);

    const payload = {
      id: editId,
      title: title,
      icon: document.getElementById("preset-save-icon")?.value?.trim() || "⭐",
      category: document.getElementById("preset-save-category")?.value?.trim() || "Пользовательские",
      description: document.getElementById("preset-save-desc")?.value?.trim() || "",
      prompt: promptVal,
      negative_prompt: negVal,
      model_id: this.state.selectedModel || null,
      variant_id: document.getElementById("variant-select")?.value || this.state.selectedVariant || null,
      ref_mode: document.getElementById("img2img-mode")?.value || "control_ref",
      strength: parseFloat(document.getElementById("img2img-strength")?.value || "0.85"),
      resolution: document.getElementById("resolution-select")?.value || "1024x1024",
      steps: stepsVal,
      cfg: cfgVal,
      sampler: document.getElementById("sampler-select")?.value || "euler",
      noise: noiseVal,
      is_private: isPrivate,
    };

    try {
      await this.apiPost("/api/presets", payload);
      this.closeSavePresetModal();
      await this.loadPresets();
      this.toast("success", editId ? `Шаблон «${title}» обновлён ✓` : `Шаблон «${title}» сохранён ✓`);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async deletePreset(presetId, event) {
    if (event) event.stopPropagation();
    if (!confirm(t("templates.confirm_delete"))) return;
    try {
      const res = await fetch(`/api/presets/${encodeURIComponent(presetId)}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      await this.loadPresets();
      this.toast("info", "Шаблон удалён ✓");
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  // ---------- Universal Custom Model Importer (Local + Hugging Face Hub) ----------
  async openCustomModelModal(tab = "local") {
    const modal = document.getElementById("custom-model-modal");
    if (!modal) return;

    this.switchCustomModelTab(tab);

    try {
      const res = await fetch("/api/custom-models/files");
      if (res.ok) {
        const data = await res.json();
        this._customFilesData = data;
        const dirPathEl = document.getElementById("custom-models-dir-path");
        if (dirPathEl) dirPathEl.textContent = data.custom_dir || "models/custom";

        this.populateCustomFileSelects(data.files || []);
      }
    } catch (err) {
      console.warn("Failed to fetch custom files:", err);
    }

    modal.style.display = "flex";
    if (tab === "hf") {
      document.getElementById("hf-search-input")?.focus();
    }
  },

  switchCustomModelTab(tab) {
    const isHf = tab === "hf";
    const tabLocal = document.getElementById("importer-tab-btn-local") || document.getElementById("custom-tab-local");
    const tabHf = document.getElementById("importer-tab-btn-hf") || document.getElementById("custom-tab-hf");
    const paneLocal = document.getElementById("custom-model-pane-local");
    const paneHf = document.getElementById("custom-model-pane-hf");

    if (tabLocal) tabLocal.classList.toggle("active", !isHf);
    if (tabHf) tabHf.classList.toggle("active", isHf);
    if (paneLocal) paneLocal.style.display = !isHf ? "block" : "none";
    if (paneHf) paneHf.style.display = isHf ? "block" : "none";
  },

  closeCustomModelModal() {
    const modal = document.getElementById("custom-model-modal");
    if (modal) modal.style.display = "none";
  },

  // ---------- LazySD-Models Cloud Catalog Hub ----------

  async openHubCatalogModal() {
    const modal = document.getElementById("hub-catalog-modal");
    if (!modal) return;
    modal.style.display = "flex";
    if (!this.state.hubCatalog || !this.state.hubCatalog.length) {
      await this.loadHubCatalog();
    } else {
      this.renderHubCatalog();
    }
    const searchInput = document.getElementById("hub-catalog-search");
    if (searchInput) {
      searchInput.value = this.state.hubCatalogSearch || "";
      searchInput.focus();
    }
  },

  closeHubCatalogModal() {
    const modal = document.getElementById("hub-catalog-modal");
    if (modal) modal.style.display = "none";
  },

  async loadHubCatalog(refresh = false) {
    const grid = document.getElementById("hub-catalog-grid");
    if (grid) {
      grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px; text-align: center;">⏳ Загрузка каталога моделей...</div>`;
    }
    try {
      const data = await this.apiGet(`/api/hub/models${refresh ? "?refresh=true" : ""}`);
      this.state.hubCatalog = data.models || [];
      const badge = document.getElementById("hub-catalog-count-badge");
      if (badge) {
        badge.textContent = `${data.count || 0} моделей (${data.installed_count || 0} уст.)`;
      }
      if (data.warning) {
        this.showToast(data.warning, "warning");
      }
      this.renderHubCatalog();
    } catch (e) {
      if (grid) {
        grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--error);">❌ Ошибка загрузки каталога: ${escapeHtml(e.message || String(e))}</div>`;
      }
    }
  },

  async refreshHubCatalog() {
    await this.loadHubCatalog(true);
    this.showToast("Каталог моделей обновлен!", "success");
  },

  onHubCatalogSearch(query) {
    this.state.hubCatalogSearch = (query || "").toLowerCase().trim();
    this.renderHubCatalog();
  },

  setHubArchFilter(arch, btnEl) {
    this.state.hubCatalogArch = arch;
    document.querySelectorAll("#hub-catalog-arch-filters .filter-chip").forEach(el => {
      el.classList.toggle("active", el === btnEl);
    });
    this.renderHubCatalog();
  },

  setHubEngineFilter(engine, btnEl) {
    this.state.hubCatalogEngine = engine;
    const parent = btnEl ? btnEl.parentElement : null;
    if (parent) {
      parent.querySelectorAll(".filter-chip").forEach(el => {
        el.classList.toggle("active", el === btnEl);
      });
    }
    this.renderHubCatalog();
  },

  renderHubCatalog() {
    const grid = document.getElementById("hub-catalog-grid");
    if (!grid) return;
    const items = this.state.hubCatalog || [];
    if (!items.length) {
      grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px; text-align: center;">Каталог пуст или недоступен.</div>`;
      return;
    }

    const q = this.state.hubCatalogSearch || "";
    const archFilter = this.state.hubCatalogArch || "all";
    const engineFilter = this.state.hubCatalogEngine || "all";

    const filtered = items.filter(m => {
      if (archFilter !== "all" && m.target_architecture !== archFilter) return false;
      if (engineFilter !== "all" && (m.engine || "quantfunc").toLowerCase() !== engineFilter.toLowerCase()) return false;
      if (q) {
        const text = `${m.name || ""} ${m.id || ""} ${m.target_architecture || ""} ${m.description || ""} ${(m.tags || []).join(" ")}`.toLowerCase();
        if (!text.includes(q)) return false;
      }
      return true;
    });

    if (!filtered.length) {
      grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px; text-align: center;">Ничего не найдено по заданным фильтрам.</div>`;
      return;
    }

    grid.innerHTML = filtered.map(m => {
      const archLabel = m.target_architecture || m.family || "all";
      const engineLabel = (m.engine || "quantfunc").toUpperCase();
      const isInstalled = !!m.installed;
      const vramMin = m.vram_min_gb ? `${m.vram_min_gb} GB` : "8 GB";
      const actionHtml = isInstalled
        ? `<span class="lora-compat-badge compatible" style="padding: 4px 10px; font-size: 11px;">✅ Установлена</span>`
        : `<button type="button" class="generate-btn" style="height: 30px; font-size: 11px; padding: 0 12px; background: var(--accent);" onclick="App.installHubCard('${escapeHtml(m.card_file || m.id + '.json')}', this)">📥 Установить карточку</button>`;

      return `
        <div class="card-editor-component-item" style="display: flex; flex-direction: column; justify-content: space-between; padding: 12px; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;">
          <div>
            <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; margin-bottom: 6px;">
              <strong style="font-size: 13px; line-height: 1.3;">${escapeHtml(m.name || m.id)}</strong>
              <span class="lora-compat-badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; font-size: 10px; white-space: nowrap;">${escapeHtml(engineLabel)}</span>
            </div>
            <div style="display: flex; gap: 6px; align-items: center; margin-bottom: 6px; flex-wrap: wrap;">
              <span class="lora-compat-badge" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; font-size: 10px;">⚡ ${escapeHtml(archLabel)}</span>
              <span class="card-tag" style="font-size: 10px;">VRAM: ${escapeHtml(vramMin)}</span>
            </div>
            <p style="margin: 0 0 10px; font-size: 11px; color: var(--text-dim); line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;">
              ${escapeHtml(m.description || "Официальная оптимизированная карточка модели.")}
            </p>
          </div>
          <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid var(--border); padding-top: 8px; margin-top: auto;">
            <span style="font-size: 10px; color: var(--text-dim);">${escapeHtml(m.card_file || "")}</span>
            ${actionHtml}
          </div>
        </div>
      `;
    }).join("");
  },

  async installHubCard(cardFile, btnEl) {
    if (btnEl) {
      btnEl.disabled = true;
      btnEl.textContent = "⏳ Установка...";
    }
    try {
      const res = await this.apiPost("/api/hub/install", { card_file: cardFile });
      this.showToast(res.message || "Карточка успешно установлена!", "success");
      if (this.state.hubCatalog) {
        const item = this.state.hubCatalog.find(x => (x.card_file === cardFile || x.id === res.model_id));
        if (item) item.installed = true;
      }
      this.renderHubCatalog();
      if (typeof this.loadModels === "function") {
        await this.loadModels();
      }
    } catch (e) {
      this.showToast("Ошибка установки карточки: " + (e.message || String(e)), "error");
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.textContent = "📥 Установить карточку";
      }
    }
  },

  populateCustomFileSelects(files) {
    const modelSel = document.getElementById("custom-model-file-select");
    const teSel = document.getElementById("custom-te-file-select");
    const vaeSel = document.getElementById("custom-vae-file-select");

    if (modelSel) {
      modelSel.innerHTML = `<option value="">${escapeHtml(t("custom_model.select_model_file"))}</option><option value="__custom__">${escapeHtml(t("custom_model.custom_path"))}</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        const roleTag = f.role === "text_encoder" ? " [Encoder]" : (f.role === "vae" ? " [VAE]" : "");
        opt.textContent = `${f.name} (${f.size_gb} GB, ${f.quant || f.ext.toUpperCase()})${roleTag}`;
        modelSel.appendChild(opt);
      }
    }

    if (teSel) {
      teSel.innerHTML = `<option value="">${escapeHtml(t("custom_model.te_not_required"))}</option><option value="__custom__">${escapeHtml(t("custom_model.custom_path"))}</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        opt.textContent = `${f.name} (${f.size_gb} GB)`;
        teSel.appendChild(opt);
      }
    }

    if (vaeSel) {
      vaeSel.innerHTML = `<option value="">${escapeHtml(t("custom_model.vae_auto"))}</option><option value="__custom__">${escapeHtml(t("custom_model.custom_path"))}</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        opt.textContent = `${f.name} (${f.size_gb} GB)`;
        vaeSel.appendChild(opt);
      }
    }
  },

  onCustomFileSelectChange(type) {
    const sel = document.getElementById(`custom-${type === 'model' ? 'model-file' : type}-file-select`);
    const input = document.getElementById(`custom-${type === 'model' ? 'model-file' : type}-file-input`);
    if (!sel || !input) return;

    if (sel.value === "__custom__") {
      input.style.display = "block";
      input.focus();
    } else {
      input.style.display = "none";
    }

    // Smart auto-detection of family, steps, CFG, and name when a model file is chosen
    if (type === "model" && sel.value && sel.value !== "__custom__") {
      const nameInput = document.getElementById("custom-model-name");
      if (nameInput && !nameInput.value.trim()) {
        const stem = sel.value.split(/[\\/]/).pop().replace(/\.[^/.]+$/, "");
        nameInput.value = stem.replace(/[-_]/g, " ");
      }
      const fileMeta = (this._customFilesData?.files || []).find(f => f.rel_path === sel.value);
      if (fileMeta) {
        const familySel = document.getElementById("custom-model-family");
        const stepsInput = document.getElementById("custom-model-steps");
        const cfgInput = document.getElementById("custom-model-cfg");
        if (familySel && fileMeta.family) familySel.value = fileMeta.family;
        if (stepsInput && fileMeta.steps) stepsInput.value = fileMeta.steps;
        if (cfgInput && fileMeta.cfg !== undefined) cfgInput.value = fileMeta.cfg;
      }
    }
  },

  onCustomEngineChange() {
    // Engine switcher hook
  },

  onCustomFamilyChange() {
    const familySel = document.getElementById("custom-model-family");
    const stepsInput = document.getElementById("custom-model-steps");
    const cfgInput = document.getElementById("custom-model-cfg");
    if (!familySel || !stepsInput || !cfgInput) return;

    const fam = familySel.value;
    if (fam === "sdxl") {
      stepsInput.value = "20";
      cfgInput.value = "7.0";
    } else if (fam === "flux") {
      stepsInput.value = "4";
      cfgInput.value = "1.0";
    } else if (fam === "qwen") {
      stepsInput.value = "20";
      cfgInput.value = "3.0";
    } else {
      stepsInput.value = "8";
      cfgInput.value = "2.5";
    }
  },

  async saveCustomModel() {
    const nameInput = document.getElementById("custom-model-name");
    const engineSel = document.getElementById("custom-model-engine");
    const familySel = document.getElementById("custom-model-family");
    const modelSel = document.getElementById("custom-model-file-select");
    const modelInput = document.getElementById("custom-model-file-input");
    const teSel = document.getElementById("custom-te-file-select");
    const teInput = document.getElementById("custom-te-file-input");
    const vaeSel = document.getElementById("custom-vae-file-select");
    const vaeInput = document.getElementById("custom-vae-file-input");
    const stepsInput = document.getElementById("custom-model-steps");
    const cfgInput = document.getElementById("custom-model-cfg");

    const name = nameInput ? nameInput.value.trim() : "";
    if (!name) {
      this.toast("error", t("custom_model.error_name") || "Введите название модели");
      return;
    }

    let modelFile = modelSel ? modelSel.value : "";
    if (modelFile === "__custom__") {
      modelFile = modelInput ? modelInput.value.trim() : "";
    }
    if (!modelFile) {
      this.toast("error", t("custom_model.error_file") || "Выберите или укажите файл модели");
      return;
    }

    let teFile = teSel ? teSel.value : "";
    if (teFile === "__custom__") teFile = teInput ? teInput.value.trim() : "";

    let vaeFile = vaeSel ? vaeSel.value : "";
    if (vaeFile === "__custom__") vaeFile = vaeInput ? vaeInput.value.trim() : "";

    const payload = {
      name: name,
      engine: engineSel ? engineSel.value : "sd_cpp",
      family: familySel ? familySel.value : "flux",
      model_file: modelFile,
      steps: parseInt(stepsInput?.value || "20", 10),
      cfg: parseFloat(cfgInput?.value || "3.5"),
      checkpoint_type: familySel?.value === "sdxl" ? "checkpoint" : "diffusion_model",
    };
    if (teFile) {
      if (familySel?.value === "qwen") payload.llm_file = teFile;
      else payload.t5_file = teFile;
    }
    if (vaeFile) payload.vae_file = vaeFile;

    try {
      const res = await fetch("/api/custom-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to register custom model");
      }

      this.toast("success", t("custom_model.saved", { name }));
      this.closeCustomModelModal();

      await this.refreshConfig();
      if (data.model && data.model.id) {
        await this.selectModelFromModal(data.model.id);
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  async onHfSearchOrInspect() {
    const raw = document.getElementById("hf-search-input")?.value?.trim() || "";
    if (!raw) {
      this.toast("warning", t("custom_model.hf_query_required"));
      return;
    }

    // Direct repo link or owner/repo format -> inspect directly
    const cleanRepo = raw
      .replace(/^https?:\/\/huggingface\.co\//i, "")
      .replace(/\/(tree|blob)\/.*$/i, "")
      .replace(/^\/+|\/+$/g, "");

    if (cleanRepo.includes("/") && !cleanRepo.includes(" ") && cleanRepo.split("/").length === 2) {
      await this.inspectHfRepo(cleanRepo);
      return;
    }

    const resultsEl = document.getElementById("hf-search-results");
    if (resultsEl) {
      resultsEl.style.display = "flex";
      resultsEl.innerHTML = `<div style="padding: 12px; font-size: 12px; color: var(--text-muted);">${escapeHtml(t("custom_model.hf_searching"))}</div>`;
    }

    try {
      const data = await this.apiGet(`/api/hf/search?q=${encodeURIComponent(raw)}`);
      const results = data.results || [];
      if (!resultsEl) return;

      if (!results.length) {
        resultsEl.innerHTML = `<div style="padding: 12px; font-size: 12px; color: var(--text-dim);">${t("custom_model.hf_nothing")}</div>`;
        return;
      }

      resultsEl.innerHTML = "";
      for (const item of results) {
        const div = document.createElement("div");
        div.className = "hf-result-item";
        div.innerHTML = `
          <div style="min-width: 0;">
            <div style="font-size: 12.5px; font-weight: 600; color: var(--text-bright);">${escapeHtml(item.repo_id)}</div>
            <div style="font-size: 11px; color: var(--text-dim);">${escapeHtml(t("custom_model.hf_counts", { downloads: item.downloads || 0, likes: item.likes || 0 }))}</div>
          </div>
          <button class="setting-btn" style="height: 26px; font-size: 11px;">${escapeHtml(t("custom_model.hf_open_files"))}</button>
        `;
        div.addEventListener("click", () => this.inspectHfRepo(item.repo_id));
        resultsEl.appendChild(div);
      }
    } catch (e) {
      if (resultsEl) {
        resultsEl.innerHTML = `<div style="padding: 12px; font-size: 12px; color: #f87171;">${escapeHtml(t("custom_model.hf_error"))}${escapeHtml(e.message)}</div>`;
      }
    }
  },

  async inspectHfRepo(repoId) {
    const inspectorEl = document.getElementById("hf-repo-inspector");
    const filesListEl = document.getElementById("hf-repo-files-list") || document.getElementById("hf-files-checklist");
    const titleEl = document.getElementById("hf-inspected-repo-title") || document.getElementById("hf-repo-title");
    if (inspectorEl) inspectorEl.style.display = "block";
    if (titleEl) titleEl.textContent = t("custom_model.hf_loading_repo", { repo: repoId });
    if (filesListEl) filesListEl.innerHTML = `<div style="font-size: 12px; color: var(--text-muted);">${escapeHtml(t("custom_model.hf_loading_meta"))}</div>`;

    try {
      const info = await this.apiGet(`/api/hf/repo-tree?repo_id=${encodeURIComponent(repoId)}`);
      this.state.hfInspectedRepo = info;

      if (titleEl) titleEl.textContent = `📦 ${info.repo_id}`;

      const badgeEl = document.getElementById("hf-inspected-repo-badge");
      if (badgeEl) {
        const layoutLabel = info.layout === "folders" ? "Folders" : (info.layout || "Repo");
        badgeEl.textContent = `${layoutLabel}${info.has_model_index ? " · model_index" : (info.has_component_configs ? " · component configs" : "")}`;
      }

      const nameInp = document.getElementById("hf-import-name") || document.getElementById("hf-custom-name");
      const modelDisplayName = info.name || info.suggested_name || repoId.split("/").pop();
      if (nameInp) nameInp.value = `${modelDisplayName} (HF)`;
      const famSel = document.getElementById("hf-import-family") || document.getElementById("hf-custom-family");
      if (famSel && (info.family || info.suggested_family)) famSel.value = info.family || info.suggested_family;
      const engineSel = document.getElementById("hf-import-engine");
      if (engineSel && info.suggested_engine) engineSel.value = info.suggested_engine;
      const folderInp = document.getElementById("hf-import-folder");
      if (folderInp) folderInp.value = info.suggested_folder || repoId.split("/").pop() || "";
      const stepsInp = document.getElementById("hf-import-steps") || document.getElementById("hf-custom-steps");
      if (stepsInp && (info.steps || info.suggested_steps)) stepsInp.value = info.steps || info.suggested_steps;
      const cfgInp = document.getElementById("hf-import-cfg") || document.getElementById("hf-custom-cfg");
      const suggestedCfg = info.cfg !== undefined ? info.cfg : info.suggested_cfg;
      if (cfgInp && suggestedCfg !== undefined) cfgInp.value = suggestedCfg;

      const metaEl = document.getElementById("hf-import-meta");
      if (metaEl) {
        const chips = [];
        if (info.pipeline_tag) chips.push(`🎯 ${info.pipeline_tag}`);
        if (info.license) chips.push(`📄 ${info.license}`);
        if (Number(info.weight_size_gb) > 0) chips.push(t("custom_model.hf_weight_size", { size: Number(info.weight_size_gb).toFixed(2) }));
        if (info.base_model) chips.push(`🧬 ${info.base_model}`);
        const counts = [
          [info.diffusion_files?.length, "transformer"],
          [info.text_encoder_files?.length, "TE"],
          [info.vae_files?.length, "VAE"],
          [info.aux_files?.length, "aux"],
        ].filter(([n]) => Number(n) > 0).map(([n, label]) => `${n} ${label}`);
        if (counts.length) chips.push(`🧩 ${counts.join(" · ")}`);
        if (Array.isArray(info.tags) && info.tags.length) chips.push(`🏷 ${info.tags.slice(0, 6).join(", ")}`);
        metaEl.innerHTML = chips.map(text => `<span class="hf-meta-chip">${escapeHtml(text)}</span>`).join("");
        metaEl.style.display = chips.length ? "flex" : "none";
      }

      const warnEl = document.getElementById("hf-import-warning-banner");
      if (warnEl) {
        const missingLabels = {
          base: t("custom_model.hf_missing_base"),
          text_encoder: t("custom_model.hf_missing_te"),
          vae: t("custom_model.hf_missing_vae"),
        };
        const warnings = [...(info.warnings || [])];
        const missing = (info.missing || []).map(item => missingLabels[item] || item);
        if (missing.length) warnings.push(t("custom_model.hf_missing", { missing: missing.join(", ") }));
        const uniqueWarnings = [...new Set(warnings.filter(Boolean))];
        warnEl.innerHTML = uniqueWarnings.map(text => `<div>⚠️ ${escapeHtml(text)}</div>`).join("");
        warnEl.style.display = uniqueWarnings.length ? "block" : "none";
      }

      const reasonEl = document.getElementById("hf-import-engine-reason");
      if (reasonEl) {
        const reasons = Array.isArray(info.engine_reasons) ? info.engine_reasons.filter(Boolean) : [];
        reasonEl.innerHTML = reasons.length
          ? `<strong>${escapeHtml(t("custom_model.hf_engine"))}${escapeHtml(info.suggested_engine || "—")}</strong> · ${reasons.map(escapeHtml).join(" ")}`
          : "";
        reasonEl.style.display = reasons.length ? "block" : "none";
      }

      const foldersEl = document.getElementById("hf-import-folders-tree");
      if (foldersEl) {
        const folders = Array.isArray(info.folders) ? info.folders : [];
        foldersEl.innerHTML = folders.length
          ? `<div style="font-weight: 600; margin-bottom: 4px; color: var(--text-bright);">${escapeHtml(t("custom_model.hf_structure"))}</div>` +
            folders.map(row => `<div class="hf-folder-row"><code>${escapeHtml(row.folder)}</code><span>${escapeHtml(t("custom_model.hf_folder_row", { role: row.role, count: row.files, size: row.size_gb }))}</span></div>`).join("")
          : "";
        foldersEl.style.display = folders.length ? "block" : "none";
      }

      const manualEl = document.getElementById("hf-import-manual-section");
      const hasTe = Array.isArray(info.text_encoder_files) && info.text_encoder_files.length > 0;
      const hasVae = Array.isArray(info.vae_files) && info.vae_files.length > 0;
      if (manualEl) {
        manualEl.style.display = (info.needs_manual || !hasTe || !hasVae) ? "flex" : "none";
      }
      const baseInp = document.getElementById("hf-import-base-repo");
      if (baseInp) {
        baseInp.value = (info.needs_manual || !info.has_component_configs) ? (info.base_model || "") : "";
      }
      const teRefInp = document.getElementById("hf-import-custom-te");
      if (teRefInp) {
        const def = info.defaults?.text_encoder;
        teRefInp.value = !hasTe && def?.repo_id && def?.path ? `${def.repo_id}::${def.path}` : "";
      }
      const vaeRefInp = document.getElementById("hf-import-custom-vae");
      if (vaeRefInp) {
        const def = info.defaults?.vae;
        vaeRefInp.value = !hasVae && def?.repo_id && def?.path ? `${def.repo_id}::${def.path}` : "";
      }

      if (filesListEl) {
        filesListEl.innerHTML = "";
        const groups = [
          { title: t("custom_model.hf_weights"), files: info.diffusion_files || [], cls: "hf-diff-checkbox", check: true, interactive: true },
          { title: t("custom_model.hf_te_files"), files: info.text_encoder_files || [], cls: "hf-role-file", role: "TE" },
          { title: t("custom_model.hf_vae_files"), files: info.vae_files || [], cls: "hf-role-file", role: "VAE" },
          { title: "Auxiliary / OrbitQuant sidecars", files: info.aux_files || [], cls: "hf-role-file", role: "AUX" },
        ];
        const allFiles = groups.reduce((sum, group) => sum + group.files.length, 0);
        if (!allFiles) {
          filesListEl.innerHTML = `<div style="font-size: 12px; color: #f87171;">${escapeHtml(t("custom_model.hf_no_files"))}</div>`;
        }
        for (const group of groups) {
          if (!group.files.length) continue;
          const heading = document.createElement("div");
          heading.className = "hf-file-group-title";
          heading.textContent = group.title;
          filesListEl.appendChild(heading);
          group.files.forEach((f, idx) => {
            const isRec = group.interactive && (f.quant === "Q8_0" || f.quant === "Q4_K_M" || idx === 0);
            const row = document.createElement("label");
            row.className = "hf-file-row";
            const checkbox = group.interactive
              ? `<input type="checkbox" class="hf-diff-checkbox" data-path="${escapeHtml(f.path)}" data-quant="${escapeHtml(f.quant || '')}" data-size="${f.size_gb || 0}" ${isRec ? "checked" : ""}>`
              : `<input type="checkbox" class="${group.cls}" data-role="${escapeHtml(group.role)}" data-path="${escapeHtml(f.path)}" checked disabled>`;
            row.innerHTML = `
              <span style="display: flex; align-items: center; gap: 8px; min-width: 0;">
                ${checkbox}
                <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(f.path || f.name)}</span>
              </span>
              <span style="font-size: 11px; color: var(--text-muted); white-space: nowrap;">${escapeHtml(group.role || f.quant || f.ext?.toUpperCase() || "")} · ${f.size_gb} GB</span>
            `;
            filesListEl.appendChild(row);
          });
        }
      }
    } catch (e) {
      if (titleEl) titleEl.textContent = t("custom_model.hf_repo_error", { repo: repoId });
      if (filesListEl) filesListEl.innerHTML = `<div style="font-size: 12px; color: #f87171;">${escapeHtml(e.message)}</div>`;
    }
  },

  async registerHfModel() {
    const info = this.state.hfInspectedRepo;
    if (!info || !info.repo_id) {
      this.toast("warning", t("custom_model.hf_repo_required"));
      return;
    }

    const checkedBoxes = Array.from(document.querySelectorAll(".hf-diff-checkbox:checked"));
    if (!checkedBoxes.length) {
      this.toast("warning", t("custom_model.hf_files_required"));
      return;
    }

    const stepsEl = document.getElementById("hf-import-steps") || document.getElementById("hf-custom-steps");
    const nameEl = document.getElementById("hf-import-name") || document.getElementById("hf-custom-name");
    const famEl = document.getElementById("hf-import-family") || document.getElementById("hf-custom-family");
    const cfgEl = document.getElementById("hf-import-cfg") || document.getElementById("hf-custom-cfg");
    const engineEl = document.getElementById("hf-import-engine");
    const folderEl = document.getElementById("hf-import-folder");
    const baseRepoEl = document.getElementById("hf-import-base-repo");
    const customTeEl = document.getElementById("hf-import-custom-te");
    const customVaeEl = document.getElementById("hf-import-custom-vae");

    const selectedFiles = checkedBoxes.map(cb => ({
      path: cb.dataset.path,
      quant: cb.dataset.quant,
      size_gb: parseFloat(cb.dataset.size || "0"),
      steps: parseInt(stepsEl?.value || "20", 10),
    }));

    const payload = {
      repo_id: info.repo_id,
      name: nameEl?.value?.trim() || `${info.name || info.suggested_name || 'Model'} (HF)`,
      family: famEl?.value || info.family || info.suggested_family || "flux",
      engine: engineEl?.value || info.suggested_engine || "sd_cpp",
      steps: parseInt(stepsEl?.value || "20", 10),
      cfg: parseFloat(cfgEl?.value || "3.5"),
      local_folder: folderEl?.value?.trim() || info.suggested_folder || "",
      base_repo_id: baseRepoEl?.value?.trim() || "",
      custom_te_ref: customTeEl?.value?.trim() || "",
      custom_vae_ref: customVaeEl?.value?.trim() || "",
      pipeline_tag: info.pipeline_tag || "",
      tags: info.tags || [],
      license: info.license || "",
      has_model_index: !!info.has_model_index,
      selected_files: selectedFiles,
      variants: selectedFiles,
      text_encoder_files: (info.text_encoder_files || []).slice(0, 4),
      selected_te_files: (info.text_encoder_files || []).slice(0, 4),
      selected_vae_files: (info.vae_files || []).slice(0, 1),
      aux_files: info.aux_files || [],
    };

    try {
      const res = await this.apiPost("/api/hf/register", payload);
      this.toast("success", t("custom_model.hf_saved", { name: payload.name }));
      this.closeCustomModelModal();
      await this.refreshConfig();
      if (res.model && res.model.id) {
        await this.selectModelFromModal(res.model.id);
      }
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async deleteCustomModel(modelId, event) {
    if (event) event.stopPropagation();
    const confirmed = confirm(t("models.delete_card_confirm", { id: modelId }));
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/custom-models/${encodeURIComponent(modelId)}?delete_files=true`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to delete");
      }
      this.toast("info", t("models.delete_card_success", { id: modelId }));
      await this.refreshConfig();
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  parseComponentRef(value) {
    const text = String(value || "").trim();
    if (!text) return null;
    const urlMatch = text.match(/^https?:\/\/(?:www\.)?huggingface\.co\/([^/]+)\/([^/]+)\/(?:blob|resolve)\/[^/]+\/([^?#]+)/i);
    if (urlMatch) {
      return { repo_id: `${urlMatch[1]}/${urlMatch[2]}`, path: decodeURIComponent(urlMatch[3]) };
    }
    if (text.includes("::")) {
      const [repoRaw, ...pathParts] = text.split("::");
      const repo = repoRaw.replace(/^https?:\/\/(?:www\.)?huggingface\.co\//i, "").replace(/\/+$/, "");
      const path = pathParts.join("::").replace(/\\/g, "/").replace(/^\/+/, "");
      if (repo.includes("/") && path) return { repo_id: repo, path };
    }
    return null;
  },

  async openModelCardEditor(modelId) {
    try {
      const card = await this.apiGet(`/api/models/${encodeURIComponent(modelId)}/card`);
      const args = card.engine_args || {};
      const vaeInfo = card.vae || null;
      const vaePath = vaeInfo?.path || args.vae_remote || args.vae_file || args.vae || "";
      const vaeLocal = /^[A-Za-z]:[\\/]/.test(vaePath) || vaePath.startsWith("/");
      const textEncoders = (card.text_encoders || []).map(te => ({
        id: te.id,
        name: te.name || te.path || te.file,
        repo_id: (te.source || {}).repo_id || te.repo || card.repo_id || "",
        path: te.path || te.file || "",
        size_gb: Number(te.size_gb || 0),
        precision: te.precision || "custom",
        recommended: Boolean(te.recommended),
      }));
      const auxFiles = (args.aux_files || []).map(aux => {
        if (aux && typeof aux === "object") {
          return { repo_id: aux.repo_id || card.repo_id || "", path: aux.path || "", size_gb: Number(aux.size_gb || 0) };
        }
        return { repo_id: card.repo_id || "", path: String(aux || ""), size_gb: 0 };
      });
      const existingModel = (this.state.models || []).find(m => m.id === modelId);
      const existingVae = existingModel?.vae || (Array.isArray(existingModel?.vae_files) ? existingModel.vae_files[0] : null);
      this.state.modelCardEditor = {
        modelId,
        repoId: card.repo_id || "",
        manifest: card,
        te: textEncoders.map(te => {
          const matched = (existingModel?.text_encoders || []).find(e => e.id === te.id || e.path === te.path || e.file === te.file);
          return {
            ...te,
            installed: Boolean(matched?.installed),
          };
        }),
        vae: vaePath ? {
          repo_id: vaeInfo?.repo_id || args.vae_repo || card.repo_id || "",
          path: vaePath,
          size_gb: Number(vaeInfo?.size_gb || 0),
          local: vaeLocal,
          installed: Boolean(existingVae?.installed),
          shared: Boolean(existingVae?.shared),
          shared_from: existingVae?.shared_from || "",
        } : null,
        aux: auxFiles,
        repoInfo: null,
      };

      const setValue = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.value = value ?? "";
      };
      setValue("model-card-editor-id", modelId);
      setValue("model-card-editor-name", card.name || "");
      setValue("model-card-editor-family", card.family || "flux");
      setValue("model-card-editor-engine", card.engine || "sd_cpp");
      setValue("model-card-editor-pipeline", card.pipeline || "text_to_image");
      setValue("model-card-editor-steps", card.scheduler_presets?.default?.steps ?? 20);
      setValue("model-card-editor-cfg", card.default_cfg ?? card.scheduler_presets?.default?.cfg ?? 1.0);
      setValue("model-card-editor-license", card.license || "");
      setValue("model-card-editor-base-repo", card.base_repo_id || "");
      setValue("model-card-editor-description", card.description || "");
      const repoEl = document.getElementById("model-card-editor-repo");
      if (repoEl) repoEl.textContent = card.repo_id || t("models.card_editor.no_repo");
      const subtitle = document.getElementById("model-card-editor-subtitle");
      if (subtitle) subtitle.textContent = `${card.name || modelId} · ${card.repo_id || t("models.card_editor.local_model")}`;
      const statusEl = document.getElementById("model-card-editor-repo-status");
      if (statusEl) statusEl.textContent = t("models.card_editor.repo_hint");
      const suggestionsEl = document.getElementById("model-card-editor-suggestions");
      if (suggestionsEl) suggestionsEl.innerHTML = "";

      this.renderModelCardEditorComponents();
      document.getElementById("model-card-editor-modal").style.display = "flex";
      if (card.repo_id) this.inspectCardRepo(true);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  closeModelCardEditor() {
    const modal = document.getElementById("model-card-editor-modal");
    if (modal) modal.style.display = "none";
    this.state.modelCardEditor = null;
  },

  renderModelCardEditorComponents() {
    const editor = this.state.modelCardEditor;
    if (!editor) return;
    const teList = document.getElementById("model-card-editor-te-list");
    const vaeList = document.getElementById("model-card-editor-vae-list");
    const auxList = document.getElementById("model-card-editor-aux-list");
    const teCount = document.getElementById("model-card-editor-te-count");
    const auxCount = document.getElementById("model-card-editor-aux-count");
    if (teCount) teCount.textContent = String(editor.te.length);
    if (auxCount) auxCount.textContent = String(editor.aux.length);

    if (teList) {
      teList.innerHTML = editor.te.length
        ? editor.te.map((te, idx) => `
            <div class="component-item">
              <div class="component-info">
                <span class="component-name">${escapeHtml(te.name || te.path || t("models.card_editor.encoder_label"))} ${te.recommended ? `<span class="component-primary">${escapeHtml(t("models.card_editor.primary"))}</span>` : ""} ${te.installed ? `<span class="component-state installed">${ICON_CHECK} ${escapeHtml(t("models.component_on_disk"))}</span>` : `<span class="component-state missing">${ICON_DOWNLOAD} ${escapeHtml(t("models.card_editor.not_installed"))}</span>`}</span>
                <span class="component-sub" title="${escapeHtml([te.repo_id, te.path, te.precision].filter(Boolean).join(" · "))}">${escapeHtml([te.repo_id, te.path, te.precision, te.size_gb ? `${te.size_gb} GB` : ""].filter(Boolean).join(" · "))}</span>
              </div>
              <div class="component-actions">
                <button type="button" class="dock-ctrl-btn" onclick="App.makeCardTePrimary(${idx})" title="${escapeHtml(t("models.card_editor.make_primary"))}">★</button>
                <button type="button" class="dock-ctrl-btn" onclick="App.removeCardComponent('te', ${idx})" title="${escapeHtml(t("models.card_editor.remove_from_card"))}">✕</button>
              </div>
            </div>`).join("")
        : `<div class="component-empty">${escapeHtml(t("models.card_editor.te_empty"))}</div>`;
    }

    if (vaeList) {
      vaeList.innerHTML = editor.vae
        ? `<div class="component-item">
             <div class="component-info">
               <span class="component-name">VAE ${editor.vae.installed ? `<span class="component-state installed">${ICON_CHECK} ${escapeHtml(t("models.component_on_disk"))}${editor.vae.shared ? ` (${escapeHtml(editor.vae.shared_from || t("common.shared"))})` : ""}</span>` : `<span class="component-state missing">${ICON_DOWNLOAD} ${escapeHtml(t("models.card_editor.not_installed"))}</span>`}</span>
               <span class="component-sub" title="${escapeHtml([editor.vae.repo_id, editor.vae.path].filter(Boolean).join(" · "))}">${escapeHtml([editor.vae.repo_id, editor.vae.path].filter(Boolean).join(" · "))}</span>
             </div>
             <div class="component-actions">
               <button type="button" class="dock-ctrl-btn" onclick="App.removeCardComponent('vae', 0)" title="${escapeHtml(t("models.card_editor.remove_vae"))}">✕</button>
             </div>
           </div>`
        : `<div class="component-empty">${escapeHtml(t("models.card_editor.vae_empty"))}</div>`;
    }

    if (auxList) {
      auxList.innerHTML = editor.aux.length
        ? editor.aux.map((aux, idx) => `
            <div class="component-item">
              <div class="component-info">
                <span class="component-name">${escapeHtml(String(aux.path || "").split("/").pop() || t("models.card_editor.sidecar_label"))}</span>
                <span class="component-sub" title="${escapeHtml([aux.repo_id, aux.path].filter(Boolean).join(" · "))}">${escapeHtml([aux.repo_id, aux.path, aux.size_gb ? `${aux.size_gb} GB` : ""].filter(Boolean).join(" · "))}</span>
              </div>
              <div class="component-actions">
                <button type="button" class="dock-ctrl-btn" onclick="App.removeCardComponent('aux', ${idx})" title="${escapeHtml(t("models.card_editor.remove_sidecar"))}">✕</button>
              </div>
            </div>`).join("")
        : `<div class="component-empty">${escapeHtml(t("models.card_editor.aux_empty"))}</div>`;
    }
  },

  makeCardTePrimary(index) {
    const editor = this.state.modelCardEditor;
    if (!editor || !editor.te[index]) return;
    editor.te.forEach((te, idx) => { te.recommended = idx === index; });
    this.renderModelCardEditorComponents();
  },

  removeCardComponent(role, index) {
    const editor = this.state.modelCardEditor;
    if (!editor) return;
    if (role === "te") editor.te.splice(index, 1);
    else if (role === "vae") editor.vae = null;
    else if (role === "aux") editor.aux.splice(index, 1);
    this.renderModelCardEditorComponents();
  },

  addCardComponent(role, component) {
    const editor = this.state.modelCardEditor;
    if (!editor || !component || !component.path) return;
    const repo = component.repo_id || editor.repoId || "";
    if (role === "vae") {
      editor.vae = {
        repo_id: repo,
        path: component.path,
        size_gb: Number(component.size_gb || 0),
      };
      this.renderModelCardEditorComponents();
      return;
    }
    if (role === "aux") {
      const exists = editor.aux.some(aux => aux.repo_id === repo && aux.path === component.path);
      if (!exists) {
        editor.aux.push({ repo_id: repo, path: component.path, size_gb: Number(component.size_gb || 0) });
      }
      this.renderModelCardEditorComponents();
      return;
    }

    const exists = editor.te.some(te => te.repo_id === repo && te.path === component.path);
    if (exists) {
      this.toast("info", t("models.card_editor.already_added"));
      return;
    }
    let baseId = String(component.path.split("/").pop() || "te").replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "te";
    let teId = baseId;
    let suffix = 2;
    while (editor.te.some(te => te.id === teId)) {
      teId = `${baseId}_${suffix}`;
      suffix += 1;
    }
    editor.te.push({
      id: teId,
      name: component.name || component.path.split("/").pop(),
      repo_id: repo,
      path: component.path,
      size_gb: Number(component.size_gb || 0),
      precision: component.precision || "custom",
      recommended: editor.te.length === 0,
    });
    this.renderModelCardEditorComponents();
  },

  addCardComponentFromInput(role) {
    const inputId = role === "te" ? "model-card-editor-te-ref" : role === "vae" ? "model-card-editor-vae-ref" : "model-card-editor-aux-ref";
    const input = document.getElementById(inputId);
    const parsed = this.parseComponentRef(input?.value || "");
    if (!parsed) {
      this.toast("warning", t("models.card_editor.invalid_ref"));
      return;
    }
    this.addCardComponent(role, { ...parsed, size_gb: 0, precision: "custom" });
    if (input) input.value = "";
  },

  async inspectCardRepo(silent = false) {
    const editor = this.state.modelCardEditor;
    if (!editor || !editor.repoId) {
      if (!silent) this.toast("warning", t("models.card_editor.repo_missing"));
      return;
    }
    const statusEl = document.getElementById("model-card-editor-repo-status");
    const suggestionsEl = document.getElementById("model-card-editor-suggestions");
    if (statusEl) statusEl.textContent = t("models.card_editor.checking_repo");
    try {
      const info = await this.apiGet(`/api/hf/repo-tree?repo_id=${encodeURIComponent(editor.repoId)}`);
      editor.repoInfo = info;
      const teFiles = info.text_encoder_files || [];
      const vaeFiles = info.vae_files || [];
      const auxFiles = info.aux_files || [];
      if (statusEl) {
        const parts = t("models.card_editor.repo_counts", {
          te: teFiles.length,
          vae: vaeFiles.length,
          aux: auxFiles.length,
        });
        statusEl.innerHTML = escapeHtml(t("models.card_editor.repo_found", {
          icon: teFiles.length ? "✅" : "⚠️",
          parts,
        })) + (teFiles.length ? "" : escapeHtml(t("models.card_editor.te_not_found")));
      }
      if (!suggestionsEl) return;
      const rows = [];
      const pushSuggestion = (role, file, label) => {
        const exists = role === "te"
          ? editor.te.some(te => te.path === file.path)
          : role === "vae"
            ? editor.vae?.path === file.path
            : editor.aux.some(aux => aux.path === file.path);
        const size = Number(file.size_gb || 0);
        rows.push(`
          <div class="card-editor-suggestion">
            <span><strong>${escapeHtml(label)}</strong> <code>${escapeHtml(file.path)}</code>${size ? ` · ${size} GB` : ""}</span>
            ${exists
              ? `<span class="component-primary">${escapeHtml(t("models.card_editor.already_in_card"))}</span>`
              : `<button type="button" class="dock-mini-btn" data-role="${role}" data-repo="${escapeHtml(editor.repoId)}" data-path="${escapeHtml(file.path)}" data-size="${size}" onclick="App.addCardSuggestion(this.dataset.role, this.dataset.repo, this.dataset.path, this.dataset.size)">${escapeHtml(t("models.card_editor.add"))}</button>`}
          </div>`);
      };
      teFiles.slice(0, 8).forEach(file => pushSuggestion("te", file, t("models.card_editor.encoder_label")));
      vaeFiles.slice(0, 4).forEach(file => pushSuggestion("vae", file, "VAE"));
      auxFiles.slice(0, 8).forEach(file => pushSuggestion("aux", file, t("models.card_editor.sidecar_label")));
      suggestionsEl.innerHTML = rows.join("") || `<div class="component-empty">${escapeHtml(t("models.card_editor.nothing_found"))}</div>`;
    } catch (e) {
      if (statusEl) statusEl.innerHTML = `<span style="color:#f87171;">${escapeHtml(t("models.card_editor.check_failed"))}${escapeHtml(e.message)}</span>`;
      if (suggestionsEl) suggestionsEl.innerHTML = "";
      if (!silent) this.toast("error", e.message);
    }
  },

  addCardSuggestion(role, repoId, path, size) {
    this.addCardComponent(role, {
      repo_id: repoId,
      path,
      size_gb: Number(size || 0),
      precision: role === "vae" ? "custom" : undefined,
    });
  },

  async saveModelCard() {
    const editor = this.state.modelCardEditor;
    if (!editor) return;
    const value = id => document.getElementById(id)?.value ?? "";
    const payload = {
      name: value("model-card-editor-name").trim(),
      family: value("model-card-editor-family").trim(),
      engine: value("model-card-editor-engine"),
      pipeline: value("model-card-editor-pipeline"),
      steps: parseInt(value("model-card-editor-steps") || "20", 10),
      cfg: parseFloat(value("model-card-editor-cfg") || "1.0"),
      license: value("model-card-editor-license").trim(),
      base_repo_id: value("model-card-editor-base-repo").trim(),
      description: value("model-card-editor-description").trim(),
      text_encoders: editor.te.map(te => ({
        id: te.id,
        name: te.name,
        repo_id: te.repo_id,
        path: te.path,
        size_gb: te.size_gb,
        precision: te.precision,
        recommended: te.recommended,
      })),
      vae: editor.vae,
      aux_files: editor.aux,
    };
    if (editor.vae?.local) {
      delete payload.vae;
    }
    try {
      await this.apiPatch(`/api/models/${encodeURIComponent(editor.modelId)}/card`, payload);
      this.toast("success", t("models.card_editor.saved"));
      this.closeModelCardEditor();
      await this.refreshConfig();
      if (this.state.selectedModel === editor.modelId) {
        await this.onModelChange(this.state.selectedVariant || null);
      }
      await this.renderModelsTab();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  openComponentPicker(role) {
    if (!this.state.modelCardEditor) return;
    this.state.componentPicker = { role, repoId: this.state.modelCardEditor.repoId || "", info: null };
    const modal = document.getElementById("component-picker-modal");
    if (modal) modal.style.display = "flex";
    const title = document.getElementById("component-picker-title");
    if (title) title.textContent = role === "vae"
      ? t("models.component_picker.title_vae")
      : role === "aux"
        ? t("models.component_picker.title_aux")
        : t("models.component_picker.title_te");
    const search = document.getElementById("component-picker-search");
    if (search) search.value = "";
    const results = document.getElementById("component-picker-results");
    if (results) { results.innerHTML = ""; results.style.display = "none"; }
    const repoInput = document.getElementById("component-picker-repo");
    if (repoInput) repoInput.value = this.state.componentPicker.repoId;
    const files = document.getElementById("component-picker-files");
    if (files) files.innerHTML = "";
    const manual = document.getElementById("component-picker-manual");
    if (manual) {
      manual.value = "";
      manual.placeholder = role === "vae"
        ? t("models.component_picker.manual_vae")
        : t("models.component_picker.manual_te");
    }
    if (this.state.componentPicker.repoId) this.inspectComponentRepo();
  },

  closeComponentPicker() {
    const modal = document.getElementById("component-picker-modal");
    if (modal) modal.style.display = "none";
    this.state.componentPicker = null;
  },

  async searchComponentRepos() {
    const picker = this.state.componentPicker;
    if (!picker) return;
    const input = document.getElementById("component-picker-search");
    const query = (input?.value || "").trim() || (picker.role === "vae" ? "flux vae" : "flux text encoder");
    const resultsEl = document.getElementById("component-picker-results");
    if (resultsEl) { resultsEl.style.display = "block"; resultsEl.innerHTML = `<div class="component-empty">${escapeHtml(t("models.component_picker.searching"))}</div>`; }
    try {
      const res = await this.apiGet(`/api/hf/search?q=${encodeURIComponent(query)}&limit=12`);
      const results = res.results || [];
      if (!resultsEl) return;
      resultsEl.innerHTML = results.length
        ? results.map(item => `
            <div class="hf-result-item" data-repo="${escapeHtml(item.repo_id)}" onclick="App.pickComponentRepo(this.dataset.repo)">
              <div>${escapeHtml(item.repo_id)}</div>
              <div class="hf-result-meta">${escapeHtml(item.pipeline_tag || "")} ${item.downloads ? `· ⬇ ${item.downloads}` : ""}</div>
            </div>`).join("")
        : `<div class="component-empty">${escapeHtml(t("models.component_picker.none"))}</div>`;
    } catch (e) {
      if (resultsEl) resultsEl.innerHTML = `<div class="component-empty" style="color:#f87171;">${escapeHtml(e.message)}</div>`;
    }
  },

  pickComponentRepo(repoId) {
    const input = document.getElementById("component-picker-repo");
    if (input) input.value = repoId;
    this.inspectComponentRepo();
  },

  async inspectComponentRepo() {
    const picker = this.state.componentPicker;
    if (!picker) return;
    const repoInput = document.getElementById("component-picker-repo");
    const repoId = (repoInput?.value || "").trim();
    const filesEl = document.getElementById("component-picker-files");
    if (!repoId) {
      this.toast("warning", t("models.component_picker.repo_required"));
      return;
    }
    if (filesEl) filesEl.innerHTML = `<div class="component-empty">${escapeHtml(t("models.component_picker.checking"))}</div>`;
    try {
      const info = await this.apiGet(`/api/hf/repo-tree?repo_id=${encodeURIComponent(repoId)}`);
      picker.repoId = info.repo_id || repoId;
      picker.info = info;
      const files = picker.role === "vae" ? (info.vae_files || []) : picker.role === "aux" ? (info.aux_files || []) : (info.text_encoder_files || []);
      if (!filesEl) return;
      if (!files.length) {
        const what = picker.role === "vae"
          ? t("models.component_picker.role_vae")
          : picker.role === "aux"
            ? t("models.component_picker.role_aux")
            : t("models.component_picker.role_te");
        filesEl.innerHTML = `<div class="component-empty">${escapeHtml(t("models.component_picker.no_files", { what }))}</div>`;
        return;
      }
      filesEl.innerHTML = files.map(file => {
        const size = Number(file.size_gb || 0);
        return `
          <div class="component-item">
            <div class="component-info">
              <span class="component-name">${escapeHtml(file.path)}</span>
              <span class="component-sub">${size ? `${size} GB` : ""} ${file.quant ? `· ${file.quant}` : ""}</span>
            </div>
            <div class="component-actions">
              <button type="button" class="dock-mini-btn" data-repo="${escapeHtml(picker.repoId)}" data-path="${escapeHtml(file.path)}" data-size="${size}" data-precision="${escapeHtml(file.quant || 'custom')}" onclick="App.pickComponentFile(this.dataset.repo, this.dataset.path, this.dataset.size, this.dataset.precision)">${escapeHtml(t("models.card_editor.add"))}</button>
            </div>
          </div>`;
      }).join("");
    } catch (e) {
      if (filesEl) filesEl.innerHTML = `<div class="component-empty" style="color:#f87171;">${escapeHtml(e.message)}</div>`;
    }
  },

  pickComponentFile(repoId, path, size, precision) {
    const picker = this.state.componentPicker;
    if (!picker) return;
    this.addCardComponent(picker.role, { repo_id: repoId, path, size_gb: Number(size || 0), precision: precision || "custom" });
    this.closeComponentPicker();
    this.toast("success", t("models.component_picker.added"));
  },

  addComponentFromManual() {
    const picker = this.state.componentPicker;
    if (!picker) return;
    const manual = document.getElementById("component-picker-manual");
    const parsed = this.parseComponentRef(manual?.value || "");
    if (!parsed) {
      this.toast("warning", t("models.card_editor.invalid_ref"));
      return;
    }
    this.addCardComponent(picker.role, { ...parsed, size_gb: 0, precision: "custom" });
    this.closeComponentPicker();
    this.toast("success", t("models.component_picker.added"));
  },

  async deleteModelFiles(modelId, variantId, btnEl) {
    const model = (this.state.models || []).find(m => m.id === modelId);
    const modelLabel = model?.name || modelId;
    const variant = (model?.transformers || []).find(v => v.id === variantId);
    const sizeText = variant && Number(variant.size_gb) > 0 ? ` (~${Number(variant.size_gb).toFixed(1)} GB)` : "";
    const what = variantId
      ? t("models.weights_variant", { id: variantId, size: sizeText })
      : t("models.weights_all", { name: modelLabel });
    const confirmed = confirm(t("models.delete_weights_confirm", { what }));
    if (!confirmed) return;

    if (btnEl) {
      btnEl.disabled = true;
      btnEl.classList.add("is-loading");
    }

    try {
      const query = variantId ? `?variant_id=${encodeURIComponent(variantId)}` : "";
      const result = await this.apiDelete(`/api/models/${encodeURIComponent(modelId)}/files${query}`);
      const freedMb = Number(result.freed_mb || 0);
      const freedText = freedMb >= 1024 ? `${(freedMb / 1024).toFixed(2)} GB` : `${freedMb.toFixed(1)} MB`;
      this.toast("success", result.message || t("models.files_deleted", { count: result.deleted_files_count || 0, size: freedText }));

      await this.refreshConfig();
      if (this.state.selectedModel === modelId) {
        if (variantId && this.state.selectedVariant === variantId) this.state.selectedVariant = null;
        await this.onModelChange(this.state.selectedVariant || null);
      }
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      this.toast("error", t("models.delete_weights_failed") + err.message);
      if (btnEl) {
        btnEl.disabled = false;
        btnEl.classList.remove("is-loading");
      }
    }
  },

  openModelVisibilityModal(modelId, event) {
    if (event) event.stopPropagation();
    const modal = document.getElementById("model-visibility-modal");
    if (!modal) return;
    const model = (this.state.models || []).find(m => m.id === modelId);
    if (!model) return;

    const idInput = document.getElementById("model-vis-id");
    const descEl = document.getElementById("model-vis-title-desc");
    const chkGuests = document.getElementById("model-vis-hidden-guests");
    const chkHost = document.getElementById("model-vis-hidden-host");

    if (idInput) idInput.value = model.id;
    if (descEl) descEl.textContent = t("models.vis_model", { name: model.name, id: model.id });
    if (chkGuests) chkGuests.checked = !!model.hidden_for_guests;
    if (chkHost) chkHost.checked = !!model.hidden_from_host;

    modal.style.display = "flex";
  },

  closeModelVisibilityModal() {
    const modal = document.getElementById("model-visibility-modal");
    if (modal) modal.style.display = "none";
  },

  async saveModelVisibility() {
    const idInput = document.getElementById("model-vis-id");
    const chkGuests = document.getElementById("model-vis-hidden-guests");
    const chkHost = document.getElementById("model-vis-hidden-host");
    const modelId = idInput?.value;
    if (!modelId) return;

    const payload = {
      hidden_for_guests: !!chkGuests?.checked,
      hidden_from_host: !!chkHost?.checked,
    };

    try {
      await this.apiPost(`/api/models/${encodeURIComponent(modelId)}/visibility`, payload);
      this.toast("success", t("models.vis_saved", { id: modelId }));
      this.closeModelVisibilityModal();
      await this.refreshConfig();
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  // ============================================================
  // Dock Box Controls (Position, Transparency, Minimized)
  // ============================================================
  initDockDrag() {
    const wrap = document.getElementById("gemini-dock-wrap");
    const handle = document.getElementById("dock-drag-handle");
    if (!wrap || !handle || this._dockDragInit) return;
    this._dockDragInit = true;

    handle.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      this.resetDockPosition();
    });

    let isDragging = false;
    let startX = 0, startY = 0, origX = 0, origY = 0;

    // Drag starts ONLY from the ⋮⋮ handle, and only when the explicit drag
    // toggle is enabled. Popovers, prompts and buttons therefore never fight
    // with the dock drag gesture.
    handle.addEventListener("pointerdown", (e) => {
      if (!this.state.dockDragEnabled) return;
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = wrap.getBoundingClientRect();
      origX = rect.left;
      origY = rect.top;
      try { handle.setPointerCapture(e.pointerId); } catch (_) {}
      wrap.classList.add("dock-dragging");
    });

    const onMove = (e) => {
      if (!isDragging) return;
      const bounds = this.getDockBounds();
      const nx = Math.max(bounds.minX, Math.min(bounds.maxX, origX + (e.clientX - startX)));
      const ny = Math.max(bounds.minY, Math.min(bounds.maxY, origY + (e.clientY - startY)));
      this.state.dockCustomX = Math.round(nx);
      this.state.dockCustomY = Math.round(ny);
      wrap.style.left = `${this.state.dockCustomX}px`;
      wrap.style.top = `${this.state.dockCustomY}px`;
    };

    const stopDrag = (e) => {
      if (!isDragging) return;
      isDragging = false;
      wrap.classList.remove("dock-dragging");
      try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
      this.saveDockSettings();
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", stopDrag);
    window.addEventListener("pointercancel", stopDrag);

    window.addEventListener("resize", () => {
      if (!this.state.dockDragEnabled) return;
      this.clampDockPosition();
      this.applyDockSettings();
      this.saveDockSettings();
    });
  },

  getDockBounds() {
    const wrap = document.getElementById("gemini-dock-wrap");
    const width = wrap ? wrap.offsetWidth : 420;
    const height = wrap ? wrap.offsetHeight : 240;
    return {
      minX: 8,
      minY: 8,
      maxX: Math.max(8, window.innerWidth - width - 8),
      maxY: Math.max(8, window.innerHeight - height - 8),
    };
  },

  clampDockPosition() {
    if (!Number.isFinite(this.state.dockCustomX) || !Number.isFinite(this.state.dockCustomY)) return;
    const bounds = this.getDockBounds();
    this.state.dockCustomX = Math.round(Math.max(bounds.minX, Math.min(bounds.maxX, this.state.dockCustomX)));
    this.state.dockCustomY = Math.round(Math.max(bounds.minY, Math.min(bounds.maxY, this.state.dockCustomY)));
  },

  initDockSettings() {
    try {
      const raw = localStorage.getItem("lazysd_dock");
      const cfg = raw ? JSON.parse(raw) : {};
      if (["center", "left", "right", "bottom"].includes(cfg.position)) this.state.dockPosition = cfg.position;
      if (Number.isFinite(Number(cfg.opacity))) this.state.dockOpacity = Number(cfg.opacity);
      if (Number.isFinite(Number(cfg.blur))) this.state.dockBlur = Number(cfg.blur);
      if (typeof cfg.minimized === "boolean") this.state.dockMinimized = cfg.minimized;
      if (typeof cfg.dragEnabled === "boolean") this.state.dockDragEnabled = cfg.dragEnabled;
      if (["legacy", "new"].includes(cfg.variant)) this.state.dockVariant = cfg.variant;
      if (typeof cfg.newDockFloat === "boolean") this.state.newDockFloat = cfg.newDockFloat;
      if (cfg.newDockCustomX != null && cfg.newDockCustomY != null
        && Number.isFinite(Number(cfg.newDockCustomX)) && Number.isFinite(Number(cfg.newDockCustomY))) {
        this.state.newDockCustomX = Number(cfg.newDockCustomX);
        this.state.newDockCustomY = Number(cfg.newDockCustomY);
      }
      if (cfg.customX != null && cfg.customY != null
        && Number.isFinite(Number(cfg.customX)) && Number.isFinite(Number(cfg.customY))) {
        this.state.dockCustomX = Number(cfg.customX);
        this.state.dockCustomY = Number(cfg.customY);
      }
    } catch (_) {}

    if (this.state.dockDragEnabled && (!Number.isFinite(this.state.dockCustomX) || !Number.isFinite(this.state.dockCustomY))) {
      const wrap = document.getElementById("gemini-dock-wrap");
      if (wrap) {
        const rect = wrap.getBoundingClientRect();
        this.state.dockCustomX = Math.round(rect.left);
        this.state.dockCustomY = Math.round(rect.top);
      }
    }
    this.clampDockPosition();
    this.applyDockSettings();
    this.initDockDrag();
    this.initNewDock();
  },

  applyDockSettings() {
    const wrap = document.getElementById("gemini-dock-wrap");
    if (!wrap) return;

    wrap.classList.remove("dock-pos-left", "dock-pos-right", "dock-pos-bottom", "dock-pos-center", "dock-minimized", "dock-free-drag");

    const hasCustomPosition = Number.isFinite(this.state.dockCustomX) && Number.isFinite(this.state.dockCustomY);
    if (this.state.dockDragEnabled && hasCustomPosition) {
      this.clampDockPosition();
      wrap.classList.add("dock-free-drag");
      wrap.style.left = `${this.state.dockCustomX}px`;
      wrap.style.top = `${this.state.dockCustomY}px`;
      wrap.style.transform = "none";
      wrap.style.bottom = "auto";
      wrap.style.right = "auto";
    } else {
      wrap.style.left = "";
      wrap.style.top = "";
      wrap.style.transform = "";
      wrap.style.bottom = "";
      wrap.style.right = "";
      if (this.state.dockPosition && this.state.dockPosition !== "center") {
        wrap.classList.add(`dock-pos-${this.state.dockPosition}`);
      }
    }
    if (this.state.dockMinimized) wrap.classList.add("dock-minimized");

    const stage = document.getElementById("canvas-stage");
    if (stage) stage.classList.toggle("dock-variant-new", this.state.dockVariant === "new");
    this.applyNewDockSettings();

    const legacyPositionRow = document.getElementById("setting-dock-position-row");
    const legacyDragRow = document.getElementById("setting-dock-legacy-drag-row");
    const legacySettingsVisible = this.state.dockVariant !== "new";
    if (legacyPositionRow) legacyPositionRow.style.display = legacySettingsVisible ? "" : "none";
    if (legacyDragRow) legacyDragRow.style.display = legacySettingsVisible ? "" : "none";

    const root = document.documentElement;
    root.style.setProperty("--dock-opacity", this.state.dockOpacity);
    root.style.setProperty("--dock-blur", `${this.state.dockBlur}px`);

    const posSelect = document.getElementById("setting-dock-pos");
    if (posSelect) posSelect.value = this.state.dockPosition || "center";

    const opSlider = document.getElementById("setting-dock-opacity");
    const opLabel = document.getElementById("setting-dock-opacity-val");
    if (opSlider) opSlider.value = this.state.dockOpacity;
    if (opLabel) opLabel.textContent = `${Math.round(this.state.dockOpacity * 100)}%`;

    const blurSlider = document.getElementById("setting-dock-blur");
    const blurLabel = document.getElementById("setting-dock-blur-val");
    if (blurSlider) blurSlider.value = this.state.dockBlur;
    if (blurLabel) blurLabel.textContent = `${this.state.dockBlur}px`;

    const dragBtn = document.getElementById("dock-drag-toggle-btn");
    if (dragBtn) {
      dragBtn.classList.toggle("active", this.state.dockDragEnabled);
      dragBtn.setAttribute("aria-pressed", this.state.dockDragEnabled ? "true" : "false");
      dragBtn.title = this.state.dockDragEnabled
        ? (t("dock.drag_disable") || "Отключить свободное перемещение")
        : (t("dock.drag_enable") || "Включить свободное перемещение");
    }
    const settingDragBtn = document.getElementById("setting-dock-drag-toggle");
    if (settingDragBtn) {
      settingDragBtn.classList.toggle("active", this.state.dockDragEnabled);
      settingDragBtn.textContent = this.state.dockDragEnabled
        ? (t("dock.drag_on_short") || "Drag включён")
        : (t("dock.drag_off_short") || "Включить Drag");
    }
  },

  saveDockSettings() {
    try {
      localStorage.setItem("lazysd_dock", JSON.stringify({
        position: this.state.dockPosition,
        opacity: this.state.dockOpacity,
        blur: this.state.dockBlur,
        minimized: this.state.dockMinimized,
        dragEnabled: this.state.dockDragEnabled,
        customX: this.state.dockCustomX,
        customY: this.state.dockCustomY,
        variant: this.state.dockVariant,
        newDockFloat: this.state.newDockFloat,
        newDockCustomX: this.state.newDockCustomX,
        newDockCustomY: this.state.newDockCustomY,
      }));
    } catch (_) {}
  },

  setDockPosition(pos) {
    this.state.dockPosition = pos;
    this.state.dockDragEnabled = false;
    this.state.dockCustomX = null;
    this.state.dockCustomY = null;
    this.applyDockSettings();
    this.saveDockSettings();
  },

  toggleDockDrag() {
    const wrap = document.getElementById("gemini-dock-wrap");
    if (!wrap) return;
    if (!this.state.dockDragEnabled) {
      if (!Number.isFinite(this.state.dockCustomX) || !Number.isFinite(this.state.dockCustomY)) {
        const rect = wrap.getBoundingClientRect();
        this.state.dockCustomX = Math.round(rect.left);
        this.state.dockCustomY = Math.round(rect.top);
      }
      this.state.dockDragEnabled = true;
      this.clampDockPosition();
    } else {
      this.state.dockDragEnabled = false;
    }
    this.applyDockSettings();
    this.saveDockSettings();
    this.toast("info", this.state.dockDragEnabled
      ? (t("dock.drag_on") || "Свободное перемещение включено: тяните за ⋮⋮")
      : (t("dock.drag_off") || "Свободное перемещение отключено"));
  },

  resetDockPosition() {
    this.state.dockDragEnabled = false;
    this.state.dockPosition = "center";
    this.state.dockCustomX = null;
    this.state.dockCustomY = null;
    this.state.dockMinimized = false;
    this.applyDockSettings();
    this.saveDockSettings();
    this.toast("info", t("dock.reset_done") || "Док возвращён в нижний центр");
  },

  cycleDockPosition() {
    const order = ["center", "left", "right", "bottom"];
    const curIdx = order.indexOf(this.state.dockPosition || "center");
    const next = order[(curIdx + 1) % order.length];
    this.setDockPosition(next);
    this.toast("info", t("settings.dock.position_toast", { pos: next }));
  },

  toggleDockMinimized() {
    this.state.dockMinimized = !this.state.dockMinimized;
    this.applyDockSettings();
    this.saveDockSettings();
  },

  setDockOpacity(val) {
    const num = parseFloat(val);
    if (isNaN(num)) return;
    this.state.dockOpacity = Math.max(0.2, Math.min(1.0, num));
    this.applyDockSettings();
    this.saveDockSettings();
  },

  setDockBlur(val) {
    const num = parseInt(val, 10);
    if (isNaN(num)) return;
    this.state.dockBlur = Math.max(0, Math.min(40, num));
    this.applyDockSettings();
    this.saveDockSettings();
  },

  // ============================================================
  // Compare Curtain / Slider (Шторка)
  // ============================================================
  toggleCompare() {
    const container = document.getElementById("compare-container");
    const mainImg = document.getElementById("preview-img");
    const btn = document.getElementById("btn-compare-toggle");
    if (!container || !mainImg) return;

    if (this.state.compareActive) {
      this.state.compareActive = false;
      container.style.display = "none";
      mainImg.style.display = "block";
      if (btn) btn.classList.remove("active");
      return;
    }

    const hasRef = Boolean(this.state.img2imgRef?.url);
    const fgSrc = this.state.currentPreviewUrl || (mainImg.src && !mainImg.src.endsWith("#") ? mainImg.src : null);
    // Prioritize imported document/reference as the baseline (left/bg)
    const bgSrc = hasRef ? this.state.img2imgRef.url : (this.state.previousPreviewUrl || null);

    if (!fgSrc) {
      this.toast("warning", "Нет изображения для сравнения. Сгенерируйте кадр.");
      return;
    }
    if (!bgSrc) {
      this.toast("warning", "Нет второго кадра для сравнения (загрузите документ/референс или сгенерируйте ещё один кадр).");
      return;
    }

    this.state.compareActive = true;
    container.style.display = "flex";
    mainImg.style.display = "none";
    if (btn) btn.classList.add("active");

    const bgImg = document.getElementById("compare-bg-img");
    const fgImg = document.getElementById("compare-fg-img");
    if (bgImg) bgImg.src = bgSrc;
    if (fgImg) fgImg.src = fgSrc;

    const leftLabel = document.getElementById("compare-label-left");
    const rightLabel = document.getElementById("compare-label-right");
    if (leftLabel) {
      leftLabel.textContent = hasRef ? "Документ / Исходник" : "Пред. кадр";
    }
    if (rightLabel) {
      rightLabel.textContent = "Результат";
    }

    this.updateCompareSlider(this.state.comparePercentage || 50);
  },

  updateCompareImages() {
    const bgImg = document.getElementById("compare-bg-img");
    const fgImg = document.getElementById("compare-fg-img");
    const leftLabel = document.getElementById("compare-label-left");
    const rightLabel = document.getElementById("compare-label-right");
    const hasRef = Boolean(this.state.img2imgRef?.url);

    if (fgImg && this.state.currentPreviewUrl) fgImg.src = this.state.currentPreviewUrl;
    if (bgImg) {
      if (hasRef) {
        bgImg.src = this.state.img2imgRef.url;
      } else if (this.state.previousPreviewUrl) {
        bgImg.src = this.state.previousPreviewUrl;
      }
    }
    if (leftLabel) {
      leftLabel.textContent = hasRef ? "Документ / Исходник" : "Пред. кадр";
    }
    if (rightLabel) {
      rightLabel.textContent = "Результат";
    }
  },

  updateCompareSlider(pct) {
    pct = Math.max(0, Math.min(100, pct));
    this.state.comparePercentage = pct;
    const clipEl = document.getElementById("compare-fg-clip");
    const barEl = document.getElementById("compare-slider-bar");
    if (clipEl) {
      clipEl.style.clipPath = `inset(0 0 0 ${pct}%)`;
    }
    if (barEl) {
      barEl.style.left = `${pct}%`;
    }
  },

  initCompareEvents() {
    const container = document.getElementById("compare-container");
    if (!container) return;

    let isDragging = false;
    const onMove = (e) => {
      if (!isDragging) return;
      const rect = container.getBoundingClientRect();
      if (!rect.width) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const x = clientX - rect.left;
      const pct = (x / rect.width) * 100;
      this.updateCompareSlider(pct);
    };

    container.addEventListener("pointerdown", (e) => {
      isDragging = true;
      try { container.setPointerCapture(e.pointerId); } catch (_) {}
      const rect = container.getBoundingClientRect();
      if (rect.width) {
        this.updateCompareSlider(((e.clientX - rect.left) / rect.width) * 100);
      }
    });

    container.addEventListener("pointermove", onMove);
    const endDrag = (e) => {
      if (isDragging) {
        isDragging = false;
        try { container.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    };
    container.addEventListener("pointerup", endDrag);
    container.addEventListener("pointercancel", endDrag);
  },

  // ============================================================
  // LoRA Management & Stacking
  // ============================================================
  currentLoraModelId() {
    const select = document.getElementById("model-select");
    return this.state.selectedModel || (select ? select.value : "") || "";
  },

  async loadLoras(notifyIncompatible = true) {
    const modelId = this.currentLoraModelId();
    const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : "";
    try {
      const res = await this.apiGet(`/api/loras${query}`);
      this.state.loraLibrary = Array.isArray(res) ? res : (res.loras || []);
      this.pruneIncompatibleLoras(notifyIncompatible);
      this.renderLoraUI();
    } catch (e) {
      console.warn("[lora] Failed to load loras:", e);
    }
  },

  loraCompatibility(item) {
    return item && item.compatibility ? item.compatibility : null;
  },

  isLoraCompatible(item) {
    const compat = this.loraCompatibility(item);
    return !compat || compat.compatible !== false;
  },

  loraCompatBadge(item) {
    const compat = this.loraCompatibility(item);
    if (!compat) return "";
    if (compat.compatible === false) {
      return `<span class="lora-compat-badge incompatible" title="${escapeHtml(compat.reason || "")}">несовместима</span>`;
    }
    const labels = { exact: "точная", target_architecture: "100% архитектура", architecture: "архитектура", family: "семейство", engine: "движок" };
    const label = labels[compat.match_level] || "совместима";
    return `<span class="lora-compat-badge compatible" title="${escapeHtml(compat.reason || "")}">${escapeHtml(label)}</span>`;
  },

  pruneIncompatibleLoras(notify = true) {
    const blocked = [];
    this.state.activeLoras = this.state.activeLoras.filter(active => {
      const item = this.state.loraLibrary.find(l => l.name === active.name);
      if (item && !this.isLoraCompatible(item)) {
        blocked.push(active.name);
        return false;
      }
      return true;
    });
    if (blocked.length && notify) {
      this.toast("warning", t("lora.hub.blocked", { names: blocked.join(", ") }));
    }
    return blocked;
  },

  renderLoraUI() {
    const count = this.state.activeLoras.length;
    const pillText = document.getElementById("dock-lora-pill-text");
    if (pillText) {
      pillText.textContent = `LoRA (${count})`;
    }
    const dockCount = document.getElementById("dock-lora-count");
    if (dockCount) {
      dockCount.textContent = `${count} активных`;
    }
    const inspectorCount = document.getElementById("inspector-lora-count");
    if (inspectorCount) {
      inspectorCount.textContent = String(count);
    }

    const dockList = document.getElementById("dock-lora-list");
    const dockEmpty = document.getElementById("dock-lora-empty");
    if (dockList && dockEmpty) {
      if (this.state.loraLibrary.length === 0) {
        dockEmpty.style.display = "block";
        dockList.innerHTML = "";
      } else {
        dockEmpty.style.display = "none";
        dockList.innerHTML = this.state.loraLibrary.map(item => {
          const activeItem = this.state.activeLoras.find(l => l.name === item.name);
          const isActive = !!activeItem;
          const incompatible = !this.isLoraCompatible(item);
          const strength = activeItem ? activeItem.strength : (Number(item.scale_default) || 1.0);
          const family = item.family ? `<span class="lora-meta-tag">${escapeHtml(item.family)}</span>` : "";
          const size = item.size_mb ? `<span class="lora-meta-tag">${item.size_mb} MB</span>` : "";
          return `
            <div class="lora-card-item ${isActive ? 'active' : ''} ${incompatible ? 'incompatible' : ''}">
              <div class="lora-card-header">
                <div class="lora-title-group" onclick="App.toggleLora('${escapeHtml(item.name)}')" style="cursor: pointer;">
                  <input type="checkbox" ${isActive ? 'checked' : ''} ${incompatible && !isActive ? 'disabled' : ''} onclick="event.stopPropagation(); App.toggleLora('${escapeHtml(item.name)}')">
                  <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
                  ${family}
                  ${size}
                  ${this.loraCompatBadge(item)}
                </div>
                <div style="display: flex; gap: 2px;">
                  <button type="button" class="dock-ctrl-btn" onclick="App.openLoraBinding('${escapeHtml(item.name)}')" title="Привязать к моделям">⚙</button>
                  <button type="button" class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить файл LoRA">✕</button>
                </div>
              </div>
              ${isActive ? `
                <div class="lora-slider-row">
                  <input type="range" min="-2.0" max="2.0" step="0.05" value="${strength}" oninput="App.setLoraStrength('${escapeHtml(item.name)}', this.value)">
                  <span class="lora-weight-val" id="lora-weight-val-${escapeHtml(item.name)}">${strength.toFixed(2)}</span>
                </div>
              ` : ''}
            </div>
          `;
        }).join("");
      }
    }

    const ndList = document.getElementById("new-dock-lora-list");
    const ndEmpty = document.getElementById("new-dock-lora-empty");
    const ndCount = document.getElementById("new-dock-lora-count");
    if (ndCount) ndCount.textContent = String(count);
    if (ndList && ndEmpty) {
      ndEmpty.style.display = this.state.loraLibrary.length ? "none" : "block";
      ndList.innerHTML = this.state.loraLibrary.map(item => {
        const activeItem = this.state.activeLoras.find(l => l.name === item.name);
        const isActive = !!activeItem;
        const incompatible = !this.isLoraCompatible(item);
        const strength = activeItem ? activeItem.strength : (Number(item.scale_default) || 1.0);
        const safe = escapeHtml(item.name);
        return `
          <div class="new-dock-lora-row ${isActive ? "active" : ""} ${incompatible ? "incompatible" : ""}">
            <div class="new-dock-lora-head" onclick="App.toggleLora('${safe}')">
              <input type="checkbox" ${isActive ? "checked" : ""} ${incompatible && !isActive ? "disabled" : ""} onclick="event.stopPropagation(); App.toggleLora('${safe}')">
              <span class="new-dock-lora-name" title="${safe}">${safe}</span>
              ${this.loraCompatBadge(item)}
            </div>
            ${isActive ? `
              <div class="new-dock-lora-slider">
                <input type="range" min="-2.0" max="2.0" step="0.05" value="${strength}" oninput="App.setLoraStrength('${safe}', this.value)">
                <span id="lora-weight-nd-${safe}">${Number(strength).toFixed(2)}</span>
              </div>` : ""}
          </div>`;
      }).join("");
    }

    const inspectorList = document.getElementById("lora-inspector-list");
    if (inspectorList) {
      if (this.state.activeLoras.length === 0) {
        inspectorList.innerHTML = `<div style="font-size: 11px; color: var(--text-dim); text-align: center; padding: 6px 0;">${escapeHtml(t("inspector.no_loras"))}</div>`;
      } else {
        inspectorList.innerHTML = this.state.activeLoras.map(item => {
          return `
            <div class="lora-card-item active">
              <div class="lora-card-header">
                <div class="lora-title-group">
                  <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
                </div>
                <button type="button" class="dock-ctrl-btn" onclick="App.toggleLora('${escapeHtml(item.name)}')" title="Отключить">✕</button>
              </div>
              <div class="lora-slider-row">
                <input type="range" min="-2.0" max="2.0" step="0.05" value="${item.strength}" oninput="App.setLoraStrength('${escapeHtml(item.name)}', this.value)">
                <span class="lora-weight-val" id="lora-weight-insp-${escapeHtml(item.name)}">${item.strength.toFixed(2)}</span>
              </div>
            </div>
          `;
        }).join("");
      }
    }

    const modalTotal = document.getElementById("lora-modal-total");
    if (modalTotal) modalTotal.textContent = String(this.state.loraLibrary.length);
    const hubCount = document.getElementById("models-hub-lora-count");
    if (hubCount) hubCount.textContent = String(this.state.loraLibrary.length);

    this.renderLoraHubGrid();
    this.renderLoraLibrary();
  },

  renderLoraHubGrid() {
    const grid = document.getElementById("lora-hub-grid");
    if (!grid) return;

    const allLoras = this.state.loraLibrary || [];
    const q = (document.getElementById("lora-hub-search")?.value || "").trim().toLowerCase();
    let list = allLoras;
    if (q) {
      list = list.filter(l =>
        (l.name || "").toLowerCase().includes(q) ||
        (l.label || "").toLowerCase().includes(q) ||
        (l.trigger || "").toLowerCase().includes(q) ||
        (l.family || "").toLowerCase().includes(q)
      );
    }

    if (list.length === 0) {
      const message = allLoras.length === 0
        ? t("lora.hub.empty")
        : t("lora.hub.no_match");
      grid.innerHTML = `<div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--text-dim);">${message}</div>`;
      return;
    }

    grid.innerHTML = list.map(item => {
      const activeItem = this.state.activeLoras.find(l => l.name === item.name);
      const isActive = !!activeItem;
      const compat = this.loraCompatibility(item);
      const incompatible = compat && compat.compatible === false;
      const family = item.family
        ? `<span class="model-badge" style="background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3);">${escapeHtml(item.family)}</span>`
        : "";
      const size = item.size_mb
        ? `<span class="model-badge" style="background: rgba(255, 255, 255, 0.05); color: var(--text-muted);">${item.size_mb} MB</span>`
        : "";
      const bound = (compat && compat.bound_models && compat.bound_models.length)
        ? `<span class="model-badge" style="background: rgba(168, 85, 247, 0.15); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.3);">🔗 ${compat.bound_models.length}</span>`
        : "";
      return `
        <div class="model-card ${isActive ? 'active' : ''}" style="display: flex; flex-direction: column; justify-content: space-between; min-height: 140px;">
          <div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
              <div style="font-weight: 600; font-size: 13.5px; color: var(--text); word-break: break-all;" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
              <div style="display: flex; gap: 4px;">
                <button class="dock-ctrl-btn" onclick="App.openLoraBinding('${escapeHtml(item.name)}')" title="Привязать к моделям и настроить архитектуру">⚙️</button>
                <button class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить">✕</button>
              </div>
            </div>
            <div style="display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px;">
              ${family}
              ${size}
              ${bound}
              ${this.loraCompatBadge(item)}
            </div>
            ${incompatible
              ? `<div class="lora-incompat-reason" title="${escapeHtml(compat.reason || "")}">${escapeHtml(compat.reason || "Несовместима с выбранной моделью")}</div>`
              : ""}
            ${item.trigger ? `<div style="font-size: 11px; color: #93c5fd; background: rgba(59, 130, 246, 0.1); border-radius: 4px; padding: 3px 6px; margin-bottom: 8px; cursor: pointer;" onclick="App.insertLoraTrigger('${escapeHtml(item.trigger)}')" title="Нажмите, чтобы вставить в промпт">💬 ${escapeHtml(item.trigger)}</div>` : ''}
          </div>
          <div style="display: flex; gap: 8px; align-items: center; margin-top: 8px;">
            <button class="card-action-btn ${isActive ? 'danger' : 'primary'}" style="flex: 1; height: 30px; font-size: 12px;" ${incompatible && !isActive ? 'disabled' : ''} onclick="App.toggleLora('${escapeHtml(item.name)}')">
              ${isActive ? 'Отключить' : incompatible ? 'Несовместима' : 'Подключить'}
            </button>
          </div>
        </div>
      `;
    }).join("");
  },

  renderLoraLibrary() {
    const listEl = document.getElementById("lora-modal-list-wrap");
    if (!listEl) return;

    const allLoras = this.state.loraLibrary || [];
    const q = (document.getElementById("lora-library-search")?.value || "").trim().toLowerCase();
    let list = allLoras;
    if (q) {
      list = list.filter(l =>
        (l.name || "").toLowerCase().includes(q) ||
        (l.label || "").toLowerCase().includes(q) ||
        (l.trigger || "").toLowerCase().includes(q)
      );
    }

    if (list.length === 0) {
      const message = allLoras.length === 0 ? "В библиотеке пока нет LoRA." : "Ничего не найдено.";
      listEl.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-dim);">${message}</div>`;
      return;
    }

    listEl.innerHTML = list.map(item => {
      const isActive = this.state.activeLoras.some(l => l.name === item.name);
      const incompatible = !this.isLoraCompatible(item);
      const meta = [
        item.size_mb ? `${item.size_mb} MB` : "",
        item.architecture || item.family || "",
        item.trigger ? `💬 ${item.trigger}` : "",
      ].filter(Boolean).join(" · ");
      return `
        <div class="lora-card-item ${isActive ? 'active' : ''} ${incompatible ? 'incompatible' : ''}" style="margin-bottom: 6px;">
          <div class="lora-card-header">
            <div class="lora-title-group" onclick="App.toggleLora('${escapeHtml(item.name)}')" style="cursor: pointer; min-width: 0;">
              <input type="checkbox" ${isActive ? 'checked' : ''} ${incompatible && !isActive ? 'disabled' : ''} onclick="event.stopPropagation(); App.toggleLora('${escapeHtml(item.name)}')">
              <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
            </div>
            <div style="display: flex; gap: 4px; align-items: center; flex-shrink: 0;">
              ${this.loraCompatBadge(item)}
              <span style="font-size: 10px; color: var(--text-dim);">${escapeHtml(meta)}</span>
              <button type="button" class="dock-ctrl-btn" onclick="App.openLoraBinding('${escapeHtml(item.name)}')" title="Привязка к моделям">⚙</button>
              <button type="button" class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить">✕</button>
            </div>
          </div>
        </div>
      `;
    }).join("");
  },

  toggleLora(name) {
    const idx = this.state.activeLoras.findIndex(l => l.name === name);
    if (idx >= 0) {
      this.state.activeLoras.splice(idx, 1);
    } else {
      const item = this.state.loraLibrary.find(l => l.name === name);
      if (item && !this.isLoraCompatible(item)) {
        const reason = this.loraCompatibility(item)?.reason || "несовместима с выбранной моделью";
        this.toast("warning", `LoRA «${name}»: ${reason}`);
        return;
      }
      const scale = Number(item?.scale_default);
      this.state.activeLoras.push({
        name: name,
        strength: Number.isFinite(scale) && scale !== 0 ? scale : 1.0,
        info: item || {},
      });
    }
    this.renderLoraUI();
  },

  setLoraStrength(name, val) {
    const num = parseFloat(val);
    if (isNaN(num)) return;
    const item = this.state.activeLoras.find(l => l.name === name);
    if (item) {
      item.strength = Math.round(num * 100) / 100;
      const el1 = document.getElementById(`lora-weight-val-${name}`);
      const el2 = document.getElementById(`lora-weight-insp-${name}`);
      if (el1) el1.textContent = item.strength.toFixed(2);
      if (el2) el2.textContent = item.strength.toFixed(2);
      const el3 = document.getElementById(`lora-weight-nd-${name}`);
      if (el3) el3.textContent = item.strength.toFixed(2);
    }
  },

  openLoraModal() {
    this.openLoraManageModal("hub");
  },

  closeLoraModal() {
    const modal = document.getElementById("lora-manage-modal");
    if (modal) modal.style.display = "none";
  },

  async onLoraFileSelected(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    const fd = new FormData();
    fd.append("file", file);

    this.toast("info", t("lora.hub.uploading", { name: file.name }));
    try {
      const res = await fetch("/api/loras/upload", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Upload failed");
      }
      this.toast("success", t("lora.hub.uploaded", { name: file.name }));
      input.value = "";
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async importLocalLora() {
    const input = document.getElementById("lora-local-path-input");
    const path = input ? input.value.trim() : "";
    if (!path) {
      this.toast("warning", t("lora.hub.path_required"));
      return;
    }
    try {
      this.toast("info", t("lora.hub.importing"));
      const res = await this.apiPost("/api/loras/import", { path: path });
      this.toast("success", t("lora.hub.imported", { name: res.lora?.name || "OK" }));
      if (input) input.value = "";
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async downloadHfLora() {
    const repoInput = document.getElementById("lora-hf-repo-input");
    const fileSelect = document.getElementById("lora-hf-file-select");
    const renameInput = document.getElementById("lora-hf-rename-input");
    const familySelect = document.getElementById("lora-hf-family-select");
    const triggerInput = document.getElementById("lora-hf-trigger-input");
    const repo = repoInput ? repoInput.value.trim() : "";
    const filename = fileSelect ? fileSelect.value : "";
    if (!repo) {
      this.toast("warning", t("lora.hub.repo_required"));
      return;
    }

    try {
      this.toast("info", t("lora.hub.downloading", { repo }));
      const res = await this.apiPost("/api/loras/download", {
        repo_id: repo,
        filename: filename || null,
        rename: renameInput?.value.trim() || null,
        family: familySelect?.value && familySelect.value !== "any" ? familySelect.value : null,
        trigger: triggerInput?.value.trim() || null,
      });
      this.toast("success", t("lora.hub.downloaded", { name: res.lora?.name }));
      if (renameInput) renameInput.value = "";
      if (triggerInput) triggerInput.value = "";
      const inspector = document.getElementById("lora-hf-inspector");
      if (inspector) inspector.style.display = "none";
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  openLoraBinding(name) {
    if (this.state.account?.is_guest) {
      this.toast("warning", t("lora.hub.guest_denied"));
      return;
    }
    const item = this.state.loraLibrary.find(l => l.name === name);
    if (!item) {
      this.toast("error", t("lora.hub.not_found", { name }));
      return;
    }
    this.state.loraBinding = {
      name,
      item,
      models: new Set(item.compatible_models || []),
      engines: new Set((item.engine || []).map(value => String(value).toLowerCase())),
    };

    const nameEl = document.getElementById("lora-binding-name");
    if (nameEl) nameEl.value = name;
    const archSelect = document.getElementById("lora-binding-architecture");
    if (archSelect) {
      const architecture = String(item.target_architecture || item.architecture || item.family || "any").toLowerCase();
      archSelect.innerHTML = `
        <option value="flux_2_klein_9b">⚡ Flux.2 Klein 9B (все движки: Nunchaku / OrbitQuant / GGUF)</option>
        <option value="flux_2_klein_4b">⚡ Flux.2 Klein 4B (Nunchaku / OrbitQuant)</option>
        <option value="flux_3_image_klein">⚡ FLUX.3 Image Klein</option>
        <option value="flux_1">⚡ FLUX.1 (Dev / Schnell)</option>
        <option value="qwen_image_2_1">⚡ Qwen-Image 2.1 (все движки)</option>
        <option value="z_image_turbo">⚡ Z-Image Turbo</option>
        <option value="sdxl">⚡ SDXL / Pony / Illustrious</option>
        <option value="any">Любая / универсальная</option>`;
      if (!Array.from(archSelect.options).some(option => option.value === architecture) && architecture) {
        archSelect.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(architecture)}">${escapeHtml(architecture)}</option>`);
      }
      archSelect.value = architecture || "any";
    }
    const scaleEl = document.getElementById("lora-binding-scale");
    if (scaleEl) scaleEl.value = item.scale_default ?? 1.0;
    const triggerEl = document.getElementById("lora-binding-trigger");
    if (triggerEl) triggerEl.value = item.trigger || "";
    const targetsEl = document.getElementById("lora-binding-targets");
    if (targetsEl) targetsEl.value = (item.target_modules || []).join(", ");

    this.renderLoraBindingEngines();
    this.renderLoraBindingModels();
    const modal = document.getElementById("lora-binding-modal");
    if (modal) modal.style.display = "flex";
  },

  closeLoraBinding() {
    const modal = document.getElementById("lora-binding-modal");
    if (modal) modal.style.display = "none";
    this.state.loraBinding = null;
  },

  renderLoraBindingEngines() {
    const bind = this.state.loraBinding;
    const host = document.getElementById("lora-binding-engines");
    if (!bind || !host) return;
    const engines = [
      ["sd_cpp", "SD.CPP (GGUF)"],
      ["quantfunc", "QuantFunc"],
      ["nunchaku", "Nunchaku (SVDQ)"],
      ["sdnq", "SDNQ"],
      ["orbitquant", "OrbitQuant (сейчас без LoRA)"],
    ];
    host.innerHTML = engines.map(([value, label]) => `
      <label class="card-editor-check ${value === "orbitquant" ? "check-disabled" : ""}">
        <input type="checkbox" value="${value}" ${bind.engines.has(value) ? "checked" : ""} onchange="App.toggleLoraBindingEngine('${value}', this.checked)">
        <span>${escapeHtml(label)}</span>
      </label>`).join("");
  },

  toggleLoraBindingEngine(engine, checked) {
    const bind = this.state.loraBinding;
    if (!bind) return;
    if (checked) bind.engines.add(engine);
    else bind.engines.delete(engine);
  },

  renderLoraBindingModels() {
    const bind = this.state.loraBinding;
    const host = document.getElementById("lora-binding-models");
    if (!bind || !host) return;
    const currentModelId = this.currentLoraModelId();
    const models = (this.state.models || []).slice().sort((a, b) => {
      if (a.id === currentModelId) return -1;
      if (b.id === currentModelId) return 1;
      return String(a.name || a.id).localeCompare(String(b.name || b.id));
    });
    host.innerHTML = models.map(model => `
      <label class="card-editor-check">
        <input type="checkbox" value="${escapeHtml(model.id)}" ${bind.models.has(model.id) ? "checked" : ""} onchange="App.toggleLoraBindingModel('${escapeHtml(model.id)}', this.checked)">
        <span>${escapeHtml(model.name || model.id)}</span>
        <span class="card-editor-check-meta">${escapeHtml([model.engine, model.family].filter(Boolean).join(" · "))}${model.id === currentModelId ? " · текущая" : ""}</span>
      </label>`).join("") || `<div class="component-empty">Модели не загружены.</div>`;
  },

  toggleLoraBindingModel(modelId, checked) {
    const bind = this.state.loraBinding;
    if (!bind) return;
    if (checked) bind.models.add(modelId);
    else bind.models.delete(modelId);
  },

  async saveLoraBinding() {
    const bind = this.state.loraBinding;
    if (!bind) return;
    const value = id => document.getElementById(id)?.value ?? "";
    const architecture = value("lora-binding-architecture").trim();
    const scaleRaw = value("lora-binding-scale").trim();
    const payload = {
      architecture: architecture === "any" ? "" : architecture,
      target_architecture: architecture === "any" ? "" : architecture,
      trigger: value("lora-binding-trigger").trim(),
      target_modules: value("lora-binding-targets").trim(),
      scale_default: scaleRaw === "" ? null : parseFloat(scaleRaw),
      engine: Array.from(bind.engines),
      compatible_models: Array.from(bind.models),
    };
    try {
      await this.apiPatch(`/api/loras/${encodeURIComponent(bind.name)}/metadata`, payload);
      this.toast("success", t("lora.hub.binding_saved"));
      this.closeLoraBinding();
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async bindLoraToCurrentModel() {
    const bind = this.state.loraBinding;
    const modelId = this.currentLoraModelId();
    if (!bind || !modelId) {
      this.toast("warning", t("lora.hub.select_model_first"));
      return;
    }
    bind.models.add(modelId);
    this.renderLoraBindingModels();
    await this.saveLoraBinding();
  },

  async clearLoraBinding() {
    const bind = this.state.loraBinding;
    if (!bind) return;
    bind.models.clear();
    bind.engines.clear();
    this.renderLoraBindingModels();
    this.renderLoraBindingEngines();
    await this.saveLoraBinding();
  },

  insertLoraTrigger(trigger) {
    const promptInput = document.getElementById("prompt-input");
    if (!promptInput || !trigger) return;
    const current = promptInput.value.trim();
    promptInput.value = current ? `${current}, ${trigger}` : trigger;
    promptInput.dispatchEvent(new Event("input", { bubbles: true }));
    this.toast("info", t("lora.hub.trigger_added", { trigger }));
  },

  async deleteLora(name) {
    if (!confirm(`Удалить LoRA «${name}» с диска?`)) return;
    try {
      await fetch(`/api/loras/${encodeURIComponent(name)}`, { method: "DELETE" });
      this.toast("info", t("lora.hub.deleted", { name }));
      const idx = this.state.activeLoras.findIndex(l => l.name === name);
      if (idx >= 0) this.state.activeLoras.splice(idx, 1);
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  // ============================================================
  // Remote Sharing (Gradio-like without Gradio)
  // ============================================================
  async loadShareStatus() {
    try {
      const data = await this.apiGet("/api/share/status");
      this.state.shareStatus = data;
      this.renderShareUI();
    } catch (_) {}
  },

  renderShareUI() {
    const status = this.state.shareStatus;
    if (!status) return;

    const badge = document.getElementById("setting-share-status-badge");
    const urlWrap = document.getElementById("setting-share-url-wrap");
    const urlInput = document.getElementById("setting-share-url");
    const toggleBtn = document.getElementById("btn-share-toggle");
    const copyBtn = document.getElementById("btn-share-copy");
    const tokenInput = document.getElementById("setting-share-token");
    const autoCheck = document.getElementById("setting-share-autostart");

    if (tokenInput && status.token) tokenInput.value = status.token;
    if (autoCheck) autoCheck.checked = !!status.auto_start;

    const providerSelect = document.getElementById("setting-share-provider");
    const activeProvider = status.tunnel?.provider || status.share_tunnel_provider || "cloudflared";
    if (providerSelect) {
      providerSelect.value = activeProvider;
      providerSelect.disabled = !!status.running;
    }

    if (status.running && status.public_url) {
      if (badge) {
        badge.className = "state-badge installed";
        const provLabel = status.tunnel?.provider_label || (activeProvider === "gradio" ? "Gradio Live" : "Cloudflare");
        badge.textContent = `✓ Активен (${provLabel})`;
      }
      if (urlWrap) urlWrap.style.display = "block";
      if (urlInput) urlInput.value = status.share_url || status.public_url;
      if (toggleBtn) toggleBtn.textContent = t("settings.share.stop");
      if (copyBtn) copyBtn.style.display = "inline-block";
    } else {
      if (badge) {
        badge.className = "state-badge missing";
        badge.textContent = t("settings.share.status_stopped");
      }
      if (urlWrap) urlWrap.style.display = "none";
      if (toggleBtn) {
        const provLabel = activeProvider === "gradio" ? "Gradio Live" : "Cloudflare";
        toggleBtn.textContent = `Запустить ${provLabel}`;
      }
      if (copyBtn) copyBtn.style.display = "none";
    }
  },

  async onShareProviderChange(provider) {
    try {
      await this.apiPost("/api/share/options", { share_tunnel_provider: provider });
      const provName = provider === "gradio" ? "Gradio Live" : "Cloudflare";
      this.toast("info", `Выбран провайдер туннеля: ${provName}`);
      await this.loadShareStatus();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async toggleShareTunnel() {
    const isRunning = this.state.shareStatus?.running;
    const btn = document.getElementById("btn-share-toggle");
    const providerSelect = document.getElementById("setting-share-provider");
    const provider = providerSelect ? providerSelect.value : (this.state.shareStatus?.tunnel?.provider || "cloudflared");
    const provName = provider === "gradio" ? "Gradio Live" : "Cloudflare";
    try {
      if (isRunning) {
        if (btn) btn.textContent = "Остановка...";
        this.toast("info", t("settings.share.stopping"));
        await this.apiPost("/api/share/stop", {});
        this.toast("info", t("settings.share.stopped"));
        await this.loadShareStatus();
      } else {
        if (btn) btn.textContent = `Запуск ${provName}...`;
        this.toast("info", `Запуск туннеля ${provName}...`);
        await this.apiPost("/api/share/start", { provider });

        // Poll for public trycloudflare URL (takes ~1-3s to establish)
        for (let i = 0; i < 8; i++) {
          await new Promise((r) => setTimeout(r, 1200));
          await this.loadShareStatus();
          const st = this.state.shareStatus;
          if (st && st.running && st.public_url) {
            this.toast("success", t("settings.share.active", { url: st.public_url }));
            return;
          }
          if (st && st.error) {
            throw new Error(st.error);
          }
        }
        this.toast("info", t("settings.share.starting_wait"));
      }
    } catch (e) {
      this.toast("error", e.message);
      await this.loadShareStatus();
    }
  },

  copyShareUrl() {
    const input = document.getElementById("setting-share-url");
    if (!input || !input.value) return;
    navigator.clipboard.writeText(input.value);
    this.toast("success", t("settings.share.copied"));
  },

  generateShareToken() {
    const token = Array.from(crypto.getRandomValues(new Uint8Array(16)))
      .map(b => b.toString(16).padStart(2, "0")).join("");
    const input = document.getElementById("setting-share-token");
    if (input) input.value = token;
  },

  async saveShareToken() {
    const input = document.getElementById("setting-share-token");
    const val = input ? input.value.trim() : "";
    try {
      await this.apiPost("/api/share/token", { token: val || null });
      this.toast("success", t("settings.share.token_saved"));
      await this.loadShareStatus();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async saveShareAutoStart(checked) {
    try {
      await this.apiPost("/api/settings", { share_auto_start: checked });
      this.toast("info", t("settings.share.autostart_toast", { state: checked ? t("common.enabled") : t("common.disabled") }));
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  // ============================================================
  // Extras & Engines (OrbitQuant & Nunchaku)
  // ============================================================
  async refreshExtrasStatus(showToast = false) {
    try {
      const data = await this.apiGet("/api/extras/status?refresh=true");
      this.state.extrasStatus = data;
      this.renderExtrasUI();
      if (showToast) {
        const isReady = data.env?.ready;
        const msg = isReady
          ? `✓ Окружение Extras готово: Python ${data.env?.python || ''}, PyTorch ${data.env?.torch || ''}`
          : `Окружение Extras: ${data.env?.summary || 'требуется установка компонентов'}`;
        this.toast(isReady ? "success" : "info", msg);
      }
    } catch (e) {
      if (showToast) {
        this.toast("error", "Не удалось опросить статус Extras: " + e.message);
      }
    }
  },

  renderExtrasUI() {
    const data = this.state.extrasStatus;
    if (!data) return;

    const nunchakuBadge = document.getElementById("setting-nunchaku-status");
    const orbitBadge = document.getElementById("setting-orbitquant-status");
    const envDesc = document.getElementById("setting-extras-env-desc");
    const customPyInput = document.getElementById("setting-extras-custom-py");

    if (customPyInput && data.env?.python_path) {
      if (!customPyInput.value && !data.env.managed) {
        customPyInput.value = data.env.python_path;
      }
    }

    if (envDesc && data.env) {
      const e = data.env;
      if (e.ready) {
        envDesc.innerHTML = `<span style="color: #4ade80; font-weight: 600;">${escapeHtml(t("settings.engines.env_ready"))}</span> Python ${e.python || '3.12'} · Torch ${e.torch || 'OK'} · CUDA ${e.cuda || 'OK'} (${e.device || 'GPU'}) · Diffusers ${e.diffusers || 'OK'}`;
      } else if (e.python_path || e.venv_exists) {
        envDesc.innerHTML = `<span style="color: #fbbf24; font-weight: 600;">${escapeHtml(t("settings.engines.env_setup"))}</span> ${escapeHtml(e.error || e.summary || t("settings.engines.env_setup_default"))}`;
      } else {
        envDesc.innerHTML = `<span style="color: var(--text-dim);">ℹ️ ${escapeHtml(e.summary || t("settings.engines.env_absent"))}</span>`;
      }
    }

    if (nunchakuBadge) {
      if (data.nunchaku_ready) {
        nunchakuBadge.innerHTML = `<span class="state-badge installed">✓ ${escapeHtml(t("common.installed"))} (${escapeHtml(data.env?.nunchaku || 'ready')})</span>`;
      } else {
        nunchakuBadge.innerHTML = `<span class="state-badge missing">${escapeHtml(t("common.not_installed"))}</span>`;
      }
    }

    if (orbitBadge) {
      if (data.orbitquant_ready) {
        orbitBadge.innerHTML = `<span class="state-badge installed">✓ ${escapeHtml(t("common.installed"))} (${escapeHtml(data.env?.orbitquant || 'ready')})</span>`;
      } else {
        orbitBadge.innerHTML = `<span class="state-badge missing">${escapeHtml(t("common.not_installed"))}</span>`;
      }
    }
  },

  async uninstallEngine(engine, label) {
    const extrasNote = (engine === "nunchaku" || engine === "orbitquant")
      ? t("settings.engine.uninstall_confirm_extras")
      : "";
    const confirmed = confirm(t("settings.engine.uninstall_confirm", { label }) + extrasNote);
    if (!confirmed) return;
    try {
      this.toast("info", t("settings.engine.uninstalling", { label }));
      const res = await this.apiPost("/api/engine/uninstall", { engine });
      this.toast("success", t("settings.engine.uninstalled", { label }));
      await this.refreshConfig();
      if (engine === "nunchaku" || engine === "orbitquant") {
        await this.refreshExtrasStatus(false);
      }
    } catch (e) {
      this.toast("error", t("settings.engine.uninstall_failed", { label }) + e.message);
    }
  },

  async saveExtrasPythonPath() {
    const input = document.getElementById("setting-extras-custom-py");
    const val = input ? input.value.trim() : "";
    try {
      this.toast("info", "Сохранение пути к Python...");
      const res = await this.apiPost("/api/extras/python", { path: val });
      this.toast("success", "✓ Путь к Extras Python обновлён");
      await this.refreshExtrasStatus(false);
    } catch (e) {
      this.toast("error", "Ошибка: " + e.message);
    }
  },

  async installExtrasPackage(target, btnEl = null, force = false) {
    const btn = btnEl || document.getElementById(`btn-install-${target}`);
    const origText = btn ? btn.textContent : `Установить ${target}`;
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Установка...";
    }
    const labelTarget = target === "runtime" ? t("settings.engine.extras_create_env") : target;
    this.toast("info", t("settings.engine.install_started", { label: labelTarget }));

    try {
      const res = await this.apiPost("/api/extras/install", { component: target, target: target, force: Boolean(force) });
      const jobId = res.job?.id || res.id;
      if (!jobId) throw new Error(res.error || "Не удалось получить ID задачи установки");

      const poll = setInterval(async () => {
        try {
          const job = await this.apiGet(`/api/extras/jobs/${jobId}`);
          const state = job.state || job.status;
          const label = job.step_label || job.stage || 'Установка';
          if (btn) btn.textContent = `${label}...`;

          if (state === "done" || state === "succeeded") {
            clearInterval(poll);
            if (btn) {
              btn.disabled = false;
              btn.textContent = origText;
            }
            this.toast("success", t("settings.engine.install_done", { label: labelTarget }));
            await this.refreshExtrasStatus(false);
          } else if (state === "error" || state === "failed" || state === "cancelled") {
            clearInterval(poll);
            if (btn) {
              btn.disabled = false;
              btn.textContent = origText;
            }
            this.toast("error", t("settings.engine.install_failed", { label: target }) + (job.error || "failed"));
          }
        } catch (_) {
          clearInterval(poll);
        }
      }, 2000);
    } catch (e) {
      if (btn) {
        btn.disabled = false;
        btn.textContent = origText;
      }
      this.toast("error", e.message);
    }
  },
};

// Global handlers for inline HTML onclick attributes
window.loadHistory = () => App.loadHistory();
window.closeHistoryModal = () => App.closeHistoryModal();
window.clearImg2ImgRef = () => App.clearImg2ImgRef();
window.updateStrengthLabel = (val) => App.updateStrengthLabel(val);
window.sendCurrentPreviewToImg2Img = () => App.sendCurrentPreviewToImg2Img();



// ============================================================
// --- INTEGRATED ZOOM & PAN ENGINE ---
// ============================================================
(function() {
/* ============================================================
   QuantFunc WebUI — Viewport Zoom & Pan Engine
   Smooth cursor-centered zoom, pan, floating HUD, and scrollbars.
   Shared across Generation, Halftone Studio GT, and Manga Studio.
   ============================================================ */


const viewports = {};
let viewportsInitialized = false;
let spacePressed = false;

function initAllViewportZoomPan(appRef = null) {
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
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      spacePressed = false;
    }
  });
}

function initViewportZoomPan(mode, cfg, appRef = null) {
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

function getViewportPanBounds(vp) {
  const vw = vp.area.clientWidth || 1;
  const vh = vp.area.clientHeight || 1;
  const baseW = vp.wrap.offsetWidth || vw * 0.8;
  const baseH = vp.wrap.offsetHeight || vh * 0.8;
  const cw = baseW * vp.zoom;
  const ch = baseH * vp.zoom;

  const marginX = Math.min(vw, baseW) * 0.28;
  const marginY = Math.min(vh, baseH) * 0.28;
  const maxPanX = vp.zoom <= 1.0
    ? marginX * 0.5
    : Math.max(marginX * 0.5, (cw - vw) * 0.5 + marginX);
  const maxPanY = vp.zoom <= 1.0
    ? marginY * 0.5
    : Math.max(marginY * 0.5, (ch - vh) * 0.5 + marginY);

  return { vw, vh, baseW, baseH, cw, ch, maxPanX, maxPanY };
}

function applyViewportTransform(vp, showHud = true) {
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

  // Let mode-specific controllers refine the label and react to zoom/pan
  // without coupling the shared viewport engine to Halftone/Manga internals.
  try {
    window.dispatchEvent(new CustomEvent("viewport-transform", {
      detail: { mode: vp.mode, zoom: vp.zoom, panX: vp.panX, panY: vp.panY },
    }));
  } catch (_) {}

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

function showViewportHudAndScrollbars(vp) {
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

function initViewportHudOpacity() {
  const saved = localStorage.getItem("lazysd_zoom_hud_opacity");
  if (saved) {
    document.documentElement.style.setProperty("--zoom-hud-opacity", saved);
  }
}

function cycleViewportHudOpacity() {
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

function stepViewportZoom(mode, dir) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  const factor = dir > 0 ? 1.25 : 1 / 1.25;
  vp.zoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
  applyViewportTransform(vp, true);
}

function setViewportZoom(mode, targetZoom, showHud = true) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  vp.zoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, Number(targetZoom) || 1.0));
  applyViewportTransform(vp, showHud);
}

function resetViewportZoom(mode) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  vp.zoom = 1.0;
  vp.panX = 0;
  vp.panY = 0;
  applyViewportTransform(vp, true);
}

function getViewportState(mode) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return null;
  return { mode: vp.mode, zoom: vp.zoom, panX: vp.panX, panY: vp.panY };
}


  Object.assign(App, {
    initAllViewportZoomPan,
    initViewportZoomPan,
    getViewportPanBounds,
    applyViewportTransform,
    showViewportHudAndScrollbars,
    initViewportHudOpacity,
    cycleViewportHudOpacity,
    stepViewportZoom,
    setViewportZoom,
    resetViewportZoom,
    getViewportState,
  });
})();

// ============================================================
// --- INTEGRATED HALFTONE STUDIO GT ENGINE (WebGL2 + 100% Full-Res) ---
// ============================================================
(function() {
  const apiPost = (url, body, retries = 0) => App.apiPost(url, body, retries);
  const apiGet = (url) => App.apiGet(url);
  const toast = (type, msg) => App.toast(type, msg);

/* ============================================================
   QuantFunc WebUI — Halftone Studio GT Engine
   Multi-layer screentone system, polygon lasso/brush masking,
   and Real-Time WebGL2 GPU Shader Engine (60–144 FPS, <0.3ms latency).
   100% WYSIWYG parity with Python backend CPU engine.
   ============================================================ */


const halftoneBaseExtraParams = {
  satellite_enabled: false,
  satellite_count: 4,
  satellite_size_ratio: 0.45,
  satellite_distance: 0.42,
  satellite_angle: 0.0,
  satellite_shape: "circle",
  satellite_stretch: 0.0,
  merge_factor: 0.0,
  aspect_x: 1.0,
  aspect_y: 1.0,
  row_offset: 0.0,
  jitter_pos: 0.0,
  jitter_size: 0.0,
  grad_type: "none",
  grad_angle: 45.0,
  grad_strength: 0.0,
  qualityMode: "full",
};

const halftonePresets = {
  manga: {
    mode: "manga", cell_size: 8, dot_shape: "circle", angle: 45, max_radius_ratio: 0.9,
    contrast: 1.25, gamma: 1.05, brightness: 0.0, black_threshold: 0.18, white_threshold: 0.86,
    invert: false, intensity: 1.0, dot_color: "#121212", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.0, preserve_lines: true, line_threshold: 0.24,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  },
  classic: {
    mode: "classic", cell_size: 10, dot_shape: "circle", angle: 45, max_radius_ratio: 0.82,
    contrast: 1.05, gamma: 1.0, brightness: 0.0, black_threshold: 0.06, white_threshold: 0.94,
    invert: false, intensity: 1.0, dot_color: "#111111", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.3, preserve_lines: false, line_threshold: 0.22,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  },
  newspaper: {
    mode: "newspaper", cell_size: 12, dot_shape: "circle", angle: 45, max_radius_ratio: 0.88,
    contrast: 1.15, gamma: 0.95, brightness: 0.0, black_threshold: 0.10, white_threshold: 0.90,
    invert: false, intensity: 1.0, dot_color: "#1f1e1c", bg_color: "#f4efe4", antialias: 2,
    dither_type: "noise", seed: 42, pre_blur: 0.4, preserve_lines: false, line_threshold: 0.25,
    paper_texture: 0.35, invert_mask: false, ...halftoneBaseExtraParams
  },
  color: {
    mode: "color", cell_size: 9, dot_shape: "circle", angle: 45, max_radius_ratio: 0.85,
    contrast: 1.10, gamma: 1.0, brightness: 0.0, black_threshold: 0.05, white_threshold: 0.95,
    invert: false, intensity: 1.0, dot_color: "#111111", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.2, preserve_lines: false, line_threshold: 0.22,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  },
  custom: {
    mode: "custom", cell_size: 8, dot_shape: "circle", angle: 45, max_radius_ratio: 0.85,
    contrast: 1.0, gamma: 1.0, brightness: 0.0, black_threshold: 0.10, white_threshold: 0.90,
    invert: false, intensity: 1.0, dot_color: "#111111", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.0, preserve_lines: true, line_threshold: 0.22,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  }
};

let glContext = null;
let glProgram = null;
let glSourceTexture = null;
let glMaskTexture = null;
let glUniformLocations = {};
// Ping-pong FBOs used by the multi-layer WebGL2 compositor.
let glFboTextures = [null, null];
let glFbos = [null, null];
let glFboSize = { w: 0, h: 0 };

// Zero-allocation reusable buffers for CPU compositor fallback
const _htBuffers = {
  _htAccR: null,
  _htBufN: 0,
};

// ============================================================
// WebGL2 Screentone Shader Sources (Ported from GenTonik & Python)
// ============================================================
const VS_SOURCE = `#version 300 es
in vec2 a_position;
uniform int u_flip_v;
out vec2 v_uv;
void main() {
    v_uv = (a_position + 1.0) * 0.5;
    // Image textures are uploaded top-row-first, while FBO textures are
    // bottom-row-first. Layer passes render into FBOs with u_flip_v=0 and the
    // final blit to the canvas uses u_flip_v=1, keeping both orientations exact.
    if (u_flip_v == 1) v_uv.y = 1.0 - v_uv.y;
    gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FS_SOURCE = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;
uniform sampler2D u_base;
uniform sampler2D u_mask;
uniform int u_has_mask;
uniform int u_invert_mask;

uniform vec2 u_resolution;
uniform float u_cell_size;
uniform int u_dot_shape; // 0..12 geometry shapes
uniform float u_angle; // radians
uniform float u_max_radius_ratio;
uniform float u_contrast;
uniform float u_gamma;
uniform float u_brightness;
uniform float u_black_threshold;
uniform float u_white_threshold;
uniform int u_invert;
uniform float u_intensity;
uniform vec3 u_dot_color;
uniform vec3 u_bg_color;
uniform int u_preserve_lines;
uniform float u_line_threshold;
uniform vec2 u_aspect;
uniform float u_row_offset;
uniform float u_jitter_pos;
uniform float u_jitter_size;

// GenTonik Planets & Satellites
uniform int u_satEnabled;
uniform int u_satCount;
uniform float u_satSizeRatio;
uniform float u_satDist;
uniform float u_satAngleBase;
uniform float u_satStretch;
uniform int u_satShape;
uniform float u_mergeFactor;

// Spatial Gradients
uniform int u_gradType; // 0=none, 1=linear, 2=radial, 3=vignette
uniform float u_gradAngle; // radians
uniform float u_gradStrength;

// Layer Opacity, Blend Mode & AA
uniform float u_opacity;
uniform int u_blend_mode; // 0=normal, 1=multiply, 2=screen, 3=overlay, 4=darken, 5=lighten
uniform int u_antialias;  // 1=1x (hard 1-bit), 2..4=AA
uniform int u_mode;       // 0=classic, 1=manga, 2=color (CMYK)
uniform int u_pass_mode;  // 0=halftone layer, 1=blit composite to canvas

// 13 Geometry Shapes distance field (exact 1:1 math match with backend/halftone.py)
float evalShapeDist(vec2 p, float halfR, int shape, out bool isLinear) {
    isLinear = false;
    vec2 ap = abs(p);

    // 0: circle
    if (shape == 0) {
        return length(p) / (halfR * 1.128379);
    }
    // 1: square
    if (shape == 1) {
        return max(ap.x, ap.y) / halfR;
    }
    // 2: diamond
    if (shape == 2) {
        return (ap.x + ap.y) / (halfR * 1.41421356);
    }
    // 3: line
    if (shape == 3) {
        isLinear = true;
        return ap.y / halfR;
    }
    // 4: crosshatch
    if (shape == 4) {
        isLinear = true;
        return min(ap.x, ap.y) / halfR;
    }
    // 5: hexagon
    if (shape == 5) {
        float hexD = max(ap.x * 0.8660254 + ap.y * 0.5, ap.y);
        return hexD / (halfR * 0.95);
    }
    // 6: cross
    if (shape == 6) {
        float d = min(max(ap.x * 2.6, ap.y * 0.85), max(ap.x * 0.85, ap.y * 2.6));
        return d / (halfR * 0.9);
    }
    // 7: ellipse
    if (shape == 7) {
        return length(vec2(p.x * 0.82, p.y * 1.22)) / (halfR * 1.05);
    }
    // 8: triangle
    if (shape == 8) {
        float ty = p.y + 0.08 * (halfR * 2.0);
        float triD = max(ap.x * 1.73205 + ty, -ty * 2.0) / 1.5;
        return max(0.0, triD) / (halfR * 0.95);
    }
    // 9: star
    if (shape == 9) {
        float r = length(p);
        float theta = atan(p.y, p.x);
        float starMod = 0.68 + 0.32 * cos(5.0 * theta - 1.570796327);
        return r / (halfR * 1.22 * max(starMod, 0.2));
    }
    // 10: heart
    if (shape == 10) {
        float hx = p.x / (halfR * 1.15);
        float hy = -p.y / (halfR * 1.15) + 0.15;
        float r = length(vec2(hx, hy - 0.35 * sqrt(abs(hx) + 0.0001)));
        return r / 0.85;
    }
    // 11: concentric
    if (shape == 11) {
        isLinear = true;
        float r = length(p) / (halfR * 1.2);
        return abs((r * 2.0) - floor(r * 2.0) - 0.5) * 2.0;
    }
    // 12: ring
    if (shape == 12) {
        isLinear = true;
        float r = length(p) / (halfR * 1.1);
        return abs(r - 0.52) * 2.3;
    }
    // Default circle
    return length(p) / (halfR * 1.128379);
}

float coverageAA(float d, float targetR, int aa, float fallbackFw) {
    if (aa <= 1) {
        return (d < targetR) ? 1.0 : 0.0;
    }
    float fw = max(fwidth(d), fallbackFw) * (float(aa) * 0.45);
    fw = max(fw, 0.0001);
    return clamp(0.5 - (d - targetR) / fw, 0.0, 1.0);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 blendLayerRgb(vec3 b, vec3 t, int mode) {
    if (mode == 1) return b * t;                                  // multiply
    if (mode == 2) return 1.0 - (1.0 - b) * (1.0 - t);            // screen
    if (mode == 3) {                                              // overlay
        vec3 res;
        res.r = (b.r < 0.5) ? (2.0 * b.r * t.r) : (1.0 - 2.0 * (1.0 - b.r) * (1.0 - t.r));
        res.g = (b.g < 0.5) ? (2.0 * b.g * t.g) : (1.0 - 2.0 * (1.0 - b.g) * (1.0 - t.g));
        res.b = (b.b < 0.5) ? (2.0 * b.b * t.b) : (1.0 - 2.0 * (1.0 - b.b) * (1.0 - t.b));
        return clamp(res, 0.0, 1.0);
    }
    if (mode == 4) return min(b, t);                              // darken
    if (mode == 5) return max(b, t);                              // lighten
    return t;                                                     // normal
}

float evalDotInk(vec2 pixelPos, vec2 center, float cov, float angleRad, vec2 pitch, float halfR, int shape, int aa, float rowOff) {
    if (cov <= 0.0001) return 0.0;
    if (cov >= 0.9999) return 1.0;
    vec2 d = pixelPos - center;
    float cosA = cos(angleRad);
    float sinA = sin(angleRad);
    float u = (d.x * cosA + d.y * sinA) / pitch.x;
    float v = (-d.x * sinA + d.y * cosA) / pitch.y;
    float cellV = floor(v);
    if (abs(rowOff) > 0.0001) {
        float oddRow = mod(abs(cellV), 2.0);
        u = u + oddRow * rowOff;
    }
    float cellU = floor(u);
    float du = (u - cellU) - 0.5;
    float dv = (v - cellV) - 0.5;
    bool isLinear = false;
    float dist = evalShapeDist(vec2(du, dv), halfR, shape, isLinear);
    float targetR = isLinear ? clamp(cov, 0.0, 1.0) : sqrt(clamp(cov, 0.0, 1.0));
    float cellPx = 1.0 / min(pitch.x, pitch.y);
    float fallbackFw = cellPx / max(halfR, 0.02);
    return coverageAA(dist, targetR, aa, fallbackFw);
}

void main() {
    vec4 srcColor = texture(u_image, v_uv);

    // Final pass: present the composited FBO to the canvas without re-toning.
    if (u_pass_mode == 1) {
        outColor = vec4(texture(u_base, v_uv).rgb, 1.0);
        return;
    }

    vec3 toneColor = vec3(0.0);

    // If CMYK Mode (u_mode == 2): Multi-channel subtractive halftoning
    if (u_mode == 2) {
        vec3 rgbNorm = srcColor.rgb;
        if (u_invert == 1) rgbNorm = 1.0 - rgbNorm;
        rgbNorm = clamp(rgbNorm + u_brightness, 0.0, 1.0);
        float span = max(0.001, u_white_threshold - u_black_threshold);
        rgbNorm = clamp((rgbNorm - u_black_threshold) / span, 0.0, 1.0);
        rgbNorm = pow(clamp(rgbNorm, 0.0, 1.0), vec3(1.0 / max(0.01, u_gamma)));
        rgbNorm = clamp((rgbNorm - 0.5) * u_contrast + 0.5, 0.0, 1.0);

        float kChan = clamp((1.0 - max(max(rgbNorm.r, rgbNorm.g), rgbNorm.b)) * 0.85, 0.0, 1.0);
        float denom = max(1.0 - kChan, 0.0001);
        float cChan = clamp((1.0 - rgbNorm.r - kChan) / denom, 0.0, 1.0);
        float mChan = clamp((1.0 - rgbNorm.g - kChan) / denom, 0.0, 1.0);
        float yChan = clamp((1.0 - rgbNorm.b - kChan) / denom, 0.0, 1.0);

        vec2 pixelPos = v_uv * u_resolution;
        vec2 center = u_resolution * 0.5;
        vec2 pitch = max(vec2(1.5), vec2(u_cell_size) * u_aspect);
        float halfR = max(0.05, 0.5 * u_max_radius_ratio);
        float effRowOffset = (u_dot_shape == 5 && abs(u_row_offset) < 0.0001) ? 0.5 : u_row_offset;

        // Screen each channel at standard offset angles: C: 15°, M: 75°, Y: 0°, K: 45°
        float cDot = evalDotInk(pixelPos, center, cChan, u_angle + 0.261799, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);
        float mDot = evalDotInk(pixelPos, center, mChan, u_angle + 1.308997, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);
        float yDot = evalDotInk(pixelPos, center, yChan, u_angle + 0.000000, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);
        float kDot = evalDotInk(pixelPos, center, kChan, u_angle + 0.785398, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);

        vec3 cmykRgb = vec3(
            u_bg_color.r * (1.0 - cDot) * (1.0 - kDot),
            u_bg_color.g * (1.0 - mDot) * (1.0 - kDot),
            u_bg_color.b * (1.0 - yDot) * (1.0 - kDot)
        );

        if (u_preserve_lines == 1) {
            vec2 texel = 1.0 / u_resolution;
            float lCenter = dot(srcColor.rgb, vec3(0.299, 0.587, 0.114));
            float lLeft   = dot(texture(u_image, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
            float lRight  = dot(texture(u_image, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
            float lUp     = dot(texture(u_image, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
            float lDown   = dot(texture(u_image, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
            float edgeMag = length(vec2(lRight - lLeft, lDown - lUp));
            if ((edgeMag > 0.08 && lCenter < 0.65) || lCenter < u_line_threshold) {
                cmykRgb = vec3(0.0);
            }
        }

        toneColor = cmykRgb;
    } else {
    // Monochromatic & Manga Modes (u_mode 0 or 1)
    float gray = dot(srcColor.rgb, vec3(0.299, 0.587, 0.114));
    float g = (u_invert == 1) ? (1.0 - gray) : gray;
    g = clamp(g + u_brightness, 0.0, 1.0);
    float span = max(0.001, u_white_threshold - u_black_threshold);
    g = clamp((g - u_black_threshold) / span, 0.0, 1.0);
    g = pow(clamp(g, 0.0, 1.0), 1.0 / max(0.01, u_gamma));
    g = clamp((g - 0.5) * u_contrast + 0.5, 0.0, 1.0);

    float inkAmount = 1.0 - g;
    if (u_mode == 1) {
        if (g >= 0.94) inkAmount = 0.0;
        if (g <= 0.06) inkAmount = 1.0;
    }

    float covMod = inkAmount;
    if (u_gradType > 0 && abs(u_gradStrength) > 0.001) {
        vec2 gCoord = v_uv - 0.5;
        float gMap = 0.5;
        if (u_gradType == 1) {
            gMap = (gCoord.x * cos(u_gradAngle) + gCoord.y * sin(u_gradAngle)) + 0.5;
        } else if (u_gradType == 2) {
            gMap = clamp(1.0 - length(gCoord) * 1.41421356, 0.0, 1.0);
        } else if (u_gradType == 3) {
            gMap = clamp(pow(length(gCoord) * 1.41421356, 1.6) - 0.25, 0.0, 1.0);
        }
        covMod = clamp(mix(covMod, covMod * gMap, u_gradStrength), 0.0, 1.0);
    }

    vec2 pixelPos = v_uv * u_resolution;
    vec2 center = u_resolution * 0.5;
    vec2 d = pixelPos - center;
    float cosA = cos(u_angle);
    float sinA = sin(u_angle);
    vec2 pitch = max(vec2(1.5), vec2(u_cell_size) * u_aspect);

    float u = (d.x * cosA + d.y * sinA) / pitch.x;
    float v = (-d.x * sinA + d.y * cosA) / pitch.y;

    float cellV = floor(v);
    float effRowOffset = (u_dot_shape == 5 && abs(u_row_offset) < 0.0001) ? 0.5 : u_row_offset;
    if (abs(effRowOffset) > 0.0001) {
        float oddRow = mod(abs(cellV), 2.0);
        u = u + oddRow * effRowOffset;
    }
    float cellU = floor(u);
    float du = (u - cellU) - 0.5;
    float dv = (v - cellV) - 0.5;

    vec2 cellCoord = vec2(cellU, cellV);
    if (u_jitter_pos > 0.001) {
        float jx = (hash21(cellCoord + 11.0) - 0.5) * u_jitter_pos * 0.55;
        float jy = (hash21(cellCoord + 29.0) - 0.5) * u_jitter_pos * 0.55;
        du -= jx;
        dv -= jy;
    }

    float halfR = max(0.05, 0.5 * u_max_radius_ratio);
    bool isLinear = false;
    float dist = evalShapeDist(vec2(du, dv), halfR, u_dot_shape, isLinear);
    float targetR = isLinear ? clamp(covMod, 0.0, 1.0) : sqrt(clamp(covMod, 0.0, 1.0));
    if (u_jitter_size > 0.001) {
        targetR *= (1.0 + (hash21(cellCoord) - 0.5) * u_jitter_size);
    }
    float cellPx = 1.0 / min(pitch.x, pitch.y);
    float fallbackFw = cellPx / max(halfR, 0.02);
    float ink = coverageAA(dist, targetR, u_antialias, fallbackFw);

    // GenTonik Planets & Satellites (1..8 slot orbit)
    if (u_satEnabled == 1 && u_satCount > 0) {
        float satDist = max(0.12, u_satDist * (1.0 - u_mergeFactor * 0.5));
        float satMaxR = max(0.08, halfR * u_satSizeRatio * (1.0 + u_mergeFactor * 0.45));
        int effCount = clamp(u_satCount, 1, 8);
        float stepAng = 6.283185307 / float(effCount);
        float satFallbackFw = cellPx / max(satMaxR, 0.02);

        for (int i = 0; i < 8; i++) {
            if (i >= effCount) break;
            float theta = u_satAngleBase + float(i) * stepAng;
            float rsx = satDist * cos(theta);
            float rsy = satDist * sin(theta);
            vec2 sPos = vec2(du - rsx, dv - rsy);

            if (u_satStretch > 0.001 && (abs(rsx) > 0.00001 || abs(rsy) > 0.00001)) {
                float dirAngle = atan(-rsy, -rsx);
                float cd = cos(dirAngle);
                float sd = sin(dirAngle);
                float rotU = (sPos.x * cd + sPos.y * sd) / (1.0 + u_satStretch);
                float rotV = -sPos.x * sd + sPos.y * cd;
                sPos.x = rotU * cd - rotV * sd;
                sPos.y = rotU * sd + rotV * cd;
            }

            bool sLinear = false;
            float sDist = evalShapeDist(sPos, satMaxR, u_satShape, sLinear);
            float sTarget = sLinear ? clamp(covMod, 0.0, 1.0) : sqrt(clamp(covMod, 0.0, 1.0));
            float satInk = coverageAA(sDist, sTarget, u_antialias, satFallbackFw);
            ink = max(ink, satInk);
        }
    }

    if (covMod <= 0.0001) ink = 0.0;
    if (u_mode == 1 && g <= 0.04) ink = 1.0;
    if (u_max_radius_ratio >= 0.92 && covMod >= 0.96) ink = 1.0;

    if (u_preserve_lines == 1) {
        vec2 texel = 1.0 / u_resolution;
        float lCenter = dot(srcColor.rgb, vec3(0.299, 0.587, 0.114));
        float lLeft   = dot(texture(u_image, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        float lRight  = dot(texture(u_image, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        float lUp     = dot(texture(u_image, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
        float lDown   = dot(texture(u_image, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
        float edgeMag = length(vec2(lRight - lLeft, lDown - lUp));
        if ((edgeMag > 0.08 && lCenter < 0.65) || lCenter < u_line_threshold) {
            ink = max(ink, 1.0);
        }
    }

    toneColor = mix(u_bg_color, u_dot_color, ink);
    }

    // Multi-pass layer compositor: blend this raster layer over the previous
    // composite (u_base), then apply the layer mask and opacity.
    vec4 baseColor = texture(u_base, v_uv);
    float maskFactor = 1.0;
    if (u_has_mask == 1) {
        vec4 mPix = texture(u_mask, v_uv);
        maskFactor = max(mPix.a, dot(mPix.rgb, vec3(0.299, 0.587, 0.114)));
        if (u_invert_mask == 1) maskFactor = 1.0 - maskFactor;
    }
    vec3 layerRgb = mix(srcColor.rgb, toneColor, clamp(u_intensity, 0.0, 1.0));
    vec3 blended = blendLayerRgb(baseColor.rgb, layerRgb, u_blend_mode);
    float totalAlpha = clamp(u_opacity * maskFactor, 0.0, 1.0);
    outColor = vec4(mix(baseColor.rgb, blended, totalAlpha), 1.0);
}`;

const HT_ORIGINAL_LAYER_ID = "hl_original";
const HT_PREVIEW_QUALITY_DEFAULT = "auto";
const HT_PERFORMANCE_MAX_SIDE = 1920;
const HT_ROI_MIN_ZOOM = 1.08;
const HT_ROI_MARGIN = 0.14;
const HT_GPU_AUTO_MIN_CAP = 2048;
const HT_GPU_MASK_MAX_SIDE = 2048;

function createHalftoneLayer(name, paramsObj = null, options = {}) {
  const opts = options || {};
  return {
    id: "hl_" + Date.now() + "_" + Math.floor(Math.random() * 10000),
    name: name || "Слой растра",
    visible: true,
    opacity: 1.0,
    blend_mode: "normal",
    params: { ...(paramsObj || halftonePresets.manga) },
    maskCanvas: null,
    maskDirty: false,
    _maskRevision: 0,
    _maskDataUrl: null,
    _maskDataUrlRevision: -1,
    _maskDataUrlCanvas: null,
    _glMaskRevision: -1,
    hasMask: Boolean(opts.hasMask),
    isBase: Boolean(opts.isBase),
    isOriginal: false,
    locked: false,
  };
}

function createHalftoneOriginalLayer() {
  return {
    id: HT_ORIGINAL_LAYER_ID,
    name: "Исходное фото",
    visible: true,
    opacity: 1.0,
    blend_mode: "normal",
    params: { ...halftonePresets.manga },
    maskCanvas: null,
    maskDirty: false,
    _maskRevision: 0,
    _maskDataUrl: null,
    _maskDataUrlRevision: -1,
    _maskDataUrlCanvas: null,
    _glMaskRevision: -1,
    hasMask: false,
    isBase: false,
    isOriginal: true,
    locked: true,
  };
}

function ensureHalftoneLayerStack(st) {
  if (!st) return null;
  st.layers = Array.isArray(st.layers) ? st.layers.filter(Boolean) : [];
  let original = st.layers.find((l) => l.isOriginal || l.id === HT_ORIGINAL_LAYER_ID);
  if (!original) {
    original = createHalftoneOriginalLayer();
    st.layers.unshift(original);
  } else {
    original.id = HT_ORIGINAL_LAYER_ID;
    original.isOriginal = true;
    original.locked = true;
    original.isBase = false;
  }
  st.layers.forEach((l) => {
    if (l === original) return;
    l.isOriginal = false;
    l.locked = false;
    if (typeof l.hasMask !== "boolean") l.hasMask = Boolean(l.maskCanvas);
  });
  if (!st.layers.some((l) => !l.isOriginal)) {
    const base = createHalftoneLayer("Базовый растр", st.params || halftonePresets.manga, { isBase: true });
    st.layers.push(base);
  }
  if (!st.layers.some((l) => l.isBase)) {
    const firstRaster = st.layers.find((l) => !l.isOriginal);
    if (firstRaster) firstRaster.isBase = true;
  }
  const active = st.layers.find((l) => l.id === st.activeLayerId);
  if (!active || active.isOriginal) {
    const firstRaster = st.layers.find((l) => !l.isOriginal);
    st.activeLayerId = firstRaster ? firstRaster.id : null;
  }
  return st.layers;
}

function resetHalftoneLayerStack(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const baseParams = { ...(st.params || halftonePresets.manga) };
  const base = createHalftoneLayer("Базовый растр", baseParams, { isBase: true });
  st.layers = [createHalftoneOriginalLayer(), base];
  st.activeLayerId = base.id;
  st.params = { ...baseParams };
  st.comparing = false;
  st.tool = "view";
  const maskCanvas = document.getElementById("ht-mask-canvas");
  if (maskCanvas) {
    const ctx = maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  }
  renderHalftoneLayersUI(app);
  return st.layers;
}

function initHalftoneStudio(appRef) {
  const app = appRef || window.App;
  if (!app) return;
  if (!app.state.halftone) {
    app.state.halftone = {
      params: { ...halftonePresets.manga },
      layers: [],
      activeLayerId: null,
      sourcePath: null,
      sourceUrl: null,
      sourceDataUrl: null,
      resultDataUrl: null,
      comparing: false,
      sourceWidth: 0,
      sourceHeight: 0,
      previewQuality: HT_PREVIEW_QUALITY_DEFAULT,
      roiDataUrl: null,
      roiRect: null,
      _roiSignature: null,
      _roiAbort: null,
      _roiSeq: 0,
      _roiTimer: null,
      _previewAbort: null,
      _previewSeq: 0,
      _editRevision: 0,
      _liveSourcePreparedUrl: null,
      _gpuCanvasMaxSide: 0,
      _viewportRaf: null,
      _layoutTimer: null,
      tool: "view",
      brushSize: 24,
      lassoMode: "replace",
      lassoPoints: [],
      isDrawingLasso: false,
      isDrawingBrush: false,
      maskDirty: false,
      liveReady: false,
      liveActive: false,
      qualityMode: "full",
      debounceTimer: null,
      _htAccR: null,
      _htBufN: 0,
    };
  }

  const st = app.state.halftone;
  st.previewQuality = st.previewQuality || HT_PREVIEW_QUALITY_DEFAULT;
  st.roiRect = st.roiRect || null;
  st._editRevision = Number(st._editRevision) || 0;
  st._previewSeq = Number(st._previewSeq) || 0;
  st._roiSeq = Number(st._roiSeq) || 0;

  ensureHalftoneLayerStack(st);
  ensureWebGLCanvas(app);
  initHalftoneViewportBridge(app);
  initHalftoneResizeObserver(app);
  renderHalftoneLayersUI(app);
  syncHalftoneControlsFromParams(st.params);
  syncHalftonePreviewQualityButtons(st);
  if (st.sourceUrl || st.sourceDataUrl) {
    requestAnimationFrame(() => layoutHalftonePreview(app));
  }
}

function syncHalftonePreviewQualityButtons(st) {
  const quality = st?.previewQuality || HT_PREVIEW_QUALITY_DEFAULT;
  ["auto", "performance", "maximum"].forEach((id) => {
    const btn = document.getElementById(`ht-preview-quality-${id}`);
    if (btn) btn.classList.toggle("active", quality === id);
  });
}

function setHalftonePreviewQuality(quality, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;

  const next = ["auto", "performance", "maximum"].includes(quality)
    ? quality
    : HT_PREVIEW_QUALITY_DEFAULT;
  if (st.previewQuality === next) return;

  st.previewQuality = next;
  syncHalftonePreviewQualityButtons(st);

  // Any cached ROI belongs to the previous calculation quality.
  st.roiDataUrl = null;
  st.roiRect = null;
  st._roiSignature = null;
  st._pendingRoiPlan = null;
  hideHalftoneRoi(app, true);

  const labels = {
    auto: "Авто — превью под экран (DPR) + детальный ROI при зуме",
    performance: "Performance — превью до 1920 px",
    maximum: "Максимум — полное разрешение источника",
  };
  toast("info", `Качество превью: ${labels[next] || next}`);

  if (st.qualityMode === "live") {
    layoutHalftonePreview(app);
    const changed = resizeHalftoneGpuCanvas(app);
    if (changed || st.liveReady) renderHalftoneWebGL(app);
  } else if (st.sourcePath || st.sourceDataUrl) {
    runHalftonePreview(true, app);
  }
}

function initHalftoneViewportBridge(appRef) {
  if (initHalftoneViewportBridge._bound) return;
  initHalftoneViewportBridge._bound = true;
  window.addEventListener("viewport-transform", (event) => {
    const detail = event?.detail;
    if (!detail || detail.mode !== "halftone") return;
    const app = window.App;
    const st = app?.state?.halftone;
    if (!st) return;

    updateHalftoneImageRendering(app);
    updateHalftoneZoomLabel(app);

    if (st.qualityMode === "live") {
      if (st._viewportRaf) return;
      st._viewportRaf = requestAnimationFrame(() => {
        st._viewportRaf = null;
        const changed = resizeHalftoneGpuCanvas(app);
        if (changed) renderHalftoneWebGL(app);
      });
    } else {
      scheduleHalftoneRoiUpdate(app, false);
    }
  });
}

function initHalftoneResizeObserver(appRef) {
  const app = appRef || window.App;
  const area = document.getElementById("ht-dropzone");
  if (!area || area._htResizeObserver) return;

  const observer = new ResizeObserver(() => scheduleHalftoneLayout(app));
  observer.observe(area);
  area._htResizeObserver = observer;

  if (!initHalftoneResizeObserver._windowBound) {
    initHalftoneResizeObserver._windowBound = true;
    window.addEventListener("resize", () => scheduleHalftoneLayout(window.App));
  }
}

function scheduleHalftoneLayout(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !(st.sourceUrl || st.sourceDataUrl)) return;
  const prevFitW = Number(st._fitWidth) || 0;
  const prevFitH = Number(st._fitHeight) || 0;
  layoutHalftonePreview(app);
  const fitChanged =
    Math.abs((Number(st._fitWidth) || 0) - prevFitW) > 0.5 ||
    Math.abs((Number(st._fitHeight) || 0) - prevFitH) > 0.5;
  // A ResizeObserver can fire because our own meta label changed the toolbar
  // height. When the fitted image size is identical, re-rendering would only
  // restart the same request and feed the loop.
  if (!fitChanged && st.resultDataUrl) {
    updateHalftoneRoiPosition(app);
    return;
  }
  if (st._layoutTimer) clearTimeout(st._layoutTimer);
  st._layoutTimer = setTimeout(() => {
    st._layoutTimer = null;
    if (!st.sourceUrl && !st.sourceDataUrl) return;
    if (st.qualityMode === "full") {
      runHalftonePreview(false, app);
    } else {
      const changed = resizeHalftoneGpuCanvas(app);
      if (changed) renderHalftoneWebGL(app);
    }
  }, 160);
}

function ensureHalftoneSourceDimensions(st) {
  if (!st) return Promise.resolve(null);
  if (st.origWidth && st.origHeight) {
    return Promise.resolve({ width: st.origWidth, height: st.origHeight });
  }
  if (st._sourceSizePromise) return st._sourceSizePromise;
  const url = st.sourceUrl || st.sourceDataUrl;
  if (!url) return Promise.resolve(null);

  st._sourceSizePromise = new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      st.origWidth = img.naturalWidth || img.width || 0;
      st.origHeight = img.naturalHeight || img.height || 0;
      st.liveWidth = st.origWidth;
      st.liveHeight = st.origHeight;
      resolve({ width: st.origWidth, height: st.origHeight });
    };
    img.onerror = () => reject(new Error("Failed to load source image"));
    img.src = url;
  });
  return st._sourceSizePromise;
}

function getHalftoneViewportMetrics() {
  const area = document.getElementById("ht-dropzone");
  if (!area) return { width: 0, height: 0, dpr: window.devicePixelRatio || 1 };
  const cs = getComputedStyle(area);
  const padX = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
  const padY = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
  return {
    width: Math.max(0, area.clientWidth - padX),
    height: Math.max(0, area.clientHeight - padY),
    dpr: window.devicePixelRatio || 1,
  };
}

function getHalftoneFitSize(st, metrics) {
  const sourceW = Number(st?.origWidth) || Number(st?.liveWidth) || 0;
  const sourceH = Number(st?.origHeight) || Number(st?.liveHeight) || 0;
  if (!sourceW || !sourceH || !metrics?.width || !metrics?.height) {
    return { width: 0, height: 0, sourceW, sourceH };
  }
  const aspect = sourceW / sourceH;
  let width = metrics.width;
  let height = width / aspect;
  if (height > metrics.height) {
    height = metrics.height;
    width = height * aspect;
  }
  return { width: Math.max(1, width), height: Math.max(1, height), sourceW, sourceH };
}

function layoutHalftonePreview(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const wrap = document.getElementById("ht-preview-wrap");
  if (!wrap) return;
  const metrics = getHalftoneViewportMetrics();
  if (!metrics.width || !metrics.height) return;
  const fit = getHalftoneFitSize(st, metrics);
  if (!fit.width || !fit.height) return;

  wrap.style.width = `${fit.width.toFixed(2)}px`;
  wrap.style.height = `${fit.height.toFixed(2)}px`;
  st._fitWidth = fit.width;
  st._fitHeight = fit.height;
  st._fitMetrics = { ...metrics };

  updateHalftoneImageRendering(app);
  updateHalftoneZoomLabel(app);
  updateHalftoneRoiPosition(app);

  if (st.qualityMode === "live" && st.liveReady) {
    const changed = resizeHalftoneGpuCanvas(app);
    if (changed) renderHalftoneWebGL(app);
  }
}

function updateHalftoneImageRendering(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const img = document.getElementById("ht-preview-img");
  if (!st || !img) return;
  const sourceW = Number(st.origWidth) || Number(st.liveWidth) || 0;
  const fitW = Number(st._fitWidth) || document.getElementById("ht-preview-wrap")?.offsetWidth || 0;
  const vp = App.getViewportState?.("halftone");
  const zoom = Number(vp?.zoom) || 1;
  const actualScale = sourceW > 0 && fitW > 0 ? (fitW / sourceW) * zoom : 1;
  st._actualPreviewScale = actualScale;
  img.classList.toggle("ht-pixelated", actualScale >= 1.15);
}

function updateHalftoneZoomLabel(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const label = document.getElementById("ht-zoom-label");
  if (!st || !label) return;
  const sourceW = Number(st.origWidth) || Number(st.liveWidth) || 0;
  const fitW = Number(st._fitWidth) || document.getElementById("ht-preview-wrap")?.offsetWidth || 0;
  const vp = App.getViewportState?.("halftone");
  const zoom = Number(vp?.zoom) || 1;
  const actualScale = sourceW > 0 && fitW > 0 ? (fitW / sourceW) * zoom : zoom;
  const pct = Math.max(1, Math.round(actualScale * 100));
  label.textContent = `${pct}%`;
  label.title = `${pct}% — реальный масштаб к источнику (клик — вписать в экран)`;
}

function getHalftoneCpuPreviewMaxSide(st) {
  if (!st) return 0;
  if (st.previewQuality === "maximum") return 0;
  if (st.previewQuality === "performance") return HT_PERFORMANCE_MAX_SIDE;
  const metrics = getHalftoneViewportMetrics();
  const fit = getHalftoneFitSize(st, metrics);
  if (!fit.width) return 1024;
  return Math.max(256, Math.round(Math.max(fit.width, fit.height) * metrics.dpr));
}

function getHalftoneCpuRoiMaxSide(st) {
  if (!st) return 0;
  if (st.previewQuality === "maximum") return 0;
  if (st.previewQuality === "performance") return HT_PERFORMANCE_MAX_SIDE;
  const metrics = getHalftoneViewportMetrics();
  return Math.max(256, Math.round(Math.max(metrics.width, metrics.height) * metrics.dpr * 1.35));
}

function getHalftoneGpuLimits(gl) {
  if (!gl) return { maxSide: 2048, maxTexture: 2048, maxRender: 2048 };
  const maxTexture = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) || 2048;
  const maxRender = Number(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)) || maxTexture;
  const dims = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
  const viewportMax = dims ? Math.max(Number(dims[0]) || 0, Number(dims[1]) || 0) : 0;
  const maxSide = Math.max(64, Math.min(maxTexture, maxRender, viewportMax || maxRender));
  return { maxSide, maxTexture, maxRender };
}

function resizeHalftoneGpuCanvas(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !glContext || !st.liveReady) return false;
  const canvas = document.getElementById("ht-live-canvas");
  if (!canvas) return false;

  const sourceW = Number(st.origWidth) || Number(st.liveWidth) || 0;
  const sourceH = Number(st.origHeight) || Number(st.liveHeight) || 0;
  if (!sourceW || !sourceH) return false;

  const metrics = getHalftoneViewportMetrics();
  const fit = getHalftoneFitSize(st, metrics);
  if (!fit.width) return false;
  const vp = App.getViewportState?.("halftone");
  const zoom = Math.max(0.01, Number(vp?.zoom) || 1);
  const dpr = metrics.dpr || 1;
  const sourceMax = Math.max(sourceW, sourceH);
  const limits = getHalftoneGpuLimits(glContext);

  let desiredMax = sourceMax;
  if (st.previewQuality === "performance") {
    // "Up to 1920" is a cap, not an upscale: never render more pixels than
    // the source can meaningfully provide.
    desiredMax = Math.min(sourceMax, HT_PERFORMANCE_MAX_SIDE);
  } else if (st.previewQuality !== "maximum") {
    const displayMax = Math.max(fit.width, fit.height) * zoom * dpr;
    const softSourceCap = Math.max(sourceMax, HT_GPU_AUTO_MIN_CAP);
    desiredMax = Math.min(displayMax, softSourceCap);
  }

  let targetMax = Math.min(Math.max(64, desiredMax), limits.maxSide);
  if (st.previewQuality !== "maximum") {
    targetMax = Math.round(targetMax / 64) * 64;
  }
  targetMax = Math.max(64, Math.min(targetMax, limits.maxSide));

  let width;
  let height;
  if (sourceW >= sourceH) {
    width = targetMax;
    height = Math.max(1, Math.round(targetMax * sourceH / sourceW));
  } else {
    height = targetMax;
    width = Math.max(1, Math.round(targetMax * sourceW / sourceH));
  }
  width = Math.max(1, Math.min(width, limits.maxSide));
  height = Math.max(1, Math.min(height, limits.maxSide));

  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  st._gpuCanvasMaxSide = targetMax;
  glContext.viewport(0, 0, width, height);
  return true;
}

function getHalftoneRoiPlan(st) {
  if (!st) return null;
  // Maximum already renders the whole source at native resolution; a second
  // ROI request would duplicate the same pixels.
  if (st.previewQuality === "maximum") return null;
  const sourceW = Number(st.origWidth) || Number(st.liveWidth) || 0;
  const sourceH = Number(st.origHeight) || Number(st.liveHeight) || 0;
  if (!sourceW || !sourceH) return null;

  const wrap = document.getElementById("ht-preview-wrap");
  const fitW = Number(st._fitWidth) || wrap?.offsetWidth || 0;
  const fitH = Number(st._fitHeight) || wrap?.offsetHeight || 0;
  if (!fitW || !fitH) return null;

  const vp = App.getViewportState?.("halftone");
  const zoom = Number(vp?.zoom) || 1;
  if (zoom < HT_ROI_MIN_ZOOM) return null;

  const metrics = getHalftoneViewportMetrics();
  if (!metrics.width || !metrics.height) return null;
  const panX = Number(vp?.panX) || 0;
  const panY = Number(vp?.panY) || 0;

  // Convert the visible viewport rectangle into un-transformed wrapper pixels.
  let left = (-metrics.width * 0.5 - panX) / zoom;
  let right = (metrics.width * 0.5 - panX) / zoom;
  let top = (-metrics.height * 0.5 - panY) / zoom;
  let bottom = (metrics.height * 0.5 - panY) / zoom;
  left = Math.max(-fitW * 0.5, left);
  right = Math.min(fitW * 0.5, right);
  top = Math.max(-fitH * 0.5, top);
  bottom = Math.min(fitH * 0.5, bottom);
  if (right - left < 2 || bottom - top < 2) return null;

  let x0 = (left + fitW * 0.5) / fitW * sourceW;
  let x1 = (right + fitW * 0.5) / fitW * sourceW;
  let y0 = (top + fitH * 0.5) / fitH * sourceH;
  let y1 = (bottom + fitH * 0.5) / fitH * sourceH;

  const marginX = (x1 - x0) * HT_ROI_MARGIN;
  const marginY = (y1 - y0) * HT_ROI_MARGIN;
  x0 -= marginX;
  x1 += marginX;
  y0 -= marginY;
  y1 += marginY;

  const x = Math.max(0, Math.floor(x0));
  const y = Math.max(0, Math.floor(y0));
  const width = Math.max(1, Math.min(sourceW - x, Math.ceil(x1) - x));
  const height = Math.max(1, Math.min(sourceH - y, Math.ceil(y1) - y));

  if (width >= sourceW * 0.97 && height >= sourceH * 0.97) return null;
  return { x, y, width, height };
}

function updateHalftoneRoiPosition(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const img = document.getElementById("ht-roi-img");
  if (!st || !img) return;

  if (st.comparing || st.qualityMode !== "full" || !st.roiDataUrl || !st.roiRect) {
    img.style.display = "none";
    return;
  }
  const vp = App.getViewportState?.("halftone");
  if ((Number(vp?.zoom) || 1) < HT_ROI_MIN_ZOOM) {
    img.style.display = "none";
    return;
  }
  const sourceW = Number(st.origWidth) || Number(st.liveWidth) || 0;
  const sourceH = Number(st.origHeight) || Number(st.liveHeight) || 0;
  const wrap = document.getElementById("ht-preview-wrap");
  const fitW = Number(st._fitWidth) || wrap?.offsetWidth || 0;
  const fitH = Number(st._fitHeight) || wrap?.offsetHeight || 0;
  if (!sourceW || !sourceH || !fitW || !fitH) {
    img.style.display = "none";
    return;
  }

  const rect = st.roiRect;
  img.style.left = `${(rect.x / sourceW) * fitW}px`;
  img.style.top = `${(rect.y / sourceH) * fitH}px`;
  img.style.right = "auto";
  img.style.bottom = "auto";
  img.style.width = `${(rect.width / sourceW) * fitW}px`;
  img.style.height = `${(rect.height / sourceH) * fitH}px`;
  img.style.display = "block";
}

function hideHalftoneRoi(appRef, abortRequest = false) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const img = document.getElementById("ht-roi-img");
  if (img) img.style.display = "none";
  if (abortRequest && st?._roiAbort) {
    try { st._roiAbort.abort(); } catch (_) {}
    st._roiAbort = null;
  }
}

function scheduleHalftoneRoiUpdate(appRef, immediate = false) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  if (st.qualityMode !== "full" || st.comparing || (!st.sourceUrl && !st.sourceDataUrl)) {
    hideHalftoneRoi(app, true);
    return;
  }
  const plan = getHalftoneRoiPlan(st);
  if (!plan) {
    hideHalftoneRoi(app, true);
    return;
  }

  st._pendingRoiPlan = plan;
  const signature = `${plan.x},${plan.y},${plan.width},${plan.height}|${st._editRevision || 0}|${st.previewQuality}`;
  if (st.roiDataUrl && st.roiRect && st._roiSignature === signature) {
    updateHalftoneRoiPosition(app);
    return;
  }

  if (st._roiTimer) clearTimeout(st._roiTimer);
  st._roiTimer = setTimeout(() => {
    st._roiTimer = null;
    updateHalftoneRoi(app);
  }, immediate ? 0 : 110);
}

async function updateHalftoneRoi(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || st.qualityMode !== "full" || st.comparing) return;
  const plan = st._pendingRoiPlan;
  if (!plan) return;

  const signature = `${plan.x},${plan.y},${plan.width},${plan.height}|${st._editRevision || 0}|${st.previewQuality}`;
  if (st.roiDataUrl && st.roiRect && st._roiSignature === signature) {
    updateHalftoneRoiPosition(app);
    return;
  }

  if (st._roiAbort) {
    try { st._roiAbort.abort(); } catch (_) {}
  }
  const controller = new AbortController();
  st._roiAbort = controller;
  const seq = ++st._roiSeq;
  const roi = { x: plan.x, y: plan.y, width: plan.width, height: plan.height };

  try {
    const res = await halftoneFetchJson("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: buildHalftoneParamsPayload(st, true),
      layers: buildHalftoneLayersPayload(st, true),
      save: false,
      preview_max_side: getHalftoneCpuRoiMaxSide(st),
      roi,
    }, controller.signal);
    if (seq !== st._roiSeq || st.qualityMode !== "full" || st.comparing) return;
    if (!res?.preview_data_url) return;

    const rect = res.roi
      ? { x: res.roi.x, y: res.roi.y, width: res.roi.width, height: res.roi.height }
      : roi;
    st.roiDataUrl = res.preview_data_url;
    st.roiRect = rect;
    st._roiSignature = `${rect.x},${rect.y},${rect.width},${rect.height}|${st._editRevision || 0}|${st.previewQuality}`;

    const img = document.getElementById("ht-roi-img");
    if (!img) return;
    img.onload = () => {
      if (seq === st._roiSeq) updateHalftoneRoiPosition(app);
    };
    img.src = res.preview_data_url;
    if (img.complete) updateHalftoneRoiPosition(app);
  } catch (e) {
    if (e?.name !== "AbortError") console.warn("[halftone] ROI preview failed:", e);
  } finally {
    if (st._roiAbort === controller) st._roiAbort = null;
  }
}

async function halftoneFetchJson(url, body, signal) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || res.statusText || `HTTP ${res.status}`);
  }
  return res.json();
}

function ensureWebGLCanvas(appRef) {
  const app = appRef || window.App;
  let canvas = document.getElementById("ht-live-canvas");
  if (!canvas) {
    const wrap = document.getElementById("ht-preview-wrap");
    if (!wrap) return null;
    canvas = document.createElement("canvas");
    canvas.id = "ht-live-canvas";
    canvas.className = "ht-live-canvas";
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    canvas.style.display = "none";
    canvas.style.zIndex = "2";
    wrap.appendChild(canvas);
  }

  if (!glContext) {
    try {
      glContext = canvas.getContext("webgl2", { alpha: false, depth: false, antialias: false, preserveDrawingBuffer: true });
      if (glContext) {
        initWebGLProgram(glContext);
      }
    } catch (e) {
      console.warn("WebGL2 initialization failed, falling back to 2D:", e);
    }
  }
  return canvas;
}

function initWebGLProgram(gl) {
  const compileShader = (src, type) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  };

  const vs = compileShader(VS_SOURCE, gl.VERTEX_SHADER);
  const fs = compileShader(FS_SOURCE, gl.FRAGMENT_SHADER);
  if (!vs || !fs) return;

  glProgram = gl.createProgram();
  gl.attachShader(glProgram, vs);
  gl.attachShader(glProgram, fs);
  gl.linkProgram(glProgram);

  if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(glProgram));
    return;
  }

  // Setup full-screen quad [-1, 1]
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
     1,  1,
  ]), gl.STATIC_DRAW);

  const posLoc = gl.getAttribLocation(glProgram, "a_position");
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // Cache uniform locations
  const uniforms = [
    "u_image", "u_resolution", "u_cell_size", "u_dot_shape", "u_angle",
    "u_max_radius_ratio", "u_contrast", "u_gamma", "u_brightness",
    "u_black_threshold", "u_white_threshold", "u_invert", "u_intensity",
    "u_dot_color", "u_bg_color", "u_preserve_lines", "u_line_threshold",
    "u_aspect", "u_row_offset", "u_jitter_pos", "u_jitter_size",
    "u_satEnabled", "u_satCount", "u_satSizeRatio", "u_satDist",
    "u_satAngleBase", "u_satStretch", "u_satShape", "u_mergeFactor",
    "u_gradType", "u_gradAngle", "u_gradStrength",
    "u_opacity", "u_blend_mode", "u_antialias", "u_mode", "u_mask", "u_has_mask", "u_invert_mask",
    "u_base", "u_flip_v", "u_pass_mode"
  ];
  glUniformLocations = {};
  for (const u of uniforms) {
    glUniformLocations[u] = gl.getUniformLocation(glProgram, u);
  }
}

function setHalftoneQualityMode(mode, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app.state.halftone;
  if (!st) return;

  const nextMode = mode === "live" ? "live" : "full";
  const changed = st.qualityMode !== nextMode;
  st.qualityMode = nextMode;
  st.comparing = false;
  const compareBtn = document.getElementById("ht-compare-btn");
  if (compareBtn) compareBtn.classList.remove("active");
  const compareBadge = document.getElementById("ht-compare-badge");
  if (compareBadge) compareBadge.style.display = "none";

  const btnFull = document.getElementById("ht-quality-full");
  const btnLive = document.getElementById("ht-quality-live");
  if (btnFull) btnFull.classList.toggle("active", st.qualityMode === "full");
  if (btnLive) btnLive.classList.toggle("active", st.qualityMode === "live");

  const canvas = document.getElementById("ht-live-canvas");
  const imgEl = document.getElementById("ht-preview-img");

  if (st.qualityMode === "full") {
    if (canvas) canvas.style.display = "none";
    if (imgEl) imgEl.style.visibility = "visible";
    hideHalftoneRoi(app, true);
    if (st.sourcePath || st.sourceDataUrl) runHalftonePreview(true, app);
  } else {
    if (imgEl) imgEl.style.visibility = "hidden";
    if (canvas) canvas.style.display = "block";
    hideHalftoneRoi(app, true);
    const srcUrl = st.sourceUrl || st.sourceDataUrl;
    if (srcUrl) {
      prepareHalftoneLiveSource(srcUrl, app);
    } else if (st.liveReady) {
      layoutHalftonePreview(app);
      resizeHalftoneGpuCanvas(app);
      renderHalftoneWebGL(app);
    }
  }

  if (changed && app.toast) {
    app.toast("info", st.qualityMode === "live" ? "⚡ Включён движок WebGL2 GPU (Живой)" : "🎨 Включён движок Точный CPU (Canvas 2D)");
  }
}

function prepareHalftoneLiveSource(srcUrl, appRef) {
  if (!srcUrl) return;
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;

  if (st.liveReady && st._liveSourcePreparedUrl === srcUrl) {
    layoutHalftonePreview(app);
    const resized = resizeHalftoneGpuCanvas(app);
    if (st.qualityMode === "live" && (resized || st.liveReady)) renderHalftoneWebGL(app);
    return;
  }

  const token = (st._liveSourceToken || 0) + 1;
  st._liveSourceToken = token;

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    if (token !== st._liveSourceToken) return;
    st.liveWidth = img.naturalWidth || img.width;
    st.liveHeight = img.naturalHeight || img.height;
    if (!st.origWidth || !st.origHeight) {
      st.origWidth = st.liveWidth;
      st.origHeight = st.liveHeight;
    }
    st.liveReady = true;
    st._liveSourcePreparedUrl = srcUrl;

    const canvas = ensureWebGLCanvas(app);
    if (!canvas || !glContext || !glProgram) {
      renderHalftoneWebGL(app);
      return;
    }

    const limits = getHalftoneGpuLimits(glContext);
    const maxTexture = limits.maxTexture;
    let uploadSource = img;
    if (Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height) > maxTexture) {
      const sc = maxTexture / Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height);
      const down = document.createElement("canvas");
      down.width = Math.max(1, Math.round((img.naturalWidth || img.width) * sc));
      down.height = Math.max(1, Math.round((img.naturalHeight || img.height) * sc));
      down.getContext("2d").drawImage(img, 0, 0, down.width, down.height);
      uploadSource = down;
    }

    // The source stays at native size (up to the hardware texture limit). The
    // drawable canvas is sized separately by resizeHalftoneGpuCanvas().
    if (!glSourceTexture) {
      glSourceTexture = glContext.createTexture();
    }
    glContext.activeTexture(glContext.TEXTURE0);
    glContext.bindTexture(glContext.TEXTURE_2D, glSourceTexture);
    glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_S, glContext.CLAMP_TO_EDGE);
    glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_T, glContext.CLAMP_TO_EDGE);
    glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MIN_FILTER, glContext.LINEAR_MIPMAP_LINEAR);
    glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MAG_FILTER, glContext.LINEAR);
    glContext.pixelStorei(glContext.UNPACK_FLIP_Y_WEBGL, false);
    glContext.texImage2D(glContext.TEXTURE_2D, 0, glContext.RGBA, glContext.RGBA, glContext.UNSIGNED_BYTE, uploadSource);
    glContext.generateMipmap(glContext.TEXTURE_2D);
    glContext.bindTexture(glContext.TEXTURE_2D, null);
    syncHalftoneMaskCanvasSize(st.liveWidth, st.liveHeight, app);

    layoutHalftonePreview(app);
    const resized = resizeHalftoneGpuCanvas(app);
    if (st.qualityMode === "live" && (resized || st.liveReady)) {
      renderHalftoneWebGL(app);
    }
  };
  img.src = srcUrl;
}

function triggerHalftoneLiveUpdate(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) return;

  // Every edit invalidates both the adaptive base and the high-detail ROI.
  st._editRevision = (Number(st._editRevision) || 0) + 1;
  st._previewSeq = (Number(st._previewSeq) || 0) + 1;
  if (st._previewAbort) {
    try { st._previewAbort.abort(); } catch (_) {}
    st._previewAbort = null;
  }
  st._roiSignature = null;
  hideHalftoneRoi(app, true);

  const canvas = document.getElementById("ht-live-canvas");
  const imgEl = document.getElementById("ht-preview-img");

  if (st.qualityMode === "full") {
    if (canvas) canvas.style.display = "none";
    if (imgEl) imgEl.style.visibility = "visible";
    if (st.debounceTimer) clearTimeout(st.debounceTimer);
    st.debounceTimer = setTimeout(() => {
      st.debounceTimer = null;
      runHalftonePreview(false, app);
    }, 120);
    return;
  }

  // Live GPU mode
  if (canvas) canvas.style.display = "block";
  if (imgEl) imgEl.style.visibility = "hidden";

  if (st.liveReady && glContext) {
    if (!st._rafPending) {
      st._rafPending = true;
      requestAnimationFrame(() => {
        st._rafPending = false;
        renderHalftoneWebGL(app);
      });
    }
  } else {
    if (st.debounceTimer) clearTimeout(st.debounceTimer);
    st.debounceTimer = setTimeout(() => {
      st.debounceTimer = null;
      runHalftonePreview(false, app);
    }, 120);
  }
}

function ensureHalftoneFramebuffers(gl, w, h) {
  const width = Math.max(1, Math.round(Number(w) || 1));
  const height = Math.max(1, Math.round(Number(h) || 1));
  if (
    glFboSize.w === width &&
    glFboSize.h === height &&
    glFbos[0] && glFbos[1] &&
    glFboTextures[0] && glFboTextures[1]
  ) {
    return true;
  }

  for (let i = 0; i < 2; i++) {
    if (glFbos[i]) {
      gl.deleteFramebuffer(glFbos[i]);
      glFbos[i] = null;
    }
    if (glFboTextures[i]) {
      gl.deleteTexture(glFboTextures[i]);
      glFboTextures[i] = null;
    }
  }

  for (let i = 0; i < 2; i++) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn("Halftone Studio: incomplete WebGL2 framebuffer");
    }
    glFboTextures[i] = tex;
    glFbos[i] = fbo;
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  glFboSize = { w: width, h: height };
  return true;
}

function bindHalftoneLayerMask(gl, lyr) {
  const maskUnit = 1;
  gl.activeTexture(gl.TEXTURE0 + maskUnit);
  gl.uniform1i(glUniformLocations.u_mask, maskUnit);

  if (!lyr || !lyr.hasMask || !lyr.maskCanvas) {
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.uniform1i(glUniformLocations.u_has_mask, 0);
    return;
  }

  if (!lyr._glMaskTexture) {
    lyr._glMaskTexture = gl.createTexture();
  }
  gl.bindTexture(gl.TEXTURE_2D, lyr._glMaskTexture);
  const maskRevision = Number(lyr._maskRevision) || 0;
  if (lyr._glMaskRevision !== maskRevision || lyr._glMaskCanvas !== lyr.maskCanvas) {
    // Painting a 4K mask should not upload 64 MB to the GPU on every frame.
    // Downscale a cached preview copy once per revision; export still uses the
    // full-resolution mask on the CPU backend.
    let upload = lyr.maskCanvas;
    const srcW = lyr.maskCanvas.width;
    const srcH = lyr.maskCanvas.height;
    const maxSide = Math.max(srcW, srcH);
    if (maxSide > HT_GPU_MASK_MAX_SIDE) {
      const sc = HT_GPU_MASK_MAX_SIDE / maxSide;
      const w = Math.max(1, Math.round(srcW * sc));
      const h = Math.max(1, Math.round(srcH * sc));
      let scratch = lyr._glMaskUploadCanvas;
      if (!scratch) {
        scratch = document.createElement("canvas");
        lyr._glMaskUploadCanvas = scratch;
      }
      if (scratch.width !== w || scratch.height !== h) {
        scratch.width = w;
        scratch.height = h;
      }
      const ctx = scratch.getContext("2d");
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(lyr.maskCanvas, 0, 0, w, h);
      upload = scratch;
    }
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, upload);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    lyr._glMaskCanvas = lyr.maskCanvas;
    lyr._glMaskRevision = maskRevision;
    lyr.maskDirty = false;
  }
  gl.uniform1i(glUniformLocations.u_has_mask, 1);
}

function applyHalftoneLayerParams(gl, p, st, width, height) {
  const params = p || {};
  gl.uniform2f(glUniformLocations.u_resolution, width, height);

  // Cell size scaled to natural dimensions. No lower clamp on the scale:
  // the adaptive canvas can be smaller than the source, and the halftone grid
  // must stay in source pixels so zooming never changes dot size.
  const scaleFactor = st.origWidth ? (width / Math.max(st.origWidth, 1)) : 1.0;
  const cell = Math.max(1.0, (Number(params.cell_size) || 8) * scaleFactor);
  gl.uniform1f(glUniformLocations.u_cell_size, cell);

  // 13 Geometry shapes map
  const shapeMap = {
    circle: 0,
    square: 1,
    diamond: 2,
    line: 3,
    crosshatch: 4,
    hexagon: 5,
    cross: 6,
    ellipse: 7,
    triangle: 8,
    star: 9,
    heart: 10,
    concentric: 11,
    ring: 12,
    stipple: 0,
    stochastic: 0,
  };
  gl.uniform1i(glUniformLocations.u_dot_shape, shapeMap[params.dot_shape] ?? 0);

  const angleRad = ((Number(params.angle) || 45) * Math.PI) / 180.0;
  gl.uniform1f(glUniformLocations.u_angle, angleRad);

  gl.uniform1f(glUniformLocations.u_max_radius_ratio, Number(params.max_radius_ratio) || 0.9);
  gl.uniform1f(glUniformLocations.u_contrast, Number(params.contrast ?? 1.25));
  gl.uniform1f(glUniformLocations.u_gamma, Number(params.gamma ?? 1.05));
  gl.uniform1f(glUniformLocations.u_brightness, Number(params.brightness ?? 0.0));
  gl.uniform1f(glUniformLocations.u_black_threshold, Number(params.black_threshold ?? 0.18));
  gl.uniform1f(glUniformLocations.u_white_threshold, Number(params.white_threshold ?? 0.86));
  gl.uniform1i(glUniformLocations.u_invert, params.invert ? 1 : 0);
  gl.uniform1f(glUniformLocations.u_intensity, Number(params.intensity ?? 1.0));

  const parseRgb = (hex, def) => {
    const s = String(hex || "").replace("#", "");
    if (s.length === 6) {
      return [parseInt(s.slice(0, 2), 16) / 255, parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255];
    }
    return def;
  };
  const dotRgb = parseRgb(params.dot_color, [0.07, 0.07, 0.07]);
  const bgRgb = parseRgb(params.bg_color, [1.0, 1.0, 1.0]);
  gl.uniform3f(glUniformLocations.u_dot_color, dotRgb[0], dotRgb[1], dotRgb[2]);
  gl.uniform3f(glUniformLocations.u_bg_color, bgRgb[0], bgRgb[1], bgRgb[2]);

  gl.uniform1i(glUniformLocations.u_preserve_lines, params.preserve_lines ? 1 : 0);
  gl.uniform1f(glUniformLocations.u_line_threshold, Number(params.line_threshold ?? 0.22));
  gl.uniform2f(glUniformLocations.u_aspect, Number(params.aspect_x ?? 1.0), Number(params.aspect_y ?? 1.0));
  gl.uniform1f(glUniformLocations.u_row_offset, Number(params.row_offset ?? 0.0));
  gl.uniform1f(glUniformLocations.u_jitter_pos, Number(params.jitter_pos ?? 0.0));
  gl.uniform1f(glUniformLocations.u_jitter_size, Number(params.jitter_size ?? 0.0));

  // GenTonik Planets & Satellites uniforms
  const satEnabled = Boolean(params.satellite_enabled);
  gl.uniform1i(glUniformLocations.u_satEnabled, satEnabled ? 1 : 0);
  gl.uniform1i(glUniformLocations.u_satCount, Number(params.satellite_count ?? 4));
  gl.uniform1f(glUniformLocations.u_satSizeRatio, Number(params.satellite_size_ratio ?? 0.45));
  gl.uniform1f(glUniformLocations.u_satDist, Number(params.satellite_distance ?? 0.42));
  const satAngleRad = ((Number(params.satellite_angle) || 0) * Math.PI) / 180.0;
  gl.uniform1f(glUniformLocations.u_satAngleBase, satAngleRad);
  gl.uniform1f(glUniformLocations.u_satStretch, Number(params.satellite_stretch ?? 0.0));
  gl.uniform1i(glUniformLocations.u_satShape, shapeMap[params.satellite_shape] ?? 0);
  gl.uniform1f(glUniformLocations.u_mergeFactor, Number(params.merge_factor ?? 0.0));

  // Spatial Gradients uniforms
  const gradMap = { none: 0, linear: 1, radial: 2, vignette: 3 };
  gl.uniform1i(glUniformLocations.u_gradType, gradMap[params.grad_type] ?? 0);
  const gradAngleRad = ((Number(params.grad_angle) || 45) * Math.PI) / 180.0;
  gl.uniform1f(glUniformLocations.u_gradAngle, gradAngleRad);
  gl.uniform1f(glUniformLocations.u_gradStrength, Number(params.grad_strength ?? 0.0));

  // AA4 is export-only; live GPU preview is capped at AA2.
  const liveAa = Math.max(1, Math.min(2, Number(params.antialias) || 2));
  gl.uniform1i(glUniformLocations.u_antialias, liveAa);
  const modeVal = params.mode === "manga" ? 1 : (params.mode === "color" ? 2 : 0);
  gl.uniform1i(glUniformLocations.u_mode, modeVal);
}

function renderHalftoneWebGL(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !st.liveReady || !glContext || !glProgram || !glSourceTexture) return;

  const gl = glContext;
  const canvas = document.getElementById("ht-live-canvas");
  if (!canvas || !canvas.width || !canvas.height) return;

  const width = canvas.width;
  const height = canvas.height;
  if (!ensureHalftoneFramebuffers(gl, width, height) || !glFbos[0] || !glFbos[1]) return;

  gl.useProgram(glProgram);
  gl.viewport(0, 0, width, height);

  // The original source texture is the same base the CPU backend composites over.
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, glSourceTexture);
  gl.uniform1i(glUniformLocations.u_image, 0);

  const rasterLayers = (Array.isArray(st.layers) ? st.layers : [])
    .filter((l) => l && !l.isOriginal && l.visible !== false);

  const blendModes = { normal: 0, multiply: 1, screen: 2, overlay: 3, darken: 4, lighten: 5 };
  let baseTexture = glSourceTexture;
  let writeIndex = 0;

  for (const lyr of rasterLayers) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, glFbos[writeIndex]);
    gl.viewport(0, 0, width, height);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, baseTexture);
    gl.uniform1i(glUniformLocations.u_base, 2);

    const p = lyr.params || st.params || {};
    applyHalftoneLayerParams(gl, p, st, width, height);
    gl.uniform1i(glUniformLocations.u_pass_mode, 0);
    gl.uniform1i(glUniformLocations.u_flip_v, 0);
    gl.uniform1f(glUniformLocations.u_opacity, typeof lyr.opacity === "number" ? Math.max(0, Math.min(1, lyr.opacity)) : 1.0);
    gl.uniform1i(glUniformLocations.u_blend_mode, blendModes[lyr.blend_mode] ?? 0);
    gl.uniform1i(glUniformLocations.u_invert_mask, p.invert_mask ? 1 : 0);
    bindHalftoneLayerMask(gl, lyr);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    baseTexture = glFboTextures[writeIndex];
    writeIndex = 1 - writeIndex;
  }

  // Final blit to the default framebuffer. Source/FBO textures are sampled
  // bottom-row-first, so flip V exactly once here to display the image upright.
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, baseTexture);
  gl.uniform1i(glUniformLocations.u_base, 2);
  gl.uniform1i(glUniformLocations.u_pass_mode, 1);
  gl.uniform1i(glUniformLocations.u_flip_v, 1);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.activeTexture(gl.TEXTURE0);
}


async function runHalftonePreview(immediate = false, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app.state.halftone;
  if (!st.sourcePath && !st.sourceDataUrl) {
    if (immediate) toast("warning", "Сначала выберите или загрузите изображение");
    return;
  }
  // The callback-based CPU preview belongs to the CPU engine. GPU mode uses
  // the shader pipeline and must not overwrite its canvas with a CPU result.
  if (st.qualityMode !== "full") return;

  try {
    await ensureHalftoneSourceDimensions(st);
  } catch (_) {}

  layoutHalftonePreview(app);

  const metaEl = document.getElementById("ht-meta-info");
  const loadOverlay = document.getElementById("ht-loading-overlay");
  const showOverlay = immediate || !st.resultDataUrl;
  // Background refreshes keep the previous status text: toggling it can reflow
  // the toolbar and feed the ResizeObserver layout loop.
  if (metaEl && showOverlay) metaEl.textContent = "⏳ Обработка...";
  if (loadOverlay && showOverlay) loadOverlay.style.display = "flex";

  hideHalftoneRoi(app, true);

  if (st._previewAbort) {
    try { st._previewAbort.abort(); } catch (_) {}
  }
  const controller = new AbortController();
  st._previewAbort = controller;
  const seq = ++st._previewSeq;

  try {
    const res = await halftoneFetchJson("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: buildHalftoneParamsPayload(st, true),
      layers: buildHalftoneLayersPayload(st, true),
      save: false,
      preview_max_side: getHalftoneCpuPreviewMaxSide(st),
    }, controller.signal);
    if (seq !== st._previewSeq || st.qualityMode !== "full") return;

    st.resultDataUrl = res.preview_data_url;
    st.origWidth = res.orig_width || res.width || st.origWidth;
    st.origHeight = res.orig_height || res.height || st.origHeight;
    st._sourceSizePromise = null;
    syncHalftoneMaskCanvasSize(st.origWidth, st.origHeight, app);

    const emptyEl = document.getElementById("ht-empty-state");
    const wrapEl = document.getElementById("ht-preview-wrap");
    const imgEl = document.getElementById("ht-preview-img");
    const canvas = document.getElementById("ht-live-canvas");

    if (emptyEl) emptyEl.style.display = "none";
    if (wrapEl) wrapEl.style.display = "inline-flex";

    layoutHalftonePreview(app);

    if (imgEl) imgEl.src = res.preview_data_url;
    if (canvas) canvas.style.display = "none";
    if (imgEl) imgEl.style.visibility = "visible";

    if (res.elapsed_sec != null && metaEl) {
      metaEl.textContent = `CPU ${Math.round(Number(res.elapsed_sec) * 1000)} ms · ${res.width}×${res.height} px`;
    }

    scheduleHalftoneRoiUpdate(app, false);
  } catch (e) {
    if (e?.name === "AbortError") return;
    toast("error", "Halftone error: " + e.message);
  } finally {
    const isActivePreview = st._previewAbort === controller;
    if (isActivePreview) st._previewAbort = null;
    // Hide the overlay when this preview finishes even if it was a silent
    // refresh, otherwise an aborted "immediate" render can leave it stuck.
    if (isActivePreview && loadOverlay) loadOverlay.style.display = "none";
  }
}

function selectHalftoneMode(mode, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const preset = halftonePresets[mode] || halftonePresets.custom;
  app.state.halftone.params = { ...app.state.halftone.params, ...preset, mode };
  const activeLyr = getActiveHalftoneLayer(app);
  if (activeLyr) activeLyr.params = app.state.halftone.params;
  syncHalftoneControlsFromParams(app.state.halftone.params);
  triggerHalftoneLiveUpdate(app);
}

function syncHalftoneControlsFromParams(p) {
  if (!p) return;
  document.querySelectorAll("#ht-mode-pills .studio-mode-pill").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === p.mode);
  });
  const setPair = (rId, nId, val) => {
    const r = document.getElementById(rId);
    const n = document.getElementById(nId);
    if (r && val !== undefined) r.value = val;
    if (n && val !== undefined) n.value = val;
  };
  setPair("ht-cell-size", "ht-cell-size-num", p.cell_size);
  setPair("ht-angle", "ht-angle-num", p.angle);
  setPair("ht-contrast", "ht-contrast-num", p.contrast);
  setPair("ht-gamma", "ht-gamma-num", p.gamma);
  setPair("ht-brightness", "ht-brightness-num", p.brightness);
  setPair("ht-max-radius", "ht-max-radius-num", p.max_radius_ratio);
  setPair("ht-black-point", "ht-black-point-num", p.black_threshold);
  setPair("ht-white-cut", "ht-white-cut-num", p.white_threshold);
  setPair("ht-intensity", "ht-intensity-num", p.intensity);

  const dotShapeSel = document.getElementById("ht-dot-shape");
  if (dotShapeSel && p.dot_shape) dotShapeSel.value = p.dot_shape;
  const invertChk = document.getElementById("ht-invert");
  if (invertChk) invertChk.checked = Boolean(p.invert);
  const linesChk = document.getElementById("ht-preserve-lines");
  if (linesChk) linesChk.checked = Boolean(p.preserve_lines);

  // Satellite & Planets controls sync
  const satBox = document.getElementById("ht-sat-controls");
  if (satBox) satBox.style.display = p.satellite_enabled ? "flex" : "none";
  const satChk = document.getElementById("ht-sat-enabled");
  if (satChk) satChk.checked = Boolean(p.satellite_enabled);
  const satCountSel = document.getElementById("ht-sat-count");
  if (satCountSel && p.satellite_count !== undefined) satCountSel.value = String(p.satellite_count);
  const satShapeSel = document.getElementById("ht-sat-shape");
  if (satShapeSel && p.satellite_shape) satShapeSel.value = p.satellite_shape;
  setPair("ht-sat-dist", "ht-sat-dist-num", p.satellite_distance);
  setPair("ht-sat-size", "ht-sat-size-num", p.satellite_size_ratio);
  setPair("ht-sat-angle", "ht-sat-angle-num", p.satellite_angle);
  setPair("ht-sat-stretch", "ht-sat-stretch-num", p.satellite_stretch);
  setPair("ht-merge-factor", "ht-merge-factor-num", p.merge_factor);

  // Grid Geometry & Gradient sync
  const aspX = document.getElementById("ht-aspect-x");
  if (aspX && p.aspect_x !== undefined) aspX.value = p.aspect_x;
  const aspY = document.getElementById("ht-aspect-y");
  if (aspY && p.aspect_y !== undefined) aspY.value = p.aspect_y;
  setPair("ht-row-offset", "ht-row-offset-num", p.row_offset);
  const jitPos = document.getElementById("ht-jitter-pos");
  if (jitPos && p.jitter_pos !== undefined) jitPos.value = p.jitter_pos;
  const jitSize = document.getElementById("ht-jitter-size");
  if (jitSize && p.jitter_size !== undefined) jitSize.value = p.jitter_size;
  const gradTypeSel = document.getElementById("ht-grad-type");
  if (gradTypeSel && p.grad_type) gradTypeSel.value = p.grad_type;
  setPair("ht-grad-angle", "ht-grad-angle-num", p.grad_angle);
  setPair("ht-grad-strength", "ht-grad-strength-num", p.grad_strength);

  // Antialias and Dither sync
  const aaSel = document.getElementById("ht-antialias");
  if (aaSel && p.antialias !== undefined) aaSel.value = String(p.antialias);
  const ditherSel = document.getElementById("ht-dither-type");
  if (ditherSel && p.dither_type) ditherSel.value = p.dither_type;
}

function onHalftoneParamInput(key, rawVal, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  let val = rawVal;
  if (["cell_size", "angle", "seed", "satellite_count", "antialias"].includes(key)) {
    val = parseInt(rawVal, 10);
    if (isNaN(val)) return;
  } else if (["contrast", "gamma", "brightness", "max_radius_ratio", "black_threshold", "white_threshold", "intensity", "satellite_distance", "satellite_size_ratio", "satellite_angle", "satellite_stretch", "aspect_x", "aspect_y", "row_offset", "jitter_pos", "jitter_size", "grad_angle", "grad_strength", "merge_factor"].includes(key)) {
    val = parseFloat(rawVal);
    if (isNaN(val)) return;
  }

  if (key === "satellite_enabled") {
    const satBox = document.getElementById("ht-sat-controls");
    if (satBox) satBox.style.display = val ? "flex" : "none";
  }

  app.state.halftone.params[key] = val;
  const activeLyr = getActiveHalftoneLayer(app);
  if (activeLyr) activeLyr.params[key] = val;

  triggerHalftoneLiveUpdate(app);
}

function clampHalftonePreviewAntialias(params) {
  const aa = parseInt(params?.antialias, 10);
  return { ...(params || {}), antialias: Math.max(1, Math.min(2, Number.isFinite(aa) ? aa : 2)) };
}

function buildHalftoneParamsPayload(st, forPreview = false) {
  const params = { ...(st?.params || {}) };
  return forPreview ? clampHalftonePreviewAntialias(params) : params;
}

function markHalftoneMaskDirty(lyr) {
  if (!lyr) return;
  lyr.maskDirty = true;
  lyr._maskRevision = (Number(lyr._maskRevision) || 0) + 1;
  lyr._maskDataUrl = null;
  lyr._maskDataUrlRevision = -1;
  lyr._maskDataUrlCanvas = null;
}

function getHalftoneMaskDataUrl(lyr) {
  if (!lyr || !lyr.hasMask || !lyr.maskCanvas) return null;
  const revision = Number(lyr._maskRevision) || 0;
  if (
    lyr._maskDataUrl &&
    lyr._maskDataUrlRevision === revision &&
    lyr._maskDataUrlCanvas === lyr.maskCanvas
  ) {
    return lyr._maskDataUrl;
  }
  try {
    lyr._maskDataUrl = lyr.maskCanvas.toDataURL("image/png");
    lyr._maskDataUrlRevision = revision;
    lyr._maskDataUrlCanvas = lyr.maskCanvas;
    return lyr._maskDataUrl;
  } catch (e) {
    console.warn("[halftone] mask serialization failed:", e);
    return null;
  }
}

function buildHalftoneLayersPayload(st, forPreview = false) {
  const layers = Array.isArray(st?.layers) ? st.layers : [];
  return layers
    .filter((l) => l && !l.isOriginal)
    .map((l) => {
      const rawParams = { ...(l.params || st.params || {}) };
      return {
        id: l.id,
        name: l.name,
        visible: l.visible !== false,
        opacity: typeof l.opacity === "number" ? Math.max(0, Math.min(1, l.opacity)) : 1.0,
        blend_mode: l.blend_mode || "normal",
        params: forPreview ? clampHalftonePreviewAntialias(rawParams) : rawParams,
        hasMask: Boolean(l.hasMask && l.maskCanvas),
        isBase: Boolean(l.isBase),
        mask: getHalftoneMaskDataUrl(l),
      };
    });
}

function getActiveHalftoneLayer(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !Array.isArray(st.layers)) return null;
  const active = st.layers.find((l) => l.id === st.activeLayerId && !l.isOriginal);
  if (active) return active;
  return st.layers.find((l) => !l.isOriginal) || null;
}

function normalizeHalftoneAngle(angle) {
  let a = Number(angle);
  if (!Number.isFinite(a)) a = 45;
  a = ((a % 180) + 180) % 180;
  return Math.round(a * 10) / 10;
}

function halftoneAngleDistance(a, b) {
  const d = Math.abs(normalizeHalftoneAngle(a) - normalizeHalftoneAngle(b));
  return Math.min(d, 180 - d);
}

function applyHalftoneMoireProtection(lyr, st, appRef) {
  if (!lyr || lyr.isOriginal || lyr.blend_mode !== "multiply" || !lyr.params) return false;
  const selfAngle = normalizeHalftoneAngle(lyr.params.angle ?? st.params?.angle ?? 45);
  const otherAngles = (st.layers || [])
    .filter((o) => o && o !== lyr && !o.isOriginal && o.visible !== false)
    .map((o) => normalizeHalftoneAngle(o.params?.angle ?? st.params?.angle ?? 45));

  let angle = selfAngle;
  for (let i = 0; i < 12; i++) {
    if (!otherAngles.some((a) => halftoneAngleDistance(a, angle) < 7.5)) break;
    angle = normalizeHalftoneAngle(angle + 15);
  }

  if (halftoneAngleDistance(angle, selfAngle) < 0.01) return false;

  lyr.params = { ...lyr.params, angle };
  if (lyr.id === st.activeLayerId) {
    st.params = { ...st.params, angle };
    syncHalftoneControlsFromParams(st.params);
  }
  const app = appRef || window.App;
  if (app?.toast) {
    app.toast("info", `🛡 Защита от муара: угол слоя «${lyr.name}» изменён на ${Math.round(angle)}°`);
  }
  return true;
}

function syncHalftoneLayerControls(appRef) {
  const app = appRef || window.App;
  const lyr = getActiveHalftoneLayer(app);
  const opacityR = document.getElementById("ht-layer-opacity");
  const opacityN = document.getElementById("ht-layer-opacity-num");
  const blendSel = document.getElementById("ht-layer-blend");
  const disabled = !lyr;

  [opacityR, opacityN, blendSel].forEach((el) => {
    if (el) el.disabled = disabled;
  });
  if (!lyr) return;

  const pct = Math.round((typeof lyr.opacity === "number" ? lyr.opacity : 1.0) * 100);
  if (opacityR) opacityR.value = pct;
  if (opacityN) opacityN.value = pct;
  if (blendSel) blendSel.value = lyr.blend_mode || "normal";
}

function renderHalftoneLayersUI(appRef) {
  const app = appRef || window.App;
  const listEl = document.getElementById("ht-layers-list");
  const st = app?.state?.halftone;
  if (!listEl || !st) return;

  const layers = Array.isArray(st.layers) ? st.layers : [];
  const blendLabels = {
    normal: "Normal",
    multiply: "Multiply",
    screen: "Screen",
    overlay: "Overlay",
    darken: "Darken",
    lighten: "Lighten",
  };

  listEl.innerHTML = layers.slice().reverse().map((layer) => {
    const isOriginal = Boolean(layer.isOriginal);
    const isAct = layer.id === st.activeLayerId && !isOriginal;
    const index = layers.indexOf(layer);
    const canUp = !isOriginal && index < layers.length - 1;
    const canDown = !isOriginal && index > 1;
    const badges = [];
    if (isOriginal) badges.push('<span class="ht-layer-tag ht-layer-tag-lock">🔒 оригинал</span>');
    if (layer.isBase) badges.push('<span class="ht-layer-tag ht-layer-tag-base">база</span>');
    if (layer.hasMask) badges.push('<span class="ht-layer-tag ht-layer-tag-mask">маска</span>');
    const meta = isOriginal
      ? "Фон / база сравнения"
      : `${blendLabels[layer.blend_mode] || "Normal"} · ${Math.round((typeof layer.opacity === "number" ? layer.opacity : 1.0) * 100)}%`;

    return `
      <div class="ht-layer-item ${isAct ? "active" : ""} ${isOriginal ? "locked" : ""}" onclick="App.selectHalftoneLayer('${layer.id}')" title="${escapeHtml(meta)}">
        <div class="ht-layer-left">
          ${isOriginal
            ? '<span class="ht-layer-lock">🔒</span>'
            : `<input type="checkbox" ${layer.visible !== false ? "checked" : ""} onclick="event.stopPropagation(); App.toggleHalftoneLayerVisible('${layer.id}')">`}
          <span class="ht-layer-name">${escapeHtml(layer.name)}</span>
          ${badges.join("")}
        </div>
        <div class="ht-layer-actions">
          ${isOriginal ? "" : `
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.moveHalftoneLayer('up', '${layer.id}')" title="Переместить слой вверх" ${canUp ? "" : "disabled"}>⬆</button>
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.moveHalftoneLayer('down', '${layer.id}')" title="Переместить слой вниз" ${canDown ? "" : "disabled"}>⬇</button>
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.duplicateHalftoneLayer('${layer.id}')" title="Дублировать слой">📋</button>
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.removeHalftoneLayer('${layer.id}')" title="Удалить слой">✕</button>
          `}
        </div>
      </div>
    `;
  }).join("");

  syncHalftoneLayerControls(app);
}

function selectHalftoneLayer(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const lyr = (st.layers || []).find((l) => l.id === id);
  if (!lyr || lyr.isOriginal) {
    renderHalftoneLayersUI(app);
    return;
  }

  st.activeLayerId = id;
  st.params = { ...(lyr.params || st.params || {}) };
  syncHalftoneControlsFromParams(st.params);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function addHalftoneLayer(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const rasterCount = (st.layers || []).filter((l) => l && !l.isOriginal).length;
  const newLyr = createHalftoneLayer(`Слой ${rasterCount + 1}`, { ...(st.params || {}) }, { hasMask: true });
  ensureLayerMaskCanvas(newLyr, false, app);
  newLyr.hasMask = true;
  markHalftoneMaskDirty(newLyr);
  st.layers.push(newLyr);
  selectHalftoneLayer(newLyr.id, app);
}

function duplicateHalftoneLayer(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const source = id
    ? (st.layers || []).find((l) => l.id === id)
    : getActiveHalftoneLayer(app);
  if (!source || source.isOriginal) {
    toast("warning", "Исходное фото нельзя дублировать");
    return;
  }

  const copy = createHalftoneLayer(`${source.name} (копия)`, source.params, {
    hasMask: Boolean(source.hasMask || source.maskCanvas),
  });
  copy.visible = source.visible !== false;
  copy.opacity = typeof source.opacity === "number" ? source.opacity : 1.0;
  copy.blend_mode = source.blend_mode || "normal";

  if (source.maskCanvas) {
    const srcCanvas = ensureLayerMaskCanvas(source, false, app);
    const dstCanvas = ensureLayerMaskCanvas(copy, false, app);
    dstCanvas.width = srcCanvas.width;
    dstCanvas.height = srcCanvas.height;
    const ctx = dstCanvas.getContext("2d");
    ctx.clearRect(0, 0, dstCanvas.width, dstCanvas.height);
    ctx.drawImage(srcCanvas, 0, 0);
    copy.hasMask = true;
    markHalftoneMaskDirty(copy);
  } else {
    copy.hasMask = false;
  }

  const srcIndex = st.layers.indexOf(source);
  st.layers.splice(srcIndex + 1, 0, copy);
  selectHalftoneLayer(copy.id, app);
}

function removeHalftoneLayer(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const target = id
    ? (st.layers || []).find((l) => l.id === id)
    : getActiveHalftoneLayer(app);
  if (!target) return;
  if (target.isOriginal) {
    toast("warning", "Исходное фото заблокировано и не может быть удалено");
    return;
  }
  if ((st.layers || []).filter((l) => l && !l.isOriginal).length <= 1) {
    toast("warning", "Нельзя удалить единственный растровый слой");
    return;
  }

  if (target._glMaskTexture && glContext) {
    glContext.deleteTexture(target._glMaskTexture);
    target._glMaskTexture = null;
  }

  const index = st.layers.indexOf(target);
  st.layers = st.layers.filter((l) => l !== target);
  if (!st.layers.some((l) => !l.isOriginal && l.isBase)) {
    const firstRaster = st.layers.find((l) => !l.isOriginal);
    if (firstRaster) firstRaster.isBase = true;
  }

  const fallback = st.layers[Math.min(index, st.layers.length - 1)];
  const next = fallback && !fallback.isOriginal ? fallback : st.layers.find((l) => !l.isOriginal);
  st.activeLayerId = next ? next.id : null;
  if (next) st.params = { ...(next.params || st.params) };
  syncHalftoneControlsFromParams(st.params);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function moveHalftoneLayer(direction, id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const layer = id
    ? (st.layers || []).find((l) => l.id === id)
    : getActiveHalftoneLayer(app);
  if (!layer || layer.isOriginal) return;

  const index = st.layers.indexOf(layer);
  const delta = direction === "up" ? 1 : (direction === "down" ? -1 : 0);
  const target = index + delta;
  if (!delta || target < 1 || target >= st.layers.length) return;

  st.layers[index] = st.layers[target];
  st.layers[target] = layer;
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function toggleHalftoneLayerVisible(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const lyr = (st.layers || []).find((l) => l.id === id);
  if (!lyr || lyr.isOriginal) return;
  lyr.visible = !lyr.visible;
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function toggleHalftoneCompare(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !st.sourceUrl) return;
  st.comparing = !st.comparing;
  const canvas = document.getElementById("ht-live-canvas");
  const img = document.getElementById("ht-preview-img");
  const badge = document.getElementById("ht-compare-badge");

  const btn = document.getElementById("ht-compare-btn");
  if (btn) btn.classList.toggle("active", st.comparing);

  if (st.comparing) {
    hideHalftoneRoi(app, true);
    if (canvas) canvas.style.display = "none";
    if (img) {
      img.style.visibility = "visible";
      img.src = st.sourceUrl;
    }
    if (badge) badge.style.display = "block";
  } else {
    if (badge) badge.style.display = "none";
    if (st.qualityMode === "live") {
      hideHalftoneRoi(app, true);
      if (img) img.style.visibility = "hidden";
      if (canvas) canvas.style.display = "block";
      renderHalftoneWebGL(app);
    } else {
      if (canvas) canvas.style.display = "none";
      if (img) {
        img.style.visibility = "visible";
        img.src = st.resultDataUrl || st.sourceUrl;
      }
      updateHalftoneRoiPosition(app);
      scheduleHalftoneRoiUpdate(app, false);
    }
  }
}

function revertHalftoneToOriginal(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !st.sourceUrl) return;
  st.comparing = true;
  hideHalftoneRoi(app, true);
  const canvas = document.getElementById("ht-live-canvas");
  const img = document.getElementById("ht-preview-img");
  const badge = document.getElementById("ht-compare-badge");

  if (canvas) canvas.style.display = "none";
  if (img) {
    img.style.visibility = "visible";
    img.src = st.sourceUrl;
  }
  if (badge) badge.style.display = "block";
  const btn = document.getElementById("ht-compare-btn");
  if (btn) btn.classList.add("active");
}

async function saveHalftoneToGallery(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Нет изображения для сохранения");
    return;
  }

  const loadOverlay = document.getElementById("ht-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";

  try {
    // Always re-render from the original source at full resolution. The live
    // canvas is a screen-sized preview and must never be used as export input.
    const res = await apiPost("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: buildHalftoneParamsPayload(st, false),
      layers: buildHalftoneLayersPayload(st, false),
      save: true,
      preview_max_side: 0,
    });
    if (res.saved) {
      toast("success", `Сохранено в галерею: ${res.saved.name} ✓`);
      if (app?.loadHistory) app.loadHistory();
    } else {
      toast("success", "Сохранено в галерею ✓");
    }
  } catch (e) {
    toast("error", "Ошибка сохранения: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

async function downloadHalftonePng(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Сначала выберите или загрузите изображение");
    return;
  }

  const loadOverlay = document.getElementById("ht-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";
  toast("info", "Рендеринг в полном разрешении для скачивания...");

  try {
    const res = await apiPost("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: buildHalftoneParamsPayload(st, false),
      layers: buildHalftoneLayersPayload(st, false),
      save: false,
      preview_max_side: 0,
    });

    const src = res?.preview_data_url;
    if (!src) throw new Error("Не удалось получить результат рендеринга");

    const a = document.createElement("a");
    a.href = src;
    a.download = `halftone_${res.width || "full"}x${res.height || "res"}_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    toast("success", `Скачано в оригинальном разрешении ${res.width}×${res.height} px ✓`);
  } catch (e) {
    toast("error", "Ошибка скачивания: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

function syncHalftoneMaskCanvasSize(w, h, appRef = null) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !w || !h) return;
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const lassoOverlay = document.getElementById("ht-lasso-overlay");
  if (maskCanvas && (maskCanvas.width !== w || maskCanvas.height !== h)) {
    maskCanvas.width = w;
    maskCanvas.height = h;
  }
  if (lassoOverlay && (lassoOverlay.width !== w || lassoOverlay.height !== h)) {
    lassoOverlay.width = w;
    lassoOverlay.height = h;
  }
}

function ensureLayerMaskCanvas(lyr, fillWhiteIfNew = false, appRef = null) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const w = st?.origWidth || st?.liveWidth || 800;
  const h = st?.origHeight || st?.liveHeight || 800;
  if (!lyr.maskCanvas) {
    lyr.maskCanvas = document.createElement("canvas");
    lyr.maskCanvas.width = w;
    lyr.maskCanvas.height = h;
    const ctx = lyr.maskCanvas.getContext("2d");
    if (fillWhiteIfNew) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.clearRect(0, 0, w, h);
    }
  } else if (lyr.maskCanvas.width !== w || lyr.maskCanvas.height !== h) {
    const old = lyr.maskCanvas;
    const next = document.createElement("canvas");
    next.width = w;
    next.height = h;
    const nCtx = next.getContext("2d");
    if (fillWhiteIfNew && !lyr.hasMask) {
      nCtx.fillStyle = "#ffffff";
      nCtx.fillRect(0, 0, w, h);
    } else {
      nCtx.drawImage(old, 0, 0, w, h);
    }
    lyr.maskCanvas = next;
  }
  return lyr.maskCanvas;
}

function renderMaskOverlayVisual(lyr, appRef = null) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const badge = document.getElementById("ht-selection-badge");
  if (!maskCanvas) return;
  const ctx = maskCanvas.getContext("2d");
  ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

  const hasMask = Boolean(lyr && lyr.hasMask && lyr.maskCanvas);
  if (badge) {
    badge.textContent = hasMask ? "Маска активна" : "Весь холст";
    badge.style.color = hasMask ? "var(--accent, #a855f7)" : "var(--text-dim)";
  }

  const showOverlay = st?.showMaskOverlay !== false;
  if (!showOverlay || !hasMask) {
    return;
  }

  ctx.save();
  ctx.drawImage(lyr.maskCanvas, 0, 0, maskCanvas.width, maskCanvas.height);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = "rgba(239, 68, 68, 0.45)";
  ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  ctx.restore();
}

function initHalftoneMaskEvents(appRef = null) {
  const app = appRef || window.App;
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const lassoOverlay = document.getElementById("ht-lasso-overlay");
  if (!maskCanvas || !lassoOverlay || maskCanvas._maskBound) return;
  maskCanvas._maskBound = true;

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(1, rect.width);
    const scaleY = canvas.height / Math.max(1, rect.height);
    return [
      (e.clientX - rect.left) * scaleX,
      (e.clientY - rect.top) * scaleY
    ];
  };

  let isPainting = false;
  let lastPos = null;

  maskCanvas.addEventListener("pointerdown", (e) => {
    const st = app?.state?.halftone;
    if (!st || (st.tool !== "brush" && st.tool !== "eraser")) return;
    const lyr = getActiveHalftoneLayer(app);
    if (!lyr) return;

    const fillWhite = (st.tool === "eraser" && !lyr.hasMask);
    const mCanvas = ensureLayerMaskCanvas(lyr, fillWhite, app);

    isPainting = true;
    lastPos = getPos(e, maskCanvas);
    try { maskCanvas.setPointerCapture(e.pointerId); } catch (_) {}

    const ctx = mCanvas.getContext("2d");
    const brushRadius = Math.max(2, ((st.brushSize || 24) * (mCanvas.width / 1000.0)) * 0.5);
    ctx.lineWidth = brushRadius * 2.0;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (st.tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#ffffff";
      ctx.fillStyle = "#ffffff";
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)";
      ctx.fillStyle = "rgba(0,0,0,1)";
    }

    ctx.beginPath();
    ctx.arc(lastPos[0], lastPos[1], brushRadius, 0, Math.PI * 2);
    ctx.fill();

    lyr.hasMask = true;
    markHalftoneMaskDirty(lyr);
    renderMaskOverlayVisual(lyr, app);
    triggerHalftoneLiveUpdate(app);
  });

  maskCanvas.addEventListener("pointermove", (e) => {
    if (!isPainting) return;
    const st = app?.state?.halftone;
    const lyr = getActiveHalftoneLayer(app);
    if (!lyr || !lyr.maskCanvas) return;
    const pos = getPos(e, maskCanvas);
    const ctx = lyr.maskCanvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(lastPos[0], lastPos[1]);
    ctx.lineTo(pos[0], pos[1]);
    ctx.stroke();
    lastPos = pos;
    markHalftoneMaskDirty(lyr);
    renderMaskOverlayVisual(lyr, app);
    triggerHalftoneLiveUpdate(app);
  });

  const stopPaint = (e) => {
    if (!isPainting) return;
    isPainting = false;
    try { maskCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
    const lyr = getActiveHalftoneLayer(app);
    if (lyr) {
      markHalftoneMaskDirty(lyr);
      renderMaskOverlayVisual(lyr, app);
      triggerHalftoneLiveUpdate(app);
    }
  };
  maskCanvas.addEventListener("pointerup", stopPaint);
  maskCanvas.addEventListener("pointercancel", stopPaint);

  let isLassoing = false;
  let lassoPts = [];

  lassoOverlay.addEventListener("pointerdown", (e) => {
    const st = app?.state?.halftone;
    if (!st || st.tool !== "lasso") return;
    const lyr = getActiveHalftoneLayer(app);
    if (!lyr) return;

    isLassoing = true;
    lassoPts = [getPos(e, lassoOverlay)];
    try { lassoOverlay.setPointerCapture(e.pointerId); } catch (_) {}
  });

  lassoOverlay.addEventListener("pointermove", (e) => {
    if (!isLassoing) return;
    lassoPts.push(getPos(e, lassoOverlay));
    const ctx = lassoOverlay.getContext("2d");
    ctx.clearRect(0, 0, lassoOverlay.width, lassoOverlay.height);
    ctx.strokeStyle = "#a855f7";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(lassoPts[0][0], lassoPts[0][1]);
    for (let i = 1; i < lassoPts.length; i++) {
      ctx.lineTo(lassoPts[i][0], lassoPts[i][1]);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  });

  const stopLasso = (e) => {
    if (!isLassoing) return;
    isLassoing = false;
    try { lassoOverlay.releasePointerCapture(e.pointerId); } catch (_) {}
    const ctx = lassoOverlay.getContext("2d");
    ctx.clearRect(0, 0, lassoOverlay.width, lassoOverlay.height);

    if (lassoPts.length > 2) {
      const lyr = getActiveHalftoneLayer(app);
      if (lyr) {
        const mode = app?.state?.halftone?.lassoMode || "replace";
        const fillWhite = (mode === "subtract" && !lyr.hasMask);
        const mCanvas = ensureLayerMaskCanvas(lyr, fillWhite, app);
        const mCtx = mCanvas.getContext("2d");

        if (mode === "replace") {
          mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);
          mCtx.globalCompositeOperation = "source-over";
          mCtx.fillStyle = "#ffffff";
          mCtx.beginPath();
          mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
          for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
          mCtx.closePath();
          mCtx.fill();
        } else if (mode === "add") {
          mCtx.globalCompositeOperation = "source-over";
          mCtx.fillStyle = "#ffffff";
          mCtx.beginPath();
          mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
          for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
          mCtx.closePath();
          mCtx.fill();
        } else if (mode === "subtract") {
          mCtx.globalCompositeOperation = "destination-out";
          mCtx.fillStyle = "rgba(0,0,0,1)";
          mCtx.beginPath();
          mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
          for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
          mCtx.closePath();
          mCtx.fill();
        } else if (mode === "intersect") {
          if (!lyr.hasMask) {
            mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);
            mCtx.globalCompositeOperation = "source-over";
            mCtx.fillStyle = "#ffffff";
            mCtx.beginPath();
            mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
            for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
            mCtx.closePath();
            mCtx.fill();
          } else {
            const tC = document.createElement("canvas");
            tC.width = mCanvas.width;
            tC.height = mCanvas.height;
            const tCtx = tC.getContext("2d");
            tCtx.fillStyle = "#ffffff";
            tCtx.beginPath();
            tCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
            for (let i = 1; i < lassoPts.length; i++) tCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
            tCtx.closePath();
            tCtx.fill();

            mCtx.globalCompositeOperation = "destination-in";
            mCtx.drawImage(tC, 0, 0);
            mCtx.globalCompositeOperation = "source-over";
          }
        }
        lyr.hasMask = true;
        markHalftoneMaskDirty(lyr);
        renderMaskOverlayVisual(lyr, app);
        triggerHalftoneLiveUpdate(app);
      }
    }
    lassoPts = [];
  };
  lassoOverlay.addEventListener("pointerup", stopLasso);
  lassoOverlay.addEventListener("pointercancel", stopLasso);
}

function setHalftoneTool(tool, appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;
  st.tool = tool || "view";

  const toolIds = ["ht-tool-view", "ht-tool-lasso", "ht-tool-brush", "ht-tool-eraser"];
  toolIds.forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.toggle("active", id === `ht-tool-${st.tool}`);
  });

  const wrap = document.getElementById("ht-preview-wrap");
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const lassoOverlay = document.getElementById("ht-lasso-overlay");

  if (wrap) {
    wrap.style.cursor = st.tool === "view" ? "grab" : "crosshair";
  }

  const isMaskPaint = (st.tool === "brush" || st.tool === "eraser");
  const isLasso = (st.tool === "lasso");
  const showMask = (st.showMaskOverlay !== false);

  if (maskCanvas) {
    maskCanvas.style.display = (showMask || isMaskPaint) ? "block" : "none";
    maskCanvas.style.cursor = isMaskPaint ? "crosshair" : "default";
    maskCanvas.style.pointerEvents = isMaskPaint ? "auto" : "none";
  }

  if (lassoOverlay) {
    lassoOverlay.style.display = isLasso ? "block" : "none";
    lassoOverlay.style.cursor = isLasso ? "crosshair" : "default";
    lassoOverlay.style.pointerEvents = isLasso ? "auto" : "none";
  }

  initHalftoneMaskEvents(app);
  renderMaskOverlayVisual(getActiveHalftoneLayer(app), app);
  toast("info", `Инструмент: ${st.tool}`);
}

function setHalftoneBrushSize(size, appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const val = parseInt(size, 10) || 24;
  if (app?.state?.halftone) app.state.halftone.brushSize = val;
  const sR = document.getElementById("ht-mask-brush");
  const sN = document.getElementById("ht-mask-brush-num");
  if (sR && sR.value != val) sR.value = val;
  if (sN && sN.value != val) sN.value = val;
}

function setHalftoneLassoMode(mode, appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const m = mode || "replace";
  if (app?.state?.halftone) app.state.halftone.lassoMode = m;

  ["replace", "add", "subtract", "intersect"].forEach(id => {
    const btn = document.getElementById(`ht-sel-${id}`);
    if (btn) btn.classList.toggle("active", id === m);
  });

  if (app?.state?.halftone?.tool === "view") {
    setHalftoneTool("lasso", app);
  }
}

function clearHalftoneMask(appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const lyr = getActiveHalftoneLayer(app);
  if (!lyr) return;

  if (lyr.maskCanvas) {
    const ctx = lyr.maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, lyr.maskCanvas.width, lyr.maskCanvas.height);
  }
  if (lyr.isBase) {
    // Base raster returns to full-canvas coverage.
    lyr.hasMask = false;
  } else {
    // Local layers stay masked; an empty mask means "not painted yet".
    ensureLayerMaskCanvas(lyr, false, app);
    lyr.hasMask = true;
  }
  markHalftoneMaskDirty(lyr);

  const maskCanvas = document.getElementById("ht-mask-canvas");
  if (maskCanvas) {
    const ctx = maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  }
  renderMaskOverlayVisual(lyr, app);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
  toast("info", lyr.isBase ? "Маска снята (скринтон на весь холст)" : "Маска очищена (локальный слой скрыт)");
}

function invertHalftoneMaskSelection(appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const lyr = getActiveHalftoneLayer(app);
  if (lyr && lyr.maskCanvas && lyr.hasMask) {
    const ctx = lyr.maskCanvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, lyr.maskCanvas.width, lyr.maskCanvas.height);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const newA = 255 - d[i + 3];
      d[i] = newA;
      d[i + 1] = newA;
      d[i + 2] = newA;
      d[i + 3] = newA;
    }
    ctx.putImageData(imgData, 0, 0);
    markHalftoneMaskDirty(lyr);
    renderMaskOverlayVisual(lyr, app);
    triggerHalftoneLiveUpdate(app);
    toast("info", "Выделение маски инвертировано");
  } else if (app?.state?.halftone?.params) {
    const p = app.state.halftone.params;
    p.invert_mask = !p.invert_mask;
    const chk = document.getElementById("ht-invert-mask");
    if (chk) chk.checked = p.invert_mask;
    triggerHalftoneLiveUpdate(app);
    toast("info", "Инверсия маски слоя переключена");
  }
}

function toggleHalftoneMaskOverlay(checked, appRef = null) {
  const app = (typeof checked === "object" && checked && checked.state) ? checked : (appRef || window.App);
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;
  st.showMaskOverlay = (typeof checked === "boolean") ? checked : !st.showMaskOverlay;
  const chk = document.getElementById("ht-show-mask-overlay");
  if (chk) chk.checked = st.showMaskOverlay;
  setHalftoneTool(st.tool, app);
}

function toggleHalftoneMaskMode(checked, appRef = null) {
  const app = (typeof checked === "object" && checked && checked.state) ? checked : (appRef || window.App);
  initHalftoneStudio(app);
  const isChecked = (typeof checked === "boolean") ? checked : true;
  setHalftoneTool(isChecked ? "brush" : "view", app);
}

function toggleHalftoneMaskEraser(appRef = null) {
  setHalftoneTool("eraser", appRef);
}

function resetHalftoneParams(appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;
  const mode = st.params.mode || "manga";
  const def = halftonePresets[mode] || halftonePresets.manga;
  st.params = { ...def };
  const lyr = getActiveHalftoneLayer(app);
  if (lyr) lyr.params = { ...st.params };
  syncHalftoneControlsFromParams(st.params);
  triggerHalftoneLiveUpdate(app);
  toast("info", "Параметры сброшены до значений пресета");
}

function onHalftoneFileUpload(input, appRef) {
  const app = appRef || window.App;
  const file = input?.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    loadHalftoneSource(e.target.result, file.name, app);
  };
  reader.readAsDataURL(file);
  input.value = "";
}

function resolveHalftoneSourceRef(src) {
  const raw = String(src || "").trim();
  const fallback = { sourceUrl: raw, sourceDataUrl: raw, sourcePath: null };
  if (!raw) return fallback;
  try {
    const parsed = new URL(raw, window.location.href);
    if (parsed.origin === window.location.origin && parsed.pathname.startsWith("/outputs/")) {
      // Gallery/history images live on disk. Passing the path avoids a huge
      // base64 round-trip and lets export read the untouched original file.
      return {
        sourceUrl: parsed.pathname + parsed.search,
        sourceDataUrl: null,
        sourcePath: parsed.pathname,
      };
    }
  } catch (_) {}
  return fallback;
}

function loadHalftoneSource(dataUrl, filename = "image.png", appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app.state.halftone;
  const sourceRef = resolveHalftoneSourceRef(dataUrl);
  st.sourceDataUrl = sourceRef.sourceDataUrl;
  st.sourceUrl = sourceRef.sourceUrl;
  st.sourcePath = sourceRef.sourcePath;

  // Reset everything tied to the previous source so stale previews, ROI
  // rectangles and GPU textures can never leak into the new document.
  st.resultDataUrl = null;
  st.roiDataUrl = null;
  st.roiRect = null;
  st._roiSignature = null;
  st._pendingRoiPlan = null;
  st._sourceSizePromise = null;
  st.origWidth = 0;
  st.origHeight = 0;
  st.liveWidth = 0;
  st.liveHeight = 0;
  st.liveReady = false;
  st._liveSourcePreparedUrl = null;
  st._liveSourceToken = (Number(st._liveSourceToken) || 0) + 1;
  st._editRevision = 0;
  st._previewSeq = (Number(st._previewSeq) || 0) + 1;
  st._roiSeq = (Number(st._roiSeq) || 0) + 1;
  st._gpuCanvasMaxSide = 0;
  st.comparing = false;
  if (st._previewAbort) {
    try { st._previewAbort.abort(); } catch (_) {}
    st._previewAbort = null;
  }
  if (st._roiTimer) {
    clearTimeout(st._roiTimer);
    st._roiTimer = null;
  }
  if (st._layoutTimer) {
    clearTimeout(st._layoutTimer);
    st._layoutTimer = null;
  }
  hideHalftoneRoi(app, true);

  resetHalftoneLayerStack(app);

  const emptyEl = document.getElementById("ht-empty-state");
  const wrapEl = document.getElementById("ht-preview-wrap");
  const imgEl = document.getElementById("ht-preview-img");
  if (emptyEl) emptyEl.style.display = "none";
  if (wrapEl) wrapEl.style.display = "inline-flex";
  if (imgEl) imgEl.src = dataUrl;

  const compareBtn = document.getElementById("ht-compare-btn");
  if (compareBtn) compareBtn.classList.remove("active");
  const badge = document.getElementById("ht-compare-badge");
  if (badge) badge.style.display = "none";

  const metaEl = document.getElementById("ht-meta-info");
  if (metaEl) metaEl.textContent = `Файл: ${filename}`;

  if (st.qualityMode === "live") {
    prepareHalftoneLiveSource(sourceRef.sourceUrl, app);
  } else {
    runHalftonePreview(true, app);
  }
}

function onStudioDrop(e, target, appRef) {
  e.preventDefault();
  e.stopPropagation();
  const app = appRef || window.App;
  const files = e.dataTransfer?.files;
  if (!files || !files.length) return;
  const file = files[0];
  if (!file.type.startsWith("image/")) {
    toast("warning", "Пожалуйста, перетащите файл изображения (PNG, JPG, WEBP)");
    return;
  }
  const reader = new FileReader();
  reader.onload = (ev) => {
    const dataUrl = ev.target.result;
    if (target === "halftone") {
      loadHalftoneSource(dataUrl, file.name, app);
    } else if (target === "manga") {
      if (app.loadMangaSource) app.loadMangaSource(dataUrl, file.name);
    } else {
      if (app.uploadImg2ImgFile) app.uploadImg2ImgFile(file);
    }
  };
  reader.readAsDataURL(file);
}

async function sendHalftoneToManga(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Нет результата Halftone Studio для передачи в Manga Screen AI");
    return;
  }

  const loadOverlay = document.getElementById("ht-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";

  try {
    const res = await apiPost("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: buildHalftoneParamsPayload(st, false),
      layers: buildHalftoneLayersPayload(st, false),
      save: false,
      preview_max_side: 0,
    });
    const src = res?.preview_data_url || st.resultDataUrl || st.sourceDataUrl;
    if (!src) throw new Error("Не удалось получить результат рендеринга");

    app.switchTab("manga");
    if (app.loadMangaSource) {
      app.loadMangaSource(src, "halftone_export.png");
    }
    toast("success", "Результат Halftone Studio передан в Manga Screen AI ✓");
  } catch (e) {
    toast("error", "Ошибка передачи в Manga AI: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

let _galleryPickerTarget = "halftone";

function onHalftoneLayerMetaChange(prop, val, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  const lyr = getActiveHalftoneLayer(app);
  if (!st || !lyr) return;
  if (prop === "opacity") {
    const v = parseFloat(val) / 100.0;
    lyr.opacity = isNaN(v) ? 1.0 : Math.max(0, Math.min(1, v));
  } else if (prop === "blend_mode") {
    lyr.blend_mode = val || "normal";
    if (lyr.blend_mode === "multiply") {
      applyHalftoneMoireProtection(lyr, st, app);
    }
  }
  syncHalftoneLayerControls(app);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function openGalleryPicker(targetTool = "halftone", appRef) {
  const app = appRef || window.App;
  _galleryPickerTarget = targetTool;
  const modal = document.getElementById("gallery-picker-modal");
  const grid = document.getElementById("gallery-picker-grid");
  const title = document.getElementById("gallery-picker-title");
  if (!modal || !grid) return;

  if (title) {
    const titles = {
      halftone: "Выберите изображение для Halftone Studio GT",
      manga: "Выберите изображение для Manga Screen AI",
      generate: "Выберите изображение для редактирования по маске",
      inpaint: "Выберите изображение для Inpaint",
    };
    title.textContent = titles[targetTool] || titles.halftone;
  }

  const items = app?.state?.historyItems || [];
  if (!items.length) {
    grid.innerHTML = '<div style="grid-column: 1/-1; padding: 24px; text-align: center; color: var(--text-dim);">Галерея генераций пуста</div>';
  } else {
    grid.innerHTML = items.map(it => {
      const thumbUrl = it.thumb_url || it.image_url || it.url;
      const fullUrl = it.image_url || it.url || thumbUrl;
      return `
        <div class="gallery-picker-item" onclick="App.selectGalleryPickerItem('${escapeHtml(fullUrl)}', '${escapeHtml(it.filename || it.name || '')}')" style="cursor: pointer; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); aspect-ratio: 1; background: var(--bg-surface); transition: transform 0.15s, border-color 0.15s;">
          <img src="${thumbUrl}" alt="History item" style="width: 100%; height: 100%; object-fit: cover; display: block;" loading="lazy">
        </div>
      `;
    }).join("");
  }

  modal.style.display = "flex";
}

function closeGalleryPicker() {
  const modal = document.getElementById("gallery-picker-modal");
  if (modal) modal.style.display = "none";
}

// Resolve the server side path of a gallery item (relative to the output dir).
function resolveOutputPath(url, filename) {
  if (filename) return filename;
  if (typeof url === "string") {
    const marker = "/outputs/";
    const idx = url.indexOf(marker);
    if (idx >= 0) {
      try {
        return decodeURIComponent(url.slice(idx + marker.length).split("?")[0]);
      } catch (_) {
        return url.slice(idx + marker.length).split("?")[0];
      }
    }
  }
  return "";
}

function selectGalleryPickerItem(url, filename, appRef) {
  closeGalleryPicker();
  const app = appRef || window.App;
  const target = _galleryPickerTarget || "halftone";
  if (target === "halftone") {
    loadHalftoneSource(url, filename || "gallery_item.png", app);
  } else if (target === "manga") {
    if (app.loadMangaSource) app.loadMangaSource(url, filename || "gallery_item.png");
  } else if (target === "generate") {
    if (app.enableGenMaskWithSource) {
      const path = resolveOutputPath(url, filename);
      app.enableGenMaskWithSource(app, { url, path, name: path || filename || "image.png" });
    }
  } else if (target === "inpaint") {
    if (app.loadInpaintSource) {
      const path = resolveOutputPath(url, filename);
      app.loadInpaintSource(url, path, path || filename || "image.png");
    }
  }
}

function openHistoryItemInTool(tool, appRef) {
  const app = appRef || window.App;
  const it = app?.state?.selectedHistoryItem;
  if (!it) return;
  const url = it.image_url || it.url || it.thumb_url;
  if (!url) return;
  if (app.closeHistoryModal) app.closeHistoryModal();
  app.switchTab(tool);
  if (tool === "halftone") {
    loadHalftoneSource(url, it.filename || "history_image.png", app);
  } else if (tool === "manga") {
    if (app.loadMangaSource) app.loadMangaSource(url, it.filename || "history_image.png");
  }
}

function useCanvasImageInTool(tool, appRef) {
  const app = appRef || window.App;
  const curImg = app?.state?.lastPreviewUrl || document.getElementById("preview-img")?.src;
  if (!curImg) {
    toast("warning", "Нет сгенерированного изображения для передачи");
    return;
  }
  app.switchTab(tool);
  if (tool === "halftone") {
    loadHalftoneSource(curImg, "generated_preview.png", app);
  } else if (tool === "manga") {
    if (app.loadMangaSource) app.loadMangaSource(curImg, "generated_preview.png");
  }
}


  Object.assign(App, {
    halftoneBaseExtraParams,
    halftonePresets,
    _htBuffers,
    createHalftoneLayer,
    initHalftoneStudio,
    setHalftoneQualityMode,
    setHalftonePreviewQuality,
    prepareHalftoneLiveSource,
    triggerHalftoneLiveUpdate,
    renderHalftoneWebGL,
    runHalftonePreview,
    selectHalftoneMode,
    syncHalftoneControlsFromParams,
    onHalftoneParamInput,
    getActiveHalftoneLayer,
    renderHalftoneLayersUI,
    selectHalftoneLayer,
    addHalftoneLayer,
    duplicateHalftoneLayer,
    removeHalftoneLayer,
    moveHalftoneLayer,
    syncHalftoneLayerControls,
    toggleHalftoneLayerVisible,
    toggleHalftoneCompare,
    revertHalftoneToOriginal,
    saveHalftoneToGallery,
    downloadHalftonePng,
    syncHalftoneMaskCanvasSize,
    ensureLayerMaskCanvas,
    renderMaskOverlayVisual,
    initHalftoneMaskEvents,
    setHalftoneTool,
    setHalftoneBrushSize,
    setHalftoneLassoMode,
    clearHalftoneMask,
    invertHalftoneMaskSelection,
    toggleHalftoneMaskOverlay,
    toggleHalftoneMaskMode,
    toggleHalftoneMaskEraser,
    resetHalftoneParams,
    onHalftoneLayerMetaChange,
    onHalftoneFileUpload,
    loadHalftoneSource,
    onStudioDrop,
    sendHalftoneToManga,
    openGalleryPicker,
    closeGalleryPicker,
    selectGalleryPickerItem,
    openHistoryItemInTool,
    useCanvasImageInTool,
  });
})();

// ============================================================
// --- INTEGRATED MANGA SCREEN AI ENGINE ---
// ============================================================
(function() {
  const apiPost = (url, body, retries = 0) => App.apiPost(url, body, retries);
  const apiGet = (url) => App.apiGet(url);
  const toast = (type, msg) => App.toast(type, msg);

/* ============================================================
   QuantFunc WebUI — Manga Studio AI Controller
   B&W stylization, G-pen lineart, AI reference pipeline, compare mode.
   ============================================================ */


function defaultMangaParams() {
  return {
    mode: "manga_lite",
    ai_enabled: false,
    model_id: "",
    variant_id: "",
    lora_name: "",
    lora_strength: 0.85,
    ai_prompt: "black and white Japanese manga illustration, clean G-pen ink lineart, high contrast screentone shading, monochrome comic art",
    ai_negative_prompt: "color, blurry, photo, 3d render, watermark, muddy shading",
    ai_denoise_strength: 0.65,
    ai_steps: 0,
    ai_cfg: 0.0,
    ai_ref_mode: "control_ref",
    levels_enabled: true,
    levels_contrast: 1.25,
    levels_gamma: 1.05,
    levels_brightness: 0.0,
    levels_descreen: 0.20,
    lines_enabled: true,
    line_strength: 1.15,
    line_threshold: 0.28,
    line_thickness: 1,
    line_invert: false,
    finish_enabled: true,
    finish_preset: "clean",
    finish_dot_size: 6,
    finish_dot_shape: "circle",
    finish_angle: 45.0,
    finish_max_radius_ratio: 0.90,
    finish_ink_threshold: 0.28,
    finish_tone_gamma: 1.05,
    finish_light_tone: 0.80,
    finish_mid_tone: 0.50,
    finish_black_point: 0.18,
    finish_region_size: 6,
    finish_gradation: 0.25,
    finish_white_cut: 0.35,
    finish_invert: false,
    panel_enabled: false,
    panel_aspect: "original",
    panel_crop_x: 0.0,
    panel_crop_y: 0.0,
    panel_border: 4,
    panel_margin: 18,
    panel_paper_tone: "off-white",
    panel_print_bleed: 0.15,
    seed: 42,
  };
}

function initMangaStudio(appRef) {
  const app = appRef || window.App;
  if (!app) return;
  if (!app.state.manga) {
    app.state.manga = {
      params: defaultMangaParams(),
      sourcePath: null,
      sourceUrl: null,
      sourceDataUrl: null,
      resultDataUrl: null,
      comparing: false,
      debounceTimer: null,
    };
  }
  populateMangaModelsAndLoras(app);
  syncMangaControlsFromParams(app.state.manga.params);
}

function populateMangaModelsAndLoras(appRef) {
  const app = appRef || window.App;
  if (!app) return;

  const modelSel = document.getElementById("manga-ai-model");
  if (modelSel) {
    const models = (app.state.models || []).filter(m => !m.disabled);
    const curVal = app.state.manga?.params?.model_id || modelSel.value || app.state.selectedModel || "";
    modelSel.innerHTML = models.map(m => {
      const eng = (m.engine || "qf").toUpperCase();
      const inst = m.installed ? "✓ " : "⬇ ";
      return `<option value="${escapeHtml(m.id)}">${inst}${escapeHtml(m.name)} [${eng}]</option>`;
    }).join("");

    if (curVal && models.some(m => m.id === curVal)) {
      modelSel.value = curVal;
      if (app.state.manga) app.state.manga.params.model_id = curVal;
    } else if (models.length) {
      modelSel.value = models[0].id;
      if (app.state.manga) app.state.manga.params.model_id = models[0].id;
    }
  }

  const loraSel = document.getElementById("manga-ai-lora");
  if (loraSel) {
    const loras = app.state.loraLibrary || [];
    const curLora = app.state.manga?.params?.lora_name || "";
    const noneText = window.t ? (window.t("manga.ai_lora_none") || "Без LoRA (базовая модель)") : "Без LoRA (базовая модель)";
    loraSel.innerHTML = `<option value="">${noneText}</option>` +
      loras.map(l => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)} (${escapeHtml(l.family || "flux")})</option>`).join("");
    loraSel.value = curLora;
  }

  checkMangaLoraCompatibility(app);
}

async function onMangaModelChange(modelId, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  if (app.state.manga) app.state.manga.params.model_id = modelId;
  await checkMangaLoraCompatibility(app);
}

async function onMangaLoraChange(loraName, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  if (app.state.manga) app.state.manga.params.lora_name = loraName || "";
  await checkMangaLoraCompatibility(app);
}

async function checkMangaLoraCompatibility(appRef) {
  const app = appRef || window.App;
  const badge = document.getElementById("manga-lora-compat-badge");
  if (!badge || !app?.state?.manga) return;

  const modelId = app.state.manga.params.model_id;
  const loraName = app.state.manga.params.lora_name;
  if (!modelId) {
    badge.textContent = "";
    return;
  }

  const modelObj = (app.state.models || []).find(m => m.id === modelId);
  if (modelObj && modelObj.engine === "sdnq" && !loraName) {
    badge.style.color = "var(--text-dim)";
    badge.textContent = "ℹ️ Движок SDNQ использует мультимодальный Qwen-Image референс без внешних LoRA.";
    return;
  }
  if (!loraName) {
    badge.style.color = "var(--text-dim)";
    badge.textContent = "✓ Работа без LoRA (семантическая стилизация базовой моделью)";
    return;
  }

  try {
    const res = await apiGet(`/api/loras/check-compatibility?model_id=${encodeURIComponent(modelId)}&lora_name=${encodeURIComponent(loraName)}`);
    if (res.compatible) {
      badge.style.color = "#34d399";
      badge.textContent = `✓ Совместимо (${res.model_family || "Flux/Qwen"})`;
    } else {
      badge.style.color = "#f87171";
      badge.textContent = `⚠️ Несовместимо: модель ${res.model_family}, а LoRA для ${res.lora_family}`;
    }
  } catch (_) {
    badge.textContent = "";
  }
}

function selectMangaMode(mode, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const p = app.state.manga.params;
  p.mode = mode;

  if (mode === "manga_lite") {
    p.ai_enabled = false;
    p.levels_enabled = true;
    p.lines_enabled = true;
    p.finish_enabled = true;
  } else if (mode === "finish_only") {
    p.ai_enabled = false;
    p.levels_enabled = false;
    p.lines_enabled = false;
    p.finish_enabled = true;
  } else if (mode === "ai_manga") {
    p.ai_enabled = true;
    p.levels_enabled = false;
    p.lines_enabled = false;
    p.finish_enabled = false;
  } else if (mode === "ai_manga_finish") {
    p.ai_enabled = true;
    p.levels_enabled = true;
    p.lines_enabled = true;
    p.finish_enabled = true;
  }

  const banner = document.getElementById("manga-fallback-banner");
  if (banner && (mode === "manga_lite" || mode === "finish_only")) {
    banner.style.display = "none";
  }

  syncMangaControlsFromParams(p);
  if (mode === "manga_lite" || mode === "finish_only") {
    runMangaPipeline(false, app);
  }
}

function syncMangaControlsFromParams(p) {
  if (!p) return;
  document.querySelectorAll("#manga-mode-pills .studio-mode-pill").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === p.mode);
  });

  const stageACard = document.getElementById("manga-stage-a-card");
  if (stageACard) {
    stageACard.style.display = (p.mode === "ai_manga" || p.mode === "ai_manga_finish" || p.ai_enabled) ? "flex" : "none";
  }

  const setChk = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.checked = Boolean(v);
  };
  setChk("manga-ai-enabled", p.ai_enabled);
  setChk("manga-levels-enabled", p.levels_enabled);
  setChk("manga-lines-enabled", p.lines_enabled);
  setChk("manga-finish-enabled", p.finish_enabled);
  setChk("manga-panel-enabled", p.panel_enabled);
  setChk("manga-line-invert", p.line_invert);

  const setPair = (rId, nId, val) => {
    const r = document.getElementById(rId);
    const n = document.getElementById(nId);
    if (r && val !== undefined) r.value = val;
    if (n && val !== undefined) n.value = val;
  };
  setPair("manga-ai-lora-weight", "manga-ai-lora-weight-num", p.lora_strength);
  setPair("manga-ai-denoise", "manga-ai-denoise-num", p.ai_denoise_strength);
  setPair("manga-levels-contrast", "manga-levels-contrast-num", p.levels_contrast);
  setPair("manga-levels-gamma", "manga-levels-gamma-num", p.levels_gamma);
  setPair("manga-levels-descreen", "manga-levels-descreen-num", p.levels_descreen);
  setPair("manga-line-strength", "manga-line-strength-num", p.line_strength);
  setPair("manga-line-threshold", "manga-line-threshold-num", p.line_threshold);
  setPair("manga-line-thickness", "manga-line-thickness-num", p.line_thickness);
  setPair("manga-finish-dot-size", "manga-finish-dot-size-num", p.finish_dot_size);
  setPair("manga-finish-ink", "manga-finish-ink-num", p.finish_ink_threshold);
  setPair("manga-finish-light-tone", "manga-finish-light-tone-num", p.finish_light_tone);
  setPair("manga-finish-mid-tone", "manga-finish-mid-tone-num", p.finish_mid_tone);
  setPair("manga-finish-black-point", "manga-finish-black-point-num", p.finish_black_point);
  setPair("manga-finish-white-cut", "manga-finish-white-cut-num", p.finish_white_cut);
  setPair("manga-panel-border", "manga-panel-border-num", p.panel_border);
  setPair("manga-panel-margin", "manga-panel-margin-num", p.panel_margin);
  setPair("manga-panel-bleed", "manga-panel-bleed-num", p.panel_print_bleed);

  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el && val !== undefined) el.value = val;
  };
  setVal("manga-ai-ref-mode", p.ai_ref_mode);
  setVal("manga-ai-prompt", p.ai_prompt);
  setVal("manga-finish-preset", p.finish_preset);
  setVal("manga-finish-dot-shape", p.finish_dot_shape);
  setVal("manga-panel-aspect", p.panel_aspect);
  setVal("manga-panel-paper", p.panel_paper_tone);
}

function onMangaParamInput(key, rawVal, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  let val = rawVal;
  if (["line_thickness", "finish_dot_size", "panel_border", "panel_margin", "seed"].includes(key)) {
    val = parseInt(rawVal, 10);
    if (isNaN(val)) return;
  } else if ([
    "lora_strength", "ai_denoise_strength", "levels_contrast", "levels_gamma", "levels_descreen",
    "line_strength", "line_threshold", "finish_ink_threshold", "finish_light_tone",
    "finish_mid_tone", "finish_black_point", "finish_white_cut", "panel_print_bleed"
  ].includes(key)) {
    val = parseFloat(rawVal);
    if (isNaN(val)) return;
  }

  app.state.manga.params[key] = val;
  syncMangaControlsFromParams(app.state.manga.params);

  if (app.state.manga.params.mode === "manga_lite" || app.state.manga.params.mode === "finish_only") {
    if (app.state.manga.debounceTimer) clearTimeout(app.state.manga.debounceTimer);
    app.state.manga.debounceTimer = setTimeout(() => {
      runMangaPipeline(false, app);
    }, 220);
  }
}

function resetMangaParams(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const curMode = app.state.manga.params?.mode || "manga_lite";
  app.state.manga.params = defaultMangaParams();
  selectMangaMode(curMode, app);
}

async function runMangaPipeline(immediate = false, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourcePath && !st.sourceDataUrl) {
    if (immediate) toast("warning", "Сначала загрузите изображение или выберите его из галереи");
    return;
  }

  const p = st.params;
  const isAi = (p.mode === "ai_manga" || p.mode === "ai_manga_finish") && p.ai_enabled;
  const metaEl = document.getElementById("manga-meta-info");
  const banner = document.getElementById("manga-fallback-banner");
  const bannerMsg = document.getElementById("manga-fallback-msg");
  const loadOverlay = document.getElementById("manga-loading-overlay");
  const loadText = document.getElementById("manga-loading-text");

  if (!isAi) {
    if (metaEl) metaEl.textContent = t("manga.processing_meta");
    if (loadText) loadText.textContent = t("manga.processing_cpu");
    if (loadOverlay) loadOverlay.style.display = "flex";

    try {
      const res = await apiPost("/api/manga/process", {
        image_path: st.sourcePath,
        image_data: st.sourceDataUrl,
        params: p,
        save: false,
        preview_max_side: 0, // Full resolution: preserves fine manga lines
      });

      st.resultDataUrl = res.preview_data_url;
      st.comparing = false;

      const emptyEl = document.getElementById("manga-empty-state");
      const wrapEl = document.getElementById("manga-preview-wrap");
      const imgEl = document.getElementById("manga-preview-img");
      const badgeEl = document.getElementById("manga-compare-badge");

      if (emptyEl) emptyEl.style.display = "none";
      if (wrapEl) wrapEl.style.display = "inline-flex";
      if (imgEl) imgEl.src = res.preview_data_url;
      if (badgeEl) badgeEl.style.display = "none";

      if (metaEl) {
        metaEl.textContent = `${res.width}×${res.height} px · ${(res.elapsed_sec * 1000).toFixed(0)} ms (CPU)`;
      }
    } catch (e) {
      if (metaEl) metaEl.textContent = t("manga.error");
      toast("error", "Manga pipeline error: " + e.message);
    } finally {
      if (loadOverlay) loadOverlay.style.display = "none";
    }
    return;
  }

  // AI GPU pipeline
  if (metaEl) metaEl.textContent = "⏳ Запуск AI Manga (GPU)...";
  if (loadText) loadText.textContent = "Запуск AI Manga (GPU)...";
  if (loadOverlay) loadOverlay.style.display = "flex";
  if (banner) banner.style.display = "none";

  try {
    const res = await apiPost("/api/manga/ai-job", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: p,
    });

    const jobId = res.job_id;
    const poll = setInterval(async () => {
      try {
        const status = await apiGet(`/api/jobs/${jobId}`);
        if (status.status === "queued") {
          app.showQueueStatus(status, jobId);
          const queueText = window.t ? window.t("manga.queue_waiting") : "В очереди GPU...";
          if (metaEl) metaEl.textContent = queueText;
          if (loadText) loadText.textContent = queueText;
          return;
        }
        app.hideQueueStatus();
        const msgText = status.message || `⏳ ${Math.round((status.progress || 0) * 100)}%`;
        if (metaEl) metaEl.textContent = msgText;
        if (loadText) loadText.textContent = msgText;

        if (status.status === "cancelled") {
          clearInterval(poll);
          app.hideQueueStatus();
          if (loadOverlay) loadOverlay.style.display = "none";
          if (metaEl) metaEl.textContent = "Отменено";
          toast("info", window.t ? window.t("queue.cancelled") : "Задача отменена");
        } else if (status.status === "complete") {
          clearInterval(poll);
          app.hideQueueStatus();
          if (loadOverlay) loadOverlay.style.display = "none";
          const filename = (status.image_path || "").split(/[\/\\]/).pop();
          const url = `/outputs/${encodeURIComponent(filename)}?t=${Date.now()}`;
          st.resultDataUrl = url;
          st.comparing = false;

          const emptyEl = document.getElementById("manga-empty-state");
          const wrapEl = document.getElementById("manga-preview-wrap");
          const imgEl = document.getElementById("manga-preview-img");
          if (emptyEl) emptyEl.style.display = "none";
          if (wrapEl) wrapEl.style.display = "inline-flex";
          if (imgEl) imgEl.src = url;

          if (metaEl) metaEl.textContent = status.message || "✓ Готово";
          const savedMsg = window.t ? (window.t("manga.saved_toast") || "Результат сохранён в галерею ✓") : "Результат сохранён в галерею ✓";
          toast("success", savedMsg);
        } else if (status.status === "error") {
          clearInterval(poll);
          app.hideQueueStatus();
          if (loadOverlay) loadOverlay.style.display = "none";
          if (metaEl) metaEl.textContent = "Ошибка AI";
          if (banner && bannerMsg) {
            bannerMsg.textContent = `Ошибка AI-этапа: ${status.error || status.message}. Режим Manga Lite доступен без GPU.`;
            banner.style.display = "flex";
          }
          toast("error", status.error || status.message || "AI Manga failed");
        }
      } catch (_) {
        clearInterval(poll);
        app.hideQueueStatus();
        if (loadOverlay) loadOverlay.style.display = "none";
      }
    }, 400);
  } catch (e) {
    if (loadOverlay) loadOverlay.style.display = "none";
    if (metaEl) metaEl.textContent = "Ошибка AI";
    if (banner && bannerMsg) {
      bannerMsg.textContent = `${e.message} — Вы можете переключиться в режим Manga Lite (CPU) без загрузки моделей.`;
      banner.style.display = "flex";
    }
    toast("error", e.message);
  }
}

function toggleMangaCompare(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourceUrl || !st.resultDataUrl) return;

  st.comparing = !st.comparing;
  const imgEl = document.getElementById("manga-preview-img");
  const badgeEl = document.getElementById("manga-compare-badge");
  const btnEl = document.getElementById("manga-compare-btn");

  if (imgEl) imgEl.src = st.comparing ? st.sourceUrl : st.resultDataUrl;
  if (badgeEl) badgeEl.style.display = st.comparing ? "block" : "none";
  if (btnEl) btnEl.classList.toggle("active", st.comparing);
}

function revertMangaToOriginal(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourceUrl) return;

  st.comparing = true;
  const imgEl = document.getElementById("manga-preview-img");
  const badgeEl = document.getElementById("manga-compare-badge");
  if (imgEl) imgEl.src = st.sourceUrl;
  if (badgeEl) badgeEl.style.display = "block";
}

async function saveMangaToGallery(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourcePath && !st.sourceDataUrl) {
    toast("warning", "Нет изображения для сохранения");
    return;
  }

  if (st.params.mode === "ai_manga" || st.params.mode === "ai_manga_finish") {
    await runMangaPipeline(true, app);
    return;
  }

  try {
    const res = await apiPost("/api/manga/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: st.params,
      save: true,
    });
    if (res.saved) {
      st.resultDataUrl = res.preview_data_url;
      const imgEl = document.getElementById("manga-preview-img");
      if (imgEl) imgEl.src = res.preview_data_url;
      const savedMsg = window.t ? (window.t("manga.saved_toast") || "Сохранено в галерею") : "Сохранено в галерею";
      toast("success", `${savedMsg}: ${res.saved.name} ✓`);
    }
  } catch (e) {
    toast("error", "Ошибка сохранения: " + e.message);
  }
}

async function downloadMangaPng(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Сначала выберите или загрузите изображение");
    return;
  }

  const loadOverlay = document.getElementById("manga-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";
  toast("info", "Рендеринг в полном разрешении для скачивания...");

  try {
    const p = { ...st.params, mode: "manga_screen", finish_enabled: false };
    const res = await apiPost("/api/manga/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: p,
      save: false,
      preview_max_side: 0,
    });

    const src = res?.preview_data_url || st.resultDataUrl;
    if (!src) throw new Error("Не удалось получить результат рендеринга");

    const a = document.createElement("a");
    a.href = src;
    a.download = `manga_${res.width || "full"}x${res.height || "res"}_${Date.now()}.png`;
    a.click();
    toast("success", `Скачано в оригинальном разрешении ${res.width}×${res.height} px ✓`);
  } catch (e) {
    toast("error", "Ошибка скачивания: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

function onMangaFileUpload(input, appRef) {
  const app = appRef || window.App;
  const file = input?.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    loadMangaSource(e.target.result, file.name, app);
  };
  reader.readAsDataURL(file);
  input.value = "";
}

function loadMangaSource(dataUrl, filename = "image.png", appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  st.sourceDataUrl = dataUrl;
  st.sourceUrl = dataUrl;
  st.sourcePath = null;

  const emptyEl = document.getElementById("manga-empty-state");
  const wrapEl = document.getElementById("manga-preview-wrap");
  const imgEl = document.getElementById("manga-preview-img");
  if (emptyEl) emptyEl.style.display = "none";
  if (wrapEl) wrapEl.style.display = "inline-flex";
  if (imgEl) imgEl.src = dataUrl;

  const metaEl = document.getElementById("manga-meta-info");
  if (metaEl) metaEl.textContent = `Файл: ${filename}`;

  runMangaPipeline(true, app);
}


  Object.assign(App, {
    defaultMangaParams,
    initMangaStudio,
    populateMangaModelsAndLoras,
    onMangaModelChange,
    onMangaLoraChange,
    checkMangaLoraCompatibility,
    selectMangaMode,
    syncMangaControlsFromParams,
    onMangaParamInput,
    resetMangaParams,
    runMangaPipeline,
    toggleMangaCompare,
    revertMangaToOriginal,
    saveMangaToGallery,
    downloadMangaPng,
    onMangaFileUpload,
    loadMangaSource,
  });
})();

// ============================================================
// --- UI HELPERS FOR INDEX.HTML INLINE CALLS ---
// ============================================================
Object.assign(App, {
  onModelsTabSearch(query) {
    this.state.modelsTabSearch = query || "";
    this.renderModelsTab();
  },

  syncModelsTabFilterButtons(btnEl = null) {
    const filter = this.state.modelsTabFilter || "all";
    document.querySelectorAll("#models-tab-filters .filter-chip, #models-filters .filter-chip").forEach(chip => {
      const value = chip.dataset.tabFilter || chip.dataset.filter || "";
      chip.classList.toggle("active", btnEl ? chip === btnEl : value === filter);
    });
  },

  setModelsTabFilter(filter, btnEl = null) {
    this.state.modelsTabFilter = filter || "all";
    this.syncModelsTabFilterButtons(btnEl);
    this.renderModelsTab();
  },

  switchModelsHubSubtab(subtab) {
    const normalized = subtab === "loras" ? "loras" : "models";
    this.state.modelsHubSubtab = normalized;
    const isLoras = normalized === "loras";

    document.getElementById("models-subtab-btn-models")?.classList.toggle("active", !isLoras);
    document.getElementById("models-subtab-btn-loras")?.classList.toggle("active", isLoras);

    const modelsPanel = document.getElementById("models-subpanel-models");
    const lorasPanel = document.getElementById("models-subpanel-loras");
    if (modelsPanel) modelsPanel.style.display = isLoras ? "none" : "block";
    if (lorasPanel) lorasPanel.style.display = isLoras ? "block" : "none";

    const addModelBtn = document.getElementById("models-hub-add-model-btn");
    const addLoraBtn = document.getElementById("models-hub-add-lora-btn");
    if (addModelBtn) addModelBtn.style.display = isLoras ? "none" : "inline-flex";
    if (addLoraBtn) addLoraBtn.style.display = isLoras ? "inline-flex" : "none";

    if (isLoras) {
      this.loadLoras();
    } else {
      this.renderModelsTab();
    }
  },

  goToModelInHub(modelId) {
    this.switchTab("models");
    this.state.modelsTabFilter = "all";
    this.state.modelsTabSearch = modelId || "";
    this.syncModelsTabFilterButtons();
    const searchInput = document.getElementById("models-tab-search");
    if (searchInput) searchInput.value = modelId || "";
    this.renderModelsTab();
  },

  async scanLocalModelPath() {
    const input = document.getElementById("custom-model-scan-path");
    const path = (input?.value || "").trim();
    if (!path) {
      this.toast("warning", t("custom_model.scan_path_required"));
      return;
    }
    try {
      this.toast("info", t("custom_model.scanning"));
      const res = await this.apiPost("/api/custom-models/sniff", { path });
      this.populateCustomFileSelects(res);
      this.toast("success", t("custom_model.scan_found", {
        arch: res.detected_architecture || t("custom_model.scan_model"),
      }));
    } catch (e) {
      this.toast("error", t("custom_model.scan_failed") + e.message);
    }
  },

  filterLoraLibrary(query) {
    this.state.loraLibrarySearch = (query || "").toLowerCase();
    this.renderLoraLibrary();
  },

  switchLoraModalTab(tab) {
    const normalized = (tab === "hf" || tab === "active" || tab === "browse" || tab === "import")
      ? (tab === "import" ? "local" : "hub")
      : (tab || "hub");
    const tabs = ["hub", "local", "library"];

    tabs.forEach(t => {
      document.getElementById(`lora-tab-btn-${t}`)?.classList.toggle("active", t === normalized);
      const pane = document.getElementById(`lora-pane-${t}`);
      if (pane) pane.style.display = (t === normalized) ? "flex" : "none";
    });

    if (normalized === "library") this.loadLoras();
  },

  openLoraManageModal(tab = "hub") {
    const modal = document.getElementById("lora-manage-modal");
    if (modal) modal.style.display = "flex";
    this.switchLoraModalTab(tab);
    this.loadLoras();
  },

  async inspectHfLora() {
    const repoInput = document.getElementById("lora-hf-repo-input");
    const repo = (repoInput?.value || "").trim();
    if (!repo) {
      this.toast("warning", t("lora.hub.repo_required"));
      return;
    }
    try {
      this.toast("info", t("lora.hub.checking_repo"));
      const res = await this.apiGet(`/api/loras/hf-inspect?repo_id=${encodeURIComponent(repo)}`);
      const files = res.files || [];
      if (!files.length) {
        this.toast("warning", t("lora.hub.repo_files_missing"));
        return;
      }

      if (repoInput) repoInput.value = res.repo_id || repo;
      const inspector = document.getElementById("lora-hf-inspector");
      if (inspector) inspector.style.display = "flex";
      const title = document.getElementById("lora-hf-inspected-title");
      if (title) title.textContent = res.repo_id || repo;
      const badge = document.getElementById("lora-hf-base-badge");
      if (badge) badge.textContent = res.base_model || res.family || "LoRA";

      const fileSelect = document.getElementById("lora-hf-file-select");
      if (fileSelect) {
        fileSelect.innerHTML = files.map(f =>
          `<option value="${escapeHtml(f.path)}">${escapeHtml(f.name)}${f.size_mb ? ` · ${f.size_mb} MB` : ""}</option>`
        ).join("");
      }

      const familySelect = document.getElementById("lora-hf-family-select");
      if (familySelect) {
        const fam = res.suggested_family || res.family || "any";
        const hasOption = Array.from(familySelect.options).some(o => o.value === fam);
        familySelect.value = hasOption ? fam : "any";
      }

      const triggerInput = document.getElementById("lora-hf-trigger-input");
      if (triggerInput) triggerInput.value = res.suggested_trigger || res.trigger || "";
      const renameInput = document.getElementById("lora-hf-rename-input");
      if (renameInput) renameInput.value = "";

      this.toast("success", t("lora.hub.files_found", { count: files.length }));
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async searchHfLoras(query) {
    const q = (query || document.getElementById("lora-hf-search-input")?.value || "").trim();
    if (!q) {
      this.toast("warning", t("lora.hub.search_required"));
      return;
    }
    try {
      this.toast("info", t("lora.hub.searching"));
      const res = await this.apiGet(`/api/loras/hf-search?q=${encodeURIComponent(q)}`);
      this.renderHfLoraSearchResults(res.results || []);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  renderHfLoraSearchResults(results) {
    const box = document.getElementById("lora-hf-search-results");
    if (!box) return;

    if (!results.length) {
      box.style.display = "block";
      box.innerHTML = '<div style="padding: 14px; text-align: center; color: var(--text-dim);">Ничего не найдено</div>';
      return;
    }

    box.style.display = "block";
    box.innerHTML = results.map(item => `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 6px 8px; border-bottom: 1px solid var(--border);">
        <div style="min-width: 0;">
          <div style="font-size: 12px; font-weight: 600; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(item.repo_id)}</div>
          <div style="font-size: 10px; color: var(--text-dim);">⬇ ${item.downloads || 0} · ❤ ${item.likes || 0}${item.base_model ? ` · ${escapeHtml(item.base_model)}` : ""}</div>
        </div>
        <button type="button" class="dock-mini-btn" onclick="App.selectHfLoraResult('${escapeHtml(item.repo_id)}')">Выбрать</button>
      </div>
    `).join("");
  },

  selectHfLoraResult(repoId) {
    const repoInput = document.getElementById("lora-hf-repo-input");
    if (repoInput) repoInput.value = repoId;
    this.inspectHfLora();
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
      this.toast("info", checked ? t("settings.dock.hidden_toast") : t("settings.dock.shown_toast"));
    } catch (e) {
      this.toast("error", e.message || t("settings.save_error"));
    }
  },

  // ============================================================
  // MASK EDITING (Generate) + INPAINT STUDIO
  // Mirror of frontend/js/modules/mask-core.js, generate-mask.js and inpaint.js
  // ============================================================

  maskT(key, fallback, vars) {
    const value = t(key, vars || {});
    if (value && value !== key) return value;
    return fallback || key;
  },

  // ---------- Mask core ----------
  pointerToCanvasPoint(canvas, event) {
    if (!canvas || !event) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    return { x, y };
  },

  syncOverlayGeometry(canvas, targetEl) {
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
  },

  createMaskSurface(dataCanvas, options = {}) {
    if (!dataCanvas) return null;

    const visualCanvas = options.visualCanvas || null;
    const ctx = dataCanvas.getContext("2d");
    const vctx = visualCanvas ? visualCanvas.getContext("2d") : null;

    const clamp = (value, min, max) => {
      const v = Number(value);
      if (!Number.isFinite(v)) return min;
      return Math.min(max, Math.max(min, v));
    };

    const st = {
      brushSize: clamp(options.brushSize ?? 64, 1, 4096),
      softness: clamp(options.softness ?? 0.5, 0, 1),
      mode: options.mode === "eraser" ? "eraser" : "brush",
      color: options.color || "rgba(239, 68, 68, 0.45)",
      eraseColor: options.eraseColor || "rgba(56, 189, 248, 0.45)",
      maxHistory: clamp(options.maxHistory ?? 24, 1, 100),
    };

    const undoStack = [];
    const redoStack = [];
    let drawing = false;
    let lastPoint = null;
    let dirty = false;
    let visualQueued = false;
    let scratch = null;

    const renderVisual = () => {
      if (!vctx || !visualCanvas) return;
      vctx.clearRect(0, 0, visualCanvas.width, visualCanvas.height);
      vctx.drawImage(dataCanvas, 0, 0, visualCanvas.width, visualCanvas.height);
      vctx.globalCompositeOperation = "source-in";
      vctx.fillStyle = st.mode === "eraser" ? st.eraseColor : st.color;
      vctx.fillRect(0, 0, visualCanvas.width, visualCanvas.height);
      vctx.globalCompositeOperation = "source-over";
    };

    const scheduleVisual = () => {
      if (!vctx || visualQueued) return;
      visualQueued = true;
      requestAnimationFrame(() => {
        visualQueued = false;
        renderVisual();
      });
    };

    const snapshot = () => {
      try {
        return dataCanvas.toDataURL("image/png");
      } catch (_) {
        return "";
      }
    };

    const pushHistory = () => {
      const snap = snapshot();
      if (!snap) return;
      undoStack.push(snap);
      while (undoStack.length > st.maxHistory) undoStack.shift();
      redoStack.length = 0;
    };

    const restoreFromDataURL = (url) => {
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
    };

    const undo = () => {
      if (!undoStack.length) return Promise.resolve(false);
      redoStack.push(snapshot());
      const prev = undoStack.pop();
      return restoreFromDataURL(prev).then((ok) => {
        if (!ok) undoStack.push(prev);
        return ok;
      });
    };

    const redo = () => {
      if (!redoStack.length) return Promise.resolve(false);
      undoStack.push(snapshot());
      const next = redoStack.pop();
      return restoreFromDataURL(next).then((ok) => {
        if (!ok) redoStack.push(next);
        return ok;
      });
    };

    const paintDot = (x, y, erase) => {
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
    };

    const strokeTo = (x, y) => {
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
    };

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

      clear(record = true) {
        if (record) pushHistory();
        ctx.clearRect(0, 0, dataCanvas.width, dataCanvas.height);
        dirty = true;
        renderVisual();
      },

      invert(record = true) {
        if (record) pushHistory();
        const w = dataCanvas.width;
        const h = dataCanvas.height;
        if (!scratch) scratch = document.createElement("canvas");
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

      continueStroke(x, y) {
        if (!drawing) return;
        strokeTo(x, y);
        dirty = true;
        scheduleVisual();
      },

      endStroke() {
        if (!drawing) return;
        drawing = false;
        lastPoint = null;
        renderVisual();
      },

      undo,
      redo,

      toDataURL() {
        return snapshot();
      },

      loadDataURL(url) {
        return restoreFromDataURL(url).then((ok) => {
          dirty = ok;
          renderVisual();
          return ok;
        });
      },

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
  },

  // ---------- Generate: mask state ----------
  getGenMaskState() {
    if (!this.state.genMask) {
      this.state.genMask = {
        enabled: false,
        surface: null,
        source: null,
        brushSize: 64,
        softness: 0.5,
        mode: "brush",
        bound: false,
        sourceLoadSeq: 0,
      };
    }
    return this.state.genMask;
  },

  getGenMaskCanvas() {
    return document.getElementById("gen-mask-canvas");
  },

  getGenMaskPreviewImage() {
    return document.getElementById("preview-img");
  },

  ensureGenMaskSurface() {
    const state = this.getGenMaskState();
    const canvas = this.getGenMaskCanvas();
    if (!state || !canvas) return null;
    if (!state.surface) {
      const dataCanvas = document.createElement("canvas");
      dataCanvas.width = 1024;
      dataCanvas.height = 1024;
      state.surface = this.createMaskSurface(dataCanvas, { visualCanvas: canvas });
      state.surface.brushSize = state.brushSize;
      state.surface.softness = state.softness;
      state.surface.mode = state.mode;
    }
    return state.surface;
  },

  syncGenMaskGeometry() {
    const state = this.getGenMaskState();
    const canvas = this.getGenMaskCanvas();
    const img = this.getGenMaskPreviewImage();
    if (!state || !canvas || !img) return;
    this.syncOverlayGeometry(canvas, img);
  },

  updateGenMaskPanel() {
    const state = this.getGenMaskState();
    if (!state) return;

    const panelEl = document.getElementById("gen-mask-panel");
    const toggleBtn = document.getElementById("gen-mask-toggle");
    const hintEl = document.getElementById("gen-mask-hint");
    const sourceEl = document.getElementById("gen-mask-source-name");
    const statusEl = document.getElementById("gen-mask-status");
    const canvas = this.getGenMaskCanvas();

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
    if (sourceEl) sourceEl.textContent = state.source?.name || this.maskT("generate.mask.no_source", "нет изображения");

    if (statusEl) {
      if (!hasSource) {
        statusEl.textContent = this.maskT("generate.mask.hint_load", "Загрузите изображение, чтобы рисовать маску");
      } else if (!state.surface || state.surface.isEmpty()) {
        statusEl.textContent = this.maskT("generate.mask.hint_empty", "Маска пуста — нарисуйте область для изменения");
      } else {
        statusEl.textContent = `${this.maskT("generate.mask.coverage", "Покрытие")}: ${Math.round(state.surface.coverage() * 100)}%`;
      }
    }

    const undoBtn = document.getElementById("gen-mask-undo");
    const redoBtn = document.getElementById("gen-mask-redo");
    if (undoBtn) undoBtn.disabled = !state.surface || !state.surface.canUndo;
    if (redoBtn) redoBtn.disabled = !state.surface || !state.surface.canRedo;
  },

  setGenMaskSource(source) {
    const state = this.getGenMaskState();
    if (!state || !source) return;
    const sourceLoadSeq = (state.sourceLoadSeq || 0) + 1;
    state.sourceLoadSeq = sourceLoadSeq;

    const img = this.getGenMaskPreviewImage();
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
      const surface = this.ensureGenMaskSurface();
      if (changed && surface) {
        surface.reset(state.source.width, state.source.height);
        this.syncGenMaskGeometry();
        this.updateGenMaskPanel();
      } else {
        this.resizeGenMaskToSource();
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

    this.updateGenMaskPanel();
    this.syncGenMaskGeometry();
  },

  clearGenMaskSource() {
    const state = this.getGenMaskState();
    if (!state) return;
    state.sourceLoadSeq = (state.sourceLoadSeq || 0) + 1;
    state.source = null;
    if (state.surface) state.surface.reset(1024, 1024);
    const img = this.getGenMaskPreviewImage();
    const placeholder = document.getElementById("preview-placeholder");
    const wrap = document.getElementById("preview-image-wrap");
    if (img) img.removeAttribute("src");
    if (wrap) wrap.style.display = "none";
    if (placeholder) placeholder.style.display = "flex";
    this.updateGenMaskPanel();
    this.syncGenMaskGeometry();
  },

  resizeGenMaskToSource() {
    const state = this.getGenMaskState();
    const surface = this.ensureGenMaskSurface();
    const img = this.getGenMaskPreviewImage();
    if (!state || !surface || !img) return;
    const width = img.naturalWidth || state.source?.width || 1024;
    const height = img.naturalHeight || state.source?.height || 1024;
    surface.resize(width, height);
    if (state.source) {
      state.source.width = width;
      state.source.height = height;
    }
    this.syncGenMaskGeometry();
    this.updateGenMaskPanel();
  },

  enableGenMaskWithSource(appRef, source) {
    const app = appRef || this;
    const state = app.getGenMaskState();
    if (!state) return;
    app.setGenMaskSource(source);
    if (source?.path && source?.url) {
      app.setImg2ImgRef({
        path: source.path,
        url: source.url,
        filename: source.name || source.path,
      });
    }
    state.enabled = true;
    app.resizeGenMaskToSource();
    app.updateGenMaskPanel();
    app.syncGenMaskGeometry();
  },

  toggleGenMask() {
    const state = this.getGenMaskState();
    if (!state) return;

    if (state.enabled) {
      state.enabled = false;
      const img = this.getGenMaskPreviewImage();
      const placeholder = document.getElementById("preview-placeholder");
      const wrap = document.getElementById("preview-image-wrap");
      if (this.state.currentPreviewUrl) {
        if (img) img.src = this.state.currentPreviewUrl;
        if (wrap) wrap.style.display = "flex";
        if (placeholder) placeholder.style.display = "none";
      } else {
        if (img) img.removeAttribute("src");
        if (wrap) wrap.style.display = "none";
        if (placeholder) placeholder.style.display = "flex";
      }
      this.updateGenMaskPanel();
      return;
    }

    const reference = this.state.img2imgRef;
    const currentUrl = reference?.url || this.state.currentPreviewUrl;
    const currentName = reference?.path || this.state.currentPreviewFilename;
    if (!currentUrl || !currentName) {
      state.enabled = true;
      this.updateGenMaskPanel();
      this.toast("warning", this.maskT("generate.mask.hint_load", "Загрузите изображение, чтобы рисовать маску"));
      return;
    }

    this.setGenMaskSource({
      url: currentUrl,
      path: currentName,
      name: reference?.filename || currentName,
    });
    state.enabled = true;
    this.updateGenMaskPanel();
    this.syncGenMaskGeometry();
  },

  setGenMaskTool(mode) {
    const state = this.getGenMaskState();
    if (!state) return;
    state.mode = mode === "eraser" ? "eraser" : "brush";
    if (state.surface) state.surface.mode = state.mode;
    document.getElementById("gen-mask-tool-brush")?.classList.toggle("active", state.mode === "brush");
    document.getElementById("gen-mask-tool-eraser")?.classList.toggle("active", state.mode === "eraser");
    this.updateGenMaskPanel();
  },

  setGenMaskBrushSize(value) {
    const state = this.getGenMaskState();
    if (!state) return;
    state.brushSize = Math.max(1, Math.min(4096, Number(value) || 64));
    if (state.surface) state.surface.brushSize = state.brushSize;
    const label = document.getElementById("gen-mask-size-val");
    if (label) label.textContent = String(Math.round(state.brushSize));
  },

  setGenMaskSoftness(value) {
    const state = this.getGenMaskState();
    if (!state) return;
    state.softness = Math.max(0, Math.min(1, Number(value)));
    if (state.surface) state.surface.softness = state.softness;
    const label = document.getElementById("gen-mask-softness-val");
    if (label) label.textContent = `${Math.round(state.softness * 100)}%`;
  },

  undoGenMask() {
    const state = this.getGenMaskState();
    if (!state || !state.surface) return;
    state.surface.undo().then(() => this.updateGenMaskPanel());
  },

  redoGenMask() {
    const state = this.getGenMaskState();
    if (!state || !state.surface) return;
    state.surface.redo().then(() => this.updateGenMaskPanel());
  },

  clearGenMask() {
    const state = this.getGenMaskState();
    if (!state || !state.surface) return;
    state.surface.clear();
    this.updateGenMaskPanel();
  },

  invertGenMask() {
    const state = this.getGenMaskState();
    if (!state || !state.surface) return;
    state.surface.invert();
    this.updateGenMaskPanel();
  },

  async uploadGenMaskFile(file) {
    if (!file) return;
    try {
      const data = await this.apiUpload(file);
      this.setGenMaskSource({
        url: data.url,
        path: data.path,
        name: file.name || data.path,
      });
      this.toast("success", this.maskT("generate.mask.loaded", "Изображение загружено"));
    } catch (e) {
      this.toast("error", e.message || "Ошибка загрузки");
    }
  },

  useCanvasImageForMask() {
    if (!this.state.currentPreviewUrl || !this.state.currentPreviewFilename) {
      this.toast("warning", this.maskT("generate.mask.hint_load", "Загрузите изображение, чтобы рисовать маску"));
      return;
    }
    this.setGenMaskSource({
      url: this.state.currentPreviewUrl,
      path: this.state.currentPreviewFilename,
      name: this.state.currentPreviewFilename,
    });
  },

  getMaskCapableModels() {
    return (this.state.models || []).filter((m) => m.supports_mask);
  },

  applyGenMaskParams(params) {
    const state = this.getGenMaskState();
    if (!state) return "none";
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
  },

  initGenMaskUI() {
    const state = this.getGenMaskState();
    if (!state) return;
    if (state.bound) {
      this.updateGenMaskPanel();
      return;
    }
    state.bound = true;

    const canvas = this.getGenMaskCanvas();
    if (canvas) {
      let painting = false;

      canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        const st = this.getGenMaskState();
        if (!st?.enabled) return;
        const surface = this.ensureGenMaskSurface();
        if (!surface) return;
        const point = this.pointerToCanvasPoint(canvas, event);
        if (!point) return;
        event.preventDefault();
        event.stopPropagation();
        painting = true;
        surface.mode = st.mode;
        surface.beginStroke(point.x, point.y);
        try { canvas.setPointerCapture(event.pointerId); } catch (_) {}
        this.updateGenMaskPanel();
      });

      canvas.addEventListener("pointermove", (event) => {
        const st = this.getGenMaskState();
        if (!st?.enabled) return;
        const point = this.pointerToCanvasPoint(canvas, event);
        if (painting && point && st.surface) {
          st.surface.continueStroke(point.x, point.y);
        }
      });

      const end = () => {
        if (!painting) return;
        painting = false;
        const st = this.getGenMaskState();
        if (st?.surface) st.surface.endStroke();
        this.updateGenMaskPanel();
      };
      canvas.addEventListener("pointerup", end);
      canvas.addEventListener("pointercancel", end);
      canvas.addEventListener("pointerleave", end);
    }

    const img = this.getGenMaskPreviewImage();
    if (img && typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(() => this.syncGenMaskGeometry());
      observer.observe(img);
      state.maskObserver = observer;
    }
    window.addEventListener("resize", () => this.syncGenMaskGeometry());

    document.getElementById("gen-mask-tool-brush")?.addEventListener("click", () => this.setGenMaskTool("brush"));
    document.getElementById("gen-mask-tool-eraser")?.addEventListener("click", () => this.setGenMaskTool("eraser"));

    const fileInput = document.getElementById("gen-mask-file-input");
    fileInput?.addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) this.uploadGenMaskFile(file);
      event.target.value = "";
    });

    this.updateGenMaskPanel();
  },

  // ---------- Inpaint studio ----------
  getInpaintState() {
    if (!this.state.inpaint) {
      this.state.inpaint = {
        source: null,
        layers: [],
        activeLayerId: null,
        history: [],
        future: [],
        brushSize: 64,
        softness: 0.5,
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
    return this.state.inpaint;
  },

  getInpaintCompositeCanvas() {
    return document.getElementById("inpaint-composite");
  },

  getInpaintMaskStack() {
    return document.getElementById("inpaint-mask-stack");
  },

  getInpaintActiveLayer() {
    const state = this.getInpaintState();
    if (!state) return null;
    return state.layers.find((l) => l.id === state.activeLayerId) || null;
  },

  getInpaintActiveMaskLayer() {
    const layer = this.getInpaintActiveLayer();
    return layer && layer.type === "mask" ? layer : null;
  },

  getInpaintLayerById(id) {
    const state = this.getInpaintState();
    if (!state) return null;
    return state.layers.find((l) => l.id === id) || null;
  },

  createInpaintMaskLayer(state, name) {
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

    const surface = this.createMaskSurface(dataCanvas, { visualCanvas: viewCanvas });

    const layer = {
      id: `mask_${Date.now().toString(36)}_${(this._inpaintLayerSeq = (this._inpaintLayerSeq || 0) + 1).toString(36)}`,
      type: "mask",
      name: name || this.maskT("inpaint.layer.mask", "Маска"),
      visible: true,
      opacity: 1,
      locked: false,
      surface,
      dataCanvas,
      viewCanvas,
      // Bumped on every mask mutation so the cached mask bbox stays valid.
      revision: 0,
      maskRev: this._inpaintLayerSeq || 0,
    };
    viewCanvas.dataset.layerId = layer.id;
    return layer;
  },

  attachInpaintMaskLayerView(layer) {
    const stack = this.getInpaintMaskStack();
    if (!stack || !layer) return;
    if (layer.viewCanvas && layer.viewCanvas.parentElement !== stack) {
      stack.appendChild(layer.viewCanvas);
    }
    layer.viewCanvas.style.display = "none";
    layer.viewCanvas.style.pointerEvents = "none";
  },

  refreshInpaintMaskLayerViews() {
    const state = this.getInpaintState();
    if (!state) return;
    for (const layer of state.layers) {
      if (layer.type !== "mask" || !layer.viewCanvas) continue;
      const isActive = layer.id === state.activeLayerId;
      layer.viewCanvas.style.display = isActive && layer.visible ? "block" : "none";
      layer.viewCanvas.style.pointerEvents = isActive && layer.visible ? "auto" : "none";
      layer.viewCanvas.style.opacity = String(layer.opacity ?? 1);
    }
  },

  renderInpaintComposite() {
    const state = this.getInpaintState();
    const canvas = this.getInpaintCompositeCanvas();
    if (!state || !canvas) return;

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
        this.drawInpaintResultLayer(ctx, layer, width, height);
      }
    }
  },

  drawInpaintResultLayer(ctx, layer, width, height) {
    if (!layer.image) return;
    const tmp = this.getInpaintScratchCanvas(width, height);
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
  },

  getInpaintScratchCanvas(width, height) {
    if (!this._inpaintScratch) this._inpaintScratch = document.createElement("canvas");
    const scratch = this._inpaintScratch;
    if (scratch.width !== width || scratch.height !== height) {
      scratch.width = width;
      scratch.height = height;
    }
    return scratch;
  },

  captureInpaintSnapshot(state) {
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
  },

  pushInpaintHistory(state, snapshot = null) {
    state.history.push(snapshot || this.captureInpaintSnapshot(state));
    while (state.history.length > 12) state.history.shift();
    state.future.length = 0;
  },

  async restoreInpaintSnapshot(snapshot) {
    const state = this.getInpaintState();
    if (!state || !snapshot) return;

    const stack = this.getInpaintMaskStack();
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
        const layer = this.createInpaintMaskLayer(state, saved.name);
        layer.id = saved.id;
        layer.visible = saved.visible;
        layer.opacity = saved.opacity;
        if (saved.maskDataUrl) await layer.surface.loadDataURL(saved.maskDataUrl);
        this.attachInpaintMaskLayerView(layer);
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
    if (!this.getInpaintLayerById(state.activeLayerId)) {
      const fallback = state.layers.find((l) => l.type === "mask") || state.layers[0] || null;
      state.activeLayerId = fallback ? fallback.id : null;
    }
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
  },

  undoInpaint() {
    const state = this.getInpaintState();
    if (!state || !state.history.length) return;
    const current = this.captureInpaintSnapshot(state);
    const previous = state.history.pop();
    state.future.push(current);
    this.restoreInpaintSnapshot(previous);
  },

  redoInpaint() {
    const state = this.getInpaintState();
    if (!state || !state.future.length) return;
    const current = this.captureInpaintSnapshot(state);
    const next = state.future.pop();
    state.history.push(current);
    this.restoreInpaintSnapshot(next);
  },

  async loadInpaintSource(url, path, name, appRef) {
    const app = appRef || this;
    const state = app.getInpaintState();
    if (!state) return false;

    const loadSeq = (state.sourceLoadSeq || 0) + 1;
    state.sourceLoadSeq = loadSeq;
    const img = new Image();
    img.crossOrigin = "anonymous";
    try {
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error("Не удалось загрузить изображение"));
        img.src = url;
      });
    } catch (err) {
      app.toast("error", err.message || "Не удалось загрузить изображение");
      return false;
    }

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
      id: `source_${Date.now().toString(36)}_1`,
      type: "source",
      name: this.maskT("inpaint.layer.source", "Исходник"),
      visible: true,
      opacity: 1,
      locked: true,
      image: img,
      url,
    };
    state.layers.push(sourceLayer);
    state.activeLayerId = sourceLayer.id;

    const stack = this.getInpaintMaskStack();
    if (stack) stack.innerHTML = "";

    const maskLayer = this.createInpaintMaskLayer(state);
    this.attachInpaintMaskLayerView(maskLayer);
    state.layers.push(maskLayer);
    state.activeLayerId = maskLayer.id;

    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
    // A new source invalidates the previously selected generation area.
    this.resetInpaintCropSelection();
    this.updateInpaintStatus();
    this.syncInpaintGeometry();
    return true;
  },

  clearInpaintSource() {
    const state = this.getInpaintState();
    if (!state) return;
    state.sourceLoadSeq = (state.sourceLoadSeq || 0) + 1;
    state.source = null;
    state.layers = [];
    state.activeLayerId = null;
    state.history.length = 0;
    state.future.length = 0;
    state.jobContext = null;

    const stack = this.getInpaintMaskStack();
    if (stack) stack.replaceChildren();
    const canvas = this.getInpaintCompositeCanvas();
    if (canvas) {
      canvas.width = 1024;
      canvas.height = 1024;
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    }

    this.renderInpaintLayersUI();
    this.resetInpaintCropSelection();
    this.updateInpaintStatus();
    this.syncInpaintGeometry();
  },

  syncInpaintGeometry() {
    const state = this.getInpaintState();
    const canvas = this.getInpaintCompositeCanvas();
    if (!state || !canvas || !state.source) return;
    if (canvas.width !== state.source.width) canvas.width = state.source.width;
    if (canvas.height !== state.source.height) canvas.height = state.source.height;
    const overlay = this.getInpaintCropOverlay();
    if (overlay) {
      if (overlay.width !== state.source.width) overlay.width = state.source.width;
      if (overlay.height !== state.source.height) overlay.height = state.source.height;
    }
    for (const layer of state.layers) {
      if (layer.type !== "mask") continue;
      if (layer.viewCanvas.width !== state.source.width) layer.viewCanvas.width = state.source.width;
      if (layer.viewCanvas.height !== state.source.height) layer.viewCanvas.height = state.source.height;
    }
  },

  renderInpaintLayersUI() {
    const state = this.getInpaintState();
    const list = document.getElementById("inpaint-layers-list");
    if (!state || !list) return;

    const layer = this.getInpaintActiveLayer();
    const opacityInput = document.getElementById("inpaint-layer-opacity");
    const opacityNum = document.getElementById("inpaint-layer-opacity-num");
    const opacity = layer ? Number(layer.opacity ?? 1) : 1;
    if (opacityInput) opacityInput.value = String(opacity);
    if (opacityNum) opacityNum.textContent = opacity.toFixed(2);

    const deleteBtn = document.getElementById("inpaint-layer-delete");
    if (deleteBtn) deleteBtn.disabled = !layer || layer.type === "source";

    const ordered = [...state.layers].reverse();

    list.innerHTML = ordered.map((l) => {
      const isActive = l.id === state.activeLayerId;
      const badge = l.type === "source" ? "🔒" : l.type === "mask" ? "🎭" : "✨";
      const thumb = this.inpaintLayerThumb(l);
      return `
      <div class="inpaint-layer-item${isActive ? " active" : ""}" data-layer-id="${l.id}" onclick="App.selectInpaintLayer('${l.id}')">
        <span class="inpaint-layer-badge">${badge}</span>
        <span class="inpaint-layer-thumb">${thumb}</span>
        <span class="inpaint-layer-name" title="${escapeHtml(l.name)}">${escapeHtml(l.name)}</span>
        <button class="inpaint-layer-eye" title="${l.visible ? "Скрыть" : "Показать"}" onclick="event.stopPropagation(); App.toggleInpaintLayerVisible('${l.id}')">${l.visible ? "👁" : "🚫"}</button>
      </div>`;
    }).join("");

    if (!state.layers.length) {
      list.innerHTML = `<div class="inpaint-layers-empty">${this.maskT("inpaint.layers.empty", "Слоёв пока нет — загрузите изображение")}</div>`;
    }
  },

  inpaintLayerThumb(layer) {
    if (layer.type === "source") {
      return layer.url ? `<img src="${escapeHtml(layer.url)}" alt="" />` : "";
    }
    if (layer.type === "result") {
      return layer.url ? `<img src="${escapeHtml(layer.url)}" alt="" />` : "";
    }
    return `<span class="inpaint-layer-thumb-mask">🎭</span>`;
  },

  selectInpaintLayer(id) {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintLayerById(id);
    if (!layer) return;
    state.activeLayerId = id;
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintLayersUI();
    this.updateInpaintStatus();
  },

  addInpaintMaskLayer() {
    const state = this.getInpaintState();
    if (!state) return;
    if (!state.source) {
      this.toast("warning", this.maskT("inpaint.hint.no_source", "Сначала загрузите изображение"));
      return;
    }
    this.pushInpaintHistory(state);
    const layer = this.createInpaintMaskLayer(
      state,
      `${this.maskT("inpaint.layer.mask", "Маска")} ${state.layers.filter((l) => l.type === "mask").length + 1}`
    );
    this.attachInpaintMaskLayerView(layer);
    state.layers.push(layer);
    state.activeLayerId = layer.id;
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintLayersUI();
    this.updateInpaintStatus();
  },

  addInpaintResultLayer() {
    const state = this.getInpaintState();
    if (!state) return;
    if (!state.source) {
      this.toast("warning", this.maskT("inpaint.hint.no_source", "Сначала загрузите изображение"));
      return;
    }
    this.pushInpaintHistory(state);
    const layer = {
      id: `result_${Date.now().toString(36)}_${(this._inpaintLayerSeq = (this._inpaintLayerSeq || 0) + 1).toString(36)}`,
      type: "result",
      name: `${this.maskT("inpaint.layer.result", "Результат")} ${state.layers.filter((l) => l.type === "result").length + 1}`,
      visible: true,
      opacity: 1,
      image: null,
      maskImage: null,
      maskImageSrc: null,
      url: null,
    };
    state.layers.push(layer);
    state.activeLayerId = layer.id;
    this.renderInpaintLayersUI();
    this.updateInpaintStatus();
  },

  removeInpaintLayer(id) {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintLayerById(id);
    if (!layer) return;
    if (layer.type === "source") {
      this.toast("warning", this.maskT("inpaint.hint.source_locked", "Слой «Исходник» удалить нельзя"));
      return;
    }
    this.pushInpaintHistory(state);
    state.layers = state.layers.filter((l) => l.id !== id);
    if (layer.viewCanvas && layer.viewCanvas.parentElement) {
      layer.viewCanvas.parentElement.removeChild(layer.viewCanvas);
    }
    if (state.activeLayerId === id) {
      const fallback = state.layers.find((l) => l.type === "mask") || state.layers[state.layers.length - 1] || null;
      state.activeLayerId = fallback ? fallback.id : null;
    }
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
    this.updateInpaintStatus();
  },

  duplicateInpaintLayer(id) {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintLayerById(id);
    if (!layer) return;
    if (layer.type === "source") {
      this.toast("warning", this.maskT("inpaint.hint.source_locked", "Слой «Исходник» дублировать нельзя"));
      return;
    }
    this.pushInpaintHistory(state);
    const copy = { ...layer };
    copy.id = `${layer.type}_${Date.now().toString(36)}_${(this._inpaintLayerSeq = (this._inpaintLayerSeq || 0) + 1).toString(36)}`;
    copy.name = `${layer.name} ${this.maskT("inpaint.layer.copy_suffix", "(копия)")}`;
    if (layer.type === "mask") {
      copy.dataCanvas = document.createElement("canvas");
      copy.dataCanvas.width = layer.dataCanvas.width;
      copy.dataCanvas.height = layer.dataCanvas.height;
      copy.dataCanvas.getContext("2d").drawImage(layer.dataCanvas, 0, 0);
      copy.viewCanvas = document.createElement("canvas");
      copy.viewCanvas.width = layer.viewCanvas.width;
      copy.viewCanvas.height = layer.viewCanvas.height;
      copy.viewCanvas.className = "inpaint-mask-canvas";
      copy.surface = this.createMaskSurface(copy.dataCanvas, { visualCanvas: copy.viewCanvas });
      copy.surface.loadDataURL(layer.surface.toDataURL());
      this.attachInpaintMaskLayerView(copy);
    }
    const index = state.layers.findIndex((l) => l.id === id);
    state.layers.splice(index + 1, 0, copy);
    state.activeLayerId = copy.id;
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
  },

  moveInpaintLayer(id, direction) {
    const state = this.getInpaintState();
    if (!state) return;
    const index = state.layers.findIndex((l) => l.id === id);
    if (index < 0) return;
    const target = direction === "up" ? index + 1 : index - 1;
    if (target < 0 || target >= state.layers.length) return;
    this.pushInpaintHistory(state);
    const [item] = state.layers.splice(index, 1);
    state.layers.splice(target, 0, item);
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
  },

  toggleInpaintLayerVisible(id) {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintLayerById(id);
    if (!layer) return;
    this.pushInpaintHistory(state);
    layer.visible = !layer.visible;
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
  },

  renameInpaintLayer(id) {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintLayerById(id);
    if (!layer) return;
    const name = window.prompt(this.maskT("inpaint.layers.rename_prompt", "Название слоя"), layer.name);
    if (name === null) return;
    this.pushInpaintHistory(state);
    layer.name = String(name).slice(0, 80) || layer.name;
    this.renderInpaintLayersUI();
  },

  setInpaintLayerOpacity(id, value) {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintLayerById(id);
    if (!layer) return;
    layer.opacity = Math.max(0, Math.min(1, Number(value)));
    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    const num = document.getElementById("inpaint-layer-opacity-num");
    if (num) num.textContent = layer.opacity.toFixed(2);
  },

  applyInpaintPreset(key) {
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
    if (this.toast) this.toast("info", `Пресет: ${p.prompt.slice(0, 32)}...`);
  },
  setInpaintTool(mode) {
    const state = this.getInpaintState();
    if (!state) return;
    state.mode = mode === "eraser" ? "eraser" : "brush";
    const layer = this.getInpaintActiveMaskLayer();
    if (layer) layer.surface.mode = state.mode;
    document.getElementById("inpaint-tool-brush")?.classList.toggle("active", state.mode === "brush");
    document.getElementById("inpaint-tool-eraser")?.classList.toggle("active", state.mode === "eraser");
  },

  setInpaintBrushSize(value) {
    const state = this.getInpaintState();
    if (!state) return;
    state.brushSize = Math.max(1, Math.min(4096, Number(value) || 64));
    const label = document.getElementById("inpaint-brush-size-val");
    if (label) label.textContent = String(Math.round(state.brushSize));
    const layer = this.getInpaintActiveMaskLayer();
    if (layer) layer.surface.brushSize = state.brushSize;
    this.updateInpaintBrushCursor();
  },

  setInpaintSoftness(value) {
    const state = this.getInpaintState();
    if (!state) return;
    state.softness = Math.max(0, Math.min(1, Number(value)));
    const label = document.getElementById("inpaint-softness-val");
    if (label) label.textContent = `${Math.round(state.softness * 100)}%`;
    const layer = this.getInpaintActiveMaskLayer();
    if (layer) layer.surface.softness = state.softness;
    this.updateInpaintBrushCursor();
  },

  clearInpaintMask() {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintActiveMaskLayer();
    if (!layer) {
      this.toast("warning", this.maskT("inpaint.hint.select_mask", "Выберите слой маски"));
      return;
    }
    this.pushInpaintHistory(state);
    layer.surface.clear(false);
    layer.revision = (layer.revision || 0) + 1;
    this.updateInpaintStatus();
  },

  invertInpaintMask() {
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintActiveMaskLayer();
    if (!layer) {
      this.toast("warning", this.maskT("inpaint.hint.select_mask", "Выберите слой маски"));
      return;
    }
    this.pushInpaintHistory(state);
    layer.surface.invert(false);
    layer.revision = (layer.revision || 0) + 1;
    this.updateInpaintStatus();
  },

  // ---------------------------------------------------------------------------
  // Inpaint: generation area (crop modes)
  // ---------------------------------------------------------------------------

  getInpaintCropState() {
    const state = this.getInpaintState();
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
  },

  getInpaintCropOverlay() {
    return document.getElementById("inpaint-crop-overlay");
  },

  // Mirrors backend/masking.py :: resolve_manual_crop_box().
  // The rectangle is expressed in the pixel coordinates of the ORIGINAL image.
  resolveInpaintManualCropBox(box, width, height) {
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
  },

  // Bounding box of the painted mask, in source pixels.
  // The result is cached - the mask only changes when a stroke is finished.
  getInpaintMaskBBox() {
    const layer = this.getInpaintActiveMaskLayer();
    if (!layer || !layer.dataCanvas) return null;
    const crop = this.getInpaintCropState();
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
  },

  // Returns a human readable problem description for the current manual
  // selection, or null when the selection is valid.
  getInpaintCropProblem() {
    const state = this.getInpaintState();
    const crop = this.getInpaintCropState();
    if (!state || !crop || crop.mode !== "manual") return null;
    if (!crop.box) {
      return this.maskT(
        "inpaint.crop.error.no_box",
        "Select the generation area on the image first"
      );
    }
    const width = state.source ? state.source.width : 0;
    const height = state.source ? state.source.height : 0;
    const resolved = this.resolveInpaintManualCropBox(crop.box, width, height);
    if (!resolved) {
      return this.maskT(
        "inpaint.crop.error.too_small",
        "The area is too small (minimum 64px per side)"
      );
    }
    const bbox = this.getInpaintMaskBBox();
    if (
      bbox &&
      (bbox[0] < resolved[0] || bbox[1] < resolved[1] ||
        bbox[2] > resolved[2] || bbox[3] > resolved[3])
    ) {
      return this.maskT(
        "inpaint.crop.error.not_covered",
        "The selection does not cover the whole mask - enlarge it"
      );
    }
    return null;
  },

  setInpaintCropMode(mode) {
    const crop = this.getInpaintCropState();
    if (!crop) return;
    crop.mode = ["auto", "parameters", "manual"].includes(mode) ? mode : "auto";
    if (crop.mode !== "manual") {
      crop.selecting = false;
      crop.dragStart = null;
      crop.dragRect = null;
    }
    this.renderInpaintCropUI();
    this.renderInpaintCropOverlay();
    this.updateInpaintCropStatus();
  },

  setInpaintCropPadding(value) {
    const crop = this.getInpaintCropState();
    if (!crop) return;
    const num = parseInt(value, 10);
    crop.padding = Number.isFinite(num) ? Math.max(0, Math.min(1024, num)) : 64;
    this.updateInpaintCropStatus();
  },

  setInpaintCropMinSize(value) {
    const crop = this.getInpaintCropState();
    if (!crop) return;
    const num = parseInt(value, 10);
    crop.minSize = Number.isFinite(num) ? Math.max(64, Math.min(4096, num)) : 256;
    this.updateInpaintCropStatus();
  },

  toggleInpaintCropSelection() {
    const state = this.getInpaintState();
    const crop = this.getInpaintCropState();
    if (!state || !crop) return;
    if (!state.source) {
      this.toast(
        "warning",
        this.maskT("inpaint.hint.no_source", "Load an image first")
      );
      return;
    }
    if (crop.mode !== "manual") this.setInpaintCropMode("manual");
    crop.selecting = !crop.selecting;
    crop.dragStart = null;
    crop.dragRect = null;
    this.renderInpaintCropUI();
    this.renderInpaintCropOverlay();
    this.updateInpaintCropStatus();
  },

  resetInpaintCropSelection() {
    const crop = this.getInpaintCropState();
    if (!crop) return;
    crop.box = null;
    crop.selecting = false;
    crop.dragStart = null;
    crop.dragRect = null;
    this.renderInpaintCropUI();
    this.renderInpaintCropOverlay();
    this.updateInpaintCropStatus();
  },

  normalizeInpaintCropRect(a, b, width, height) {
    if (!a || !b) return null;
    const w = Math.max(1, Math.round(Number(width) || 0));
    const h = Math.max(1, Math.round(Number(height) || 0));
    const x0 = Math.max(0, Math.min(w, Math.min(a.x, b.x)));
    const y0 = Math.max(0, Math.min(h, Math.min(a.y, b.y)));
    const x1 = Math.max(0, Math.min(w, Math.max(a.x, b.x)));
    const y1 = Math.max(0, Math.min(h, Math.max(a.y, b.y)));
    return [x0, y0, x1, y1];
  },

  renderInpaintCropUI() {
    const crop = this.getInpaintCropState();
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
      desc.textContent = this.maskT(key, "");
    }

    const selectBtn = document.getElementById("inpaint-crop-select");
    if (selectBtn) {
      selectBtn.classList.toggle("active", !!crop.selecting);
      selectBtn.textContent = crop.selecting
        ? this.maskT("inpaint.crop.selecting", "Cancel selection")
        : this.maskT("inpaint.crop.select", "Select area");
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

    this.updateInpaintCropOverlayVisibility();
  },

  updateInpaintCropOverlayVisibility() {
    const overlay = this.getInpaintCropOverlay();
    if (!overlay) return;
    const state = this.getInpaintState();
    const crop = this.getInpaintCropState();
    if (!state || !crop) return;
    const visible = crop.mode === "manual" && !!state.source;
    overlay.style.display = visible ? "block" : "none";
    overlay.style.pointerEvents = crop.selecting && visible ? "auto" : "none";
    overlay.classList.toggle("selecting", !!crop.selecting && visible);
  },

  renderInpaintCropOverlay() {
    const overlay = this.getInpaintCropOverlay();
    if (!overlay) return;
    const state = this.getInpaintState();
    const crop = this.getInpaintCropState();
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

    // Dim everything outside of the selected generation area.
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.beginPath();
    ctx.rect(0, 0, width, height);
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.fill("evenodd");
    ctx.restore();

    const scale = Math.max(1, width / 1024);
    const problem = this.getInpaintCropProblem();
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
  },

  updateInpaintCropStatus() {
    const crop = this.getInpaintCropState();
    if (!crop) return;

    const coords = document.getElementById("inpaint-crop-coords");
    if (coords) {
      if (crop.mode !== "manual") {
        coords.textContent = "";
      } else if (!crop.box) {
        coords.textContent = this.maskT(
          "inpaint.crop.no_selection",
          "No area selected"
        );
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
      const problem = this.getInpaintCropProblem();
      warning.style.display = problem ? "block" : "none";
      if (problem) warning.textContent = problem;
    }
  },

  initInpaintCropEvents() {
    const overlay = this.getInpaintCropOverlay();
    if (!overlay) return;
    if (overlay.dataset.eventsBound === "1") return;
    overlay.dataset.eventsBound = "1";

    overlay.addEventListener("pointerdown", (event) => {
      const crop = this.getInpaintCropState();
      if (!crop || !crop.selecting || event.button !== 0) return;
      const state = this.getInpaintState();
      if (!state || !state.source) return;
      event.preventDefault();
      event.stopPropagation();
      const point = this.pointerToCanvasPoint(overlay, event);
      if (!point) return;
      crop.dragStart = point;
      crop.dragRect = null;
      try { overlay.setPointerCapture(event.pointerId); } catch (err) { /* ignore */ }
      this.renderInpaintCropOverlay();
    });

    overlay.addEventListener("pointermove", (event) => {
      const crop = this.getInpaintCropState();
      if (!crop || !crop.selecting || !crop.dragStart) return;
      const state = this.getInpaintState();
      const point = this.pointerToCanvasPoint(overlay, event);
      if (!point) return;
      crop.dragRect = this.normalizeInpaintCropRect(
        crop.dragStart,
        point,
        state && state.source ? state.source.width : 0,
        state && state.source ? state.source.height : 0
      );
      this.renderInpaintCropOverlay();
      this.updateInpaintCropStatus();
    });

    const finish = (event, discard) => {
      const crop = this.getInpaintCropState();
      if (!crop || !crop.selecting || !crop.dragStart) return;
      const state = this.getInpaintState();
      const point = this.pointerToCanvasPoint(overlay, event) || crop.dragStart;
      const rect = this.normalizeInpaintCropRect(
        crop.dragStart,
        point,
        state && state.source ? state.source.width : 0,
        state && state.source ? state.source.height : 0
      );
      crop.dragStart = null;
      crop.dragRect = null;
      crop.selecting = false;
      try { overlay.releasePointerCapture(event.pointerId); } catch (err) { /* ignore */ }
      if (!discard && rect && rect[2] - rect[0] >= 1 && rect[3] - rect[1] >= 1) {
        crop.box = rect;
      }
      this.renderInpaintCropUI();
      this.renderInpaintCropOverlay();
      this.updateInpaintCropStatus();
      const problem = this.getInpaintCropProblem();
      if (problem) this.toast("warning", problem);
    };

    overlay.addEventListener("pointerup", (event) => finish(event, false));
    overlay.addEventListener("pointercancel", (event) => finish(event, true));
  },

  updateInpaintStatus() {
    const state = this.getInpaintState();
    if (!state) return;
    const statusEl = document.getElementById("inpaint-status");
    const hintEl = document.getElementById("inpaint-empty-hint");
    const clearBtn = document.getElementById("inpaint-clear-source");
    const layer = this.getInpaintActiveMaskLayer();

    if (hintEl) hintEl.style.display = state.source ? "none" : "flex";
    if (clearBtn) clearBtn.disabled = !state.source;

    if (!statusEl) return;
    if (!state.source) {
      statusEl.textContent = this.maskT("inpaint.hint.no_source", "Сначала загрузите изображение");
      return;
    }
    if (!layer) {
      statusEl.textContent = this.maskT("inpaint.hint.select_mask", "Выберите слой маски");
      return;
    }
    const coverage = Math.round(layer.surface.coverage() * 100);
    statusEl.textContent = `${this.maskT("inpaint.status.coverage", "Покрытие маски")}: ${coverage}%`;
  },

  updateInpaintBrushCursor() {
    const cursor = document.getElementById("inpaint-brush-cursor");
    if (!cursor) return;
    const state = this.getInpaintState();
    if (!state) return;
    const layer = this.getInpaintActiveMaskLayer();
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
  },

  initInpaintMaskEvents() {
    const state = this.getInpaintState();
    const stack = this.getInpaintMaskStack();
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
      return id ? this.getInpaintLayerById(id) : null;
    };

    stack.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const layer = layerFromEvent(event);
      if (!layer || layer.type !== "mask") return;
      event.preventDefault();
      event.stopPropagation();
      const point = this.pointerToCanvasPoint(layer.viewCanvas, event);
      if (!point) return;
      painting = true;
      strokeSnapshot = this.captureInpaintSnapshot(state);
      strokeMaskData = layer.surface.toDataURL();
      layer.surface.mode = state.mode;
      layer.surface.beginStroke(point.x, point.y);
      try { layer.viewCanvas.setPointerCapture(event.pointerId); } catch (_) {}
    });

    stack.addEventListener("pointermove", (event) => {
      const layer = layerFromEvent(event);
      if (!layer) return;
      const point = this.pointerToCanvasPoint(layer.viewCanvas, event);
      if (point) layer.surface.continueStroke(point.x, point.y);
      this.updateInpaintBrushCursor();
      const cursor = document.getElementById("inpaint-brush-cursor");
      if (cursor) {
        cursor.style.display = "block";
        cursor.style.left = `${event.clientX}px`;
        cursor.style.top = `${event.clientY}px`;
      }
    });

    const endStroke = () => {
      if (!painting) return;
      painting = false;
      const layer = this.getInpaintActiveMaskLayer();
      if (layer) layer.surface.endStroke();
      if (layer) layer.revision = (layer.revision || 0) + 1;
      if (layer && strokeMaskData !== layer.surface.toDataURL()) {
        this.pushInpaintHistory(state, strokeSnapshot);
      }
      strokeSnapshot = null;
      strokeMaskData = null;
      this.updateInpaintStatus();
    };
    stack.addEventListener("pointerup", endStroke);
    stack.addEventListener("pointercancel", endStroke);
    stack.addEventListener("pointerleave", () => {
      const cursor = document.getElementById("inpaint-brush-cursor");
      if (cursor) cursor.style.display = "none";
    });
  },

  async generateInpaint() {
    const state = this.getInpaintState();
    if (!state) return;
    if (state.generating || state.applyingResult) {
      this.toast("warning", this.maskT("inpaint.hint.generating", "Генерация уже выполняется"));
      return;
    }

    if (!state.source || !state.source.path) {
      this.toast("warning", this.maskT("inpaint.hint.no_source", "Сначала загрузите изображение"));
      return;
    }

    const layer = this.getInpaintActiveMaskLayer();
    if (!layer || !layer.visible) {
      this.toast("warning", this.maskT("inpaint.hint.select_mask", "Выберите видимый слой маски"));
      return;
    }
    if (layer.surface.isEmpty()) {
      this.toast("warning", this.maskT("inpaint.hint.empty_mask", "Маска пуста — нарисуйте область для изменения"));
      return;
    }

    const prompt = document.getElementById("inpaint-prompt")?.value?.trim() || "";
    if (!prompt) {
      this.toast("warning", this.maskT("inpaint.hint.no_prompt", "Введите промпт"));
      return;
    }

    const modelId = document.getElementById("inpaint-model-select")?.value || this.state.selectedModel;
    const model = (this.state.models || []).find((m) => m.id === modelId);
    if (!model) {
      this.toast("warning", this.maskT("inpaint.hint.no_model", "Выберите модель"));
      return;
    }
    if (!model.supports_mask) {
      const compatible = (this.state.models || []).filter((m) => m.supports_mask);
      const names = compatible.map((m) => m.name).join(", ") || this.maskT("inpaint.hint.no_compatible", "нет в наличии");
      this.toast("error", this.maskT(
        "inpaint.hint.model_unsupported",
        `Модель «${model.name}» не поддерживает маску. Поддерживают: ${names}`,
        { name: model.name, list: names }
      ));
      return;
    }

    const selectedVariantId = document.getElementById("inpaint-variant-select")?.value || "";
    const selectedVariant = state.inpaintModelState?.transformers?.find((v) => v.id === selectedVariantId);
    const stepsInput = parseInt(document.getElementById("inpaint-steps")?.value || "", 10);
    const steps = Number.isFinite(stepsInput) ? stepsInput : (Number(selectedVariant?.steps) || 4);
    const cfg = parseFloat(document.getElementById("inpaint-cfg")?.value || "1.0");
    const seedRaw = document.getElementById("inpaint-seed")?.value?.trim();
    const seed = seedRaw === "" || seedRaw === "random" ? null : parseInt(seedRaw, 10);
    const teSelect = document.getElementById("inpaint-text-encoder-select");
    const customTeInput = document.getElementById("inpaint-custom-text-encoder-input");
    const customTeVal = customTeInput && customTeInput.style.display !== "none"
      ? customTeInput.value.trim()
      : "";
    const vaeSelect = document.getElementById("inpaint-vae-select");
    const customVaeInput = document.getElementById("inpaint-custom-vae-input");
    const customVaeVal = customVaeInput && customVaeInput.style.display !== "none"
      ? customVaeInput.value.trim()
      : "";

    // Generation area (crop mode).
    const crop = this.getInpaintCropState() || { mode: "auto", padding: 64, minSize: 256, box: null };
    if (crop.mode === "manual") {
      const problem = this.getInpaintCropProblem();
      if (problem) {
        this.toast("warning", problem);
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

    const params = {
      model_id: modelId,
      variant_id: selectedVariantId || undefined,
      prompt,
      width: state.source.width,
      height: state.source.height,
      steps: Number.isFinite(steps) ? steps : 4,
      cfg: Number.isFinite(cfg) ? cfg : 1.0,
      seed: Number.isFinite(seed) ? seed : null,
      sampler: document.getElementById("inpaint-sampler")?.value || "euler",
      init_image_path: state.source.path,
      init_image_strength: 1.0,
      mask: maskDataUrl,
      mask_strength: 1.0,
      mask_grow: 6,
      mask_blur: 0.0,
      mask_crop_mode: crop.mode || "auto",
      mask_crop_padding: Math.round(Number(crop.padding) || 0),
      mask_crop_min_size: Math.round(Number(crop.minSize) || 256),
      mask_crop_box: cropBox,
    };

    if (customTeVal) {
      params.custom_text_encoder_path = customTeVal;
    } else if (teSelect?.value && teSelect.value !== "custom") {
      params.text_encoder_id = teSelect.value;
    }

    if (customVaeVal) {
      params.custom_vae_path = customVaeVal;
    } else if (vaeSelect?.value && vaeSelect.value !== "auto" && vaeSelect.value !== "custom") {
      if (vaeSelect.value.startsWith("file:")) {
        params.custom_vae_path = vaeSelect.value.slice(5);
      } else {
        params.vae = vaeSelect.value;
      }
    }

    state.generating = true;
    const btn = document.getElementById("inpaint-generate-btn");
    if (btn) btn.disabled = true;

    try {
      const result = await this.apiPost("/api/generate", params);
      state.jobId = result.job_id;
      state.jobContext.jobId = result.job_id;
      this.startInpaintPolling(result.job_id);
    } catch (e) {
      state.generating = false;
      state.jobContext = null;
      if (btn) btn.disabled = false;
      const data = e?.data || {};
      const compatible = Array.isArray(data.compatible_models) ? data.compatible_models : [];
      if (compatible.length) {
        this.toast("error", `${e.message} ${this.maskT("inpaint.hint.switch_model", "Переключитесь на:")} ${compatible.map((m) => m.name).join(", ")}`);
      } else {
        this.toast("error", e.message || "Ошибка генерации");
      }
    }
  },

  startInpaintPolling(jobId) {
    const state = this.getInpaintState();
    if (!state) return;
    if (state.pollTimer) clearInterval(state.pollTimer);

    state.pollTimer = setInterval(async () => {
      try {
        const status = await this.apiGet(`/api/jobs/${jobId}`);
        const progressEl = document.getElementById("inpaint-progress");
        if (progressEl) {
          progressEl.style.display = status.status === "complete" ? "none" : "flex";
          progressEl.textContent = `${Math.round((status.progress || 0) * 100)}%`;
        }
        if (status.status === "complete") {
          this.stopInpaintPolling();
          await this.applyInpaintResult(status);
        } else if (status.status === "error") {
          this.stopInpaintPolling();
          state.jobContext = null;
          this.toast("error", status.error || status.message || this.maskT("inpaint.hint.generation_error", "Ошибка генерации"));
        } else if (status.status === "cancelled") {
          this.stopInpaintPolling();
          state.jobContext = null;
          this.toast("info", this.maskT("inpaint.hint.cancelled", "Генерация отменена"));
        }
      } catch (e) {
        this.stopInpaintPolling();
        state.jobContext = null;
        this.toast("error", e.message || "Ошибка опроса задания");
      }
    }, 400);
  },

  stopInpaintPolling() {
    const state = this.getInpaintState();
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
  },

  async applyInpaintResult(status) {
    const state = this.getInpaintState();
    if (!state) return;

    const context = state.jobContext;
    if (
      !context
      || state.jobId !== context.jobId
      || state.source?.token !== context.sourceToken
      || state.sourceLoadSeq !== context.sourceToken
    ) {
      state.jobContext = null;
      this.toast("warning", this.maskT("inpaint.hint.source_changed", "Источник изменился; результат не добавлен"));
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
      this.toast("warning", this.maskT("inpaint.hint.source_changed", "Источник изменился; результат не добавлен"));
      return;
    }
    this.pushInpaintHistory(state);

    const layer = {
      id: `result_${Date.now().toString(36)}_${(this._inpaintLayerSeq = (this._inpaintLayerSeq || 0) + 1).toString(36)}`,
      type: "result",
      name: `${this.maskT("inpaint.layer.result", "Результат")} ${state.layers.filter((l) => l.type === "result").length + 1}`,
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

    this.refreshInpaintMaskLayerViews();
    this.renderInpaintComposite();
    this.renderInpaintLayersUI();
    this.updateInpaintStatus();
    this.toast("success", this.maskT("inpaint.hint.done", "Слой результата создан"));
    } finally {
      state.applyingResult = false;
    }
  },

  renderInpaintModels() {
    const select = document.getElementById("inpaint-model-select");
    if (!select) return;

    const models = this.state.models || [];
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
    } else if (this.state.selectedModel) {
      select.value = this.state.selectedModel;
    }
    this.updateInpaintModelWarning();
  },

  updateInpaintModelWarning() {
    const select = document.getElementById("inpaint-model-select");
    const warnEl = document.getElementById("inpaint-model-warning");
    if (!select || !warnEl) return;
    const model = (this.state.models || []).find((m) => m.id === select.value);
    if (!model || model.supports_mask) {
      warnEl.style.display = "none";
      return;
    }
    const compatible = (this.state.models || []).filter((m) => m.supports_mask);
    warnEl.style.display = "flex";
    warnEl.innerHTML = `⚠ ${this.maskT("inpaint.hint.model_unsupported_short", "Модель не поддерживает маску")} — ${compatible.map((m) => escapeHtml(m.name)).join(", ") || "—"}`;
  },

  // Resolve the recommended steps/CFG for a variant.  Distilled variants
  // use the model card's "default" preset; undistilled "base" variants have
  // a separate preset (e.g. Flux.3 Klein Base needs 20 steps / CFG 3.5).
  resolveInpaintVariantDefaults(model, variant) {
    const presets = (model && model.scheduler_presets) || {};
    const isBase = /(^|[_-])base([_-]|$)/i.test(String(variant?.id || ""));
    const preset = (isBase && presets.base) || presets.default || {};
    const steps = Number(variant?.steps ?? preset.steps);
    const cfg = Number(preset.cfg ?? preset.guidance_scale ?? model?.default_cfg);
    return {
      steps: Number.isFinite(steps) && steps > 0 ? Math.round(steps) : null,
      cfg: Number.isFinite(cfg) && cfg > 0 ? cfg : null,
    };
  },

  // Inpaint uses a masked img2img run, so it must inherit the selected
  // model's step/CFG defaults.  The panel used to hardcode 4 steps / CFG 1.0,
  // which silently degraded 16-28 step models (Flux.2/3 Klein, Qwen-Image)
  // into blurry/unrelated patches.
  applyInpaintModelDefaults() {
    const select = document.getElementById("inpaint-model-select");
    if (!select) return;
    const model = (this.state.models || []).find((m) => m.id === select.value);
    if (!model) return;

    const state = this.getInpaintState();
    const loadedState = state?.inpaintModelState;
    const variants = Array.isArray(loadedState?.transformers) && loadedState.model_id === model.id
      ? loadedState.transformers
      : (Array.isArray(model.transformers) ? model.transformers : []);
    const variant = variants.find((v) => v.installed) || variants[0];
    const defaults = this.resolveInpaintVariantDefaults(model, variant);

    const stepsInput = document.getElementById("inpaint-steps");
    if (stepsInput && defaults.steps != null) {
      stepsInput.value = String(defaults.steps);
    }
    const cfgInput = document.getElementById("inpaint-cfg");
    if (cfgInput && defaults.cfg != null) {
      cfgInput.value = defaults.cfg.toFixed(1);
    }
  },

  // Mirror the Generate tab's quantization / text-encoder pickers.  The
  // inpaint request is handled by the same backend endpoint, so it accepts
  // variant_id, text_encoder_id and custom_text_encoder_path too.
  async loadInpaintModelState(modelId, preferredVariantId = null) {
    const state = this.getInpaintState();
    if (!state || !modelId) return;
    const requestId = (state._modelStateRequestId = (state._modelStateRequestId || 0) + 1);
    let modelState;
    try {
      modelState = await this.apiGet(`/api/models/${encodeURIComponent(modelId)}/state`);
    } catch (e) {
      console.warn("[inpaint] Failed to load model state:", e);
      return;
    }
    if (requestId !== state._modelStateRequestId) return;
    const select = document.getElementById("inpaint-model-select");
    if (!select || select.value !== modelId) return;

    state.inpaintModelState = modelState;
    this.renderInpaintVariantSelect(modelState, preferredVariantId);
    this.renderInpaintTextEncoderSelect(modelState);
    this.renderInpaintVaeSelect(modelState);
  },

  renderInpaintVariantSelect(modelState, preferredVariantId = null) {
    const select = document.getElementById("inpaint-variant-select");
    if (!select) return;
    const variants = Array.isArray(modelState?.transformers) ? modelState.transformers : [];
    select.innerHTML = "";
    if (!variants.length) {
      select.disabled = true;
      return;
    }
    select.disabled = false;

    // Recommended variants first, matching the Generate tab's grouping.
    const ordered = [
      ...variants.filter((v) => v.recommended),
      ...variants.filter((v) => !v.recommended),
    ];
    for (const v of ordered) {
      const opt = document.createElement("option");
      opt.value = v.id;
      const isInstalled = Boolean(v.installed) || v.state === "cached" || v.state === "installed";
      const mark = isInstalled ? "✓" : "⬇";
      const backend = v.backend || modelState.gpu_tier || "";
      opt.textContent = `${mark} ${v.id}${backend ? ` (${backend}, ${v.steps} steps)` : ` (${v.steps} steps)`}`;
      opt.title = isInstalled ? v.id : `${v.id} — ${v.state_label || "not installed"}`;
      // Prevent selecting files that are not on disk; the engine would fail
      // later with "diffusion model file not found".
      opt.disabled = !isInstalled;
      select.appendChild(opt);
    }

    const preferred = preferredVariantId ? variants.find((v) => v.id === preferredVariantId) : null;
    const installed = variants.find((v) => v.installed || v.state === "cached" || v.state === "installed");
    const preferredReady = preferred && (preferred.installed || preferred.state === "cached" || preferred.state === "installed");
    const active = preferredReady
      ? preferred
      : (installed || variants.find((v) => v.recommended) || variants[0]);
    if (active) {
      select.value = active.id;
      const model = (this.state.models || []).find((m) => m.id === modelState?.model_id);
      const defaults = this.resolveInpaintVariantDefaults(model, active);
      const stepsInput = document.getElementById("inpaint-steps");
      if (stepsInput && defaults.steps != null) {
        stepsInput.value = String(defaults.steps);
      }
      const cfgInput = document.getElementById("inpaint-cfg");
      if (cfgInput && defaults.cfg != null) {
        cfgInput.value = defaults.cfg.toFixed(1);
      }
    }
  },

  renderInpaintTextEncoderSelect(modelState) {
    const block = document.getElementById("inpaint-text-encoder-block");
    const select = document.getElementById("inpaint-text-encoder-select");
    const customInput = document.getElementById("inpaint-custom-text-encoder-input");
    if (!select) return;
    const encoders = Array.isArray(modelState?.text_encoders) ? modelState.text_encoders : [];
    select.innerHTML = "";
    if (!encoders.length) {
      select.disabled = true;
      if (block) block.style.display = "none";
      if (customInput) {
        customInput.style.display = "none";
        customInput.value = "";
      }
      return;
    }

    select.disabled = false;
    if (block) block.style.display = "flex";
    const ordered = [
      ...encoders.filter((te) => te.installed),
      ...encoders.filter((te) => !te.installed),
    ];
    for (const te of ordered) {
      const opt = document.createElement("option");
      opt.value = te.id;
      opt.textContent = `${te.installed ? "✓" : "⬇"} ${te.name || te.id}`;
      opt.title = te.file || te.id;
      opt.disabled = te.installed === false;
      select.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "Custom file / ComfyUI...";
    select.appendChild(customOpt);

    const active = ordered.find((te) => te.installed) || ordered.find((te) => te.recommended) || ordered[0];
    if (active) select.value = active.id;
    if (customInput) {
      customInput.style.display = select.value === "custom" ? "block" : "none";
      customInput.value = "";
    }
  },

  renderInpaintVaeSelect(modelState) {
    const block = document.getElementById("inpaint-vae-block");
    const select = document.getElementById("inpaint-vae-select");
    const customInput = document.getElementById("inpaint-custom-vae-input");
    if (!select) return;
    const model = (this.state.models || []).find((m) => m.id === modelState?.model_id);
    const vaeList = (Array.isArray(modelState?.vae_files) && modelState.vae_files.length)
      ? modelState.vae_files
      : (modelState?.vae ? [modelState.vae] : []);
    const isQwen = Boolean(model && (model.family === "qwen" || model.id.includes("qwen")));
    select.innerHTML = "";
    if (!isQwen && !vaeList.length) {
      select.disabled = true;
      if (block) block.style.display = "none";
      if (customInput) {
        customInput.style.display = "none";
        customInput.value = "";
      }
      return;
    }

    select.disabled = false;
    if (block) block.style.display = "flex";
    const autoOpt = document.createElement("option");
    autoOpt.value = "auto";
    autoOpt.textContent = t("controls.vae_auto") || "Auto (model default)";
    select.appendChild(autoOpt);
    if (isQwen) {
      for (const value of ["texture_fix", "official"]) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = t(`controls.vae_${value}`) || value;
        select.appendChild(opt);
      }
    }
    for (const vae of vaeList.filter((v) => v.installed && v.local_file)) {
      const fileName = String(vae.path || vae.local_file).split("/").pop() || "VAE";
      const opt = document.createElement("option");
      opt.value = `file:${vae.local_file}`;
      opt.textContent = fileName;
      opt.title = vae.local_file;
      select.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "Custom file / ComfyUI...";
    select.appendChild(customOpt);

    const desired = this.state.selectedVae
      || (isQwen ? (this.state.settings?.qwen_vae_type || "texture_fix") : "auto");
    select.value = [...select.options].some((opt) => opt.value === desired) ? desired : "auto";
    if (customInput) {
      customInput.style.display = "none";
      customInput.value = "";
    }
  },

  onInpaintVariantChange() {
    const state = this.getInpaintState();
    const select = document.getElementById("inpaint-variant-select");
    const variant = state?.inpaintModelState?.transformers?.find((v) => v.id === select?.value);
    if (!variant) return;
    const modelSelect = document.getElementById("inpaint-model-select");
    const model = (this.state.models || []).find((m) => m.id === modelSelect?.value);
    const defaults = this.resolveInpaintVariantDefaults(model, variant);
    const stepsInput = document.getElementById("inpaint-steps");
    if (stepsInput && defaults.steps != null) {
      stepsInput.value = String(defaults.steps);
    }
    const cfgInput = document.getElementById("inpaint-cfg");
    if (cfgInput && defaults.cfg != null) {
      cfgInput.value = defaults.cfg.toFixed(1);
    }
  },

  onInpaintTextEncoderChange() {
    const select = document.getElementById("inpaint-text-encoder-select");
    const customInput = document.getElementById("inpaint-custom-text-encoder-input");
    if (!select || !customInput) return;
    const isCustom = select.value === "custom";
    customInput.style.display = isCustom ? "block" : "none";
    if (isCustom) customInput.focus();
  },

  onInpaintVaeChange() {
    const select = document.getElementById("inpaint-vae-select");
    const customInput = document.getElementById("inpaint-custom-vae-input");
    if (!select || !customInput) return;
    const isCustom = select.value === "custom";
    customInput.style.display = isCustom ? "block" : "none";
    if (isCustom) customInput.focus();
  },

  useCanvasImageInInpaint() {
    const url = this.state.currentPreviewUrl;
    const path = this.state.currentPreviewFilename;
    if (!url || !path) {
      this.toast("warning", this.maskT("inpaint.hint.no_source", "Сначала загрузите изображение"));
      return;
    }
    this.loadInpaintSource(url, path, path);
  },

  async uploadInpaintFile(file) {
    if (!file) return;
    const allowed = ["image/png", "image/jpeg", "image/webp"];
    const extension = file.name?.split(".").pop()?.toLowerCase();
    if (!allowed.includes(file.type) && !["png", "jpg", "jpeg", "webp"].includes(extension)) {
      this.toast("error", "Поддерживаются только PNG, JPEG и WebP");
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      this.toast("error", "Размер файла не должен превышать 25 МБ");
      return;
    }
    try {
      const data = await this.apiUpload(file);
      const loaded = await this.loadInpaintSource(data.url, data.path, file.name || data.path);
      if (loaded) this.toast("success", this.maskT("inpaint.hint.loaded", "Изображение загружено"));
    } catch (e) {
      this.toast("error", e.message || "Ошибка загрузки");
    }
  },

  initInpaintStudio() {
    const state = this.getInpaintState();
    if (!state) return;

    this.initInpaintMaskEvents();
    this.initInpaintCropEvents();
    this.renderInpaintModels();
    const modelSelect = document.getElementById("inpaint-model-select");
    // Seed steps/CFG + quantization/text-encoder lists once per page load.
    // Later tab switches must keep whatever the user typed in the panel.
    if (!state._inpaintDefaultsApplied) {
      state._inpaintDefaultsApplied = true;
      this.applyInpaintModelDefaults();
      this.loadInpaintModelState(modelSelect?.value || this.state.selectedModel);
    }
    if (modelSelect && !modelSelect.dataset.defaultsBound) {
      modelSelect.dataset.defaultsBound = "1";
      modelSelect.addEventListener("change", () => {
        this.applyInpaintModelDefaults();
        this.loadInpaintModelState(modelSelect.value);
      });
    }
    this.renderInpaintLayersUI();
    this.renderInpaintCropUI();
    this.updateInpaintCropStatus();
    this.updateInpaintStatus();

    const brushBtn = document.getElementById("inpaint-tool-brush");
    if (brushBtn && !brushBtn.dataset.bound) {
      brushBtn.dataset.bound = "1";
      brushBtn.addEventListener("click", () => this.setInpaintTool("brush"));
    }
    const eraserBtn = document.getElementById("inpaint-tool-eraser");
    if (eraserBtn && !eraserBtn.dataset.bound) {
      eraserBtn.dataset.bound = "1";
      eraserBtn.addEventListener("click", () => this.setInpaintTool("eraser"));
    }

    const fileInput = document.getElementById("inpaint-file-input");
    if (fileInput && !fileInput.dataset.bound) {
      fileInput.dataset.bound = "1";
      fileInput.addEventListener("change", (event) => {
        const file = event.target.files?.[0];
        if (file) this.uploadInpaintFile(file);
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
        if (file) this.uploadInpaintFile(file);
        else this.toast("error", "Перетащите изображение PNG, JPEG или WebP");
      });
    }

    if (!state._initialized) {
      state._initialized = true;
      this.setInpaintTool("brush");
      this.setInpaintBrushSize(state.brushSize);
      this.setInpaintSoftness(state.softness);
    }
  },

});


// ---------- Resilient Boot ----------
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => App.init());
} else {
  App.init();
}

// Refresh dynamically rendered texts when the interface language changes.
window.addEventListener("qf:locale-changed", () => {
  App.renderInpaintCropUI();
  App.updateInpaintCropStatus();
  App.updateInpaintStatus();
  App.renderInpaintLayersUI();
});

// Export for inline scripts
window.App = App;
