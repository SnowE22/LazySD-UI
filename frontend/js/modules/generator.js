/* ============================================================
   QuantFunc WebUI — Generation Controller & Gallery
   Prompts, parameters, Img2Img, active jobs polling, history modal.
   ============================================================ */

import { apiGet, apiPost, apiDelete, apiUpload, toast, escapeHtml } from "../api.js";
import { applyGenMaskParams, getMaskCapableModels } from "./generate-mask.js";

function t(key, fallback, vars) {
  const fn = typeof window !== "undefined" ? window.t : null;
  if (typeof fn === "function") {
    const value = fn(key, vars || {});
    if (value && value !== key) return value;
  }
  return fallback || key;
}

export async function onGenerate(appRef) {
  const app = appRef || window.App;
  if (!app) return;

  const prompt = document.getElementById("prompt-input")?.value?.trim();
  if (!prompt) {
    toast("warning", "Введите текстовый промпт для генерации");
    return;
  }

  if (!app.state.selectedModel) {
    toast("warning", "Выберите модель");
    return;
  }

  const activeModel = app.state.models.find(m => m.id === app.state.selectedModel);
  if (activeModel && activeModel.disabled) {
    toast("error", activeModel.disabled_reason || "Модель временно отключена.");
    return;
  }

  if (activeModel && activeModel.installed === false) {
    toast("warning", `Модель "${activeModel.name}" не скачана. Перейдите в Model Hub для загрузки.`);
    const noticeEl = document.getElementById("gen-uninstalled-notice");
    const noticeText = document.getElementById("gen-uninstalled-notice-text");
    if (noticeText) noticeText.textContent = `Модель «${activeModel.name}» не скачана. Перейдите в Model Hub для загрузки.`;
    if (noticeEl) noticeEl.style.display = "flex";
    return;
  }

  // Check text encoder installation
  const teSelect = document.getElementById("text-encoder-select");
  const modelState = app.state.modelState;
  const activeTe = modelState?.text_encoders?.find(t => t.id === teSelect?.value);
  if (activeTe && !activeTe.installed && teSelect?.value !== "custom") {
    toast("warning", `Текстовый энкодер "${activeTe.name}" не скачан. Скачайте его в Model Hub.`);
    const noticeEl = document.getElementById("gen-uninstalled-notice");
    const noticeText = document.getElementById("gen-uninstalled-notice-text");
    if (noticeText) noticeText.textContent = `Текстовый энкодер «${activeTe.name}» не скачан. Скачайте его в Model Hub.`;
    if (noticeEl) noticeEl.style.display = "flex";
    return;
  }

  // Check FLUX.3 Nunchaku + diffusers-main requirement
  if (activeModel?.family === "flux3") {
    const extrasEnv = app.state.extrasStatus?.env;
    if (extrasEnv?.nunchaku && extrasEnv.has_flux3 === false) {
      toast("error", "FLUX.3 SVDQuant требует Nunchaku >= 1.4.0. Обновите Nunchaku в Настройки → Движки или выберите 'Flux.2 Klein (Nunchaku)'.");
      return;
    }
    if (extrasEnv?.diffusers && extrasEnv.has_diffusers_flux3 === false) {
      toast("error", "FLUX.3 требует diffusers из GitHub main (стабильный 0.41.0 не поддерживает FLUX.3). Переустановите Nunchaku в Настройки → Движки — установщик поставит diffusers main.");
      return;
    }
  }

  const res = parseResolution();
  const variantSelect = document.getElementById("variant-select");
  const selectedVariantId = variantSelect?.value || app.state.selectedVariant;
  const currentVariant = app.state.modelState?.transformers?.find(v => v.id === selectedVariantId);
  const backend = currentVariant ? currentVariant.backend : "lighting";

  const cfgVal = parseFloat(document.getElementById("cfg-input")?.value);
  const noiseVal = parseFloat(document.getElementById("noise-input")?.value);
  const negPrompt = document.getElementById("negative-prompt-input")?.value?.trim() || "";
  const samplerVal = document.getElementById("sampler-select")?.value || "euler";

  const params = {
    model_id: app.state.selectedModel,
    variant_id: selectedVariantId,
    prompt: prompt,
    negative_prompt: negPrompt,
    seed: parseSeed(),
    width: res.width,
    height: res.height,
    steps: parseInt(document.getElementById("steps-input")?.value || (currentVariant ? String(currentVariant.steps) : "4"), 10),
    cfg: !isNaN(cfgVal) ? cfgVal : 1.0,
    noise: !isNaN(noiseVal) ? noiseVal : 0.0,
    sampler: samplerVal,
    backend: backend,
  };

  if (app.state.activeLoras && app.state.activeLoras.length > 0) {
    params.loras = app.state.activeLoras.map(l => ({
      name: l.name,
      strength: l.strength !== undefined ? parseFloat(l.strength) : 1.0,
    }));
  }

  if (app.state.img2imgRef) {
    params.init_image_path = app.state.img2imgRef.path;
    params.init_image_strength = parseFloat(document.getElementById("img2img-strength")?.value || "0.85");
    params.ref_mode = document.getElementById("img2img-mode")?.value || "control_ref";
    const cnInput = document.getElementById("controlnet-custom-path");
    const cnPath = cnInput && cnInput.style.display !== "none" ? cnInput.value.trim() : "";
    if (cnPath) params.control_net_path = cnPath;
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
      if (vaeSelect && vaeSelect.value) params.vae = vaeSelect.value;
    }
  }

  // Mask editing (Generate): the mask defines the edited region, everything
  // outside of it is restored by the backend compositor.
  const maskStatus = applyGenMaskParams(app, params);
  if (maskStatus === "empty") {
    toast("warning", t("generate.mask.empty_block", "Маска пуста — нарисуйте область или отключите режим «Маска»"));
    return;
  }
  if (maskStatus === "ok") {
    const maskModel = (app.state.models || []).find((m) => m.id === params.model_id);
    if (maskModel && !maskModel.supports_mask) {
      const capable = getMaskCapableModels(app);
      const names = capable.map((m) => m.name).join(", ") || t("generate.mask.no_capable", "нет в наличии");
      toast("error", t("generate.mask.model_unsupported", `Модель «${maskModel.name}» не поддерживает маску. Поддерживают: ${names}`, { name: maskModel.name, list: names }));
      return;
    }
    params.init_image_strength = 1.0;
  }

  setGenerating(true, app);
  try {
    const result = await apiPost("/api/generate", params);
    app.state.currentJob = result.job_id;
    startPolling(result.job_id, app);
  } catch (e) {
    console.error("[app] Generate failed:", e);
    toast("error", "Ошибка генерации: " + e.message);
    setGenerating(false, app);
  }
}

export function setGenerating(isGen, appRef) {
  const btn = document.getElementById("btn-generate");
  const cancelBtn = document.getElementById("btn-cancel");
  const dockBtn = document.getElementById("dock-generate-btn");
  const progressWrap = document.getElementById("progress-wrap");
  const dockProgress = document.getElementById("dock-progress-bar");

  if (btn) btn.disabled = isGen;
  if (dockBtn) {
    dockBtn.disabled = isGen;
    dockBtn.textContent = isGen ? "⏳ Генерация..." : "✨ Сгенерировать";
  }
  if (cancelBtn) cancelBtn.style.display = isGen ? "inline-flex" : "none";
  if (progressWrap) progressWrap.style.display = isGen ? "block" : "none";
  if (dockProgress) dockProgress.style.display = isGen ? "block" : "none";
}

export async function cancelJob(appRef) {
  const app = appRef || window.App;
  if (!app?.state?.currentJob) return;
  try {
    await apiPost(`/api/jobs/${app.state.currentJob}/cancel`);
    toast("info", "Генерация отменена");
  } catch (e) {
    toast("error", "Ошибка отмены: " + e.message);
  }
}

export function startPolling(jobId, appRef) {
  const app = appRef || window.App;
  if (app?.state?.pollInterval) clearInterval(app.state.pollInterval);

  const poll = async () => {
    try {
      const status = await apiGet(`/api/jobs/${jobId}`);
      updateProgress(status);

      if (status.status === "complete") {
        clearInterval(app.state.pollInterval);
        app.state.pollInterval = null;
        setGenerating(false, app);
        onGenerationComplete(status, app);
      } else if (status.status === "error" || status.status === "cancelled") {
        clearInterval(app.state.pollInterval);
        app.state.pollInterval = null;
        setGenerating(false, app);
        toast("error", status.error || status.message || "Генерация прервана");
      }
    } catch (_) {
      clearInterval(app.state.pollInterval);
      app.state.pollInterval = null;
      setGenerating(false, app);
    }
  };

  app.state.pollInterval = setInterval(poll, 350);
}

function updateProgress(status) {
  const pBar = document.getElementById("progress-bar-inner");
  const pText = document.getElementById("progress-text");
  const dockFill = document.getElementById("dock-progress-fill");
  const pct = Math.round((status.progress || 0) * 100);

  if (pBar) pBar.style.width = `${pct}%`;
  if (dockFill) dockFill.style.width = `${pct}%`;
  if (pText) pText.textContent = status.message || `${pct}%`;
}

export function onGenerationComplete(status, appRef) {
  const app = appRef || window.App;
  const filename = (status.image_path || "").split(/[\/\\]/).pop();
  const url = `/outputs/${encodeURIComponent(filename)}?t=${Date.now()}`;

  if (app) {
    app.state.previousPreviewUrl = app.state.currentPreviewUrl;
    app.state.currentPreviewUrl = url;
    app.state.currentPreviewFilename = filename;
    app.state.lastPreviewName = filename;
    app.state.lastPreviewUrl = url;
  }

  const emptyEl = document.getElementById("empty-state");
  const previewImg = document.getElementById("preview-img");
  const wrap = document.getElementById("preview-image-wrap");

  if (emptyEl) emptyEl.style.display = "none";
  if (wrap) wrap.style.display = "inline-flex";
  if (previewImg && !app?.state?.genMask?.enabled) previewImg.src = url;

  toast("success", "Изображение готово ✓");
  loadHistory(app);
}

export function parseResolution() {
  const sel = document.getElementById("resolution-select")?.value || "1024x1024";
  const [w, h] = sel.split("x").map(x => parseInt(x, 10));
  return { width: w || 1024, height: h || 1024 };
}

export function parseSeed() {
  const val = document.getElementById("seed-input")?.value?.trim();
  if (!val || val === "-1") return Math.floor(Math.random() * 2147483647);
  const n = parseInt(val, 10);
  return isNaN(n) ? Math.floor(Math.random() * 2147483647) : n;
}

export function randomizeSeed() {
  const inp = document.getElementById("seed-input");
  if (inp) inp.value = "-1";
}

export async function uploadImg2ImgFile(file, appRef) {
  const app = appRef || window.App;
  try {
    toast("info", "Загрузка изображения референса...");
    const data = await apiUpload(file);
    setImg2ImgRef(data.path, data.url, app);
    toast("success", "Референс загружен ✓");
  } catch (e) {
    toast("error", "Ошибка загрузки: " + e.message);
  }
}

export function setImg2ImgRef(path, url, appRef) {
  const app = appRef || window.App;
  if (app) app.state.img2imgRef = { path, url };
  const preview = document.getElementById("img2img-preview");
  const empty = document.getElementById("img2img-empty");
  const container = document.getElementById("img2img-ref-container");

  if (preview) {
    preview.src = url;
    preview.style.display = "block";
  }
  if (empty) empty.style.display = "none";
  if (container) container.classList.add("has-image");
  if (app?.state?.genMask?.enabled && app.setGenMaskSource) {
    app.setGenMaskSource({ path, url, name: path });
  }
}

export function clearImg2ImgRef(appRef) {
  const app = appRef || window.App;
  if (app) app.state.img2imgRef = null;
  const preview = document.getElementById("img2img-preview");
  const empty = document.getElementById("img2img-empty");
  const container = document.getElementById("img2img-ref-container");

  if (preview) {
    preview.src = "";
    preview.style.display = "none";
  }
  if (empty) empty.style.display = "flex";
  if (container) container.classList.remove("has-image");
  if (app?.state?.genMask?.enabled) {
    if (app.state.currentPreviewUrl && app.state.currentPreviewFilename) {
      app.setGenMaskSource({
        path: app.state.currentPreviewFilename,
        url: app.state.currentPreviewUrl,
        name: app.state.currentPreviewFilename,
      });
    } else {
      app.clearGenMaskSource?.();
    }
  }
}

export function updateStrengthLabel(val) {
  const lbl = document.getElementById("img2img-strength-val");
  if (lbl) lbl.textContent = parseFloat(val).toFixed(2);
}

export function sendCurrentPreviewToImg2Img(appRef) {
  const app = appRef || window.App;
  if (!app?.state?.currentPreviewUrl || !app?.state?.currentPreviewFilename) {
    toast("warning", "Нет сгенерированного изображения для отправки в Img2Img");
    return;
  }
  setImg2ImgRef(app.state.currentPreviewFilename, app.state.currentPreviewUrl, app);
  toast("success", "Текущее изображение установлено как референс Img2Img");
}

export async function loadHistory(appRef) {
  const app = appRef || window.App;
  try {
    const items = await apiGet("/api/history");
    if (app) {
      app.state.historyItems = items || [];
      renderHistory(app);
    }
  } catch (e) {
    console.warn("Could not load history:", e);
  }
}

export function renderHistory(appRef) {
  const app = appRef || window.App;
  const grid = document.getElementById("history-grid");
  if (!grid || !app) return;

  const items = app.state.historyItems || [];
  if (!items.length) {
    grid.innerHTML = '<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px 0;">История пуста</div>';
    return;
  }

  grid.innerHTML = items.map((item, idx) => `
    <div class="history-card" onclick="App.openHistoryModal(${idx})">
      <img src="${escapeHtml(item.url)}" alt="Gen ${idx}" loading="lazy">
      <div class="history-card-caption">${escapeHtml(item.name || item.prompt || "")}</div>
    </div>
  `).join("");
}

export function openHistoryModal(index, appRef) {
  const app = appRef || window.App;
  const item = app?.state?.historyItems?.[index];
  if (!item) return;

  app.state.selectedHistoryItem = item;
  const modal = document.getElementById("history-modal");
  const img = document.getElementById("history-modal-img");
  const promptBox = document.getElementById("history-modal-prompt-box");

  if (img) img.src = item.url;
  if (promptBox) {
    promptBox.innerHTML = `
      <div style="font-size: 13px; color: var(--text); line-height: 1.4; margin-bottom: 8px;">${escapeHtml(item.prompt || "—")}</div>
      <div style="font-size: 11px; color: var(--text-dim);">${escapeHtml(item.model_name || item.model_id || "")} · ${item.width || 1024}×${item.height || 1024} px</div>
    `;
  }
  if (modal) modal.style.display = "flex";
}

export function closeHistoryModal() {
  const modal = document.getElementById("history-modal");
  if (modal) modal.style.display = "none";
}

export function toggleHistoryPrompt() {
  const box = document.getElementById("history-modal-prompt-box");
  const btn = document.getElementById("history-prompt-toggle");
  if (!box || !btn) return;
  const isHidden = box.style.display === "none";
  box.style.display = isHidden ? "block" : "none";
  btn.textContent = isHidden ? "Свернуть ▴" : "Развернуть ▾";
}

export function reuseHistoryItem(appRef) {
  const app = appRef || window.App;
  const item = app?.state?.selectedHistoryItem;
  if (!item) return;

  closeHistoryModal();
  const promptInp = document.getElementById("prompt-input");
  if (promptInp && item.prompt) promptInp.value = item.prompt;

  if (item.model_id) {
    const sel = document.getElementById("model-select");
    if (sel) {
      sel.value = item.model_id;
      if (app.onModelChange) app.onModelChange(item.variant_id);
    }
  }

  if (item.seed !== undefined) {
    const seedInp = document.getElementById("seed-input");
    if (seedInp) seedInp.value = String(item.seed);
  }
  if (app.updateDockPills) app.updateDockPills();
  toast("success", "Параметры скопированы в генератор ✓");
}

export async function deleteHistoryItem(appRef) {
  const app = appRef || window.App;
  const item = app?.state?.selectedHistoryItem;
  if (!item) return;
  if (!confirm(`Удалить изображение ${item.name}?`)) return;

  try {
    await apiDelete(`/api/history/${encodeURIComponent(item.name)}`);
    closeHistoryModal();
    toast("success", "Изображение удалено");
    loadHistory(app);
  } catch (e) {
    toast("error", "Ошибка удаления: " + e.message);
  }
}
