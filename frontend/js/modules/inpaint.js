(function() {
  'use strict';

  const getApp = () => window.App || {};

  async function apiPost(path, body = null, retries = 0) {
    const a = getApp();
    if (typeof a.apiPost === 'function') {
      return a.apiPost(path, body, retries);
    }
    const res = await fetch(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : null
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function apiGet(path, retries = 2) {
    const a = getApp();
    if (typeof a.apiGet === 'function') {
      return a.apiGet(path, retries);
    }
    const res = await fetch(path);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function showToast(kind, message) {
    const a = getApp();
    if (typeof a.toast === 'function') {
      a.toast(kind, message);
    } else if (typeof window.toast === 'function') {
      window.toast(kind, message);
    } else {
      console.log(`[${kind}] ${message}`);
    }
  }

  
  const app = window.App || {};
  if (!app.state) app.state = {};
  app.state.inpaint = {
    tool: 'brush', // 'view', 'brush', 'eraser'
    brushSize: 32,
    maskBlur: 4,
    denoiseStrength: 0.75,
    steps: 20,
    cfgScale: 7.0,
    seed: -1,
    invertMask: false,
    showMaskOverlay: true,
    sourceDataUrl: null,
    sourcePath: null,
    resultDataUrl: null,
    comparing: false,
    isDrawing: false,
    lastX: 0,
    lastY: 0
  };

  function getCanvas() {
    return document.getElementById('inpaint-mask-canvas');
  }

  function getCtx() {
    const canvas = getCanvas();
    return canvas ? canvas.getContext('2d') : null;
  }

  function getCoords(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / (rect.width || 1);
    const scaleY = canvas.height / (rect.height || 1);
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      scale: scaleX
    };
  }

  function drawStroke(x1, y1, x2, y2, tool, brushSize, scale) {
    const ctx = getCtx();
    if (!ctx) return;
    ctx.save();
    if (tool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = 'rgba(0,0,0,1)';
      ctx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'rgba(239, 68, 68, 1.0)';
      ctx.fillStyle = 'rgba(239, 68, 68, 1.0)';
    }
    const strokeWidth = Math.max(1, brushSize * scale);
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x2, y2, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  Object.assign(app, {
    initInpaintStudio(appRef) {
      const st = app.state.inpaint;
      const canvas = getCanvas();

      if (!canvas || st._inpaintEventsBound) return;
      st._inpaintEventsBound = true;

      canvas.style.pointerEvents = (st.tool === 'view') ? 'none' : 'auto';

      canvas.addEventListener('pointerdown', e => {
        if (st.tool === 'view' || e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
        const p = getCoords(e, canvas);
        st.lastX = p.x;
        st.lastY = p.y;
        st.isDrawing = true;
        drawStroke(p.x, p.y, p.x, p.y, st.tool, st.brushSize, p.scale);
      });

      canvas.addEventListener('pointermove', e => {
        if (!st.isDrawing) return;
        e.preventDefault();
        e.stopPropagation();
        const p = getCoords(e, canvas);
        drawStroke(st.lastX, st.lastY, p.x, p.y, st.tool, st.brushSize, p.scale);
        st.lastX = p.x;
        st.lastY = p.y;
      });

      const stopDrawing = (e) => {
        if (!st.isDrawing) return;
        st.isDrawing = false;
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      };
      canvas.addEventListener('pointerup', stopDrawing);
      canvas.addEventListener('pointercancel', stopDrawing);

      document.getElementById('inpaint-dropzone').addEventListener('dragover', e => {
        e.preventDefault();
      });

      document.getElementById('inpaint-dropzone').addEventListener('drop', async (e) => {
        const file = e.dataTransfer.files[0];
        if (!file) return;
        app.onInpaintFileUpload({ target: { files: [file] } });
      });

      if (app.initViewportZoomPan) app.initViewportZoomPan('inpaint', { areaId: 'inpaint-dropzone', wrapId: 'inpaint-preview-wrap', hudId: 'inpaint-zoom-hud', labelId: 'inpaint-zoom-label', scrollXId: 'inpaint-scrollbar-x', thumbXId: 'inpaint-scrollbar-thumb-x', scrollYId: 'inpaint-scrollbar-y', thumbYId: 'inpaint-scrollbar-thumb-y' }, app);

      // Auto-toggle crisp rendering when zooming in
      window.addEventListener('viewport-transform', (e) => {
        if (e.detail?.mode === 'inpaint') {
          const img = document.getElementById('inpaint-preview-img');
          const canvas = document.getElementById('inpaint-mask-canvas');
          const zoom = Number(e.detail.zoom) || 1;
          const isZoomed = zoom >= 1.05;
          if (img) img.classList.toggle('crisp-zoom', isZoomed);
          if (canvas) canvas.classList.toggle('crisp-zoom', isZoomed);
        }
      });

      // Synchronize inpaint inspector with main generator state
      app.syncInpaintInspector();
    },

    syncInpaintInspector() {
      if (app.updateModelTrigger) app.updateModelTrigger();

      const mainVariant = document.getElementById('variant-select');
      const inpaintVariant = document.getElementById('inpaint-variant-select');
      if (mainVariant && inpaintVariant) {
        inpaintVariant.innerHTML = mainVariant.innerHTML;
        inpaintVariant.value = mainVariant.value;
        inpaintVariant.onchange = () => {
          mainVariant.value = inpaintVariant.value;
          mainVariant.dispatchEvent(new Event('change'));
        };
      }

      const mainTe = document.getElementById('text-encoder-select');
      const inpaintTe = document.getElementById('inpaint-text-encoder-select');
      if (mainTe && inpaintTe) {
        inpaintTe.innerHTML = mainTe.innerHTML;
        inpaintTe.value = mainTe.value;
        inpaintTe.onchange = () => {
          mainTe.value = inpaintTe.value;
          if (app.onTextEncoderChange) app.onTextEncoderChange();
        };
      }

      const mainVae = document.getElementById('vae-select');
      const inpaintVae = document.getElementById('inpaint-vae-select');
      if (mainVae && inpaintVae) {
        inpaintVae.innerHTML = mainVae.innerHTML;
        inpaintVae.value = mainVae.value;
        inpaintVae.onchange = () => {
          mainVae.value = inpaintVae.value;
          if (app.onVaeChange) app.onVaeChange();
        };
      }

      const mainLoraCount = document.getElementById('inspector-lora-count');
      const inpaintLoraCount = document.getElementById('inpaint-lora-count');
      const mainLoraList = document.getElementById('lora-inspector-list');
      const inpaintLoraList = document.getElementById('inpaint-lora-list');
      if (mainLoraCount && inpaintLoraCount) inpaintLoraCount.textContent = mainLoraCount.textContent;
      if (mainLoraList && inpaintLoraList) inpaintLoraList.innerHTML = mainLoraList.innerHTML;
    },

    toggleInpaintSharpness() {
      const img = document.getElementById('inpaint-preview-img');
      const canvas = document.getElementById('inpaint-mask-canvas');
      const btn = document.getElementById('inpaint-sharp-btn');
      const label = document.getElementById('inpaint-sharp-label');
      if (!img) return;
      const isSharp = img.classList.toggle('inpaint-sharp');
      if (canvas) canvas.classList.toggle('inpaint-sharp', isSharp);
      if (btn) btn.classList.toggle('active', isSharp);
      if (label) label.textContent = isSharp ? 'Резко (Вкл)' : 'Чётко';
      showToast('info', isSharp ? 'Пиксельная чёткость включена (без размытия при зуме)' : 'Автоматическое сглаживание');
    },

    setInpaintTool(tool) {
      const st = app.state.inpaint;
      st.tool = tool;
      ['view', 'brush', 'eraser'].forEach(t => {
        const btn = document.getElementById(`inpaint-tool-${t}`);
        if (btn) btn.classList.toggle('active', t === tool);
      });
      const canvas = getCanvas();
      if (canvas) {
        canvas.style.cursor = (tool === 'view') ? 'grab' : 'crosshair';
        canvas.style.pointerEvents = (tool === 'view') ? 'none' : 'auto';
      }
    },

    setInpaintBrushSize(size) {
      const val = parseInt(size, 10) || 32;
      const st = app.state.inpaint;
      st.brushSize = val;
      const range = document.getElementById('inpaint-brush-size');
      const num = document.getElementById('inpaint-brush-val');
      if (range && range.value != val) range.value = val;
      if (num) num.textContent = `${val}px`;
    },

    clearInpaintMask() {
      const ctx = getCtx();
      if (!ctx) return;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    },

    invertInpaintMask() {
      const canvas = getCanvas();
      const ctx = getCtx();
      if (!canvas || !ctx) return;
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 3] > 32) {
          d[i] = 0; d[i+1] = 0; d[i+2] = 0; d[i+3] = 0;
        } else {
          d[i] = 239; d[i+1] = 68; d[i+2] = 68; d[i+3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    },

    toggleInpaintMaskOverlay(show) {
      const canvas = getCanvas();
      if (canvas) canvas.style.display = show ? 'block' : 'none';
    },

    async onInpaintFileUpload(event) {
      const file = event?.target?.files?.[0];
      if (!file) return;
      showToast('info', 'Загрузка изображения...');
      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        if (res.ok) {
          const data = await res.json();
          app.state.inpaint.sourcePath = data.path;
        }
      } catch (err) {
        console.warn('Direct upload error:', err);
      }
      const reader = new FileReader();
      reader.onload = (e) => app.loadInpaintSource(e.target.result, file.name);
      reader.readAsDataURL(file);
    },

    loadInpaintSource(url, filename, appRef) {
      const st = app.state.inpaint;
      st.sourceDataUrl = url;
      st.resultDataUrl = null;
      st.comparing = false;

      const previewImg = document.getElementById('inpaint-preview-img');
      const metaInfo = document.getElementById('inpaint-meta-info');
      const emptyState = document.getElementById('inpaint-empty-state');
      const previewWrap = document.getElementById('inpaint-preview-wrap');
      const compareBadge = document.getElementById('inpaint-compare-badge');

      if (compareBadge) compareBadge.style.display = 'none';

      if (previewImg) {
        previewImg.onload = () => {
          const canvas = getCanvas();
          if (canvas) {
            canvas.width = previewImg.naturalWidth;
            canvas.height = previewImg.naturalHeight;
            canvas.style.pointerEvents = (st.tool === 'view') ? 'none' : 'auto';
            const ctx = canvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
          if (metaInfo) metaInfo.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight} px`;
          if (emptyState) emptyState.style.display = 'none';
          if (previewWrap) previewWrap.style.display = 'inline-flex';
        };
        previewImg.src = url;
      }
    },

    async runInpaint() {
      const st = app.state.inpaint;

      if (!st.sourceDataUrl) {
        showToast('warning', 'Сначала загрузите изображение');
        return;
      }

      const prompt = document.getElementById('inpaint-prompt')?.value?.trim() || '';
      const negative_prompt = document.getElementById('inpaint-negative-prompt')?.value?.trim() || '';
      const strength = parseFloat(document.getElementById('inpaint-strength')?.value) || st.denoiseStrength;
      const mask_blur = parseInt(document.getElementById('inpaint-mask-blur')?.value, 10) || st.maskBlur;
      const steps = parseInt(document.getElementById('inpaint-steps')?.value, 10) || st.steps;
      const cfg_scale = parseFloat(document.getElementById('inpaint-cfg')?.value) || st.cfgScale;
      const seedVal = document.getElementById('inpaint-seed')?.value?.trim();
      const seed = (seedVal && seedVal !== '-1' && !isNaN(parseInt(seedVal, 10))) ? parseInt(seedVal, 10) : null;
      const model = app.state.selectedModel || document.getElementById('inpaint-model-select')?.value;
      const variant = document.getElementById('inpaint-variant-select')?.value || app.state.selectedVariant;

      const canvas = getCanvas();
      const maskDataUrl = canvas ? canvas.toDataURL('image/png') : '';

      const overlay = document.getElementById('inpaint-loading-overlay');
      const progressText = document.getElementById('inpaint-loading-text');

      if (overlay) overlay.style.display = 'flex';
      if (progressText) progressText.textContent = 'Инициализация генерации...';

      try {
        const lorasList = (app.state.activeLoras && app.state.activeLoras.length > 0)
          ? app.state.activeLoras.map(l => ({ name: l.name, strength: parseFloat(l.strength || 1.0) }))
          : (app.state.loras || []);

        const res = await apiPost('/api/generate', {
          mode: 'inpaint',
          model_id: model,
          variant_id: variant,
          prompt,
          negative_prompt,
          init_image_path: st.sourcePath || undefined,
          ref_image: st.sourceDataUrl,
          init_image_strength: strength,
          mask_image: maskDataUrl,
          mask_blur: mask_blur,
          invert_mask: st.invertMask,
          width: canvas ? canvas.width : 1024,
          height: canvas ? canvas.height : 1024,
          steps,
          cfg_scale,
          seed,
          loras: lorasList
        });

        const jobId = res?.job_id;
        if (!jobId) throw new Error('Не получен job_id');

        const pollInterval = setInterval(async () => {
          try {
            const status = await apiGet(`/api/jobs/${jobId}`);
            if (progressText && status) {
              const step = status.step || 0;
              const total = status.total_steps || steps;
              const pct = status.percentage || Math.round((step / Math.max(1, total)) * 100);
              progressText.textContent = `Генерация: шаг ${step}/${total} (${pct}%)`;
            }
            if (status.status === 'complete') {
              clearInterval(pollInterval);
              if (overlay) overlay.style.display = 'none';
              const outUrl = status.image_path.startsWith('/') ? status.image_path : '/' + status.image_path;
              st.resultDataUrl = outUrl;
              const previewImg = document.getElementById('inpaint-preview-img');
              if (previewImg) previewImg.src = outUrl;
              const canvas = getCanvas();
              if (canvas) canvas.style.display = 'none';
              const showMaskChk = document.getElementById('inpaint-show-mask');
              if (showMaskChk) showMaskChk.checked = false;
              showToast('success', 'Inpaint успешно сгенерирован! ✓');
              if (app.loadHistory) app.loadHistory();
            } else if (status.status === 'error') {
              clearInterval(pollInterval);
              if (overlay) overlay.style.display = 'none';
              showToast('error', 'Ошибка: ' + (status.error || status.message || ''));
            } else if (status.status === 'cancelled') {
              clearInterval(pollInterval);
              if (overlay) overlay.style.display = 'none';
              showToast('info', 'Генерация отменена');
            }
          } catch (err) {
            console.error('[inpaint] poll error:', err);
          }
        }, 250);
      } catch (e) {
        if (overlay) overlay.style.display = 'none';
        showToast('error', 'Ошибка запуска: ' + e.message);
      }
    },

    toggleInpaintCompare() {
      const st = app.state.inpaint;
      if (!st.resultDataUrl || !st.sourceDataUrl) return;

      st.comparing = !st.comparing;

      const previewImg = document.getElementById('inpaint-preview-img');
      const compareBadge = document.getElementById('inpaint-compare-badge');

      if (previewImg) previewImg.src = st.comparing ? st.sourceDataUrl : st.resultDataUrl;
      if (compareBadge) compareBadge.style.display = st.comparing ? 'block' : 'none';
    },

    downloadInpaintPng() {
      const st = app.state.inpaint;
      const url = st.resultDataUrl || st.sourceDataUrl;
      if (!url) return;

      const a = document.createElement('a');
      a.href = url;
      a.download = `inpaint_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  });
})();