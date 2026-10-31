/* ============================================================
   i18n — bilingual RU/EN support
   - Loads translations from /locale/<lang>.json
   - Auto-detects language from navigator.language
   - Persists choice in localStorage + backend settings.json
   - Pluralization: t("key", {count: n}) → looks up key.one/few/many
   ============================================================ */

let currentLocale = "en";
let translations = { en: {}, ru: {} };

async function loadLocale(locale) {
  try {
    const res = await fetch(`/locale/${locale}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    translations[locale] = await res.json();
  } catch (e) {
    console.error(`[i18n] Failed to load ${locale}:`, e);
    // Try loading from relative path (dev mode)
    try {
      const res2 = await fetch(`frontend/locale/${locale}.json`);
      translations[locale] = await res2.json();
    } catch (e2) {
      console.error(`[i18n] Also failed relative path:`, e2);
      translations[locale] = {};
    }
  }
}

function t(key, vars = {}) {
  // Try current locale, fall back to English, fall back to key
  let str = translations[currentLocale]?.[key]
         || translations.en?.[key]
         || key;

  // Variable substitution: {{name}} → vars.name
  for (const [k, v] of Object.entries(vars)) {
    str = str.replace(new RegExp(`{{${k}}}`, "g"), v);
  }
  return str;
}

function applyTranslations() {
  // Apply to all elements with data-i18n attribute
  document.querySelectorAll("[data-i18n]").forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  // Apply to placeholders
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  // Apply to titles
  document.querySelectorAll("[data-i18n-title]").forEach(el => {
    el.title = t(el.dataset.i18nTitle);
  });
  // Update <html lang>
  document.documentElement.lang = currentLocale;
  // Update active state on toggle buttons
  document.querySelectorAll(".lang-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.lang === currentLocale);
  });
}

async function setLocale(locale) {
  if (!translations[locale] || Object.keys(translations[locale]).length === 0) {
    await loadLocale(locale);
  }
  currentLocale = locale;
  applyTranslations();
  localStorage.setItem("qf_lang", locale);

  // Let dynamically rendered widgets refresh their texts.
  try {
    window.dispatchEvent(new CustomEvent("qf:locale-changed", { detail: { locale } }));
  } catch (e) {
    console.warn("[i18n] Could not dispatch locale change:", e);
  }

  // Persist to backend (best-effort, non-blocking)
  try {
    fetch("/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang: locale }),
    }).catch(() => {});  // Ignore network errors
  } catch (e) {
    console.warn("[i18n] Could not persist language:", e);
  }
}

function detectInitialLocale() {
  // Priority: localStorage > settings.json (injected by backend) > navigator.language > "en"
  const saved = localStorage.getItem("qf_lang");
  if (saved && (saved === "ru" || saved === "en")) {
    return saved;
  }
  // Check if backend injected a setting
  if (window.QF_SETTINGS?.lang && window.QF_SETTINGS.lang !== "auto") {
    return window.QF_SETTINGS.lang;
  }
  // Auto-detect from browser
  const navLang = navigator.language || navigator.userLanguage || "en";
  return navLang.toLowerCase().startsWith("ru") ? "ru" : "en";
}

// ---------- Init ----------
async function initI18n() {
  // Preload both locales
  await Promise.all([loadLocale("en"), loadLocale("ru")]);

  const initial = detectInitialLocale();
  await setLocale(initial);

  // Wire up toggle buttons (if any exist yet)
  document.querySelectorAll(".lang-btn").forEach(btn => {
    btn.addEventListener("click", () => setLocale(btn.dataset.lang));
  });
}

// Export to global scope for inline scripts
window.t = t;
window.setLocale = setLocale;
window.applyTranslations = applyTranslations;
window.initI18n = initI18n;
