# QuantFunc WebUI — Статус проекта и план доработки

**Дата:** 2026-09-15
**Версия:** Phase 2.5 (API-key проблема РЕШЕНА, registry расширен)
**Передача:** для следующего чата/исполнителя

> **Актуальность на 2026-10-23:** документ исторический, часть пунктов ниже уже
> реализована (E2E-генерация, Models/History/Templates, LoRA, ControlNet, Qwen/Ideogram,
> система слоёв Halftone). Актуальный журнал коммитов — [`CHANGELOG.md`](CHANGELOG.md),
> архитектура — [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md).
>
> Последние коммиты: `07d074a` (Inpaint crop modes: auto/params/manual),
> `00d4702` (Inpaint bbox crop + Qwen), `9acb452` (FLUX inpaint masks Nunchaku/OrbitQuant/SD.CPP),
> `507dd90` (New Dock: Ref/ControlNet/LoRA), `73ffbfd` (Fast HF downloads + mirror).
> Полный журнал: [`CHANGELOG.md`](CHANGELOG.md). Перед планированием сверяйся с `git log --oneline -20`.

---

## 1. Краткое описание проекта

Локальный WebUI для **QuantFunc Engine** — закрытого C++/CUDA движка диффузионного вывода.
Построен на **FastAPI + uvicorn** с полностью кастомным HTML/CSS/JS интерфейсом
(тёмная тема, IDE-grid layout, bilingual RU/EN).

**Не требует ComfyUI и не требует Gradio.** Использует QuantFunc engine binary напрямую
через subprocess CLI.

Архитектура:
```
Custom HTML/CSS/JS  →  FastAPI + uvicorn  →  Python backend
                                              ↓
                                       Subprocess Worker
                                              ↓
                                       QuantFunc Engine (libquantfunc)
                                              ↓
                                       CUDA  →  NVIDIA GPU
```

---

## 2. ✅ БЛОКИРУЮЩАЯ ОШИБКА РЕШЕНА (была в Phase 2)

### Что было
```
RuntimeError: Engine error: [ComponentImpl.cpp:1888] Flux2Klein: checkObfuscation failed:
This model is key-protected. An API key is required to load it. Please provide --apikey <key>.
```

### Причина (установлена 2026-09-15)

По официальной документе https://www.quantfunc.com/docs/model-loading-and-apikey (раздел 6):

- API-ключ (формат `qf_xxxxxxxx`) нужен для license-verification и расшифровки
  key-protected моделей. License-сервер: `https://service.quantfunc.com`.
- **Во время free trial (public beta) все клиенты используют один общий тестовый ключ.**
  ComfyUI-плагин распространяет его в файле `bin/<os>/config.json` рядом с движком:
  ```json
  {
      "server_url": "https://service.quantfunc.com",
      "api_key": "qf_4a256a9dfbfd04fd6fcf6bfc48781557aa1fed478d7a3c62214e91d0fbebba95"
  }
  ```
- Наш `download_engine()` качал только `quantfunc.dll` + `quantfunc.exe`, но НЕ `config.json`
  (в ModelScope-репо QuantFunc/Plugin его тоже нет — он живёт только в GitHub-репо плагина).
  Движок без ключа падал на `checkObfuscation`.

### Что сделано (все пункты раздела 4 из Phase 2)

1. **`backend/engine.py`**:
   - `ensure_config_json(force=False)` — ставит `bin/<os>/config.json`: скачивает из
     GitHub-репо плагина (`raw.githubusercontent.com/RealJonathanYip/ComfyUI-QuantFunc/main/bin/<os>/config.json`),
     при офлайне пишет встроенный fallback (`BUNDLED_TEST_CONFIG`). Существующий файл
     без `force=True` никогда не перезаписывается.
   - `read_engine_api_key()`, `has_api_key_available()`, `engine_config_json_path()`.
   - `download_engine()` теперь вызывает `ensure_config_json()` после установки бинарников.
2. **`backend/worker.py`**:
   - Pre-flight: если модель `key_protected` и ключа нигде нет — понятная ошибка ДО запуска engine.
   - `--apikey <key>` добавляется в args, если `settings.quantfunc_api_key` задан.
   - Детектор `_KEY_PROTECTED_RE` ловит `checkObfuscation`/`key-protected` в выводе engine
     и выдаёт инструкцию по фиксу вместо криптичной ошибки.
3. **`backend/settings.py`**: поле `quantfunc_api_key: Optional[str]`; `update()` принимает
   пустую строку для явной очистки секретов (quantfunc_api_key, hf/ms токены).
4. **`app.py`**: `quantfunc_api_key` в whitelist PATCH /api/settings; новый endpoint
   `POST /api/engine/ensure-config` (`{"force": true}` — перезаписать тестовым ключом);
   в `/api/config` добавлены `api_auth` (config_present/config_has_key/user_key_set)
   и `key_protected` у каждой модели; `init_state()` вызывает `ensure_config_json()` на старте.
5. **`frontend/index.html`**: Settings → Engine → поле «QuantFunc API key» (password) +
   кнопка-статус/восстановление «Restore test key» (`restoreBundledKey()`) + подсказка.
   `saveSettings()` отправляет `quantfunc_api_key` (пустая строка = очистить).
6. **i18n**: `settings.qf_api_key`, `settings.qf_api_key.hint`, `settings.qf_api_key.restore`,
   `errors.key_protected` (en + ru).

### Как теперь получить ключ (если свой)

- Public beta: ничего не нужно — тестовый ключ ставится автоматически.
- Свой ключ: https://www.quantfunc.com → Get API key (30 сек, бесплатно в beta)
  → Settings → Engine → QuantFunc API key.
- Официальная докa: https://www.quantfunc.com/docs/model-loading-and-apikey
- Discord: https://discord.gg/jCp9TpFWcn

---

## 3. ✅ Registry расширен: 3 модели

| Модель | id | Файлов | Размер transformer | key_protected | Комментарий |
|--------|----|--------|--------------------|---------------|-------------|
| Flux.2 Klein 9B | `flux_klein_9b` | 6 | 5.3 ГБ | **true** | Флагман, реальный SHA-256 |
| Flux.2 Klein 4B | `flux_klein_4b` | 6 | 2.1 ГБ | **true** | Компактный Klein |
| Z-Image Turbo | `z_image_turbo` | 5 | 3.5 ГБ | **false** | Самая лёгкая — **рекомендуется для первого e2e теста** |

- Все SHA-256 и размеры взяты из API ModelScope (`Sha256` поля метаданных) — реальные.
- Z-Image: офиц. CLI-пример из README репо использует `--steps 4 --model-backend lighting`.
- Klein-специфичные `klein_tier` фильтры сохранены для 4B; у Z-Image 40x/30x-below варианты
  помечены `klein_tier` по аналогии.
- **Qwen-Image-Series НЕ добавлял**: трансформеры ~13.7 ГБ — тяжёлая для RTX 5060 Ti 16GB
  связка; добавить при необходимости по тому же шаблону (repo `QuantFunc/Qwen-Image-Series`,
  есть prequant для SVDQ).

---

## 4. Что сделано ранее (Phase 2 — без изменений)

Backend: settings.py (Pydantic persisting), gpu.py (nvidia-smi→SM, tier), security.py
(path traversal, prompt, safe_subprocess_args), cache.py (state machine, SHA-256),
engine.py (download+verify, CUDA deps), registry loader + JSON, worker.py (subprocess,
progress parsing, CUDA PATH injection), sources/{base,huggingface,modelscope}.py.

Frontend: index.html (IDE-grid), styles.css (dark violet), app.js (REST client, polling,
toasts), i18n.js (RU/EN), dropzone.js, locale/{en,ru}.json.

История багов #1–#10 из Phase 2 остаётся актуальной (см. предыдущую версию документа,
архив: QuantFunc-WebUI-v1 zip + отчёт Phase 1).

### Тесты

`tests/smoke_test.py` — теперь **7 тестов**: imports, settings (+clearing секретов),
GPU, registry (3 модели, key_protected, resolve для SM 120/86), security, **engine auth
(config.json: установка, формат ключа, fallback, детектор key-protected)**, engine status.
Последний прогон: **7/7 passed** + интеграционная проверка живого сервера
(/api/config, /api/models, ensure-config force, PATCH set/clear ключа, HTML содержит
поле ключа, локали валидны) — все проверки прошли.

---

## 5. Что нужно доделать (по приоритетам)

### 🔴 КРИТИЧНО (следующий шаг — e2e на машине с GPU)

- [ ] **End-to-end генерация на Windows-машине с RTX 5060 Ti**:
  1. `python app.py` — при старте появится `[startup] Engine auth config: ... (source: ...)`
  2. Settings → Download engine (если ещё не стоит) — config.json поставится автоматически
  3. **Рекомендую начать с Z-Image Turbo** (~6.5 ГБ суммарно, key_protected=false)
  4. Generate → проверить PNG в outputs/ и preview в UI
  5. Повторить для Klein 4B / 9B (проверить, что тестовый ключ прокатывает для Klein)
- [ ] Если тестовый ключ перестанет действовать (окончание beta) — свой ключ с quantfunc.com

### 🟡 ВАЖНО (Phase 3 — UI polish)

- [ ] Models tab — карточки моделей, GPU tier filter, download buttons, статус файлов
- [ ] History tab — thumbnails, re-run, delete, clear all
- [ ] Templates tab — CRUD промптов, категории, теги, переменные `{{character}}`
- [ ] Persistent History — SQLite вместо RAM-only
- [ ] Image Edit / Img2Img / Inpaint pipelines + подключение dropzone к ним

### 🟢 НИЗКИЙ ПРИОРИТЕТ (Phase 4-6)

- [ ] Qwen-Image / Qwen-Image-Edit / Ideogram-4 / Qwen-Image-Layered в registry
  (все репо существуют: QuantFunc/Qwen-Image-Series, QuantFunc/Qwen-Image-Edit-Series;
   Auto Loader на сайте подтверждает серии: Qwen-Image-Edit, Qwen-Image, Z-Image,
   Klein-4B, Klein-9B)
- [ ] LoRA stacking UI, ControlNet, Layered generation
- [ ] ctypes C API вместо subprocess (9 символов C API)
- [ ] Template import/export, offline update improvements

---

## 6. Структура проекта (новое помечено ➕)

```
QuantFunc-WebUI/
├── app.py                              # FastAPI entry point (~870 строк) — ensure-config, api_auth
├── requirements.txt
├ start.bat                              # Windows launcher
├── settings.json                       # Создаётся при первом запуске (+ quantfunc_api_key) ➕
├── backend/
│   ├── settings.py                     # + quantfunc_api_key, clearable secrets ➕
│   ├── gpu.py / security.py / cache.py # без изменений
│   ├── engine.py                       # + ensure_config_json, BUNDLED_TEST_CONFIG, has_api_key_available ➕
│   ├── worker.py                       # + --apikey, pre-flight key check, _KEY_PROTECTED_RE ➕
│   ├── sources/                        # без изменений
│   └── registry/
│       ├── loader.py                   # + key_protected в ModelSpec ➕
│       └── models/
│           ├── flux_klein_9b.json      # + key_protected: true ➕
│           ├── flux_klein_4b.json      # новая модель ➕
│           └── z_image_turbo.json      # новая модель ➕
├── frontend/
│   ├── index.html                      # + поле API ключа, restoreBundledKey(), updateApiKeyStatus() ➕
│   ├── css/styles.css                  # без изменений
│   ├── js/ (app.js, i18n.js, dropzone.js)
│   └── locale/{en,ru}.json             # + 4 новых ключа ➕
├── bin/<os>/config.json                # создаётся автоматически при старте/скачивании engine ➕
├── models/cache/                       # ~6.5–10 ГБ на модель
├── outputs/
└── tests/smoke_test.py                 # 7 тестов ➕
```

---

## 7. Ключевые технические решения (добавление)

### Engine auth (config.json vs --apikey)
- Приоритет: `settings.quantfunc_api_key` (передаётся как `--apikey`) → иначе движок сам
  читает `bin/<os>/config.json` (сервер + тестовый ключ).
- `ensure_config_json(force=False)` никогда не перезаписывает существующий файл —
  пользовательские правки безопасны. `force=True` — только по кнопке «Восстановить».
- Тестовый ключ — публичный (идёт в поставке плагина), хранить его в коде допустимо;
  после окончания beta пользователи вносят свой ключ.

### GPU detection / CUDA DLL discovery / Lazy download / Security
- Без изменений — см. предыдущую версию документа или код.

---

## 8. Target environment

- **OS:** Windows 10/11 (основной target), Linux (best-effort)
- **GPU:** NVIDIA GeForce RTX 5060 Ti 16 ГБ (Blackwell, SM 120, tier 50x-above, FP4 native)
- **CUDA:** v13.1 (`G:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64`)
- **Driver:** 610.62
- **Python:** 3.12 в venv
- **Использование:** некоммерческое, команда игроделов, single-user

---

## 9. Как запустить

```cmd
cd QuantFunc-WebUI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Открыть: `http://127.0.0.1:7860`

### Первый запуск
1. Settings → Download engine (~120 МБ с ModelScope; config.json поставится сам)
2. Settings → CUDA bin path: `G:\...\v13.1\bin\x64` (если auto-detect не сработал)
3. Generate → **Z-Image Turbo** → Download model (~6.5 ГБ) → Generate
4. Klein 9B теперь тоже должен работать (тестовый ключ из config.json)

---

## 10. Ссылки и ресурсы

- **Docs API key (ОФИЦИАЛЬНАЯ, раздел 6 — про ключ):** https://www.quantfunc.com/docs/model-loading-and-apikey
- **Get API key:** https://www.quantfunc.com (кнопка Get API key)
- **GitHub QuantFunc plugin:** https://github.com/RealJonathanYip/ComfyUI-QuantFunc
  - config.json с тестовым ключом: `bin/<os>/config.json` в этом репо
- **ModelScope QuantFunc:** https://www.modelscope.cn/models/QuantFunc/Plugin
- **Репо моделей:** QuantFunc/Klein-9B-Series, QuantFunc/Klein-4B-Series,
  QuantFunc/Z-Image-Series, QuantFunc/Qwen-Image-Series, QuantFunc/Qwen-Image-Edit-Series
- **QuantFunc Discord:** https://discord.gg/jCp9TpFWcn
- **License server:** https://service.quantfunc.com
- **Reference HF Space:** https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast

---

## 11. ПРИОРИТЕТ для следующего исполнителя

**Шаг 1:** Запустить на Windows-машине с GPU, скачать engine, сгенерировать картинку
Z-Image Turbo (лёгкая, без ключа), затем Klein 4B/9B (с тестовым ключом из config.json).

**Шаг 2:** Если что-то не так с ключом — Settings → «Восстановить тестовый ключ»,
или свой ключ с quantfunc.com. Логи: смотреть `[engine]`/`[worker]` в консоли.

**Шаг 3:** Phase 3 — UI polish (Models/History/Templates tabs, SQLite history).

**Шаг 4:** Phase 4-6 — pipelines, больше моделей (Qwen-Image ~13.7 ГБ трансформеры —
учесть VRAM), ctypes upgrade.

---

## 12. Известные ограничения

1. Qwen-Image / Qwen-Image-Edit / Ideogram-4 — не в registry (добавляются JSON-файлом)
2. Только text-to-image — Image Edit / Img2Img / Inpaint не реализованы
3. History в RAM — не сохраняется между запусками
4. Templates не реализованы — только placeholder в sidebar
5. LoRA не поддерживается
6. No cancel для model download — только для inference
7. Single-user — нет concurrent job support
8. Тестовый ключ может перестать действовать после окончания public beta — тогда свой
   ключ с quantfunc.com (UI уже готов)
