# AGENT_HANDOFF.md — Полная база знаний и руководство для ИИ-агентов

> **Для любого ИИ-агента / бота (Gemini, Claude, GPT, GLM и др.):**  
> Этот документ содержит исчерпывающее описание архитектуры, кодовой базы, подводных камней, неочевидных багов и принятых инженерных решений в репозитории **LazySD-UI (QuantFunc-WebUI)**.  
> **Обязательно прочитай этот файл перед внесением любых изменений в код!**

---

## 1. О проекте: философия и концепция

**LazySD-UI** — локальный автономный легковесный веб-интерфейс для запуска квантованных диффузионных нейросетей и студия полутоновой обработки манги и скринтонов.

### Ключевые принципы:
1. **Никакого ComfyUI**: никаких запутанных спагетти-графов и нод. Все сложные пайплайны (3-этапная изоляция VRAM, Text Encoder ➔ Transformer ➔ VAE, слияние LoRA, CMYK-сепарация) зашиты в лаконичные бэкенд-воркеры.
2. **Никакого Gradio**: фронтенд написан на чистом нативном HTML5, CSS3 и ванильном JavaScript без тяжелых фреймворков и бандлеров.
3. **Две главные роли приложения**:
   - **Генератор диффузии**: запуск квантованных моделей (GGUF, SDNQ, Nunchaku/SVDQuant, OrbitQuant) с минимальным потреблением VRAM.
   - **Студия манги и скринтонов**: **Halftone Studio GT** (интеграция движка GenTonik: 13 форм точек, орбиты спутников, 4-канальный CMYK, пространственные градиенты, WebGL2 GPU + CPU) и **Manga Screen AI** (G-Pen инкинг, уровни, финишный растр).

---

## 2. Структура репозитория

```text
G:\QuantFunc-WebUI\
├── app.py                      # FastAPI бэкенд: REST API эндпоинты, маршрутизация, раздача статики
├── start.bat                   # Лаунчер Windows (создание venv, запуск uvicorn на 127.0.0.1:7860)
├── requirements.txt            # Python-зависимости (FastAPI, uvicorn, opencv, pillow, safetensors и др.)
├── AGENT_HANDOFF.md            # Этот файл: знания и архитектура для ИИ-разработчиков
├── CHANGELOG.md                # Журнал коммитов и изменений для людей и ИИ-агентов
├── README.md                   # Публичная документация для пользователей GitHub
│
├── backend/                    # Python бэкенд (модули логики)
│   ├── worker.py               # Оркестратор задач генерации, отмена процессов, JSON-sidecar
│   ├── generation_queue.py     # Глобальная FIFO-очередь GPU: один активный job + отмена
│   ├── presence.py             # Учёт подключённых аккаунтов (TTL 90 с, heartbeat)
│   ├── history_paging.py       # Пагинация и clamping страниц History API
│   ├── extras.py               # ExtrasWorker: интеграция с SVDQuant / Nunchaku (Flux.2, Flux.3, Tonera)
│   ├── sdnq_engine.py          # 3-этапный изолированный запуск SDNQ / Diffusers (Qwen-Image-2.1)
│   ├── sd_cpp.py               # stable-diffusion.cpp (GGUF-модели: Qwen, Flux.1, SDXL Turbo)
│   ├── halftone.py             # Ядро Halftone Studio GT на CPU: векторная математика (NumPy + OpenCV + PIL)
│   ├── manga_pipeline.py       # Manga Screen AI конвейер: контуры G-Pen, тональные уровни, финиш
│   ├── flowmatch_samplers.py   # FlowMatch сэмплеры (Euler, Euler A, Heun), интерполяция sample_sigmas
│   ├── gpu.py                  # Автоопределение GPU, VRAM и SM-архитектуры (Ampere, Ada, Blackwell sm_120)
│   ├── model_sniffer.py        # Сниффер локальных весов и автоопределение моделей/LoRA
│   ├── lora.py                 # Управление библиотекой LoRA, совместимость и скачивание
│   ├── custom_models.py        # Импорт пользовательских моделей и HF-инспектор
│   ├── model_files.py          # Безопасное удаление весов по манифесту (общие файлы защищены)
│   ├── security.py             # Валидация путей (Path Traversal protection) и безопасность
│   ├── settings.py             # Настройки приложения и авторизация аккаунтов (Хост / Гость)
│   └── registry/               # JSON-спецификации моделей и пресетов
│       ├── loader.py           # Загрузчик и валидатор ModelSpec
│       └── models/             # JSON-файлы описания поддерживаемых моделей
│
├── frontend/                   # Клиентская часть WebUI
│   ├── index.html              # Разметка интерфейса (вкладки: Генератор, Halftone GT, Manga AI, Хаб)
│   ├── css/
│   │   └── styles.css          # Все стили: IDE-сетка, тёмная тема, Gemini Dock, оверлеи масок
│   ├── locale/
│   │   ├── en.json             # Английская локализация (i18n)
│   │   └── ru.json             # Русская локализация (i18n)
│   └── js/
│       ├── api.js              # Обертки apiGet, apiPost, apiUpload, тосты, экранирование
│       ├── state.js            # Реактивный стейт приложения (App.state)
│       ├── main.js             # Модульная сборка ES6
│       ├── app.js              # АКТИВНЫЙ монолитный бандл, подключаемый в index.html (ВАЖНО!)
│       └── modules/            # Чистые ES6-модули:
│           ├── dock.js         # Gemini Dock, свободный перенос (drag), выбор энкодеров
│           ├── generator.js    # Параметры генерации, валидация версий Nunchaku, запуск задач
│           ├── halftone.js     # Halftone Studio GT (шейдер WebGL2, холсты масок, лассо, кисть)
│           ├── manga.js        # Manga Screen AI клиентская логика
│           ├── models-hub.js   # Модальное окно и вкладка Model Hub
│           ├── lora-hub.js     # Библиотека LoRA
│           ├── settings.js     # Настройки движков и аккаунтов
│           └── zoom-pan.js     # Зум колесом вокруг курсора, панорамирование, opacity HUD
│
├── models/                     # Папки хранения весов моделей
│   ├── cache/                  # Скачанные трансформеры и базовые модели
│   │   └── Qwen3-4B-text-Nunchaku # Общий канонический репозиторий текстового энкодера
│   ├── loras/                  # Пользовательские LoRA (.safetensors)
│   └── model_visibility.json   # Пользовательские флаги видимости/скрытия моделей
│
├── outputs/                    # Сгенерированные изображения и JSON метаданные
└── tests/                      # Набор тестов (pytest)
    ├── smoke_test.py           # Тесты безопасности, эндпоинтов API, регуляторов, GPU
    ├── test_es6_modules.py     # Тесты модульности frontend
    ├── test_halftone_and_manga.py # Тесты математики полутонов и манги
    ├── test_latest_fixes.py    # Интеграционные тесты последних исправлений
    ├── test_model_files_and_hf_layout.py # Безопасное удаление, общие файлы, Fibo folder-layout
    ├── test_model_card_and_lora_binding.py # Карточки кастомных моделей и явная привязка LoRA
    └── test_queue_presence_history.py # GPU-очередь, presence, пагинация History, Dock/UI-контракты
```

---

## 3. Архитектура бэкенда и движков генерации

### 3.1. Движок SVDQuant / Nunchaku (`backend/extras.py`)
- Запускается через подпроцесс Python в изолированном окружении `envs/extras` (или системном Python).
- **Поддерживаемые модели**:
  - `flux_2_klein_9b_nunchaku`, `flux_2_klein_4b_nunchaku`, `flux_2_klein_9b_kv_nunchaku`, `flux_2_klein_9b_orbitquant`.
  - `qwen_image_2_1_nunchaku`.
  - `flux_3_image_klein_nunchaku`.
- **FLUX.3 Nunchaku**: требует одновременно **Nunchaku >= 1.4.0** и **diffusers из GitHub main**
  (`Flux3Transformer2DModel`, `ModularPipeline`). Стабильный PyPI `diffusers 0.41.0`
  не подходит. Загрузка идёт через `Flux3DistilledModularPipeline`, локальный
  `modular_model_index.json`, INT4-энкодер `NunchakuQwen3VLEncoderModel`; блок
  `prompt_upsampler` отключается, пока WebUI не управляет отдельной 9B-моделью.
- **Логирование CLI**: перед запуском выводится информационная строка вида:
  `[INFO] [SVDQ] Launching NUNCHAKU: 1024x1024, steps=16, cfg=1.0, sampler=euler, eta=0.0, seed=...`
- **Отмена задачи**: вызов `Worker.cancel()` транслируется в `ExtrasWorker.cancel()`, который принудительно завершает подпроцесс (`proc.terminate()`), освобождая GPU VRAM.
- **Текстовые энкодеры**:
  - `flux_2_klein_*`: общий `tonera/Qwen3-4B-text-Nunchaku` в `models/cache/Qwen3-4B-text-Nunchaku`.
  - `qwen_image_2_1_*`: Qwen3-VL-8B INT4.
  - `flux_3_image_klein_nunchaku`: Qwen3-VL-4B INT4 (`NunchakuQwen3VLEncoderModel`).

### 3.2. Движок SDNQ / Diffusers (`backend/sdnq_engine.py`)
- Реализует 3-этапный запуск для генерации на картах с 16 ГБ VRAM:
  1. Загрузка Text Encoder ➔ генерация эмбеддингов ➔ удаление энкодера и `torch.cuda.empty_cache()`.
  2. Загрузка INT8 квантованного Transformer ➔ денойзинг ➔ удаление трансформера и `torch.cuda.empty_cache()`.
  3. Загрузка VAE ➔ декодирование латента в RGB.
- **Совместимость с Blackwell (RTX 50xx, `sm_120`)**:
  В `sdnq_engine.py` принудительно задано `SDNQ_USE_TORCH_COMPILE=0`, чтобы избежать падения Triton for Windows с ошибкой `3221225477` (Access Violation).

### 3.3. Движок SD.CPP (`backend/sd_cpp.py`)
- Обёртка над `bin/windows/sd_cpp/sd-cli.exe`.
- Применяет GGUF-модели (Qwen-Image-2.1, Flux.1 Schnell).
- Требует установленного `Visual C++ 2015–2022 Redistributable (x64)`.

### 3.4. GPU-очередь и лимиты CPU

> ⚠️ **Единственный воркер GPU — только через очередь.**
> Нельзя запускать `worker.generate(...)` из нового эндпоинта напрямую: сначала
> `STATE.register_job(..., serialized=True)`, затем в фоновом потоке
> `STATE.acquire_generation(job_id)`, а освобождение слота — в `finally` через
> `STATE.release_generation(job_id)`.

- `backend/generation_queue.py` — потокобезопасная FIFO-очередь с **одним
  активным** job. `enqueue()` возвращает 1-based позицию, `acquire()` блокирует
  поток до его очереди, `cancel()` убирает ожидающий job или сообщает, что job
  уже активен, `release()` освобождает слот.
- Сериализованы ровно две GPU-точки: `POST /api/generate` (универсальный
  генератор) и `POST /api/manga/ai-job` (AI-этап Manga). Halftone Studio,
  `/api/manga/process`, скачивания моделей и LoRA не входят в GPU-очередь.
- `AppState` хранит для job поля `owner_id`, `owner_name`, `kind`, `shared`,
  `serialized`, `queue_position`, `queue_size`, `queue_active`.
  `_sync_queue_positions()` зеркалит состояние очереди в job metadata.
  - `queue_position == 0` + `queue_active == true` — job владеет GPU.
  - `queue_position >= 1` — job ждёт; `queue_size` считает только ожидающих,
    без активного.
- Отмена ожидающего job удаляет его из `_pending` и ставит маркер `_cancelled`;
  его поток при `acquire()` увидит маркер и завершится, **не трогая чужой
  активный worker**. Отмена активного job возвращает `was_active=True`, после
  чего эндпоинт вызывает `Worker.cancel()`.
- Если cancel приходит между активацией слота и записью `INITIALIZING`,
  `acquire_generation()` не перезаписывает `CANCELLED`, освобождает слот и
  возвращает `False`; run_job дополнительно проверяет статус перед
  `worker.generate()`.
- `update_job()` не перезаписывает `JobStatus.CANCELLED` поздним
  `error/complete` от остановленного воркера.
- CPU-тяжёлые Halftone/Manga-пайплайны ограничены общим
  `TOOL_CPU_MAX_CONCURRENCY = 4` (`_TOOL_CPU_SEMAPHORE`): одновременно максимум
  4 задачи, но они не блокируют GPU-очередь.
- Фронтенд: глобальный `#queue-overlay` (до `</body>`, поэтому виден на всех
  вкладках) показывает позицию и кнопку отмены. Первый poll решает, показывать
  ли оверлей: при `queued` — показать, при `initializing/loading/generating` —
  скрыть. Не показывай очередь до ответа сервера, иначе будет ложная вспышка
  «GPU busy» у job, который сразу получил слот.

---

## 3.5. Архитектура Inpaint Studio и режимы обрезки (`backend/masking.py`)

Inpaint Studio позволяет перерисовывать фрагменты изображения с помощью маскирования:
1. **Полярность маски**: единая во всех движках — **белый (255)** = область перегенерации, **чёрный (0)** = неизменная исходная область.
2. **Растушёвка (`feather_mask`)**: расширение (дилатация) и размытие по Гауссу краёв маски для мягкого и естественного перехода между сгенерированной областью и оригиналом.
3. **Локальный кроп (`resolve_crop_box`)**:
   - Вместо прогона всей многомегапиксельной картинки через диффузию, бэкенд обрезает кадр и маску до прямоугольной области (`crop_box`), выполняет инференс на компактном разрешении (кратно 16px, минимум 256px) и затем накладывает сгенерированный фрагмент обратно в оригинал через маску.
   - За пределами маски каждый пиксель остаётся побитово идентичным оригиналу.
4. **Три режима Crop Mode**:
   - `auto`: bounding box маски с паддингом 64px и min 256px.
   - `parameters`: пользовательский паддинг (0-1024px) и min_side (64-4096px).
   - `manual`: произвольный прямоугольник, нарисованный пользователем поверх изображения. Обязан полностью перекрывать маску (проверяется на бэкенде).

---

## 4. Ядро Halftone Studio GT & Скринтоны (`backend/halftone.py`)

Движок скринтонов обеспечивает **100% WYSIWYG-паритет** между шейдером WebGL2 в браузере и CPU-рендером на Python.

### 4.1. Геометрические формы растра (13 форм)
Функции `evalShapeDist` (в GLSL) и `_eval_shape_distance` (в Python) используют идентичные формулы расстояния в нормализованных координатах ячейки `du, dv ∈ [-0.5, 0.5]`:
- `0: circle` — евклидово расстояние $\frac{\sqrt{u^2 + v^2}}{R \cdot 1.12838}$
- `1: square` — расстояние Чебышёва $\frac{\max(|u|, |v|)}{R}$
- `2: diamond` — манхэттенское расстояние $\frac{|u| + |v|}{R \cdot \sqrt{2}}$
- `3: line` — параллельные линии $\frac{|v|}{R}$ (линейный радиус)
- `4: crosshatch` — сетка $\frac{\min(|u|, |v|)}{R}$ (линейный радиус)
- `5: hexagon` — правильный шестиугольник / соты GT
- `6: cross` — крест GT
- `7: ellipse` — эллипс с деформацией осей
- `8: triangle` — равносторонний треугольник
- `9: star` — 5-конечная звезда с полярной модуляцией
- `10: heart` — кардиоида / сердце
- `11: concentric` — концентрические кольца (линейный радиус)
- `12: ring` — одиночное кольцо

### 4.2. Спутники и Планеты (GenTonik Orbit Engine)
Вокруг каждой базовой точки («планеты») генерируется от 1 до 8 спутников:
- `satellite_distance`: радиус орбиты от центра ячейки.
- `satellite_size_ratio`: размер спутника относительно планеты.
- `satellite_stretch`: направленное вытягивание спутника вдоль вектора к планете.
- `merge_factor`: динамическое разрастание и слияние точек при росте плотности тона.
- **Субпиксельный антиалиасинг (AA)**: в шейдере используется `fallbackFw = cellPx / max(satMaxR, 0.02)` для предотвращения исчезновения производных `fwidth()` внутри циклов.

### 4.3. 4-канальный CMYK
Субтрактивное растрирование с офсетными углами: Cyan (15°), Magenta (75°), Yellow (0°), Black (45°). Линии контура (`preserve_lines`) накладываются поверх всех 4 каналов.

### 4.4. Декодирование маски в Python (`decode_mask_to_float`)
> **Критический нюанс:**  
> Маска из браузера отправляется как RGBA PNG (`toDataURL`). Прозрачные пиксели имеют `rgba(0,0,0,0)`.  
> Если просто вызвать `Image.open().convert("L")`, Pillow проигнорирует альфа-канал!  
> В `decode_mask_to_float` реализован учёт обоих каналов:
> ```python
> arr = np.array(opened_img.convert("RGBA"), dtype=np.float32) / 255.0
> rgb_lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
> alpha_chan = arr[:, :, 3]
> m_f = np.maximum(rgb_lum, alpha_chan)
> ```

### 4.5. ROI-рендер и экспорт (`POST /api/halftone/process`)

- `render_halftone_screen(..., roi=(x, y, w, h))` рендерит только видимую область
  источника. Маска обрезается тем же прямоугольником (`_crop_mask_to_roi`), а
  фаза растра сохраняется через `_pattern_frame_for_roi`: в кадр передаются
  `offset_x/offset_y`, поэтому точки в ROI лежат ровно там же, где в полном
  рендере. Без этого при зуме растр «прыгал» бы относительно базового превью.
- `roi` валидируется в `app.py`: координаты клампятся к источнику, нулевые
  ширина/высота отклоняются. `preview_max_side` ограничивает рабочую сторону
  ROI; если ROI меньше лимита, он рендерится в нативном разрешении.
- `save: true` **игнорирует** `roi` и принудительно ставит
  `preview_max_side = 0`. Сохранение в галерею, скачивание и передача в Manga
  всегда пересчитывают полный кадр из оригинала, а не берут live-превью.
- Ответ содержит фактический `roi` (после клампа) — фронтенд позиционирует по
  нему оверлей, не полагаясь на запрошенный прямоугольник.
- Галерея/History передают `/outputs/...` как `image_path`, а не как
  `image_data`; иначе base64 раздувает запрос и теряется исходное разрешение.

---

## 5. Фронтенд: двойная архитектура и правила синхронизации

> ⚠️ **КРАЙНЕ ВАЖНО: ДВОЙНОЙ ФАЙЛ JS**  
> В текущей версии `frontend/index.html` (строка 3904) подключает монолитный скрипт:  
> `<script src="/js/app.js?v=..."></script>`.  
> В то же время в проекте существует набор современных ES6-модулей в `frontend/js/modules/` (`halftone.js`, `dock.js`, `generator.js` и т.д.).  
> **ПРАВИЛО ДЛЯ АГЕНТА:** Любые исправления в функциях студии, генератора, дока или масок **ОБЯЗАТЕЛЬНО ДОЛЖНЫ ВНОСИТЬСЯ И В `frontend/js/app.js`, И В `frontend/js/modules/`!**  
> Перед завершением работы обязательно проверяй оба файла через `node --check`.

### 5.1. Иерархия Z-Index и Pointer Events на холсте
Внутри `#ht-preview-wrap` размещены 5 слоёв:
1. `<img>` (`#ht-preview-img`) — базовое превью CPU (`z-index: 1`).
2. `<canvas id="ht-live-canvas">` — живой рендер WebGL2:
   `position: absolute; inset: 0; z-index: 2; pointer-events: none;` (**pointer-events ДОЛЖЕН БЫТЬ NONE**, иначе он перехватывает клики у маски!).
3. `<img id="ht-roi-img">` (`.ht-roi-img`) — детальный ROI-оверлей CPU:
   `position: absolute; z-index: 3; pointer-events: none;` Показывается только в
   CPU-режиме (`qualityMode === "full"`) при зуме ≥ `1.08`, не в compare.
4. `<canvas id="ht-mask-canvas">` — оверлей маски (рубиновый полупрозрачный слой):
   `position: absolute; inset: 0; z-index: 15;`
   - `pointer-events: auto` в режимах `brush` и `eraser`.
   - `pointer-events: none` в режимах `view` и `lasso`.
5. `<canvas id="ht-lasso-overlay">` — оверлей пунктирного контура лассо:
   `position: absolute; inset: 0; z-index: 20;`
   - `pointer-events: auto` в режиме `lasso`.
   - `pointer-events: none` в остальных режимах.

### 5.2. Координаты маски и размер холстов
- Элементы `<canvas id="ht-mask-canvas">` и `<canvas id="ht-lasso-overlay">` по умолчанию создаются размером 300×150.
- Функция `syncHalftoneMaskCanvasSize(w, h)` обязана вызываться при загрузке картинки (`loadHalftoneSource`, `prepareHalftoneLiveSource`, `runHalftonePreview`) и выставлять `width` и `height` холстов равными оригинальным размерам изображения (`origWidth × origHeight`).

### 5.3. Загрузка текстуры маски в WebGL2
- В вершинном шейдере ось Y уже перевёрнута (`v_uv.y = 1.0 - v_uv.y`).
- Поэтому при вызове `gl.texImage2D` для `glMaskTexture` параметр `UNPACK_FLIP_Y_WEBGL` **ОБЯЗАТЕЛЬНО ДОЛЖЕН БЫТЬ `false`**:
  `gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);`  
  (Если поставить `true`, маска будет наложена вверх ногами, а глобальное состояние WebGL перевернёт последующие текстуры!).

### 5.4. Качество превью, адаптивный CPU/GPU и ROI

Превью Halftone не должно рендерить полный кадр на каждый чих. Есть **два
независимых переключателя**:

- `qualityMode` — движок: `full` = CPU Canvas 2D, `live` = WebGL2 GPU.
- `previewQuality` — качество превью: `auto` (по умолчанию), `performance`,
  `maximum`.

Значения `previewQuality` (кнопки `#ht-preview-quality-*`):

| Режим | CPU (`qualityMode="full"`) | GPU (`qualityMode="live"`) |
|---|---|---|
| `auto` | база: `max(fitW, fitH) × devicePixelRatio` (мин. 256 px); при зуме — детальный ROI | `fit × zoom × DPR`, но не больше `max(sourceMax, 2048)` и лимитов GPU |
| `performance` | `min(sourceMax, 1920)` | `min(sourceMax, 1920)` — это **потолок, не апскейл** |
| `maximum` | полное разрешение источника (`preview_max_side = 0`) | `sourceMax`, ограниченный только GPU |

- **CPU Auto + зум.** Базовое превью остаётся дешёвым и растягивается
  (`image-rendering: pixelated` при фактическом масштабе ≥ 1.15), а поверх него
  запрашивается ROI видимой области. ROI считается в координатах источника:
  видимый прямоугольник вьюпорта переводится в пиксели обёртки, затем в пиксели
  источника, с запасом `HT_ROI_MARGIN = 14%`. Порог — `HT_ROI_MIN_ZOOM = 1.08`.
  `preview_max_side` для ROI = `max(viewportW, viewportH) × DPR × 1.35`; если ROI
  меньше этого лимита, он приходит в нативном разрешении и браузер его только
  увеличивает — это честная детализация, а не пересчёт экранного приближения.
- **Позиционирование ROI.** `#ht-roi-img` лежит внутри той же
  `#ht-preview-wrap`, поэтому наследует её transform. Смещение и размер
  считаются как `rect.x / sourceW × fitW` и т.д. Не добавляй ROI отдельный
  transform/скролл — он разъедется с базой при пан/зуме.
- **Асинхронность.** У ROI свои `_roiAbort`, `_roiSeq`, `_roiTimer` и
  `_roiSignature`. Подпись включает `x,y,w,h | _editRevision | previewQuality`,
  поэтому кэш автоматически инвалидируется при правке параметров, кисти или
  смене качества. Старый ответ не должен перезаписывать новый — всегда
  сравнивай `seq` после `await`. `setHalftonePreviewQuality()` очищает
  `roiDataUrl/roiRect/_roiSignature` и скрывает оверлей.
- **GPU sizing.** `resizeHalftoneGpuCanvas()` считает целевой максимум по
  формуле режима, затем клампит к `MAX_TEXTURE_SIZE`, `MAX_RENDERBUFFER_SIZE` и
  `MAX_VIEWPORT_DIMS` (`getHalftoneGpuLimits`). В Auto/Performance размер
  округляется до 64 px. Никакого искусственного потолка 2048 нет: 4K-источник
  в Auto растёт до `fit × zoom × DPR` и упирается только в железо. При
  Performance 1536-источник остаётся 1536 (никакого апскейла до 1920).
- **Маски.** На каждом слое есть `_maskRevision`, `_maskDataUrl`,
  `_maskDataUrlRevision` и `_glMaskRevision`. Data URL маски кэшируется и
  пересобирается только при росте ревизии; GPU-текстура маски один раз
  даунскейлится до `HT_GPU_MASK_MAX_SIDE = 2048` на ревизию. Не вызывай
  `toDataURL()` маски в каждом кадре — это главный источник фризов кисти.
- **AA.** В live-превью AA клампится до 2 (`AA2`). `AA4` допустим только в
  экспорте: `buildHalftoneParamsPayload(st, forPreview)` и
  `buildHalftoneLayersPayload(st, forPreview)` не должны резать параметры при
  `forPreview = false`.
- **Экспорт.** Сохранение/скачивание/передача в Manga вызывают backend с
  `save: true` (для download/handoff — `preview_max_side: 0`), поэтому всегда
  получают полный кадр исходного разрешения. Не экспортируй `ht-preview-img`
  или `resultDataUrl`: там adaptive-превью.

#### Ловушки адаптивной вёрстки

- `#ht-preview-wrap` — единственный блок с абсолютно позиционированными
  слоями. Правила `#ht-preview-wrap img { position:absolute; inset:0; ... }`
  **нельзя** переносить на общий `.studio-preview-wrap`: Manga использует тот
  же класс и ожидает обычный `img` в потоке (иначе картинка обрезается/прыгает).
- Ширина `#ht-meta-info` фиксирована (`190px`). Текст «CPU 123 ms · 568×379 px»
  не должен менять высоту тулбара. Если убрать фиксацию, `ResizeObserver`
  увидит изменение `#ht-dropzone`, перезапустит превью, снова поменяет
  метку — и получится бесконечный цикл, который обрывает ROI-запросы.
- `scheduleHalftoneLayout()` обязан выходить, если `fitWidth/fitHeight` не
  изменились. Оверлей «⏳ Обработка...» показывай только когда реально показан
  loading-overlay, и снимай его в `finally` активного превью — иначе после
  отменённого `maximum`-рендера он залипает.
- `#ht-preview-wrap` задаёт явные `width/height` из `fit size`; `flex: 0 0 auto`
  и `contain: layout paint` обязательны, иначе wrapper растягивается под
  контейнер и координаты ROI/маски уезжают.

---

## 6. Плавающий Gemini Dock, подключения и History

### 6.1. Свободный перенос Dock (Drag)

- Расположен в `#gemini-dock-wrap`; в свободном режиме получает класс
  `.dock-free-drag` и inline `left/top`.
- **Drag — отдельный явный режим**, а не поведение всей коробки:
  - кнопка `#dock-drag-toggle-btn` (✥) в доке и
    `#setting-dock-drag-toggle` в настройках вызывают `App.toggleDockDrag()`;
  - состояние `dockDragEnabled` сохраняется в
    `localStorage["lazysd_dock"]` вместе с `position`, `opacity`, `blur`,
    `minimized`, `customX`, `customY`.
- Перетаскивание начинается **только** с ручки
  `<div id="dock-drag-handle">⋮⋮</div>` и только при включённом Drag.
  Кнопки, prompt-поля и поповеры больше не конкурируют за pointer-события.
- При `pointermove` обновляются обе оси:
  ```js
  nx = clamp(origX + dx, 8, innerWidth  - dockWidth  - 8);
  ny = clamp(origY + dy, 8, innerHeight - dockHeight - 8);
  ```
  `wrap.style.transform = "none"; bottom/right = "auto"` — иначе CSS-позиция
  перебивает inline-координаты.
- **Историческая ловушка:** в `.gemini-dock-wrap.dock-free-drag` был
  `left: auto !important`. Он отменял `${customX}px`, из-за чего док двигался
  только влево и по вертикали. Этот `!important` удалён; не возвращай его.
- `App.setDockPosition(pos)` (пресеты center/left/right/bottom) выключает Drag и
  очищает свободные координаты. `App.resetDockPosition()` и двойной клик по
  ручке сбрасывают в нижний центр (`dockDragEnabled=false`, `dockPosition=center`,
  `dockCustomX/Y=null`).
- При повторном включении Drag возвращается последняя сохранённая свободная
  позиция (если она есть), затем клампится к текущему окну.
- `frontend/js/app.js` — активная реализация; `frontend/js/modules/dock.js`
  обязана повторять те же id/методы и CSS-класс.
- Поповеры Dock (`#dock-popover-model|res|params|presets`) обязаны быть
  **вне normal flow**: `.dock-popover` имеет `position:absolute;
  bottom:calc(100% + 8px); left:0`. Иначе в drag-режиме обёртка привязана к
  `top`, и открытие поповера увеличивает её высоту, сдвигая саму панель вниз.
  Для положения у верхней границы `toggleDockPopover()` добавляет
  `.dock-popover-below` (`top:calc(100% + 8px)`); геометрия Dock при этом
  не меняется.
- Сортировка текстовых энкодеров: установленные всегда первыми
  (`te.installed ? -1 : 1`) и выбираются по умолчанию.

### 6.2. Мониторинг подключённых пользователей (presence)

- `backend/presence.py` хранит уникальные аккаунты по ключу `role:id`; один
  пользователь в нескольких вкладках считается одним подключением. Гости с
  разными id считаются отдельно.
- `ShareAccessMiddleware` вызывает `record_activity(caller)` на `/`, `/index.html`
  и `/api/*`. Запись считается онлайн, пока `last_seen` не старше
  `STALE_AFTER_SECONDS = 90`.
- `GET /api/presence` и `POST /api/presence/heartbeat` доступны и гостям.
  Гость получает только `{enabled,count,users:[],stale_after}`, хост —
  полный список. Токены/пароли не сохраняются.
- Фронтенд: heartbeat каждые 20 с, индикатор `#presence-indicator` в шапке
  виден всегда (кроме выключенного мониторинга), настройка
  `monitor_connections` — в Settings. Список пользователей рендерится только
  для хоста.
- При выключении мониторинга registry очищается; при включении отсчёт
  начинается заново.

### 6.3. History: пагинация, фильтры и защита от зависаний

- `GET /api/history`:
  - без query-параметров возвращает **старый массив** (обратная совместимость);
  - с `page`, `page_size`, `filter` или `q` возвращает
    `{items,total,page,page_size,pages}`.
- `page_size` по умолчанию 24, UI предлагает 12/24/48; сервер принимает
  1…100 (FastAPI `Query(ge=1, le=100)`). `backend/history_paging.py` клампит
  `page` к последней странице, поэтому удаление последней работы на странице
  не оставляет пустую/ошибочную страницу.
- Фильтры: `all`, `mine`, `guests`, `shared`; поиск `q` идёт по prompt,
  model_id, filename и owner_name. Гостевая изоляция по `owner_id/shared`
  применяется до пагинации.
- `CacheManager.list_outputs(limit=None)` возвращает все выходные файлы,
  отсортированные по mtime (новые первыми).
- Фронтенд: `historyRequestSeq` отбрасывает устаревшие ответы при быстром
  переключении страниц; `finally` снимает `history-loading`. Без этого старый
  запрос мог завершиться позже нового и подвесить/перерисовать страницу.
- Не вызывай `loadHistory()` повторно из рендера и не убирай `finally`:
  именно эти две вещи приводили к зависаниям.
- `App._fetchWithRetry()` повторяет запрос только для HTTP `502/503/504` и
  сетевых ошибок: `apiGet` — 2 раза, превью Halftone (`POST
  /api/halftone/process` с `save:false`) — 1 раз. Мутирующие POST
  `/api/generate` и `/api/manga/ai-job` **не повторяются**, иначе можно создать
  дубликат GPU-задачи. Ошибка `Bad Gateway` при живом сервере означает, что
  запрос не дошёл до uvicorn: сначала проверь процесс на порту и access-log.

### 6.4. Models Hub: фильтры, поиск и текстовые энкодеры

- Активная разметка — `#models-tab-filters`; у кнопок атрибут
  `data-tab-filter`, а не `data-filter`. Старый селектор
  `#models-hub-filters .filter-pill` не существует и делал фильтры
  неработающими.
- `App.setModelsTabFilter(filter, btnEl)` выставляет
  `state.modelsTabFilter`, обновляет `.active` по `data-tab-filter` и вызывает
  `renderModelsTab()`. `onModelsTabSearch()` обновляет
  `state.modelsTabSearch`.
- `renderModelsTab()` теперь `async`; для `encoders` она уходит в
  `renderEncodersTab()` и `/api/encoders`. Остальные фильтры:
  - `installed` — `model.installed === true`;
  - `custom` — `model.is_custom === true`;
  - `hidden` — `hidden_from_host` **или** `hidden_for_guests`;
  - `svdq` — движок Nunchaku/SVDQ или SVDQ-трансформер;
  - `sd_cpp` — `engine === "sd_cpp"` или GGUF;
  - `sdnq`, `klein`/Flux.2, `qwen`, `z-image` — по engine/family/id.
- Поиск идёт по id, name, family, engine, description, текстовым энкодерам и
  id трансформеров, поэтому запрос `klein` находит не только модели с Klein в
  названии.
- `renderEncodersTab()` показывает 13 общих энкодеров с признаком установки,
  размером и кнопкой `App.downloadSharedEncoder()`; после завершения
  скачивания кэш `state.encoders` сбрасывается, вызывается `refreshConfig()`.
- `frontend/js/modules/models-hub.js` должен повторять фильтры и селектор
  активной кнопки; это проверяется статическим тестом.

### 6.5. Безопасное удаление весов и панель загрузок

- Источник истины при удалении — **манифест модели**, а не каталог репозитория.
  Логика живёт в `backend/model_files.py`:
  - `model_file_refs(model)` собирает все `(repo_id, relative_path)` из
    `transformers`, `text_encoders`, `engine_args.vae*` и `aux_files`. Для VAE
    добавляются оба пути: source (`vae_repo`/`vae_remote`) и локальный target
    (`primary_repo`/`vae`), иначе скачанный в primary-репозиторий VAE не
    защищался бы от удаления другой моделью.
  - `collect_delete_targets(model, all_models, cache, variant_id=None)` сначала
    собирает ссылки **всех других** моделей в `protected`. Совпавшие файлы не
    удаляются, а попадают в `skipped_shared`. При удалении одного варианта
    sidecar-файлы `orbitquant_*`/`svdq_*` добавляются только если в его папке
    нет другого варианта.
  - `delete_targets()` раскрывает шарды `-00001-of-0000N`, удаляет
    `.incomplete`, проверяет `resolve().is_relative_to(cache.cache_dir)` и
    убирает только пустые каталоги. Никакого `shutil.rmtree` по репозиторию.
- `DELETE /api/models/{model_id}/files?variant_id=...`:
  - гость → 403, неизвестный вариант → 404, модель без HF-источника → 400;
  - ответ: `deleted_files`, `deleted_files_count`, `freed_bytes`, `freed_mb`,
    `removed_dirs`, `protected_shared_count`, `skipped_shared`, `message`.
- `backend/custom_models.py:delete_custom_model(..., delete_files=True)` строит
  `SimpleNamespace`-адаптер из JSON и вызывает те же helpers. Удалять каталог
  репозитория нельзя: в нём могут лежать общий энкодер, VAE, sidecar'ы или
  варианты другой модели. Это проверяет `test_custom_model_delete_uses_safe_helpers`.
- UI: кнопки `.btn-download` / `.btn-delete-files` — настоящие `<button>` с SVG;
  установленный вариант получает `.variant-state installed` → «На диске» и
  отдельную кнопку удаления. `<details id="downloads-panel">` всегда стартует
  закрытой, open-state не сохраняется (localStorage больше не используется),
  показывает счётчик, прогресс, отмену и «Очистить завершённые»
  (`App.registerDownload`, `App.renderDownloads`, `App.clearFinishedDownloads`).
  `App.deleteModelFiles(modelId, variantId, btnEl)` подтверждает удаление,
  показывает `skipped_shared`/`freed_mb` и вызывает `refreshConfig()`.

### 6.6. HF-импорт folder-layout релизов (OrbitQuant/SVDQ)

- `backend/model_sniffer.py` различает `layout=folders` и
  `has_component_configs`. Если есть `transformer/config.json` и
  `vae/config.json`, релиз считается самодостаточным даже без корневого
  `model_index.json`; `missing` не должен содержать `base`. Вместо этого UI
  предупреждает о кастомном pipeline-коде.
- `pipeline_code` (корневые `.py`, например `fibo2_orbitquant.py`) — только
  предупреждение. Импорт скачивает `.py` и `fonts/**` для полноты, но никогда
  их не импортирует и не исполняет.
- `inspect_hf_repo()` отдаёт `pipeline_tag`, нормализованный `base_model`,
  `license`, `weight_size_gb`, `has_component_configs`, `pipeline_code`;
  инспектор показывает meta-chips, дерево папок и группы TE/VAE/aux.
- Импорт обязан скачивать декларированные sidecar-файлы
  `transformer/orbitquant_*.safetensors` и `transformer/svdq_*`: без них
  квантованный трансформер не собрать. Они также удаляются вместе с вариантом.
- Сгенерированный манифест пишется в
  `backend/registry/models/custom_hf_<...>.json` и по умолчанию остаётся
  неотслеживаемым пользовательским артефактом. Не коммить манифест модели,
  runtime которой ещё не поддерживается: пользователь скачает гигабайты и
  получит ошибку движка.

### 6.7. Редактируемые карточки и скачивание компонентов

- Редактировать и **удалять** можно все модели реестра, включая встроенные:
  `_find_custom_manifest()` больше не отказывает манифестам без `is_custom`,
  `GET/PATCH /api/models/{id}/card` возвращают `editable: true`. Гость
  получает 403. `delete_custom_model()` больше не содержит
  `protected_system_models`: удаление встроенной карточки убирает её манифест
  из `backend/registry/models/`, а `delete_files=true` дополнительно удаляет
  только файлы из манифеста (общие TE/VAE/aux защищены).
- `PATCH /api/models/{id}/card` принимает поля `name`, `description`, `family`,
  `pipeline`, `engine`, `steps`, `cfg`, `license`, `base_repo_id`, `tags`,
  `text_encoders`, `vae`, `aux_files`. Валидация в
  `backend/custom_models.py:_apply_card_updates()`:
  - `steps` 1–150, `cfg` 0–30, известный `engine`, `pipeline`;
  - репозитории нормализуются к `author/repo`;
  - пути компонентов только относительные, без `..`, `:` в первом сегменте и
    выхода за пределы репозитория;
  - TE/VAE/aux принимаются как `{repo_id, path}` или как вложенный
    `source.file_path` — не заставляй пользователя дублировать путь.
- Сохранение атомарное: `*.tmp` → `replace`, затем `reload_registry()` и
  `get_registry().get(model_id)`. При любой ошибке файл восстанавливается из
  побайтового бэкапа и реестр перезагружается. Не сохраняй манифест напрямую
  через `open(..., "w")` без этой проверки.
- `engine_args` — производное поле. При сохранении TE пересобирает
  `llm/llm_repo/llm_file`, VAE — `vae/vae_repo/vae_file/vae_remote`, aux —
  `aux_files/aux_repo`. Если список очищен, старые ключи удаляются; иначе
  движок продолжит искать удалённый файл.
- `_reconcile_engine_args()` при смене `engine` переносит основной вес в
  `diffusion_model` (SDNQ/Nunchaku/OrbitQuant), в `checkpoint` для SDXL
  (SD.CPP) или `diffusion_model` для остальных SD.CPP-моделей, удаляет
  устаревшие ключи и гарантирует `base_models` для Extras-движков. Без этого
  карточка после переключения движка продолжала бы искать старый checkpoint.
- VAE-резолвер живёт в `backend/model_files.py`:
  - `vae_payload()` / `vae_payload_from_args()` возвращают `repo_id`, `path`,
    `installed`, `local_repo_id`, `local_path`, `inferred` и (при найденном
    файле) `resolved_repo_id`/`resolved_path`;
  - установленность проверяется по primary-репозиторию и отдельному
    VAE-репозиторию: `vae_remote`, `vae_file`, `vae`, basename в корне,
    `vae/<basename>`;
  - если VAE в манифесте не объявлен, но в папке модели есть
    `vae/*.safetensors` (или `*vae*.safetensors`/`ae.safetensors` в корне),
    возвращается inferred-VAE. Не отключай inference: без него старые
    манифесты снова начнут показывать «не указан VAE», хотя файл уже скачан;
  - `vae_download_refs()` отделяет source (`vae_repo`/`vae_remote`) от
    локального target (`vae`/`vae_file` в primary-репозитории).
- `POST /api/models/{id}/component-download` скачивает ровно один компонент:
  `role=text_encoder` + `id`, `role=vae`, или `role=aux` + `index`. Endpoint
  повторно санитизирует и source-, и local-путь (traversal, абсолютный путь,
  `:`), проверяет локальный `destination.is_file()` и возвращает
  `already_installed`, иначе запускает фоновую job с прогрессом. Для VAE
  источник берётся из `vae_repo`/`vae_remote`, а файл кладётся в
  `primary_repo/vae` (или `vae_file`) — как в `backend/sd_cpp.py`; если
  source-копия уже есть, она копируется без сети. Гость получает 403.
- `POST /api/models/{id}/component-delete` удаляет один TE/VAE/aux:
  `{role, index, id?, force?}`. `POST /api/encoders/delete` удаляет общий
  энкодер: `{repo_id, file_path, file?, id?, force?}`. Оба endpoint'а сначала
  считают ссылки **других** моделей (`split_shared_refs`) и при пересечении
  возвращают `status="shared"` + список файлов, ничего не удаляя. Только
  повторный запрос с `force=true` удаляет общий файл. Inferred repo-VAE
  (`vae_component_refs(..., cache)`) тоже можно удалить, даже если он не
  объявлен в манифесте.
- `/api/models` и `/api/models/{id}/state` обязаны отдавать TE с `repo_id`,
  `path`, `precision`, `size_gb`, `installed`, VAE с `repo_id`, `path`,
  `installed`, `inferred`, `local_path`, и полный `aux_files`. Раньше Fibo
  показывал только transformer, потому что TE/VAE искались лишь в
  `engine_args.llm*`, а не в repo-local путях манифеста.
- UI: секции карточки — `<details class="model-card-section">`
  (Transformers / TE / VAE / Aux), у компонентов кнопки
  `App.downloadModelComponent()` и `App.deleteModelComponent()`. Секция VAE
  рендерится у каждой карточки, даже если VAE не объявлен (тогда — подсказка
  открыть редактор). Кнопки «Редактировать» и «Удалить» доступны всем, кроме
  гостя; гость не видит и «Видимость». Вкладка «Энкодеры» показывает Delete
  для скачанных и Download для отсутствующих (`App.deleteSharedEncoder`).
- Все `<details>` (инспектор, карточки, редактор карточки, downloads-panel)
  по умолчанию закрыты и не сохраняют open-state: пользователь открывает то,
  что ему нужно. Не возвращай `open` в разметку и не добавляй localStorage
  для аккордеонов.
- Модалки управления моделями — `#model-card-editor-modal`,
  `#component-picker-modal`, `#model-visibility-modal`,
  `#custom-model-modal`; состояние редактора лежит в
  `App.state.modelCardEditor` и `App.state.componentPicker`. Весь новый UI
  (карточки, редактор, picker, видимость, энкодеры, движки) использует
  `data-i18n`/`t()` и ключи `models.*`/`settings.engine.*`; не добавляй
  русские строки прямо в JS.
- Выбор компонента: HF не умеет искать репозитории по роли файла. Поэтому
  сначала ищется репозиторий (`/api/hf/search`), затем инспектируется его
  дерево, а найденные файлы фильтруются по TE/VAE/aux. Не обещай фильтр
  «только энкодеры» на уровне поиска репозиториев.
- При изменении `frontend/js/app.js` обновляй `?v=` у `<script>` в
  `frontend/index.html`; иначе браузер может показать старую карточку и
  спрятать новые кнопки/секции.

### 6.8. Управление движками: Install / Update / Delete

- В Settings → «Движки» у SD.CPP, SDNQ, Nunchaku и OrbitQuant есть три
  действия: Install (повторная установка), Update (переустановка/обновление)
  и Delete. У QuantFunc Delete нет — его бинарник и кейс обновляются отдельно.
  Кнопки `.setting-btn.danger` локализованы (`settings.engine.*`).
- `POST /api/engine/uninstall` с `{engine}`:
  - `sd_cpp` — удаляет каталог движка (`get_sd_cpp_dir()`);
  - `sdnq` — удаляет локальный source (`get_local_sdnq_dir()`) и пытается
    `pip uninstall sdnq` в найденном окружении;
  - `nunchaku` / `orbitquant` — только `pip uninstall` в управляемом
    `envs/extras` venv.
  Внешний `extras_python` не модифицируется: endpoint возвращает отказ, чтобы
  не сломать чужой SD.Next/venv. Удаление движка не трогает кэш весов и
  карточки моделей.
- Extras install/update принимает `force`: Nunchaku переустанавливает wheel с
  `--force-reinstall --no-deps`, OrbitQuant — `pip install --upgrade`.
  Общий `envs/extras` (torch/diffusers) при удалении компонента сохраняется.
- После uninstall/установки UI обновляет `refreshConfig()` и, для
  Nunchaku/OrbitQuant, `refreshExtrasStatus(false)`.

### 6.9. Явная привязка LoRA к моделям

- Модельно-независимая LoRA — неправильная модель поведения. В метаданных
  `.loras_meta.json` хранятся `architecture`, `compatible_models`, `engine`,
  `target_modules`, `rank`, `scale_default`, `trigger`, `base`, `repo_id`.
- `check_lora_compatibility()` соблюдает порядок:
  1. `compatible_models` — точная привязка; совпадение даёт `match_level=exact`
     и перекрывает несовпадение family;
  2. если привязка есть, но модели в ней нет — LoRA несовместима, даже когда
     family совпадает (не возвращай старый family-heuristic раньше привязки);
  3. `engine` — allow-list движков;
  4. `architecture`/`family` — только legacy-fallback для LoRA без привязки.
- `suggest_model_bindings(base_model)` сопоставляет HF `base_model` с
  `sources.primary.repo_id`, `base_repo_id` и `upstream_model`. Импорт LoRA с
  HF автоматически заполняет `compatible_models`, а `info["suggested_models"]`
  возвращается в ответе.
- `GET /api/loras?model_id=...` добавляет каждому элементу `compatibility`
  (`compatible`, `supported_by_engine`, `match_level`, `reason`,
  `bound_models`, `bound_engines`). Без `model_id` поле не добавляется —
  старые клиенты должны продолжать работать.
- `PATCH /api/loras/{name}/metadata` — host-only. `update_lora_metadata()`
  нормализует списки (строка с `,`/`;` или массив), ограничивает
  `scale_default` диапазоном `MIN_STRENGTH..MAX_STRENGTH`, проверяет имя без
  path traversal и существование файла.
- UI: `App.loadLoras()` запрашивает `/api/loras?model_id=...`,
  `pruneIncompatibleLoras()` снимает активные несовместимые адаптеры при смене
  модели, `loraCompatBadge()` рисует бейдж, incompatible-чекбоксы и кнопка
  подключения disabled. Модалка привязки — `#lora-binding-modal`,
  состояние — `App.state.loraBinding`.
- `/api/generate` обязан повторно вызвать `check_lora_compatibility()` для
  каждой LoRA: UI можно обойти, а движок всё равно получит несовместимый
  адаптер. При ошибке — 400 с причиной.
- `LORA_SUPPORTED_ENGINES = ("sd_cpp", "quantfunc", "nunchaku", "sdnq")`.
  OrbitQuant динамические LoRA пока не поддерживает — не добавляй его в этот
  список без реализации в движке.

---

## 7. Архитектурный план реализации системы слоёв (Layer System Plan)

> **Статус: реализовано.** Стек слоёв, виртуальный оригинал, маски, FBO-композитор WebGL2 и защита от муара встроены в `frontend/js/app.js` и `frontend/js/modules/halftone.js`; бэкенд пропускает виртуальный оригинал и сохраняет метаданные слоёв (`hasMask`, `isBase`).

### Концепция стека слоёв:
```text
----------------------------------------------------------------------
👁 [Слой 3: Блик / Спецэффект] (Маска: Лассо неба, Режим: Screen, 75%)
👁 [Слой 2: Тени одежды]       (Маска: Кисть складок, Режим: Multiply, 100%, Угол: 60°)
👁 [Слой 1: Базовый растр]     (Маска: Нет / На всё тело, Режим: Normal, 100%, Угол: 45°)
🔒 [Слой 0: Исходное фото]     (Оригинал / Background — заблокирован, база сравнения)
----------------------------------------------------------------------
[Управление]: ➕ Слой | ⬆ Вверх | ⬇ Вниз | 📋 Дубликат | 🗑 Удалить
```

### Задачи по этапам:
1. **Слой 0 (Оригинал)**:
   - Всегда присутствует внизу списка слоёв под замочком (`🔒`).
   - Кнопка «До / После» (`↔`) сравнивает итоговый композит строго со Слоем 0.
2. **Слой 1 (Базовый растр)**:
   - Создаётся автоматически при загрузке картинки. Покрывает 100% холста.
   - Маска по умолчанию отсутствует (`hasMask = false`).
3. **Слои 2+ (Локальные зоны)**:
   - Кнопка `➕ Слой` создает новый слой с *пустой* маской. Он не затирает Слой 1.
   - Растр этого слоя проявляется только там, где пользователь рисует кистью или применил выделение лассо.
4. **Кнопки ⬆ и ⬇**:
   - Перемещение элементов внутри `st.layers`.
   - Порядок в массиве определяет порядок наложения (верхний перекрывает нижний).
5. **Многопроходный WebGL2 FBO Композитор**:
   - Для стабильного живого рендеринга всех видимых слоёв перевести шейдер с одиночной отрисовки на Framebuffer Objects (FBO): последовательный рендер каждого видимого слоя с учётом его маски, непрозрачности и blend mode (`Normal`, `Multiply`, `Screen`, `Overlay`).
6. **Защита от муара (Moiré Protection)**:
   - При создании теневых слоёв (`Multiply`) автоматически предлагать смещение угла наклона точек (например, 45° для базы, 60°/75° для тени), предотвращая интерференционную сетку.

---

## 8. Известные ловушки, баги и правила их предотвращения

1. **Стилизация чекбоксов в CSS**:
   - **НИКОГДА** не пиши в CSS общие правила `input { ... }` без псевдоклассов исключения. Всегда пиши:
     `input:not([type="checkbox"]):not([type="radio"]):not([type="range"]) { ... }`
   - Для чекбоксов обязательно задавай: `appearance: auto; accent-color: var(--accent);`.
2. **Окончания строк Windows (CRLF vs LF)**:
   - Репозиторий находится на Windows, файлы часто содержат `\r\n`.
   - При строковых заменах через Python всегда открывай файл с точным указанием `newline=''` или используй нормализацию.
3. **Шаблоны строковых литералов в JS**:
   - Внутри шейдеров `VS_SOURCE` и `FS_SOURCE` используются шаблонные строки с обратными кавычками (`` ` ``). Будь предельно внимателен: любая незакрытая кавычка ломает парсер всего файла `app.js` до следующей функции!
4. **Совместимость FLUX.3: Nunchaku >= 1.4.0 _и_ diffusers main**:
   - Веса `tonera/flux-3-image-klein-nunchaku` содержат архитектуру `mode_blocks.image...`, которая поддерживается только в **Nunchaku >= 1.4.0** (PyTorch 2.14).
   - Даже с Nunchaku 1.4.1 стабильный `diffusers 0.41.0` падает с `NunchakuFlux3Transformer2DModel requires diffusers with Flux3Transformer2DModel`: нужен GitHub main (`0.42.0.dev0`), где есть `Flux3Transformer2DModel` и `ModularPipeline`.
   - FLUX.3 нельзя загружать через `Flux2KleinPipeline`; используй `ModularPipeline.from_pretrained(...).blocks`, удали блок `prompt_upsampler`, затем `init_pipeline`, `update_components`, `load_components`, `to("cuda")`.
   - Установщик Extras сам ставит diffusers main, если проба не находит `has_diffusers_flux3`/`has_modular_pipeline`. При ручной установке: `pip install -U "git+https://github.com/huggingface/diffusers.git"`.
5. **Рассинхронизация `app.js` и `modules/*`**:
   - Активный фронтенд — монолит `frontend/js/app.js`, но HTML и ES6-модули могут содержать более новые id/методы. Именно так «пропала» LoRA в LoRA Hub (коммит `2e1b352`): панель не переключалась, а `renderLoraHubGrid`/`renderLoraLibrary` отсутствовали в монолите.
   - После любой правки UI проверяй, что методы и id совпадают в `index.html`, `app.js` и `modules/*`. Регрессия ловится тестом `test_lora_hub_frontend_sync` в `tests/test_latest_fixes.py`.
   - Папка LoRA — это `models_dir/loras`, то есть при текущих настройках `models/cache/loras`, а не `models/loras`.
6. **Новый GPU-эндпоинт без очереди**:
   - Любой вызов `worker.generate(...)` обязан предваряться
     `STATE.register_job(..., serialized=True)` и `STATE.acquire_generation(job_id)`
     в фоновом потоке, а `STATE.release_generation(job_id)` — стоять в `finally`.
   - Иначе две вкладки снова стартуют два diffusion-пайплайна одновременно и
     получают CUDA OOM / повреждение состояния движка. Тест
     `test_queue_presence_history.py` проверяет FIFO и отмену.
7. **Dock Drag и CSS `left: auto !important`**:
   - В `.gemini-dock-wrap.dock-free-drag` **нельзя возвращать**
     `left: auto !important`: он отменяет inline `${customX}px`, и док едет
     только по одной оси. При этом `right:auto`, `bottom:auto` и
     `transform:none` с `!important` там нужны — они нейтрализуют preset-классы
     и не мешают inline `left/top`. Тянуть разрешено только за
     `#dock-drag-handle` при включённом `dockDragEnabled`.
   - **Не возвращай `.dock-popover` в normal flow.** В drag-режиме wrap
     привязан к `top`, и растущий поповер сдвигает саму панель вниз. Поповер
     должен оставаться `position:absolute`; для нехватки места сверху
     используется `.dock-popover-below`.
8. **History: гонка запросов, потеря `finally` и 502**:
   - `loadHistory()` инкрементирует `historyRequestSeq`; ответ применяется
     только если seq всё ещё актуален. Снятие `history-loading` обязано быть в
     `finally` — иначе любой сетевой сбой навсегда оставляет скелетон/зависание.
   - API нельзя переключать на объект без query-параметров: старые клиенты
     ожидают массив. Пагинация включается наличием `page/page_size/filter/q`.
   - Повторы `_fetchWithRetry` спасают только от транзиентного шлюза. Если
     `502` повторяется, проверь, что сервер слушает порт: такие запросы
     отсутствуют в access-log.
9. **Models Hub: селектор фильтров**:
   - Кнопки находятся в `#models-tab-filters` и используют
     `data-tab-filter`. Не возвращай несуществующий
     `#models-hub-filters .filter-pill`; иначе active-состояние не
     обновляется, а фильтрация выглядит неработающей.
   - Ветка `encoders` обязана запрашивать `/api/encoders` и использовать
     `renderEncodersTab()`, а не фильтровать `state.models`.
10. **Presence middleware не должен ломать запросы**:
    - `record_activity()` вызывается best-effort и глотает исключения. Не
      поднимай ошибку из middleware при проблемах с registry: иначе упадёт
      авторизация/статика. Гостю не отдавай список пользователей, только count.
11. **Удаление весов: никогда не удаляй каталог репозитория**:
    - Только `backend/model_files.py` и точные пути из манифеста. `shutil.rmtree`
      по `models/cache/<repo>` удалит общие энкодеры, VAE, sidecar'ы и варианты
      других моделей.
    - Перед удалением сверяй `(repo_id, path)` со ссылками **всех** моделей
      реестра; защищённые файлы возвращай в `skipped_shared`.
    - Удаление варианта не должно трогать TE/VAE другого варианта той же модели;
      sidecar'ы `orbitquant_*`/`svdq_*` удаляй только вместе с последним
      вариантом в папке.
    - Гостю удаление запрещено; путь обязан оставаться внутри `models/cache`.
12. **Кодовая страница Windows и логи со стрелками**:
    - На cp1251/cp866 символ `→` или эмодзи в `print()` вызывает
      `UnicodeEncodeError` и может убить сервер до старта. В `app.py` уже стоит
      `stdout/stderr.reconfigure(errors="replace")`, в `start.bat` —
      `chcp 65001`, `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`.
    - Не убирай эти настройки и не добавляй `print()` до их выполнения.
13. **Карточки моделей: встроенные манифесты теперь тоже редактируются**:
    - `GET/PATCH /api/models/{id}/card` доступны для всех моделей реестра, но
      только хосту; гость получает 403. Не возвращай старую проверку
      `is_custom`/`custom_*` — пользователь явно просил сделать редактируемыми
      все карточки.
    - Удаление встроенных моделей через UI по-прежнему запрещено (кнопка
      только у `is_custom`), а `delete_custom_model()` сохраняет защиту
      базовых системных моделей. Не путай редактирование с удалением.
    - Сохраняй через временный файл и восстановление бэкапа. Прямая запись
      JSON оставит битый манифест при ошибке валидации или reload.
    - `engine_args` синхронизируй с TE/VAE/aux. Иначе UI показывает новый
      энкодер, а движок грузит старый (или удалённый) файл.
    - VAE ищи через `backend/model_files.py:vae_payload*()`, а не локальной
      проверкой `vae_repo/vae_remote`: реальные файлы часто лежат в
      primary-репозитории (`vae`, basename) или в `vae/<basename>`. Для
      скачивания используй `vae_download_refs()`: source и local target могут
      быть разными.
14. **LoRA должна быть привязана к модели**:
    - Не возвращай модель-агностичное поведение: при наличии
      `compatible_models` точная привязка важнее family. Это осознанное
      требование пользователя, а не баг.
    - Совместимость проверяется в двух местах: `/api/loras?model_id=` для UI
      и `/api/generate` для фактического запуска. Не удаляй вторую проверку.
    - При смене модели снимай активные несовместимые LoRA, иначе следующий
      запуск упадёт с 400 или, хуже, отправит адаптер не той архитектуры.

---

## 9. Как тестировать изменения

Перед отправкой любого ответа пользователю выполни следующие шаги верификации:

1. **Проверка синтаксиса JavaScript**:
   ```powershell
   python -c "
   import subprocess, os
   for root, _, files in os.walk(r'G:\QuantFunc-WebUI\frontend\js'):
       for f in files:
           if f.endswith('.js'):
               p = os.path.join(root, f)
               res = subprocess.run(['node', '--check', p], capture_output=True, text=True)
               assert res.returncode == 0, f'Error in {f}: {res.stderr}'
   print('All JS files passed node --check!')
   "
   ```
2. **Запуск тестов pytest**:
   ```powershell
   python -m pytest G:\QuantFunc-WebUI\tests -q
   ```
   *Все 66 тестов (включая smoke_test, test_latest_fixes, test_halftone_and_manga,
   test_flux3_nunchaku_support, test_lora_hub_frontend_sync,
   test_model_files_and_hf_layout, test_model_card_and_lora_binding и
   test_queue_presence_history) должны проходить со статусом PASS (100%).*
   `venv\Scripts\python.exe` не содержит pytest — используй глобальный `python`.
3. **Проверка JSON-локалей и компиляции Python**:
   ```powershell
   python -c "import json; from pathlib import Path; [json.loads(Path(p).read_text(encoding='utf-8')) for p in ['frontend/locale/ru.json','frontend/locale/en.json']]; print('JSON OK')"
   python -m py_compile app.py backend/*.py
   ```
4. **Проверка git-статуса**:
   ```powershell
   git -C G:\QuantFunc-WebUI status -s
   ```

---

## 10. История коммитов (для агентов)

> Полный журнал с деталями: [`CHANGELOG.md`](CHANGELOG.md).
> Актуальный список всегда: `git log --oneline -20`.

### 2026-10-30 — 2026-11-01 (Inpaint Studio, Crop Modes, New Dock, Fast HF)
| Коммит | Описание |
|---|---|
| `07d074a` | `feat(inpaint)`: три режима области генерации: `auto` (стандартный bbox с padding 64, min 256), `parameters` (пользовательский padding 0-1024 и min side 64-4096) и `manual` (ручной прямоугольник с выравниванием по 16px сетке). Проверка покрытия маски в `resolve_crop_box()`, инспектор и холст оверлея, автотесты `tests/test_inpaint_crop_modes.py`. |
| `00d4702` | `fix(inpaint)`: локальный кроп генерации до bounding box маски с обратным бесшовным вклеиванием в оригинал. Маски для Qwen-Image-2.1 GGUF через SD.CPP `img2img+mask`, селективный предохранитель QuantFunc только для FLUX Klein, скролл инспекторов. |
| `630145d` | `fix(generate)`: обход фильтра `backend=lighting` для движков Nunchaku, OrbitQuant и SD.CPP (устранена ошибка `No compatible transformer found`). |
| `9acb452` | `feat(mask)`: инпейнт для FLUX.2 Klein (Nunchaku, OrbitQuant) и SD.CPP GGUF. `backend/masking.py`: единая полярность, `feather_mask()`, `prepare_mask_arrays()`, `Flux2KleinInpaintPipeline`, флаг `supports_mask: true`. |
| `5de3882` | `fix(mask)`: стабилизация Inpaint Studio: изолированные модули `mask-core.js`, `generate-mask.js`, `inpaint.js`, сохранение состояния источника, кисть/ластик/размытие, трансляция зума. |
| `507dd90` | `feat(new-dock)`: панель Reference / ControlNet в новом доке (Ctrl+V, миниатюра, выбор режима, сила, внешний ControlNet) и секция LoRA прямо под блоком VAE. |
| `b386e28` | `fix(ui)`: `refreshConfig()` с `cache: "no-store"` и изоляцией шагов рендера, автосброс кэша энкодеров и перерендер Hub после правки карточки модели. |
| `73ffbfd` | `feat(settings)`: переключатель быстрой загрузки (Xet/hf_transfer по версии библиотеки) и поле зеркала HF в Settings → Network. |
| `f9a5c9b` | `fix(download)`: учёт `HF_HUB_ENABLE_HF_TRANSFER` из окружения при установленном `hf_transfer`. |
| `58fdc2a` | `feat(webui)`: альтернативный компактный док LazySD Studio (`new-dock.js`) с режимами Pin/Float и шторкой параметров. |

### 2026-10-25
| Коммит | Описание |
|---|---|
| `0657d3e` | `fix(models)`: общий VAE-резолвер в `backend/model_files.py`, поиск уже скачанных VAE в primary/отдельном репозитории и `vae`/`vae/<basename>`, inferred-VAE для старых карточек, VAE-секция у каждой карточки, скачивание VAE в локальный target primary-репозитория, редактирование всех карточек (built-in + custom, guest-only ограничение), cache-busting `app.js`; 66 тестов. |
| `cf19e39` | `feat(models)`: редактируемые карточки кастомных моделей (host-only, атомарное сохранение с бэкапом), repo-local TE/VAE/aux в API и карточке, скачивание отдельного компонента, модалки редактора и выбора компонента, явная привязка LoRA (`compatible_models`, `engine`, `architecture`, `scale_default`), автопривязка по HF `base_model`, проверка совместимости в UI и `/api/generate`; 60 тестов. |
| `1033602` | `feat(models)`: безопасное удаление весов только по манифесту с защитой общих файлов, `backend/model_files.py`, удаление кастомных моделей без `rmtree`, настоящие кнопки скачивания/удаления, сворачиваемая панель загрузок, aux-sidecar'ы и `.py`/`fonts` при HF-импорте, self-contained folder-layout без `model_index.json` (Fibo), UTF-8 запуск на Windows; 44 теста. |

### 2026-10-23
| Коммит | Описание |
|---|---|
| `73b9ce8` | `fix(ui)`: Dock-поповеры выведены из normal flow (`position:absolute`, `dock-popover-below`), исправлены селектор и реальная фильтрация Models Hub, добавлены карточки/скачивание энкодеров, retry для транзиентных 502/503/504 в GET и Halftone preview; 36 тестов. |
| `d322737` | `feat(queue)`: FIFO GPU-очередь для `/api/generate` и `/api/manga/ai-job`, queue-overlay, лимит CPU 4, presence-мониторинг с heartbeat, серверная пагинация/фильтры History, отдельный Drag-режим Dock и исправление `left:auto`; 36 тестов. |
| `0470af4` | `fix(flux3)`: FLUX.3 Nunchaku через diffusers main и `ModularPipeline`; проба/установка `has_diffusers_flux3`, локальный `modular_model_index.json`, `NunchakuQwen3VLEncoderModel`, обход prompt upsampler, UI-проверка и тест `test_flux3_nunchaku_support`. |
| `2e1b352` | `fix(lora)`: восстановлен LoRA Hub — переключение панели, `renderLoraHubGrid`/`renderLoraLibrary`, счётчик, вкладки модалки, HF-эндпоинты, путь `models_dir/loras`, регрессионный тест. |
| `9ece51d` | `feat(halftone)`: стек слоёв, маски, виртуальный оригинал, WebGL2 FBO-композитор, защита от муара. |

### 2026-10-16 — 2026-10-17
| Коммит | Описание |
|---|---|
| `dcc5891` | Пресеты в доке/модалке, probe Nunchaku, синхронизация HF-токена. |
| `27bb549` | Безопасность удалённого доступа и share-URL. |
| `7d3b401` | Phase 1–2: аккаунты, пресеты, видимость моделей, HF Hub, свои модели. |
| `9ca1294` | Статистика CLI, HF-токен, изоляция гостей, шаринг. |
| `2bb835e` | cloudflared, Extras install payload, UI-пояснения. |
| `e174f76` | Открытие локальных папок в Проводнике Windows. |
| `da394dd` | Compare curtain, multi-LoRA, док, статистика, OrbitQuant/Nunchaku. |
| `b658bae` | Gemini Dock, Topaz-инспектор, сортировка моделей, HF explorer, пресеты. |
| `8784641` | ControlNet/Reference, live VRAM/RAM, history modal, prompt resize. |
| `aa4980e` | Переименование в LazySD UI, FIT preview, sidebar, отключение QuantFunc по умолчанию. |

**Правило:** после каждого значимого коммита добавляй запись в `CHANGELOG.md` и, если меняется архитектура или поведение, обновляй соответствующий раздел этого файла.
