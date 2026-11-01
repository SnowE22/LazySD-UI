# CHANGELOG — LazySD-UI / QuantFunc-WebUI

Журнал коммитов для людей и ИИ-агентов. Перед началом работы сверяйся с
`git log --oneline -20`, потому что этот файл обновляется не автоматически.

> **Последний код-коммит:** `07d074a` — feat(inpaint): add auto, parameterized and manual crop modes (2026-11-01).
> Документация (этот файл и ссылки в MD) идёт отдельным коммитом `docs: ...`, его хеш смотри в `git log -1`.

---

## 2026-11-01

### `HEAD` — feat(inpaint): aspect ratio clamping, safe crop scaling, softness cursor and manga presets

- **Клампинг аспект-соотношения кропа (`max_aspect_ratio: 2.0`)**: `backend/masking.py:compute_crop_box()` теперь ограничивает максимальное соотношение сторон области инпейнта до $\le 2:1$, симметрично расширяя меньшую сторону наружу в пределах границ кадра. Это устраняет артефакты дублирования объектов и разрывы латентов модели при узких вертикальных или горизонтальных масках.
- **Безопасный лимит разрешения инференса (`scale_crop_for_safe_inference`)**: при инпейнте крупных областей на 4K-изображениях кроп и маска автоматически масштабируются для движка до безопасного размера ($\le 1024$ px, кратно 16 px), предотвращая CUDA OOM. Функция `composite_crop_into_image` бесшовно масштабирует результат Lanczos-фильтром обратно в координаты исходного кадра с сохранением попиксельной идентичности нетронутых зон.
- **Двухконтурный курсор кисти (Softness Preview)**: в Inpaint Studio при включенной мягкости кисти (`softness > 0`) курсор отображает внешнее пунктирное кольцо (граница затухания) и внутреннее сплошное кольцо (100% ядро), давая наглядный контроль над растушевкой маски.
- **Оптимизация Undo/Redo (RAM protection)**: глубина стека снимков холста в Inpaint Studio ограничена 12 шагами (вместо 20), снижая пиковое потребление оперативной памяти вкладки браузера на 4K-сканах в 2 раза.
- **Быстрые пресеты для манги**: добавлены кнопки быстрого заполнения промптов над полем ввода («💬 Текст/SFX», «░ Скринтон», «✒️ Лайнарт», «🖼️ Фон») со специализированными положительными и негативными промптами для клининга страниц.

---

### `930f7e2` — feat(inpaint): port SnowE22 UX and memory optimizations

Портированы удачные решения и оптимизации из форка [SnowE22/LazySD-UI](https://github.com/SnowE22/LazySD-UI) (автор: `SnowE22 <deramur19191@gmail.com>`):
- **Персистентность в Inpaint Studio (`localStorage`)**: промпт, негативный промпт, шаги, CFG, сила изменения (denoise strength) и сэмплер теперь автоматически сохраняются в `localStorage` и восстанавливаются при обновлении страницы или перезапуске браузера.
- **Поддержка негативного промпта, Denoise strength и LoRA**: в Inpaint Studio добавлены поле негативного промпта и ввод силы изменения (Denoise strength), а также проброс активного стека LoRA в генерацию.
- **Неразрывный выбор моделей (Tab preservation)**: выбор модели в UI больше не выбивает пользователя принудительно на вкладку Generate, если он находится в Inpaint Studio (`this.state.currentTab !== "inpaint"`).
- **Pixel-sharp зум для манги и пиксель-арта**: добавлен режим четкого масштабирования (`image-rendering: pixelated`) при зуме холста $\ge 1.25\times$, устраняющий размытие штрихов и скринтонов.
- **VRAM-оптимизация для SD.CPP**: на GPU с $\le 12$ ГБ VRAM для `sd-cli.exe` автоматически активируются `--backend te=cpu`, `--params-backend cpu` и `--vae-tiling` для предотвращения падений с OOM при инференсе больших GGUF-моделей.

---

### `06be18f` — fix(models): family-level shared VAE fallback and Card Editor disk status

- **Семейный VAE fallback (`find_family_shared_vae`)**: FLUX-модели (`klein`, `flux`, `flux3`, `moss-image`) и Qwen-модели автоматически переиспользуют уже скачанные VAE того же семейства из кэша (например, `FLUX.2-klein-9B-Nunchaku` или `flux-3-image-klein-nunchaku`), исключая повторные загрузки одного и того же 16-канального VAE для каждой вариации модели.
- **Статус в Model Hub**: секция VAE на карточках моделей теперь учитывает общесемейные VAE и отображает бейдж `✓ На диске (из <shared_model>)` без ложной кнопки `[Скачать]`; предупреждение `[нужно скачать]` отображается только если VAE действительно отсутствует везде на диске.
- **Индикаторы наличия в Редакторе карточек**: компоненты VAE и текстовые энкодеры в Model Card Editor теперь отображают актуальный статус на диске (`✓ На диске` / `⬇ Не скачан`) и источник вместо сырого текстового пути.
- **Офлайн fallback базового пайплайна в Nunchaku**: при отсутствии `model_index.json` в локальной директории модели `ExtrasWorker` задействует базовую конфигурацию пайплайна от родственной модели того же семейства, позволяя инференсу работать полностью локально без сетевых запросов к Hugging Face.
- **Автотесты**: добавлены тесты разрешения семейных VAE и проверки UI-контрактов в `tests/test_model_files_and_hf_layout.py`.

---

### `868d33b` — fix(models): detect manually downloaded weights in subfolders and unblock Nunchaku generation

- **Поддержка подпапок весов (`transformer/`, `vae/`, `text_encoder/`)**: добавлен метод `CacheManager.find_existing_file()`, который находит веса и компоненты как в корне кэша репозитория, так и в стандартных поддиректориях Hugging Face diffusers.
- **Толерантность к размеру в `check_file()`**: файлы с девиацией размера (до 15% между десятичными GB в манифестах и бинарными GiB на диске, например 5.80 GB vs 6.22 GiB) больше не помечаются ошибочно как `corrupted`.
- **Авто-выбор установленного варианта при генерации**: если `variant_id` явно не передан из UI, бэкенд автоматически выбирает установленный на диске совместимый вариант трансформера (например, Blackwell FP4) вместо дефолтного нескачанного INT4.
- **Nunchaku локальное разрешение путей**: `ExtrasWorker` теперь надёжно находит локальный файл трансформера и текстового энкодера на диске, предотвращая зависание на попытке повторной онлайн-загрузки с Hugging Face.
- **UI Model Hub и Dock**: бейдж «нужно скачать» на секции текстовых энкодеров теперь отображается только если не установлен ни один совместимый энкодер; в Dock автоматически выбирается установленный вариант весов.

---

### `07d074a` — feat(inpaint): add auto, parameterized and manual crop modes

Три режима области генерации для Inpaint Studio (сокращают время диффузии и VRAM):
- **auto**: автоматический bounding box нарисованной маски со стандартным паддингом 64 px и минимальной стороной 256 px.
- **parameters**: настраиваемый паддинг (0–1024 px) и минимальный размер стороны (64–4096 px).
- **manual**: ручное выделение прямоугольной области пользователем с выравниванием по 16-пиксельной сетке движков.
- Строгая валидация покрытия: ручной прямоугольник обязан целиком закрывать нарисованную маску (проверяется общей функцией `resolve_crop_box()` в `backend/masking.py`), исключая неполный инпейнт.
- Инспектор «Область генерации» (`Generation area`), оверлейный холст с зумом/панорамированием, интерактивные бейджи X/Y/W/H и блокировка запуска при некорректном выделении.
- 13 автотестов в `tests/test_inpaint_crop_modes.py`.

---

## 2026-10-31

### `00d4702` — fix(inpaint): crop masked generation to the mask bbox and unblock Qwen

- Локальный кроп при инпейнте: исходное изображение и маска обрезаются до области маски (с выравниванием и min 256 px), движок диффузии выполняет инференс только на этой области, а результат бесшовно вклеивается обратно в исходный кадр при оригинальном разрешении (bit-identical снаружи маски).
- Поддержка масок для Qwen-Image-2.1 GGUF через SD.CPP (режим `img2img + mask`).
- Предохранитель QuantFunc ограничен только проблемными карточками FLUX Klein, чтобы Qwen и другие исправные карточки оставались доступными.
- Восстановлена плавная прокрутка инспекторов во вкладках Generate и Inpaint.

### `630145d` — fix(generate): bypass backend filter for non-QuantFunc engines

- Обход фильтра `backend=lighting` для не-QuantFunc движков: Nunchaku, OrbitQuant и SD.CPP используют имя движка как бэкенд трансформера.
- Устранена ошибка `"No compatible transformer found for backend=lighting"`, блокировавшая запуск инпейнта на движках Nunchaku/OrbitQuant/SD.CPP.

### `9acb452` — feat(mask): enable FLUX inpaint masks for Nunchaku, OrbitQuant and SD.CPP

- Расширение инпейнта на FLUX.2 Klein (Nunchaku, OrbitQuant) и SD.CPP GGUF.
- Модуль `backend/masking.py`: единая полярность маски (белый = перегенерировать, чёрный = сохранить), функции растушёвки `feather_mask()` (дилатация + размытие по Гауссу) и подготовка массивов `prepare_mask_arrays()`.
- Nunchaku / OrbitQuant: переключение на `Flux2KleinInpaintPipeline` при наличии маски, композитинг результата с исходником на полном разрешении.
- SD.CPP: автопередача `-i <source>`, `--strength <mask_strength>` и `--mask <file>`.
- Флаг `supports_mask: true` в манифестах совместимых моделей.

---

## 2026-10-30

### `5de3882` — fix(mask): harden inpaint workflow and preserve source state

- Комплексная стабилизация Inpaint Studio: изолированные ES6-модули `mask-core.js`, `generate-mask.js`, `inpaint.js`.
- Сохранение исходного состояния изображения, холст маскирования с кистью/ластиком/размытием, корректная трансляция координат при Pan/Zoom.
- Автотесты синхронизации фронтенда с бэкендом `tests/test_mask_frontend_sync.py` и базовые тесты маскирования `tests/test_masking.py`.

### `507dd90` — feat(new-dock): add reference / ControlNet bar and LoRA section under VAE

- Новый компактный док (`new-dock.js`): добавлена верхняя панель Reference / ControlNet (миниатюра, выбор режима Native Control / Canny / Img2Img, слайдер силы, поле внешнего ControlNet и кнопка очистки).
- Поддержка вставки референсов через `Ctrl+V`.
- Секция LoRA в поповере модели нового дока (размещена непосредственно под селектором VAE) с чекбоксами активации, слайдерами силы и бейджами совместимости.

### `b386e28` — fix(ui): refreshConfig uses no-store, isolates render steps, resets encoder cache; re-render Hub after model card save

- Защита от зависания UI: `refreshConfig()` запрашивает `/api/config` с `cache: "no-store"`, изолирует каждый шаг рендера в `try/catch` (ошибка одного элемента не ломает остальные).
- Автоматический сброс кэша энкодеров и принудительный перерендер вкладки Models Hub сразу после сохранения карточки модели.

### `73ffbfd` — feat(settings): fast HF downloads toggle (Xet/hf_transfer by hub version) and HF mirror endpoint in Settings > Network

- Настройки сети в Settings → Network: переключатель быстрой загрузки и поле кастомного зеркала (endpoint) Hugging Face (например, `https://hf-mirror.com`).
- Автоматический выбор ускоренного бэкенда в зависимости от версии `huggingface_hub` (Xet для 1.x, hf_transfer для 0.x).

### `f9a5c9b` — fix(download): honor HF_HUB_ENABLE_HF_TRANSFER from env when hf_transfer is installed

- Устранено принудительное отключение `HF_HUB_ENABLE_HF_TRANSFER=0` в `app.py`, блокировавшее многопоточный загрузчик из `start.bat`.

### `f3ccf95` — perf(download): enable hf_transfer multi-threaded HF downloads

- Добавлена зависимость `hf-transfer>=0.1.6` в `requirements.txt` и опция `HF_HUB_ENABLE_HF_TRANSFER=1` в `start.bat`.

### `30ae859` & `7213efc` — fix(webui): restore model list and keep dock controls visible

- Исправлено отображение списка моделей и выпадающих меню в поповерах альтернативного дока.

### `58fdc2a` — feat(webui): add alternate LazySD Studio dock

- Добавлен альтернативный плавающий компактный док LazySD Studio (`frontend/js/modules/new-dock.js`) с режимами Pin/Float, шторкой параметров и поповерами Model/Resolution/Params.

---

## 2026-10-27

### `6be7d5f` — feat(webui): multi-VAE selection, localization and UI/engine fixes

Мульти-выбор VAE:
- Карточки моделей отображают полный список найденных в кэше VAE (основной + дополнительные со статусом «На диске»).
- В плавающий Gemini Dock добавлен селектор VAE с динамической синхронизацией с инспектором параметров.
- Генерация корректно передаёт `custom_vae_path` в бэкенд, а глобальная секция VAE удалена из Settings во избежание путаницы.
- Добавлены автотесты в `tests/test_vae_files.py`.

Локализация RU/EN (i18n):
- Достигнут 100% паритет локалей: по 950 ключей в `frontend/locale/en.json` и `ru.json`.
- Полностью локализованы вкладки Models, Generate, Templates, инспектор параметров, док и системные уведомления `localizedDisabledReason()`.

Инспектор параметров и UI:
- Компактная верстка инспектора в стиле Topaz AI: убрано паразитное вертикальное растягивание аккордеонов (`flex: 0 0 auto`).
- Исправлено обрезание полей ввода и выпадающих списков, поля числовых значений CFG/Steps/Seed зафиксированы на 62 px.
- Поповеры выбора моделей переведены в `position: absolute` со стилем `.dock-popover-below`, устранив прыжки дока при открытии меню.

Управление компонентами и движками:
- Добавлено безопасное удаление отдельных VAE, текстовых энкодеров и aux-компонентов через `/api/models/{id}/component-delete` и `/api/encoders/delete` с защитой shared-файлов (`split_shared_refs`).
- Добавлено удаление движков SD.CPP и SDNQ через `/api/engine/uninstall`.
- Возможность редактирования путей `models_dir` и `output_dir` прямо из настроек UI.

Halftone Studio GT & Manga Screen AI:
- Реализована поддержка ROI (Region of Interest) с бесшовным продолжением растровой сетки (`_pattern_frame_for_roi`).
- Полноразмерный экспорт и превью с адаптивным антиалиасингом (AA2/AA4) без принудительного даунскейла.
- Автоматические повторы (retry) при временных кодах 502/503/504 для истории и превью.

---

## 2026-10-25

### `0657d3e` — fix(models): cached VAE detection + editable built-in cards

VAE в карточках моделей:
- `backend/model_files.py` получил общие helpers `vae_payload()`,
  `vae_payload_from_args()`, `vae_download_refs()`, `safe_repo_rel()`.
  Все три API-ответа (`/api/models`, `/api/models/{id}/state`, карточка)
  используют один и тот же резолвер вместо трёх копий `_is_vae_installed()`.
- Установленный VAE теперь ищется не только по `vae_repo`/`vae_remote`, но и
  в реальных cache-layout'ах: `vae`, `vae_file`, basename в корне репозитория,
  `vae/<basename>`, плюс отдельный VAE-репозиторий. Поэтому уже скачанные
  `flux2-vae.safetensors`, `qwen_image_2.1_vae_bf16.safetensors` и
  `vae/diffusion_pytorch_model.safetensors` показываются как «На диске», а не
  «Скачать».
- Если в манифесте VAE не объявлен, но в папке модели есть
  `vae/*.safetensors` (или `*vae*.safetensors`/`ae.safetensors` в корне),
  карточка выводит inferred-VAE с `inferred: true`. Секция VAE теперь
  рендерится у каждой карточки; без объявления и без файла показывается
  подсказка открыть редактор.
- `POST /api/models/{id}/component-download` для `role=vae` берёт источник из
  `vae_repo`/`vae_remote`, но скачивает/копирует файл в локальный путь
  `vae`/`vae_file` основного репозитория — так же, как `backend/sd_cpp.py`.
  Если файл уже лежит в source-репозитории, он копируется без сети.
- Удаление по манифесту учитывает и source-путь VAE, и локальный target,
  поэтому общий VAE защищается ссылками других моделей.

Редактирование карточек:
- `GET/PATCH /api/models/{id}/card` теперь работают для **всех** моделей
  реестра, включая встроенные; `editable: true` возвращается всегда.
  Гость по-прежнему получает 403. Удаление встроенных моделей не разрешено:
  кнопка «Удалить» остаётся только у `is_custom`.
- Редактор получает нормализованный `vae` из backend, включая inferred-VAE,
  и не перезаписывает абсолютный локальный VAE при сохранении других полей.
- Атомарное сохранение с бэкапом, `reload_registry()` и проверкой
  `registry.get(model_id)` сохранено без изменений.
- Обновлён cache-busting `frontend/index.html` (`app.js?v=20261025_03`),
  чтобы браузер не показывал старую карточку после обновления.

Проверка:
- `python -m pytest -q` — **66 passed** (60 старых + 6 новых: VAE в primary
  repo, VAE в отдельном repo, inferred-VAE, local download target,
  absolute-path rejection, редактирование built-in карточки).
- `node --check frontend/js/app.js`, `python -m py_compile`, `git diff --check`
  чистые.
- Live на `127.0.0.1:7871`: 24 модели; пять явно объявленных VAE показывают
  `installed=true`, Fibo остаётся `false`; у всех 24 карточек есть кнопка
  «Редактировать» и секция VAE.

### `cf19e39` — feat(models): editable custom cards, component downloads, explicit LoRA binding

Карточки моделей и компоненты:
- `GET /api/models/{id}/card` отдаёт редактируемый манифест кастомной модели,
  `PATCH /api/models/{id}/card` сохраняет изменения. Гость получает 403,
  встроенные манифесты — 404 «защищены»; редактировать можно только
  `is_custom`/`custom_*` модели.
- В редакторе доступны имя, описание, семейство, движок, pipeline, шаги, CFG,
  лицензия, базовая модель, список TE, VAE и aux-файлы. `steps`/`cfg`
  синхронизируются с `scheduler_presets.default` и `default_cfg`;
  `engine_args.llm*`, `vae*`, `aux_files` пересобираются из карточки.
- Пути компонентов проверяются: только относительные, без `..`, абсолютных
  путей и выхода за пределы репозитория; репозитории нормализуются к
  `author/repo`. Числовые поля ограничены (`steps` 1–150, `cfg` 0–30).
- Сохранение атомарное: временный файл, проверка `reload_registry()` и
  `registry.get(model_id)`, при ошибке — восстановление из бэкапа.
- `POST /api/models/{id}/component-download` скачивает ровно один
  repo-local компонент (`role=text_encoder|vae|aux`, для aux — `index`).
  Endpoint отклоняет traversal/абсолютные пути, возвращает
  `already_installed` для готового файла и запускает обычную job-прогрессию.
- `/api/models` и `/api/models/{id}/state` теперь содержат TE с `repo_id`,
  `path`, `precision`, `size_gb`, `installed`; VAE с `repo_id`, `path`,
  `installed`; список aux-файлов и `primary_repo_id`.
- В карточке модели появились сворачиваемые секции
  «Transformers / Текстовые энкодеры / VAE / Aux» с кнопками скачивания и
  удаления. Для кастомных моделей доступна кнопка «Редактировать».
- Редактор умеет проверять выбранный HF-репозиторий: показывает, какие TE,
  VAE и sidecar-файлы реально есть в дереве, отмечает уже добавленные и
  предлагает недостающие. Если энкодера в репозитории нет, его можно добавить
  вручную ссылкой или через поиск с фильтрацией по роли файла.

LoRA — явная привязка вместо модельно-независимых адаптеров:
- Метаданные LoRA расширены: `architecture`, `compatible_models`, `engine`,
  `target_modules`, `rank`, `scale_default`, `trigger`, `base`, `repo_id`.
- `check_lora_compatibility()` сначала проверяет точную привязку
  `compatible_models`, затем allow-list движков, и только потом — семейство
  архитектур. Точная привязка перекрывает несовпадение family.
- При импорте LoRA с HF поле `base_model` автоматически сопоставляется с
  `sources.primary.repo_id`/`base_repo_id` моделей реестра
  (`suggest_model_bindings`), поэтому адаптер сразу получает привязку.
- `GET /api/loras?model_id=...` возвращает `compatibility`
  (`compatible`, `match_level`, `reason`, `bound_models`, `bound_engines`).
  `PATCH /api/loras/{name}/metadata` доступен только хосту.
- UI показывает бейджи «точная / архитектура / семейство / движок» и
  «несовместима», блокирует чекбоксы и кнопку подключения, а при смене модели
  снимает активные несовместимые LoRA с предупреждением.
- `/api/generate` повторно проверяет совместимость и отклоняет запрос с
  несовместимой LoRA (400), даже если UI был обойдён.
- OrbitQuant пока не поддерживает динамические LoRA:
  `LORA_SUPPORTED_ENGINES = (sd_cpp, quantfunc, nunchaku, sdnq)`, для
  OrbitQuant-моделей все адаптеры помечаются несовместимыми.

Fibo 2.0 OrbitQuant W4A8:
- Репозиторные TE и VAE теперь видны в карточке: `text_encoder/model.safetensors`
  (bf16, 3.19 GB) и `vae/diffusion_pytorch_model.safetensors`, плюс два
  sidecar-файла `transformer/orbitquant_codebooks.safetensors` и
  `transformer/orbitquant_rotations.safetensors`.
- Runtime по-прежнему не поддерживается: установлен OrbitQuant 0.14.0, а W4A8
  требует `orbitquant>=0.15.0`, `BriaFibo2Pipeline`, `fibo2_orbitquant.py`,
  `manifest.json`, `processor/`, `scheduler/` и `fonts/`. Локальный манифест
  `custom_hf_fibo_20_edit_turbo_orbitquant_w4a8_hf.json` намеренно не коммитится.

Проверка:
- `python -m pytest -q` — **60 passed** (44 старых + 16 новых в
  `tests/test_model_card_and_lora_binding.py`, включая отказ component-download
  на `../`, `C:/` и `/etc/passwd`); `node --check`, `py_compile` и
  `git diff --check` чистые.
- Live на `127.0.0.1:7871`: 24 модели; карточка Fibo показывает 4 секции и
  5 кнопок скачивания; редактор — 1 TE, 1 VAE, 2 aux и предложения из
  репозитория; поиск компонента возвращает 12 HF-репозиториев; модалка
  привязки LoRA — 5 движков и 24 модели; при выборе Fibo три LoRA получают
  бейдж «несовместима» и disabled-чекбоксы, при возврате на FLUX.3 —
  снова совместимы.
- Bump cache-bust: `/css/styles.css?v=20261025_02`, `/js/app.js?v=20261025_02`.

---

### `1033602` — feat(models): safe weight deletion, download queue UI, folder-layout HF imports

Безопасное удаление весов:
- `DELETE /api/models/{id}/files` принимает необязательный `variant_id` и удаляет
  только точные файлы из манифеста (вариант, шарды `-0000N-of-0000M`, `.incomplete`),
  а не весь каталог репозитория. Общая логика вынесена в `backend/model_files.py`
  (`collect_delete_targets`, `delete_targets`, `model_file_refs`).
- Файл, на который ссылается другая модель реестра, никогда не удаляется. Ответ
  содержит `deleted_files`, `deleted_files_count`, `freed_bytes`, `freed_mb`,
  `removed_dirs`, `protected_shared_count`, `skipped_shared` и `message`.
- Пустые каталоги удаляются только после успешного удаления файлов; путь обязан
  находиться внутри `models/cache`.
- Удаление кастомной модели (`?delete_files=true`) использует те же helpers и
  больше не вызывает `shutil.rmtree` для каталога репозитория. Подтверждение в UI
  объясняет, что общие энкодеры, VAE и файлы других моделей останутся на диске.
- При удалении одного варианта sidecar-файлы `orbitquant_*`/`svdq_*` удаляются
  вместе с ним только если в той же папке нет другого варианта. Repo-local TE/VAE
  удаляются, общие энкодеры из других репозиториев защищены.

Models Hub / загрузки:
- Кнопки скачивания и удаления — настоящие `<button>` с inline SVG-иконками;
  установленный вариант показывает «На диске» и отдельную кнопку удаления.
- `<details id="downloads-panel">` — сворачиваемая панель загрузок: счётчик
  активных, прогресс, отмена, «Очистить завершённые», сохранение open-state.
- `installed` у вариантов вычисляется по фактическому файлу; в state/config
  добавлены `path`, `repo_id`, `installed`.

HF-импорт и folder-layout:
- Релизы без корневого `model_index.json`, но с `transformer/config.json` и
  `vae/config.json`, считаются самодостаточными. Вместо ложного `missing: base`
  UI предупреждает о кастомном pipeline-коде.
- Импорт скачивает декларированные `orbitquant_*`/`svdq_*` sidecar-файлы, а для
  кастомных моделей — корневые `.py` и `fonts/**` (файлы не исполняются).
- `inspect_hf_repo()` возвращает `pipeline_tag`, нормализованный `base_model`,
  `license`, `weight_size_gb`, `has_component_configs`, `pipeline_code`;
  инспектор показывает meta-chips, дерево папок и группы TE/VAE/aux.
- Проверен `WaveCut/Fibo-2.0-edit-turbo-OrbitQuant-W4A8`: `layout=folders`,
  `engine=orbitquant`, `family=flux`, 4 шага, CFG 1.0, `missing=[]`, 7.53 GB,
  base `briaai/Fibo-2.0-edit-turbo`, лицензия `cc-by-nc-4.0`.
  Манифест создан локально для проверки и **не входит в коммит**: W4A8 требует
  `orbitquant>=0.15.0` и `BriaFibo2Pipeline`, а установленный OrbitQuant 0.14.0
  модель не запустит.

Запуск на Windows:
- `app.py` переводит `stdout/stderr` в `errors="replace"`, а `start.bat` задаёт
  `chcp 65001`, `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`. Раньше сообщение со
  стрелкой (`→`) роняло сервер с `UnicodeEncodeError` на кодовой странице cp1251.

Проверка:
- `python -m pytest tests -q` — **44 passed**; `node --check frontend/js/app.js`,
  `py_compile` и live-проверка на `127.0.0.1:7871` успешны.
- Live: 24 модели, Fibo в списке как Custom, 57 кнопок скачивания, 7 кнопок
  удаления, панель «Загрузки 0 / Очистить завершённые»; удаление несуществующего
  варианта даёт 404, пустое удаление — 200 с `protected_shared_count=76`.
- Bump cache-bust: `/css/styles.css?v=20261025_01`, `/js/app.js?v=20261025_01`.

---

## 2026-10-23

### `73b9ce8` — fix(ui): keep Dock stationary, repair Models Hub filters, tolerate transient 502

Исправлены регрессии, найденные вживую на `127.0.0.1:7871`: сдвиг Dock при
открытии выбора модели в Drag-режиме, неработающие теги/поиск Models Hub и
ошибки `Bad Gateway` в History/Halftone.

Dock:
- Причина сдвига: поповеры лежали в normal flow перед `.gemini-dock`; в
  drag-режиме обёртка привязана к `top`, поэтому её рост увеличивал `top` самой
  панели. Например, при `dockCustomY=120` панель уезжала на `top=378.5`.
- `.dock-popover` теперь `position: absolute; bottom: calc(100% + 8px); left: 0`
  и не влияет на геометрию обёртки. Для случая, когда сверху нет места,
  `toggleDockPopover()` добавляет `.dock-popover-below`
  (`top: calc(100% + 8px)`).
- Проверено в браузере: Dock остаётся на `top=120` при открытии поповера и
  вверху, и внизу экрана; смещение — `0 px`. Ручка Drag и inline `left/top`
  не меняются.

Models Hub:
- Исправлен селектор активной кнопки: реальная разметка —
  `#models-tab-filters .filter-chip` с `data-tab-filter`, а не
  `#models-hub-filters .filter-pill`.
- `renderModelsTab()` реально фильтрует `all`, `installed`, `custom`, `hidden`,
  `svdq`, `sd_cpp`, `sdnq`, `klein`, `qwen`, `z-image`; поиск идёт по id,
  названию, family, engine, описанию, текстовым энкодерам и вариантам.
- `hidden` учитывает оба флага: `hidden_from_host` и `hidden_for_guests`.
- Ветка `encoders` запрашивает `/api/encoders` и рисует отдельные карточки
  (`renderEncodersTab`) с кнопкой скачивания; после скачивания сбрасывается
  кэш энкодеров и вызывается `refreshConfig()`.
- `frontend/js/modules/models-hub.js` синхронизирован: тот же селектор,
  `hidden`-фильтр и обновление active по `data-tab-filter`.

History/Halftone и сетевые ошибки:
- Живые запросы `/api/history?page=1..3` и `/api/halftone/process` на CPU
  вернули HTTP 200; в `logs/verify/quantfunc.log` не было ни 502, ни traceback.
  Ошибки `Bad Gateway` возникали, когда проверочный сервер был недоступен:
  запросы вообще не доходили до uvicorn.
- Добавлен `_fetchWithRetry`: GET-запросы повторяются до 2 раз, превью
  Halftone — 1 раз, только для `502/503/504` и сетевых ошибок. Мутирующие
  POST-генерации не повторяются, чтобы не создать дубликаты задач.
- Проверено инъекцией ответов: два подряд 502 на History и один 502 на
  Halftone preview успешно перекрываются повтором, UI не показывает ошибку.

Проверка:
- `python -m pytest tests -q` → **36 passed**; статические guard-тесты
  дополнены проверками Dock-поповеров, фильтров и retry.
- Все 14 frontend JS-файлов прошли `node --check`; `compileall` и оба locale
  JSON валидны.
- Живые фильтры: 23 all, 7 installed, 0 custom, 1 hidden, 13 encoders, 7 SVDQ,
  6 SD.CPP, 1 SDNQ, 11 Klein, 7 Qwen, 2 Z-Image; поиск `klein` — 10 карточек.
- История: страницы 1–3 по 24 элемента, без тостов; Halftone CPU: 1024×1024,
  1536×1536 и 2048×2048, HTTP 200, без `Halftone error`.
- Bump cache-bust: `/js/app.js?v=20261023_04`, `/css/styles.css?v=20261023_04`.

### `d322737` — feat(queue): serialize GPU jobs, add presence monitor, fix History and Dock drag

Основная задача: не допускать параллельного запуска GPU-генерации из нескольких
вкладок/аккаунтов, показать понятную очередь, добавить мониторинг подключений и
починить зависания History, а также конфликт Drag-режима Gemini Dock.

GPU-очередь:
- Новый `backend/generation_queue.py`: FIFO-очередь с одним активным job и
  поддержкой отмены ожидающих/активных задач.
- `AppState` хранит `owner_id`, `owner_name`, `kind`, `shared`, выдаёт
  `queue_position`, `queue_size`, `queue_active`; активная задача получает слот
  только после `acquire_generation()`, освобождает в `finally`.
- Сериализованы ровно две GPU-точки: `POST /api/generate` и
  `POST /api/manga/ai-job`. Скачивания, Halftone и Manga CPU-обработка остаются
  параллельными.
- Отмена ожидающей задачи удаляет её из очереди и не трогает чужой активный
  воркер; отмена активной останавливает worker. Поздняя ошибка воркера не
  перезаписывает статус `cancelled`. Если cancel пришёл между активацией слота
  и записью `initializing`, слот освобождается, а job остаётся `cancelled`.
- Для Halftone/Manga CPU-пайплайнов добавлен общий семафор
  `TOOL_CPU_MAX_CONCURRENCY = 4` (3–4 одновременные задачи).
- Глобальный `#queue-overlay` показывает позицию, число ожидающих и кнопку
  отмены на всех вкладках; исчезает, как только job переходит из `queued` в
  `initializing/loading/generating`.

Подключения:
- Новый `backend/presence.py`: уникальные аккаунты по `role:id`, TTL 90 секунд.
- `ShareAccessMiddleware` отмечает активность, фронтенд шлёт heartbeat каждые
  20 секунд; добавлены `/api/presence` и `/api/presence/heartbeat` (гостям
  разрешены, но список пользователей видит только хост).
- Настройка `monitor_connections` (по умолчанию включена) в Settings, PATCH и
  `/api/config`; в шапке появился постоянный индикатор `#presence-indicator`.
- В настройках хоста — счётчик онлайн и список подключённых пользователей.

History:
- Новый `backend/history_paging.py`; `/api/history` принимает `page`,
  `page_size`, `filter`, `q` и возвращает
  `{items,total,page,page_size,pages}`. Размер страницы 12/24/48, серверный
  максимум 100, номер страницы клампится к последней.
- Без query-параметров сохранён старый ответ-массив для совместимости.
- `CacheManager.list_outputs(limit=None)` отдаёт все файлы для пагинации.
- Фронтенд: серверные фильтры, debounce поиска, выбор размера страницы,
  кнопки страниц, защита от гонки устаревших ответов (`historyRequestSeq`) и
  снятие `history-loading` в `finally` — зависаний при ошибке больше нет.

Dock:
- Drag вынесен в отдельный явный режим (`#dock-drag-toggle-btn` и кнопка в
  настройках). Стартует только с ручки `⋮⋮`, а не со всей коробки/поповеров.
- Убран `left: auto !important` из `.gemini-dock-wrap.dock-free-drag`, который
  перебивал inline-координаты и вызывал движение только в одну сторону.
- Оба axes клампятся в границы окна; смена preset-позиции выключает drag;
  координаты и флаг сохраняются в `localStorage["lazysd_dock"]`.
- Добавлены сброс (`#dock-reset-btn`, двойной клик по ручке) и восстановление
  прежней свободной позиции при повторном включении Drag.

Проверка:
- `pytest tests -q` → **36 passed** (добавлен
  `tests/test_queue_presence_history.py`).
- Все 14 JS-файлов прошли `node --check`; оба locale JSON валидны;
  `python -m py_compile app.py backend/*.py` проходит.
- Живой запуск на `127.0.0.1:7871`: два FLUX.3-задания 512×512/4 шага —
  A `loading`, B `queued position=1`; отмена B не затронула A; A завершился за
  34.0 с, слот освобождён.
- В браузере: History 249 работ / 11 страниц, переход на страницу 2, размер
  страницы 12, поиск по prompt; очередь-оверлей, presence-индикатор и Dock
  drag/clamp/reset проверены через реальный UI.
- Bump cache-bust: `/js/app.js?v=20261023_03`, `/css/styles.css?v=20261023_03`.

### `0470af4` — fix(flux3): load Nunchaku FLUX.3 via diffusers ModularPipeline

Причина ошибки `[generate] Job cd33523f failed`:
`NunchakuFlux3Transformer2DModel requires diffusers with Flux3Transformer2DModel`.
В `envs/extras` стоял стабильный `diffusers 0.41.0` с PyPI: в нём есть
`Flux2KleinPipeline`, но нет `Flux3Transformer2DModel` и `ModularPipeline`.
FLUX.3 появился только в GitHub main (`0.42.0.dev0`, проверено на коммите
`48cdbaef5ee3d718aaee392b44f2099ae511c4a5`). Установщик Extras проверял только
`has_flux2_klein`, поэтому не обновлял diffusers для FLUX.3.

Исправлено:
- `backend/extras.py`: проба окружения теперь сообщает `has_diffusers_flux3` и
  `has_modular_pipeline`; установщик ставит diffusers из GitHub main, если этих
  классов нет.
- FLUX.3 Nunchaku загружается через `ModularPipeline` / `Flux3DistilledModularPipeline`,
  а не через `Flux2KleinPipeline`.
- Локальный `modular_model_index.json` распознаётся как индекс пайплайна.
- INT4-энкодер FLUX.3 загружается классом `NunchakuQwen3VLEncoderModel`
  (`Qwen3-VL-4B`), а не Qwen2-VL/Qwen3-text.
- Блок `prompt_upsampler` отключается: отдельная 9B-модель апсемплера пока не
  управляется WebUI, поэтому промпт идёт в текстовый энкодер как есть.
- Обработка результата ModularPipeline: список изображений / `output="images"`,
  без неподдерживаемых `guidance_scale` и `callback_on_step_end`.
- `frontend/js/app.js` и `modules/generator.js`: перед генерацией FLUX.3
  проверяют не только Nunchaku >= 1.4.0, но и наличие diffusers main; показывают
  понятное сообщение о переустановке Extras.
- Добавлен тест `test_flux3_nunchaku_support`.
- Bump cache-bust: `/js/app.js?v=20261023_02`.

Проверка:
- `pytest tests -q` → **29 passed**.
- Все JS-файлы (9) прошли `node --check`.
- Реальные задания через WebUI/backend: 512×512, 4 шага — 25.8 с;
  1024×1024, 4 шага — 45.6 с. Оба изображения сохранены в `outputs/`.

### `2e1b352` — fix(lora): restore LoRA Hub rendering and local library detection

Причина: активный монолит `frontend/js/app.js` рассинхронизировался с
`frontend/index.html` и `frontend/js/modules/lora-hub.js`.

Исправлено:
- `switchModelsHubSubtab('loras')` теперь реально показывает `#models-subpanel-loras`
  и скрывает панель моделей.
- Добавлены рабочие `renderLoraHubGrid()` и `renderLoraLibrary()`; LoRA Hub
  показывает локальные файлы, счётчик, фильтр, триггеры и кнопки подключения.
- Исправлены вкладки модального окна (`hub` / `local` / `library`).
- HF-запросы переведены на реальные эндпоинты `/api/loras/hf-inspect` и
  `/api/loras/hf-search`; скачивание читает `#lora-hf-file-select`.
- `app.py`: открытие папки `loras` ведёт в `models_dir/loras`
  (фактически `models/cache/loras`).
- `modules/lora-hub.js` синхронизирован с активной версией.
- Добавлен регрессионный тест `test_lora_hub_frontend_sync`.
- Bump cache-bust: `/js/app.js?v=20261023_01`.

Проверка: `pytest tests -q` → **28 passed**; все JS прошли `node --check`;
в браузере LoRA Hub показывает 1 локальную LoRA
(`[FLUX.2.Klein]LineArt_Redmond.safetensors`).

### `9ece51d` — feat(halftone): add layer stack, masks, and WebGL2 FBO compositor

- Система слоёв Halftone Studio GT: заблокированный виртуальный оригинал,
  автоматический базовый растр, локальные слои с пустыми масками.
- Многопроходный WebGL2 FBO-композитор с масками, opacity и blend modes.
- Перемещение слоёв ⬆/⬇, дублирование, удаление, бейджи base/mask.
- Защита от муара для `Multiply` (сдвиг угла на 15°).
- Бэкенд пропускает виртуальный оригинал и сохраняет `hasMask`/`isBase`.
- Тест `test_halftone_layer_stack_system`; на тот момент **27 passed**.

---

## 2026-10-17

| Коммит | Что изменилось |
|---|---|
| `dcc5891` | Кнопка сохранения пресета в доке и модальном окне; исправлен probe Nunchaku; HF-токен синхронизирован с CLI env. |
| `27bb549` | Безопасность удалённого доступа: экран входа, удалён backdoor `guest_public`, чистые share-URL. |
| `7d3b401` | Phase 1–2: аккаунты, compare curtain, пресеты, видимость моделей, HF Hub и пользовательские модели. |
| `9ca1294` | Компактная статистика CLI-движка, настройка HuggingFace-токена, изоляция гостей и шаринг. |
| `2bb835e` | Автоустановка cloudflared, исправление payload установки Extras и пояснения в UI. |
| `e174f76` | Возврат открытия локальных папок outputs/models в Проводнике Windows. |

## 2026-10-16

| Коммит | Что изменилось |
|---|---|
| `da394dd` | Compare curtain, multi-LoRA stacking, стили/докинг дока, CPU/GPU-статистика, OrbitQuant & Nunchaku, шаринг без Gradio. |
| `b658bae` | Плавающий Gemini Dock, Topaz-инспектор, 3-уровневая сортировка моделей, HF explorer, Presets hub. |
| `8784641` | Нативный ControlNet/Reference для Flux и Qwen, live VRAM/RAM, исправления history modal и prompt resize, RTX 5070 Ti cudart. |
| `aa4980e` | Переименование в LazySD UI, FIT preview, раскладка настроек, сворачиваемый sidebar, отключение QuantFunc-моделей по умолчанию. |

---

## Правила ведения журнала

1. Каждый значимый коммит добавляй в этот файл отдельной записью: хеш, тип,
   дата, что изменено и как проверено.
2. Если коммит меняет поведение Halftone/LoRA/Manga, обнови соответствующий
   раздел `AGENT_HANDOFF.md`.
3. Для активного фронтенда источник истины — `frontend/js/app.js`.
   `frontend/js/modules/*` обязан оставаться синхронизированным.
4. Перед коммитом запускай:
   ```powershell
   node --check frontend/js/app.js
   python -m pytest tests -q
   ```
