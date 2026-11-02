# Промты для выбора и реализации генератора промпта Inpaint

Файл содержит:
1. промт для выбора варианта реализации;
2. общие правила для любого варианта;
3. промты на реализацию вариантов A, B и C;
4. промт для ревью результата.

Перед запуском любого промта приложите или укажите:
- `docs/INPAINT_PROMPT_GENERATOR_TZ.md`;
- доступ к репозиторию `D:\LazySD-UI`;
- ограничение по времени/объёму, если оно есть.

---

## 1. Промт выбора варианта

```text
Ты senior full-stack инженер. Работаешь в репозитории D:\LazySD-UI
(FastAPI + vanilla JS, без сборщика).

Задача: выбрать оптимальный вариант реализации генератора промпта для
Inpaint по выделенной области и затем реализовать его.

Сначала изучи:
- docs/INPAINT_PROMPT_GENERATOR_TZ.md;
- backend/masking.py;
- backend/settings.py;
- app.py: endpoints /api/generate, /api/models, /api/models/{id}/state,
  /api/settings, существующие job/download механизмы;
- frontend/index.html, frontend/js/app.js,
  frontend/js/modules/inpaint.js, frontend/css/styles.css;
- как устроены скачивание моделей и прогресс.

Проверь окружение:
- установлена ли Ollama и запущена ли она;
- какие vision-модели доступны через Ollama /api/tags;
- есть ли ограничения по VRAM/GPU;
- какие зависимости уже есть в requirements.txt.

Варианты:
A. Только эвристика по цветам/текстуре, без VLM. 2–3 AI-часа.
B. Ollama vision + эвристический fallback. 6–10 AI-часов.
C. Self-contained llama.cpp (llama-server + GGUF + mmproj). 13–20 AI-часов.

Критерии выбора:
1. Если Ollama доступна и пользователь не запретил внешний runtime —
   выбирай B: это даёт реальный vision при разумных сроках.
2. Если Ollama недоступна, но нужен максимум качества и автономность —
   выбирай C.
3. Если нужен быстрый результат без загрузок и зависимостей —
   выбирай A.

По умолчанию выбирай B, если нет явных противопоказаний.

Верни:
- выбранный вариант и обоснование;
- что уже есть в кодовой базе и что придётся добавить;
- риски и способы их снизить;
- план по шагам с оценкой в AI-часах;
- список файлов, которые будут изменены.

Не пиши код на этом шаге. Дождись подтверждения.
```

---

## 2. Общие правила для всех вариантов реализации

```text
Общие требования:
- Не ломать существующие Inpaint и Generate.
- Не добавлять тяжёлые Python-зависимости без необходимости.
- Всё работает локально; никаких облачных API.
- Новые строки UI — через существующий i18n (ru/en).
- Пути к изображениям валидировать: только uploads/, outputs/, cache/.
- Ограничить instruction 500 символами, prompt — 1000.
- Не логировать содержимое изображений и base64.
- Добавить тесты и проверить синтаксис (py_compile, node --check).
- Не делать commit и push без отдельной команды.
- В конце отчитаться: что изменено, как проверено, что осталось.
```

---

## 3. Промт на реализацию варианта A (только эвристика)

```text
Реализуй вариант A из docs/INPAINT_PROMPT_GENERATOR_TZ.md:
эвристический генератор промпта для Inpaint без vision-модели.

Функционал:
1. В Inpaint-панели добавить поле «Что сделать в этой области?» и кнопку
   «✨ Сгенерировать промпт».
2. Добавить POST /api/inpaint/prompt.
3. Backend:
   - валидирует source_path и маску;
   - декодирует маску через backend/masking.py;
   - находит bbox и область кропа;
   - анализирует доминирующие цвета области и кольца вокруг;
   - считает яркость, насыщенность, std, плотность краёв;
   - классифицирует область: white_bubble / flat_color / colored_art;
   - разбирает instruction по ключевым словам;
   - собирает prompt и negative_prompt по шаблонам;
   - кэширует результат по хэшу входа;
   - возвращает { prompt, negative_prompt, source: "heuristic", analysis }.
4. Frontend:
   - кнопка активна только при непустой маске и источнике;
   - спиннер/disable на время запроса;
   - заполняет #inpaint-prompt и #inpaint-negative-prompt;
   - toast при ошибке;
   - сохраняет последнюю инструкцию в localStorage.

Файлы:
- backend/prompt_gen.py
- backend/prompt_fallback.py
- app.py
- frontend/index.html
- frontend/js/app.js
- frontend/js/modules/inpaint.js
- frontend/css/styles.css
- tests/test_prompt_fallback.py
- tests/test_prompt_gen_api.py

Критерии приёмки:
- без интернета и моделей кнопка возвращает осмысленный промпт;
- пустая маска/битый путь/огромное изображение обработаны;
- существующие тесты проходят;
- UI не ломает текущую разметку Inpaint.
```

---

## 4. Промт на реализацию варианта B (Ollama vision + fallback)

```text
Реализуй вариант B из docs/INPAINT_PROMPT_GENERATOR_TZ.md:
Ollama vision + эвристический fallback.

Часть 1. Backend vision:
1. backend/vision_models.py:
   - каталог qwen2.5vl:3b, qwen2.5vl:7b, minicpm-v:8b, llava:7b;
   - статус через Ollama /api/tags и capabilities;
   - installed / not_installed / downloading / error.
2. backend/vision_runtime.py:
   - проверка доступности Ollama;
   - вызов /api/chat с images (base64) и keep_alive="0s";
   - таймауты, ошибки, fallback;
   - остановка/выгрузка модели перед diffusion.
3. Endpoints:
   - GET /api/vision/models;
   - POST /api/vision/models/{id}/download (Ollama /api/pull, прогресс);
   - DELETE /api/vision/models/{id};
   - GET /api/vision/status;
   - POST /api/inpaint/prompt.

Часть 2. Подготовка изображений:
- overview: вся страница max 1024 px, маска красная полупрозрачная;
- crop: область маски + padding, max 1024 px, без overlay;
- ограничение размера payload.

Часть 3. VLM:
- system prompt: prompt engineer для SD/Flux, вернуть только JSON
  {prompt, negative_prompt}, описывать цвета/стиль/текстуру, не
  транскрибировать текст;
- temperature 0.2, num_predict 200–250;
- парсинг JSON с fallback на первую строку.

Часть 4. Fallback:
- если Ollama недоступна/модель не скачана/таймаут — использовать
  эвристику из варианта A;
- вернуть source="heuristic" и warnings.

Часть 5. UI:
- поле инструкции, селект vision-модели, кнопка генерации промпта;
- кнопка «Скачать» для неустановленной модели с прогрессом;
- раздел «Vision / Prompt Assistant» в Models/Settings;
- статус VLM/эвристика/кэш;
- настройки: vision_enabled, vision_model, ollama_url, keep_alive,
  timeout, max_image_dim, fallback_enabled.

Файлы:
- backend/prompt_gen.py
- backend/prompt_fallback.py
- backend/vision_models.py
- backend/vision_runtime.py
- backend/settings.py
- app.py
- frontend/index.html
- frontend/js/app.js
- frontend/js/modules/inpaint.js
- frontend/css/styles.css
- tests/test_prompt_fallback.py
- tests/test_prompt_gen_api.py
- tests/test_vision_models.py

Критерии приёмки:
- без vision-модели работает fallback;
- с qwen2.5vl:3b возвращается VLM-промпт, source="vlm";
- скачивание/удаление/прогресс работают из UI;
- VLM выгружается перед запуском diffusion;
- ошибки Ollama/timeout/OOM обработаны;
- существующие тесты Inpaint/Generate проходят.
```

---

## 5. Промт на реализацию варианта C (self-contained llama.cpp)

```text
Реализуй вариант C из docs/INPAINT_PROMPT_GENERATOR_TZ.md:
полностью автономный vision-runtime на llama.cpp, без Ollama.

Часть 1. Runtime:
1. Добавить скачивание llama-server + DLL как отдельного runtime
   (по аналогии с sd.cpp): CUDA/Vulkan/CPU сборки, checksum, версия.
2. backend/vision_runtime.py:
   - запуск llama-server с -m model.gguf --mmproj mmproj.gguf;
   - выбор порта, health-check, таймаут;
   - остановка перед diffusion и после простоя;
   - OpenAI-compatible /v1/chat/completions с base64 images.

Часть 2. Модели:
- каталог vision-моделей: LLM GGUF + mmproj GGUF;
- скачивание через backend/sources/huggingface.py;
- проверка пары model/mmproj, размера, VRAM;
- статусы и прогресс.

Часть 3. Prompt pipeline:
- та же подготовка overview/crop и system prompt, что в варианте B;
- fallback-эвристика при недоступном runtime;
- кэш.

Часть 4. UI:
- раздел Vision Models со скачиванием/удалением/выбором;
- поле инструкции и кнопка генерации промпта в Inpaint;
- настройки runtime (GPU layers, context, порт, таймаут).

Критерии приёмки:
- runtime и модели скачиваются из UI;
- VLM-промпт генерируется без Ollama;
- fallback работает;
- VLM не конфликтует с diffusion по VRAM;
- тесты и ручной прогон на цветной странице.
```

---

## 6. Промт для ревью после реализации

```text
Проведи ревью реализации генератора промпта Inpaint.

Проверь:
1. Соответствие docs/INPAINT_PROMPT_GENERATOR_TZ.md.
2. Безопасность: валидация путей, размеры изображений, отсутствие
   внешних сетевых вызовов кроме локальной Ollama/llama-server.
3. Обработку ошибок: пустая маска, Ollama down, модель не скачана,
   таймаут, OOM, битый JSON от VLM.
4. VRAM: VLM выгружается до старта diffusion.
5. Кэш и его инвалидацию.
6. UI: состояния загрузки, disable кнопок, i18n, отсутствие регрессий.
7. Тесты: покрытие fallback, API, парсинга JSON, статусов моделей.
8. Качество промптов на примерах: белый баббл, цветные брызги,
   текст поверх персонажа.

Верни список проблем по приоритетам (blocker/major/minor) и конкретные
предложения по исправлению. Не меняй код без отдельной команды.
```
