# ТЗ: генератор промпта для Inpaint по выделенной области

## 1. Цель

Добавить в Inpaint-панель помощника, который по пожеланию пользователя
(«убрать текст», «восстановить красный фон», «сделать пустой баббл») и
контексту всей страницы формирует точный промпт и negative prompt для
основной diffusion-модели.

Пользователь не должен вручную придумывать prompt engineering. Он пишет
намерение, получает готовый промпт, при необходимости редактирует его и
запускает генерацию.

## 2. Варианты реализации

### Вариант A. Только эвристика

- Не требует vision-модели и дополнительных загрузок.
- Анализирует цвета, контраст и текстуру области маски и окружающего кольца.
- По ключевым словам пожелания собирает промпт по шаблону.
- Работает всегда, но не понимает сюжет и персонажей.

### Вариант B. Ollama vision + fallback (рекомендуемый)

- Если в Ollama доступна vision-модель, отправляет ей всю страницу,
  страницу с маской и кроп области.
- Если vision-модели нет или Ollama недоступна — автоматически использует
  вариант A.
- Модели скачиваются внутри UI через Ollama `/api/pull` с прогрессом.

### Вариант C. Self-contained llama.cpp

- Собственный `llama-server` + GGUF + mmproj, без внешней Ollama.
- Полностью автономно, но дороже по разработке и поддержке.

Вариант B считается основным. A — обязательный fallback. C — расширение
после стабилизации B.

## 3. Пользовательские сценарии

1. Пользователь загружает цветную страницу манги.
2. Рисует маску вокруг японского текста или SFX.
3. В поле «Что сделать в этой области?» пишет: «убрать текст, восстановить
   красные брызги».
4. Нажимает «Сгенерировать промпт».
5. Получает:
   - `prompt`: `seamless dark red and black ink splash background, matching
     surrounding red, manga brush texture, clean edges, no text`
   - `negative_prompt`: `text, japanese, kanji, kana, sfx, seams, blurry,
     color shift`
6. Проверяет/редактирует промпт.
7. Нажимает «Сгенерировать в маске».

Дополнительные сценарии:

- vision-модель не скачана → предложение скачать `qwen2.5vl:3b` прямо из панели;
- Ollama не запущена → понятная ошибка и инструкция;
- маска пустая → кнопка неактивна / предупреждение;
- пользователь хочет только fallback → настройка «Не использовать VLM».

## 4. Архитектура

### 4.1. Компоненты

- Frontend:
  - `frontend/index.html` — поля и кнопки в Inpaint, блок vision-моделей.
  - `frontend/js/app.js` — основная логика (фактически загружается `app.js`).
  - `frontend/js/modules/inpaint.js` — зеркальная модульная реализация.
  - `frontend/css/styles.css` — стили.
- Backend:
  - `app.py` — HTTP endpoints.
  - `backend/prompt_gen.py` — оркестрация генерации промпта.
  - `backend/prompt_fallback.py` — эвристический анализ изображения.
  - `backend/vision_models.py` — каталог и статусы vision-моделей.
  - `backend/vision_runtime.py` — Ollama-адаптер (и задел под llama.cpp).
  - `backend/settings.py` — новые настройки.
- Тесты:
  - `tests/test_prompt_fallback.py`
  - `tests/test_prompt_gen_api.py`
  - `tests/test_vision_models.py`

### 4.2. Поток данных

```text
Inpaint UI
  → POST /api/inpaint/prompt
      source_path, mask(data URL/path), instruction,
      crop_mode, crop_box, crop_padding, vision_model
  → backend:
      1. validate source + decode mask
      2. build page overview + mask overlay + region crop
      3. try VLM (Ollama) if enabled and installed
      4. else heuristic fallback
      5. cache result
  → { prompt, negative_prompt, source, model, warnings }
  → UI fills prompt/negative fields
```

## 5. Backend: API

### 5.1. `POST /api/inpaint/prompt`

Request:

```json
{
  "source_path": "uploads/page.png",
  "mask": "data:image/png;base64,...",
  "instruction": "убрать японский текст, восстановить красный фон",
  "crop_mode": "auto",
  "crop_box": null,
  "crop_padding": 64,
  "vision_model": "qwen2.5vl:3b",
  "use_vlm": true
}
```

Response:

```json
{
  "prompt": "seamless dark red and black ink splash background, ...",
  "negative_prompt": "text, japanese, kanji, kana, sfx, seams, blurry",
  "source": "vlm",
  "model": "qwen2.5vl:3b",
  "warnings": [],
  "analysis": {
    "dominant_colors": ["#b51f2a", "#1a1014", "#f2f2f2"],
    "region_type": "colored_art",
    "mask_box": [1856, 1984, 2752, 3008]
  }
}
```

Ошибки:

- `400` — пустая маска, неверный путь, слишком большое изображение.
- `409` — vision-модель не установлена, но `use_vlm=true` и fallback запрещён.
- `503` — Ollama недоступна.
- `504` — таймаут VLM.

Fallback-поведение: при `use_vlm=true` и недоступной VLM вернуть
эвристический результат с `source: "heuristic"` и `warnings`, если
пользователь не запретил fallback.

### 5.2. `GET /api/vision/models`

Возвращает:

```json
{
  "provider": "ollama",
  "available": true,
  "models": [
    {
      "id": "qwen2.5vl:3b",
      "name": "Qwen2.5-VL 3B",
      "size_gb": 3.2,
      "vram_min_gb": 6,
      "recommended": true,
      "installed": false,
      "capabilities": ["vision"],
      "status": "not_installed"
    }
  ]
}
```

### 5.3. `POST /api/vision/models/{id}/download`

- Запускает загрузку через Ollama `/api/pull`.
- Возвращает `job_id`.
- Прогресс отдаётся существующим механизмом job/status или SSE.
- Повторный вызов при активной загрузке возвращает тот же job.

### 5.4. `DELETE /api/vision/models/{id}`

- Удаляет модель через Ollama `/api/delete`.
- Нельзя удалить активную модель без подтверждения.

### 5.5. `GET /api/vision/status`

- `ollama_available`, `ollama_version`, `active_model`, `vram_note`.

## 6. Backend: vision-модели

Каталог (минимум):

| ID | Размер | VRAM | Примечание |
|---|---:|---:|---|
| `qwen2.5vl:3b` | ~3.2 ГБ | 6+ ГБ | рекомендуется для 8 ГБ |
| `qwen2.5vl:7b` | ~6 ГБ | 12+ ГБ | лучшее качество |
| `minicpm-v:8b` | ~5.5 ГБ | 12+ ГБ | альтернатива |
| `llava:7b` | ~4.7 ГБ | 8+ ГБ | fallback-совместимость |

Требования:

- определять установку через Ollama `/api/tags` и `capabilities`;
- показывать `installed`, `downloading`, `not_installed`, `error`;
- поддержать отмену загрузки, если Ollama API это позволяет;
- не смешивать модели из разных семейств.

## 7. Backend: подготовка изображений

1. Валидировать `source_path` (только `uploads/`, `outputs/`, `cache/`).
2. Декодировать маску через `backend/masking.py`
   (`decode_mask_to_float`, `prepare_mask_arrays`).
3. Найти bbox маски (`mask_bbox`).
4. `crop_mode`:
   - `auto` — bbox + 64 px;
   - `parameters` — bbox + `crop_padding`;
   - `manual` — `crop_box`;
   - `full` — вся страница.
5. Подготовить:
   - `overview` — вся страница, max сторона 1024 px, маска красная
     полупрозрачная (alpha 0.35–0.45);
   - `crop` — область маски + padding, max сторона 768–1024 px, без overlay.
6. Ограничить payload: PNG/JPEG, max ~8 МБ на изображение.
7. Не отправлять исходный 4K-файл целиком.

## 8. Backend: VLM-запрос

Провайдер по умолчанию — Ollama:

```http
POST {ollama_url}/api/chat
{
  "model": "qwen2.5vl:3b",
  "stream": false,
  "keep_alive": "0s",
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<instruction + context>", "images": [base64_overview, base64_crop]}
  ],
  "options": {"temperature": 0.2, "num_predict": 220}
}
```

System prompt (сокращённо):

```text
You are a prompt engineer for Stable Diffusion / Flux inpainting.
You receive: the whole page, the same page with the edited region highlighted,
and a close-up of that region.
Return ONLY valid JSON:
{"prompt": "...", "negative_prompt": "..."}
The prompt must describe what should appear inside the masked area:
colors, texture, style, continuity with surroundings.
Do not transcribe or reproduce any text from the image.
Do not include "no text" instructions as text to render; use the negative prompt.
Max 60 words.
```

Парсинг:

- убрать markdown-обёртки;
- извлечь JSON;
- fallback: взять первую непустую строку;
- ограничить `prompt` 1000 символами, `negative_prompt` 500;
- запретить пустой prompt.

## 9. Backend: эвристический fallback

Без внешних ML-зависимостей, только `Pillow` + `numpy`.

1. Взять область маски и кольцо вокруг (расширение bbox на 10–25%).
2. Уменьшить до 64×64 для анализа.
3. Получить палитру:
   - `Image.quantize(colors=5)` или ручная квантизация RGB;
   - отсортировать по площади;
   - игнорировать почти прозрачные/шумовые пиксели.
4. Посчитать:
   - среднюю яркость;
   - среднюю насыщенность;
   - std яркости (плоская заливка vs сложный рисунок);
   - плотность краёв (Sobel/простая разность).
5. Классифицировать область:
   - `white_bubble` — яркость > 0.82, насыщенность < 0.15, низкий std;
   - `flat_color` — std ниже порога;
   - `colored_art` — иначе.
6. Разобрать инструкцию по ключевым словам:
   - `удалить|убрать|clean|remove|стереть` → очистка;
   - `заменить|replace|вместо` → замена;
   - `фон|background|залить|fill` → заливка фона;
   - `текст|text|sfx|надпись` → добавить в negative.
7. Собрать prompt:
   - `white_bubble`: `clean white speech bubble interior, smooth white fill,
     subtle paper texture, seamless edges, no text`
   - `flat_color`: `seamless <color> fill, matching surrounding <color>,
     smooth paper texture, clean edges`
   - `colored_art`: `seamless <color> and <color2> background, matching
     surrounding manga brush texture, clean edges`
   - если есть явная замена: `<instruction>, <style>, seamless, no text`
8. Negative prompt по умолчанию:
   `text, japanese, kanji, kana, letters, sfx, speech bubble outline,
   watermark, signature, seams, blurry, low contrast, color shift`

## 10. Кэширование

- Ключ: `sha1(source_path + mtime + mask_hash + instruction + model + crop params)`.
- Хранить в памяти (LRU 50) или в `cache/prompt_gen/`.
- TTL: до изменения исходника/маски.
- При попадании возвращать `source` и `cached: true`.

## 11. VRAM и жизненный цикл VLM

- VLM и diffusion-модель не должны занимать VRAM одновременно.
- После ответа VLM отправлять `keep_alive: "0s"` (выгрузка).
- Перед стартом `/api/generate` проверять и при необходимости останавливать
  vision-модель.
- Настройка `vision_keep_alive` (по умолчанию `0s`).
- При OOM: вернуть понятную ошибку, предложить модель 3B или fallback.

## 12. Frontend: Inpaint

Добавить в панель:

- textarea/input «Что сделать в этой области?»;
- селект vision-модели;
- кнопку «✨ Сгенерировать промпт»;
- статус: `VLM`, `эвристика`, `кэш`;
- кнопку «Скачать модель», если выбрана неустановленная;
- спиннер/disable на время запроса;
- обработку ошибок через существующий toast.

Поведение:

- кнопка активна только при непустой маске и источнике;
- после ответа заполнять `#inpaint-prompt` и `#inpaint-negative-prompt`;
- не перезаписывать промпт без действия пользователя (только по кнопке);
- сохранять последнюю инструкцию в localStorage;
- все строки — через i18n (ru/en).

## 13. Frontend: модели

- Раздел «Vision / Prompt Assistant» в Models или Settings.
- Карточки моделей: имя, размер, VRAM, статус, кнопки
  «Скачать» / «Удалить» / «Выбрать».
- Прогресс загрузки: проценты + текущий файл/операция.
- Кнопка «Обновить статус».
- Если Ollama недоступна — баннер с инструкцией.

## 14. Настройки

Добавить в `Settings`:

- `vision_enabled: bool = True`
- `vision_provider: str = "ollama"`
- `vision_model: str = "qwen2.5vl:3b"`
- `vision_ollama_url: str = "http://127.0.0.1:11434"`
- `vision_keep_alive: str = "0s"`
- `vision_timeout_sec: int = 120`
- `vision_max_image_dim: int = 1024`
- `vision_fallback_enabled: bool = True`

Все поля — через существующий PATCH `/api/settings`, с валидаторами.

## 15. Нефункциональные требования

- Полностью локально; никаких внешних API кроме локальной Ollama.
- Время ответа: VLM < 15 с на GPU, fallback < 300 мс, кэш < 100 мс.
- Не ломать существующий Inpaint и Generate.
- Не добавлять тяжёлые Python-зависимости.
- Работать без vision-модели (fallback).
- Логи без содержимого изображений и без токенов.
- Ограничение instruction: 500 символов.

## 16. Тесты и критерии приёмки

Тесты:

- `test_prompt_fallback.py`: палитра, классификация, шаблоны, negative.
- `test_prompt_gen_api.py`: mock Ollama, JSON-парсинг, ошибки, кэш.
- `test_vision_models.py`: статусы, installed/not_installed, capabilities.
- Frontend: ручная проверка кнопки, прогресса, ошибок.

Критерии приёмки:

1. Без vision-модели кнопка возвращает осмысленный fallback-промпт.
2. С `qwen2.5vl:3b` возвращается VLM-промпт, `source="vlm"`.
3. Модель скачивается/удаляется из UI с прогрессом.
4. VLM выгружается до запуска diffusion.
5. Пустая маска, битый путь, Ollama down, таймаут — обработаны.
6. Промпт и negative подставляются в Inpaint и редактируются.
7. Существующие тесты Inpaint/Generate проходят.

## 17. Этапы и оценка

| Этап | AI-часы |
|---|---:|
| Fallback + endpoint + кнопка | 2–3 |
| Ollama-адаптер + vision-модели | 2–3 |
| UI моделей, прогресс, настройки | 1–2 |
| Тесты, QA, документация | 1–2 |
| **Итого (вариант B)** | **6–10** |

Вариант A: 2–3 AI-часа.
Вариант C (llama.cpp): +7–10 AI-часов к B.

## 18. Вне scope

- Обучение/дообучение моделей.
- Автоматическое распознавание и перевод текста.
- Массовая пакетная обработка страниц.
- Облачные VLM API.
- Генерация промпта без маски.
