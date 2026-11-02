"""Model Hub integration for LazySD-Models catalog.

Provides discovery and 1-click installation of verified model cards from
the official or configured community repository (https://github.com/nnmgrp/LazySD-Models).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from backend.registry import get_registry, reload_registry
from backend.settings import get_settings

DEFAULT_HUB_URL = "https://raw.githubusercontent.com/nnmgrp/LazySD-Models/main"
CACHE_TTL_SECONDS = 600  # 10 minutes

_cached_catalog: Optional[dict[str, Any]] = None
_cache_time: float = 0.0

SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+\.json$")


def get_hub_base_url() -> str:
    """Return the raw repository base URL configured in settings or fallback to default."""
    try:
        settings = get_settings()
        custom = getattr(settings, "model_hub_url", "")
        url = (custom or "").strip().rstrip("/")
        if url:
            return url
    except Exception:
        pass
    return DEFAULT_HUB_URL


def fetch_hub_catalog(refresh: bool = False) -> dict[str, Any]:
    """Fetch index.json from the hub repository, caching for 10 minutes.

    Cross-references entries with local registry models and disk files
    to mark each card as installed or available for download.
    """
    global _cached_catalog, _cache_time
    now = time.time()
    base_url = get_hub_base_url()

    error_message: Optional[str] = None
    raw_catalog = None

    if not refresh and _cached_catalog is not None and (now - _cache_time) < CACHE_TTL_SECONDS:
        raw_catalog = _cached_catalog
    else:
        index_url = f"{base_url}/index.json"
        req = urllib.request.Request(
            index_url,
            headers={"User-Agent": "LazySD-UI/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} from hub: {index_url}")
                raw_catalog = json.loads(resp.read().decode("utf-8"))
                _cached_catalog = raw_catalog
                _cache_time = now
        except Exception as e:
            error_message = str(e)
            if _cached_catalog is not None:
                raw_catalog = _cached_catalog

    if raw_catalog is None:
        return {
            "schema_version": 1,
            "source": base_url,
            "count": 0,
            "installed_count": 0,
            "models": [],
            "error": error_message or "Не удалось загрузить каталог моделей.",
        }

    registry = get_registry()
    installed_ids = set(registry.models.keys())
    models_dir = Path(__file__).parent / "registry" / "models"
    local_files = {p.name.lower() for p in models_dir.glob("*.json")}

    models_out = []
    for item in raw_catalog.get("models", []):
        card_file = item.get("card_file") or f"{item.get('id')}.json"
        is_installed = (
            item.get("id") in installed_ids or
            card_file.lower() in local_files
        )
        entry = dict(item)
        entry["installed"] = is_installed
        entry["card_file"] = card_file
        models_out.append(entry)

    res = {
        "schema_version": raw_catalog.get("schema_version", 1),
        "source": base_url,
        "count": len(models_out),
        "installed_count": sum(1 for m in models_out if m["installed"]),
        "models": models_out,
    }
    if error_message:
        res["warning"] = f"Используется кэшированная версия. Ошибка обновления: {error_message}"
    return res


def install_hub_card(card_file: str) -> dict[str, Any]:
    """Download a verified model card JSON from the hub into backend/registry/models."""
    safe_name = str(card_file or "").strip()
    if not safe_name or not SAFE_FILENAME_RE.match(safe_name) or ".." in safe_name:
        raise ValueError(f"Недопустимое имя файла карточки: {card_file}")

    base_url = get_hub_base_url()
    card_url = f"{base_url}/models/{safe_name}"

    req = urllib.request.Request(
        card_url,
        headers={"User-Agent": "LazySD-UI/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} fetching {card_url}")
            content = resp.read().decode("utf-8")
            data = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки карточки из {card_url}: {e}")

    mid = data.get("id")
    if not mid or not isinstance(mid, str):
        raise ValueError(f"Карточка {safe_name} не содержит корректного поля 'id'")

    models_dir = Path(__file__).parent / "registry" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / safe_name

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    # Force registry reload so model immediately appears in UI and model picker
    reload_registry()

    return {
        "success": True,
        "card_file": safe_name,
        "model_id": mid,
        "name": data.get("name", mid),
        "message": f"Карточка модели «{data.get('name', mid)}» успешно установлена.",
    }
