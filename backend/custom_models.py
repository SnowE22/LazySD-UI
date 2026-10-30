"""
Custom Model Manager — handles user-imported GGUF/Safetensors models and Hugging Face Hub discovery.

Allows users to:
1. Drop arbitrary .gguf / .safetensors files into models/custom/ or specify absolute paths on PC.
2. Auto-detect model family, quantization, role, steps, and CFG.
3. Search Hugging Face Hub and inspect any repository's GGUF/Safetensors file tree to register
   downloadable model profiles in one click.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import requests

from .settings import get_settings
from .registry.loader import get_registry, reload_registry


def get_custom_models_dir() -> Path:
    """Return path to models/custom/ directory, creating it if necessary."""
    settings = get_settings()
    custom_dir = settings.models_dir.parent / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir


def detect_file_metadata(filename: str) -> dict[str, Any]:
    """Guess family, role, quantization, steps, and CFG from a weight filename."""
    name_lower = filename.lower()

    # Role detection
    role = "diffusion_model"
    if any(k in name_lower for k in ("vae", "ae.safetensors", "ae.sft", "pig")):
        role = "vae"
    elif any(k in name_lower for k in ("t5", "clip", "text_encoder", "llm", "qwen3", "qwen2.5", "mmproj", "umt5")):
        role = "text_encoder"
    elif any(k in name_lower for k in ("sdxl", "sd_xl", "v1-5", "sd15", "juggernaut", "pony", "illustrious")):
        role = "checkpoint"

    # Family detection
    family = "flux"
    if "moss" in name_lower:
        family = "moss-image"
    elif "qwen" in name_lower:
        family = "qwen"
    elif any(k in name_lower for k in ("sdxl", "sd_xl", "pony", "illustrious")):
        family = "sdxl"
    elif any(k in name_lower for k in ("z-image", "z_image", "zimage")):
        family = "z-image"
    elif any(k in name_lower for k in ("klein", "flux")):
        family = "flux"

    # Quantization detection
    quant_match = re.search(
        r"(q8_0|q6_k|q5_k_m|q5_k_s|q5_0|q4_k_m|q4_k_s|q4_0|q3_k_m|q3_k_s|q2_k|iq4_nl|iq4_xs|bf16|fp16|f16|fp8|int8|int4)",
        name_lower,
    )
    quant = quant_match.group(1).upper() if quant_match else "CUSTOM"

    # Default steps & CFG
    is_distilled_fast = any(k in name_lower for k in ("schnell", "turbo", "lightning", "hyper")) and "base" not in name_lower
    if family == "sdxl":
        steps = 1 if is_distilled_fast else 20
        cfg = 1.0 if is_distilled_fast else 7.0
    elif family == "moss-image":
        steps = 8 if is_distilled_fast else 50
        cfg = 1.0 if is_distilled_fast else 3.0
    elif family == "qwen":
        steps = 8 if is_distilled_fast else 20
        cfg = 1.0 if is_distilled_fast else 3.0
    elif family == "flux":
        if "klein" in name_lower and "base" not in name_lower:
            steps = 4
            cfg = 1.0
        elif is_distilled_fast:
            steps = 4
            cfg = 1.0
        else:
            steps = 20
            cfg = 3.5
    else:
        steps = 20
        cfg = 3.5

    return {
        "role": role,
        "family": family,
        "quant": quant,
        "steps": steps,
        "cfg": cfg,
    }


def list_custom_files() -> list[dict]:
    """Scan models/custom/ for compatible weights files with auto-detected metadata."""
    custom_dir = get_custom_models_dir()
    supported_exts = {".gguf", ".safetensors", ".ckpt", ".pt", ".bin"}
    files = []

    for path in sorted(custom_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported_exts:
            try:
                st = path.stat()
                size_bytes = st.st_size
                size_gb = round(size_bytes / (1024 ** 3), 2)
                rel_path = str(path.relative_to(custom_dir)).replace("\\", "/")
                ext = path.suffix.lower().lstrip(".")
                meta = detect_file_metadata(path.name)

                files.append({
                    "name": path.name,
                    "rel_path": rel_path,
                    "abs_path": str(path.resolve()),
                    "size_bytes": size_bytes,
                    "size_gb": size_gb,
                    "ext": ext,
                    "role": meta["role"],
                    "family": meta["family"],
                    "quant": meta["quant"],
                    "steps": meta["steps"],
                    "cfg": meta["cfg"],
                })
            except Exception:
                continue

    return files


def slugify(text: str) -> str:
    """Generate a clean slug for model ID."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


def register_custom_model(data: dict) -> dict:
    """Register a custom model configuration as a ModelSpec JSON manifest.
    Supports passing either a file path OR a folder path (auto-binding sibling text encoders, VAE, and tokenizer).
    """
    from .model_sniffer import scan_local_path

    name = data.get("name", "").strip()
    model_file = (data.get("model_file") or data.get("local_path") or "").strip().strip('"').strip("'")
    if not model_file:
        raise ValueError("Model file or folder path is required")

    custom_dir = get_custom_models_dir()
    file_path = Path(model_file)
    if not file_path.is_absolute():
        resolved_file = custom_dir / model_file
        if not resolved_file.exists():
            raise ValueError(f"File or folder not found in models/custom/: {model_file}")
        abs_target = resolved_file.resolve()
    else:
        if not file_path.exists():
            raise ValueError(f"File or folder not found: {file_path}")
        abs_target = file_path.resolve()

    # Scan target (file or folder) so sibling TE/VAE/variants in the same folder are auto-discovered
    sniff_res: Optional[dict[str, Any]] = None
    try:
        sniff_res = scan_local_path(str(abs_target))
    except Exception:
        sniff_res = None

    all_diff_files: list[Path] = []
    if abs_target.is_dir():
        base_folder = abs_target
        if sniff_res:
            comps = sniff_res.get("components", {})
            diff_items = comps.get("diffusion_model", []) + comps.get("checkpoint", [])
            for item in diff_items:
                cand = (base_folder / item["path"]).resolve()
                if cand.is_file():
                    all_diff_files.append(cand)
        if not all_diff_files:
            for ext in ("*.gguf", "*.safetensors", "*.ckpt"):
                for f in sorted(base_folder.rglob(ext)):
                    if f.is_file() and detect_file_metadata(f.name)["role"] in ("diffusion_model", "checkpoint"):
                        all_diff_files.append(f.resolve())
        if not all_diff_files:
            raise ValueError(f"В папке {abs_target} не найдено весов модели (.gguf / .safetensors)")
        abs_file = all_diff_files[0]
    else:
        abs_file = abs_target
        base_folder = abs_target.parent
        all_diff_files = [abs_file]

    if not name:
        name = base_folder.name if abs_target.is_dir() else abs_file.stem

    st = abs_file.stat()
    size_gb = round(st.st_size / (1024 ** 3), 2)
    meta = detect_file_metadata(abs_file.name)
    if sniff_res and sniff_res.get("suggested_family") and meta["family"] == "flux":
        if sniff_res["suggested_family"] != "flux":
            meta = detect_file_metadata(sniff_res["suggested_family"] + "_" + abs_file.name)

    raw_slug = slugify(name) or "model"
    model_id = f"custom_{raw_slug}"

    engine = data.get("engine") or (sniff_res.get("suggested_engine") if sniff_res else None) or "sd_cpp"
    family = (data.get("family") or (sniff_res.get("suggested_family") if sniff_res else None) or meta["family"] or "flux").lower()
    pipeline = data.get("pipeline", "text_to_image")
    steps = int(data.get("steps") or meta["steps"])
    cfg = float(data.get("cfg") if data.get("cfg") is not None else meta["cfg"])
    sampler = data.get("sampler", "euler")
    checkpoint_type = data.get("checkpoint_type") or ("checkpoint" if meta["role"] == "checkpoint" else "diffusion_model")
    local_folder = (data.get("local_folder") or base_folder.name).strip()

    engine_args: dict[str, Any] = {}
    if checkpoint_type == "checkpoint":
        engine_args["checkpoint"] = str(abs_file)
    else:
        engine_args["diffusion_model"] = str(abs_file)

    llm_path = (data.get("llm_file") or "").strip().strip('"').strip("'")
    if llm_path:
        p = Path(llm_path)
        engine_args["llm"] = str(p if p.is_absolute() else (custom_dir / llm_path).resolve())

    t5_path = (data.get("t5_file") or "").strip().strip('"').strip("'")
    if t5_path:
        p = Path(t5_path)
        engine_args["t5xxl"] = str(p if p.is_absolute() else (custom_dir / t5_path).resolve())

    clip_path = (data.get("clip_file") or "").strip().strip('"').strip("'")
    if clip_path:
        p = Path(clip_path)
        engine_args["clip_l"] = str(p if p.is_absolute() else (custom_dir / clip_path).resolve())

    vae_path = (data.get("vae_file") or "").strip().strip('"').strip("'")
    if vae_path:
        p = Path(vae_path)
        engine_args["vae"] = str(p if p.is_absolute() else (custom_dir / vae_path).resolve())

    # Auto-bind sibling text_encoder / vae / tokenizer discovered in the same directory if not manually set
    sibling_tes: list[dict[str, Any]] = []
    if sniff_res:
        comps = sniff_res.get("components", {})
        for te in comps.get("text_encoder", []):
            te_abs = (base_folder / te["path"]).resolve()
            if not te_abs.is_file():
                continue
            te_lower = te_abs.name.lower()
            if "clip" in te_lower and not engine_args.get("clip_l"):
                engine_args["clip_l"] = str(te_abs)
            elif "t5" in te_lower and "umt5" not in te_lower and not engine_args.get("t5xxl") and family == "flux" and "klein" not in abs_file.name.lower():
                engine_args["t5xxl"] = str(te_abs)
            else:
                if not engine_args.get("llm"):
                    engine_args["llm"] = str(te_abs)
                sibling_tes.append({
                    "id": slugify(te_abs.stem) or "custom_llm",
                    "name": f"{te_abs.name} ({te.get('size_gb', 0)} GB)",
                    "file": str(te_abs),
                    "precision": (te.get("quant") or "custom").lower(),
                    "recommended": len(sibling_tes) == 0,
                })
        if not engine_args.get("vae"):
            for v in comps.get("vae", []):
                v_abs = (base_folder / v["path"]).resolve()
                if v_abs.is_file():
                    engine_args["vae"] = str(v_abs)
                    break

    tok_cand = base_folder / "tokenizer.json"
    if tok_cand.is_file() and not engine_args.get("tokenizer"):
        engine_args["tokenizer"] = str(tok_cand.resolve())

    if data.get("vae_format"):
        engine_args["vae_format"] = data["vae_format"]

    if data.get("flow_shift"):
        engine_args["flow_shift"] = float(data["flow_shift"])

    transformers_list = []
    for idx, df in enumerate(all_diff_files):
        df_meta = detect_file_metadata(df.name)
        df_size = round(df.stat().st_size / (1024 ** 3), 2)
        transformers_list.append({
            "id": slugify(df.stem) if len(all_diff_files) > 1 else (df_meta["quant"].lower() if df_meta["quant"] != "CUSTOM" else "custom"),
            "path": str(df),
            "tier": "universal",
            "backend": df_meta["quant"].lower() if df_meta["quant"] != "CUSTOM" else engine,
            "steps": steps,
            "size_gb": df_size,
            "recommended": idx == 0,
            "note": f"Local file: {df.name} ({df_size} GB)",
        })

    manifest = {
        "id": model_id,
        "name": name if "(Custom)" in name else f"{name} (Custom)",
        "family": family,
        "pipeline": pipeline,
        "engine": engine,
        "engine_version": engine if engine != "quantfunc" else "0.0.17",
        "is_custom": True,
        "model_path": str(abs_file),
        "local_folder": local_folder,
        "default_cfg": cfg,
        "description": data.get("description") or f"Custom {family.upper()} model ({abs_file.name}, {size_gb} GB, engine: {engine}).",
        "engine_args": engine_args,
        "transformers": transformers_list,
        "requirements": {
            "vram_min_gb": int(data.get("vram_min_gb", 8)),
            "vram_recommended_gb": 12,
            "gpu_min_sm": 75,
            "default_vram_mode": "auto",
        },
        "scheduler_presets": {
            "default": {
                "sampler": sampler,
                "steps": steps,
                "cfg": cfg,
            }
        },
    }

    if sibling_tes:
        manifest["text_encoders"] = sibling_tes
    elif engine_args.get("llm"):
        manifest["text_encoders"] = [
            {
                "id": "custom_llm",
                "name": Path(engine_args["llm"]).name,
                "file": engine_args["llm"],
                "precision": "custom",
                "recommended": True,
            }
        ]

    registry_dir = Path(__file__).parent / "registry" / "models"
    out_json = registry_dir / f"{model_id}.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    reload_registry()
    return manifest


def _get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from settings."""
    settings = get_settings()
    return getattr(settings, "huggingface_token", None) or getattr(settings, "hf_token", None)


def search_hf_models(query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search Hugging Face Hub for GGUF / image generation models."""
    q = (query or "").strip()
    if not q:
        q = "flux gguf"
    hf_token = _get_hf_token()
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    params = {
        "search": q,
        "limit": max(1, min(limit, 25)),
        "sort": "downloads",
        "direction": "-1",
    }
    resp = requests.get("https://huggingface.co/api/models", params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    items = resp.json()
    results = []
    for item in items:
        repo_id = item.get("id") or item.get("modelId")
        if not repo_id:
            continue
        tags = item.get("tags") or []
        results.append({
            "repo_id": repo_id,
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "pipeline_tag": item.get("pipeline_tag") or "",
            "tags": [t for t in tags if t in ("gguf", "diffusers", "flux", "sdxl", "text-to-image", "image-to-image")][:6],
            "last_modified": item.get("lastModified"),
        })
    return results


def inspect_hf_repo(repo_id: str) -> dict[str, Any]:
    """Inspect a Hugging Face repository: classify every weight by folder + name, detect layout/engine."""
    from .model_sniffer import SUPPORTED_ENGINES, normalize_hf_repo, sniff_files

    clean_repo = normalize_hf_repo(repo_id)
    if not clean_repo or "/" not in clean_repo:
        raise ValueError("Укажите корректный Hugging Face репозиторий в формате author/repo-name")

    hf_token = _get_hf_token()
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    url = f"https://huggingface.co/api/models/{clean_repo}?blobs=true"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 404:
        raise ValueError(f"Репозиторий не найден на Hugging Face: {clean_repo}")
    resp.raise_for_status()
    info = resp.json()

    siblings = info.get("siblings") or []
    tags = [str(t) for t in (info.get("tags") or [])]
    files = [{"path": s.get("rfilename") or "", "size": s.get("size") or 0} for s in siblings]
    sniff = sniff_files(files, repo_id=clean_repo, tags=tags)
    comps = sniff["components"]

    repo_meta = detect_file_metadata(clean_repo)
    suggested_name = clean_repo.split("/")[-1].replace("-", " ").replace("_", " ")
    card = info.get("cardData") or {}

    raw_base = card.get("base_model")
    if isinstance(raw_base, list):
        raw_base = next((str(item) for item in raw_base if item), "")
    base_model = (raw_base or "").strip() if isinstance(raw_base, str) else ""
    # HF tags carry the same binding as base_model:quantized for many quantized releases.
    if not base_model:
        for tag in tags:
            if tag.startswith("base_model:") and not tag.startswith("base_model:quantized:"):
                base_model = tag.split(":", 1)[1].strip()
                break

    raw_license = card.get("license")
    if isinstance(raw_license, list):
        raw_license = next((str(item) for item in raw_license if item), "")
    license_name = str(raw_license or "").strip()
    if not license_name:
        license_name = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), "")

    total_weight_gb = round(sum(float(f.get("size_gb") or 0) for f in [
        *comps["diffusion_model"], *comps["checkpoint"],
        *comps["text_encoder"], *comps["vae"], *comps["aux"],
    ]), 2)

    return {
        "repo_id": clean_repo,
        "name": suggested_name,
        "suggested_name": suggested_name,
        "family": sniff["suggested_family"] or repo_meta["family"],
        "suggested_family": sniff["suggested_family"] or repo_meta["family"],
        "pipeline_tag": info.get("pipeline_tag") or "",
        "steps": repo_meta["steps"],
        "suggested_steps": repo_meta["steps"],
        "cfg": repo_meta["cfg"],
        "suggested_cfg": repo_meta["cfg"],
        "tags": tags[:20],
        "base_model": base_model,
        "license": license_name,
        "weight_size_gb": total_weight_gb,
        # Legacy keys (older UI code paths)
        "diffusion_files": comps["diffusion_model"] + comps["checkpoint"],
        "text_encoder_files": comps["text_encoder"],
        "vae_files": comps["vae"],
        # Folder-aware analysis
        "aux_files": comps["aux"],
        "layout": sniff["layout"],
        "has_model_index": sniff["has_model_index"],
        "has_component_configs": sniff.get("has_component_configs", False),
        "pipeline_code": sniff.get("pipeline_code", []),
        "folders": sniff["folders"],
        "suggested_engine": sniff["suggested_engine"],
        "engine_reasons": sniff["engine_reasons"],
        "missing": sniff["missing"],
        "needs_manual": sniff["needs_manual"],
        "warnings": sniff["warnings"],
        "suggested_folder": sniff["suggested_folder"],
        "defaults": sniff["defaults"],
        "engines": [{"id": k, "label": v} for k, v in SUPPORTED_ENGINES.items()],
    }


def register_hf_custom_model(data: dict[str, Any]) -> dict[str, Any]:
    """Register a downloadable model profile directly from a Hugging Face repository.
    Preserves subfolder paths (e.g. text_encoder/..., vae/..., transformer/..., split_files/...)
    and auto-binds repo-internal text encoders and VAEs when present.
    """
    from .model_sniffer import (
        SUPPORTED_ENGINES, safe_folder_name, parse_hf_file_ref, normalize_hf_repo
    )
    from .cache import set_repo_folder_alias

    raw_repo = (data.get("repo_id") or "").strip().strip('"').strip("'")
    repo_id = normalize_hf_repo(raw_repo) or raw_repo.replace("https://huggingface.co/", "").strip("/")
    if not repo_id or "/" not in repo_id:
        raise ValueError("Некорректный repo_id Hugging Face")

    name = (data.get("name") or repo_id.split("/")[-1]).strip()
    family = (data.get("family") or detect_file_metadata(repo_id)["family"] or "flux").lower()
    steps = int(data.get("steps") or detect_file_metadata(repo_id)["steps"] or 20)
    cfg = float(data.get("cfg") if data.get("cfg") is not None else detect_file_metadata(repo_id)["cfg"])
    selected_files: list[dict[str, Any]] = data.get("selected_files") or data.get("variants") or []
    if not selected_files:
        raise ValueError("Выберите хотя бы один файл модели (.gguf / .safetensors) из репозитория")

    engine = (data.get("engine") or data.get("suggested_engine") or "sd_cpp").lower()
    if engine not in SUPPORTED_ENGINES:
        engine = "sd_cpp"

    raw_folder = (data.get("local_folder") or data.get("folder_name") or repo_id.split("/")[-1]).strip()
    local_folder = safe_folder_name(raw_folder, repo_id.split("/")[-1])
    set_repo_folder_alias(repo_id, local_folder)

    base_repo_id = (data.get("base_repo_id") or "").strip()
    if base_repo_id:
        base_repo_id = normalize_hf_repo(base_repo_id) or base_repo_id

    raw_slug = slugify(name) or slugify(repo_id.split("/")[-1])
    model_id = f"custom_hf_{raw_slug}"

    pipeline_tag = str(data.get("pipeline_tag") or "").lower()
    pipeline = "image_edit" if "image-to-image" in pipeline_tag or "image_edit" in pipeline_tag else "text_to_image"

    transformers = []
    for idx, vf in enumerate(selected_files):
        fpath = (vf.get("path") or vf.get("name") or "").replace("\\", "/")
        if not fpath:
            continue
        meta = detect_file_metadata(fpath)
        parts = Path(fpath).parts
        vid = slugify(f"{parts[-2]}_{Path(fpath).stem}") if len(parts) > 1 else slugify(Path(fpath).stem)
        vid = vid or f"var_{idx}"
        base_vid = vid
        suffix = 2
        while any(t["id"] == vid for t in transformers):
            vid = f"{base_vid}_{suffix}"
            suffix += 1
        transformers.append({
            "id": vid,
            "path": fpath,
            "tier": "universal",
            "backend": meta["quant"].lower() if meta["quant"] != "CUSTOM" else ("gguf" if fpath.endswith(".gguf") else engine),
            "steps": steps,
            "size_gb": float(vf.get("size_gb") or 4.0),
            "recommended": idx == 0,
            "repo_id": repo_id,
            "note": f"{fpath} ({meta['quant']})",
        })

    text_encoders = list(data.get("text_encoders") or [])
    engine_args: dict[str, Any] = {
        "diffusion_model": transformers[0]["path"],
    }

    # 1. Bind repo-internal Text Encoder files if discovered in repo or passed from UI
    repo_te_files: list[dict[str, Any]] = data.get("selected_te_files") or data.get("text_encoder_files") or []
    for idx, tef in enumerate(repo_te_files):
        te_path = (tef.get("path") or tef.get("file") or tef.get("name") or "").replace("\\", "/")
        if not te_path:
            continue
        te_meta = detect_file_metadata(te_path)
        te_id = slugify(Path(te_path).stem) or f"te_{idx}"
        te_size = float(tef.get("size_gb") or 0.0)
        te_source = tef.get("source") if isinstance(tef.get("source"), dict) else {}
        te_repo = te_source.get("repo_id") or tef.get("repo") or repo_id
        te_entry = {
            "id": te_id,
            "name": f"{Path(te_path).name} ({te_size} GB)" if te_size else Path(te_path).name,
            "repo": te_repo,
            "file": te_path,
            "path": te_path,
            "size_gb": te_size,
            "precision": te_meta["quant"].lower() if te_meta["quant"] != "CUSTOM" else "bf16",
            "recommended": idx == 0,
            "source": {
                "repo_id": te_repo,
                "file_path": te_source.get("file_path") or te_path,
            },
        }
        if not any((te.get("file") == te_path and (te.get("repo") or (te.get("source") or {}).get("repo_id")) == te_repo)
                   for te in text_encoders):
            text_encoders.append(te_entry)
        if idx == 0 and not engine_args.get("llm"):
            engine_args["llm"] = te_path
            engine_args["llm_repo"] = te_repo
            engine_args["llm_file"] = te_path

    # 2. Bind repo-internal VAE files if discovered in repo or passed from UI
    repo_vae_files: list[dict[str, Any]] = data.get("selected_vae_files") or data.get("vae_files") or []
    for vf in repo_vae_files:
        v_path = (vf.get("path") or vf.get("file") or vf.get("name") or "").replace("\\", "/")
        if v_path:
            v_source = vf.get("source") if isinstance(vf.get("source"), dict) else {}
            v_repo = v_source.get("repo_id") or vf.get("repo") or repo_id
            engine_args["vae"] = v_path
            engine_args["vae_repo"] = v_repo
            engine_args["vae_file"] = v_path
            engine_args["vae_remote"] = v_source.get("file_path") or v_path
            break

    # 2b. OrbitQuant / SVDQ sidecars (codebooks, rotations, manifests) must travel with the weights.
    aux_entries: list[dict[str, Any]] = []
    for aux in data.get("aux_files") or []:
        aux_path = (aux.get("path") if isinstance(aux, dict) else str(aux or "")).replace("\\", "/")
        if not aux_path:
            continue
        aux_source = aux.get("source") if isinstance(aux, dict) and isinstance(aux.get("source"), dict) else {}
        aux_entries.append({
            "path": aux_path,
            "repo_id": aux_source.get("repo_id") or (aux.get("repo") if isinstance(aux, dict) else "") or repo_id,
            "size_gb": float(aux.get("size_gb") or 0.0) if isinstance(aux, dict) else 0.0,
        })
    if aux_entries:
        engine_args["aux_files"] = aux_entries
        engine_args["aux_repo"] = repo_id

    # 3. Handle manual / custom TE input from user (overrides repo-internal / defaults)
    custom_te_ref = (data.get("custom_te_ref") or data.get("custom_te_url") or "").strip()
    if custom_te_ref:
        parsed_te = parse_hf_file_ref(custom_te_ref)
        if parsed_te:
            te_item = {
                "id": slugify(Path(parsed_te["path"]).stem) or "custom_te",
                "name": Path(parsed_te["path"]).name,
                "repo": parsed_te["repo_id"],
                "file": parsed_te["path"],
                "path": parsed_te["path"],
                "precision": "custom",
                "recommended": True,
                "source": {
                    "repo_id": parsed_te["repo_id"],
                    "file_path": parsed_te["path"],
                },
            }
            text_encoders = [te_item]
            engine_args["llm"] = parsed_te["path"]
            engine_args["llm_repo"] = parsed_te["repo_id"]
            engine_args["llm_file"] = parsed_te["path"]
        else:
            engine_args["llm"] = custom_te_ref

    # 4. Handle manual / custom VAE input from user
    custom_vae_ref = (data.get("custom_vae_ref") or data.get("custom_vae_url") or "").strip()
    if custom_vae_ref:
        parsed_vae = parse_hf_file_ref(custom_vae_ref)
        if parsed_vae:
            engine_args["vae"] = parsed_vae["path"]
            engine_args["vae_repo"] = parsed_vae["repo_id"]
            engine_args["vae_file"] = parsed_vae["path"]
            engine_args["vae_remote"] = parsed_vae["path"]
        else:
            engine_args["vae"] = custom_vae_ref

    # 5. Family defaults if still unset
    if not engine_args.get("vae"):
        if family == "moss-image":
            engine_args["vae"] = "flux2-vae.safetensors"
            engine_args["vae_repo"] = "unsloth/FLUX.2-VAE"
            engine_args["vae_file"] = "flux2-vae.safetensors"
            engine_args["vae_remote"] = "flux2-vae.safetensors"
        elif family == "flux":
            engine_args["vae"] = "ae.safetensors"
            engine_args["vae_repo"] = "ffxvs/vae-flux"
            engine_args["vae_file"] = "ae.safetensors"
            engine_args["vae_remote"] = "ae.safetensors"
            if "klein" in repo_id.lower() or "klein" in name.lower():
                engine_args["vae"] = "flux2-vae.safetensors"
                engine_args["vae_repo"] = "Lightricks/LTX-2-comfy"
                engine_args["vae_file"] = "vae/flux2-vae.safetensors"
                engine_args["vae_remote"] = "vae/flux2-vae.safetensors"
        elif family == "qwen":
            engine_args["vae"] = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
            engine_args["vae_repo"] = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
            engine_args["vae_file"] = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
            engine_args["vae_remote"] = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"

    if not text_encoders and not engine_args.get("llm"):
        if family == "moss-image":
            text_encoders = [
                {
                    "id": "qwen3_5_9b_q4_k_xl",
                    "name": "Qwen3.5-9B Q4_K_XL (5.6 GB, Рекомендуется)",
                    "repo": "unsloth/Qwen3.5-9B-GGUF",
                    "file": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                    "path": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                    "size_gb": 5.6,
                    "precision": "q4_k_xl",
                    "recommended": True,
                    "source": {
                        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
                        "file_path": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
                    },
                },
            ]
            engine_args["llm"] = text_encoders[0]["file"]
            engine_args["llm_repo"] = text_encoders[0]["repo"]
            engine_args["llm_file"] = text_encoders[0]["file"]
            engine_args["tokenizer"] = "tokenizer.json"
            engine_args["tokenizer_repo"] = "OpenMOSS-Team/MOSS-Image-Base"
            engine_args["tokenizer_remote"] = "tokenizer/tokenizer.json"
        elif family == "flux" and ("klein" in repo_id.lower() or "klein" in name.lower()):
            is_9b = "9b" in repo_id.lower() or "9b" in name.lower()
            if is_9b:
                text_encoders = [
                    {
                        "id": "qwen3_8b_q8_0",
                        "name": "Qwen3-8B Q8_0 (8.7 GB, Максимальная точность)",
                        "repo": "Qwen/Qwen3-8B-GGUF",
                        "file": "Qwen3-8B-Q8_0.gguf",
                        "path": "Qwen3-8B-Q8_0.gguf",
                        "size_gb": 8.7,
                        "precision": "q8_0",
                        "recommended": True,
                        "source": {
                            "repo_id": "Qwen/Qwen3-8B-GGUF",
                            "file_path": "Qwen3-8B-Q8_0.gguf",
                        },
                    },
                ]
            else:
                text_encoders = [
                    {
                        "id": "qwen3vl_4b_q8_0",
                        "name": "Qwen3-VL-4B Q8_0 (4.3 GB, Рекомендуется)",
                        "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                        "file": "Qwen3VL-4B-Instruct-Q8_0.gguf",
                        "path": "Qwen3VL-4B-Instruct-Q8_0.gguf",
                        "size_gb": 4.3,
                        "precision": "q8_0",
                        "recommended": True,
                        "source": {
                            "repo_id": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                            "file_path": "Qwen3VL-4B-Instruct-Q8_0.gguf",
                        },
                    },
                ]
            engine_args["llm"] = text_encoders[0]["file"]
            engine_args["llm_repo"] = text_encoders[0]["repo"]
            engine_args["llm_file"] = text_encoders[0]["file"]
        elif family == "qwen":
            text_encoders = [
                {
                    "id": "qwen3vl_8b_q4_k_xl",
                    "name": "Qwen3-VL-8B Q4_K_XL (5.1 GB, Рекомендуется)",
                    "repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                    "file": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                    "path": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                    "size_gb": 5.1,
                    "precision": "q4_k_xl",
                    "recommended": True,
                    "source": {
                        "repo_id": "unsloth/Qwen3-VL-8B-GGUF",
                        "file_path": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                    },
                },
            ]
            engine_args["llm"] = text_encoders[0]["file"]
            engine_args["llm_repo"] = text_encoders[0]["repo"]
            engine_args["llm_file"] = text_encoders[0]["file"]

    base_models_map: dict[str, str] = {}
    if engine in ("sdnq", "nunchaku", "orbitquant") or data.get("has_model_index") or base_repo_id:
        base_models_map = {"50x-above": ".", "50x-below": "."}

    manifest = {
        "id": model_id,
        "name": f"{name} (HF)" if not name.endswith("(HF)") else name,
        "family": family,
        "pipeline": pipeline,
        "engine": engine,
        "engine_version": engine,
        "is_custom": True,
        "local_folder": local_folder,
        "base_repo_id": base_repo_id,
        "has_model_index": bool(data.get("has_model_index")),
        "hf_pipeline_tag": pipeline_tag,
        "tags": list(data.get("tags") or [])[:20],
        "license": data.get("license") or "",
        "default_cfg": cfg,
        "description": data.get("description") or f"Импортировано из Hugging Face: {repo_id} (движок: {engine})",
        "sources": {
            "primary": {
                "type": "huggingface",
                "repo_id": repo_id,
            }
        },
        "base_models": base_models_map,
        "engine_args": engine_args,
        "transformers": transformers,
        "text_encoders": text_encoders,
        "requirements": {
            "vram_min_gb": 8,
            "vram_recommended_gb": 12,
            "gpu_min_sm": 75,
            "default_vram_mode": "auto",
        },
        "scheduler_presets": {
            "default": {
                "sampler": "euler",
                "steps": steps,
                "cfg": cfg,
            }
        },
    }

    registry_dir = Path(__file__).parent / "registry" / "models"
    out_json = registry_dir / f"{model_id}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    reload_registry()
    return manifest


# ---------------------------------------------------------------------------
# Model card editing (built-in and custom manifests)
# ---------------------------------------------------------------------------

CARD_ENGINES = ("sd_cpp", "sdnq", "nunchaku", "orbitquant", "quantfunc")
CARD_PIPELINES = ("text_to_image", "image_edit", "img2img", "inpaint")
CARD_FAMILIES = ("flux", "klein", "qwen", "sdxl", "z-image", "moss-image", "ideogram", "qwen-layered")


def _card_registry_dir() -> Path:
    return Path(__file__).parent / "registry" / "models"


def _find_custom_manifest(model_id: str) -> tuple[Path, dict[str, Any]]:
    """Return the JSON path + payload for a registry model (built-in or custom)."""
    registry_dir = _card_registry_dir()
    candidates: list[Path] = []
    direct = registry_dir / f"{model_id}.json"
    if direct.exists():
        candidates.append(direct)
    else:
        candidates.extend(sorted(registry_dir.glob("*.json")))

    for path in candidates:
        if path.name == "registry_schema.json":
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if str(data.get("id") or path.stem) != model_id:
            continue
        return path, data

    raise ValueError(f"Модель {model_id} не найдена в реестре")


def _primary_repo(manifest: dict[str, Any]) -> str:
    sources = manifest.get("sources") or {}
    primary = sources.get("primary") or {}
    return str(primary.get("repo_id") or primary.get("repo") or "").strip()


def _clean_repo_id(value: Any, field: str, fallback: str = "") -> str:
    from .model_sniffer import normalize_hf_repo

    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        raw = fallback
    repo = normalize_hf_repo(raw) if raw else ""
    if not repo or "/" not in repo:
        raise ValueError(f"{field}: укажите репозиторий в формате author/repo")
    return repo


def _clean_component_path(value: Any, field: str) -> str:
    path = str(value or "").replace("\\", "/").strip()
    if path.startswith("/"):
        raise ValueError(f"{field}: укажите относительный путь внутри репозитория")
    if not path:
        raise ValueError(f"{field}: укажите путь к файлу внутри репозитория")
    parts = [part for part in path.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"{field}: путь не должен выходить за пределы репозитория")
    if ":" in parts[0]:
        raise ValueError(f"{field}: укажите относительный путь внутри репозитория")
    return "/".join(parts)


def _clean_size(value: Any) -> float:
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        size = 0.0
    return round(max(0.0, size), 2)


def _normalize_text_encoders(items: Any, primary_repo: str) -> list[dict[str, Any]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("text_encoders должен быть списком")
    if len(items) > 12:
        raise ValueError("Слишком много текстовых энкодеров (максимум 12)")

    encoders: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Некорректная запись текстового энкодера")
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        repo = _clean_repo_id(
            source.get("repo_id") or item.get("repo") or item.get("repo_id"),
            f"Энкодер #{index + 1}",
            fallback=primary_repo,
        )
        path = _clean_component_path(
            item.get("path") or item.get("file") or source.get("file_path"),
            f"Энкодер #{index + 1}",
        )
        te_id = slugify(str(item.get("id") or Path(path).stem)) or f"te_{index + 1}"
        base_id = te_id
        suffix = 2
        while te_id in used_ids:
            te_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(te_id)
        name = str(item.get("name") or Path(path).name).strip()[:160] or Path(path).name
        encoders.append({
            "id": te_id,
            "name": name,
            "repo": repo,
            "file": path,
            "path": path,
            "size_gb": _clean_size(item.get("size_gb")),
            "precision": str(item.get("precision") or "custom").strip()[:32] or "custom",
            "recommended": bool(item.get("recommended", index == 0)),
            "source": {"repo_id": repo, "file_path": path},
        })
    if encoders and not any(te.get("recommended") for te in encoders):
        encoders[0]["recommended"] = True
    return encoders


def _normalize_vae(item: Any, primary_repo: str) -> Optional[dict[str, Any]]:
    if not item:
        return None
    if not isinstance(item, dict):
        raise ValueError("Некорректная запись VAE")
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    repo = _clean_repo_id(
        source.get("repo_id") or item.get("repo") or item.get("repo_id"),
        "VAE",
        fallback=primary_repo,
    )
    path = _clean_component_path(item.get("path") or item.get("file") or source.get("file_path"), "VAE")
    return {
        "repo_id": repo,
        "path": path,
        "size_gb": _clean_size(item.get("size_gb")),
    }


def _normalize_aux_files(items: Any, primary_repo: str) -> list[dict[str, Any]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("aux_files должен быть списком")
    if len(items) > 24:
        raise ValueError("Слишком много вспомогательных файлов (максимум 24)")

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        if isinstance(item, dict):
            source = item.get("source") if isinstance(item.get("source"), dict) else {}
            repo = _clean_repo_id(
                source.get("repo_id") or item.get("repo_id") or item.get("repo"),
                f"Aux #{index + 1}",
                fallback=primary_repo,
            )
            path = _clean_component_path(
                item.get("path") or item.get("file") or source.get("file_path"),
                f"Aux #{index + 1}",
            )
            size = _clean_size(item.get("size_gb"))
        else:
            repo = _clean_repo_id("", f"Aux #{index + 1}", fallback=primary_repo)
            path = _clean_component_path(item, f"Aux #{index + 1}")
            size = 0.0
        key = (repo, path)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"path": path, "repo_id": repo, "size_gb": size})
    return entries


def _primary_weight_path(manifest: dict[str, Any]) -> str:
    """Return the first transformer path declared by a model card."""
    for variant in manifest.get("transformers") or []:
        if isinstance(variant, dict) and variant.get("path"):
            return str(variant["path"])
    return ""


def _reconcile_engine_args(manifest: dict[str, Any], engine: str) -> None:
    """Point engine_args at the card's primary weights after an engine switch.

    The registry uses one shared transformer list, but each runtime reads a
    different manifest field. Keeping this mapping in sync is what lets a card
    migrate between SD.CPP, SDNQ, Nunchaku, OrbitQuant, and QuantFunc.
    """
    engine_args = manifest.setdefault("engine_args", {})
    if not isinstance(engine_args, dict):
        engine_args = {}
        manifest["engine_args"] = engine_args

    weight = _primary_weight_path(manifest)
    family = str(manifest.get("family") or "").strip().lower()

    if engine in ("sdnq", "nunchaku", "orbitquant"):
        engine_args.pop("checkpoint", None)
        if weight:
            engine_args["diffusion_model"] = weight
        if not manifest.get("base_models"):
            manifest["base_models"] = {"50x-above": ".", "50x-below": "."}
    elif engine == "sd_cpp" and weight:
        if family == "sdxl":
            engine_args["checkpoint"] = weight
            engine_args.pop("diffusion_model", None)
        else:
            engine_args["diffusion_model"] = weight
            engine_args.pop("checkpoint", None)


def _apply_card_updates(manifest: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Validate editable card fields and keep engine_args in sync."""
    if not isinstance(updates, dict):
        raise ValueError("Некорректные данные карточки")

    primary_repo = _primary_repo(manifest)
    engine_changed = False

    if "name" in updates:
        name = str(updates.get("name") or "").strip()
        if not name:
            raise ValueError("Название модели не может быть пустым")
        manifest["name"] = name[:160]

    if "description" in updates:
        manifest["description"] = str(updates.get("description") or "").strip()[:2000]

    if "family" in updates:
        family = str(updates.get("family") or "").strip().lower()
        if not family or len(family) > 40:
            raise ValueError("Некорректное семейство модели")
        manifest["family"] = family

    if "pipeline" in updates:
        pipeline = str(updates.get("pipeline") or "").strip().lower()
        if pipeline not in CARD_PIPELINES:
            raise ValueError(f"Неизвестный pipeline: {pipeline}")
        manifest["pipeline"] = pipeline

    if "supports_mask" in updates:
        manifest["supports_mask"] = bool(updates.get("supports_mask"))

    if "engine" in updates:
        engine = str(updates.get("engine") or "").strip().lower()
        if engine not in CARD_ENGINES:
            raise ValueError(f"Неизвестный движок: {engine}")
        manifest["engine"] = engine
        manifest["engine_version"] = engine
        engine_changed = True

    if "license" in updates:
        manifest["license"] = str(updates.get("license") or "").strip()[:160]

    if "base_repo_id" in updates:
        raw_base = str(updates.get("base_repo_id") or "").strip()
        manifest["base_repo_id"] = _clean_repo_id(raw_base, "Базовая модель") if raw_base else ""

    if "tags" in updates:
        raw_tags = updates.get("tags")
        if isinstance(raw_tags, str):
            raw_tags = [tag.strip() for tag in raw_tags.split(",")]
        if not isinstance(raw_tags, list):
            raise ValueError("tags должен быть списком")
        manifest["tags"] = [str(tag).strip()[:40] for tag in raw_tags if str(tag).strip()][:30]

    steps = manifest.get("scheduler_presets", {}).get("default", {}).get("steps", 20)
    cfg = manifest.get("default_cfg", 1.0)
    if "steps" in updates:
        try:
            steps = int(updates.get("steps"))
        except (TypeError, ValueError):
            raise ValueError("steps должен быть целым числом")
        if not 1 <= steps <= 150:
            raise ValueError("steps должен быть от 1 до 150")
    if "cfg" in updates:
        try:
            cfg = float(updates.get("cfg"))
        except (TypeError, ValueError):
            raise ValueError("cfg должен быть числом")
        if not 0.0 <= cfg <= 30.0:
            raise ValueError("cfg должен быть от 0 до 30")

    engine_args = manifest.setdefault("engine_args", {})
    if not isinstance(engine_args, dict):
        engine_args = {}
        manifest["engine_args"] = engine_args

    if engine_changed:
        _reconcile_engine_args(manifest, engine)
        engine_args = manifest["engine_args"]

    if "text_encoders" in updates:
        encoders = _normalize_text_encoders(updates.get("text_encoders"), primary_repo)
        manifest["text_encoders"] = encoders
        for key in ("llm", "llm_repo", "llm_file", "llm_remote"):
            engine_args.pop(key, None)
        if encoders:
            primary_te = next((te for te in encoders if te.get("recommended")), encoders[0])
            engine_args["llm"] = primary_te["path"]
            engine_args["llm_repo"] = primary_te["repo"]
            engine_args["llm_file"] = primary_te["path"]

    if "vae" in updates:
        vae = _normalize_vae(updates.get("vae"), primary_repo)
        for key in ("vae", "vae_repo", "vae_file", "vae_remote"):
            engine_args.pop(key, None)
        if vae:
            engine_args["vae"] = vae["path"]
            engine_args["vae_repo"] = vae["repo_id"]
            engine_args["vae_file"] = vae["path"]
            engine_args["vae_remote"] = vae["path"]

    if "aux_files" in updates:
        aux = _normalize_aux_files(updates.get("aux_files"), primary_repo)
        if aux:
            engine_args["aux_files"] = aux
            engine_args["aux_repo"] = primary_repo
        else:
            engine_args.pop("aux_files", None)
            engine_args.pop("aux_repo", None)

    manifest["default_cfg"] = round(float(cfg), 3)
    presets = manifest.setdefault("scheduler_presets", {})
    if not isinstance(presets, dict):
        presets = {}
        manifest["scheduler_presets"] = presets
    default_preset = presets.get("default") if isinstance(presets.get("default"), dict) else {}
    default_preset["steps"] = int(steps)
    default_preset["cfg"] = round(float(cfg), 3)
    presets["default"] = default_preset

    return manifest


def get_custom_model_card(model_id: str) -> dict[str, Any]:
    """Return the raw editable manifest of any registry model (host only)."""
    _path, manifest = _find_custom_manifest(model_id)
    repo_id = _primary_repo(manifest)
    card = {
        **manifest,
        "repo_id": repo_id,
        "editable": True,
    }

    from .cache import CacheManager
    from .model_files import vae_payload_from_args

    payload = vae_payload_from_args(manifest.get("engine_args") or {}, repo_id, CacheManager())
    if payload is not None:
        card["vae"] = payload
    return card


def update_custom_model_card(model_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Validate + atomically save card edits, then reload the registry."""
    path, manifest = _find_custom_manifest(model_id)
    updated = _apply_card_updates(manifest, updates)

    backup = path.read_bytes()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
        reload_registry()
        get_registry().get(model_id)
    except Exception as exc:
        try:
            path.write_bytes(backup)
            reload_registry()
        except Exception:
            pass
        raise ValueError(f"Не удалось сохранить карточку модели: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        **updated,
        "repo_id": _primary_repo(updated),
        "editable": True,
    }


def delete_custom_model(model_id: str, delete_files: bool = False) -> bool:
    """Delete any model manifest and optionally its downloaded files.

    Built-in registry cards are deletable too: a fresh checkout ships useful
    defaults, but users must be able to remove cards they do not need.
    """
    registry_dir = Path(__file__).parent / "registry" / "models"

    target_json: Optional[Path] = None
    target_data: Optional[dict] = None

    # 1. Match directly by filename
    cand = registry_dir / f"{model_id}.json"
    if cand.exists():
        target_json = cand
    else:
        # 2. Match by "id" inside registry files
        for f in sorted(registry_dir.glob("*.json")):
            if f.name == "registry_schema.json":
                continue
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    d = json.load(jf)
                    if d.get("id") == model_id:
                        target_json = f
                        target_data = d
                        break
            except Exception:
                continue

    if not target_json or not target_json.exists():
        raise ValueError(f"Модель {model_id} не найдена в реестре")

    if target_data is None:
        try:
            with open(target_json, "r", encoding="utf-8") as jf:
                target_data = json.load(jf)
        except Exception:
            target_data = {}

    # Optionally delete downloaded files. Never remove a repository directory:
    # another model may use the same repo for its encoder, VAE or transformer.
    if delete_files:
        try:
            from .cache import CacheManager
            from .model_files import collect_delete_targets, delete_targets

            sources_data = target_data.get("sources") or {}
            primary = sources_data.get("primary") or {}
            repo_id = primary.get("repo_id") or primary.get("repo") or ""
            adapter = SimpleNamespace(
                id=target_data.get("id") or model_id,
                sources=SimpleNamespace(primary=SimpleNamespace(repo_id=repo_id)) if repo_id else None,
                transformers=[
                    SimpleNamespace(
                        id=t.get("id") or "",
                        path=t.get("path") or "",
                        repo_id=t.get("repo_id"),
                    )
                    for t in target_data.get("transformers", [])
                    if isinstance(t, dict)
                ],
                text_encoders=target_data.get("text_encoders") or [],
                engine_args=target_data.get("engine_args") or {},
            )
            cache = CacheManager()
            targets, _protected, skipped = collect_delete_targets(adapter, get_registry().list_all(), cache)
            result = delete_targets(cache, targets)
            print(f"[delete] Removed {result['deleted_files_count']} cached file(s) for {model_id}")
            if skipped:
                print(f"[delete] Kept {len(skipped)} shared file(s) for {model_id}")
        except Exception as e:
            print(f"[delete] Warning cleaning cache for {model_id}: {e}")

        # Local custom files in models/custom/
        try:
            custom_dir = get_custom_models_dir()
            for t in target_data.get("transformers", []):
                t_path = t.get("path")
                if t_path:
                    cand_file = Path(t_path)
                    if not cand_file.is_absolute():
                        cand_file = custom_dir / t_path
                    if cand_file.exists() and custom_dir in cand_file.parents:
                        cand_file.unlink()
                        print(f"[delete] Removed custom weight file: {cand_file}")
        except Exception as e:
            print(f"[delete] Warning cleaning custom files for {model_id}: {e}")

    # Remove manifest JSON
    target_json.unlink()

    # Clean up from visibility map
    try:
        from .model_visibility import remove_model_from_visibility
        remove_model_from_visibility(model_id)
    except Exception:
        pass

    reload_registry()
    return True
