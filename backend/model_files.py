"""Safe, manifest-driven deletion of downloaded model files.

It also resolves VAE source/local targets and detects already-downloaded VAEs
in both the primary model repository and a separate VAE repository.

The cache directory is shared by many models and by repo-local encoders/VAEs.
A model must therefore never be deleted by removing a repository directory:
only exact files referenced by its manifest may be removed, and files that are
still referenced by another registry model must be kept.

This module is deliberately free of FastAPI/registry imports so it can be used
by both the HTTP API and the custom-model registry code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


VAE_SUFFIXES = frozenset({".safetensors", ".pt", ".pth", ".bin", ".ckpt"})


def clean_rel(value: Any) -> str:
    """Normalize a manifest-relative path to POSIX form without a leading slash."""
    return str(value or "").replace("\\", "/").lstrip("/")


def safe_repo_rel(value: Any) -> str:
    """Return a safe repository-relative path or an empty string."""
    raw = str(value or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/"):
        return ""
    rel = clean_rel(raw)
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or ":" in parts[0]:
        return ""
    return "/".join(parts)


def _vae_declared_paths(engine_args: dict[str, Any]) -> list[str]:
    """VAE paths as declared in a manifest, most specific first."""
    paths: list[str] = []
    for key in ("vae_remote", "vae_file", "vae"):
        rel = safe_repo_rel(engine_args.get(key))
        if rel and rel not in paths:
            paths.append(rel)
    return paths


def _vae_local_path(engine_args: dict[str, Any], source_path: str) -> str:
    """Local VAE path used by the runtimes (relative to the primary repo)."""
    for key in ("vae", "vae_file"):
        rel = safe_repo_rel(engine_args.get(key))
        if rel:
            return rel
    return Path(source_path).name if source_path else ""


def _vae_candidate_paths(declared: list[str]) -> list[str]:
    """Repository-relative places a declared VAE may already be cached."""
    candidates: list[str] = []
    for rel in declared:
        name = Path(rel).name
        for candidate in (rel, name, f"vae/{name}"):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _infer_repo_vae(repo_dir: Path) -> str:
    """Find an already-downloaded repo-local VAE when a card does not declare one."""
    try:
        if not repo_dir.is_dir():
            return ""
    except OSError:
        return ""

    def rank(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if name.startswith("diffusion_pytorch_model"):
            priority = 0
        elif "texture" not in name:
            priority = 1
        else:
            priority = 2
        return priority, 0 if "vae" in name else 1, name

    vae_dir = repo_dir / "vae"
    try:
        if vae_dir.is_dir():
            files = [
                path for path in vae_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VAE_SUFFIXES
            ]
            if files:
                return f"vae/{min(files, key=rank).name}"
    except OSError:
        pass

    try:
        root_files = [
            path for path in repo_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VAE_SUFFIXES
            and ("vae" in path.name.lower() or path.name.lower() == "ae.safetensors")
        ]
        if root_files:
            return min(root_files, key=rank).name
    except OSError:
        pass
    return ""


def vae_download_refs(model, primary_repo: str) -> Optional[dict[str, str]]:
    """Source and local destination for a manifest-declared VAE download.

    Runtimes download ``vae_remote`` from ``vae_repo`` but store the file in the
    primary model repository under ``vae`` (or ``vae_file``).
    """
    engine_args = getattr(model, "engine_args", {}) or {}
    declared = _vae_declared_paths(engine_args)
    if not declared:
        return None
    source_repo = str(engine_args.get("vae_repo") or primary_repo or "")
    local_repo = primary_repo or source_repo
    local_path = _vae_local_path(engine_args, declared[0])
    if not source_repo or not local_repo or not local_path:
        return None
    return {
        "source_repo": source_repo,
        "source_path": declared[0],
        "local_repo": local_repo,
        "local_path": local_path,
    }


FAMILY_GROUPS: dict[str, str] = {
    "klein": "flux",
    "flux": "flux",
    "flux2": "flux",
    "flux3": "flux",
    "moss-image": "flux",
    "qwen": "qwen",
    "sdxl": "sdxl",
}

FAMILY_DIR_PATTERNS: dict[str, list[str]] = {
    "flux": ["flux", "klein", "moss"],
    "qwen": ["qwen"],
    "sdxl": ["sdxl"],
}

VAE_FAMILY_CANDIDATES: dict[str, list[str]] = {
    "flux": [
        "vae/diffusion_pytorch_model.safetensors",
        "vae/flux2-vae.safetensors",
        "flux2-vae.safetensors",
        "ae.safetensors",
        "diffusion_pytorch_model.safetensors",
    ],
    "qwen": [
        "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
        "qwen_image_2.1_vae_bf16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    "sdxl": [
        "sdxl_vae.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ],
}


def find_family_shared_vae(
    family: str,
    cache,
    exclude_repo: str = "",
) -> Optional[dict[str, Any]]:
    """Locate an already-downloaded VAE from another model in the same architectural family.

    FLUX models (FLUX.1, FLUX.2 Klein 4B/9B, FLUX.3 Image Klein) share an identical
    16-channel VAE. Qwen models share the Qwen VAE. When a model card specifies a
    VAE that has not yet been downloaded into its own folder, this fallback reuses
    any existing compatible VAE on disk to avoid redundant downloads.
    """
    family_norm = str(family or "").strip().lower()
    group = FAMILY_GROUPS.get(family_norm)
    if not group:
        for k, v in FAMILY_GROUPS.items():
            if k in family_norm:
                group = v
                break
    if not group:
        return None

    candidates = VAE_FAMILY_CANDIDATES.get(group, [])
    patterns = FAMILY_DIR_PATTERNS.get(group, [])
    cache_dir = getattr(cache, "cache_dir", None)
    if not cache_dir or not cache_dir.is_dir():
        return None

    ex_names = {exclude_repo.lower(), exclude_repo.split("/")[-1].lower()} if exclude_repo else set()

    dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.lower() not in ex_names]
    dirs.sort(key=lambda d: (0 if family_norm and family_norm in d.name.lower() else 1, d.name.lower()))

    for d in dirs:
        d_lower = d.name.lower()
        if patterns and not any(p in d_lower for p in patterns):
            continue

        for rel in candidates:
            fpath = d / rel
            try:
                if fpath.is_file() and fpath.stat().st_size > 1024 * 1024:
                    return {
                        "repo_id": d.name,
                        "path": rel,
                        "installed": True,
                        "shared": True,
                        "shared_from": d.name,
                        "local_repo_id": d.name,
                        "local_path": rel,
                        "local_file": fpath.as_posix(),
                        "inferred": False,
                        "resolved_repo_id": d.name,
                        "resolved_path": rel,
                    }
            except OSError:
                continue
    return None


def vae_payload_from_args(
    engine_args: dict[str, Any],
    primary_repo: str,
    cache,
    family: str = "",
) -> Optional[dict[str, Any]]:
    """Describe the VAE for a model card: source, local target and cache state.

    A declared VAE is considered installed when any of its known local layouts
    exists, either in the primary repository or in the separate VAE repository.
    When a card has no VAE declaration, an already-downloaded repo-local VAE is
    inferred so older manifests still show the component.
    """
    args = engine_args or {}
    primary_repo = str(primary_repo or "")
    declared = _vae_declared_paths(args)

    for key in ("vae_remote", "vae_file", "vae"):
        raw = str(args.get(key) or "").strip()
        if not raw:
            continue
        try:
            absolute = Path(raw)
            if absolute.is_absolute() and absolute.is_file():
                return {
                    "repo_id": str(args.get("vae_repo") or primary_repo),
                    "path": absolute.as_posix(),
                    "installed": True,
                    "local_repo_id": primary_repo,
                    "local_path": absolute.as_posix(),
                    "local_file": absolute.as_posix(),
                    "inferred": False,
                }
        except OSError:
            continue

    if not declared:
        if not primary_repo:
            return None
        inferred = _infer_repo_vae(cache.repo_dir_for(primary_repo))
        if not inferred:
            if family:
                shared = find_family_shared_vae(family, cache, exclude_repo=primary_repo)
                if shared:
                    return shared
            return None
        return {
            "repo_id": primary_repo,
            "path": inferred,
            "installed": True,
            "local_repo_id": primary_repo,
            "local_path": inferred,
            "local_file": (cache.repo_dir_for(primary_repo) / inferred).as_posix(),
            "inferred": True,
        }

    source_repo = str(args.get("vae_repo") or primary_repo)
    local_path = _vae_local_path(args, declared[0])
    source_path = declared[0]
    candidate_paths = _vae_candidate_paths(declared)

    repos: list[str] = []
    for repo in (primary_repo, source_repo):
        if repo and repo not in repos:
            repos.append(repo)

    for repo in repos:
        repo_dir = cache.repo_dir_for(repo)
        for rel in candidate_paths:
            try:
                if (repo_dir / rel).is_file():
                    return {
                        "repo_id": source_repo or repo,
                        "path": source_path,
                        "installed": True,
                        "local_repo_id": primary_repo or repo,
                        "local_path": local_path,
                        "local_file": (repo_dir / rel).as_posix(),
                        "inferred": False,
                        "resolved_repo_id": repo,
                        "resolved_path": rel,
                    }
            except OSError:
                continue

    # Fallback to family-level shared VAE
    if family:
        shared = find_family_shared_vae(family, cache, exclude_repo=primary_repo)
        if shared:
            return {
                "repo_id": source_repo or primary_repo,
                "path": source_path,
                "installed": True,
                "shared": True,
                "shared_from": shared.get("shared_from"),
                "local_repo_id": shared.get("local_repo_id"),
                "local_path": shared.get("local_path"),
                "local_file": shared.get("local_file"),
                "inferred": False,
                "resolved_repo_id": shared.get("resolved_repo_id"),
                "resolved_path": shared.get("resolved_path"),
            }

    return {
        "repo_id": source_repo or primary_repo,
        "path": source_path,
        "installed": False,
        "local_repo_id": primary_repo or source_repo,
        "local_path": local_path,
        "inferred": False,
    }


def vae_payload(model, primary_repo: str, cache) -> Optional[dict[str, Any]]:
    """Model-object wrapper around :func:`vae_payload_from_args`."""
    family = getattr(model, "family", "") or ""
    return vae_payload_from_args(getattr(model, "engine_args", {}) or {}, primary_repo, cache, family=family)


def _vae_files_in_repo(repo_id: str, repo_dir: Path) -> list[dict[str, Any]]:
    """Every VAE-looking file already present in one repository cache."""
    found: list[dict[str, Any]] = []
    try:
        if not repo_dir.is_dir():
            return found
    except OSError:
        return found

    def add(rel: str) -> None:
        rel = clean_rel(rel)
        if not rel:
            return
        found.append({
            "repo_id": repo_id,
            "path": rel,
            "installed": True,
            "local_repo_id": repo_id,
            "local_path": rel,
            "local_file": (repo_dir / rel).as_posix(),
            "inferred": True,
            "resolved_repo_id": repo_id,
            "resolved_path": rel,
        })

    vae_dir = repo_dir / "vae"
    try:
        if vae_dir.is_dir():
            for path in sorted(vae_dir.iterdir(), key=lambda p: p.name.lower()):
                if path.is_file() and path.suffix.lower() in VAE_SUFFIXES:
                    add(f"vae/{path.name}")
    except OSError:
        pass

    try:
        for path in sorted(repo_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in VAE_SUFFIXES:
                continue
            name = path.name.lower()
            if "vae" in name or name == "ae.safetensors":
                add(path.name)
    except OSError:
        pass
    return found


def vae_payload_list(model, primary_repo: str, cache) -> list[dict[str, Any]]:
    """Declared VAE first, followed by every other VAE found on disk.

    Model cards use this to show all installed VAEs, not just the manifest one.
    Extra entries are read-only and marked ``inferred`` so the UI does not offer
    a delete action that is not backed by a manifest declaration.
    """
    engine_args = getattr(model, "engine_args", {}) or {}
    primary_repo = str(primary_repo or "")
    family = getattr(model, "family", "") or ""
    declared = vae_payload_from_args(engine_args, primary_repo, cache, family=family)

    repos: list[str] = []
    for repo in (primary_repo, str(engine_args.get("vae_repo") or "")):
        if repo and repo not in repos:
            repos.append(repo)

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    if declared:
        seen.add(Path(str(declared.get("resolved_path") or declared.get("path") or "")).name.lower())
        seen.add(Path(str(declared.get("path") or "")).name.lower())
        entries.append(declared)

    for repo in repos:
        try:
            repo_dir = cache.repo_dir_for(repo)
        except Exception:
            continue
        for item in _vae_files_in_repo(repo, repo_dir):
            name = Path(item["path"]).name.lower()
            if name in seen:
                continue
            seen.add(name)
            entries.append(item)

    return entries


def model_repo_id(model) -> str:
    sources = getattr(model, "sources", None)
    primary = getattr(sources, "primary", None) if sources else None
    return str(getattr(primary, "repo_id", "") or "")


def encoder_component_refs(encoder: dict[str, Any], primary_repo: str) -> list[tuple[str, str]]:
    """Possible cache locations for one text-encoder entry.

    Encoders may live in their own repository or inside the primary model
    repository (sometimes under ``text_encoders/``). The runtime probes all of
    these locations, so deletion and shared-file protection must too.
    """
    source = encoder.get("source") if isinstance(encoder.get("source"), dict) else {}
    repo = str(source.get("repo_id") or encoder.get("repo") or primary_repo or "")
    local_name = clean_rel(encoder.get("path") or encoder.get("file"))
    remote_name = clean_rel(source.get("file_path") or local_name)

    refs: list[tuple[str, str]] = []
    for candidate_repo in (repo, primary_repo):
        if not candidate_repo:
            continue
        for path in (local_name, remote_name):
            if not path:
                continue
            for rel in (path, f"text_encoders/{path}"):
                key = (str(candidate_repo), rel)
                if key not in refs:
                    refs.append(key)
    return refs


def vae_component_refs(
    engine_args: dict[str, Any],
    primary_repo: str,
    cache=None,
) -> list[tuple[str, str]]:
    """Possible cache locations for the VAE declared by a model card.

    Older manifests may not declare a VAE but still have a repo-local
    ``vae/*`` file that the UI inferred. When ``cache`` is provided, that
    inferred file is returned as a deletion target too.
    """
    args = engine_args or {}
    primary_repo = str(primary_repo or "")
    declared = _vae_declared_paths(args)
    if not declared:
        if cache is not None and primary_repo:
            inferred = _infer_repo_vae(cache.repo_dir_for(primary_repo))
            if inferred:
                return [(primary_repo, inferred)]
        return []

    source_repo = str(args.get("vae_repo") or primary_repo or "")
    local_path = _vae_local_path(args, declared[0])
    refs: list[tuple[str, str]] = []
    for repo in (source_repo, primary_repo):
        if not repo:
            continue
        for rel in _vae_candidate_paths(declared):
            key = (repo, rel)
            if key not in refs:
                refs.append(key)
    if primary_repo and local_path:
        key = (primary_repo, local_path)
        if key not in refs:
            refs.append(key)
    return refs


def aux_component_refs(engine_args: dict[str, Any], primary_repo: str, index: int = 0) -> list[tuple[str, str]]:
    """Cache locations for one Aux/sidecar entry (``index`` is zero-based)."""
    args = engine_args or {}
    items = args.get("aux_files") or []
    if not isinstance(items, list) or index < 0 or index >= len(items):
        return []
    item = items[index]
    if isinstance(item, dict):
        repo = str(item.get("repo_id") or primary_repo or "")
        path = clean_rel(item.get("path"))
    else:
        repo = str(primary_repo or "")
        path = clean_rel(item)
    if repo and path:
        return [(repo, path)]
    return []


def model_file_refs(model) -> set[tuple[str, str]]:
    """Every ``(repo_id, relative_path)`` referenced by one model manifest."""
    primary_repo = model_repo_id(model)
    refs: set[tuple[str, str]] = set()

    for variant in getattr(model, "transformers", []) or []:
        repo = getattr(variant, "repo_id", None) or primary_repo
        path = clean_rel(getattr(variant, "path", ""))
        if repo and path:
            refs.add((str(repo), path))

    for encoder in getattr(model, "text_encoders", []) or []:
        refs.update(encoder_component_refs(encoder, primary_repo))

    engine_args = getattr(model, "engine_args", {}) or {}
    vae_repo = engine_args.get("vae_repo") or primary_repo
    vae_path = safe_repo_rel(
        engine_args.get("vae_remote") or engine_args.get("vae_file") or engine_args.get("vae")
    )
    if vae_repo and vae_path:
        refs.add((str(vae_repo), vae_path))
    local_vae = _vae_local_path(engine_args, vae_path)
    if primary_repo and local_vae:
        refs.add((primary_repo, local_vae))

    for aux in engine_args.get("aux_files") or []:
        if isinstance(aux, dict):
            repo = aux.get("repo_id") or primary_repo
            path = clean_rel(aux.get("path"))
        else:
            repo, path = primary_repo, clean_rel(aux)
        if repo and path:
            refs.add((str(repo), path))

    return refs


def split_shared_refs(
    model,
    all_models: list[Any],
    refs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split component refs into safe-to-delete and shared-with-other-models.

    ``model`` is the card whose component is being removed. A file referenced by
    any other registered model is never deleted unless the caller explicitly
    opts in after showing the warning.
    """
    model_id = getattr(model, "id", None)
    protected: set[tuple[str, str]] = set()
    for other in all_models:
        if getattr(other, "id", None) != model_id:
            protected.update(model_file_refs(other))

    safe: list[tuple[str, str]] = []
    shared: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        if ref in protected:
            shared.append(ref)
        else:
            safe.append(ref)
    return safe, shared


def collect_delete_targets(
    model,
    all_models: list[Any],
    cache,
    variant_id: Optional[str] = None,
) -> tuple[list[tuple[str, str]], set[tuple[str, str]], set[str]]:
    """Collect exact files to delete for a model or one variant.

    Returns ``(targets, protected, skipped_shared)``. ``protected`` contains
    every file referenced by another model; ``skipped_shared`` contains the
    subset that would otherwise have been deleted.
    """
    primary_repo = model_repo_id(model)
    if not primary_repo:
        return [], set(), set()

    protected: set[tuple[str, str]] = set()
    for other in all_models:
        if getattr(other, "id", None) != getattr(model, "id", None):
            protected.update(model_file_refs(other))

    targets: list[tuple[str, str]] = []
    skipped_shared: set[str] = set()

    def add(repo: str, rel_value: Any, protect: bool = True) -> None:
        rel = clean_rel(rel_value)
        if not repo or not rel:
            return
        key = (repo, rel)
        if key in targets:
            return
        if protect and key in protected:
            skipped_shared.add(f"{repo}/{rel}")
            return
        targets.append(key)

    selected_variant = None
    if variant_id:
        selected_variant = next(
            (variant for variant in getattr(model, "transformers", []) or []
             if getattr(variant, "id", None) == variant_id),
            None,
        )
        if selected_variant is None:
            raise KeyError(variant_id)

    if selected_variant is not None:
        add(getattr(selected_variant, "repo_id", None) or primary_repo, selected_variant.path)

        # Keep OrbitQuant/SVDQ sidecars while any other variant still lives in
        # the same folder. They are not listed in the manifest but are required
        # to reconstruct the transformer.
        variant_repo = getattr(selected_variant, "repo_id", None) or primary_repo
        variant_dir = cache.path_for(variant_repo, selected_variant.path).parent
        siblings = [
            variant for variant in getattr(model, "transformers", []) or []
            if getattr(variant, "id", None) != getattr(selected_variant, "id", None)
            and cache.path_for(
                getattr(variant, "repo_id", None) or primary_repo, variant.path
            ).parent == variant_dir
        ]
        try:
            inside_cache = variant_dir.resolve().is_relative_to(cache.cache_dir.resolve())
        except OSError:
            inside_cache = False
        if not siblings and inside_cache and variant_dir.is_dir():
            repo_dir = cache.repo_dir_for(variant_repo)
            for sidecar in variant_dir.iterdir():
                low = sidecar.name.lower()
                if not (sidecar.is_file() and (low.startswith("orbitquant_") or low.startswith("svdq_"))):
                    continue
                try:
                    rel_sidecar = sidecar.relative_to(repo_dir).as_posix()
                except ValueError:
                    continue
                add(variant_repo, rel_sidecar)
        return targets, protected, skipped_shared

    for variant in getattr(model, "transformers", []) or []:
        add(getattr(variant, "repo_id", None) or primary_repo, variant.path)

    # Repo-local encoders/VAEs belong to this model. Shared encoders in other
    # repositories are never touched because they are protected above.
    for encoder in getattr(model, "text_encoders", []) or []:
        source = encoder.get("source") or {}
        repo = source.get("repo_id") or encoder.get("repo") or primary_repo
        if repo == primary_repo:
            add(repo, encoder.get("path") or encoder.get("file"))

    engine_args = getattr(model, "engine_args", {}) or {}
    vae_repo = engine_args.get("vae_repo") or primary_repo
    vae_path = safe_repo_rel(
        engine_args.get("vae_remote") or engine_args.get("vae_file") or engine_args.get("vae")
    )
    local_vae = _vae_local_path(engine_args, vae_path)
    if local_vae:
        add(primary_repo, local_vae)
    if vae_repo == primary_repo:
        add(vae_repo, vae_path)

    for aux in engine_args.get("aux_files") or []:
        if isinstance(aux, dict):
            repo = aux.get("repo_id") or primary_repo
            path = aux.get("path")
        else:
            repo, path = primary_repo, aux
        if repo == primary_repo:
            add(repo, path)

    return targets, protected, skipped_shared


def delete_targets(cache, targets: list[tuple[str, str]]) -> dict[str, Any]:
    """Delete exact cached files (and their shards/partials), then prune empty dirs."""
    cache_root = cache.cache_dir.resolve()
    deleted_files: list[str] = []
    removed_dirs: list[str] = []
    freed_bytes = 0
    touched_dirs: set[Path] = set()

    for repo, rel in targets:
        existing_path = cache.find_existing_file(repo, rel)
        base_path = existing_path or cache.path_for(repo, rel)
        try:
            if not base_path.resolve().is_relative_to(cache_root):
                continue
        except OSError:
            continue

        candidates = [base_path]
        match = re.match(
            r"^(?P<stem>.+)-00001-of-(?P<total>\d{5})(?P<ext>\.safetensors|\.gguf)$",
            base_path.name,
            re.I,
        )
        if match:
            pattern = f"{match.group('stem')}-*-of-{match.group('total')}{match.group('ext')}"
            candidates.extend(sorted(base_path.parent.glob(pattern)))

        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                if not candidate.resolve().is_relative_to(cache_root):
                    continue
            except OSError:
                continue

            freed_bytes += candidate.stat().st_size
            candidate.unlink(missing_ok=True)
            deleted_files.append(candidate.relative_to(cache_root).as_posix())
            touched_dirs.add(candidate.parent)

            partial = candidate.with_suffix(candidate.suffix + ".incomplete")
            if partial.is_file():
                freed_bytes += partial.stat().st_size
                partial.unlink(missing_ok=True)
                deleted_files.append(partial.relative_to(cache_root).as_posix())

    # Remove only empty directories, never a directory that still has files.
    for start in sorted(touched_dirs, key=lambda path: len(path.parts), reverse=True):
        current = start
        while current != cache_root and current.is_relative_to(cache_root) and current.is_dir():
            try:
                if any(current.iterdir()):
                    break
                current.rmdir()
                removed_dirs.append(current.relative_to(cache_root).as_posix())
            except OSError:
                break
            current = current.parent

    return {
        "deleted_files": deleted_files,
        "deleted_files_count": len(deleted_files),
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 1),
        "removed_dirs": removed_dirs,
    }
