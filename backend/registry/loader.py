"""
ModelRegistry — loads model definitions from JSON files in backend/registry/models/.

Each JSON file describes one model with its:
    - base_model subdirs (50x-above / 50x-below)
    - transformer variants (multiple — filtered by GPU tier + backend + steps)
    - optional prequant weights
    - optional precision configs
    - requirements (VRAM, SM)
    - scheduler presets

Adding a new model = creating a new .json file. No backend code changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..gpu import tier_for_sm, klein_tier_for_sm, TIER_50X_ABOVE


# ---------- Data classes ----------
@dataclass
class TransformerVariant:
    """A specific transformer file in a model series."""
    id: str
    path: str                       # Relative path inside the source repo
    tier: str                       # "50x-above" | "50x-below"
    backend: str                    # "lighting" | "svdq"
    steps: int                      # 4 (lighting-4steps) or 20 (svdq full)
    prequant: bool = False
    size_gb: float = 0.0
    size_bytes: int = 0
    sha256: Optional[str] = None
    recommended: bool = False
    recommended_for_gpu: list[str] = field(default_factory=list)
    klein_tier: Optional[str] = None  # Only for Klein family: "30x-below" / "40x" / "50x"
    note: Optional[str] = None  # Optional descriptive note
    repo_id: Optional[str] = None  # Optional custom remote repo_id

    @property
    def size_label(self) -> str:
        """Human-readable size."""
        if self.size_gb > 0:
            return f"{self.size_gb:.1f} GB"
        if self.size_bytes > 0:
            return f"{self.size_bytes / 1024**3:.1f} GB"
        return "unknown size"

    @property
    def short_id(self) -> str:
        """Short ID for UI display."""
        return self.id.replace("-50x-above", "").replace("-50x-below", "")


@dataclass
class PrequantWeights:
    """Optional prequant modulation weights for Lighting backend."""
    id: str
    path: str
    tier: str
    size_gb: float = 0.0
    klein_tier: Optional[str] = None


@dataclass
class PrecisionConfig:
    """Per-layer precision config JSON file."""
    id: str
    path: str
    tier: str
    klein_tier: Optional[str] = None  # Only for Klein family


@dataclass
class ModelRequirements:
    """Hardware requirements for a model."""
    vram_min_gb: int = 8
    vram_recommended_gb: int = 16
    gpu_min_sm: int = 75
    gpu_min_tier: str = "50x-below"
    default_vram_mode: str = "auto"


@dataclass
class ModelSourceSpec:
    """A source repository for a model (HF or ModelScope)."""
    type: str               # "huggingface" or "modelscope"
    repo_id: str            # "QuantFunc/Klein-9B-Series"


@dataclass
class ModelSources:
    """Primary + mirror sources for a model."""
    primary: ModelSourceSpec
    mirror: Optional[ModelSourceSpec] = None


@dataclass
class ModelSpec:
    """Full specification of a model — loaded from JSON."""
    id: str
    name: str
    family: str                     # "qwen" | "klein" | "z-image" | "ideogram" | "qwen-layered"
    pipeline: str                   # "text_to_image" | "image_edit" | "img2img" | "inpaint"
    description: str = ""
    upstream_model: str = ""
    upstream_license: str = ""
    commercial_use: bool = False

    sources: Optional[ModelSources] = None
    base_models: dict[str, str] = field(default_factory=dict)  # tier → subdir path

    transformers: list[TransformerVariant] = field(default_factory=list)
    prequant_weights: list[PrequantWeights] = field(default_factory=list)
    precision_configs: list[PrecisionConfig] = field(default_factory=list)

    requirements: ModelRequirements = field(default_factory=ModelRequirements)
    scheduler_presets: dict = field(default_factory=dict)

    # Key-protection: model weights are encrypted and need an API key
    # (bundled free-trial test key or a user key from quantfunc.com).
    key_protected: bool = False

    # Mask / inpaint support. True when the model can be driven with an explicit
    # inpaint mask (white = edit, black = preserve). Models declaring
    # pipeline "inpaint" are mask capable by definition.
    supports_mask: bool = False

    # Target engine (e.g. "quantfunc" or "sd_cpp")
    engine: str = "quantfunc"
    engine_args: dict = field(default_factory=dict)

    # Target engine version (e.g. "0.0.12" for FP4 Klein models, "0.0.17" for Qwen-Image-2.1)
    engine_version: Optional[str] = None

    # Available text encoder variants (e.g. w4a8, int8, bf16)
    text_encoders: list[dict] = field(default_factory=list)

    # Custom user-imported model flag
    is_custom: bool = False

    # Host-defined metadata for imported models
    local_folder: str = ""                 # Host-named folder inside models/cache
    base_repo_id: str = ""                 # Repo that supplies encoder/VAE/config (diffusers layout)
    download_patterns: list[str] = field(default_factory=list)  # Snapshot patterns for the base task

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSpec":
        """Build ModelSpec from a JSON dict."""
        sources = None
        if "sources" in data and data["sources"]:
            s = data["sources"]
            p_data = dict(s.get("primary", {}))
            if "repo" in p_data and "repo_id" not in p_data:
                p_data["repo_id"] = p_data.pop("repo")
            primary = ModelSourceSpec(**p_data) if p_data else None

            mirror = None
            if s.get("mirror"):
                m_data = dict(s["mirror"])
                if "repo" in m_data and "repo_id" not in m_data:
                    m_data["repo_id"] = m_data.pop("repo")
                mirror = ModelSourceSpec(**m_data)
            if primary:
                sources = ModelSources(primary=primary, mirror=mirror)

        transformers = [TransformerVariant(**t) for t in data.get("transformers", [])]
        prequant = [PrequantWeights(**p) for p in data.get("prequant_weights", [])]
        precision = [PrecisionConfig(**p) for p in data.get("precision_configs", [])]
        requirements = ModelRequirements(**data.get("requirements", {}))

        is_custom = bool(data.get("is_custom", False))
        local_folder = str(data.get("local_folder") or "").strip()
        if is_custom and local_folder and sources and sources.primary and sources.primary.repo_id:
            from ..cache import set_repo_folder_alias
            set_repo_folder_alias(sources.primary.repo_id, local_folder)

        return cls(
            id=data["id"],
            name=data["name"],
            family=data["family"],
            pipeline=data["pipeline"],
            description=data.get("description", ""),
            upstream_model=data.get("upstream", {}).get("model", ""),
            upstream_license=data.get("upstream", {}).get("license", ""),
            commercial_use=data.get("upstream", {}).get("commercial_use", False),
            sources=sources,
            base_models=data.get("base_models", {}),
            transformers=transformers,
            prequant_weights=prequant,
            precision_configs=precision,
            requirements=requirements,
            scheduler_presets=data.get("scheduler_presets", {}),
            key_protected=bool(data.get("key_protected", False)),
            supports_mask=bool(
                data.get("supports_mask", False) or data.get("pipeline") == "inpaint"
            ),
            engine=data.get("engine", "quantfunc"),
            engine_args=data.get("engine_args", {}),
            engine_version=data.get("engine_version"),
            text_encoders=data.get("text_encoders", []),
            is_custom=is_custom,
            local_folder=local_folder,
            base_repo_id=str(data.get("base_repo_id") or "").strip(),
            download_patterns=list(data.get("download_patterns") or []),
        )

    # ---------- Helpers ----------
    def get_text_encoder(self, te_id: Optional[str] = None) -> Optional[dict]:
        """Return text encoder dict by id or default recommended."""
        if not self.text_encoders:
            return None
        if te_id:
            for te in self.text_encoders:
                if te.get("id") == te_id:
                    return te
        for te in self.text_encoders:
            if te.get("recommended"):
                return te
        return self.text_encoders[0]

    def get_default_cfg(self) -> float:
        """Return default CFG / guidance_scale from scheduler_presets."""
        if not self.scheduler_presets:
            return 1.0
        for key in ("default", "lighting-4steps"):
            preset = self.scheduler_presets.get(key)
            if isinstance(preset, dict):
                if preset.get("cfg") is not None:
                    return float(preset["cfg"])
                if preset.get("guidance_scale") is not None:
                    return float(preset["guidance_scale"])
        for preset in self.scheduler_presets.values():
            if isinstance(preset, dict):
                if preset.get("cfg") is not None:
                    return float(preset["cfg"])
                if preset.get("guidance_scale") is not None:
                    return float(preset["guidance_scale"])
        return 1.0

    def get_base_model_path(self, tier: str) -> Optional[str]:
        """Get the base model subdir path for a GPU tier."""
        return self.base_models.get(tier)

    def get_transformers_for_tier(
        self,
        gpu_sm: int,
        backend: Optional[str] = None,
    ) -> list[TransformerVariant]:
        """Filter transformers by GPU tier (+ optional backend)."""
        tier = tier_for_sm(gpu_sm)
        klein_tier = klein_tier_for_sm(gpu_sm)

        result = []
        for t in self.transformers:
            # Filter by main tier
            if t.tier != tier and t.tier not in ("universal", "50x"):
                continue

            # Per-file sub-tier filter (30x-below / 40x / 50x) when GPU SM is known
            if t.klein_tier and klein_tier != "unknown" and t.klein_tier != klein_tier:
                continue

            # Optional: filter by backend
            if backend and t.backend != backend:
                continue

            result.append(t)

        # Fallback if strict sub-tier filter produced no results
        if not result:
            for t in self.transformers:
                if t.tier != tier and t.tier not in ("universal", "50x"):
                    continue
                if backend and t.backend != backend:
                    continue
                result.append(t)

        return result

    def get_recommended_transformers(self, gpu_sm: int) -> list[TransformerVariant]:
        """Get recommended transformers for current GPU."""
        all_for_tier = self.get_transformers_for_tier(gpu_sm)
        return [t for t in all_for_tier if t.recommended]

    def resolve_required_files(
        self,
        gpu_sm: int,
        backend: str = "lighting",
        steps: int = 4,
        use_prequant: bool = False,
        variant_id: Optional[str] = None,
    ) -> dict:
        """Resolve concrete file paths required for a generation.

        Returns dict with keys:
            - base_model_path: str
            - transformer_path: str
            - prequant_weights_path: Optional[str]
            - precision_config_path: Optional[str]
        """
        tier = tier_for_sm(gpu_sm)

        # Base model
        base_path = self.get_base_model_path(tier)
        if not base_path:
            raise ValueError(f"No base model for tier {tier} in {self.id}")

        transformer = None
        # 1. Exact match by variant_id if provided
        if variant_id:
            tier_candidates = self.get_transformers_for_tier(gpu_sm)
            for t in tier_candidates:
                if t.id == variant_id:
                    transformer = t
                    backend = t.backend
                    break
            if not transformer:
                for t in self.transformers:
                    if t.id == variant_id:
                        transformer = t
                        backend = t.backend
                        break

        # 2. Match by backend + steps (+ prequant flag)
        if not transformer:
            candidates = self.get_transformers_for_tier(gpu_sm, backend=backend)
            for t in candidates:
                if t.steps == steps and t.prequant == use_prequant:
                    transformer = t
                    break
            if not transformer:
                # Fallback: any matching backend + steps
                for t in candidates:
                    if t.steps == steps:
                        transformer = t
                        break
            if not transformer:
                # Last resort: any matching backend
                if candidates:
                    transformer = candidates[0]
        if not transformer:
            raise ValueError(
                f"No transformer found for backend={backend}, steps={steps}, "
                f"tier={tier}, gpu_sm={gpu_sm}"
            )

        # Optional prequant
        prequant_path = None
        if use_prequant and self.prequant_weights:
            for pw in self.prequant_weights:
                if pw.tier == tier:
                    prequant_path = pw.path
                    break

        # Optional precision config (only for Lighting backend)
        precision_path = None
        if backend == "lighting" and self.precision_configs:
            klein_tier = klein_tier_for_sm(gpu_sm)
            for pc in self.precision_configs:
                if pc.tier == tier:
                    if pc.klein_tier and pc.klein_tier != klein_tier:
                        continue
                    precision_path = pc.path
                    break

        return {
            "base_model_path": base_path,
            "transformer_path": transformer.path,
            "transformer_id": transformer.id,
            "prequant_weights_path": prequant_path,
            "precision_config_path": precision_path,
            "backend": backend,
            "steps": steps,
        }


# ---------- Registry ----------
class ModelRegistry:
    """Loads and holds all model specs from JSON files."""

    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.models: dict[str, ModelSpec] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all .json files from registry_dir."""
        if not self.registry_dir.exists():
            print(f"[registry] WARNING: registry dir does not exist: {self.registry_dir}")
            return

        for json_path in sorted(self.registry_dir.glob("*.json")):
            if json_path.name == "registry_schema.json":
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                spec = ModelSpec.from_dict(data)
                self.models[spec.id] = spec
                print(f"[registry] Loaded: {spec.id} ({len(spec.transformers)} transformers)")
            except Exception as e:
                print(f"[registry] ERROR loading {json_path.name}: {e}")

    def get(self, model_id: str) -> ModelSpec:
        """Get a model spec by ID (or by primary repo_id / upstream_model alias)."""
        if model_id in self.models:
            return self.models[model_id]
        for m in self.models.values():
            if (m.sources and m.sources.primary and m.sources.primary.repo_id == model_id) or m.upstream_model == model_id:
                return m
        raise KeyError(f"Model not found: {model_id!r}. Available: {list(self.models.keys())}")

    def list_all(self) -> list[ModelSpec]:
        """List all loaded model specs."""
        return list(self.models.values())

    def list_for_gpu(self, gpu_sm: int) -> list[ModelSpec]:
        """Filter models compatible with the current GPU."""
        return [
            m for m in self.models.values()
            if m.requirements.gpu_min_sm <= gpu_sm
        ]

    def list_for_pipeline(self, pipeline: str) -> list[ModelSpec]:
        """Filter models by pipeline type (text_to_image, image_edit, etc.)."""
        return [m for m in self.models.values() if m.pipeline == pipeline]

    def list_mask_capable(self) -> list[ModelSpec]:
        """List models that can be driven with an explicit inpaint mask."""
        return [m for m in self.models.values() if model_supports_mask(m)]


# ---------- Mask capability helpers ----------
# QuantFunc cards that ship known-defective (striped) upstream weights.  The
# global ``enable_quantfunc_models`` switch only gates these cards; other QF
# models (Qwen-Image / Qwen-Image-Edit, Z-Image, ...) stay available so
# mask-capable Qwen works without enabling the defective FLUX weights.
QF_DEFECTIVE_MODEL_IDS = frozenset({"flux_klein_4b", "flux_klein_9b"})


def quantfunc_model_disabled(spec, qf_enabled: bool) -> bool:
    """True when a QuantFunc card is blocked by the global safety switch."""
    if spec is None:
        return False
    if getattr(spec, "engine", "quantfunc") != "quantfunc":
        return False
    if qf_enabled:
        return False
    return getattr(spec, "id", "") in QF_DEFECTIVE_MODEL_IDS


def model_supports_mask(spec) -> bool:
    """True when the model card AND its engine can handle an inpaint mask.

    A card has to opt in (``supports_mask: true`` or ``pipeline: "inpaint"``)
    and the target engine must expose a documented mask argument.
    """
    if spec is None:
        return False
    if not getattr(spec, "supports_mask", False):
        return False
    from ..masking import engine_mask_support
    return engine_mask_support(getattr(spec, "engine", "quantfunc"))


# ---------- Singleton ----------
_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global ModelRegistry singleton."""
    global _registry_instance
    if _registry_instance is None:
        registry_dir = Path(__file__).parent / "models"
        _registry_instance = ModelRegistry(registry_dir)
    return _registry_instance


def reload_registry() -> ModelRegistry:
    """Force reload from disk."""
    global _registry_instance
    registry_dir = Path(__file__).parent / "models"
    _registry_instance = ModelRegistry(registry_dir)
    return _registry_instance
