"""
Worker — subprocess wrapper for QuantFunc CLI.

Spawns quantfunc / quantfunc.exe as a subprocess, captures stdout/stderr,
parses progress events, returns the generated image.

Architecture (per technical report):
    Gradio UI → backend.worker.generate() → subprocess.run([quantfunc, ...]) → PNG file

CRITICAL: All subprocess arguments go through backend.security.safe_subprocess_args().
NEVER use shell=True or os.system().
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("worker")

from .cache import CacheManager
from .engine import engine_cli_path, is_engine_installed, has_api_key_available
from .gpu import GPUInfo
from .registry import get_registry, ModelSpec
from .security import (
    safe_subprocess_args,
    validate_int,
    validate_path,
    validate_prompt,
    validate_resolution,
)
from .settings import VRAMMode, get_settings


# ---------- Status enum ----------
class JobStatus:
    QUEUED = "queued"
    LOADING = "loading"
    DOWNLOADING = "downloading"
    INITIALIZING = "initializing"
    GENERATING = "generating"
    SAVING = "saving"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# ---------- Progress event ----------
@dataclass
class ProgressEvent:
    """A progress event emitted during generation."""
    status: str                       # JobStatus value
    progress: float = 0.0             # 0.0 to 1.0
    message: str = ""
    step: Optional[int] = None        # Current denoising step
    total_steps: Optional[int] = None  # Total denoising steps
    speed_bps: Optional[float] = None  # Download speed (bytes/sec) — only when downloading
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    image_path: Optional[str] = None  # Final image path (when complete)
    raw_image_path: Optional[str] = None  # Raw engine output (before mask compositing)
    error: Optional[str] = None       # Error message (when error)


# ---------- Generate request ----------
@dataclass
class GenerateRequest:
    """Input parameters for a generate call."""
    model_id: str
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None        # None = random
    width: int = 1024
    height: int = 1024
    steps: int = 4
    cfg: float = 1.0
    noise: float = 0.0
    backend: str = "lighting"         # "lighting" | "svdq"
    use_prequant: bool = False
    init_image_path: Optional[Path] = None  # For img2img / image edit
    init_image_strength: float = 0.7

    # ---- Mask / inpaint -------------------------------------------------
    # Mask semantics: white = regenerate, black = keep, grey = soft blend.
    # The mask may be sent as a PNG data URL (browser) or as a file path.
    mask_data_url: Optional[str] = None
    mask_path: Optional[Path] = None
    mask_strength: float = 1.0           # multiplies the mask (0 disables inpaint)
    mask_grow: int = 6                   # dilation in px (ComfyUI VAEEncodeForInpaint)
    mask_blur: float = 0.0               # gaussian blur sigma in px
    mask_invert: bool = False            # swap black/white before use
    lora_paths: list[Path] = field(default_factory=list)
    lora_strengths: list[float] = field(default_factory=list)
    output_filename: Optional[str] = None
    variant_id: Optional[str] = None
    sampler: Optional[str] = None
    text_encoder_id: Optional[str] = None
    custom_text_encoder_path: Optional[str] = None
    vae: Optional[str] = None
    custom_vae_path: Optional[str] = None
    ref_mode: str = "control_ref"      # "control_ref" | "controlnet_canny" | "img2img"
    control_net_path: Optional[str] = None
    owner_id: str = "host"
    owner_name: str = "Хост"
    shared: bool = True


# ---------- Result ----------
@dataclass
class GenerateResult:
    """Result of a successful generation."""
    image_path: Path
    seed_used: int
    generation_time_sec: float
    model_id: str
    backend: str
    steps: int
    prompt: str


# ---------- Progress parsing ----------
# Engine emits lines like:
#   [step 1/4]
#   [step 2/4]
#   [progress 0.45]
#   [complete] /path/to/output.png
_STEP_RE = re.compile(r"\[step\s+(\d+)/(\d+)\]", re.IGNORECASE)
_PROGRESS_RE = re.compile(r"\[progress\s+([0-9.]+)\]", re.IGNORECASE)
_COMPLETE_RE = re.compile(r"\[complete\]\s*(\S+)", re.IGNORECASE)
_ERROR_RE = re.compile(r"\[error\]\s*(.+)|^(?:Error|Exception):\s*(.+)", re.IGNORECASE)

# Engine-side key-protection failure (checkObfuscation in ComponentImpl.cpp).
# Matched against raw output lines so we can give actionable advice.
_KEY_PROTECTED_RE = re.compile(
    r"checkObfuscation|key-protected|API key is required|provide --apikey|"
    r"api[_ ]?key.*(invalid|required|missing)",
    re.IGNORECASE,
)

_KEY_PROTECTED_HELP = (
    "This model is key-protected. Fix it in one of these ways:\n"
    "  1. Settings → Engine → 'Restore bundled test key' — restores "
    "bin/<os>/config.json with the shared free-trial test key (works during "
    "the public beta, no registration needed).\n"
    "  2. Settings → Engine → 'QuantFunc API key' — enter your own key "
    "(get one free at https://www.quantfunc.com)."
)


def parse_engine_output(line: str) -> Optional[ProgressEvent]:
    """Parse a single line of engine stdout. Returns ProgressEvent or None."""
    line = line.strip()
    if not line:
        return None

    # Step progress
    m = _STEP_RE.search(line)
    if m:
        step, total = int(m.group(1)), int(m.group(2))
        progress = step / total if total > 0 else 0.0
        return ProgressEvent(
            status=JobStatus.GENERATING,
            progress=progress,
            step=step,
            total_steps=total,
            message=f"Step {step}/{total}",
        )

    # Explicit progress
    m = _PROGRESS_RE.search(line)
    if m:
        return ProgressEvent(
            status=JobStatus.GENERATING,
            progress=float(m.group(1)),
            message=f"{float(m.group(1)) * 100:.0f}%",
        )

    # Completion
    m = _COMPLETE_RE.search(line)
    if m:
        return ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=m.group(1),
            message="Complete",
        )

    # Error
    m = _ERROR_RE.search(line)
    if m:
        err_msg = m.group(1) or m.group(2)
        return ProgressEvent(
            status=JobStatus.ERROR,
            error=err_msg.strip(),
            message=err_msg.strip(),
        )

    # Unrecognized line — treat as info message
    return ProgressEvent(
        status=JobStatus.GENERATING,
        message=line,
    )


# ---------- Worker ----------
class Worker:
    """Subprocess wrapper for QuantFunc CLI. One job at a time."""

    def __init__(self):
        self.cache = CacheManager()
        self._current_process: Optional[subprocess.Popen] = None
        self._sd_worker = None
        self._sdnq_worker = None
        self._cancel_flag = threading.Event()
        self._lock = threading.Lock()

    # ---------- Pre-flight checks ----------
    def _check_engine(self, version: Optional[str] = None) -> None:
        """Verify engine binary is installed for target version."""
        if not is_engine_installed(version):
            ver_str = f" v{version}" if version else ""
            raise RuntimeError(
                f"QuantFunc engine{ver_str} not installed. "
                "Go to Settings → Download engine to install it."
            )

    def _check_cuda_libs(self) -> None:
        """Verify critical CUDA DLLs are available. If missing, give clear instructions.

        cuDNN is OPTIONAL — warn but don't block, engine may not need it
        (or may report the missing DLL in its own error message).
        """
        from .engine import check_cuda_dependencies
        result = check_cuda_dependencies()
        if not result["installed"]:
            missing = result["missing"]
            dep_filename = result.get("dep_bundle_filename", "")
            raise RuntimeError(
                f"CUDA libraries missing: {', '.join(missing)}. "
                f"The QuantFunc engine needs CUDA runtime to work. "
                f"Go to Settings → Download CUDA deps to download {dep_filename} "
                f"(~1 GB, extracts automatically), or install CUDA Toolkit "
                f"from https://developer.nvidia.com/cuda-toolkit"
            )
        # Optional warning for cuDNN (don't block)
        if result.get("missing_optional"):
            missing_opt = result["missing_optional"]
            print(f"[worker] WARNING: Optional CUDA libs missing: {', '.join(missing_opt)} "
                  f"(cuDNN — engine may fail if it needs it)")

    def _check_gpu(self, gpu_info: GPUInfo, model: ModelSpec) -> None:
        """Verify GPU is compatible with model."""
        if gpu_info.sm < model.requirements.gpu_min_sm:
            raise RuntimeError(
                f"GPU SM {gpu_info.sm} is below minimum required {model.requirements.gpu_min_sm} "
                f"for model {model.name}."
            )
        if gpu_info.vram_gb < model.requirements.vram_min_gb:
            raise RuntimeError(
                f"VRAM {gpu_info.vram_gb:.1f} GB is below minimum {model.requirements.vram_min_gb} GB "
                f"for model {model.name}."
            )

    def _ensure_qwen_image_2_1_configs(self, model_dir: Path) -> None:
        """Ensure QwenImage21Pipeline configs are present and valid."""
        midx_path = model_dir / "model_index.json"
        t_cfg_path = model_dir / "transformer" / "config.json"
        s_cfg_path = model_dir / "scheduler" / "scheduler_config.json"

        need_midx = False
        if not midx_path.exists():
            need_midx = True
        else:
            try:
                data = json.loads(midx_path.read_text(encoding="utf-8"))
                if data.get("_class_name") != "QwenImage21Pipeline" or "transformer" not in data:
                    need_midx = True
            except Exception:
                need_midx = True

        if need_midx:
            midx = {
                "_class_name": "QwenImage21Pipeline",
                "_diffusers_version": "0.37.0.dev0",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage21"],
            }
            midx_path.write_text(json.dumps(midx, indent=2), encoding="utf-8")

        if not t_cfg_path.exists():
            t_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            t_cfg = {
                "_class_name": "QwenImage21Transformer2DModel",
                "_diffusers_version": "0.37.0.dev0",
                "attention_head_dim": 128,
                "axes_dims_rope": [16, 56, 56],
                "context_in_dim": 4096,
                "in_channels": 64,
                "num_attention_heads": 32,
                "num_layers": 32,
                "out_channels": 64,
                "patch_size": 1,
                "mlp_ratio": 3,
                "eps": 1e-06,
                "causal_condition": True,
            }
            t_cfg_path.write_text(json.dumps(t_cfg, indent=2), encoding="utf-8")

        if not s_cfg_path.exists():
            s_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            s_cfg = {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "_diffusers_version": "0.37.0.dev0",
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": 0.02,
                "time_shift_type": "exponential",
                "use_dynamic_shifting": True,
            }
            s_cfg_path.write_text(json.dumps(s_cfg, indent=2), encoding="utf-8")

        # Ensure tokenizer directory and files exist (vocab.json, merges.txt, tokenizer.json)
        tok_dir = model_dir / "tokenizer"
        tok_file = tok_dir / "tokenizer.json"
        root_tok_file = model_dir / "tokenizer.json"
        if not tok_file.exists() and not root_tok_file.exists():
            try:
                tok_dir.mkdir(parents=True, exist_ok=True)
                from .sources import HuggingFaceSource
                hf = HuggingFaceSource(token=self._get_token("huggingface"))
                for fname in ["tokenizer.json", "vocab.json", "merges.txt"]:
                    try:
                        hf.download_file(
                            repo_id="Qwen/Qwen3-VL-8B-Instruct",
                            file_path=fname,
                            local_dir=tok_dir,
                        )
                    except Exception as e:
                        print(f"[worker] Warning downloading {fname}: {e}")
            except Exception as e:
                print(f"[worker] Could not auto-download tokenizer files: {e}")

    def _resolve_local_paths(
        self,
        model: ModelSpec,
        required_files: dict,
        gpu_info: GPUInfo,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        request: Optional[GenerateRequest] = None,
    ) -> dict:
        """Resolve remote file paths to local cache paths.

        Downloads missing files if needed (lazy download on Generate).
        """
        from .sources import get_source

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        settings = get_settings()
        source_spec = model.sources.primary
        if settings.download_source_priority == "hf_first" and model.sources.mirror and model.sources.mirror.type == "huggingface":
            source_spec = model.sources.mirror
        elif settings.download_source_priority == "ms_first" and model.sources.mirror and model.sources.mirror.type == "modelscope":
            source_spec = model.sources.mirror
        source = get_source(source_spec.type, token=self._get_token(source_spec.type))

        # Repo ID for cache lookup
        repo_id = source_spec.repo_id
        local_repo_dir = self.cache.repo_dir_for(repo_id)

        # Check each required file
        required_paths = {}
        need_base_download = False
        need_transformer_download = False
        need_precision_download = False

        # Base model — check if directory exists with model_index.json
        base_path = required_files["base_model_path"]
        local_base = local_repo_dir / base_path

        # Custom VAE or Qwen-Image-2.1 VAE handling
        custom_vae_str = getattr(request, "custom_vae_path", None) if request else None
        if custom_vae_str:
            custom_vae_p = Path(custom_vae_str)
            if not custom_vae_p.is_file():
                raise RuntimeError(f"Custom VAE file not found: {custom_vae_p}")
            required_paths["vae"] = custom_vae_p

        if model.id == "qwen_image_2_1" or getattr(model, "text_encoders", None):
            if model.id == "qwen_image_2_1":
                self._ensure_qwen_image_2_1_configs(local_base)

                if not required_paths.get("vae"):
                    settings = get_settings()
                    vae_type = (getattr(request, "vae", None) if request else None) or getattr(settings, "qwen_vae_type", "texture_fix")
                    if vae_type == "official":
                        vae_filename = "qwen_image_2.1_vae_bf16.safetensors"
                        vae_repo = "Comfy-Org/Qwen-Image-2.1"
                        vae_remote = "vae/qwen_image_2.1_vae_bf16.safetensors"
                    else:
                        vae_filename = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
                        vae_repo = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
                        vae_remote = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"

                    # Check if VAE already exists in local_base, vae/ subdir, or other model caches
                    vae_candidates = [
                        local_base / vae_filename,
                        local_base / "vae" / vae_filename,
                        settings.models_dir / "Qwen-Image-2.1-4bit" / vae_filename,
                        settings.models_dir / "Qwen-Image-2.1-4bit" / "vae" / vae_filename,
                        settings.models_dir / "Qwen-Image-2.1-GGUF" / vae_filename,
                        settings.models_dir / "vae" / vae_filename,
                    ]
                    vae_file = next((p for p in vae_candidates if p.exists()), local_base / vae_filename)

                    # Auto-download VAE (644 MB) if missing
                    if not vae_file.exists():
                        emit(ProgressEvent(
                            status=JobStatus.DOWNLOADING,
                            progress=0.1,
                            message=f"Downloading Qwen-Image-2.1 VAE ({vae_filename}, 644 MB)...",
                        ))
                        from .sources import HuggingFaceSource
                        hf = HuggingFaceSource(token=self._get_token("huggingface"))
                        hf.download_file(
                            repo_id=vae_repo,
                            file_path=vae_remote,
                            local_dir=local_base,
                        )
                        downloaded_vae = local_base / vae_remote
                        if downloaded_vae.exists() and not (local_base / vae_filename).exists():
                            downloaded_vae.rename(local_base / vae_filename)
                            vae_file = local_base / vae_filename
                        elif (local_base / vae_filename).exists():
                            vae_file = local_base / vae_filename
                        elif downloaded_vae.exists():
                            vae_file = downloaded_vae

                    if vae_file.exists():
                        required_paths["vae"] = vae_file

            # Determine text encoder to use
            te_file_path = None
            te_precision = "int4"

            custom_path_str = getattr(request, "custom_text_encoder_path", None) if request else None
            if custom_path_str:
                custom_p = Path(custom_path_str)
                if custom_p.exists():
                    te_file_path = custom_p
                    te_precision = "int4"
                else:
                    raise RuntimeError(f"Custom text encoder file not found: {custom_p}")
            else:
                te_id = getattr(request, "text_encoder_id", None) if request else None
                # Check if te_id matches an existing file directly
                if te_id:
                    cand = next((p for p in [
                        local_base / te_id,
                        local_base / "text_encoders" / te_id,
                        settings.models_dir / "text_encoders" / te_id,
                        settings.models_dir / "Qwen-Image-2.1-4bit" / te_id,
                        settings.models_dir / "Qwen-Image-2.1-GGUF" / te_id,
                    ] if p.exists()), None)
                    if cand:
                        te_file_path = cand
                        te_precision = "int4"

                if not te_file_path:
                    te_spec = model.get_text_encoder(te_id)
                    if not te_spec and getattr(model, "text_encoders", None):
                        te_spec = model.text_encoders[0]

                    target_filename = te_spec.get("file", "qwen3vl_8b_bf16.safetensors") if te_spec else "qwen3vl_8b_bf16.safetensors"
                    te_precision = te_spec.get("precision", "int4") if te_spec else "int4"

                    # Check if file already exists in local_base or standard paths
                    candidate_paths = [
                        local_base / target_filename,
                        local_base / "text_encoders" / target_filename,
                        local_base / "text_encoder" / target_filename,
                    ]

                    found_path = next((p for p in candidate_paths if p.exists()), None)
                    if found_path:
                        te_file_path = found_path
                    else:
                        # Auto-download the selected text encoder
                        te_source = te_spec.get("source", {}) if te_spec else {}
                        repo = te_source.get("repo_id", "Comfy-Org/Qwen-Image-2.1")
                        remote_file = te_source.get("file_path", f"text_encoders/{target_filename}")
                        size_gb = te_spec.get("size_gb", 16.33) if te_spec else 16.33

                        emit(ProgressEvent(
                            status=JobStatus.DOWNLOADING,
                            progress=0.15,
                            message=f"Downloading text encoder ({target_filename}, ~{size_gb:.1f} GB)...",
                        ))
                        from .sources import HuggingFaceSource
                        hf = HuggingFaceSource(token=self._get_token("huggingface"))
                        hf.download_file(
                            repo_id=repo,
                            file_path=remote_file,
                            local_dir=local_base,
                        )
                        downloaded_te = local_base / "text_encoders" / target_filename
                        final_te = local_base / target_filename
                        if downloaded_te.exists() and not final_te.exists():
                            downloaded_te.rename(final_te)
                            te_file_path = final_te
                        elif final_te.exists():
                            te_file_path = final_te
                        elif downloaded_te.exists():
                            te_file_path = downloaded_te
                        else:
                            raise RuntimeError(f"Failed to download text encoder {target_filename} from {repo}")

            if te_file_path:
                required_paths["text_encoder"] = te_file_path
                required_paths["text_precision"] = te_precision

        if not (local_base / "model_index.json").exists():
            need_base_download = True
        required_paths["model_dir"] = local_base

        # Resolve tokenizer directory if available
        for tok_candidate in [
            local_base / "tokenizer",
            local_base,
            local_repo_dir / "tokenizer",
            local_repo_dir,
        ]:
            if (tok_candidate / "tokenizer.json").exists() or (tok_candidate / "vocab.json").exists():
                required_paths["tokenizer"] = tok_candidate
                break

        # Transformer file (top-level transformer/ dir, NOT inside base-model)
        transformer_path = required_files["transformer_path"]
        local_transformer = local_repo_dir / transformer_path
        if not local_transformer.exists():
            need_transformer_download = True
        required_paths["transformer"] = local_transformer

        # Optional precision config
        if required_files.get("precision_config_path"):
            precision_path = required_files["precision_config_path"]
            local_precision = local_repo_dir / precision_path
            if not local_precision.exists():
                need_precision_download = True
            required_paths["precision_config"] = local_precision

        # Optional prequant (Klein 9B doesn't have these, but keep for future models)
        if required_files.get("prequant_weights_path"):
            prequant_path = required_files["prequant_weights_path"]
            local_prequant = local_repo_dir / prequant_path
            required_paths["mod_weights"] = local_prequant
            if not local_prequant.exists():
                # Will be included in the download
                pass

        # ---------- Download if needed ----------
        revision = "master" if source_spec.type == "modelscope" else "main"

        # 1. Download base model (entire subdir recursively)
        if need_base_download:
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.1,
                message=f"Downloading base model ({base_path.rstrip('/')})...",
            ))
            base_subdir = base_path.rstrip("/")
            if base_subdir in ("", "."):
                allow_patterns = [
                    "model_index.json", "config.json", "*.json",
                    "text_encoder/**", "text_encoder*/**",
                    "vae/**", "scheduler/**", "tokenizer/**", "processor/**",
                ]
            else:
                allow_patterns = [f"{base_subdir}/**"]
            source.download_snapshot(
                repo_id=repo_id,
                local_dir=local_repo_dir,
                allow_patterns=allow_patterns,
                revision=revision,
            )
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.5,
                message="Base model downloaded ✓",
            ))

        # 2. Download transformer file (single large file)
        if need_transformer_download:
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.5,
                message=f"Downloading transformer ({transformer_path})...",
            ))
            source.download_file(
                repo_id=repo_id,
                file_path=transformer_path,
                local_dir=local_repo_dir,
                revision=revision,
            )
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.9,
                message="Transformer downloaded ✓",
            ))

        # 3. Download precision config (small JSON file)
        if need_precision_download:
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.95,
                message="Downloading precision config...",
            ))
            source.download_file(
                repo_id=repo_id,
                file_path=required_files["precision_config_path"],
                local_dir=local_repo_dir,
                revision=revision,
            )

        return required_paths

    def _get_token(self, source_type: str) -> Optional[str]:
        """Get auth token for a source type."""
        settings = get_settings()
        if source_type == "huggingface":
            return settings.huggingface_token
        elif source_type == "modelscope":
            return settings.modelscope_token
        return None

    # ---------- Generate ----------
    def generate(
        self,
        request: GenerateRequest,
        gpu_info: GPUInfo,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> GenerateResult:
        """Run a single text-to-image generation job.

        Args:
            request: GenerateRequest with prompt + model + params
            gpu_info: Detected GPU info
            progress_callback: Optional callback for progress events

        Returns:
            GenerateResult with image path + metadata

        Raises:
            RuntimeError: If engine not installed, GPU incompatible, or inference fails
        """
        self._cancel_flag.clear()

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        # Load model from registry
        registry = get_registry()
        model = registry.get(request.model_id)

        # Dispatch: stable-diffusion.cpp engine
        if getattr(model, "engine", "quantfunc") == "sd_cpp":
            from .sd_cpp import SDCppWorker, is_sd_cpp_installed
            if not is_sd_cpp_installed():
                raise RuntimeError(
                    "Engine stable-diffusion.cpp (SD.CPP) is not installed. "
                    "Please go to Settings → Engine to download it."
                )
            sd_worker = SDCppWorker()
            self._sd_worker = sd_worker
            try:
                return sd_worker.generate(request, model, gpu_info, progress_callback=emit)
            finally:
                self._sd_worker = None

        # Dispatch: SDNQ (SD.Next Quantization) engine
        if getattr(model, "engine", "quantfunc") == "sdnq":
            from .sdnq_engine import SDNQWorker, is_sdnq_installed
            if not is_sdnq_installed():
                raise RuntimeError(
                    "SDNQ runtime environment is not installed. "
                    "Please go to Settings → Engine to install/verify SDNQ."
                )
            sdnq_worker = SDNQWorker()
            self._sdnq_worker = sdnq_worker
            try:
                return sdnq_worker.generate(request, model, gpu_info, progress_callback=emit)
            finally:
                self._sdnq_worker = None

        # Dispatch: Nunchaku (vitoom) / OrbitQuant engines
        if getattr(model, "engine", "quantfunc") in ("nunchaku", "orbitquant"):
            from .extras import ExtrasWorker
            extras_worker = ExtrasWorker()
            self._extras_worker = extras_worker
            try:
                return extras_worker.generate(request, model, gpu_info, progress_callback=emit)
            finally:
                self._extras_worker = None

        # Determine engine version:
        engine_ver = getattr(model, "engine_version", None)
        if not engine_ver:
            # Auto-route: models with FP4 precision (Flux.2 Klein and Z-Image on 50x-above tier)
            # were quantized with FP4 text encoder and precision maps which engine v0.0.17 removed.
            # They run with native NVFP4 support in v0.0.12.
            if model.family in ("klein", "z-image") and gpu_info.tier == "50x-above":
                engine_ver = "0.0.12"
            else:
                engine_ver = "0.0.17"

        # ----- Phase 1: Pre-flight checks -----
        emit(ProgressEvent(status=JobStatus.LOADING, message=f"Checking engine (v{engine_ver})..."))
        self._check_engine(version=engine_ver)

        emit(ProgressEvent(status=JobStatus.LOADING, message="Checking CUDA libraries..."))
        self._check_cuda_libs()

        emit(ProgressEvent(status=JobStatus.LOADING, message=f"Checking GPU compatibility for {model.name}..."))
        self._check_gpu(gpu_info, model)

        # ----- Phase 1.5: Auth pre-flight (key-protected models) -----
        if getattr(model, "key_protected", False) and not has_api_key_available():
            raise RuntimeError(
                f"{model.name} is key-protected, but no API key is available. "
                + _KEY_PROTECTED_HELP
            )

        # ----- Phase 2: Resolve required files (lazy download) -----
        emit(ProgressEvent(status=JobStatus.LOADING, message="Resolving required files..."))
        required_files = model.resolve_required_files(
            gpu_sm=gpu_info.sm,
            backend=request.backend,
            steps=request.steps,
            use_prequant=request.use_prequant,
            variant_id=request.variant_id,
        )

        emit(ProgressEvent(status=JobStatus.DOWNLOADING, message="Checking cache for missing files..."))
        local_paths = self._resolve_local_paths(model, required_files, gpu_info, progress_callback=emit, request=request)

        # ----- Phase 3: Validate inputs -----
        emit(ProgressEvent(status=JobStatus.INITIALIZING, message="Validating inputs..."))
        prompt = validate_prompt(request.prompt)
        if not prompt:
            raise ValueError("Prompt is empty after validation")

        seed = request.seed if request.seed is not None else int(time.time()) & 0x7FFFFFFF
        seed = validate_int(seed, "seed", 0, 2**31 - 1)

        width, height = validate_resolution(request.width, request.height)

        # Output path
        settings = get_settings()

        # ----- Phase 3.5: Mask (inpaint) preparation -----
        # The mask is decoded here so that both the engine argument and the
        # final compositing use exactly the same coverage values.
        mask_ctx = None
        raw_mask_source = request.mask_data_url or (
            str(request.mask_path) if request.mask_path else None
        )
        if raw_mask_source:
            mask_ctx = self._prepare_mask(request, settings, raw_mask_source)

        if request.output_filename:
            from .security import validate_filename
            validate_filename(request.output_filename)
            output_path = settings.output_dir / request.output_filename
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.output_dir / f"qf_{timestamp}_{seed}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate that model_dir and transformer are within cache (path traversal protection)
        local_paths["model_dir"] = validate_path(str(local_paths["model_dir"]), base_dir=settings.models_dir, must_exist=True)
        local_paths["transformer"] = validate_path(str(local_paths["transformer"]), base_dir=settings.models_dir, must_exist=True)
        if local_paths.get("tokenizer"):
            local_paths["tokenizer"] = validate_path(str(local_paths["tokenizer"]), base_dir=settings.models_dir, must_exist=True)
        if local_paths.get("vae"):
            if getattr(request, "custom_vae_path", None):
                vae_p = Path(local_paths["vae"]).resolve()
                if not vae_p.is_file():
                    raise ValueError(f"Custom VAE file not found: {vae_p}")
                local_paths["vae"] = vae_p
            else:
                local_paths["vae"] = validate_path(str(local_paths["vae"]), base_dir=settings.models_dir, must_exist=True)

        # ----- Phase 4: Build subprocess args -----
        cli_path = engine_cli_path(version=engine_ver)
        engine_dir = cli_path.parent
        vram_budget = self._compute_vram_budget(gpu_info, model)

        args = [
            str(cli_path),
            "--model-dir", str(local_paths["model_dir"]),
            "--transformer", str(local_paths["transformer"]),
            "--model-backend", request.backend,
            "--prompt", prompt,
            "--output", str(output_path),
            "--steps", str(request.steps),
            "--seed", str(seed),
            "--width", str(width),
            "--height", str(height),
        ]

        # Guidance scale / CFG & True CFG
        cfg = getattr(request, "cfg", None)
        if cfg is None:
            cfg = model.get_default_cfg()
        if cfg is not None and cfg > 0.0:
            args.extend(["--guidance-scale", str(cfg)])
            has_neg = bool(getattr(request, "negative_prompt", None) and request.negative_prompt.strip())
            if model.family == "qwen" or model.id.startswith("qwen") or has_neg or request.init_image_path:
                args.extend(["--true-cfg-scale", str(cfg)])

        # Negative prompt
        if getattr(request, "negative_prompt", None) and request.negative_prompt.strip():
            from .security import validate_negative_prompt
            neg_prompt = validate_negative_prompt(request.negative_prompt)
            if neg_prompt:
                args.extend(["--negative-prompt", neg_prompt])

        # Sampler & Noise / Eta
        # NOTE: Binary disassembly of quantfunc.exe (v0.0.12 & v0.0.17) confirmed that all
        # DiT pipelines (Flux2Klein, QwenImage, QwenImageEdit, ZImage) pack latents into a
        # 3D sequence tensor shape=[B, seq_len, C] (e.g. [1, 4096, 128]) before calling
        # sampler->denoise(). Inside Sampler.cpp, stochastic/ancestral samplers (euler_a,
        # dpm++2m_sde, ddim) hardcode 4D indexing `x.shape[3]` when eta > 0, which throws
        # `TensorShape::operator[]: index 3 out of range for size 3 shape=[1, 4096, 128]`.
        # Deterministic samplers (euler, heun, dpm++2m, and unipc in v0.0.17) operate
        # elementwise on 3D packed latents without indexing shape[3].
        noise_val = getattr(request, "noise", 0.0) or 0.0
        req_sampler = (getattr(request, "sampler", None) or "euler").lower().strip()
        qf_sampler_map = {
            "euler": "euler",
            "euler_a": "euler",
            "lcm": "euler",
            "ddim": "euler",
            "heun": "heun",
            "dpm++2m": "dpm++2m",
            "dpm++2m_sde": "dpm++2m",
            "dpm2_a": "dpm++2m",
            "unipc": "dpm++2m" if engine_ver == "0.0.12" else "unipc",
        }
        sampler_val = qf_sampler_map.get(req_sampler, "euler")
        if sampler_val != req_sampler or noise_val > 0.0:
            logger.info(
                "[worker] quantfunc.exe uses 3D packed latents [B, L, C]; "
                "mapped sampler '%s' (noise=%.2f) -> '%s' (eta=0)",
                req_sampler,
                noise_val,
                sampler_val,
            )
        args.extend(["--sampler", str(sampler_val)])

        if settings.auto_optimize:
            args.append("--auto-optimize")

        if getattr(settings, "rotation_block_size", 0) and int(settings.rotation_block_size) > 0:
            args.extend(["--rotation-block-size", str(int(settings.rotation_block_size))])

        if vram_budget > 0:
            args.extend(["--vram-budget", str(vram_budget)])

        if local_paths.get("vae"):
            args.extend(["--vae", str(local_paths["vae"])])

        if local_paths.get("mod_weights"):
            args.extend(["--mod-weights", str(local_paths["mod_weights"])])

        if local_paths.get("precision_config"):
            args.extend(["--precision-config", str(local_paths["precision_config"])])

        if local_paths.get("text_encoder"):
            args.extend(["--text-encoder", str(local_paths["text_encoder"])])

        if local_paths.get("text_precision"):
            args.extend(["--text-precision", str(local_paths["text_precision"])])

        if local_paths.get("tokenizer"):
            args.extend(["--tokenizer", str(local_paths["tokenizer"])])

        if request.init_image_path:
            resolved_init = validate_path(str(request.init_image_path), base_dir=settings.output_dir, must_exist=True)
            from .security import validate_float
            strength = validate_float(request.init_image_strength, "init_image_strength", min_val=0.01, max_val=1.0)
            args.extend([
                "--ref-image", str(resolved_init),
                "--edit-strength", str(strength),
            ])

        # ----- Mask (inpaint) engine arguments -----
        if mask_ctx is not None:
            from .masking import engine_mask_arg
            from .security import validate_float as _validate_float

            mask_arg = engine_mask_arg("quantfunc")
            if not mask_arg:
                raise RuntimeError(
                    "The selected engine does not support inpaint masks. "
                    "Choose a mask-capable model or generate without a mask."
                )
            mask_strength = _validate_float(request.mask_strength, "mask_strength", min_val=0.0, max_val=1.0)
            mask_grow = max(0, min(64, int(request.mask_grow or 0)))
            mask_blur = _validate_float(request.mask_blur, "mask_blur", min_val=0.0, max_val=64.0)
            args.extend([mask_arg, str(mask_ctx["file"])])
            args.extend(["--mask-strength", str(mask_strength)])
            args.extend(["--mask-grow", str(mask_grow)])
            if mask_blur > 0:
                args.extend(["--mask-blur", str(mask_blur)])
            print(
                "[worker] Inpaint mask: %s (%sx%s), strength=%.2f grow=%d blur=%.1f",
                mask_ctx["file"].name, mask_ctx["source_size"][0], mask_ctx["source_size"][1],
                mask_strength, mask_grow, mask_blur,
            )

        # QuantFunc API key — user key from Settings takes precedence.
        # Without it, key-protected models fail at engine checkObfuscation.
        # If not set here, the engine falls back to bin/<os>/config.json itself.
        if settings.quantfunc_api_key and settings.quantfunc_api_key.strip():
            args.extend(["--apikey", settings.quantfunc_api_key.strip()])

        # LoRA stacking
        for lora_path, lora_strength in zip(request.lora_paths, request.lora_strengths):
            validate_path(str(lora_path), base_dir=settings.models_dir, must_exist=True)
            args.extend(["--lora", f"{lora_path}:{lora_strength}"])

        # Final safety check
        args = safe_subprocess_args(args)

        # ----- Phase 5: Spawn subprocess -----
        emit(ProgressEvent(
            status=JobStatus.GENERATING,
            progress=0.0,
            message=f"Starting inference (seed={seed})",
        ))

        start_time = time.time()

        try:
            # Build environment for subprocess: take current env, prepend CUDA paths
            # so the engine binary can find cublas64_13.dll, cudnn64_9.dll, etc.
            # This is CRITICAL on Windows — the engine binary's DLL loader looks up
            # DLLs by walking PATH in order. Without explicit PATH, CUDA Toolkit
            # install location is invisible to the subprocess.
            import os as _os
            from .engine import get_cuda_search_paths

            env = _os.environ.copy()
            cuda_paths = get_cuda_search_paths()
            existing_path = env.get("PATH", "")
            # Prepend engine dir and CUDA dirs so they take precedence
            new_path_parts = [str(engine_dir), str(settings.engine_dir)] + [str(p) for p in cuda_paths if p.exists()]
            if existing_path:
                new_path_parts.append(existing_path)
            env["PATH"] = _os.pathsep.join(new_path_parts)
            if settings.huggingface_token:
                env["HF_TOKEN"] = settings.huggingface_token
                env["HUGGING_FACE_HUB_TOKEN"] = settings.huggingface_token

            print(f"[worker] Launching QuantFunc engine (v{engine_ver}): width={width}, height={height}, steps={request.steps}, backend={request.backend}")
            print(f"[worker] CLI args: {' '.join(args)}")
            print(f"[worker] subprocess PATH (first 3): {new_path_parts[:3]}")
            print(f"[worker] subprocess cwd: {engine_dir}")

            self._current_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line-buffered
                encoding="utf-8",
                errors="replace",
                cwd=str(engine_dir),  # Engine needs its dir as CWD
                env=env,  # Inject CUDA paths
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to start engine CLI at {cli_path}: {e}")

        # ----- Phase 6: Read output line by line -----
        full_output_lines = []  # Collect all lines for error reporting
        engine_reported_complete = False
        try:
            for line in self._current_process.stdout:
                if self._cancel_flag.is_set():
                    self._current_process.terminate()
                    emit(ProgressEvent(status=JobStatus.CANCELLED, message="Cancelled by user"))
                    raise RuntimeError("Generation cancelled by user")

                full_output_lines.append(line.rstrip())

                # Detect key-protection failure early and explain the fix
                if _KEY_PROTECTED_RE.search(line):
                    self._current_process.terminate()
                    raise RuntimeError(
                        f"Engine error (key-protected model): {line.strip()}\n"
                        + _KEY_PROTECTED_HELP
                    )

                event = parse_engine_output(line)
                if event:
                    emit(event)
                    if event.status == JobStatus.ERROR:
                        raise RuntimeError(f"Engine error: {event.error}")
                    if event.status == JobStatus.COMPLETE and event.image_path:
                        engine_reported_complete = True
                        output_path = Path(event.image_path)

        finally:
            if self._current_process:
                # Wait for process to finish, then check exit code
                self._current_process.wait(timeout=30)
                return_code = self._current_process.returncode
                self._current_process = None
            else:
                return_code = -1  # Unknown

        # ----- Phase 6.5: Check process exit code -----
        # If the engine crashed, return_code will be non-zero and no output file will exist.
        if return_code != 0 and not engine_reported_complete:
            tail = "\n".join(full_output_lines[-20:]) if full_output_lines else "(no output)"
            explicit_errors = [
                l.strip() for l in full_output_lines
                if l.strip().startswith("Error:") or "[error]" in l.lower()
            ]
            if explicit_errors:
                primary = explicit_errors[-1]
                raise RuntimeError(
                    f"Engine error: {primary}\n"
                    f"--- Engine output (last 20 lines) ---\n{tail}"
                )
            raise RuntimeError(
                f"Engine process exited with code {return_code} (crashed). "
                f"This usually means a DLL is missing (e.g., cublas64_13.dll, cudnn64_9.dll). "
                f"Go to Settings → Download CUDA deps to install CUDA runtime libraries.\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        # ----- Phase 7: Verify output -----
        if not output_path.exists():
            tail = "\n".join(full_output_lines[-20:]) if full_output_lines else "(no output)"
            raise RuntimeError(
                f"Engine reported completion but output file not found: {output_path}\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        elapsed = time.time() - start_time

        emit(ProgressEvent(
            status=JobStatus.SAVING,
            message=f"Saving to {output_path.name}...",
        ))

        # Copy to output dir if engine wrote elsewhere
        final_path = output_path
        if output_path.parent != settings.output_dir:
            from .cache import CacheManager
            cm = CacheManager()
            final_path = cm.save_output(output_path)

        # ----- Phase 7.5: Mask compositing (guaranteed preservation) -----
        # result = generated * mask + original * (1 - mask)
        # The composite is produced at the ORIGINAL resolution, so every pixel
        # outside the mask stays bit-identical to the source image. The raw
        # engine output is kept next to it for the Inpaint "Result" layer.
        raw_fragment_rel = None
        if mask_ctx is not None:
            from PIL import Image
            from .masking import composite_with_mask

            try:
                with Image.open(final_path) as gen_file:
                    generated_rgb = gen_file.convert("RGB")
                source_image = mask_ctx["source_image"]
                composite = composite_with_mask(
                    generated_rgb, source_image, mask_ctx["array"]
                )
                composite.save(final_path, format="PNG")

                frag_dir = settings.output_dir / "masks"
                frag_dir.mkdir(parents=True, exist_ok=True)
                frag_path = frag_dir / f"{final_path.stem}_raw.png"
                generated_rgb.resize(source_image.size, Image.LANCZOS).save(frag_path, format="PNG")
                raw_fragment_rel = f"masks/{frag_path.name}"
                print(f"[worker] Mask composite applied -> {final_path.name} (fragment: {raw_fragment_rel})")
            except Exception as comp_err:
                print(f"[worker] Mask compositing failed: {comp_err}")
                raise RuntimeError(f"Mask compositing failed: {comp_err}") from comp_err

        # Write JSON sidecar with generation metadata
        json_sidecar = final_path.with_suffix(".json")
        try:
            meta = {
                "filename": final_path.name,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "model_id": request.model_id,
                "variant_id": getattr(request, "variant_id", None),
                "seed": seed,
                "width": width,
                "height": height,
                "steps": request.steps,
                "cfg": cfg,
                "noise": noise_val,
                "sampler": sampler_val,
                "vae": getattr(request, "vae", None),
                "backend": request.backend,
                "owner_id": getattr(request, "owner_id", "host"),
                "owner_name": getattr(request, "owner_name", "Хост"),
                "shared": getattr(request, "shared", True),
                "generation_time_sec": round(elapsed, 2),
                "timestamp": int(time.time()),
            }
            if mask_ctx is not None:
                meta.update({
                    "mask_applied": True,
                    "mask_strength": float(request.mask_strength),
                    "mask_grow": int(request.mask_grow or 0),
                    "mask_blur": float(request.mask_blur or 0.0),
                    "mask_source": str(mask_ctx["source_path"].name),
                    "mask_raw_fragment": raw_fragment_rel,
                    "composite": "generated*mask + original*(1-mask)",
                })
            else:
                meta["mask_applied"] = False
            json_sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as meta_err:
            print(f"[worker] Failed to write sidecar metadata {json_sidecar}: {meta_err}")

        emit(ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=str(final_path),
            raw_image_path=raw_fragment_rel,
            message=f"Done in {elapsed:.1f}s",
        ))

        return GenerateResult(
            image_path=final_path,
            seed_used=seed,
            generation_time_sec=elapsed,
            model_id=request.model_id,
            backend=request.backend,
            steps=request.steps,
            prompt=prompt,
        )

    def cancel(self) -> None:
        """Cancel the current generation job."""
        self._cancel_flag.set()
        if hasattr(self, "_extras_worker") and self._extras_worker:
            try:
                self._extras_worker.cancel()
            except Exception:
                pass
        if hasattr(self, "_sd_worker") and self._sd_worker:
            try:
                self._sd_worker.cancel()
            except Exception:
                pass
        if hasattr(self, "_sdnq_worker") and self._sdnq_worker:
            try:
                self._sdnq_worker.cancel()
            except Exception:
                pass
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    # ---------- Mask (inpaint) ----------
    def _prepare_mask(self, request: GenerateRequest, settings, raw_source: str) -> dict:
        """Decode the incoming mask and persist it as a PNG for the engine.

        The decoded coverage array lives in SOURCE IMAGE coordinates, so it is
        used both for the engine argument and for the final compositing.
        """
        from PIL import Image
        from .masking import (
            MaskError,
            decode_mask_to_float,
            mask_file_name,
            save_mask_file,
        )

        if not request.init_image_path:
            raise ValueError(
                "Mask editing requires a source image (init_image_path is missing)"
            )

        source_path = validate_path(
            str(request.init_image_path),
            base_dir=settings.output_dir,
            must_exist=True,
        )
        with Image.open(source_path) as src_file:
            source_image = src_file.copy()

        if request.mask_path:
            mask_path = validate_path(
                str(request.mask_path),
                base_dir=settings.output_dir,
                must_exist=True,
            )
            raw_source = str(mask_path)
        source_size = source_image.size

        try:
            mask_array = decode_mask_to_float(
                raw_source,
                target_size=source_size,
                invert_mask=bool(request.mask_invert),
            )
        except MaskError as exc:
            raise ValueError(str(exc)) from exc

        if mask_array is None:
            raise ValueError("Mask payload is empty")

        mask_file = save_mask_file(
            mask_array,
            settings.output_dir / "masks",
            mask_file_name("inpaint"),
        )
        return {
            "array": mask_array,
            "file": mask_file,
            "source_path": source_path,
            "source_size": source_size,
            "source_image": source_image,
        }

    def _compute_vram_budget(self, gpu_info: GPUInfo, model: ModelSpec) -> int:
        """Compute --vram-budget value in MB based on settings + GPU."""
        settings = get_settings()
        mode = settings.vram_mode

        if mode == VRAMMode.AUTO or gpu_info.vram_gb == 0:
            return 0  # No budget — let engine decide
        elif mode == VRAMMode.CONSERVATIVE:
            return int(gpu_info.vram_gb * 0.5 * 1024)
        elif mode == VRAMMode.BALANCED:
            return int(gpu_info.vram_gb * 0.75 * 1024)
        elif mode == VRAMMode.MAXIMUM:
            return int(gpu_info.vram_gb * 1024)
        return 0


# ---------- Singleton worker ----------
_worker_instance: Optional[Worker] = None


def get_worker() -> Worker:
    """Get the global Worker singleton."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = Worker()
    return _worker_instance
