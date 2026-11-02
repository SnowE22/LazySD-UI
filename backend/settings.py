"""
Settings store for QuantFunc WebUI.

Persists user configuration to settings.json in the project root.
All paths are configurable — engine binary, model cache, outputs.

Single-user, global settings (one machine = one user).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------- Platform detection ----------
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform.startswith("darwin")

PLATFORM_NAME = "windows" if IS_WINDOWS else "linux" if IS_LINUX else "macos"


# ---------- Default paths ----------
def _default_engine_dir() -> Path:
    """Engine binary lives in ./bin/<platform>/ — same convention as plugin."""
    return Path(__file__).parent.parent / "bin" / PLATFORM_NAME


def _default_models_dir() -> Path:
    return Path(__file__).parent.parent / "models" / "cache"


def _default_output_dir() -> Path:
    return Path(__file__).parent.parent / "outputs"


def _default_cache_dir() -> Path:
    return Path(__file__).parent.parent / "cache"


# ---------- Settings model ----------
class VRAMMode(str):
    AUTO = "auto"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    MAXIMUM = "maximum"

    @classmethod
    def values(cls):
        return [cls.AUTO, cls.CONSERVATIVE, cls.BALANCED, cls.MAXIMUM]


class DownloadSourcePriority(str):
    AUTO = "auto"          # Try HF first, fall back to MS
    HF_FIRST = "hf_first"
    MS_FIRST = "ms_first"

    @classmethod
    def values(cls):
        return [cls.AUTO, cls.HF_FIRST, cls.MS_FIRST]


class Settings(BaseModel):
    """Persistent user settings — stored as settings.json."""

    # ---------- Paths (configurable) ----------
    models_dir: Path = Field(default_factory=_default_models_dir)
    output_dir: Path = Field(default_factory=_default_output_dir)
    cache_dir: Path = Field(default_factory=_default_cache_dir)
    engine_dir: Path = Field(default_factory=_default_engine_dir)

    # ---------- Engine version ----------
    engine_version: str = "0.0.17"
    engine_platform: str = PLATFORM_NAME  # windows / linux
    cuda_version: str = "13"              # "12" or "13" — affects which .dll/.so to use

    # ---------- UI ----------
    lang: str = "auto"                    # "ru" | "en" | "auto"
    theme: str = "dark"                   # Only "dark" supported in Phase 2
    hide_uninstalled_in_generator: bool = False
    monitor_connections: bool = True       # Count online accounts + show header indicator

    # ---------- GPU / VRAM ----------
    vram_mode: str = VRAMMode.AUTO
    auto_optimize: bool = True
    # Cached GPU info — refreshed on startup or when user clicks "Re-detect"
    gpu_name: Optional[str] = None
    gpu_sm: Optional[int] = None          # 89 for RTX 4090, 120 for RTX 5060 Ti / 5090
    gpu_vram_mb: Optional[int] = None
    gpu_driver_version: Optional[str] = None

    # ---------- Network ----------
    download_source_priority: str = DownloadSourcePriority.AUTO
    auto_update_engine: bool = False      # Offline-first — disabled by default
    huggingface_token: Optional[str] = None
    modelscope_token: Optional[str] = None
    # Fast HF downloads: hf_transfer (huggingface_hub 0.x) or Xet high-performance (1.x).
    # Off by default: Xet previously crashed with an allocator error on this setup.
    hf_transfer: bool = False
    # Optional HF mirror base URL (e.g. https://hf-mirror.com). Empty = official.
    # NOTE: the HF token is sent to this host too.
    hf_endpoint: Optional[str] = None
    model_hub_url: str = "https://raw.githubusercontent.com/nnmgrp/LazySD-Models/main"

    # ---------- QuantFunc auth ----------
    # Optional user API key (format "qf_..."). Passed to the engine as --apikey.
    # When empty/None the engine falls back to bin/<os>/config.json which ships
    # with the plugin and contains the shared free-trial test key.
    # Get your own key at https://www.quantfunc.com (free during public beta).
    quantfunc_api_key: Optional[str] = None

    # ---------- Worker ----------
    worker_restart_on_crash: bool = True
    worker_max_restarts: int = 3
    worker_log_level: str = "INFO"        # DEBUG / INFO / WARNING / ERROR

    # ---------- Advanced ----------
    fbcache_cond_threshold: Optional[float] = None
    fbcache_uncond_threshold: Optional[float] = None
    rotation_block_size: int = 256        # For SVDQ rotation quantization
    log_level: str = "INFO"

    # ---------- CUDA library paths ----------
    # Optional: explicit path to CUDA Toolkit bin (where cublas64_13.dll lives).
    # If None, we auto-detect by checking standard install locations on Windows.
    # Examples:
    #   "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin"
    #   "G:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64"
    cuda_bin_path: Optional[str] = None

    # ---------- VAE Selection ----------
    qwen_vae_type: str = "texture_fix"    # "texture_fix" | "official"

    # ---------- QuantFunc experimental toggle ----------
    # Disabled by default due to corrupted/striped upstream weights.
    enable_quantfunc_models: bool = False

    # ---------- Extras runtime (OrbitQuant / Nunchaku) ----------
    # Optional: python.exe of an existing environment that already has torch +
    # diffusers (+ nunchaku / orbitquant). Empty = use the managed venv in envs/extras.
    extras_python: Optional[str] = None

    # ---------- Sharing (Gradio-style `share=True`, but without Gradio) ----------
    # Remote visitors (anything arriving via LAN/tunnel, i.e. not plain loopback)
    # must present `share_token`. Remote sessions are "guests": they can generate
    # and browse, but can't touch settings/engines/models unless share_allow_admin.
    share_token: Optional[str] = None
    share_tunnel_provider: str = "cloudflared"   # see backend/share.py PROVIDERS
    share_auto_start: bool = False               # start the tunnel with the server
    share_allow_admin: bool = False              # let remote visitors use admin endpoints

    # ---------- Vision / Inpaint Prompt Assistant ----------
    vision_enabled: bool = True
    vision_provider: str = "ollama"
    vision_model: str = "qwen2.5vl:3b"
    vision_ollama_url: str = "http://127.0.0.1:11434"
    vision_keep_alive: str = "5m"
    vision_timeout_sec: int = 120
    vision_max_image_dim: int = 640
    vision_fallback_enabled: bool = True

    # ---------- State ----------
    first_run_completed: bool = False

    # ---------- Validators ----------
    @field_validator("vram_mode")
    @classmethod
    def _validate_vram_mode(cls, v: str) -> str:
        if v not in VRAMMode.values():
            raise ValueError(f"vram_mode must be one of {VRAMMode.values()}, got {v!r}")
        return v

    @field_validator("download_source_priority")
    @classmethod
    def _validate_source_priority(cls, v: str) -> str:
        if v not in DownloadSourcePriority.values():
            raise ValueError(f"download_source_priority must be one of {DownloadSourcePriority.values()}, got {v!r}")
        return v

    @field_validator("lang")
    @classmethod
    def _validate_lang(cls, v: str) -> str:
        if v not in ("ru", "en", "auto"):
            raise ValueError(f"lang must be 'ru', 'en', or 'auto', got {v!r}")
        return v

    @field_validator("qwen_vae_type")
    @classmethod
    def _validate_qwen_vae_type(cls, v: str) -> str:
        if v not in ("texture_fix", "official"):
            raise ValueError(f"qwen_vae_type must be 'texture_fix' or 'official', got {v!r}")
        return v

    @field_validator("vision_provider")
    @classmethod
    def _validate_vision_provider(cls, v: str) -> str:
        allowed = ("auto", "ollama", "gguf", "llamacpp", "nunchaku", "heuristic")
        if v not in allowed:
            raise ValueError(f"vision_provider must be one of {allowed}, got {v!r}")
        return v

    @field_validator("vision_timeout_sec")
    @classmethod
    def _validate_vision_timeout_sec(cls, v: int) -> int:
        val = int(v)
        if val < 5 or val > 600:
            raise ValueError("vision_timeout_sec must be between 5 and 600")
        return val

    @field_validator("vision_max_image_dim")
    @classmethod
    def _validate_vision_max_image_dim(cls, v: int) -> int:
        val = int(v)
        if val < 256 or val > 2048:
            raise ValueError("vision_max_image_dim must be between 256 and 2048")
        return val

    # ---------- Path helpers ----------
    @property
    def engine_binary_path(self) -> Path:
        """Path to libquantfunc.so / quantfunc.dll for current platform + CUDA version."""
        if IS_WINDOWS:
            # quantfunc.dll (CUDA 13) or quantfunc-12.dll (CUDA 12)
            if self.cuda_version == "12":
                return self.engine_dir / "quantfunc-12.dll"
            return self.engine_dir / "quantfunc.dll"
        else:
            # libquantfunc.so (CUDA 13) or libquantfunc-12.so (CUDA 12)
            if self.cuda_version == "12":
                return self.engine_dir / "libquantfunc-12.so"
            return self.engine_dir / "libquantfunc.so"

    @property
    def engine_cli_path(self) -> Path:
        """Path to standalone CLI quantfunc / quantfunc.exe."""
        if IS_WINDOWS:
            exe_name = "quantfunc-12.exe" if self.cuda_version == "12" else "quantfunc.exe"
        else:
            exe_name = "quantfunc-12" if self.cuda_version == "12" else "quantfunc"
        return self.engine_dir / exe_name

    @property
    def settings_file_path(self) -> Path:
        return Path(__file__).parent.parent / "settings.json"

    # ---------- Persistence ----------
    def save(self) -> None:
        """Save settings to settings.json."""
        data = self.model_dump(mode="json")
        # Convert Path objects to strings for JSON serialization
        for key in ("models_dir", "output_dir", "cache_dir", "engine_dir"):
            if key in data:
                data[key] = str(Path(data[key]).resolve())
        # Don't persist tokens in plaintext if user has keyring (Phase 5)
        # For Phase 2 — store as-is (single-user machine)

        with open(self.settings_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from settings.json, or create defaults on first run."""
        path = cls().settings_file_path
        if not path.exists():
            # First run — create with defaults
            settings = cls()
            settings.save()
            return settings

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Convert string paths back to Path
            for key in ("models_dir", "output_dir", "cache_dir", "engine_dir"):
                if key in data and data[key]:
                    data[key] = Path(data[key])
            return cls(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Corrupt settings — fall back to defaults
            print(f"[settings] WARNING: settings.json corrupt ({e}), using defaults")
            settings = cls()
            settings.save()
            return settings

    def update(self, **kwargs) -> "Settings":
        """Update fields and persist. Returns self for chaining.

        Secret/token fields accept empty strings to allow explicit clearing
        ("" means "remove the stored value"). Other fields skip None values
        to avoid accidental resets when a caller omits a key.
        """
        clearable = {"quantfunc_api_key", "huggingface_token", "modelscope_token",
                     "share_token", "extras_python", "hf_endpoint"}
        for key, value in kwargs.items():
            if key not in ("models_dir", "output_dir", "cache_dir", "engine_dir"):
                if not hasattr(self, key):
                    continue
                if value is None and key not in clearable:
                    continue
                setattr(self, key, value)
            else:
                if value is None:
                    continue
                setattr(self, key, Path(value).expanduser().resolve())
        self.save()
        if "huggingface_token" in kwargs:
            sync_hf_token_to_system(self.huggingface_token)
        return self

    def ensure_directories_exist(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in (self.models_dir, self.output_dir, self.cache_dir, self.engine_dir):
            path.mkdir(parents=True, exist_ok=True)


def sync_hf_token_to_system(token: Optional[str] = None) -> None:
    """Sync HuggingFace token to OS environment variables and huggingface_hub login."""
    tok = (token or "").strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False, skip_if_logged_in=False)
        except Exception:
            pass
    else:
        if token == "":
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            try:
                from huggingface_hub import logout
                logout()
            except Exception:
                pass


# ---------- Singleton accessor ----------
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global Settings singleton. Loads from disk on first access."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.load()
        _settings_instance.ensure_directories_exist()
        sync_hf_token_to_system(_settings_instance.huggingface_token)
    return _settings_instance


def reload_settings() -> Settings:
    """Force reload from disk. Use after external modifications."""
    global _settings_instance
    _settings_instance = Settings.load()
    _settings_instance.ensure_directories_exist()
    sync_hf_token_to_system(_settings_instance.huggingface_token)
    return _settings_instance
