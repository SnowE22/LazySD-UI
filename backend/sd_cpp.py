"""
Integration for stable-diffusion.cpp (SD.CPP / sd-cli.exe).

Provides lightweight, standalone C++/CUDA inference for GGUF and safetensors
diffusion models (SDXL, SD 1.5, SD 3.5, Flux.1 GGUF) with quantized text encoders
(T5-XXL Q4, CLIP-L) without PyTorch dependencies.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .gpu import GPUInfo
from .registry.loader import ModelSpec

logger = logging.getLogger("sd_cpp")
from .security import (
    validate_filename,
    validate_int,
    validate_prompt,
    validate_resolution,
)
from .settings import get_settings


# Default SD.CPP binary directory
def get_sd_cpp_dir() -> Path:
    settings = get_settings()
    return settings.engine_dir / "sd_cpp"


def get_sd_cpp_cli_path() -> Path:
    return get_sd_cpp_dir() / "sd-cli.exe"


def is_sd_cpp_installed() -> bool:
    return get_sd_cpp_cli_path().exists()


def get_sd_cpp_version() -> str:
    """Return installed or known SD.CPP release version."""
    cli_path = get_sd_cpp_cli_path()
    if not cli_path.exists():
        return "not installed"
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        res = subprocess.run(
            [str(cli_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cli_path.parent),
            creationflags=creationflags,
        )
        out = (res.stdout or res.stderr or "").strip()
        m = re.search(r"commit\s+([0-9a-fA-F]+)", out)
        if m:
            return f"master-{m.group(1)} (cu12)"
    except Exception:
        pass
    return "master-f0aac52 (cu12)"


SD_CPP_WINDOWS_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-973-f0aac52/sd-master-f0aac52-bin-win-cuda12-x64.zip"
SD_CPP_CUDART_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-973-f0aac52/cudart-sd-bin-win-cu12-x64.zip"


def _resolve_latest_sd_cpp_url() -> tuple[str, str]:
    """Query GitHub API for the latest leejet/stable-diffusion.cpp CUDA 12 Windows release."""
    try:
        import requests
        r = requests.get(
            "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest",
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            win_url = None
            cudart_url = None
            for asset in data.get("assets", []):
                name = (asset.get("name") or "").lower()
                url = asset.get("browser_download_url")
                if not url:
                    continue
                if "bin-win-cuda12-x64.zip" in name:
                    win_url = url
                elif "cudart-sd-bin-win-cu12-x64.zip" in name:
                    cudart_url = url
            if win_url:
                return win_url, (cudart_url or SD_CPP_CUDART_URL)
    except Exception as e:
        logger.debug("Could not query latest SD.CPP release from GitHub API: %s", e)
    return SD_CPP_WINDOWS_URL, SD_CPP_CUDART_URL


def _sd_cpp_has_working_cuda(cli_path: Path) -> bool:
    """Verify whether sd-cli.exe can load ggml-cuda.dll and see a CUDA device."""
    if not cli_path.exists():
        return False
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        res = subprocess.run(
            [str(cli_path), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cli_path.parent),
            creationflags=creationflags,
        )
        return "CUDA" in (res.stdout or "")
    except Exception:
        return False


def download_sd_cpp(
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    force: bool = False,
) -> Path:
    """Download and extract stable-diffusion.cpp standalone CUDA 12 build.

    Also ensures CUDA 12 runtime DLLs (cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll)
    are present if the host system does not have CUDA Toolkit 12 in PATH (e.g. clean RTX 5070 Ti PC).
    """
    target_dir = get_sd_cpp_dir()
    cli_path = get_sd_cpp_cli_path()

    import zipfile
    from .engine import _download_file_with_progress

    target_dir.mkdir(parents=True, exist_ok=True)
    win_url, cudart_url = _resolve_latest_sd_cpp_url() if force else (SD_CPP_WINDOWS_URL, SD_CPP_CUDART_URL)

    if not cli_path.exists() or force:
        temp_zip = target_dir / "sd_master_cuda12.zip.tmp"
        try:
            logger.info("Downloading SD.CPP from %s", win_url)
            _download_file_with_progress(
                win_url,
                temp_zip,
                progress_callback=progress_callback,
            )

            logger.info("Extracting SD.CPP zip to %s", target_dir)
            with zipfile.ZipFile(temp_zip, "r") as z:
                z.extractall(target_dir)

            if not cli_path.exists():
                raise RuntimeError(f"sd-cli.exe not found after extracting {temp_zip}")
        finally:
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                except OSError:
                    pass

    # Check if CUDA backend loads properly; if not (missing cudart64_12.dll / cublas64_12.dll on fresh PC),
    # download the official cudart-sd-bin-win-cu12-x64.zip bundle into target_dir.
    if not _sd_cpp_has_working_cuda(cli_path) and not (target_dir / "cublas64_12.dll").exists():
        cudart_zip = target_dir / "cudart_cu12.zip.tmp"
        try:
            logger.info("CUDA 12 runtime DLLs not found in system PATH; downloading %s", SD_CPP_CUDART_URL)
            _download_file_with_progress(
                SD_CPP_CUDART_URL,
                cudart_zip,
                progress_callback=progress_callback,
            )
            with zipfile.ZipFile(cudart_zip, "r") as z:
                z.extractall(target_dir)
        except Exception as e:
            logger.warning("Could not download optional cudart bundle: %s", e)
        finally:
            if cudart_zip.exists():
                try:
                    cudart_zip.unlink()
                except OSError:
                    pass

    logger.info("SD.CPP ready at %s", cli_path)
    return cli_path


def uninstall_sd_cpp() -> dict:
    """Remove the managed SD.CPP directory (binary, CUDA DLLs, downloads)."""
    target_dir = get_sd_cpp_dir()
    removed = target_dir.exists()
    if removed:
        shutil.rmtree(target_dir)
        logger.info("Removed SD.CPP directory: %s", target_dir)
    return {
        "engine": "sd_cpp",
        "removed": removed,
        "path": str(target_dir),
        "message": "SD.CPP удалён" if removed else "SD.CPP уже не установлен",
    }


# Progress matching regex for stable-diffusion.cpp
# Example outputs:
#   "| 3/20 [00:02<00:11,  1.45s/it]"
#   "generating image: 5/20"
#   "sampling: 45%"
_STEP_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_PERCENT_PROGRESS_RE = re.compile(r"(\d+)\s*%")


def _ensure_moss_gguf_qkv_fused(gguf_path: Path, emit: Optional[Callable] = None) -> None:
    """Losslessly fuse split layers.{i}.proj.{q,k,v}.weight into layers.{i}.proj.qkv.weight
    for RebelAI MOSS-Image GGUF checkpoints so stable-diffusion.cpp can load them natively."""
    import struct

    try:
        with open(gguf_path, "rb") as f:
            head_probe = f.read(65536)
        if b"moss_image" not in head_probe or b"layers.0.proj.qkv.weight" in head_probe:
            return
        if b"layers.0.proj.q.weight" not in head_probe:
            return
    except Exception:
        return

    logger.info("[sd_cpp] Fusing split Q/K/V tensors in MOSS-Image GGUF: %s", gguf_path.name)
    if emit:
        from .worker import JobStatus, ProgressEvent
        emit(ProgressEvent(
            status=JobStatus.LOADING,
            progress=0.04,
            message=f"Converting {gguf_path.name} Q/K/V tensors for SD.CPP (one-time)...",
        ))

    with open(gguf_path, "rb") as f:
        magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            return
        kv_start = 24

        def r_str() -> str:
            (length,) = struct.unpack("<Q", f.read(8))
            return f.read(length).decode("utf-8")

        def skip_val(vtype: int) -> None:
            sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if vtype in sz:
                f.read(sz[vtype])
            elif vtype == 8:
                r_str()
            elif vtype == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                for _ in range(cnt):
                    skip_val(et)

        for _ in range(n_kv):
            _ = r_str()
            (vt,) = struct.unpack("<I", f.read(4))
            skip_val(vt)
        kv_end = f.tell()
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)

        tensors = []
        for _ in range(n_tensors):
            name = r_str()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, tt, off))
        header_end = f.tell()
        data_offset = (header_end + 31) // 32 * 32

    file_size = gguf_path.stat().st_size
    sorted_t = sorted(tensors, key=lambda x: x[3])
    sizes = {}
    for idx, (name, dims, tt, off) in enumerate(sorted_t):
        next_off = sorted_t[idx + 1][3] if idx + 1 < len(sorted_t) else (file_size - data_offset)
        sizes[name] = next_off - off

    t_map = {t[0]: t for t in tensors}
    new_entries = []
    consumed = set()

    for name, dims, tt, off in tensors:
        if name in consumed:
            continue
        m = re.match(r"^(layers\.\d+\.proj)\.(q|k|v)\.weight$", name)
        if m:
            prefix = m.group(1)
            q_name = f"{prefix}.q.weight"
            k_name = f"{prefix}.k.weight"
            v_name = f"{prefix}.v.weight"
            if q_name in t_map and k_name in t_map and v_name in t_map:
                consumed.update((q_name, k_name, v_name))
                q_t, k_t, v_t = t_map[q_name], t_map[k_name], t_map[v_name]
                fused_dims = [q_t[1][0], q_t[1][1] + k_t[1][1] + v_t[1][1]]
                new_entries.append((
                    f"{prefix}.qkv.weight",
                    fused_dims,
                    q_t[2],
                    [(q_t[3], sizes[q_name]), (k_t[3], sizes[k_name]), (v_t[3], sizes[v_name])],
                ))
                continue
        new_entries.append((name, dims, tt, [(off, sizes[name])]))

    # Compute new relative data offsets (32-byte aligned)
    cur_rel_off = 0
    header_tensor_parts = []
    entry_Plans = []
    for name, dims, tt, chunks in new_entries:
        aligned_off = (cur_rel_off + 31) // 32 * 32
        pad_before = aligned_off - cur_rel_off
        cur_rel_off = aligned_off
        total_len = sum(length for _, length in chunks)
        name_b = name.encode("utf-8")
        t_hdr = [struct.pack("<Q", len(name_b)), name_b, struct.pack("<I", len(dims))]
        for d in dims:
            t_hdr.append(struct.pack("<Q", d))
        t_hdr.append(struct.pack("<IQ", tt, cur_rel_off))
        header_tensor_parts.append(b"".join(t_hdr))
        entry_Plans.append((pad_before, chunks))
        cur_rel_off += total_len

    new_header = (
        struct.pack("<4sIQQ", b"GGUF", ver, len(new_entries), n_kv)
        + kv_bytes
        + b"".join(header_tensor_parts)
    )
    new_data_offset = (len(new_header) + 31) // 32 * 32
    header_pad = b"\x00" * (new_data_offset - len(new_header))

    tmp_path = gguf_path.with_suffix(".gguf.tmp")
    with open(gguf_path, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(new_header)
        dst.write(header_pad)
        for pad_before, chunks in entry_Plans:
            if pad_before > 0:
                dst.write(b"\x00" * pad_before)
            for src_rel_off, length in chunks:
                src.seek(data_offset + src_rel_off)
                remaining = length
                while remaining > 0:
                    buf = src.read(min(remaining, 8 * 1024 * 1024))
                    if not buf:
                        break
                    dst.write(buf)
                    remaining -= len(buf)

    tmp_path.replace(gguf_path)
    logger.info("[sd_cpp] Successfully fused Q/K/V tensors in %s (%d tensors)", gguf_path.name, len(new_entries))


class SDCppWorker:
    """Executes diffusion inference via sd-cli.exe."""

    def __init__(self):
        self._cancel_flag = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self):
        """Cancel current running process."""
        self._cancel_flag.set()
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def generate(
        self,
        request,
        model: ModelSpec,
        gpu_info: Optional[GPUInfo],
        progress_callback: Optional[Callable] = None,
    ):
        """Run generation via sd-cli.exe.

        Args:
            request: GenerateRequest instance
            model: ModelSpec for the selected model
            gpu_info: GPUInfo
            progress_callback: Callable taking ProgressEvent

        Returns:
            GenerateResult instance
        """
        from .worker import GenerateResult, JobStatus, ProgressEvent

        self._cancel_flag.clear()

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        cli_path = get_sd_cpp_cli_path()
        if not cli_path.exists():
            raise RuntimeError(
                f"stable-diffusion.cpp binary not found at {cli_path}. "
                "Ensure bin/windows/sd_cpp/sd-cli.exe is present."
            )

        # 1. Validate inputs
        emit(ProgressEvent(status=JobStatus.INITIALIZING, message="Validating inputs for SD.CPP..."))
        prompt = validate_prompt(request.prompt)
        if not prompt:
            raise ValueError("Prompt is empty after validation")

        seed = request.seed if request.seed is not None and request.seed >= 0 else int(time.time()) & 0x7FFFFFFF
        seed = validate_int(seed, "seed", 0, 2**31 - 1)

        width, height = validate_resolution(request.width, request.height)

        # 2. Setup output path
        settings = get_settings()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = request.output_filename or f"sd_{timestamp}_{seed}.png"
        validate_filename(out_filename)
        output_path = settings.output_dir / out_filename

        # 3. Resolve model paths
        from .cache import CacheManager
        cache = CacheManager()
        primary_repo = model.sources.primary.repo_id if model.sources else ""
        repo_dir = cache.repo_dir_for(primary_repo) if primary_repo else settings.models_dir

        cmd = [str(cli_path), "-M", "img_gen"]

        def resolve_model_file(rel_or_abs: str) -> Path:
            p = Path(rel_or_abs)
            if p.is_absolute() and p.exists():
                return p
            if (repo_dir / rel_or_abs).exists():
                return repo_dir / rel_or_abs
            try:
                from .custom_models import get_custom_models_dir
                custom_candidate = get_custom_models_dir() / rel_or_abs
                if custom_candidate.exists():
                    return custom_candidate
            except Exception:
                pass
            if (repo_dir / "text_encoders" / rel_or_abs).exists():
                return repo_dir / "text_encoders" / rel_or_abs
            if (repo_dir / "vae" / rel_or_abs).exists():
                return repo_dir / "vae" / rel_or_abs
            # Check secondary repos (e.g. text encoders from unsloth/Qwen3-8B-GGUF or variants from akhaliq/...)
            sec_repos = [
                t.repo_id for t in getattr(model, "transformers", []) if getattr(t, "repo_id", None)
            ] + [
                te.get("source", {}).get("repo_id")
                for te in getattr(model, "text_encoders", [])
                if isinstance(te.get("source"), dict) and te.get("source", {}).get("repo_id")
            ]
            for cand_repo in sec_repos:
                alt_dir = cache.repo_dir_for(cand_repo)
                if (alt_dir / rel_or_abs).exists():
                    return alt_dir / rel_or_abs
                if (alt_dir / "text_encoders" / rel_or_abs).exists():
                    return alt_dir / "text_encoders" / rel_or_abs
                if (alt_dir / "vae" / rel_or_abs).exists():
                    return alt_dir / "vae" / rel_or_abs
            return p if p.is_absolute() else repo_dir / rel_or_abs

        # Checkpoint vs standalone diffusion-model
        ckpt_rel = model.engine_args.get("checkpoint")
        if ckpt_rel:
            ckpt_path = resolve_model_file(ckpt_rel)
            cmd.extend(["-m", str(ckpt_path)])
        else:
            # Standalone diffusion model (e.g. DiT GGUF)
            diff_rel = model.engine_args.get("diffusion_model")
            if getattr(request, "variant_id", None):
                for t in getattr(model, "transformers", []):
                    if t.id == request.variant_id:
                        diff_rel = t.path
                        break
            if diff_rel:
                diff_path = resolve_model_file(diff_rel)
                if not diff_path.exists():
                    raise RuntimeError(
                        f"Diffusion model file not found: {diff_path.name}. "
                        "Please download it from the Models tab."
                    )
                if diff_path.suffix.lower() == ".gguf":
                    _ensure_moss_gguf_qkv_fused(diff_path, emit=emit)
                cmd.extend(["--diffusion-model", str(diff_path)])

            # Text encoders (GGUF / safetensors)
            t5_rel = model.engine_args.get("t5xxl")
            if t5_rel:
                cmd.extend(["--t5xxl", str(resolve_model_file(t5_rel))])

            clip_rel = model.engine_args.get("clip_l")
            if clip_rel:
                cmd.extend(["--clip_l", str(resolve_model_file(clip_rel))])

            # LLM text encoder (for multimodal / Qwen-Image / Flux 2)
            llm_rel = model.engine_args.get("llm")
            if getattr(request, "custom_text_encoder_path", None):
                custom_te = Path(request.custom_text_encoder_path)
                if custom_te.exists():
                    cmd.extend(["--llm", str(custom_te)])
                else:
                    raise RuntimeError(f"Custom text encoder file not found: {custom_te}")
            elif getattr(request, "text_encoder_id", None):
                te_id = request.text_encoder_id
                direct_cand = resolve_model_file(te_id)
                if direct_cand.exists():
                    cmd.extend(["--llm", str(direct_cand)])
                elif getattr(model, "text_encoders", None):
                    te_spec = model.get_text_encoder(te_id)
                    if te_spec and te_spec.get("file"):
                        te_path = resolve_model_file(te_spec["file"])
                        if not te_path.exists():
                            raise RuntimeError(
                                f"Selected text encoder not found: {te_spec['file']}. "
                                "Please download it from the Models tab."
                            )
                        cmd.extend(["--llm", str(te_path)])
                    elif llm_rel:
                        cmd.extend(["--llm", str(resolve_model_file(llm_rel))])
                elif llm_rel:
                    cmd.extend(["--llm", str(resolve_model_file(llm_rel))])
            elif llm_rel:
                llm_path = resolve_model_file(llm_rel)
                if not llm_path.exists() and getattr(model, "text_encoders", None):
                    # Smart fallback: check if another registered text encoder is installed
                    for te in model.text_encoders:
                        cand = resolve_model_file(te.get("file", ""))
                        if cand.exists():
                            logger.info(f"[sd_cpp] Default LLM {llm_rel} not found; falling back to installed {te.get('file')}")
                            llm_path = cand
                            break
                if not llm_path.exists():
                    raise RuntimeError(
                        f"Text encoder file not found: {llm_rel}. "
                        "Please download it from the Models tab."
                    )
                cmd.extend(["--llm", str(llm_path)])

            # VAE
            vae_rel = None
            vae_type = getattr(request, "vae", None) or getattr(settings, "qwen_vae_type", "texture_fix")
            custom_vae_str = getattr(request, "custom_vae_path", None)
            if custom_vae_str:
                custom_vae_p = Path(custom_vae_str)
                if not custom_vae_p.is_file():
                    raise RuntimeError(f"Custom VAE file not found: {custom_vae_p}")
                cmd.extend(["--vae", str(custom_vae_p)])
            else:
                vae_rel = model.engine_args.get("vae")
                if model.family == "qwen" or model.id.startswith("qwen"):
                    if vae_type == "official":
                        vae_rel = "qwen_image_2.1_vae_bf16.safetensors"
                    else:
                        vae_rel = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"

            if vae_rel:
                vae_search_paths = [
                    resolve_model_file(vae_rel),
                    settings.models_dir / "Klein-4B-Series" / "klein-4b-series-50x-above-base-model" / "vae" / "diffusion_pytorch_model.safetensors",
                    settings.models_dir / "Klein-9B-Series" / "klein-9b-series-50x-above-base-model" / "vae" / "diffusion_pytorch_model.safetensors",
                    settings.models_dir / "FLUX.1-schnell-gguf" / "ae.safetensors",
                    settings.models_dir / "FLUX.1-schnell-gguf" / vae_rel,
                    settings.models_dir / "Qwen-Image-2.1-4bit" / vae_rel,
                    settings.models_dir / "Qwen-Image-2.1-4bit" / "vae" / vae_rel,
                    settings.models_dir / "Qwen-Image-2.1-GGUF" / vae_rel,
                    settings.models_dir / "vae" / vae_rel,
                ]
                if vae_type == "official":
                    vae_search_paths.append(
                        settings.models_dir / "OzzyGT--Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit" / "vae" / "diffusion_pytorch_model.safetensors"
                    )
                candidate_vae = next((p for p in vae_search_paths if p.exists()), resolve_model_file(vae_rel))

                if not candidate_vae.exists():
                    # Auto-download Qwen VAE if requested and missing
                    if model.family == "qwen" or model.id.startswith("qwen"):
                        from .sources import HuggingFaceSource
                        hf = HuggingFaceSource(token=settings.huggingface_token)
                        if vae_type == "texture_fix":
                            emit(ProgressEvent(
                                status=JobStatus.DOWNLOADING,
                                progress=0.1,
                                message=f"Downloading Texture-Fix VAE ({vae_rel}, 644 MB)...",
                            ))
                            hf.download_file(
                                repo_id="madebyollin/texture-fix-vae-for-qwen-image-2.1",
                                file_path="texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
                                local_dir=repo_dir,
                            )
                            candidate_vae = repo_dir / vae_rel
                        else:
                            emit(ProgressEvent(
                                status=JobStatus.DOWNLOADING,
                                progress=0.1,
                                message="Downloading Official Qwen-Image-2.1 VAE (644 MB)...",
                            ))
                            hf.download_file(
                                repo_id="OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit",
                                file_path="vae/diffusion_pytorch_model.safetensors",
                                local_dir=repo_dir,
                            )
                            dl_vae = repo_dir / "vae" / "diffusion_pytorch_model.safetensors"
                            if dl_vae.exists():
                                candidate_vae = repo_dir / vae_rel
                                if not candidate_vae.exists():
                                    dl_vae.rename(candidate_vae)
                    elif model.engine_args.get("vae_repo"):
                        from .sources import HuggingFaceSource
                        hf = HuggingFaceSource(token=settings.huggingface_token)
                        vae_remote = model.engine_args.get("vae_remote", f"vae/{vae_rel}")
                        emit(ProgressEvent(
                            status=JobStatus.DOWNLOADING,
                            progress=0.1,
                            message=f"Downloading VAE ({vae_rel})...",
                        ))
                        hf.download_file(
                            repo_id=model.engine_args["vae_repo"],
                            file_path=vae_remote,
                            local_dir=repo_dir,
                        )
                        target_v = repo_dir / vae_rel
                        dl_v = repo_dir / vae_remote
                        if dl_v.exists() and not target_v.exists() and dl_v != target_v:
                            import shutil
                            target_v.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(dl_v, target_v)
                        candidate_vae = target_v if target_v.exists() else dl_v
                    else:
                        raise RuntimeError(
                            f"VAE file not found: {vae_rel}. "
                            "Please download it from the Models tab."
                        )
                cmd.extend(["--vae", str(candidate_vae)])
                if model.family == "qwen" or model.id.startswith("qwen"):
                    cmd.append("--vae-tiling")

            # VAE latent format override
            if model.engine_args.get("vae_format"):
                cmd.extend(["--vae-format", model.engine_args["vae_format"]])

            # Flow shift
            if model.engine_args.get("flow_shift"):
                cmd.extend(["--flow-shift", str(model.engine_args["flow_shift"])])

            # External Tokenizer (required for MOSS-Image / Qwen3.5 text encoder)
            tok_rel = model.engine_args.get("tokenizer")
            is_moss_or_qwen35 = (
                "moss" in model.id.lower()
                or "moss" in (model.family or "").lower()
                or "qwen3.5" in (llm_rel or "").lower()
                or "moss" in (diff_rel or "").lower()
            )
            if tok_rel or is_moss_or_qwen35:
                tok_rel = tok_rel or "tokenizer.json"
                tok_search = [
                    resolve_model_file(tok_rel),
                    repo_dir / "tokenizer.json",
                    repo_dir / "tokenizer" / "tokenizer.json",
                    repo_dir / "tokenizer_qwen3.5.json",
                    settings.models_dir / "MOSS-Image_GGUFs" / "tokenizer.json",
                    settings.models_dir / "MOSS-Image_GGUFs" / "tokenizer" / "tokenizer.json",
                ]
                tok_path = next((p for p in tok_search if p.exists() and p.is_file()), None)
                if tok_path is None:
                    from .sources import HuggingFaceSource
                    hf = HuggingFaceSource(token=settings.huggingface_token)
                    tok_repo = model.engine_args.get("tokenizer_repo", "OpenMOSS-Team/MOSS-Image-Base")
                    tok_remote = model.engine_args.get("tokenizer_remote", "tokenizer/tokenizer.json")
                    emit(ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=0.12,
                        message="Downloading Qwen3.5 Tokenizer (tokenizer.json) for MOSS-Image...",
                    ))
                    hf.download_file(
                        repo_id=tok_repo,
                        file_path=tok_remote,
                        local_dir=repo_dir,
                    )
                    dl_tok = repo_dir / tok_remote
                    target_tok = repo_dir / "tokenizer.json"
                    if dl_tok.exists() and not target_tok.exists() and dl_tok != target_tok:
                        import shutil
                        shutil.copy2(dl_tok, target_tok)
                    tok_path = target_tok if target_tok.exists() else dl_tok
                if tok_path and tok_path.exists():
                    cmd.extend(["--tokenizer", str(tok_path)])

        # ----- Inpaint mask preparation -----
        # SD.CPP applies --mask to the init image (white = regenerate,
        # black = preserve), so a mask always forces classic img2img mode.
        mask_source = getattr(request, "mask_data_url", None) or (
            str(request.mask_path) if getattr(request, "mask_path", None) else None
        )
        mask_ctx = None
        if mask_source:
            from PIL import Image
            from .masking import (
                MaskError,
                mask_file_name,
                prepare_mask_arrays,
                save_mask_file,
            )

            raw_init = getattr(request, "init_image_path", None)
            if not raw_init:
                raise ValueError(
                    "Mask editing requires a source image (init_image_path is missing)"
                )
            source_path = Path(raw_init)
            if not source_path.is_absolute():
                source_path = settings.output_dir / source_path
            if not source_path.exists():
                raise ValueError(f"Source image not found: {source_path.name}")

            with Image.open(source_path) as src_file:
                source_image = src_file.copy()
            try:
                raw_mask, engine_mask = prepare_mask_arrays(
                    mask_source,
                    target_size=source_image.size,
                    invert_mask=bool(getattr(request, "mask_invert", False)),
                    grow=int(getattr(request, "mask_grow", 0) or 0),
                    blur=float(getattr(request, "mask_blur", 0.0) or 0.0),
                )
            except MaskError as exc:
                raise ValueError(str(exc)) from exc
            mask_file = save_mask_file(
                engine_mask,
                settings.output_dir / "masks",
                mask_file_name("sd_cpp_mask"),
            )
            mask_ctx = {
                "array": raw_mask,
                "file": mask_file,
                "source_image": source_image,
            }
            emit(ProgressEvent(
                status=JobStatus.INITIALIZING,
                message=f"Mask ready ({source_image.size[0]}x{source_image.size[1]})",
            ))

        # ``mask_strength`` may legitimately be 0.0, so ``or`` must NOT be
        # used here — it would silently turn 0 into 1.
        raw_mask_strength = getattr(request, "mask_strength", None)
        mask_strength_value = 1.0 if raw_mask_strength is None else float(raw_mask_strength)

        # Reference Image: Native Control/Ref (-r / --control-image / --control-net) vs Classic Noise Img2Img (-i + --strength)
        if getattr(request, "init_image_path", None):
            ref_path = Path(request.init_image_path)
            if not ref_path.is_absolute():
                ref_path = settings.output_dir / ref_path
            if ref_path.exists():
                ref_mode = (getattr(request, "ref_mode", None) or "control_ref").strip().lower()
                strength = float(getattr(request, "init_image_strength", 0.85) or 0.85)
                cn_str = (getattr(request, "control_net_path", None) or "").strip()
                cn_path: Optional[Path] = None
                if cn_str:
                    cand_cn = Path(cn_str)
                    if not cand_cn.is_absolute():
                        for p in (resolve_model_file(cn_str), settings.models_dir / cn_str, settings.models_dir / "controlnet" / cn_str):
                            if p.exists():
                                cand_cn = p
                                break
                    if cand_cn.exists():
                        cn_path = cand_cn

                if mask_ctx is not None:
                    # Masked img2img: SD.CPP only honours --mask together with
                    # --init-img.  mask_strength maps to the denoising strength.
                    cmd.extend(["-i", str(ref_path)])
                    cmd.extend(["--strength", str(round(max(0.01, min(1.0, mask_strength_value)), 2))])
                    cmd.extend(["--mask", str(mask_ctx["file"])])
                elif ref_mode == "img2img":
                    # Classic noise-based Img2Img (--init-img + --strength)
                    cmd.extend(["-i", str(ref_path)])
                    cmd.extend(["--strength", str(round(strength, 2))])
                else:
                    # Step isolation: NO --init-img (-i) and NO --strength!
                    # Model executes 100% of steps from clean white noise while guided by ControlNet / Reference latents.
                    if cn_path is not None:
                        cmd.extend([
                            "--control-net", str(cn_path),
                            "--control-image", str(ref_path),
                            "--control-strength", str(round(strength, 2)),
                        ])
                        if ref_mode == "controlnet_canny":
                            cmd.extend(["--image-preprocess", "target=control,canny=true"])
                    elif model.family == "qwen" or model.id.startswith("qwen"):
                        # Native Qwen-Image-2.1 Reference / Control conditioning (-r + --llm_vision)
                        mmproj_candidates = [
                            repo_dir / "mmproj-BF16.gguf",
                            repo_dir / "mmproj-F16.gguf",
                            settings.models_dir / "unsloth--Qwen3-VL-8B-Instruct-GGUF" / "mmproj-BF16.gguf",
                            settings.models_dir / "unsloth--Qwen3-VL-8B-Instruct-GGUF" / "mmproj-F16.gguf",
                        ]
                        mmproj_path = next((p for p in mmproj_candidates if p.exists()), None)
                        if mmproj_path is None:
                            from .sources import HuggingFaceSource
                            hf = HuggingFaceSource(token=settings.huggingface_token)
                            emit(ProgressEvent(
                                status=JobStatus.DOWNLOADING,
                                progress=0.12,
                                message="Downloading Qwen3-VL Vision Projector (mmproj-F16.gguf, ~1.1 GB) for Native Control/Reference...",
                            ))
                            hf.download_file(
                                repo_id="unsloth/Qwen3-VL-8B-Instruct-GGUF",
                                file_path="mmproj-F16.gguf",
                                local_dir=repo_dir,
                            )
                            mmproj_path = repo_dir / "mmproj-F16.gguf"
                        w32 = max(32, (width // 32) * 32)
                        h32 = max(32, (height // 32) * 32)
                        preproc = f"target=ref,mode=crop-resize,width={w32},height={h32}"
                        if ref_mode == "controlnet_canny":
                            preproc += ",canny=true"
                        cmd.extend([
                            "--llm_vision", str(mmproj_path),
                            "-r", str(ref_path),
                            "--image-preprocess", preproc,
                        ])
                    elif model.id in ("flux_2_klein_base_9b_gguf", "flux_3_image_klein_gguf") or "klein" in model.id or model.family == "flux":
                        # Native Flux.2 / Flux.3 in-context Reference / Control conditioning (-r)
                        cmd.extend(["-r", str(ref_path)])
                        if ref_mode == "controlnet_canny":
                            cmd.extend(["--image-preprocess", "target=ref,canny=true"])
                    else:
                        # Fallback for SDXL without external ControlNet file
                        cmd.extend(["-i", str(ref_path), "--strength", str(round(strength, 2))])

        # Guidance / CFG scale
        req_cfg = getattr(request, "cfg", None)
        if req_cfg is None:
            req_cfg = getattr(request, "guidance_scale", None)

        variant_id = getattr(request, "variant_id", "") or ""
        is_base = ("base" in variant_id.lower()) or ("base" in model.id.lower())

        if not is_base and (model.engine_args.get("guidance_type") == "distilled" or model.family in ("flux", "sd3")):
            cfg_val = req_cfg if req_cfg is not None else model.get_default_cfg()
            # Always pass --cfg-scale 1.0 alongside --guidance so sd-cli doesn't use default cfg_scale=7.0
            cmd.extend(["--cfg-scale", "1.0", "--guidance", str(cfg_val)])
        else:
            default_cfg = model.scheduler_presets.get("base", {}).get("cfg") if is_base and "base" in model.scheduler_presets else model.get_default_cfg()
            cfg_val = req_cfg if req_cfg is not None else default_cfg
            cmd.extend(["--cfg-scale", str(cfg_val)])

        # LoRA injection: append prompt tags <lora:stem:strength> and specify --lora-model-dir
        lora_paths = getattr(request, "lora_paths", None) or []
        lora_strengths = getattr(request, "lora_strengths", None) or []
        if lora_paths:
            from .lora import get_loras_dir, lora_prompt_tags
            lora_tags = lora_prompt_tags(lora_paths, lora_strengths)
            if lora_tags:
                prompt = f"{prompt} {lora_tags}".strip()
            cmd.extend(["--lora-model-dir", str(get_loras_dir())])

        # Prompt & Negative prompt
        cmd.extend(["-p", prompt])
        neg_str = (getattr(request, "negative_prompt", None) or "").strip()
        if neg_str:
            cmd.extend(["-n", neg_str])
        elif (model.family in ("qwen", "klein") or model.id.startswith("qwen") or is_base) and cfg_val > 1.0:
            # Ensure unconditional pass is active in sd.cpp when CFG > 1.0
            cmd.extend(["-n", " "])

        # Dimensions, steps, seed
        cmd.extend([
            "-W", str(width),
            "-H", str(height),
            "--steps", str(request.steps or 20),
            "-s", str(seed),
            "-o", str(output_path),
        ])

        # Flash Attention for diffusion model
        cmd.append("--diffusion-fa")

        # Sampling method & Noise / Eta
        noise_val = float(getattr(request, "noise", 0.0) or 0.0)
        ui_sampler = getattr(request, "sampler", None)
        if not ui_sampler and getattr(model, "scheduler_presets", None) and "default" in model.scheduler_presets:
            ui_sampler = model.scheduler_presets["default"].get("sampler")
        if not ui_sampler:
            ui_sampler = "euler"
        if noise_val > 0.0 and ui_sampler == "euler":
            ui_sampler = "euler_a"

        sd_cpp_sampler_map = {
            "euler": "euler",
            "euler_a": "euler_a",
            "heun": "heun",
            "dpm++2m": "dpm++2m",
            "dpm++2m_sde": "er_sde",
            "dpm2": "dpm2",
            "dpm2_a": "dpm++2s_a",
            "unipc": "res_multistep",
            "lcm": "lcm",
        }
        mapped_sampler = sd_cpp_sampler_map.get(ui_sampler, "euler")
        cmd.extend(["--sampling-method", mapped_sampler])
        # Always pass --eta explicitly when noise is specified or stochastic sampler is active
        if noise_val > 0.0 or mapped_sampler in ("euler_a", "er_sde", "dpm++2s_a", "lcm"):
            cmd.extend(["--eta", str(noise_val)])

        # Logging
        logger.info(f"[sd_cpp] Launching SD.CPP: {width}x{height}, steps={request.steps}, cfg={cfg_val}, sampler={mapped_sampler}, eta={noise_val}, seed={seed}")
        logger.debug(f"[sd_cpp] CLI args: {' '.join(cmd)}")

        # 4. Environment PATH
        env = os.environ.copy()
        sd_cpp_dir = get_sd_cpp_dir()
        cuda_dirs = [
            r"G:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
            r"G:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64",
            str(settings.engine_dir),
            str(sd_cpp_dir),
        ]
        valid_cuda = [d for d in cuda_dirs if os.path.exists(d)]
        env["PATH"] = ";".join(valid_cuda + [env.get("PATH", "")])
        if settings.huggingface_token:
            env["HF_TOKEN"] = settings.huggingface_token
            env["HUGGING_FACE_HUB_TOKEN"] = settings.huggingface_token

        start_time = time.time()
        emit(ProgressEvent(
            status=JobStatus.GENERATING,
            progress=0.05,
            message="Starting SD.CPP inference...",
        ))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(sd_cpp_dir),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._current_process = proc

        total_steps = request.steps or 20
        captured_lines = []

        try:
            for line in proc.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                captured_lines.append(line_str)
                logger.debug(f"[sd_cpp] {line_str}")

                # Check cancel
                if self._cancel_flag.is_set():
                    proc.terminate()
                    raise RuntimeError("Generation cancelled by user")

                # Parse progress
                step_match = _STEP_PROGRESS_RE.search(line_str)
                if step_match:
                    try:
                        cur_step = int(step_match.group(1))
                        tot_steps = int(step_match.group(2))
                        pct = min(0.95, max(0.05, cur_step / tot_steps))
                        emit(ProgressEvent(
                            status=JobStatus.GENERATING,
                            progress=pct,
                            step=cur_step,
                            total_steps=tot_steps,
                            message=f"Sampling step {cur_step}/{tot_steps} ({int(pct*100)}%)",
                        ))
                    except ValueError:
                        pass
                else:
                    pct_match = _PERCENT_PROGRESS_RE.search(line_str)
                    if pct_match:
                        try:
                            val = int(pct_match.group(1))
                            emit(ProgressEvent(
                                status=JobStatus.GENERATING,
                                progress=val / 100.0,
                                message=f"Sampling ({val}%)",
                            ))
                        except ValueError:
                            pass

            proc.wait()
        finally:
            self._current_process = None

        if proc.returncode != 0:
            tail = "\n".join(captured_lines[-20:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SD.CPP process exited with code {proc.returncode}.\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        if not output_path.exists():
            tail = "\n".join(captured_lines[-20:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SD.CPP finished but output file not found: {output_path}\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        # ----- Mask compositing (guaranteed preservation) -----
        # result = generated * mask + original * (1 - mask), at the ORIGINAL
        # resolution.  The raw engine output is kept as the Inpaint fragment.
        raw_fragment_rel = None
        if mask_ctx is not None:
            from PIL import Image
            from .masking import composite_with_mask

            try:
                with Image.open(output_path) as gen_file:
                    generated_rgb = gen_file.convert("RGB")
                source_image = mask_ctx["source_image"]
                composite = composite_with_mask(
                    generated_rgb, source_image, mask_ctx["array"]
                )
                composite.save(output_path, format="PNG")

                frag_dir = settings.output_dir / "masks"
                frag_dir.mkdir(parents=True, exist_ok=True)
                frag_path = frag_dir / f"{output_path.stem}_raw.png"
                generated_rgb.resize(source_image.size, Image.LANCZOS).save(
                    frag_path, format="PNG"
                )
                raw_fragment_rel = f"masks/{frag_path.name}"
                logger.info(
                    "[sd_cpp] Mask composite applied -> %s (fragment: %s)",
                    output_path.name,
                    raw_fragment_rel,
                )
            except Exception as comp_err:
                raise RuntimeError(f"Mask compositing failed: {comp_err}") from comp_err

        elapsed = time.time() - start_time

        # Save metadata sidecar
        sidecar_path = output_path.with_suffix(".json")
        try:
            meta = {
                "filename": output_path.name,
                "prompt": prompt,
                "negative_prompt": getattr(request, "negative_prompt", None),
                "model_id": request.model_id,
                "variant_id": getattr(request, "variant_id", None),
                "seed": seed,
                "width": width,
                "height": height,
                "steps": request.steps,
                "cfg": cfg_val,
                "noise": noise_val,
                "sampler": ui_sampler,
                "vae": vae_type,
                "backend": "sd_cpp",
                "engine": "stable-diffusion.cpp",
                "owner_id": getattr(request, "owner_id", "host"),
                "owner_name": getattr(request, "owner_name", "Хост"),
                "shared": getattr(request, "shared", True),
                "generation_time_sec": round(elapsed, 2),
                "timestamp": int(time.time()),
            }
            if mask_ctx is not None:
                meta.update({
                    "mask_applied": True,
                    "mask_strength": mask_strength_value,
                    "mask_grow": int(getattr(request, "mask_grow", 0) or 0),
                    "mask_blur": float(getattr(request, "mask_blur", 0.0) or 0.0),
                    "mask_source": mask_ctx["file"].name,
                    "mask_raw_fragment": raw_fragment_rel,
                    "composite": "generated*mask + original*(1-mask)",
                })
            else:
                meta["mask_applied"] = False
            sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[sd_cpp] Failed to write sidecar metadata: {e}")

        emit(ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=str(output_path),
            raw_image_path=raw_fragment_rel,
            message=f"Done in {elapsed:.1f}s",
        ))

        return GenerateResult(
            image_path=output_path,
            seed_used=seed,
            generation_time_sec=elapsed,
            model_id=request.model_id,
            backend="sd_cpp",
            steps=request.steps,
            prompt=prompt,
        )
