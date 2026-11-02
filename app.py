"""



QuantFunc WebUI — main entry point.







Architecture: gradio.Server + FastAPI + custom index.html (NOT gr.Blocks).







This follows the pattern from the HF Space prithivMLmods/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast,



which serves a hand-written index.html at / and exposes JSON API endpoints.







The UI is fully custom HTML/CSS/JS — no Gradio components are rendered.



Backend communication happens via REST endpoints defined below.







Run:



    python app.py







Then open: http://127.0.0.1:7860



"""



from __future__ import annotations







import asyncio



import json



import math



import os



os.environ["HF_HUB_DISABLE_XET"] = "1"



try:
    import hf_transfer as _hf_transfer_probe  # noqa: F401
    _HF_TRANSFER_OK = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
except Exception:
    _HF_TRANSFER_OK = False
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if _HF_TRANSFER_OK else "0"



try:



    import huggingface_hub.constants as _hfc



    _hfc.HF_HUB_DISABLE_XET = True



    _hfc.HF_HUB_ENABLE_HF_TRANSFER = _HF_TRANSFER_OK



except Exception:



    pass


def hf_fast_backend():
    """Which fast-download backend the installed huggingface_hub supports.

    huggingface_hub 1.x dropped hf_transfer (Xet replaced it); 0.x still uses hf_transfer.
    Returns (backend_name, package_installed).
    """
    import importlib.util
    try:
        import huggingface_hub
        major = int(str(huggingface_hub.__version__).split(".")[0])
    except Exception:
        major = 0
    if major >= 1:
        return "xet", importlib.util.find_spec("hf_xet") is not None
    return "hf_transfer", importlib.util.find_spec("hf_transfer") is not None


def apply_hf_download_settings(fast_enabled=False, endpoint=None):
    """Apply fast-download toggle + HF mirror to env and huggingface_hub constants.

    Safe to call at runtime: huggingface_hub reads these constants per call.
    Returns (fast_active, backend, endpoint_in_use).
    """
    backend, installed = hf_fast_backend()
    active = bool(fast_enabled) and installed
    if backend == "xet":
        # Xet was disabled by default after an allocator crash (see git f26a624),
        # so it is strictly opt-in via the Settings toggle.
        os.environ["HF_HUB_DISABLE_XET"] = "0" if active else "1"
        if active:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        else:
            os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if active else "0"
    ep = str(endpoint or "").strip().rstrip("/")
    if ep and not ep.lower().startswith(("http://", "https://")):
        ep = ""
    used = ep or "https://huggingface.co"
    if ep:
        os.environ["HF_ENDPOINT"] = ep
    else:
        os.environ.pop("HF_ENDPOINT", None)
    try:
        import huggingface_hub.constants as hfc
        if backend == "xet":
            hfc.HF_HUB_DISABLE_XET = not active
        else:
            hfc.HF_HUB_ENABLE_HF_TRANSFER = active
        hfc.ENDPOINT = used
        hfc.HUGGINGFACE_CO_URL_TEMPLATE = used + "/{repo_id}/resolve/{revision}/{filename}"
    except Exception:
        pass
    return active, backend, used


import sys



import threading



import time



import uuid



from pathlib import Path



from typing import Optional


# Legacy Windows consoles (cp1251/cp866) cannot encode arrows/emoji used in
# startup messages. Never let a log line crash the server before it binds.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError, ValueError):
        pass





# Add project root to path



PROJECT_ROOT = Path(__file__).parent.resolve()



sys.path.insert(0, str(PROJECT_ROOT))







# ---------- Backend imports ----------



from backend.cache import CacheManager



from backend.model_files import vae_download_refs, vae_payload, vae_payload_list



from backend.engine import (



    check_engine_status,



    download_engine,



    engine_config_json_path,



    ensure_config_json,



    is_engine_installed,



    list_installed_engine_versions,



    read_engine_api_key,



    verify_engine,



)



from backend.sd_cpp import is_sd_cpp_installed, get_sd_cpp_version



from backend.sdnq_engine import is_sdnq_installed, get_sdnq_version, get_sdnq_path



from backend.gpu import detect_gpu, get_system_memory_stats, GPUInfo



from backend.generation_queue import GenerationQueue
from backend.history_paging import paginate
from backend.presence import get_presence
from backend.registry import get_registry
from backend.registry.loader import model_supports_mask, quantfunc_model_disabled



from backend.settings import get_settings



from backend.worker import (



    GenerateRequest,



    JobStatus,



    ProgressEvent,



    get_worker,



)







# ---------- Web framework ----------



# NOTE: We use FastAPI + uvicorn directly (no gradio.Blocks).



# The HF Space reference (prithivMLmods/Flux.2-Klein-KV-Edit) uses the same



# pattern: gradio.Server is replaced by a hand-rolled FastAPI app serving



# custom index.html + JSON API endpoints. This avoids gradio's heavy



# dependencies (huggingface_hub version conflicts, etc.).



from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Query



from fastapi.responses import HTMLResponse, JSONResponse



from fastapi.staticfiles import StaticFiles











# ============================================================



# State — held in module scope (single-user, global)



# ============================================================



class AppState:
    """Global app state - single-user, single-process."""

    def __init__(self):
        self.gpu_info: Optional[GPUInfo] = None
        self.jobs: dict[str, dict] = {}   # job_id -> {status, progress, ...}
        self._job_lock = threading.Lock()
        self.generation_queue = GenerationQueue()

    def register_job(
        self,
        job_id: str,
        owner_id: str = "host",
        owner_name: str = "Хост",
        kind: str = "download",
        serialized: bool = False,
        shared: bool = False,
    ) -> None:
        now = time.time()
        with self._job_lock:
            self.jobs[job_id] = {
                "status": JobStatus.QUEUED,
                "progress": 0.0,
                "message": "Queued",
                "started_at": now,
                "created_at": now,
                "finished_at": None,
                "image_path": None,
                "error": None,
                "step": 0,
                "total_steps": 0,
                "owner_id": owner_id or "host",
                "owner_name": owner_name or "Хост",
                "kind": kind,
                "shared": bool(shared),
                "queue_position": 0,
                "queue_size": 0,
                "queue_active": False,
                "serialized": bool(serialized),
            }
            if serialized:
                self.generation_queue.enqueue(job_id)
        if serialized:
            self._sync_queue_positions()

    def _sync_queue_positions(self) -> None:
        """Mirror the FIFO queue into JSON-safe job metadata."""
        snap = self.generation_queue.snapshot()
        pending = snap.get("pending") or []
        total = len(pending)
        active = snap.get("active")
        with self._job_lock:
            for index, pending_id in enumerate(pending, start=1):
                job = self.jobs.get(pending_id)
                if not job:
                    continue
                job.update({
                    "status": JobStatus.QUEUED,
                    "progress": 0.0,
                    "message": f"Queue position {index} of {total}",
                    "queue_position": index,
                    "queue_size": total,
                    "queue_active": False,
                })
            if active and active in self.jobs:
                self.jobs[active].update({
                    "queue_position": 0,
                    "queue_size": total,
                    "queue_active": True,
                })

    def update_job(self, job_id: str, event: ProgressEvent) -> None:
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            # Never overwrite a user cancellation with a late error from a
            # worker that was terminated mid-generation.
            if job.get("status") == JobStatus.CANCELLED and event.status != JobStatus.CANCELLED:
                return
            job.update({
                "status": event.status,
                "progress": event.progress,
                "message": event.message,
                "step": event.step,
                "total_steps": event.total_steps,
                "image_path": event.image_path,
                "raw_image_path": getattr(event, "raw_image_path", None),
                "error": event.error,
            })
            if event.status in (JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELLED):
                job["finished_at"] = time.time()

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._job_lock:
            job = self.jobs.get(job_id)
            return dict(job) if job is not None else None

    def job_status(self, job_id: str) -> Optional[str]:
        with self._job_lock:
            job = self.jobs.get(job_id)
            return job.get("status") if job else None

    def acquire_generation(self, job_id: str) -> bool:
        """Wait for this job's turn, then mark it active."""
        acquired = self.generation_queue.acquire(job_id)
        if not acquired:
            # acquire() returns False when a cancel marker was consumed. If the
            # cancel arrived after this job had already become active, release
            # the slot here; otherwise no later worker could release it.
            self.generation_queue.release(job_id)
            with self._job_lock:
                job = self.jobs.get(job_id)
                if job and job.get("status") not in (
                    JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELLED
                ):
                    job.update({
                        "status": JobStatus.CANCELLED,
                        "message": "Cancelled before generation started",
                        "finished_at": time.time(),
                    })
            self._sync_queue_positions()
            return False
        cancelled = False
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job is None:
                cancelled = True
            elif job.get("status") == JobStatus.CANCELLED:
                # A cancel request can arrive after GenerationQueue made this
                # job active but before we wrote INITIALIZING. Never resurrect
                # a user-cancelled job by overwriting that status.
                cancelled = True
            else:
                job.update({
                    "status": JobStatus.INITIALIZING,
                    "progress": 0.0,
                    "message": "GPU queue slot acquired - preparing...",
                    "queue_position": 0,
                    "queue_active": True,
                })
        if cancelled:
            self.generation_queue.release(job_id)
            self._sync_queue_positions()
            return False
        self._sync_queue_positions()
        return True

    def release_generation(self, job_id: str) -> None:
        self.generation_queue.release(job_id)
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job:
                job.update({"queue_position": 0, "queue_active": False})
        self._sync_queue_positions()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued/active job. Returns True if it owned the GPU slot."""
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job.get("status") in (JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELLED):
                return False
        was_active = self.generation_queue.cancel(job_id)
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job is not None:
                job.update({
                    "status": JobStatus.CANCELLED,
                    "message": "Cancelled by user",
                    "queue_position": 0,
                    "finished_at": time.time(),
                })
        self._sync_queue_positions()
        return was_active


STATE = AppState()

# Halftone Studio and Manga CPU pipelines are intentionally parallel, but a
# burst from several browser tabs must not saturate every CPU core / RAM.
TOOL_CPU_MAX_CONCURRENCY = 4
_TOOL_CPU_SEMAPHORE = threading.BoundedSemaphore(TOOL_CPU_MAX_CONCURRENCY)


def _run_cpu_limited(func, *args, **kwargs):
    """Run CPU-heavy tool work under the shared 4-slot semaphore."""
    with _TOOL_CPU_SEMAPHORE:
        return func(*args, **kwargs)


async def _run_cpu_tool(func, *args, **kwargs):
    return await asyncio.to_thread(_run_cpu_limited, func, *args, **kwargs)











# ============================================================



# Startup — detect GPU, check engine, load registry



# ============================================================



def init_state() -> None:



    """Run at startup: detect GPU, check engine, load model registry."""



    print("=" * 60)



    print("QuantFunc WebUI — starting up")



    print("=" * 60)







    # Load settings



    settings = get_settings()



    print(f"[startup] Settings loaded from: {settings.settings_file_path}")
    _hf_on, _hf_be, _hf_ep = apply_hf_download_settings(
        getattr(settings, "hf_transfer", False), getattr(settings, "hf_endpoint", None))
    print(f"[startup] HF downloads: fast={'on' if _hf_on else 'off'} ({_hf_be}), endpoint={_hf_ep}")



    print(f"[startup] Models dir: {settings.models_dir}")



    print(f"[startup] Engine dir: {settings.engine_dir}")



    print(f"[startup] Language: {settings.lang}")







    # Detect GPU



    print("[startup] Detecting GPU...")



    STATE.gpu_info = detect_gpu()



    if STATE.gpu_info:



        g = STATE.gpu_info



        print(f"[startup] GPU: {g.name}")



        print(f"[startup]   SM: {g.sm} -> tier={g.tier}, klein_tier={g.klein_tier}")



        print(f"[startup]   VRAM: {g.vram_mb} MB ({g.vram_gb:.1f} GB)")



        print(f"[startup]   Driver: {g.driver_version}")



        # Persist to settings



        settings.update(



            gpu_name=g.name,



            gpu_sm=g.sm,



            gpu_vram_mb=g.vram_mb,



            gpu_driver_version=g.driver_version,



        )



    else:



        print("[startup] WARNING: No NVIDIA GPU detected. WebUI will start but generation will fail.")







    # Check engine



    print("[startup] Checking engine...")



    engine_status = check_engine_status()



    print(f"[startup] Engine status: {engine_status.status} — {engine_status.message}")







    # Load model registry



    print("[startup] Loading model registry...")



    registry = get_registry()



    print(f"[startup] Loaded {len(registry.models)} models: {list(registry.models.keys())}")







    # Ensure engine auth config exists (bin/<os>/config.json with the bundled



    # free-trial test key). Without it key-protected models fail at



    # checkObfuscation. Best-effort — never blocks startup.



    try:



        cfg_path, cfg_source = ensure_config_json()



        print(f"[startup] Engine auth config: {cfg_path} (source: {cfg_source})")



    except Exception as e:



        print(f"[startup] WARNING: could not ensure engine config.json: {e}")







    print("=" * 60)



    print(f"[startup] Ready. Open http://127.0.0.1:7860")



    print("=" * 60)











# ============================================================



# Build the main HTML page & Engine Status



# ============================================================



def _get_engines_info() -> dict:



    from backend.engine import is_engine_installed, get_engine_paths



    from backend.sd_cpp import is_sd_cpp_installed, get_sd_cpp_cli_path, get_sd_cpp_version



    from backend.extras import is_orbitquant_ready as is_orbitquant_installed, is_nunchaku_ready as is_nunchaku_installed







    _, cli17 = get_engine_paths("0.0.17")



    _, cli12 = get_engine_paths("0.0.12")



    cli_sd = get_sd_cpp_cli_path()







    return {



        "quantfunc_0_17": {



            "id": "quantfunc_0_17",



            "engine": "quantfunc",



            "version": "0.0.17",



            "name": "QuantFunc v0.0.17",



            "title": "QuantFunc v0.0.17 (Primary SVDQuant)",



            "installed": is_engine_installed("0.0.17"),



            "path": str(cli17),



            "description": "Primary engine for Qwen-Image-2.1 and standard SVDQuant models.",



        },



        "quantfunc_0_12": {



            "id": "quantfunc_0_12",



            "engine": "quantfunc",



            "version": "0.0.12",



            "name": "QuantFunc v0.0.12",



            "title": "QuantFunc v0.0.12 (Legacy NVFP4)",



            "installed": is_engine_installed("0.0.12"),



            "path": str(cli12),



            "description": "Legacy engine with native NVFP4 support (Flux.2 Klein FP4, Z-Image).",



        },



        "sd_cpp": {



            "id": "sd_cpp",



            "engine": "sd_cpp",



            "version": get_sd_cpp_version(),



            "name": "stable-diffusion.cpp",



            "title": "stable-diffusion.cpp (GGUF Runtime)",



            "installed": is_sd_cpp_installed(),



            "path": str(cli_sd),



            "description": "Standalone C++/CUDA runner for GGUF models (Flux.1 Schnell, SDXL, Unsloth Q4).",



        },



        "sdnq": {



            "id": "sdnq",



            "engine": "sdnq",



            "version": get_sdnq_version(),



            "name": "SDNQ (Diffusers)",



            "title": "SDNQ (SD.Next Quantization + Diffusers)",



            "installed": is_sdnq_installed(),



            "path": get_sdnq_path(),



            "description": "PyTorch + Triton INT8/INT4/FP8 quantization with Hadamard rotation & SVD (Qwen-Image-2.1 Turbo INT8).",



        },



        "orbitquant": {



            "id": "orbitquant",



            "engine": "orbitquant",



            "version": "W4A4",



            "name": "OrbitQuant",



            "title": "OrbitQuant W4A4 (WaveCut)",



            "installed": is_orbitquant_installed(),



            "path": "Diffusers + OrbitQuant runtime",



            "description": "Quantized rotation transforms + 4-bit weights and activations for Klein and Z-Image.",



        },



        "nunchaku": {



            "id": "nunchaku",



            "engine": "nunchaku",



            "version": "1.3.0",



            "name": "Nunchaku SVDQ",



            "title": "Nunchaku SVDQ (vitoom-nunchaku)",



            "installed": is_nunchaku_installed(),



            "path": "vitoom-nunchaku wheels",



            "description": "Ultra-fast SVDQuant 4-bit kernel engine for FLUX.2 Klein (INT4 / FP4 Blackwell).",



        },



    }











def _api_auth_status() -> dict:



    """Engine auth status for the frontend (config.json + user key)."""



    cfg_present = engine_config_json_path().exists()



    has_cfg_key = read_engine_api_key() is not None



    return {



        "config_present": cfg_present,



        "config_has_key": has_cfg_key,



        "user_key_set": bool(get_settings().quantfunc_api_key and get_settings().quantfunc_api_key.strip()),



    }











def build_index_html() -> str:



    """Read frontend/index.html and inject settings + GPU info as JSON."""



    html_path = PROJECT_ROOT / "frontend" / "index.html"



    if not html_path.exists():



        return "<html><body><h1>frontend/index.html not found</h1></body></html>"







    with open(html_path, "r", encoding="utf-8") as f:



        html = f.read()







    # Inject initial config (so JS doesn't need to fetch on load)



    settings = get_settings()



    gpu_dict = None



    if STATE.gpu_info:



        gpu_dict = {



            "name": STATE.gpu_info.name,



            "sm": STATE.gpu_info.sm,



            "vram_mb": STATE.gpu_info.vram_mb,



            "driver_version": STATE.gpu_info.driver_version,



            "tier": STATE.gpu_info.tier,



            "klein_tier": STATE.gpu_info.klein_tier,



        }







    engine_status = check_engine_status()



    installed_engines = list_installed_engine_versions()



    if is_sd_cpp_installed():



        installed_engines.append("sd_cpp")



    if is_sdnq_installed():



        installed_engines.append("sdnq")







    config_json = json.dumps({



        "settings": {



            "lang": settings.lang,



            "vram_mode": settings.vram_mode,



            "auto_optimize": settings.auto_optimize,



            "models_dir": str(settings.models_dir),



            "output_dir": str(settings.output_dir),



            "engine_version": settings.engine_version,



            "engine_platform": settings.engine_platform,



            "cuda_version": settings.cuda_version,



            "qwen_vae_type": getattr(settings, "qwen_vae_type", "texture_fix"),



        },



        "gpu": gpu_dict,



        "engine_status": {



            "status": engine_status.status,



            "message": engine_status.message,



        },



        "installed_engines": installed_engines,



        "engines": _get_engines_info(),



        "sd_cpp_installed": is_sd_cpp_installed(),



        "sd_cpp_version": get_sd_cpp_version(),



        "sdnq_installed": is_sdnq_installed(),



        "sdnq_version": get_sdnq_version(),



        "api_auth": _api_auth_status(),



    }, default=str)







    # Inject before </head>



    html = html.replace(



        "</head>",



        f'<script>window.QF_INITIAL_CONFIG = {config_json};</script></head>'



    )



    return html











# ============================================================



# FastAPI app — built on top of Gradio



# ============================================================



def _get_token_for_source(source_type: str) -> Optional[str]:



    """Get auth token for a source type from settings."""



    settings = get_settings()



    if source_type == "huggingface":



        return settings.huggingface_token



    elif source_type == "modelscope":



        return settings.modelscope_token



    return None











def _parse_int_in_range(value, name: str, bounds) -> int:

    """Parse an integer request field and check it against an inclusive range."""

    try:

        number = int(value)

    except (TypeError, ValueError) as exc:

        raise ValueError(f"{name} must be an integer") from exc

    low, high = bounds

    if not low <= number <= high:

        raise ValueError(f"{name} must be between {low} and {high}")

    return number




def _parse_mask_crop_box(value) -> Optional[list]:

    """Validate the manual generation-area rectangle of a generate request.

    The rectangle is a ``[x0, y0, x1, y1]`` list (a mapping with the same keys
    is accepted as well) expressed in the pixel coordinates of the ORIGINAL
    image.  ``None`` means "not set" and is only allowed for the automatic crop
    modes.
    """

    if value is None:

        return None

    if isinstance(value, dict):

        values = [value.get("x0"), value.get("y0"), value.get("x1"), value.get("y1")]

    elif isinstance(value, (list, tuple)) and len(value) == 4:

        values = list(value)

    else:

        raise ValueError("mask_crop_box must be [x0, y0, x1, y1]")

    if any(item is None for item in values):

        raise ValueError("mask_crop_box must contain four coordinates")

    numbers = []

    for item in values:

        try:

            number = float(item)

        except (TypeError, ValueError) as exc:

            raise ValueError("mask_crop_box coordinates must be numbers") from exc

        if not math.isfinite(number):

            raise ValueError("mask_crop_box coordinates must be finite numbers")

        numbers.append(int(round(number)))

    x0, y0, x1, y1 = numbers

    if x0 > x1:

        x0, x1 = x1, x0

    if y0 > y1:

        y0, y1 = y1, y0

    return [x0, y0, x1, y1]




def create_app() -> FastAPI:



    """Create the FastAPI app with custom routes + Gradio mount."""







    # Use Gradio's FastAPI underpinning



    app = FastAPI(title="QuantFunc WebUI", version="0.1.0")







    # Access / Share Middleware & Extras / LoRA extensions



    from backend.share import ShareAccessMiddleware, register_share_routes



    from backend.extras import register_extras_routes







    app.add_middleware(ShareAccessMiddleware)



    register_share_routes(app)



    register_extras_routes(app)







    @app.middleware("http")



    async def add_dev_no_cache_headers(request: Request, call_next):



        response = await call_next(request)



        p = request.url.path



        if p in ("/", "/index.html") or p.startswith(("/js/", "/css/", "/locale/")):



            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"



            response.headers["Pragma"] = "no-cache"



            response.headers["Expires"] = "0"



        return response







    # ---------- Static file mounts ----------



    frontend_dir = PROJECT_ROOT / "frontend"



    app.mount("/css", StaticFiles(directory=str(frontend_dir / "css")), name="css")



    app.mount("/js", StaticFiles(directory=str(frontend_dir / "js")), name="js")



    app.mount("/locale", StaticFiles(directory=str(frontend_dir / "locale")), name="locale")







    # Output dir mount (for serving generated images)



    settings = get_settings()



    settings.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists before mount



    app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")







    def _is_te_installed(te: dict, repo_dir: Optional[Path], cache: CacheManager) -> bool:
        te_f = te.get("file", "")
        if not te_f:
            return False
        if repo_dir:
            for sub in ("", "text_encoders", "text_encoder"):
                p = (repo_dir / sub / te_f) if sub else (repo_dir / te_f)
                if p.exists():
                    return True
            if te.get("id") == "bf16" and (repo_dir / "qwen3vl_8b_bf16.safetensors").exists():
                return True
        t_repo = (te.get("source") or {}).get("repo_id") or te.get("repo")
        if t_repo:
            found = cache.find_existing_file(t_repo, te_f)
            if found and found.is_file():
                return True
            sec_dir = cache.repo_dir_for(t_repo)
            for sub in ("", "text_encoders", "text_encoder"):
                p = (sec_dir / sub / te_f) if sub else (sec_dir / te_f)
                if p.exists():
                    return True
        return False

    def _repo_file_installed(repo_id: str, file_path: str, cache: CacheManager) -> bool:
        if not repo_id or not file_path:
            return False
        try:
            found = cache.find_existing_file(repo_id, file_path)
            return bool(found and found.is_file())
        except Exception:
            return False

    def _is_aux_installed(aux: dict, primary_repo: str, cache: CacheManager) -> bool:
        if not isinstance(aux, dict):
            return False
        return _repo_file_installed(aux.get("repo_id") or primary_repo, str(aux.get("path") or ""), cache)

    def _transformer_installed(model, variant, cache: CacheManager) -> bool:
        try:
            repo_id = getattr(variant, "repo_id", None) or (model.sources.primary.repo_id if model.sources and model.sources.primary else "")
            v_path = getattr(variant, "path", None)
            if not repo_id or not v_path:
                return False
            found = cache.find_existing_file(repo_id, v_path)
            if found and found.is_file():
                return True
            if getattr(model, "is_custom", False):
                from backend.custom_models import get_custom_models_dir
                custom_dir = get_custom_models_dir()
                p = Path(v_path)
                if p.is_file() or (custom_dir / v_path).is_file():
                    return True
            return False
        except Exception:
            return False



    def _model_is_installed(model, cache: CacheManager, gpu_tier: str) -> bool:



        try:



            has_hf_source = bool(model.sources and model.sources.primary and model.sources.primary.repo_id)



            if getattr(model, "is_custom", False) and not has_hf_source:



                from backend.custom_models import get_custom_models_dir



                custom_dir = get_custom_models_dir()



                ckpt_rel = model.engine_args.get("checkpoint") if getattr(model, "engine_args", None) else None



                if ckpt_rel:



                    p = Path(ckpt_rel)



                    return p.exists() or (custom_dir / ckpt_rel).exists()



                diff_rel = model.engine_args.get("diffusion_model") if getattr(model, "engine_args", None) else None



                if diff_rel:



                    p = Path(diff_rel)



                    return p.exists() or (custom_dir / diff_rel).exists()



                for t in model.transformers:



                    p = Path(t.path)



                    if p.exists() or (custom_dir / t.path).exists():



                        return True



                return False







            if getattr(model, "engine", "quantfunc") == "sd_cpp":



                repo_id = model.sources.primary.repo_id if model.sources else ""



                repo_dir = cache.repo_dir_for(repo_id) if repo_id else get_settings().models_dir



                ckpt_rel = model.engine_args.get("checkpoint")



                if ckpt_rel and (repo_dir / ckpt_rel).exists():



                    return True



                diff_rel = model.engine_args.get("diffusion_model")



                diff_ok = False



                if diff_rel and (repo_dir / diff_rel).exists():



                    diff_ok = True



                else:



                    for t in model.transformers:



                        t_repo = getattr(t, "repo_id", None)



                        if (repo_dir / t.path).exists() or (t_repo and (cache.repo_dir_for(t_repo) / t.path).exists()):



                            diff_ok = True



                            break



                if not diff_ok:



                    return False



                if getattr(model, "text_encoders", None):



                    if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):



                        return False



                elif model.engine_args.get("llm"):



                    llm_rel = model.engine_args["llm"]



                    if not ((repo_dir / llm_rel).exists() or (repo_dir / "text_encoders" / llm_rel).exists()):



                        return False



                return True







            if getattr(model, "engine", "quantfunc") == "sdnq":
                repo_id = model.sources.primary.repo_id if model.sources else ""
                repo_dir = cache.repo_dir_for(repo_id) if repo_id else get_settings().models_dir
                if not (repo_dir / "model_index.json").exists():
                    return False
                vae_ok = (repo_dir / "vae" / "diffusion_pytorch_model.safetensors").exists() or bool(cache.find_existing_file(repo_id, "vae/diffusion_pytorch_model.safetensors"))
                if not vae_ok:
                    return False
                if getattr(model, "text_encoders", None):
                    if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):
                        return False
                return any(_transformer_installed(model, t, cache) for t in getattr(model, "transformers", []))

            if getattr(model, "engine", "quantfunc") == "orbitquant":
                repo_id = model.sources.primary.repo_id if model.sources else ""
                repo_dir = cache.repo_dir_for(repo_id) if repo_id else get_settings().models_dir
                has_model = (repo_dir / "model.safetensors").exists()
                has_tr = any(_transformer_installed(model, t, cache) for t in getattr(model, "transformers", []))
                if not (has_model or has_tr):
                    return False
                if getattr(model, "text_encoders", None):
                    if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):
                        return False
                return True

            if getattr(model, "engine", "quantfunc") == "nunchaku":
                repo_id = model.sources.primary.repo_id if model.sources else ""
                repo_dir = cache.repo_dir_for(repo_id) if repo_id else get_settings().models_dir
                has_tr = any(_transformer_installed(model, t, cache) for t in getattr(model, "transformers", []))
                if not has_tr:
                    return False
                if getattr(model, "text_encoders", None):
                    if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):
                        return False
                return True







            # QuantFunc engine models



            if not model.sources:



                return False



            base_path = model.get_base_model_path(gpu_tier) if gpu_tier != "unknown" else None



            if not base_path and model.base_models:



                base_path = list(model.base_models.values())[0]



            if base_path:



                if base_path in ("", "."):



                    local_base = cache.repo_dir_for(model.sources.primary.repo_id)



                else:



                    local_base = cache.repo_dir_for(model.sources.primary.repo_id) / base_path



                has_diffusers_index = (local_base / "model_index.json").exists()

                # Imported folder-layout releases (OrbitQuant/SVDQ) often ship component
                # configs but no root model_index.json. Treat them as installed when the
                # transformer and VAE configs are present.
                has_component_layout = (
                    (local_base / "transformer" / "config.json").exists()
                    and (local_base / "vae" / "config.json").exists()
                )
                if not has_diffusers_index and not has_component_layout:

                    return False



            if getattr(model, "text_encoders", None):



                repo_dir = cache.repo_dir_for(model.sources.primary.repo_id)



                if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):



                    return False



            primary_dir = cache.repo_dir_for(model.sources.primary.repo_id)



            for t in model.transformers:



                if primary_dir and (primary_dir / t.path).exists():



                    return True



                t_repo = getattr(t, "repo_id", None)



                if t_repo and (cache.repo_dir_for(t_repo) / t.path).exists():



                    return True



            return False



        except Exception:



            return False







    # ---------- API: Config ----------



    @app.get("/api/config")



    async def api_config(request: Request = None):



        """Initial config for frontend."""



        settings = get_settings()



        registry = get_registry()



        cache = CacheManager()



        gpu_tier = STATE.gpu_info.tier if STATE.gpu_info else "unknown"







        caller = {"role": "host", "id": "host", "username": "Хост", "is_guest": False}



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



        is_guest = caller.get("is_guest", False)







        from backend.model_visibility import load_visibility_map



        vis_map = load_visibility_map()







        # Convert models to JSON-serializable list



        models_list = []



        qf_enabled = getattr(settings, "enable_quantfunc_models", False)



        for m in registry.list_all():



            vinfo = vis_map.get(m.id, {})



            hidden_for_guests = vinfo.get("hidden_for_guests", False)



            hidden_from_host = vinfo.get("hidden_from_host", False)







            if is_guest and (hidden_for_guests or hidden_from_host):



                continue







            repo_id = m.sources.primary.repo_id if m.sources else ""



            repo_dir = cache.repo_dir_for(repo_id) if repo_id else None



            default_cfg = m.get_default_cfg()



            qf_disabled = quantfunc_model_disabled(m, qf_enabled)



            models_list.append({



                "id": m.id,



                "name": m.name,



                "family": m.family,
                "target_architecture": m.get_target_architecture() if hasattr(m, "get_target_architecture") else getattr(m, "target_architecture", m.family),



                "pipeline": m.pipeline,
                "supports_mask": model_supports_mask(m),



                "engine": getattr(m, "engine", "quantfunc"),



                "engine_version": m.engine_version,



                "is_custom": getattr(m, "is_custom", False),



                "local_folder": getattr(m, "local_folder", "") or getattr(m, "engine_args", {}).get("local_folder", ""),



                "base_repo_id": getattr(m, "base_repo_id", ""),



                "primary_repo_id": repo_id,



                "vae": vae_payload(m, repo_id, cache),


                "vae_files": vae_payload_list(m, repo_id, cache),



                "aux_files": [



                    {



                        "path": aux.get("path") if isinstance(aux, dict) else str(aux),



                        "repo_id": (aux.get("repo_id") if isinstance(aux, dict) else "") or repo_id,



                        "size_gb": float(aux.get("size_gb") or 0) if isinstance(aux, dict) else 0.0,



                        "installed": _is_aux_installed(aux, repo_id, cache),



                    }



                    for aux in ((getattr(m, "engine_args", {}) or {}).get("aux_files") or [])



                ],



                "model_path": getattr(m, "model_path", "") or getattr(m, "engine_args", {}).get("diffusion_model") or getattr(m, "engine_args", {}).get("checkpoint") or (m.transformers[0].path if m.transformers else ""),



                "installed": _model_is_installed(m, cache, gpu_tier),



                "disabled": qf_disabled,



                "disabled_reason": "Временно не поддерживается (сырые веса upstream с полосами). Включите в Настройках." if qf_disabled else None,



                "description": m.description,



                "key_protected": bool(m.key_protected),



                "default_cfg": default_cfg,



                "hidden_for_guests": hidden_for_guests,



                "hidden_from_host": hidden_from_host,



                "requirements": {



                    "vram_min_gb": m.requirements.vram_min_gb,



                    "vram_recommended_gb": m.requirements.vram_recommended_gb,



                    "gpu_min_sm": m.requirements.gpu_min_sm,



                    "default_vram_mode": m.requirements.default_vram_mode,



                },



                "text_encoders": [



                    {



                        "id": te.get("id"),



                        "name": te.get("name"),



                        "file": te.get("file"),



                        "path": te.get("path") or te.get("file"),



                        "repo_id": (te.get("source") or {}).get("repo_id") or te.get("repo") or repo_id,



                        "size_gb": te.get("size_gb"),



                        "precision": te.get("precision"),



                        "recommended": te.get("recommended", False),



                        "installed": _is_te_installed(te, repo_dir, cache),



                    }



                    for te in getattr(m, "text_encoders", [])



                ],



                "transformers": [



                    {



                        "id": t.id,



                        "backend": t.backend,



                        "steps": t.steps,



                        "size_gb": t.size_gb,



                        "recommended": t.recommended,



                        "note": t.note,



                        "installed": _transformer_installed(m, t, cache),



                    }



                    for t in m.transformers



                ],



            })







        models_list.sort(key=lambda item: (



            1 if item.get("disabled") else 0,



            0 if item.get("installed") else 1,



            (item.get("name") or "").lower(),



        ))







        installed_engines = list_installed_engine_versions()



        if is_sd_cpp_installed():



            installed_engines.append("sd_cpp")



        if is_sdnq_installed():



            installed_engines.append("sdnq")







        settings_dict = {



            "lang": settings.lang,



            "vram_mode": settings.vram_mode,



            "auto_optimize": settings.auto_optimize,



            "models_dir": "[Скрыто]" if is_guest else str(settings.models_dir),



            "output_dir": "[Скрыто]" if is_guest else str(settings.output_dir),



            "engine_version": settings.engine_version,



            "engine_platform": settings.engine_platform,



            "cuda_version": settings.cuda_version,



            "qwen_vae_type": getattr(settings, "qwen_vae_type", "texture_fix"),



            "enable_quantfunc_models": qf_enabled,



            "hide_uninstalled_in_generator": getattr(settings, "hide_uninstalled_in_generator", False),



            "monitor_connections": bool(getattr(settings, "monitor_connections", True)),



        }



        if not is_guest:



            settings_dict.update({



                "huggingface_token": settings.huggingface_token or "",



                "modelscope_token": settings.modelscope_token or "",



                "quantfunc_api_key": settings.quantfunc_api_key or "",



                "download_source_priority": settings.download_source_priority,
                "hf_transfer": bool(getattr(settings, "hf_transfer", False)),
                "hf_fast_backend": hf_fast_backend()[0],
                "hf_transfer_available": hf_fast_backend()[1],
                "hf_endpoint": getattr(settings, "hf_endpoint", None) or "",



                "auto_update_engine": settings.auto_update_engine,



                "share_token": settings.share_token or "",



                "share_auto_start": getattr(settings, "share_auto_start", False),



                "share_allow_admin": getattr(settings, "share_allow_admin", False),



                "extras_python": settings.extras_python or "",



            })







        return {



            "account": caller,



            "settings": settings_dict,



            "gpu": {



                "name": STATE.gpu_info.name,



                "sm": STATE.gpu_info.sm,



                "vram_mb": STATE.gpu_info.vram_mb,



                "driver_version": STATE.gpu_info.driver_version,



                "tier": STATE.gpu_info.tier,



                "klein_tier": STATE.gpu_info.klein_tier,



            } if STATE.gpu_info else None,



            "engine_status": {



                "status": check_engine_status().status,



                "message": check_engine_status().message,



            },



            "installed_engines": installed_engines,



            "engines": _get_engines_info(),



            "sd_cpp_installed": is_sd_cpp_installed(),



            "sd_cpp_version": get_sd_cpp_version(),



            "sdnq_installed": is_sdnq_installed(),



            "sdnq_version": get_sdnq_version(),



            "api_auth": _api_auth_status() if not is_guest else {"key_available": False, "source": "guest"},



            "system_stats": get_system_memory_stats(),



            "models": models_list,



            "presence": get_presence().snapshot(include_users=not is_guest),



        }







    # ---------- API: Connected users / presence ----------


    @app.get("/api/presence")


    async def api_presence(request: Request = None):


        caller = _resolve_caller(request)


        return get_presence().snapshot(include_users=not caller.get("is_guest", False))


    @app.post("/api/presence/heartbeat")


    async def api_presence_heartbeat(request: Request = None):


        caller = _resolve_caller(request)


        presence = get_presence()


        presence.touch(caller)


        return presence.snapshot(include_users=not caller.get("is_guest", False))


    # ---------- API: Live System Stats (VRAM & RAM) ----------



    @app.get("/api/system/stats")



    async def api_system_stats():



        return get_system_memory_stats()







    # ---------- API: Open local directory / file in OS file manager ----------



    @app.post("/api/system/open-folder")



    @app.post("/api/open-folder")



    async def api_open_folder(request: Request, body: dict = None):



        """Open specified local directory in Windows Explorer (or OS file manager).







        Payload options:



          - folder_type: "outputs" (default) | "models" | "loras" | "custom" | "presets" | "root"



          - filename: optional filename inside target directory (if found, will be selected/highlighted)



          - path: optional explicit relative or absolute path (validated against allowed root directories)



        """



        body = body or {}



        folder_type = str(body.get("folder_type") or "outputs").lower()



        filename = body.get("filename")



        explicit_path = body.get("path")



        settings = get_settings()







        # Check if caller is remote tunnel guest



        from backend.share import is_remote_scope



        if is_remote_scope(request.scope):



            return {



                "status": "remote",



                "message": "Remote session — cannot open local file manager on server.",



                "path": str(settings.output_dir),



            }







        target_dir: Path



        if explicit_path:



            p = Path(explicit_path).resolve()



            allowed_roots = [



                PROJECT_ROOT.resolve(),



                settings.output_dir.resolve(),



                settings.models_dir.resolve(),



                settings.cache_dir.resolve(),



            ]



            if not any(p == r or r in p.parents for r in allowed_roots):



                raise HTTPException(400, "Path is outside allowed directories")



            target_dir = p



        else:



            if folder_type in ("outputs", "output", "download", "downloads"):



                target_dir = settings.output_dir



            elif folder_type in ("models", "model", "cache"):



                target_dir = settings.models_dir



            elif folder_type in ("loras", "lora"):



                target_dir = settings.models_dir / "loras"



            elif folder_type == "custom":



                target_dir = PROJECT_ROOT / "models" / "custom"



            elif folder_type in ("presets", "preset"):



                target_dir = PROJECT_ROOT / "models" / "presets"



            elif folder_type == "root":



                target_dir = PROJECT_ROOT



            else:



                target_dir = settings.output_dir







        target_dir.mkdir(parents=True, exist_ok=True)



        target_file: Optional[Path] = None



        if filename:



            from backend.security import validate_filename



            try:



                safe_fname = validate_filename(filename)



                candidate = target_dir / safe_fname



                if candidate.exists() and candidate.is_file():



                    target_file = candidate



            except Exception:



                target_file = None







        import subprocess



        opened = False



        err_msg = None



        try:



            if sys.platform == "win32":



                if target_file and target_file.exists():



                    subprocess.Popen(["explorer", f"/select,{str(target_file.resolve())}"])



                else:



                    os.startfile(str(target_dir.resolve()))



                opened = True



            elif sys.platform == "darwin":



                if target_file and target_file.exists():



                    subprocess.Popen(["open", "-R", str(target_file.resolve())])



                else:



                    subprocess.Popen(["open", str(target_dir.resolve())])



                opened = True



            else:



                dir_to_open = target_dir if target_dir.is_dir() else target_dir.parent



                subprocess.Popen(["xdg-open", str(dir_to_open.resolve())])



                opened = True



        except Exception as e:



            opened = False



            err_msg = str(e)



            print(f"[system] Failed to open folder {target_dir}: {e}")







        return {



            "status": "ok" if opened else "error",



            "message": err_msg,



            "path": str((target_file or target_dir).resolve()),



            "folder": str(target_dir.resolve()),



            "selected_file": target_file.name if target_file else None,



        }







    @app.get("/api/system/open-folder")



    @app.get("/api/open-folder")



    async def api_open_folder_get(



        request: Request,



        folder_type: Optional[str] = Query("outputs"),



        filename: Optional[str] = Query(None),



        path: Optional[str] = Query(None),



    ):



        return await api_open_folder(



            request,



            body={"folder_type": folder_type, "filename": filename, "path": path},



        )







    # ---------- API: Models ----------



    @app.get("/api/models")



    async def api_models(request: Request = None):



        registry = get_registry()



        cache = CacheManager()



        gpu_tier = STATE.gpu_info.tier if STATE.gpu_info else "unknown"







        caller = {"role": "host", "id": "host", "username": "Хост", "is_guest": False}



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



        is_guest = caller.get("is_guest", False)







        from backend.model_visibility import load_visibility_map



        vis_map = load_visibility_map()







        models_out = []



        settings = get_settings()



        qf_enabled = getattr(settings, "enable_quantfunc_models", False)



        for m in registry.list_all():



            vinfo = vis_map.get(m.id, {})



            hidden_for_guests = vinfo.get("hidden_for_guests", False)



            hidden_from_host = vinfo.get("hidden_from_host", False)







            if is_guest and (hidden_for_guests or hidden_from_host):



                continue







            repo_id = m.sources.primary.repo_id if m.sources else ""



            repo_dir = cache.repo_dir_for(repo_id) if repo_id else None



            default_cfg = m.get_default_cfg()



            qf_disabled = quantfunc_model_disabled(m, qf_enabled)



            models_out.append({



                "id": m.id,



                "name": m.name,



                "family": m.family,
                "target_architecture": m.get_target_architecture() if hasattr(m, "get_target_architecture") else getattr(m, "target_architecture", m.family),



                "pipeline": m.pipeline,
                "supports_mask": model_supports_mask(m),



                "engine": getattr(m, "engine", "quantfunc"),



                "engine_version": m.engine_version,



                "is_custom": getattr(m, "is_custom", False),



                "local_folder": getattr(m, "local_folder", "") or getattr(m, "engine_args", {}).get("local_folder", ""),



                "base_repo_id": getattr(m, "base_repo_id", ""),



                "primary_repo_id": repo_id,



                "vae": vae_payload(m, repo_id, cache),


                "vae_files": vae_payload_list(m, repo_id, cache),



                "aux_files": [



                    {



                        "path": aux.get("path") if isinstance(aux, dict) else str(aux),



                        "repo_id": (aux.get("repo_id") if isinstance(aux, dict) else "") or repo_id,



                        "size_gb": float(aux.get("size_gb") or 0) if isinstance(aux, dict) else 0.0,



                        "installed": _is_aux_installed(aux, repo_id, cache),



                    }



                    for aux in ((getattr(m, "engine_args", {}) or {}).get("aux_files") or [])



                ],



                "model_path": getattr(m, "model_path", "") or getattr(m, "engine_args", {}).get("diffusion_model") or getattr(m, "engine_args", {}).get("checkpoint") or (m.transformers[0].path if m.transformers else ""),



                "installed": _model_is_installed(m, cache, gpu_tier),



                "disabled": qf_disabled,



                "disabled_reason": "Временно не поддерживается (сырые веса upstream с полосами). Включите в Настройках." if qf_disabled else None,



                "description": m.description,



                "key_protected": bool(m.key_protected),



                "default_cfg": default_cfg,



                "hidden_for_guests": hidden_for_guests,



                "hidden_from_host": hidden_from_host,



                "requirements": {



                    "vram_min_gb": m.requirements.vram_min_gb,



                    "vram_recommended_gb": m.requirements.vram_recommended_gb,



                    "gpu_min_sm": m.requirements.gpu_min_sm,



                },



                "text_encoders": [



                    {



                        "id": te.get("id"),



                        "name": te.get("name"),



                        "file": te.get("file"),



                        "path": te.get("path") or te.get("file"),



                        "repo_id": (te.get("source") or {}).get("repo_id") or te.get("repo") or repo_id,



                        "size_gb": te.get("size_gb"),



                        "precision": te.get("precision"),



                        "recommended": te.get("recommended", False),



                        "installed": _is_te_installed(te, repo_dir, cache),



                    }



                    for te in getattr(m, "text_encoders", [])



                ],



                "transformers": [



                    {



                        "id": t.id,



                        "backend": t.backend,



                        "steps": t.steps,



                        "size_gb": t.size_gb,



                        "recommended": t.recommended,



                        "note": t.note,



                        "installed": _transformer_installed(m, t, cache),



                    }



                    for t in m.transformers



                ],



            })



        models_out.sort(key=lambda item: (



            1 if item.get("disabled") else 0,



            0 if item.get("installed") else 1,



            (item.get("name") or "").lower(),



        ))



        return models_out







    @app.get("/api/models/{model_id}/card")


    async def api_model_card(model_id: str, request: Request = None):


        """Return editable manifest data for a registry model card (host only)."""


        if request:


            from backend.accounts import resolve_account_from_scope


            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)


            if caller.get("is_guest", False):


                raise HTTPException(403, "Guests cannot edit model cards")


        from backend.custom_models import get_custom_model_card


        try:


            return get_custom_model_card(model_id)


        except ValueError as e:


            raise HTTPException(404, str(e))


    @app.patch("/api/models/{model_id}/card")


    async def api_update_model_card(model_id: str, body: dict, request: Request = None):


        """Update a model card and keep engine_args in sync (host only)."""


        if request:


            from backend.accounts import resolve_account_from_scope


            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)


            if caller.get("is_guest", False):


                raise HTTPException(403, "Guests cannot edit model cards")


        from backend.custom_models import update_custom_model_card


        try:


            manifest = update_custom_model_card(model_id, body or {})


            return {"status": "ok", "model": manifest}


        except ValueError as e:


            raise HTTPException(400, str(e))


    @app.get("/api/models/{model_id}/state")



    async def api_model_state(model_id: str, request: Request = None):



        """Check cache state for a model's required files."""



        if request:



            from backend.accounts import resolve_account_from_scope



            from backend.model_visibility import get_model_visibility



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                vinfo = get_model_visibility(model_id)



                if vinfo.get("hidden_for_guests", False) or vinfo.get("hidden_from_host", False):



                    raise HTTPException(404, f"Model not found: {model_id}")







        registry = get_registry()



        try:



            model = registry.get(model_id)



        except KeyError:



            raise HTTPException(404, f"Model not found: {model_id}")







        # Determine GPU tier (or fall back to showing all transformers)



        if STATE.gpu_info:



            gpu_sm = STATE.gpu_info.sm



            gpu_tier = STATE.gpu_info.tier



            transformers_for_gpu = model.get_transformers_for_tier(gpu_sm)



        else:



            # No GPU — show all transformers (UI-only, generation will fail later)



            gpu_sm = 0



            gpu_tier = "unknown"



            transformers_for_gpu = model.transformers







        has_hf_source = bool(model.sources and model.sources.primary and model.sources.primary.repo_id)



        if getattr(model, "is_custom", False) and not has_hf_source:



            from backend.custom_models import get_custom_models_dir



            custom_dir = get_custom_models_dir()



            result = []



            for t in model.transformers:



                p = Path(t.path)



                f_exists = p.exists() or (custom_dir / t.path).exists()



                result.append({



                    "id": t.id,



                    "backend": t.backend,



                    "steps": t.steps,



                    "prequant": t.prequant,



                    "size_gb": t.size_gb,



                    "tier": t.tier,



                    "klein_tier": t.klein_tier,



                    "recommended": t.recommended,



                    "recommended_for_gpu": t.recommended_for_gpu,



                    "state": "cached" if f_exists else "missing",



                    "state_label": "Local ✓" if f_exists else "Missing",



                })



            te_result = []



            for te in getattr(model, "text_encoders", []):



                p = Path(te.get("file", ""))



                te_exists = p.exists() or (custom_dir / te.get("file", "")).exists()



                te_result.append({



                    "id": te.get("id"),



                    "name": te.get("name"),



                    "file": te.get("file"),



                    "path": te.get("path") or te.get("file"),



                    "repo_id": (te.get("source") or {}).get("repo_id") or te.get("repo") or "",



                    "size_gb": te.get("size_gb"),



                    "precision": te.get("precision"),



                    "recommended": te.get("recommended", False),



                    "installed": te_exists,



                })



            custom_engine_args = getattr(model, "engine_args", {}) or {}



            vae_rel = str(custom_engine_args.get("vae") or custom_engine_args.get("vae_file") or "")



            vae_installed = bool(vae_rel and (Path(vae_rel).is_file() or (custom_dir / vae_rel).is_file()))



            aux_result = []



            for aux in custom_engine_args.get("aux_files") or []:



                aux_path = str((aux.get("path") if isinstance(aux, dict) else aux) or "")



                aux_result.append({



                    "path": aux_path,



                    "repo_id": (aux.get("repo_id") if isinstance(aux, dict) else "") or "",



                    "size_gb": float(aux.get("size_gb") or 0) if isinstance(aux, dict) else 0.0,



                    "installed": bool(aux_path and (Path(aux_path).is_file() or (custom_dir / aux_path).is_file())),



                })



            return {



                "model_id": model_id,



                "model_name": model.name,



                "gpu_tier": gpu_tier,



                "gpu_sm": gpu_sm,



                "base_model_installed": True,



                "vae": ({



                    "repo_id": custom_engine_args.get("vae_repo") or "",



                    "path": vae_rel,



                    "installed": vae_installed,



                } if vae_rel else None),



                "aux_files": aux_result,



                "text_encoders": te_result,



                "transformers": result,



            }







        # Check cache state for each



        cache = CacheManager()



        source_spec = model.sources.primary



        repo_id = source_spec.repo_id



        local_repo_dir = cache.repo_dir_for(repo_id)







        result = []



        for t in transformers_for_gpu:



            t_repo = getattr(t, "repo_id", None) or repo_id
            use_repo = t_repo if (cache.find_existing_file(t_repo, t.path) or (cache.repo_dir_for(t_repo) / t.path).exists()) else repo_id



            entry = cache.check_file(



                repo_id=use_repo,



                file_path=t.path,



                expected_size=t.size_bytes or 0,



                expected_sha256=t.sha256,



                compute_checksum=False,  # Fast check on first load



            )



            is_t_installed = entry.state.value == "installed"



            result.append({



                "id": t.id,



                "backend": t.backend,



                "steps": t.steps,



                "prequant": t.prequant,



                "size_gb": t.size_gb,



                "tier": t.tier,



                "klein_tier": t.klein_tier,



                "recommended": t.recommended,



                "recommended_for_gpu": t.recommended_for_gpu,



                "path": t.path,



                "repo_id": getattr(t, "repo_id", None) or repo_id,



                "state": entry.state.value,



                "state_label": entry.state_label,



                "installed": is_t_installed,



            })







        # Also check base model dir



        base_path = model.get_base_model_path(gpu_tier) if gpu_tier != "unknown" else None



        if not base_path and model.base_models:



            # Fall back to first available base model



            base_path = list(model.base_models.values())[0]



        base_installed = False



        eng_type = getattr(model, "engine", "quantfunc")



        if eng_type == "sd_cpp" or not model.base_models:



            base_installed = True



        elif eng_type in ("sdnq", "nunchaku", "orbitquant"):



            local_base = local_repo_dir if (not base_path or base_path in ("", ".")) else (local_repo_dir / base_path)



            has_cfg = (



                (local_base / "model_index.json").exists()



                or (local_base / "modular_model_index.json").exists()



                or (local_base / "config.json").exists()



            )



            has_vae = (



                (local_base / "vae" / "diffusion_pytorch_model.safetensors").exists()



                or (local_base / "vae" / "diffusion_pytorch_model.bin").exists()



            )



            base_installed = has_cfg and has_vae



        elif base_path:



            local_base = local_repo_dir if base_path in ("", ".") else (local_repo_dir / base_path)



            base_installed = (local_base / "model_index.json").exists()







        # Check text encoders state if model specifies them



        te_result = []



        known_te_files = set()



        for te in getattr(model, "text_encoders", []):



            te_f = te.get("file")



            known_te_files.add(te_f)



            known_te_files.add(Path(te_f).name)



            installed = _is_te_installed(te, local_repo_dir, cache)



            te_result.append({



                "id": te.get("id"),



                "name": te.get("name"),



                "file": te_f,



                "path": te.get("path") or te_f,



                "repo_id": (te.get("source") or {}).get("repo_id") or te.get("repo") or repo_id,



                "size_gb": te.get("size_gb"),



                "precision": te.get("precision"),



                "recommended": te.get("recommended", False),



                "installed": installed,



            })







        # Dynamically discover any other custom/local encoders in the directory (skip for sdnq pipeline folders)



        if getattr(model, "engine", "") != "sdnq" and (getattr(model, "text_encoders", None) or model.family == "qwen" or getattr(model, "engine", "") == "sd_cpp"):



            known_files = set(known_te_files)



            known_files.add("model_index.json")



            for tr in getattr(model, "transformers", []):



                known_files.add(tr.path)



                known_files.add(Path(tr.path).name)







            scan_dirs = [local_repo_dir, local_repo_dir / "text_encoders"]



            seen_extra = set()



            for sdir in scan_dirs:



                if sdir and sdir.exists():



                    for pattern in ("*.gguf", "*.safetensors"):



                        for f in sdir.glob(pattern):



                            fname = f.name



                            if fname not in known_files and fname not in seen_extra:



                                # Skip known non-encoders (VAE or transformer files)



                                if "vae" in fname.lower() or ("base" in fname.lower() and "quantfunc" in fname.lower()):



                                    continue



                                seen_extra.add(fname)



                                sz_gb = round(f.stat().st_size / (1024**3), 2)



                                te_result.append({



                                    "id": fname,



                                    "name": f"{fname} (~{sz_gb} GB, Local)",



                                    "file": fname,



                                    "size_gb": sz_gb,



                                    "precision": "custom",



                                    "recommended": False,



                                    "installed": True,



                                })







        return {



            "model_id": model_id,



            "model_name": model.name,



            "gpu_tier": gpu_tier,



            "gpu_sm": gpu_sm,



            "base_model_installed": base_installed,



            "vae": vae_payload(model, repo_id, cache),


            "vae_files": vae_payload_list(model, repo_id, cache),



            "aux_files": [



                {



                    "path": aux.get("path") if isinstance(aux, dict) else str(aux),



                    "repo_id": (aux.get("repo_id") if isinstance(aux, dict) else "") or repo_id,



                    "size_gb": float(aux.get("size_gb") or 0) if isinstance(aux, dict) else 0.0,



                    "installed": _is_aux_installed(aux, repo_id, cache),



                }



                for aux in ((getattr(model, "engine_args", {}) or {}).get("aux_files") or [])



            ],



            "text_encoders": te_result,



            "transformers": result,



        }







    # ---------- API: Custom Models ----------



    @app.get("/api/custom-models/files")



    async def api_custom_models_files(request: Request = None):



        """List local files available in models/custom/ directory."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot view custom model files")



        from backend.custom_models import list_custom_files, get_custom_models_dir



        return {



            "custom_dir": str(get_custom_models_dir()),



            "files": list_custom_files(),



        }







    @app.post("/api/custom-models")



    async def api_register_custom_model(body: dict, request: Request = None):



        """Register a new custom model profile."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot register models")



        from backend.custom_models import register_custom_model



        try:



            res = register_custom_model(body)



            return {"ok": True, "model": res}



        except Exception as e:



            raise HTTPException(400, str(e))







    @app.post("/api/custom-models/sniff")



    async def api_sniff_custom_model(body: dict, request: Request = None):



        """Sniff a local folder or Hugging Face repo: returns detected engines, components, folders, and warnings."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot sniff models")



        from backend.model_sniffer import scan_local_path



        from backend.custom_models import inspect_hf_repo



        path = (body.get("path") or "").strip()



        repo_id = (body.get("repo_id") or "").strip()



        try:



            if repo_id:



                info = inspect_hf_repo(repo_id)



                return {"ok": True, **info}



            if path:



                info = scan_local_path(path)



                return {"ok": True, **info}



            raise HTTPException(400, "Specify 'path' (local) or 'repo_id' (Hugging Face)")



        except Exception as e:



            raise HTTPException(400, str(e))







    @app.delete("/api/custom-models/{model_id}")



    async def api_delete_custom_model(model_id: str, delete_files: bool = False, request: Request = None):



        """Delete an imported custom model profile."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot delete models")



        from backend.custom_models import delete_custom_model



        try:



            ok = delete_custom_model(model_id, delete_files=delete_files)



            if not ok:



                raise HTTPException(404, "Custom model manifest not found")



            return {"ok": True}



        except Exception as e:



            raise HTTPException(400, str(e))







    # ---------- API: Hugging Face Hub Search & Import ----------



    @app.get("/api/hf/search")



    async def api_hf_search(q: str = "flux gguf", limit: int = 12):



        """Search Hugging Face Hub for GGUF / diffusion models."""



        from backend.custom_models import search_hf_models



        try:



            results = search_hf_models(q, limit=limit)



            return {"ok": True, "results": results}



        except Exception as e:



            raise HTTPException(400, f"Ошибка поиска Hugging Face: {e}")







    @app.get("/api/hf/repo-tree")



    async def api_hf_repo_tree(repo_id: Optional[str] = None, repo: Optional[str] = None):



        """Inspect weight files (.gguf / .safetensors) in a Hugging Face repository."""



        target_repo = repo_id or repo



        if not target_repo:



            raise HTTPException(400, "repo_id is required")



        from backend.custom_models import inspect_hf_repo



        try:



            info = inspect_hf_repo(target_repo)



            return {"ok": True, **info}



        except Exception as e:



            raise HTTPException(400, str(e))







    @app.post("/api/hf/register")



    async def api_hf_register_model(body: dict, request: Request = None):



        """Register a downloadable model profile from a Hugging Face repository."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot register models")



        from backend.custom_models import register_hf_custom_model



        try:



            manifest = register_hf_custom_model(body)



            return {"ok": True, "model": manifest}



        except Exception as e:



            raise HTTPException(400, str(e))







    # ---------- API: Model Visibility (Host Control) ----------



    @app.get("/api/models/visibility")



    async def api_get_models_visibility(request: Request = None):



        """Get visibility settings for models."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot view model visibility")



        from backend.model_visibility import load_visibility_map



        return {"visibility": load_visibility_map()}







    @app.post("/api/models/{model_id}/visibility")



    async def api_set_model_visibility(model_id: str, body: dict, request: Request = None):



        """Update visibility settings for a model (hidden for guests / hidden from host)."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot change model visibility")



        from backend.model_visibility import set_model_visibility



        hidden_for_guests = bool(body.get("hidden_for_guests", False))



        hidden_from_host = bool(body.get("hidden_from_host", False))



        res = set_model_visibility(model_id, hidden_for_guests=hidden_for_guests, hidden_from_host=hidden_from_host)



        return {"ok": True, "model_id": model_id, "visibility": res}



    @app.post("/api/models/{model_id}/open-folder")

    async def api_model_open_folder(model_id: str, request: Request = None):

        """Open the specific directory for model_id in Windows Explorer (or OS file manager)."""

        from backend.share import is_remote_scope

        if request and is_remote_scope(request.scope):

            return {"status": "remote", "message": "Удалённая сессия: открытие папки на сервере недоступно."}



        registry = get_registry()

        model = registry.get(model_id)

        if not model:

            raise HTTPException(404, f"Model '{model_id}' not found")



        cache = CacheManager()

        target_dir = None

        # 1. Check local_folder on the model (custom or standard)
        loc_folder = getattr(model, "local_folder", "") or getattr(model, "engine_args", {}).get("local_folder", "")
        if loc_folder:
            p = cache.cache_dir / loc_folder
            if p.exists():
                target_dir = p
            elif Path(loc_folder).is_absolute() and Path(loc_folder).exists():
                target_dir = Path(loc_folder)

        # 2. Check engine diffusion_model or checkpoint file
        if not target_dir:
            diff_model = getattr(model, "engine_args", {}).get("diffusion_model") or getattr(model, "engine_args", {}).get("checkpoint")
            if diff_model:
                p = Path(diff_model)
                if p.is_absolute() and p.exists():
                    target_dir = p.parent if p.is_file() else p
                elif (cache.cache_dir / diff_model).exists():
                    p2 = cache.cache_dir / diff_model
                    target_dir = p2.parent if p2.is_file() else p2

        # 3. Check transformer variant file paths on disk
        if not target_dir and model.transformers:
            for t in model.transformers:
                tp = Path(t.path)
                if tp.is_absolute() and tp.exists():
                    target_dir = tp.parent
                    break
                p1 = cache.cache_dir / t.path
                if p1.exists():
                    target_dir = p1.parent
                    break

        # 4. Check model_path
        if not target_dir and getattr(model, "model_path", ""):
            mp = Path(model.model_path)
            if mp.is_absolute() and mp.exists():
                target_dir = mp.parent
            elif (cache.cache_dir / model.model_path).exists():
                target_dir = (cache.cache_dir / model.model_path).parent

        # 5. Check primary source repo dir
        if not target_dir and model.sources and model.sources.primary and model.sources.primary.repo_id:
            cand = cache.repo_dir_for(model.sources.primary.repo_id)
            if cand.exists():
                target_dir = cand
            else:
                target_dir = cand

        if not target_dir or not target_dir.exists():
            target_dir = cache.cache_dir



        target_dir.mkdir(parents=True, exist_ok=True)

        try:

            if sys.platform.startswith("win"):

                os.startfile(str(target_dir.resolve()))

            elif sys.platform == "darwin":

                subprocess.Popen(["open", str(target_dir.resolve())])

            else:

                subprocess.Popen(["xdg-open", str(target_dir.resolve())])

            return {"status": "ok", "path": str(target_dir.resolve())}

        except Exception as e:

            raise HTTPException(500, f"Failed to open folder: {e}")


    @app.delete("/api/models/{model_id}/files")

    async def api_model_delete_files(model_id: str, variant_id: Optional[str] = None, request: Request = None):

        """Delete downloaded weights for one variant (or the whole model) without touching shared files.

        The endpoint removes only exact files referenced by the model manifest. Repositories are
        never removed recursively: a repo may also hold shared encoders, VAEs, auxiliary sidecars
        or variants belonging to other models.
        """

        if request:

            from backend.accounts import resolve_account_from_scope

            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)

            if caller.get("is_guest", False):

                raise HTTPException(403, "Гостевой режим: удаление моделей защищено хостом.")



        registry = get_registry()

        try:

            model = registry.get(model_id)

        except KeyError:

            model = None

        if not model:

            raise HTTPException(404, f"Model '{model_id}' not found")



        from backend.model_files import collect_delete_targets, delete_targets

        cache = CacheManager()
        primary_repo = model.sources.primary.repo_id if model.sources and model.sources.primary else ""
        if not primary_repo:
            raise HTTPException(400, "У модели нет источника Hugging Face — удалять нечего")

        try:
            targets, protected, skipped_shared = collect_delete_targets(
                model, registry.list_all(), cache, variant_id
            )
        except KeyError:
            raise HTTPException(404, f"Variant '{variant_id}' not found for model '{model_id}'")

        selected_variant = next(
            (variant for variant in getattr(model, "transformers", []) or []
             if getattr(variant, "id", None) == variant_id),
            None,
        ) if variant_id else None
        result = delete_targets(cache, targets)

        if not result["deleted_files"]:
            message = ("Файлы используются другой моделью — удаление пропущено" if skipped_shared
                       else "Файлы не найдены — возможно, веса уже удалены")
        elif selected_variant:
            message = f"Веса варианта «{selected_variant.id}» удалены"
        else:
            message = "Веса модели удалены"

        return {
            "status": "ok",
            "model_id": model_id,
            "variant_id": variant_id,
            **result,
            "protected_shared_count": len(protected),
            "skipped_shared": sorted(skipped_shared),
            "message": message,
        }


    @app.post("/api/models/{model_id}/component-delete")
    async def api_model_component_delete(model_id: str, body: dict, request: Request = None):
        """Delete one downloaded VAE, text encoder, or Aux/sidecar component.

        Shared files are protected by default. The response ``shared`` status lets
        the UI show which other models depend on the file; a second explicit
        request with ``force=true`` is required to remove it anyway.
        """
        if request:
            from backend.accounts import resolve_account_from_scope
            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)
            if caller.get("is_guest", False):
                raise HTTPException(403, "Guests cannot delete model components")

        body = body or {}
        role = str(body.get("role") or "").strip().lower()
        if role not in ("text_encoder", "vae", "aux"):
            raise HTTPException(400, "role must be text_encoder, vae, or aux")
        force = bool(body.get("force", False))

        registry = get_registry()
        try:
            model = registry.get(model_id)
        except KeyError:
            model = None
        if not model:
            raise HTTPException(404, f"Model '{model_id}' not found")

        primary_repo = model.sources.primary.repo_id if (model.sources and model.sources.primary) else ""
        from backend.model_files import (
            aux_component_refs,
            delete_targets,
            encoder_component_refs,
            split_shared_refs,
            vae_component_refs,
        )

        refs: list[tuple[str, str]] = []
        cache = CacheManager()
        if role == "text_encoder":
            encoders = getattr(model, "text_encoders", []) or []
            enc_id = str(body.get("id") or "").strip()
            index = int(body.get("index") or 0)
            encoder = next((te for te in encoders if enc_id and str(te.get("id")) == enc_id), None)
            if encoder is None and 0 <= index < len(encoders):
                encoder = encoders[index]
            if encoder is None:
                raise HTTPException(404, "Text encoder not found in this model card")
            refs = encoder_component_refs(encoder, primary_repo)
        elif role == "vae":
            refs = vae_component_refs(getattr(model, "engine_args", {}) or {}, primary_repo, cache)
        else:
            refs = aux_component_refs(
                getattr(model, "engine_args", {}) or {},
                primary_repo,
                int(body.get("index") or 0),
            )

        if not refs:
            raise HTTPException(404, "Component is not declared in this model card")

        safe, shared = split_shared_refs(model, registry.list_all(), refs)
        if shared and not force:
            return {
                "status": "shared",
                "role": role,
                "shared": [f"{repo}/{path}" for repo, path in shared],
                "message": "Файл используется другими моделями. Подтвердите удаление ещё раз.",
            }

        result = delete_targets(cache, refs if force else safe)
        if not result["deleted_files"]:
            return {
                "status": "not_found",
                "role": role,
                "shared": [f"{repo}/{path}" for repo, path in shared],
                **result,
                "message": "Файл не найден — возможно, он уже удалён",
            }
        return {
            "status": "ok",
            "role": role,
            "shared": [f"{repo}/{path}" for repo, path in shared],
            **result,
            "message": "Компонент удалён из локального кэша",
        }


    # ---------- API: Presets & Templates ----------



    @app.get("/api/presets")



    async def api_list_presets(request: Request = None):



        """Return generation presets filtered by caller permissions."""



        from backend.presets import list_presets



        from backend.accounts import resolve_account_from_scope



        caller = (request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)) if request else None



        return {"presets": list_presets(caller)}







    @app.post("/api/presets")



    async def api_save_preset(body: dict, request: Request = None):



        """Save a new or updated preset."""



        from backend.presets import save_user_preset



        from backend.accounts import resolve_account_from_scope



        caller = (request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)) if request else None



        try:



            preset = save_user_preset(body, caller)



            return {"ok": True, "preset": preset}



        except PermissionError as pe:



            raise HTTPException(403, str(pe))



        except Exception as e:



            raise HTTPException(400, str(e))







    @app.delete("/api/presets/{preset_id}")



    async def api_delete_preset(preset_id: str, request: Request = None):



        """Delete a preset by ID."""



        from backend.presets import delete_user_preset



        from backend.accounts import resolve_account_from_scope



        caller = (request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)) if request else None



        try:



            ok = delete_user_preset(preset_id, caller)



            if not ok:



                raise HTTPException(404, "Шаблон не найден")



            return {"ok": True}



        except PermissionError as pe:



            raise HTTPException(403, str(pe))







    @app.post("/api/models/{model_id}/component-download")


    async def api_model_component_download(model_id: str, body: dict, request: Request = None):


        """Download exactly one repository-local component (TE / VAE / aux) for a model."""


        if request:


            from backend.accounts import resolve_account_from_scope


            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)


            if caller.get("is_guest", False):


                raise HTTPException(403, "Guests cannot trigger downloads")


        registry = get_registry()


        try:


            model = registry.get(model_id)


        except KeyError:


            raise HTTPException(404, f"Model not found: {model_id}")


        role = str((body or {}).get("role") or "").strip().lower()


        if role not in ("text_encoder", "vae", "aux"):


            raise HTTPException(400, "role must be text_encoder, vae or aux")


        primary_repo = ""


        if model.sources and model.sources.primary:


            primary_repo = model.sources.primary.repo_id or ""


        repo_id = ""
        file_path = ""
        local_repo_id = ""
        local_path = ""
        if role == "text_encoder":


            te = model.get_text_encoder((body or {}).get("id"))


            if not te:


                raise HTTPException(404, "Text encoder not found in model card")


            source = te.get("source") or {}


            repo_id = source.get("repo_id") or te.get("repo") or primary_repo


            file_path = source.get("file_path") or te.get("path") or te.get("file") or ""


        elif role == "vae":


            refs = vae_download_refs(model, primary_repo)


            if not refs:


                raise HTTPException(400, "VAE has no downloadable repository/path binding")


            repo_id = refs["source_repo"]


            file_path = refs["source_path"]


            local_repo_id = refs["local_repo"]


            local_path = refs["local_path"]


        else:


            args = getattr(model, "engine_args", {}) or {}


            aux_list = args.get("aux_files") or []


            try:
                index = int((body or {}).get("index") or 0)
            except (TypeError, ValueError):
                raise HTTPException(400, "index must be an integer")


            if index < 0 or index >= len(aux_list):


                raise HTTPException(404, "Aux file not found in model card")


            aux = aux_list[index]


            if isinstance(aux, dict):


                repo_id = aux.get("repo_id") or primary_repo


                file_path = aux.get("path") or ""


            else:


                repo_id = primary_repo


                file_path = str(aux)


        file_path = str(file_path or "").replace("\\", "/").strip()
        parts = [part for part in file_path.split("/") if part not in ("", ".")]
        if (
            not repo_id
            or not parts
            or file_path.startswith("/")
            or any(part == ".." for part in parts)
            or ":" in parts[0]
        ):
            raise HTTPException(400, "Component has no safe repository/path binding")
        file_path = "/".join(parts)

        local_repo_id = local_repo_id or repo_id
        local_path = str(local_path or file_path).replace("\\", "/").strip()
        local_parts = [part for part in local_path.split("/") if part not in ("", ".")]
        if (
            not local_repo_id
            or not local_parts
            or local_path.startswith("/")
            or any(part == ".." for part in local_parts)
            or ":" in local_parts[0]
        ):
            raise HTTPException(400, "Component has no safe local destination")
        local_path = "/".join(local_parts)


        cache = CacheManager()


        destination = cache.repo_dir_for(local_repo_id) / local_path


        source_cached = cache.repo_dir_for(repo_id) / file_path


        if destination.is_file():


            return {
                "status": "already_installed",
                "repo_id": repo_id,
                "path": file_path,
                "local_repo_id": local_repo_id,
                "local_path": local_path,
            }


        job_id = str(uuid.uuid4())[:8]


        STATE.register_job(job_id)


        source_type = "huggingface"


        if model.sources and model.sources.primary and model.sources.primary.type:


            source_type = model.sources.primary.type


        def run_component_download():


            try:


                from backend.sources import get_source


                STATE.update_job(job_id, ProgressEvent(


                    status=JobStatus.DOWNLOADING, progress=0.05,


                    message=f"Downloading {role}: {file_path}",


                ))


                source = get_source(source_type, token=_get_token_for_source(source_type))


                local_dir = cache.repo_dir_for(local_repo_id)


                if source_cached.is_file() and source_cached.resolve() != destination.resolve():


                    import shutil


                    destination.parent.mkdir(parents=True, exist_ok=True)


                    shutil.copy2(source_cached, destination)


                else:


                    def _on_comp_progress(dl_b, total_b, speed):
                        pct = (dl_b / float(total_b)) if total_b > 0 else 0.0
                        speed_mb = (speed or 0.0) / (1024 * 1024)
                        speed_str = f" ({speed_mb:.1f} MB/s)" if speed_mb >= 0.1 else ""
                        STATE.update_job(job_id, ProgressEvent(
                            status=JobStatus.DOWNLOADING, progress=pct,
                            downloaded_bytes=dl_b, total_bytes=total_b, speed_bps=speed,
                            message=f"Downloading {role}: {file_path}{speed_str}",
                        ))

                    downloaded = Path(source.download_file(
                        repo_id, file_path, local_dir, progress_callback=_on_comp_progress, job_id=job_id
                    ))


                    if role == "vae" and downloaded.resolve() != destination.resolve() and not destination.is_file():


                        import shutil


                        destination.parent.mkdir(parents=True, exist_ok=True)


                        shutil.copy2(downloaded, destination)


                STATE.update_job(job_id, ProgressEvent(


                    status=JobStatus.COMPLETE, progress=1.0,


                    message=f"Downloaded {role}: {file_path}",


                ))


            except Exception as exc:


                STATE.update_job(job_id, ProgressEvent(


                    status=JobStatus.ERROR, progress=0.0,


                    message=f"Download failed: {exc}",


                ))


        threading.Thread(target=run_component_download, daemon=True).start()


        return {
            "status": "started",
            "job_id": job_id,
            "repo_id": repo_id,
            "path": file_path,
            "local_repo_id": local_repo_id,
            "local_path": local_path,
        }


    @app.post("/api/models/{model_id}/download")



    async def api_model_download(model_id: str, body: Optional[dict] = None, request: Request = None):



        """Download missing files for a model. Runs in background thread."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot trigger model downloads")







        if not STATE.gpu_info:



            raise HTTPException(400, "GPU not detected")







        registry = get_registry()



        try:



            model = registry.get(model_id)



        except KeyError:



            raise HTTPException(404, f"Model not found: {model_id}")







        if not model.sources or not model.sources.primary:



            raise HTTPException(400, "Эта модель является локальной и не требует скачивания из сети.")







        settings = get_settings()



        if quantfunc_model_disabled(
            model, bool(getattr(settings, "enable_quantfunc_models", False))
        ):



            raise HTTPException(400, "Эта QuantFunc-модель временно отключена из-за дефектов весов (FLUX Klein). Включите QuantFunc-модели в Настройках, если хотите протестировать.")







        # Determine which variant to download



        body = body or {}



        variant_id = body.get("variant_id")



        include_base = body.get("include_base", True)



        include_precision = body.get("include_precision", True)



        include_transformer = body.get("include_transformer", True)







        gpu_sm = STATE.gpu_info.sm if STATE.gpu_info else 0



        gpu_tier = STATE.gpu_info.tier if STATE.gpu_info else "50x-below"



        transformers_for_gpu = model.get_transformers_for_tier(gpu_sm) if gpu_sm else model.transformers



        selected = None



        if variant_id:



            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)



        if not selected and transformers_for_gpu:



            # Pick first recommended, or first available



            selected = next((t for t in transformers_for_gpu if t.recommended), None) or transformers_for_gpu[0]



        if include_transformer and not selected and not body.get("text_encoder_id"):



            raise HTTPException(400, f"No compatible transformer found for GPU SM={gpu_sm}")







        # Create a download job (reuse the job system)



        job_id = str(uuid.uuid4())[:8]



        STATE.register_job(job_id)







        def run_download():



            """Background download thread. ALWAYS updates job state, even on crash."""



            try:



                # Imports inside try so import errors get caught and reported to UI



                from backend.sources import get_source



                from backend.cache import CacheManager







                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.DOWNLOADING,



                    progress=0.0,



                    message=f"Preparing to download {model.name} files...",



                ))







                settings = get_settings()



                source_spec = model.sources.primary



                if settings.download_source_priority == "hf_first" and model.sources.mirror and model.sources.mirror.type == "huggingface":



                    source_spec = model.sources.mirror



                elif settings.download_source_priority == "ms_first" and model.sources.mirror and model.sources.mirror.type == "modelscope":



                    source_spec = model.sources.mirror



                source = get_source(source_spec.type, token=_get_token_for_source(source_spec.type))



                repo_id = source_spec.repo_id



                revision = "master" if source_spec.type == "modelscope" else "main"



                cache = CacheManager()



                local_repo_dir = cache.repo_dir_for(repo_id)



                local_repo_dir.mkdir(parents=True, exist_ok=True)







                tasks = []



                if include_base and getattr(model, "engine", "quantfunc") != "sd_cpp" and model.base_models:



                    base_path = model.get_base_model_path(gpu_tier)



                    if base_path:



                        local_base = local_repo_dir if base_path in ("", ".") else (local_repo_dir / base_path)



                        eng = getattr(model, "engine", "quantfunc")



                        is_svdq_or_nunchaku = eng in ("nunchaku", "orbitquant", "sdnq")







                        # For SVDQ/Nunchaku/OrbitQuant/SDNQ: check for configs + VAE



                        if is_svdq_or_nunchaku:



                            has_cfg = (



                                (local_base / "model_index.json").exists()



                                or (local_base / "modular_model_index.json").exists()



                                or (local_base / "config.json").exists()



                            )



                            has_vae = (



                                (local_base / "vae" / "diffusion_pytorch_model.safetensors").exists()



                                or (local_base / "vae" / "diffusion_pytorch_model.bin").exists()



                            )



                            base_ready = has_cfg and has_vae



                        else:



                            base_ready = (local_base / "model_index.json").exists()







                        if not base_ready:



                            sub = base_path.rstrip("/")



                            base_target_repo = getattr(model, "base_repo_id", None) or repo_id



                            if is_svdq_or_nunchaku:



                                # Whitelist ONLY configs, scheduler, tokenizer/processor, and VAE.



                                # NEVER download unquantized 9-16 GB text_encoder/** or transformer/** weights!



                                patterns = [



                                    "model_index.json", "modular_model_index.json", "config.json", "configuration.json",



                                    "*.json", "*/*.json", "*/*.jinja", "*/*.txt",



                                    "scheduler/**", "tokenizer/**", "tokenizer*/**", "processor/**",



                                    "vae/config.json", "vae/*.json", "vae/diffusion_pytorch_model.safetensors",

                                    # Imported folder releases may ship their own pipeline script/fonts.

                                    # They are downloaded for completeness and never executed automatically.

                                    *(["*.py", "fonts/**"] if getattr(model, "is_custom", False) else []),



                                ]



                                tasks.append(("base", ".", patterns, base_target_repo))



                            elif sub in ("", "."):



                                patterns = [



                                    "model_index.json", "config.json", "*.json",



                                    "text_encoder/**", "text_encoder*/**",



                                    "vae/**", "scheduler/**", "tokenizer/**", "tokenizer*/**", "processor/**",



                                ]



                                tasks.append(("base", ".", patterns, base_target_repo))



                            else:



                                tasks.append(("base", sub, [f"{sub}/**"], base_target_repo))



                        else:



                            STATE.update_job(job_id, ProgressEvent(



                                status=JobStatus.DOWNLOADING, progress=0.1,



                                message="Base model already installed ✓",



                            ))



                    else:



                        raise RuntimeError(f"No base model for tier {gpu_tier}")







                # Transformer



                if include_transformer and selected:



                    local_transformer = local_repo_dir / selected.path



                    if not local_transformer.exists():



                        t_repo = getattr(selected, "repo_id", None) or repo_id



                        tasks.append(("transformer", selected.path, selected.path, t_repo))



                    else:



                        STATE.update_job(job_id, ProgressEvent(



                            status=JobStatus.DOWNLOADING, progress=0.5,



                            message="Transformer already installed ✓",



                        ))







                # Precision config



                if include_precision:



                    klein_tier = STATE.gpu_info.klein_tier if STATE.gpu_info else None



                    for pc in model.precision_configs:



                        if pc.tier == gpu_tier:



                            if pc.klein_tier and klein_tier and pc.klein_tier != klein_tier:



                                continue



                            local_pc = local_repo_dir / pc.path



                            if not local_pc.exists():



                                tasks.append(("precision", pc.path, pc.path))



                            break







                # Text encoder (if model specifies text_encoders)



                if getattr(model, "text_encoders", None):



                    te_id = body.get("text_encoder_id")



                    te_spec = model.get_text_encoder(te_id)



                    if te_spec:



                        te_f = te_spec.get("file") or te_spec.get("path")



                        if te_f:



                            te_source = te_spec.get("source") or {}



                            te_repo = te_source.get("repo_id") or te_spec.get("repo") or repo_id



                            sec_dir = cache.repo_dir_for(te_repo) if te_repo else None



                            is_te_already_downloaded = (



                                (local_repo_dir / te_f).exists()



                                or (local_repo_dir / "text_encoders" / Path(te_f).name).exists()



                                or (sec_dir and ((sec_dir / te_f).exists() or (sec_dir / "text_encoders" / Path(te_f).name).exists()))



                            )



                            if not is_te_already_downloaded:



                                te_remote = te_source.get("file_path") or (te_f if "/" in te_f else f"text_encoders/{te_f}")



                                tasks.append(("text_encoder", te_remote, te_remote, te_repo, te_f))







                # VAE (if model specifies engine_args.vae)



                eng_args = getattr(model, "engine_args", None) or {}



                vae_rel = eng_args.get("vae")



                if vae_rel:



                    local_vae = local_repo_dir / vae_rel



                    existing_vae = next((



                        p for p in [



                            settings.models_dir / "Qwen-Image-2.1-4bit" / vae_rel,



                            settings.models_dir / "Qwen-Image-2.1-4bit" / "vae" / Path(vae_rel).name,



                        ] if p.exists() and p.is_file()



                    ), None)



                    if not local_vae.exists():



                        if existing_vae:



                            import shutil



                            local_vae.parent.mkdir(parents=True, exist_ok=True)



                            shutil.copy2(existing_vae, local_vae)



                            print(f"[download] Reused existing VAE from {existing_vae} -> {local_vae}")



                        else:



                            vae_repo = eng_args.get("vae_repo", repo_id)



                            vae_remote = eng_args.get("vae_remote") or eng_args.get("vae_file") or (vae_rel if "/" in vae_rel else f"vae/{vae_rel}")



                            tasks.append(("vae", vae_remote, vae_remote, vae_repo, vae_rel))







                # Tokenizer (if model specifies engine_args.tokenizer_repo)



                tok_rel = eng_args.get("tokenizer")



                tok_repo = eng_args.get("tokenizer_repo")



                tok_remote = eng_args.get("tokenizer_remote")



                if tok_rel and tok_repo and tok_remote:



                    local_tok = local_repo_dir / tok_rel



                    if not local_tok.exists():



                        tasks.append(("tokenizer", tok_remote, tok_remote, tok_repo, tok_rel))







                # Auxiliary quant sidecars (OrbitQuant codebooks/rotations, SVDQ metadata).

                # They are small but the transformer cannot be reconstructed without them.

                for aux in eng_args.get("aux_files") or []:

                    if isinstance(aux, dict):

                        aux_rel = str(aux.get("path") or "").replace("\\", "/").lstrip("/")

                        aux_repo = aux.get("repo_id") or repo_id

                    else:

                        aux_rel = str(aux or "").replace("\\", "/").lstrip("/")

                        aux_repo = repo_id

                    if not aux_rel:

                        continue

                    local_aux = local_repo_dir / aux_rel

                    if not local_aux.exists():

                        tasks.append(("aux", aux_rel, aux_rel, aux_repo))



                total_tasks = len(tasks)



                if total_tasks == 0:



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.COMPLETE,



                        progress=1.0,



                        message="All files already installed ✓",



                    ))



                    return







                for i, item in enumerate(tasks):



                    kind = item[0]



                    file_path = item[1]



                    pattern = item[2]



                    progress_base = i / total_tasks



                    progress_end = (i + 1) / total_tasks



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.DOWNLOADING,



                        progress=progress_base,



                        message=f"[{i+1}/{total_tasks}] Downloading {kind}: {file_path}...",



                    ))







                    if kind == "base":



                        # Snapshot download with allow_pattern for recursive dir



                        allow_pat = pattern if isinstance(pattern, list) else [pattern]



                        base_repo = item[3] if len(item) > 3 and item[3] else repo_id



                        base_src = get_source("huggingface", token=_get_token_for_source("huggingface")) if base_repo != repo_id else source



                        base_src.download_snapshot(



                            repo_id=base_repo,



                            local_dir=local_repo_dir,



                            allow_patterns=allow_pat,



                            revision="main" if base_repo != repo_id else revision,



                        )



                    elif kind in ("text_encoder", "vae", "tokenizer"):



                        te_repo = item[3] if len(item) > 3 else repo_id



                        te_fname = item[4] if len(item) > 4 else file_path



                        te_src = get_source("huggingface", token=_get_token_for_source("huggingface"))



                        # Store in dedicated shared folder for this repo so multiple models by the author share it without copying!



                        target_dest_dir = cache.repo_dir_for(te_repo) if (te_repo and te_repo != repo_id) else local_repo_dir



                        target_dest_dir.mkdir(parents=True, exist_ok=True)



                        def _on_te_progress(dl_b, total_b, speed):
                            item_pct = (dl_b / float(total_b)) if total_b > 0 else 0.0
                            overall = progress_base + item_pct * (progress_end - progress_base)
                            speed_mb = (speed or 0.0) / (1024 * 1024)
                            speed_str = f" ({speed_mb:.1f} MB/s)" if speed_mb >= 0.1 else ""
                            STATE.update_job(job_id, ProgressEvent(
                                status=JobStatus.DOWNLOADING, progress=overall,
                                downloaded_bytes=dl_b, total_bytes=total_b, speed_bps=speed,
                                message=f"[{i+1}/{total_tasks}] Downloading {kind}: {file_path}{speed_str}",
                            ))

                        te_src.download_file(
                            repo_id=te_repo,
                            file_path=file_path,
                            local_dir=target_dest_dir,
                            revision="main",
                            progress_callback=_on_te_progress,
                            job_id=job_id,
                        )



                        dl_file = target_dest_dir / file_path



                        target_rel_file = target_dest_dir / te_fname



                        if dl_file.exists() and not target_rel_file.exists() and dl_file.resolve() != target_rel_file.resolve():



                            import shutil



                            target_rel_file.parent.mkdir(parents=True, exist_ok=True)



                            shutil.move(str(dl_file), str(target_rel_file))



                    else:



                        # Single file download (preserves subfolder hierarchy inside local_repo_dir)



                        target_repo = item[3] if len(item) > 3 else repo_id



                        target_source = get_source("huggingface", token=_get_token_for_source("huggingface")) if len(item) > 3 else source



                        def _on_target_progress(dl_b, total_b, speed):
                            item_pct = (dl_b / float(total_b)) if total_b > 0 else 0.0
                            overall = progress_base + item_pct * (progress_end - progress_base)
                            speed_mb = (speed or 0.0) / (1024 * 1024)
                            speed_str = f" ({speed_mb:.1f} MB/s)" if speed_mb >= 0.1 else ""
                            STATE.update_job(job_id, ProgressEvent(
                                status=JobStatus.DOWNLOADING, progress=overall,
                                downloaded_bytes=dl_b, total_bytes=total_b, speed_bps=speed,
                                message=f"[{i+1}/{total_tasks}] Downloading {kind}: {file_path}{speed_str}",
                            ))

                        target_source.download_file(
                            repo_id=target_repo,
                            file_path=file_path,
                            local_dir=local_repo_dir,
                            revision=revision,
                            progress_callback=_on_target_progress,
                            job_id=job_id,
                        )







                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.DOWNLOADING,



                        progress=progress_end,



                        message=f"[{i+1}/{total_tasks}] {kind} downloaded ✓",



                    ))







                if model.id == "qwen_image_2_1":



                    try:



                        get_worker()._ensure_qwen_image_2_1_configs(local_repo_dir)



                    except Exception as e:



                        print(f"[download] Warning ensuring Qwen 2.1 configs: {e}")







                if getattr(model, "family", "") == "moss-image" and (local_repo_dir / selected.path).suffix.lower() == ".gguf":



                    try:



                        from backend.sd_cpp import _ensure_moss_gguf_qkv_fused



                        _ensure_moss_gguf_qkv_fused(local_repo_dir / selected.path)



                    except Exception as e:



                        print(f"[download] Warning fusing MOSS-Image QKV tensors: {e}")







                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.COMPLETE,



                    progress=1.0,



                    message=f"All {total_tasks} files downloaded ✓",



                ))







            except Exception as e:



                # CRITICAL: Always update job state on failure, otherwise



                # frontend will poll forever waiting for a status that never comes.



                import traceback



                error_detail = f"{type(e).__name__}: {e}"



                print(f"[download] Job {job_id} failed: {error_detail}")



                traceback.print_exc()



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.ERROR,



                    error=error_detail,



                    message=f"Download failed: {error_detail}",



                ))







        thread = threading.Thread(target=run_download, daemon=True)



        thread.start()







        return {"job_id": job_id, "status": "downloading", "variant_id": selected.id if selected else None}







    @app.get("/api/encoders")



    async def api_list_encoders():



        """List all text encoders across all registered models with installation status."""



        cache = CacheManager()



        registry = get_registry()



        models = registry.list_all()







        encoders_map = {}



        for m in models:



            te_list = getattr(m, "text_encoders", []) or []



            repo_id = m.sources.primary.repo_id if (m.sources and m.sources.primary) else ""



            m_repo_dir = cache.repo_dir_for(repo_id) if repo_id else None







            for te in te_list:



                key = te.get("file") or te.get("id")



                if not key:



                    continue



                source = te.get("source") or {}



                te_repo = source.get("repo_id") if isinstance(source, dict) else (te.get("repo") or repo_id)



                te_file = te.get("file") or te.get("path") or ""



                te_remote = source.get("file_path") if isinstance(source, dict) else te_file



                installed = _is_te_installed(te, m_repo_dir, cache)







                if key not in encoders_map:



                    encoders_map[key] = {



                        "id": te.get("id"),



                        "name": te.get("name"),



                        "file": te_file,



                        "repo_id": te_repo,



                        "file_path": te_remote,



                        "size_gb": te.get("size_gb"),



                        "precision": te.get("precision", "int4"),



                        "installed": installed,



                        "used_by": [],



                    }



                else:



                    if installed:



                        encoders_map[key]["installed"] = True







                encoders_map[key]["used_by"].append({



                    "id": m.id,



                    "name": m.name,



                    "family": m.family,
                    "target_architecture": m.get_target_architecture() if hasattr(m, "get_target_architecture") else getattr(m, "target_architecture", m.family),



                    "engine": getattr(m, "engine", "quantfunc"),



                })







        return list(encoders_map.values())


    @app.post("/api/encoders/delete")
    async def api_delete_encoder(body: dict, request: Request = None):
        """Delete a downloaded shared text encoder by repository + file reference.

        Encoders are shared across model cards, so the endpoint gathers every
        local copy referenced by matching cards. Files also used by an unrelated
        model (for example as an Aux/sidecar) require ``force=true``.
        """
        if request:
            from backend.accounts import resolve_account_from_scope
            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)
            if caller.get("is_guest", False):
                raise HTTPException(403, "Guests cannot delete encoders")

        body = body or {}
        repo_id = str(body.get("repo_id") or "").strip()
        file_path = str(body.get("file_path") or body.get("file") or "").strip()
        file_name = str(body.get("file") or "").strip()
        enc_id = str(body.get("id") or "").strip()
        force = bool(body.get("force", False))

        from backend.model_files import (
            delete_targets,
            encoder_component_refs,
            model_file_refs,
            safe_repo_rel,
        )

        refs: list[tuple[str, str]] = []
        for raw in (file_path, file_name):
            rel = safe_repo_rel(raw)
            if repo_id and rel:
                refs.append((repo_id, rel))

        registry = get_registry()
        matched_ids: set[str] = set()
        used_by: list[dict[str, str]] = []
        for model in registry.list_all():
            primary_repo = model.sources.primary.repo_id if (model.sources and model.sources.primary) else ""
            for encoder in getattr(model, "text_encoders", []) or []:
                source = encoder.get("source") if isinstance(encoder.get("source"), dict) else {}
                enc_repo = str(source.get("repo_id") or encoder.get("repo") or primary_repo or "")
                enc_file = str(encoder.get("file") or encoder.get("path") or "")
                enc_remote = str(source.get("file_path") or enc_file)
                enc_key = str(encoder.get("id") or "")
                if repo_id and enc_repo != repo_id:
                    continue
                if not (
                    (file_name and file_name in (enc_file, enc_remote))
                    or (file_path and file_path in (enc_file, enc_remote))
                    or (enc_id and enc_id == enc_key)
                ):
                    continue
                matched_ids.add(model.id)
                used_by.append({"id": model.id, "name": model.name})
                for ref in encoder_component_refs(encoder, primary_repo):
                    if ref not in refs:
                        refs.append(ref)

        if not refs:
            raise HTTPException(404, "Encoder reference is empty")

        protected: set[tuple[str, str]] = set()
        for model in registry.list_all():
            if model.id not in matched_ids:
                protected.update(model_file_refs(model))

        safe = [ref for ref in refs if ref not in protected]
        shared = [ref for ref in refs if ref in protected]
        if shared and not force:
            return {
                "status": "shared",
                "shared": [f"{repo}/{path}" for repo, path in shared],
                "used_by": used_by,
                "message": "Энкодер используется другими моделями. Подтвердите удаление ещё раз.",
            }

        cache = CacheManager()
        result = delete_targets(cache, refs if force else safe)
        if not result["deleted_files"]:
            return {
                "status": "not_found",
                "shared": [f"{repo}/{path}" for repo, path in shared],
                "used_by": used_by,
                **result,
                "message": "Файл энкодера не найден — возможно, он уже удалён",
            }
        return {
            "status": "ok",
            "shared": [f"{repo}/{path}" for repo, path in shared],
            "used_by": used_by,
            **result,
            "message": "Энкодер удалён из локального кэша",
        }







    @app.post("/api/encoders/download")



    async def api_download_encoder(body: dict, request: Request = None):



        """Download a shared text encoder into its shared repo directory."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot trigger downloads")







        repo_id = body.get("repo_id")



        file_path = body.get("file_path") or body.get("file")



        if not repo_id or not file_path:



            raise HTTPException(400, "repo_id and file_path are required")







        job_id = str(uuid.uuid4())[:8]



        STATE.register_job(job_id)







        def run_dl():



            try:



                from backend.sources import get_source



                cache = CacheManager()



                target_dir = cache.repo_dir_for(repo_id)



                target_dir.mkdir(parents=True, exist_ok=True)



                dest_file = target_dir / Path(file_path).name







                if dest_file.exists():



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.COMPLETE,



                        progress=1.0,



                        message="Text encoder already installed ✓",



                    ))



                    return







                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.DOWNLOADING,



                    progress=0.1,



                    message=f"Downloading text encoder {Path(file_path).name} from {repo_id}...",



                ))







                src = get_source("huggingface", token=_get_token_for_source("huggingface"))



                src.download_file(



                    repo_id=repo_id,



                    file_path=file_path,



                    local_dir=target_dir,



                    revision="main",



                )



                dl_file = target_dir / file_path



                if dl_file.exists() and not dest_file.exists() and dl_file.resolve() != dest_file.resolve():



                    import shutil



                    shutil.move(str(dl_file), str(dest_file))







                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.COMPLETE,



                    progress=1.0,



                    message=f"Text encoder {Path(file_path).name} installed ✓",



                ))



            except Exception as e:



                print(f"[download] Error downloading encoder {file_path}: {e}")



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.FAILED,



                    progress=0.0,



                    message=f"Download failed: {e}",



                ))







        threading.Thread(target=run_dl, daemon=True).start()



        return {"job_id": job_id, "status": "started"}







    # ---------- API: Upload reference image for Img2Img ----------



    @app.post("/api/upload")



    async def api_upload(file: UploadFile = File(...)):



        """Upload a reference image for Img2Img / Image Edit mode."""



        if not file.filename:



            raise HTTPException(400, "No file provided")







        ext = Path(file.filename).suffix.lower()



        if ext not in (".png", ".jpg", ".jpeg", ".webp"):



            raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: .png, .jpg, .jpeg, .webp")







        settings = get_settings()



        uploads_dir = settings.output_dir / "uploads"



        uploads_dir.mkdir(parents=True, exist_ok=True)







        from backend.security import validate_filename



        safe_stem = validate_filename(f"ref_{uuid.uuid4().hex[:8]}{ext}")



        target_path = uploads_dir / safe_stem







        try:



            content = await file.read()



            if len(content) > 25 * 1024 * 1024:



                raise HTTPException(400, "File too large. Max 25 MB.")







            with open(target_path, "wb") as f:



                f.write(content)



        except HTTPException:



            raise



        except Exception as e:



            raise HTTPException(500, f"Failed to save upload: {e}")







        rel_path = f"uploads/{safe_stem}"



        return {



            "status": "ok",



            "filename": safe_stem,



            "path": rel_path,



            "url": f"/outputs/uploads/{safe_stem}",



        }







    # ---------- API: LoRA library ----------



    @app.get("/api/loras")



    async def api_list_loras(model_id: Optional[str] = None):



        """List all discovered LoRA files and metadata, optionally with model compatibility."""



        from backend.lora import list_loras, check_lora_compatibility



        loras = list_loras()



        model = None



        if model_id:



            try:



                model = get_registry().get(model_id)



            except KeyError:



                model = None



        if model is not None:



            for item in loras:



                item["compatibility"] = check_lora_compatibility(item.get("name"), model)



        return {"loras": loras}







    @app.post("/api/loras/upload")



    async def api_upload_lora(file: UploadFile = File(...)):



        """Upload a LoRA file directly into the models_dir/loras library."""



        if not file.filename:



            raise HTTPException(400, "No file provided")



        from backend.lora import save_uploaded_lora



        try:



            res = save_uploaded_lora(file.filename, file.file)



            return {"status": "ok", "lora": res}



        except Exception as e:



            raise HTTPException(400, f"Upload failed: {e}")







    @app.post("/api/loras/import")



    async def api_import_lora(body: dict):



        """Import a LoRA file from a local path on the host computer."""



        path = body.get("path")



        if not path:



            raise HTTPException(400, "Missing 'path'")



        rename = body.get("rename")



        from backend.lora import import_local_lora



        try:



            res = import_local_lora(path, rename)



            return {"status": "ok", "lora": res}



        except Exception as e:



            raise HTTPException(400, f"Import failed: {e}")







    @app.get("/api/loras/hf-search")



    async def api_lora_hf_search(q: str = "flux lora"):



        """Search Hugging Face Hub for LoRAs."""



        from backend.lora import search_hf_loras



        return {"results": search_hf_loras(q)}







    @app.get("/api/loras/hf-inspect")



    async def api_lora_hf_inspect(repo_id: str):



        """Inspect a Hugging Face LoRA repo: candidate files, family, trigger words."""



        from backend.lora import inspect_hf_lora



        settings = get_settings()



        try:



            return inspect_hf_lora(repo_id, token=settings.huggingface_token)



        except Exception as e:



            raise HTTPException(400, str(e))







    @app.patch("/api/loras/{name}/metadata")



    async def api_update_lora_metadata(name: str, body: dict, request: Request = None):



        """Update host-configured architecture, trigger and explicit model bindings."""



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Guests cannot edit LoRA bindings")



        from backend.lora import update_lora_metadata



        try:



            result = update_lora_metadata(name, body or {})



            return {"status": "ok", **result}



        except Exception as e:



            raise HTTPException(400, str(e))







    @app.post("/api/loras/download")



    async def api_download_lora(body: dict):



        """Download a LoRA file from Hugging Face."""



        repo_id = body.get("repo_id")



        if not repo_id:



            raise HTTPException(400, "Missing 'repo_id'")



        filename = body.get("filename")



        rename = body.get("rename")



        family = body.get("family")



        architecture = body.get("architecture")



        trigger = body.get("trigger")



        compatible_models = body.get("compatible_models")



        engine = body.get("engine")



        target_modules = body.get("target_modules")



        scale_default = body.get("scale_default")



        settings = get_settings()



        from backend.lora import download_hf_lora



        try:



            res = download_hf_lora(



                repo_id,



                filename=filename,



                rename=rename,



                family=family,



                architecture=architecture,



                trigger=trigger,



                compatible_models=compatible_models,



                engine=engine,



                target_modules=target_modules,



                scale_default=scale_default,



                token=settings.huggingface_token,



            )



            return {"status": "ok", "lora": res}



        except Exception as e:



            raise HTTPException(400, f"Download failed: {e}")







    @app.delete("/api/loras/{name}")



    async def api_delete_lora(name: str):



        """Delete a LoRA file from the models_dir/loras library."""



        from backend.lora import delete_lora



        try:



            delete_lora(name)



            return {"status": "ok", "deleted": name}



        except Exception as e:



            raise HTTPException(400, f"Delete failed: {e}")







    # ---------- API: Generate ----------



    @app.post("/api/generate")



    async def api_generate(body: dict, request: Request = None):



        """Start a generation job. Returns job_id for polling."""



        # Resolve caller identity first



        caller = {"role": "host", "id": "host", "username": "Хост", "is_guest": False}



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)







        owner_id = caller.get("id", "host")



        owner_name = caller.get("username", "Хост")



        default_shared = not caller.get("is_guest", False)



        shared = bool(body.get("shared", default_shared))







        # Generation area (crop) parameters ------------------------------

        from backend.masking import (

            MASK_CROP_MIN_SIZE,

            MASK_CROP_MIN_SIZE_RANGE,

            MASK_CROP_MODES,

            MASK_CROP_PADDING,

            MASK_CROP_PADDING_RANGE,

        )



        # Parse request



        try:



            model_id = body["model_id"]



            prompt = body["prompt"]



            variant_id = body.get("variant_id")



            raw_seed = body.get("seed")



            seed = int(raw_seed) if raw_seed not in (None, "", "random") else None



            width = int(body.get("width", 1024))



            height = int(body.get("height", 1024))



            steps = int(body.get("steps", 4))



            backend = body.get("backend", "lighting")



            raw_init_path = body.get("init_image_path")



            init_image_strength = float(body.get("init_image_strength", 0.7))

            mask_present = "mask" in body or "mask_path" in body
            raw_mask = body.get("mask")
            raw_mask_path = body.get("mask_path")
            if mask_present:
                if "mask" in body and "mask_path" in body:
                    raise ValueError("Provide either mask or mask_path, not both")
                if "mask" in body and (
                    not isinstance(raw_mask, str)
                    or not raw_mask.startswith("data:image/")
                ):
                    raise ValueError("mask must be a non-empty image data URL")
                if "mask_path" in body and (
                    not isinstance(raw_mask_path, str) or not raw_mask_path.strip()
                ):
                    raise ValueError("mask_path must be a non-empty path")

            mask_data_url = raw_mask if mask_present and "mask" in body else None

            mask_strength = float(body.get("mask_strength", 1.0))

            mask_grow = int(body.get("mask_grow", 6))

            mask_blur = float(body.get("mask_blur", 0.0))

            raw_mask_invert = body.get("mask_invert", False)
            if mask_present and not isinstance(raw_mask_invert, bool):
                raise ValueError("mask_invert must be a boolean")
            mask_invert = raw_mask_invert if mask_present else bool(raw_mask_invert)

            # Generation area (crop) -------------------------------------

            mask_crop_mode = str(body.get("mask_crop_mode") or "auto").strip().lower()

            if mask_crop_mode not in MASK_CROP_MODES:

                raise ValueError(

                    "mask_crop_mode must be one of: " + ", ".join(MASK_CROP_MODES)

                )

            mask_crop_padding = _parse_int_in_range(

                body.get("mask_crop_padding", MASK_CROP_PADDING),

                "mask_crop_padding",

                MASK_CROP_PADDING_RANGE,

            )

            mask_crop_min_size = _parse_int_in_range(

                body.get("mask_crop_min_size", MASK_CROP_MIN_SIZE),

                "mask_crop_min_size",

                MASK_CROP_MIN_SIZE_RANGE,

            )

            mask_crop_box = _parse_mask_crop_box(body.get("mask_crop_box"))

            if mask_crop_mode == "manual" and mask_crop_box is None:

                raise ValueError(

                    "mask_crop_box is required when mask_crop_mode is 'manual'"

                )



        except (KeyError, TypeError, ValueError) as e:



            raise HTTPException(400, f"Invalid request: {e}")







        # Enforce model visibility for guests before checking GPU



        if caller.get("is_guest", False):



            from backend.model_visibility import get_model_visibility



            vinfo = get_model_visibility(model_id)



            if vinfo.get("hidden_for_guests", False) or vinfo.get("hidden_from_host", False):



                raise HTTPException(403, "Выбранная модель скрыта хостом и недоступна для использования.")







        if not STATE.gpu_info:



            raise HTTPException(400, "GPU not detected")







        # Normalize init_image_path



        init_image_path = None



        if raw_init_path:



            norm = str(raw_init_path).strip().replace("\\", "/")



            if norm.startswith("/outputs/"):



                norm = norm[len("/outputs/"):]



            elif norm.startswith("outputs/"):



                norm = norm[len("outputs/"):]



            elif norm.startswith("/"):



                norm = norm[1:]



            if norm:



                init_image_path = norm







        # Look up variant to determine backend + use_prequant



        registry = get_registry()



        try:



            model = registry.get(model_id)



        except KeyError:



            raise HTTPException(404, f"Model not found: {model_id}")







        model_engine = getattr(model, "engine", "quantfunc")



        if model_engine == "sd_cpp":



            if not is_sd_cpp_installed():



                raise HTTPException(400, "SD.CPP engine not installed. Go to Settings → Download SD.CPP.")



        elif model_engine == "sdnq":



            if not is_sdnq_installed():



                raise HTTPException(400, "SDNQ runtime not installed. Go to Settings → Install/Verify SDNQ.")



        elif model_engine == "orbitquant":



            from backend.extras import is_orbitquant_ready



            if not is_orbitquant_ready():



                raise HTTPException(400, "OrbitQuant runtime not installed. Go to Settings → Extras & Engines.")



        elif model_engine == "nunchaku":



            from backend.extras import is_nunchaku_ready



            if not is_nunchaku_ready():



                raise HTTPException(400, "Nunchaku runtime not installed. Go to Settings → Extras & Engines.")



        else:



            settings = get_settings()



            if quantfunc_model_disabled(
                model, bool(getattr(settings, "enable_quantfunc_models", False))
            ):



                raise HTTPException(400, "Эта QuantFunc-модель временно отключена из-за дефектов весов (FLUX Klein). Включите QuantFunc-модели в Настройках, если хотите протестировать.")



            if not (is_engine_installed(model.engine_version) or is_engine_installed()):



                raise HTTPException(400, "QuantFunc engine not installed. Go to Settings → Download engine.")







        # ----- Mask (inpaint) validation -----
        # The mask capability is checked BEFORE the job starts, so the user gets
        # a clear error (plus the list of compatible models) instead of a silent
        # full-image generation.
        mask_requested = mask_present
        if mask_requested:
            if not init_image_path:
                raise HTTPException(400, "Mask editing requires a source image. Load an image (gallery / upload / current canvas) first.")
            if not model_supports_mask(model):
                compatible = [m for m in registry.list_all() if model_supports_mask(m)]
                compatible_names = ", ".join(f"{m.name} ({m.id})" for m in compatible) or "none available yet"
                raise HTTPException(400, detail={
                    "error": "mask_unsupported",
                    "message": f"Model '{model.name}' does not support inpaint masks. Switch to a mask-capable model: {compatible_names}.",
                    "model_id": model.id,
                    "compatible_models": [
                        {"id": m.id, "name": m.name, "engine": getattr(m, "engine", "quantfunc")}
                        for m in compatible
                    ],
                })
            try:
                import math
                from backend.masking import MaskError, decode_mask_to_float, is_mask_empty
                from backend.security import validate_path

                if not all(math.isfinite(value) for value in (mask_strength, mask_blur)):
                    raise ValueError("Mask parameters must be finite numbers")
                if not 0.0 <= mask_strength <= 1.0:
                    raise ValueError("mask_strength must be between 0 and 1")
                if not 0.0 <= mask_blur <= 64.0:
                    raise ValueError("mask_blur must be between 0 and 64")
                if not 0 <= mask_grow <= 128:
                    raise ValueError("mask_grow must be between 0 and 128")

                mask_source = mask_data_url
                if raw_mask_path is not None:
                    validated_mask_path = validate_path(
                        raw_mask_path,
                        base_dir=Path(get_settings().output_dir),
                        must_exist=True,
                    )
                    raw_mask_path = str(validated_mask_path)
                    mask_source = validated_mask_path
                decoded_mask = decode_mask_to_float(
                    mask_source,
                    invert_mask=mask_invert,
                )
                if is_mask_empty(decoded_mask, threshold=0.0):
                    raise ValueError("Mask is empty")
            except (MaskError, ValueError, OSError) as exc:
                raise HTTPException(400, f"Invalid mask: {exc}") from exc
        if model.pipeline == "image_edit" and not init_image_path:



            raise HTTPException(400, "Reference image is required for image_edit models (upload an image or select one from preview/history)")







        # Find the selected transformer



        # `backend` (lighting/svdq) is a QuantFunc-engine concept. Nunchaku,
        # OrbitQuant, SD.CPP and SDNQ cards expose the ENGINE name as the
        # transformer backend, so the request backend must not filter them out
        # (the Inpaint studio sends no backend/variant_id and would otherwise
        # fail with "No compatible transformer found for backend=lighting").
        backend_filter = backend if model_engine == "quantfunc" else None
        transformers_for_gpu = model.get_transformers_for_tier(STATE.gpu_info.sm, backend=backend_filter)



        selected = None



        if variant_id:



            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)



            if not selected:



                selected = next((t for t in model.transformers if t.id == variant_id), None)



        if not selected and transformers_for_gpu:
            installed_for_gpu = [t for t in transformers_for_gpu if _transformer_installed(model, t, cache)]
            selected = installed_for_gpu[0] if installed_for_gpu else transformers_for_gpu[0]



        if not selected:



            raise HTTPException(400, f"No compatible transformer found for backend={backend}")







        # Determine steps and CFG



        effective_steps = int(body["steps"]) if "steps" in body and body["steps"] not in (None, "") else selected.steps



        cfg_val = model.get_default_cfg()



        if "cfg" in body and body["cfg"] not in (None, ""):



            try:



                cfg_val = float(body["cfg"])



            except (ValueError, TypeError):



                pass







        # Determine negative prompt, noise & sampler



        negative_prompt = body.get("negative_prompt", "") or ""



        noise_val = 0.0



        if "noise" in body and body["noise"] not in (None, ""):



            try:



                noise_val = float(body["noise"])



            except (ValueError, TypeError):



                pass







        raw_sampler = str(body.get("sampler") or "").strip().lower()



        allowed_samplers = {



            "euler", "euler_a", "heun", "dpm++2m", "dpm++2m_sde",



            "dpm2", "dpm2_a", "lcm", "unipc"



        }



        sampler = raw_sampler if raw_sampler in allowed_samplers else None







        raw_ref_mode = str(body.get("ref_mode") or "control_ref").strip().lower()



        ref_mode = raw_ref_mode if raw_ref_mode in ("control_ref", "controlnet_canny", "img2img") else "control_ref"



        control_net_path = (body.get("control_net_path") or "").strip() or None







        # Resolve LoRA stack



        raw_loras = body.get("loras") or []



        lora_paths: list[Path] = []



        lora_strengths: list[float] = []



        if raw_loras:



            from backend.lora import resolve_lora_selection



            try:



                lora_paths, lora_strengths, _ = resolve_lora_selection(raw_loras)



            except ValueError as e:



                raise HTTPException(400, f"LoRA error: {e}")



            from backend.lora import check_lora_compatibility



            for item in raw_loras:



                if not isinstance(item, dict):



                    continue



                compat = check_lora_compatibility(item.get("name"), model)



                if not compat.get("compatible", False):



                    raise HTTPException(400, f"LoRA «{item.get('name')}»: {compat.get('reason', 'несовместима')}")







        # Build GenerateRequest



        request = GenerateRequest(



            model_id=model_id,



            prompt=prompt,



            negative_prompt=negative_prompt,



            seed=seed,



            width=width,



            height=height,



            steps=effective_steps,



            cfg=cfg_val,



            noise=noise_val,



            backend=selected.backend,



            use_prequant=selected.prequant,



            init_image_path=Path(init_image_path) if init_image_path else None,



            init_image_strength=init_image_strength,
            mask_data_url=mask_data_url,
            mask_path=Path(str(raw_mask_path)) if raw_mask_path else None,
            mask_strength=mask_strength,
            mask_grow=mask_grow,
            mask_blur=mask_blur,
            mask_invert=mask_invert,

            mask_crop_mode=mask_crop_mode,

            mask_crop_padding=mask_crop_padding,

            mask_crop_min_size=mask_crop_min_size,

            mask_crop_box=mask_crop_box,



            lora_paths=lora_paths,



            lora_strengths=lora_strengths,



            variant_id=selected.id,



            sampler=sampler,



            text_encoder_id=body.get("text_encoder_id"),



            custom_text_encoder_path=body.get("custom_text_encoder_path"),



            vae=body.get("vae"),



            custom_vae_path=body.get("custom_vae_path"),



            ref_mode=ref_mode,



            control_net_path=control_net_path,



            owner_id=owner_id,



            owner_name=owner_name,



            shared=shared,



        )







        # Create job



        job_id = str(uuid.uuid4())[:8]



        STATE.register_job(job_id, owner_id=owner_id, owner_name=owner_name, kind="generation", serialized=True, shared=shared)







        # Run in background thread



        def run_job():
            if not STATE.acquire_generation(job_id):
                return
            try:



                worker = get_worker()
                if STATE.job_status(job_id) == JobStatus.CANCELLED:
                    return







                def progress_cb(event: ProgressEvent):



                    STATE.update_job(job_id, event)







                result = worker.generate(request, STATE.gpu_info, progress_callback=progress_cb)



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.COMPLETE,



                    progress=1.0,



                    image_path=str(result.image_path),



                    message=f"Done in {result.generation_time_sec:.1f}s",



                ))



            except Exception as e:



                # CRITICAL: Always update job state on failure, otherwise



                # frontend will poll forever waiting for a status that never comes.



                import traceback



                error_detail = f"{type(e).__name__}: {e}"



                print(f"[generate] Job {job_id} failed: {error_detail}")



                traceback.print_exc()



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.ERROR,



                    error=error_detail,



                    message=f"Generation failed: {error_detail}",



                ))

            finally:
                STATE.release_generation(job_id)



        thread = threading.Thread(target=run_job, daemon=True)



        thread.start()







        return {"job_id": job_id, "status": "queued"}







    @app.get("/api/jobs/{job_id}")



    async def api_job_status(job_id: str):



        job = STATE.get_job(job_id)



        if not job:



            raise HTTPException(404, f"Job not found: {job_id}")



        return job







    @app.post("/api/jobs/{job_id}/cancel")


    async def api_job_cancel(job_id: str, request: Request = None):


        job = STATE.get_job(job_id)


        if not job:


            raise HTTPException(404, f"Job not found: {job_id}")


        caller = _resolve_caller(request)


        if caller.get("is_guest", False) and job.get("owner_id") != caller.get("id"):


            raise HTTPException(403, "Нельзя отменить задачу другого пользователя.")


        if job.get("status") in (JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELLED):


            return {"status": job.get("status"), "was_active": False}


        was_active = STATE.cancel_job(job_id)

        try:
            from backend.downloader import get_download_manager
            get_download_manager().cancel(job_id)
        except Exception:
            pass

        if was_active and job.get("kind") == "generation":


            try:


                get_worker().cancel()


            except Exception as exc:


                print(f"[jobs] Worker cancel failed for {job_id}: {exc}")


        return {"status": "cancelled", "was_active": bool(was_active)}


    @app.post("/api/jobs/{job_id}/pause")
    async def api_job_pause(job_id: str, request: Request = None):
        caller = _resolve_caller(request)
        job = STATE.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        if caller.get("is_guest", False) and job.get("owner_id") != caller.get("id"):
            raise HTTPException(403, "Нельзя приостановить задачу другого пользователя.")

        from backend.downloader import get_download_manager
        mgr = get_download_manager()
        paused = mgr.pause(job_id)
        if paused:
            STATE.update_job(job_id, ProgressEvent(
                status="paused",
                message="Загрузка приостановлена",
            ))
            return {"status": "paused", "job_id": job_id}
        return {"status": "error", "message": "Загрузка не может быть приостановлена"}


    @app.post("/api/jobs/{job_id}/resume")
    async def api_job_resume(job_id: str, request: Request = None):
        caller = _resolve_caller(request)
        job = STATE.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        if caller.get("is_guest", False) and job.get("owner_id") != caller.get("id"):
            raise HTTPException(403, "Нельзя возобновить задачу другого пользователя.")

        from backend.downloader import get_download_manager
        mgr = get_download_manager()
        resumed = mgr.resume(job_id)
        if resumed:
            STATE.update_job(job_id, ProgressEvent(
                status="downloading",
                message="Возобновление загрузки...",
            ))
            return {"status": "resumed", "job_id": job_id}
        return {"status": "error", "message": "Загрузка не может быть возобновлена"}


    @app.get("/api/downloads")
    async def api_list_downloads():
        from backend.downloader import get_download_manager
        return get_download_manager().list_tasks()


    @app.get("/api/downloads/{task_id}")
    async def api_get_download(task_id: str):
        from backend.downloader import get_download_manager
        task = get_download_manager().get_task(task_id)
        if not task:
            raise HTTPException(404, f"Download task not found: {task_id}")
        return task.to_dict()


    @app.post("/api/downloads/{task_id}/pause")
    async def api_pause_download(task_id: str):
        from backend.downloader import get_download_manager
        ok = get_download_manager().pause(task_id)
        return {"status": "paused" if ok else "error", "task_id": task_id}


    @app.post("/api/downloads/{task_id}/resume")
    async def api_resume_download(task_id: str):
        from backend.downloader import get_download_manager
        ok = get_download_manager().resume(task_id)
        return {"status": "resumed" if ok else "error", "task_id": task_id}


    @app.post("/api/downloads/{task_id}/cancel")
    async def api_cancel_download(task_id: str):
        from backend.downloader import get_download_manager
        ok = get_download_manager().cancel(task_id)
        return {"status": "cancelled" if ok else "error", "task_id": task_id}


    # ---------- API: Settings ----------



    @app.patch("/api/settings")



    async def api_settings_update(body: dict, request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                allowed_guest = {"lang", "theme", "hide_uninstalled_in_generator"}



                unauthorized = [k for k in body.keys() if k not in allowed_guest]



                if unauthorized:



                    raise HTTPException(403, "Гостевой режим: системные настройки и API-ключи защищены хостом.")







        settings = get_settings()



        # Whitelist fields that can be updated



        allowed = {



            "lang", "theme", "hide_uninstalled_in_generator", "monitor_connections",



            "vram_mode", "auto_optimize",



            "download_source_priority", "auto_update_engine",
            "hf_transfer", "hf_endpoint",



            "huggingface_token", "modelscope_token",



            "quantfunc_api_key",



            "fbcache_cond_threshold", "fbcache_uncond_threshold",



            "rotation_block_size", "log_level",



            "worker_restart_on_crash",



            "cuda_bin_path",



            "qwen_vae_type",



            "enable_quantfunc_models",



            "share_token", "share_auto_start", "share_allow_admin",



            "extras_python",



            "models_dir", "output_dir",



        }



        updates = {k: v for k, v in body.items() if k in allowed}



        if "models_dir" in updates or "output_dir" in updates:

            for path_key in ("models_dir", "output_dir"):

                if path_key not in updates:

                    continue

                raw_path = str(updates[path_key] or "").strip()

                if not raw_path:

                    raise HTTPException(400, f"{path_key} cannot be empty")

                resolved = Path(raw_path).expanduser()

                try:

                    resolved.mkdir(parents=True, exist_ok=True)

                except OSError as exc:

                    raise HTTPException(400, f"Cannot use {path_key}: {exc}") from exc

                updates[path_key] = resolved



        if updates:



            if "hf_endpoint" in updates:
                _ep = str(updates["hf_endpoint"] or "").strip()
                if _ep and not _ep.lower().startswith(("http://", "https://")):
                    raise HTTPException(400, "hf_endpoint must start with http:// or https://")
                updates["hf_endpoint"] = _ep.rstrip("/")
            settings.update(**updates)
            if "hf_transfer" in updates or "hf_endpoint" in updates:
                apply_hf_download_settings(
                    getattr(settings, "hf_transfer", False), getattr(settings, "hf_endpoint", None))



            if "monitor_connections" in updates and not settings.monitor_connections:



                try:



                    get_presence().clear()



                except Exception:



                    pass



            return {"status": "saved", "updated": list(updates.keys())}



        return {"status": "no_changes"}







    # ---------- API: Engine ----------



    @app.get("/api/engine/status")



    async def api_engine_status():



        """Get status of all supported engines."""



        return {



            "status": "ok",



            "engines": _get_engines_info(),



        }







    @app.post("/api/engine/download")



    async def api_engine_download(body: Optional[dict] = None):



        """Download engine binary (QuantFunc v0.0.17 / v0.0.12, SD.CPP, or SDNQ).







        Body (optional):



            {"engine": "quantfunc", "version": "0.0.17"} or



            {"engine": "quantfunc", "version": "0.0.12"} or



            {"engine": "sd_cpp", "force": false} or



            {"engine": "sdnq", "force": false}



        """



        body = body or {}



        engine_type = body.get("engine", "quantfunc")



        version = body.get("version")







        try:



            if engine_type == "sd_cpp":



                from backend.sd_cpp import download_sd_cpp



                binary_path = await asyncio.to_thread(download_sd_cpp, force=bool(body.get("force", False)))



                return {



                    "status": "ok",



                    "engine": "sd_cpp",



                    "message": f"SD.CPP engine installed: {binary_path}",



                    "path": str(binary_path),



                }



            elif engine_type == "sdnq":



                from backend.sdnq_engine import install_sdnq, get_sdnq_version



                sdnq_path = await asyncio.to_thread(install_sdnq, force=bool(body.get("force", False)))



                return {



                    "status": "ok",



                    "engine": "sdnq",



                    "message": f"SDNQ ready ({get_sdnq_version()}): {sdnq_path}",



                    "path": str(sdnq_path),



                }



            else:



                target_ver = version or get_settings().engine_version



                binary_path = await asyncio.to_thread(download_engine, version=target_ver)



                return {



                    "status": "ok",



                    "engine": "quantfunc",



                    "version": target_ver,



                    "message": f"QuantFunc engine v{target_ver} installed and verified: {binary_path}",



                    "path": str(binary_path),



                }



        except Exception as e:



            raise HTTPException(500, f"Engine download failed: {e}")


    @app.post("/api/engine/uninstall")
    async def api_engine_uninstall(body: Optional[dict] = None, request: Request = None):
        """Remove SD.CPP, SDNQ, Nunchaku, or OrbitQuant (QuantFunc stays intact)."""
        if request:
            from backend.accounts import resolve_account_from_scope
            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)
            if caller.get("is_guest", False):
                raise HTTPException(403, "Guests cannot uninstall engines")

        body = body or {}
        engine_type = str(body.get("engine") or "").strip().lower()
        try:
            if engine_type == "sd_cpp":
                from backend.sd_cpp import uninstall_sd_cpp
                result = await asyncio.to_thread(uninstall_sd_cpp)
            elif engine_type == "sdnq":
                from backend.sdnq_engine import uninstall_sdnq
                result = await asyncio.to_thread(uninstall_sdnq)
            elif engine_type in ("nunchaku", "orbitquant"):
                from backend.extras import uninstall_component
                result = await asyncio.to_thread(uninstall_component, engine_type)
            else:
                raise HTTPException(400, "engine must be sd_cpp, sdnq, nunchaku, or orbitquant")
            return {"status": "ok", **result}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Engine uninstall failed: {e}")







    @app.post("/api/engine/verify")



    async def api_engine_verify(body: Optional[dict] = None):



        """Verify engine SHA-256 against remote verify.json."""



        body = body or {}



        version = body.get("version")



        target_ver = version or get_settings().engine_version



        result = await asyncio.to_thread(verify_engine, version=target_ver)



        return {



            "status": result.status,



            "version": target_ver,



            "message": result.message,



            "local_sha256": result.local_sha256,



            "remote_sha256": result.remote_sha256,



            "remote_version": result.remote_version,



        }







    @app.post("/api/engine/ensure-config")



    async def api_engine_ensure_config(body: Optional[dict] = None):



        """Install/restore bin/<os>/config.json with the bundled free-trial test key.







        Body (optional): {"force": true} — overwrite existing config.json



        (use when the bundled test key was deleted or corrupted).



        """



        try:



            force = bool((body or {}).get("force", False))



            cfg_path, source = ensure_config_json(force=force)



            return {



                "status": "ok",



                "message": f"Engine auth config ready ({source}): {cfg_path}",



                "path": str(cfg_path),



                "source": source,



            }



        except Exception as e:



            raise HTTPException(500, f"Failed to install config.json: {e}")







    @app.get("/api/cuda/status")



    async def api_cuda_status():



        """Check if CUDA runtime libraries are available."""



        from backend.engine import check_cuda_dependencies



        return check_cuda_dependencies()







    @app.post("/api/cuda/download")



    async def api_cuda_download():



        """Download and extract CUDA dependency bundle from ModelScope (~1 GB).







        Downloads cu13-dep-win32.zip (or equivalent for platform/CUDA version),



        extracts cublas64_XX.dll, cudnn64_9.dll, etc. into engine_dir.



        """



        from backend.engine import download_cuda_deps, check_cuda_dependencies







        job_id = str(uuid.uuid4())[:8]



        STATE.register_job(job_id)







        def run_cuda_download():



            try:



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.DOWNLOADING,



                    progress=0.0,



                    message="Downloading CUDA deps bundle (~1 GB)...",



                ))







                result = download_cuda_deps(progress_callback=lambda d, t, s: {



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.DOWNLOADING,



                        progress=d / t if t > 0 else 0,



                        downloaded_bytes=d,



                        total_bytes=t,



                        speed_bps=s,



                        message=f"Downloading CUDA deps: {d / 1024 / 1024:.0f} / {t / 1024 / 1024:.0f} MB",



                    ))



                })







                # Verify after extraction



                cuda_status = check_cuda_dependencies()



                if cuda_status["installed"]:



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.COMPLETE,



                        progress=1.0,



                        message=f"CUDA deps installed: {result['files_count']} files extracted ✓",



                    ))



                else:



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.ERROR,



                        error=f"CUDA deps downloaded but still missing: {cuda_status['missing']}",



                        message=f"Downloaded but still missing: {cuda_status['missing']}",



                    ))







            except Exception as e:



                import traceback



                error_detail = f"{type(e).__name__}: {e}"



                print(f"[cuda] Download job {job_id} failed: {error_detail}")



                traceback.print_exc()



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.ERROR,



                    error=error_detail,



                    message=f"CUDA deps download failed: {error_detail}",



                ))







        thread = threading.Thread(target=run_cuda_download, daemon=True)



        thread.start()







        return {"job_id": job_id, "status": "downloading"}







    # ---------- API: History ----------



    _history_sidecar_cache: dict[str, tuple[float, dict]] = {}







    @app.get("/api/history")



    async def api_history(
        request: Request = None,
        page: Optional[int] = Query(None, ge=1),
        page_size: Optional[int] = Query(None, ge=1, le=100),
        owner_filter: str = Query("all", alias="filter"),
        q: str = Query(""),
    ):



        caller = {"role": "host", "id": "host", "username": "Хост", "is_guest": False}



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)







        is_guest = caller.get("is_guest", False)



        caller_id = caller.get("id", "host")







        cache = CacheManager()



        outputs = cache.list_outputs(limit=None)



        result = []



        for p in outputs:



            try:



                st = p.stat()  # cache: avoid two syscalls per file



            except OSError:



                continue



            sidecar = p.with_suffix(".json")



            meta = {}



            sidecar_key = str(sidecar)



            try:



                sc_st = sidecar.stat()



                cached_entry = _history_sidecar_cache.get(sidecar_key)



                if cached_entry and cached_entry[0] == sc_st.st_mtime:



                    meta = cached_entry[1]



                else:



                    meta = json.loads(sidecar.read_text(encoding="utf-8"))



                    _history_sidecar_cache[sidecar_key] = (sc_st.st_mtime, meta)



            except FileNotFoundError:



                _history_sidecar_cache.pop(sidecar_key, None)



            except Exception as meta_err:



                print(f"[history] Failed to parse sidecar for {p.name}: {meta_err}")







            owner_id = meta.get("owner_id", "host")



            owner_name = meta.get("owner_name", "Хост")



            is_shared = meta.get("shared", True)







            # Isolation check: guest sees their own generations OR shared ones



            if is_guest and owner_id != caller_id and not is_shared:



                continue







            item = {



                "path": str(p),



                "name": p.name,



                "size": st.st_size,



                "modified": st.st_mtime,



                "url": f"/outputs/{p.name}",



                "prompt": meta.get("prompt", ""),



                "negative_prompt": meta.get("negative_prompt", ""),



                "model_id": meta.get("model_id", ""),



                "variant_id": meta.get("variant_id"),



                "seed": meta.get("seed"),



                "width": meta.get("width"),



                "height": meta.get("height"),



                "steps": meta.get("steps"),



                "cfg": meta.get("cfg"),



                "noise": meta.get("noise"),



                "sampler": meta.get("sampler"),



                "vae": meta.get("vae"),



                "backend": meta.get("backend"),



                "owner_id": owner_id,



                "owner_name": owner_name,



                "shared": is_shared,



                "generation_time_sec": meta.get("generation_time_sec"),



                "tool": meta.get("tool"),



                "tool_params": meta.get("tool_params"),



                "source_image": meta.get("source_image"),



                "mode": meta.get("mode"),



                "loras": meta.get("loras"),



            }



            result.append(item)



        owner_filter = (owner_filter or "all").strip().lower()

        query = (q or "").strip().lower()

        if owner_filter == "mine":

            result = [item for item in result if item.get("owner_id") == caller_id]

        elif owner_filter == "guests":

            result = [item for item in result if item.get("owner_id") not in ("host", "host_remote")]

        elif owner_filter == "shared":

            result = [item for item in result if item.get("shared")]

        if query:

            result = [

                item for item in result

                if query in (item.get("prompt") or "").lower()

                or query in (item.get("model_id") or "").lower()

                or query in (item.get("name") or "").lower()

                or query in (item.get("owner_name") or "").lower()

            ]

        paginated = page is not None or page_size is not None or owner_filter != "all" or bool(query)

        if paginated:

            return paginate(result, page or 1, page_size or 24)

        return result


    @app.delete("/api/history/{filename}")



    async def api_history_delete(filename: str, request: Request = None):



        from backend.security import validate_filename



        try:



            validate_filename(filename)



        except ValueError as e:



            raise HTTPException(400, f"Invalid filename: {e}")







        output_dir = get_settings().output_dir



        target_file = output_dir / filename



        sidecar = target_file.with_suffix(".json")







        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False) and sidecar.exists():



                try:



                    meta = json.loads(sidecar.read_text(encoding="utf-8"))



                    if meta.get("owner_id") != caller.get("id"):



                        raise HTTPException(403, "Гость может удалять только свои собственные генерации")



                except HTTPException:



                    raise



                except Exception:



                    pass







        try:



            target_file.unlink()



            if sidecar.exists():



                sidecar.unlink()



            return {"status": "ok", "message": f"Deleted {filename}"}



        except FileNotFoundError:



            raise HTTPException(404, f"File not found: {filename}")



        except Exception as e:



            raise HTTPException(500, f"Failed to delete file: {e}")







    @app.post("/api/history/{filename}/share")



    async def api_history_share_toggle(filename: str, request: Request = None, body: Optional[dict] = None):



        from backend.security import validate_filename



        try:



            validate_filename(filename)



        except ValueError as e:



            raise HTTPException(400, f"Invalid filename: {e}")







        output_dir = get_settings().output_dir



        target_file = output_dir / filename



        sidecar = target_file.with_suffix(".json")



        if not sidecar.exists():



            raise HTTPException(404, "Metadata sidecar not found")







        caller = {"role": "host", "id": "host", "username": "Хост", "is_guest": False}



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)







        try:



            meta = json.loads(sidecar.read_text(encoding="utf-8"))



        except Exception as e:



            raise HTTPException(500, f"Failed to read sidecar: {e}")







        if caller.get("is_guest", False) and meta.get("owner_id") != caller.get("id"):



            raise HTTPException(403, "Вы можете делиться только своими собственными генерациями")







        body = body or {}



        if "shared" in body:



            new_shared = bool(body["shared"])



        else:



            new_shared = not bool(meta.get("shared", False))







        meta["shared"] = new_shared



        sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")



        return {"status": "ok", "filename": filename, "shared": new_shared}







    # ---------- API: Accounts Management (Host only) ----------



    @app.get("/api/accounts")



    async def api_accounts_list(request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Управление аккаунтами доступно только хосту.")



        from backend.accounts import get_accounts_manager



        mgr = get_accounts_manager()



        return {"status": "ok", "accounts": mgr.list_accounts()}







    @app.post("/api/accounts")



    async def api_accounts_create(body: dict, request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Создание аккаунтов доступно только хосту.")



        username = body.get("username", "Гость")



        password = body.get("password")



        from backend.accounts import get_accounts_manager



        mgr = get_accounts_manager()



        acc = mgr.create_guest_account(username, password=password)



        return {"status": "ok", "account": acc}







    @app.patch("/api/accounts/{account_id}")



    async def api_accounts_update(account_id: str, body: dict, request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Редактирование аккаунтов доступно только хосту.")



        new_username = body.get("username")



        new_password = body.get("password")



        from backend.accounts import get_accounts_manager



        mgr = get_accounts_manager()



        updated = mgr.update_guest_account(account_id, new_username=new_username, new_password=new_password)



        if not updated:



            raise HTTPException(404, "Аккаунт не найден")



        return {"status": "ok", "account": updated}







    @app.delete("/api/accounts/{account_id}")



    async def api_accounts_delete(account_id: str, request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Удаление аккаунтов доступно только хосту.")



        from backend.accounts import get_accounts_manager



        mgr = get_accounts_manager()



        if not mgr.delete_guest_account(account_id):



            raise HTTPException(404, "Аккаунт не найден")



        return {"status": "ok", "message": f"Аккаунт {account_id} удалён"}







    @app.post("/api/accounts/{account_id}/token")



    async def api_accounts_regen_token(account_id: str, request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Смена токена доступна только хосту.")



        from backend.accounts import get_accounts_manager



        mgr = get_accounts_manager()



        new_tok = mgr.regenerate_guest_token(account_id)



        if not new_tok:



            raise HTTPException(404, "Аккаунт не найден")



        return {"status": "ok", "token": new_tok}







    # ---------- API: Remote Guest Auth & Self-Service ----------



    @app.post("/api/auth/login")



    async def api_auth_login(body: dict):



        username = (body.get("username") or "").strip()



        password = (body.get("password") or "").strip()



        if not username or not password:



            raise HTTPException(400, "Укажите имя пользователя и пароль")







        from backend.share import COOKIE_NAME, token_ok



        settings = get_settings()







        if settings.share_token and settings.share_allow_admin and token_ok(password, settings.share_token) and username.lower() in ("host", "хост", "admin", ""):



            tok = settings.share_token



            acc_info = {"role": "host", "id": "host_remote", "username": "Хост", "is_guest": False}



        else:



            from backend.accounts import get_accounts_manager



            acc = get_accounts_manager().verify_login(username, password)



            if not acc:



                raise HTTPException(401, "Неверное имя пользователя или пароль")



            tok = acc["token"]



            acc_info = {"role": "guest", "id": acc["id"], "username": acc.get("username", "Гость"), "is_guest": True}







        res = JSONResponse({"status": "ok", "account": acc_info, "token": tok})



        res.set_cookie(COOKIE_NAME, tok, path="/", httponly=True, samesite="lax", max_age=2592000)



        return res







    @app.post("/api/auth/logout")



    async def api_auth_logout():



        from backend.share import COOKIE_NAME



        res = JSONResponse({"status": "ok", "message": "Вы вышли из системы"})



        res.delete_cookie(COOKIE_NAME, path="/")



        return res







    @app.post("/api/account/change-password")



    async def api_account_change_password(body: dict, request: Request = None):



        """Guest self-service: update their own account password."""



        from backend.accounts import resolve_account_from_scope, get_accounts_manager



        from backend.share import COOKIE_NAME



        caller = (request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)) if request else {}



        account_id = caller.get("id")



        if not account_id or account_id in ("anonymous", "host"):



            if account_id == "host":



                raise HTTPException(400, "Пароль хоста настраивается в параметрах туннеля.")



            raise HTTPException(401, "Необходима авторизация")







        new_pwd = (body.get("new_password") or "").strip()



        if len(new_pwd) < 3:



            raise HTTPException(400, "Пароль должен содержать не менее 3 символов")







        mgr = get_accounts_manager()



        ok = mgr.change_guest_password(account_id, new_pwd)



        if not ok:



            raise HTTPException(404, "Аккаунт не найден")







        updated_acc = mgr.get_account(account_id)



        new_tok = updated_acc.get("token")



        res = JSONResponse({"status": "ok", "message": "Пароль успешно обновлён", "token": new_tok})



        res.set_cookie(COOKIE_NAME, new_tok, path="/", httponly=True, samesite="lax", max_age=2592000)



        return res







    # ---------- API: Test Hugging Face Token (Host only) ----------



    @app.post("/api/hf/test-token")



    async def api_hf_test_token(body: dict, request: Request = None):



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



            if caller.get("is_guest", False):



                raise HTTPException(403, "Проверка токенов доступна только хосту.")



        token = body.get("token") or get_settings().huggingface_token



        if not token:



            return {"status": "empty", "valid": False, "message": "Токен не указан"}



        try:



            from huggingface_hub import HfApi



            api = HfApi(token=token)



            user_info = await asyncio.to_thread(api.whoami)



            username = user_info.get("name", "Unknown")



            from backend.settings import sync_hf_token_to_system



            sync_hf_token_to_system(token)



            return {"status": "ok", "valid": True, "user": username, "message": f"Токен активен (пользователь: {username})"}



        except Exception as e:



            return {"status": "invalid", "valid": False, "message": f"Ошибка проверки: {e}"}







    # ---------- Helpers: Safe Image Resolution & Tool Output Saving ----------



    def _resolve_caller(request: Optional[Request]) -> dict:



        caller = {"role": "host", "id": "host", "username": "Хост", "is_guest": False}



        if request:



            from backend.accounts import resolve_account_from_scope



            caller = request.scope.get("state", {}).get("account") or resolve_account_from_scope(request.scope)



        return caller







    def _normalize_output_rel_path(raw_path: str) -> str:



        norm = str(raw_path or "").strip().replace("\\", "/")



        if norm.startswith("/outputs/"):



            norm = norm[len("/outputs/"):]



        elif norm.startswith("outputs/"):



            norm = norm[len("outputs/"):]



        elif norm.startswith("/"):



            norm = norm[1:]



        return norm







    def _resolve_source_image(image_path: Optional[str], image_data: Optional[str]):



        from backend.halftone import load_image_safe



        from backend.security import validate_path







        settings = get_settings()



        if image_path and str(image_path).strip():



            norm = _normalize_output_rel_path(image_path)



            if not norm or ".." in norm.split("/"):



                raise HTTPException(400, "Invalid source image path")



            try:



                resolved = validate_path(



                    settings.output_dir / norm,



                    base_dir=settings.output_dir,



                    must_exist=True,



                )



            except (ValueError, FileNotFoundError) as e:



                raise HTTPException(400, f"Invalid or missing source image: {e}")



            try:



                img = load_image_safe(resolved)



            except Exception as e:



                raise HTTPException(400, f"Failed to decode source image: {e}")



            return img, f"/outputs/{norm}", norm







        if image_data and str(image_data).strip():



            try:



                img = load_image_safe(str(image_data))



            except Exception as e:



                raise HTTPException(400, f"Failed to decode base64 image data: {e}")



            return img, "inline_upload", None







        raise HTTPException(400, "Source image is required ('image_path' or 'image_data')")







    def _save_tool_output_image(



        img,



        prefix: str,



        tool_name: str,



        tool_params: dict,



        source_image_ref: str,



        caller: dict,



        elapsed_sec: float = 0.0,



        extra_meta: Optional[dict] = None,



    ) -> dict:



        from datetime import datetime



        from PIL.PngImagePlugin import PngInfo



        from backend.security import validate_filename







        settings = get_settings()



        output_dir = settings.output_dir



        output_dir.mkdir(parents=True, exist_ok=True)







        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")



        uid = uuid.uuid4().hex[:6]



        fname = validate_filename(f"{prefix}_{ts_str}_{uid}.png")



        dest_path = output_dir / fname



        sidecar_path = dest_path.with_suffix(".json")







        owner_id = caller.get("id", "host")



        owner_name = caller.get("username", "Хост")



        default_shared = not caller.get("is_guest", False)







        meta = {



            "prompt": (extra_meta or {}).get("prompt") or f"[{tool_name}] {tool_params.get('mode', 'custom')}",



            "negative_prompt": (extra_meta or {}).get("negative_prompt", ""),



            "model_id": (extra_meta or {}).get("model_id") or tool_name,



            "variant_id": (extra_meta or {}).get("variant_id") or tool_params.get("mode", "custom"),



            "seed": tool_params.get("seed", 42),



            "width": img.width,



            "height": img.height,



            "steps": (extra_meta or {}).get("steps", 1),



            "cfg": (extra_meta or {}).get("cfg", 1.0),



            "backend": (extra_meta or {}).get("backend", "cpu"),



            "tool": tool_name,



            "mode": tool_params.get("mode", "custom"),



            "tool_params": tool_params,



            "source_image": source_image_ref,



            "owner_id": owner_id,



            "owner_name": owner_name,



            "shared": bool((extra_meta or {}).get("shared", default_shared)),



            "timestamp": time.time(),



            "generation_time_sec": round(float(elapsed_sec), 3),



        }



        if extra_meta:



            for k, v in extra_meta.items():



                if k not in meta and v is not None:



                    meta[k] = v







        png_info = PngInfo()



        png_info.add_text("lazy_sd_tool", tool_name)



        png_info.add_text("lazy_sd_params", json.dumps(meta, ensure_ascii=False))



        img.save(dest_path, format="PNG", pnginfo=png_info)



        sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")







        st = dest_path.stat()



        return {



            "path": str(dest_path),



            "name": fname,



            "size": st.st_size,



            "modified": st.st_mtime,



            "url": f"/outputs/{fname}",



            "prompt": meta["prompt"],



            "negative_prompt": meta["negative_prompt"],



            "model_id": meta["model_id"],



            "variant_id": meta["variant_id"],



            "seed": meta["seed"],



            "width": img.width,



            "height": img.height,



            "steps": meta["steps"],



            "cfg": meta["cfg"],



            "backend": meta["backend"],



            "tool": tool_name,



            "mode": meta["mode"],



            "tool_params": tool_params,



            "source_image": source_image_ref,



            "owner_id": owner_id,



            "owner_name": owner_name,



            "shared": meta["shared"],



            "generation_time_sec": meta["generation_time_sec"],



            "loras": meta.get("loras"),



        }







    # ---------- API: LoRA Compatibility Check ----------



    @app.get("/api/loras/check-compatibility")



    async def api_check_lora_compatibility(lora_name: str = Query(...), model_id: str = Query(...)):



        """Check whether a LoRA file is compatible with a model architecture and engine."""



        from backend.lora import check_lora_compatibility



        registry = get_registry()



        try:



            model = registry.get(model_id)



        except KeyError:



            raise HTTPException(404, f"Model not found: {model_id}")



        res = check_lora_compatibility(lora_name, model)



        return res







    # ---------- API: Halftone Studio (CPU Algorithmic Engine) ----------



    @app.get("/api/halftone/presets")



    async def api_halftone_presets():



        from backend.halftone import HALFTONE_PRESETS



        return {"presets": HALFTONE_PRESETS}







    @app.post("/api/halftone/process")



    async def api_halftone_process(body: dict, request: Request = None):



        """Process an image with Halftone Studio GT on CPU (no diffusion model loaded)."""



        from backend.halftone import (



            HalftoneParams,



            process_halftone_image,



            process_halftone_layers,



            image_to_data_url,



        )







        caller = _resolve_caller(request)



        raw_params = body.get("params") or {}



        if not isinstance(raw_params, dict):



            print("[halftone] ERROR: 'params' must be an object", flush=True)



            raise HTTPException(400, "'params' must be an object")







        try:



            ht_params = HalftoneParams.from_dict(raw_params)



        except (ValueError, TypeError) as e:



            print(f"[halftone] ERROR (invalid params): {e} | raw_params={raw_params}", flush=True)



            raise HTTPException(400, f"Invalid halftone parameters: {e}")







        try:



            src_img, src_ref, _ = _resolve_source_image(



                body.get("image_path"),



                body.get("image_data"),



            )



        except HTTPException as e:



            print(f"[halftone] ERROR (source image): {e.detail}", flush=True)



            raise



        orig_w, orig_h = src_img.size

        roi_rect = None
        roi_payload = body.get("roi")
        if isinstance(roi_payload, dict):
            try:
                roi_x = int(round(float(roi_payload.get("x", 0))))
                roi_y = int(round(float(roi_payload.get("y", 0))))
                roi_w = int(round(float(roi_payload.get("width", roi_payload.get("w", 0)))))
                roi_h = int(round(float(roi_payload.get("height", roi_payload.get("h", 0)))))
            except (TypeError, ValueError):
                raise HTTPException(400, "Invalid ROI: x/y/width/height must be numbers")
            if roi_w <= 0 or roi_h <= 0:
                raise HTTPException(400, "Invalid ROI: width and height must be positive")
            roi_x = max(0, min(orig_w - 1, roi_x))
            roi_y = max(0, min(orig_h - 1, roi_y))
            roi_w = max(1, min(orig_w - roi_x, roi_w))
            roi_h = max(1, min(orig_h - roi_y, roi_h))
            roi_rect = (roi_x, roi_y, roi_w, roi_h)

        save = bool(body.get("save", False))



        try:
            preview_max_side = int(body.get("preview_max_side", 0))
        except (ValueError, TypeError):
            preview_max_side = 0



        if save:
            preview_max_side = 0  # Always full resolution on save/export
            roi_rect = None  # Exports always use the complete source
        roi_arg = roi_rect

        mask_data = body.get("mask")



        layers_data = body.get("layers")



        t0 = time.perf_counter()



        try:



            if isinstance(layers_data, list) and len(layers_data) > 0:



                out_img = await _run_cpu_tool(
                    process_halftone_layers,
                    src_img,
                    layers_data,
                    preview_max_side,
                    roi=roi_arg,
                )



            else:



                out_img = await _run_cpu_tool(
                    process_halftone_image,
                    src_img,
                    ht_params,
                    mask_data,
                    preview_max_side,
                    roi=roi_arg,
                )



        except ValueError as e:



            print(f"[halftone] ERROR (ValueError): {e}", flush=True)



            raise HTTPException(400, f"Halftone processing error: {e}")



        except Exception as e:



            import traceback



            print(f"[halftone] ERROR (Exception): {e}\n{traceback.format_exc()}", flush=True)



            raise HTTPException(500, f"Halftone processing failed: {e}")



        elapsed = time.perf_counter() - t0







        saved_item = None



        if save:



            shared = bool(body.get("shared", not caller.get("is_guest", False)))



            tool_params_dict = ht_params.to_dict()



            if isinstance(layers_data, list) and len(layers_data) > 0:



                # Store layer configurations (without heavy base64 masks) in metadata



                clean_layers = []



                for lyr in layers_data:



                    if isinstance(lyr, dict):



                        clean_layers.append({



                            "id": lyr.get("id"),



                            "name": lyr.get("name"),



                            "visible": lyr.get("visible", True),



                            "opacity": lyr.get("opacity", 1.0),



                            "blend_mode": lyr.get("blend_mode", "normal"),



                            "hasMask": bool(lyr.get("hasMask", False)),



                            "isBase": bool(lyr.get("isBase", False)),



                            "params": lyr.get("params") or {},



                        })



                tool_params_dict["layers"] = clean_layers



            saved_item = _save_tool_output_image(



                out_img,



                prefix="ht",



                tool_name="halftone_studio",



                tool_params=tool_params_dict,



                source_image_ref=src_ref,



                caller=caller,



                elapsed_sec=elapsed,



                extra_meta={"shared": shared},



            )







        data_url = await asyncio.to_thread(image_to_data_url, out_img, "PNG")



        return {



            "status": "ok",



            "preview_data_url": data_url,



            "width": out_img.width,



            "height": out_img.height,



            "orig_width": orig_w,
            "orig_height": orig_h,
            "preview_max_side": preview_max_side,
            "roi": (
                {"x": roi_rect[0], "y": roi_rect[1], "width": roi_rect[2], "height": roi_rect[3]}
                if roi_rect else None
            ),
            "params": ht_params.to_dict(),



            "elapsed_sec": round(elapsed, 3),



            "saved": saved_item,



        }







    # ---------- API: Manga Screen AI & Manga Finish ----------



    @app.post("/api/manga/process")



    async def api_manga_process(body: dict, request: Request = None):



        """Run CPU algorithmic Manga pipeline (Manga Lite / Finish Only / custom stage combination)."""



        from backend.manga_pipeline import MangaPipelineParams, process_manga_algorithmic



        from backend.halftone import image_to_data_url







        caller = _resolve_caller(request)



        raw_params = body.get("params") or {}



        if not isinstance(raw_params, dict):



            print("[manga] ERROR: 'params' must be an object", flush=True)



            raise HTTPException(400, "'params' must be an object")







        try:



            mp_params = MangaPipelineParams.from_dict(raw_params)



        except (ValueError, TypeError) as e:



            print(f"[manga] ERROR (invalid params): {e}", flush=True)



            raise HTTPException(400, f"Invalid manga parameters: {e}")







        try:



            src_img, src_ref, _ = _resolve_source_image(



                body.get("image_path"),



                body.get("image_data"),



            )



        except HTTPException as e:



            print(f"[manga] ERROR (source image): {e.detail}", flush=True)



            raise



        orig_w, orig_h = src_img.size







        save = bool(body.get("save", False))



        try:
            preview_max_side = int(body.get("preview_max_side", 0))
        except (ValueError, TypeError):
            preview_max_side = 0



        if save:



            preview_max_side = 0







        mask_data = body.get("mask")



        t0 = time.perf_counter()



        try:



            out_img = await _run_cpu_tool(



                process_manga_algorithmic,



                src_img,



                mp_params,



                mask_data,



                preview_max_side,



            )



        except ValueError as e:



            print(f"[manga] ERROR (ValueError): {e}", flush=True)



            raise HTTPException(400, f"Manga processing error: {e}")



        except Exception as e:



            import traceback



            print(f"[manga] ERROR (Exception): {e}\n{traceback.format_exc()}", flush=True)



            raise HTTPException(500, f"Manga processing failed: {e}")



        elapsed = time.perf_counter() - t0







        saved_item = None



        if save:



            shared = bool(body.get("shared", not caller.get("is_guest", False)))



            saved_item = _save_tool_output_image(



                out_img,



                prefix="manga",



                tool_name="manga_screen_ai",



                tool_params=mp_params.to_dict(),



                source_image_ref=src_ref,



                caller=caller,



                elapsed_sec=elapsed,



                extra_meta={"shared": shared},



            )







        data_url = await asyncio.to_thread(image_to_data_url, out_img, "PNG")



        return {



            "status": "ok",



            "preview_data_url": data_url,



            "width": out_img.width,



            "height": out_img.height,



            "orig_width": orig_w,



            "orig_height": orig_h,



            "params": mp_params.to_dict(),



            "elapsed_sec": round(elapsed, 3),



            "saved": saved_item,



        }







    @app.post("/api/manga/ai-job")



    async def api_manga_ai_job(body: dict, request: Request = None):



        """Start an asynchronous AI Manga or AI Manga + Finish job using existing Img2Img/Ref worker."""



        from backend.manga_pipeline import MangaPipelineParams, process_manga_algorithmic



        from backend.halftone import load_image_safe



        from backend.lora import check_lora_compatibility, resolve_lora_selection



        from backend.security import validate_filename







        caller = _resolve_caller(request)



        owner_id = caller.get("id", "host")



        owner_name = caller.get("username", "Хост")



        default_shared = not caller.get("is_guest", False)



        shared = bool(body.get("shared", default_shared))







        raw_params = body.get("params") or {}



        if not isinstance(raw_params, dict):



            raise HTTPException(400, "'params' must be an object")







        try:



            mp_params = MangaPipelineParams.from_dict(raw_params)



        except (ValueError, TypeError) as e:



            raise HTTPException(400, f"Invalid manga parameters: {e}")







        if mp_params.mode not in ("ai_manga", "ai_manga_finish") and not mp_params.ai_enabled:



            raise HTTPException(400, "Mode does not require AI job; use /api/manga/process for CPU modes.")







        model_id = (mp_params.model_id or body.get("model_id") or "").strip()



        if not model_id:



            raise HTTPException(400, "Please select an AI model for AI Manga mode.")







        if caller.get("is_guest", False):



            from backend.model_visibility import get_model_visibility



            vinfo = get_model_visibility(model_id)



            if vinfo.get("hidden_for_guests", False) or vinfo.get("hidden_from_host", False):



                raise HTTPException(403, "Выбранная модель скрыта хостом и недоступна для использования.")







        if not STATE.gpu_info:



            raise HTTPException(400, "GPU not detected. Switch to 'Manga Lite' or 'Finish Only' for CPU processing.")







        registry = get_registry()



        try:



            model = registry.get(model_id)



        except KeyError:



            raise HTTPException(404, f"Model not found: {model_id}")







        model_engine = getattr(model, "engine", "quantfunc")



        if model_engine == "sd_cpp":



            if not is_sd_cpp_installed():



                raise HTTPException(400, "SD.CPP engine not installed. Go to Settings → Download SD.CPP, or use 'Manga Lite'.")



        elif model_engine == "sdnq":



            if not is_sdnq_installed():



                raise HTTPException(400, "SDNQ runtime not installed. Go to Settings → Install/Verify SDNQ, or use 'Manga Lite'.")



        elif model_engine == "orbitquant":



            from backend.extras import is_orbitquant_ready



            if not is_orbitquant_ready():



                raise HTTPException(400, "OrbitQuant runtime not installed.")



        elif model_engine == "nunchaku":



            from backend.extras import is_nunchaku_ready



            if not is_nunchaku_ready():



                raise HTTPException(400, "Nunchaku runtime not installed.")



        else:



            settings = get_settings()



            if quantfunc_model_disabled(
                model, bool(getattr(settings, "enable_quantfunc_models", False))
            ):



                raise HTTPException(400, "Эта QuantFunc-модель временно отключена (дефектные веса FLUX Klein). Выберите модель SD.CPP или SDNQ, включите QuantFunc-модели в Настройках либо используйте режим 'Manga Lite'.")



            if not (is_engine_installed(model.engine_version) or is_engine_installed()):



                raise HTTPException(400, "QuantFunc engine not installed.")







        # Resolve source image (if base64 inline, save a new copy in outputs/uploads without touching any existing file)



        src_img, src_ref, norm_rel = _resolve_source_image(



            body.get("image_path"),



            body.get("image_data"),



        )



        settings = get_settings()



        if not norm_rel:



            uploads_dir = settings.output_dir / "uploads"



            uploads_dir.mkdir(parents=True, exist_ok=True)



            safe_stem = validate_filename(f"ref_{uuid.uuid4().hex[:8]}.png")



            src_img.save(uploads_dir / safe_stem, format="PNG")



            norm_rel = f"uploads/{safe_stem}"



            src_ref = f"/outputs/{norm_rel}"







        # Validate LoRA compatibility if a LoRA is specified



        lora_paths: list[Path] = []



        lora_strengths: list[float] = []



        lora_meta_list: list[dict] = []



        if mp_params.lora_name:



            compat = check_lora_compatibility(mp_params.lora_name, model)



            if not compat.get("compatible", False):



                raise HTTPException(400, f"LoRA incompatible: {compat.get('reason', 'Unknown incompatibility')}")



            try:



                lora_paths, lora_strengths, lora_meta_list = resolve_lora_selection([



                    {"name": mp_params.lora_name, "strength": mp_params.lora_strength}



                ])



            except ValueError as e:



                raise HTTPException(400, f"LoRA error: {e}")







        # Find transformer variant



        variant_id = mp_params.variant_id or body.get("variant_id")



        transformers_for_gpu = model.get_transformers_for_tier(STATE.gpu_info.sm)



        selected = None



        if variant_id:



            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)



            if not selected:



                selected = next((t for t in model.transformers if t.id == variant_id), None)



        if not selected and transformers_for_gpu:



            selected = transformers_for_gpu[0]



        if not selected and model.transformers:



            selected = model.transformers[0]



        if not selected:



            raise HTTPException(400, f"No compatible transformer found for model {model_id}")







        # Compute target dimensions (preserve aspect ratio aligned to 64, capped to 1024 max side for safe VRAM)



        orig_w, orig_h = src_img.size



        max_ai_side = int(body.get("max_ai_side", 1024))



        max_ai_side = max(512, min(1536, max_ai_side))



        scale = min(1.0, float(max_ai_side) / float(max(orig_w, orig_h, 1)))



        ai_w = max(256, int(round((orig_w * scale) / 64.0)) * 64)



        ai_h = max(256, int(round((orig_h * scale) / 64.0)) * 64)







        effective_steps = mp_params.ai_steps if mp_params.ai_steps > 0 else selected.steps



        effective_cfg = mp_params.ai_cfg if mp_params.ai_cfg > 0 else model.get_default_cfg()







        gen_request = GenerateRequest(



            model_id=model_id,



            prompt=mp_params.ai_prompt,



            negative_prompt=mp_params.ai_negative_prompt,



            seed=mp_params.seed,



            width=ai_w,



            height=ai_h,



            steps=effective_steps,



            cfg=effective_cfg,



            noise=0.0,



            backend=selected.backend,



            use_prequant=selected.prequant,



            init_image_path=Path(norm_rel),



            init_image_strength=mp_params.ai_denoise_strength,



            lora_paths=lora_paths,



            lora_strengths=lora_strengths,



            variant_id=selected.id,



            ref_mode=mp_params.ai_ref_mode,



            owner_id=owner_id,



            owner_name=owner_name,



            shared=shared,



        )







        job_id = str(uuid.uuid4())[:8]



        STATE.register_job(job_id, owner_id=owner_id, owner_name=owner_name, kind="generation", serialized=True, shared=shared)



        mask_data = body.get("mask")







        def run_manga_ai_job():
            if not STATE.acquire_generation(job_id):
                return
            t0 = time.perf_counter()
            try:



                worker = get_worker()
                if STATE.job_status(job_id) == JobStatus.CANCELLED:
                    return







                def progress_cb(event: ProgressEvent):



                    # Scale AI progress to 0.0 - 0.85 if finish is enabled



                    if mp_params.mode == "ai_manga_finish" or mp_params.finish_enabled:



                        scaled_prog = min(0.85, float(event.progress or 0.0) * 0.85)



                        STATE.update_job(job_id, ProgressEvent(



                            status=event.status if event.status != JobStatus.COMPLETE else JobStatus.GENERATING,



                            progress=scaled_prog,



                            step=event.step,



                            total_steps=event.total_steps,



                            message=f"[Stage A: AI Manga] {event.message}",



                        ))



                    else:



                        STATE.update_job(job_id, event)







                ai_result = worker.generate(gen_request, STATE.gpu_info, progress_callback=progress_cb)







                if mp_params.mode == "ai_manga_finish" or mp_params.finish_enabled:



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.GENERATING,



                        progress=0.90,



                        message="[Stage B–E] Applying Manga Finish (Screentone, Ink & Panel)...",



                    ))



                    ai_img = load_image_safe(ai_result.image_path)



                    # Resize back to original dimensions if requested and panel crop is original



                    if ai_img.size != (orig_w, orig_h):



                        ai_img = ai_img.resize((orig_w, orig_h), resample=3)  # Image.Resampling.LANCZOS







                    final_img = _run_cpu_limited(



                        process_manga_algorithmic,



                        ai_img,



                        mp_params,



                        mask_source=mask_data,



                        preview_max_side=0,



                    )



                    total_elapsed = time.perf_counter() - t0



                    saved = _save_tool_output_image(



                        final_img,



                        prefix="manga",



                        tool_name="manga_screen_ai",



                        tool_params=mp_params.to_dict(),



                        source_image_ref=src_ref,



                        caller=caller,



                        elapsed_sec=total_elapsed,



                        extra_meta={



                            "prompt": mp_params.ai_prompt,



                            "negative_prompt": mp_params.ai_negative_prompt,



                            "model_id": model_id,



                            "variant_id": selected.id,



                            "steps": effective_steps,



                            "cfg": effective_cfg,



                            "backend": selected.backend,



                            "loras": lora_meta_list,



                            "ai_intermediate_image": ai_result.image_path.name,



                            "shared": shared,



                        },



                    )



                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.COMPLETE,



                        progress=1.0,



                        image_path=saved["path"],



                        message=f"AI Manga + Finish complete in {total_elapsed:.1f}s",



                    ))



                else:



                    # Pure AI Manga mode: enrich existing sidecar with tool metadata



                    sidecar = ai_result.image_path.with_suffix(".json")



                    if sidecar.exists():



                        try:



                            meta = json.loads(sidecar.read_text(encoding="utf-8"))



                            meta["tool"] = "manga_screen_ai"



                            meta["mode"] = mp_params.mode



                            meta["tool_params"] = mp_params.to_dict()



                            meta["source_image"] = src_ref



                            if lora_meta_list:



                                meta["loras"] = lora_meta_list



                            sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")



                        except Exception as meta_err:



                            print(f"[manga-ai] Failed to enrich sidecar: {meta_err}")







                    STATE.update_job(job_id, ProgressEvent(



                        status=JobStatus.COMPLETE,



                        progress=1.0,



                        image_path=str(ai_result.image_path),



                        message=f"AI Manga complete in {ai_result.generation_time_sec:.1f}s",



                    ))



            except Exception as e:



                import traceback



                error_detail = f"{type(e).__name__}: {e}"



                print(f"[manga-ai] Job {job_id} failed: {error_detail}")



                traceback.print_exc()



                STATE.update_job(job_id, ProgressEvent(



                    status=JobStatus.ERROR,



                    error=error_detail,



                    message=f"AI Manga failed: {error_detail}",



                ))

            finally:
                STATE.release_generation(job_id)



        thread = threading.Thread(target=run_manga_ai_job, daemon=True)



        thread.start()



        return {"job_id": job_id, "status": "queued", "mode": mp_params.mode}







    # ---------- Root: serve index.html ----------



    @app.get("/", response_class=HTMLResponse)



    async def homepage():



        return build_index_html()







    @app.get("/index.html", response_class=HTMLResponse)



    async def index_html_alias():



        return build_index_html()







    return app











# ============================================================



# Main



# ============================================================



def find_available_port(host: str, start_port: int, max_attempts: int = 50) -> int:
    """Find the first available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.bind((host if host != "0.0.0.0" else "127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def _free_port_if_in_use(host: str, port: int) -> None:



    """If a previous QuantFunc-WebUI Python instance is still listening on `port`, terminate it cleanly."""



    import os



    import socket



    import time







    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:



        s.settimeout(0.3)



        if s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) != 0:



            return  # Port is already free







    my_pid = os.getpid()



    try:



        import psutil



        for conn in psutil.net_connections(kind="inet"):



            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:



                pid = conn.pid



                if pid and pid != my_pid:



                    try:



                        proc = psutil.Process(pid)



                        pname = (proc.name() or "").lower()



                        if "python" in pname:



                            print(f"[startup] Port {port} is held by previous Python process (PID {pid}) — terminating it...")



                            proc.terminate()



                            try:



                                proc.wait(timeout=3.0)



                            except psutil.TimeoutExpired:



                                proc.kill()



                            time.sleep(0.4)



                    except Exception as e:



                        print(f"[startup] Could not terminate PID {pid} on port {port}: {e}")



    except Exception:



        pass











def print_compact_engine_fleet_cli() -> None:



    """Print a reduced, compact 5-engine status table directly in the CLI."""



    try:



        from backend.engine import is_engine_installed



        from backend.sd_cpp import is_sd_cpp_installed, get_sd_cpp_version



        from backend.sdnq_engine import is_sdnq_installed, get_sdnq_version



        from backend.extras import is_orbitquant_ready, is_nunchaku_ready, inspect_env







        qf_installed = is_engine_installed()



        qf_status = "v0.0.17 [Ready]" if qf_installed else "[Not installed]"







        sd_cpp_ok = is_sd_cpp_installed()



        sd_cpp_ver = get_sd_cpp_version() or "master"



        sd_cpp_status = f"{sd_cpp_ver} [Ready]" if sd_cpp_ok else "[Not installed]"







        sdnq_ok = is_sdnq_installed()



        sdnq_ver = get_sdnq_version() or "v0.2.8"



        sdnq_status = f"{sdnq_ver} [Ready]" if sdnq_ok else "[Not installed]"







        env = inspect_env()



        nunchaku_ver = env.get("nunchaku")



        nunchaku_status = f"v{nunchaku_ver} [Ready]" if nunchaku_ver else ("[Available in Extras]" if is_nunchaku_ready() else "[Not installed]")







        orbit_ver = env.get("orbitquant")



        orbit_status = f"v{orbit_ver} [Ready]" if orbit_ver else ("[Available in Extras]" if is_orbitquant_ready() else "[Not installed]")







        print("┌" + "─" * 60 + "┐")



        print("│ Engine Fleet Status (5 engines):                           │")



        print(f"│  • QuantFunc:   {qf_status:<20} (Native SVDQ/FP4/FP8)     │")



        print(f"│  • SD.CPP:      {sd_cpp_status:<20} (GGUF / sd-cli)             │")



        print(f"│  • SDNQ:        {sdnq_status:<20} (Diffusers / 4-bit Turbo)     │")



        print(f"│  • Nunchaku:    {nunchaku_status:<20} (SVDQ / FLUX.2 Klein)     │")



        print(f"│  • OrbitQuant:  {orbit_status:<20} (BF16 / W4A16 Native)         │")



        print("└" + "─" * 60 + "┘\n")



    except Exception as e:



        print(f"[engines] Status probe failed: {e}\n")











def main():



    """Entry point — initialize state, build app, launch server."""



    import argparse



    from backend.logging_config import setup_logging







    parser = argparse.ArgumentParser(description="QuantFunc WebUI server")



    parser.add_argument(



        "--verbose", "-v",



        action="store_true",



        default=False,



        help="Show detailed debug output and HTTP access logs in console (default: quiet)",



    )



    parser.add_argument(



        "--host",



        type=str,



        default="127.0.0.1",



        help="Host to bind to (default: 127.0.0.1)",



    )



    parser.add_argument(



        "--port",



        type=int,



        default=7860,

        help="Port to bind to (default: 7860)",

    )

    parser.add_argument(
        "--auto-port",
        action="store_true",
        default=True,
        help="Automatically pick the next free port if the requested port is already in use (default: enabled)",
    )

    parser.add_argument(
        "--no-auto-port",
        dest="auto_port",
        action="store_false",
        help="Disable automatic port fallback; fail if the requested port is busy",
    )



    parser.add_argument(



        "--log-dir",



        type=str,



        default=None,



        help="Directory to store log files (default: logs/)",



    )







    args = parser.parse_args()







    import socket
    requested_port = args.port
    is_busy = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.bind((args.host if args.host != "0.0.0.0" else "127.0.0.1", args.port))
        except OSError:
            is_busy = True

    if is_busy:
        if args.auto_port:
            free_port = find_available_port(args.host, requested_port + 1)
            print(f"[startup] Port {requested_port} is already in use by another process (e.g. ComfyUI / Automatic1111 / WebUI).")
            print(f"[startup] Automatically switched to next free port: http://{args.host}:{free_port}")
            args.port = free_port
        else:
            _free_port_if_in_use(args.host, args.port)







    log_file, uvicorn_log_config = setup_logging(verbose=args.verbose, log_dir=args.log_dir)







    init_state()



    from backend.share import set_bind, get_share_manager



    set_bind(args.host, args.port)



    app = create_app()







    settings = get_settings()



    if getattr(settings, "share_auto_start", False):



        try:



            get_share_manager().start()



        except Exception as e:



            print(f"[share] Auto-start failed: {e}")







    import uvicorn







    gpu_desc = (



        f"{STATE.gpu_info.name} (SM {STATE.gpu_info.sm}, {STATE.gpu_info.vram_gb:.1f} GB VRAM)"



        if STATE.gpu_info else "No NVIDIA GPU detected"



    )







    print("\n" + "=" * 62)



    print(" QuantFunc WebUI")



    print(f" URL:     http://{args.host}:{args.port}")



    print(f" GPU:     {gpu_desc}")



    print(f" Logs:    {log_file} (all activity logged silently)")



    if args.verbose:



        print(" Mode:    VERBOSE (detailed console logs enabled)")



    else:



        print(" Mode:    QUIET console (use --verbose for detailed HTTP/debug logs)")



    print("=" * 62 + "\n")







    print_compact_engine_fleet_cli()







    uvicorn.run(



        app,



        host=args.host,



        port=args.port,



        log_config=uvicorn_log_config,



        log_level="debug" if args.verbose else "info",



        reload=False,



    )











if __name__ == "__main__":



    main()

