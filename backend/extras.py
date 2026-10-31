"""
Managed "extras" runtime for the PyTorch-based engines that don't ship their own
binaries: **OrbitQuant** and **Nunchaku (vitoom fork)**.

Why a separate venv?  The Nunchaku wheels from ``tonera/vitoom-nunchaku`` are built
against an exact torch minor + CUDA (e.g. torch 2.12 + cu13.0) and won't load into
the SD.Next environment used by SDNQ (a different torch). So we keep an isolated
``envs/extras`` venv; users who already have a suitable environment can point
``settings.extras_python`` at it instead.

Nothing here runs automatically: installs are explicit, cancellable background jobs
with a streaming log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

IS_WINDOWS = sys.platform.startswith("win")

def get_settings():
    from .settings import get_settings as _get
    return _get()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = PROJECT_ROOT / "envs"
VENV_DIR = ENVS_DIR / "extras"
WHEELS_DIR = ENVS_DIR / "wheels"
NUNCHAKU_REPO = "tonera/vitoom-nunchaku"
ORBITQUANT_REQ = "orbitquant[hf,kernels]>=0.6.0"
_WHEEL_CACHE = PROJECT_ROOT / "cache" / "extras_wheels.json"

# Used when the wheel index can't be reached and no plan can be computed.
FALLBACK_TORCH = "2.12"
FALLBACK_CUDA_TAG = "cu130"

_WHEEL_RE = re.compile(r"^nunchaku-([^+]+)\+cu([\d.]+)torch([\d.]+)-(cp\d+)-cp\d+-([\w]+)\.whl$")


# ------------------------------------------------------------
# Environment discovery
# ------------------------------------------------------------
def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def extras_python() -> Optional[Path]:
    """python executable used for OrbitQuant/Nunchaku jobs (override > managed venv)."""
    override = (get_settings().extras_python or "").strip().strip('"')
    if override:
        p = Path(override)
        return p if p.is_file() else None
    p = venv_python()
    return p if p.is_file() else None


_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0], "py_tag": "cp%d%d" % sys.version_info[:2]}
def _try(name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = None
        out[name + "_error"] = type(e).__name__ + ": " + str(e)[:200]
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["sm"] = "%d%d" % torch.cuda.get_device_capability(0)
except Exception as e:
    out["torch"] = None
    out["torch_error"] = type(e).__name__ + ": " + str(e)[:200]
def _diff():
    import diffusers
    v = diffusers.__version__
    out["has_flux2_klein"] = hasattr(diffusers, "Flux2KleinPipeline")
    out["has_diffusers_flux3"] = hasattr(diffusers, "Flux3Transformer2DModel")
    out["has_modular_pipeline"] = hasattr(diffusers, "ModularPipeline")
    out["has_zimage"] = hasattr(diffusers, "ZImagePipeline")
    return v
_try("diffusers", _diff)
def _nun():
    import nunchaku
    try:
        from importlib.metadata import version
        v = version("nunchaku")
    except Exception:
        v = getattr(nunchaku, "__version__", getattr(nunchaku, "version", "installed"))
    out["has_flux3"] = hasattr(nunchaku, "NunchakuFlux3Transformer2DModel")
    try:
        from nunchaku.utils import get_precision
        out["nunchaku_precision"] = get_precision()
    except Exception as e:
        out["nunchaku_precision_error"] = str(e)[:120]
    return v or "installed"
_try("nunchaku", _nun)
def _orb():
    import orbitquant
    try:
        from importlib.metadata import version
        v = version("orbitquant")
    except Exception:
        v = getattr(orbitquant, "__version__", getattr(orbitquant, "version", "installed"))
    return v or "installed"
_try("orbitquant", _orb)
print("__PROBE__" + json.dumps(out))
"""

_probe_cache: dict = {"key": None, "t": 0.0, "value": None}


def inspect_env(force: bool = False) -> dict:
    """Probe the extras python (cached ~60s). Never raises."""
    py = extras_python()
    base = {
        "python_path": str(py) if py else None,
        "managed": not bool((get_settings().extras_python or "").strip()),
        "venv_exists": venv_python().is_file(),
        "ready": False,
        "python": None,
        "python_version": None,
        "torch": None,
        "torch_version": None,
        "cuda": None,
        "cuda_available": False,
        "device": None,
        "diffusers": None,
        "has_flux2_klein": False,
        "has_diffusers_flux3": False,
        "has_modular_pipeline": False,
        "has_zimage": False,
        "nunchaku": None,
        "nunchaku_precision": None,
        "orbitquant": None,
        "py_tag": None,
        "error": None,
        "summary": "Изолированное окружение (envs/extras) пока не создано",
    }
    if not py:
        base["error"] = "Extras environment not installed"
        base["summary"] = "Изолированное окружение (envs/extras) пока не создано. Создаётся автоматически при установке Nunchaku или OrbitQuant."
        return base

    key = str(py)
    if not force and _probe_cache["key"] == key and time.time() - _probe_cache["t"] < 60 and _probe_cache["value"]:
        return dict(_probe_cache["value"])

    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
        res = subprocess.run([str(py), "-c", _PROBE], capture_output=True, text=True, timeout=120,
                             creationflags=flags)
        line = next((l for l in res.stdout.splitlines() if l.startswith("__PROBE__")), None)
        if not line:
            base["error"] = (res.stderr or res.stdout or "probe produced no output").strip()[-300:]
            base["summary"] = f"Ошибка опроса: {base['error']}"
        else:
            data = json.loads(line[len("__PROBE__"):])
            base.update({k: v for k, v in data.items() if v is not None or k in base})
            base["torch_version"] = base.get("torch")
            base["python_version"] = base.get("python")
            base["ready"] = bool(data.get("torch") and data.get("diffusers") and data.get("cuda_available"))
            if not base["ready"]:
                base["error"] = data.get("torch_error") or (
                    "CUDA is not available in this environment" if data.get("torch") else "torch/diffusers missing")
                base["summary"] = f"Окружение не готово: {base['error']}"
            else:
                base["summary"] = f"Готово: Python {base.get('python')} · PyTorch {base.get('torch')} · CUDA {base.get('cuda')} ({base.get('device', 'GPU')})"
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"
        base["summary"] = f"Ошибка окружения: {base['error']}"

    _probe_cache.update(key=key, t=time.time(), value=dict(base))
    return base


def invalidate_probe() -> None:
    _probe_cache.update(key=None, t=0.0, value=None)


def is_orbitquant_ready() -> bool:
    e = inspect_env()
    return bool(e["ready"] and e["orbitquant"])


def is_nunchaku_ready() -> bool:
    e = inspect_env()
    return bool(e["ready"] and e["nunchaku"])


# ------------------------------------------------------------
# Nunchaku wheel planning
# ------------------------------------------------------------
def _version_key(ver: str):
    nums = tuple(int(x) for x in re.findall(r"\d+", ver.split(".dev")[0])[:3])
    is_dev = ".dev" in ver
    dev = int(re.search(r"dev(\d+)", ver).group(1)) if is_dev and re.search(r"dev(\d+)", ver) else 0
    return (nums, 0 if is_dev else 1, dev)


def _platform_tag() -> str:
    import platform
    if IS_WINDOWS:
        return "win_amd64"
    return "linux_aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "linux_x86_64"


def list_wheel_files(refresh: bool = False) -> tuple[list[str], Optional[str]]:
    """Wheel filenames in the vitoom-nunchaku repo (cached on disk for offline use)."""
    err = None
    if not refresh and _WHEEL_CACHE.is_file() and time.time() - _WHEEL_CACHE.stat().st_mtime < 6 * 3600:
        try:
            return json.loads(_WHEEL_CACHE.read_text(encoding="utf-8")), None
        except Exception:
            pass
    try:
        from huggingface_hub import HfApi
        files = [f for f in HfApi(token=get_settings().huggingface_token or None).list_repo_files(NUNCHAKU_REPO)
                 if f.endswith(".whl")]
        _WHEEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _WHEEL_CACHE.write_text(json.dumps(files), encoding="utf-8")
        return files, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    if _WHEEL_CACHE.is_file():
        try:
            return json.loads(_WHEEL_CACHE.read_text(encoding="utf-8")), err
        except Exception:
            pass
    return [], err


def plan_nunchaku_wheel(py_tag: Optional[str] = None, files: Optional[list[str]] = None) -> dict:
    """Pick the best wheel for this OS + python. Pure function when `files` is given."""
    if py_tag is None:
        env = inspect_env() if extras_python() else {}
        py_tag = env.get("py_tag") or f"cp{sys.version_info.major}{sys.version_info.minor}"
    list_err = None
    if files is None:
        files, list_err = list_wheel_files()

    plat = _platform_tag()
    cands = []
    for f in files:
        m = _WHEEL_RE.match(f)
        if not m:
            continue
        ver, cu, torch_v, py, p = m.groups()
        if py == py_tag and p == plat:
            cands.append((f, ver, cu, torch_v))
    if not cands:
        return {
            "wheel": None, "py_tag": py_tag, "platform": plat,
            "torch_pin": FALLBACK_TORCH, "cuda_tag": FALLBACK_CUDA_TAG,
            "reason": (f"No Nunchaku wheel for {plat} + {py_tag}" +
                       (f" ({list_err})" if list_err else ". Use Python 3.11-3.13 on Windows."))
        }
    cands.sort(key=lambda c: (_version_key(c[1]), tuple(int(x) for x in c[3].split(".")), float(c[2])), reverse=True)
    f, ver, cu, torch_v = cands[0]
    return {
        "wheel": f, "version": ver, "py_tag": py_tag, "platform": plat,
        "torch_pin": torch_v, "cuda_tag": "cu" + cu.replace(".", ""),
        "reason": f"Nunchaku {ver} needs torch {torch_v} + CUDA {cu}",
    }


# ------------------------------------------------------------
# Install jobs
# ------------------------------------------------------------
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_PROCS: dict[str, subprocess.Popen] = {}
_MAX_LOG = 500


def _log(job: dict, line: str) -> None:
    job["log"].append(line.rstrip())
    if len(job["log"]) > _MAX_LOG:
        del job["log"][: len(job["log"]) - _MAX_LOG]


class _Cancelled(Exception):
    pass


def _run(job: dict, cmd: list[str], env: Optional[dict] = None) -> int:
    if job["cancel"].is_set():
        raise _Cancelled()
    _log(job, "$ " + " ".join(cmd))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
    hf_tok = get_settings().huggingface_token
    extra_vars = {}
    if hf_tok:
        extra_vars["HF_TOKEN"] = hf_tok
        extra_vars["HUGGING_FACE_HUB_TOKEN"] = hf_tok
    full_env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONIOENCODING": "utf-8", **extra_vars, **(env or {})}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=full_env, creationflags=flags)
    _PROCS[job["id"]] = proc
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            _log(job, line)
            if job["cancel"].is_set():
                proc.terminate()
                raise _Cancelled()
        return proc.wait()
    finally:
        _PROCS.pop(job["id"], None)


def _step(job: dict, label: str) -> None:
    job["step"] += 1
    job["step_label"] = label
    _log(job, f"=== [{job['step']}/{job['steps_total']}] {label} ===")
    print(f"[extras-install] [{job['step']}/{job['steps_total']}] {label}...", flush=True)


def _pip(job: dict, py: Path, *args: str) -> None:
    if _run(job, [str(py), "-m", "pip", *args]) != 0:
        raise RuntimeError(f"pip {' '.join(args[:3])} ... failed (see log)")


def _ensure_runtime(job: dict, plan: dict) -> Path:
    """Create the venv + torch + diffusers stack if the extras env isn't ready."""
    if (get_settings().extras_python or "").strip():
        py = extras_python()
        if not py:
            raise RuntimeError("settings.extras_python points to a missing file")
        return py

    py = venv_python()
    if not py.is_file():
        _step(job, "Creating virtual environment")
        ENVS_DIR.mkdir(parents=True, exist_ok=True)
        if _run(job, [sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
            raise RuntimeError("venv creation failed")
        _pip(job, py, "install", "--upgrade", "pip", "wheel", "setuptools")

    env = inspect_env(force=True)
    if not env.get("torch") or not env.get("cuda_available"):
        pin = plan.get('torch_pin')
        cu = plan.get('cuda_tag') or "cu128"
        _step(job, f"Installing PyTorch ({pin}, {cu})")
        idx = f"https://download.pytorch.org/whl/{cu}"
        code = _run(job, [str(py), "-m", "pip", "install", f"torch=={pin}.*", "torchvision", "--index-url", idx])
        if code != 0:
            _log(job, f"Exact torch=={pin}.* not found in {idx}; attempting latest compatible PyTorch with {cu}...")
            code2 = _run(job, [str(py), "-m", "pip", "install", "torch", "torchvision", "--index-url", idx])
            if code2 != 0:
                _run(job, [str(py), "-m", "pip", "install", "torch", "torchvision"])
    if not env.get("diffusers"):
        _step(job, "Installing diffusers / transformers / accelerate / peft")
        _pip(job, py, "install", "diffusers", "transformers", "accelerate", "safetensors",
             "sentencepiece", "protobuf", "huggingface_hub", "pillow", "numpy", "peft", "pydantic")
    env = inspect_env(force=True)
    needs_diffusers_main = (
        not env.get("has_flux2_klein")
        or not env.get("has_diffusers_flux3")
        or not env.get("has_modular_pipeline")
    )
    if env.get("diffusers") and needs_diffusers_main:
        # FLUX.2/FLUX.3 support is not in the latest stable PyPI release yet.
        # Install diffusers from GitHub main; the managed env already has the runtime deps.
        _log(job, "diffusers lacks FLUX.3/ModularPipeline support; installing GitHub main (needs git)...")
        if _run(job, [str(py), "-m", "pip", "install", "--upgrade",
                      "git+https://github.com/huggingface/diffusers.git"]) != 0:
            _log(job, "WARNING: could not install diffusers from git. FLUX.3 Nunchaku models need diffusers main.")
        else:
            env = inspect_env(force=True)
            _log(job, f"diffusers is now {env.get('diffusers')} "
                      f"(flux3={env.get('has_diffusers_flux3')}, modular={env.get('has_modular_pipeline')})")
    invalidate_probe()
    return py


def _count_steps(component: str, need_runtime: bool) -> int:
    n = 3 if need_runtime else 0
    return n + {"runtime": 1, "nunchaku": 3, "orbitquant": 2}.get(component, 1)


def _job_main(job: dict, component: str) -> None:
    try:
        plan = plan_nunchaku_wheel()
        job["plan"] = plan
        if component == "nunchaku" and not plan.get("wheel"):
            raise RuntimeError(plan["reason"])

        job["steps_total"] = _count_steps(component, need_runtime=not inspect_env().get("ready"))
        job["state"] = "running"
        py = _ensure_runtime(job, plan)

        if component == "nunchaku":
            env_cur = inspect_env()
            req_torch = plan.get("torch_pin")
            cur_torch = env_cur.get("torch_version") or env_cur.get("torch") or ""
            if req_torch and not str(cur_torch).startswith(str(req_torch)):
                cu = plan.get("cuda_tag") or "cu130"
                _step(job, f"Upgrading PyTorch to {req_torch} ({cu}) for Nunchaku {plan.get('version')}")
                idx = f"https://download.pytorch.org/whl/{cu}"
                _run(job, [str(py), "-m", "pip", "install", f"torch=={req_torch}.*", "torchvision", "--index-url", idx])

            _step(job, f"Downloading {plan['wheel']}")
            from huggingface_hub import hf_hub_download
            WHEELS_DIR.mkdir(parents=True, exist_ok=True)
            wheel = hf_hub_download(NUNCHAKU_REPO, plan["wheel"], local_dir=str(WHEELS_DIR),
                                    token=get_settings().huggingface_token or None)
            _step(job, "Installing Nunchaku wheel")
            install_args = ["install"]
            if job.get("force"):
                install_args += ["--force-reinstall", "--no-deps"]
            install_args.append(wheel)
            _pip(job, py, *install_args)
            _step(job, "Verifying")
            invalidate_probe()
            env = inspect_env(force=True)
            if not env.get("nunchaku"):
                raise RuntimeError("Nunchaku installed but import failed: " + str(env.get("nunchaku_error")))
            _log(job, f"OK: nunchaku {env['nunchaku']} (precision {env.get('nunchaku_precision')})")

        elif component == "orbitquant":
            _step(job, "Installing OrbitQuant")
            if job.get("force"):
                _pip(job, py, "install", "--upgrade", ORBITQUANT_REQ)
            else:
                _pip(job, py, "install", ORBITQUANT_REQ)
            _step(job, "Provisioning native kernels (optional)")
            cli = py.parent / ("orbitquant.exe" if IS_WINDOWS else "orbitquant")
            if cli.is_file():
                if _run(job, [str(cli), "kernels-install"]) != 0:
                    _log(job, "Kernel provisioning failed; OrbitQuant will fall back to the Triton path.")
            invalidate_probe()
            env = inspect_env(force=True)
            if not env.get("orbitquant"):
                raise RuntimeError("OrbitQuant installed but import failed: " + str(env.get("orbitquant_error")))
            _log(job, f"OK: orbitquant {env['orbitquant']}")

        else:  # runtime only
            _step(job, "Verifying")
            env = inspect_env(force=True)
            if not env.get("ready"):
                raise RuntimeError(env.get("error") or "Runtime not ready")
            _log(job, f"OK: torch {env['torch']} (CUDA {env['cuda']}), diffusers {env['diffusers']}")

        job["state"] = "done"
        print(f"[extras-install] Component '{component}' installed successfully [OK]", flush=True)
    except _Cancelled:
        job["state"] = "cancelled"
        _log(job, "Cancelled by user")
        print(f"[extras-install] Component '{component}' installation cancelled", flush=True)
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        _log(job, "ERROR: " + job["error"])
        print(f"[extras-install] Component '{component}' failed: {job['error']}", flush=True)
    finally:
        job["ended"] = time.time()
        invalidate_probe()


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("cancel",)}


def start_install(component: str, force: bool = False) -> dict:
    comp = str(component or "").lower().strip()
    if comp not in ("runtime", "nunchaku", "orbitquant"):
        raise ValueError(f"Неизвестный компонент '{comp}'. Допустимы: runtime | nunchaku | orbitquant")
    with _JOBS_LOCK:
        if any(j["state"] in ("queued", "running") for j in _JOBS.values()):
            raise RuntimeError("Другая установка уже выполняется в фоне")
        job = {
            "id": uuid.uuid4().hex[:8], "component": comp, "state": "queued", "step": 0,
            "steps_total": 1, "step_label": "Queued", "log": [], "error": None,
            "started": time.time(), "ended": None, "plan": None, "force": bool(force),
            "cancel": threading.Event(),
        }
        _JOBS[job["id"]] = job
    threading.Thread(target=_job_main, args=(job, comp), daemon=True).start()
    return _public(job)


def uninstall_component(component: str) -> dict:
    """Uninstall Nunchaku or OrbitQuant from the managed extras venv.

    The shared torch/diffusers runtime is kept: the other quantized engine may
    still need it. External environments are deliberately not modified
    automatically because they are owned by the user.
    """
    comp = str(component or "").lower().strip()
    if comp not in ("nunchaku", "orbitquant"):
        raise ValueError(f"Неизвестный компонент '{comp}'. Допустимы: nunchaku | orbitquant")
    if (get_settings().extras_python or "").strip():
        raise RuntimeError(
            "Extras Python указывает на внешнее окружение. Удалите пакет вручную "
            "или очистите путь Extras Python, чтобы использовать управляемое окружение."
        )
    with _JOBS_LOCK:
        if any(j["state"] in ("queued", "running") for j in _JOBS.values()):
            raise RuntimeError("Дождитесь завершения текущей установки перед удалением")
    py = venv_python()
    if not py.is_file():
        invalidate_probe()
        return {
            "engine": comp,
            "removed": False,
            "message": "Управляемое окружение Extras не создано — удалять нечего",
        }

    package = "nunchaku" if comp == "nunchaku" else "orbitquant"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
    proc = subprocess.run(
        [str(py), "-m", "pip", "uninstall", "-y", package],
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=flags,
    )
    invalidate_probe()
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()[-2000:]
    if proc.returncode != 0:
        raise RuntimeError(f"pip uninstall {package} failed: {output or proc.returncode}")
    return {
        "engine": comp,
        "removed": True,
        "package": package,
        "output": output,
        "message": f"{comp} удалён из окружения Extras",
    }


def get_job(job_id: str) -> Optional[dict]:
    j = _JOBS.get(job_id)
    return _public(j) if j else None


def cancel_job(job_id: str) -> bool:
    j = _JOBS.get(job_id)
    if not j or j["state"] not in ("queued", "running"):
        return False
    j["cancel"].set()
    p = _PROCS.get(job_id)
    if p and p.poll() is None:
        try:
            p.terminate()
        except Exception:
            pass
    return True


def latest_job() -> Optional[dict]:
    if not _JOBS:
        return None
    return _public(max(_JOBS.values(), key=lambda j: j["started"]))


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
def register_extras_routes(app) -> None:
    import asyncio

    from fastapi import HTTPException

    @app.get("/api/extras/status")
    async def extras_status(refresh: bool = False):
        env = await asyncio.to_thread(inspect_env, refresh)
        plan = await asyncio.to_thread(plan_nunchaku_wheel)
        return {
            "env": env,
            "nunchaku_ready": bool(env.get("nunchaku")),
            "orbitquant_ready": bool(env.get("orbitquant")),
            "nunchaku_plan": plan,
            "job": latest_job(),
        }

    @app.post("/api/extras/install")
    async def extras_install(body: dict = None):
        body = body or {}
        comp = str(body.get("component") or body.get("target") or body.get("pkg") or "").strip().lower()
        if not comp:
            raise HTTPException(400, "Параметр 'component' или 'target' обязателен (runtime | nunchaku | orbitquant)")
        try:
            job = start_install(comp, force=bool(body.get("force", False)))
            return {"status": "ok", "job": job, "id": job["id"], "state": job["state"], **job}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/extras/jobs/{job_id}")
    async def extras_job(job_id: str):
        j = get_job(job_id)
        if not j:
            raise HTTPException(404, "Job not found")
        return {"job": j, "id": j["id"], "state": j["state"], "status": j["state"], **j}

    @app.post("/api/extras/jobs/{job_id}/cancel")
    async def extras_job_cancel(job_id: str):
        return {"cancelled": cancel_job(job_id)}

    @app.post("/api/extras/python")
    async def extras_set_python(body: dict):
        """Point the extras runtime at an existing python.exe (or clear with empty path)."""
        path = str((body or {}).get("path") or "").strip().strip('"')
        if path:
            p = Path(path)
            if not p.is_file() or "python" not in p.name.lower():
                raise HTTPException(400, "Path must be an existing python executable")
        get_settings().update(extras_python=path or None)
        invalidate_probe()
        return {"env": await asyncio.to_thread(inspect_env, True)}


# ------------------------------------------------------------
# Extras Worker (Nunchaku vitoom + OrbitQuant inference + LoRA)
# ------------------------------------------------------------
_STEP_RE = re.compile(r"\[step\s+(\d+)/(\d+)\]")
_STATUS_RE = re.compile(r"\[status\]\s*(.+)")
_ERROR_RE = re.compile(r"\[error\]\s*(.+)")


class ExtrasWorker:
    """Executes Nunchaku (tonera/vitoom-nunchaku) and OrbitQuant inference in the managed extras venv."""

    def __init__(self):
        self._cancel_flag = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self):
        self._cancel_flag.set()
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def generate(self, request, model, gpu_info, progress_callback=None):
        import tempfile
        from .cache import CacheManager
        from .security import validate_filename, validate_int, validate_prompt, validate_resolution
        from .worker import GenerateResult, JobStatus, ProgressEvent

        self._cancel_flag.clear()

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        engine = getattr(model, "engine", "nunchaku")
        py = extras_python()
        if not py or not py.is_file():
            raise RuntimeError(
                f"Extras runtime environment for {engine.upper()} is not installed. "
                "Please install it in Settings → Engines."
            )

        emit(ProgressEvent(status=JobStatus.INITIALIZING, message=f"Validating inputs for {engine.upper()}..."))
        prompt = validate_prompt(request.prompt)
        if not prompt:
            raise ValueError("Prompt is empty after validation")

        seed = request.seed if request.seed is not None and request.seed >= 0 else int(time.time()) & 0x7FFFFFFF
        seed = validate_int(seed, "seed", 0, 2**31 - 1)
        width, height = validate_resolution(request.width, request.height)

        settings = get_settings()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = request.output_filename or f"{engine}_{timestamp}_{seed}.png"
        validate_filename(out_filename)
        output_path = settings.output_dir / out_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cache = CacheManager()
        primary_repo = model.sources.primary.repo_id if model.sources else ""
        repo_dir = cache.repo_dir_for(primary_repo) if primary_repo else settings.models_dir

        # Resolve transformer weight file
        tr_rel = model.engine_args.get("diffusion_model") if getattr(model, "engine_args", None) else None
        variant_id = getattr(request, "variant_id", None)
        for t in getattr(model, "transformers", []):
            if variant_id and t.id == variant_id:
                tr_rel = t.path
                break
            if not tr_rel and t.recommended:
                tr_rel = t.path
        if not tr_rel and getattr(model, "transformers", None):
            tr_rel = model.transformers[0].path

        tr_path = (repo_dir / tr_rel) if tr_rel else repo_dir
        if tr_rel and not tr_path.exists():
            # Fallback: look for any svdq-*.safetensors in repo_dir for nunchaku
            cands = list(repo_dir.glob("*.safetensors")) if repo_dir.exists() else []
            if cands:
                tr_path = cands[0]

        ref_image_path = None
        if getattr(request, "init_image_path", None):
            rp = Path(request.init_image_path)
            if not rp.is_absolute():
                rp = settings.output_dir / rp
            if rp.exists():
                ref_image_path = str(rp)

        # ----- Inpaint mask preparation (crop-to-mask) -----
        # The engine only sees a crop around the painted area, so masked runs
        # cost a fraction of a full-frame generation.  The raw mask is kept for
        # the guaranteed compositing; the engine receives the cropped, grown
        # mask (white = edit).
        mask_source = getattr(request, "mask_data_url", None) or (
            str(request.mask_path) if getattr(request, "mask_path", None) else None
        )
        mask_ctx = None
        if mask_source:
            if not ref_image_path:
                raise ValueError(
                    "Mask editing requires a source image (init_image_path is missing)"
                )
            from PIL import Image
            from .masking import (
                MaskError,
                compute_crop_box,
                crop_array,
                crop_image,
                mask_file_name,
                prepare_mask_arrays,
                save_image_file,
                save_mask_file,
            )

            with Image.open(ref_image_path) as src_file:
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

            crop_box = compute_crop_box(engine_mask)
            if crop_box is None:
                raise ValueError("Mask is empty - paint the area to edit")
            crop_w = crop_box[2] - crop_box[0]
            crop_h = crop_box[3] - crop_box[1]
            init_crop = crop_image(source_image, crop_box)
            raw_crop = crop_array(raw_mask, crop_box)
            engine_crop = crop_array(engine_mask, crop_box)

            mask_file = save_mask_file(
                engine_crop,
                settings.output_dir / "masks",
                mask_file_name(f"{engine}_mask"),
            )
            init_file = save_image_file(
                init_crop,
                settings.output_dir / "masks",
                mask_file_name(f"{engine}_init_{seed}"),
            )
            mask_ctx = {
                "array": raw_mask,
                "crop_array": raw_crop,
                "box": crop_box,
                "crop_size": (crop_w, crop_h),
                "file": mask_file,
                "init_file": init_file,
                "source_image": source_image,
            }
            emit(ProgressEvent(
                status=JobStatus.INITIALIZING,
                message=(
                    f"Mask ready: {crop_w}x{crop_h} crop "
                    f"at ({crop_box[0]},{crop_box[1]}) of "
                    f"{source_image.size[0]}x{source_image.size[1]}"
                ),
            ))

        # ``mask_strength`` may legitimately be 0.0 (degenerate "no change"
        # request), so ``or`` must NOT be used here — it would turn 0 into 1.
        raw_mask_strength = getattr(request, "mask_strength", None)
        mask_strength_value = 1.0 if raw_mask_strength is None else float(raw_mask_strength)

        lora_paths = [str(p) for p in (getattr(request, "lora_paths", None) or [])]
        lora_strengths = [float(s) for s in (getattr(request, "lora_strengths", None) or [])]
        steps = int(request.steps or 4)
        cfg = float(getattr(request, "cfg", None) if getattr(request, "cfg", None) is not None else model.get_default_cfg())

        # Resolve Text Encoder (Priority: custom path -> model text_encoders / author shared repo -> repo local)
        te_path = None
        custom_te = getattr(request, "custom_text_encoder_path", None)
        if custom_te and Path(custom_te).exists():
            te_path = str(Path(custom_te))
        else:
            te_id = getattr(request, "text_encoder_id", None)
            te_spec = model.get_text_encoder(te_id) if hasattr(model, "get_text_encoder") else None
            if not te_spec and getattr(model, "text_encoders", None):
                te_spec = next((t for t in model.text_encoders if t.get("recommended")), model.text_encoders[0])
            if te_spec:
                source = te_spec.get("source", {})
                te_repo = source.get("repo_id") if isinstance(source, dict) else None
                te_file = te_spec.get("file", "")
                if te_repo and te_file:
                    shared_p = cache.repo_dir_for(te_repo) / te_file
                    if shared_p.exists():
                        te_path = str(shared_p)
                    else:
                        # Auto-download into shared author repository folder (zero duplication!)
                        emit(ProgressEvent(
                            status=JobStatus.DOWNLOADING,
                            progress=0.15,
                            message=f"Downloading shared text encoder ({te_file})...",
                        ))
                        from .sources import get_source
                        src_inst = get_source("huggingface", token=settings.huggingface_token)
                        shared_dir = cache.repo_dir_for(te_repo)
                        shared_dir.mkdir(parents=True, exist_ok=True)
                        src_inst.download_file(
                            repo_id=te_repo,
                            file_path=source.get("file_path", te_file),
                            local_dir=shared_dir,
                            revision="main",
                        )
                        te_path = str(shared_p)
                elif te_file:
                    cand = repo_dir / te_file
                    if cand.exists():
                        te_path = str(cand)

        # Base model fallback
        base_model_repo = getattr(model, "base_repo_id", None)
        if not base_model_repo and getattr(model, "base_models", None):
            gpu_tier = gpu_info.tier if gpu_info else "50x-above"
            base_model_repo = model.base_models.get(gpu_tier) or model.base_models.get("50x-above") or model.base_models.get("50x-below")

        payload = {
            "engine": engine,
            "family": getattr(model, "family", "klein"),
            "repo_id": primary_repo,
            "model_dir": str(repo_dir),
            "transformer_path": str(tr_path),
            "text_encoder_path": te_path,
            "base_model_repo": base_model_repo,
            "prompt": prompt,
            "negative_prompt": getattr(request, "negative_prompt", None),
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "width": mask_ctx["crop_size"][0] if mask_ctx else width,
            "height": mask_ctx["crop_size"][1] if mask_ctx else height,
            "output_path": str(output_path),
            "init_image_path": str(mask_ctx["init_file"]) if mask_ctx else ref_image_path,
            "mask_path": str(mask_ctx["file"]) if mask_ctx else None,
            "mask_strength": mask_strength_value,
            "lora_paths": lora_paths,
            "lora_strengths": lora_strengths,
        }

        fd, job_json_path = tempfile.mkstemp(prefix=f"{engine}_job_", suffix=".json")
        os.close(fd)
        Path(job_json_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONIOENCODING"] = "utf-8"
        if settings.huggingface_token:
            env["HF_TOKEN"] = settings.huggingface_token
            env["HUGGING_FACE_HUB_TOKEN"] = settings.huggingface_token

        cmd = [str(py), "-u", "-m", "backend.extras", "--run-job", job_json_path]
        start_time = time.time()
        sampler = getattr(request, "sampler", None) or "euler"
        eta = float(getattr(request, "noise", 0.0) or 0.0)
        print(f"[INFO] [SVDQ] Launching {engine.upper()}: {width}x{height}, steps={steps}, cfg={cfg}, sampler={sampler}, eta={eta}, seed={seed}", flush=True)
        emit(ProgressEvent(status=JobStatus.LOADING, progress=0.05, message=f"Starting {engine.upper()} pipeline..."))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._current_process = proc
        captured_lines: list[str] = []

        try:
            for line in proc.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                captured_lines.append(line_str)
                if self._cancel_flag.is_set():
                    proc.terminate()
                    raise RuntimeError("Generation cancelled by user")

                m_step = _STEP_RE.search(line_str)
                if m_step:
                    cur_s, tot_s = int(m_step.group(1)), int(m_step.group(2))
                    pct = min(0.95, max(0.10, cur_s / max(tot_s, 1)))
                    emit(ProgressEvent(
                        status=JobStatus.GENERATING,
                        progress=pct,
                        step=cur_s,
                        total_steps=tot_s,
                        message=f"Sampling step {cur_s}/{tot_s} ({int(pct * 100)}%)",
                    ))
                    continue

                m_status = _STATUS_RE.match(line_str)
                if m_status:
                    emit(ProgressEvent(status=JobStatus.LOADING, progress=0.08, message=m_status.group(1).strip()))
                    continue

                m_err = _ERROR_RE.match(line_str)
                if m_err:
                    raise RuntimeError(f"{engine.upper()} error: {m_err.group(1).strip()}")
            proc.wait()
        finally:
            self._current_process = None
            Path(job_json_path).unlink(missing_ok=True)

        if proc.returncode != 0 or not output_path.exists():
            tail = "\n".join(captured_lines[-40:]) if captured_lines else "(no output)"
            raise RuntimeError(f"{engine.upper()} process exited with code {proc.returncode}.\n{tail}")

        # ----- Mask compositing (guaranteed preservation) -----
        # The engine ran on the crop; paste the composited crop back into the
        # ORIGINAL frame.  Outside the crop the source stays bit-identical.
        raw_fragment_rel = None
        if mask_ctx is not None:
            from PIL import Image
            from .masking import (
                composite_crop_into_image,
                paste_crop_into_image,
            )

            try:
                with Image.open(output_path) as gen_file:
                    generated_rgb = gen_file.convert("RGB")
                source_image = mask_ctx["source_image"]
                composite = composite_crop_into_image(
                    generated_rgb,
                    source_image,
                    mask_ctx["crop_array"],
                    mask_ctx["box"],
                )
                composite.save(output_path, format="PNG")

                frag_dir = settings.output_dir / "masks"
                frag_dir.mkdir(parents=True, exist_ok=True)
                frag_path = frag_dir / f"{output_path.stem}_raw.png"
                paste_crop_into_image(
                    generated_rgb, source_image, mask_ctx["box"]
                ).save(frag_path, format="PNG")
                raw_fragment_rel = f"masks/{frag_path.name}"
                print(
                    f"[extras] Mask composite applied -> {output_path.name} "
                    f"(crop {mask_ctx['crop_size'][0]}x{mask_ctx['crop_size'][1]}, "
                    f"fragment: {raw_fragment_rel})",
                    flush=True,
                )
            except Exception as comp_err:
                raise RuntimeError(f"Mask compositing failed: {comp_err}") from comp_err

        elapsed = time.time() - start_time
        sidecar_path = output_path.with_suffix(".json")
        try:
            meta = {
                "filename": output_path.name,
                "prompt": prompt,
                "negative_prompt": getattr(request, "negative_prompt", None),
                "model_id": request.model_id,
                "variant_id": variant_id,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "backend": engine,
                "engine": engine,
                "loras": [{"name": Path(p).name, "strength": s} for p, s in zip(lora_paths, lora_strengths)],
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
                    "mask_crop": {
                        "box": list(mask_ctx["box"]),
                        "size": list(mask_ctx["crop_size"]),
                    },
                    "mask_raw_fragment": raw_fragment_rel,
                    "composite": "generated*mask + original*(1-mask)",
                })
            else:
                meta["mask_applied"] = False
            sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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
            backend=engine,
            steps=steps,
            prompt=prompt,
        )


def _run_extras_job_cli(job_json_path: str) -> int:
    try:
        import torch
        from PIL import Image

        payload = json.loads(Path(job_json_path).read_text(encoding="utf-8"))
        engine = payload["engine"]
        family = (payload.get("family") or "klein").lower()
        model_dir = Path(payload["model_dir"])
        tr_path = Path(payload["transformer_path"])
        repo_id = payload.get("repo_id") or str(model_dir)
        prompt = payload["prompt"]
        steps = int(payload.get("steps", 4))
        cfg = float(payload.get("cfg", 1.0))
        seed = int(payload.get("seed", 42))
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        output_path = Path(payload["output_path"])
        init_image_path = payload.get("init_image_path")
        mask_path = payload.get("mask_path")
        raw_strength = payload.get("mask_strength", 1.0)
        mask_strength = 1.0 if raw_strength is None else float(raw_strength)
        has_mask = bool(mask_path and Path(mask_path).is_file())
        lora_paths = payload.get("lora_paths") or []
        lora_strengths = payload.get("lora_strengths") or []

        is_flux3 = ("flux3" in family or "flux-3" in family
                    or "flux-3" in str(repo_id).lower() or "flux3" in str(repo_id).lower())
        if has_mask and (is_flux3 or not ("klein" in family or "flux2" in family)):
            raise RuntimeError(
                "Inpaint masks are only wired for FLUX.2 Klein in the extras runtime "
                f"(engine={engine}, family={family})."
            )
        has_local_pipeline = ((model_dir / "model_index.json").exists()
                              or (model_dir / "modular_model_index.json").exists())
        base_source = str(model_dir) if has_local_pipeline else (payload.get("base_model_repo") or repo_id)
        te_path_str = payload.get("text_encoder_path")
        te_path = Path(te_path_str) if te_path_str else None

        te_obj = None
        if te_path and te_path.exists():
            print(f"[status] Loading Text Encoder from {te_path.name}...", flush=True)
            if is_flux3 and ("qwen3-vl" in te_path.name.lower() or "qwen3_vl" in te_path.name.lower()):
                try:
                    from nunchaku.models.text_encoders import NunchakuQwen3VLEncoderModel
                    te_obj = NunchakuQwen3VLEncoderModel.from_pretrained(str(te_path), torch_dtype=torch.bfloat16)
                except Exception as e:
                    print(f"[warning] Could not load NunchakuQwen3VLEncoderModel: {e}", flush=True)
            elif "qwen3-vl" in te_path.name.lower() or "qwen2-vl" in te_path.name.lower():
                try:
                    from nunchaku.models.text_encoders import NunchakuQwen2VLTextEncoderModel
                    te_obj = NunchakuQwen2VLTextEncoderModel.from_pretrained(str(te_path), torch_dtype=torch.bfloat16)
                except Exception as e:
                    print(f"[warning] Could not load NunchakuQwen2VLTextEncoderModel: {e}", flush=True)
            elif "qwen3" in te_path.name.lower():
                try:
                    from nunchaku.models.text_encoders import NunchakuQwen3TextEncoderModel
                    te_obj = NunchakuQwen3TextEncoderModel.from_pretrained(str(te_path), torch_dtype=torch.bfloat16)
                except Exception as e:
                    print(f"[warning] Could not load NunchakuQwen3TextEncoderModel: {e}", flush=True)

        modular_flux3 = False
        if engine == "nunchaku":
            print("[status] Loading Nunchaku (vitoom) SVDQ Transformer...", flush=True)
            if is_flux3:
                import diffusers
                if not hasattr(diffusers, "Flux3Transformer2DModel"):
                    raise RuntimeError(
                        f"FLUX.3 SVDQuant requires diffusers with Flux3Transformer2DModel, but the extras env has "
                        f"diffusers {getattr(diffusers, '__version__', 'unknown')} without it. "
                        "Reinstall/update the Nunchaku extras environment: the app installs diffusers from GitHub main "
                        "for FLUX.3 support (stable PyPI diffusers 0.41.0 does not include it)."
                    )
            import nunchaku
            if "qwen" in family:
                from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as TrCls
            elif "z" in family:
                from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel as TrCls
            elif is_flux3:
                if not hasattr(nunchaku, "NunchakuFlux3Transformer2DModel"):
                    try:
                        import importlib.metadata
                        ver = importlib.metadata.version("nunchaku")
                    except Exception:
                        ver = getattr(nunchaku, "__version__", "unknown")
                    raise RuntimeError(
                        f"NunchakuFlux3Transformer2DModel is not available in the installed Nunchaku package ({ver}). "
                        "FLUX.3 SVDQuant requires Nunchaku >= 1.4.0. Please use 'Flux.2 Klein (Nunchaku)' or update your Nunchaku environment."
                    )
                TrCls = getattr(nunchaku, "NunchakuFlux3Transformer2DModel")
            elif "flux2" in family or "klein" in family:
                from nunchaku import NunchakuFlux2Transformer2DModel as TrCls
            else:
                from nunchaku import NunchakuFluxTransformer2dModel as TrCls

            tr_arg = str(tr_path) if tr_path.is_file() else f"{repo_id}/{tr_path.name}"
            transformer = TrCls.from_pretrained(tr_arg, torch_dtype=torch.bfloat16)

            # Native vitoom-nunchaku LoRA injection (SVDQLoRAMixin)
            if lora_paths:
                print(f"[status] Applying {len(lora_paths)} LoRA adapter(s) to Nunchaku Transformer...", flush=True)
                if len(lora_paths) == 1:
                    s_val = float(lora_strengths[0]) if lora_strengths else 1.0
                    transformer.update_lora_params(str(lora_paths[0]), strength=s_val)
                else:
                    try:
                        from nunchaku.lora.flux.compose import compose_lora
                        pairs = [(str(p), float(s)) for p, s in zip(lora_paths, lora_strengths)]
                        composed_sd = compose_lora(pairs)
                        transformer.update_lora_params(composed_sd, strength=1.0)
                    except Exception:
                        s_val = float(lora_strengths[0]) if lora_strengths else 1.0
                        transformer.update_lora_params(str(lora_paths[0]), strength=s_val)

            print(f"[status] Loading Diffusers pipeline from base '{base_source}'...", flush=True)
            if is_flux3:
                # FLUX.3 ships as a ModularPipeline (see diffusers docs / tonera model card).
                # The prompt upsampler is a separate 9B model which this UI does not manage yet,
                # so drop that block and feed the prompt to the text encoder as-is.
                from diffusers import ModularPipeline
                blocks = ModularPipeline.from_pretrained(base_source).blocks
                if blocks.sub_blocks.pop("prompt_upsampler", None) is not None:
                    print("[status] Prompt upsampler disabled; using the prompt as-is.", flush=True)
                pipe = blocks.init_pipeline(base_source)
                pipe.update_components(transformer=transformer)
                if te_obj is not None:
                    pipe.update_components(text_encoder=te_obj)
                pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
                pipe.to("cuda")
                modular_flux3 = True
            else:
                pipe_kwargs = {
                    "transformer": transformer,
                    "torch_dtype": torch.bfloat16,
                }
                if te_obj is not None:
                    pipe_kwargs["text_encoder"] = te_obj

                if has_mask:
                    from diffusers import Flux2KleinInpaintPipeline
                    pipe = Flux2KleinInpaintPipeline.from_pretrained(base_source, **pipe_kwargs)
                elif "klein" in family or "flux" in family:
                    from diffusers import Flux2KleinPipeline
                    pipe = Flux2KleinPipeline.from_pretrained(base_source, **pipe_kwargs)
                else:
                    from diffusers import DiffusionPipeline
                    pipe = DiffusionPipeline.from_pretrained(base_source, **pipe_kwargs)
                pipe.enable_model_cpu_offload()
        else:
            print(f"[status] Loading OrbitQuant W4A4 pipeline from base '{base_source}'...", flush=True)
            import orbitquant  # noqa: F401
            pipe_kwargs = {
                "torch_dtype": torch.bfloat16,
            }
            if te_obj is not None:
                pipe_kwargs["text_encoder"] = te_obj
            if has_mask:
                from diffusers import Flux2KleinInpaintPipeline
                pipe = Flux2KleinInpaintPipeline.from_pretrained(base_source, **pipe_kwargs)
            else:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(base_source, **pipe_kwargs)
            pipe.enable_model_cpu_offload()

        def on_step_end(pipe_inst, step_idx, timestep, callback_kwargs):
            print(f"[step {step_idx + 1}/{steps}]", flush=True)
            return callback_kwargs

        if modular_flux3:
            if abs(cfg - 1.0) > 0.01:
                print(f"[warning] FLUX.3 distilled ignores cfg={cfg}; using the pipeline default.", flush=True)
            call_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "generator": torch.Generator(device="cpu").manual_seed(seed),
                "output": "images",
            }
            if init_image_path and Path(init_image_path).is_file():
                call_kwargs["image"] = Image.open(init_image_path).convert("RGB")

            print(f"[status] Sampling ({steps} steps, FLUX.3 ModularPipeline)...", flush=True)
            result = pipe(**call_kwargs)
            if isinstance(result, dict):
                images = result.get("images") or []
            elif isinstance(result, (list, tuple)):
                images = result
            else:
                images = [result]
            if not images:
                raise RuntimeError("FLUX.3 pipeline returned no images")
            images[0].save(str(output_path))
        else:
            call_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "generator": torch.Generator(device="cpu").manual_seed(seed),
                "callback_on_step_end": on_step_end,
            }
            if init_image_path and Path(init_image_path).is_file():
                call_kwargs["image"] = Image.open(init_image_path).convert("RGB")
            if has_mask:
                call_kwargs["mask_image"] = Image.open(mask_path).convert("L")
                call_kwargs["strength"] = max(0.01, min(1.0, mask_strength))

            print(f"[status] Sampling ({steps} steps)...", flush=True)
            out = pipe(**call_kwargs).images[0]
            out.save(str(output_path))
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[error] {e}", flush=True)
        return 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-job":
        sys.exit(_run_extras_job_cli(sys.argv[2]))
