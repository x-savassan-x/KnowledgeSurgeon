import os
import platform
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import save_hardware_results


SAFE_PROVIDER_ENV_KEYS = [
    "VAST_CONTAINERLABEL",
    "VAST_INSTANCE_ID",
    "VAST_MACHINE_ID",
    "VAST_HOST_ID",
    "RUNPOD_POD_ID",
    "RUNPOD_DC_ID",
    "PAPERSPACE_MACHINE_ID",
    "AWS_INSTANCE_ID",
    "GCP_INSTANCE_ID",
]


def _run_command(args: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "command": args,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": completed.returncode == 0,
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _read_text(path: str | Path, limit: int = 100_000) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            return file.read(limit).strip()
    except Exception:
        return None


def _linux_cpu_model() -> str | None:
    cpuinfo = _read_text("/proc/cpuinfo")
    if not cpuinfo:
        return None
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _memory_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    meminfo = _read_text("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "SwapTotal"}:
                parts = value.strip().split()
                if parts:
                    kib = float(parts[0])
                    info[f"{key.lower()}_gb"] = kib / 1024 / 1024
    if "memtotal_gb" not in info:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            info["memtotal_gb"] = pages * page_size / 1024**3
        except Exception:
            pass
    return info


def _disk_info() -> dict[str, Any]:
    usage = shutil.disk_usage(Path.cwd())
    info: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
    }
    lsblk = _run_command(["lsblk", "-J", "-o", "NAME,MODEL,SIZE,TYPE,MOUNTPOINT"], timeout=5)
    if lsblk.get("ok"):
        try:
            info["lsblk"] = json.loads(str(lsblk.get("stdout", "")))
        except Exception:
            info["lsblk_raw"] = lsblk.get("stdout")
    return info


def _parse_nvidia_smi_csv(output: str, fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append(dict(zip(fields, values)))
    return rows


def _nvidia_gpu_info() -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "memory_used_mib",
        "pcie_link_gen_current",
        "pcie_link_width_current",
    ]
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,memory.used,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ]
    result = _run_command(query, timeout=10)
    if not result.get("ok"):
        basic = _run_command(["nvidia-smi"], timeout=10)
        return {
            "available": False,
            "query_error": result,
            "nvidia_smi": basic,
        }

    gpus = _parse_nvidia_smi_csv(str(result.get("stdout", "")), fields)
    for gpu in gpus:
        for key in ("memory_total_mib", "memory_free_mib", "memory_used_mib"):
            try:
                gpu[key] = float(gpu[key])
                gpu[key.replace("_mib", "_gb")] = gpu[key] / 1024
            except Exception:
                pass
        for key in ("index", "pcie_link_gen_current", "pcie_link_width_current"):
            try:
                gpu[key] = int(str(gpu[key]))
            except Exception:
                pass

    topo = _run_command(["nvidia-smi", "topo", "-m"], timeout=10)
    return {
        "available": True,
        "count": len(gpus),
        "gpus": gpus,
        "topology": topo.get("stdout") if topo.get("ok") else None,
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append({
                "index": idx,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })
        info["devices"] = devices
    return info


def _provider_env() -> dict[str, str]:
    return {
        key: value
        for key in SAFE_PROVIDER_ENV_KEYS
        if (value := os.environ.get(key))
    }


def collect_hardware_info(
    notes: str = "",
    manual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": {
            "model": _linux_cpu_model() or platform.processor(),
            "logical_count": os.cpu_count(),
        },
        "memory": _memory_info(),
        "disk": _disk_info(),
        "gpu": _nvidia_gpu_info(),
        "torch": _torch_info(),
        "provider_env": _provider_env(),
    }
    if notes:
        payload["notes"] = notes
    if manual:
        payload["manual"] = manual
    return payload


def save_run_hardware(
    model_key: str,
    milestones: list[int],
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    notes: str = "",
    manual: dict[str, Any] | None = None,
    evaluation_dir: Path | str | None = None,
) -> Path:
    hardware = collect_hardware_info(
        notes=notes,
        manual=manual,
    )
    return save_hardware_results(
        hardware=hardware,
        model_key=model_key,
        milestones=milestones,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
