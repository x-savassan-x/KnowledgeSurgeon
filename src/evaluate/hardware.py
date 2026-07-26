import json
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

from .io import save_hardware_results


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


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _compact_cpu_list(cpu_ids: list[int]) -> str:
    if not cpu_ids:
        return ""

    ranges: list[str] = []
    start = previous = cpu_ids[0]
    for cpu_id in cpu_ids[1:]:
        if cpu_id == previous + 1:
            previous = cpu_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _powershell_json(script: str) -> Any:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if not executable:
        return None
    result = _run_command(
        [executable, "-NoProfile", "-NonInteractive", "-Command", script],
        timeout=10,
    )
    if not result.get("ok"):
        return None
    try:
        return json.loads(str(result.get("stdout", "")))
    except (json.JSONDecodeError, TypeError):
        return None


def _cpu_affinity() -> dict[str, Any]:
    try:
        cpu_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return {}
    return {
        "effective_logical_count": len(cpu_ids),
        "affinity": _compact_cpu_list(cpu_ids),
    }


def _cpu_quota_cores() -> float | None:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, _, period = cpu_max.partition(" ")
        if quota != "max":
            quota_value = _as_float(quota)
            period_value = _as_float(period)
            if quota_value is not None and period_value:
                return quota_value / period_value

    quota_value = _as_float(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period_value = _as_float(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota_value is not None and quota_value > 0 and period_value:
        return quota_value / period_value
    return None


def _linux_cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "model": platform.processor() or None,
        "logical_count": os.cpu_count(),
    }
    lscpu = _run_command(["lscpu", "-J"], timeout=5)
    if lscpu.get("ok"):
        try:
            rows = json.loads(str(lscpu.get("stdout", ""))).get("lscpu", [])
        except (json.JSONDecodeError, AttributeError):
            rows = []
        values = {
            str(row.get("field", "")).rstrip(":"): row.get("data")
            for row in rows
            if isinstance(row, dict)
        }
        field_map = {
            "Architecture": "architecture",
            "Vendor ID": "vendor_id",
            "Model name": "model",
            "Socket(s)": "socket_count",
            "Core(s) per socket": "cores_per_socket",
            "Thread(s) per core": "threads_per_core",
            "NUMA node(s)": "numa_node_count",
            "CPU max MHz": "max_mhz",
            "CPU min MHz": "min_mhz",
            "L3 cache": "l3_cache",
        }
        integer_fields = {
            "socket_count",
            "cores_per_socket",
            "threads_per_core",
            "numa_node_count",
        }
        float_fields = {"max_mhz", "min_mhz"}
        for source, target in field_map.items():
            value = values.get(source)
            if value in (None, ""):
                continue
            if target in integer_fields:
                value = _as_int(value)
            elif target in float_fields:
                value = _as_float(value)
            if value is not None:
                info[target] = value

    sockets = _as_int(info.get("socket_count"))
    cores_per_socket = _as_int(info.get("cores_per_socket"))
    if sockets is not None and cores_per_socket is not None:
        info["physical_core_count"] = sockets * cores_per_socket

    info.update(_cpu_affinity())
    quota_cores = _cpu_quota_cores()
    if quota_cores is not None:
        info["quota_cores"] = quota_cores
    return {key: value for key, value in info.items() if value not in (None, "")}


def _windows_cpu_info() -> dict[str, Any]:
    rows = _powershell_json(
        "Get-CimInstance Win32_Processor | "
        "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
        "SocketDesignation,MaxClockSpeed,L3CacheSize | "
        "ConvertTo-Json -Compress"
    )
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        return _linux_cpu_info()

    def values(key: str) -> list[Any]:
        return [row.get(key) for row in rows if isinstance(row, dict) and row.get(key) is not None]

    info: dict[str, Any] = {
        "model": next(iter(values("Name")), platform.processor()),
        "logical_count": sum(_as_int(value) or 0 for value in values("NumberOfLogicalProcessors")),
        "physical_core_count": sum(_as_int(value) or 0 for value in values("NumberOfCores")),
        "socket_count": len({str(value) for value in values("SocketDesignation")}),
    }
    max_clock = max((_as_float(value) or 0 for value in values("MaxClockSpeed")), default=0)
    l3_cache = max((_as_float(value) or 0 for value in values("L3CacheSize")), default=0)
    if max_clock:
        info["max_mhz"] = max_clock
    if l3_cache:
        info["l3_cache"] = f"{l3_cache:g} KB"
    return {key: value for key, value in info.items() if value not in (None, "")}


def _cpu_info() -> dict[str, Any]:
    if platform.system() == "Windows":
        return _windows_cpu_info()
    return _linux_cpu_info()


def _memory_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    meminfo = _read_text("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "SwapTotal"}:
                parts = value.strip().split()
                if parts:
                    kib = float(parts[0])
                    info[f"{key.lower()}_gb"] = kib / 1024 / 1024
    if "memtotal_gb" not in info:
        if platform.system() == "Windows":
            memory = _powershell_json(
                "Get-CimInstance Win32_ComputerSystem | "
                "Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress"
            )
            if isinstance(memory, dict):
                total_bytes = _as_float(memory.get("TotalPhysicalMemory"))
                if total_bytes is not None:
                    info["memtotal_gb"] = total_bytes / 1024**3
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
    info: dict[str, Any] = {"filesystem_total_gb": usage.total / 1024**3}
    lsblk = _run_command(
        ["lsblk", "-J", "-b", "-o", "NAME,MODEL,SIZE,TYPE,ROTA,TRAN"],
        timeout=5,
    )
    if lsblk.get("ok"):
        try:
            blockdevices = json.loads(str(lsblk.get("stdout", ""))).get("blockdevices", [])
            devices = []
            for device in blockdevices:
                if not isinstance(device, dict) or device.get("type") != "disk":
                    continue
                size_bytes = _as_float(device.get("size"))
                devices.append({
                    "name": device.get("name"),
                    "model": str(device.get("model") or "").strip() or None,
                    "size_gb": size_bytes / 1024**3 if size_bytes is not None else None,
                    "rotational": bool(_as_int(device.get("rota"))),
                    "transport": device.get("tran"),
                })
            if devices:
                info["devices"] = [
                    {key: value for key, value in device.items() if value not in (None, "")}
                    for device in devices
                ]
        except Exception:
            pass
    if "devices" not in info and platform.system() == "Windows":
        rows = _powershell_json(
            "Get-CimInstance Win32_DiskDrive | "
            "Select-Object DeviceID,Model,Size,MediaType | ConvertTo-Json -Compress"
        )
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows, list):
            devices = []
            for device in rows:
                if not isinstance(device, dict):
                    continue
                size_bytes = _as_float(device.get("Size"))
                devices.append({
                    "name": device.get("DeviceID"),
                    "model": str(device.get("Model") or "").strip() or None,
                    "size_gb": size_bytes / 1024**3 if size_bytes is not None else None,
                    "media_type": device.get("MediaType"),
                })
            if devices:
                info["devices"] = [
                    {key: value for key, value in device.items() if value not in (None, "")}
                    for device in devices
                ]
    return info


def _parse_nvidia_smi_csv(output: str, fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append({
            field: None if value.lower() in {"", "n/a", "[n/a]", "not supported"} else value
            for field, value in zip(fields, values)
        })
    return rows


def _gpu_numa_node(bus_id: str | None) -> int | None:
    if not bus_id:
        return None
    parts = bus_id.split(":")
    if len(parts) != 3:
        return None
    linux_bus_id = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    numa_node = _as_int(_read_text(f"/sys/bus/pci/devices/{linux_bus_id}/numa_node"))
    return numa_node if numa_node is not None and numa_node >= 0 else None


def _nvidia_gpu_info() -> dict[str, Any]:
    core_fields = [
        "index",
        "name",
        "uuid",
        "pci_bus_id",
        "driver_version",
        "memory_total_mib",
        "pcie_link_gen_max",
        "pcie_link_width_max",
    ]
    optional_fields = [
        "index",
        "power_limit_w",
        "clocks_max_sm_mhz",
        "clocks_max_memory_mhz",
        "compute_mode",
        "mig_mode",
        "ecc_mode",
        "vbios_version",
    ]
    core_query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,"
        "pcie.link.gen.max,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ]
    core_result = _run_command(core_query, timeout=10)
    if not core_result.get("ok"):
        return {"available": False}

    gpus = _parse_nvidia_smi_csv(str(core_result.get("stdout", "")), core_fields)
    optional_query = [
        "nvidia-smi",
        "--query-gpu=index,power.limit,clocks.max.sm,clocks.max.memory,"
        "compute_mode,mig.mode.current,ecc.mode.current,vbios_version",
        "--format=csv,noheader,nounits",
    ]
    optional_result = _run_command(optional_query, timeout=10)
    if optional_result.get("ok"):
        optional_rows = _parse_nvidia_smi_csv(
            str(optional_result.get("stdout", "")),
            optional_fields,
        )
        optional_by_index = {row.get("index"): row for row in optional_rows}
        for gpu in gpus:
            optional = optional_by_index.get(gpu.get("index"), {})
            gpu.update({key: value for key, value in optional.items() if key != "index"})

    for gpu in gpus:
        memory_total_mib = _as_float(gpu.get("memory_total_mib"))
        if memory_total_mib is not None:
            gpu["memory_total_mib"] = memory_total_mib
            gpu["memory_total_gb"] = memory_total_mib / 1024
        for key in (
            "index",
            "pcie_link_gen_max",
            "pcie_link_width_max",
        ):
            gpu[key] = _as_int(gpu.get(key))
        for key in ("power_limit_w", "clocks_max_sm_mhz", "clocks_max_memory_mhz"):
            gpu[key] = _as_float(gpu.get(key))
        numa_node = _gpu_numa_node(gpu.get("pci_bus_id"))
        if numa_node is not None:
            gpu["numa_node"] = numa_node
        cleaned = {key: value for key, value in gpu.items() if value is not None}
        gpu.clear()
        gpu.update(cleaned)

    topo = _run_command(["nvidia-smi", "topo", "-m"], timeout=10)
    topology = topo.get("stdout") if topo.get("ok") else None
    if isinstance(topology, str):
        topology = re.sub(r"\x1b\[[0-9;]*m", "", topology)
    return {
        "available": True,
        "count": len(gpus),
        "gpus": gpus,
        "topology": topology,
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
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


def collect_hardware_info(
    provider_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
        },
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "disk": _disk_info(),
        "gpu": _nvidia_gpu_info(),
        "torch": _torch_info(),
    }
    if provider_specs:
        payload["provider_specs"] = {
            key: value
            for key, value in provider_specs.items()
            if value not in (None, "")
        }
    return payload


def save_run_hardware(
    model_key: str,
    milestones: list[int],
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    provider_specs: dict[str, Any] | None = None,
    evaluation_dir: Path | str | None = None,
) -> Path:
    hardware = collect_hardware_info(
        provider_specs=provider_specs,
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
