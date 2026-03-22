import time
from pathlib import Path

from typing import Any, Dict, List

try:
    import pynvml
except Exception:
    pynvml = None

try:
    import docker
except Exception:
    docker = None


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        return ""

def _read_cgroup_first_line(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except Exception:
        return ""

def _docker_id_from_cgroup(line: str) -> str:
    # systemd: /system.slice/docker-<ID>.scope
    if "docker-" in line and ".scope" in line:
        x = line.split("docker-")[-1]
        return x.split(".scope")[0]
    # cgroupfs: /docker/<ID>
    if "/docker/" in line:
        return line.split("/docker/")[-1].split("/")[0]
    return ""

def _procs_to_list(procs, pynvml):
    out = []
    for p in procs or []:
        used = getattr(p, "usedGpuMemory", None)
        used_mib = None if used in (None, pynvml.NVML_VALUE_NOT_AVAILABLE) else used / (1024 * 1024)
        out.append({"pid": int(p.pid), "used_mib": used_mib})
    return out

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default



def get_gpu_metrics() -> Dict[str, Any]:
    if pynvml is None:
        return {"ok": False, "error": "pynvml not installed"}

    docker_client = None
    if docker is not None:
        try:
            docker_client = docker.from_env()
        except Exception:
            docker_client = None

    # cache docker_id -> {name,image} per call (cheap)
    docker_cache: Dict[str, Dict[str, str]] = {}

    def _docker_meta(docker_id: str) -> Dict[str, str]:
        if not docker_id:
            return {"docker_name": "", "docker_image": ""}
        if docker_id in docker_cache:
            return docker_cache[docker_id]
        if docker_client is None:
            docker_cache[docker_id] = {"docker_name": "", "docker_image": ""}
            return docker_cache[docker_id]
        try:
            c = docker_client.containers.get(docker_id)
            meta = {
                "docker_name": (c.name or ""),
                "docker_image": str(getattr(c.image, "tags", [""])[0] if getattr(c, "image", None) else ""),
            }
        except Exception:
            meta = {"docker_name": "", "docker_image": ""}
        docker_cache[docker_id] = meta
        return meta

    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()

        gpus: List[Dict[str, Any]] = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)

            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            power_w = None
            power_limit_w = None
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power_w = None

            try:
                power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                power_limit_w = None

            sm_clock = None
            mem_clock = None
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            except Exception:
                sm_clock = None
            try:
                mem_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            except Exception:
                mem_clock = None

            total_mib = mem.total / (1024 * 1024)
            used_mib = mem.used / (1024 * 1024)
            free_mib = mem.free / (1024 * 1024)

            # --- processes ---
            compute_procs = []
            graphics_procs = []
            try:
                compute_procs = _procs_to_list(pynvml.nvmlDeviceGetComputeRunningProcesses_v2(h), pynvml)
            except Exception:
                compute_procs = []
            try:
                graphics_procs = _procs_to_list(pynvml.nvmlDeviceGetGraphicsRunningProcesses_v2(h), pynvml)
            except Exception:
                graphics_procs = []

            # enrich pids with cmdline + docker
            def _enrich(plist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                out = []
                for p in plist:
                    pid = int(p.get("pid") or 0)
                    cg = _read_cgroup_first_line(pid) if pid else ""
                    did = _docker_id_from_cgroup(cg) if cg else ""
                    meta = _docker_meta(did)
                    out.append({
                        **p,
                        "cmdline": _read_cmdline(pid) if pid else "",
                        "cgroup": cg,
                        "docker_id": did,
                        "docker_name": meta.get("docker_name", ""),
                        "docker_image": meta.get("docker_image", ""),
                    })
                return out

            compute_procs = _enrich(compute_procs)
            graphics_procs = _enrich(graphics_procs)

            gpus.append({
                "index": i,
                "name": name,
                "util_gpu_pct": _safe_int(util.gpu),
                "util_mem_pct": _safe_int(util.memory),
                "mem_total_mib": _safe_float(total_mib),
                "mem_used_mib": _safe_float(used_mib),
                "mem_free_mib": _safe_float(free_mib),
                "temp_c": _safe_int(temp) if temp is not None else None,
                "power_w": _safe_float(power_w) if power_w is not None else None,
                "power_limit_w": _safe_float(power_limit_w) if power_limit_w is not None else None,
                "sm_clock_mhz": _safe_int(sm_clock) if sm_clock is not None else None,
                "mem_clock_mhz": _safe_int(mem_clock) if mem_clock is not None else None,
                "compute_procs": compute_procs,
                "graphics_procs": graphics_procs,
            })

        return {"ok": True, "ts": int(time.time()), "gpus": gpus}

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass