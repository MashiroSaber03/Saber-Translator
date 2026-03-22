import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "logs", "system_monitor")


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_cmdline(p: psutil.Process) -> str:
    try:
        parts = p.cmdline()
        return " ".join(parts)[:2000]
    except Exception:
        return ""


def _bytes_to_gb(b: int) -> float:
    return float(b) / (1024.0**3)


def _get_disk_usage(path: str) -> Optional[Dict[str, Any]]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gb": round(_bytes_to_gb(usage.free), 3),
            "used_percent": round(usage.used / usage.total * 100.0, 2) if usage.total else None,
        }
    except Exception:
        return None


def _parse_nvidia_smi_csv(text: str) -> Optional[Dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    gpus: List[Dict[str, Any]] = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5:
            continue
        name, util, mem_used, mem_total, temp = parts[:5]
        def _to_int(v: str) -> Optional[int]:
            try:
                return int(float(v))
            except Exception:
                return None
        gpus.append(
            {
                "name": name,
                "utilization_gpu_percent": _to_int(util),
                "memory_used_mib": _to_int(mem_used),
                "memory_total_mib": _to_int(mem_total),
                "temperature_c": _to_int(temp),
            }
        )
    if not gpus:
        return None
    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
    }


def _get_nvidia_smi() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        parsed = _parse_nvidia_smi_csv(out)
        return {"available": parsed is not None, "data": parsed, "error": None}
    except FileNotFoundError:
        return {"available": False, "data": None, "error": "nvidia-smi not found"}
    except subprocess.CalledProcessError as e:
        return {"available": False, "data": None, "error": (e.output or str(e))[:2000]}
    except Exception as e:
        return {"available": False, "data": None, "error": str(e)[:2000]}


@dataclass(frozen=True)
class ProcAgg:
    name: str
    count: int
    total_rss_bytes: int
    max_rss_bytes: int
    max_rss_pid: Optional[int]
    total_private_bytes: Optional[int]
    max_private_bytes: Optional[int]
    max_private_pid: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_rss_bytes": self.total_rss_bytes,
            "total_rss_gb": round(_bytes_to_gb(self.total_rss_bytes), 3),
            "max_rss_bytes": self.max_rss_bytes,
            "max_rss_gb": round(_bytes_to_gb(self.max_rss_bytes), 3),
            "max_rss_pid": self.max_rss_pid,
            "total_private_bytes": self.total_private_bytes,
            "total_private_gb": round(_bytes_to_gb(self.total_private_bytes), 3) if self.total_private_bytes is not None else None,
            "max_private_bytes": self.max_private_bytes,
            "max_private_gb": round(_bytes_to_gb(self.max_private_bytes), 3) if self.max_private_bytes is not None else None,
            "max_private_pid": self.max_private_pid,
        }


def _aggregate_processes(
    procs: Sequence[psutil.Process],
    group_name: str,
    *,
    exclude_pids: Sequence[int] = (),
) -> ProcAgg:
    total_rss = 0
    max_rss = 0
    max_rss_pid: Optional[int] = None
    total_private: Optional[int] = 0
    max_private: Optional[int] = 0
    max_private_pid: Optional[int] = None

    count = 0
    for p in procs:
        if p.pid in exclude_pids:
            continue
        try:
            mi = p.memory_info()
            rss = int(mi.rss)
        except Exception:
            continue

        count += 1
        total_rss += rss
        if rss > max_rss:
            max_rss = rss
            max_rss_pid = p.pid

        private_bytes: Optional[int]
        try:
            mfi = p.memory_full_info()
            private_bytes = getattr(mfi, "private", None)
        except Exception:
            private_bytes = None

        if private_bytes is None:
            total_private = None
            max_private = None
            max_private_pid = None
        elif total_private is not None:
            total_private += int(private_bytes)
            if max_private is not None and int(private_bytes) > int(max_private):
                max_private = int(private_bytes)
                max_private_pid = p.pid

    return ProcAgg(
        name=group_name,
        count=count,
        total_rss_bytes=total_rss,
        max_rss_bytes=max_rss,
        max_rss_pid=max_rss_pid,
        total_private_bytes=total_private,
        max_private_bytes=max_private,
        max_private_pid=max_private_pid,
    )


def _find_by_name(names: Sequence[str]) -> List[psutil.Process]:
    names_l = {n.lower() for n in names}
    out: List[psutil.Process] = []
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info.get("name") or "").lower() in names_l:
                out.append(p)
        except Exception:
            continue
    return out


def _find_by_cmdline_substrings(substrings: Sequence[str]) -> List[psutil.Process]:
    subs_l = [s.lower() for s in substrings if s]
    if not subs_l:
        return []
    out: List[psutil.Process] = []
    for p in psutil.process_iter():
        try:
            cmd = _safe_cmdline(p).lower()
            if not cmd:
                continue
            if any(s in cmd for s in subs_l):
                out.append(p)
        except Exception:
            continue
    return out


def _open_log_file(out_dir: str, base_name: str, part: int) -> Tuple[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    suffix = f".part{part:02d}" if part > 0 else ""
    path = os.path.join(out_dir, f"{base_name}{suffix}.jsonl")
    f = open(path, "a", encoding="utf-8")
    return path, f


def main() -> int:
    parser = argparse.ArgumentParser(prog="system_monitor", description="系统/进程资源监控（JSONL）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument("--danger-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--track-edge", action="store_true", default=True)
    parser.add_argument("--edge-process-name", default="msedge.exe")
    parser.add_argument("--track-backend", action="store_true", default=True)
    parser.add_argument("--backend-cmd-substring", action="append", default=["app.py"])
    parser.add_argument("--low-mem-gb", type=float, default=2.0)
    parser.add_argument("--edge-max-rss-gb", type=float, default=6.0)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    session_dir = os.path.join(out_dir, datetime.now().strftime("%Y%m%d"))
    base_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    part = 0
    log_path, f = _open_log_file(session_dir, base_name, part)

    try:
        psutil.cpu_percent(interval=None)
        sample_idx = 0
        last_edge_count: Optional[int] = None

        while True:
            now_iso = _iso_now()
            now_unix = time.time()

            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            cpu = psutil.cpu_percent(interval=None)

            system_payload: Dict[str, Any] = {
                "memory": {
                    "total_bytes": int(vm.total),
                    "available_bytes": int(vm.available),
                    "used_bytes": int(vm.used),
                    "percent": float(vm.percent),
                    "available_gb": round(_bytes_to_gb(int(vm.available)), 3),
                    "used_gb": round(_bytes_to_gb(int(vm.used)), 3),
                },
                "pagefile": {
                    "total_bytes": int(sm.total),
                    "used_bytes": int(sm.used),
                    "free_bytes": int(sm.free),
                    "percent": float(sm.percent),
                    "used_gb": round(_bytes_to_gb(int(sm.used)), 3),
                },
                "cpu": {
                    "percent": float(cpu),
                    "logical_count": psutil.cpu_count(logical=True),
                },
            }

            disk_payload = {
                "system_drive": _get_disk_usage(os.environ.get("SystemDrive", "C:") + "\\"),
                "out_dir_drive": _get_disk_usage(out_dir),
            }

            proc_payload: Dict[str, Any] = {}
            exclude_pids = [os.getpid()]

            edge_agg: Optional[ProcAgg] = None
            if args.track_edge:
                edge_procs = _find_by_name([args.edge_process_name])
                edge_agg = _aggregate_processes(edge_procs, "edge", exclude_pids=exclude_pids)
                proc_payload["edge"] = edge_agg.to_dict()

                edge_count = edge_agg.count
                if last_edge_count is not None and last_edge_count > 0 and edge_count == 0:
                    proc_payload["edge_event"] = "disappeared"
                last_edge_count = edge_count

            backend_agg: Optional[ProcAgg] = None
            if args.track_backend:
                backend_procs = _find_by_cmdline_substrings(args.backend_cmd_substring)
                backend_agg = _aggregate_processes(backend_procs, "backend", exclude_pids=exclude_pids)
                backend_payload = backend_agg.to_dict()
                if backend_agg.max_rss_pid is not None:
                    try:
                        p = psutil.Process(backend_agg.max_rss_pid)
                        backend_payload["max_cmdline"] = _safe_cmdline(p)
                    except Exception:
                        pass
                proc_payload["backend"] = backend_payload

            gpu_payload = _get_nvidia_smi()

            record = {
                "ts": now_iso,
                "t": now_unix,
                "sample": sample_idx,
                "system": system_payload,
                "disk": disk_payload,
                "gpu": gpu_payload,
                "process": proc_payload,
                "meta": {
                    "host": os.environ.get("COMPUTERNAME"),
                    "user": os.environ.get("USERNAME"),
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            sample_idx += 1
            if args.max_samples and sample_idx >= args.max_samples:
                break

            danger = False
            if vm.available <= int(args.low_mem_gb * 1024**3):
                danger = True
            if edge_agg is not None and edge_agg.max_rss_bytes >= int(args.edge_max_rss_gb * 1024**3):
                danger = True

            interval = float(args.danger_interval_sec if danger else args.interval_sec)

            max_bytes = int(args.max_file_mb * 1024 * 1024)
            try:
                if f.tell() >= max_bytes:
                    f.close()
                    part += 1
                    log_path, f = _open_log_file(session_dir, base_name, part)
            except Exception:
                pass

            time.sleep(max(0.1, interval))

        return 0
    finally:
        try:
            f.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

