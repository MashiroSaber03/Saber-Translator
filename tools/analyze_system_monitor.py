import argparse
import json
import math
import os
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Tuple


def _gb(b: Optional[int]) -> Optional[float]:
    if b is None:
        return None
    return float(b) / (1024.0**3)


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _update_min(best: Tuple[Optional[float], Optional[Dict[str, Any]]], value: Optional[float], rec: Dict[str, Any]):
    cur, _ = best
    if value is None:
        return best
    if cur is None or value < cur:
        return value, rec
    return best


def _update_max(best: Tuple[Optional[float], Optional[Dict[str, Any]]], value: Optional[float], rec: Dict[str, Any]):
    cur, _ = best
    if value is None:
        return best
    if cur is None or value > cur:
        return value, rec
    return best


def _fmt_ts(rec: Optional[Dict[str, Any]]) -> str:
    if not rec:
        return "-"
    return str(rec.get("ts") or "-")


def main() -> int:
    parser = argparse.ArgumentParser(description="解析 system_monitor.jsonl 并输出摘要")
    parser.add_argument("--file", required=True)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--around", default="", help="本地时间 HH:MM，例如 14:37")
    args = parser.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.exists(path):
        raise SystemExit(f"file not found: {path}")

    around_hhmm = args.around.strip()
    around_target: Optional[Tuple[int, int]] = None
    if around_hhmm:
        try:
            hh, mm = around_hhmm.split(":")
            around_target = (int(hh), int(mm))
        except Exception:
            around_target = None

    count = 0
    first_rec: Optional[Dict[str, Any]] = None
    last_rec: Optional[Dict[str, Any]] = None

    min_avail_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_swap_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_edge_max_rss_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_edge_max_private_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_edge_total_rss_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_backend_max_rss_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_backend_max_private_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)
    max_gpu_mem_used_gb: Tuple[Optional[float], Optional[Dict[str, Any]]] = (None, None)

    tail: Deque[Dict[str, Any]] = deque(maxlen=max(1, int(args.tail)))

    around_best: Tuple[Optional[int], Optional[Dict[str, Any]]] = (None, None)
    edge_disappear_at: Optional[Dict[str, Any]] = None
    prev_edge_count: Optional[int] = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            count += 1
            if first_rec is None:
                first_rec = rec
            last_rec = rec
            tail.append(rec)

            avail_b = _get(rec, "system.memory.available_bytes")
            if isinstance(avail_b, (int, float)):
                min_avail_gb = _update_min(min_avail_gb, _gb(int(avail_b)), rec)

            swap_b = _get(rec, "system.pagefile.used_bytes")
            if isinstance(swap_b, (int, float)):
                max_swap_gb = _update_max(max_swap_gb, _gb(int(swap_b)), rec)

            edge_max_rss = _get(rec, "process.edge.max_rss_bytes")
            if isinstance(edge_max_rss, (int, float)):
                max_edge_max_rss_gb = _update_max(max_edge_max_rss_gb, _gb(int(edge_max_rss)), rec)

            edge_total_rss = _get(rec, "process.edge.total_rss_bytes")
            if isinstance(edge_total_rss, (int, float)):
                max_edge_total_rss_gb = _update_max(max_edge_total_rss_gb, _gb(int(edge_total_rss)), rec)

            edge_max_priv = _get(rec, "process.edge.max_private_gb")
            if edge_max_priv is None:
                edge_max_priv_bytes = _get(rec, "process.edge.max_private_bytes")
                if isinstance(edge_max_priv_bytes, (int, float)):
                    max_edge_max_private_gb = _update_max(max_edge_max_private_gb, _gb(int(edge_max_priv_bytes)), rec)

            backend_max_rss = _get(rec, "process.backend.max_rss_bytes")
            if isinstance(backend_max_rss, (int, float)):
                max_backend_max_rss_gb = _update_max(max_backend_max_rss_gb, _gb(int(backend_max_rss)), rec)

            backend_max_priv_bytes = _get(rec, "process.backend.max_private_bytes")
            if isinstance(backend_max_priv_bytes, (int, float)):
                max_backend_max_private_gb = _update_max(max_backend_max_private_gb, _gb(int(backend_max_priv_bytes)), rec)

            gpu_used_mib = _get(rec, "gpu.data.gpus.0.memory_used_mib")
            if isinstance(gpu_used_mib, (int, float)) and gpu_used_mib >= 0:
                max_gpu_mem_used_gb = _update_max(max_gpu_mem_used_gb, float(gpu_used_mib) / 1024.0, rec)

            edge_count = _get(rec, "process.edge.count")
            if isinstance(edge_count, (int, float)):
                edge_count_i = int(edge_count)
                if prev_edge_count is not None and prev_edge_count > 0 and edge_count_i == 0 and edge_disappear_at is None:
                    edge_disappear_at = rec
                prev_edge_count = edge_count_i

            if around_target is not None:
                ts = _parse_ts(str(rec.get("ts") or ""))
                if ts is not None:
                    if ts.hour == around_target[0]:
                        delta = abs(ts.minute - around_target[1]) * 60 + abs(ts.second)
                        best_delta, _ = around_best
                        if best_delta is None or delta < best_delta:
                            around_best = (delta, rec)

    print(f"file: {path}")
    print(f"records: {count}")
    print(f"range: {_fmt_ts(first_rec)} -> {_fmt_ts(last_rec)}")
    print("")

    print("=== Peaks ===")
    print(f"min available_gb: {min_avail_gb[0]} at {_fmt_ts(min_avail_gb[1])}")
    print(f"max pagefile_used_gb: {max_swap_gb[0]} at {_fmt_ts(max_swap_gb[1])}")
    print(f"max edge max_rss_gb: {max_edge_max_rss_gb[0]} at {_fmt_ts(max_edge_max_rss_gb[1])}")
    print(f"max edge total_rss_gb: {max_edge_total_rss_gb[0]} at {_fmt_ts(max_edge_total_rss_gb[1])}")
    print(f"max edge max_private_gb: {max_edge_max_private_gb[0]} at {_fmt_ts(max_edge_max_private_gb[1])}")
    print(f"max backend max_rss_gb: {max_backend_max_rss_gb[0]} at {_fmt_ts(max_backend_max_rss_gb[1])}")
    print(f"max backend max_private_gb: {max_backend_max_private_gb[0]} at {_fmt_ts(max_backend_max_private_gb[1])}")
    print(f"max gpu mem_used_gb: {max_gpu_mem_used_gb[0]} at {_fmt_ts(max_gpu_mem_used_gb[1])}")

    if edge_disappear_at is not None:
        print("")
        print(f"edge disappeared event at: {_fmt_ts(edge_disappear_at)}")

    if around_best[1] is not None:
        rec = around_best[1]
        print("")
        print(f"=== Around {around_hhmm} ===")
        avail_b = _get(rec, "system.memory.available_bytes")
        swap_b = _get(rec, "system.pagefile.used_bytes")
        edge_m = _get(rec, "process.edge.max_rss_bytes")
        backend_m = _get(rec, "process.backend.max_rss_bytes")
        gpu_used_mib = _get(rec, "gpu.data.gpus.0.memory_used_mib")
        print(f"ts: {rec.get('ts')}")
        print(f"available_gb: {round(_gb(int(avail_b)) or 0, 3) if isinstance(avail_b,(int,float)) else None}")
        print(f"pagefile_used_gb: {round(_gb(int(swap_b)) or 0, 3) if isinstance(swap_b,(int,float)) else None}")
        print(f"edge max_rss_gb: {round(_gb(int(edge_m)) or 0, 3) if isinstance(edge_m,(int,float)) else None}")
        print(f"backend max_rss_gb: {round(_gb(int(backend_m)) or 0, 3) if isinstance(backend_m,(int,float)) else None}")
        print(f"gpu mem_used_mib: {gpu_used_mib}")

    print("")
    print(f"=== Tail ({len(tail)}) ===")
    for rec in tail:
        ts = rec.get("ts")
        avail_b = _get(rec, "system.memory.available_bytes")
        swap_b = _get(rec, "system.pagefile.used_bytes")
        edge_c = _get(rec, "process.edge.count")
        edge_m = _get(rec, "process.edge.max_rss_bytes")
        backend_m = _get(rec, "process.backend.max_rss_bytes")
        gpu_used_mib = _get(rec, "gpu.data.gpus.0.memory_used_mib")
        print(
            ts,
            "avail_gb",
            round(_gb(int(avail_b)) or 0, 2) if isinstance(avail_b, (int, float)) else None,
            "swap_gb",
            round(_gb(int(swap_b)) or 0, 2) if isinstance(swap_b, (int, float)) else None,
            "edge_cnt",
            int(edge_c) if isinstance(edge_c, (int, float)) else None,
            "edge_max_gb",
            round(_gb(int(edge_m)) or 0, 2) if isinstance(edge_m, (int, float)) else None,
            "backend_max_gb",
            round(_gb(int(backend_m)) or 0, 2) if isinstance(backend_m, (int, float)) else None,
            "gpu_mem_mib",
            gpu_used_mib,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

