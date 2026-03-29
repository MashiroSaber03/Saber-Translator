import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _norm_slash(p: str) -> str:
    return p.replace("\\", "/")


def _default_data_root() -> str:
    return os.path.join(REPO_ROOT, "data")


def _default_sessions_root() -> str:
    return os.path.join(_default_data_root(), "sessions")


def _session_dir_from_session_path(session_path: str) -> str:
    sp = _norm_slash(session_path).lstrip("/")
    if sp.startswith("bookshelf/"):
        return os.path.join(_default_data_root(), sp)
    return os.path.join(_default_sessions_root(), sp)


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iter_page_dirs(session_dir: str) -> Iterable[str]:
    images_dir = os.path.join(session_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    page_dirs: List[str] = []
    for name in os.listdir(images_dir):
        p = os.path.join(images_dir, name)
        if os.path.isdir(p):
            page_dirs.append(p)

    def _sort_key(x: str) -> Any:
        base = os.path.basename(x)
        return int(base) if base.isdigit() else base

    return sorted(page_dirs, key=_sort_key)


def _backup_file(src: str, backup_root: str) -> None:
    rel = os.path.relpath(src, REPO_ROOT)
    dst = os.path.join(backup_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


@dataclass
class RerenderStats:
    pages_scanned: int = 0
    pages_rendered: int = 0
    pages_skipped_no_meta: int = 0
    pages_skipped_no_bubbles: int = 0
    pages_skipped_no_clean: int = 0
    pages_skipped_filter: int = 0
    pages_failed: int = 0


def _contains_target(meta: Dict[str, Any], contains: str) -> bool:
    if not contains:
        return True
    bubbles = meta.get("bubbleStates")
    if isinstance(bubbles, list):
        for b in bubbles:
            if isinstance(b, dict):
                t = b.get("translatedText")
                if isinstance(t, str) and contains in t:
                    return True
    for key in ("bubbleTexts", "translatedTexts", "textboxTexts"):
        arr = meta.get(key)
        if isinstance(arr, list):
            for v in arr:
                if isinstance(v, str) and contains in v:
                    return True
    return False


def rerender_session(
    session_dir: str,
    *,
    contains: str,
    dry_run: bool,
    backup_root: Optional[str],
) -> RerenderStats:
    from src.core.config_models import BubbleState
    from src.core.rendering import render_bubbles_unified

    stats = RerenderStats()

    for page_dir in _iter_page_dirs(session_dir):
        stats.pages_scanned += 1
        meta_path = os.path.join(page_dir, "meta.json")
        meta = _safe_read_json(meta_path)
        if meta is None:
            stats.pages_skipped_no_meta += 1
            continue

        bubbles_raw = meta.get("bubbleStates")
        if not isinstance(bubbles_raw, list) or not bubbles_raw:
            stats.pages_skipped_no_bubbles += 1
            continue

        if not _contains_target(meta, contains):
            stats.pages_skipped_filter += 1
            continue

        clean_path = os.path.join(page_dir, "clean.png")
        if not os.path.isfile(clean_path):
            stats.pages_skipped_no_clean += 1
            continue

        translated_path = os.path.join(page_dir, "translated.png")

        try:
            bubble_states: List[BubbleState] = []
            for b in bubbles_raw:
                if isinstance(b, dict):
                    bubble_states.append(BubbleState.from_dict(b))

            if dry_run:
                stats.pages_rendered += 1
                continue

            if backup_root and os.path.isfile(translated_path):
                _backup_file(translated_path, backup_root)

            base = Image.open(clean_path).convert("RGBA")
            render_bubbles_unified(base, bubble_states)
            base.save(translated_path, format="PNG", optimize=True)

            stats.pages_rendered += 1
        except Exception:
            stats.pages_failed += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rerender_translated_images_in_session",
        description="根据 session/images/<page>/meta.json 的 bubbleStates，基于 clean.png 批量重渲染 translated.png",
    )
    parser.add_argument("--session-path", default="", help="会话路径（例如 bookshelf/<book_id>/chapters/<chapter_id>/session）")
    parser.add_argument("--session-dir", default="", help="会话目录（绝对路径，优先级最高）")
    parser.add_argument("--contains", default="", help="仅重渲染译文包含该字符串的页面（例如 阳太）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    parser.add_argument("--no-backup", action="store_true", help="不备份原 translated.png（默认会备份）")
    args = parser.parse_args()

    if not args.session_dir and not args.session_path:
        print("需要提供 --session-path 或 --session-dir")
        return 2

    if args.session_dir:
        session_dir = os.path.abspath(args.session_dir)
    else:
        session_dir = _session_dir_from_session_path(args.session_path)

    if not os.path.isdir(session_dir):
        print(f"session 目录不存在: {session_dir}")
        return 2

    backup_root = None
    if not args.dry_run and not args.no_backup:
        backup_root = os.path.join(REPO_ROOT, "logs", "rerender_backups", _timestamp())
        os.makedirs(backup_root, exist_ok=True)

    stats = rerender_session(
        session_dir,
        contains=args.contains,
        dry_run=args.dry_run,
        backup_root=backup_root,
    )

    print("=== Done ===")
    print(f"session_dir: {session_dir}")
    print(f"pages_scanned: {stats.pages_scanned}")
    print(f"pages_rendered: {stats.pages_rendered}")
    print(f"pages_skipped_no_meta: {stats.pages_skipped_no_meta}")
    print(f"pages_skipped_no_bubbles: {stats.pages_skipped_no_bubbles}")
    print(f"pages_skipped_no_clean: {stats.pages_skipped_no_clean}")
    print(f"pages_skipped_filter: {stats.pages_skipped_filter}")
    print(f"pages_failed: {stats.pages_failed}")
    print(f"mode: {'dry-run' if args.dry_run else 'write'}")
    print(f"backup: {'disabled' if args.no_backup or args.dry_run else backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
