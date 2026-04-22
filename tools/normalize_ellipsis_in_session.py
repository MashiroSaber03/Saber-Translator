import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ELLIPSIS_TARGET = "……"
ELLIPSIS_RUN_RE = re.compile(r"…{3,}")
DOT_RUN_RE = re.compile(r"・{2,}")


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


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iter_page_dirs(session_dir: str) -> Iterable[str]:
    images_dir = os.path.join(session_dir, "images")
    if not os.path.isdir(images_dir):
        return []

    page_dirs: List[str] = []
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if os.path.isdir(path):
            page_dirs.append(path)

    def _sort_key(path: str) -> Any:
        name = os.path.basename(path)
        return int(name) if name.isdigit() else name

    return sorted(page_dirs, key=_sort_key)


def _backup_file(src: str, backup_root: str) -> None:
    rel = os.path.relpath(src, REPO_ROOT)
    dst = os.path.join(backup_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _normalize_ellipsis_text(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0

    replacements = 0

    def _replace_ellipsis(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        return ELLIPSIS_TARGET

    def _replace_dots(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        return ELLIPSIS_TARGET

    normalized = ELLIPSIS_RUN_RE.sub(_replace_ellipsis, text)
    normalized = DOT_RUN_RE.sub(_replace_dots, normalized)
    return normalized, replacements


def _normalize_bubble_states(meta: Dict[str, Any]) -> Tuple[int, int]:
    bubbles = meta.get("bubbleStates")
    if not isinstance(bubbles, list):
        return 0, 0

    bubble_changes = 0
    replacement_count = 0

    for bubble in bubbles:
        if not isinstance(bubble, dict):
            continue
        val = bubble.get("translatedText")
        if not isinstance(val, str):
            continue
        normalized, replacements = _normalize_ellipsis_text(val)
        if replacements > 0 and normalized != val:
            bubble["translatedText"] = normalized
            bubble_changes += 1
            replacement_count += replacements

    return bubble_changes, replacement_count


def _normalize_text_arrays(meta: Dict[str, Any]) -> Tuple[int, int]:
    array_changes = 0
    replacement_count = 0

    for key in ("bubbleTexts", "translatedTexts"):
        arr = meta.get(key)
        if not isinstance(arr, list):
            continue

        changed = False
        for idx, val in enumerate(arr):
            if not isinstance(val, str):
                continue
            normalized, replacements = _normalize_ellipsis_text(val)
            if replacements > 0 and normalized != val:
                arr[idx] = normalized
                array_changes += 1
                replacement_count += replacements
                changed = True
        if changed:
            meta[key] = arr

    return array_changes, replacement_count


@dataclass
class NormalizeStats:
    pages_scanned: int = 0
    pages_changed: int = 0
    bubbles_changed: int = 0
    array_entries_changed: int = 0
    replacements: int = 0
    pages_rendered: int = 0
    pages_skipped_no_meta: int = 0
    pages_skipped_no_bubbles: int = 0
    pages_skipped_no_clean: int = 0
    pages_failed: int = 0
    changed_pages: List[int] = None

    def __post_init__(self) -> None:
        if self.changed_pages is None:
            self.changed_pages = []


def normalize_session(
    session_dir: str,
    *,
    dry_run: bool,
    backup_root: Optional[str],
) -> NormalizeStats:
    BubbleState = None
    render_bubbles_unified = None
    Image = None

    if not dry_run:
        from PIL import Image as PILImage
        from src.core.config_models import BubbleState as BubbleStateModel
        from src.core.rendering import render_bubbles_unified as render_func

        Image = PILImage
        BubbleState = BubbleStateModel
        render_bubbles_unified = render_func

    stats = NormalizeStats()

    for page_dir in _iter_page_dirs(session_dir):
        stats.pages_scanned += 1
        page_name = os.path.basename(page_dir)
        page_index = int(page_name) if page_name.isdigit() else -1

        meta_path = os.path.join(page_dir, "meta.json")
        meta = _safe_read_json(meta_path)
        if meta is None:
            stats.pages_skipped_no_meta += 1
            continue

        bubble_changes, bubble_replacements = _normalize_bubble_states(meta)
        array_changes, array_replacements = _normalize_text_arrays(meta)
        total_replacements = bubble_replacements + array_replacements

        if total_replacements <= 0:
            continue

        stats.pages_changed += 1
        stats.bubbles_changed += bubble_changes
        stats.array_entries_changed += array_changes
        stats.replacements += total_replacements
        stats.changed_pages.append(page_index)

        bubbles_raw = meta.get("bubbleStates")
        if not isinstance(bubbles_raw, list) or not bubbles_raw:
            stats.pages_skipped_no_bubbles += 1
            if not dry_run:
                if backup_root:
                    _backup_file(meta_path, backup_root)
                _safe_write_json(meta_path, meta)
            continue

        clean_path = os.path.join(page_dir, "clean.png")
        if not os.path.isfile(clean_path):
            stats.pages_skipped_no_clean += 1
            if not dry_run:
                if backup_root:
                    _backup_file(meta_path, backup_root)
                _safe_write_json(meta_path, meta)
            continue

        if dry_run:
            stats.pages_rendered += 1
            continue

        translated_path = os.path.join(page_dir, "translated.png")

        try:
            if backup_root:
                _backup_file(meta_path, backup_root)
                if os.path.isfile(translated_path):
                    _backup_file(translated_path, backup_root)

            _safe_write_json(meta_path, meta)

            bubble_states: List[BubbleState] = []
            for bubble in bubbles_raw:
                if isinstance(bubble, dict):
                    bubble_states.append(BubbleState.from_dict(bubble))

            base = Image.open(clean_path).convert("RGBA")
            render_bubbles_unified(base, bubble_states)
            base.save(translated_path, format="PNG", optimize=True)
            stats.pages_rendered += 1
        except Exception:
            stats.pages_failed += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="normalize_ellipsis_in_session",
        description="规范化 session 译文中的省略号，并只重渲染发生变化的页面。",
    )
    parser.add_argument("--session-path", default="", help="会话路径，例如 bookshelf/<book_id>/chapters/<chapter_id>/session")
    parser.add_argument("--session-dir", default="", help="会话目录绝对路径，优先级高于 session-path")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    parser.add_argument("--no-backup", action="store_true", help="不备份被修改的 meta.json 和 translated.png")
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
        backup_root = os.path.join(REPO_ROOT, "logs", "ellipsis_normalize_backups", _timestamp())
        os.makedirs(backup_root, exist_ok=True)

    stats = normalize_session(
        session_dir,
        dry_run=args.dry_run,
        backup_root=backup_root,
    )

    changed_pages_preview = [p + 1 for p in stats.changed_pages[:20] if p >= 0]

    print("=== Done ===")
    print(f"session_dir: {session_dir}")
    print(f"pages_scanned: {stats.pages_scanned}")
    print(f"pages_changed: {stats.pages_changed}")
    print(f"bubbles_changed: {stats.bubbles_changed}")
    print(f"array_entries_changed: {stats.array_entries_changed}")
    print(f"replacements: {stats.replacements}")
    print(f"pages_rendered: {stats.pages_rendered}")
    print(f"pages_skipped_no_meta: {stats.pages_skipped_no_meta}")
    print(f"pages_skipped_no_bubbles: {stats.pages_skipped_no_bubbles}")
    print(f"pages_skipped_no_clean: {stats.pages_skipped_no_clean}")
    print(f"pages_failed: {stats.pages_failed}")
    print(f"mode: {'dry-run' if args.dry_run else 'write'}")
    print(f"backup: {'disabled' if args.no_backup or args.dry_run else backup_root}")
    print(f"changed_pages_preview(1-based): {changed_pages_preview}")
    return 0 if stats.pages_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
