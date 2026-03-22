import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
    except FileNotFoundError:
        return None
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


@dataclass
class ReplaceStats:
    files_scanned: int = 0
    files_changed: int = 0
    replacements: int = 0


def _replace_in_text(text: str, old: str, new: str) -> Tuple[str, int]:
    if not text or old not in text:
        return text, 0
    return text.replace(old, new), text.count(old)


def _replace_in_bubble_states(meta: Dict[str, Any], old: str, new: str) -> int:
    count = 0
    bubbles = meta.get("bubbleStates")
    if not isinstance(bubbles, list):
        return 0

    for b in bubbles:
        if not isinstance(b, dict):
            continue

        for key in ("translatedText", "textboxText"):
            val = b.get(key)
            if isinstance(val, str):
                replaced, n = _replace_in_text(val, old, new)
                if n:
                    b[key] = replaced
                    count += n

    return count


def _replace_in_text_arrays(meta: Dict[str, Any], old: str, new: str) -> int:
    count = 0
    for key in ("bubbleTexts", "textboxTexts", "translatedTexts"):
        arr = meta.get(key)
        if not isinstance(arr, list):
            continue
        changed_any = False
        for i, v in enumerate(arr):
            if isinstance(v, str):
                replaced, n = _replace_in_text(v, old, new)
                if n:
                    arr[i] = replaced
                    count += n
                    changed_any = True
        if changed_any:
            meta[key] = arr
    return count


def _replace_all_strings(obj: Any, old: str, new: str) -> Tuple[Any, int]:
    if isinstance(obj, str):
        replaced, n = _replace_in_text(obj, old, new)
        return replaced, n
    if isinstance(obj, list):
        total = 0
        out = []
        for it in obj:
            new_it, n = _replace_all_strings(it, old, new)
            total += n
            out.append(new_it)
        return out, total
    if isinstance(obj, dict):
        total = 0
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            new_v, n = _replace_all_strings(v, old, new)
            total += n
            out[k] = new_v
        return out, total
    return obj, 0


def _iter_session_dirs_for_book(book_id: str) -> Iterable[str]:
    chapters_root = os.path.join(_default_data_root(), "bookshelf", book_id, "chapters")
    if not os.path.isdir(chapters_root):
        return []
    out: List[str] = []
    for chapter_id in os.listdir(chapters_root):
        session_dir = os.path.join(chapters_root, chapter_id, "session")
        if os.path.isdir(session_dir):
            out.append(session_dir)
    return sorted(out)


def _iter_meta_files(session_dir: str) -> Iterable[str]:
    session_meta_path = os.path.join(session_dir, "session_meta.json")
    session_meta = _safe_read_json(session_meta_path) or {}
    total_pages = session_meta.get("total_pages")

    images_dir = os.path.join(session_dir, "images")
    if not os.path.isdir(images_dir):
        return []

    meta_files: List[str] = []
    if isinstance(total_pages, int) and total_pages >= 0:
        for idx in range(total_pages):
            p = os.path.join(images_dir, str(idx), "meta.json")
            if os.path.isfile(p):
                meta_files.append(p)
        return meta_files

    for entry in os.listdir(images_dir):
        p = os.path.join(images_dir, entry, "meta.json")
        if os.path.isfile(p):
            meta_files.append(p)
    return sorted(meta_files, key=lambda x: int(os.path.basename(os.path.dirname(x))) if os.path.basename(os.path.dirname(x)).isdigit() else x)


def _backup_file(src: str, backup_root: str) -> None:
    rel = os.path.relpath(src, REPO_ROOT)
    dst = os.path.join(backup_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def replace_in_session_dir(
    session_dir: str,
    old: str,
    new: str,
    *,
    dry_run: bool,
    backup_root: Optional[str],
    all_strings: bool,
) -> ReplaceStats:
    stats = ReplaceStats()
    for meta_path in _iter_meta_files(session_dir):
        stats.files_scanned += 1
        meta = _safe_read_json(meta_path)
        if meta is None:
            continue

        original_meta = meta
        changed = 0
        if all_strings:
            meta, changed = _replace_all_strings(meta, old, new)
        else:
            changed += _replace_in_bubble_states(meta, old, new)
            changed += _replace_in_text_arrays(meta, old, new)

        if changed <= 0:
            continue

        stats.files_changed += 1
        stats.replacements += changed

        if dry_run:
            continue

        if backup_root:
            _backup_file(meta_path, backup_root)

        if meta is not original_meta:
            _safe_write_json(meta_path, meta)
        else:
            _safe_write_json(meta_path, meta)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="replace_translations_in_book",
        description="批量替换书架漫画已翻译文本中的指定字符串（默认只改译文相关字段）",
    )
    parser.add_argument("--book-id", default="", help="书架 book_id（替换该书所有章节）")
    parser.add_argument("--session-path", default="", help="会话路径（例如 bookshelf/<book_id>/chapters/<chapter_id>/session）")
    parser.add_argument("--session-dir", default="", help="会话目录（绝对路径，优先级最高）")
    parser.add_argument("--from", dest="old", required=True, help="要替换的旧字符串")
    parser.add_argument("--to", dest="new", required=True, help="替换成的新字符串")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份（默认会备份被修改文件）")
    parser.add_argument("--all-strings", action="store_true", help="替换 meta.json 内所有字符串字段（更激进）")
    args = parser.parse_args()

    if args.old == args.new:
        print("from 和 to 相同，无需替换。")
        return 0

    session_dirs: List[str] = []
    if args.session_dir:
        sd = os.path.abspath(args.session_dir)
        if not os.path.isdir(sd):
            print(f"session-dir 不存在: {sd}")
            return 2
        session_dirs = [sd]
    elif args.session_path:
        sd = _session_dir_from_session_path(args.session_path)
        if not os.path.isdir(sd):
            print(f"session-path 对应目录不存在: {sd}")
            return 2
        session_dirs = [sd]
    elif args.book_id:
        session_dirs = list(_iter_session_dirs_for_book(args.book_id))
        if not session_dirs:
            print(f"未找到该 book_id 的章节会话目录: {args.book_id}")
            return 2
    else:
        print("需要提供 --book-id 或 --session-path 或 --session-dir")
        return 2

    backup_root = None
    if not args.dry_run and not args.no_backup:
        backup_root = os.path.join(REPO_ROOT, "logs", "replace_backups", _timestamp())
        os.makedirs(backup_root, exist_ok=True)

    total = ReplaceStats()
    for sd in session_dirs:
        stats = replace_in_session_dir(
            sd,
            args.old,
            args.new,
            dry_run=args.dry_run,
            backup_root=backup_root,
            all_strings=args.all_strings,
        )
        total.files_scanned += stats.files_scanned
        total.files_changed += stats.files_changed
        total.replacements += stats.replacements

    print("=== Done ===")
    print(f"session_dirs: {len(session_dirs)}")
    print(f"files_scanned: {total.files_scanned}")
    print(f"files_changed: {total.files_changed}")
    print(f"replacements: {total.replacements}")
    if args.dry_run:
        print("mode: dry-run (no writes)")
    else:
        print("mode: write")
        print(f"backup: {'disabled' if args.no_backup else backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
