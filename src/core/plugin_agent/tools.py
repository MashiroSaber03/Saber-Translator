from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.plugins.manager import PluginManager

from .models import LockedPluginTarget


def _preview_text(content: str, limit: int = 1200) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "\n...[truncated]..."


class PluginAgentToolExecutor:
    def __init__(
        self,
        *,
        target: LockedPluginTarget,
        skill_markdown: str,
        on_write: Optional[Callable[[str, str], None]] = None,
        on_delete: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.target = target
        self.skill_markdown = skill_markdown
        self.on_write = on_write
        self.on_delete = on_delete
        self.is_cancelled = is_cancelled or (lambda: False)

    def _ensure_not_cancelled(self) -> None:
        if self.is_cancelled():
            raise RuntimeError("任务已取消")

    def _resolve_path(self, relative_path: str = "") -> Path:
        self._ensure_not_cancelled()
        normalized = relative_path.replace("\\", "/").strip()
        candidate = (Path(self.target.plugin_dir) / normalized).resolve()
        target_root = Path(self.target.plugin_dir).resolve()
        try:
            candidate.relative_to(target_root)
        except ValueError as exc:
            raise ValueError("只能访问当前锁定的插件目录") from exc
        return candidate

    def list_files(self, relative_path: str = "") -> Dict[str, Any]:
        target_path = self._resolve_path(relative_path)
        if not target_path.exists():
            return {
                "success": True,
                "base_path": relative_path or ".",
                "entries": [],
            }

        entries: List[Dict[str, Any]] = []
        for child in sorted(target_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            rel_path = str(child.relative_to(Path(self.target.plugin_dir))).replace("\\", "/")
            entries.append(
                {
                    "path": rel_path,
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return {
            "success": True,
            "base_path": relative_path or ".",
            "entries": entries,
        }

    def read_file(self, relative_path: str) -> Dict[str, Any]:
        target_path = self._resolve_path(relative_path)
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"文件不存在: {relative_path}")
        content = target_path.read_text(encoding="utf-8")
        return {
            "success": True,
            "path": str(target_path.relative_to(Path(self.target.plugin_dir))).replace("\\", "/"),
            "content": content,
            "preview": _preview_text(content),
        }

    def write_file(self, relative_path: str, content: str) -> Dict[str, Any]:
        target_path = self._resolve_path(relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        rel_path = str(target_path.relative_to(Path(self.target.plugin_dir))).replace("\\", "/")
        if self.on_write:
            self.on_write(rel_path, content)
        return {
            "success": True,
            "path": rel_path,
            "preview": _preview_text(content),
            "size": len(content.encode("utf-8")),
        }

    def delete_file(self, relative_path: str) -> Dict[str, Any]:
        target_path = self._resolve_path(relative_path)
        if not target_path.exists():
            raise FileNotFoundError(f"文件不存在: {relative_path}")
        if target_path.is_dir():
            raise ValueError("只允许删除文件，不允许删除目录")
        target_path.unlink()
        rel_path = str(target_path.relative_to(Path(self.target.plugin_dir))).replace("\\", "/")
        if self.on_delete:
            self.on_delete(rel_path)
        return {
            "success": True,
            "path": rel_path,
        }

    def read_skill(self) -> Dict[str, Any]:
        return {
            "success": True,
            "content": self.skill_markdown,
            "preview": _preview_text(self.skill_markdown, limit=2000),
        }

    def validate_plugin(self) -> Dict[str, Any]:
        plugin_dir = Path(self.target.plugin_dir)
        if not plugin_dir.exists():
            return {
                "success": False,
                "error": "插件目录尚未创建",
            }

        manager = PluginManager(plugin_dirs=[str(plugin_dir.parent)])
        try:
            result = manager.validate_plugin_source_path(str(plugin_dir), plugin_name=plugin_dir.name)
            return result
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }

    def run_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        if tool_name == "list_files":
            return self.list_files(args.get("path", ""))
        if tool_name == "read_file":
            return self.read_file(args.get("path", ""))
        if tool_name == "write_file":
            return self.write_file(args.get("path", ""), args.get("content", ""))
        if tool_name == "delete_file":
            return self.delete_file(args.get("path", ""))
        if tool_name == "read_skill":
            return self.read_skill()
        if tool_name == "validate_plugin":
            return self.validate_plugin()
        raise ValueError(f"未知工具: {tool_name}")
