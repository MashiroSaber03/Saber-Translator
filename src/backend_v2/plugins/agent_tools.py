"""Six worktree-scoped tools available to the Plugin Agent Worker."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.backend_v2.plugins.package import build_archive, parse_archive
from src.backend_v2.redaction import redact_sensitive_text


class PluginAgentWorktreeTools:
    MAX_TEXT_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        *,
        worktree: Path,
        skill_markdown: str,
        cancelled: Callable[[], bool],
        on_write: Callable[[str, str], None] | None = None,
        on_delete: Callable[[str], None] | None = None,
    ) -> None:
        self.worktree = worktree.resolve()
        self.skill_markdown = skill_markdown
        self.cancelled = cancelled
        self.on_write = on_write
        self.on_delete = on_delete

    def is_cancelled(self) -> bool:
        return bool(self.cancelled())

    def run_tool(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = args or {}
        if tool_name == "list_files":
            return self.list_files(str(payload.get("path", "")))
        if tool_name == "read_file":
            return self.read_file(str(payload.get("path", "")))
        if tool_name == "write_file":
            return self.write_file(
                str(payload.get("path", "")),
                str(payload.get("content", "")),
            )
        if tool_name == "delete_file":
            return self.delete_file(str(payload.get("path", "")))
        if tool_name == "read_skill":
            return self.read_skill()
        if tool_name == "validate_plugin":
            return self.validate_plugin()
        raise ValueError(f"unknown Plugin Agent tool: {tool_name}")

    def list_files(self, relative: str = "") -> dict[str, Any]:
        target = self._path(relative)
        if not target.exists():
            return {
                "success": True,
                "base_path": relative or ".",
                "entries": [],
            }
        if not target.is_dir():
            raise ValueError("list_files target must be a directory")
        entries = []
        for child in sorted(
            target.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        ):
            entries.append(
                {
                    "path": child.relative_to(
                        self.worktree
                    ).as_posix(),
                    "name": child.name,
                    "type": (
                        "directory" if child.is_dir() else "file"
                    ),
                    "size": (
                        child.stat().st_size
                        if child.is_file()
                        else None
                    ),
                }
            )
        return {
            "success": True,
            "base_path": relative or ".",
            "entries": entries,
        }

    def read_file(self, relative: str) -> dict[str, Any]:
        target = self._path(relative)
        if not target.is_file():
            raise FileNotFoundError(f"file not found: {relative}")
        if target.stat().st_size > self.MAX_TEXT_BYTES:
            raise ValueError("Plugin Agent text file exceeds 2 MiB")
        content = target.read_text(encoding="utf-8")
        return {
            "success": True,
            "path": target.relative_to(self.worktree).as_posix(),
            "content": content,
            "preview": _preview(content),
        }

    def write_file(
        self,
        relative: str,
        content: str,
    ) -> dict[str, Any]:
        target = self._path(relative)
        encoded = content.encode("utf-8")
        if not relative or len(encoded) > self.MAX_TEXT_BYTES:
            raise ValueError(
                "write_file requires a path and at most 2 MiB UTF-8 text"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        name = target.relative_to(self.worktree).as_posix()
        if self.on_write is not None:
            self.on_write(name, content)
        return {
            "success": True,
            "path": name,
            "size": len(encoded),
            "preview": _preview(content),
        }

    def delete_file(self, relative: str) -> dict[str, Any]:
        target = self._path(relative)
        if not target.is_file():
            raise ValueError("delete_file only accepts an existing file")
        name = target.relative_to(self.worktree).as_posix()
        target.unlink()
        if self.on_delete is not None:
            self.on_delete(name)
        return {"success": True, "path": name}

    def read_skill(self) -> dict[str, Any]:
        return {
            "success": True,
            "content": self.skill_markdown,
            "preview": _preview(self.skill_markdown, limit=2000),
        }

    def validate_plugin(self) -> dict[str, Any]:
        try:
            parsed = parse_archive(build_archive(self.worktree))
            python_files = sorted(self.worktree.rglob("*.py"))
            for path in python_files:
                source = path.read_text(encoding="utf-8")
                compile(
                    source,
                    path.relative_to(self.worktree).as_posix(),
                    "exec",
                )
            return {
                "success": True,
                "plugin_id": parsed.manifest.plugin_id,
                "package_version": (
                    parsed.manifest.package_version
                ),
                "hooks": list(parsed.manifest.hooks),
                "python_files": len(python_files),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": redact_sensitive_text(exc),
            }

    def _path(self, relative: str) -> Path:
        if self.is_cancelled():
            raise RuntimeError("Plugin Agent job is cancelling")
        normalized = relative.replace("\\", "/").strip().lstrip("/")
        target = (self.worktree / Path(normalized)).resolve()
        try:
            target.relative_to(self.worktree)
        except ValueError as exc:
            raise ValueError(
                "Plugin Agent tools cannot leave the worktree"
            ) from exc
        return target


def _preview(content: str, *, limit: int = 1200) -> str:
    return (
        content
        if len(content) <= limit
        else content[:limit] + "\n...[truncated]..."
    )
