"""Six worktree-scoped tools available to the Plugin Agent Worker."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from src.backend_v2.plugins.contract import validate_hook_source_contract
from src.backend_v2.plugins.package import build_archive, parse_archive
from src.backend_v2.redaction import redact_sensitive_text
from src.core.plugin_agent.controller import PluginAgentControlRequested
from src.shared.memory_errors import is_memory_allocation_error


class PluginAgentWorktreeTools:
    def __init__(
        self,
        *,
        worktree: Path,
        skill_markdown: str,
        control_requested: Callable[[], bool],
        on_write: Callable[[str, str], None] | None = None,
        on_delete: Callable[[str], None] | None = None,
    ) -> None:
        self.worktree = worktree.resolve()
        self.skill_markdown = skill_markdown
        self.control_requested = control_requested
        self.on_write = on_write
        self.on_delete = on_delete

    def is_control_requested(self) -> bool:
        requested = self.control_requested()
        if not isinstance(requested, bool):
            raise TypeError("Plugin Agent control callback must return bool")
        return requested

    def check_control(self) -> None:
        if self.is_control_requested():
            raise PluginAgentControlRequested(
                "plugin agent job control requested"
            )

    def run_tool(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.check_control()
        if not isinstance(args, dict):
            raise ValueError("Plugin Agent tool args must be an object")
        payload = args
        if tool_name == "list_files":
            if set(payload) not in (set(), {"path"}):
                raise ValueError("list_files only accepts path")
            relative = payload.get("path", "")
            if not isinstance(relative, str):
                raise ValueError("list_files.path must be a string")
            return self.list_files(relative)
        if tool_name == "read_file":
            return self.read_file(self._required_path(payload, "read_file"))
        if tool_name == "write_file":
            if set(payload) != {"path", "content"}:
                raise ValueError("write_file requires path and content")
            relative = self._required_path(payload, "write_file", exact=False)
            content = payload["content"]
            if not isinstance(content, str):
                raise ValueError("write_file.content must be a string")
            return self.write_file(
                relative,
                content,
            )
        if tool_name == "delete_file":
            return self.delete_file(self._required_path(payload, "delete_file"))
        if tool_name == "read_skill":
            self._require_empty_args(payload, "read_skill")
            return self.read_skill()
        if tool_name == "validate_plugin":
            self._require_empty_args(payload, "validate_plugin")
            return self.validate_plugin()
        raise ValueError(f"unknown Plugin Agent tool: {tool_name}")

    @staticmethod
    def _require_empty_args(payload: dict[str, Any], tool_name: str) -> None:
        if payload:
            raise ValueError(f"{tool_name} does not accept arguments")

    @staticmethod
    def _required_path(
        payload: dict[str, Any],
        tool_name: str,
        *,
        exact: bool = True,
    ) -> str:
        if exact and set(payload) != {"path"}:
            raise ValueError(f"{tool_name} requires only path")
        relative = payload.get("path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"{tool_name}.path must be a non-empty string")
        return relative

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
            self.check_control()
            parsed = parse_archive(build_archive(self.worktree))
            python_files = sorted(self.worktree.rglob("*.py"))
            for path in python_files:
                source = path.read_text(encoding="utf-8")
                compile(
                    source,
                    path.relative_to(self.worktree).as_posix(),
                    "exec",
                )
            module_path = parsed.manifest.entrypoint.rsplit(":", 1)[0]
            entrypoint = self.worktree.joinpath(
                *module_path.replace("\\", "/").split("/")
            )
            validate_hook_source_contract(
                parsed.manifest,
                entrypoint.read_text(encoding="utf-8"),
                filename=module_path,
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
        except PluginAgentControlRequested:
            raise
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            return {
                "success": False,
                "error": redact_sensitive_text(exc),
            }

    def _path(self, relative: str) -> Path:
        self.check_control()
        if (
            not isinstance(relative, str)
            or relative != relative.strip()
            or "\\" in relative
            or relative.startswith("/")
        ):
            raise ValueError("Plugin Agent tool paths must be relative POSIX paths")
        normalized = PurePosixPath(relative or ".")
        if ".." in normalized.parts:
            raise ValueError("Plugin Agent tools cannot leave the worktree")
        target = self.worktree.joinpath(*normalized.parts).resolve()
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
