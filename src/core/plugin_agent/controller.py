from __future__ import annotations

import itertools
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from src.shared.ai_providers import (
    PLUGIN_AGENT_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_requires_api_key,
    provider_supports_capability,
)
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
    OpenAICompatibleSyncExecutor,
    build_openai_compatible_runtime_options,
)
from src.shared.openai_options import OpenAICompatibleOptions

from .models import PluginAgentSession

logger = logging.getLogger("PluginAgent.Controller")
_ASSISTANT_MESSAGE_PATTERN = re.compile(r'"assistant_message"\s*:\s*"')
_PLANNING_FIELDS = {"assistant_message", "target_proposal"}
_EXECUTION_FIELDS = {"assistant_message", "action"}
_ACTION_FIELDS = {"tool", "args"}
_TARGET_FIELDS = {
    "plugin_id",
    "display_name",
    "supported_steps",
    "supported_modes",
}
_TOOLS = {
    "list_files",
    "read_file",
    "write_file",
    "delete_file",
    "read_skill",
    "validate_plugin",
    "finish",
}
_TARGET_STEPS = {
    "job",
    "pipeline",
    "detect",
    "ocr",
    "color",
    "translate",
    "ai_translate",
    "inpaint",
    "render",
}
_TARGET_MODES = {"standard", "hq", "proofread", "remove_text"}


class PluginAgentControlRequested(RuntimeError):
    """The durable job requested pause or cancellation at a safe point."""


def _decode_json_string_prefix(raw_text: str) -> tuple[str, bool]:
    decoded: list[str] = []
    index = 0
    while index < len(raw_text):
        character = raw_text[index]
        if character == '"':
            return "".join(decoded), True
        if character != "\\":
            decoded.append(character)
            index += 1
            continue

        index += 1
        if index >= len(raw_text):
            return "".join(decoded), False
        escaped = raw_text[index]
        if escaped == "n":
            decoded.append("\n")
            index += 1
            continue
        if escaped == "r":
            decoded.append("\r")
            index += 1
            continue
        if escaped == "t":
            decoded.append("\t")
            index += 1
            continue
        if escaped == "b":
            decoded.append("\b")
            index += 1
            continue
        if escaped == "f":
            decoded.append("\f")
            index += 1
            continue
        if escaped in {'"', "\\", "/"}:
            decoded.append(escaped)
            index += 1
            continue
        if escaped == "u":
            if index + 4 >= len(raw_text):
                return "".join(decoded), False
            code = raw_text[index + 1 : index + 5]
            if not all(character in "0123456789abcdefABCDEF" for character in code):
                return "".join(decoded), False
            decoded.append(chr(int(code, 16)))
            index += 5
            continue
        decoded.append(escaped)
        index += 1
    return "".join(decoded), False


class PluginAgentController:
    def __init__(self, transport: OpenAICompatibleChatTransport | None = None) -> None:
        self.transport = transport or OpenAICompatibleChatTransport()
        self.executor = OpenAICompatibleSyncExecutor(self.transport)

    def plan_turn(
        self,
        session: PluginAgentSession,
        skill_markdown: str,
        agent_config: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = self._build_planning_system_prompt(session)
        messages = self._build_chat_messages(session, system_prompt, skill_markdown)
        return self._call_agent_json(messages, agent_config, label="PluginAgent-Planning")

    def execute(
        self,
        session: PluginAgentSession,
        skill_markdown: str,
        agent_config: dict[str, Any],
        tool_executor,
        emit_event,
    ) -> dict[str, Any]:
        tool_history: list[dict[str, Any]] = []
        last_validation: dict[str, Any] | None = None

        for iteration in itertools.count(1):
            if tool_executor.is_control_requested():
                raise PluginAgentControlRequested(
                    "plugin agent job control requested"
                )

            system_prompt = self._build_execution_system_prompt(
                session,
                tool_history,
                iteration,
            )
            messages = self._build_execution_messages(session, system_prompt, skill_markdown)
            stream_id = f"execution-{iteration}"
            last_stream_content = ""

            def emit_streaming_assistant(raw_stream_content: str, *, force: bool = False) -> None:
                nonlocal last_stream_content
                content, _completed = self._extract_assistant_message_prefix(raw_stream_content)
                if content is None or content == last_stream_content:
                    return
                delta = content[len(last_stream_content) :] if content.startswith(last_stream_content) else content
                last_stream_content = content
                if not delta and not force:
                    return
                emit_event(
                    "assistant_delta",
                    {
                        "stream_id": stream_id,
                        "phase": "execution",
                        "delta": delta,
                        "content": content,
                    },
                )

            def check_control() -> None:
                if tool_executor.is_control_requested():
                    raise PluginAgentControlRequested(
                        "plugin agent job control requested"
                    )

            def handle_stream_chunk(_chunk: str, content: str) -> None:
                check_control()
                emit_streaming_assistant(content)

            envelope = self._call_agent_json(
                messages,
                agent_config,
                label="PluginAgent-Execution",
                on_stream_chunk=handle_stream_chunk,
                before_request=check_control,
                require_action=True,
            )
            check_control()

            assistant_message = envelope["assistant_message"].strip()
            if assistant_message:
                emit_streaming_assistant(json.dumps({"assistant_message": assistant_message}, ensure_ascii=False), force=True)
                emit_event(
                    "assistant",
                    {
                        "stream_id": stream_id,
                        "message": assistant_message,
                        "phase": "execution",
                    },
                )

            action = envelope["action"]
            tool_name = action["tool"]

            if tool_name == "finish":
                if action["args"]:
                    raise OpenAICompatibleBusinessRetryableError(
                        "Agent finish 动作不能包含参数"
                    )
                final_validation = last_validation or tool_executor.validate_plugin()
                self._validate_tool_result("validate_plugin", final_validation)
                if not final_validation["success"]:
                    error = final_validation.get("error")
                    raise ValueError(
                        error if isinstance(error, str) and error else "插件校验失败"
                    )
                return {
                    "assistant_message": assistant_message or "插件任务完成。",
                    "validation": final_validation,
                }

            tool_args = action["args"]
            group_id = f"tool-{iteration}"
            emit_event("tool_call", self._build_tool_call_payload(tool_name, tool_args, group_id))
            tool_result = tool_executor.run_tool(tool_name, tool_args)
            self._validate_tool_result(tool_name, tool_result)
            emit_event("tool_result", self._build_tool_result_payload(tool_name, tool_result, group_id))

            if tool_name == "validate_plugin":
                last_validation = tool_result
                emit_event("validation", self._build_validation_payload(tool_result))
            elif tool_name in {"write_file", "delete_file"}:
                last_validation = None

            tool_history.append(
                {
                    "tool": tool_name,
                    "args": dict(tool_args),
                    "result": dict(tool_result),
                }
            )

    def _call_agent_json(
        self,
        messages: list[dict[str, Any]],
        agent_config: dict[str, Any],
        *,
        label: str,
        on_stream_chunk: Callable[[str, str], None] | None = None,
        before_request: Callable[[], None] | None = None,
        require_action: bool = False,
    ) -> dict[str, Any]:
        expected_config_fields = {
            "provider",
            "credential_version_id",
            "api_key",
            "model_name",
            "custom_base_url",
            "openai_options",
        }
        if not isinstance(agent_config, dict) or set(agent_config) != expected_config_fields:
            raise ValueError("agent_config 字段无效")
        string_fields = ("provider", "api_key", "model_name", "custom_base_url")
        if any(not isinstance(agent_config[field], str) for field in string_fields):
            raise ValueError("agent_config 文本字段必须是字符串")
        credential_version_id = agent_config["credential_version_id"]
        if credential_version_id is not None and not isinstance(
            credential_version_id,
            str,
        ):
            raise ValueError("agent_config.credential_version_id 必须是字符串或 null")
        provider = normalize_provider_id(agent_config["provider"])
        api_key = agent_config["api_key"]
        model_name = agent_config["model_name"]
        custom_base_url = agent_config["custom_base_url"] or None
        openai_options = agent_config["openai_options"]
        if not isinstance(openai_options, OpenAICompatibleOptions):
            raise ValueError("agent_config.openai_options 必须是 OpenAICompatibleOptions")

        if not provider_supports_capability(provider, PLUGIN_AGENT_CAPABILITY):
            raise ValueError(f"不支持的插件 Agent 服务商: {provider}")

        manifest = get_provider_manifest(provider)
        if provider_requires_api_key(provider, custom_base_url) and not api_key:
            raise ValueError(f"{manifest.display_name} 需要 API Key")
        if manifest.requires_model and not model_name:
            raise ValueError(f"{manifest.display_name} 需要模型名称")
        if manifest.requires_base_url and not custom_base_url:
            raise ValueError(f"{manifest.display_name} 需要 Base URL")

        result = self.executor.execute(
            UnifiedChatRequest(
                provider=provider,
                api_key=api_key,
                model=model_name,
                credential_version_id=credential_version_id,
                base_url=custom_base_url,
                capability=PLUGIN_AGENT_CAPABILITY,
                openai_options=openai_options,
                runtime_options=build_openai_compatible_runtime_options(
                    timeout=180.0,
                    print_stream_output=openai_options.execution.use_stream,
                    stream_output_label=label,
                    on_stream_chunk=on_stream_chunk,
                ),
                messages=messages,
            ),
            capability=PLUGIN_AGENT_CAPABILITY,
            parser=lambda content: self._parse_agent_envelope(
                content,
                require_action=require_action,
            ),
            logger_instance=logger,
            before_request=before_request,
        )
        return result.parsed

    @staticmethod
    def _parse_agent_envelope(
        content: str,
        *,
        require_action: bool = False,
    ) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OpenAICompatibleBusinessRetryableError(
                f"Agent JSON 解析失败: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise OpenAICompatibleBusinessRetryableError(
                "Agent 返回结果必须是 JSON 对象"
            )
        expected_fields = _EXECUTION_FIELDS if require_action else _PLANNING_FIELDS
        if set(parsed) != expected_fields:
            raise OpenAICompatibleBusinessRetryableError(
                "Agent 返回结果字段与当前阶段不匹配"
            )
        assistant_message = parsed["assistant_message"]
        if not isinstance(assistant_message, str):
            raise OpenAICompatibleBusinessRetryableError(
                "Agent assistant_message 必须是字符串"
            )
        if not require_action:
            target = parsed["target_proposal"]
            if target is not None and (
                not isinstance(target, dict) or set(target) != _TARGET_FIELDS
            ):
                raise OpenAICompatibleBusinessRetryableError(
                    "Agent target_proposal 字段无效"
                )
            if target is not None:
                plugin_id = target["plugin_id"]
                display_name = target["display_name"]
                if (
                    not isinstance(plugin_id, str)
                    or not plugin_id.strip()
                    or not isinstance(display_name, str)
                    or not display_name.strip()
                ):
                    raise OpenAICompatibleBusinessRetryableError(
                        "Agent target_proposal 文本字段无效"
                    )
                PluginAgentController._validate_target_values(
                    target["supported_steps"],
                    allowed=_TARGET_STEPS,
                    field="supported_steps",
                )
                PluginAgentController._validate_target_values(
                    target["supported_modes"],
                    allowed=_TARGET_MODES,
                    field="supported_modes",
                )
            return parsed

        action = parsed["action"]
        if not isinstance(action, dict) or set(action) != _ACTION_FIELDS:
            raise OpenAICompatibleBusinessRetryableError(
                "Agent 工具动作字段无效"
            )
        tool_name = action["tool"]
        if not isinstance(tool_name, str) or tool_name not in _TOOLS:
            raise OpenAICompatibleBusinessRetryableError(
                "Agent 未返回受支持的工具动作"
            )
        if not isinstance(action["args"], dict):
            raise OpenAICompatibleBusinessRetryableError(
                "Agent 工具动作 args 必须是 JSON 对象"
            )
        PluginAgentController._validate_tool_args(tool_name, action["args"])
        return parsed

    @staticmethod
    def _require_tool_result(result: object) -> None:
        if not isinstance(result, dict) or not isinstance(
            result.get("success"),
            bool,
        ):
            raise TypeError("Plugin Agent 工具必须返回带布尔 success 的对象")

    @classmethod
    def _validate_tool_result(
        cls,
        tool_name: str,
        result: object,
    ) -> None:
        cls._require_tool_result(result)
        if not isinstance(result, dict):
            raise TypeError("Plugin Agent 工具结果必须是对象")
        success = result["success"]
        if tool_name == "validate_plugin":
            expected = (
                {
                    "success",
                    "plugin_id",
                    "package_version",
                    "hooks",
                    "python_files",
                }
                if success
                else {"success", "error"}
            )
        else:
            expected = {
                "list_files": {"success", "base_path", "entries"},
                "read_file": {"success", "path", "content", "preview"},
                "write_file": {"success", "path", "size", "preview"},
                "delete_file": {"success", "path"},
                "read_skill": {"success", "content", "preview"},
            }[tool_name]
        if set(result) != expected:
            raise TypeError(f"Plugin Agent {tool_name} 工具结果字段无效")

        if not success:
            if not isinstance(result["error"], str) or not result["error"]:
                raise TypeError("Plugin Agent 校验错误必须是非空字符串")
            return
        if tool_name == "list_files":
            if not isinstance(result["base_path"], str) or not isinstance(
                result["entries"],
                list,
            ):
                raise TypeError("Plugin Agent 目录结果无效")
            for entry in result["entries"]:
                if (
                    not isinstance(entry, dict)
                    or set(entry) != {"path", "name", "type", "size"}
                    or not isinstance(entry["path"], str)
                    or not isinstance(entry["name"], str)
                    or entry["type"] not in {"file", "directory"}
                    or not (
                        (entry["type"] == "directory" and entry["size"] is None)
                        or (
                            entry["type"] == "file"
                            and not isinstance(entry["size"], bool)
                            and isinstance(entry["size"], int)
                            and entry["size"] >= 0
                        )
                    )
                ):
                    raise TypeError("Plugin Agent 目录条目无效")
            return
        if tool_name in {"read_file", "write_file", "delete_file"}:
            if not isinstance(result["path"], str) or not result["path"]:
                raise TypeError("Plugin Agent 文件工具结果路径无效")
        if tool_name in {"read_file", "read_skill"}:
            if not isinstance(result["content"], str):
                raise TypeError("Plugin Agent 读取结果内容无效")
        if tool_name in {"read_file", "write_file", "read_skill"}:
            if not isinstance(result["preview"], str):
                raise TypeError("Plugin Agent 工具结果预览无效")
        if tool_name == "write_file" and (
            isinstance(result["size"], bool)
            or not isinstance(result["size"], int)
            or result["size"] < 0
        ):
            raise TypeError("Plugin Agent 写入结果大小无效")
        if tool_name == "validate_plugin" and (
            not isinstance(result["plugin_id"], str)
            or not result["plugin_id"]
            or not isinstance(result["package_version"], str)
            or not result["package_version"]
            or not isinstance(result["hooks"], list)
            or any(not isinstance(hook, str) for hook in result["hooks"])
            or isinstance(result["python_files"], bool)
            or not isinstance(result["python_files"], int)
            or result["python_files"] < 1
        ):
            raise TypeError("Plugin Agent 校验成功结果无效")

    @staticmethod
    def _validate_target_values(
        value: object,
        *,
        allowed: set[str],
        field: str,
    ) -> None:
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or item not in allowed for item in value)
            or len(set(value)) != len(value)
        ):
            raise OpenAICompatibleBusinessRetryableError(
                f"Agent target_proposal.{field} 无效"
            )

    @staticmethod
    def _validate_tool_args(tool_name: str, args: dict[str, Any]) -> None:
        expected_fields = {
            "list_files": (set(), {"path"}),
            "read_file": ({"path"},),
            "write_file": ({"path", "content"},),
            "delete_file": ({"path"},),
            "read_skill": (set(),),
            "validate_plugin": (set(),),
            "finish": (set(),),
        }[tool_name]
        if set(args) not in expected_fields:
            raise OpenAICompatibleBusinessRetryableError(
                f"Agent {tool_name} 工具参数字段无效"
            )
        if "path" in args and (
            not isinstance(args["path"], str) or not args["path"].strip()
        ):
            raise OpenAICompatibleBusinessRetryableError(
                f"Agent {tool_name}.path 必须是非空字符串"
            )
        if "content" in args and not isinstance(args["content"], str):
            raise OpenAICompatibleBusinessRetryableError(
                "Agent write_file.content 必须是字符串"
            )

    def _build_planning_system_prompt(self, session: PluginAgentSession) -> str:
        locked_target = session.locked_target.plugin_id if session.locked_target else "未锁定"
        return (
            "你是 Saber Translator 的内置插件编程 Agent。\n"
            "当前阶段是需求分析与方案确认阶段，严禁产生任何写文件动作。\n"
            "你会收到项目内置插件 skill，因此必须遵守其规则。\n"
            f"当前会话模式: {session.mode}\n"
            f"当前锁定目标: {locked_target}\n\n"
            "请只返回 JSON 对象，结构如下：\n"
            "{\n"
            '  "assistant_message": "给用户看的简洁中文回复，可用 Markdown",\n'
            '  "target_proposal": null 或 {\n'
            '    "plugin_id": "snake_case_id",\n'
            '    "display_name": "显示名称",\n'
            '    "supported_steps": ["ocr"],\n'
            '    "supported_modes": ["standard"]\n'
            "  }\n"
            "}\n\n"
            "规则：\n"
            "- 你的回答必须且只能是一个合法 JSON 对象。\n"
            "- JSON 的第一个非空字符必须是 `{`，最后一个非空字符必须是 `}`。\n"
            "- JSON 外面禁止输出任何额外内容，包括解释、前言、结语、Markdown 代码块、反引号、注释和空行。\n"
            "- 不要输出“下面是 JSON”“好的”之类的文字。\n"
            '- assistant_message 必须放在返回 JSON 的第一个字段。\n'
            "- modify 模式下不要重新选择其他插件。\n"
            "- create 模式下，若信息不足可让用户补充；若信息足够则给出一个明确 target_proposal。\n"
            "- assistant_message 要指出插件将作用于哪些步骤、预计做什么、缺什么信息。\n"
        )

    def _build_chat_messages(
        self,
        session: PluginAgentSession,
        system_prompt: str,
        skill_markdown: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\n以下是项目内置的插件开发 skill，请始终以它为准：\n\n"
                    + skill_markdown
                ),
            },
        ]
        for item in session.messages:
            messages.append({"role": item.role, "content": item.content})
        return messages

    def _build_execution_system_prompt(
        self,
        session: PluginAgentSession,
        tool_history: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        if session.locked_target is None:
            raise ValueError("执行阶段缺少 locked_target")
        history_json = (
            json.dumps(tool_history, ensure_ascii=False, indent=2)
            if tool_history
            else "[]"
        )
        return (
            "你是 Saber Translator 的内置插件编程 Agent，正在执行插件开发任务。\n"
            "你只能操作当前锁定插件目录，不能访问项目其他目录，不能切换到第二个插件。\n"
            "一次只返回一个工具动作，不要同时返回多个动作。\n"
            f"当前迭代: {iteration}\n"
            f"锁定插件: {session.locked_target.plugin_id}\n"
            f"插件目录: {session.locked_target.plugin_dir}\n"
            f"会话模式: {session.mode}\n"
            f"完整工具历史: {history_json}\n\n"
            "可用工具：list_files, read_file, write_file, delete_file, read_skill, validate_plugin, finish\n"
            "请只返回 JSON 对象，结构如下：\n"
            "{\n"
            '  "assistant_message": "给用户看的当前动作说明",\n'
            '  "action": {\n'
            '    "tool": "write_file",\n'
            '    "args": {"path": "plugin.py", "content": "...完整文件内容..."}\n'
            "  }\n"
            "}\n\n"
            "规则：\n"
            "- 你的回答必须且只能是一个合法 JSON 对象。\n"
            "- JSON 的第一个非空字符必须是 `{`，最后一个非空字符必须是 `}`。\n"
            "- JSON 外面禁止输出任何额外内容，包括解释、前言、结语、Markdown 代码块、反引号、注释和空行。\n"
            "- 不要输出“下面是 JSON”“好的”之类的文字。\n"
            '- assistant_message 必须放在返回 JSON 的第一个字段，action 必须紧随其后。\n'
            "- 修改文件时必须提供完整文件内容，不要只给 diff。\n"
            "- finish 前至少应完成一次 validate_plugin 并确保成功。\n"
            "- 优先保持实现简单、符合项目插件规范。\n"
        )

    def _build_execution_messages(
        self,
        session: PluginAgentSession,
        system_prompt: str,
        skill_markdown: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\n以下是项目内置的插件开发 skill，请严格遵守：\n\n"
                    + skill_markdown
                ),
            },
        ]
        for item in session.messages:
            messages.append({"role": item.role, "content": item.content})
        return messages

    @staticmethod
    def _extract_assistant_message_prefix(raw_text: str) -> tuple[str | None, bool]:
        match = _ASSISTANT_MESSAGE_PATTERN.search(raw_text)
        if not match:
            return None, False
        return _decode_json_string_prefix(raw_text[match.end() :])

    @staticmethod
    def _build_tool_call_payload(
        tool_name: str,
        tool_args: dict[str, Any],
        group_id: str,
    ) -> dict[str, Any]:
        path = tool_args.get("path", "")
        if tool_name == "write_file":
            summary = f"准备写入文件 {path or '未指定路径'}"
        elif tool_name == "read_file":
            summary = f"读取文件 {path or '未指定路径'}"
        elif tool_name == "list_files":
            summary = f"查看目录 {path or '.'}"
        elif tool_name == "delete_file":
            summary = f"删除文件 {path or '未指定路径'}"
        elif tool_name == "read_skill":
            summary = "读取内置插件开发 skill"
        elif tool_name == "validate_plugin":
            summary = "校验当前插件实现"
        else:
            summary = f"执行工具 {tool_name}"

        args_preview: dict[str, Any] = {}
        if path:
            args_preview["path"] = path
        if tool_name == "write_file":
            args_preview["content_length"] = len(tool_args["content"])

        return {
            "group_id": group_id,
            "tool": tool_name,
            "summary": summary,
            "args_preview": args_preview,
        }

    @classmethod
    def _build_tool_result_payload(
        cls,
        tool_name: str,
        tool_result: dict[str, Any],
        group_id: str,
    ) -> dict[str, Any]:
        success = tool_result["success"]
        raw_path = tool_result.get("path")
        if raw_path is not None and not isinstance(raw_path, str):
            raise TypeError("Plugin Agent 工具结果 path 必须是字符串")
        path = raw_path or ""
        summary = cls._summarize_tool_result(tool_name, tool_result, success)
        changed_files: list[str] = []
        file_previews: dict[str, str] = {}
        if tool_name == "write_file" and path:
            changed_files.append(path)
            preview = tool_result.get("preview")
            if isinstance(preview, str):
                file_previews[path] = preview
        elif tool_name == "delete_file" and path:
            changed_files.append(path)
        return {
            "group_id": group_id,
            "tool": tool_name,
            "summary": summary,
            "success": success,
            "changed_files": changed_files,
            "file_previews": file_previews,
            "debug_result": dict(tool_result),
        }

    @staticmethod
    def _build_validation_payload(
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        PluginAgentController._require_tool_result(validation_result)
        success = validation_result["success"]
        if success:
            plugin_id = validation_result.get("plugin_id")
            if plugin_id is not None and not isinstance(plugin_id, str):
                raise TypeError("Plugin Agent 校验结果 plugin_id 必须是字符串")
            plugin_label = plugin_id or "当前插件"
            package_version = validation_result.get("package_version")
            if package_version is not None and not isinstance(
                package_version,
                str,
            ):
                raise TypeError(
                    "Plugin Agent 校验结果 package_version 必须是字符串"
                )
            if package_version:
                plugin_label = f"{plugin_label} {package_version}"
            summary = f"插件校验通过：{plugin_label}"
        else:
            error = validation_result.get("error")
            if error is not None and not isinstance(error, str):
                raise TypeError("Plugin Agent 校验结果 error 必须是字符串")
            summary = f"插件校验失败：{error or '未知错误'}"
        return {
            "summary": summary,
            "success": success,
            "details": validation_result,
        }

    @staticmethod
    def _summarize_tool_result(
        tool_name: str,
        tool_result: dict[str, Any],
        success: bool,
    ) -> str:
        if tool_name == "write_file":
            return f"{'已写入' if success else '写入失败'} {tool_result.get('path') or '文件'}"
        if tool_name == "read_file":
            return f"{'已读取' if success else '读取失败'} {tool_result.get('path') or '文件'}"
        if tool_name == "list_files":
            entries = tool_result.get("entries") or []
            return f"目录扫描完成，共 {len(entries)} 项"
        if tool_name == "delete_file":
            return f"{'已删除' if success else '删除失败'} {tool_result.get('path') or '文件'}"
        if tool_name == "read_skill":
            return "已读取内置插件开发 skill"
        if tool_name == "validate_plugin":
            return "插件校验通过" if success else f"插件校验失败：{tool_result.get('error') or '未知错误'}"
        return f"{tool_name} {'执行成功' if success else '执行失败'}"
