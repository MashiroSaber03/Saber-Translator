from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.shared.ai_providers import (
    PLUGIN_AGENT_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
)
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleSyncExecutor,
    build_openai_compatible_runtime_options,
    parse_json_block_from_text,
)
from src.shared.openai_options import OpenAICompatibleOptions

from .models import PluginAgentSession

logger = logging.getLogger("PluginAgent.Controller")
_ASSISTANT_MESSAGE_PATTERN = re.compile(r'"assistant_message"\s*:\s*"')


def _decode_json_string_prefix(raw_text: str) -> Tuple[str, bool]:
    decoded: List[str] = []
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
    def __init__(self, transport: Optional[OpenAICompatibleChatTransport] = None) -> None:
        self.transport = transport or OpenAICompatibleChatTransport()
        self.executor = OpenAICompatibleSyncExecutor(self.transport)

    def plan_turn(
        self,
        session: PluginAgentSession,
        skill_markdown: str,
        agent_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = self._build_planning_system_prompt(session)
        messages = self._build_chat_messages(session, system_prompt, skill_markdown)
        return self._call_agent_json(messages, agent_config, label="PluginAgent-Planning")

    def execute(
        self,
        session: PluginAgentSession,
        skill_markdown: str,
        agent_config: Dict[str, Any],
        tool_executor,
        emit_event,
    ) -> Dict[str, Any]:
        recent_results: List[Dict[str, Any]] = []
        last_validation: Optional[Dict[str, Any]] = None

        for iteration in range(1, 13):
            if tool_executor.is_cancelled():
                raise RuntimeError("任务已取消")

            system_prompt = self._build_execution_system_prompt(session, recent_results, iteration)
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

            envelope = self._call_agent_json(
                messages,
                agent_config,
                label="PluginAgent-Execution",
                on_stream_chunk=lambda _chunk, content: emit_streaming_assistant(content),
            )

            assistant_message = str(envelope.get("assistant_message") or "").strip()
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

            action = envelope.get("action") or {}
            tool_name = str(action.get("tool") or "").strip()
            if not tool_name:
                raise ValueError("Agent 未返回有效工具动作")

            if tool_name == "finish":
                final_validation = last_validation or tool_executor.validate_plugin()
                if not final_validation.get("success"):
                    raise ValueError(final_validation.get("error") or "插件校验失败")
                return {
                    "assistant_message": assistant_message or "插件任务完成。",
                    "refresh_plugins": True,
                    "validation": final_validation,
                }

            tool_args = action.get("args") if isinstance(action.get("args"), dict) else {}
            group_id = f"tool-{iteration}"
            emit_event("tool_call", self._build_tool_call_payload(tool_name, tool_args, group_id))
            tool_result = tool_executor.run_tool(tool_name, tool_args)
            emit_event("tool_result", self._build_tool_result_payload(tool_name, tool_result, group_id))

            if tool_name == "validate_plugin":
                last_validation = tool_result
                emit_event("validation", self._build_validation_payload(tool_result))

            recent_results.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": self._shrink_tool_result(tool_result),
                }
            )
            recent_results = recent_results[-8:]

        raise RuntimeError("Agent 超过最大迭代次数仍未完成")

    def _call_agent_json(
        self,
        messages: List[Dict[str, Any]],
        agent_config: Dict[str, Any],
        *,
        label: str,
        on_stream_chunk: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        provider = normalize_provider_id(agent_config.get("provider"))
        api_key = agent_config.get("api_key", "")
        model_name = agent_config.get("model_name", "")
        custom_base_url = agent_config.get("custom_base_url") or None
        openai_options = agent_config.get("openai_options")
        if not isinstance(openai_options, OpenAICompatibleOptions):
            raise ValueError("agent_config.openai_options 必须是 OpenAICompatibleOptions")

        if not provider_supports_capability(provider, PLUGIN_AGENT_CAPABILITY):
            raise ValueError(f"不支持的插件 Agent 服务商: {provider}")

        manifest = get_provider_manifest(provider)
        if manifest.requires_api_key and not api_key:
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
                force_json_output=openai_options.request.force_json_output,
            ),
            logger_instance=logger,
        )
        return result.parsed

    @staticmethod
    def _parse_agent_envelope(content: str, *, force_json_output: bool) -> Dict[str, Any]:
        if force_json_output:
            parsed = json.loads(content)
        else:
            parsed = parse_json_block_from_text(content)
        if not isinstance(parsed, dict):
            raise ValueError("Agent 返回结果必须是 JSON 对象")
        return parsed

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
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": "以下是项目内置的插件开发 skill，请始终以它为准：\n\n" + skill_markdown,
            },
        ]
        for item in session.messages[-8:]:
            messages.append({"role": item.role, "content": item.content})
        return messages

    def _build_execution_system_prompt(
        self,
        session: PluginAgentSession,
        recent_results: List[Dict[str, Any]],
        iteration: int,
    ) -> str:
        if session.locked_target is None:
            raise ValueError("执行阶段缺少 locked_target")
        recent_json = json.dumps(recent_results, ensure_ascii=False, indent=2) if recent_results else "[]"
        return (
            "你是 Saber Translator 的内置插件编程 Agent，正在执行插件开发任务。\n"
            "你只能操作当前锁定插件目录，不能访问项目其他目录，不能切换到第二个插件。\n"
            "一次只返回一个工具动作，不要同时返回多个动作。\n"
            f"当前迭代: {iteration}/12\n"
            f"锁定插件: {session.locked_target.plugin_id}\n"
            f"插件目录: {session.locked_target.plugin_dir}\n"
            f"会话模式: {session.mode}\n"
            f"近期工具结果: {recent_json}\n\n"
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
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": "以下是项目内置的插件开发 skill，请严格遵守：\n\n" + skill_markdown,
            },
        ]
        for item in session.messages[-10:]:
            messages.append({"role": item.role, "content": item.content})
        return messages

    @staticmethod
    def _shrink_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
        raw = dict(result)
        content = raw.get("content")
        if isinstance(content, str) and len(content) > 1200:
            raw["content"] = content[:1200] + "\n...[truncated]..."
        preview = raw.get("preview")
        if isinstance(preview, str) and len(preview) > 1200:
            raw["preview"] = preview[:1200] + "\n...[truncated]..."
        return raw

    @staticmethod
    def _extract_assistant_message_prefix(raw_text: str) -> Tuple[Optional[str], bool]:
        match = _ASSISTANT_MESSAGE_PATTERN.search(raw_text)
        if not match:
            return None, False
        return _decode_json_string_prefix(raw_text[match.end() :])

    @staticmethod
    def _build_tool_call_payload(tool_name: str, tool_args: Dict[str, Any], group_id: str) -> Dict[str, Any]:
        path = str(tool_args.get("path") or tool_args.get("base_path") or "").strip()
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

        args_preview: Dict[str, Any] = {}
        if path:
            args_preview["path"] = path
        if tool_name == "write_file":
            args_preview["content_length"] = len(str(tool_args.get("content") or ""))

        return {
            "group_id": group_id,
            "tool": tool_name,
            "summary": summary,
            "args_preview": args_preview,
        }

    @classmethod
    def _build_tool_result_payload(cls, tool_name: str, tool_result: Dict[str, Any], group_id: str) -> Dict[str, Any]:
        success = bool(tool_result.get("success", True))
        path = str(tool_result.get("path") or "").strip()
        summary = cls._summarize_tool_result(tool_name, tool_result, success)
        changed_files: List[str] = []
        file_previews: Dict[str, str] = {}
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
            "debug_result": cls._shrink_tool_result(tool_result),
            "result": cls._shrink_tool_result(tool_result),
        }

    @staticmethod
    def _build_validation_payload(validation_result: Dict[str, Any]) -> Dict[str, Any]:
        success = bool(validation_result.get("success"))
        if success:
            plugin_meta = validation_result.get("plugin") or {}
            plugin_label = str(plugin_meta.get("display_name") or plugin_meta.get("id") or "当前插件")
            summary = f"插件校验通过：{plugin_label}"
        else:
            summary = f"插件校验失败：{validation_result.get('error') or '未知错误'}"
        return {
            "summary": summary,
            "success": success,
            "details": validation_result,
        }

    @staticmethod
    def _summarize_tool_result(tool_name: str, tool_result: Dict[str, Any], success: bool) -> str:
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
