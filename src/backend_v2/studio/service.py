"""API-executor handlers for durable Character Studio operations."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Iterator, Protocol

from sqlalchemy import Engine, select

from src.backend_v2.operations.repository import (
    OperationFence,
    OperationFenced,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import assets
from src.backend_v2.studio.repository import StudioRepository
from src.backend_v2.studio.pure import (
    apply_regex_scripts,
    match_lorebook,
    run_state_tasks,
    select_provider_section,
    sort_lorebook_hits,
    validate_current_document,
)


class StudioAlgorithms(Protocol):
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
        analysis_context: Mapping[str, Any] | None = None,
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> Mapping[str, Any]: ...

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system: str,
        config: Mapping[str, Any],
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> str: ...

    def summarize(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> Mapping[str, Any]: ...


class DefaultStudioAlgorithms:
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
        analysis_context: Mapping[str, Any] | None = None,
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> Mapping[str, Any]:
        context_json = json.dumps(
            dict(analysis_context or {}),
            ensure_ascii=False,
        )
        document_json = json.dumps(document, ensure_ascii=False)
        if section == "review":
            prompt = (
                "请结合漫画分析压缩上下文审查以下 Character Studio 角色文档，"
                "重点检查角色事实是否与原作一致、各字段是否自洽且可实际使用。"
                "只输出一个 JSON 对象，且只允许包含："
                "summary（字符串）、issues（字符串数组）、"
                "suggestions（字符串数组）。不要回传原文档。\n\n"
                f"漫画分析压缩上下文：\n{context_json}\n\n"
                f"当前角色文档：\n{document_json}"
            )
        else:
            contracts = {
                "identity": (
                    '{"identity":{"name":"角色名","aliases":[],'
                    '"description":"角色简介","personality":"性格",'
                    '"scenario":"当前场景"}}'
                ),
                "greetings": (
                    '{"coreMessages":{"first_message":"第一人称开场白",'
                    '"message_example":"示例对话","alternate_greetings":[],'
                    '"system_prompt":"","post_history_instructions":"",'
                    '"creator_notes":"","character_version":"2.0.0"}}'
                ),
                "lorebook": (
                    '{"lorebook":{"name":"世界书名称","entries":['
                    '{"id":"稳定唯一ID","keys":["触发词"],'
                    '"secondary_keys":[],"comment":"条目名称",'
                    '"content":"原作事实","constant":false,'
                    '"selective":false,"enabled":true,'
                    '"position":"before_char","priority":100,'
                    '"probability":100,"prevent_recursion":true}]}}'
                ),
                "regex": '{"regexScripts":[]}',
                "state-tasks": '{"stateTasks":[]}',
                "translate": (
                    '{"identity":{"name":"","aliases":[],'
                    '"description":"","personality":"","scenario":""},'
                    '"coreMessages":{"first_message":"",'
                    '"message_example":"","alternate_greetings":[],'
                    '"system_prompt":"","post_history_instructions":"",'
                    '"creator_notes":"","character_version":"2.0.0"},'
                    '"lorebook":{"name":"","entries":[]},'
                    '"regexScripts":[],"stateTasks":[]}'
                ),
                "full": (
                    '{"identity":{"name":"角色名","aliases":[],'
                    '"description":"角色简介","personality":"性格",'
                    '"scenario":"场景"},'
                    '"coreMessages":{"first_message":"第一人称开场白",'
                    '"message_example":"示例对话",'
                    '"alternate_greetings":[],'
                    '"system_prompt":"角色扮演约束",'
                    '"post_history_instructions":"",'
                    '"creator_notes":"基于原作分析生成",'
                    '"character_version":"2.0.0"},'
                    '"lorebook":{"name":"角色世界书","entries":[]},'
                    '"regexScripts":[],"stateTasks":[]}'
                ),
            }
            instruction = (
                "把当前角色文档中的自然语言内容完整翻译为中文；"
                "保留 ID、正则表达式、模板变量和数据结构"
                if section == "translate"
                else "依据漫画分析中的原作事实生成并补全指定区段"
            )
            full_requirement = (
                "identity、coreMessages、lorebook、regexScripts、"
                "stateTasks 五个顶层键必须全部出现；"
                "至少补全角色简介、性格、场景、第一人称开场白和世界书，"
                "没有必要生成脚本或任务时也必须返回对应空数组。"
                if section == "full"
                else ""
            )
            prompt = (
                f"请{instruction}。目标角色是当前文档的角色名或 source_character，"
                "不要混入其他角色的设定。漫画分析压缩上下文是生成事实依据，"
                "必须实际使用；当前文档已有的非空内容应在不冲突时保留。"
                f"只输出 JSON 对象，顶层结构必须为：{contracts[section]}。"
                f"{full_requirement}"
                "不要回传数据库元数据、revision、status、meta 或解释文字。\n\n"
                f"漫画分析压缩上下文：\n{context_json}\n\n"
                f"当前角色文档：\n{document_json}"
            )
        result = self._chat_json(
            prompt,
            config=config,
            on_chunk=on_chunk,
        )
        if not isinstance(result, Mapping):
            raise ValueError("Studio generation did not return a JSON object")
        return dict(result)

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system: str,
        config: Mapping[str, Any],
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> str:
        remote_messages: list[dict[str, Any]] = []
        system_parts = [system] if system else []
        has_image_attachments = False
        for index, raw in enumerate(messages):
            if not isinstance(raw, Mapping):
                raise ValueError(f"Studio chat message {index} must be an object")
            role = _required_string(raw.get("role"), f"Studio chat message {index} role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Studio chat message {index} role is invalid")
            content = _string(raw.get("content"), f"Studio chat message {index} content")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            attachments = raw.get("attachmentDataUrls", [])
            if not isinstance(attachments, list) or not all(
                isinstance(value, str) and value
                for value in attachments
            ):
                raise ValueError(
                    f"Studio chat message {index} attachmentDataUrls is invalid"
                )
            if role == "user" and attachments:
                has_image_attachments = True
                parts: list[dict[str, Any]] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": value},
                    }
                    for value in attachments
                ]
                parts.append({"type": "text", "text": content})
                remote_messages.append({"role": role, "content": parts})
            else:
                remote_messages.append({"role": role, "content": content})
        if system_parts:
            remote_messages.insert(
                0,
                {"role": "system", "content": "\n\n".join(system_parts)},
            )
        result = self._complete(
            remote_messages,
            config=config,
            temperature=0.7,
            force_json=False,
            on_chunk=on_chunk,
            prefer_vlm=has_image_attachments,
        )
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Studio chat did not return response text")
        return result

    def summarize(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            "总结以下角色对话，保留事实、关系、变量变化和未解决事项。"
            "输出 JSON 对象，至少包含 summary。\n\n"
            + json.dumps(list(messages), ensure_ascii=False)
        )
        result = self._chat_json(
            prompt,
            config=config,
            on_chunk=on_chunk,
        )
        if not isinstance(result, Mapping):
            raise ValueError("Studio summary did not return a JSON object")
        summary = result.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Studio summary did not return summary text")
        return {"summary": summary.strip()}

    def _chat_json(
        self,
        prompt: str,
        *,
        config: Mapping[str, Any],
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> object:
        text = self._complete(
            [{"role": "user", "content": prompt}],
            config=config,
            temperature=0.3,
            force_json=True,
            on_chunk=on_chunk,
        )
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    @staticmethod
    def _complete(
        messages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
        temperature: float,
        force_json: bool,
        on_chunk: Callable[[str, str], None] | None,
        prefer_vlm: bool = False,
    ) -> str:
        from src.shared.ai_transport import (
            OpenAICompatibleChatTransport,
            UnifiedChatRequest,
        )
        from src.shared.openai_execution import (
            build_openai_compatible_runtime_options,
        )
        from src.shared.openai_options import OpenAICompatibleOptions

        section = _provider_config(config, prefer_vlm=prefer_vlm)
        provider = _required_string(
            section.get("provider"),
            "Studio provider",
        )
        model = _required_string(section.get("model"), "Studio model")
        if not provider or not model:
            raise ValueError("Studio chat provider/model is not configured")
        options = OpenAICompatibleOptions.from_dict(
            _required_mapping(
                section.get("openai_options"),
                "Studio openai_options",
            )
        )
        options.request.force_json_output = force_json
        if options.request.temperature is None:
            options.request.temperature = temperature
        request = UnifiedChatRequest(
            provider=provider,
            api_key=_string(section.get("api_key"), "Studio api_key"),
            model=model,
            credential_version_id=(
                _required_string(
                    section["credential_version_id"],
                    "Studio credential_version_id",
                )
                if section.get("credential_version_id") is not None
                else None
            ),
            messages=[dict(message) for message in messages],
            base_url=(
                _required_string(section["base_url"], "Studio base_url")
                if section.get("base_url") is not None
                else None
            ),
            openai_options=options,
            runtime_options=build_openai_compatible_runtime_options(
                timeout=_positive_number(
                    section.get("timeout_seconds"),
                    "Studio timeout_seconds",
                ),
                on_stream_chunk=on_chunk,
            ),
        )
        return OpenAICompatibleChatTransport().complete(request)


class StudioOperationService:
    def __init__(
        self,
        *,
        engine: Engine,
        data_root: Path | None = None,
        repository: StudioRepository | None = None,
        algorithms: StudioAlgorithms | None = None,
    ) -> None:
        self.engine = engine
        self.storage = (
            AssetStorageService(data_root, engine)
            if data_root is not None
            else None
        )
        self.repository = repository or StudioRepository(engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or DefaultStudioAlgorithms()

    def handle(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = _required_mapping(
            operation.get("request"),
            "Studio operation request",
        )
        kind = _required_string(
            operation.get("kind"),
            "Studio operation kind",
        )
        on_chunk = self._event_callback(fence)
        if kind == "studio_generate":
            _exact_keys(
                request,
                {"config", "document", "section", "analysisContext"},
                "Studio generation request",
            )
            config = self._with_credentials(
                _required_mapping(
                    request.get("config"),
                    "Studio generation config",
                ),
            )
            document = _current_document(request.get("document"))
            section = _required_string(
                request.get("section"),
                "Studio generation section",
            )
            raw_analysis = request.get("analysisContext")
            analysis_context = (
                None
                if raw_analysis is None
                else _required_mapping(
                    raw_analysis,
                    "Studio analysis context",
                )
            )
            generated = self.algorithms.generate(
                document,
                section=section,
                config=config,
                analysis_context=analysis_context,
                on_chunk=on_chunk,
            )
            if section == "review":
                return self.repository.publish_generate(
                    fence,
                    generated_document=document,
                    review=_normalize_review(generated),
                )
            _validate_generated_payload(
                document,
                generated,
                section=section,
            )
            merged = _apply_generated_section(
                document,
                generated,
                section=section,
            )
            return self.repository.publish_generate(
                fence,
                generated_document=merged,
            )
        if kind == "studio_chat":
            _exact_keys(
                request,
                {
                    "config",
                    "document",
                    "messages",
                    "runtimeState",
                    "summaryBlocks",
                    "summaryThroughMessageId",
                    "variables",
                },
                "Studio chat request",
            )
            return self._chat(
                fence,
                request,
                input_assets=_required_mapping(
                    operation.get("inputs"),
                    "Studio operation inputs",
                ),
                config=_required_mapping(
                    request.get("config"),
                    "Studio chat config",
                ),
            )
        if kind == "studio_summary":
            _exact_keys(
                request,
                {"config", "messages"},
                "Studio summary request",
            )
            config = self._with_credentials(
                _required_mapping(
                    request.get("config"),
                    "Studio summary config",
                ),
            )
            messages = _operation_messages(
                request.get("messages"),
                label="Studio summary messages",
                require_nonempty=True,
            )
            summary = self.algorithms.summarize(
                messages,
                config=config,
                on_chunk=on_chunk,
            )
            if not isinstance(summary, Mapping):
                raise ValueError("Studio summary did not return a JSON object")
            if set(summary) != {"summary"}:
                raise ValueError("Studio summary fields are invalid")
            summary_text = summary.get("summary")
            if not isinstance(summary_text, str) or not summary_text.strip():
                raise ValueError("Studio summary did not return summary text")
            return self.repository.publish_summary(
                fence,
                summary={"summary": summary_text.strip()},
            )
        raise ValueError(f"unsupported Studio operation: {kind}")

    def _chat(
        self,
        fence: OperationFence,
        request: Mapping[str, Any],
        *,
        input_assets: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        document = _current_document(request.get("document"))
        messages = _operation_messages(
            request.get("messages"),
            label="Studio chat messages",
            require_nonempty=True,
        )
        last = messages[-1]
        if last["role"] != "user":
            raise ValueError("Studio chat last message must be a user message")
        raw_user = last["content"]
        runtime_state = deepcopy(
            _required_mapping(
                request.get("runtimeState"),
                "Studio chat runtimeState",
            )
        )
        variables = deepcopy(
            _required_mapping(
                request.get("variables"),
                "Studio chat variables",
            )
        )
        session_work = {
            "variables": variables,
            "_runtime": runtime_state,
        }
        _, prompt_user, regex_hits = apply_regex_scripts(
            raw_user,
            document["regexScripts"],
            placement=1,
            respect_run_on_edit=True,
        )
        lorebook_hits = sort_lorebook_hits(
            match_lorebook(
                document["lorebook"]["entries"],
                prompt_user,
                session=session_work,
            )
        )
        runtime_log: list[dict[str, Any]] = list(regex_hits)
        runtime_log.extend(
            {
                "type": "lorebook",
                "id": entry["id"],
                "comment": entry["comment"],
            }
            for entry in lorebook_hits
        )
        runtime_log.extend(
            run_state_tasks(
                session_work,
                document["stateTasks"],
                event="message_received",
            )
        )
        summaries = _summary_blocks(request.get("summaryBlocks"))
        system = _build_system_prompt(
            document=document,
            variables=variables,
            summaries=summaries,
            lorebook_hits=lorebook_hits,
        )
        allowed_asset_ids: set[str] = set()
        for role, value in input_assets.items():
            if not isinstance(role, str) or not role.startswith("attachment:"):
                raise ValueError("Studio operation input role is invalid")
            allowed_asset_ids.add(
                _required_string(value, "Studio operation input asset id")
            )
        asset_data_urls = self._asset_data_urls(allowed_asset_ids)
        summarized_through = request.get("summaryThroughMessageId")
        if summarized_through is not None and not isinstance(
            summarized_through,
            str,
        ):
            raise ValueError(
                "Studio chat summaryThroughMessageId must be a string or null"
            )
        if summarized_through is not None and not any(
            item["messageId"] == summarized_through for item in messages
        ):
            raise ValueError(
                "Studio chat summaryThroughMessageId does not identify a message"
            )
        include = summarized_through is None
        conversation: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not include:
                if message["messageId"] == summarized_through:
                    include = True
                continue
            attachment_urls: list[str] = []
            for attachment in message["attachments"]:
                asset_id = attachment["assetId"]
                if asset_id not in allowed_asset_ids:
                    raise ValueError(
                        "Studio chat attachment is not bound to the operation"
                    )
                data_url = asset_data_urls.get(asset_id)
                if data_url is None:
                    raise ValueError("Studio chat attachment is unavailable")
                attachment_urls.append(data_url)
            conversation.append(
                {
                    "role": message["role"],
                    "content": (
                        prompt_user
                        if index == len(messages) - 1
                        else message["content"]
                    ),
                    "attachmentDataUrls": attachment_urls,
                }
            )
        self.repository.operations.append_event(
            fence,
            event_type="prompt_ready",
            payload={
                "messageCount": len(conversation),
                "attachmentCount": sum(
                    len(item["attachmentDataUrls"])
                    for item in conversation
                ),
            },
        )
        assistant = self.algorithms.chat(
            messages=conversation,
            system=system,
            config=self._with_credentials(
                config,
                prefer_vlm=any(
                    item["attachmentDataUrls"] for item in conversation
                ),
            ),
            on_chunk=self._event_callback(fence),
        )
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError("Studio chat did not return response text")
        visible_assistant, _, output_hits = apply_regex_scripts(
            assistant,
            document["regexScripts"],
            placement=2,
            respect_run_on_edit=True,
        )
        runtime_log.extend(output_hits)
        runtime_log.extend(
            run_state_tasks(
                session_work,
                document["stateTasks"],
                event="message_sent",
            )
        )
        return self.repository.publish_chat(
            fence,
            content=visible_assistant,
            runtime_log=runtime_log,
            variables=session_work["variables"],
            runtime_state=session_work["_runtime"],
        )

    def prompt_preview(
        self,
        *,
        document: Mapping[str, Any],
        session: Mapping[str, Any],
    ) -> dict[str, Any]:
        document = _current_document(document)
        messages = _operation_messages(
            session.get("messages"),
            label="Studio session messages",
        )
        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        last_user = (
            messages[last_user_index]["content"]
            if last_user_index is not None
            else ""
        )
        _, prompt_user, _regex_hits = apply_regex_scripts(
            last_user,
            document["regexScripts"],
            placement=1,
            respect_run_on_edit=True,
        )
        work = {
            "variables": deepcopy(
                _required_mapping(
                    session.get("variables"),
                    "Studio session variables",
                )
            ),
            "_runtime": deepcopy(
                _required_mapping(
                    session.get("runtimeState"),
                    "Studio session runtimeState",
                )
            ),
        }
        hits = sort_lorebook_hits(
            match_lorebook(
                document["lorebook"]["entries"],
                prompt_user,
                session=work,
            )
        )
        system = _build_system_prompt(
            document=document,
            variables=work["variables"],
            summaries=_summary_blocks(session.get("summaryBlocks")),
            lorebook_hits=hits,
        )
        summarized_through = session.get("summaryThroughMessageId")
        if summarized_through is not None and not isinstance(
            summarized_through,
            str,
        ):
            raise ValueError(
                "Studio session summaryThroughMessageId must be a string or null"
            )
        if summarized_through is not None and not any(
            message["messageId"] == summarized_through
            for message in messages
        ):
            raise ValueError(
                "Studio session summaryThroughMessageId does not identify a message"
            )
        include = summarized_through is None
        visible: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not include:
                if message.get("messageId") == summarized_through:
                    include = True
                continue
            visible.append(
                {
                    "role": message["role"],
                    "content": (
                        prompt_user
                        if index == last_user_index
                        else message["content"]
                    ),
                    "assetIds": [
                        attachment["assetId"]
                        for attachment in message["attachments"]
                    ],
                }
            )
        lorebook_hits = []
        for index, entry in enumerate(hits):
            entry_id = entry["id"]
            comment = entry["comment"]
            lorebook_hits.append({"id": entry_id, "comment": comment})
        return {
            "system": system,
            "messages": visible,
            "lorebookHits": lorebook_hits,
        }

    def agent_chunks(
        self,
        *,
        document: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        cancelled: threading.Event,
    ) -> Iterator[str]:
        document = _current_document(document)
        config = _required_mapping(config, "Studio agent config")
        agent_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            raw = _required_mapping(
                message,
                f"Studio agent message {index}",
            )
            _exact_keys(
                raw,
                {"role", "content"},
                f"Studio agent message {index}",
            )
            role = _required_string(
                raw.get("role"),
                f"Studio agent message {index} role",
            )
            if role not in {"user", "assistant"}:
                raise ValueError(f"Studio agent message {index} role is invalid")
            agent_messages.append(
                {
                    "role": role,
                    "content": _string(
                        raw.get("content"),
                        f"Studio agent message {index} content",
                    ),
                }
            )
        chunks: queue.Queue[object] = queue.Queue(maxsize=128)
        done = object()
        emitted_stream_chunk = threading.Event()
        streamed_text = ""

        class AgentDisconnected(RuntimeError):
            pass

        def on_chunk(chunk: str, full_text: str) -> None:
            nonlocal streamed_text
            if not isinstance(chunk, str) or not isinstance(full_text, str):
                raise ValueError("Studio agent returned an invalid stream chunk")
            if not chunk:
                return
            if full_text != streamed_text + chunk:
                raise ValueError("Studio agent stream content is inconsistent")
            streamed_text = full_text
            emitted_stream_chunk.set()
            while not cancelled.is_set():
                try:
                    chunks.put(chunk, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise AgentDisconnected("Studio agent connection closed")

        def publish_control(item: object) -> None:
            while not cancelled.is_set():
                try:
                    chunks.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def run() -> None:
            try:
                system = (
                    "你是 Character Studio 卡片助手。根据当前角色卡提出具体改进。"
                    "需要结构化修改时输出 ```json:patch 代码块，内容必须是对象，"
                    "可用顶层字段仅限 set、greeting_add、worldbook_add、"
                    "worldbook_update、worldbook_delete、regex_add、regex_update、"
                    "regex_delete、task_add、task_update、task_delete。"
                    "普通字段修改放入 set，键使用点路径，例如 "
                    '{"set":{"identity.description":"新的简介"}}；'
                    "不要输出 RFC 6902 的操作数组。需要视觉预览时可输出 "
                    "```html 代码块。不要声称已直接保存文档。\n\n当前文档：\n"
                    + json.dumps(document, ensure_ascii=False)
                )
                result = self.algorithms.chat(
                    messages=agent_messages,
                    system=system,
                    config=self._with_credentials(config),
                    on_chunk=on_chunk,
                )
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("Studio agent did not return response text")
                # The saved model configuration may intentionally disable
                # streaming. In that mode the transport returns the complete
                # response without invoking ``on_chunk``; the SSE endpoint
                # still has to deliver that response to the browser.
                if emitted_stream_chunk.is_set():
                    if result != streamed_text:
                        raise ValueError("Studio agent stream result is inconsistent")
                else:
                    publish_control(result)
            except Exception as exc:
                if not isinstance(exc, AgentDisconnected):
                    publish_control(exc)
            finally:
                publish_control(done)

        thread = threading.Thread(
            target=run,
            name="studio-transient-agent",
            daemon=True,
        )
        thread.start()
        try:
            while True:
                item = chunks.get()
                if item is done:
                    return
                if isinstance(item, Exception):
                    raise item
                if not isinstance(item, str):
                    raise TypeError("Studio agent produced an invalid stream item")
                yield item
        finally:
            cancelled.set()

    def _event_callback(
        self,
        fence: OperationFence,
    ) -> Callable[[str, str], None]:
        previous_full_text = ""

        def emit(chunk: str, full_text: str) -> None:
            nonlocal previous_full_text
            if not isinstance(chunk, str) or not isinstance(full_text, str):
                raise ValueError("Studio provider returned an invalid stream chunk")
            if not chunk:
                return
            if full_text != previous_full_text + chunk:
                raise ValueError("Studio provider stream content is inconsistent")
            previous_full_text = full_text
            self.repository.operations.append_event(
                fence,
                event_type="chunk",
                payload={
                    "text": chunk,
                    "totalCharacters": len(full_text),
                },
            )

        return emit

    def _asset_data_urls(self, asset_ids: set[str]) -> dict[str, str]:
        if self.storage is None or not asset_ids:
            return {}
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        assets.c.id,
                        assets.c.relative_path,
                        assets.c.mime_type,
                        assets.c.integrity_status,
                    ).where(assets.c.id.in_(tuple(asset_ids)))
                ).mappings()
            )
        result: dict[str, str] = {}
        for row in rows:
            if row["integrity_status"] != "ok":
                continue
            path = self.storage.resolve_relative_path(
                str(row["relative_path"])
            )
            if path.is_file():
                result[str(row["id"])] = (
                    f"data:{row['mime_type']};base64,"
                    + base64.b64encode(path.read_bytes()).decode("ascii")
                )
        return result

    def _with_credentials(
        self,
        config: Mapping[str, Any],
        *,
        prefer_vlm: bool = False,
    ) -> dict[str, Any]:
        section_name, _ = select_provider_section(
            config,
            prefer_vlm=prefer_vlm,
        )
        try:
            return self.credentials.resolve_credential_sections(
                config,
                (section_name,),
            )
        except LookupError as exc:
            raise OperationFenced(
                "Studio credential version no longer exists"
            ) from exc


def _provider_config(
    config: Mapping[str, Any],
    *,
    prefer_vlm: bool = False,
) -> dict[str, Any]:
    _, section = select_provider_section(
        config,
        prefer_vlm=prefer_vlm,
    )
    raw_options = section.get("openai_options")
    options = (
        {}
        if raw_options is None
        else _required_mapping(raw_options, "Studio provider openai_options")
    )
    timeout = section.get("timeout_seconds")
    base_url = section.get("custom_base_url")
    if base_url == "":
        base_url = None
    return {
        "provider": section.get("provider", ""),
        "api_key": section.get("api_key", ""),
        "model": section.get("model_name", ""),
        "base_url": base_url,
        "openai_options": options,
        "timeout_seconds": 120 if timeout is None else timeout,
    }


def _build_system_prompt(
    *,
    document: Mapping[str, Any],
    variables: Mapping[str, Any],
    summaries: object,
    lorebook_hits: Sequence[Mapping[str, Any]],
) -> str:
    identity = document["identity"]
    core = document["coreMessages"]
    lorebook_text = "\n".join(
        entry["content"] for entry in lorebook_hits
    )
    return "\n\n".join(
        value
        for value in (
            core["system_prompt"],
            f"角色：{identity['name']}",
            identity["description"],
            identity["personality"],
            identity["scenario"],
            core["post_history_instructions"],
            f"变量：{json.dumps(dict(variables), ensure_ascii=False)}",
            f"会话摘要：{json.dumps(summaries, ensure_ascii=False)}",
            f"世界书：{lorebook_text}",
        )
        if value
    )


def _normalize_review(generated: Mapping[str, Any]) -> dict[str, Any]:
    if not set(generated).issubset({"summary", "issues", "suggestions"}):
        raise ValueError("Studio review fields are invalid")
    summary = generated.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Studio review did not return a summary")

    def string_list(field: str) -> list[str]:
        value = generated.get(field)
        if value is None:
            return []
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"Studio review {field} must be a string array")
        return [item.strip() for item in value if item.strip()]

    return {
        "summary": summary.strip(),
        "issues": string_list("issues"),
        "suggestions": string_list("suggestions"),
    }


def _validate_generated_payload(
    document: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    section: str,
) -> None:
    section_fields: dict[str, tuple[str, type]] = {
        "identity": ("identity", Mapping),
        "greetings": ("coreMessages", Mapping),
        "lorebook": ("lorebook", Mapping),
        "regex": ("regexScripts", list),
        "state-tasks": ("stateTasks", list),
    }
    if section in section_fields:
        field, expected_type = section_fields[section]
        if set(generated) != {field}:
            raise ValueError("Studio generation fields are invalid")
        if not isinstance(generated.get(field), expected_type):
            expected = "an object" if expected_type is Mapping else "an array"
            raise ValueError(
                f"Studio generation field {field} must be {expected}"
            )
        return
    if section not in {"full", "translate"}:
        raise ValueError("unsupported Studio generation section")
    frozen = set(document["status"]["frozen_sections"])
    section_keys = {
        "identity": "identity",
        "greetings": "coreMessages",
        "lorebook": "lorebook",
        "regex": "regexScripts",
        "state-tasks": "stateTasks",
    }
    if not set(generated).issubset(set(section_keys.values())):
        raise ValueError("Studio generation fields are invalid")
    missing = [
        key
        for section_name, key in section_keys.items()
        if section_name not in frozen and key not in generated
    ]
    if missing:
        raise ValueError(
            "Studio full-document generation omitted required fields: "
            + ", ".join(missing)
        )
    for key in ("identity", "coreMessages", "lorebook"):
        if key in generated and not isinstance(generated[key], Mapping):
            raise ValueError(
                f"Studio generation field {key} must be an object"
            )
    for key in ("regexScripts", "stateTasks"):
        if key in generated and not isinstance(generated[key], list):
            raise ValueError(
                f"Studio generation field {key} must be an array"
            )


def _apply_generated_section(
    document: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    section: str,
) -> dict[str, Any]:
    result = deepcopy(dict(document))
    frozen = set(result["status"]["frozen_sections"])
    if section in frozen:
        return result
    if section == "identity":
        identity = dict(generated["identity"])
        result["identity"] = {
            **result["identity"],
            **identity,
        }
        name = result["identity"].get("name")
        if isinstance(name, str) and name.strip():
            result["title"] = name.strip()
            result["identity"]["name"] = name.strip()
            result["meta"]["title"] = name.strip()
    elif section == "greetings":
        value = dict(generated["coreMessages"])
        result["coreMessages"] = {
            **result["coreMessages"],
            **value,
        }
    elif section == "lorebook":
        result["lorebook"] = dict(generated["lorebook"])
    elif section == "regex":
        result["regexScripts"] = list(generated.get("regexScripts", []))
    elif section == "state-tasks":
        result["stateTasks"] = list(generated.get("stateTasks", []))
    elif section in {"translate", "full"}:
        section_keys = {
            "identity": "identity",
            "greetings": "coreMessages",
            "lorebook": "lorebook",
            "regex": "regexScripts",
            "state-tasks": "stateTasks",
        }
        for section_name, key in section_keys.items():
            if key in generated and section_name not in frozen:
                result[key] = deepcopy(generated[key])
        name = result["identity"].get("name")
        if (
            isinstance(name, str)
            and name.strip()
            and "identity" not in frozen
        ):
            result["title"] = name.strip()
            result["identity"]["name"] = name.strip()
            result["meta"]["title"] = name.strip()
    else:
        raise ValueError("unsupported Studio generation section")
    return result


def _required_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _exact_keys(
    value: Mapping[str, Any],
    fields: set[str],
    label: str,
) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields are invalid")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _required_string(value: object, label: str) -> str:
    result = _string(value, label)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _current_document(value: object) -> dict[str, Any]:
    document = _required_mapping(value, "Studio document")
    book_id = _required_string(
        document.get("bookId"),
        "Studio document bookId",
    )
    return validate_current_document(document, book_id=book_id)


def _operation_messages(
    value: object,
    *,
    label: str,
    require_nonempty: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    if require_nonempty and not value:
        raise ValueError(f"{label} must not be empty")
    result: list[dict[str, Any]] = []
    message_ids: set[str] = set()
    for index, raw in enumerate(value):
        message = _required_mapping(raw, f"{label}[{index}]")
        message_id = _required_string(
            message.get("messageId"),
            f"{label}[{index}].messageId",
        )
        if message_id in message_ids:
            raise ValueError(f"{label} contains duplicate message IDs")
        message_ids.add(message_id)
        role = _required_string(
            message.get("role"),
            f"{label}[{index}].role",
        )
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{label}[{index}].role is invalid")
        content = _string(
            message.get("content"),
            f"{label}[{index}].content",
        )
        attachments = message.get("attachments")
        if not isinstance(attachments, list):
            raise ValueError(f"{label}[{index}].attachments must be an array")
        normalized_attachments: list[dict[str, Any]] = []
        for attachment_index, raw_attachment in enumerate(attachments):
            attachment = _required_mapping(
                raw_attachment,
                f"{label}[{index}].attachments[{attachment_index}]",
            )
            _required_string(
                attachment.get("assetId"),
                f"{label}[{index}].attachments[{attachment_index}].assetId",
            )
            normalized_attachments.append(attachment)
        result.append(
            {
                **message,
                "messageId": message_id,
                "role": role,
                "content": content,
                "attachments": normalized_attachments,
            }
        )
    return result


def _summary_blocks(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Studio summaryBlocks must be an array")
    result = []
    for index, raw in enumerate(value):
        block = _required_mapping(raw, f"Studio summaryBlocks[{index}]")
        _exact_keys(block, {"summary"}, f"Studio summaryBlocks[{index}]")
        result.append(
            {
                "summary": _required_string(
                    block.get("summary"),
                    f"Studio summaryBlocks[{index}].summary",
                )
            }
        )
    return result
