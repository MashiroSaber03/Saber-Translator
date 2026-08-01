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
        has_image_attachments = False
        if system:
            remote_messages.append({"role": "system", "content": system})
        for raw in messages:
            role = str(raw.get("role", "assistant"))
            content = str(raw.get("content", ""))
            attachments = raw.get("attachmentDataUrls", [])
            if role == "user" and isinstance(attachments, list) and attachments:
                has_image_attachments = True
                parts: list[dict[str, Any]] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": str(value)},
                    }
                    for value in attachments
                    if isinstance(value, str)
                ]
                parts.append({"type": "text", "text": content})
                remote_messages.append({"role": role, "content": parts})
            else:
                remote_messages.append({"role": role, "content": content})
        return self._complete(
            remote_messages,
            config=config,
            temperature=0.7,
            force_json=False,
            on_chunk=on_chunk,
            prefer_vlm=has_image_attachments,
        )

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
        return (
            dict(result)
            if isinstance(result, Mapping)
            else {"summary": str(result)}
        )

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
        provider = str(section.get("provider", ""))
        model = str(section.get("model", ""))
        if not provider or not model:
            raise ValueError("Studio chat provider/model is not configured")
        options = OpenAICompatibleOptions.from_dict(
            _object(section.get("openai_options"))
        )
        options.request.force_json_output = force_json
        if options.request.temperature is None:
            options.request.temperature = temperature
        request = UnifiedChatRequest(
            provider=provider,
            api_key=str(section.get("api_key", "")),
            model=model,
            credential_version_id=(
                str(section["credential_version_id"])
                if section.get("credential_version_id")
                else None
            ),
            messages=[dict(message) for message in messages],
            base_url=(
                str(section["base_url"])
                if section.get("base_url")
                else None
            ),
            openai_options=options,
            runtime_options=build_openai_compatible_runtime_options(
                timeout=float(section.get("timeout_seconds", 120) or 120),
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
        request = _object(operation.get("request"))
        kind = str(operation["kind"])
        on_chunk = self._event_callback(fence)
        if kind == "studio_generate":
            config = self._with_credentials(
                _object(request.get("config")),
            )
            document = _object(request.get("document"))
            section = str(request.get("section", ""))
            generated = self.algorithms.generate(
                document,
                section=section,
                config=config,
                analysis_context=_object(
                    request.get("analysisContext")
                ),
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
            return self._chat(
                fence,
                request,
                input_assets=_object(operation.get("inputs")),
                config=_object(request.get("config")),
            )
        if kind == "studio_summary":
            config = self._with_credentials(
                _object(request.get("config")),
            )
            messages = request.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("Studio summary messages are invalid")
            summary = self.algorithms.summarize(
                messages,
                config=config,
                on_chunk=on_chunk,
            )
            return self.repository.publish_summary(
                fence,
                summary=summary,
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
        document = _object(request.get("document"))
        messages = request.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("Studio chat has no messages")
        last = _object(messages[-1])
        raw_user = str(last.get("content", ""))
        runtime_state = deepcopy(_object(request.get("runtimeState")))
        variables = deepcopy(_object(request.get("variables")))
        session_work = {
            "variables": variables,
            "_runtime": runtime_state,
        }
        _, prompt_user, regex_hits = apply_regex_scripts(
            raw_user,
            document.get("regexScripts", []),
            placement=1,
            respect_run_on_edit=True,
        )
        lorebook_hits = sort_lorebook_hits(
            match_lorebook(
                _object(document.get("lorebook")).get("entries", []),
                prompt_user,
                session=session_work,
            )
        )
        runtime_log: list[dict[str, Any]] = list(regex_hits)
        runtime_log.extend(
            {
                "type": "lorebook",
                "id": entry.get("id"),
                "comment": entry.get("comment", ""),
            }
            for entry in lorebook_hits
        )
        runtime_log.extend(
            run_state_tasks(
                session_work,
                document.get("stateTasks", []),
                event="message_received",
            )
        )
        summaries = request.get("summaryBlocks", [])
        system = _build_system_prompt(
            document=document,
            variables=variables,
            summaries=summaries,
            lorebook_hits=lorebook_hits,
        )
        allowed_asset_ids = {
            str(value)
            for key, value in input_assets.items()
            if str(key).startswith("attachment:")
        }
        asset_data_urls = self._asset_data_urls(allowed_asset_ids)
        summarized_through = request.get("summaryThroughMessageId")
        include = summarized_through is None
        conversation: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            item = _object(message)
            if not include:
                if item.get("messageId") == summarized_through:
                    include = True
                continue
            attachment_urls: list[str] = []
            for attachment in item.get("attachments", []) or []:
                asset_id = str(_object(attachment).get("assetId", ""))
                if asset_id and asset_id in allowed_asset_ids:
                    data_url = asset_data_urls.get(asset_id)
                    if data_url is not None:
                        attachment_urls.append(data_url)
            conversation.append(
                {
                    "role": str(item.get("role", "assistant")),
                    "content": (
                        prompt_user
                        if index == len(messages) - 1
                        else str(item.get("content", ""))
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
        visible_assistant, _, output_hits = apply_regex_scripts(
            assistant,
            document.get("regexScripts", []),
            placement=2,
            respect_run_on_edit=True,
        )
        runtime_log.extend(output_hits)
        runtime_log.extend(
            run_state_tasks(
                session_work,
                document.get("stateTasks", []),
                event="message_sent",
            )
        )
        return self.repository.publish_chat(
            fence,
            content=visible_assistant,
            runtime_log=runtime_log,
            variables=_object(session_work.get("variables")),
            runtime_state=_object(session_work.get("_runtime")),
        )

    def prompt_preview(
        self,
        *,
        document: Mapping[str, Any],
        session: Mapping[str, Any],
    ) -> dict[str, Any]:
        messages = session.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], Mapping)
                and messages[index].get("role") == "user"
            ),
            None,
        )
        last_user = (
            str(messages[last_user_index].get("content", ""))
            if last_user_index is not None
            and isinstance(messages[last_user_index], Mapping)
            else ""
        )
        _, prompt_user, _regex_hits = apply_regex_scripts(
            last_user,
            document.get("regexScripts", []),
            placement=1,
            respect_run_on_edit=True,
        )
        work = {
            "variables": deepcopy(_object(session.get("variables"))),
            "_runtime": deepcopy(_object(session.get("runtimeState"))),
        }
        hits = sort_lorebook_hits(
            match_lorebook(
                _object(document.get("lorebook")).get("entries", []),
                prompt_user,
                session=work,
            )
        )
        system = _build_system_prompt(
            document=document,
            variables=_object(work.get("variables")),
            summaries=session.get("summaryBlocks", []),
            lorebook_hits=hits,
        )
        summarized_through = session.get("summaryThroughMessageId")
        include = summarized_through is None
        visible: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            if not include:
                if message.get("messageId") == summarized_through:
                    include = True
                continue
            visible.append(
                {
                    "role": str(message.get("role", "assistant")),
                    "content": (
                        prompt_user
                        if index == last_user_index
                        else str(message.get("content", ""))
                    ),
                    "assetIds": [
                        str(attachment.get("assetId"))
                        for attachment in message.get("attachments", [])
                        if isinstance(attachment, Mapping)
                        and attachment.get("assetId")
                    ],
                }
            )
        return {
            "system": system,
            "messages": visible,
            "lorebookHits": [
                {
                    "id": entry.get("id"),
                    "comment": entry.get("comment", ""),
                }
                for entry in hits
            ],
        }

    def agent_chunks(
        self,
        *,
        document: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        cancelled: threading.Event,
    ) -> Iterator[str]:
        chunks: queue.Queue[object] = queue.Queue(maxsize=128)
        done = object()
        emitted_stream_chunk = threading.Event()

        class AgentDisconnected(RuntimeError):
            pass

        def on_chunk(chunk: str, _full_text: str) -> None:
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
                    messages=messages,
                    system=system,
                    config=self._with_credentials(config),
                    on_chunk=on_chunk,
                )
                # The saved model configuration may intentionally disable
                # streaming. In that mode the transport returns the complete
                # response without invoking ``on_chunk``; the SSE endpoint
                # still has to deliver that response to the browser.
                if result and not emitted_stream_chunk.is_set():
                    publish_control(str(result))
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
                yield str(item)
        finally:
            cancelled.set()

    def _event_callback(
        self,
        fence: OperationFence,
    ) -> Callable[[str, str], None]:
        def emit(chunk: str, full_text: str) -> None:
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
    return {
        "provider": section.get("provider", ""),
        "api_key": section.get("api_key", ""),
        "model": section.get("model_name", ""),
        "base_url": section.get("custom_base_url"),
        "openai_options": _object(section.get("openai_options")),
        "timeout_seconds": section.get("timeout_seconds", 120),
    }


def _build_system_prompt(
    *,
    document: Mapping[str, Any],
    variables: Mapping[str, Any],
    summaries: object,
    lorebook_hits: Sequence[Mapping[str, Any]],
) -> str:
    identity = _object(document.get("identity"))
    core = _object(document.get("coreMessages"))
    lorebook_text = "\n".join(
        str(entry.get("content", "")) for entry in lorebook_hits
    )
    return "\n\n".join(
        value
        for value in (
            str(core.get("system_prompt", "")),
            f"角色：{identity.get('name', document.get('title', ''))}",
            str(identity.get("description", "")),
            str(identity.get("personality", "")),
            str(identity.get("scenario", "")),
            str(core.get("post_history_instructions", "")),
            f"变量：{json.dumps(dict(variables), ensure_ascii=False)}",
            f"会话摘要：{json.dumps(summaries, ensure_ascii=False)}",
            f"世界书：{lorebook_text}",
        )
        if value
    )


def _normalize_review(generated: Mapping[str, Any]) -> dict[str, Any]:
    nested = generated.get("review")
    source = dict(nested) if isinstance(nested, Mapping) else dict(generated)
    summary = str(source.get("summary") or "").strip()
    if not summary:
        raise ValueError("Studio review did not return a summary")

    def string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            rendered
            for item in value
            if (rendered := str(item).strip())
        ]

    return {
        "summary": summary,
        "issues": string_list(source.get("issues")),
        "suggestions": string_list(source.get("suggestions")),
    }


def _validate_generated_payload(
    document: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    section: str,
) -> None:
    if section not in {"full", "translate"}:
        return
    frozen = set(
        _object(document.get("status")).get("frozen_sections", [])
    )
    section_keys = {
        "identity": "identity",
        "greetings": "coreMessages",
        "lorebook": "lorebook",
        "regex": "regexScripts",
        "state-tasks": "stateTasks",
    }
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
    frozen = set(
        _object(result.get("status")).get("frozen_sections", [])
    )
    if section in frozen:
        return result
    if section == "identity":
        identity = _object(generated.get("identity")) or dict(generated)
        result["identity"] = {
            **_object(result.get("identity")),
            **identity,
        }
        name = str(_object(result["identity"]).get("name", "")).strip()
        if name:
            result["title"] = name
            result.setdefault("meta", {})["title"] = name
    elif section == "greetings":
        value = _object(generated.get("coreMessages")) or dict(generated)
        result["coreMessages"] = {
            **_object(result.get("coreMessages")),
            **value,
        }
    elif section == "lorebook":
        result["lorebook"] = (
            _object(generated.get("lorebook")) or dict(generated)
        )
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
        name = str(_object(result.get("identity")).get("name", "")).strip()
        if name and "identity" not in frozen:
            result["title"] = name
            result.setdefault("meta", {})["title"] = name
    else:
        raise ValueError("unsupported Studio generation section")
    return result


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
