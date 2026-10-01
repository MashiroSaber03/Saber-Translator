"""API-executor handlers for durable Character Studio operations."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from contextvars import copy_context
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
from src.shared.user_logging import log_result


_GENERATION_SECTION_LABELS = {
    "identity": "基础身份",
    "greetings": "开场白",
    "lorebook": "世界书",
    "regex": "正则脚本",
    "state-tasks": "状态任务",
    "translate": "文本翻译",
    "full": "完整角色卡",
    "review": "角色卡审阅",
}


class StudioAlgorithms(Protocol):
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
        analysis_context: Mapping[str, Any] | None = None,
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
    ) -> Mapping[str, Any]: ...


class DefaultStudioAlgorithms:
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
        analysis_context: Mapping[str, Any] | None = None,
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
            lorebook_contract = (
                '{"name":"角色世界书","entries":['
                '{"id":"稳定唯一ID","keys":["触发词"],'
                '"secondary_keys":[],"comment":"条目名称",'
                '"content":"原作事实","constant":false,'
                '"selective":false,"enabled":true,'
                '"position":"before_char","priority":100,'
                '"depth":4,"children":[],'
                '"probability":100,"prevent_recursion":true}]}'
            )
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
                "lorebook": '{"lorebook":' + lorebook_contract + '}',
                "regex": '{"regexScripts":[]}',
                "state-tasks": (
                    '{"stateTasks":[{"id":"稳定唯一ID","name":"任务名",'
                    '"triggerTiming":"initialization","interval":1,'
                    '"commands":"/setvar key=score 0","disabled":false}]}'
                ),
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
                    '"lorebook":' + lorebook_contract + ','
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
            lorebook_requirement = (
                "世界书 entries 中的每个条目都必须包含非负整数 depth 和数组 children；"
                "没有子条目时 children 必须返回空数组。"
                "所有条目及子条目都必须遵循同一完整字段结构，条目名称使用 comment，不要使用 title。"
                if section in {"lorebook", "full"}
                else ""
            )
            prompt = (
                f"请{instruction}。目标角色是当前文档的角色名或 source_character，"
                "不要混入其他角色的设定。漫画分析压缩上下文是生成事实依据，"
                "必须实际使用；当前文档已有的非空内容应在不冲突时保留。"
                f"只输出 JSON 对象，顶层结构必须为：{contracts[section]}。"
                f"{full_requirement}"
                f"{lorebook_requirement}"
                "状态任务仅支持每行一条 /setvar key=变量名 值 或 /addvar key=变量名 数字；"
                "变量名使用英文字母、数字、下划线；不要生成 JavaScript 或 STscript 包装。"
                "triggerTiming 使用 initialization、message_received 或 message_sent。"
                "没有需要的任务时 stateTasks 返回空数组。"
                "不要回传数据库元数据、revision、status、meta 或解释文字。\n\n"
                f"漫画分析压缩上下文：\n{context_json}\n\n"
                f"当前角色文档：\n{document_json}"
            )
        def validate_generated(result: object) -> None:
            if not isinstance(result, Mapping):
                raise ValueError("Studio generation did not return a JSON object")
            if section == "review":
                _normalize_review(result)
                return
            _validate_generated_payload(document, result, section=section)
            merged = _apply_generated_section(document, result, section=section)
            validate_current_document(merged, book_id=merged["bookId"], title=merged.get("title"))

        result = self._chat_json(
            prompt,
            config=config,
            validator=validate_generated,
        )
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
        for index, raw in enumerate(messages):
            if not isinstance(raw, Mapping):
                raise ValueError(f"Studio chat message {index} must be an object")
            role = _required_string(raw.get("role"), f"Studio chat message {index} role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Studio chat message {index} role is invalid")
            content = _string(raw.get("content"), f"Studio chat message {index} content")
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
        if system:
            remote_messages.insert(
                0,
                {"role": "system", "content": system},
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
    ) -> Mapping[str, Any]:
        prompt = (
            "总结以下角色对话，保留事实、关系、变量变化和未解决事项。"
            "输出 JSON 对象，至少包含 summary。\n\n"
            + json.dumps(list(messages), ensure_ascii=False)
        )
        result = self._chat_json(
            prompt,
            config=config,
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
        validator: Callable[[object], None] | None = None,
    ) -> object:
        from src.shared.openai_execution import (
            OpenAICompatibleBusinessRetryableError,
            parse_json_block_from_text,
        )

        def parse(text: str) -> object:
            try:
                result = parse_json_block_from_text(text)
                if validator is not None:
                    validator(result)
                return result
            except (ValueError, TypeError) as exc:
                raise OpenAICompatibleBusinessRetryableError(str(exc)) from exc

        return self._complete(
            [{"role": "user", "content": prompt}],
            config=config,
            temperature=0.3,
            force_json=True,
            on_chunk=None,
            parser=parse,
        )

    @staticmethod
    def _complete(
        messages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
        temperature: float,
        force_json: bool,
        on_chunk: Callable[[str, str], None] | None,
        prefer_vlm: bool = False,
        parser: Callable[[str], object] | None = None,
    ) -> Any:
        from src.shared.ai_transport import UnifiedChatRequest
        from src.shared.openai_execution import (
            OpenAICompatibleSyncExecutor,
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
                on_stream_chunk=on_chunk,
            ),
        )
        result = OpenAICompatibleSyncExecutor().execute(
            request,
            capability=request.capability,
            parser=parser,
        )
        return result.parsed


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
            # JSON 结果校验完成后整体发布；逐字事件只用于聊天文本。
            generated = self.algorithms.generate(
                document,
                section=section,
                config=config,
                analysis_context=analysis_context,
            )
            if section == "review":
                review = _normalize_review(generated)
                published = self.repository.publish_generate(
                    fence,
                    generated_document=document,
                    review=review,
                )
                review_details = [review["summary"]]
                review_details.extend(
                    f"问题：{value}" for value in review["issues"]
                )
                review_details.extend(
                    f"建议：{value}" for value in review["suggestions"]
                )
                log_result("角色卡审阅完成", details=review_details)
                return published
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
            published = self.repository.publish_generate(
                fence,
                generated_document=merged,
            )
            section_label = _GENERATION_SECTION_LABELS.get(section, section)
            log_result(
                f"角色设定「{section_label}」已保存｜生成 {len(generated)} 个字段",
                details=("字段：" + "、".join(sorted(generated)),),
            )
            return published
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
                } | ({"rewriteMessageId"} if "rewriteMessageId" in request else set()),
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
            )
            if not isinstance(summary, Mapping):
                raise ValueError("Studio summary did not return a JSON object")
            if set(summary) != {"summary"}:
                raise ValueError("Studio summary fields are invalid")
            summary_text = summary.get("summary")
            if not isinstance(summary_text, str) or not summary_text.strip():
                raise ValueError("Studio summary did not return summary text")
            published = self.repository.publish_summary(
                fence,
                summary={"summary": summary_text.strip()},
            )
            log_result(
                f"对话摘要已保存｜{len(summary_text.strip())} 个字符",
                details=(summary_text.strip(),),
            )
            return published
        raise ValueError(f"unsupported Studio operation: {kind}")

    def _chat(
        self,
        fence: OperationFence,
        request: Mapping[str, Any],
        *,
        input_assets: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        messages = _operation_messages(
            request.get("messages"), label="Studio chat messages", require_nonempty=True,
        )
        if messages[-1]["role"] != "user":
            raise ValueError("Studio chat must end with a user message")
        system, conversation, _, session_work, runtime_log = _prepare_chat_prompt(request)
        allowed_asset_ids: set[str] = set()
        for role, value in input_assets.items():
            if not isinstance(role, str) or not role.startswith("attachment:"):
                raise ValueError("Studio operation input role is invalid")
            allowed_asset_ids.add(
                _required_string(value, "Studio operation input asset id")
            )
        asset_data_urls = self._asset_data_urls(allowed_asset_ids)
        remote_messages: list[dict[str, Any]] = []
        for message in conversation:
            attachment_urls = []
            for asset_id in message["assetIds"]:
                if asset_id not in allowed_asset_ids:
                    raise ValueError("Studio chat attachment is not bound to the operation")
                if asset_id not in asset_data_urls:
                    raise ValueError("Studio chat attachment is unavailable")
                attachment_urls.append(asset_data_urls[asset_id])
            remote_messages.append({
                "role": message["role"],
                "content": message["content"],
                "attachmentDataUrls": attachment_urls,
            })
        self.repository.operations.append_event(
            fence,
            event_type="prompt_ready",
            payload={
                "messageCount": len(remote_messages),
                "attachmentCount": sum(
                    len(item["attachmentDataUrls"]) for item in remote_messages
                ),
            },
        )
        assistant = self.algorithms.chat(
            messages=remote_messages,
            system=system,
            config=self._with_credentials(
                config,
                prefer_vlm=any(item["attachmentDataUrls"] for item in remote_messages),
            ),
            on_chunk=self._event_callback(fence),
        )
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError("Studio chat did not return response text")
        document = _current_document(request.get("document"))
        visible_assistant, _, output_hits = apply_regex_scripts(
            assistant, document["regexScripts"], placement=2, respect_run_on_edit=True,
        )
        runtime_log.extend(output_hits)
        runtime_log.extend(run_state_tasks(
            session_work, document["stateTasks"], event="message_sent",
        ))
        published = self.repository.publish_chat(
            fence,
            content=visible_assistant,
            runtime_log=runtime_log,
            variables=session_work["variables"],
            runtime_state=session_work["_runtime"],
        )
        log_result(
            f"角色回复已保存｜{len(visible_assistant)} 个字符",
            details=(visible_assistant,),
        )
        return published

    def prompt_preview(
        self,
        *,
        document: Mapping[str, Any],
        session: Mapping[str, Any],
    ) -> dict[str, Any]:
        # 已生成的回复使用原请求快照；未聊天的会话显示当前配置。
        request = self.repository.get_chat_prompt_request(str(session["sessionId"]))
        has_request = request is not None
        if request is None:
            request = {"document": document, **session}
        system, messages, hits, _, _ = _prepare_chat_prompt(request, apply_user_input=has_request)
        return {
            "source": "request" if has_request else "current_config",
            "system": system,
            "messages": messages,
            "lorebookHits": [
                {"id": entry["id"], "comment": entry["comment"]} for entry in hits
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

        producer_context = copy_context()
        thread = threading.Thread(
            target=lambda: producer_context.run(run),
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
    base_url = section.get("custom_base_url")
    if base_url == "":
        base_url = None
    return {
        "provider": section.get("provider", ""),
        "api_key": section.get("api_key", ""),
        "credential_version_id": section.get("credential_version_id"),
        "model": section.get("model_name", ""),
        "base_url": base_url,
        "openai_options": options,
    }


def _prepare_chat_prompt(
    request: Mapping[str, Any],
    *,
    apply_user_input: bool = True,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build model and preview context from the same immutable input."""
    document = _current_document(request.get("document"))
    messages = _operation_messages(request.get("messages"), label="Studio chat messages")
    work = {
        "variables": deepcopy(_required_mapping(request.get("variables"), "Studio chat variables")),
        "_runtime": deepcopy(_required_mapping(request.get("runtimeState"), "Studio chat runtimeState")),
    }
    work["_runtime"].pop("matched_lorebook_ids", None)
    last_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index]["role"] == "user"),
        None,
    ) if apply_user_input else None
    raw_user = messages[last_user_index]["content"] if last_user_index is not None else ""
    _, prompt_user, regex_hits = apply_regex_scripts(
        raw_user, document["regexScripts"], placement=1, respect_run_on_edit=True,
    )
    hits = sort_lorebook_hits(match_lorebook(document["lorebook"]["entries"], prompt_user))
    logs = list(regex_hits)
    logs.extend(
        {"type": "lorebook", "id": entry["id"], "comment": entry["comment"]} for entry in hits
    )
    if last_user_index is not None:
        logs.extend(run_state_tasks(work, document["stateTasks"], event="message_received"))
    identity = document["identity"]
    core = document["coreMessages"]
    system = "\n\n".join(value for value in (
        core["system_prompt"],
        "\n".join(entry["content"] for entry in hits if entry["position"] == "before_char"),
        f"角色：{identity['name']}",
        identity["description"],
        identity["personality"],
        identity["scenario"],
        "\n".join(entry["content"] for entry in hits if entry["position"] == "after_char"),
        f"示例对话：\n{core['message_example']}" if core["message_example"] else "",
        f"变量：{json.dumps(work['variables'], ensure_ascii=False)}",
        f"会话摘要：{json.dumps(_summary_blocks(request.get('summaryBlocks')), ensure_ascii=False)}",
    ) if value)
    through = request.get("summaryThroughMessageId")
    if through is not None and (not isinstance(through, str) or not any(
        item["messageId"] == through for item in messages
    )):
        raise ValueError("Studio chat summaryThroughMessageId does not identify a message")
    include = through is None
    history: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not include:
            if message["messageId"] == through:
                include = True
            continue
        history.append({
            "role": message["role"],
            "content": prompt_user if index == last_user_index else message["content"],
            "assetIds": [attachment["assetId"] for attachment in message["attachments"]],
        })
    # 深度按保留的真实聊天消息计数，不把其他注入条目算进深度。
    injections: dict[int, list[dict[str, Any]]] = {}
    for entry in hits:
        if entry["position"] == "at_depth":
            position = max(0, len(history) - entry["depth"])
            if history and position == len(history) and history[-1]["role"] == "user":
                history[-1]["content"] += f"\n\n[补充上下文]\n{entry['content']}"
                continue
            injections.setdefault(position, []).append({
                "role": "system", "content": entry["content"], "assetIds": [],
            })
    # 生成请求以用户消息结束；回复要求在该消息的正文之后，不另造末尾 system 消息。
    if core["post_history_instructions"] and history and history[-1]["role"] == "user":
        history[-1]["content"] += f"\n\n[回复要求]\n{core['post_history_instructions']}"
    conversation: list[dict[str, Any]] = []
    for index in range(len(history) + 1):
        conversation.extend(injections.get(index, []))
        if index < len(history):
            conversation.append(history[index])
    if core["post_history_instructions"] and (not history or history[-1]["role"] != "user"):
        conversation.append({
            "role": "system", "content": core["post_history_instructions"], "assetIds": [],
        })
    return system, conversation, hits, work, logs


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
