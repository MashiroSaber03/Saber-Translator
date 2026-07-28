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
from src.backend_v2.storage.schema import assets, credential_versions
from src.backend_v2.studio.repository import StudioRepository
from src.backend_v2.studio.pure import (
    apply_regex_scripts,
    match_lorebook,
    run_state_tasks,
    sort_lorebook_hits,
)


class StudioAlgorithms(Protocol):
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
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
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> Mapping[str, Any]:
        prompt = (
            f"请为 Character Studio 文档生成 {section} 区段。"
            "只输出 JSON；保留未要求修改的字段。\n\n"
            + json.dumps(document, ensure_ascii=False)
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
        if system:
            remote_messages.append({"role": "system", "content": system})
        for raw in messages:
            role = str(raw.get("role", "assistant"))
            content = str(raw.get("content", ""))
            attachments = raw.get("attachmentDataUrls", [])
            if role == "user" and isinstance(attachments, list) and attachments:
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
    ) -> str:
        from src.shared.ai_transport import (
            OpenAICompatibleChatTransport,
            UnifiedChatRequest,
        )
        from src.shared.openai_execution import (
            build_openai_compatible_runtime_options,
        )
        from src.shared.openai_options import OpenAICompatibleOptions

        section = _provider_config(config)
        provider = str(section.get("provider", ""))
        model = str(section.get("model", ""))
        if not provider or not model:
            raise ValueError("Studio chat provider/model is not configured")
        options = OpenAICompatibleOptions.from_dict(
            _object(section.get("openai_options"))
        )
        options.execution.use_stream = on_chunk is not None
        options.request.force_json_output = force_json
        if options.request.temperature is None:
            options.request.temperature = temperature
        request = UnifiedChatRequest(
            provider=provider,
            api_key=str(section.get("api_key", "")),
            model=model,
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
        self.algorithms = algorithms or DefaultStudioAlgorithms()

    def handle(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request = _object(operation.get("request"))
        config = self._with_credentials(_object(request.get("config")))
        kind = str(operation["kind"])
        on_chunk = self._event_callback(fence)
        if kind == "studio_generate":
            document = _object(request.get("document"))
            section = str(request.get("section", ""))
            generated = self.algorithms.generate(
                document,
                section=section,
                config=config,
                on_chunk=on_chunk,
            )
            if section == "review":
                return self.repository.publish_generate(
                    fence,
                    generated_document=document,
                    review=generated,
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
                config=config,
            )
        if kind == "studio_summary":
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
        visible_user, prompt_user, regex_hits = apply_regex_scripts(
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
        if not allowed_asset_ids:
            allowed_asset_ids = set()
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
                    data_url = self._asset_data_url(asset_id)
                    if data_url is not None:
                        attachment_urls.append(data_url)
            conversation.append(
                {
                    "role": str(item.get("role", "assistant")),
                    "content": (
                        visible_user
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
            config=config,
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
        last_user = next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if isinstance(message, Mapping)
                and message.get("role") == "user"
            ),
            "",
        )
        work = {
            "variables": deepcopy(_object(session.get("variables"))),
            "_runtime": deepcopy(_object(session.get("runtimeState"))),
        }
        hits = sort_lorebook_hits(
            match_lorebook(
                _object(document.get("lorebook")).get("entries", []),
                last_user,
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
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if not include:
                if message.get("messageId") == summarized_through:
                    include = True
                continue
            visible.append(
                {
                    "role": str(message.get("role", "assistant")),
                    "content": str(message.get("content", "")),
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

    def resolve_runtime_config(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._with_credentials(config)

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

        class AgentDisconnected(RuntimeError):
            pass

        def on_chunk(chunk: str, _full_text: str) -> None:
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
                    "需要结构化修改时输出 ```json:patch 代码块，操作仅可使用 "
                    "add/remove/replace/move/copy/test；需要视觉预览时可输出 "
                    "```html 代码块。不要声称已直接保存文档。\n\n当前文档：\n"
                    + json.dumps(document, ensure_ascii=False)
                )
                self.algorithms.chat(
                    messages=messages,
                    system=system,
                    config=self._with_credentials(config),
                    on_chunk=on_chunk,
                )
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

    def _asset_data_url(self, asset_id: str) -> str | None:
        if self.storage is None:
            return None
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    assets.c.relative_path,
                    assets.c.mime_type,
                    assets.c.integrity_status,
                ).where(assets.c.id == asset_id)
            ).mappings().one_or_none()
        if row is None or row["integrity_status"] != "ok":
            return None
        path = self.storage.resolve_relative_path(str(row["relative_path"]))
        if not path.is_file():
            return None
        return (
            f"data:{row['mime_type']};base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii")
        )

    def _with_credentials(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        for key in ("chat", "vlm"):
            section = _object(result.get(key))
            credential_id = section.pop("credentialVersionId", None)
            if credential_id:
                with self.engine.connect() as connection:
                    value = connection.execute(
                        select(credential_versions.c.secret_json).where(
                            credential_versions.c.id == credential_id
                        )
                    ).scalar_one_or_none()
                if value is None:
                    raise OperationFenced(
                        "Studio credential version no longer exists"
                    )
                secret = json.loads(value)
                if isinstance(secret, Mapping):
                    section.update(secret)
            result[key] = section
        return result


def _provider_config(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _object(config.get("chat"))
    if not section.get("provider"):
        section = _object(config.get("vlm"))
    return {
        "provider": section.get("provider", ""),
        "api_key": section.get("api_key", section.get("apiKey", "")),
        "model": section.get(
            "model_name",
            section.get("modelName", ""),
        ),
        "base_url": section.get(
            "custom_base_url",
            section.get("base_url"),
        ),
        "openai_options": _object(section.get("openai_options")),
        "timeout_seconds": section.get(
            "timeout_seconds",
            section.get("timeoutSeconds", 120),
        ),
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
        result["regexScripts"] = list(
            generated.get(
                "regexScripts",
                generated.get("regex_scripts", []),
            )
        )
    elif section == "state-tasks":
        result["stateTasks"] = list(
            generated.get("stateTasks", generated.get("state_tasks", []))
        )
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
