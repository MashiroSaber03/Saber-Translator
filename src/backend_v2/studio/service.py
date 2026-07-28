"""API-executor handlers for durable Character Studio operations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any, Protocol

from sqlalchemy import Engine, select

from src.backend_v2.operations.repository import (
    OperationFence,
    OperationFenced,
)
from src.backend_v2.storage.schema import credential_versions
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
    ) -> Mapping[str, Any]: ...

    def chat(
        self,
        *,
        prompt: str,
        system: str,
        config: Mapping[str, Any],
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
    ) -> Mapping[str, Any]:
        prompt = (
            f"请为 Character Studio 文档生成 {section} 区段。"
            "只输出 JSON；保留未要求修改的字段。\n\n"
            + json.dumps(document, ensure_ascii=False)
        )
        result = self._chat_json(prompt, config=config)
        if not isinstance(result, Mapping):
            raise ValueError("Studio generation did not return a JSON object")
        return dict(result)

    def chat(
        self,
        *,
        prompt: str,
        system: str,
        config: Mapping[str, Any],
    ) -> str:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        section = _provider_config(config)
        client = ChatClient(ChatLLMConfig.from_dict(section))

        async def execute() -> str:
            try:
                return await client.generate(
                    prompt,
                    system=system or None,
                    temperature=0.7,
                )
            finally:
                await client.close()

        return asyncio.run(execute())

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
        result = self._chat_json(prompt, config=config)
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
    ) -> object:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        client = ChatClient(
            ChatLLMConfig.from_dict(_provider_config(config))
        )

        async def execute() -> object:
            try:
                return await client.generate_json(
                    prompt,
                    temperature=0.3,
                )
            finally:
                await client.close()

        return asyncio.run(execute())


class StudioOperationService:
    def __init__(
        self,
        *,
        engine: Engine,
        repository: StudioRepository | None = None,
        algorithms: StudioAlgorithms | None = None,
    ) -> None:
        self.engine = engine
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
        if kind == "studio_generate":
            document = _object(request.get("document"))
            section = str(request.get("section", ""))
            generated = self.algorithms.generate(
                document,
                section=section,
                config=config,
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
            return self._chat(fence, request, config=config)
        if kind == "studio_summary":
            messages = request.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("Studio summary messages are invalid")
            summary = self.algorithms.summarize(
                messages,
                config=config,
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
        identity = _object(document.get("identity"))
        core = _object(document.get("coreMessages"))
        summaries = request.get("summaryBlocks", [])
        lorebook_text = "\n".join(
            str(entry.get("content", "")) for entry in lorebook_hits
        )
        system = "\n\n".join(
            value
            for value in (
                str(core.get("system_prompt", "")),
                f"角色：{identity.get('name', document.get('title', ''))}",
                str(identity.get("description", "")),
                str(identity.get("personality", "")),
                str(identity.get("scenario", "")),
                f"变量：{json.dumps(variables, ensure_ascii=False)}",
                f"会话摘要：{json.dumps(summaries, ensure_ascii=False)}",
                f"世界书：{lorebook_text}",
            )
            if value
        )
        history = "\n".join(
            f"{message.get('role')}: "
            f"{visible_user if index == len(messages) - 1 else message.get('content', '')}"
            for index, message in enumerate(messages)
        )
        assistant = self.algorithms.chat(
            prompt=history,
            system=system,
            config=config,
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
    }


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
        for key in (
            "identity",
            "coreMessages",
            "lorebook",
            "regexScripts",
            "stateTasks",
        ):
            if key in generated and key not in frozen:
                result[key] = deepcopy(generated[key])
        name = str(_object(result.get("identity")).get("name", "")).strip()
        if name:
            result["title"] = name
            result.setdefault("meta", {})["title"] = name
    else:
        raise ValueError("unsupported Studio generation section")
    return result


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
