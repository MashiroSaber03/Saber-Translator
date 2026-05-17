"""
Core Character Studio services.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import base64
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.manga_insight.config_utils import create_chat_client, has_provider_model_config, load_insight_config
from src.core.manga_insight.embedding_client import ChatClient
from src.core.manga_insight.storage import AnalysisStorage
from src.core.manga_insight.vlm_client import VLMClient
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
    parse_json_block_from_text,
)

from .adapters import (
    build_export_bundle,
    create_empty_document,
    ensure_document_shape,
    export_png_bytes,
    import_document_payload,
)
from .agent import build_agent_context
from .runtime import (
    apply_regex_scripts,
    initialize_runtime_session,
    match_lorebook,
    run_state_tasks,
    sort_lorebook_hits,
)
from .store import CharacterStudioStore
from .validators import build_diagnostics_report

logger = logging.getLogger("MangaInsight.CharacterStudio")


SECTION_PROMPTS = {
    "identity": """你是角色卡编辑助手。请基于压缩摘要与目标角色名，为该角色补全角色设定。输出 JSON 对象，字段 only: name, description, personality, scenario。所有字段必须是字符串。""",
    "greetings": """你是角色卡问候语设计助手。请基于压缩摘要与目标角色名，为该角色补全问候语和对话元信息。输出 JSON 对象，字段 only: first_message, message_example, alternate_greetings, system_prompt, post_history_instructions, creator_notes, character_version。alternate_greetings 必须为字符串数组。""",
    "lorebook": """你是角色世界书设计助手。请基于压缩摘要与目标角色名，为该角色构建世界书。输出 JSON 对象，字段 only: lorebook，其中 lorebook 必须包含 name, entries。entries 为数组，每项必须使用以下精确字段：comment, keys, secondary_keys, content, enabled, constant, selective, priority, position, depth, children。keys 必须是非空字符串数组。不要输出 name、title、key 这类替代字段；如果你拿不准，就返回空 entries 数组。""",
    "regex": """你是角色脚本助手。请基于压缩摘要与目标角色名，为该角色生成运行时正则脚本。输出 JSON 对象，字段 only: regexScripts。regexScripts 为数组，每项必须使用以下精确字段：scriptName, findRegex, replaceString, placement, markdownOnly, promptOnly, runOnEdit, disabled。placement 只能是数字数组，推荐使用 [2]；1 表示用户输入/提示文本侧，2 表示角色回复/显示文本侧。不要输出 name、regex、replacement、condition 这类替代字段；当前不支持 condition 条件表达式。如果拿不准，就返回空数组。""",
    "state-tasks": """你是状态任务助手。请基于压缩摘要与目标角色名，为该角色生成状态任务。输出 JSON 对象，字段 only: stateTasks。stateTasks 为数组，每项必须使用以下精确字段：name, triggerTiming, interval, commands, disabled。triggerTiming 只能使用 initialization、message_received、message_sent 三个值。commands 必须是非空字符串，格式推荐为 <<taskjs>> ... <</taskjs>>；如果拿不准如何写可执行 commands，就返回空数组。不要输出 description、trigger、action 这类替代字段。""",
    "review": """你是角色卡审查员。请基于压缩摘要与目标角色名，审查当前角色卡是否忠于原始剧情信息。输出 JSON 对象，字段 only: summary, issues, suggestions。summary 必须为字符串；issues 与 suggestions 必须为字符串数组。""",
    "translate": """你是专业翻译与角色卡整理助手。请参考压缩摘要与当前角色卡草稿，将角色卡正文翻译或整理为更自然的中文。输出 JSON 对象，字段 only: identity, coreMessages。identity 必须包含完整字段：name, description, personality, scenario。coreMessages 必须包含完整字段：first_message, message_example, alternate_greetings, system_prompt, post_history_instructions, creator_notes, character_version。不要省略字段，不要返回 null。""",
    "full": """你是角色卡构建助手。请基于压缩摘要与目标角色名，一次性生成完整角色卡初稿。输出 JSON 对象，字段 only: identity, coreMessages, lorebook, regexScripts, stateTasks。

identity 必须包含：name, description, personality, scenario。
coreMessages 必须包含：first_message, message_example, alternate_greetings, system_prompt, post_history_instructions, creator_notes, character_version。
lorebook 必须包含：name, entries。entries 中每项必须使用精确字段：comment, keys, secondary_keys, content, enabled, constant, selective, priority, position, depth, children。keys 必须是非空字符串数组。不要用 name、title、key 代替。
regexScripts 必须是数组。每项必须使用精确字段：scriptName, findRegex, replaceString, placement, markdownOnly, promptOnly, runOnEdit, disabled。placement 推荐使用 [2]；1 表示用户输入/提示文本侧，2 表示角色回复/显示文本侧。不要用 name、regex、replacement、condition 代替；当前不支持 condition。如果拿不准，就返回空数组。
stateTasks 必须是数组。每项必须使用精确字段：name, triggerTiming, interval, commands, disabled。triggerTiming 只能使用 initialization、message_received、message_sent。commands 必须是非空字符串，推荐使用 <<taskjs>> ... <</taskjs>>。不要用 description、trigger、action 代替。如果拿不准，就返回空数组。

请直接输出 JSON，不要输出解释、Markdown 或代码块外文字。""",
}

FACT_DRIVEN_SECTIONS = {"identity", "greetings", "lorebook", "regex", "state-tasks", "translate", "full"}


class CharacterStudioService:
    def __init__(self, book_id: str, config=None):
        self.book_id = book_id
        self.storage = AnalysisStorage(book_id)
        self.store = CharacterStudioStore(book_id)
        self.config = config or load_insight_config()
        self._dialogue_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def _create_chat_client(self) -> Optional[ChatClient]:
        try:
            llm = create_chat_client(self.config)
            if llm and has_provider_model_config(llm.config.provider, llm.config.model, llm.config.api_key):
                return llm
        except Exception as exc:
            logger.warning("初始化 Character Studio ChatClient 失败: %s", exc)
        return None

    def _create_vlm_client(self) -> Optional[VLMClient]:
        try:
            vlm = VLMClient(self.config.vlm, getattr(self.config, "prompts", None))
            if has_provider_model_config(vlm.config.provider, vlm.config.model, vlm.config.api_key):
                return vlm
        except Exception as exc:
            logger.warning("初始化 Character Studio VLMClient 失败: %s", exc)
        return None

    @staticmethod
    def _normalize_name(value: str) -> str:
        import re

        value = (value or "").strip().lower()
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[·•\\-_\\.,!?:;()\\[\\]\"'“”‘’]", "", value)
        return value

    @staticmethod
    def _extract_dialogues_from_analysis(analysis: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for dlg in analysis.get("dialogues", []) or []:
            speaker = dlg.get("speaker_name") or dlg.get("character") or dlg.get("speaker") or ""
            text = dlg.get("text") or dlg.get("translated_text") or ""
            if text:
                items.append({"speaker": speaker, "text": text, "page": page_num})
        for panel in analysis.get("panels", []) or []:
            for dlg in panel.get("dialogues", []) or []:
                speaker = dlg.get("speaker_name") or dlg.get("character") or dlg.get("speaker") or ""
                text = dlg.get("text") or dlg.get("translated_text") or ""
                if text:
                    items.append({"speaker": speaker, "text": text, "page": page_num})
        return items

    async def _build_dialogue_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._dialogue_cache is not None:
            return self._dialogue_cache
        cache: Dict[str, List[Dict[str, Any]]] = {}
        page_nums = await self.storage.list_pages()
        for page_num in page_nums:
            analysis = await self.storage.load_page_analysis(page_num)
            if not analysis:
                continue
            dialogs = self._extract_dialogues_from_analysis(analysis, page_num)
            for item in dialogs:
                speaker = (item.get("speaker") or "").strip()
                if not speaker:
                    continue
                key = self._normalize_name(speaker)
                cache.setdefault(key, []).append(item)
        self._dialogue_cache = cache
        return cache

    async def _collect_dialogues_for_character(self, names: List[str]) -> List[Dict[str, Any]]:
        cache = await self._build_dialogue_cache()
        normalized_keys = {self._normalize_name(name) for name in names if name}
        result: List[Dict[str, Any]] = []
        for key, lines in cache.items():
            if any(key == target or key in target or target in key for target in normalized_keys if target):
                result.extend(lines)
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in result:
            token = (item.get("page"), item.get("speaker"), item.get("text"))
            if token in seen:
                continue
            seen.add(token)
            deduped.append(item)
        return deduped

    async def get_candidates(self) -> Dict[str, Any]:
        timeline = await self.storage.load_timeline()
        if not timeline:
            raise ValueError("未找到时间线缓存，请先生成增强时间线。")
        characters = timeline.get("characters", [])
        if not isinstance(characters, list) or len(characters) == 0:
            raise ValueError("时间线中未包含角色数据，请先生成增强时间线。")

        candidates: List[Dict[str, Any]] = []
        for character in characters:
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            aliases = [a for a in character.get("aliases", []) if isinstance(a, str) and a.strip()]
            dialogues = await self._collect_dialogues_for_character([name] + aliases)
            sample_pages = sorted({int(item.get("page", 0)) for item in dialogues if int(item.get("page", 0)) > 0})[:8]
            candidates.append({
                "name": name,
                "aliases": aliases,
                "first_appearance": int(character.get("first_appearance", 0) or 0),
                "dialogue_count": len(dialogues),
                "has_dialogues": len(dialogues) > 0,
                "sample_pages": sample_pages,
            })
        return {
            "book_id": self.book_id,
            "candidates": candidates,
            "count": len(candidates),
            "generated_at": datetime.now().isoformat(),
        }

    async def get_index_payload(self) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        has_timeline = True
        try:
            candidate_payload = await self.get_candidates()
            candidates = candidate_payload.get("candidates", [])
        except ValueError:
            has_timeline = False
        index_data = await self.store.load_index()
        return {
            "book_id": self.book_id,
            "documents": index_data.get("documents", []),
            "candidates": candidates,
            "count": len(index_data.get("documents", [])),
            "has_timeline": has_timeline,
        }

    async def create_document(self, *, candidate_name: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
        if candidate_name:
            document = await self._build_document_from_candidate(candidate_name)
        else:
            document = create_empty_document(self.book_id, title=title or "新角色", origin_type="manual")
        await self.store.save_document(document)
        return document

    async def _build_document_from_candidate(self, candidate_name: str) -> Dict[str, Any]:
        timeline = await self.storage.load_timeline() or {}
        characters = timeline.get("characters", []) or []
        if not any(item.get("name") == candidate_name for item in characters):
            raise ValueError(f"未找到候选角色: {candidate_name}")
        document = create_empty_document(self.book_id, title=candidate_name, origin_type="analysis")
        document["origin"]["source_character"] = candidate_name
        document["identity"]["name"] = candidate_name
        document["meta"]["title"] = candidate_name
        return ensure_document_shape(document, book_id=self.book_id)

    async def save_document(self, doc_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = await self.store.load_document(doc_id)
        if not current:
            raise ValueError("文档不存在")
        merged = copy.deepcopy(current)
        for key, value in payload.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
        merged["id"] = doc_id
        merged["bookId"] = self.book_id
        merged = ensure_document_shape(merged, book_id=self.book_id)
        await self.store.save_document(merged)
        return merged

    async def generate_section(self, doc_id: str, section: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        if section not in SECTION_PROMPTS:
            raise ValueError(f"不支持的 section: {section}")
        frozen_sections = set(document.get("status", {}).get("frozen_sections", []) or [])
        if section in frozen_sections:
            return document
        if section == "review":
            generated = await self._generate_review_payload(document)
            updated = self._apply_section_payload(document, section, generated)
            await self.store.save_document(updated)
            return updated

        if section in FACT_DRIVEN_SECTIONS:
            updated = await self._generate_fact_driven_section(document, section, frozen_sections=frozen_sections)
            await self.store.save_document(updated)
            return updated

        raise ValueError(f"不支持的 section: {section}")

    async def _generate_review_payload(self, document: Dict[str, Any]) -> Dict[str, Any]:
        role_name = str(document.get("identity", {}).get("name", "") or document.get("meta", {}).get("title", "")).strip()
        if not role_name:
            raise ValueError("请先填写角色名，再使用 AI 审查。")

        compressed_context = await self._ensure_compressed_context()
        prompt = self._build_review_prompt(document, compressed_context)
        client = self._create_chat_client()
        if not client:
            raise ValueError("未配置可用的 AI 对话模型，无法审查角色卡。")
        try:
            generated = await self._generate_structured_payload(
                client=client,
                prompt=prompt,
                temperature=0.4,
                parser=self._parse_review_payload,
            )
        except Exception as exc:
            logger.error("生成审查报告失败: %s", exc, exc_info=True)
            raise ValueError(f"AI 审查失败：{exc}") from exc
        finally:
            try:
                await client.close()
            except Exception:
                pass

        return generated

    async def _generate_fact_driven_section(
        self,
        document: Dict[str, Any],
        section: str,
        *,
        frozen_sections: set[str],
    ) -> Dict[str, Any]:
        role_name = str(document.get("identity", {}).get("name", "") or document.get("meta", {}).get("title", "")).strip()
        if not role_name:
            raise ValueError("请先填写角色名，再使用 AI 补全。")

        compressed_context = await self._ensure_compressed_context()
        prompt = self._build_fact_generation_prompt(document, section, compressed_context)
        client = self._create_chat_client()
        if not client:
            raise ValueError("未配置可用的 AI 对话模型，无法补全角色卡。")

        try:
            generated = await self._generate_structured_payload(
                client=client,
                prompt=prompt,
                temperature=0.45,
                parser=lambda content: self._parse_fact_driven_payload(
                    content,
                    document=document,
                    section=section,
                    frozen_sections=frozen_sections,
                ),
            )
        except Exception as exc:
            logger.error("生成 section %s 失败: %s", section, exc, exc_info=True)
            raise ValueError(f"AI 生成失败：{exc}") from exc
        finally:
            try:
                await client.close()
            except Exception:
                pass

        updated = self._apply_section_payload(document, section, generated, frozen_sections=frozen_sections)
        return updated

    async def _ensure_compressed_context(self) -> str:
        compressed = await self.storage.load_compressed_context()
        context = str((compressed or {}).get("context", "") or "").strip()
        if context:
            return context

        from src.core.manga_insight.features.hierarchical_summary import HierarchicalSummaryGenerator

        summary_client = self._create_chat_client()
        try:
            generator = HierarchicalSummaryGenerator(
                book_id=self.book_id,
                storage=self.storage,
                llm_client=summary_client,
                prompts_config=getattr(self.config, "prompts", None),
            )
            await generator.generate_hierarchical_overview()
        finally:
            if summary_client:
                try:
                    await summary_client.close()
                except Exception:
                    pass

        compressed = await self.storage.load_compressed_context()
        context = str((compressed or {}).get("context", "") or "").strip()
        if context:
            return context
        raise ValueError("未找到可用于角色工坊补全的压缩摘要，请先完成漫画分析或生成概览。")

    def _build_fact_generation_prompt(self, document: Dict[str, Any], section: str, compressed_context: str) -> str:
        snapshot = {
            "identity": document.get("identity", {}),
            "coreMessages": document.get("coreMessages", {}),
            "lorebook": document.get("lorebook", {}),
            "regexScripts": document.get("regexScripts", []),
            "stateTasks": document.get("stateTasks", []),
            "frozen_sections": document.get("status", {}).get("frozen_sections", []),
        }
        role_name = document.get("identity", {}).get("name", "") or document.get("meta", {}).get("title", "")
        return (
            SECTION_PROMPTS[section]
            + "\n\n目标角色名:\n"
            + str(role_name)
            + "\n\n压缩摘要（唯一外部事实资料库）:\n"
            + compressed_context
            + "\n\n当前角色卡草稿（仅作风格、一致性与保留内容参考，不可当作新的事实来源）:\n"
            + json.dumps(snapshot, ensure_ascii=False, indent=2)
        )

    def _build_review_prompt(self, document: Dict[str, Any], compressed_context: str) -> str:
        role_name = document.get("identity", {}).get("name", "") or document.get("meta", {}).get("title", "")
        review_target = {
            "identity": document.get("identity", {}),
            "coreMessages": document.get("coreMessages", {}),
            "lorebook": document.get("lorebook", {}),
            "regexScripts": document.get("regexScripts", []),
            "stateTasks": document.get("stateTasks", []),
        }
        return (
            SECTION_PROMPTS["review"]
            + "\n\n目标角色名:\n"
            + str(role_name)
            + "\n\n压缩摘要（唯一外部事实资料库）:\n"
            + compressed_context
            + "\n\n当前角色卡内容（审查对象）:\n"
            + json.dumps(review_target, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _require_string(payload: Dict[str, Any], key: str, *, allow_empty: bool = False) -> None:
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"AI 生成结果缺少字符串字段: {key}")
        if not allow_empty and not value.strip():
            raise ValueError(f"AI 生成结果字段为空: {key}")

    @staticmethod
    def _require_list(payload: Dict[str, Any], key: str) -> List[Any]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"AI 生成结果缺少数组字段: {key}")
        return value

    def _validate_identity_payload(self, payload: Dict[str, Any]) -> None:
        self._require_string(payload, "name")
        self._require_string(payload, "description")
        self._require_string(payload, "personality")
        self._require_string(payload, "scenario")

    def _validate_core_messages_payload(self, payload: Dict[str, Any]) -> None:
        self._require_string(payload, "first_message")
        self._require_string(payload, "message_example")
        alternates = self._require_list(payload, "alternate_greetings")
        if not all(isinstance(item, str) for item in alternates):
            raise ValueError("AI 生成结果字段类型错误: alternate_greetings")
        self._require_string(payload, "system_prompt", allow_empty=True)
        self._require_string(payload, "post_history_instructions", allow_empty=True)
        self._require_string(payload, "creator_notes", allow_empty=True)
        self._require_string(payload, "character_version", allow_empty=True)

    def _validate_review_payload(self, payload: Dict[str, Any]) -> None:
        self._require_string(payload, "summary", allow_empty=False)
        issues = self._require_list(payload, "issues")
        suggestions = self._require_list(payload, "suggestions")
        if not all(isinstance(item, str) for item in issues):
            raise ValueError("AI 审查结果字段类型错误: issues")
        if not all(isinstance(item, str) for item in suggestions):
            raise ValueError("AI 审查结果字段类型错误: suggestions")

    def _validate_generation_payload(self, section: str, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict) or not payload:
            raise ValueError("AI 生成结果不是有效 JSON 对象。")

        if section == "review":
            self._validate_review_payload(payload)
            return

        if section == "identity":
            self._validate_identity_payload(payload)
            return

        if section == "greetings":
            core_messages = payload.get("coreMessages", payload)
            if not isinstance(core_messages, dict):
                raise ValueError("AI 生成结果缺少 coreMessages。")
            self._validate_core_messages_payload(core_messages)
            return

        if section == "lorebook":
            lorebook = payload.get("lorebook", payload)
            if not isinstance(lorebook, dict):
                raise ValueError("AI 生成结果缺少 lorebook。")
            if "name" in lorebook and not isinstance(lorebook.get("name"), str):
                raise ValueError("AI 生成结果字段类型错误: lorebook.name")
            self._require_list(lorebook, "entries")
            return

        if section == "regex":
            scripts = payload.get("regexScripts", payload.get("regex_scripts"))
            if not isinstance(scripts, list):
                raise ValueError("AI 生成结果缺少 regexScripts。")
            return

        if section == "state-tasks":
            tasks = payload.get("stateTasks", payload.get("state_tasks"))
            if not isinstance(tasks, list):
                raise ValueError("AI 生成结果缺少 stateTasks。")
            return

        if section == "translate":
            identity = payload.get("identity")
            core_messages = payload.get("coreMessages")
            if not isinstance(identity, dict) or not isinstance(core_messages, dict):
                raise ValueError("AI 生成结果缺少 identity 或 coreMessages。")
            self._validate_identity_payload(identity)
            self._validate_core_messages_payload(core_messages)
            return

        if section == "full":
            identity = payload.get("identity")
            core_messages = payload.get("coreMessages")
            lorebook = payload.get("lorebook")
            regex_scripts = payload.get("regexScripts")
            state_tasks = payload.get("stateTasks")
            if not isinstance(identity, dict):
                raise ValueError("AI 生成结果缺少 identity。")
            if not isinstance(core_messages, dict):
                raise ValueError("AI 生成结果缺少 coreMessages。")
            if not isinstance(lorebook, dict):
                raise ValueError("AI 生成结果缺少 lorebook。")
            if not isinstance(regex_scripts, list):
                raise ValueError("AI 生成结果缺少 regexScripts。")
            if not isinstance(state_tasks, list):
                raise ValueError("AI 生成结果缺少 stateTasks。")
            self._validate_identity_payload(identity)
            self._validate_core_messages_payload(core_messages)
            if "name" in lorebook and not isinstance(lorebook.get("name"), str):
                raise ValueError("AI 生成结果字段类型错误: lorebook.name")
            self._require_list(lorebook, "entries")
            return

    def _parse_review_payload(self, content: str) -> Dict[str, Any]:
        generated = parse_json_block_from_text(content)
        if not isinstance(generated, dict):
            raise OpenAICompatibleBusinessRetryableError("AI 审查结果不是 JSON 对象。")
        try:
            self._validate_generation_payload("review", generated)
        except ValueError as exc:
            raise OpenAICompatibleBusinessRetryableError(str(exc)) from exc
        return generated

    def _parse_fact_driven_payload(
        self,
        content: str,
        *,
        document: Dict[str, Any],
        section: str,
        frozen_sections: set[str],
    ) -> Dict[str, Any]:
        generated = parse_json_block_from_text(content)
        if not isinstance(generated, dict):
            raise OpenAICompatibleBusinessRetryableError("AI 生成结果不是 JSON 对象。")
        generated = self._normalize_generated_payload(section, generated)
        try:
            self._validate_generation_payload(section, generated)
            updated = self._apply_section_payload(document, section, generated, frozen_sections=frozen_sections)
            diagnostics = build_diagnostics_report(updated)
            if diagnostics.get("errors"):
                raise ValueError(f"AI 生成结果校验失败：{'；'.join(diagnostics['errors'])}")
        except ValueError as exc:
            raise OpenAICompatibleBusinessRetryableError(str(exc)) from exc
        return generated

    async def _generate_structured_payload(
        self,
        *,
        client: ChatClient,
        prompt: str,
        temperature: float,
        parser,
    ) -> Dict[str, Any]:
        if hasattr(client, "generate_parsed"):
            return await client.generate_parsed(
                prompt,
                parser=parser,
                temperature=temperature,
            )
        return parser(await client.generate(prompt=prompt, temperature=temperature))

    @staticmethod
    def _coerce_string_list(value: Any) -> List[str]:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if not isinstance(value, list):
            return []
        results: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                results.append(text)
        return results

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off", ""}:
                return False
        return default

    def _normalize_regex_placement(self, value: Any) -> List[int]:
        raw_values = value if isinstance(value, list) else [value]
        placements: List[int] = []
        for item in raw_values:
            placement = self._coerce_int(item, -1)
            if placement in {1, 2}:
                placements.append(placement)
        return placements or [2]

    def _normalize_trigger_timing(self, value: Any) -> tuple[str, bool]:
        raw = str(value or "").strip().lower()
        mapping = {
            "initialization": "initialization",
            "init": "initialization",
            "startup": "initialization",
            "message_received": "message_received",
            "message_receive": "message_received",
            "received": "message_received",
            "user_message": "message_received",
            "conversation_start": "message_received",
            "message_sent": "message_sent",
            "message_send": "message_sent",
            "sent": "message_sent",
            "assistant_message": "message_sent",
            "conversation_end": "message_sent",
        }
        normalized = mapping.get(raw)
        if normalized:
            return normalized, False
        return "message_sent", True

    def _normalize_lorebook_entry(self, entry: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None

        comment = str(
            entry.get("comment")
            or entry.get("name")
            or entry.get("title")
            or ""
        ).strip()
        keys = self._coerce_string_list(entry.get("keys", entry.get("key", [])))
        if not comment and keys:
            comment = keys[0]
        if not keys and comment:
            keys = [comment]
        secondary_keys = self._coerce_string_list(
            entry.get("secondary_keys", entry.get("secondaryKeys", entry.get("keysecondary", [])))
        )
        children = [
            normalized
            for child in entry.get("children", []) or []
            for normalized in [self._normalize_lorebook_entry(child)]
            if normalized
        ]

        return {
            "id": str(entry.get("id") or ""),
            "comment": comment,
            "keys": keys,
            "secondary_keys": secondary_keys,
            "content": str(entry.get("content", entry.get("description", "")) or "").strip(),
            "enabled": self._coerce_bool(entry.get("enabled"), True),
            "constant": self._coerce_bool(entry.get("constant"), False),
            "selective": self._coerce_bool(entry.get("selective"), True),
            "priority": self._coerce_int(entry.get("priority", entry.get("order", 100)), 100),
            "position": str(entry.get("position", "before_char") or "before_char"),
            "depth": self._coerce_int(entry.get("depth", 4), 4),
            "probability": self._coerce_int(entry.get("probability", 100), 100),
            "prevent_recursion": self._coerce_bool(entry.get("prevent_recursion", entry.get("preventRecursion")), True),
            "use_regex": self._coerce_bool(entry.get("use_regex", entry.get("useRegex")), False),
            "match_persona_description": self._coerce_bool(entry.get("match_persona_description"), True),
            "match_character_description": self._coerce_bool(entry.get("match_character_description"), True),
            "match_character_personality": self._coerce_bool(entry.get("match_character_personality"), True),
            "match_character_depth_prompt": self._coerce_bool(entry.get("match_character_depth_prompt"), True),
            "match_scenario": self._coerce_bool(entry.get("match_scenario"), True),
            "children": children,
        }

    def _normalize_lorebook_payload(self, lorebook: Any) -> Dict[str, Any]:
        if not isinstance(lorebook, dict):
            return {"name": "", "entries": []}
        raw_entries = lorebook.get("entries", [])
        if isinstance(raw_entries, dict):
            raw_entries = list(raw_entries.values())
        normalized_entries = [
            normalized
            for entry in raw_entries or []
            for normalized in [self._normalize_lorebook_entry(entry)]
            if normalized
        ]
        return {
            "name": str(lorebook.get("name", "") or "").strip(),
            "entries": normalized_entries,
        }

    def _normalize_regex_scripts_payload(self, scripts: Any) -> List[Dict[str, Any]]:
        if not isinstance(scripts, list):
            return []
        normalized_scripts: List[Dict[str, Any]] = []
        for idx, item in enumerate(scripts):
            if not isinstance(item, dict):
                continue
            placement = self._normalize_regex_placement(item.get("placement", [2]))
            has_unsupported_condition = str(item.get("condition", "") or "").strip() != ""
            normalized_scripts.append({
                "id": str(item.get("id") or ""),
                "scriptName": str(item.get("scriptName") or item.get("name") or f"脚本{idx + 1}").strip(),
                "findRegex": str(item.get("findRegex") or item.get("regex") or item.get("pattern") or "").strip(),
                "replaceString": str(item.get("replaceString") or item.get("replacement") or item.get("replace") or "").strip(),
                "placement": placement,
                "markdownOnly": self._coerce_bool(item.get("markdownOnly"), False),
                "promptOnly": self._coerce_bool(item.get("promptOnly"), False),
                "runOnEdit": self._coerce_bool(item.get("runOnEdit"), True),
                "disabled": self._coerce_bool(item.get("disabled"), False) or has_unsupported_condition,
            })
        return normalized_scripts

    def _build_placeholder_task_commands(self, task: Dict[str, Any]) -> str:
        lines = [
            "<<taskjs>>",
            "// AI generated placeholder task metadata. Please replace with executable STscript logic if needed.",
        ]
        description = str(task.get("description", "") or "").strip()
        trigger = str(task.get("trigger", "") or "").strip()
        action = str(task.get("action", "") or "").strip()
        if description:
            lines.append(f"// description: {description}")
        if trigger:
            lines.append(f"// trigger: {trigger}")
        if action:
            lines.append(f"// action: {action}")
        lines.append("<</taskjs>>")
        return "\n".join(lines)

    def _normalize_state_tasks_payload(self, tasks: Any) -> List[Dict[str, Any]]:
        if not isinstance(tasks, list):
            return []
        normalized_tasks: List[Dict[str, Any]] = []
        for idx, item in enumerate(tasks):
            if not isinstance(item, dict):
                continue
            trigger_timing, unsupported_trigger = self._normalize_trigger_timing(
                item.get("triggerTiming") or item.get("trigger")
            )
            commands = str(item.get("commands", "") or "").strip()
            used_placeholder = not commands
            if not commands:
                commands = self._build_placeholder_task_commands(item)
            normalized_tasks.append({
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or f"任务{idx + 1}").strip(),
                "triggerTiming": trigger_timing,
                "interval": self._coerce_int(item.get("interval", 0), 0),
                "commands": commands,
                "disabled": self._coerce_bool(item.get("disabled"), False) or used_placeholder or unsupported_trigger,
            })
        return normalized_tasks

    def _normalize_generated_payload(self, section: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(payload)

        if section == "lorebook":
            lorebook = normalized.get("lorebook", normalized)
            return {"lorebook": self._normalize_lorebook_payload(lorebook)}

        if section == "regex":
            scripts = normalized.get("regexScripts", normalized.get("regex_scripts", normalized))
            return {"regexScripts": self._normalize_regex_scripts_payload(scripts)}

        if section == "state-tasks":
            tasks = normalized.get("stateTasks", normalized.get("state_tasks", normalized))
            return {"stateTasks": self._normalize_state_tasks_payload(tasks)}

        if section == "full":
            lorebook = self._normalize_lorebook_payload(normalized.get("lorebook", {}))
            normalized["lorebook"] = lorebook
            normalized["regexScripts"] = self._normalize_regex_scripts_payload(
                normalized.get("regexScripts", normalized.get("regex_scripts", []))
            )
            normalized["stateTasks"] = self._normalize_state_tasks_payload(
                normalized.get("stateTasks", normalized.get("state_tasks", []))
            )
            return normalized

        return normalized

    def _apply_section_payload(
        self,
        document: Dict[str, Any],
        section: str,
        payload: Dict[str, Any],
        *,
        frozen_sections: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        updated = copy.deepcopy(document)
        frozen = frozen_sections or set(updated.get("status", {}).get("frozen_sections", []) or [])
        if section == "identity":
            updated["identity"].update({
                "name": payload.get("name", updated["identity"]["name"]),
                "description": payload.get("description", updated["identity"]["description"]),
                "personality": payload.get("personality", updated["identity"]["personality"]),
                "scenario": payload.get("scenario", updated["identity"]["scenario"]),
            })
            updated["meta"]["title"] = updated["identity"]["name"]
        elif section == "greetings":
            core_messages = payload.get("coreMessages", payload)
            updated["coreMessages"]["first_message"] = core_messages.get("first_message", updated["coreMessages"]["first_message"])
            updated["coreMessages"]["message_example"] = core_messages.get("message_example", updated["coreMessages"]["message_example"])
            updated["coreMessages"]["system_prompt"] = core_messages.get("system_prompt", updated["coreMessages"]["system_prompt"])
            updated["coreMessages"]["post_history_instructions"] = core_messages.get("post_history_instructions", updated["coreMessages"]["post_history_instructions"])
            updated["coreMessages"]["creator_notes"] = core_messages.get("creator_notes", updated["coreMessages"]["creator_notes"])
            updated["coreMessages"]["character_version"] = core_messages.get("character_version", updated["coreMessages"]["character_version"])
            alternates = core_messages.get("alternate_greetings", updated["coreMessages"]["alternate_greetings"])
            updated["coreMessages"]["alternate_greetings"] = alternates if isinstance(alternates, list) else []
        elif section == "lorebook":
            lorebook = payload.get("lorebook", payload)
            if isinstance(lorebook, dict):
                updated["lorebook"]["name"] = lorebook.get("name", updated["lorebook"]["name"])
            entries = payload.get("entries", payload.get("lorebook", {}).get("entries", []))
            if isinstance(entries, list):
                updated["lorebook"]["entries"] = entries
        elif section == "regex":
            scripts = payload.get("regex_scripts", payload.get("regexScripts", []))
            if isinstance(scripts, list):
                updated["regexScripts"] = scripts
        elif section == "state-tasks":
            tasks = payload.get("state_tasks", payload.get("stateTasks", []))
            if isinstance(tasks, list):
                updated["stateTasks"] = tasks
        elif section == "translate":
            identity = payload.get("identity")
            core_messages = payload.get("coreMessages")
            if isinstance(identity, dict):
                updated["identity"].update(identity)
                updated["meta"]["title"] = updated["identity"]["name"]
            if isinstance(core_messages, dict):
                updated["coreMessages"].update(core_messages)
        elif section == "full":
            identity = payload.get("identity")
            core_messages = payload.get("coreMessages")
            lorebook = payload.get("lorebook")
            if isinstance(identity, dict) and "identity" not in frozen:
                updated["identity"].update(identity)
                updated["meta"]["title"] = updated["identity"]["name"]
            if isinstance(core_messages, dict) and "greetings" not in frozen:
                updated["coreMessages"].update(core_messages)
            if isinstance(lorebook, dict) and "lorebook" not in frozen:
                updated["lorebook"]["name"] = lorebook.get("name", updated["lorebook"]["name"])
                if isinstance(lorebook.get("entries"), list):
                    updated["lorebook"]["entries"] = lorebook["entries"]
            regex_scripts = payload.get("regexScripts")
            if isinstance(regex_scripts, list) and "regex" not in frozen:
                updated["regexScripts"] = regex_scripts
            state_tasks = payload.get("stateTasks")
            if isinstance(state_tasks, list) and "state-tasks" not in frozen:
                updated["stateTasks"] = state_tasks
        elif section == "review":
            updated.setdefault("exportArtifacts", {})
            updated["exportArtifacts"]["last_review"] = {
                "summary": str(payload.get("summary", "") or ""),
                "issues": list(payload.get("issues", []) or []),
                "suggestions": list(payload.get("suggestions", []) or []),
                "generated_at": datetime.now().isoformat(),
            }
        return ensure_document_shape(updated, book_id=self.book_id)

    async def validate_document(self, doc_id: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        report = build_diagnostics_report(document)
        document["status"]["last_validated_at"] = datetime.now().isoformat()
        document["exportArtifacts"]["last_diagnostics"] = report
        await self.store.save_document(document)
        return report

    async def export_document(self, doc_id: str, format_name: str) -> Tuple[bytes, str, str]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        bundle = build_export_bundle(document)
        title = document["meta"]["title"] or document["identity"]["name"] or doc_id
        safe_title = self.storage.safe_card_filename(title)
        if format_name == "v3":
            payload = json.dumps(bundle["v3"], ensure_ascii=False, indent=2).encode("utf-8")
            return payload, "application/json", f"{safe_title}.json"
        if format_name == "v2":
            payload = json.dumps(bundle["v2"], ensure_ascii=False, indent=2).encode("utf-8")
            return payload, "application/json", f"{safe_title}.v2.json"
        if format_name == "worldbook":
            payload = json.dumps(bundle["worldbook"], ensure_ascii=False, indent=2).encode("utf-8")
            return payload, "application/json", f"{safe_title}.worldbook.json"
        if format_name == "png":
            png_bytes = export_png_bytes(document, base_image_path=document["avatar"].get("asset_path"))
            return png_bytes, "image/png", f"{safe_title}.png"
        raise ValueError(f"不支持的导出格式: {format_name}")

    async def import_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        document = import_document_payload(self.book_id, payload)
        await self.store.save_document(document)
        return document

    async def import_png(self, png_bytes: bytes) -> Dict[str, Any]:
        from .png_codec import CharacterStudioPngCodec

        payload = CharacterStudioPngCodec.read_card_png(png_bytes)
        if not payload:
            raise ValueError("PNG 中未找到角色卡数据")
        return await self.import_payload(payload)

    async def import_image(self, *, title: str, extension: str, image_bytes: bytes) -> Dict[str, Any]:
        document = create_empty_document(self.book_id, title=title or "导入图片角色", origin_type="imported")
        avatar_path = await self.store.save_avatar_asset(document["id"], extension, image_bytes)
        document["avatar"]["mode"] = "asset"
        document["avatar"]["asset_path"] = avatar_path
        await self.store.save_document(document)
        return document

    async def import_worldbook_into_document(self, doc_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        imported = import_document_payload(self.book_id, payload)
        document["lorebook"]["entries"].extend(imported["lorebook"]["entries"])
        await self.store.save_document(document)
        return document

    async def run_agent(self, doc_id: str, message: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        chat_index = await self.store.load_chat_index(doc_id)
        active_chat_id = str(chat_index.get("active_session_id") or "").strip()
        session = {
            "doc_id": doc_id,
            "messages": [],
            "variables": {},
            "log": [],
        }
        if active_chat_id:
            chat_session = await self.store.load_chat_session(doc_id, active_chat_id)
            if chat_session:
                last_message = (chat_session.get("messages", []) or [{}])[-1]
                session = {
                    "doc_id": doc_id,
                    "messages": [
                        {"role": item.get("role", "assistant"), "content": item.get("content", "")}
                        for item in (chat_session.get("messages", []) or [])
                    ],
                    "variables": copy.deepcopy(chat_session.get("variables", {})),
                    "log": copy.deepcopy(last_message.get("runtime_log", [])),
                }
        compressed_context = await self._ensure_compressed_context()
        context = build_agent_context(document, session, compressed_context)
        prompt = (
            "你是 Manga Insight Character Studio 的角色卡助手。"
            "你可以审查角色卡、建议世界书、建议问候语、建议正则脚本和状态任务。"
            "如需修改，请输出 ```json:patch 代码块，并严格遵循上下文里列出的 patch 契约与字段命名。"
            "\n\n上下文如下：\n"
            + context
            + "\n\n用户请求:\n"
            + message
        )
        client = self._create_chat_client()
        if not client:
            raise ValueError("未配置可用的 AI 对话模型，无法使用卡片助手。")
        try:
            response_text = await client.generate(prompt=prompt, temperature=0.6)
        except Exception as exc:
            logger.error("Agent 调用失败: %s", exc, exc_info=True)
            raise ValueError(f"卡片助手调用失败：{exc}") from exc
        finally:
            try:
                await client.close()
            except Exception:
                pass
        return {
            "content": response_text,
            "context": context,
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _new_session_id() -> str:
        return f"chat_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _new_message_id() -> str:
        return f"msg_{uuid.uuid4().hex[:12]}"

    def _build_chat_greetings(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        greetings: List[Dict[str, Any]] = []
        first_message = str(document.get("coreMessages", {}).get("first_message", "") or "").strip()
        if first_message:
            greetings.append({
                "greeting_id": "first_message",
                "label": "主问候",
                "content": first_message,
                "source": {"type": "first_message", "index": 0},
            })
        for index, item in enumerate(document.get("coreMessages", {}).get("alternate_greetings", []) or [], start=1):
            content = str(item or "").strip()
            if not content:
                continue
            greetings.append({
                "greeting_id": f"alternate_{index}",
                "label": f"备用问候 {index}",
                "content": content,
                "source": {"type": "alternate_greetings", "index": index - 1},
            })
        return greetings

    def _create_chat_message(
        self,
        *,
        role: str,
        content: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        runtime_log: Optional[List[Dict[str, Any]]] = None,
        variables_snapshot: Optional[Dict[str, Any]] = None,
        generation_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = self._now_iso()
        return {
            "message_id": self._new_message_id(),
            "role": role,
            "content": str(content or ""),
            "attachments": copy.deepcopy(attachments or []),
            "runtime_log": copy.deepcopy(runtime_log or []),
            "variables_snapshot": copy.deepcopy(variables_snapshot or {}),
            "generation_meta": copy.deepcopy(generation_meta or {}),
            "created_at": now,
            "updated_at": now,
        }

    def _build_session_summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        messages = session.get("messages", []) or []
        last_message = messages[-1] if messages else {}
        return {
            "session_id": session.get("session_id"),
            "title": session.get("title", "新对话"),
            "message_count": len(messages),
            "updated_at": session.get("updated_at"),
            "archived_at": session.get("archived_at"),
            "last_message_excerpt": str(last_message.get("content", "") or "")[:120],
        }

    def _create_empty_chat_session(
        self,
        doc_id: str,
        *,
        title: str = "新对话",
        greeting_source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = self._now_iso()
        return {
            "session_id": self._new_session_id(),
            "doc_id": doc_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "archived_at": None,
            "greeting_source": copy.deepcopy(greeting_source or {"type": "first_message", "index": 0}),
            "summary_blocks": [],
            "messages": [],
            "variables": {},
            "_runtime": {
                "event_counts": {
                    "message_received": 0,
                    "message_sent": 0,
                },
                "matched_lorebook_ids": [],
            },
            "last_prompt_preview": "",
        }

    def _reset_chat_runtime(self, document: Dict[str, Any], session: Dict[str, Any]) -> None:
        initialized = initialize_runtime_session(document)
        session["variables"] = copy.deepcopy(initialized.get("variables", {}))
        session["_runtime"] = copy.deepcopy(initialized.get("_runtime", {
            "event_counts": {"message_received": 0, "message_sent": 0},
            "matched_lorebook_ids": [],
        }))

    def _create_opening_chat_session(
        self,
        document: Dict[str, Any],
        *,
        greeting_content: Optional[str] = None,
        greeting_source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self._create_empty_chat_session(
            document["id"],
            greeting_source=greeting_source,
        )
        self._reset_chat_runtime(document, session)
        opening = str(
            greeting_content
            if greeting_content is not None
            else document.get("coreMessages", {}).get("first_message", "")
        ).strip()
        if opening:
            session["messages"].append(
                self._create_chat_message(
                    role="assistant",
                    content=opening,
                    variables_snapshot=session.get("variables", {}),
                    generation_meta={"kind": "opening"},
                )
            )
        return session

    async def _bootstrap_chat_session(self, document: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = document["id"]
        index = await self.store.load_chat_index(doc_id)
        active_session_id = str(index.get("active_session_id") or "").strip()
        if active_session_id:
            session = await self.store.load_chat_session(doc_id, active_session_id)
            if session:
                return session

        greetings = self._build_chat_greetings(document)
        greeting = greetings[0] if greetings else None
        session = self._create_opening_chat_session(
            document,
            greeting_content=greeting["content"] if greeting else "",
            greeting_source=greeting["source"] if greeting else None,
        )

        index["active_session_id"] = session["session_id"]
        index["archived_session_ids"] = list(index.get("archived_session_ids", []) or [])
        await self.store.save_chat_session(doc_id, session)
        await self.store.save_chat_index(doc_id, index)
        return session

    async def _load_chat_session_or_raise(self, doc_id: str, session_id: str) -> Dict[str, Any]:
        session = await self.store.load_chat_session(doc_id, session_id)
        if not session:
            raise ValueError("聊天会话不存在")
        return session

    def _attachment_to_data_url(self, asset_path: str, mime_type: str) -> str:
        with open(asset_path, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:{mime_type};base64,{payload}"

    def _build_chat_system_prompt(
        self,
        document: Dict[str, Any],
        session: Dict[str, Any],
        lorebook_hits: List[Dict[str, Any]],
    ) -> str:
        parts = [
            document.get("coreMessages", {}).get("system_prompt", "") or "你需要始终忠于当前角色设定。",
            f"角色名: {document['identity'].get('name', '')}",
            f"角色描述: {document['identity'].get('description', '')}",
            f"角色人格: {document['identity'].get('personality', '')}",
            f"当前场景: {document['identity'].get('scenario', '')}",
            f"额外要求: {document['coreMessages'].get('post_history_instructions', '')}",
            f"当前状态变量: {json.dumps(session.get('variables', {}), ensure_ascii=False)}",
        ]
        summary_blocks = session.get("summary_blocks", []) or []
        if summary_blocks:
            parts.append("会话摘要:")
            for block in summary_blocks:
                parts.append(f"- {block.get('content', '')}")
        if lorebook_hits:
            parts.append("命中的世界书条目:")
            for entry in lorebook_hits:
                content = str(entry.get("content", "") or "")
                depth = max(int(entry.get("depth", 4) or 4), 1)
                parts.append(f"- {entry.get('comment', '')}: {content[:depth * 160]}")
        return "\n".join(part for part in parts if str(part).strip())

    def _build_chat_request_messages(
        self,
        document: Dict[str, Any],
        session: Dict[str, Any],
        lorebook_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{
            "role": "system",
            "content": self._build_chat_system_prompt(document, session, lorebook_hits),
        }]
        covered_message_ids = {
            str(message_id)
            for block in (session.get("summary_blocks", []) or [])
            for message_id in (block.get("covered_message_ids", []) or [])
        }
        for item in session.get("messages", []) or []:
            if str(item.get("message_id", "")) in covered_message_ids:
                continue
            role = str(item.get("role", "assistant") or "assistant")
            attachments = item.get("attachments", []) or []
            if role == "user" and attachments:
                content_parts: List[Dict[str, Any]] = []
                for attachment in attachments:
                    asset_path = str(attachment.get("asset_path", "") or "")
                    mime_type = str(attachment.get("mime_type", "image/png") or "image/png")
                    if asset_path and os.path.exists(asset_path):
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": self._attachment_to_data_url(asset_path, mime_type)},
                        })
                content_parts.append({"type": "text", "text": str(item.get("content", "") or "")})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": role, "content": str(item.get("content", "") or "")})
        session["last_prompt_preview"] = json.dumps(messages, ensure_ascii=False, indent=2)
        return messages

    @staticmethod
    def _emit_chat_event(
        on_event: Optional[Callable[[Dict[str, Any]], None]],
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        if on_event:
            on_event({"type": event_type, **payload})

    async def get_chat_state(self, doc_id: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = await self._bootstrap_chat_session(document)
        index = await self.store.load_chat_index(doc_id)
        archived_sessions: List[Dict[str, Any]] = []
        for archived_id in index.get("archived_session_ids", []) or []:
            archived = await self.store.load_chat_session(doc_id, str(archived_id))
            if archived:
                archived_sessions.append(self._build_session_summary(archived))
        archived_sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return {
            "doc_id": doc_id,
            "active_session": session,
            "archived_sessions": archived_sessions,
            "available_greetings": self._build_chat_greetings(document),
        }

    async def create_new_chat_session(
        self,
        doc_id: str,
        *,
        greeting_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        index = await self.store.load_chat_index(doc_id)
        active_id = str(index.get("active_session_id") or "").strip()
        if active_id:
            active = await self.store.load_chat_session(doc_id, active_id)
            if active:
                active["archived_at"] = self._now_iso()
                active["updated_at"] = active["archived_at"]
                await self.store.save_chat_session(doc_id, active)
                archived_ids = [str(item) for item in (index.get("archived_session_ids", []) or [])]
                if active_id not in archived_ids:
                    archived_ids.insert(0, active_id)
                index["archived_session_ids"] = archived_ids
        greetings = self._build_chat_greetings(document)
        greeting = next((item for item in greetings if item["greeting_id"] == greeting_id), None)
        if not greeting:
            greeting = greetings[0] if greetings else None
        session = self._create_opening_chat_session(
            document,
            greeting_content=greeting["content"] if greeting else "",
            greeting_source=greeting["source"] if greeting else None,
        )
        index["active_session_id"] = session["session_id"]
        await self.store.save_chat_session(doc_id, session)
        await self.store.save_chat_index(doc_id, index)
        return await self.get_chat_state(doc_id)

    async def switch_chat_session(self, doc_id: str, session_id: str) -> Dict[str, Any]:
        target_session = await self._load_chat_session_or_raise(doc_id, session_id)
        index = await self.store.load_chat_index(doc_id)
        active_id = str(index.get("active_session_id") or "").strip()
        archived_ids = [str(item) for item in (index.get("archived_session_ids", []) or [])]
        if active_id and active_id != session_id:
            active_session = await self.store.load_chat_session(doc_id, active_id)
            if active_session:
                active_session["archived_at"] = self._now_iso()
                active_session["updated_at"] = active_session["archived_at"]
                await self.store.save_chat_session(doc_id, active_session)
            if active_id not in archived_ids:
                archived_ids.insert(0, active_id)
        archived_ids = [item for item in archived_ids if item != session_id]
        index["active_session_id"] = session_id
        index["archived_session_ids"] = archived_ids
        target_session["archived_at"] = None
        target_session["updated_at"] = self._now_iso()
        await self.store.save_chat_session(doc_id, target_session)
        await self.store.save_chat_index(doc_id, index)
        return await self.get_chat_state(doc_id)

    async def _generate_assistant_message(
        self,
        document: Dict[str, Any],
        session: Dict[str, Any],
        *,
        raw_user_content: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        visible_user, prompt_user, user_hits = apply_regex_scripts(
            raw_user_content,
            document.get("regexScripts", []),
            placement=1,
            respect_run_on_edit=True,
        )
        session["messages"][-1]["content"] = visible_user
        session["messages"][-1]["generation_meta"]["original_content"] = raw_user_content
        lorebook_hits = sort_lorebook_hits(
            match_lorebook(
                document.get("lorebook", {}).get("entries", []),
                prompt_user,
                session=session,
            )
        )
        runtime_log: List[Dict[str, Any]] = list(user_hits)
        for entry in lorebook_hits:
            runtime_log.append({
                "type": "lorebook",
                "comment": entry.get("comment", ""),
                "keys": entry.get("keys", []),
                "position": entry.get("position", "before_char"),
                "depth": entry.get("depth", 4),
            })
        runtime_log.extend(run_state_tasks(session, document.get("stateTasks", []), event="message_received"))
        prompt_messages = self._build_chat_request_messages(document, session, lorebook_hits)
        self._emit_chat_event(on_event, "runtime", {
            "runtime_log": runtime_log,
            "variables": copy.deepcopy(session.get("variables", {})),
        })

        client = self._create_vlm_client()
        if not client:
            raise ValueError("未配置可用的 VLM 模型，无法进行角色聊天。")

        streamed_chunks: List[str] = []

        def handle_chunk(delta: str) -> None:
            streamed_chunks.append(delta)
            self._emit_chat_event(on_event, "assistant_delta", {
                "delta": delta,
                "content": "".join(streamed_chunks),
            })

        try:
            assistant_text = await client.generate_messages(
                prompt_messages,
                on_stream_chunk=handle_chunk,
            )
        finally:
            try:
                await client.close()
            except Exception:
                pass

        visible_assistant, _, assistant_hits = apply_regex_scripts(
            assistant_text,
            document.get("regexScripts", []),
            placement=2,
            respect_run_on_edit=True,
        )
        runtime_log.extend(assistant_hits)
        runtime_log.extend(run_state_tasks(session, document.get("stateTasks", []), event="message_sent"))
        assistant_message = self._create_chat_message(
            role="assistant",
            content=visible_assistant,
            runtime_log=runtime_log,
            variables_snapshot=session.get("variables", {}),
            generation_meta={"prompt_preview": session.get("last_prompt_preview", "")},
        )
        session["messages"].append(assistant_message)
        session["updated_at"] = self._now_iso()
        self._emit_chat_event(on_event, "assistant_done", {
            "message_id": assistant_message["message_id"],
            "content": assistant_message["content"],
        })
        return session

    async def send_chat_message(
        self,
        doc_id: str,
        *,
        session_id: str,
        content: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        if not str(content or "").strip() and not (attachments or []):
            raise ValueError("消息内容不能为空")
        session = await self._load_chat_session_or_raise(doc_id, session_id)

        stored_attachments: List[Dict[str, Any]] = []
        for index, attachment in enumerate(attachments or []):
            raw_bytes = attachment.get("bytes")
            if not isinstance(raw_bytes, (bytes, bytearray)):
                continue
            filename = str(attachment.get("filename") or f"attachment_{index}.bin")
            mime_type = str(attachment.get("mime_type") or "application/octet-stream")
            asset_path = await self.store.save_chat_attachment(
                doc_id,
                session_id,
                filename=filename,
                data=bytes(raw_bytes),
            )
            stored_attachments.append({
                "attachment_id": f"att_{uuid.uuid4().hex[:10]}",
                "filename": filename,
                "mime_type": mime_type,
                "asset_path": asset_path,
                "created_at": self._now_iso(),
            })

        user_message = self._create_chat_message(
            role="user",
            content=content,
            attachments=stored_attachments,
            variables_snapshot=session.get("variables", {}),
            generation_meta={"original_content": content},
        )
        session["messages"].append(user_message)
        await self._generate_assistant_message(
            document,
            session,
            raw_user_content=content,
            on_event=on_event,
        )
        await self.store.save_chat_session(doc_id, session)
        return {
            "session": session,
            "available_greetings": self._build_chat_greetings(document),
        }

    async def edit_chat_message(
        self,
        doc_id: str,
        *,
        session_id: str,
        message_id: str,
        new_content: str,
    ) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = await self._load_chat_session_or_raise(doc_id, session_id)
        messages = session.get("messages", []) or []
        target_index = next((idx for idx, item in enumerate(messages) if item.get("message_id") == message_id), -1)
        if target_index < 0:
            raise ValueError("消息不存在")
        target = messages[target_index]
        target["content"] = new_content
        target["updated_at"] = self._now_iso()
        target.setdefault("generation_meta", {})["original_content"] = new_content
        session["messages"] = messages[: target_index + 1]
        self._reset_chat_runtime(document, session)
        session["updated_at"] = self._now_iso()
        await self.store.save_chat_session(doc_id, session)
        return {"session": session}

    async def delete_chat_message(
        self,
        doc_id: str,
        *,
        session_id: str,
        message_id: str,
    ) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = await self._load_chat_session_or_raise(doc_id, session_id)
        messages = session.get("messages", []) or []
        target_index = next((idx for idx, item in enumerate(messages) if item.get("message_id") == message_id), -1)
        if target_index < 0:
            raise ValueError("消息不存在")
        session["messages"] = messages[:target_index]
        self._reset_chat_runtime(document, session)
        session["updated_at"] = self._now_iso()
        await self.store.save_chat_session(doc_id, session)
        return {"session": session}

    async def regenerate_chat_message(
        self,
        doc_id: str,
        *,
        session_id: str,
        anchor_message_id: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = await self._load_chat_session_or_raise(doc_id, session_id)
        messages = session.get("messages", []) or []
        anchor_index = next((idx for idx, item in enumerate(messages) if item.get("message_id") == anchor_message_id), -1)
        if anchor_index < 0:
            raise ValueError("消息不存在")
        if messages[anchor_index].get("role") == "assistant":
            user_index = next(
                (idx for idx in range(anchor_index - 1, -1, -1) if messages[idx].get("role") == "user"),
                -1,
            )
            if user_index < 0:
                raise ValueError("未找到可重生的用户消息")
        else:
            user_index = anchor_index
        session["messages"] = messages[: user_index + 1]
        self._reset_chat_runtime(document, session)
        raw_user_content = str(
            session["messages"][-1].get("generation_meta", {}).get("original_content")
            or session["messages"][-1].get("content", "")
            or ""
        )
        await self._generate_assistant_message(
            document,
            session,
            raw_user_content=raw_user_content,
            on_event=on_event,
        )
        await self.store.save_chat_session(doc_id, session)
        return {"session": session}

    async def summarize_chat_session(
        self,
        doc_id: str,
        *,
        session_id: str,
        cutoff_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = await self._load_chat_session_or_raise(doc_id, session_id)
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        messages = session.get("messages", []) or []
        if cutoff_message_id:
            target_index = next((idx for idx, item in enumerate(messages) if item.get("message_id") == cutoff_message_id), -1)
            if target_index < 0:
                raise ValueError("消息不存在")
            source_messages = messages[: target_index + 1]
        else:
            source_messages = messages
        if not source_messages:
            raise ValueError("当前会话没有可总结的消息")
        prompt = (
            f"请总结角色 {document['identity'].get('name', '')} 与用户的聊天进展。"
            "用中文输出一段紧凑摘要，保留关键剧情、关系变化、状态变量和未完成事项。\n\n"
            + "\n".join(f"{item.get('role')}: {item.get('content')}" for item in source_messages)
        )
        client = self._create_vlm_client()
        if not client:
            raise ValueError("未配置可用的 VLM 模型，无法总结聊天。")
        try:
            summary_text = await client.generate_messages([{"role": "user", "content": prompt}])
        finally:
            try:
                await client.close()
            except Exception:
                pass
        session.setdefault("summary_blocks", []).append({
            "summary_id": f"sum_{uuid.uuid4().hex[:10]}",
            "content": summary_text,
            "created_at": self._now_iso(),
            "covered_message_ids": [item.get("message_id") for item in source_messages],
        })
        session["updated_at"] = self._now_iso()
        await self.store.save_chat_session(doc_id, session)
        return {"session": session}

    async def export_chat_session(self, doc_id: str, *, session_id: str) -> Dict[str, Any]:
        session = await self._load_chat_session_or_raise(doc_id, session_id)
        payload = copy.deepcopy(session)
        for message in payload.get("messages", []) or []:
            for attachment in message.get("attachments", []) or []:
                asset_path = str(attachment.get("asset_path", "") or "")
                if asset_path and os.path.exists(asset_path):
                    with open(asset_path, "rb") as handle:
                        attachment["blob_base64"] = base64.b64encode(handle.read()).decode("utf-8")
                attachment.pop("asset_path", None)
        return payload

    async def import_chat_session(self, doc_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        imported = copy.deepcopy(payload or {})
        imported["doc_id"] = doc_id
        imported["session_id"] = self._new_session_id()
        imported["created_at"] = self._now_iso()
        imported["updated_at"] = imported["created_at"]
        imported["archived_at"] = None
        for message in imported.get("messages", []) or []:
            message["message_id"] = self._new_message_id()
            for index, attachment in enumerate(message.get("attachments", []) or []):
                blob_base64 = str(attachment.pop("blob_base64", "") or "")
                if not blob_base64:
                    continue
                restored = await self.store.save_chat_attachment(
                    doc_id,
                    imported["session_id"],
                    filename=str(attachment.get("filename") or f"attachment_{index}.bin"),
                    data=base64.b64decode(blob_base64),
                )
                attachment["asset_path"] = restored
        index = await self.store.load_chat_index(doc_id)
        active_id = str(index.get("active_session_id") or "").strip()
        if active_id:
            current = await self.store.load_chat_session(doc_id, active_id)
            if current:
                current["archived_at"] = self._now_iso()
                current["updated_at"] = current["archived_at"]
                await self.store.save_chat_session(doc_id, current)
                archived_ids = [str(item) for item in (index.get("archived_session_ids", []) or [])]
                if active_id not in archived_ids:
                    archived_ids.insert(0, active_id)
                index["archived_session_ids"] = archived_ids
        index["active_session_id"] = imported["session_id"]
        await self.store.save_chat_session(doc_id, imported)
        await self.store.save_chat_index(doc_id, index)
        return await self.get_chat_state(doc_id)

    async def get_chat_prompt_preview(self, doc_id: str, *, session_id: str) -> Dict[str, Any]:
        session = await self._load_chat_session_or_raise(doc_id, session_id)
        return {
            "session_id": session_id,
            "prompt_preview": session.get("last_prompt_preview", ""),
        }
