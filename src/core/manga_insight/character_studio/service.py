"""
Core Character Studio services.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.manga_insight.config_utils import create_chat_client, has_provider_model_config, load_insight_config
from src.core.manga_insight.embedding_client import ChatClient
from src.core.manga_insight.storage import AnalysisStorage
from src.core.manga_insight.utils.json_parser import parse_llm_json

from .adapters import (
    build_export_bundle,
    create_empty_document,
    ensure_document_shape,
    export_png_bytes,
    import_document_payload,
)
from .agent import build_agent_context
from .preview import (
    apply_regex_scripts,
    initialize_preview_session,
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
    "lorebook": """你是角色世界书设计助手。请基于压缩摘要与目标角色名，为该角色构建世界书。输出 JSON 对象，字段 only: lorebook，其中 lorebook 必须包含 name, entries。entries 为数组，每项至少包含 comment, keys, secondary_keys, content, enabled, constant, selective, priority, position, depth, children。""",
    "regex": """你是角色脚本助手。请基于压缩摘要与目标角色名，为该角色生成运行时正则脚本。输出 JSON 对象，字段 only: regexScripts。regexScripts 为数组，每项包含 scriptName, findRegex, replaceString, placement, markdownOnly, promptOnly, runOnEdit, disabled。""",
    "state-tasks": """你是状态任务助手。请基于压缩摘要与目标角色名，为该角色生成状态任务。输出 JSON 对象，字段 only: stateTasks。stateTasks 为数组，每项包含 name, triggerTiming, interval, commands, disabled。""",
    "review": """你是角色卡审查员。请基于压缩摘要与目标角色名，审查当前角色卡是否忠于原始剧情信息。输出 JSON 对象，字段 only: summary, issues, suggestions。summary 必须为字符串；issues 与 suggestions 必须为字符串数组。""",
    "translate": """你是专业翻译与角色卡整理助手。请参考压缩摘要与当前角色卡草稿，将角色卡正文翻译或整理为更自然的中文。输出 JSON 对象，字段 only: identity, coreMessages。""",
    "full": """你是角色卡构建助手。请基于压缩摘要与目标角色名，一次性生成完整角色卡初稿。输出 JSON 对象，字段 only: identity, coreMessages, lorebook, regexScripts, stateTasks。

identity 必须包含：name, description, personality, scenario。
coreMessages 必须包含：first_message, message_example, alternate_greetings, system_prompt, post_history_instructions, creator_notes, character_version。
lorebook 必须包含：name, entries。
regexScripts 必须是数组。
stateTasks 必须是数组。

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
            response = await client.generate(prompt=prompt, temperature=0.4)
            generated = parse_llm_json(response, default={})
        except Exception as exc:
            logger.error("生成审查报告失败: %s", exc, exc_info=True)
            raise ValueError(f"AI 审查失败：{exc}") from exc
        finally:
            try:
                await client.close()
            except Exception:
                pass

        self._validate_generation_payload("review", generated)
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
            response = await client.generate(prompt=prompt, temperature=0.45)
            generated = parse_llm_json(response, default={})
        except Exception as exc:
            logger.error("生成 section %s 失败: %s", section, exc, exc_info=True)
            raise ValueError(f"AI 生成失败：{exc}") from exc
        finally:
            try:
                await client.close()
            except Exception:
                pass

        self._validate_generation_payload(section, generated)
        updated = self._apply_section_payload(document, section, generated, frozen_sections=frozen_sections)
        diagnostics = build_diagnostics_report(updated)
        if diagnostics.get("errors"):
            raise ValueError(f"AI 生成结果校验失败：{'；'.join(diagnostics['errors'])}")
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

    async def preview_reset(self, doc_id: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = initialize_preview_session(document)
        await self.store.save_preview_session(doc_id, session)
        return session

    async def preview_chat(self, doc_id: str, message: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = await self.store.load_preview_session(doc_id)
        if not session.get("messages"):
            session = initialize_preview_session(document)
            opening = document["coreMessages"]["first_message"]
            if opening:
                session["messages"].append({"role": "assistant", "content": opening})

        visible_user, prompt_user, user_hits = apply_regex_scripts(
            message,
            document.get("regexScripts", []),
            placement=1,
            respect_run_on_edit=True,
        )
        lorebook_hits = sort_lorebook_hits(
            match_lorebook(
                document.get("lorebook", {}).get("entries", []),
                prompt_user,
                session=session,
            )
        )
        session["messages"].append({"role": "user", "content": visible_user})
        session["log"].extend(user_hits)
        for entry in lorebook_hits:
            session["log"].append({
                "type": "lorebook",
                "comment": entry.get("comment", ""),
                "keys": entry.get("keys", []),
                "position": entry.get("position", "before_char"),
                "depth": entry.get("depth", 4),
            })
        session["log"].extend(run_state_tasks(session, document.get("stateTasks", []), event="message_received"))

        prompt_lines = [
            document["coreMessages"].get("system_prompt", ""),
            f"角色名: {document['identity']['name']}",
            f"角色描述: {document['identity']['description']}",
            f"角色人格: {document['identity']['personality']}",
            f"当前场景: {document['identity']['scenario']}",
            f"额外要求: {document['coreMessages']['post_history_instructions']}",
        ]
        if lorebook_hits:
            prompt_lines.append("命中的世界书条目:")
            for entry in lorebook_hits:
                content = str(entry.get("content", "") or "")
                depth = max(int(entry.get("depth", 4) or 4), 1)
                prompt_lines.append(f"- {entry.get('comment', '')}: {content[:depth * 160]}")
        history_text = "\n".join(
            f"{item.get('role')}: {item.get('content')}" for item in session["messages"][-8:]
        )
        prompt_lines.append("最近对话:")
        prompt_lines.append(history_text)
        prompt_lines.append(f"用户输入: {prompt_user}")
        assistant_text = "我收到你的信息了，我们继续推进。"
        client = self._create_chat_client()
        if client:
            try:
                assistant_text = await client.generate(prompt="\n".join(prompt_lines), temperature=0.7)
            except Exception as exc:
                logger.warning("预览聊天调用失败，回退默认回复: %s", exc)
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
        session["messages"].append({"role": "assistant", "content": visible_assistant})
        session["log"].extend(assistant_hits)
        session["log"].extend(run_state_tasks(session, document.get("stateTasks", []), event="message_sent"))
        await self.store.save_preview_session(doc_id, session)
        return session

    async def run_agent(self, doc_id: str, message: str) -> Dict[str, Any]:
        document = await self.store.load_document(doc_id)
        if not document:
            raise ValueError("文档不存在")
        session = await self.store.load_preview_session(doc_id)
        compressed_context = await self._ensure_compressed_context()
        context = build_agent_context(document, session, compressed_context)
        prompt = (
            "你是 Manga Insight Character Studio 的角色卡助手。"
            "你可以审查角色卡、建议世界书、建议问候语、建议正则脚本和状态任务。"
            "如需修改，请输出 ```json:patch 代码块。"
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
