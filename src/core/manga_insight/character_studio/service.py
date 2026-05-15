"""
Core Character Studio services.
"""

from __future__ import annotations

import copy
import io
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
    "identity": """你是角色卡编辑助手。根据以下角色 dossier，输出 JSON 对象，字段仅包含 name, description, personality, scenario。""",
    "greetings": """你是角色卡问候语设计助手。根据以下角色 dossier，输出 JSON 对象，字段仅包含 first_message, alternate_greetings。alternate_greetings 为字符串数组。""",
    "lorebook": """你是角色世界书设计助手。根据以下角色 dossier，输出 JSON 对象，字段仅包含 entries。entries 为世界书条目数组，每个条目包含 comment, keys, content。""",
    "regex": """你是角色脚本助手。根据以下角色 dossier，输出 JSON 对象，字段仅包含 regex_scripts。每项包含 scriptName, findRegex, replaceString, placement, markdownOnly, promptOnly, runOnEdit, disabled。""",
    "state-tasks": """你是状态任务助手。根据以下角色 dossier，输出 JSON 对象，字段仅包含 state_tasks。每项包含 name, triggerTiming, interval, commands, disabled。""",
    "review": """你是角色卡审查员。请输出 JSON 对象，字段 only: summary, issues, suggestions。issues/suggestions 均为数组。""",
    "translate": """你是专业翻译。把以下角色卡内容翻译成中文并保持 JSON 结构。""",
}


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
                "description": character.get("description", "") or "",
                "arc": character.get("arc", "") or "",
                "dialogue_count": len(dialogues),
                "has_dialogues": len(dialogues) > 0,
                "sample_pages": sample_pages,
                "relationship_count": len(character.get("relationships", []) or []),
                "key_moment_count": len(character.get("key_moments", []) or []),
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
        overview = await self.storage.load_overview() or {}
        characters = timeline.get("characters", []) or []
        target = next((item for item in characters if item.get("name") == candidate_name), None)
        if not target:
            raise ValueError(f"未找到候选角色: {candidate_name}")
        document = create_empty_document(self.book_id, title=candidate_name, origin_type="analysis")
        document["origin"]["source_character"] = candidate_name
        document["identity"].update({
            "name": candidate_name,
            "aliases": [a for a in target.get("aliases", []) if isinstance(a, str)],
            "description": target.get("description", "") or "",
            "personality": target.get("arc", "") or "",
            "scenario": (overview.get("book_summary") or overview.get("summary") or "")[:600],
        })
        document["meta"]["tags"] = ["manga-insight", candidate_name]
        document["grounding"].update({
            "timeline_mode": timeline.get("mode", ""),
            "relationships": target.get("relationships", []) or [],
            "key_moments": target.get("key_moments", []) or [],
        })
        dialogues = await self._collect_dialogues_for_character([candidate_name] + document["identity"]["aliases"])
        document["coreMessages"]["first_message"] = dialogues[0]["text"] if dialogues else f"我是{candidate_name}。我们开始吧。"
        document["coreMessages"]["message_example"] = (
            "<START>\n{{user}}: 我们接下来怎么办？\n{{char}}: 先稳住局势，再寻找关键线索。"
        )
        document["coreMessages"]["alternate_greetings"] = [
            f"你来得正好，我是{candidate_name}，我们可以直接进入正题。",
            f"别浪费时间了，我是{candidate_name}。你想先确认哪一部分？",
        ]
        document["grounding"]["sample_pages"] = sorted({int(item.get("page", 0)) for item in dialogues if int(item.get("page", 0)) > 0})[:8]
        document["lorebook"]["entries"] = [
            {
                "id": "entry_origin",
                "comment": "角色基底",
                "keys": [candidate_name] + document["identity"]["aliases"][:3],
                "secondary_keys": [],
                "content": "\n".join(filter(None, [
                    document["identity"]["description"],
                    f"角色成长线：{target.get('arc', '')}" if target.get("arc") else "",
                ])),
                "enabled": True,
                "constant": False,
                "selective": True,
                "priority": 100,
                "position": "before_char",
                "depth": 4,
                "children": [],
            }
        ]
        document["regexScripts"] = [{
            "id": "regex_state_hide",
            "scriptName": "隐藏状态块",
            "findRegex": "<state>[\\s\\S]*?</state>",
            "replaceString": "",
            "placement": [2],
            "markdownOnly": False,
            "promptOnly": False,
            "runOnEdit": True,
            "disabled": False,
        }]
        document["stateTasks"] = [{
            "id": "task_init",
            "name": "初始化好感度",
            "triggerTiming": "initialization",
            "interval": 0,
            "commands": "<<taskjs>>\nawait STscript('/setvar key=trust_score 20');\n<</taskjs>>",
            "disabled": False,
        }]
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
        if section in set(document.get("status", {}).get("frozen_sections", []) or []):
            return document
        dossier = self._build_dossier(document)
        prompt = (
            SECTION_PROMPTS[section]
            + "\n\n角色 dossier:\n"
            + json.dumps(dossier, ensure_ascii=False, indent=2)
        )
        client = self._create_chat_client()
        generated: Dict[str, Any] = {}
        if client:
            try:
                response = await client.generate(prompt=prompt, temperature=0.5)
                generated = parse_llm_json(response, default={})
            except Exception as exc:
                logger.warning("生成 section %s 失败，回退默认模板: %s", section, exc)
            finally:
                try:
                    await client.close()
                except Exception:
                    pass
        if not generated:
            generated = self._fallback_generation(document, section)
        updated = self._apply_section_payload(document, section, generated)
        await self.store.save_document(updated)
        return updated

    def _build_dossier(self, document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "identity": document.get("identity", {}),
            "grounding": document.get("grounding", {}),
            "lorebook_count": len(document.get("lorebook", {}).get("entries", [])),
            "regex_count": len(document.get("regexScripts", [])),
            "task_count": len(document.get("stateTasks", [])),
            "coreMessages": document.get("coreMessages", {}),
        }

    def _fallback_generation(self, document: Dict[str, Any], section: str) -> Dict[str, Any]:
        name = document["identity"]["name"] or "角色"
        if section == "identity":
            return {
                "name": name,
                "description": document["identity"]["description"] or f"{name} 是故事中的关键角色。",
                "personality": document["identity"]["personality"] or "冷静、观察力强、会在关键时刻做出决断。",
                "scenario": document["identity"]["scenario"] or "故事推进阶段，冲突尚未完全解决。",
            }
        if section == "greetings":
            return {
                "first_message": document["coreMessages"]["first_message"] or f"我是{name}。先告诉我，现在最紧急的事情是什么？",
                "alternate_greetings": document["coreMessages"]["alternate_greetings"] or [f"你来了。我们可以继续推进下一步了。"],
            }
        if section == "lorebook":
            return {
                "entries": document["lorebook"]["entries"] or [],
            }
        if section == "regex":
            return {
                "regex_scripts": document["regexScripts"] or [],
            }
        if section == "state-tasks":
            return {
                "state_tasks": document["stateTasks"] or [],
            }
        if section == "review":
            report = build_diagnostics_report(document)
            return {
                "summary": "角色卡基础结构可用，请根据诊断结果继续完善。",
                "issues": report.get("errors", []),
                "suggestions": report.get("warnings", []),
            }
        if section == "translate":
            return {
                "identity": document["identity"],
                "coreMessages": document["coreMessages"],
            }
        return {}

    def _apply_section_payload(self, document: Dict[str, Any], section: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        updated = copy.deepcopy(document)
        if section == "identity":
            updated["identity"].update({
                "name": payload.get("name", updated["identity"]["name"]),
                "description": payload.get("description", updated["identity"]["description"]),
                "personality": payload.get("personality", updated["identity"]["personality"]),
                "scenario": payload.get("scenario", updated["identity"]["scenario"]),
            })
            updated["meta"]["title"] = updated["identity"]["name"]
        elif section == "greetings":
            updated["coreMessages"]["first_message"] = payload.get("first_message", updated["coreMessages"]["first_message"])
            alternates = payload.get("alternate_greetings", updated["coreMessages"]["alternate_greetings"])
            updated["coreMessages"]["alternate_greetings"] = alternates if isinstance(alternates, list) else []
        elif section == "lorebook":
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
        context = build_agent_context(document, session)
        prompt = (
            "你是 Manga Insight Character Studio 的角色卡助手。"
            "你可以审查角色卡、建议世界书、建议问候语、建议正则脚本和状态任务。"
            "如需修改，请输出 ```json:patch 代码块。"
            "\n\n当前文档摘要:\n"
            + context
            + "\n\n用户请求:\n"
            + message
        )
        response_text = "我建议先补充问候语和世界书条目。"
        client = self._create_chat_client()
        if client:
            try:
                response_text = await client.generate(prompt=prompt, temperature=0.6)
            except Exception as exc:
                logger.warning("Agent 调用失败，回退默认回复: %s", exc)
            finally:
                try:
                    await client.close()
                except Exception:
                    pass
        return {
            "content": response_text,
            "context": context,
        }
