"""
角色卡生成器。
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.manga_insight.book_pages import build_book_pages_manifest
from src.core.manga_insight.config_models import MangaInsightConfig
from src.core.manga_insight.config_utils import load_insight_config
from src.core.manga_insight.embedding_client import ChatClient
from src.core.manga_insight.storage import AnalysisStorage
from src.core.manga_insight.utils.json_parser import parse_llm_json

from .mappers import apply_generated_fields, build_card_template
from .models import CharacterCandidate, CharacterCardV2Draft
from .png_codec import CharacterCardPngCodec
from .validator import build_compatibility_report, normalize_card_v2, validate_card_v2

logger = logging.getLogger("MangaInsight.CharacterCards")


CARD_POLISH_PROMPT = """你是角色卡编辑助手。请基于下列信息生成 SillyTavern 角色卡字段（JSON）。

角色名：{name}
别名：{aliases}
角色简介：{description}
角色成长线：{arc}
故事概览：{overview}
压缩语境：{compressed_context}
对话样本：
{dialogues}

请输出 JSON，仅包含以下字段：
{{
  "description": "...",
  "personality": "...",
  "scenario": "...",
  "first_mes": "...",
  "mes_example": "...",
  "creator_notes": "...",
  "system_prompt": "...",
  "post_history_instructions": "...",
  "alternate_greetings": ["...", "..."],
  "tags": ["...", "..."]
}}

要求：
1) 中文输出；
2) 设定稳定，避免胡编；
3) 保留角色口吻；
4) 不要输出 markdown。
"""


class CharacterCardGenerator:
    """角色卡业务层。"""

    def __init__(self, book_id: str, config: Optional[MangaInsightConfig] = None):
        self.book_id = book_id
        self.storage = AnalysisStorage(book_id)
        self.config = config or load_insight_config()
        self._dialogue_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def _init_chat_client(self) -> Optional[ChatClient]:
        try:
            if self.config.chat_llm and not self.config.chat_llm.use_same_as_vlm:
                if self.config.chat_llm.api_key:
                    return ChatClient(self.config.chat_llm)
                return None
            if self.config.vlm and self.config.vlm.api_key:
                return ChatClient(self.config.vlm)
            return None
        except Exception as e:
            logger.warning(f"初始化角色卡 ChatClient 失败: {e}")
            return None

    async def _load_timeline_required(self) -> Dict[str, Any]:
        timeline = await self.storage.load_timeline()
        if not timeline:
            raise ValueError("未找到时间线缓存，请先生成增强时间线。")
        if not isinstance(timeline.get("characters"), list) or len(timeline.get("characters", [])) == 0:
            raise ValueError("时间线中未包含角色数据，请使用增强模式重新生成时间线。")
        return timeline

    @staticmethod
    def _normalize_name(value: str) -> str:
        value = (value or "").strip().lower()
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[·•\-\_\.\,\!\?\:\;\(\)\[\]\"'“”‘’]", "", value)
        return value

    @classmethod
    def _speaker_matches(cls, speaker: str, names: List[str]) -> bool:
        s = cls._normalize_name(speaker)
        if not s:
            return False
        for name in names:
            n = cls._normalize_name(name)
            if not n:
                continue
            if s == n or s in n or n in s:
                return True
        return False

    @staticmethod
    def _extract_dialogues_from_analysis(analysis: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        # 顶层 dialogues
        for dlg in analysis.get("dialogues", []) or []:
            speaker = dlg.get("speaker_name") or dlg.get("character") or dlg.get("speaker") or ""
            text = dlg.get("text") or dlg.get("translated_text") or ""
            if text:
                items.append({"speaker": speaker, "text": text, "page": page_num})

        # panel 中 dialogues
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
        result: List[Dict[str, Any]] = []

        normalized_keys = {self._normalize_name(name) for name in names if name}
        for key, lines in cache.items():
            if not key:
                continue
            for n in normalized_keys:
                if not n:
                    continue
                if key == n or key in n or n in key:
                    result.extend(lines)
                    break

        # 去重
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
        timeline = await self._load_timeline_required()
        candidates: List[CharacterCandidate] = []

        for character in timeline.get("characters", []):
            name = character.get("name", "").strip()
            if not name:
                continue
            aliases = [a for a in character.get("aliases", []) if isinstance(a, str) and a.strip()]
            lines = await self._collect_dialogues_for_character([name] + aliases)
            pages = sorted({int(item.get("page", 0)) for item in lines if int(item.get("page", 0)) > 0})[:8]

            candidate = CharacterCandidate(
                name=name,
                aliases=aliases,
                first_appearance=int(character.get("first_appearance", 0) or 0),
                description=character.get("description", "") or "",
                arc=character.get("arc", "") or "",
                dialogue_count=len(lines),
                has_dialogues=len(lines) > 0,
                sample_pages=pages,
            )
            candidates.append(candidate)

        return {
            "book_id": self.book_id,
            "candidates": [c.to_dict() for c in candidates],
            "count": len(candidates),
            "generated_at": datetime.now().isoformat(),
        }

    async def _polish_with_llm(
        self,
        character: Dict[str, Any],
        overview_text: str,
        compressed_context: str,
        dialogues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        client = self._init_chat_client()
        if not client:
            return {}

        try:
            sampled_dialogues = []
            for item in dialogues[:10]:
                speaker = item.get("speaker", "")
                text = item.get("text", "")
                page = item.get("page", 0)
                sampled_dialogues.append(f"- 第{page}页 {speaker}: {text}")
            dialogue_text = "\n".join(sampled_dialogues) if sampled_dialogues else "（暂无可用对话）"

            prompt = CARD_POLISH_PROMPT.format(
                name=character.get("name", ""),
                aliases=", ".join(character.get("aliases", [])[:8]) or "无",
                description=character.get("description", "") or "无",
                arc=character.get("arc", "") or "无",
                overview=(overview_text or "无")[:900],
                compressed_context=(compressed_context or "无")[:1400],
                dialogues=dialogue_text[:1800],
            )

            response = await client.generate(prompt=prompt, temperature=0.4)
            parsed = parse_llm_json(response, default={})
            if not isinstance(parsed, dict):
                return {}
            return parsed
        except Exception as e:
            logger.warning(f"LLM 角色卡润色失败，回退模板：{e}")
            return {}
        finally:
            try:
                await client.close()
            except Exception:
                pass

    async def generate_drafts(self, character_names: List[str], style: str = "balanced") -> Dict[str, Any]:
        if not isinstance(character_names, list) or len(character_names) == 0:
            raise ValueError("character_names 不能为空")
        normalized_names: List[str] = []
        for idx, name in enumerate(character_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"character_names[{idx}] 必须为非空字符串")
            normalized_names.append(name.strip())
        character_names = normalized_names

        timeline = await self._load_timeline_required()
        overview = await self.storage.load_overview()
        compressed = await self.storage.load_compressed_context()

        overview_text = (
            (overview or {}).get("book_summary")
            or (overview or {}).get("summary")
            or ""
        )
        compressed_text = (compressed or {}).get("context", "")

        timeline_characters = {c.get("name"): c for c in timeline.get("characters", []) if c.get("name")}

        cards: List[CharacterCardV2Draft] = []
        missing: List[str] = []

        for name in character_names:
            char = timeline_characters.get(name)
            if not char:
                missing.append(name)
                continue

            names = [char.get("name", "")] + [a for a in char.get("aliases", []) if isinstance(a, str)]
            dialogues = await self._collect_dialogues_for_character(names)
            base_card = build_card_template(
                character=char,
                overview_text=overview_text or compressed_text,
                dialogues=dialogues,
                timeline_data=timeline,
            )

            if style == "balanced":
                generated_fields = await self._polish_with_llm(
                    character=char,
                    overview_text=overview_text,
                    compressed_context=compressed_text,
                    dialogues=dialogues,
                )
                base_card = apply_generated_fields(base_card, generated_fields)

            source_stats = {
                "dialogue_count": len(dialogues),
                "sample_pages": sorted(
                    {int(item.get("page", 0)) for item in dialogues if int(item.get("page", 0)) > 0}
                )[:10],
                "relationship_count": len(char.get("relationships", []) or []),
                "key_moment_count": len(char.get("key_moments", []) or []),
                "timeline_mode": timeline.get("mode", ""),
            }

            cards.append(
                CharacterCardV2Draft(
                    character=name,
                    card=base_card,
                    source_stats=source_stats,
                )
            )

        draft = {
            "book_id": self.book_id,
            "style": style,
            "generated_at": datetime.now().isoformat(),
            "cards": [card.to_dict() for card in cards],
            "missing_characters": missing,
        }
        await self.storage.save_character_card_draft(draft)
        return draft

    async def compile_cards(
        self,
        draft: Dict[str, Any],
        character_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(draft, dict):
            raise ValueError("draft 必须为对象")
        cards = draft.get("cards")
        if not isinstance(cards, list):
            raise ValueError("draft.cards 缺失或类型错误")
        if character_names is not None:
            if not isinstance(character_names, list):
                raise ValueError("character_names 必须为数组")
            normalized_names: List[str] = []
            for idx, name in enumerate(character_names):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"character_names[{idx}] 必须为非空字符串")
                normalized_names.append(name.strip())
            character_names = normalized_names

        target_set = set(character_names or [])
        include_all = len(target_set) == 0

        compiled_cards: Dict[str, Dict[str, Any]] = {}
        compatibility_reports: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        warnings: List[str] = []

        for item in cards:
            if not isinstance(item, dict):
                errors.append("草稿项格式错误：cards 内元素必须为对象")
                continue

            character_raw = item.get("character", "")
            if not isinstance(character_raw, str) or not character_raw.strip():
                warnings.append("草稿项缺少 character，已跳过")
                continue
            character = character_raw.strip()
            if not include_all and character not in target_set:
                continue

            card = item.get("card", {})
            if not isinstance(card, dict):
                errors.append(f"[{character}] card 字段必须为对象")
                continue
            normalized_card, normalize_warnings = normalize_card_v2(card)
            for warning in normalize_warnings:
                warnings.append(f"[{character}] {warning}")

            validation = validate_card_v2(normalized_card)
            for err in validation.get("errors", []):
                errors.append(f"[{character}] {err}")
            for warning in validation.get("warnings", []):
                warnings.append(f"[{character}] {warning}")

            if validation.get("valid"):
                compatibility_reports[character] = build_compatibility_report(normalized_card)
                compiled_cards[character] = normalized_card
                await self.storage.save_compiled_character_card(character, normalized_card)

        result = {
            "book_id": self.book_id,
            "valid": len(errors) == 0 and len(compiled_cards) > 0,
            "errors": errors,
            "warnings": warnings,
            "compiled_cards": compiled_cards,
            "compatibility_reports": compatibility_reports,
            "compiled_count": len(compiled_cards),
            "compiled_at": datetime.now().isoformat(),
        }
        return result

    async def get_compat_report(self, character_name: str) -> Dict[str, Any]:
        card = await self._get_card_for_export(character_name)
        normalized_card, normalize_warnings = normalize_card_v2(card)
        report = build_compatibility_report(normalized_card)
        if normalize_warnings:
            report["warnings"] = report.get("warnings", []) + normalize_warnings
        return {
            "book_id": self.book_id,
            "character": character_name,
            "report": report,
            "checked_at": datetime.now().isoformat(),
        }

    async def _get_card_for_export(self, character_name: str) -> Dict[str, Any]:
        if not isinstance(character_name, str) or not character_name.strip():
            raise ValueError("character_name 必须为非空字符串")
        character_name = character_name.strip()

        card = await self.storage.load_compiled_character_card(character_name)
        if card:
            return card

        draft = await self.storage.load_character_card_draft()
        if not draft:
            raise ValueError("未找到角色卡草稿，请先生成草稿。")

        compile_result = await self.compile_cards(draft, [character_name])
        if not compile_result.get("valid"):
            detail = "; ".join(compile_result.get("errors", [])) or "目标角色编译失败"
            raise ValueError(detail)
        if character_name not in compile_result.get("compiled_cards", {}):
            raise ValueError(f"未找到角色 {character_name} 的可导出编译结果")
        return compile_result["compiled_cards"][character_name]

    async def _guess_base_image(self, character_name: str) -> Optional[str]:
        timeline = await self.storage.load_timeline() or {}
        first_page = 0
        for char in timeline.get("characters", []) or []:
            if char.get("name") == character_name:
                first_page = int(char.get("first_appearance", 0) or 0)
                break

        manifest = build_book_pages_manifest(self.book_id)
        images = manifest.get("all_images", [])
        if first_page > 0 and first_page <= len(images):
            path = images[first_page - 1].get("path")
            if path and os.path.exists(path):
                return path
        if images:
            path = images[0].get("path")
            if path and os.path.exists(path):
                return path
        return None

    async def export_png(self, character_name: str) -> Dict[str, Any]:
        if not isinstance(character_name, str) or not character_name.strip():
            raise ValueError("character_name 必须为非空字符串")
        character_name = character_name.strip()

        card = await self._get_card_for_export(character_name)
        base_image_path = await self._guess_base_image(character_name)

        png_bytes = CharacterCardPngCodec.write_card_png(
            card=card,
            base_image_path=base_image_path,
            mirror_ccv3=True,
        )
        ok, err = CharacterCardPngCodec.validate_roundtrip(card, png_bytes)
        if not ok:
            raise ValueError(err)

        file_path = await self.storage.save_character_card_png(character_name, png_bytes)
        log_item = {
            "character": character_name,
            "file_path": file_path,
            "size_bytes": len(png_bytes),
            "exported_at": datetime.now().isoformat(),
        }
        await self.storage.append_character_card_export_log(log_item)

        return {
            "character": character_name,
            "png_bytes": png_bytes,
            "file_path": file_path,
            "size_bytes": len(png_bytes),
            "card": card,
        }

    async def export_batch_zip(self, character_names: List[str]) -> Dict[str, Any]:
        if not isinstance(character_names, list) or len(character_names) == 0:
            raise ValueError("缺少角色列表")
        normalized_names: List[str] = []
        for idx, name in enumerate(character_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"character_names[{idx}] 必须为非空字符串")
            normalized_names.append(name.strip())

        zip_buffer = io.BytesIO()
        exported: List[Dict[str, Any]] = []
        used_names: Dict[str, int] = {}

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for character_name in normalized_names:
                result = await self.export_png(character_name)
                safe_name = self.storage.safe_card_filename(character_name)
                count = used_names.get(safe_name, 0)
                used_names[safe_name] = count + 1
                suffix = f"_{count + 1}" if count > 0 else ""
                filename = f"{safe_name}{suffix}.png"
                zf.writestr(filename, result["png_bytes"])
                exported.append(
                    {
                        "character": character_name,
                        "filename": filename,
                        "size_bytes": result["size_bytes"],
                    }
                )

        zip_buffer.seek(0)
        return {
            "zip_bytes": zip_buffer.getvalue(),
            "count": len(exported),
            "items": exported,
        }
