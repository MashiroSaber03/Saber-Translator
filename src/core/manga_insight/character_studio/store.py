"""
Character Studio storage helpers.
"""

from __future__ import annotations

import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from src.core.manga_insight.storage import AnalysisStorage

from .adapters import ensure_document_shape


class CharacterStudioStore:
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.storage = AnalysisStorage(book_id)
        self.base_path = self.storage.base_path
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for relative in (
            "character_studio",
            "character_studio/documents",
            "character_studio/assets",
            "character_studio/assets/avatars",
            "character_studio/assets/chat",
            "character_studio/chat_sessions",
            "character_studio/imports",
            "character_studio/exports",
        ):
            os.makedirs(os.path.join(self.base_path, relative), exist_ok=True)

    async def load_index(self) -> Dict[str, Any]:
        data = await self.storage._load_json("character_studio/index.json", None)
        if not data:
            return {"book_id": self.book_id, "documents": []}
        if not isinstance(data.get("documents"), list):
            data["documents"] = []
        data.setdefault("book_id", self.book_id)
        return data

    async def save_index(self, index_data: Dict[str, Any]) -> bool:
        payload = dict(index_data or {})
        payload["book_id"] = self.book_id
        return await self.storage._save_json("character_studio/index.json", payload)

    async def load_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        payload = await self.storage._load_json(f"character_studio/documents/{doc_id}.json", None)
        if not payload:
            return None
        return ensure_document_shape(payload, book_id=self.book_id)

    async def list_documents(self) -> List[Dict[str, Any]]:
        index_data = await self.load_index()
        return index_data.get("documents", [])

    async def save_document(self, document: Dict[str, Any]) -> bool:
        normalized = ensure_document_shape(document, book_id=self.book_id)
        ok = await self.storage._save_json(
            f"character_studio/documents/{normalized['id']}.json",
            normalized,
        )
        if not ok:
            return False
        index_data = await self.load_index()
        documents = [item for item in index_data.get("documents", []) if item.get("id") != normalized["id"]]
        documents.insert(0, self._build_summary(normalized))
        index_data["documents"] = sorted(
            documents,
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )
        await self.save_index(index_data)
        return True

    async def delete_document(self, doc_id: str) -> bool:
        index_data = await self.load_index()
        index_data["documents"] = [item for item in index_data.get("documents", []) if item.get("id") != doc_id]
        await self.save_index(index_data)
        deleted = await self.storage._delete_file_if_exists(
            os.path.join(self.base_path, "character_studio", "documents", f"{doc_id}.json")
        )
        chat_dir = os.path.join(self.base_path, "character_studio", "chat_sessions", doc_id)
        if os.path.isdir(chat_dir):
            shutil.rmtree(chat_dir, ignore_errors=True)
        chat_asset_dir = os.path.join(self.base_path, "character_studio", "assets", "chat", doc_id)
        if os.path.isdir(chat_asset_dir):
            shutil.rmtree(chat_asset_dir, ignore_errors=True)
        return deleted

    async def save_avatar_asset(self, doc_id: str, extension: str, image_bytes: bytes) -> str:
        ext = (extension or ".png").lower()
        filename = f"{doc_id}{ext}"
        path = os.path.join(self.base_path, "character_studio", "assets", "avatars", filename)
        ok = await self.storage._save_binary(path, image_bytes)
        if not ok:
            raise RuntimeError("保存头像失败")
        return path

    async def save_export_binary(self, doc_id: str, format_name: str, data: bytes) -> str:
        filename = f"{doc_id}.{format_name}"
        path = os.path.join(self.base_path, "character_studio", "exports", filename)
        ok = await self.storage._save_binary(path, data)
        if not ok:
            raise RuntimeError("保存导出文件失败")
        return path

    def _chat_doc_dir(self, doc_id: str) -> str:
        return os.path.join(self.base_path, "character_studio", "chat_sessions", doc_id)

    async def load_chat_index(self, doc_id: str) -> Dict[str, Any]:
        data = await self.storage._load_json(
            f"character_studio/chat_sessions/{doc_id}/index.json",
            None,
        )
        return data or {
            "doc_id": doc_id,
            "active_session_id": None,
            "archived_session_ids": [],
        }

    async def save_chat_index(self, doc_id: str, index_data: Dict[str, Any]) -> bool:
        payload = dict(index_data or {})
        payload["doc_id"] = doc_id
        return await self.storage._save_json(
            f"character_studio/chat_sessions/{doc_id}/index.json",
            payload,
        )

    async def load_chat_session(self, doc_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        payload = await self.storage._load_json(
            f"character_studio/chat_sessions/{doc_id}/{session_id}.json",
            None,
        )
        return payload or None

    async def save_chat_session(self, doc_id: str, session_data: Dict[str, Any]) -> bool:
        session_id = str(session_data.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("chat session 缺少 session_id")
        os.makedirs(self._chat_doc_dir(doc_id), exist_ok=True)
        return await self.storage._save_json(
            f"character_studio/chat_sessions/{doc_id}/{session_id}.json",
            session_data,
        )

    async def save_chat_attachment(
        self,
        doc_id: str,
        session_id: str,
        *,
        filename: str,
        data: bytes,
    ) -> str:
        safe_name = os.path.basename(filename or "attachment.bin")
        unique_name = f"{session_id}_{uuid.uuid4().hex[:10]}_{safe_name}"
        session_asset_dir = os.path.join(
            self.base_path,
            "character_studio",
            "assets",
            "chat",
            doc_id,
            session_id,
        )
        os.makedirs(session_asset_dir, exist_ok=True)
        path = os.path.join(session_asset_dir, unique_name)
        ok = await self.storage._save_binary(path, data)
        if not ok:
            raise RuntimeError("保存聊天附件失败")
        return path

    @staticmethod
    def _build_summary(document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": document["id"],
            "title": document["meta"]["title"],
            "origin": document["origin"]["type"],
            "source_character": document["origin"].get("source_character"),
            "updated_at": document["meta"]["updated_at"],
            "tags": document["meta"].get("tags", []),
            "is_favorite": bool(document["status"].get("is_favorite", False)),
            "has_avatar": bool(document["avatar"].get("asset_path")),
            "sample_pages": document["grounding"].get("sample_pages", []),
        }
