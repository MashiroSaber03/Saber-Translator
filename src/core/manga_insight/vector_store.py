"""
Manga Insight 向量存储

基于 ChromaDB 的向量存储实现。
"""

import logging
import os
from typing import List, Dict, Optional, Any

from src.shared.path_helpers import resource_path
from .storage import get_insight_storage_path

logger = logging.getLogger("MangaInsight.VectorStore")

# 尝试导入 ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB 未安装，向量存储功能将不可用")


class MangaVectorStore:
    """漫画向量存储"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        # 使用统一的路径系统
        self.storage_path = os.path.join(get_insight_storage_path(book_id), "embeddings")
        
        self.client = None
        self.pages_collection = None
        self.dialogues_collection = None
        self.scenes_collection = None
        self.events_collection = None  # 新增: 事件集合 (key_events)
        
        if CHROMADB_AVAILABLE:
            self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """初始化 ChromaDB"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.storage_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 创建不同的 collection
            self.pages_collection = self.client.get_or_create_collection(
                name="pages",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.dialogues_collection = self.client.get_or_create_collection(
                name="dialogues",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.scenes_collection = self.client.get_or_create_collection(
                name="scenes",
                metadata={"hnsw:space": "cosine"}
            )
            
            # 新增: 事件集合 (用于存储 key_events)
            self.events_collection = self.client.get_or_create_collection(
                name="events",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.debug(f"ChromaDB 初始化成功: {self.storage_path}")
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        return self.client is not None
    
    async def add_page_embedding(
        self,
        page_num: int,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        添加页面向量
        
        Args:
            page_num: 页码
            embedding: 向量
            metadata: 元数据
        
        Returns:
            bool: 是否成功
        """
        if not self.is_available():
            return False
        
        try:
            doc_id = f"page_{page_num}"
            document = metadata.get("summary", metadata.get("page_summary", ""))
            
            # 清理 metadata，只保留基本类型
            clean_metadata = {
                "page_num": page_num,
                "book_id": self.book_id,
                "type": "page"
            }
            if "chapter_id" in metadata:
                clean_metadata["chapter_id"] = str(metadata["chapter_id"])
            if "parent_batch" in metadata:
                clean_metadata["parent_batch"] = str(metadata["parent_batch"])
            
            self.pages_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                documents=[document]
            )
            return True
        except Exception as e:
            logger.error(f"添加页面向量失败: {e}")
            return False
    
    async def add_dialogue_embedding(
        self,
        dialogue_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        添加对话向量
        
        Args:
            dialogue_id: 对话ID
            embedding: 向量
            metadata: 元数据
        
        Returns:
            bool: 是否成功
        """
        if not self.is_available():
            return False
        
        try:
            document = metadata.get("translated_text", metadata.get("text", ""))
            
            clean_metadata = {
                "page_num": metadata.get("page_num", 0),
                "speaker": str(metadata.get("speaker_name", "")),
                "book_id": self.book_id
            }
            
            self.dialogues_collection.upsert(
                ids=[dialogue_id],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                documents=[document]
            )
            return True
        except Exception as e:
            logger.error(f"添加对话向量失败: {e}")
            return False
    
    async def add_scene_embedding(
        self,
        scene_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """添加场景向量"""
        if not self.is_available():
            return False
        
        try:
            document = metadata.get("description", "")
            
            clean_metadata = {
                "page_num": metadata.get("page_num", 0),
                "location": str(metadata.get("location", "")),
                "book_id": self.book_id
            }
            
            self.scenes_collection.upsert(
                ids=[scene_id],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                documents=[document]
            )
            return True
        except Exception as e:
            logger.error(f"添加场景向量失败: {e}")
            return False
    
    async def search_pages(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        搜索相关页面
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            where: 过滤条件
        
        Returns:
            List[Dict]: 搜索结果
        """
        if not self.is_available():
            return []
        
        try:
            results = self.pages_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            return self._format_results(results)
        except Exception as e:
            logger.error(f"搜索页面失败: {e}")
            return []
    
    async def search_dialogues(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """搜索相关对话"""
        if not self.is_available():
            return []
        
        try:
            results = self.dialogues_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            return self._format_results(results)
        except Exception as e:
            logger.error(f"搜索对话失败: {e}")
            return []
    
    async def search_scenes(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """搜索相关场景"""
        if not self.is_available():
            return []
        
        try:
            results = self.scenes_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            return self._format_results(results)
        except Exception as e:
            logger.error(f"搜索场景失败: {e}")
            return []
    
    # ============================================================
    # 新增: 事件相关方法 (用于 key_events 细粒度检索)
    # ============================================================
    
    async def add_event_embedding(
        self,
        event_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        添加事件向量
        
        Args:
            event_id: 事件ID (如 "event_1_5_0")
            embedding: 向量
            metadata: 元数据 (包含 content, parent_batch, start_page, end_page)
        
        Returns:
            bool: 是否成功
        """
        if not self.is_available() or self.events_collection is None:
            return False
        
        try:
            document = metadata.get("content", "")
            
            clean_metadata = {
                "event_id": event_id,
                "book_id": self.book_id,
                "type": "event",
                "parent_batch": str(metadata.get("parent_batch", "")),
                "start_page": int(metadata.get("start_page", 0)),
                "end_page": int(metadata.get("end_page", 0))
            }
            
            self.events_collection.upsert(
                ids=[event_id],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                documents=[document]
            )
            return True
        except Exception as e:
            logger.error(f"添加事件向量失败: {e}")
            return False
    
    async def search_events(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        搜索相关事件
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            where: 过滤条件
        
        Returns:
            List[Dict]: 搜索结果
        """
        if not self.is_available() or self.events_collection is None:
            return []
        
        try:
            results = self.events_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            return self._format_results(results)
        except Exception as e:
            logger.error(f"搜索事件失败: {e}")
            return []
    
    async def delete_all_events(self) -> bool:
        """删除所有事件向量"""
        if not self.is_available() or self.events_collection is None:
            return False
        
        try:
            # 获取所有 ID 并删除
            all_data = self.events_collection.get()
            if all_data and all_data.get("ids"):
                self.events_collection.delete(ids=all_data["ids"])
            return True
        except Exception as e:
            logger.error(f"删除事件向量失败: {e}")
            return False
    
    async def delete_all_pages(self) -> bool:
        """删除所有页面向量"""
        if not self.is_available() or self.pages_collection is None:
            return False
        
        try:
            all_data = self.pages_collection.get()
            if all_data and all_data.get("ids"):
                self.pages_collection.delete(ids=all_data["ids"])
            return True
        except Exception as e:
            logger.error(f"删除页面向量失败: {e}")
            return False
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """格式化搜索结果"""
        formatted = []
        
        if not results or not results.get("ids"):
            return formatted
        
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []
        
        for i, doc_id in enumerate(ids):
            item = {
                "id": doc_id,
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 0,
                "score": 1 - (distances[i] if i < len(distances) else 0)
            }
            formatted.append(item)
        
        return formatted
    
    async def delete_page_embeddings(self, page_nums: List[int]) -> bool:
        """删除指定页面的向量"""
        if not self.is_available():
            return False
        
        try:
            ids = [f"page_{num}" for num in page_nums]
            self.pages_collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"删除页面向量失败: {e}")
            return False
    
    async def delete_all(self) -> bool:
        """删除所有向量"""
        if not self.is_available():
            return False
        
        try:
            # 重新创建 collections
            self.client.delete_collection("pages")
            self.client.delete_collection("dialogues")
            self.client.delete_collection("scenes")
            try:
                self.client.delete_collection("events")
            except Exception:
                pass  # events collection 可能不存在
            
            self._initialize_chromadb()
            return True
        except Exception as e:
            logger.error(f"删除所有向量失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.is_available():
            return {"available": False}
        
        try:
            return {
                "available": True,
                "pages_count": self.pages_collection.count() if self.pages_collection else 0,
                "dialogues_count": self.dialogues_collection.count() if self.dialogues_collection else 0,
                "scenes_count": self.scenes_collection.count() if self.scenes_collection else 0,
                "events_count": self.events_collection.count() if self.events_collection else 0
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"available": False, "error": str(e)}
