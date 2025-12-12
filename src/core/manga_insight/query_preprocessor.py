"""
查询预处理模块
参考 RAGFlow: rag/nlp/query.py, rag/nlp/term_weight.py

实现功能：
1. 去除疑问词
2. 关键词匹配分数计算
"""
import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("MangaInsight.QueryPreprocessor")


class QueryPreprocessor:
    """
    查询预处理器
    照搬 RAGFlow 的核心逻辑：去疑问词、词权重
    """
    
    # ========== 照搬 RAGFlow query.py 的 rmWWW 正则 ==========
    QUESTION_PATTERNS = [
        # 中文疑问词 (照搬 RAGFlow)
        (r"是*(怎么办|什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*", ""),
        # 英文停用词 (照搬 RAGFlow)
        (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
        (r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ", " ")
    ]
    
    def __init__(self):
        pass
    
    def preprocess(self, question: str) -> Dict:
        """
        预处理查询
        
        Args:
            question: 原始问题
            
        Returns:
            {
                "original": 原始问题,
                "clean_query": 清理后的问题 (用于 embedding),
                "keywords": 关键词列表 (用于关键词匹配),
                "tokens_with_weight": [(词, 权重), ...]
            }
        """
        if not question or not question.strip():
            return {
                "original": question,
                "clean_query": "",
                "keywords": [],
                "tokens_with_weight": []
            }
        
        original = question.strip()
        
        # 1. 标准化 (照搬 RAGFlow)
        text = self._normalize(original)
        
        # 2. 去除疑问词 (照搬 RAGFlow rmWWW)
        text = self._remove_question_words(text)
        
        # 3. 不分词，直接用清理后的文本作为关键词
        # 整个清理后的文本就是用于匹配的关键词
        keywords = [text] if text else []
        
        # 4. 计算词权重 (简化版 RAGFlow term_weight)
        tokens_with_weight = self._compute_weights(keywords)
        
        # 5. 重建清理后的查询
        clean_query = " ".join(keywords) if keywords else text
        
        result = {
            "original": original,
            "clean_query": clean_query,
            "keywords": keywords,
            "tokens_with_weight": tokens_with_weight
        }
        
        logger.debug(f"查询预处理: '{original}' -> keywords={keywords}")
        return result
    
    def _normalize(self, text: str) -> str:
        """标准化文本 - 照搬 RAGFlow"""
        # 全角转半角
        text = self._strQ2B(text)
        # 统一小写
        text = text.lower()
        # 去除多余空白和标点
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"[,，。？?！!;；:：\"""'''()（）【】{}<>《》·~～`@#$%^&*+=|\\/_.\\[\\]-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _strQ2B(self, text: str) -> str:
        """全角转半角 - 照搬 RAGFlow"""
        result = []
        for char in text:
            code = ord(char)
            if code == 0x3000:  # 全角空格
                code = 0x0020
            elif 0xFF01 <= code <= 0xFF5E:  # 全角字符
                code -= 0xFEE0
            result.append(chr(code))
        return "".join(result)
    
    def _remove_question_words(self, text: str) -> str:
        """去除疑问词 - 照搬 RAGFlow rmWWW"""
        original = text
        for pattern, replacement in self.QUESTION_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        # 如果处理后为空，返回原文
        return text if text else original
    
    def _compute_weights(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """
        计算词权重 - 简化版 RAGFlow term_weight
        
        权重逻辑：
        - 长词权重更高 (信息量大)
        - 中文词权重更高
        - 纯数字降权
        - 专有名词模式加权
        """
        if not tokens:
            return []
        
        weights = []
        for token in tokens:
            w = 1.0
            
            # 长度加权 (参考 RAGFlow 的 IDF 思想，长词通常更具体)
            if len(token) >= 4:
                w *= 1.5
            elif len(token) >= 3:
                w *= 1.3
            elif len(token) >= 2:
                w *= 1.1
            
            # 中文加权
            if re.search(r"[\u4e00-\u9fff]", token):
                w *= 1.2
            
            # 纯数字降权 (但保留，因为"第5页"这种很重要)
            if re.match(r"^[0-9]+$", token):
                w *= 0.7
            
            # 可能是人名/角色名的模式加权 (2-4字中文)
            if re.match(r"^[\u4e00-\u9fff]{2,4}$", token):
                w *= 1.1
            
            weights.append((token, w))
        
        # 归一化
        total = sum(w for _, w in weights)
        if total > 0:
            weights = [(t, round(w / total, 4)) for t, w in weights]
        
        return weights
    
    def compute_keyword_score(
        self, 
        content: str, 
        keywords: List[str],
        tokens_with_weight: List[Tuple[str, float]] = None
    ) -> float:
        """
        计算关键词匹配分数 - 照搬 RAGFlow query.py 的 similarity
        
        Args:
            content: 待匹配的文档内容
            keywords: 关键词列表
            tokens_with_weight: 带权重的词列表 (可选)
            
        Returns:
            float: 匹配分数 (0-1)
        """
        if not content or not keywords:
            return 0.0
        
        content_lower = content.lower()
        
        if tokens_with_weight:
            # 加权匹配 (照搬 RAGFlow similarity)
            matched_weight = 0.0
            total_weight = sum(w for _, w in tokens_with_weight)
            
            for token, weight in tokens_with_weight:
                if token.lower() in content_lower:
                    matched_weight += weight
            
            return matched_weight / total_weight if total_weight > 0 else 0.0
        else:
            # 简单匹配
            hits = sum(1 for kw in keywords if kw.lower() in content_lower)
            return hits / len(keywords) if keywords else 0.0


# 全局单例
_preprocessor = None


def get_preprocessor() -> QueryPreprocessor:
    """获取预处理器单例"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = QueryPreprocessor()
    return _preprocessor
