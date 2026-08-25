import requests
import base64
import logging
import threading
import time
from typing import List

from src.shared.user_logging import inline_log_text, user_log

# 配置日志
logger = logging.getLogger(__name__)
REQUEST_TIMEOUT_SECONDS = 120.0

class BaiduOCRInterface:
    # 百度OCR API端点
    API_ENDPOINTS = {
        "standard": "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic",           # 标准版
        "high_precision": "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",    # 高精度版
    }
    
    # 语言映射 - 百度OCR语言类型参数值
    # 参考文档: https://cloud.baidu.com/doc/OCR/s/zk3h7xz52
    LANGUAGE_MAPPING = {
        "japanese": "JAP",   # 日语（必须大写）
        "korean": "KOR",     # 韩语（必须大写）
        "chinese": "CHN_ENG", # 中文和英文
        "en": "ENG",         # 当前前端使用的英文代码
        "english": "ENG",    # 英文
        "french": "FRE",     # 法语（必须大写）
        "german": "GER",     # 德语（必须大写）
        "spanish": "SPA",    # 西班牙语（必须大写）
        "portuguese": "POR", # 葡萄牙语（必须大写）
        "italian": "ITA",    # 意大利语（必须大写）
        "russian": "RUS",    # 俄语（必须大写）
    }
    
    def __init__(self, api_key: str, secret_key: str, version: str = "standard"):
        """
        初始化百度OCR接口
        
        Args:
            api_key: 百度OCR API Key
            secret_key: 百度OCR Secret Key
            version: OCR版本，"standard"(标准版)或"high_precision"(高精度版)
        """
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("百度 OCR API Key 不能为空")
        if not isinstance(secret_key, str) or not secret_key:
            raise ValueError("百度 OCR Secret Key 不能为空")
        if version not in self.API_ENDPOINTS:
            raise ValueError(f"不支持的百度OCR版本: {version}")
        self.api_key = api_key
        self.secret_key = secret_key
        self.version = version
        self.access_token = None
        self.last_request_time = 0  # 上次请求时间戳
        self._request_interval_lock = threading.Lock()
        
    def _get_access_token(self) -> str:
        """获取百度API访问令牌"""
        response = requests.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("百度访问令牌响应必须是对象")
        access_token = result.get('access_token')
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"获取百度访问令牌失败: {result}")
        return access_token
    
    def _ensure_request_interval(self, min_interval_ms: int = 500):
        """
        确保请求之间有足够的时间间隔，防止触发QPS限制
        
        Args:
            min_interval_ms: 最小请求间隔(毫秒)
        """
        with self._request_interval_lock:
            current_time = time.time() * 1000  # 转换为毫秒
            elapsed = current_time - self.last_request_time

            if elapsed < min_interval_ms:
                sleep_time = (min_interval_ms - elapsed) / 1000
                logger.debug(f"强制请求延迟 {sleep_time:.2f}s 以避免QPS限制")
                time.sleep(sleep_time)

            self.last_request_time = time.time() * 1000
    
    def recognize_text(self, image_bytes: bytes, language: str = "auto") -> List[str]:
        """
        识别图像中的文本
        
        Args:
            image_bytes: 图像字节数据
            language: 语言代码或 'auto_detect' 表示自动检测
            
        Returns:
            识别出的文本列表
        """
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("百度 OCR 图像字节不能为空")
        if not isinstance(language, str) or not language:
            raise ValueError("百度 OCR 语言必须是非空字符串")

        # 确保我们有访问令牌
        if not self.access_token:
            self.access_token = self._get_access_token()
        
        # 准备API端点
        if self.version not in self.API_ENDPOINTS:
            raise ValueError(f"不支持的百度OCR版本: {self.version}")
        endpoint = self.API_ENDPOINTS[self.version]
        
        # 准备请求参数
        params = {
            "access_token": self.access_token
        }
        
        # 准备请求数据
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data = {
            "image": image_base64
        }
        
        # 添加语言参数（如果不是自动检测）
        if language != "auto" and language != "auto_detect":
            # 如果是直接指定的百度OCR语言代码（如CHN_ENG, JAP等）
            lang_code = language
            # 如果language是源语言代码，尝试转换为百度OCR需要的语言代码
            if language.lower() in self.LANGUAGE_MAPPING:
                lang_code = self.LANGUAGE_MAPPING[language.lower()]
            data["language_type"] = lang_code
            logger.debug(f"设置百度OCR语言类型为: {lang_code} (源语言: {language})")
        else:
            # 如果是auto或auto_detect，不设置language_type参数，让API自动检测
            logger.debug("使用自动检测语言，不设置language_type参数")
        
        # 确保请求间隔
        self._ensure_request_interval()
        
        # 发送请求
        max_retries = 3
        retry_delay = 1.0  # 初始重试延迟(秒)
        
        for retry in range(max_retries):
            try:
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                logger.debug(f"发送百度OCR请求 (尝试 {retry+1}/{max_retries})")
                # 不记录完整请求参数，仅记录端点和是否有语言设置
                logger.debug(f"请求端点: {endpoint.split('/')[-1]}, 语言设置: {'有' if 'language_type' in data else '无'}")
                
                response = requests.post(
                    endpoint,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise RuntimeError("百度OCR响应必须是对象")
                
                if 'error_code' in result:
                    error_code = result.get('error_code')
                    error_msg = result.get('error_msg', '未知错误')
                    logger.error(f"百度OCR API错误: {result}")
                    
                    # 处理不同类型的错误
                    if error_code in [110, 111]:  # 令牌过期
                        logger.debug("访问令牌过期，正在重新获取")
                        user_log("warning", "百度 OCR 访问令牌已过期，正在自动刷新")
                        self.access_token = self._get_access_token()
                        params["access_token"] = self.access_token
                        continue  # 使用新令牌重试
                            
                    elif error_code == 18:  # QPS限制
                        if retry < max_retries - 1:
                            wait_time = retry_delay * (retry + 1)
                            logger.debug(f"触发QPS限制，等待 {wait_time} 秒后重试")
                            user_log(
                                "warning",
                                f"百度 OCR 已达到 QPS 上限，等待 {wait_time} 秒后重试",
                            )
                            time.sleep(wait_time)
                            continue  # 等待后重试
                        else:
                            raise RuntimeError("百度OCR达到最大重试次数，QPS限制仍然存在")
                    
                    raise RuntimeError(
                        f"百度OCR错误: {error_code} - {error_msg}"
                    )
                
                # 提取识别文本
                words_result = result.get("words_result")
                if not isinstance(words_result, list):
                    raise RuntimeError("百度OCR words_result 必须是数组")
                text_results = []
                for index, item in enumerate(words_result):
                    if not isinstance(item, dict) or not isinstance(
                        item.get("words"),
                        str,
                    ):
                        raise RuntimeError(
                            f"百度OCR words_result[{index}] 格式无效"
                        )
                    text_results.append(item["words"])
                
                logger.debug(f"百度OCR识别成功，返回 {len(text_results)} 个文本结果")
                return text_results
            
            except (requests.RequestException, ValueError) as e:
                logger.error(f"百度OCR识别时出错: {str(e)}")
                if retry < max_retries - 1:
                    wait_time = retry_delay * (retry + 1)
                    logger.debug(f"百度OCR请求将在 {wait_time} 秒后重试")
                    user_log(
                        "warning",
                        f"百度 OCR 请求失败，等待 {wait_time} 秒后重试｜"
                        f"{inline_log_text(e)}",
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError("百度OCR请求重试耗尽") from e
        
        raise RuntimeError("百度OCR请求重试耗尽")

# 单例实例
_baidu_ocr_instance = None
_baidu_ocr_lock = threading.Lock()

def get_baidu_ocr(api_key: str, secret_key: str, version: str = "standard") -> BaiduOCRInterface:
    """
    获取百度OCR实例（单例模式）
    
    Args:
        api_key: 百度OCR API Key
        secret_key: 百度OCR Secret Key
        version: OCR版本，"standard"(标准版)或"high_precision"(高精度版)
        
    Returns:
        BaiduOCRInterface实例
    """
    global _baidu_ocr_instance

    with _baidu_ocr_lock:
        if (
            _baidu_ocr_instance is None
            or _baidu_ocr_instance.api_key != api_key
            or _baidu_ocr_instance.secret_key != secret_key
            or _baidu_ocr_instance.version != version
        ):
            _baidu_ocr_instance = BaiduOCRInterface(
                api_key,
                secret_key,
                version,
            )
        return _baidu_ocr_instance

def recognize_text_with_baidu_ocr(
    image_bytes: bytes,
    language: str = "auto",
    api_key: str | None = None,
    secret_key: str | None = None,
    version: str = "standard",
) -> List[str]:
    """
    使用百度OCR识别文本
    
    Args:
        image_bytes: 图像字节数据
        language: 语言代码
        api_key: 百度OCR API Key
        secret_key: 百度OCR Secret Key
        version: OCR版本，"standard"(标准版)或"high_precision"(高精度版)
        
    Returns:
        识别出的文本列表
    """
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("百度OCR缺少 API Key")
    if not isinstance(secret_key, str) or not secret_key:
        raise ValueError("百度OCR缺少 Secret Key")
    return get_baidu_ocr(api_key, secret_key, version).recognize_text(
        image_bytes,
        language,
    )
