import requests
import base64
import json
import logging
import time
from typing import List, Dict, Tuple, Optional, Any

# 配置日志
logger = logging.getLogger(__name__)

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
        "japan": "JAP",      # 兼容"japan"的语言代码
        "korean": "KOR",     # 韩语（必须大写）
        "chinese": "CHN_ENG", # 中文和英文
        "english": "ENG",    # 英文
        "en": "ENG",         # 兼容"en"的语言代码
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
        self.api_key = api_key
        self.secret_key = secret_key
        self.version = version
        self.access_token = None
        self.last_request_time = 0  # 上次请求时间戳
        
    def _get_access_token(self) -> Optional[str]:
        """获取百度API访问令牌"""
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        
        try:
            response = requests.post(url)
            result = response.json()
            if 'access_token' in result:
                return result['access_token']
            else:
                logger.error(f"获取百度访问令牌失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取百度访问令牌时出错: {str(e)}")
            return None
    
    def _ensure_request_interval(self, min_interval_ms: int = 500):
        """
        确保请求之间有足够的时间间隔，防止触发QPS限制
        
        Args:
            min_interval_ms: 最小请求间隔(毫秒)
        """
        current_time = time.time() * 1000  # 转换为毫秒
        elapsed = current_time - self.last_request_time
        
        if elapsed < min_interval_ms:
            # 计算需要等待的时间
            sleep_time = (min_interval_ms - elapsed) / 1000  # 转回秒
            logger.info(f"强制请求延迟 {sleep_time:.2f}s 以避免QPS限制")
            time.sleep(sleep_time)
            
        # 更新上次请求时间
        self.last_request_time = time.time() * 1000
    
    def recognize_text(self, image_bytes: bytes, language: str = "auto") -> List[str]:
        """
        识别图像中的文本
        
        Args:
            image_bytes: 图像字节数据
            language: 语言代码 (japanese, chinese, english, 等)
            
        Returns:
            识别出的文本列表
        """
        # 确保我们有访问令牌
        if not self.access_token:
            self.access_token = self._get_access_token()
            if not self.access_token:
                return []
        
        # 准备API端点
        endpoint = self.API_ENDPOINTS.get(self.version, self.API_ENDPOINTS["standard"])
        
        # 准备请求参数
        params = {
            "access_token": self.access_token
        }
        
        # 准备请求数据
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data = {
            "image": image_base64
        }
        
        # 添加语言参数（如果不是自动）
        if language != "auto" and language in self.LANGUAGE_MAPPING:
            mapped_language = self.LANGUAGE_MAPPING[language]
            # 百度OCR要求语言代码必须大写
            data["language_type"] = mapped_language.upper()
            logger.info(f"设置百度OCR语言类型为: {mapped_language} (原始语言代码: {language})")
        else:
            # 如果是auto或未知语言，不设置language_type参数，让API自动检测
            logger.info("未指定语言或使用自动检测，不设置language_type参数")
        
        # 确保请求间隔
        self._ensure_request_interval()
        
        # 发送请求
        max_retries = 3
        retry_delay = 1.0  # 初始重试延迟(秒)
        
        for retry in range(max_retries):
            try:
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                logger.info(f"发送百度OCR请求 (尝试 {retry+1}/{max_retries})")
                # 不记录完整请求参数，仅记录端点和是否有语言设置
                logger.info(f"请求端点: {endpoint.split('/')[-1]}, 语言设置: {'有' if 'language_type' in data else '无'}")
                
                response = requests.post(endpoint, params=params, data=data, headers=headers)
                result = response.json()
                
                if 'error_code' in result:
                    error_code = result.get('error_code')
                    error_msg = result.get('error_msg', '未知错误')
                    logger.error(f"百度OCR API错误: {result}")
                    
                    # 处理不同类型的错误
                    if error_code in [110, 111]:  # 令牌过期
                        logger.info("访问令牌过期，重新获取...")
                        self.access_token = self._get_access_token()
                        if self.access_token:
                            params["access_token"] = self.access_token
                            continue  # 使用新令牌重试
                        else:
                            return []
                            
                    elif error_code == 18:  # QPS限制
                        if retry < max_retries - 1:
                            wait_time = retry_delay * (retry + 1)  # 指数退避
                            logger.info(f"触发QPS限制，等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                            continue  # 等待后重试
                        else:
                            logger.error("达到最大重试次数，QPS限制仍然存在")
                            return []
                    
                    elif error_code == 216100:  # 语言参数错误
                        logger.warning("语言类型参数无效，尝试移除语言参数使用自动检测...")
                        # 移除语言参数
                        if 'language_type' in data:
                            del data['language_type']
                            continue  # 重试，但不设置语言参数
                        else:
                            logger.error("即使不设置语言参数也出错")
                            return []
                    else:
                        # 其他错误直接返回空结果
                        logger.error(f"未处理的百度OCR错误: {error_code} - {error_msg}")
                        return []
                
                # 提取识别文本
                text_results = []
                if 'words_result' in result:
                    for item in result['words_result']:
                        if 'words' in item:
                            text_results.append(item['words'])
                
                logger.info(f"百度OCR识别成功，返回 {len(text_results)} 个文本结果")
                return text_results
            
            except Exception as e:
                logger.error(f"百度OCR识别时出错: {str(e)}")
                if retry < max_retries - 1:
                    wait_time = retry_delay * (retry + 1)
                    logger.info(f"将在 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    return []
        
        return []  # 所有重试都失败

# 单例实例
_baidu_ocr_instance = None

def get_baidu_ocr(api_key: str = None, secret_key: str = None, version: str = "standard") -> Optional[BaiduOCRInterface]:
    """
    获取百度OCR实例（单例模式）
    
    Args:
        api_key: 百度OCR API Key
        secret_key: 百度OCR Secret Key
        version: OCR版本，"standard"(标准版)或"high_precision"(高精度版)
        
    Returns:
        BaiduOCRInterface实例或None（如果创建失败）
    """
    global _baidu_ocr_instance
    
    # 如果实例不存在或者参数改变，创建新实例
    if (api_key and secret_key) and (_baidu_ocr_instance is None or 
                                    _baidu_ocr_instance.api_key != api_key or
                                    _baidu_ocr_instance.secret_key != secret_key or
                                    _baidu_ocr_instance.version != version):
        try:
            _baidu_ocr_instance = BaiduOCRInterface(api_key, secret_key, version)
            # 检查令牌是否可以获取（作为连接测试）
            if _baidu_ocr_instance._get_access_token() is None:
                logger.error("无法获取百度OCR访问令牌，请检查API密钥")
                _baidu_ocr_instance = None
        except Exception as e:
            logger.error(f"初始化百度OCR时出错: {str(e)}")
            _baidu_ocr_instance = None
    
    return _baidu_ocr_instance

def recognize_text_with_baidu_ocr(image_bytes: bytes, language: str = "auto", api_key: str = None, 
                               secret_key: str = None, version: str = "standard") -> List[str]:
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
    ocr = get_baidu_ocr(api_key, secret_key, version)
    if ocr:
        return ocr.recognize_text(image_bytes, language)
    return []

def test_baidu_ocr_connection(api_key: str, secret_key: str) -> Dict[str, Any]:
    """
    测试百度OCR连接
    
    Args:
        api_key: 百度OCR API Key
        secret_key: 百度OCR Secret Key
        
    Returns:
        测试结果字典 {"success": bool, "message": str}
    """
    try:
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        response = requests.post(url)
        result = response.json()
        
        if 'access_token' in result:
            return {"success": True, "message": "百度OCR连接测试成功！"}
        else:
            error_msg = result.get('error_description', '未知错误')
            return {"success": False, "message": f"连接失败: {error_msg}"}
    except Exception as e:
        return {"success": False, "message": f"连接测试出错: {str(e)}"} 