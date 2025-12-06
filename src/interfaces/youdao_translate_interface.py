import requests
import hashlib
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class YoudaoTranslateInterface:
    """有道翻译API接口"""
    
    def __init__(self, app_key="", app_secret=""):
        self.app_key = app_key
        self.app_secret = app_secret
        self.api_url = "https://openapi.youdao.com/api"
        
    def translate(self, text, from_lang="auto", to_lang="zh-CHS"):
        """
        调用有道翻译API进行翻译
        
        参数:
            text: 待翻译文本
            from_lang: 源语言，默认auto自动检测
            to_lang: 目标语言，默认zh-CHS(简体中文)
        
        返回:
            翻译结果文本，如果出错则返回原文
        """
        if not self.app_key or not self.app_secret:
            logger.error("有道翻译API密钥未设置")
            return text
            
        try:
            # 准备请求参数
            salt = str(uuid.uuid1())
            curtime = str(int(time.time()))
            
            # 计算input参数
            input_text = self._truncate(text)
            
            # 计算签名
            sign_str = self.app_key + input_text + salt + curtime + self.app_secret
            sign = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()
            
            # 构建请求参数
            params = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appKey': self.app_key,
                'salt': salt,
                'sign': sign,
                'signType': 'v3',
                'curtime': curtime
            }
            
            # 发送请求
            response = requests.post(self.api_url, params=params)
            result = response.json()
            
            # 处理结果
            if 'translation' in result and result['translation']:
                return result['translation'][0]
            else:
                error_code = result.get('errorCode', 'unknown')
                logger.error(f"有道翻译API返回错误，错误码: {error_code}")
                return text
                
        except Exception as e:
            logger.error(f"有道翻译API调用异常: {str(e)}")
            return text
    
    def _truncate(self, q):
        """
        按照有道API要求截取输入字符
        input = q前10个字符 + q长度 + q后10个字符（当q长度大于20）
        或 input = q字符串（当q长度小于等于20）
        """
        if q is None:
            return None
            
        size = len(q)
        if size <= 20:
            return q
        else:
            return q[:10] + str(size) + q[-10:] 