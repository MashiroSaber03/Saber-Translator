import requests
import random
import json
from hashlib import md5
import logging
from time import sleep

logger = logging.getLogger(__name__)

class BaiduTranslateInterface:
    """百度翻译API接口封装"""
    
    API_URL = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    
    def __init__(self, app_id=None, app_key=None):
        self.app_id = app_id
        self.app_key = app_key
        
    def set_credentials(self, app_id, app_key):
        """设置百度翻译API的appid和appkey"""
        self.app_id = app_id
        self.app_key = app_key
        
    def translate(self, text, from_lang='auto', to_lang='zh', max_retries=3, retry_delay=1):
        """
        调用百度翻译API翻译文本
        
        参数:
            text (str): 要翻译的文本
            from_lang (str): 源语言，默认为'auto'自动检测
            to_lang (str): 目标语言，默认为'zh'中文
            max_retries (int): 最大重试次数
            retry_delay (float): 重试延迟时间（秒）
            
        返回:
            str: 翻译后的文本
        """
        if not self.app_id or not self.app_key:
            raise ValueError("百度翻译API未配置appid和appkey，请在设置中配置")
        
        # 生成签名
        salt = random.randint(32768, 65536)
        sign = self._make_md5(self.app_id + text + str(salt) + self.app_key)
        
        # 构建请求参数
        params = {
            'appid': self.app_id,
            'q': text,
            'from': from_lang,
            'to': to_lang,
            'salt': salt,
            'sign': sign
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        # 发送请求并处理重试
        for attempt in range(max_retries):
            try:
                response = requests.post(self.API_URL, params=params, headers=headers)
                result = response.json()
                
                # 检查返回的错误码
                if 'error_code' in result:
                    error_code = result['error_code']
                    error_msg = result.get('error_msg', '未知错误')
                    logger.error(f"百度翻译API错误 (错误码: {error_code}): {error_msg}")
                    
                    # 处理特定错误码
                    if error_code == '52003':  # 未授权用户
                        raise ValueError("百度翻译API认证失败，请检查appid和appkey是否正确")
                    elif error_code == '54003':  # 访问频率受限
                        if attempt < max_retries - 1:
                            logger.warning(f"百度翻译API访问频率受限，{retry_delay}秒后重试")
                            sleep(retry_delay)
                            continue
                    
                    raise Exception(f"百度翻译API错误 (错误码: {error_code}): {error_msg}")
                
                # 正常情况下，提取翻译结果
                if 'trans_result' in result:
                    translated_texts = [item['dst'] for item in result['trans_result']]
                    return '\n'.join(translated_texts)
                else:
                    return ""
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"百度翻译API请求异常: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"尝试重新连接百度翻译API ({attempt + 1}/{max_retries})")
                    sleep(retry_delay)
                else:
                    raise Exception(f"百度翻译API连接失败: {str(e)}")
        
        # 如果所有重试都失败
        raise Exception("百度翻译API请求失败，已达到最大重试次数")
    
    def test_connection(self):
        """测试百度翻译API连接"""
        if not self.app_id or not self.app_key:
            return False, "未配置百度翻译API的appid和appkey"
        
        try:
            result = self.translate("Hello world", from_lang="en", to_lang="zh")
            return True, "百度翻译API连接成功"
        except Exception as e:
            return False, f"百度翻译API连接失败: {str(e)}"
    
    @staticmethod
    def _make_md5(s, encoding='utf-8'):
        """生成MD5签名"""
        return md5(s.encode(encoding)).hexdigest()

# 语言代码映射，将程序内部语言代码映射到百度API支持的语言代码
LANGUAGE_CODE_MAP = {
    'zh': 'zh',       # 中文
    'en': 'en',       # 英语
    'ja': 'jp',       # 日语 (百度API使用jp)
    'ko': 'kor',      # 韩语
    'fr': 'fra',      # 法语
    'es': 'spa',      # 西班牙语
    'it': 'it',       # 意大利语
    'de': 'de',       # 德语
    'ru': 'ru',       # 俄语
    'pt': 'pt',       # 葡萄牙语
    'vi': 'vie',      # 越南语
    'th': 'th',       # 泰语
    'auto': 'auto',   # 自动检测
}

# 百度翻译API单例
baidu_translate = BaiduTranslateInterface() 