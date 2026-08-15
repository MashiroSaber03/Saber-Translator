import math
import random
from hashlib import md5

import requests


class BaiduTranslateInterface:
    """百度翻译API接口封装"""

    API_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    def __init__(self, app_id: str, app_key: str):
        if not isinstance(app_id, str) or not app_id.strip():
            raise ValueError("百度翻译API未配置appid，请在设置中配置")
        if not isinstance(app_key, str) or not app_key.strip():
            raise ValueError("百度翻译API未配置appkey，请在设置中配置")
        self.app_id = app_id
        self.app_key = app_key

    def translate(
        self,
        text: str,
        from_lang: str = "auto",
        to_lang: str = "zh",
        *,
        timeout: float = 30.0,
    ) -> str:
        """
        调用百度翻译API翻译文本
        
        参数:
            text (str): 要翻译的文本
            from_lang (str): 源语言，默认为'auto'自动检测
            to_lang (str): 目标语言，默认为'zh'中文
            timeout (float): 单次网络请求超时（秒）
            
        返回:
            str: 翻译后的文本
        """
        if not isinstance(text, str) or not text:
            raise ValueError("百度翻译文本必须是非空字符串")
        if not isinstance(from_lang, str) or not from_lang:
            raise ValueError("百度翻译源语言必须是非空字符串")
        if not isinstance(to_lang, str) or not to_lang:
            raise ValueError("百度翻译目标语言必须是非空字符串")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("百度翻译请求超时必须大于零")

        # 生成签名
        salt = random.randint(32768, 65536)
        sign = self._make_md5(self.app_id + text + str(salt) + self.app_key)
        
        # 构建请求参数
        form = {
            "appid": self.app_id,
            "q": text,
            "from": from_lang,
            "to": to_lang,
            "salt": salt,
            "sign": sign,
        }

        response = requests.post(
            self.API_URL,
            data=form,
            timeout=float(timeout),
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("百度翻译API返回值必须是对象")

        if "error_code" in result:
            error_code = result["error_code"]
            error_msg = result.get("error_msg", "未知错误")
            raise RuntimeError(
                f"百度翻译API错误 (错误码: {error_code}): {error_msg}"
            )

        translated = result.get("trans_result")
        if not isinstance(translated, list) or not translated:
            raise RuntimeError("百度翻译API未返回翻译结果")
        translated_texts: list[str] = []
        for item in translated:
            if not isinstance(item, dict) or not isinstance(item.get("dst"), str):
                raise RuntimeError("百度翻译API返回的翻译条目格式错误")
            value = item["dst"].strip()
            if not value:
                raise RuntimeError("百度翻译API返回空翻译结果")
            translated_texts.append(value)
        return "\n".join(translated_texts)

    @staticmethod
    def _make_md5(value: str, encoding: str = "utf-8") -> str:
        """生成MD5签名"""
        return md5(value.encode(encoding)).hexdigest()
