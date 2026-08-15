import hashlib
import math
import time
import uuid

import requests


class YoudaoTranslateInterface:
    """有道翻译API接口"""

    API_URL = "https://openapi.youdao.com/api"

    def __init__(self, app_key: str, app_secret: str):
        if not isinstance(app_key, str) or not app_key.strip():
            raise ValueError("有道翻译API未配置AppKey")
        if not isinstance(app_secret, str) or not app_secret.strip():
            raise ValueError("有道翻译API未配置AppSecret")
        self.app_key = app_key
        self.app_secret = app_secret

    def translate(
        self,
        text: str,
        from_lang: str = "auto",
        to_lang: str = "zh-CHS",
        *,
        timeout: float = 30.0,
    ) -> str:
        """
        调用有道翻译API进行翻译
        
        参数:
            text: 待翻译文本
            from_lang: 源语言，默认auto自动检测
            to_lang: 目标语言，默认zh-CHS(简体中文)
        
        返回:
            翻译结果文本
        """
        if not isinstance(text, str) or not text:
            raise ValueError("有道翻译文本必须是非空字符串")
        if not isinstance(from_lang, str) or not from_lang:
            raise ValueError("有道翻译源语言必须是非空字符串")
        if not isinstance(to_lang, str) or not to_lang:
            raise ValueError("有道翻译目标语言必须是非空字符串")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("有道翻译请求超时必须大于零")

        salt = str(uuid.uuid4())
        curtime = str(int(time.time()))
        input_text = self._truncate(text)
        sign_str = self.app_key + input_text + salt + curtime + self.app_secret
        sign = hashlib.sha256(sign_str.encode("utf-8")).hexdigest()
        form = {
            "q": text,
            "from": from_lang,
            "to": to_lang,
            "appKey": self.app_key,
            "salt": salt,
            "sign": sign,
            "signType": "v3",
            "curtime": curtime,
        }
        response = requests.post(
            self.API_URL,
            data=form,
            timeout=float(timeout),
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("有道翻译API返回值必须是对象")
        error_code = result.get("errorCode")
        if error_code != "0":
            raise RuntimeError(f"有道翻译API返回错误，错误码: {error_code}")
        translated = result.get("translation")
        if (
            not isinstance(translated, list)
            or not translated
            or not isinstance(translated[0], str)
            or not translated[0].strip()
        ):
            raise RuntimeError("有道翻译API未返回有效翻译结果")
        return translated[0].strip()

    @staticmethod
    def _truncate(q: str) -> str:
        """
        按照有道API要求截取输入字符
        input = q前10个字符 + q长度 + q后10个字符（当q长度大于20）
        或 input = q字符串（当q长度小于等于20）
        """
        size = len(q)
        if size <= 20:
            return q
        return q[:10] + str(size) + q[-10:]
