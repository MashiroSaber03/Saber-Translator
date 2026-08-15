"""
翻译页非 OpenAI 兼容服务适配层。
"""

from __future__ import annotations

import uuid

import requests

from src.interfaces.baidu_translate_interface import BaiduTranslateInterface
from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface
from src.shared import constants


def translate_with_caiyun(text: str, target_language: str, api_key: str) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("彩云小译文本必须是非空字符串")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("彩云小译需要 API Key")
    try:
        trans_type = {
            "zh": "auto2zh",
            "en": "zh2en",
            "japan": "zh2ja",
        }[target_language]
    except KeyError as exc:
        raise ValueError(f"彩云小译不支持目标语言: {target_language}") from exc

    response = requests.post(
        "https://api.interpreter.caiyunai.com/v1/translator",
        headers={
            "Content-Type": "application/json",
            "X-Authorization": f"token {api_key}",
        },
        json={
            "source": [text],
            "trans_type": trans_type,
            "request_id": f"comic_translator_{uuid.uuid4()}",
            "detect": True,
            "media": "text",
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("彩云小译返回值必须是对象")
    target = result.get("target")
    if (
        not isinstance(target, list)
        or len(target) != 1
        or not isinstance(target[0], str)
        or not target[0].strip()
    ):
        raise RuntimeError("彩云小译未返回有效翻译结果")
    return target[0].strip()


def translate_with_baidu(text: str, target_language: str, app_id: str, app_key: str) -> str:
    try:
        to_lang = constants.PROJECT_TO_BAIDU_TRANSLATE_LANG_MAP[target_language]
    except KeyError as exc:
        raise ValueError(f"百度翻译不支持目标语言: {target_language}") from exc
    return BaiduTranslateInterface(app_id, app_key).translate(text, "auto", to_lang)


def translate_with_youdao(text: str, target_language: str, app_key: str, app_secret: str) -> str:
    try:
        to_lang = constants.PROJECT_TO_YOUDAO_TRANSLATE_LANG_MAP[target_language]
    except KeyError as exc:
        raise ValueError(f"有道翻译不支持目标语言: {target_language}") from exc
    return YoudaoTranslateInterface(app_key, app_secret).translate(text, "auto", to_lang)
