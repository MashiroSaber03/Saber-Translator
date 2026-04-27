"""
翻译页非 OpenAI 兼容服务适配层。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import requests

from src.interfaces.baidu_translate_interface import BaiduTranslateInterface
from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface
from src.shared import constants

logger = logging.getLogger("SharedAIAdapters")

_baidu_translate = BaiduTranslateInterface()
_youdao_translate = YoudaoTranslateInterface()


def run_local_chat_completion(provider: str, model_name: str, messages: List[Dict[str, Any]], timeout: float = 120.0) -> str:
    provider = provider.lower()
    if provider == "ollama":
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model_name, "messages": messages, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()

    if provider == "sakura":
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": model_name, "messages": messages},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            raise ValueError("Sakura 返回空 choices")
        return choices[0]["message"]["content"].strip()

    raise ValueError(f"不支持的本地模型服务商: {provider}")


def translate_with_caiyun(text: str, target_language: str, api_key: str, model_hint: str = "") -> str:
    trans_type = "auto2zh"
    if target_language == "en":
        trans_type = "zh2en"
    elif target_language == "ja":
        trans_type = "zh2ja"
    if "japan" in model_hint or "ja" in model_hint:
        trans_type = "ja2zh"
    elif "en" in model_hint:
        trans_type = "en2zh"

    response = requests.post(
        "http://api.interpreter.caiyunai.com/v1/translator",
        headers={
            "Content-Type": "application/json",
            "X-Authorization": f"token {api_key}",
        },
        json={
            "source": [text],
            "trans_type": trans_type,
            "request_id": f"comic_translator_{int(time.time())}",
            "detect": True,
            "media": "text",
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("target"):
        return result["target"][0].strip()
    raise ValueError(f"彩云小译返回格式错误: {result}")


def translate_with_baidu(text: str, target_language: str, app_id: str, app_key: str) -> str:
    _baidu_translate.set_credentials(app_id, app_key)
    to_lang = constants.PROJECT_TO_BAIDU_TRANSLATE_LANG_MAP.get(target_language, "zh")
    return _baidu_translate.translate(text, "auto", to_lang)


def translate_with_youdao(text: str, target_language: str, app_key: str, app_secret: str) -> str:
    _youdao_translate.app_key = app_key
    _youdao_translate.app_secret = app_secret
    to_lang = constants.PROJECT_TO_YOUDAO_TRANSLATE_LANG_MAP.get(target_language, "zh-CHS")
    return _youdao_translate.translate(text, "auto", to_lang)


def fetch_local_models(provider: str) -> List[Dict[str, str]]:
    provider = provider.lower()
    if provider == "ollama":
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        return [
            {"id": model.get("name", ""), "name": model.get("name", "")}
            for model in data.get("models", [])
            if model.get("name")
        ]

    if provider == "sakura":
        response = requests.get("http://localhost:8080/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        return [
            {"id": model.get("id", ""), "name": model.get("id", "")}
            for model in data.get("data", [])
            if model.get("id")
        ]

    raise ValueError(f"不支持获取模型列表的本地服务商: {provider}")


def test_caiyun_connection(token: str) -> tuple[bool, str]:
    response = requests.post(
        "https://api.interpreter.caiyunai.com/v1/translator",
        headers={
            "Content-Type": "application/json",
            "X-Authorization": f"token {token}",
        },
        json={
            "source": ["Hello"],
            "trans_type": "en2zh",
            "request_id": "test",
            "detect": True,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    targets = data.get("target") or []
    if not targets:
        raise ValueError("未获得预期的翻译结果")
    return True, targets[0]
