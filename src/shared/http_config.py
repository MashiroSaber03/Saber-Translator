"""
AI 服务访问统一网络策略

集中维护 User-Agent 伪装、超时、代理策略；供同步/异步两套客户端复用，
一处修改全局生效。

项目约定：
- 所有访问外部 AI 服务的 HTTP 客户端必须通过本模块构造，
  或通过 src.shared.openai_helpers.create_openai_client() 创建。
- 禁止直接 `from openai import OpenAI` 实例化 OpenAI 客户端。
- 禁止裸 `httpx.Client(...)` / `httpx.AsyncClient(...)` 访问 AI 服务。
"""
from __future__ import annotations

import ipaddress
import os
import socket
from typing import Optional
from urllib.parse import urlsplit

# Chrome UA，用于绕过套 CF 的中转站 WAF 基础 UA 黑名单
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

BROWSER_HEADERS = {"User-Agent": BROWSER_USER_AGENT}


def public_outbound_policy_enabled() -> bool:
    return os.environ.get("SABER_V2_PROFILE", "local").lower() == "public"


def validate_public_outbound_url(url: str) -> None:
    """Reject URLs that could make a public deployment reach a private network."""

    if not public_outbound_policy_enabled():
        return
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError("Base URL 格式无效") from exc
    if parsed.scheme != "https":
        raise ValueError("公开模式只允许 HTTPS Base URL")
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Base URL 不允许包含凭据或片段")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
        }
    except OSError as exc:
        raise ValueError("Base URL 域名无法解析") from exc
    if not addresses:
        raise ValueError("Base URL 域名无法解析")
    for address in addresses:
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError("Base URL 解析结果无效") from exc
        if not resolved.is_global:
            raise ValueError("公开模式禁止访问内网、回环或保留地址")


def is_local_service(base_url: Optional[str]) -> bool:
    """检测是否为本地服务（localhost / 127.0.0.1 / 0.0.0.0 / ::1）。"""
    if not base_url:
        return False
    if not isinstance(base_url, str):
        raise TypeError("base_url 必须是字符串或 null")
    try:
        hostname = urlsplit(base_url).hostname
    except ValueError:
        return False
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def build_httpx_kwargs(base_url: Optional[str], timeout) -> dict:
    """
    构造 httpx.Client / httpx.AsyncClient 的统一 kwargs。

    - 本地服务：trust_env=False，避免系统代理干扰本机回环访问
    - 远程服务：trust_env=True，允许用户系统代理生效
    - 全部注入浏览器 UA，用于绕过基础 WAF UA 黑名单
    """
    if base_url:
        validate_public_outbound_url(base_url)
    return {
        "timeout": timeout,
        "trust_env": not is_local_service(base_url),
        "headers": dict(BROWSER_HEADERS),
        "follow_redirects": False,
    }
