"""Validation and defaults shared by saved settings and native diagnostics."""

from urllib.parse import urlsplit

DOMAIN = "font_matching"
ONLINE_PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "gpt-6-luna"),
    "jev": ("https://api.typesafe.ai/v1", "jev-1.14.0"),
}
VALEN_IDS = {"Valen-0.8B": "valen_08b", "Valen-2B": "valen_2b"}
PROVIDERS = frozenset((*ONLINE_PROVIDERS, "valen"))


def model_supported(provider: str, model: str) -> bool:
    if provider == "valen":
        return model in VALEN_IDS
    if provider == "openai":
        return model == "gpt-6-luna" or model.startswith("gpt-6-luna-")
    if provider == "jev":
        return model == "jev-latest" or model.startswith(("jev-1.13.", "jev-1.14."))
    return False


def endpoint(provider: str, value: str) -> str:
    base = value.strip().rstrip("/") or ONLINE_PROVIDERS[provider][0]
    try:
        url = urlsplit(base)
        port = url.port
        if (url.scheme not in {"http", "https"} or not url.hostname or
                url.username is not None or url.password is not None or url.query or url.fragment
                or (port is not None and port <= 0)):
            raise ValueError
    except ValueError:
        raise ValueError("字体匹配服务地址须为不含凭据、查询参数或片段的 HTTP(S) API 基础地址") from None
    return base if url.path else base + "/v1"
