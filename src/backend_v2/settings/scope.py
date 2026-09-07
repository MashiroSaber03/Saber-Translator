"""Storage isolation for the extension's single configuration."""

from copy import deepcopy
from typing import Any

from sqlalchemy.sql.elements import ColumnElement

from src.backend_v2.storage.defaults import (
    DEFAULT_TEXT_STYLE,
    TEXT_STYLE_DEFAULTS_SCHEMA_VERSION,
    default_translation_settings,
)

EXTENSION_PREFIX = "browser_extension:"
EXTENSION_DOMAINS = frozenset(
    {"text_style_defaults", "browser_dom_agent"}
)


class SettingsScope:
    def __init__(self, browser_extension: bool = False) -> None:
        self.browser_extension = browser_extension
        self.prefix = EXTENSION_PREFIX if browser_extension else ""

    def storage_domain(self, domain: str) -> str:
        if domain.startswith(EXTENSION_PREFIX) or (
            self.browser_extension and domain not in EXTENSION_DOMAINS
        ):
            raise ValueError(f"unsupported settings domain: {domain}")
        return self.prefix + domain

    def public_domain(self, domain: str) -> str:
        return domain.removeprefix(self.prefix) if self.prefix else domain

    def condition(self, column: ColumnElement[str]) -> ColumnElement[bool]:
        extension = column.startswith(EXTENSION_PREFIX, autoescape=True)
        return column.in_([self.prefix + domain for domain in EXTENSION_DOMAINS]) if self.browser_extension else ~extension

    def add_factory_defaults(
        self, document: dict[str, Any], domains: tuple[str, ...]
    ) -> None:
        if not self.browser_extension:
            return
        existing = {row["domain"] for row in document["settings"]}
        for domain, payload, version in (
            ("browser_dom_agent", default_translation_settings()["browserDomAgent"], 1),
            ("text_style_defaults", DEFAULT_TEXT_STYLE, TEXT_STYLE_DEFAULTS_SCHEMA_VERSION),
        ):
            if domain not in existing and (not domains or domain in domains):
                document["settings"].append(
                    {
                        "domain": domain,
                        "payload": deepcopy(payload),
                        "revision": 0,
                        "schemaVersion": version,
                    }
                )
