"""Local token-gated HTTP facade used by the Manifest V3 extension."""

from __future__ import annotations

from typing import Any, Protocol
import ipaddress
import logging

from flask import Blueprint, Response, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from src.backend_v2.api.request_helpers import (
    json_body,
    integer_value,
    required_boolean,
    required_string,
    validate_multipart_fields,
)
from src.backend_v2.browser_extension.auth import BrowserExtensionAccess
from src.backend_v2.browser_extension.dom_agent import BrowserDomAgentUnavailable
from src.backend_v2.browser_extension.service import (
    BrowserSessionConflict,
    BrowserSessionNotFound,
    BrowserSessionService,
)
from src.backend_v2.content.image_import import UnsupportedImage
from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.storage.assets import AssetQuotaExceeded


LOGGER = logging.getLogger("BrowserExtensionApi")


class DomDetector(Protocol):
    def detect(self, payload: dict[str, Any]) -> dict[str, object]: ...


def _bearer_token() -> str:
    value = request.headers.get("Authorization", "")
    prefix = "Bearer "
    return value[len(prefix) :] if value.startswith(prefix) else ""


def _is_loopback(value: str | None) -> bool:
    if not value:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value.casefold() == "localhost"


def _browser_error(
    code: str,
    message: str,
    status: int,
    *,
    retryable: bool,
):
    return jsonify(
        {
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            }
        }
    ), status


def create_browser_extension_blueprint(
    *,
    service: BrowserSessionService,
    access: BrowserExtensionAccess,
    dom_detector: DomDetector | None = None,
) -> Blueprint:
    blueprint = Blueprint(
        "browser_extension_v2",
        __name__,
        url_prefix="/api/v2/browser-extension",
    )

    @blueprint.before_request
    def authorize_request():
        if request.endpoint == "browser_extension_v2.get_result":
            if not _is_loopback(request.remote_addr):
                return _browser_error(
                    "loopback_required",
                    "browser extension results only accept loopback clients",
                    403,
                    retryable=False,
                )
            return None
        if not access.enabled:
            return _browser_error(
                "integration_disabled",
                "browser extension integration is disabled",
                503,
                retryable=True,
            )
        if not _is_loopback(request.remote_addr):
            return _browser_error(
                "loopback_required",
                "browser extension API only accepts loopback clients",
                403,
                retryable=False,
            )
        if request.method == "OPTIONS":
            return Response(status=204)
        if not access.valid_token(_bearer_token()):
            return _browser_error(
                "invalid_extension_token",
                "browser extension token is invalid",
                401,
                retryable=False,
            )
        return None

    @blueprint.after_request
    def add_extension_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/v2/browser-extension"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.errorhandler(BrowserSessionNotFound)
    def session_not_found(error: BrowserSessionNotFound):
        return _browser_error(
            "not_found",
            str(error),
            404,
            retryable=False,
        )

    @blueprint.errorhandler(BrowserSessionConflict)
    @blueprint.errorhandler(JobConflict)
    def session_conflict(error: BaseException):
        return _browser_error(
            "session_conflict",
            str(error),
            409,
            retryable=True,
        )

    @blueprint.errorhandler(UnsupportedImage)
    def unsupported_image(error: UnsupportedImage):
        return _browser_error(
            "unsupported_image",
            str(error),
            422,
            retryable=False,
        )

    @blueprint.errorhandler(AssetQuotaExceeded)
    def asset_quota_exceeded(error: AssetQuotaExceeded):
        return _browser_error(
            "asset_quota_exceeded",
            str(error),
            413,
            retryable=False,
        )

    @blueprint.errorhandler(BrowserDomAgentUnavailable)
    def dom_agent_unavailable(error: BrowserDomAgentUnavailable):
        return _browser_error(
            "dom_agent_unavailable",
            str(error),
            503,
            retryable=False,
        )

    @blueprint.errorhandler(ValueError)
    def validation_error(error: ValueError):
        return _browser_error(
            "validation_error",
            str(error),
            422,
            retryable=False,
        )

    @blueprint.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        status = int(error.code or 500)
        return _browser_error(
            "browser_http_error",
            str(error.description),
            status,
            retryable=status >= 500,
        )

    @blueprint.errorhandler(Exception)
    def unexpected_error(error: Exception):
        LOGGER.exception(
            "浏览器扩展接口发生未处理错误：%s",
            error,
        )
        return _browser_error(
            "browser_internal_error",
            "browser extension request failed",
            500,
            retryable=True,
        )

    @blueprint.get("/status")
    def status() -> Response:
        return jsonify({"status": "ready"})

    @blueprint.get("/library-books")
    def library_books() -> Response:
        return jsonify({"items": service.library_books()})

    @blueprint.post("/sessions")
    def create_session():
        body = json_body(
            allowed_keys={
                "pageUrl",
                "pageTitle",
                "mode",
                "glossaryEnabled",
                "autoTermsEnabled",
            }
        )
        return jsonify(
            service.create(
                page_url=required_string(body, "pageUrl"),
                page_title=required_string(body, "pageTitle"),
                mode=required_string(body, "mode"),
                glossary_enabled=required_boolean(body, "glossaryEnabled"),
                auto_terms_enabled=required_boolean(body, "autoTermsEnabled"),
            )
        ), 201

    @blueprint.get("/sessions/<session_id>")
    def get_session(session_id: str) -> Response:
        touch = request.args.get("touch", "false")
        if touch not in {"true", "false"}:
            raise ValueError("touch must be true or false")
        return jsonify(service.get(session_id, touch=touch == "true"))

    @blueprint.patch("/sessions/<session_id>")
    def patch_session(session_id: str) -> Response:
        body = json_body(
            allowed_keys={
                "mode",
                "glossaryEnabled",
                "autoTermsEnabled",
            }
        )
        if not body:
            raise ValueError("at least one session setting is required")
        return jsonify(
            service.update(
                session_id,
                mode=(required_string(body, "mode") if "mode" in body else None),
                glossary_enabled=(
                    required_boolean(body, "glossaryEnabled")
                    if "glossaryEnabled" in body
                    else None
                ),
                auto_terms_enabled=(
                    required_boolean(body, "autoTermsEnabled")
                    if "autoTermsEnabled" in body
                    else None
                ),
            )
        )

    @blueprint.post("/sessions/<session_id>/start")
    def start_session(session_id: str) -> Response:
        return jsonify(service.start(session_id)), 202

    @blueprint.post("/sessions/<session_id>/pages")
    def upload_page(session_id: str):
        validate_multipart_fields(
            allowed_form_keys={
                "clientPageKey",
                "ordinal",
                "logicalPath",
                "sourceUrl",
            },
            allowed_file_keys={"file"},
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        client_page_key = str(request.form.get("clientPageKey", "")).strip()
        logical_path = str(request.form.get("logicalPath", "")).strip()
        source_url_value = request.form.get("sourceUrl")
        result = service.add_page(
            session_id=session_id,
            client_page_key=client_page_key,
            ordinal=integer_value(
                request.form.get("ordinal"),
                "ordinal",
                minimum=1,
                maximum=1_000_000,
            ),
            logical_path=logical_path,
            source_url=(str(source_url_value) if source_url_value else None),
            upload=upload.stream,
        )
        return jsonify(result), 201

    @blueprint.post("/sessions/<session_id>/pages/<browser_page_id>/retry")
    def retry_page(session_id: str, browser_page_id: str) -> Response:
        return jsonify(
            service.retry(
                session_id=session_id,
                browser_page_id=browser_page_id,
            )
        ), 202

    @blueprint.post(
        "/sessions/<session_id>/pages/<browser_page_id>/result-capability"
    )
    def result_capability(session_id: str, browser_page_id: str) -> Response:
        asset = service.translated_asset(
            session_id=session_id,
            browser_page_id=browser_page_id,
        )
        expires_at, signature = access.sign_result(
            session_id=session_id,
            browser_page_id=browser_page_id,
            asset_id=asset.id,
        )
        result_url = (
            f"/api/v2/browser-extension/results/{asset.id}"
            f"?session={session_id}&page={browser_page_id}"
            f"&expires={expires_at}&signature={signature}"
        )
        return jsonify(
            {
                "url": result_url,
                "assetId": asset.id,
                "expiresAt": expires_at,
            }
        )

    @blueprint.get("/sessions/<session_id>/terms")
    def get_terms(session_id: str) -> Response:
        return jsonify(service.terms(session_id))

    @blueprint.post("/sessions/<session_id>/cancel")
    def cancel_session(session_id: str) -> Response:
        return jsonify(service.cancel(session_id))

    @blueprint.post("/sessions/<session_id>/import")
    def import_session(session_id: str) -> Response:
        body = json_body(
            allowed_keys={
                "destination",
                "bookTitle",
                "targetBookId",
                "chapterTitle",
            }
        )
        return jsonify(
            service.import_to_library(
                session_id,
                destination=required_string(body, "destination"),
                book_title=(
                    required_string(body, "bookTitle")
                    if "bookTitle" in body
                    else None
                ),
                target_book_id=(
                    required_string(body, "targetBookId")
                    if "targetBookId" in body
                    else None
                ),
                chapter_title=required_string(body, "chapterTitle"),
            )
        )

    @blueprint.post("/dom-detection")
    def detect_dom() -> Response:
        if dom_detector is None:
            return _browser_error(
                "dom_agent_unavailable",
                "Browser DOM Agent is not configured",
                503,
                retryable=False,
            )
        body = json_body(allowed_keys={"pageUrl", "pageTitle", "nodes"})
        try:
            result = dom_detector.detect(body)
        except (ValueError, BrowserDomAgentUnavailable):
            raise
        except Exception as error:
            LOGGER.warning("Browser DOM Agent 调用失败", exc_info=True)
            return _browser_error(
                "dom_agent_failed",
                str(error) or "Browser DOM Agent request failed",
                503,
                retryable=True,
            )
        return jsonify(result)

    @blueprint.get("/results/<asset_id>")
    def get_result(asset_id: str):
        session_id = request.args.get("session", "")
        browser_page_id = request.args.get("page", "")
        signature = request.args.get("signature", "")
        try:
            expires_at = int(request.args.get("expires", "0"))
        except ValueError:
            expires_at = 0
        if not access.verify_result(
            session_id=session_id,
            browser_page_id=browser_page_id,
            asset_id=asset_id,
            expires_at=expires_at,
            signature=signature,
        ):
            return _browser_error(
                "invalid_result_capability",
                "result capability is invalid or expired",
                403,
                retryable=True,
            )
        try:
            asset = service.validate_result_binding(
                session_id=session_id,
                browser_page_id=browser_page_id,
                asset_id=asset_id,
            )
        except BrowserSessionNotFound as error:
            return _browser_error(
                "result_not_found",
                str(error),
                404,
                retryable=True,
            )
        response = send_file(
            service.storage.resolve_relative_path(asset.relative_path),
            mimetype=asset.mime_type,
            conditional=True,
            etag=asset.checksum,
        )
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

    return blueprint
