"""Serve the production Vue SPA."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, abort, send_from_directory

from src.backend_v2.paths import project_root


def _vue_dist_root() -> Path:
    return project_root() / "src" / "backend_v2" / "static" / "vue"


def create_web_blueprint() -> Blueprint:
    blueprint = Blueprint("web", __name__)

    def serve_index() -> Response:
        root = _vue_dist_root()
        if not (root / "index.html").is_file():
            abort(503, description="Vue production assets are not built")
        return send_from_directory(root, "index.html")

    @blueprint.get("/")
    def index() -> Response:
        return serve_index()

    @blueprint.get("/js/<path:filename>")
    def javascript(filename: str) -> Response:
        return send_from_directory(_vue_dist_root() / "js", filename)

    @blueprint.get("/assets/<path:filename>")
    def assets(filename: str) -> Response:
        return send_from_directory(_vue_dist_root() / "assets", filename)

    @blueprint.get("/<path:path>")
    def spa_fallback(path: str) -> Response:
        if path == "api" or path.startswith("api/"):
            abort(404)
        root = _vue_dist_root()
        candidate = root / path
        if candidate.is_file():
            return send_from_directory(root, path)
        return serve_index()

    return blueprint
