"""Serve the production Vue SPA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flask import Blueprint, Response, abort, send_from_directory

from src.backend_v2.editing.layer_styles import manga_styles_path
from src.backend_v2.paths import project_root


_PHOTOCRAFT_HOST_TAG = '<script type="module" src="./host.js"></script>'
_MANGA_STYLES_SEED_TAG = '<script src="./manga-styles-seed.js"></script>'
_MANGA_STYLES_SEED_KEY = "saber.photocraft.mangaStylesSeedVersion"
_PHOTOCRAFT_PREFS_KEY = "photocraft.preferences"


def _vue_dist_root() -> Path:
    return project_root() / "src" / "backend_v2" / "static" / "vue"


def _manga_styles_path() -> Path:
    return manga_styles_path(project_root())


def _no_store_js(script: str) -> Response:
    response = Response(script, mimetype="application/javascript; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"
    return response


def _serve_manga_styles_seed_script() -> Response:
    styles_path = _manga_styles_path()
    if not styles_path.is_file():
        return _no_store_js("/* no manga styles */\n")
    try:
        raw_bytes = styles_path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
        manga_groups = data.get("mangaGroups")
        builtin_groups = data.get("builtinGroups")
        if not isinstance(manga_groups, list) or not isinstance(builtin_groups, list):
            return _no_store_js("/* invalid manga styles payload */\n")
    except (OSError, UnicodeDecodeError, ValueError):
        return _no_store_js("/* unreadable manga styles */\n")

    seed_version = f"manga-v1-{hashlib.sha256(raw_bytes).hexdigest()[:12]}"
    # Escape "</" to avoid closing tags if ever inspected inside HTML contexts.
    manga_json = json.dumps(manga_groups, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    builtin_json = json.dumps(builtin_groups, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    script = (
        "(function () {\n"
        "  try {\n"
        f"    var SEED_KEY = {json.dumps(_MANGA_STYLES_SEED_KEY)};\n"
        f"    var PREFS_KEY = {json.dumps(_PHOTOCRAFT_PREFS_KEY)};\n"
        f"    var SEED_VERSION = {json.dumps(seed_version)};\n"
        "    if (typeof localStorage === 'undefined') { return; }\n"
        "    var raw = localStorage.getItem(PREFS_KEY);\n"
        "    var prefs = null;\n"
        "    if (raw) {\n"
        "      try {\n"
        "        var parsed = JSON.parse(raw);\n"
        "        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) { prefs = parsed; }\n"
        "      } catch (_) {}\n"
        "    }\n"
        "    if (!prefs) { prefs = { version: 1 }; }\n"
        "    if (!prefs.presets || typeof prefs.presets !== 'object' || Array.isArray(prefs.presets)) {\n"
        "      prefs.presets = { swatches: null };\n"
        "    } else if (!Object.prototype.hasOwnProperty.call(prefs.presets, 'swatches')) {\n"
        "      prefs.presets.swatches = null;\n"
        "    }\n"
        "    if (localStorage.getItem(SEED_KEY) === SEED_VERSION && Array.isArray(prefs.presets.styles)) {\n"
        "      return;\n"
        "    }\n"
        f"    var mangaGroups = {manga_json};\n"
        f"    var builtinGroups = {builtin_json};\n"
        "    var baseGroups = Array.isArray(prefs.presets.styles) ? prefs.presets.styles.slice() : builtinGroups.slice();\n"
        "    var seen = new Set();\n"
        "    var existingByName = Object.create(null);\n"
        "    for (var i = 0; i < baseGroups.length; i++) {\n"
        "      var grp = baseGroups[i];\n"
        "      if (!grp || typeof grp !== 'object') { continue; }\n"
        "      if (typeof grp.name === 'string' && !existingByName[grp.name]) {\n"
        "        existingByName[grp.name] = grp;\n"
        "      }\n"
        "      if (Array.isArray(grp.items)) {\n"
        "        for (var j = 0; j < grp.items.length; j++) {\n"
        "          var item = grp.items[j];\n"
        "          if (item && typeof item.name === 'string') { seen.add(item.name); }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "    var prepended = [];\n"
        "    for (var m = 0; m < mangaGroups.length; m++) {\n"
        "      var mg = mangaGroups[m];\n"
        "      if (!mg || typeof mg.name !== 'string' || !Array.isArray(mg.items)) { continue; }\n"
        "      var added = [];\n"
        "      for (var k = 0; k < mg.items.length; k++) {\n"
        "        var preset = mg.items[k];\n"
        "        if (preset && typeof preset.name === 'string' && !seen.has(preset.name)) {\n"
        "          seen.add(preset.name);\n"
        "          added.push(preset);\n"
        "        }\n"
        "      }\n"
        "      if (!added.length) { continue; }\n"
        "      var target = existingByName[mg.name];\n"
        "      if (target) {\n"
        "        if (!Array.isArray(target.items)) { target.items = []; }\n"
        "        for (var a = 0; a < added.length; a++) { target.items.push(added[a]); }\n"
        "      } else {\n"
        "        prepended.push({ name: mg.name, items: added });\n"
        "      }\n"
        "    }\n"
        "    prefs.presets.styles = prepended.concat(baseGroups);\n"
        "    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));\n"
        "    localStorage.setItem(SEED_KEY, SEED_VERSION);\n"
        "  } catch (_) {}\n"
        "})();\n"
    )
    return _no_store_js(script)


def _serve_photocraft_index(root: Path) -> Response:
    index_path = root / "index.html"
    if not index_path.is_file():
        abort(404)
    html = index_path.read_text(encoding="utf-8")
    if _manga_styles_path().is_file() and _PHOTOCRAFT_HOST_TAG in html and _MANGA_STYLES_SEED_TAG not in html:
        html = html.replace(_PHOTOCRAFT_HOST_TAG, f"{_MANGA_STYLES_SEED_TAG}\n  {_PHOTOCRAFT_HOST_TAG}", 1)
    response = Response(html, mimetype="text/html; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"
    return response


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

    @blueprint.get("/photocraft/<path:filename>")
    def photocraft(filename: str) -> Response:
        if filename == "manga-styles-seed.js":
            return _serve_manga_styles_seed_script()
        root = project_root() / "src" / "backend_v2" / "static" / "photocraft"
        if filename == "index.html":
            return _serve_photocraft_index(root)
        response = send_from_directory(root, filename, max_age=0)
        if filename.endswith(".wasm"):
            response.mimetype = "application/wasm"
        return response

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
