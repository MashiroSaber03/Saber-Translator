"""Small, filesystem-only adapter to the pinned PhotoCraft Rust engine.

HTTP callers never provide engine commands or filesystem paths. Assets and fonts
are resolved by the existing storage/ownership services before invoking Rust.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import base64
import hashlib
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from io import BytesIO
from zipfile import ZipFile

from sqlalchemy import insert, or_, select, update

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.rendering.fonts import resolve_font_path
from src.backend_v2.serialization import canonical_json
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import assets, bubbles, fonts, page_assets, pages
from src.backend_v2.timestamps import utcnow
from src.core.config_models import BubbleState, validate_bubble_payload
from src.shared.path_helpers import resource_path


# Native layers own these fields. Everything else (OCR, source text, repair
# preferences, detection geometry) remains database business data.
NATIVE_BUBBLE_FIELDS = frozenset({
    "translatedText", "coords", "position", "fontSize", "textColor", "textDirection",
    "rotationAngle", "inlineAlign", "blockAlign", "lineSpacing", "strokeEnabled",
    "strokeColor", "strokeWidth", "richText",
})


@dataclass
class EditingResult:
    document: bytes | None
    bubbles: list[dict]
    width: int
    height: int
    translated: bytes | None = None
    clean: bytes | None = None
    text: bytes | None = None
    user_mask: bytes | None = None


class NativeEditor:
    def __init__(self, storage: AssetStorageService):
        self.storage = storage

    def font_files(self, connection):
        return [{"id": row.id, "path": resolve_font_path(connection, self.storage, row.id)}
                for row in connection.execute(select(fonts.c.id).where(or_(
                    fonts.c.owner_user_id.is_(None), fonts.c.owner_user_id == effective_owner_id(),
                )))]

    @staticmethod
    def _legacy_document(payload: bytes) -> str | None:
        """Return the migration needed; only old text metadata needs the stored text."""
        with ZipFile(BytesIO(payload)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        text = dict(manifest["document"].get("metadata", {}).get("text", []))
        metadata = json.loads(text.get("saber.page", "{}"))
        if any("translatedText" in entry.get("payload", {}) or "richText" in entry.get("payload", {})
               for entry in metadata.get("bubbles", {}).values()):
            return "text"
        layers = list(manifest["document"].get("layers", []))
        while layers:
            layer = layers.pop()
            if layer["id"] == metadata.get("background"):
                if layer["content"]["kind"] != "raster":
                    return "background"
                break
            layers.extend(layer.get("content", {}).get("children", []))
        return None

    @staticmethod
    def page_bubbles(connection, page_id):
        return [{"bubbleId": row.id, "fontId": row.font_id, "payload": json.loads(row.payload_json)}
                for row in connection.execute(select(bubbles).where(
                    bubbles.c.page_id == page_id).order_by(bubbles.c.ordinal))]

    def read_document(self, connection, page_id):
        path = connection.execute(select(assets.c.relative_path).join(page_assets,
            page_assets.c.asset_id == assets.c.id).where(
                page_assets.c.page_id == page_id, page_assets.c.role == "editor_document")).scalar_one_or_none()
        if path is not None:
            payload = self.storage.resolve_relative_path(path).read_bytes()
            if not self._legacy_document(payload):
                return payload
        # Read-only compatibility: old published images are never changed here.
        return self.prepare(connection, page_id).document

    def prepare_commit(self, connection, page_id, revision, documents, *, fields=None, **producer):
        """Prepare under a read snapshot; return the existing transaction's publisher."""
        def native_values(items):
            return [(item["bubbleId"], item["fontId"],
                     {key: item["payload"].get(key) for key in NATIVE_BUBBLE_FIELDS}) for item in items]

        explicit = any(set(keys).intersection(NATIVE_BUBBLE_FIELDS | {"fontId"})
                       for keys in (fields or {}).values())
        if not explicit and native_values(documents) == native_values(self.page_bubbles(connection, page_id)):
            return lambda writer: self._publish_bubbles(writer, page_id, revision, documents)
        result = self.prepare(connection, page_id, documents=documents, fields=fields)
        return lambda writer: self.publish(writer, page_id, revision, result, **producer)

    @staticmethod
    def invoke(request: dict) -> dict:
        executable = Path(resource_path("native/saber-editor.exe" if os.name == "nt" else "native/saber-editor"))
        if not executable.is_file():
            raise RuntimeError("缺少编辑内核，请退出后重新启动以准备内核；发行包请重新完整解压")
        result = subprocess.run([str(executable)], input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                                env={**os.environ, "PHOTOCRAFT_NO_SYSTEM_FONTS": "1"},
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            answer = json.loads(result.stdout)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("PhotoCraft 编辑内核未返回有效结果") from exc
        if result.returncode or not answer.get("ok"):
            raise ValueError(answer.get("error", "PhotoCraft 编辑失败"))
        return answer["document"]

    @staticmethod
    def render_image(image, payloads):
        """Pure algorithm entry point for existing pipeline/plug-in callers."""
        from PIL import Image

        with TemporaryDirectory() as directory:
            root = Path(directory)
            image.save(root / "source.png", format="PNG")
            font_paths = {str(payload["fontFamily"]) for payload in payloads}
            documents = []
            for index, payload in enumerate(payloads):
                value = dict(payload)
                font = value.pop("fontFamily")
                documents.append(dict(bubbleId=str(index), fontId=font, payload=value))
            NativeEditor.invoke(dict(pageId="render", source=str(root / "source.png"),
                                     bubbles=documents, fonts=[dict(id=path, path=path) for path in font_paths],
                                     png=str(root / "result.png")))
            with Image.open(root / "result.png") as output:
                return output.convert("RGB")

    def prepare(self, connection, page_id: str, *, documents=None, uploaded: bytes | None = None,
                render: bool = False, repair: bytes | None = None, text_only: bool = False,
                mask_override: bytes | None = None, background: bytes | None = None, fields=None) -> EditingResult:
        page = connection.execute(select(pages).where(pages.c.id == page_id)).mappings().one()
        pointer_rows = list(connection.execute(select(page_assets.c.role,
                            assets.c.relative_path, assets.c.width, assets.c.height)
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(page_assets.c.page_id == page_id)))
        pointers = {row.role: self.storage.resolve_relative_path(row.relative_path) for row in pointer_rows}
        records = {row.role: row for row in pointer_rows}
        current = pointers["editor_document"].read_bytes() if "editor_document" in pointers else None
        legacy = self._legacy_document(current) if current is not None else None
        writing = (documents is not None or uploaded is not None or repair is not None
                   or mask_override is not None or background is not None or current is None or legacy is not None)
        stored = self.page_bubbles(connection, page_id) if documents is None and (writing or not render and not text_only) else []
        if documents is None and uploaded is None and (current is None or legacy == "text"):
            documents = stored
        business = {item["bubbleId"]: item["payload"] for item in (documents if documents is not None else stored)}
        temporary_root = self.storage.data_root / "temp" / "editing"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            request = {"pageId": page_id, "fonts": self.font_files(connection),
                       "defaultFontId": page["default_font_id"]}
            if writing:
                request["output"] = str(root / "document.pcraft")
            if uploaded is not None:
                (root / "input.pcraft").write_bytes(uploaded)
                request["project"] = str(root / "input.pcraft")
            elif "editor_document" in pointers:
                request["project"] = str(pointers["editor_document"])
            else:
                request["source"] = str(pointers["source"])
                if "clean" in pointers:
                    request["base"] = str(pointers["clean"])
            if background is not None:
                (root / "background.png").write_bytes(background)
                if "project" in request:
                    request["baseUpdate"] = str(root / "background.png")
                else:
                    request["base"] = str(root / "background.png")
            if documents is not None:
                request["bubbles"] = [{**item, "fontId": item["fontId"] or page["default_font_id"],
                                       "fields": sorted((fields or {}).get(item["bubbleId"], []))}
                                      for item in documents]
            if uploaded is None and (current is None or legacy or mask_override is not None):
                request["userMask"] = str(pointers["user_mask"]) if "user_mask" in pointers else None
                if mask_override is not None:
                    (root / "user-mask.png").write_bytes(mask_override)
                    request["userMask"] = str(root / "user-mask.png")
            if repair is not None:
                (root / "repair.png").write_bytes(repair)
                request["repair"] = str(root / "repair.png")
            if render:
                request.update(png=str(root / "translated.png"), clean=str(root / "clean.png"))
            if text_only:
                request["text"] = str(root / "text.png")
            projection = self.invoke(request)
            source = records["source"]
            if (projection["width"], projection["height"]) != (source.width, source.height):
                raise ValueError("漫画页面需要保持原图尺寸，请使用图层变换调整内容")
            projected = []
            for bubble in projection["bubbles"]:
                defaults = BubbleState().to_dict()
                defaults.pop("fontFamily")
                if bubble["bubbleId"] not in business:
                    defaults["autoTextDirection"] = bubble["payload"].get("textDirection", defaults["autoTextDirection"])
                defaults.update({key: value for key, value in business.get(bubble["bubbleId"], {}).items()
                                 if key not in NATIVE_BUBBLE_FIELDS})
                defaults.update(bubble["payload"])
                bubble["payload"] = validate_bubble_payload(defaults, render=False)
                projected.append(bubble)
            user_mask = base64.b64decode(projection["userMask"], validate=True) if projection.get("userMask") else None
            if user_mask is not None:
                from src.backend_v2.operations.masks import read_user_mask

                with read_user_mask(BytesIO(user_mask)) as mask:
                    if mask.size != (projection["width"], projection["height"]):
                        raise ValueError("手动修复范围尺寸与原图不一致")
            return EditingResult((root / "document.pcraft").read_bytes() if writing else
                                 (current if not render and not text_only else None), projected,
                                 projection["width"], projection["height"],
                                 (root / "translated.png").read_bytes() if render else None,
                                 (root / "clean.png").read_bytes() if render else None,
                                 (root / "text.png").read_bytes() if text_only else None, user_mask)

    def publish(self, connection, page_id: str, revision: int, result: EditingResult,
                *, render_request_id=None, operation_id=None, job_step_id=None):
        source_revision = connection.execute(select(pages.c.source_revision).where(pages.c.id == page_id)).scalar_one()
        if result.document is not None:
            self._publish_bubbles(connection, page_id, revision, result.bubbles)
        current_mask = connection.execute(select(page_assets.c.asset_id, assets.c.checksum).join(
            assets, assets.c.id == page_assets.c.asset_id).where(page_assets.c.page_id == page_id,
                                                              page_assets.c.role == "user_mask")).first()
        checksum = hashlib.sha256(result.user_mask).hexdigest() if result.user_mask is not None else None
        if result.document is not None and checksum != (current_mask.checksum if current_mask else None):
            connection.execute(page_assets.delete().where(page_assets.c.page_id == page_id, page_assets.c.role == "user_mask"))
            if result.user_mask is not None:
                mask = self.storage.publish_bytes(result.user_mask, extension="png", mime_type="image/png",
                                                  width=result.width, height=result.height, connection=connection)
                connection.execute(insert(page_assets).values(page_id=page_id, role="user_mask", asset_id=mask.id,
                    input_source_revision=source_revision, input_document_revision=revision,
                    producer_operation_id=operation_id, producer_job_step_id=job_step_id,
                    producer_render_request_id=render_request_id))
        for role, payload, extension, mime in (
            ("translated", result.translated, "png", "image/png"),
            ("clean", result.clean, "png", "image/png"),
            ("editor_document", result.document, "pcraft", "application/x-photocraft"),
        ):
            if payload is None:
                continue
            asset = self.storage.publish_bytes(payload, extension=extension, mime_type=mime,
                                              width=result.width, height=result.height, connection=connection)
            values = dict(asset_id=asset.id, input_source_revision=source_revision,
                          input_document_revision=revision, parent_asset_id=None,
                          producer_job_step_id=job_step_id, producer_operation_id=operation_id,
                          producer_render_request_id=render_request_id)
            present = connection.execute(select(page_assets.c.asset_id).where(
                page_assets.c.page_id == page_id, page_assets.c.role == role)).scalar_one_or_none()
            if present:
                connection.execute(update(page_assets).where(page_assets.c.page_id == page_id,
                                   page_assets.c.role == role).values(**values))
            else:
                connection.execute(insert(page_assets).values(page_id=page_id, role=role, **values))

    @staticmethod
    def _publish_bubbles(connection, page_id, revision, documents):
        """Only document commits replace the cached native text projection."""
        ids = [item["bubbleId"] for item in documents]
        existing = set(connection.execute(select(bubbles.c.id).where(bubbles.c.page_id == page_id)).scalars())
        connection.execute(bubbles.delete().where(bubbles.c.page_id == page_id, bubbles.c.id.not_in(ids)))
        # Move old ordinals aside before assigning the document's layer order.
        maximum = connection.execute(select(bubbles.c.ordinal).where(bubbles.c.page_id == page_id)
                                     .order_by(bubbles.c.ordinal.desc()).limit(1)).scalar_one_or_none() or 0
        connection.execute(update(bubbles).where(bubbles.c.page_id == page_id).values(ordinal=bubbles.c.ordinal + maximum + len(ids) + 1))
        for ordinal, item in enumerate(documents, 1):
            values = dict(ordinal=ordinal, font_id=item["fontId"], payload_json=canonical_json(item["payload"]),
                          updated_revision=revision, updated_at=utcnow())
            if item["bubbleId"] in existing:
                connection.execute(update(bubbles).where(bubbles.c.id == item["bubbleId"], bubbles.c.page_id == page_id).values(**values))
            else:
                connection.execute(insert(bubbles).values(id=item["bubbleId"], page_id=page_id, **values))
