"""Build PhotoCraft manga layer-style preset bundle via native/saber-editor.

Reproduces ``src/backend_v2/resources/photocraft_manga_styles.json`` using PhotoCraft's
own Rust engine serialization (``layer.layerStyle.replace`` and ``style.presets.apply``)
so preset payloads always stay schema-compatible without editing third_party/photocraft.

Usage:
    python scripts/build_manga_layer_styles.py
    python scripts/build_manga_layer_styles.py --preview-dir /tmp/style-previews
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from zipfile import ZipFile
import zlib

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "backend_v2" / "resources" / "photocraft_manga_styles.json"


def _saber_editor_binary(root: Path = PROJECT_ROOT) -> Path:
    return root / "native" / ("saber-editor.exe" if os.name == "nt" else "saber-editor")


def _make_solid_png(w: int, h: int, rgba: tuple[int, int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgba) * w
    raw = row * h
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _make_split_png(
    w: int,
    h: int,
    top_rgba: tuple[int, int, int, int] = (255, 255, 255, 255),
    bot_rgba: tuple[int, int, int, int] = (138, 144, 154, 255),
) -> bytes:
    """Split light/gray background so both dark and white strokes are visible."""
    img = Image.new("RGBA", (w, h), top_rgba)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, h // 2, w, h], fill=bot_rgba)
    for y in range(0, h, 14):
        for x in range(w - 42, w, 14):
            c = (212, 216, 222, 255) if ((x // 14 + y // 14) % 2 == 0) else (175, 180, 188, 255)
            draw.rectangle([x, y, min(x + 13, w - 1), min(y + 13, h - 1)], fill=c)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_saber(req: dict, *, executable: Path) -> dict:
    proc = subprocess.run(
        [str(executable)],
        input=json.dumps(req, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=60,
        env={**os.environ, "PHOTOCRAFT_NO_SYSTEM_FONTS": "1"},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"saber-editor failed ({proc.returncode}): "
            f"{proc.stdout.decode(errors='replace')} {proc.stderr.decode(errors='replace')}"
        )
    return json.loads(proc.stdout.decode("utf-8"))


# 28 manga typesetting styles across 3 groups
MANGA_GROUPS = [
    {
        "name": "漫画拟声词 · 立体特效",
        "items": [
            {
                "name": "黑底立体白字（图款·标准）",
                "fill_opacity": 1.0,
                "sample": ("ぎゅーっ", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 95, "useGlobalLight": False, "distance": 3, "spread": 100, "size": 3, "knocksOut": True}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 95, "useGlobalLight": False, "distance": 6, "spread": 100, "size": 3, "knocksOut": True}},
                ],
            },
            {
                "name": "黑底立体白字（图款·加厚）",
                "fill_opacity": 1.0,
                "sample": ("ぎゅーっ", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 4, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 98, "useGlobalLight": False, "distance": 3, "spread": 100, "size": 4, "knocksOut": True}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 98, "useGlobalLight": False, "distance": 6, "spread": 100, "size": 4, "knocksOut": True}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 98, "useGlobalLight": False, "distance": 9, "spread": 100, "size": 4, "knocksOut": True}},
                ],
            },
            {
                "name": "黑底立体白字（45°斜影）",
                "fill_opacity": 1.0,
                "sample": ("咚咔！", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 135, "useGlobalLight": False, "distance": 3, "spread": 100, "size": 3, "knocksOut": True}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 135, "useGlobalLight": False, "distance": 6, "spread": 100, "size": 3, "knocksOut": True}},
                ],
            },
            {
                "name": "黑底立体（保留原字色）",
                "fill_opacity": 1.0,
                "sample": ("砰轰！", "#ff4d6d"),
                "effects": [
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 95, "useGlobalLight": False, "distance": 3, "spread": 100, "size": 3, "knocksOut": True}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 95, "useGlobalLight": False, "distance": 6, "spread": 100, "size": 3, "knocksOut": True}},
                ],
            },
            {
                "name": "白底立体黑字（暗背景用）",
                "fill_opacity": 1.0,
                "sample": ("咔哒", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#111111", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal", "angle": 95, "useGlobalLight": False, "distance": 3, "spread": 100, "size": 3, "knocksOut": True}},
                    {"kind": "dropShadow", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal", "angle": 95, "useGlobalLight": False, "distance": 6, "spread": 100, "size": 3, "knocksOut": True}},
                ],
            },
            {
                "name": "双层白黑描边（黑字爆发）",
                "fill_opacity": 1.0,
                "sample": ("什…什么", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#111111", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 6, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "双层黑白描边（白字防杂底）",
                "fill_opacity": 1.0,
                "sample": ("哗啦啦", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 7, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "三层漫画冲击（白黑红）",
                "fill_opacity": 1.0,
                "sample": ("绝不！", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 6, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 9, "position": "outside", "color": "#d92626", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "硬边错位重影（黑白）",
                "fill_opacity": 1.0,
                "sample": ("嗡—", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 100, "blend": "Normal", "angle": 135, "useGlobalLight": False, "distance": 7, "spread": 100, "size": 3, "knocksOut": True}},
                ],
            },
            {
                "name": "镂空黑边字（空心）",
                "fill_opacity": 0.0,
                "sample": ("呆滞…", "#000000"),
                "effects": [
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "镂空白边字（暗底空心）",
                "fill_opacity": 0.0,
                "sample": ("唔…", "#ffffff"),
                "effects": [
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 85, "blend": "Normal", "angle": 120, "useGlobalLight": False, "distance": 0, "spread": 10, "size": 7, "knocksOut": True}},
                ],
            },
        ],
    },
    {
        "name": "漫画嵌字 · 常规描边与光晕",
        "items": [
            {
                "name": "细白描边（2px·框外小字）",
                "fill_opacity": 1.0,
                "sample": ("小声嘟囔", "#111111"),
                "effects": [
                    {"kind": "stroke", "params": {"size": 2, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "中白描边（4px·通用嵌字）",
                "fill_opacity": 1.0,
                "sample": ("等一下！", "#111111"),
                "effects": [
                    {"kind": "stroke", "params": {"size": 4, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "粗白描边（7px·复杂背景）",
                "fill_opacity": 1.0,
                "sample": ("别过来！", "#111111"),
                "effects": [
                    {"kind": "stroke", "params": {"size": 7, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "柔和白光晕（网点背景用）",
                "fill_opacity": 1.0,
                "sample": ("沙沙声", "#111111"),
                "effects": [
                    {"kind": "outerGlow", "params": {"color": "#ffffff", "blend": "Normal", "opacity": 100, "spread": 35, "size": 10, "range": 50}},
                ],
            },
            {
                "name": "白字细黑边（2px·旁白）",
                "fill_opacity": 1.0,
                "sample": ("那年夏天", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 2, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "白字粗黑边（4px·强调）",
                "fill_opacity": 1.0,
                "sample": ("好痛！！", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 4, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "白字黑光晕（暗场·回忆）",
                "fill_opacity": 1.0,
                "sample": ("不可能…", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "outerGlow", "params": {"color": "#000000", "blend": "Normal", "opacity": 90, "spread": 25, "size": 12, "range": 50}},
                ],
            },
            {
                "name": "内心独白（灰字柔白光）",
                "fill_opacity": 1.0,
                "sample": ("（果然…）", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#333333", "opacity": 100, "blend": "Normal"}},
                    {"kind": "outerGlow", "params": {"color": "#ffffff", "blend": "Normal", "opacity": 95, "spread": 20, "size": 10, "range": 50}},
                ],
            },
            {
                "name": "彩漫字幕白字（细边柔影）",
                "fill_opacity": 1.0,
                "sample": ("第1话完", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 2, "position": "outside", "color": "#1f1f1f", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 55, "blend": "Multiply", "angle": 120, "useGlobalLight": False, "distance": 3, "spread": 0, "size": 6, "knocksOut": True}},
                ],
            },
        ],
    },
    {
        "name": "漫画氛围 · 渐变与情绪字",
        "items": [
            {
                "name": "黑白竖渐变（经典拟声）",
                "fill_opacity": 1.0,
                "sample": ("轰隆隆", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#ffffff", "to": "#1a1a1a", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 5, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                ],
            },
            {
                "name": "黑灰压抑渐变（阴郁/低沉）",
                "fill_opacity": 1.0,
                "sample": ("完了…", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#080808", "to": "#888888", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 60, "blend": "Multiply", "angle": 90, "useGlobalLight": False, "distance": 0, "spread": 15, "size": 10, "knocksOut": True}},
                ],
            },
            {
                "name": "恐怖黑雾字（惊悚/黑化）",
                "fill_opacity": 1.0,
                "sample": ("死吧…", "#000000"),
                "effects": [
                    {"kind": "colorOverlay", "params": {"color": "#111111", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 2, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "outerGlow", "params": {"color": "#000000", "blend": "Normal", "opacity": 92, "spread": 20, "size": 16, "range": 50}},
                ],
            },
            {
                "name": "血红爆发渐变（愤怒/惊悚）",
                "fill_opacity": 1.0,
                "sample": ("可恶！", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#520000", "to": "#ff2424", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#000000", "opacity": 100, "blend": "Normal"}},
                    {"kind": "outerGlow", "params": {"color": "#7a0000", "blend": "Normal", "opacity": 75, "spread": 10, "size": 12, "range": 50}},
                ],
            },
            {
                "name": "少女粉嫩渐变（恋爱/可爱）",
                "fill_opacity": 1.0,
                "sample": ("心动♥", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#ff6ea8", "to": "#fff0f6", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 5, "position": "outside", "color": "#ff478c", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#ff6ea8", "opacity": 45, "blend": "Multiply", "angle": 120, "useGlobalLight": False, "distance": 3, "spread": 0, "size": 6, "knocksOut": True}},
                ],
            },
            {
                "name": "高能金黄渐变（热血/标题）",
                "fill_opacity": 1.0,
                "sample": ("必杀！", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#ff6600", "to": "#fff066", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#1a0c00", "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 6, "position": "outside", "color": "#ffffff", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 90, "blend": "Normal", "angle": 120, "useGlobalLight": False, "distance": 5, "spread": 100, "size": 3, "knocksOut": True}},
                ],
            },
            {
                "name": "冰蓝清爽渐变（冷场/水效）",
                "fill_opacity": 1.0,
                "sample": ("飕—", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#1565c0", "to": "#e0f7ff", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "stroke", "params": {"size": 3, "position": "outside", "color": "#0f3a6b", "opacity": 100, "blend": "Normal"}},
                    {"kind": "outerGlow", "params": {"color": "#ffffff", "blend": "Normal", "opacity": 90, "spread": 30, "size": 8, "range": 50}},
                ],
            },
            {
                "name": "金属银灰浮雕（封面/标题）",
                "fill_opacity": 1.0,
                "sample": ("最终章", "#000000"),
                "effects": [
                    {"kind": "gradientOverlay", "params": {"from": "#f5f7fa", "to": "#3f4650", "angle": 90, "style": "linear", "scale": 100, "opacity": 100, "blend": "Normal"}},
                    {"kind": "bevelEmboss", "params": {"style": "inner", "size": 4, "depth": 200, "soften": 1, "angle": 120, "useGlobalLight": False}},
                    {"kind": "stroke", "params": {"size": 2, "position": "outside", "color": "#1f2328", "opacity": 100, "blend": "Normal"}},
                    {"kind": "dropShadow", "params": {"color": "#000000", "opacity": 55, "blend": "Multiply", "angle": 120, "useGlobalLight": False, "distance": 4, "spread": 0, "size": 6, "knocksOut": True}},
                ],
            },
        ],
    },
]

BUILTIN_SPEC = [
    ("Basics", [
        ("Default Style (None)", None),
        ("Drop Shadow", None),
        ("Soft Shadow", None),
        ("Black Stroke", None),
        ("White Stroke", None),
        ("Inner Shadow", None),
        ("Outer Glow", None),
        ("Emboss", None),
    ]),
    ("Text Effects", [
        ("Neon Blue", None),
        ("Neon Pink", None),
        ("Chrome", None),
        ("Gold", None),
        ("Comic Outline", None),
        ("Letterpress", None),
        ("Hollow", 0.0),
    ]),
    ("Buttons", [
        ("Glass", None),
        ("Gel Green", None),
        ("Pressed", None),
        ("Pill Red", None),
        ("Flat Shadow", None),
        ("Satin Plum", None),
    ]),
]


def extract_serialized_effects(tmp: Path, commands: list[dict], *, executable: Path) -> list[dict]:
    src = tmp / "tiny.png"
    if not src.exists():
        src.write_bytes(_make_solid_png(32, 32, (255, 255, 255, 255)))
    out_pcraft = tmp / "out.pcraft"
    req = {
        "pageId": "extract",
        "source": str(src),
        "commands": [
            {"id": "layer.new.layer", "params": {"name": "target"}},
            *commands,
        ],
        "output": str(out_pcraft),
    }
    _run_saber(req, executable=executable)
    with ZipFile(out_pcraft) as zf:
        doc = json.loads(zf.read("manifest.json"))["document"]
    target = [layer for layer in doc["layers"] if layer["name"] == "target"][0]
    return target["effects"]["items"]


def _resolve_preview_font(explicit_font: Path | None) -> Path:
    if explicit_font is not None:
        return explicit_font
    fonts_dir = PROJECT_ROOT / "src" / "backend_v2" / "resources" / "fonts"
    for candidate in (
        "[toolbox]圆体-简繁-Heavy(v2.4).ttf",
        "汉仪正圆-75W.TTF",
        "msyh.ttc",
        "simhei.ttf",
        "Arial_Unicode.ttf",
    ):
        path = fonts_dir / candidate
        if path.is_file():
            return path
    found = sorted(fonts_dir.glob("*.tt*")) + sorted(fonts_dir.glob("*.otf"))
    if not found:
        raise FileNotFoundError(f"No usable preview font found in {fonts_dir}")
    return found[0]


def _render_sample_tile(
    tmp: Path,
    idx: int,
    text: str,
    text_color: str,
    effects_cmd: list[dict],
    fill_opacity: float,
    font_path: Path,
    direction: str,
    w: int,
    h: int,
    bg_bytes: bytes,
    font_size: int,
    *,
    executable: Path,
) -> Image.Image:
    src = tmp / f"bg_{idx}.png"
    src.write_bytes(bg_bytes)
    out_png = tmp / f"tile_{idx}.png"
    commands = [{"id": "layer.layerStyle.replace", "params": {"effects": effects_cmd}}]
    if fill_opacity < 1.0:
        commands.append({"id": "layer.layerStyle.blendingOptions", "params": {"fillOpacity": round(fill_opacity * 100.0)}})
    req = {
        "pageId": f"tile-{idx}",
        "source": str(src),
        "fonts": [{"id": "f1", "path": str(font_path)}],
        "defaultFontId": "f1",
        "bubbles": [
            {
                "bubbleId": "b1",
                "fontId": "f1",
                "payload": {
                    "translatedText": text,
                    "coords": [8, 8, w - 8, h - 8],
                    "position": {"x": 0, "y": 0},
                    "fontSize": font_size,
                    "textColor": text_color,
                    "textDirection": direction,
                    "rotationAngle": 0,
                    "inlineAlign": "center",
                    "blockAlign": "center",
                    "lineSpacing": 1.15,
                    "strokeEnabled": False,
                    "strokeColor": "#000000",
                    "strokeWidth": 0,
                },
            }
        ],
        "commands": commands,
        "png": str(out_png),
    }
    _run_saber(req, executable=executable)
    return Image.open(out_png).convert("RGBA")


def build_styles_payload(*, executable: Path | None = None) -> dict:
    binary = executable or _saber_editor_binary()
    if not binary.is_file():
        raise FileNotFoundError(f"native saber-editor binary not found: {binary}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manga_groups_out = []
        for group in MANGA_GROUPS:
            items_out = []
            for item in group["items"]:
                serialized = extract_serialized_effects(
                    tmp,
                    [{"id": "layer.layerStyle.replace", "params": {"effects": item["effects"]}}],
                    executable=binary,
                )
                items_out.append({
                    "name": item["name"],
                    "effects": serialized,
                    "blend": None,
                    "fill_opacity": item["fill_opacity"],
                })
            manga_groups_out.append({"name": group["name"], "items": items_out})

        builtin_groups_out = []
        for group_name, specs in BUILTIN_SPEC:
            items_out = []
            for style_name, fill_op in specs:
                serialized = extract_serialized_effects(
                    tmp,
                    [{"id": "style.presets.apply", "params": {"preset": style_name, "group": group_name}}],
                    executable=binary,
                )
                items_out.append({
                    "name": style_name,
                    "effects": serialized,
                    "blend": None,
                    "fill_opacity": fill_op,
                })
            builtin_groups_out.append({"name": group_name, "items": items_out})

    return {
        "format": "saber-photocraft-styles",
        "version": 1,
        "mangaGroups": manga_groups_out,
        "builtinGroups": builtin_groups_out,
    }


def render_previews(preview_dir: Path, *, font_path: Path, executable: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        label_font = ImageFont.truetype(str(font_path), 15)
        sub_font = ImageFont.truetype(str(font_path), 13)

        urgent_items = MANGA_GROUPS[0]["items"][:4]
        closeup = Image.new("RGBA", (220 * 4 + 40, 440), (246, 247, 249, 255))
        cdraw = ImageDraw.Draw(closeup)
        for col, item in enumerate(urgent_items):
            tile_jp = _render_sample_tile(
                tmp, 100 + col, "ぎゅーっ", "#ff4d6d" if "保留原字色" in item["name"] else "#000000",
                item["effects"], item["fill_opacity"], font_path, "vertical", 110, 360,
                _make_solid_png(110, 360, (255, 255, 255, 255)), 56, executable=executable,
            )
            tile_cn = _render_sample_tile(
                tmp, 200 + col, "紧握！", "#ff4d6d" if "保留原字色" in item["name"] else "#000000",
                item["effects"], item["fill_opacity"], font_path, "vertical", 110, 360,
                _make_solid_png(110, 360, (255, 255, 255, 255)), 52, executable=executable,
            )
            x0 = 16 + col * 224
            closeup.paste(tile_jp, (x0, 16))
            closeup.paste(tile_cn, (x0 + 108, 16))
            cdraw.rectangle([x0, 16, x0 + 217, 375], outline=(210, 214, 220, 255), width=1)
            cdraw.text((x0 + 4, 386), item["name"], fill=(30, 32, 38, 255), font=label_font)
            cdraw.text((x0 + 4, 410), f"#{col + 1}", fill=(110, 118, 128, 255), font=sub_font)
        closeup.save(preview_dir / "urgent_closeup.png")

        tile_w, tile_h = 240, 108
        split_bg = _make_split_png(tile_w, tile_h)
        cols = 4
        all_items = [(g["name"], it) for g in MANGA_GROUPS for it in g["items"]]
        rows = (len(all_items) + cols - 1) // cols
        cell_w, cell_h = tile_w + 18, tile_h + 36
        sheet = Image.new("RGBA", (cols * cell_w + 24, rows * cell_h + 24), (240, 242, 246, 255))
        sdraw = ImageDraw.Draw(sheet)
        for idx, (_g_name, item) in enumerate(all_items):
            r, c = divmod(idx, cols)
            x0, y0 = 12 + c * cell_w + 9, 12 + r * cell_h + 6
            txt, tcol = item["sample"]
            tile = _render_sample_tile(
                tmp, 300 + idx, txt, tcol,
                item["effects"], item["fill_opacity"], font_path, "horizontal", tile_w, tile_h,
                split_bg, 42, executable=executable,
            )
            sheet.paste(tile, (x0, y0))
            sdraw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=(195, 200, 208, 255), width=1)
            sdraw.text((x0, y0 + tile_h + 4), f"{idx + 1}. {item['name']}", fill=(24, 28, 34, 255), font=sub_font)
        sheet.save(preview_dir / "all_styles_sheet.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PhotoCraft manga layer styles JSON via saber-editor.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional directory to render preview PNGs")
    parser.add_argument("--font", type=Path, default=None, help="Font path used only with --preview-dir")
    args = parser.parse_args()

    executable = _saber_editor_binary()
    payload = build_styles_payload(executable=executable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manga_count = sum(len(g["items"]) for g in payload["mangaGroups"])
    builtin_count = sum(len(g["items"]) for g in payload["builtinGroups"])
    print(f"Wrote {args.output} ({manga_count} manga + {builtin_count} builtin presets)")

    if args.preview_dir is not None:
        font_path = _resolve_preview_font(args.font)
        render_previews(args.preview_dir, font_path=font_path, executable=executable)
        print(f"Rendered preview sheets to {args.preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
