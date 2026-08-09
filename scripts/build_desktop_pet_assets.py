#!/usr/bin/env python3
"""Build and validate the Saber desktop-pet atlas from generated row strips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CELL_WIDTH = 192
CELL_HEIGHT = 208
COLUMNS = 8
KEY_COLOR = np.array([0, 255, 0], dtype=np.float32)
TRANSPARENT_THRESHOLD = 30.0
OPAQUE_THRESHOLD = 105.0
SUBJECT_ALPHA_THRESHOLD = 180
GENERATED_STATES = (
    "idle",
    "greeting",
    "starting",
    "waiting",
    "translating",
    "analyzing",
    "transfer",
    "paused",
    "success",
    "warning",
    "failed",
    "drag_right",
)
EXPECTED_STATES = (*GENERATED_STATES, "drag_left")


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def remove_chroma(image: Image.Image) -> Image.Image:
    """Remove the generated flat green background with a soft, despilled matte."""

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[:, :, :3].astype(np.float32)
    distance = np.max(np.abs(rgb - KEY_COLOR), axis=2)
    non_green = np.maximum(rgb[:, :, 0], rgb[:, :, 2])
    dominance = rgb[:, :, 1] - non_green
    key_like = (distance <= 32.0) | (dominance >= 16.0)

    ratio = (distance - TRANSPARENT_THRESHOLD) / (
        OPAQUE_THRESHOLD - TRANSPARENT_THRESHOLD
    )
    distance_alpha = 255.0 * _smoothstep(ratio)
    denominator = np.maximum(1.0, 255.0 - non_green)
    dominance_alpha = 255.0 * (1.0 - np.clip(dominance / denominator, 0.0, 1.0))
    matte = np.where(key_like, np.minimum(distance_alpha, dominance_alpha), 255.0)
    matte *= rgba[:, :, 3].astype(np.float32) / 255.0
    matte[matte <= 8.0] = 0.0

    partial = key_like & (matte < 252.0) & (matte > 0.0)
    green_cap = np.maximum(rgb[:, :, 0], rgb[:, :, 2]) - 1.0
    rgb[:, :, 1] = np.where(partial, np.minimum(rgb[:, :, 1], green_cap), rgb[:, :, 1])
    rgba[:, :, :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    rgba[:, :, 3] = np.clip(matte, 0.0, 255.0).astype(np.uint8)

    output = Image.fromarray(rgba, mode="RGBA")
    alpha = output.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.65))
    output.putalpha(alpha)
    return output


def _extract_subjects(strip: Image.Image) -> list[Image.Image]:
    """Find the eight character bodies without assuming perfectly equal slots."""

    rgba = np.asarray(strip.convert("RGBA"), dtype=np.uint8)
    mask = (rgba[:, :, 3] > SUBJECT_ALPHA_THRESHOLD).astype(np.uint8)
    return _extract_subjects_by_clusters(rgba, mask)


def _extract_subjects_by_clusters(
    rgba: np.ndarray,
    mask: np.ndarray,
) -> list[Image.Image]:
    """Split a row when adjacent hair pixels merge two otherwise distinct poses."""

    width = rgba.shape[1]
    x_values = np.arange(width, dtype=np.float64)
    weights = mask.sum(axis=0).astype(np.float64)
    centers = np.linspace(width / (COLUMNS * 2), width - width / (COLUMNS * 2), COLUMNS)
    for _iteration in range(64):
        assignments = np.argmin(np.abs(x_values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in range(COLUMNS):
            selected = assignments == index
            total = float(weights[selected].sum())
            if total:
                updated[index] = float(np.sum(x_values[selected] * weights[selected]) / total)
        if np.max(np.abs(updated - centers)) < 0.05:
            centers = updated
            break
        centers = updated

    boundaries = [0]
    boundaries.extend(round((centers[index] + centers[index + 1]) / 2) for index in range(7))
    boundaries.append(width)
    kernel = np.ones((3, 3), dtype=np.uint8)
    frames: list[Image.Image] = []
    for left, right in zip(boundaries, boundaries[1:]):
        crop = rgba[:, left:right].copy()
        crop_mask = (crop[:, :, 3] > SUBJECT_ALPHA_THRESHOLD).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(crop_mask, 8)
        if count <= 1:
            raise ValueError("could not separate eight character poses")
        areas = stats[1:, cv2.CC_STAT_AREA]
        subject_label = int(np.argmax(areas)) + 1
        keep = (labels == subject_label).astype(np.uint8)
        keep = cv2.dilate(keep, kernel, iterations=2).astype(bool)
        crop[~keep] = 0
        image = Image.fromarray(crop, mode="RGBA")
        bbox = image.getbbox()
        if bbox is None:
            raise ValueError("clustered character pose is empty")
        frames.append(image.crop(bbox))
    return frames


def _normalize_row(frames: list[Image.Image]) -> list[Image.Image]:
    bounds = [frame.getbbox() for frame in frames]
    if any(bounds_item is None for bounds_item in bounds):
        raise ValueError("generated strip contains an empty frame")
    concrete = [item for item in bounds if item is not None]
    max_width = max(right - left for left, _top, right, _bottom in concrete)
    max_height = max(bottom - top for _left, top, _right, bottom in concrete)
    scale = min((CELL_WIDTH - 10) / max_width, (CELL_HEIGHT - 10) / max_height)

    normalized: list[Image.Image] = []
    for frame, bbox in zip(frames, concrete):
        sprite = frame.crop(bbox)
        sprite = sprite.resize(
            (
                max(1, round(sprite.width * scale)),
                max(1, round(sprite.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        cell = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
        left = (CELL_WIDTH - sprite.width) // 2
        top = CELL_HEIGHT - 5 - sprite.height
        cell.alpha_composite(sprite, (left, top))
        normalized.append(cell)
    return normalized


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("pet manifest rows are missing")
    states = [row.get("state") for row in rows if isinstance(row, dict)]
    row_indexes = [row.get("row") for row in rows if isinstance(row, dict)]
    if states != list(EXPECTED_STATES) or row_indexes != list(range(len(EXPECTED_STATES))):
        raise ValueError("pet manifest must declare the 13 project states in canonical order")
    if any(row.get("frameCount") != COLUMNS for row in rows):
        raise ValueError("every project pet row must contain exactly eight frames")
    return payload, rows


def _load_rows(source_dir: Path) -> dict[str, list[Image.Image]]:
    output: dict[str, list[Image.Image]] = {}
    for state in GENERATED_STATES:
        path = source_dir / f"{state}.png"
        if not path.is_file():
            raise FileNotFoundError(f"missing generated strip: {path}")
        with Image.open(path) as opened:
            transparent = remove_chroma(opened)
        try:
            output[state] = _normalize_row(_extract_subjects(transparent))
        except ValueError as error:
            raise ValueError(f"{state}: {error}") from error
    output["drag_left"] = [
        frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for frame in output["drag_right"]
    ]
    return output


def _assemble(rows: list[dict[str, Any]], frames: dict[str, list[Image.Image]]) -> Image.Image:
    atlas = Image.new(
        "RGBA",
        (CELL_WIDTH * COLUMNS, CELL_HEIGHT * len(rows)),
        (0, 0, 0, 0),
    )
    for row in rows:
        state = str(row["state"])
        y = int(row["row"]) * CELL_HEIGHT
        for column, frame in enumerate(frames[state]):
            atlas.alpha_composite(frame, (column * CELL_WIDTH, y))
    return atlas


def _checker(size: tuple[int, int], square: int = 12) -> Image.Image:
    background = Image.new("RGB", size, "#fffafb")
    draw = ImageDraw.Draw(background)
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill="#f3e7eb")
    return background


def _write_contact_sheet(
    rows: list[dict[str, Any]],
    frames: dict[str, list[Image.Image]],
    output: Path,
) -> None:
    scale = 0.5
    cell_width = round(CELL_WIDTH * scale)
    cell_height = round(CELL_HEIGHT * scale)
    label_height = 22
    sheet = Image.new(
        "RGB",
        (cell_width * COLUMNS, (cell_height + label_height) * len(rows)),
        "#fffafb",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * (cell_height + label_height)
        draw.rectangle((0, y, sheet.width, y + label_height - 1), fill="#3f3038")
        draw.text((7, y + 5), str(row["state"]), fill="#ffffff", font=font)
        for column, frame in enumerate(frames[str(row["state"])]):
            preview = frame.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            background = _checker((cell_width, cell_height))
            background.paste(preview, (0, 0), preview)
            x = column * cell_width
            sheet.paste(background, (x, y + label_height))
            draw.rectangle(
                (x, y + label_height, x + cell_width - 1, y + label_height + cell_height - 1),
                outline="#e6a6bb",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="WEBP", quality=92, method=6)


def _write_animation_previews(
    rows: list[dict[str, Any]],
    frames: dict[str, list[Image.Image]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        state = str(row["state"])
        durations = [int(value) for value in row["durationsMs"]]
        previews: list[Image.Image] = []
        for frame in frames[state]:
            background = _checker(frame.size)
            background.paste(frame, (0, 0), frame)
            previews.append(background)
        previews[0].save(
            output_dir / f"{state}.gif",
            save_all=True,
            append_images=previews[1:],
            duration=durations,
            loop=0,
            disposal=2,
        )


def _validate_atlas(
    atlas: Image.Image,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_size = (CELL_WIDTH * COLUMNS, CELL_HEIGHT * len(rows))
    if atlas.size != expected_size:
        raise ValueError(f"atlas is {atlas.size}, expected {expected_size}")

    rgba = np.asarray(atlas, dtype=np.uint8)
    empty: list[str] = []
    visible_counts: dict[str, list[int]] = {}
    for row in rows:
        state = str(row["state"])
        y = int(row["row"]) * CELL_HEIGHT
        counts: list[int] = []
        for column in range(COLUMNS):
            alpha = rgba[
                y : y + CELL_HEIGHT,
                column * CELL_WIDTH : (column + 1) * CELL_WIDTH,
                3,
            ]
            count = int(np.count_nonzero(alpha > 16))
            counts.append(count)
            if count < 1000:
                empty.append(f"{state}:{column}")
        visible_counts[state] = counts
    if empty:
        raise ValueError(f"atlas contains empty or tiny frames: {', '.join(empty)}")

    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3]
    key_distance = np.max(np.abs(rgb - np.array([0, 255, 0], dtype=np.int16)), axis=2)
    opaque_key_pixels = int(np.count_nonzero((alpha > 200) & (key_distance < 40)))
    if opaque_key_pixels:
        raise ValueError(f"atlas retains {opaque_key_pixels} opaque chroma-key pixels")
    return {
        "ok": True,
        "size": list(atlas.size),
        "rows": len(rows),
        "columns": COLUMNS,
        "visiblePixels": visible_counts,
        "opaqueChromaPixels": opaque_key_pixels,
    }


def build(source_dir: Path, manifest_path: Path, output: Path, qa_dir: Path) -> None:
    _payload, rows = _load_manifest(manifest_path)
    frames = _load_rows(source_dir)
    atlas = _assemble(rows, frames)
    report = _validate_atlas(atlas, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output, format="WEBP", lossless=True, quality=100, method=6)
    _write_contact_sheet(rows, frames, qa_dir / "contact-sheet.webp")
    _write_animation_previews(rows, frames, qa_dir / "animations")
    (qa_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qa-dir", required=True, type=Path)
    args = parser.parse_args()
    build(
        args.source_dir.expanduser().resolve(),
        args.manifest.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.qa_dir.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
