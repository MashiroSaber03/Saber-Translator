from PIL import Image, ImageDraw

from scripts.build_desktop_pet_assets import (
    CELL_HEIGHT, CELL_WIDTH, _normalize_row, _write_animation_previews,
)


def test_walk_registration_does_not_follow_foot_reach() -> None:
    frames = []
    for foot_x in (5, 25, 45, 65, 85, 65, 45, 25):
        frame = Image.new("RGBA", (110, 180))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((35, 0, 65, 100), fill="brown")
        draw.polygon([(45, 100), (55, 100), (foot_x + 10, 179), (foot_x, 179)], fill="black")
        frames.append(frame)
    original = _normalize_row(frames)
    aligned = _normalize_row(frames, align_head=True)
    centers = []
    for before, after in zip(original, aligned):
        alpha = after.getchannel("A")
        left, _, right, _ = alpha.crop((0, 0, CELL_WIDTH, CELL_HEIGHT // 3)).getbbox()
        centers.append((left + right) / 2)
        assert sum(before.getchannel("A").histogram()[1:]) == sum(alpha.histogram()[1:])
    assert max(centers) - min(centers) <= 1


def test_animation_previews_respect_one_shot_and_loop_timing(tmp_path) -> None:
    frames = [Image.new("RGBA", (192, 208), color) for color in ("red", "blue")]
    rows = [
        {"state": "idle", "durationsMs": [1200, 60], "loop": True},
        {"state": "greeting", "durationsMs": [100, 200], "loop": False},
    ]
    _write_animation_previews(rows, {"idle": frames, "greeting": frames}, tmp_path)
    with Image.open(tmp_path / "idle.gif") as image:
        assert image.info["loop"] == 0
        assert image.info["duration"] == 1200
        image.seek(1)
        assert image.info["duration"] == 60
    with Image.open(tmp_path / "greeting.gif") as image:
        assert "loop" not in image.info
