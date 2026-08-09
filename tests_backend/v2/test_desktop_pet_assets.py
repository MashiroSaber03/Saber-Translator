from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from src.backend_v2.desktop.pet import PetManifest, PetWindow
from src.backend_v2.desktop.pet_state import PetState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PET_ROOT = PROJECT_ROOT / "src" / "backend_v2" / "desktop" / "assets" / "pet" / "saber_chan"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_project_pet_atlas_matches_custom_manifest() -> None:
    manifest = PetManifest.load(PET_ROOT / "pet.json")
    atlas = QImage(str(manifest.spritesheet_path))

    assert not atlas.isNull()
    manifest.validate_image(atlas)
    assert atlas.width() == 1536
    assert atlas.height() == 2704
    assert set(manifest.animations) == set(PetState)


def test_drag_left_is_exact_mirror_of_drag_right() -> None:
    manifest = PetManifest.load(PET_ROOT / "pet.json")
    atlas = QImage(str(manifest.spritesheet_path))
    right_row = manifest.animations[PetState.DRAG_RIGHT].row
    left_row = manifest.animations[PetState.DRAG_LEFT].row

    for column in range(manifest.columns):
        right = atlas.copy(
            column * manifest.cell_width,
            right_row * manifest.cell_height,
            manifest.cell_width,
            manifest.cell_height,
        )
        left = atlas.copy(
            column * manifest.cell_width,
            left_row * manifest.cell_height,
            manifest.cell_width,
            manifest.cell_height,
        )
        for y in range(manifest.cell_height):
            for x in range(manifest.cell_width):
                left_pixel = left.pixelColor(x, y)
                right_pixel = right.pixelColor(manifest.cell_width - 1 - x, y)
                assert left_pixel.alpha() == right_pixel.alpha()
                if left_pixel.alpha():
                    assert left_pixel == right_pixel


def test_non_looping_task_reaction_holds_its_last_frame() -> None:
    _app()
    pet = PetWindow(
        PET_ROOT / "pet.json",
        fallback_logo=PROJECT_ROOT / "pic" / "logo.png",
    )
    pet.set_base_state(PetState.SUCCESS)
    animation = pet._manifest.animations[PetState.SUCCESS]  # type: ignore[union-attr]

    for _index in range(animation.frame_count):
        pet._timer.stop()
        pet._advance_frame()

    assert pet._visible_state == PetState.SUCCESS
    assert pet._frame_index == animation.frame_count - 1
    assert not pet._timer.isActive()
    pet.close()
