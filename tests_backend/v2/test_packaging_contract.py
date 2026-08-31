from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from src.backend_v2.dispatch import _parser


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_packaged_entrypoint_defaults_to_desktop_shell() -> None:
    defaults = _parser().parse_args([])
    assert defaults.role == "desktop"
    assert defaults.host == "127.0.0.1"


def test_production_spec_uses_only_the_backend_first_entrypoint() -> None:
    spec = (PROJECT_ROOT / "app.spec").read_text(encoding="utf-8")

    assert "'saber_v2.py'" in spec
    assert "name='Saber-Translator'" in spec
    assert "'src.backend_v2.api.entrypoint'" in spec
    assert "'src.backend_v2.worker.entrypoint'" in spec
    assert "'src.backend_v2.launcher.entrypoint'" in spec
    assert "'src.backend_v2.desktop.entrypoint'" in spec
    assert "'PySide6'" not in spec.split("excludes =", 1)[1].split("]", 1)[0]
    assert "'desktop', 'assets'" in spec
    assert "'desktop', 'assets', 'app-icon.ico'" in spec
    assert "hide_console='hide-early'" in spec
    assert "'openapi', 'v2.yaml'" in spec
    assert "'src', 'backend_v2', 'static'" in spec
    assert "'src', 'backend_v2', 'resources'" in spec
    assert "'src', 'shared', 'prompt_defaults_factory.json'" in spec
    assert "datas.append((pic_path, 'pic'))" not in spec
    assert "'rapidocr'" in spec
    assert "accelerate" not in spec
    assert "required packaging dependency is missing" in spec
    assert "shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo')" in spec
    assert "os.stat(bundle_plugins_path).st_mode | stat.S_IWRITE" in spec
    assert "'plugin.json'" in spec
    assert "collect_all({pkg}) FAILED" not in spec
    assert "except:" not in spec
    assert "'app.py'" not in spec
    assert "'src.app" not in spec
    assert not (PROJECT_ROOT / "saber_v2.spec").exists()


def test_production_spec_imports_its_dependency_probe() -> None:
    spec = (PROJECT_ROOT / "app.spec").read_text(encoding="utf-8")

    assert "from importlib.util import find_spec" in spec


def test_production_spec_does_not_bundle_foreign_windows_icu() -> None:
    spec = (PROJECT_ROOT / "app.spec").read_text(encoding="utf-8")

    assert "os.path.basename(entry[0]).lower() != 'icuuc.dll'" in spec


def test_production_spec_has_no_missing_project_hidden_imports() -> None:
    spec_path = PROJECT_ROOT / "app.spec"
    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))
    hidden_imports: set[str] = set()

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "hiddenimports"
            and isinstance(node.value, ast.List)
        ):
            continue
        hidden_imports.update(
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and (element.value == "src" or element.value.startswith("src."))
        )

    missing = []
    for module_name in sorted(hidden_imports):
        module_path = PROJECT_ROOT.joinpath(*module_name.split("."))
        if not module_path.is_dir() and not module_path.with_suffix(".py").is_file():
            missing.append(module_name)

    assert missing == []


def test_production_spec_lists_only_current_ctd_runtime_modules() -> None:
    spec = (PROJECT_ROOT / "app.spec").read_text(encoding="utf-8")

    for module_name in (
        "src.interfaces.ctd.basemodel",
        "src.interfaces.ctd.detector",
        "src.interfaces.ctd.utils.db_utils",
        "src.interfaces.ctd.utils.imgproc_utils",
        "src.interfaces.ctd.utils.yolov5_utils",
        "src.interfaces.ctd.yolov5.common",
        "src.interfaces.ctd.yolov5.yolo",
    ):
        assert repr(module_name) in spec
    assert "完整 - 包含所有子模块" not in spec


def test_release_workflow_builds_the_triggering_revision() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "github.event.inputs.branch" not in workflow
    assert "default: 'release'" not in workflow
    assert "target_commitish: ${{ github.sha }}" in workflow
    assert "EVENT_NAME: ${{ github.event_name }}" in workflow
    assert 'echo "PRERELEASE=true" >> "$GITHUB_OUTPUT"' in workflow
    assert "prerelease: ${{ steps.version.outputs.PRERELEASE == 'true' }}" in workflow
    assert "git status --porcelain -- src/backend_v2/static/vue" in workflow
    assert "python -m pytest tests_backend -q" in workflow
    assert "npm test" in workflow
    assert "npm run lint:ui:audit" in workflow
    assert 'echo "Build result: ${{ needs.build.result }}"' in workflow
    assert 'if [ "${{ needs.build.result }}" = "success" ]; then' in workflow


def test_direct_runtime_dependencies_are_declared() -> None:
    direct_dependencies = {
        "fonttools": "fontTools",
        "img2pdf": "img2pdf",
        "numpy": "numpy",
        "packaging": "packaging",
        "reportlab": "reportlab",
    }
    for filename in ("requirements-cpu.txt", "requirements-gpu.txt"):
        requirements = {
            line.strip().split("#", 1)[0].strip().lower()
            for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
        }
        assert direct_dependencies.keys() <= requirements

    for import_name in direct_dependencies.values():
        assert importlib.util.find_spec(import_name) is not None


def test_manga_ocr_is_pinned_to_the_transformers_5_compatible_release() -> None:
    for filename in ("requirements-cpu.txt", "requirements-gpu.txt"):
        requirements = {
            line.strip().split("#", 1)[0].strip().lower()
            for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
        }
        assert "manga-ocr==0.1.16" in requirements
        assert "transformers==5.15.0" in requirements
