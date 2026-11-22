"""Build the pinned Rust core for Python workers and the browser editor.

Run after installing Rust 1.95.0, the wasm32-unknown-unknown target, and
wasm-bindgen-cli 0.2.129. End users only need the resulting packaged files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys
import tomllib
from collections.abc import Iterator
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "photocraft"
RUNTIME_APPS = ("saber-editor", "photocraft-web")
WEB_HOST_FILES = ("index.html", "host.js")
sys.path.insert(0, str(ROOT))

from src.backend_v2.editor_runtime import (
    NOTICE_FILES, STAMP, bundle_name, checksum, pinned_build, runtime_files, runtime_ready, runtime_stamp,
)


def runtime_crate_inputs(source: Path, workspace: dict) -> Iterator[Path]:
    workspace_dependencies = workspace["workspace"].get("dependencies", {})
    pending = [source / "apps" / name for name in RUNTIME_APPS]
    seen = set()
    while pending:
        crate = pending.pop().resolve()
        if crate in seen:
            continue
        seen.add(crate)
        manifest = crate / "Cargo.toml"
        yield manifest
        config = tomllib.loads(manifest.read_text(encoding="utf-8"))
        for table in (config, *config.get("target", {}).values()):
            for section in ("dependencies", "build-dependencies"):
                for name, dependency in table.get(section, {}).items():
                    if not isinstance(dependency, dict):
                        continue
                    base = crate
                    if dependency.get("workspace"):
                        dependency = workspace_dependencies.get(name)
                        base = source
                    if isinstance(dependency, dict) and "path" in dependency:
                        pending.append(base / dependency["path"])
        for path in (crate / "src").rglob("*"):
            if (path.is_file() and "tests" not in path.relative_to(crate).parts
                    and path.name != "tests.rs" and not path.name.endswith("_tests.rs")):
                yield path
        build = config.get("package", {}).get("build", "build.rs")
        if isinstance(build, str) and (crate / build).is_file():
            yield crate / build


def source_fingerprint(root: Path) -> str:
    """Content identity independent of checkout location and Git line endings."""
    root = root.resolve()
    source = root / "third_party/photocraft"
    workspace = source / "Cargo.toml"
    inputs = {workspace, source / "Cargo.lock", root / "scripts/build_editor.py",
              root / "src/backend_v2/editor_runtime.py", root / ".github/workflows/editor-runtime.yml"}
    inputs.update(runtime_crate_inputs(source, tomllib.loads(workspace.read_text(encoding="utf-8"))))
    inputs.update(source / name for name in NOTICE_FILES)
    for directory in ("assets", ".cargo", "contributors"):
        inputs.update(path for path in (source / directory).rglob("*") if path.is_file())
    inputs.update(root / "scripts/editor" / name for name in WEB_HOST_FILES)
    digest = hashlib.sha256()
    for path in sorted(inputs, key=lambda item: item.relative_to(root).as_posix()):
        data = path.read_bytes()
        if b"\0" not in data:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                data = data.replace(b"\r\n", b"\n")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()[:24]


def package_runtime(root: Path, output: Path, build: str) -> Path:
    if not runtime_ready(root, build):
        raise ValueError("Build the complete matching native and web runtime before packaging")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / bundle_name(build)
    with ZipFile(archive, "w", ZIP_DEFLATED) as package:
        for path in (*runtime_files(), STAMP):
            package.write(root / path, path.as_posix())
    archive.with_suffix(".zip.sha256").write_text(f"{checksum(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def run(*command):
    subprocess.run(command, cwd=SOURCE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--web-only", action="store_true")
    parser.add_argument("--pin", action="store_true", help="Update the source fingerprint; no Rust required")
    parser.add_argument("--check", action="store_true", help="Verify and print the pinned fingerprint; no Rust required")
    parser.add_argument("--package", type=Path, help="Write the source-user runtime ZIP to this directory")
    parser.add_argument("--package-only", action="store_true", help="Package an already complete build without compiling")
    args = parser.parse_args()
    build = source_fingerprint(ROOT)
    if args.pin:
        (ROOT / "editor-runtime.json").write_text(json.dumps({"build": build}, indent=2) + "\n", encoding="utf-8")
        print(build)
        return
    if pinned_build(ROOT) != build:
        raise SystemExit("Editor source changed: run python scripts/build_editor.py --pin and commit editor-runtime.json")
    if args.check:
        print(build)
        return
    if args.package_only:
        if not args.package:
            parser.error("--package-only requires --package")
        print(package_runtime(ROOT, args.package, build))
        return
    if args.native_only and args.web_only:
        parser.error("--native-only and --web-only cannot be combined")
    cargo = shutil.which("cargo")
    if not cargo:
        raise SystemExit("Install Rust 1.95.0 and add cargo to PATH")
    (ROOT / STAMP).unlink(missing_ok=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", SOURCE / "target")).resolve()
    if not args.web_only:
        run(cargo, "build", "--locked", "--release", "-p", "saber-editor")
        name = "saber-editor.exe" if os.name == "nt" else "saber-editor"
        (ROOT / "native").mkdir(exist_ok=True)
        native_output = target / os.environ.get("CARGO_BUILD_TARGET", "") / "release" / name
        shutil.copy2(native_output, ROOT / "native" / name)
    if not args.native_only:
        bindgen = shutil.which("wasm-bindgen")
        if not bindgen:
            raise SystemExit("Install wasm-bindgen-cli 0.2.129: cargo install wasm-bindgen-cli --version 0.2.129 --locked")
        run(cargo, "build", "--locked", "--profile", "wasm-release", "--target", "wasm32-unknown-unknown", "-p", "photocraft-web")
        output = ROOT / "src" / "backend_v2" / "static" / "photocraft"
        output.mkdir(exist_ok=True, parents=True)
        run(bindgen, str(target / "wasm32-unknown-unknown" / "wasm-release" / "photocraft-web.wasm"),
            "--target", "web", "--out-name", "photocraft", "--out-dir", str(output))
        for filename in WEB_HOST_FILES:
            shutil.copy2(ROOT / "scripts" / "editor" / filename, output / filename)
    if not args.native_only and not args.web_only:
        for name in NOTICE_FILES:
            notice = ROOT / "native/licenses" / name
            notice.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, notice)
        (ROOT / STAMP).write_text(json.dumps(runtime_stamp(ROOT, build), indent=2) + "\n", encoding="utf-8")
    if args.package:
        print(package_runtime(ROOT, args.package, build))


if __name__ == "__main__":
    main()
