"""Prepare an optional Python environment and own one offline Valen process."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager, suppress
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading

from src.backend_v2.font_matching.config import VALEN_IDS
from src.backend_v2.font_matching.decisions import Decision, DecisionError, decode_response
from src.backend_v2.launcher.windows_job import ChildProcessJob
from src.shared.model_catalog import MODELS_BY_ID, ModelFile, file_matches, installed_root, models_root
from src.shared.path_helpers import resource_path
from src.storage_migrator.control import DataRootAlreadyLocked, DataRootLock, reject_links

TRANSFORMERS = "5.17.0"


def runtime_python(root: Path) -> Path:
    directory = root / f".valen-python-{TRANSFORMERS}"
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_ready(root: Path | None = None) -> bool:
    root = models_root() if root is None else root
    python = runtime_python(root)
    try:
        return python.is_file() and (python.parent.parent / ".ready").read_text("ascii").strip() == _torch_build()
    except (OSError, UnicodeError):
        return False


@lru_cache(maxsize=1)
def _torch_build() -> str:
    if sys.platform == "darwin":
        return "native"
    try:
        driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        count, version = ctypes.c_int(), ctypes.c_int()
        if (driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0
                or count.value < 1 or driver.cuDriverGetVersion(ctypes.byref(version)) != 0):
            return "cpu"
        for minimum, build in ((13000, "cu130"), (12080, "cu128"), (12060, "cu126")):
            if version.value >= minimum:
                return build
    except (OSError, AttributeError):
        pass
    return "cpu"


@lru_cache(maxsize=48)
def _verified_file(path: Path, file: ModelFile, _file_state: tuple[int, ...]) -> bool:
    return file_matches(path, file)


def _setup_python() -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    windows = os.name == "nt"
    candidates = [("py", ("-3.12",)), ("python3.12", ()), ("python", ())] if windows else [("python3.12", ())]
    for name, args in candidates:
        python = shutil.which(name)
        if not python:
            continue
        command = [python, *args]
        with suppress(OSError, subprocess.SubprocessError):
            version = subprocess.check_output(
                [*command, "-I", "--version"], timeout=5,
                stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if windows else 0,
            )
            if version.startswith(b"Python 3.12."):
                return command
    raise DecisionError("准备 Valen 环境需要系统安装 Python 3.12")


@contextmanager
def _owned_process(cmd: list[str], **kwargs):
    with ChildProcessJob() as job:
        process = subprocess.Popen(
            cmd, **kwargs,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        assigned = False
        try:
            job.assign(process)
            assigned = True
            yield process
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    if assigned:
                        job.close()
                    else:
                        process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait()


def prepare_runtime(root: Path, check: Callable[[], None]) -> None:
    """Called by an explicit model-management task only."""
    if runtime_ready(root):
        return
    python = runtime_python(root)
    directory = python.parent.parent
    reject_links(directory)
    build = _torch_build()
    probe = ("import torch, torchvision, transformers, PIL, av; "
             "from transformers import AutoModel, AutoProcessor; "
             f"assert transformers.__version__ == {TRANSFORMERS!r}")
    torch_index = [] if build == "native" else ["--index-url", f"https://download.pytorch.org/whl/{build}"]
    suffix = "" if build == "native" else f"+{build}"
    if build.startswith("cu"):
        probe += "; assert torch.cuda.is_available()"
    pip = [str(python), "-Im", "pip", "install", "--no-cache-dir", "--only-binary=:all:"]
    commands = [
        ("创建独立 Python 环境", [*_setup_python(), "-Im", "venv", str(directory)]),
        ("安装 PyTorch", [*pip, *torch_index, f"torch==2.11.0{suffix}", f"torchvision==0.26.0{suffix}"]),
        ("安装 Transformers", [*pip, f"transformers=={TRANSFORMERS}", "pillow", "av"]),
        ("验证独立环境", [str(python), "-Ic", probe]),
    ]
    for label, command in commands:
        check()
        try:
            with _owned_process(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) as process:
                while True:
                    check()
                    try:
                        code = process.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
        except OSError:
            raise DecisionError(f"{label}失败，请检查 Python 3.12 是否可用") from None
        if code != 0:
            raise DecisionError(f"{label}失败，请检查 Python 3.12、磁盘空间与下载网络")
    (directory / ".ready").write_text(build, "ascii")


class ValenDecisions:
    provider = "valen"
    batch_size = 1

    def __init__(self, model: str, check: Callable[[], None], *, scratch: Path) -> None:
        self.model = model
        self.check = check
        self.root = models_root()
        self.scratch = scratch

    @contextmanager
    def open(self):
        if not runtime_ready(self.root):
            raise DecisionError("请先在字体匹配设置中准备 Valen 推理环境")
        model_id = VALEN_IDS[self.model]
        model = MODELS_BY_ID[model_id]
        source = installed_root(model_id)
        # Verify imported checkpoints as well as installer downloads before
        # Transformers is allowed to execute their pinned Python code.
        for file in model.files:
            self.check()
            path = source / file.path
            reject_links(path.parent)
            reject_links(path)
            if not path.is_file():
                raise DecisionError("Valen 目录不完整或版本不符，请在模型管理中校验或修复")
            stat = path.stat()
            state = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if not _verified_file(path, file, state):
                raise DecisionError("Valen 目录不完整或版本不符，请在模型管理中校验或修复")
        lock = DataRootLock(self.root / ".valen-inference")
        try:
            lock.acquire()
        except DataRootAlreadyLocked:
            raise DecisionError("另一个 Saber 实例正在使用 Valen，请稍后重试") from None
        child = resource_path("src/backend_v2/font_matching/valen_child.py")
        # Do not forward public broker credentials, provider keys or PYTHONPATH.
        allowed = {
            "systemroot", "windir", "path", "temp", "tmp", "home", "userprofile",
            "ld_library_path", "dyld_library_path", "cuda_visible_devices", "saber_valen_device",
            "systemdrive", "appdata", "localappdata", "programdata", "programfiles", "programfiles(x86)",
            "commonprogramfiles", "commonprogramfiles(x86)", "processor_architecture", "number_of_processors",
            "comspec", "pathext", "homedrive", "homepath",
            "user", "username", "logname", "lname",
        }
        env = {k: v for k, v in os.environ.items() if k.lower() in allowed}
        try:
            with TemporaryDirectory(prefix="valen-views-", dir=self.scratch) as temporary:
                self.images = Path(temporary)
                env.update(HF_HOME=str(self.images), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                with _owned_process(
                    [str(runtime_python(self.root)), "-Iu", child, str(source / model.directory),
                     str(sum(file.size for file in model.files if file.path.endswith(".safetensors")))],
                    cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                ) as process:
                    self.process = process
                    self.replies = queue.Queue()
                    reader = threading.Thread(target=self._read_replies, daemon=True)
                    reader.start()
                    try:
                        ready = self._reply()
                        self.cache_identity = (model.revision, "valen-native-single-v1", TRANSFORMERS, ready["device"],
                                               ready["precision"], ready["torch"])
                        yield self
                    finally:
                        if process.stdin:
                            with suppress(BrokenPipeError):
                                process.stdin.close()
                        with suppress(subprocess.TimeoutExpired):
                            process.wait(timeout=1)
        finally:
            lock.release()

    def _read_replies(self) -> None:
        with self.process.stdout as stream:
            for line in stream:
                self.replies.put(line)
        self.replies.put(None)

    def _reply(self) -> dict:
        while True:
            self.check()
            try:
                raw = self.replies.get(timeout=0.5)
            except queue.Empty:
                continue
            if raw is None:
                raise DecisionError("Valen 推理进程退出，请检查设备内存和独立环境")
            try:
                reply = json.loads(raw)
            except ValueError:
                raise DecisionError("Valen 运行环境返回了无效结果") from None
            if "error" in reply:
                raise DecisionError(reply["error"])
            return reply

    def score(self, views, questions, context, check) -> Decision:
        answers, usage = {}, {}
        base = [{"type": "text", "text": context}]

        def media(index: int, image: bytes) -> dict:
            path = self.images / f"{index}.png"
            path.write_bytes(image)
            return {"type": "image_url", "image_url": {"url": str(path)}}

        base.extend(media(index, image) for index, image in enumerate(views))
        for question in questions:
            check()
            state = base + ([media(len(views), question.image)] if question.image else [])
            request = {
                "state": {"messages": [{"role": "user", "content": state}]},
                "questions": {
                    question.name: {"type": "score", "instructions": question.instruction, "criteria": question.levels},
                },
            }
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self.process.stdin.flush()
            except OSError:
                raise DecisionError("Valen 推理进程连接中断") from None
            result = decode_response(self._reply(), self.provider, self.model)
            answers.update(result.answers)
            for key, value in result.usage.items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
        return Decision(self.model, answers, usage)
