# -*- mode: python ; coding: utf-8 -*-
"""Backend-first production PyInstaller specification.

Build with ``pyinstaller app.spec --noconfirm`` after the Vue production build.
The resulting executable defaults to the native desktop shell, which owns the
same isolated API and Worker lifecycle used by the terminal Launcher role.
"""

import os
import shutil
from importlib.util import find_spec
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all, copy_metadata

block_cipher = None

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC))

# ===================== 初始化收集列表 =====================
datas = []
binaries = []
hiddenimports = []
module_collection_mode = {
    # TorchScript/inspect 需要在运行时访问原始 .py 文件；仅有 PYZ 内字节码不够。
    'litelama': 'pyz+py',
    'kornia': 'pyz+py',
}

# ===================== 项目资源文件 =====================
# 1. v2 静态资源、内置字体、契约和存储基线
datas.append((os.path.join(PROJECT_ROOT, 'src', 'backend_v2', 'static'), os.path.join('src', 'backend_v2', 'static')))
datas.append((os.path.join(PROJECT_ROOT, 'src', 'backend_v2', 'resources'), os.path.join('src', 'backend_v2', 'resources')))
datas.append((os.path.join(PROJECT_ROOT, 'src', 'backend_v2', 'desktop', 'assets'), os.path.join('src', 'backend_v2', 'desktop', 'assets')))
datas.append((os.path.join(PROJECT_ROOT, 'src', 'shared', 'text_style_defaults_factory.json'), os.path.join('src', 'shared')))
datas.append((os.path.join(PROJECT_ROOT, 'src', 'shared', 'prompt_defaults_factory.json'), os.path.join('src', 'shared')))
datas.append((os.path.join(PROJECT_ROOT, 'src', 'shared', 'ai_provider_manifest.json'), os.path.join('src', 'shared')))
datas.append((os.path.join(PROJECT_ROOT, 'src', 'backend_v2', 'plugins', 'plugin_builder_skill.md'), os.path.join('src', 'backend_v2', 'plugins')))
datas.append((os.path.join(PROJECT_ROOT, 'openapi', 'v2.yaml'), 'openapi'))
datas.append((
    os.path.join(PROJECT_ROOT, 'src', 'backend_v2', 'storage', 'migrations'),
    os.path.join('src', 'backend_v2', 'storage', 'migrations'),
))

# 2. 用户设置、提示词和凭据只存在于 v2 data root 的数据库中，
# 不把任何运行时配置或密钥打进应用包。

# 3. 模型文件 - 包含所有模型
models_path = os.path.join(PROJECT_ROOT, 'models')
if os.path.exists(models_path):
    datas.append((models_path, 'models'))

# 4. 插件目录
plugins_path = os.path.join(PROJECT_ROOT, 'plugins')

# ===================== 关键: 使用 collect_all 完整收集库 =====================
# transformers 使用动态导入，必须用 collect_all 完整收集
critical_packages = [
    'transformers',      # 关键! 解决动态导入问题
    'manga_ocr',
    'tokenizers',
    'huggingface_hub',
    'safetensors',
    'sentencepiece',         # PaddleOCR-VL tokenizer 必需
    'rapidocr_onnxruntime',  # PaddleOCR ONNX 版本
    'onnxruntime',           # ONNX 推理引擎 (GPU/CPU 模块名相同)
    'ultralytics',           # YOLO 检测器
    'chromadb',              # 向量数据库 (manga_insight)
    'gallery_dl',            # 网页导入 - Gallery-DL 引擎
]

for pkg in critical_packages:
    if find_spec(pkg) is None:
        raise ModuleNotFoundError(f"required packaging dependency is missing: {pkg}")
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports
    print(f"[SPEC] collect_all({pkg}): OK")

# accelerate 只由 GPU 版的 transformers device_map 使用。CPU 依赖集不安装
# 它，因此仅在当前构建环境实际存在时收集。
optional_packages = ['accelerate']
installed_optional_packages = [
    pkg for pkg in optional_packages if find_spec(pkg) is not None
]
for pkg in installed_optional_packages:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports
    print(f"[SPEC] collect_all({pkg}): OK (optional)")

# 其他库的数据文件
for pkg in ['unidic_lite', 'fugashi', 'litelama']:
    datas += collect_data_files(pkg)
    print(f"[SPEC] collect_data_files({pkg}): OK")

# 收集元数据
metadata_packages = [
    'transformers',
    'tokenizers',
    'huggingface_hub',
    'safetensors',
    'manga_ocr',
    'sentencepiece',
] + installed_optional_packages
for pkg in metadata_packages:
    datas += copy_metadata(pkg)

# ===================== 隐藏导入 =====================
hiddenimports += [
    # Backend-first role dispatcher and persistence
    'src.backend_v2', 'src.backend_v2.dispatch', 'src.backend_v2.paths',
    'src.backend_v2.runtime_identity', 'src.backend_v2.import_guard',
    'src.backend_v2.api', 'src.backend_v2.api.app', 'src.backend_v2.api.entrypoint',
    'src.backend_v2.worker', 'src.backend_v2.worker.entrypoint',
    'src.backend_v2.launcher', 'src.backend_v2.launcher.entrypoint',
    'src.backend_v2.launcher.windows_job',
    'src.backend_v2.desktop', 'src.backend_v2.desktop.entrypoint',
    'src.backend_v2.desktop.settings', 'src.backend_v2.desktop.task_client',
    'src.backend_v2.desktop.pet_state', 'src.backend_v2.desktop.pet',
    'src.backend_v2.desktop.theme', 'src.backend_v2.desktop.window',
    'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtNetwork',
    'PySide6.QtWidgets',
    'sqlalchemy', 'alembic', 'waitress',

    # Flask 相关
    'flask', 'werkzeug', 'werkzeug.serving', 'jinja2', 'itsdangerous', 'click',
    
    # Worker 算法实现（仍由 v2 services 在 Worker 内按需导入）
    'src',
    'src.core', 'src.core.detection', 'src.core.ocr', 'src.core.translation', 'src.core.inpainting',
    'src.core.rendering', 'src.core.config_models',
    'src.core.large_image_detection',  # 大图片检测包装器
    
    # backend v2 仍复用的 Manga Insight 模型和传输适配器
    'src.core.manga_insight', 'src.core.manga_insight.config_models',
    'src.core.manga_insight.embedding_client', 'src.core.manga_insight.vlm_client',
    'src.core.manga_insight.clients', 'src.core.manga_insight.clients.image_gen_client',

    # backend v2 immutable plugin agent metadata/controller
    'src.core.plugin_agent', 'src.core.plugin_agent.controller', 'src.core.plugin_agent.models',
    
    # core.detector (关键 - 检测器框架)
    'src.core.detector', 'src.core.detector.registry', 'src.core.detector.base',
    'src.core.detector.data_types', 'src.core.detector.geometry', 'src.core.detector.postprocess',
    'src.core.detector.textline_merge',
    'src.core.detector.panel_detector', 'src.core.detector.smart_sort',  # 面板检测和智能排序
    'src.core.detector.backends', 'src.core.detector.backends.ctd_backend',
    'src.core.detector.backends.default_backend', 'src.core.detector.backends.yolo_backend',
    
    # interfaces 基础
    'src.interfaces', 'src.interfaces.manga_ocr_interface', 'src.interfaces.paddle_ocr_onnx_interface',
    'src.interfaces.baidu_ocr_interface', 'src.interfaces.baidu_translate_interface',
    'src.interfaces.youdao_translate_interface', 'src.interfaces.lama_interface', 'src.interfaces.vision_interface',
    
    # interfaces.default (DBNet 检测器)
    'src.interfaces.default', 'src.interfaces.default.DBHead',
    'src.interfaces.default.DBNet_resnet34', 'src.interfaces.default.imgproc',
    
    # interfaces.lama_mpe
    'src.interfaces.lama_mpe_interface',
    
    # interfaces.ocr_48px (48px OCR 和颜色提取)
    'src.interfaces.ocr_48px', 'src.interfaces.ocr_48px.core',
    'src.interfaces.ocr_48px.interface', 'src.interfaces.ocr_48px.xpos',
    
    # interfaces.paddleocr_vl (PaddleOCR-VL 日漫专用 OCR)
    'src.interfaces.paddleocr_vl_interface',
    
    # core.color_extractor (颜色提取模块)
    'src.core.color_extractor',
    
    # interfaces.ctd (当前 CTD 推理图与后处理)
    'src.interfaces.ctd', 'src.interfaces.ctd.detector', 'src.interfaces.ctd.basemodel',
    # ctd.utils 子模块
    'src.interfaces.ctd.utils', 'src.interfaces.ctd.utils.db_utils', 'src.interfaces.ctd.utils.imgproc_utils',
    'src.interfaces.ctd.utils.yolov5_utils',
    # ctd.yolov5 子模块
    'src.interfaces.ctd.yolov5',
    'src.interfaces.ctd.yolov5.common',
    'src.interfaces.ctd.yolov5.yolo',
    
    # shared runtime helpers
    'src.shared', 'src.shared.constants', 'src.shared.path_helpers',
    'src.shared.image_helpers',
    'src.shared.openai_helpers',  # OpenAI 客户端辅助函数
    
    # PyTorch
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.utils', 'torch.utils.data', 'torch.jit', 'torch.cuda',
    'torchvision', 'torchvision.transforms', 'torchvision.models', 'torchvision.ops',
    
    # RapidOCR (PaddleOCR ONNX 版本)
    'rapidocr_onnxruntime', 'onnxruntime',
    
    # MangaOCR
    'manga_ocr', 'manga_ocr.ocr',
    
    # 图像处理
    'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont', 'cv2', 'numpy', 'scipy', 'scipy.ndimage',
    
    # 其他
    'litelama', 'openai', 'httpx', 'yaml', 'colorama', 'loguru', 'requests', 'urllib3', 'certifi',
    'tqdm', 'regex', 'filelock', 'packaging', 'psutil',
    'fugashi', 'unidic_lite', 'jaconv', 'einops', 'kornia', 'omegaconf', 'polars',
    'shapely', 'pyclipper', 'networkx', 'multiprocessing', 'concurrent.futures',
    'freetype',  # 字体回退支持 (rendering.py)
    
    # PaddleOCR-VL tokenizer 依赖；GPU 构建的 accelerate 由上方按环境收集
    'sentencepiece',
    
    # manga_insight 依赖
    'chromadb',
    
    # ultralytics/YOLO 相关
    'ultralytics', 'pandas', 'dill',
    
    # asyncio (textline_merge 需要)
    'asyncio',
    
    # 电子书处理
    'mobi', 'fitz', 'pymupdf',
    
    # utils 模块
    'src.utils', 'src.utils.image_rearrange',
]

# Collect submodules
print("[SPEC] Collecting submodules...")
for mod in ['src.backend_v2', 'flask', 'werkzeug', 'jinja2', 'torch', 'torchvision', 'onnxruntime', 'safetensors', 'ultralytics', 'networkx', 'kornia', 'litelama']:
    hiddenimports += collect_submodules(mod)

# ===================== 排除项 =====================
excludes = [
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2',
    'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx', 'docutils',
    # 不需要的子模块（避免警告）
    'onnx', 'tensorboard', 'timm',
    'onnxruntime.quantization',  # 量化功能不需要
    'torch.utils.tensorboard',   # 训练可视化不需要
]

# ===================== Analysis =====================
print("[SPEC] Starting analysis...")
a = Analysis(
    ['saber_v2.py'],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode=module_collection_mode,
)

# ===================== 打包 =====================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Saber-Translator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    hide_console='hide-early',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, 'src', 'backend_v2', 'desktop', 'assets', 'app-icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Saber-Translator',
)

bundle_plugins_path = os.path.join(coll.name, 'plugins')
if os.path.exists(bundle_plugins_path):
    shutil.rmtree(bundle_plugins_path)
if os.path.exists(plugins_path):
    shutil.copytree(
        plugins_path,
        bundle_plugins_path,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'),
    )
    for name in os.listdir(bundle_plugins_path):
        candidate = os.path.join(bundle_plugins_path, name)
        if (
            os.path.isdir(candidate)
            and not os.path.isfile(os.path.join(candidate, 'plugin.json'))
        ):
            shutil.rmtree(candidate)
