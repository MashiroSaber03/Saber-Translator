# -*- mode: python ; coding: utf-8 -*-
"""
Saber-Translator PyInstaller Spec 文件
打包命令: pyinstaller app.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all, copy_metadata

block_cipher = None

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC))

# ===================== 初始化收集列表 =====================
datas = []
binaries = []
hiddenimports = []

# ===================== 项目资源文件 =====================
# 1. 静态资源 (CSS, JS, 字体, 图标)
datas.append((os.path.join(PROJECT_ROOT, 'src', 'app', 'static'), os.path.join('src', 'app', 'static')))

# 2. HTML 模板
datas.append((os.path.join(PROJECT_ROOT, 'src', 'app', 'templates'), os.path.join('src', 'app', 'templates')))

# 3. 配置文件 - 不打包用户运行时配置
# user_settings.json, prompts.json, model_history.json 等会在运行时自动生成
# 不打包以避免泄露 API 密钥等敏感信息
# config 目录会在运行时由程序自动创建

# 4. 模型文件 - 包含所有模型
models_path = os.path.join(PROJECT_ROOT, 'models')
if os.path.exists(models_path):
    datas.append((models_path, 'models'))

# 5. 插件目录
plugins_path = os.path.join(PROJECT_ROOT, 'plugins')
if os.path.exists(plugins_path):
    datas.append((plugins_path, 'plugins'))

# 6. src/plugins 目录 (内置插件)
src_plugins_path = os.path.join(PROJECT_ROOT, 'src', 'plugins')
if os.path.exists(src_plugins_path):
    datas.append((src_plugins_path, os.path.join('src', 'plugins')))

# 7. YOLOv5 接口目录 (模型和仓库)
yolov5_path = os.path.join(PROJECT_ROOT, 'src', 'interfaces', 'yolov5')
if os.path.exists(yolov5_path):
    # models 目录
    yolov5_models = os.path.join(yolov5_path, 'models')
    if os.path.exists(yolov5_models):
        datas.append((yolov5_models, os.path.join('src', 'interfaces', 'yolov5', 'models')))
    # repo 目录
    yolov5_repo = os.path.join(yolov5_path, 'repo')
    if os.path.exists(yolov5_repo):
        datas.append((yolov5_repo, os.path.join('src', 'interfaces', 'yolov5', 'repo')))

# 8. 图片资源
pic_path = os.path.join(PROJECT_ROOT, 'pic')
if os.path.exists(pic_path):
    datas.append((pic_path, 'pic'))

# ===================== 关键: 使用 collect_all 完整收集库 =====================
# transformers 使用动态导入，必须用 collect_all 完整收集
critical_packages = [
    'transformers',      # 关键! 解决动态导入问题
    'manga_ocr',
    'tokenizers',
    'huggingface_hub',
    'safetensors',
    'rapidocr_onnxruntime',  # PaddleOCR ONNX 版本
    'onnxruntime',           # ONNX 推理引擎 (GPU/CPU 模块名相同)
    'ultralytics',           # YOLO 检测器
]

for pkg in critical_packages:
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
        print(f"[SPEC] collect_all({pkg}): OK")
    except Exception as e:
        print(f"[SPEC] collect_all({pkg}) 失败: {e}")

# 其他库的数据文件
for pkg in ['rapidocr_onnxruntime', 'unidic_lite', 'fugashi', 'litelama']:
    try:
        datas += collect_data_files(pkg)
        print(f"[SPEC] collect_data_files({pkg}): OK")
    except Exception as e:
        print(f"[SPEC] collect_data_files({pkg}) 失败: {e}")

# 收集元数据
for pkg in ['transformers', 'tokenizers', 'huggingface_hub', 'safetensors', 'manga_ocr']:
    try:
        datas += copy_metadata(pkg)
    except:
        pass

# ===================== 隐藏导入 =====================
hiddenimports += [
    # Flask 相关
    'flask', 'flask_cors', 'werkzeug', 'werkzeug.serving', 'jinja2', 'itsdangerous', 'click',
    
    # ========== 项目内部模块 (完整版) ==========
    # app 基础
    'src', 'src.app', 'src.app.routes',
    
    # app.api
    'src.app.api', 'src.app.api.config_api', 'src.app.api.session_api', 
    'src.app.api.bookshelf_api', 'src.app.api.api_docs',
    
    # app.api.system (完整)
    'src.app.api.system', 'src.app.api.system.tests', 'src.app.api.system.cleanup',
    'src.app.api.system.downloads', 'src.app.api.system.files', 
    'src.app.api.system.fonts', 'src.app.api.system.plugins',
    
    # app.api.translation (完整)
    'src.app.api.translation', 'src.app.api.translation.routes',
    'src.app.api.translation.translate_api', 'src.app.api.translation.high_quality_api',
    
    'src.app.error_handlers', 'src.app.route_redirects',
    
    # core (完整)
    'src.core', 'src.core.detection', 'src.core.ocr', 'src.core.translation', 'src.core.inpainting',
    'src.core.rendering', 'src.core.processing', 'src.core.session_manager', 'src.core.bookshelf_manager',
    'src.core.pdf_processor', 'src.core.config_models', 'src.core.types_enhanced',
    
    # core.detector (关键 - 检测器框架)
    'src.core.detector', 'src.core.detector.registry', 'src.core.detector.base',
    'src.core.detector.data_types', 'src.core.detector.geometry', 'src.core.detector.postprocess',
    'src.core.detector.textline_merge',
    'src.core.detector.backends', 'src.core.detector.backends.ctd_backend',
    'src.core.detector.backends.default_backend', 'src.core.detector.backends.yolo_backend',
    'src.core.detector.backends.yolov5_backend',
    
    # interfaces 基础
    'src.interfaces', 'src.interfaces.manga_ocr_interface', 'src.interfaces.paddle_ocr_interface', 'src.interfaces.paddle_ocr_onnx_interface',
    'src.interfaces.baidu_ocr_interface', 'src.interfaces.baidu_translate_interface',
    'src.interfaces.youdao_translate_interface', 'src.interfaces.lama_interface', 'src.interfaces.vision_interface',
    
    # interfaces.default (DBNet 检测器)
    'src.interfaces.default', 'src.interfaces.default.DBHead',
    'src.interfaces.default.DBNet_resnet34', 'src.interfaces.default.imgproc',
    
    # interfaces.yolov5
    'src.interfaces.yolov5', 'src.interfaces.yolov5.detector',
    
    # interfaces.lama_mpe
    'src.interfaces.lama_mpe_interface',
    
    # interfaces.ctd (完整 - 包含所有子模块)
    'src.interfaces.ctd', 'src.interfaces.ctd.detector', 'src.interfaces.ctd.basemodel', 'src.interfaces.ctd.textmask',
    # ctd.core 子模块
    'src.interfaces.ctd.core', 'src.interfaces.ctd.core.generic', 'src.interfaces.ctd.core.generic2',
    'src.interfaces.ctd.core.textblock', 'src.interfaces.ctd.core.textline_merge',
    # ctd.utils 子模块
    'src.interfaces.ctd.utils', 'src.interfaces.ctd.utils.db_utils', 'src.interfaces.ctd.utils.imgproc_utils',
    'src.interfaces.ctd.utils.weight_init', 'src.interfaces.ctd.utils.yolov5_utils',
    # ctd.yolov5 子模块
    'src.interfaces.ctd.yolov5',
    'src.interfaces.ctd.yolov5.common',
    'src.interfaces.ctd.yolov5.yolo',
    
    # shared (完整)
    'src.shared', 'src.shared.constants', 'src.shared.path_helpers', 'src.shared.config_loader',
    'src.shared.exceptions', 'src.shared.image_helpers', 'src.shared.performance', 'src.shared.types', 'src.shared.validators',
    
    # plugins
    'src.plugins', 'src.plugins.base', 'src.plugins.manager', 'src.plugins.hooks',
    
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
    'tqdm', 'regex', 'filelock', 'packaging', 'psutil', 'PyPDF2',
    'fugashi', 'unidic_lite', 'jaconv', 'einops', 'kornia', 'omegaconf', 'polars',
    'shapely', 'pyclipper', 'networkx', 'multiprocessing', 'concurrent.futures',
    
    # ultralytics/YOLO 相关
    'ultralytics', 'pandas', 'dill',
    
    # asyncio (textline_merge 需要)
    'asyncio',
    
    # YOLOv5 repo 依赖
    'matplotlib', 'seaborn',
]

# 收集子模块
print("[SPEC] 收集子模块...")
for mod in ['flask', 'werkzeug', 'jinja2', 'torch', 'torchvision', 'onnxruntime', 'safetensors', 'ultralytics', 'networkx', 'kornia']:
    try:
        hiddenimports += collect_submodules(mod)
    except:
        pass

# ===================== 排除项 =====================
# 注意：matplotlib 不能排除，YOLOv5 repo 需要它
excludes = [
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx', 'docutils',
    # 不需要的子模块（避免警告）
    'onnx', 'tensorboard', 'timm',
    'onnxruntime.quantization',  # 量化功能不需要
    'torch.utils.tensorboard',   # 训练可视化不需要
]

# ===================== 分析 =====================
print("[SPEC] 开始分析...")
a = Analysis(
    ['app.py'],
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
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, 'src', 'app', 'static', 'favicon.ico') if os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'app', 'static', 'favicon.ico')) else None,
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
