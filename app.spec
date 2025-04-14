# -*- mode: python ; coding: utf-8 -*-

import os
import manga_ocr
import unidic_lite
import glob
import sys
import shutil
import importlib.util
import importlib.metadata

block_cipher = None

# 设置工作目录
current_dir = os.getcwd()

# 确保src在sys.path中，以便脚本可以导入自定义模块
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 动态获取 src/app/static 文件夹中的所有字体文件列表
static_folder = os.path.join('src', 'app', 'static')
font_files = glob.glob(os.path.join(static_folder, '*.[tT][tT][fF]')) + glob.glob(os.path.join(static_folder, '*.[tT][tT][cC]'))

# 尝试在运行前删除任何现有的rapidfuzz.__pyinstaller.py文件
rapidfuzz_pyinstaller_path = None
try:
    import rapidfuzz
    rapidfuzz_dir = os.path.dirname(rapidfuzz.__file__)
    rapidfuzz_pyinstaller_path = os.path.join(rapidfuzz_dir, "__pyinstaller.py")
    if os.path.exists(rapidfuzz_pyinstaller_path):
        print(f"正在备份并修改问题文件: {rapidfuzz_pyinstaller_path}")
        # 备份文件
        shutil.copy2(rapidfuzz_pyinstaller_path, rapidfuzz_pyinstaller_path + ".bak")
        # 创建正确的钩子文件
        with open(rapidfuzz_pyinstaller_path, 'w') as f:
            f.write("# Fixed pyinstaller hook file\n")
            f.write("def get_hook_dirs():\n")
            f.write("    return []\n")
        print("已修复rapidfuzz.__pyinstaller.py文件")
except Exception as e:
    print(f"修复rapidfuzz钩子文件失败: {e}")

# 尝试定位ollama库位置
ollama_path = None
try:
    import ollama
    ollama_path = os.path.dirname(ollama.__file__)
    print(f"找到ollama库路径: {ollama_path}")
except ImportError:
    print("找不到ollama库，请确保已安装")

# 尝试定位httpx库位置
httpx_path = None
try:
    import httpx
    httpx_path = os.path.dirname(httpx.__file__)
    print(f"找到httpx库路径: {httpx_path}")
except ImportError:
    print("找不到httpx库，请确保已安装")

# 尝试定位cv2库位置
cv2_path = None
try:
    import cv2
    cv2_path = os.path.dirname(cv2.__file__)
    print(f"找到cv2库路径: {cv2_path}")
except ImportError:
    print("找不到cv2库，请确保已安装")

# 尝试定位paddleocr库位置以及其tools和ppocr目录
paddleocr_path = None
try:
    import paddleocr
    paddleocr_path = os.path.dirname(paddleocr.__file__)
    print(f"找到paddleocr库路径: {paddleocr_path}")
    
    # 获取tools和ppocr的路径
    tools_path = os.path.join(paddleocr_path, "tools")
    ppocr_path = os.path.join(paddleocr_path, "ppocr")
    
    if not os.path.exists(tools_path):
        print(f"警告: 找不到tools目录: {tools_path}")
    else:
        print(f"找到tools目录: {tools_path}")
        
    if not os.path.exists(ppocr_path):
        print(f"警告: 找不到ppocr目录: {ppocr_path}")
    else:
        print(f"找到ppocr目录: {ppocr_path}")
        
        # 检查ppocr/utils目录和字典文件
        ppocr_utils_path = os.path.join(ppocr_path, "utils")
        if os.path.exists(ppocr_utils_path):
            print(f"找到ppocr/utils目录: {ppocr_utils_path}")
            dict_files = glob.glob(os.path.join(ppocr_utils_path, "*_dict.txt"))
            print(f"找到字典文件: {dict_files}")
    
except ImportError:
    print("找不到paddleocr库，请确保已安装")

# 尝试定位paddle库位置
paddle_path = None
try:
    import paddle
    paddle_path = os.path.dirname(paddle.__file__)
    print(f"找到paddle库路径: {paddle_path}")
except ImportError:
    print("找不到paddle库，请确保已安装")

# 尝试定位rapidfuzz库位置
rapidfuzz_path = None
try:
    import rapidfuzz
    rapidfuzz_path = os.path.dirname(rapidfuzz.__file__)
    print(f"找到rapidfuzz库路径: {rapidfuzz_path}")
except ImportError:
    print("找不到rapidfuzz库，这可能不会影响核心功能")

# 尝试定位litelama库位置
litelama_path = None
try:
    import litelama
    litelama_path = os.path.dirname(litelama.__file__)
    print(f"找到litelama库路径: {litelama_path}")
except ImportError:
    print("找不到litelama库，请确保已安装")

# 预下载PaddleOCR模型并确保模型文件夹存在
paddleocr_models_dir = os.path.join(current_dir, 'models', 'paddle_ocr')
os.makedirs(os.path.join(paddleocr_models_dir, 'det_en'), exist_ok=True)
os.makedirs(os.path.join(paddleocr_models_dir, 'rec_en'), exist_ok=True)
os.makedirs(os.path.join(paddleocr_models_dir, 'cls'), exist_ok=True)

# 注释掉可能导致错误的预初始化代码
print("跳过PaddleOCR预初始化以避免'No module named src'错误")
# try:
#     print("尝试预先初始化PaddleOCR以下载模型...")
#     from src.interfaces.paddle_ocr_interface import PaddleOCRHandler
#     ocr_handler = PaddleOCRHandler()
#     ocr_handler.initialize("en")
#     print("PaddleOCR初始化成功，模型应已下载")
# except Exception as e:
#     print(f"PaddleOCR预初始化失败: {e}，请确保手动下载模型文件")

# 检查PaddleOCR模型文件是否存在
paddle_ocr_model_files = []
if os.path.exists(paddleocr_models_dir):
    print(f"找到PaddleOCR模型目录: {paddleocr_models_dir}")
    for root, dirs, files in os.walk(paddleocr_models_dir):
        for file in files:
            model_file = os.path.join(root, file)
            rel_path = os.path.relpath(model_file, current_dir)
            target_path = os.path.dirname(rel_path)
            paddle_ocr_model_files.append((model_file, target_path))
            print(f"添加PaddleOCR模型文件: {model_file} -> {target_path}")
else:
    print(f"警告: 找不到PaddleOCR模型目录: {paddleocr_models_dir}")
    print("请确保在运行PyInstaller之前已下载PaddleOCR模型")

# 创建所有需要的数据文件列表
datas = [
    # 注意：config目录不再打包，而是在post_build_process中创建一个空目录
    
    # 打包 src/app/templates 文件夹
    (os.path.join('src', 'app', 'templates'), os.path.join('src', 'app', 'templates')),
    
    # 打包 src/app/static 文件夹，包含 js 和 css 文件
    (os.path.join('src', 'app', 'static'), os.path.join('src', 'app', 'static')),
    
    # 打包 pic 文件夹，包含项目所需图片资源
    ('pic', 'pic'),
    
    # 动态添加所有的字体文件，目标路径为 src/app/static 文件夹
    *((f, os.path.join('src', 'app', 'static')) for f in font_files),
    
    # 打包 weights 文件夹
    ('weights', 'weights'),
    
    # 打包 models 文件夹 (包含MI-GAN模型)
    ('models', 'models'),
    
    # 添加PaddleOCR模型文件
    *paddle_ocr_model_files,
    
    # 打包 manga_ocr_model 文件夹
    ('manga_ocr_model', 'manga_ocr_model'),
    
    # 打包核心源代码目录
    ('src', 'src'),
    
    # 打包 ultralytics_yolov5_master 文件夹
    ('ultralytics_yolov5_master', 'ultralytics_yolov5_master'),
    
    # 打包 sd-webui-cleaner 文件夹（包含LAMA模型和脚本）
    ('sd-webui-cleaner', 'sd-webui-cleaner'),
    
    # 打包 plugins 文件夹（用户自定义插件目录）
    ('plugins', 'plugins'),
    
    # 打包 temp_paddleocr 文件夹
    ('temp_paddleocr', 'temp_paddleocr'),
    
    # 打包 scripts 文件夹（确保修复脚本也被包含）
    ('scripts', 'scripts'),
    
    # 动态获取 unidic_lite 词典文件夹路径并打包
    (os.path.join(os.path.dirname(unidic_lite.__file__), 'dicdir'), 'unidic_lite/dicdir'),
    
    # 动态获取 manga_ocr assets 文件夹路径并打包
    (os.path.join(os.path.dirname(manga_ocr.__file__), 'assets'), 'manga_ocr/assets'),
]

# 添加ollama和httpx库
if ollama_path:
    datas.append((ollama_path, 'ollama'))
if httpx_path:
    datas.append((httpx_path, 'httpx'))
if cv2_path:
    datas.append((cv2_path, 'cv2'))

# 添加paddleocr和paddle库
if paddleocr_path:
    datas.append((paddleocr_path, 'paddleocr'))
    # 单独添加tools目录
    if os.path.exists(tools_path):
        for root, dirs, files in os.walk(tools_path):
            rel_path = os.path.relpath(root, paddleocr_path)
            for file in files:
                file_path = os.path.join(root, file)
                target_path = os.path.join('paddleocr', rel_path)
                datas.append((file_path, target_path))
    
    # 单独添加ppocr目录
    if os.path.exists(ppocr_path):
        for root, dirs, files in os.walk(ppocr_path):
            rel_path = os.path.relpath(root, paddleocr_path)
            for file in files:
                file_path = os.path.join(root, file)
                target_path = os.path.join('paddleocr', rel_path)
                datas.append((file_path, target_path))

# 添加litelama库及其配置文件
if litelama_path:
    datas.append((litelama_path, 'litelama'))
    # 特别确保config.yaml文件被包含
    config_file = os.path.join(litelama_path, 'config.yaml')
    if os.path.exists(config_file):
        datas.append((config_file, 'litelama'))

if paddle_path:
    datas.append((paddle_path, 'paddle'))
if rapidfuzz_path:
    datas.append((rapidfuzz_path, 'rapidfuzz'))

# 创建排除列表，排除引起警告的模块
excludes_list = [
    'venv', 
    '_pycache_',
    'tensorboard',  # 排除tensorboard，因为项目并不需要它
    'torch.utils.tensorboard',  # 排除PyTorch tensorboard支持
    'torch.distributed.elastic.multiprocessing.redirects',  # 排除重定向功能
]

# 定义所有需要的隐藏导入
hidden_imports = [
    'ultralytics',
    # ollama相关
    'ollama',
    'ollama._client',
    'ollama._types',
    # httpx相关
    'httpx',
    'httpx._api',
    'httpx._auth',
    'httpx._client',
    'httpx._config',
    'httpx._models',
    'httpx._transports',
    'httpx._types',
    # httpx依赖
    'certifi',
    'h11',
    'idna',
    'anyio',
    'sniffio',
    
    # PaddleOCR相关
    'paddleocr',
    'tools',
    'tools.infer',
    'tools.program',
    'ppocr',
    'ppocr.data',
    'ppocr.postprocess',
    'ppocr.utils',
    'ppocr.modeling',
    
    # Paddle相关
    'paddle',
    'paddle.fluid',
    'paddle.inference',
    'paddle.dataset',
    'paddle.vision',
    'paddle.vision.ops',
    'paddle.nn',
    'paddle.nn.functional',
    'paddle.optimizer',
    'paddle.framework',
    'paddle.static',
    'paddle.tensor',
    'paddle.distributed',
    'paddle.distributed.fleet',
    
    # LAMA相关
    'litelama',
    
    # OpenCV相关
    'cv2',
    
    # 其他依赖
    'imgaug',
    'lanms',
    'lmdb',
    
    # rapidfuzz相关
    'rapidfuzz',
    'rapidfuzz.fuzz',
    'rapidfuzz.process',
    'rapidfuzz.utils',
    'numpy',
    
    # MI-GAN相关
    'onnxruntime',
    
    # 重构后的模块
    'src.app',
    'src.app.api',
    'src.app.api.translate_api',
    'src.app.api.config_api',
    'src.app.api.system_api',
    'src.core',
    'src.interfaces',
    'src.shared',
    'src.shared.config_loader',
    'src.shared.constants',
    'src.shared.path_helpers',
    'src.plugins',
    'src.plugins.base',
    'src.plugins.hooks',
    'src.plugins.manager',
    
    # Flask相关
    'flask_cors',
    'werkzeug',
    'werkzeug.utils',
    'colorama',
    'engineio',
    'eventlet',
    'gevent',
    'PyPDF2',
]

# 尝试执行scripts/fix_paddle_files.py脚本(如果存在)
try:
    fix_paddle_script = os.path.join(current_dir, 'scripts', 'fix_paddle_files.py')
    if os.path.exists(fix_paddle_script):
        print(f"正在执行PaddleOCR修复脚本: {fix_paddle_script}")
        # 使用utf-8编码打开文件以解决编码问题
        with open(fix_paddle_script, 'r', encoding='utf-8') as f:
            script_content = f.read()
            exec(script_content)
        print("PaddleOCR修复脚本执行完成")
    else:
        print(f"警告: 找不到PaddleOCR修复脚本: {fix_paddle_script}")
except Exception as e:
    print(f"执行PaddleOCR修复脚本失败: {e}")

a = Analysis(
    ['app.py'],
    pathex=[current_dir],  # 使用当前目录作为pathex
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes_list,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 定义打包后的处理函数
def post_build_process():
    # 定义DISTPATH
    DISTPATH = os.path.join(current_dir, 'dist')
    target_dir = os.path.join(DISTPATH, 'Saber-Translator', '_internal')
    
    # 确保tools目录存在
    tools_dir = os.path.join(target_dir, 'tools')
    if not os.path.exists(tools_dir):
        os.makedirs(tools_dir)
        print(f"创建目录: {tools_dir}")
    
    # 创建tools/__init__.py文件
    with open(os.path.join(tools_dir, '__init__.py'), 'w') as f:
        f.write("# Empty init file for PyInstaller\n")
    print(f"创建文件: {os.path.join(tools_dir, '__init__.py')}")
    
    # 检查并创建ppocr目录
    ppocr_dir = os.path.join(target_dir, 'ppocr')
    if not os.path.exists(ppocr_dir):
        os.makedirs(ppocr_dir)
        print(f"创建目录: {ppocr_dir}")
    
    # 创建ppocr/__init__.py文件
    with open(os.path.join(ppocr_dir, '__init__.py'), 'w') as f:
        f.write("# Empty init file for PyInstaller\n")
    print(f"创建文件: {os.path.join(ppocr_dir, '__init__.py')}")
    
    # 检查并创建ppocr/utils目录
    ppocr_utils_dir = os.path.join(ppocr_dir, 'utils')
    if not os.path.exists(ppocr_utils_dir):
        os.makedirs(ppocr_utils_dir)
        print(f"创建目录: {ppocr_utils_dir}")
    
    # 创建ppocr/utils/__init__.py文件
    with open(os.path.join(ppocr_utils_dir, '__init__.py'), 'w') as f:
        f.write("# Empty init file for PyInstaller\n")
    print(f"创建文件: {os.path.join(ppocr_utils_dir, '__init__.py')}")
    
    # 检查并创建ppocr/utils/dict目录
    ppocr_dict_dir = os.path.join(ppocr_utils_dir, 'dict')
    if not os.path.exists(ppocr_dict_dir):
        os.makedirs(ppocr_dict_dir)
        print(f"创建目录: {ppocr_dict_dir}")
    
    # 复制paddleocr中的字典文件到ppocr/utils目录
    src_dict_dir = os.path.join(target_dir, 'paddleocr', 'ppocr', 'utils')
    if os.path.exists(src_dict_dir):
        dict_files = [f for f in os.listdir(src_dict_dir) if f.endswith('.txt')]
        for dict_file in dict_files:
            src_file = os.path.join(src_dict_dir, dict_file)
            dst_file = os.path.join(ppocr_utils_dir, dict_file)
            shutil.copy2(src_file, dst_file)
            print(f"复制字典文件: {src_file} -> {dst_file}")
    else:
        print(f"警告: 找不到源字典目录: {src_dict_dir}")
    
    # 检查paddleocr/ppocr/utils/dict目录是否存在
    src_dict_subdir = os.path.join(target_dir, 'paddleocr', 'ppocr', 'utils', 'dict')
    if os.path.exists(src_dict_subdir):
        # 复制dict子目录中的所有文件
        dict_subdir_files = os.listdir(src_dict_subdir)
        print(f"在dict子目录中找到文件: {dict_subdir_files}")
        
        for file in dict_subdir_files:
            src_file = os.path.join(src_dict_subdir, file)
            dst_file = os.path.join(ppocr_dict_dir, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"复制字典子目录文件: {src_file} -> {dst_file}")
    else:
        print(f"警告: 找不到dict子目录: {src_dict_subdir}")
        
        # 如果找不到韩文字典文件，手动创建一个基础版本
        korean_dict_file = os.path.join(ppocr_dict_dir, 'korean_dict.txt')
        if not os.path.exists(korean_dict_file):
            # 这是一个简单的韩文字典文件，实际使用时可能需要更全面的内容
            korean_chars = [
                "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하",
                "거", "너", "더", "러", "머", "버", "서", "어", "저", "처", "커", "터", "퍼", "허",
                "고", "노", "도", "로", "모", "보", "소", "오", "조", "초", "코", "토", "포", "호",
                "구", "누", "두", "루", "무", "부", "수", "우", "주", "추", "쿠", "투", "푸", "후",
                "그", "느", "드", "르", "므", "브", "스", "으", "즈", "츠", "크", "트", "프", "흐",
                "기", "니", "디", "리", "미", "비", "시", "이", "지", "치", "키", "티", "피", "히",
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
            ]
            with open(korean_dict_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(korean_chars))
            print(f"创建基础韩文字典文件: {korean_dict_file}")
    
    # 确保模型目录结构存在
    paddle_models_dir = os.path.join(target_dir, 'models', 'paddle_ocr')
    os.makedirs(os.path.join(paddle_models_dir, 'det_en'), exist_ok=True)
    os.makedirs(os.path.join(paddle_models_dir, 'rec_en'), exist_ok=True)
    os.makedirs(os.path.join(paddle_models_dir, 'cls'), exist_ok=True)
    
    # 复制源目录中所有的PaddleOCR模型文件到目标位置
    src_models_dir = os.path.join(current_dir, 'models', 'paddle_ocr')
    if os.path.exists(src_models_dir):
        for root, dirs, files in os.walk(src_models_dir):
            for file in files:
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(root, src_models_dir)
                dst_dir = os.path.join(paddle_models_dir, rel_path)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, file)
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"复制PaddleOCR模型文件: {src_file} -> {dst_file}")
                except Exception as e:
                    print(f"复制模型文件失败: {e}")
    
    # 在exe文件同级目录下创建必要的文件夹，而不是在_internal目录中
    app_root_dir = os.path.join(DISTPATH, 'Saber-Translator')
    
    # 创建config文件夹在EXE同级目录
    config_dir = os.path.join(app_root_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    print(f"在EXE同级目录创建config文件夹: {config_dir}")
    
    # 创建日志和临时文件夹
    os.makedirs(os.path.join(app_root_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(app_root_dir, 'temp'), exist_ok=True)
    
    # 创建插件目录
    os.makedirs(os.path.join(app_root_dir, 'plugins'), exist_ok=True)
    
    # 检查并创建litelama目录
    litelama_dir = os.path.join(target_dir, 'litelama')
    if not os.path.exists(litelama_dir):
        os.makedirs(litelama_dir)
        print(f"创建目录: {litelama_dir}")
    
    # 创建litelama/__init__.py文件
    with open(os.path.join(litelama_dir, '__init__.py'), 'w') as f:
        f.write("# Empty init file for PyInstaller\n")
    print(f"创建文件: {os.path.join(litelama_dir, '__init__.py')}")

    # 确保litelama/config.yaml存在
    litelama_config = os.path.join(litelama_dir, 'config.yaml')
    if not os.path.exists(litelama_config):
        # 从源目录复制配置文件
        src_config = os.path.join(current_dir, 'venv', 'Lib', 'site-packages', 'litelama', 'config.yaml')
        if os.path.exists(src_config):
            shutil.copy2(src_config, litelama_config)
            print(f"复制litelama配置文件: {src_config} -> {litelama_config}")
        else:
            print(f"警告: 找不到litelama源配置文件: {src_config}")
            # 创建一个基本的配置文件
            with open(litelama_config, 'w') as f:
                f.write("# Basic litelama config file\n")
                f.write("generator:\n")
                f.write("  kind: ffc_resnet\n")
                f.write("  input_nc: 4\n")
                f.write("  output_nc: 3\n")
                f.write("  ngf: 64\n")
                f.write("  n_downsampling: 3\n")
                f.write("  n_blocks: 18\n")
                f.write("  add_out_act: sigmoid\n")
            print(f"创建基本的litelama配置文件: {litelama_config}")
    
    print("修复完成!")

# 处理rapidfuzz警告：手动排除导致问题的模块
for module in list(a.pure):
    if 'rapidfuzz.__pyinstaller' in module[0]:
        print(f"从pure列表中移除问题模块: {module[0]}")
        a.pure.remove(module)
    elif 'torch.utils.tensorboard' in module[0]:
        print(f"从pure列表中移除问题模块: {module[0]}")
        a.pure.remove(module)
    elif 'torch.distributed.elastic.multiprocessing.redirects' in module[0]:
        print(f"从pure列表中移除问题模块: {module[0]}")
        a.pure.remove(module)

# 打印有关打包的信息用于调试
print("="*80)
print("打包信息:")
print(f"Python版本: {sys.version}")
print(f"binaries数量: {len(a.binaries)}")
print(f"datas数量: {len(a.datas)}")
print(f"pure数量: {len(a.pure)}")
print(f"hiddenimports: {hidden_imports}")
print("="*80)

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
    upx=True,
    console=True,
    icon=os.path.join('src', 'app', 'static', 'favicon.ico'),  # 更新图标路径
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Saber-Translator',
)

# 执行打包后的处理过程
try:
    post_build_process()
except Exception as e:
    print(f"执行打包后处理过程时出错: {e}")
