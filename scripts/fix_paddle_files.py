import os
import sys
import glob
import shutil
import tarfile
import urllib.request
from pathlib import Path
import time
import logging

# 尝试导入共享模块，如果失败则回退
try:
    # 项目根目录在sys.path中
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from src.shared.path_helpers import resource_path
except ImportError:
    # 如果导入失败（例如，直接运行脚本），定义一个本地版本
    def resource_path(relative_path):
        base_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录
        return os.path.join(base_path, relative_path)
    print("警告：无法导入共享路径助手，使用本地回退版本。")

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PaddleOCRModelDownloader")

def download_and_extract_model(url, save_dir):
    """下载并解压模型文件"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        temp_file = os.path.join(save_dir, "temp_model.tar")
        
        # 显示下载进度
        def _progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r下载进度: {percent}%")
            sys.stdout.flush()
        
        # 下载文件
        logger.info(f"开始下载模型: {url}")
        urllib.request.urlretrieve(url, temp_file, _progress)
        logger.info("\n下载完成")
        
        # 解压文件
        logger.info(f"正在解压模型文件到 {save_dir}")
        with tarfile.open(temp_file) as tar:
            tar.extractall(save_dir)
        
        # 删除临时文件
        os.remove(temp_file)
        logger.info("模型解压完成")
        
        # 移动文件到正确位置
        extracted_dir = None
        for item in os.listdir(save_dir):
            item_path = os.path.join(save_dir, item)
            if os.path.isdir(item_path) and "infer" in item:
                extracted_dir = item_path
                break
        
        if extracted_dir:
            # 将内部文件移至save_dir
            for item in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, item)
                dst = os.path.join(save_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            
            # 删除空目录
            os.rmdir(extracted_dir)
        
        return True
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        return False

def download_paddle_ocr_models():
    """下载PaddleOCR所需的所有模型"""
    # 创建模型存储目录
    models_dir = resource_path(os.path.join("models", "paddle_ocr"))
    
    # 创建模型子目录
    det_en_dir = os.path.join(models_dir, "det_en")
    rec_en_dir = os.path.join(models_dir, "rec_en")
    cls_dir = os.path.join(models_dir, "cls")
    
    # 确保目录存在
    os.makedirs(det_en_dir, exist_ok=True)
    os.makedirs(rec_en_dir, exist_ok=True)
    os.makedirs(cls_dir, exist_ok=True)
    
    # 英文检测模型
    if not os.path.exists(os.path.join(det_en_dir, "inference.pdmodel")):
        logger.info("下载英文文本检测模型...")
        en_det_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
        if not download_and_extract_model(en_det_url, det_en_dir):
            logger.error("下载英文检测模型失败")
    else:
        logger.info("英文检测模型已存在")
    
    # 英文识别模型
    if not os.path.exists(os.path.join(rec_en_dir, "inference.pdmodel")):
        logger.info("下载英文文本识别模型...")
        en_rec_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"
        if not download_and_extract_model(en_rec_url, rec_en_dir):
            logger.error("下载英文识别模型失败")
    else:
        logger.info("英文识别模型已存在")
    
    # 方向分类模型
    if not os.path.exists(os.path.join(cls_dir, "inference.pdmodel")):
        logger.info("下载文本方向分类模型...")
        cls_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
        if not download_and_extract_model(cls_url, cls_dir):
            logger.error("下载方向分类模型失败")
    else:
        logger.info("方向分类模型已存在")
    
    # 验证模型是否下载成功
    if (os.path.exists(os.path.join(det_en_dir, "inference.pdmodel")) and 
        os.path.exists(os.path.join(rec_en_dir, "inference.pdmodel")) and 
        os.path.exists(os.path.join(cls_dir, "inference.pdmodel"))):
        logger.info("所有模型下载成功!")
        return True
    else:
        logger.error("部分模型下载失败")
        return False

def fix_paddle_files():
    """修复PaddleOCR文件路径问题"""
    # 获取环境变量
    site_packages_path = None
    
    # 获取当前Python解释器的site-packages路径
    for path in sys.path:
        if "site-packages" in path:
            site_packages_path = path
            break
    
    if not site_packages_path:
        logger.error("找不到site-packages目录，请检查Python环境")
        return False
    
    # 查找paddleocr路径
    paddleocr_path = os.path.join(site_packages_path, "paddleocr")
    if not os.path.exists(paddleocr_path):
        logger.error("找不到paddleocr模块，请先安装paddleocr")
        return False
    
    logger.info(f"找到paddleocr路径: {paddleocr_path}")
    
    # 检查并修复__init__.py文件
    init_file = os.path.join(paddleocr_path, "__init__.py")
    if os.path.exists(init_file):
        logger.info(f"检查并修复 {init_file}")
        
        # 读取文件内容
        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 在打包环境中避免自动更新下载模型文件
        if "auto_download_flag = False" not in content:
            logger.info("修正auto_download_flag设置")
            content = content.replace("auto_download_flag = True", "auto_download_flag = False")
            
            # 写回文件
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info("已禁用自动下载模型功能")
    
    # 检查模型目录
    tools_path = os.path.join(paddleocr_path, "tools")
    if not os.path.isdir(tools_path):
        logger.warning(f"tools目录不存在: {tools_path}")
        os.makedirs(tools_path, exist_ok=True)
        logger.info(f"已创建tools目录: {tools_path}")
        
        # 创建__init__.py文件
        with open(os.path.join(tools_path, "__init__.py"), "w") as f:
            f.write("# Auto-generated init file\n")
        logger.info(f"已创建tools/__init__.py文件")
    
    # 检查ppocr目录
    ppocr_path = os.path.join(paddleocr_path, "ppocr")
    if not os.path.isdir(ppocr_path):
        logger.warning(f"ppocr目录不存在: {ppocr_path}")
        os.makedirs(ppocr_path, exist_ok=True)
        logger.info(f"已创建ppocr目录: {ppocr_path}")
        
        # 创建__init__.py文件
        with open(os.path.join(ppocr_path, "__init__.py"), "w") as f:
            f.write("# Auto-generated init file\n")
        logger.info(f"已创建ppocr/__init__.py文件")
    
    logger.info("PaddleOCR文件路径修复完成")
    return True

if __name__ == "__main__":
    logger.info("=== PaddleOCR修复工具 ===")
    logger.info("1. 修复PaddleOCR文件结构")
    fix_result = fix_paddle_files()
    if fix_result:
        logger.info("PaddleOCR文件结构修复成功")
    else:
        logger.error("PaddleOCR文件结构修复失败")
    
    logger.info("2. 下载模型文件")
    download_result = download_paddle_ocr_models()
    if download_result:
        logger.info("模型文件下载成功，现在可以使用PyInstaller进行打包了")
    else:
        logger.error("模型文件下载失败，请检查网络连接或手动下载")
    
    logger.info("=== 处理完成 ===")
    
    # 等待用户按键退出
    input("按回车键退出...")