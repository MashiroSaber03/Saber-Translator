"""
PaddleOCR 接口 - 重定向到 ONNX 实现

此文件保留用于向后兼容，实际实现已迁移到 paddle_ocr_onnx_interface.py
使用 RapidOCR + ONNX 模型，无需安装 PaddlePaddle，打包更简单

工作流程（与原版相同）：
1. CTD 检测气泡（粗粒度）
2. det 模型检测气泡内的文本行（细粒度）
3. rec 模型识别单行（48×W 输入）
"""

# 直接从 ONNX 接口导入所有内容
from src.interfaces.paddle_ocr_onnx_interface import (
    PaddleOCRHandlerONNX as PaddleOCRHandler,
    PaddleOCRHandlerONNX as PaddleOCRHandlerV5,
    get_paddle_ocr_handler,
)

__all__ = ['PaddleOCRHandler', 'PaddleOCRHandlerV5', 'get_paddle_ocr_handler']


# 测试代码
if __name__ == '__main__':
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("PaddleOCR 接口测试 (ONNX 版本)")
    print("=" * 60)
    
    handler = get_paddle_ocr_handler()
    
    print("\n[测试] 初始化中文模型...")
    if handler.initialize("chinese"):
        print("✅ 初始化成功")
    else:
        print("❌ 初始化失败")
        print("请先运行: python download_paddle_onnx_models.py")
