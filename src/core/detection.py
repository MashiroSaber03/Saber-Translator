import logging
import cv2
import numpy as np
import os
import sys

try:
    from src.interfaces.yolo_interface import detect_bubbles # 正常导入方式
    # 确保得到config_loader模块
except ModuleNotFoundError:
    # 在直接运行脚本时找不到src模块，添加项目根目录到sys.path
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from src.interfaces.yolo_interface import detect_bubbles # 再次尝试导入

logger = logging.getLogger("CoreDetection")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_bubble_coordinates(image_pil, conf_threshold=0.6):
    """
    检测 PIL 图像中的气泡并返回排序后的坐标列表。

    Args:
        image_pil (PIL.Image.Image): 输入的 PIL 图像对象。
        conf_threshold (float): YOLOv5 检测的置信度阈值。

    Returns:
        list: 包含气泡坐标元组 (x1, y1, x2, y2) 的列表，按宽度降序排列。
              如果检测失败或未找到气泡，则返回空列表。
    """
    try:
        # 1. 将 PIL Image 转换为 OpenCV BGR 格式
        img_np = np.array(image_pil.convert('RGB')) # 确保是 RGB
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. 调用 YOLO 接口进行检测
        boxes, scores, class_ids = detect_bubbles(img_cv, conf_threshold=conf_threshold)

        if boxes is None or len(boxes) == 0:
            logger.info("未检测到气泡。")
            return []

        # 3. 提取并格式化坐标
        bubble_coords = []
        logger.info(f"检测到 {len(boxes)} 个气泡候选框。")
        for i in range(len(boxes)):
            # 确保坐标是整数
            x1, y1, x2, y2 = map(int, boxes[i])
            # 基本的坐标有效性检查 (可选，但推荐)
            if x1 < x2 and y1 < y2:
                bubble_coords.append((x1, y1, x2, y2))
            else:
                logger.warning(f"检测到无效坐标框，已跳过: [{x1}, {y1}, {x2}, {y2}]")


        # 4. 按宽度降序排序 (YOLO 输出可能无序，排序有助于后续处理)
        # 宽度 = x2 - x1
        bubble_coords.sort(key=lambda coord: coord[2] - coord[0], reverse=True)

        logger.info(f"最终获取并排序了 {len(bubble_coords)} 个有效气泡坐标。")
        return bubble_coords

    except Exception as e:
        logger.error(f"获取气泡坐标时出错: {e}", exc_info=True)
        return []

# --- 测试代码 ---
if __name__ == '__main__':
    from PIL import Image
    try:
        from src.shared.path_helpers import resource_path # 需要导入
    except ModuleNotFoundError:
        # 路径已根目录为准
        from src.shared.path_helpers import resource_path
    import os
    
    # 启用入日志输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("--- 测试气泡检测核心逻辑 ---")
    test_image_path = resource_path('pic/before1.png') # 使用你的测试图片路径
    if os.path.exists(test_image_path):
        print(f"加载测试图片: {test_image_path}")
        try:
            img_pil = Image.open(test_image_path)
            print("开始检测坐标...")
            coords = get_bubble_coordinates(img_pil, conf_threshold=0.5)
            print(f"检测完成，找到 {len(coords)} 个气泡坐标:")
            # for i, coord in enumerate(coords):
            #     print(f"  - 气泡 {i+1}: {coord}")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    else:
        print(f"错误：测试图片未找到 {test_image_path}")