import torch
import os
import logging
import sys

# 添加项目根目录到Python路径，使模块导入在直接运行脚本时也能工作
try:
    from src.shared.path_helpers import resource_path  # 正常导入方式
except ModuleNotFoundError:
    # 在直接运行脚本时找不到src模块，添加项目根目录到sys.path
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from src.shared.path_helpers import resource_path  # 再次尝试导入

logger = logging.getLogger("YOLOInterface")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 如果需要独立日志

# --- 全局变量存储加载的模型 ---
_yolo_model = None
_model_conf = 0.6 # 默认置信度阈值

def load_yolo_model(weights_name='best.pt', repo_dir_name='ultralytics_yolov5_master', conf_threshold=0.6):
    """
    加载 YOLOv5 模型。如果模型已加载，则直接返回。

    Args:
        weights_name (str): 权重文件名 (例如 'best.pt').
        repo_dir_name (str): YOLOv5 本地仓库目录名.
        conf_threshold (float): 置信度阈值.

    Returns:
        torch.nn.Module or None: 加载的模型或 None (如果失败).
    """
    global _yolo_model, _model_conf

    if _yolo_model is not None:
        # 如果置信度发生变化，更新模型配置
        if _model_conf != conf_threshold:
            logger.info(f"更新 YOLOv5 置信度阈值: {_model_conf} -> {conf_threshold}")
            _yolo_model.conf = conf_threshold
            _model_conf = conf_threshold
        # logger.debug("YOLOv5 模型已加载，直接返回。")
        return _yolo_model

    try:
        weights_path = resource_path(os.path.join('weights', weights_name))
        repo_dir = resource_path(repo_dir_name)

        if not os.path.exists(weights_path):
            logger.error(f"YOLOv5 权重文件未找到: {weights_path}")
            return None
        if not os.path.exists(repo_dir):
            logger.error(f"YOLOv5 仓库目录未找到: {repo_dir}")
            return None

        logger.info(f"开始加载 YOLOv5 模型: {weights_path}")
        # 强制重新加载，避免缓存问题，并指定本地源
        # 使用 trust_repo=True (如果你的 torch 版本支持) 或适应旧版 API
        try:
             # 尝试使用 trust_repo=True (适用于较新版本 PyTorch Hub)
             _yolo_model = torch.hub.load(repo_or_dir=repo_dir, model='custom', path=weights_path, source='local', force_reload=False, trust_repo=True)
        except TypeError:
             # 回退到旧版 API (没有 trust_repo 参数)
             logger.warning("当前 PyTorch Hub 版本不支持 trust_repo=True，尝试旧版 API。")
             _yolo_model = torch.hub.load(repo_or_dir=repo_dir, model='custom', path=weights_path, source='local', force_reload=False)


        _yolo_model.conf = conf_threshold
        _model_conf = conf_threshold # 保存当前置信度
        logger.info(f"YOLOv5 模型加载成功，置信度设置为: {conf_threshold}")
        return _yolo_model
    except Exception as e:
        logger.error(f"加载 YOLOv5 模型失败: {e}", exc_info=True)
        _yolo_model = None
        return None

def detect_bubbles(image_cv, conf_threshold=0.6):
    """
    使用加载的 YOLOv5 模型检测图像中的气泡。

    Args:
        image_cv (numpy.ndarray): OpenCV BGR 格式的图像。
        conf_threshold (float): 本次检测使用的置信度阈值。

    Returns:
        tuple: 包含 boxes, scores, class_ids 的元组，如果失败则返回 ([], [], [])。
               boxes: numpy 数组，形状 (N, 4)，每个框为 [x1, y1, x2, y2]。
               scores: numpy 数组，形状 (N,)。
               class_ids: numpy 数组，形状 (N,)。
    """
    model = load_yolo_model(conf_threshold=conf_threshold) # 加载或获取模型，并设置置信度
    if model is None:
        return [], [], []

    try:
        results = model(image_cv) # 执行推理

        # 解析结果
        predictions = results.xyxy[0].cpu().numpy() # shape: (N, 6), [x1, y1, x2, y2, conf, class]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5]

        # 根据置信度过滤 (虽然模型内部已过滤，但这里可以再次确认或调整)
        # valid_detections = scores > model.conf # 使用模型当前的置信度
        # boxes = boxes[valid_detections]
        # scores = scores[valid_detections]
        # class_ids = class_ids[valid_detections]

        logger.info(f"YOLOv5 检测到 {len(boxes)} 个候选框 (阈值: {model.conf})")
        return boxes, scores, class_ids

    except Exception as e:
        logger.error(f"YOLOv5 推理失败: {e}", exc_info=True)
        return [], [], []

# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试 YOLOv5 接口 ---")
    # 需要一个测试图片路径
    test_image_path = resource_path('pic/before1.png') # 使用你的测试图片路径
    if os.path.exists(test_image_path):
        import cv2
        print(f"加载测试图片: {test_image_path}")
        img = cv2.imread(test_image_path)
        if img is not None:
            print("开始检测...")
            detected_boxes, detected_scores, detected_classes = detect_bubbles(img, conf_threshold=0.5)
            print(f"检测完成，找到 {len(detected_boxes)} 个气泡。")
            # for i in range(len(detected_boxes)):
            #     print(f"  - Box: {detected_boxes[i]}, Score: {detected_scores[i]:.4f}, Class: {detected_classes[i]}")
        else:
            print(f"错误：无法加载测试图片 {test_image_path}")
    else:
        print(f"错误：测试图片未找到 {test_image_path}")

    # 测试模型复用
    print("\n再次调用检测 (应复用模型)...")
    detect_bubbles(img, conf_threshold=0.7) # 改变置信度
    print("再次调用检测 (应复用模型)...")
    detect_bubbles(img, conf_threshold=0.7) # 相同置信度