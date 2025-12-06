"""
CTD (Comic Text Detector) 工具函数
提供 CTD 模型所需的预处理和后处理函数
"""
import cv2
import numpy as np
import torch
import logging

from .utils.imgproc_utils import letterbox

logger = logging.getLogger("CTDDetector")


def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    """
    预处理图像用于CTD模型推理
    与 manga-image-translator 完全一致
    """
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))  # HWC to CHW
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)


def postprocess_mask(img, thresh=None):
    """
    后处理掩码输出
    与 manga-image-translator 完全一致
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)
