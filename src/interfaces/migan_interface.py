import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os
import logging
import platform
import sys
import gc # 用于 reset 函数

# 确认路径助手和常量导入正确
from src.shared.path_helpers import resource_path, get_debug_dir
# from src.shared import constants # 如果需要常量可以导入

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MIGAN")

# 全局变量定义 MI-GAN 可用性状态
MIGAN_AVAILABLE = False

# 记录系统环境信息，便于调试
logger.info(f"系统信息: {platform.platform()}")
logger.info(f"Python版本: {sys.version}")
logger.info(f"ONNX Runtime版本: {ort.__version__}")
logger.info(f"NumPy版本: {np.__version__}")
logger.info(f"OpenCV版本: {cv2.__version__}")

class MiganInpainter:
    def __init__(self):
        # 初始化模型
        logger.info("初始化MI-GAN模型...")
        model_path = resource_path(os.path.join("models", "migan_pipeline_v2.onnx"))
        
        # 如果默认路径不存在，尝试其他可能的路径
        if not os.path.exists(model_path):
            logger.warning(f"默认路径未找到模型: {model_path}")
            # 尝试直接在根目录下查找
            model_path = resource_path("migan_pipeline_v2.onnx")
            if not os.path.exists(model_path):
                # 尝试在weights目录下查找
                model_path = resource_path(os.path.join("weights", "migan_pipeline_v2.onnx"))
                if not os.path.exists(model_path):
                    # 如果仍未找到，报错退出
                    logger.error(f"MI-GAN模型文件未找到，已尝试多个路径")
                    raise FileNotFoundError(f"MI-GAN模型文件未找到。请确保已下载模型文件并将其放在models文件夹或weights文件夹或当前目录中。")
        
        logger.info(f"找到模型文件：{model_path}")
        try:
            # 创建ONNX会话
            self.session = ort.InferenceSession(model_path)
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 输出更详细的模型信息以便调试
            logger.info(f"MI-GAN模型已成功加载")
            logger.info(f"模型输入: {self.input_names}")
            logger.info(f"模型输出: {self.output_names}")
            
            # 获取每个输入的详细信息
            for input in self.session.get_inputs():
                logger.info(f"输入 '{input.name}': 形状={input.shape}, 类型={input.type}")
            
            # 获取每个输出的详细信息
            for output in self.session.get_outputs():
                logger.info(f"输出 '{output.name}': 形状={output.shape}, 类型={output.type}")
                
        except Exception as e:
            logger.error(f"加载MI-GAN模型时出错: {e}")
            raise

    def inpaint(self, image, mask, blend_edges=True, strength=1.0):
        """
        使用MI-GAN模型进行图像修复
        
        参数:
        - image: PIL Image对象，原始图像
        - mask: PIL Image对象，掩码，黑色(0)表示需要修复的区域，白色(255)表示保留的区域
        - blend_edges: 是否进行边缘融合
        - strength: 修复强度，范围0-1，值越大修复效果越明显
        
        返回:
        - 修复后的PIL Image对象
        """
        try:
            logger.info(f"开始执行MI-GAN修复，边缘融合: {blend_edges}, 修复强度: {strength}")
            # 确保图像和掩码都是PIL Image
            if not isinstance(image, Image.Image):
                logger.info("将输入图像转换为PIL Image")
                image = Image.fromarray(image)
            if not isinstance(mask, Image.Image):
                logger.info("将输入掩码转换为PIL Image")
                mask = Image.fromarray(mask)
            
            # 保存输入以便调试
            debug_dir = get_debug_dir()
            bubbles_dir = get_debug_dir("bubbles")
            os.makedirs(bubbles_dir, exist_ok=True)
            
            timestamp = int(cv2.getTickCount())
            image.save(os.path.join(bubbles_dir, f"input_image_{timestamp}.png"))
            mask.save(os.path.join(bubbles_dir, f"input_mask_{timestamp}.png"))
            
            # 验证输入有效性
            if not self._validate_inputs(image, mask):
                logger.error("输入验证失败，返回原始图像")
                return image
                
            # 将图像和掩码转换为numpy数组
            image_np = np.array(image).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            
            # 确保图像是RGB格式（3通道）
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                logger.info("将RGBA图像转换为RGB格式")
                # 创建RGB图像，保留透明度信息
                rgb_image = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
                # 使用alpha混合将背景设为白色
                alpha = image_np[:, :, 3:4] / 255.0
                rgb_image = image_np[:, :, :3] * alpha + 255 * (1 - alpha)
                image_np = rgb_image.astype(np.uint8)
                logger.info(f"转换后图像形状: {image_np.shape}")
                
            # 确保掩码是单通道
            if len(mask_np.shape) > 2 and mask_np.shape[2] > 1:
                logger.info("将多通道掩码转换为单通道")
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            
            # 格式化输入以匹配模型期望的格式
            formatted_inputs = self._format_model_inputs(image_np, mask_np)
            
            # 运行推理
            logger.info("执行MI-GAN模型推理...")
            outputs = self.session.run(self.output_names, formatted_inputs)
            
            # 获取结果（修复后的图像）
            inpainted_image = outputs[0]
            logger.info(f"原始模型输出形状: {inpainted_image.shape}, 类型: {inpainted_image.dtype}")
            
            # 处理输出并返回
            inpainted_image = self._process_outputs(inpainted_image, image_np, mask_np, blend_edges, strength)
            return inpainted_image
            
        except Exception as e:
            logger.error(f"MI-GAN修复过程中出错: {e}", exc_info=True)
            # 出错时返回原图
            logger.info("错误发生，返回原始图像")
            return image
    
    def _process_outputs(self, inpainted_image, original_image, mask_np, blend_edges, strength):
        """处理模型输出并返回最终图像"""
        try:
            # 正确处理输出张量，确保它是标准的图像格式 [height, width, channels]
            if len(inpainted_image.shape) == 4:  # [batch, channels, height, width]
                # 去掉批次维度
                inpainted_image = inpainted_image[0]
                logger.info(f"移除批次维度后形状: {inpainted_image.shape}")
                
            if len(inpainted_image.shape) == 3:
                # 检查通道维度的位置
                if inpainted_image.shape[0] == 3 or inpainted_image.shape[0] == 1:
                    # 当前是 [channels, height, width] 需要转置为 [height, width, channels]
                    logger.info("转换输出从NCHW到NHWC格式")
                    inpainted_image = np.transpose(inpainted_image, (1, 2, 0))
                    logger.info(f"转置后形状: {inpainted_image.shape}")
            
            # 确保数据类型正确
            inpainted_image = inpainted_image.astype(np.uint8)
            logger.info(f"最终输出形状: {inpainted_image.shape}, 类型: {inpainted_image.dtype}")
            
            # 确保输出图像大小与输入图像相匹配
            if inpainted_image.shape[:2] != original_image.shape[:2]:
                logger.warning(f"输出图像大小 {inpainted_image.shape[:2]} 与输入图像 {original_image.shape[:2]} 不一致，进行调整")
                # 调整大小以匹配原始图像
                inpainted_image = cv2.resize(inpainted_image, (original_image.shape[1], original_image.shape[0]))
                logger.info(f"调整后形状: {inpainted_image.shape}")
            
            # 保存输出以便调试
            debug_dir = get_debug_dir()
            output_img = Image.fromarray(inpainted_image)
            output_img.save(os.path.join(debug_dir, f"output_raw_{int(cv2.getTickCount())}.png"))
            
            # 混合原始图像和修复图像，只替换掩码指定的区域
            mask_norm = mask_np / 255.0  # 归一化掩码到0-1
            # 扩展掩码维度以匹配图像通道
            if len(mask_norm.shape) == 2 and len(inpainted_image.shape) == 3:
                mask_norm = np.expand_dims(mask_norm, axis=2)
                mask_norm = np.repeat(mask_norm, inpainted_image.shape[2], axis=2)
            
            # 原始区域权重 + 修复区域权重 = 1
            # 掩码为255(白色)的地方保留原图，为0(黑色)的地方用修复结果
            blended = original_image * mask_norm + inpainted_image * (1 - mask_norm)
            blended = blended.astype(np.uint8)
            
            # 保存混合后的图像
            blended_img = Image.fromarray(blended)
            blended_img.save(os.path.join(debug_dir, f"blended_result_{int(cv2.getTickCount())}.png"))
            
            # 仅当启用边缘融合时进行后处理
            if blend_edges:
                # 后处理：对结果进行边缘融合，使其更自然
                # 但仍确保只在文字区域进行修复
                inpainted_image = self._post_process_result(original_image, inpainted_image, mask_np, strength)
                
                # 保存后处理后的结果
                post_processed_img = Image.fromarray(inpainted_image)
                post_processed_img.save(os.path.join(debug_dir, f"output_postprocessed_{int(cv2.getTickCount())}.png"))
                
                # 将结果转换为PIL Image
                logger.info("MI-GAN修复完成 (带边缘融合)")
                return post_processed_img
            else:
                # 返回仅修复指定区域的结果
                logger.info("MI-GAN修复完成 (无边缘融合)")
                return blended_img
        except Exception as e:
            logger.error(f"处理输出时出错: {e}", exc_info=True)
            # 如果出错，返回原始图像
            return Image.fromarray(original_image)
            
    def _format_model_inputs(self, image_np, mask_np):
        """
        格式化输入以匹配模型期望的格式
        
        参数:
        - image_np: 3D图像数组 [height, width, channels] 或 2D图像数组 [height, width]
        - mask_np: 2D或3D掩码数组 [height, width] 或 [height, width, 1]
        
        返回:
        - 格式化后的输入字典
        """
        try:
            # 获取模型的输入格式
            input_details = {input.name: input for input in self.session.get_inputs()}
            
            # 处理图像格式
            # 检查图像维度，处理黑白图像
            if len(image_np.shape) == 2:  # 黑白图像 [height, width]
                logger.info(f"检测到黑白图像: {image_np.shape}")
                # 首先添加通道维度: [height, width, 1]
                image_np = np.expand_dims(image_np, axis=2)
                # 复制通道以创建RGB图像: [height, width, 3]
                image_np = np.repeat(image_np, 3, axis=2)
                logger.info(f"转换黑白图像为RGB: {image_np.shape}")
            
            # 添加批次维度
            if len(image_np.shape) == 3:  # [height, width, channels]
                # 首先添加批次维度: [1, height, width, channels]
                image_batch = np.expand_dims(image_np, axis=0)
                logger.info(f"添加批次维度后图像形状: {image_batch.shape}")
                
                # 检查是否需要转置到 [batch, channels, height, width] 格式
                if 'image' in input_details and len(input_details['image'].shape) == 4:
                    if input_details['image'].shape[1] == 3 or input_details['image'].shape[1] == 1:
                        # 模型期望 NCHW 格式
                        logger.info("转换图像为 NCHW 格式")
                        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
                        logger.info(f"转置后图像形状: {image_batch.shape}")
            else:
                # 非二维或三维图像，可能已经是正确格式
                image_batch = image_np
            
            # 处理掩码格式
            if len(mask_np.shape) == 2:  # [height, width]
                # 添加批次维度: [1, height, width]
                mask_batch = np.expand_dims(mask_np, axis=0)
                logger.info(f"添加批次维度后掩码形状: {mask_batch.shape}")
                
                # 检查是否需要添加通道维度和转置
                if 'mask' in input_details:
                    if len(input_details['mask'].shape) == 4:
                        if input_details['mask'].shape[1] == 1:
                            # 模型期望 [batch, channels, height, width] 格式
                            logger.info("转换掩码为 NCHW 格式")
                            mask_batch = np.expand_dims(mask_batch, axis=1)  # [1, 1, height, width]
                            logger.info(f"加通道后掩码形状: {mask_batch.shape}")
            else:
                # 掩码可能已经是正确格式
                mask_batch = mask_np
            
            # 返回格式化后的输入
            inputs = {
                'image': image_batch,
                'mask': mask_batch
            }
            
            for name, data in inputs.items():
                logger.info(f"最终 {name} 形状: {data.shape}")
                
            return inputs
            
        except Exception as e:
            logger.error(f"格式化模型输入时出错: {e}", exc_info=True)
            # 如果出错，使用更安全的格式尝试
            try:
                # 确保图像是3通道RGB
                if len(image_np.shape) == 2:
                    # 是灰度图像，转为RGB
                    image_rgb = np.stack([image_np] * 3, axis=-1)
                else:
                    image_rgb = image_np
                
                # 确保维度正确 [batch, channels, height, width]
                if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                    # [height, width, channels] -> [batch, channels, height, width]
                    image_batch = np.transpose(np.expand_dims(image_rgb, axis=0), (0, 3, 1, 2))
                else:
                    image_batch = np.expand_dims(image_rgb, axis=0)
                
                # 确保掩码也正确 [batch, 1, height, width]
                mask_batch = np.expand_dims(np.expand_dims(mask_np, axis=0), axis=1)
                
                return {
                    'image': image_batch,
                    'mask': mask_batch
                }
            except Exception as e2:
                logger.error(f"备用格式化也失败: {e2}", exc_info=True)
                return {
                    'image': np.expand_dims(image_np, axis=0),
                    'mask': np.expand_dims(mask_np, axis=0)
                }

    def _post_process_result(self, original_image, inpainted_image, mask_np, strength):
        """
        对修复结果进行后处理，使边缘更自然

        参数:
        - original_image: 原始图像
        - inpainted_image: 修复后的图像
        - mask_np: 掩码
        - strength: 修复强度，范围0-5.0，值越大修复效果越明显

        返回:
        - 后处理后的图像
        """
        try:
            # 确保掩码是单通道
            if len(mask_np.shape) > 2 and mask_np.shape[2] > 1:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            
            # 获取原始图像和修复图像的尺寸
            if original_image.shape != inpainted_image.shape:
                logger.warning(f"原始图像形状 {original_image.shape} 与修复图像形状 {inpainted_image.shape} 不一致")
                # 调整尺寸使其匹配
                inpainted_image = cv2.resize(inpainted_image, (original_image.shape[1], original_image.shape[0]))
            
            # 只对掩码中的黑色区域(0)进行修复，即仅修复文字区域
            # 创建初始掩码 - 确保只在0值的区域应用修复效果
            initial_mask = mask_np.copy()
            
            # 参考MI-GAN项目的后处理方法，创建多层次的渐变掩码
            # 创建一个渐变掩码，从中心向外缓慢渐变
            blend_mask = initial_mask.copy()
            
            # 调整内核大小，使混合更自然
            kernel_size = max(3, int(5 * strength / 5.0))  # 根据修复强度调整内核大小
            blur_sigma = min(2.0, strength / 2.5)  # 根据修复强度调整模糊程度
            
            logger.info(f"后处理参数：内核大小={kernel_size}, 模糊强度={blur_sigma}, 修复强度={strength}")
            
            # 多次应用不同大小的高斯模糊，创建更自然的渐变效果
            # 这类似于MI-GAN项目中的多尺度处理
            blend_mask_small = cv2.GaussianBlur(blend_mask, (3, 3), 0.5)
            blend_mask_medium = cv2.GaussianBlur(blend_mask, (kernel_size, kernel_size), blur_sigma)
            blend_mask_large = cv2.GaussianBlur(blend_mask, (kernel_size*2+1, kernel_size*2+1), blur_sigma*2)
            
            # 加权融合不同尺度的掩码
            blend_mask = blend_mask_small * 0.5 + blend_mask_medium * 0.3 + blend_mask_large * 0.2
            
            # 确保模糊后的掩码不会影响原始掩码之外的区域
            # 创建一个安全边界
            safety_mask = initial_mask.copy()
            
            # 根据修复强度调整安全区域
            safety_padding = max(1, int(strength))
            kernel = np.ones((safety_padding, safety_padding), np.uint8)
            safety_mask = cv2.dilate(safety_mask, kernel, iterations=1)
            
            # 确保模糊掩码只在安全区域内部有效
            blend_mask[safety_mask == 255] = 255
            
            # 归一化掩码值到0-1之间
            blend_mask_norm = blend_mask / 255.0 if blend_mask.max() > 1 else blend_mask
            
            # 应用修复强度，支持更大范围的强度值(0-5.0)
            # 将修复强度值映射到合理范围
            effective_strength = min(1.0, strength / 5.0) if strength > 1.0 else strength
            
            # 掩码为0的区域应用修复效果，为255的区域保持原始图像
            blend_factor = (1.0 - blend_mask_norm) * effective_strength
            
            # 应用非线性增强，提高细节表现
            # 参考StyleGAN和MI-GAN的非线性激活
            if strength > 1.0:
                # 对于高强度值，使用更激进的非线性增强
                power = min(3.0, strength / 2.0)  # 允许最高3次方
                blend_factor = np.power(blend_factor, 1.0/power)  # 反向幂函数使效果更明显
                logger.info(f"应用非线性增强，幂次={1.0/power}")
            
            # 掩码必须在正确的形状上进行广播
            if len(blend_factor.shape) == 2 and len(original_image.shape) == 3:
                # 将掩码扩展为3D以匹配图像通道
                blend_factor = np.expand_dims(blend_factor, axis=2)
                # 重复通道维度
                blend_factor = np.repeat(blend_factor, 3, axis=2)
            
            # 保存调试用的掩码图像
            debug_dir = get_debug_dir()
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存各种掩码以便调试
            cv2.imwrite(os.path.join(debug_dir, "pp_initial_mask.png"), initial_mask)
            cv2.imwrite(os.path.join(debug_dir, "pp_blend_mask.png"), blend_mask)
            blend_factor_visual = (blend_factor * 255).astype(np.uint8)
            if len(blend_factor_visual.shape) == 3:
                blend_factor_visual = blend_factor_visual[:,:,0]
            cv2.imwrite(os.path.join(debug_dir, "pp_blend_factor.png"), blend_factor_visual)
            
            # 常规混合 - 混合两个图像：blend_factor值越高的区域使用修复后的图像，值为0的区域保持原始图像不变
            blended_image = blend_factor * inpainted_image + (1.0 - blend_factor) * original_image
            
            # 确保结果是uint8类型
            blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
            
            return blended_image
            
        except Exception as e:
            logger.error(f"后处理融合步骤出错: {e}", exc_info=True)
            # 如果出错，简单地只替换掩码指定的区域
            try:
                # 原始掩码：0表示需要修复的区域
                mask_norm = mask_np / 255.0  # 归一化掩码到0-1
                # 扩展掩码维度以匹配图像通道
                if len(mask_norm.shape) == 2 and len(inpainted_image.shape) == 3:
                    mask_norm = np.expand_dims(mask_norm, axis=2)
                    mask_norm = np.repeat(mask_norm, inpainted_image.shape[2], axis=2)
                
                # 掩码为255(白色)的地方保留原图，为0(黑色)的地方用修复结果
                blended = original_image * mask_norm + inpainted_image * (1 - mask_norm)
                return blended.astype(np.uint8)
            except:
                # 如果还是出错，返回原始修复图像
                return inpainted_image

    def _validate_inputs(self, image, mask):
        """
        验证输入图像和掩码的有效性
        
        参数:
        - image: 输入图像
        - mask: 输入掩码
        
        返回:
        - 是否有效
        """
        try:
            # 检查图像和掩码尺寸是否相同
            if image.size != mask.size:
                logger.error(f"图像大小{image.size}与掩码大小{mask.size}不匹配")
                return False
                
            # 检查图像是否为空
            if image.size[0] == 0 or image.size[1] == 0:
                logger.error("图像尺寸为零")
                return False
                
            # 检查掩码是否有效区域
            mask_np = np.array(mask)
            zeros = np.sum(mask_np == 0)
            total = mask_np.size
            
            # 如果全白或全黑，则没有意义进行修复
            if zeros == 0 or zeros == total:
                logger.error(f"掩码无有效区域: 黑色像素={zeros}, 总像素={total}")
                return False
                
            # 检查掩码比例，太小的区域可能效果不好
            if zeros / total < 0.01:
                logger.warning(f"修复区域过小 (仅 {zeros/total*100:.2f}% 的像素), 可能效果不明显")
                # 虽然区域小，仍然可以尝试修复
                
            # 通过所有验证
            return True
            
        except Exception as e:
            logger.error(f"验证输入时出错: {e}", exc_info=True)
            return False

    def _resize_to_match(self, image, original_shape):
        """
        调整图像大小以匹配原始图像
        
        参数:
        - image: 输入图像
        - original_shape: 原始图像的形状
        
        返回:
        - 调整后的图像
        """
        try:
            # 获取图像的当前大小
            current_height, current_width = image.shape[:2]
            
            # 计算调整比例
            scale_height = original_shape[0] / current_height
            scale_width = original_shape[1] / current_width
            
            # 计算新的图像大小
            new_height = int(current_height * scale_height)
            new_width = int(current_width * scale_width)
            
            # 调整图像大小
            resized_image = cv2.resize(image, (new_width, new_height))
            
            return resized_image
        except Exception as e:
            logger.error(f"调整图像大小时出错: {e}", exc_info=True)
            return image

# 创建一个单例实例
migan = None

def get_migan_inpainter():
    global migan
    if migan is None:
        try:
            logger.info("初始化MI-GAN单例实例")
            migan = MiganInpainter()
        except Exception as e:
            logger.error(f"初始化MI-GAN失败: {e}", exc_info=True)
            return None
    return migan

def reset_migan_inpainter():
    """
    重置MI-GAN修复器，释放资源
    
    如果之前已经初始化了migan实例，该函数会将其设为None，以便后续重新创建
    同时尝试回收内存
    """
    global migan
    if migan is not None:
        logger.info("重置MI-GAN修复器实例")
        migan = None
        
        # 尝试强制进行垃圾回收
        gc.collect()
        
        try:
            if 'onnxruntime' in sys.modules:
                # 如果可能，也清理ONNX运行时资源
                for provider in ort.get_available_providers():
                    ort.disable_provider(provider)
                # 清理ONNX会话列表
                ort.disable_telemetry_events()
        except Exception as e:
            logger.warning(f"清理ONNX资源时出错: {e}")
        
        logger.info("MI-GAN修复器已重置")
        return True
    return False

def is_migan_available():
    """
    检查MI-GAN功能是否可用
    
    返回:
        bool: 如果MI-GAN可用返回True，否则返回False
    """
    global MIGAN_AVAILABLE
    return MIGAN_AVAILABLE

# 初始化MI-GAN可用性 - 尝试获取MI-GAN实例，如果成功则设置MIGAN_AVAILABLE为True
try:
    # 只是检查是否可以成功初始化，而不实际保存实例
    temp_inpainter = MiganInpainter()
    if temp_inpainter:
        MIGAN_AVAILABLE = True
        logger.info("MI-GAN功能测试成功，可用状态：可用")
        # 不保留实例，等到真正需要时再创建
        del temp_inpainter
except Exception as e:
    MIGAN_AVAILABLE = False
    logger.warning(f"MI-GAN功能测试失败，可用状态：不可用。原因: {e}")