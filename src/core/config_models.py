"""
配置数据模型

使用 dataclass 定义配置对象，减少函数参数传递的复杂度。
包含统一的气泡状态模型 BubbleState，用于前后端统一管理气泡渲染参数。
"""

from dataclasses import dataclass, field, fields as dataclass_fields, asdict
from typing import Optional, List, Tuple, Dict, Any
from src.shared import constants


# ============================================================
# BubbleState: 统一的气泡状态模型
# ============================================================

@dataclass
class BubbleState:
    """
    统一的单个气泡状态模型。
    
    所有翻译方法、编辑模式、渲染操作都只操作这个状态。
    前后端共用，通过 to_dict() 和 from_dict() 进行序列化。
    
    命名约定:
    - Python后端使用下划线命名 (snake_case)
    - 前端使用驼峰命名 (camelCase)
    - from_dict() 支持自动转换
    """
    # === 文本内容 ===
    original_text: str = ""           # 原文
    translated_text: str = ""         # 译文
    textbox_text: str = ""            # 文本框解释文本
    
    # === 坐标信息 ===
    coords: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    polygon: List[List[int]] = field(default_factory=list)  # 多边形顶点
    
    # === 渲染参数 ===
    font_size: int = constants.DEFAULT_FONT_SIZE
    font_family: str = constants.DEFAULT_FONT_RELATIVE_PATH
    text_direction: str = constants.DEFAULT_TEXT_DIRECTION  # "vertical" | "horizontal"
    auto_text_direction: str = constants.DEFAULT_TEXT_DIRECTION  # 自动检测的排版方向（始终在检测时计算，不受用户选择影响）
    text_color: str = constants.DEFAULT_TEXT_COLOR
    fill_color: str = constants.DEFAULT_FILL_COLOR       # 单个气泡的填充色
    rotation_angle: float = constants.DEFAULT_ROTATION_ANGLE  # 旋转角度（度）
    position_offset: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    
    # === 描边参数 ===
    stroke_enabled: bool = constants.DEFAULT_STROKE_ENABLED
    stroke_color: str = constants.DEFAULT_STROKE_COLOR
    stroke_width: int = constants.DEFAULT_STROKE_WIDTH
    
    # === 修复参数 ===
    inpaint_method: str = "solid"  # "solid" | "lama"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（用于JSON序列化，发送到前端）。
        使用驼峰命名以便前端直接使用。
        """
        return {
            # 文本内容
            "originalText": self.original_text,
            "translatedText": self.translated_text,
            "textboxText": self.textbox_text,
            # 坐标信息
            "coords": list(self.coords),
            "polygon": self.polygon,
            # 渲染参数
            "fontSize": self.font_size,
            "fontFamily": self.font_family,
            "textDirection": self.text_direction,
            "autoTextDirection": self.auto_text_direction,  # 自动检测的排版方向
            "textColor": self.text_color,
            "fillColor": self.fill_color,
            "rotationAngle": self.rotation_angle,
            "position": self.position_offset,
            # 描边参数
            "strokeEnabled": self.stroke_enabled,
            "strokeColor": self.stroke_color,
            "strokeWidth": self.stroke_width,
            # 修复参数
            "inpaintMethod": self.inpaint_method,
        }
    
    def to_render_dict(self) -> Dict[str, Any]:
        """
        转换为后端渲染函数需要的字典格式（使用下划线命名）。
        用于兼容现有的 render_all_bubbles 函数。
        """
        return {
            "fontSize": self.font_size,
            "fontFamily": self.font_family,
            "text_direction": self.text_direction,
            "position_offset": self.position_offset,
            "text_color": self.text_color,
            "rotation_angle": self.rotation_angle,
            "stroke_enabled": self.stroke_enabled,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BubbleState":
        """
        从字典创建 BubbleState（支持前端驼峰命名自动转换）。
        
        Args:
            data: 来自前端的字典数据（可能使用驼峰命名）
            
        Returns:
            BubbleState 实例
        """
        # 驼峰命名 -> 下划线命名 映射
        camel_to_snake = {
            # 文本内容
            "originalText": "original_text",
            "translatedText": "translated_text",
            "textboxText": "textbox_text",
            "text": "translated_text",  # 兼容旧的 'text' 字段
            # 坐标信息
            "coords": "coords",
            "polygon": "polygon",
            # 渲染参数
            "fontSize": "font_size",
            "fontFamily": "font_family",
            "textDirection": "text_direction",
            "text_direction": "text_direction",  # 兼容后端命名
            "autoTextDirection": "auto_text_direction",  # 自动检测的排版方向
            "auto_text_direction": "auto_text_direction",  # 兼容后端命名
            "textColor": "text_color",
            "text_color": "text_color",  # 兼容后端命名
            "fillColor": "fill_color",
            "rotationAngle": "rotation_angle",
            "rotation_angle": "rotation_angle",  # 兼容后端命名
            "position": "position_offset",
            "position_offset": "position_offset",  # 兼容后端命名
            # 描边参数
            "strokeEnabled": "stroke_enabled",
            "stroke_enabled": "stroke_enabled",
            "strokeColor": "stroke_color",
            "stroke_color": "stroke_color",
            "strokeWidth": "stroke_width",
            "stroke_width": "stroke_width",
            # 修复参数
            "inpaintMethod": "inpaint_method",
        }
        
        # 转换字典键名
        converted = {}
        for key, value in data.items():
            snake_key = camel_to_snake.get(key, key)
            converted[snake_key] = value
        
        # 只保留 BubbleState 定义的字段
        valid_field_names = {f.name for f in dataclass_fields(cls)}
        filtered = {}
        for k, v in converted.items():
            if k in valid_field_names:
                filtered[k] = v
        
        # 处理 coords 可能是列表的情况
        if "coords" in filtered and isinstance(filtered["coords"], list):
            filtered["coords"] = tuple(filtered["coords"])
        
        return cls(**filtered)
    
    def update_from_dict(self, data: Dict[str, Any]) -> "BubbleState":
        """
        使用字典中的值更新当前状态（部分更新）。
        
        Args:
            data: 要更新的字段字典
            
        Returns:
            更新后的 self（支持链式调用）
        """
        temp = BubbleState.from_dict(data)
        for field_info in dataclass_fields(self):
            new_value = getattr(temp, field_info.name)
            # 只更新非默认值的字段
            default_instance = BubbleState()
            if new_value != getattr(default_instance, field_info.name):
                setattr(self, field_info.name, new_value)
        return self


def create_bubble_states_from_response(
    bubble_coords: List[Tuple[int, int, int, int]],
    original_texts: List[str],
    translated_texts: List[str],
    bubble_angles: List[float],
    global_config: "RenderConfig",
    auto_directions: Optional[List[str]] = None
) -> List[BubbleState]:
    """
    从翻译响应数据创建 BubbleState 列表。
    
    这是核心工厂函数，用于在翻译完成后初始化统一状态。
    
    Args:
        bubble_coords: 气泡坐标列表
        original_texts: 原文列表
        translated_texts: 译文列表
        bubble_angles: 检测到的旋转角度列表
        global_config: 全局渲染配置
        auto_directions: 自动检测的排版方向列表 ('v' 或 'h')
        
    Returns:
        BubbleState 列表
    """
    states = []
    for i in range(len(bubble_coords)):
        # 确定自动检测的排版方向（始终计算并保存）
        if auto_directions and i < len(auto_directions):
            auto_direction = "vertical" if auto_directions[i] == "v" else "horizontal"
        else:
            # 没有自动检测结果时，根据宽高比判断
            x1, y1, x2, y2 = bubble_coords[i]
            auto_direction = "vertical" if (y2 - y1) > (x2 - x1) else "horizontal"
        
        # 实际使用的排版方向：跟随全局设置（用户选择）
        direction = global_config.text_direction
        
        # 确定旋转角度
        angle = bubble_angles[i] if i < len(bubble_angles) else global_config.rotation_angle
        
        state = BubbleState(
            original_text=original_texts[i] if i < len(original_texts) else "",
            translated_text=translated_texts[i] if i < len(translated_texts) else "",
            coords=tuple(bubble_coords[i]),
            font_size=global_config.font_size,
            font_family=global_config.font_family,
            text_direction=direction,
            auto_text_direction=auto_direction,  # 保存自动检测的方向
            text_color=global_config.text_color,
            rotation_angle=angle,
            stroke_enabled=global_config.stroke_enabled,
            stroke_color=global_config.stroke_color,
            stroke_width=global_config.stroke_width,
        )
        states.append(state)
    
    return states


def bubble_states_to_api_response(states: List[BubbleState]) -> List[Dict]:
    """
    将 BubbleState 列表转换为 API 响应格式。
    
    Args:
        states: BubbleState 列表
        
    Returns:
        用于 JSON 响应的字典列表
    """
    return [state.to_dict() for state in states]


def bubble_states_from_api_request(data_list: List[Dict]) -> List[BubbleState]:
    """
    从 API 请求数据创建 BubbleState 列表。
    
    Args:
        data_list: 前端发送的气泡状态列表
        
    Returns:
        BubbleState 列表
    """
    return [BubbleState.from_dict(d) for d in data_list]


# ============================================================
# 原有配置类（保持不变）
# ============================================================


@dataclass
class OCRConfig:
    """OCR 配置"""
    engine: str = 'manga_ocr'
    language: str = 'japan'
    
    # 百度 OCR 相关
    baidu_api_key: Optional[str] = None
    baidu_secret_key: Optional[str] = None
    baidu_version: str = 'standard'
    baidu_language: str = 'auto_detect'
    
    # AI Vision OCR 相关
    use_ai_vision_ocr: bool = False
    ai_vision_provider: Optional[str] = None
    ai_vision_api_key: Optional[str] = None
    ai_vision_model_name: Optional[str] = None
    ai_vision_prompt: Optional[str] = None
    use_ai_vision_json_format: bool = False
    custom_ai_vision_base_url: Optional[str] = None
    rpm_limit_ai_vision: int = constants.DEFAULT_rpm_AI_VISION_OCR
    
    # 其他选项
    skip_ocr: bool = False
    provided_coords: Optional[list] = None
    provided_angles: Optional[list] = None  # 前端提供的角度列表


@dataclass
class TranslationConfig:
    """翻译配置"""
    target_language: str = 'zh'
    source_language: str = 'japan'
    
    # 模型配置
    model_provider: str = constants.DEFAULT_MODEL_PROVIDER
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    custom_base_url: Optional[str] = None
    
    # 提示词配置
    prompt_content: Optional[str] = None
    use_textbox_prompt: bool = False
    textbox_prompt_content: Optional[str] = None
    
    # JSON 格式输出
    use_json_format: bool = False
    
    # 速率限制
    rpm_limit: int = constants.DEFAULT_rpm_TRANSLATION
    
    # 重试机制
    max_retries: int = constants.DEFAULT_TRANSLATION_MAX_RETRIES
    
    # 其他选项
    skip_translation: bool = False

    # 是否启用合并多轮翻译 仅deepseek支持
    translate_merge: bool = False
    chat_session: Optional[str] = None


@dataclass
class InpaintingConfig:
    """图像修复配置"""
    use_inpainting: bool = False
    use_lama: bool = False
    lama_model: str = 'lama_mpe'  # LAMA 模型选择: 'lama_mpe' (速度优化) 或 'litelama' (通用)
    remove_only: bool = False  # 仅消除文字模式
    fill_color: str = constants.DEFAULT_FILL_COLOR
    use_precise_mask: bool = False  # 使用模型生成的精确文字掩膜（仅 CTD/Default 支持）
    mask_dilate_size: int = 10  # 掩膜膨胀大小（像素）
    mask_box_expand_ratio: int = 20  # 标注框区域扩大比例（%）


@dataclass
class RenderConfig:
    """文本渲染配置"""
    font_family: str = constants.DEFAULT_FONT_RELATIVE_PATH
    font_size: int = 25
    auto_font_size: bool = False  # 仅用于首次翻译时是否自动计算字号，计算后此标记不再使用
    text_direction: str = constants.DEFAULT_TEXT_DIRECTION
    auto_text_direction: bool = False  # 自动排版：根据检测结果自动判断每个气泡的排版方向
    text_color: str = constants.DEFAULT_TEXT_COLOR
    rotation_angle: int = constants.DEFAULT_ROTATION_ANGLE
    
    stroke_enabled: bool = constants.DEFAULT_STROKE_ENABLED
    stroke_color: str = constants.DEFAULT_STROKE_COLOR
    stroke_width: int = constants.DEFAULT_STROKE_WIDTH


@dataclass
class TranslationRequest:
    """
    完整的翻译请求配置
    
    整合所有子配置，用于 process_image_translation 函数。
    """
    # 必需参数
    # image_pil 不包含在dataclass中，因为它是Image对象，不适合序列化
    
    # 子配置对象
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    translation_config: TranslationConfig = field(default_factory=TranslationConfig)
    inpainting_config: InpaintingConfig = field(default_factory=InpaintingConfig)
    render_config: RenderConfig = field(default_factory=RenderConfig)
    
    # 检测配置
    conf_threshold: float = 0.6
    detector_type: str = constants.DEFAULT_DETECTOR  # 文本检测器类型 ('ctd' 或 'yolo')
    
    # 文本框扩展参数
    box_expand_ratio: float = 0.0      # 整体扩展比例 (%)
    box_expand_top: float = 0.0        # 上边额外扩展 (%)
    box_expand_bottom: float = 0.0     # 下边额外扩展 (%)
    box_expand_left: float = 0.0       # 左边额外扩展 (%)
    box_expand_right: float = 0.0      # 右边额外扩展 (%)
    
    # 调试选项
    show_detection_debug: bool = False  # 是否显示检测框调试（在翻译结果上画出原始检测框）
    
    # 错误处理
    ignore_connection_errors: bool = True
    
    # 预设文本（跳过OCR时使用）
    preset_texts: Optional[List[str]] = None
    preset_coords: Optional[List[Tuple[int, int, int, int]]] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TranslationRequest':
        """
        从字典创建配置对象（用于API请求解析）
        
        Args:
            data: 包含配置参数的字典（通常来自 Flask request.get_json()）
            
        Returns:
            TranslationRequest 实例
        """
        # OCR 配置
        ocr_config = OCRConfig(
            engine=data.get('ocr_engine', 'manga_ocr'),
            language=data.get('source_language', 'japan'),
            baidu_api_key=data.get('baidu_api_key'),
            baidu_secret_key=data.get('baidu_secret_key'),
            baidu_version=data.get('baidu_version', 'standard'),
            baidu_language=data.get('baidu_ocr_language', 'auto_detect'),
            use_ai_vision_ocr=data.get('use_ai_vision_ocr', False),
            ai_vision_provider=data.get('ai_vision_provider'),
            ai_vision_api_key=data.get('ai_vision_api_key'),
            ai_vision_model_name=data.get('ai_vision_model_name'),
            ai_vision_prompt=data.get('ai_vision_ocr_prompt'),
            use_ai_vision_json_format=data.get('use_json_format_ai_vision_ocr', False),
            custom_ai_vision_base_url=data.get('custom_ai_vision_base_url'),
            rpm_limit_ai_vision=int(data.get('rpm_limit_ai_vision_ocr', constants.DEFAULT_rpm_AI_VISION_OCR)),
            skip_ocr=data.get('skip_ocr', False),
            provided_coords=data.get('bubble_coords'),
            provided_angles=data.get('bubble_angles'),
        )
        
        # 翻译配置
        translation_config = TranslationConfig(
            target_language=data.get('target_language', 'zh'),
            source_language=data.get('source_language', 'japan'),
            model_provider=data.get('model_provider', constants.DEFAULT_MODEL_PROVIDER),
            model_name=data.get('model_name'),
            api_key=data.get('api_key'),
            custom_base_url=data.get('custom_base_url'),
            prompt_content=data.get('prompt_content'),
            use_textbox_prompt=data.get('use_textbox_prompt', False),
            textbox_prompt_content=data.get('textbox_prompt_content'),
            use_json_format=data.get('use_json_format_translation', False),
            rpm_limit=int(data.get('rpm_limit_translation', constants.DEFAULT_rpm_TRANSLATION)),
            max_retries=int(data.get('max_retries', constants.DEFAULT_TRANSLATION_MAX_RETRIES)),
            skip_translation=data.get('skip_translation', False),
            translate_merge=data.get('translate_merge', False),
            chat_session=data.get('chat_session', None),
        )
        
        # 修复配置
        inpainting_config = InpaintingConfig(
            use_inpainting=data.get('use_inpainting', False),
            use_lama=data.get('use_lama', False),
            lama_model=data.get('lamaModel', 'lama_mpe'),  # 'lama_mpe' 或 'litelama'
            remove_only=data.get('remove_only', False),
            fill_color=data.get('fillColor', constants.DEFAULT_FILL_COLOR),
            use_precise_mask=data.get('usePreciseMask', False),
            mask_dilate_size=int(data.get('maskDilateSize', 20)),
            mask_box_expand_ratio=int(data.get('maskBoxExpandRatio', 20)),
        )
        
        # 渲染配置
        render_config = RenderConfig(
            font_family=data.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH),
            font_size=int(data.get('fontSize', 25)) if not data.get('autoFontSize') else 25,
            auto_font_size=data.get('autoFontSize', False),
            text_direction=data.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
            auto_text_direction=data.get('autoTextDirection', False),  # 自动排版
            text_color=data.get('textColor', constants.DEFAULT_TEXT_COLOR),
            rotation_angle=int(data.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)),
            stroke_enabled=data.get('strokeEnabled', constants.DEFAULT_STROKE_ENABLED),
            stroke_color=data.get('strokeColor', constants.DEFAULT_STROKE_COLOR),
            stroke_width=int(data.get('strokeWidth', constants.DEFAULT_STROKE_WIDTH)),
        )
        
        return cls(
            ocr_config=ocr_config,
            translation_config=translation_config,
            inpainting_config=inpainting_config,
            render_config=render_config,
            conf_threshold=float(data.get('conf_threshold', 0.6)),
            detector_type=data.get('detector_type', constants.DEFAULT_DETECTOR),
            # 文本框扩展参数
            box_expand_ratio=float(data.get('box_expand_ratio', 0)),
            box_expand_top=float(data.get('box_expand_top', 0)),
            box_expand_bottom=float(data.get('box_expand_bottom', 0)),
            box_expand_left=float(data.get('box_expand_left', 0)),
            box_expand_right=float(data.get('box_expand_right', 0)),
            # 调试选项
            show_detection_debug=data.get('showDetectionDebug', False),
            ignore_connection_errors=data.get('ignore_connection_errors', True),
            preset_texts=data.get('preset_texts'),
            preset_coords=data.get('preset_coords'),
        )
    
    def to_dict(self) -> dict:
        """
        转换为字典（用于序列化）
        
        Returns:
            包含所有配置参数的字典
        """
        return {
            'ocr_config': {
                'engine': self.ocr_config.engine,
                'language': self.ocr_config.language,
                'use_ai_vision_ocr': self.ocr_config.use_ai_vision_ocr,
                'skip_ocr': self.ocr_config.skip_ocr,
            },
            'translation_config': {
                'target_language': self.translation_config.target_language,
                'source_language': self.translation_config.source_language,
                'model_provider': self.translation_config.model_provider,
                'skip_translation': self.translation_config.skip_translation,
                'translate_merge': self.translation_config.translate_merge,
                'chat_session': self.translation_config.chat_session,
            },
            'inpainting_config': {
                'use_inpainting': self.inpainting_config.use_inpainting,
                'use_lama': self.inpainting_config.use_lama,
                'remove_only': self.inpainting_config.remove_only,
            },
            'render_config': {
                'font_family': self.render_config.font_family,
                'font_size': self.render_config.font_size,
                'text_direction': self.render_config.text_direction,
            },
            'conf_threshold': self.conf_threshold,
        }
