"""
自定义异常类定义

为漫画翻译项目定义统一的异常体系，便于错误处理和追踪。
"""


class ComicTranslatorException(Exception):
    """
    漫画翻译器基础异常类
    
    所有自定义异常都应继承此类，便于统一捕获和处理。
    """
    def __init__(self, message: str, details: dict = None):
        """
        Args:
            message: 错误消息
            details: 额外的错误详情（可选）
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class DetectionException(ComicTranslatorException):
    """
    气泡检测相关异常
    
    当 YOLO 模型加载失败、检测超时或检测结果异常时抛出。
    """
    pass


class OCRException(ComicTranslatorException):
    """
    OCR 识别相关异常
    
    当 OCR 引擎初始化失败、识别超时或识别结果异常时抛出。
    """
    pass


class TranslationException(ComicTranslatorException):
    """
    翻译服务相关异常
    
    当翻译 API 调用失败、响应格式错误或超时时抛出。
    """
    pass


class InpaintingException(ComicTranslatorException):
    """
    图像修复相关异常
    
    当 LAMA 模型加载失败、修复失败时抛出。
    """
    pass


class RenderingException(ComicTranslatorException):
    """
    文本渲染相关异常
    
    当字体加载失败、渲染参数错误或渲染失败时抛出。
    """
    pass


class SessionException(ComicTranslatorException):
    """
    会话管理相关异常
    
    当会话保存/加载/删除失败时抛出。
    """
    pass


class PluginException(ComicTranslatorException):
    """
    插件系统相关异常
    
    当插件加载、初始化或执行失败时抛出。
    """
    pass


class ConfigurationException(ComicTranslatorException):
    """
    配置相关异常
    
    当配置文件读取失败、参数验证失败时抛出。
    """
    pass


class APIException(ComicTranslatorException):
    """
    API 调用相关异常
    
    当外部 API（百度OCR、有道翻译等）调用失败时抛出。
    """
    def __init__(self, message: str, api_name: str = None, status_code: int = None, response_body: str = None):
        """
        Args:
            message: 错误消息
            api_name: API 服务名称
            status_code: HTTP 状态码
            response_body: 响应体内容
        """
        details = {}
        if api_name:
            details['api_name'] = api_name
        if status_code:
            details['status_code'] = status_code
        if response_body:
            details['response_body'] = response_body[:200]  # 限制长度
        super().__init__(message, details)
        self.api_name = api_name
        self.status_code = status_code
        self.response_body = response_body


class ValidationException(ComicTranslatorException):
    """
    数据验证异常
    
    当输入参数验证失败时抛出。
    """
    pass


class ResourceNotFoundException(ComicTranslatorException):
    """
    资源未找到异常
    
    当模型文件、字体文件或配置文件未找到时抛出。
    """
    def __init__(self, message: str, resource_type: str = None, resource_path: str = None):
        """
        Args:
            message: 错误消息
            resource_type: 资源类型（model, font, config等）
            resource_path: 资源路径
        """
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_path:
            details['resource_path'] = resource_path
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_path = resource_path
