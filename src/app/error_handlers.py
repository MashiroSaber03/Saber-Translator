"""
Flask错误处理器

统一处理Flask应用中的各种错误和异常
"""

import logging
import traceback
from flask import jsonify, request
from werkzeug.exceptions import HTTPException

from src.shared.exceptions import (
    ComicTranslatorException,
    DetectionException,
    OCRException,
    TranslationException,
    InpaintingException,
    RenderingException,
    SessionException,
    PluginException,
    ConfigurationException,
    APIException,
    ValidationException,
    ResourceNotFoundException
)

logger = logging.getLogger("ErrorHandler")


def register_error_handlers(app):
    """
    注册所有错误处理器到Flask应用
    
    Args:
        app: Flask应用实例
    """
    
    @app.errorhandler(ValidationException)
    def handle_validation_error(error: ValidationException):
        """处理数据验证错误"""
        logger.warning(f"验证错误: {error}")
        return jsonify({
            'error': 'ValidationError',
            'message': str(error),
            'details': error.details
        }), 400
    
    @app.errorhandler(ResourceNotFoundException)
    def handle_resource_not_found(error: ResourceNotFoundException):
        """处理资源未找到错误"""
        logger.error(f"资源未找到: {error}")
        return jsonify({
            'error': 'ResourceNotFound',
            'message': str(error),
            'resource_type': error.resource_type,
            'resource_path': error.resource_path
        }), 404
    
    @app.errorhandler(DetectionException)
    def handle_detection_error(error: DetectionException):
        """处理气泡检测错误"""
        logger.error(f"检测错误: {error}", exc_info=True)
        return jsonify({
            'error': 'DetectionError',
            'message': f'气泡检测失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(OCRException)
    def handle_ocr_error(error: OCRException):
        """处理OCR识别错误"""
        logger.error(f"OCR错误: {error}", exc_info=True)
        return jsonify({
            'error': 'OCRError',
            'message': f'文字识别失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(TranslationException)
    def handle_translation_error(error: TranslationException):
        """处理翻译服务错误"""
        logger.error(f"翻译错误: {error}", exc_info=True)
        return jsonify({
            'error': 'TranslationError',
            'message': f'翻译失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(InpaintingException)
    def handle_inpainting_error(error: InpaintingException):
        """处理图像修复错误"""
        logger.error(f"修复错误: {error}", exc_info=True)
        return jsonify({
            'error': 'InpaintingError',
            'message': f'图像修复失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(RenderingException)
    def handle_rendering_error(error: RenderingException):
        """处理文本渲染错误"""
        logger.error(f"渲染错误: {error}", exc_info=True)
        return jsonify({
            'error': 'RenderingError',
            'message': f'文本渲染失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(SessionException)
    def handle_session_error(error: SessionException):
        """处理会话管理错误"""
        logger.error(f"会话错误: {error}")
        return jsonify({
            'error': 'SessionError',
            'message': f'会话操作失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(PluginException)
    def handle_plugin_error(error: PluginException):
        """处理插件系统错误"""
        logger.error(f"插件错误: {error}")
        return jsonify({
            'error': 'PluginError',
            'message': f'插件操作失败: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(ConfigurationException)
    def handle_configuration_error(error: ConfigurationException):
        """处理配置错误"""
        logger.error(f"配置错误: {error}")
        return jsonify({
            'error': 'ConfigurationError',
            'message': f'配置错误: {str(error)}',
            'details': error.details
        }), 500
    
    @app.errorhandler(APIException)
    def handle_api_error(error: APIException):
        """处理API调用错误"""
        logger.error(f"API错误 [{error.api_name}]: {error}")
        return jsonify({
            'error': 'APIError',
            'message': str(error),
            'api_name': error.api_name,
            'status_code': error.status_code
        }), 502  # Bad Gateway
    
    @app.errorhandler(ComicTranslatorException)
    def handle_comic_translator_error(error: ComicTranslatorException):
        """处理通用的漫画翻译器错误"""
        logger.error(f"应用错误: {error}", exc_info=True)
        return jsonify({
            'error': 'ApplicationError',
            'message': str(error),
            'details': error.details
        }), 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        """处理HTTP标准异常"""
        logger.warning(f"HTTP错误 {error.code}: {error.description}")
        return jsonify({
            'error': error.name,
            'message': error.description,
            'code': error.code
        }), error.code
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """处理404错误"""
        logger.warning(f"404 - 页面未找到: {request.url}")
        return jsonify({
            'error': 'NotFound',
            'message': f'请求的资源未找到: {request.path}'
        }), 404
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """处理500内部服务器错误"""
        logger.error("500 - 内部服务器错误", exc_info=True)
        return jsonify({
            'error': 'InternalServerError',
            'message': '服务器内部错误，请稍后重试'
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        """处理未捕获的异常"""
        logger.critical(
            f"未捕获的异常: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
        
        # 记录完整的堆栈跟踪
        tb_str = ''.join(traceback.format_tb(error.__traceback__))
        logger.critical(f"堆栈跟踪:\n{tb_str}")
        
        return jsonify({
            'error': 'UnexpectedError',
            'message': '发生未预期的错误',
            'type': type(error).__name__
        }), 500
    
    logger.info("错误处理器已注册")
