"""
API路由重定向配置

用于保持向后兼容性，将旧的根路径API重定向到新的/api/前缀路径
"""

from flask import redirect, request

# 重定向映射表: {旧路径: 新路径}
REDIRECT_MAPPING = {
    # 配置API
    '/get_prompts': '/api/get_prompts',
    '/get_textbox_prompts': '/api/get_textbox_prompts',
    '/get_used_models': '/api/get_used_models',
    '/get_model_info': '/api/get_model_info',
    '/save_model_info': '/api/save_model_info',
    '/get_prompt_content': '/api/get_prompt_content',
    '/save_prompt': '/api/save_prompt',
    '/reset_prompt_to_default': '/api/reset_prompt_to_default',
    '/delete_prompt': '/api/delete_prompt',
    '/get_textbox_prompt_content': '/api/get_textbox_prompt_content',
    '/save_textbox_prompt': '/api/save_textbox_prompt',
    '/reset_textbox_prompt_to_default': '/api/reset_textbox_prompt_to_default',
    '/delete_textbox_prompt': '/api/delete_textbox_prompt',
    
    # 翻译API
    '/translate_image': '/api/translate_image',
    '/re_render_image': '/api/re_render_image',
    '/re_render_single_bubble': '/api/re_render_single_bubble',
    '/apply_settings_to_all_images': '/api/apply_settings_to_all_images',
    '/translate_single_text': '/api/translate_single_text',
    
    # 系统API
    '/upload_pdf': '/api/upload_pdf',
    '/clean_debug_files': '/api/clean_debug_files',
    '/test_ollama_connection': '/api/test_ollama_connection',
    '/test_lama_repair': '/api/test_lama_repair',
    '/test_params': '/api/test_params',
    '/test_sakura_connection': '/api/test_sakura_connection',
    '/test_baidu_translate_connection': '/api/test_baidu_translate_connection',
}


def create_redirect_handler(new_path: str):
    """
    创建重定向处理函数
    
    Args:
        new_path: 新的路径
        
    Returns:
        Flask视图函数
    """
    def handler():
        # 保持查询参数
        query_string = request.query_string.decode()
        if query_string:
            target = f"{new_path}?{query_string}"
        else:
            target = new_path
        # 使用307保持原始HTTP方法
        return redirect(target, code=307)
    
    # 设置函数名以避免冲突
    handler.__name__ = f"redirect_{new_path.replace('/', '_')}"
    return handler


def register_redirects(app):
    """
    批量注册重定向路由
    
    Args:
        app: Flask应用实例
    """
    count = 0
    for old_path, new_path in REDIRECT_MAPPING.items():
        # 自动处理GET和POST方法
        methods = ['GET', 'POST']
        app.add_url_rule(
            old_path,
            endpoint=f"redirect{old_path.replace('/', '_')}",
            view_func=create_redirect_handler(new_path),
            methods=methods
        )
        count += 1
    
    return count
