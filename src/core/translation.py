import logging
import time
import requests # 用于 Ollama 和 Sakura
import json # 用于解析错误响应
from openai import OpenAI # 用于 SiliconFlow 和 DeepSeek
import os # 用于测试代码读取环境变量
import sys
from pathlib import Path
from src.shared import constants # 导入常量
from src.interfaces.baidu_translate_interface import baidu_translate, BaiduTranslateInterface # 导入百度翻译接口
from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface # 导入有道翻译接口
import re # 增加re模块导入

# 添加项目根目录到 Python 路径以解决导入问题
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 现在可以安全地导入项目内模块
try:
    from src.shared import constants # 导入常量
    from src.interfaces.baidu_translate_interface import baidu_translate # 导入百度翻译接口
except ImportError as e:
    print(f"警告: 无法导入相关模块: {e}")
    # 为测试创建默认常量
    class Constants: 
        DEFAULT_PROMPT = "你是一个好用的翻译助手。请将我的非中文语句段落连成一句或几句话并翻译成中文。"
        BAIDU_TRANSLATE_ENGINE_ID = 'baidu_translate'
        PROJECT_TO_BAIDU_TRANSLATE_LANG_MAP = {'auto': 'auto', 'zh': 'zh', 'en': 'en'}
    constants = Constants()
    # 创建空百度翻译对象
    class MockBaiduTranslate:
        def translate(self, *args, **kwargs):
            return "百度翻译测试"
        def set_credentials(self, *args, **kwargs):
            pass
    baidu_translate = MockBaiduTranslate()

# 全局变量用于缓存API实例
baidu_translate = BaiduTranslateInterface()
youdao_translate = YoudaoTranslateInterface()

logger = logging.getLogger("CoreTranslation")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- rpm Limiting Globals for Translation ---
_translation_rpm_last_reset_time_container = [0]
_translation_rpm_request_count_container = [0]
# ------------------------------------------

def _enforce_rpm_limit(rpm_limit: int, service_name: str, last_reset_time_ref: list, request_count_ref: list):
    """
    执行rpm（每分钟请求数）限制检查和等待。
    使用列表作为引用类型来修改外部的 last_reset_time 和 request_count。

    Args:
        rpm_limit (int): 每分钟最大请求数。如果为0或负数，则不限制。
        service_name (str): 服务名称，用于日志记录。
        last_reset_time_ref (list): 包含上次重置时间的列表 (e.g., [timestamp])。
        request_count_ref (list): 包含当前请求计数的列表 (e.g., [count])。
    """
    if rpm_limit <= 0:
        return # 无限制

    current_time = time.time()

    # 检查是否需要重置窗口
    if current_time - last_reset_time_ref[0] >= 60:
        logger.info(f"rpm: {service_name} - 1分钟窗口已过，重置计数器和时间。")
        last_reset_time_ref[0] = current_time
        request_count_ref[0] = 0

    # 检查是否达到rpm限制
    if request_count_ref[0] >= rpm_limit:
        time_to_wait = 60 - (current_time - last_reset_time_ref[0])
        if time_to_wait > 0:
            logger.info(f"rpm: {service_name} - 已达到每分钟 {rpm_limit} 次请求上限。将等待 {time_to_wait:.2f} 秒...")
            time.sleep(time_to_wait)
            # 等待结束后，这是一个新的窗口
            last_reset_time_ref[0] = time.time() # 更新为当前时间
            request_count_ref[0] = 0
        else:
            # 理论上不应该到这里，因为上面的窗口重置逻辑会处理
            logger.info(f"rpm: {service_name} - 窗口已过但计数未重置，立即重置。")
            last_reset_time_ref[0] = current_time
            request_count_ref[0] = 0
    
    # 如果是窗口内的第一次请求，设置窗口开始时间
    if request_count_ref[0] == 0 and last_reset_time_ref[0] == 0: # 或者 last_reset_time_ref[0] 远早于 current_time - 60
        last_reset_time_ref[0] = current_time
        logger.info(f"rpm: {service_name} - 启动新的1分钟请求窗口。")

    request_count_ref[0] += 1
    logger.debug(f"rpm: {service_name} - 当前窗口请求计数: {request_count_ref[0]}/{rpm_limit if rpm_limit > 0 else '无限制'}")

# 添加安全JSON解析函数
def _safely_extract_from_json(json_str, field_name):
    """
    安全地从JSON字符串中提取特定字段，处理各种异常情况。
    
    Args:
        json_str (str): JSON格式的字符串
        field_name (str): 要提取的字段名
        
    Returns:
        str: 提取的文本，如果失败则返回简化处理的原始文本
    """
    # 尝试直接解析
    try:
        data = json.loads(json_str)
        if field_name in data:
            return data[field_name]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    
    # 解析失败，尝试使用正则表达式提取
    try:
        # 匹配 "field_name": "内容" 或 "field_name":"内容" 的模式
        pattern = r'"' + re.escape(field_name) + r'"\s*:\s*"(.+?)"'
        # 多行模式，使用DOTALL
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            # 反转义提取的文本
            extracted = match.group(1)
            # 处理转义字符
            extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return extracted
    except Exception:
        pass
    
    # 如果依然失败，尝试清理明显的JSON结构，仅保留文本内容
    try:
        # 删除常见JSON结构字符
        cleaned = re.sub(r'[{}"\[\]]', '', json_str)
        # 删除字段名和冒号
        cleaned = re.sub(fr'{field_name}\s*:', '', cleaned)
        # 删除多余空白
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception:
        # 所有方法都失败，返回原始文本
        return json_str

def translate_single_text(text, target_language, model_provider, 
                          api_key=None, model_name=None, prompt_content=None, 
                          use_json_format=False, custom_base_url=None,
                          rpm_limit_translation: int = constants.DEFAULT_rpm_TRANSLATION,
                          jsonPromptMode: str = 'normal'): # <--- 新增rpm参数
    """
    使用指定的大模型翻译单段文本。

    Args:
        text (str): 需要翻译的原始文本。
        target_language (str): 目标语言代码 (例如 'zh')。
        model_provider (str): 模型提供商 ('siliconflow', 'deepseek', 'volcano', 'ollama', 'sakura', 'caiyun', 'baidu_translate', 'youdao_translate')。
        api_key (str, optional): API 密钥 (对于非本地部署是必需的)。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词。如果为 None，使用默认提示词。
        use_json_format (bool): 是否期望并解析JSON格式的响应。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL。
        rpm_limit_translation (int): 翻译服务的每分钟请求数限制。
    Returns:
        str: 翻译后的文本，如果失败则返回 "翻译失败: [原因]"。
    """
    if not text or not text.strip():
        return ""

    if prompt_content is None:
        # 根据是否使用 JSON 格式选择默认提示词
        if use_json_format:
            prompt_content = constants.DEFAULT_TRANSLATE_JSON_PROMPT
        else:
            prompt_content = constants.DEFAULT_PROMPT
    elif use_json_format and '"translated_text"' not in prompt_content: # 如果用户传入了自定义提示词但不是JSON格式
        logger.warning("期望JSON格式输出，但提供的翻译提示词可能不是JSON格式。")

    logger.info(f"开始翻译文本: '{text[:30]}...' (服务商: {model_provider}, rpm: {rpm_limit_translation if rpm_limit_translation > 0 else '无'})")

    max_retries = 3
    retry_count = 0
    translated_text = "翻译失败: 未知错误"

    # --- rpm Enforcement ---
    # 使用容器来传递引用
    _enforce_rpm_limit(
        rpm_limit_translation,
        f"Translation ({model_provider})",
        _translation_rpm_last_reset_time_container,
        _translation_rpm_request_count_container
    )
    # ---------------------

    while retry_count < max_retries:
        try:
            if model_provider == 'siliconflow':
                # SiliconFlow (硅基流动) 使用 OpenAI 兼容 API
                if not api_key:
                    raise ValueError("SiliconFlow需要API Key")
                client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text},
                    ]
                )
                translated_text = response.choices[0].message.content.strip()
                
            elif model_provider == 'deepseek':
                # DeepSeek 也使用 OpenAI 兼容 API
                if not api_key:
                    raise ValueError("DeepSeek需要API Key")
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text},
                    ]
                )
                translated_text = response.choices[0].message.content.strip()
                
            elif model_provider == 'volcano':
                # 火山引擎，也使用 OpenAI 兼容 API
                if not api_key: raise ValueError("火山引擎需要 API Key")
                client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text},
                    ]
                )
                translated_text = response.choices[0].message.content.strip()

            elif model_provider == 'caiyun':
                if not api_key: raise ValueError("彩云小译需要 API Key")
                url = "http://api.interpreter.caiyunai.com/v1/translator"
                # 确定翻译方向，默认为 auto2zh（自动检测源语言翻译到中文）
                trans_type = "auto2zh"
                if target_language == 'en':
                    trans_type = "zh2en"
                elif target_language == 'ja':
                    trans_type = "zh2ja"
                # 也可以基于源语言确定翻译方向
                if 'japan' in model_name or 'ja' in model_name:
                    trans_type = "ja2zh"
                elif 'en' in model_name:
                    trans_type = "en2zh"
                
                headers = {
                    "Content-Type": "application/json",
                    "X-Authorization": f"token {api_key}"
                }
                payload = {
                    "source": [text],
                    "trans_type": trans_type,
                    "request_id": f"comic_translator_{int(time.time())}",
                    "detect": True,
                    "media": "text"
                }
                
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                if "target" in result and len(result["target"]) > 0:
                    translated_text = result["target"][0].strip()
                else:
                    raise ValueError(f"彩云小译返回格式错误: {result}")

            elif model_provider == 'sakura':
                url = "http://localhost:8080/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                sakura_prompt = "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": sakura_prompt},
                        {"role": "user", "content": f"将下面的日文文本翻译成中文：{text}"}
                    ]
                }
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                translated_text = result['choices'][0]['message']['content'].strip()

            elif model_provider == 'ollama':
                url = "http://localhost:11434/api/chat"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text}
                    ],
                    "stream": False
                }
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                if "message" in result and "content" in result["message"]:
                    translated_text = result["message"]["content"].strip()
                else:
                    raise ValueError(f"Ollama返回格式错误: {result}")
                    
            elif model_provider == constants.BAIDU_TRANSLATE_ENGINE_ID:
                # 百度翻译API
                if not api_key or (isinstance(api_key, str) and not api_key.strip()):
                    raise ValueError("百度翻译API需要appid")
                if not model_name or (isinstance(model_name, str) and not model_name.strip()):
                    raise ValueError("百度翻译API需要appkey")
                    
                # 设置百度翻译接口的认证信息
                baidu_translate.set_credentials(api_key, model_name)
                
                # 将项目内部语言代码转换为百度翻译API支持的语言代码
                from_lang = 'auto'  # 默认自动检测源语言
                to_lang = constants.PROJECT_TO_BAIDU_TRANSLATE_LANG_MAP.get(target_language, 'zh')
                
                # 调用百度翻译接口
                translated_text = baidu_translate.translate(text, from_lang, to_lang)
            
            elif model_provider == constants.YOUDAO_TRANSLATE_ENGINE_ID:
                # 有道翻译API
                if not api_key or (isinstance(api_key, str) and not api_key.strip()):
                    raise ValueError("有道翻译API需要AppKey")
                if not model_name or (isinstance(model_name, str) and not model_name.strip()):
                    raise ValueError("有道翻译API需要AppSecret")
                    
                # 设置有道翻译接口的认证信息
                youdao_translate.app_key = api_key
                youdao_translate.app_secret = model_name
                
                # 将项目内部语言代码转换为有道翻译API支持的语言代码
                from_lang = 'auto'  # 默认自动检测源语言
                to_lang = constants.PROJECT_TO_YOUDAO_TRANSLATE_LANG_MAP.get(target_language, 'zh-CHS')
                
                # 调用有道翻译接口
                translated_text = youdao_translate.translate(text, from_lang, to_lang)
            elif model_provider.lower() == 'gemini':
                if not api_key:
                    raise ValueError("Gemini 需要 API Key")
                if not model_name:
                    raise ValueError("Gemini 需要模型名称 (例如 gemini-1.5-flash-latest)")

                client = OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" # 根据教程
                )
                
                gemini_messages = []
                # System prompt 对于 Gemini 的 OpenAI 兼容层是否有效需要测试
                # 教程中的 chat completion 示例包含 system role
                if prompt_content:
                    gemini_messages.append({"role": "system", "content": prompt_content})
                # 用户输入是实际的待翻译文本
                gemini_messages.append({"role": "user", "content": text}) 

                logger.debug(f"Gemini 文本翻译请求 (模型: {model_name}): {json.dumps(gemini_messages, ensure_ascii=False)}")

                response = client.chat.completions.create(
                    model=model_name,
                    messages=gemini_messages,
                )
                translated_text = response.choices[0].message.content.strip()
                logger.info(f"Gemini 文本翻译成功，模型: {model_name}")
                logger.info(f"Gemini 翻译结果 (前100字符): {translated_text[:100]}")
            elif model_provider == constants.CUSTOM_OPENAI_PROVIDER_ID:
                if not api_key:
                    raise ValueError("自定义 OpenAI 兼容服务需要 API Key")
                if not model_name:
                    raise ValueError("自定义 OpenAI 兼容服务需要模型名称")
                if not custom_base_url: # 检查 custom_base_url
                    raise ValueError("自定义 OpenAI 兼容服务需要 Base URL")

                logger.info(f"使用自定义 OpenAI 兼容服务: Base URL='{custom_base_url}', Model='{model_name}'")
                client = OpenAI(api_key=api_key, base_url=custom_base_url) # 使用 custom_base_url
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text},
                    ],
                )
                translated_text = response.choices[0].message.content.strip()
            else:
                raise ValueError(f"不支持的翻译服务提供商: {model_provider}")
                
            if use_json_format:
                try:
                    extracted_text = _safely_extract_from_json(translated_text, "translated_text")
                    logger.info(f"成功从JSON响应中提取翻译文本: '{extracted_text}'")
                    return extracted_text
                except Exception as e:
                    logger.warning(f"无法将翻译结果解析为JSON，将尝试提取文本。原始响应: {translated_text}")
                    return _safely_extract_from_json(translated_text, "translated_text")
            
            break
            
        except Exception as e:
            retry_count += 1
            error_message = str(e)
            logger.error(f"翻译失败（尝试 {retry_count}/{max_retries}，服务商: {model_provider}）: {error_message}", exc_info=True)
            translated_text = f"翻译失败: {error_message}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"{model_provider} API 错误详情: {error_detail}")
                except json.JSONDecodeError:
                    logger.error(f"{model_provider} API 原始错误响应 (状态码 {e.response.status_code}): {e.response.text}")

            if "API key" in error_message or "appid" in error_message or "appkey" in error_message or "authentication" in error_message.lower() or "Base URL" in error_message: # 新增 "Base URL" 检查
                break # 凭证或配置错误，不重试
            if retry_count < max_retries:
                time.sleep(1)
    
    # 记录翻译结果
    if "翻译失败" in translated_text:
        logger.warning(f"最终翻译失败: '{text}' -> '{translated_text}'")
    else:
        logger.info(f"最终翻译成功: '{text[:30]}...' -> '{translated_text[:30]}...'")
        
    return translated_text


# 添加测试用的 Mock 翻译提供商
def translate_with_mock(text, target_language, api_key=None, model_name=None, prompt_content=None):
    """只用于测试的模拟翻译提供商"""
    if not text or not text.strip():
        return ""
        
    # 简单添加目标语言作为前缀
    translated = f"[测试{target_language}] {text[:15]}..."
    
    # 如果文本为日语，模拟一些简单的翻译规则
    if text and any(ord(c) > 0x3000 for c in text):
        if target_language.lower() in ["chinese", "zh"]:
            translated = f"中文翻译: {text[:15]}..."
        elif target_language.lower() in ["english", "en"]:
            translated = f"English translation: {text[:15]}..."
    
    logger.info(f"Mock 翻译: '{text[:20]}...' -> '{translated}'")
    return translated


def translate_text_list(texts, target_language, model_provider, 
                        api_key=None, model_name=None, prompt_content=None, 
                        use_json_format=False, custom_base_url=None,
                        rpm_limit_translation: int = constants.DEFAULT_rpm_TRANSLATION): # <--- 新增rpm参数
    """
    翻译文本列表中的每一项。

    Args:
        texts (list): 包含待翻译文本字符串的列表。
        target_language (str): 目标语言代码。
        model_provider (str): 模型提供商。
        api_key (str, optional): API 密钥。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词。
        use_json_format (bool): 是否期望并解析JSON格式的响应。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL。
        rpm_limit_translation (int): 翻译服务的每分钟请求数限制。
    Returns:
        list: 包含翻译后文本的列表，顺序与输入列表一致。失败的项包含错误信息。
    """
    translated_texts = []
    if not texts:
        return translated_texts

    logger.info(f"开始批量翻译 {len(texts)} 个文本片段 (使用 {model_provider}, rpm: {rpm_limit_translation if rpm_limit_translation > 0 else '无'})...")
    
    # 特殊处理模拟翻译提供商
    if model_provider.lower() == 'mock':
        logger.info("使用模拟翻译提供商")
        for i, text in enumerate(texts):
            translated = translate_with_mock(
                text,
                target_language,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content
            )
            translated_texts.append(translated)
    else:    
        # 正常翻译流程
        for i, text in enumerate(texts):
            translated = translate_single_text(
                text,
                target_language,
                model_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content,
                use_json_format=use_json_format,
                custom_base_url=custom_base_url,
                rpm_limit_translation=rpm_limit_translation # <--- 传递参数
            )
            translated_texts.append(translated)
    
    logger.info("批量翻译完成。")
    return translated_texts

# --- 测试代码 ---
if __name__ == '__main__':
    # 设置基本的日志配置，以便在测试时查看日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("--- 测试翻译核心逻辑 ---")
    test_text_jp = "これはテストです。"
    test_text_en = "This is a test."

    # --- 配置你的测试 API Key 和模型 ---
    test_api_key_sf = os.environ.get("TEST_SILICONFLOW_API_KEY", None)
    test_model_sf = "alibaba/Qwen1.5-14B-Chat"

    test_api_key_ds = os.environ.get("TEST_DEEPSEEK_API_KEY", None)
    test_model_ds = "deepseek-chat"

    test_api_key_volcano = os.environ.get("TEST_VOLCANO_API_KEY", None)
    test_model_volcano = "deepseek-v3-250324"

    test_model_ollama = "llama3"
    test_model_sakura = "sakura-14b-qwen2.5-v1.0"
    # ------------------------------------

    print(f"\n测试 SiliconFlow ({test_model_sf}):")
    if test_api_key_sf:
        result_sf = translate_single_text(test_text_en, 'zh', 'siliconflow', test_api_key_sf, test_model_sf)
        print(f"  '{test_text_en}' -> '{result_sf}'")
    else:
        print("  跳过 SiliconFlow 测试，未设置 TEST_SILICONFLOW_API_KEY 环境变量。")

    print(f"\n测试 DeepSeek ({test_model_ds}):")
    if test_api_key_ds:
        result_ds = translate_single_text(test_text_en, 'zh', 'deepseek', test_api_key_ds, test_model_ds)
        print(f"  '{test_text_en}' -> '{result_ds}'")
    else:
        print("  跳过 DeepSeek 测试，未设置 TEST_DEEPSEEK_API_KEY 环境变量。")
        
    # 测试百度翻译
    test_baidu_app_id = os.environ.get("TEST_BAIDU_TRANSLATE_APP_ID", None)
    test_baidu_app_key = os.environ.get("TEST_BAIDU_TRANSLATE_APP_KEY", None)
    
    print(f"\n测试 百度翻译 API:")
    if test_baidu_app_id and test_baidu_app_key:
        result_baidu = translate_single_text(test_text_en, 'zh', constants.BAIDU_TRANSLATE_ENGINE_ID, test_baidu_app_id, test_baidu_app_key)
        print(f"  '{test_text_en}' -> '{result_baidu}'")
        
        result_baidu_jp = translate_single_text(test_text_jp, 'zh', constants.BAIDU_TRANSLATE_ENGINE_ID, test_baidu_app_id, test_baidu_app_key)
        print(f"  '{test_text_jp}' -> '{result_baidu_jp}'")
    else:
        print("  跳过百度翻译测试，未设置 TEST_BAIDU_TRANSLATE_APP_ID 或 TEST_BAIDU_TRANSLATE_APP_KEY 环境变量。")

    print(f"\n测试 火山引擎 ({test_model_volcano}):")
    if test_api_key_volcano:
        try:
            result_volcano = translate_single_text(test_text_en, 'zh', 'volcano', test_api_key_volcano, test_model_volcano)
            print(f"  '{test_text_en}' -> '{result_volcano}'")
        except Exception as e:
            print(f"  火山引擎测试出错: {e}")
    else:
        print("  跳过火山引擎测试，未设置 TEST_VOLCANO_API_KEY 环境变量。")

    print(f"\n测试 Ollama ({test_model_ollama}):")
    try:
        requests.get("http://localhost:11434")
        result_ollama = translate_single_text(test_text_en, 'zh', 'ollama', model_name=test_model_ollama)
        print(f"  '{test_text_en}' -> '{result_ollama}'")
    except requests.exceptions.ConnectionError:
        print("  跳过 Ollama 测试，无法连接到 http://localhost:11434。")
    except Exception as e:
         print(f"  Ollama 测试出错: {e}")

    print(f"\n测试 Sakura ({test_model_sakura}):")
    try:
        requests.get("http://localhost:8080")
        result_sakura = translate_single_text(test_text_jp, 'zh', 'sakura', model_name=test_model_sakura)
        print(f"  '{test_text_jp}' -> '{result_sakura}'")
    except requests.exceptions.ConnectionError:
        print("  跳过 Sakura 测试，无法连接到 http://localhost:8080。")
    except Exception as e:
         print(f"  Sakura 测试出错: {e}")

    print("\n--- 测试批量翻译 ---")
    test_list = ["Hello", "World", "これはペンです"]
    # 尝试使用 Ollama 进行批量测试，如果 Ollama 不可用，则此部分会失败
    try:
        requests.get("http://localhost:11434")
        translated_list = translate_text_list(test_list, 'zh', 'ollama', model_name=test_model_ollama)
        print(f"批量翻译结果 ({len(translated_list)}):")
        for i, t in enumerate(translated_list):
            print(f"  '{test_list[i]}' -> '{t}'")
    except requests.exceptions.ConnectionError:
        print("  跳过批量翻译测试，无法连接到 Ollama。")
    except Exception as e:
        print(f"  批量翻译测试出错: {e}")