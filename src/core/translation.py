import logging
import time
import requests
import json
import re
import os
import sys
from pathlib import Path
from openai import OpenAI
from src.shared.openai_helpers import create_openai_client

# 添加项目根目录到 Python 路径
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 导入项目内模块
from src.shared import constants
from src.interfaces.baidu_translate_interface import BaiduTranslateInterface
from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface

# 全局API实例缓存
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

def translate_single_text(text, target_language, model_provider, 
                          api_key=None, model_name=None, prompt_content=None, 
                          use_json_format=False, custom_base_url=None,
                          rpm_limit_translation: int = constants.DEFAULT_rpm_TRANSLATION,
                          max_retries: int = constants.DEFAULT_TRANSLATION_MAX_RETRIES):
    """
    使用指定的大模型翻译单段文本。
    
    注意：此函数用于非 LLM 提供商（如百度翻译）和编辑模式的单气泡重翻译。
    批量翻译请使用 translate_text_list() 函数。

    Args:
        text (str): 需要翻译的原始文本。
        target_language (str): 目标语言代码 (例如 'zh')。
        model_provider (str): 模型提供商。
        api_key (str, optional): API 密钥 (对于非本地部署是必需的)。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词。如果为 None，使用默认提示词。
        use_json_format (bool): [已弃用] 此参数不再使用，保留仅为向后兼容。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL。
        rpm_limit_translation (int): 翻译服务的每分钟请求数限制。
        max_retries (int): 翻译失败时的最大重试次数。
    Returns:
        str: 翻译后的文本，如果失败则返回 "翻译失败: [原因]"。
    """
    if not text or not text.strip():
        return ""

    if prompt_content is None:
        prompt_content = constants.DEFAULT_PROMPT


    logger.info(f"开始翻译文本: '{text[:30]}...' (服务商: {model_provider}, rpm: {rpm_limit_translation if rpm_limit_translation > 0 else '无'}, 重试: {max_retries})")

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
                client = create_openai_client(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
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
                client = create_openai_client(api_key=api_key, base_url="https://api.deepseek.com/v1")
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
                client = create_openai_client(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
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

                client = create_openai_client(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # 根据教程
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
                client = create_openai_client(api_key=api_key, base_url=custom_base_url)  # 使用辅助函数自动处理代理
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


def _assemble_batch_prompt(texts: list, custom_prompt: str = None, use_json_format: bool = False) -> tuple:
    """
    将多个文本组装成批量翻译的 prompt
    
    Args:
        texts: 待翻译的文本列表
        custom_prompt: 自定义提示词 (如果为 None，使用默认批量翻译模板)
        use_json_format: 是否使用 JSON 输出格式
        
    Returns:
        tuple: (messages_list, batch_size) - 消息列表和批次大小
    """
    # 构建消息列表
    messages = []
    
    if use_json_format:
        # --- JSON 模式 ---
        # 1. System prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = constants.BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. Few-shot learning: JSON 格式示例
        if hasattr(constants, 'BATCH_TRANSLATE_JSON_SAMPLE_INPUT') and hasattr(constants, 'BATCH_TRANSLATE_JSON_SAMPLE_OUTPUT'):
            messages.append({"role": "user", "content": constants.BATCH_TRANSLATE_JSON_SAMPLE_INPUT})
            messages.append({"role": "assistant", "content": constants.BATCH_TRANSLATE_JSON_SAMPLE_OUTPUT})
            logger.debug("已添加 JSON 模式翻译示例")
        
        # 3. User prompt：构建 JSON 格式的输入
        import json
        texts_json = {"texts": [{"id": i+1, "text": text} for i, text in enumerate(texts)]}
        user_prompt = constants.BATCH_TRANSLATE_JSON_USER_TEMPLATE + "\n" + json.dumps(texts_json, ensure_ascii=False, indent=2)
        messages.append({"role": "user", "content": user_prompt})
    else:
        # --- 纯文本模式 (默认) ---
        # 1. System prompt
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = constants.BATCH_TRANSLATE_SYSTEM_TEMPLATE
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. Few-shot learning: 添加翻译示例 (日译中)
        if hasattr(constants, 'BATCH_TRANSLATE_SAMPLE_INPUT') and hasattr(constants, 'BATCH_TRANSLATE_SAMPLE_OUTPUT'):
            messages.append({"role": "user", "content": constants.BATCH_TRANSLATE_SAMPLE_INPUT})
            messages.append({"role": "assistant", "content": constants.BATCH_TRANSLATE_SAMPLE_OUTPUT})
            logger.debug("已添加日译中翻译示例")
        
        # 3. User prompt：将所有文本编号并合并
        user_prompt = constants.BATCH_TRANSLATE_USER_TEMPLATE
        for i, text in enumerate(texts):
            user_prompt += f"\n<|{i+1}|>{text}"
        messages.append({"role": "user", "content": user_prompt})
    
    return messages, len(texts)




def _parse_batch_response(response_text: str, expected_count: int) -> list:
    """
    解析批量翻译的响应
    
    Args:
        response_text: LLM 返回的响应文本
        expected_count: 期望的翻译数量
        
    Returns:
        list: 解析后的翻译列表
    """
    # --- 响应清理 ---
    
    # 1. 去除 <think>...</think> 标签及内容 (某些模型的思考过程)
    cleaned_text = re.sub(r'(</think>)?<think>.*?</think>', '', response_text, flags=re.DOTALL)
    
    # 2. 删除多余的空行
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
    
    # 3. 仅保留 <|1|> 到 <|max|> 范围内的行，删除前后的解释性文字
    lines = cleaned_text.splitlines()
    min_index_line = -1
    max_index_line = -1
    has_numeric_prefix = False
    
    for index, line in enumerate(lines):
        match = re.search(r'<\|(\d+)\|>', line)
        if match:
            has_numeric_prefix = True
            current_index = int(match.group(1))
            if current_index == 1:
                min_index_line = index
            if max_index_line == -1:
                max_index_line = index
            else:
                prev_match = re.search(r'<\|(\d+)\|>', lines[max_index_line])
                if prev_match and current_index > int(prev_match.group(1)):
                    max_index_line = index
    
    if has_numeric_prefix and min_index_line != -1:
        # 只保留从 <|1|> 开始到最大编号行的内容
        modified_lines = lines[min_index_line:max_index_line + 1]
        cleaned_text = "\n".join(modified_lines)
    
    # 4. 修复前缀和翻译内容之间的空格问题
    fixed_lines = []
    for line in cleaned_text.strip().split('\n'):
        # 匹配 <|数字|> 前缀格式，去除前缀后的多余空格
        match = re.match(r'^(<\|\d+\|>)\s+(.*)$', line.strip())
        if match:
            prefix = match.group(1)
            content = match.group(2)
            fixed_lines.append(f"{prefix}{content}")
        else:
            fixed_lines.append(line)
    cleaned_text = '\n'.join(fixed_lines)
    
    # --- 分割解析 ---
    
    # 特殊情况：单个查询但响应可能被分成多段 (在分割前检查)
    if expected_count == 1:
        # 检查是否存在多个编号
        all_indices = re.findall(r'<\|(\d+)\|>', cleaned_text)
        if len(all_indices) > 1:
            # 检查是否有超过 1 的索引（说明模型错误地分割了单个翻译）
            has_invalid = any(int(idx) > 1 for idx in all_indices)
            if has_invalid:
                # 合并所有翻译，移除所有编号
                merged = re.sub(r'<\|\d+\|>', '', cleaned_text).strip()
                logger.warning("检测到单查询被分割，已合并翻译结果")
                return [merged]
    
    # 使用正则表达式分割响应：<|1|>...<|2|>...
    translations = re.split(r'<\|\d+\|>', cleaned_text)
    
    # 清理每个翻译的前后空格
    translations = [t.strip() for t in translations]
    
    # 移除第一个空元素（如果存在）
    if translations and not translations[0]:
        translations = translations[1:]
    
    return translations


def _parse_batch_json_response(response_text: str, expected_count: int) -> list:
    """
    解析 JSON 格式的批量翻译响应
    
    Args:
        response_text: LLM 返回的响应文本 (应为 JSON 格式)
        expected_count: 期望的翻译数量
        
    Returns:
        list: 解析后的翻译列表
    """
    import json
    
    # 1. 去除 <think>...</think> 标签及内容
    cleaned_text = re.sub(r'(</think>)?<think>.*?</think>', '', response_text, flags=re.DOTALL)
    
    # 2. 尝试提取 JSON 部分（可能被包裹在 ```json ... ``` 中）
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接找到 JSON 对象
        json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning("无法从响应中提取 JSON，回退到纯文本解析")
            return _parse_batch_response(response_text, expected_count)
    
    # 3. 解析 JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}，回退到纯文本解析")
        return _parse_batch_response(response_text, expected_count)
    
    # 4. 提取翻译结果
    translations = []
    
    # 支持两种格式:
    # 格式1: {"translations": [{"id": 1, "text": "..."}, ...]}
    # 格式2: {"TextList": [{"ID": 1, "text": "..."}, ...]} (备用格式)
    
    if 'translations' in data:
        items = data['translations']
    elif 'TextList' in data:
        items = data['TextList']
    else:
        logger.warning("JSON 格式不正确，找不到 translations 或 TextList 字段")
        return _parse_batch_response(response_text, expected_count)
    
    # 按 id 排序并提取文本
    try:
        # 统一 id 字段名称 (支持 'id' 和 'ID')
        for item in items:
            item_id = item.get('id') or item.get('ID')
            item_text = item.get('text', '')
            translations.append((item_id, item_text))
        
        # 按 id 排序
        translations.sort(key=lambda x: x[0] if x[0] else 0)
        translations = [t[1] for t in translations]
        
    except Exception as e:
        logger.warning(f"提取翻译结果失败: {e}，回退到纯文本解析")
        return _parse_batch_response(response_text, expected_count)
    
    logger.debug(f"JSON 模式解析成功: {len(translations)} 条翻译")
    return translations


def _translate_batch_with_llm(texts: list, model_provider: str,
                               api_key: str, model_name: str, custom_prompt: str = None,
                               custom_base_url: str = None, max_retries: int = 2,
                               use_json_format: bool = False) -> list:
    """
    使用 LLM 进行批量翻译
    
    Args:
        texts: 待翻译的文本列表
        model_provider: 模型提供商
        api_key: API 密钥
        model_name: 模型名称
        custom_prompt: 自定义提示词
        custom_base_url: 自定义 API Base URL
        max_retries: 最大重试次数
        use_json_format: 是否使用 JSON 输出格式
        
    Returns:
        list: 翻译结果列表
    """
    if not texts:
        return []
    
    # 初始化结果列表
    results = [''] * len(texts)
    
    # 组装消息列表 (包含 system prompt、few-shot 示例、user prompt)
    messages, batch_size = _assemble_batch_prompt(texts, custom_prompt, use_json_format)
    
    logger.info(f"批量翻译请求: {batch_size} 个文本片段 (消息数: {len(messages)})")
    
    # 确定 API 客户端配置
    base_url_map = {
        'siliconflow': 'https://api.siliconflow.cn/v1',
        'deepseek': 'https://api.deepseek.com/v1',
        'volcano': 'https://ark.cn-beijing.volces.com/api/v3',
        constants.CUSTOM_OPENAI_PROVIDER_ID: custom_base_url,
    }
    
    # Gemini 使用特殊的 base_url
    if model_provider.lower() == 'gemini':
        base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    else:
        base_url = base_url_map.get(model_provider, custom_base_url)
    
    if not base_url and model_provider not in ['ollama', 'sakura']:
        logger.error(f"未知的模型提供商: {model_provider}")
        return texts  # 返回原文
    
    # 重试循环
    for attempt in range(max_retries + 1):
        try:
            if model_provider == 'ollama':
                # Ollama 特殊处理
                url = "http://localhost:11434/api/chat"
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False
                }
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                response_text = result.get("message", {}).get("content", "").strip()
                
            elif model_provider == 'sakura':
                # Sakura 特殊处理
                url = "http://localhost:8080/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": messages
                }
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
            else:
                # OpenAI 兼容 API
                client = create_openai_client(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=120
                )
                response_text = response.choices[0].message.content.strip()
            
            logger.debug(f"批量翻译响应:\n{response_text[:500]}...")
            
            # 解析响应 (根据模式选择解析函数)
            if use_json_format:
                translations = _parse_batch_json_response(response_text, len(texts))
            else:
                translations = _parse_batch_response(response_text, len(texts))
            
            # 验证响应数量
            if len(translations) != len(texts):
                logger.warning(f"[尝试 {attempt+1}/{max_retries+1}] 翻译数量不匹配: 期望 {len(texts)}, 实际 {len(translations)}")
                
                # 如果翻译数量少于期望，填充原文
                if len(translations) < len(texts):
                    translations.extend(texts[len(translations):])
                # 如果翻译数量多于期望，截断
                elif len(translations) > len(texts):
                    translations = translations[:len(texts)]
            
            # 验证非空翻译
            empty_count = sum(1 for src, trans in zip(texts, translations) 
                             if src.strip() and not trans.strip())
            if empty_count > 0:
                logger.warning(f"[尝试 {attempt+1}/{max_retries+1}] 检测到 {empty_count} 个空翻译")
                if attempt < max_retries:
                    continue  # 重试
            
            # 返回结果
            for i, trans in enumerate(translations):
                results[i] = trans if trans else texts[i]
            
            logger.info(f"批量翻译成功: {len(texts)} 个文本片段")
            return results
            
        except Exception as e:
            logger.error(f"[尝试 {attempt+1}/{max_retries+1}] 批量翻译失败: {e}", exc_info=True)
            if attempt < max_retries:
                time.sleep(1)
                continue
    
    # 所有重试都失败，返回原文
    logger.error("批量翻译所有重试都失败，返回原文")
    return texts


def translate_text_list(texts, target_language, model_provider, 
                        api_key=None, model_name=None, prompt_content=None, 
                        use_json_format=False, custom_base_url=None,
                        rpm_limit_translation: int = constants.DEFAULT_rpm_TRANSLATION,
                        max_retries: int = constants.DEFAULT_TRANSLATION_MAX_RETRIES):
    """
    翻译文本列表 - 使用批量翻译策略
    
    将一页内所有气泡的文本合并为一个请求发送给 LLM，使用 <|n|> 格式编号，
    一次 API 调用翻译整页内容，大幅提升效率和翻译一致性。
    
    注意：目标语言现在由提示词控制（默认翻译为中文），如需修改请编辑 
    constants.BATCH_TRANSLATE_SYSTEM_TEMPLATE 中的提示词。

    Args:
        texts (list): 包含待翻译文本字符串的列表。
        target_language (str): [已弃用] 目标语言代码，现由提示词控制。
        model_provider (str): 模型提供商。
        api_key (str, optional): API 密钥。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词，可覆盖默认提示词。
        use_json_format (bool): 是否使用 JSON 格式输出。True 时使用结构化 JSON 格式，False 时使用 <|n|> 编号格式。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL。
        rpm_limit_translation (int): 翻译服务的每分钟请求数限制。
        max_retries (int): 翻译失败时的最大重试次数。
    Returns:
        list: 包含翻译后文本的列表，顺序与输入列表一致。失败的项包含错误信息。
    """
    if not texts:
        return []
    
    # 过滤空文本，记录索引
    non_empty_indices = []
    non_empty_texts = []
    final_translations = [''] * len(texts)
    
    for i, text in enumerate(texts):
        if text and text.strip():
            non_empty_indices.append(i)
            non_empty_texts.append(text)
        else:
            final_translations[i] = ''
    
    if not non_empty_texts:
        return final_translations
    
    logger.info(f"开始批量翻译 {len(non_empty_texts)} 个文本片段 (使用 {model_provider}, rpm: {rpm_limit_translation if rpm_limit_translation > 0 else '无'})...")
    
    # 特殊处理模拟翻译提供商
    if model_provider.lower() == 'mock':
        logger.info("使用模拟翻译提供商")
        for i, text in enumerate(non_empty_texts):
            translated = translate_with_mock(
                text,
                target_language,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content
            )
            final_translations[non_empty_indices[i]] = translated
        logger.info("批量翻译完成。")
        return final_translations
    
    # 检查是否为支持批量翻译的提供商 (LLM)
    llm_providers = {'siliconflow', 'deepseek', 'volcano', 'ollama', 'sakura', 'gemini', 
                     'custom_openai'}  # 使用字符串而不是常量，便于 .lower() 比较
    
    if model_provider.lower() in llm_providers:
        # --- rpm 限制 ---
        _enforce_rpm_limit(
            rpm_limit_translation,
            f"BatchTranslation ({model_provider})",
            _translation_rpm_last_reset_time_container,
            _translation_rpm_request_count_container
        )
        
        # 使用批量翻译
        # 将文本按字符数分批，避免超过 token 限制
        max_chars = constants.BATCH_TRANSLATE_MAX_CHARS_PER_REQUEST
        batches = []
        current_batch = []
        current_chars = 0
        
        for text in non_empty_texts:
            text_len = len(text) + 10  # +10 用于 <|n|> 标记
            if current_chars + text_len > max_chars and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(text)
            current_chars += text_len
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"文本已分为 {len(batches)} 个批次进行翻译")
        
        # 翻译每个批次
        all_translations = []
        for batch_idx, batch in enumerate(batches):
            logger.info(f"正在翻译批次 {batch_idx + 1}/{len(batches)} ({len(batch)} 个文本)...")
            
            batch_translations = _translate_batch_with_llm(
                batch,
                model_provider,
                api_key,
                model_name,
                custom_prompt=prompt_content,
                custom_base_url=custom_base_url,
                max_retries=max_retries,
                use_json_format=use_json_format
            )
            all_translations.extend(batch_translations)
            
            # 如果有多个批次，在批次之间稍微等待
            if len(batches) > 1 and batch_idx < len(batches) - 1:
                time.sleep(0.5)
        
        # 将翻译结果写回最终列表
        for i, trans in enumerate(all_translations):
            if i < len(non_empty_indices):
                final_translations[non_empty_indices[i]] = trans
        
    else:
        # 非 LLM 提供商 (如百度翻译、有道翻译)，使用原有的逐个翻译逻辑
        logger.info(f"提供商 {model_provider} 不支持批量翻译，使用逐个翻译模式")
        for i, text in enumerate(non_empty_texts):
            translated = translate_single_text(
                text,
                target_language,
                model_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content,
                use_json_format=use_json_format,
                custom_base_url=custom_base_url,
                rpm_limit_translation=rpm_limit_translation,
                max_retries=max_retries
            )
            final_translations[non_empty_indices[i]] = translated
    
    logger.info(f"批量翻译完成。成功 {len([t for t in final_translations if t])} / {len(texts)}")
    return final_translations

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