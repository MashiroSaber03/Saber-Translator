import logging
import time
import requests # 用于 Ollama 和 Sakura
import json # 用于解析错误响应
from openai import OpenAI # 用于 SiliconFlow 和 DeepSeek
import os # 用于测试代码读取环境变量
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径以解决导入问题
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 现在可以安全地导入项目内模块
try:
    from src.shared import constants # 导入常量
except ImportError as e:
    print(f"警告: 无法导入constants模块: {e}")
    # 为测试创建默认常量
    class Constants: 
        DEFAULT_PROMPT = "你是一个好用的翻译助手。请将我的非中文语句段落连成一句或几句话并翻译成中文。"
    constants = Constants()

logger = logging.getLogger("CoreTranslation")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def translate_single_text(text, target_language, model_provider, api_key=None, model_name=None, prompt_content=None):
    """
    使用指定的大模型翻译单段文本。

    Args:
        text (str): 需要翻译的原始文本。
        target_language (str): 目标语言代码 (例如 'zh')。
        model_provider (str): 模型提供商 ('siliconflow', 'deepseek', 'ollama', 'sakura')。
        api_key (str, optional): API 密钥 (对于非本地部署是必需的)。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词。如果为 None，使用默认提示词。

    Returns:
        str: 翻译后的文本，如果失败则返回 "翻译失败: [原因]"。
    """
    if not text or not text.strip():
        return ""

    if prompt_content is None:
        prompt_content = constants.DEFAULT_PROMPT

    max_retries = 3
    retry_count = 0
    translated_text = "翻译失败: 未知错误" # 默认失败信息

    while retry_count < max_retries:
        try:
            logger.info(f"尝试翻译 (尝试 {retry_count + 1}/{max_retries}) 使用 {model_provider}/{model_name}: '{text[:30]}...'")
            start_time = time.time()

            if model_provider == 'siliconflow':
                if not api_key: raise ValueError("SiliconFlow 需要 API Key")
                client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text},
                    ],
                    timeout=30
                )
                translated_text = response.choices[0].message.content.strip()

            elif model_provider == 'deepseek':
                if not api_key: raise ValueError("DeepSeek 需要 API Key")
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_content},
                        {"role": "user", "content": text},
                    ],
                    timeout=30
                )
                translated_text = response.choices[0].message.content.strip()

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
                response = requests.post(url, headers=headers, json=payload, timeout=30)
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
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                if "message" in result and "content" in result["message"]:
                    translated_text = result["message"]["content"].strip()
                else:
                    raise ValueError(f"Ollama API 返回格式错误: {result}")

            else:
                logger.error(f"未知的翻译模型提供商: {model_provider}")
                translated_text = "翻译失败: 未知服务商"

            end_time = time.time()
            logger.info(f"翻译成功 (耗时: {end_time - start_time:.2f}s): '{translated_text[:30]}...'")
            return translated_text # 成功则跳出循环并返回

        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            logger.warning(f"翻译连接失败 (尝试 {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                translated_text = "翻译失败: 连接错误"
            else:
                time.sleep(1)
        except requests.exceptions.Timeout as e:
            retry_count += 1
            logger.warning(f"翻译超时 (尝试 {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                translated_text = "翻译失败: 超时"
            else:
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            retry_count += 1
            err_msg = str(e)
            status_code = e.response.status_code if e.response is not None else "N/A"
            logger.error(f"翻译 API 请求失败 (尝试 {retry_count}/{max_retries}, Status: {status_code}): {err_msg}", exc_info=False) # 减少日志冗余
            if e.response is not None:
                try:
                    err_detail = e.response.json()
                    err_msg += f" - Detail: {err_detail}"
                except json.JSONDecodeError:
                    err_msg += f" - Response: {e.response.text[:100]}" # 只记录部分响应文本
            if retry_count >= max_retries:
                translated_text = f"翻译失败: API错误 (Status: {status_code})" # 提供状态码
            else:
                time.sleep(1)
        except ValueError as e: # 捕获特定错误，如 API Key 缺失或格式错误
             retry_count = max_retries # 不再重试此类错误
             logger.error(f"翻译配置或格式错误: {e}", exc_info=True)
             translated_text = f"翻译失败: {e}"
        except Exception as e:
            retry_count += 1
            logger.error(f"翻译过程中发生未知错误 (尝试 {retry_count}/{max_retries}): {e}", exc_info=True)
            if retry_count >= max_retries:
                translated_text = f"翻译失败: {type(e).__name__}"
            else:
                time.sleep(1)

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


def translate_text_list(texts, target_language, model_provider, api_key=None, model_name=None, prompt_content=None):
    """
    翻译文本列表中的每一项。

    Args:
        texts (list): 包含待翻译文本字符串的列表。
        target_language (str): 目标语言代码。
        model_provider (str): 模型提供商。
        api_key (str, optional): API 密钥。
        model_name (str, optional): 模型名称。
        prompt_content (str, optional): 自定义提示词。

    Returns:
        list: 包含翻译后文本的列表，顺序与输入列表一致。失败的项包含错误信息。
    """
    translated_texts = []
    if not texts:
        return translated_texts

    logger.info(f"开始批量翻译 {len(texts)} 个文本片段 (使用 {model_provider})...")
    
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
                prompt_content=prompt_content
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

    print(f"\n测试 Ollama ({test_model_ollama}):")
    try:
        requests.get("http://localhost:11434", timeout=1)
        result_ollama = translate_single_text(test_text_en, 'zh', 'ollama', model_name=test_model_ollama)
        print(f"  '{test_text_en}' -> '{result_ollama}'")
    except requests.exceptions.ConnectionError:
        print("  跳过 Ollama 测试，无法连接到 http://localhost:11434。")
    except Exception as e:
         print(f"  Ollama 测试出错: {e}")

    print(f"\n测试 Sakura ({test_model_sakura}):")
    try:
        requests.get("http://localhost:8080", timeout=1)
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
        requests.get("http://localhost:11434", timeout=1)
        translated_list = translate_text_list(test_list, 'zh', 'ollama', model_name=test_model_ollama)
        print(f"批量翻译结果 ({len(translated_list)}):")
        for i, t in enumerate(translated_list):
            print(f"  '{test_list[i]}' -> '{t}'")
    except requests.exceptions.ConnectionError:
        print("  跳过批量翻译测试，无法连接到 Ollama。")
    except Exception as e:
        print(f"  批量翻译测试出错: {e}")