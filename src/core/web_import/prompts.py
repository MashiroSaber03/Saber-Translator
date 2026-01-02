"""
网页漫画导入 - 提示词模板
"""

# 默认提取提示词
DEFAULT_EXTRACTION_PROMPT = """你是一个专业的漫画数据提取助手。请针对当前网页执行以下提取任务:

## 1. 交互行为
- 请模拟用户行为，缓慢向下滚动页面至底部，以触发所有采用"懒加载"技术的漫画图片。
- 在滚动过程中，请确保等待图片加载完成，识别并提取真实的漫画内容图片。

## 2. 提取逻辑
- **图片过滤**: 忽略所有加载占位图（如 loading.gif、spacer.gif）、广告图或图标，仅提取属于漫画正文的图片。
- **属性识别**: 优先提取 `data-src`、`data-original`、`original` 或 `file` 等包含真实高清原图地址的属性。如果这些属性不存在，再提取 `src` 属性。
- **元数据**: 提取漫画的名称（comic_title）和当前章节的名称（chapter_title）。

## 3. 数据结构
- 必须按图片在页面中显示的先后顺序提取，并为每张图片分配一个从 1 开始的 `page_number`（页码序号）。
- 最终结果以 JSON 格式输出，包含漫画名称、章节名以及包含序号和图片链接的列表。

## 4. 输出格式 (Valid JSON Only)
严格按照以下 JSON 格式输出，不要包含 Markdown 代码块标记（如 ```json）：

{
  "comic_title": "漫画名称",
  "chapter_title": "第X话 章节标题",
  "pages": [
    {"page_number": 1, "image_url": "https://..."},
    {"page_number": 2, "image_url": "https://..."}
  ],
  "total_pages": 1
}"""

# 系统提示词后缀 (强制 JSON 输出)
JSON_OUTPUT_SUFFIX = """

IMPORTANT: You must respond with valid JSON format only. Do not include any markdown code block markers like ```json or ```. Just output the raw JSON object."""


def get_system_prompt(custom_prompt: str = None, force_json: bool = True) -> str:
    """
    获取完整的系统提示词
    
    Args:
        custom_prompt: 自定义提示词，如果为空则使用默认提示词
        force_json: 是否强制 JSON 输出
    
    Returns:
        完整的系统提示词
    """
    prompt = custom_prompt if custom_prompt else DEFAULT_EXTRACTION_PROMPT
    
    if force_json:
        prompt += JSON_OUTPUT_SUFFIX
    
    return prompt
