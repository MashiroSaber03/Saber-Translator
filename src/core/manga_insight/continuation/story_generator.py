"""
Manga Insight 剧情生成器

负责三层递进的剧情生成：
1. 全话脚本生成
2. 每页剧情细化
3. 生图提示词生成
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any

from ..config_utils import load_insight_config
from ..storage import AnalysisStorage
from ..embedding_client import ChatClient
from .models import ChapterScript, PageContent
from .character_manager import CharacterManager

logger = logging.getLogger("MangaInsight.Continuation.StoryGenerator")


class StoryGenerator:
    """剧情生成器"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.config = load_insight_config()
        self.storage = AnalysisStorage(book_id)
        self.chat_client = ChatClient(self.config.chat_llm)
        self.char_manager = CharacterManager(book_id)
    
    async def prepare_continuation_data(self) -> Dict[str, Any]:
        """
        准备续写所需的数据，检查必要内容是否存在（不自动生成）
        
        Returns:
            Dict: {"ready": bool, "generating": bool, "message": str}
        """
        result = {
            "ready": False,
            "generating": False,
            "message": ""
        }
        
        # 1. 检查故事概要是否存在（不自动生成，提示用户手动生成）
        story_summary = await self.storage.load_template_overview("story_summary")
        
        if not story_summary or not story_summary.get("content"):
            logger.info("故事概要不存在，提示用户手动生成")
            result["message"] = "续写功能需要故事概要，请先在「概览」页面选择「故事概要」模板并点击生成"
            return result
        
        # 2. 检查时间线是否存在
        timeline_data = await self.storage.load_timeline()
        
        if not timeline_data or not timeline_data.get("events"):
            result["message"] = "时间线数据不存在或为空，请先完成漫画分析"
            return result
        
        result["ready"] = True
        result["generating"] = False
        result["message"] = "数据准备完成"
        return result
    
    async def generate_chapter_script(
        self,
        user_direction: str = "",
        page_count: int = 15
    ) -> ChapterScript:
        """
        生成全话脚本（第一层）
        
        Args:
            user_direction: 用户指定的续写方向
            page_count: 续写页数
            
        Returns:
            ChapterScript: 生成的脚本
        """
        # 获取必要数据
        story_summary = await self.storage.load_template_overview("story_summary")
        timeline_data = await self.storage.load_timeline()
        
        if not story_summary or not story_summary.get("content"):
            raise ValueError("故事概要不存在，请先调用 prepare_continuation_data")
        
        if not timeline_data:
            raise ValueError("时间线数据不存在")
        
        # 获取最近几页的分析结果作为参考
        reference_pages = await self._get_recent_page_analyses(5)
        
        # 构建提示词
        prompt = self._build_chapter_script_prompt(
            manga_summary=story_summary.get("content", ""),
            timeline_data=timeline_data,
            reference_pages=reference_pages,
            user_direction=user_direction,
            page_count=page_count
        )
        
        # 调用 LLM
        logger.info(f"正在生成全话脚本，页数: {page_count}")
        response = await self.chat_client.generate(prompt)
        
        # 解析响应，提取标题
        script_text = response.strip()
        chapter_title = self._extract_chapter_title(script_text)
        
        return ChapterScript(
            chapter_title=chapter_title,
            page_count=page_count,
            script_text=script_text
        )
    
    async def generate_page_details(
        self,
        chapter_script: ChapterScript,
        page_number: int
    ) -> PageContent:
        """
        生成单页剧情细化（第二层）
        
        Args:
            chapter_script: 全话脚本
            page_number: 页码
            
        Returns:
            PageContent: 页面内容
        """
        timeline_data = await self.storage.load_timeline()
        
        prompt = self._build_page_details_prompt(
            chapter_script=chapter_script.script_text,
            page_number=page_number,
            timeline_data=timeline_data
        )
        
        logger.info(f"正在生成第 {page_number} 页剧情")
        response = await self.chat_client.generate(prompt)
        
        # 解析 JSON 响应
        page_data = self._parse_json_response(response)
        
        return PageContent(
            page_number=page_data.get("page_number", page_number),
            characters=page_data.get("characters", []),
            character_forms=page_data.get("character_forms", []),
            description=page_data.get("description", ""),
            dialogues=page_data.get("dialogues", []),
            mood=page_data.get("mood", "")
        )
    
    async def generate_image_prompt(
        self,
        page_content: PageContent,
        character_descriptions: Dict[str, str]
    ) -> str:
        """
        生成生图提示词（第三层）
        
        Args:
            page_content: 页面内容
            character_descriptions: 角色描述字典 {角色名: 描述}
            
        Returns:
            str: 生图提示词
        """
        prompt = self._build_image_prompt_prompt(
            page_content=page_content,
            character_descriptions=character_descriptions
        )
        
        logger.info(f"正在生成第 {page_content.page_number} 页的生图提示词")
        response = await self.chat_client.generate(prompt)
        
        return response.strip()
    
    async def generate_all_image_prompts(
        self,
        pages: List[PageContent],
        character_descriptions: Dict[str, str]
    ) -> List[PageContent]:
        """
        批量生成所有页面的生图提示词
        
        Args:
            pages: 页面内容列表
            character_descriptions: 角色描述字典
            
        Returns:
            List[PageContent]: 更新后的页面内容列表
        """
        for page in pages:
            try:
                image_prompt = await self.generate_image_prompt(page, character_descriptions)
                page.image_prompt = image_prompt
            except Exception as e:
                logger.error(f"生成第 {page.page_number} 页提示词失败: {e}")
                page.image_prompt = f"生成失败: {str(e)}"
        return pages
    
    async def _get_recent_page_analyses(self, count: int = 5) -> List[Dict]:
        """获取最近几页的分析结果"""
        pages = []
        # 获取所有batch列表
        batch_list = await self.storage.list_batches()
        if not batch_list:
            return pages
        
        # 加载批次数据并提取页面
        all_pages = []
        for batch_info in batch_list:
            batch = await self.storage.load_batch_analysis(
                batch_info["start_page"],
                batch_info["end_page"]
            )
            if batch and "pages" in batch:
                for page in batch["pages"]:
                    all_pages.append(page)
        
        # 排序并取最后几页
        all_pages.sort(key=lambda x: x.get("page_number", 0))
        return all_pages[-count:] if len(all_pages) >= count else all_pages
    
    def _build_chapter_script_prompt(
        self,
        manga_summary: str,
        timeline_data: Dict,
        reference_pages: List[Dict] = None,
        user_direction: str = "",
        page_count: int = 15
    ) -> str:
        """生成全话脚本的提示词"""
        # 使用 CharacterManager 获取树状角色信息（包含形态）
        characters_info = self.char_manager.format_characters_for_prompt()
        
        # 如果 CharacterManager 没有角色，回退到 timeline_data
        if characters_info == "（暂无角色信息）" and timeline_data and "characters" in timeline_data:
            characters_list = []
            for char in timeline_data["characters"]:
                char_text = f"- {char['name']}"
                if char.get('aliases'):
                    char_text += f"（别名：{', '.join(char['aliases'][:3])}）"
                characters_list.append(char_text)
            characters_info = "\n".join(characters_list)
        
        # 格式化时间线事件（从timeline.events提取最近的事件）
        timeline_text = ""
        if timeline_data and "events" in timeline_data:
            recent_events = timeline_data["events"][-10:]  # 取最近10个事件
            events_list = []
            for event in recent_events:
                page_range = event.get('page_range', {})
                page_info = f"第{page_range.get('start', '?')}"
                if page_range.get('end') and page_range['end'] != page_range.get('start'):
                    page_info += f"-{page_range['end']}"
                page_info += "页"
                
                event_text = f"{page_info}：{event.get('event', '')}"
                events_list.append(event_text)
            timeline_text = "\n".join(events_list)
        
        # 格式化参考页面信息（使用page_summary）
        reference_pages_text = ""
        if reference_pages:
            pages_info = "\n\n".join([
                f"第{p['page_number']}页：\n{p.get('page_summary', '')}"
                for p in reference_pages
            ])
            reference_pages_text = f"\n\n## 原作页面分析（前{len(reference_pages)}页，供参考叙事风格）\n\n{pages_info}"
        
        direction_text = f"\n\n用户期望的剧情方向：\n{user_direction}" if user_direction else ""
        
        return f"""你是一位经验丰富的漫画编剧。请基于以下漫画的背景信息，续写下一话的剧情脚本。

# 漫画背景信息

## 故事概要
{manga_summary}

## 主要角色
{characters_info}

## 时间线（最近的剧情发展）
{timeline_text}
{reference_pages_text}
{direction_text}

# 任务要求

请为这部漫画创作续写的一话剧情脚本，共{page_count}页。

## 输出格式

【第XX话 - 标题】

第1页：
场景：[具体场景描述]
人物：[角色名(形态名), 角色名(形态名)]  ← 必须标注形态
剧情：[该页的剧情内容]
对话：（如果有）
- 角色A：「对话内容」
- 角色B：「对话内容」

第2页：
...

## 创作要求

1. **剧情连贯**：自然衔接前面的故事，保持角色性格一致
2. **节奏合理**：每页的内容量适中，重要场景可以多页展开
3. **对话真实**：符合角色性格和漫画风格，自然流畅
4. **场景描述清晰**：为后续的画面生成提供足够的信息
5. **情节完整**：这一话要有起承转合，可以是一个完整的小故事或者推进主线剧情
6. **漫画分页思维**：考虑画面呈现，重要镜头、转折点适合分页
7. **风格继承**：参考原作页面分析，保持相似的叙事节奏和分镜习惯
8. **形态明确**：如果角色有多个形态，每个出场角色需标注使用的形态，格式为「角色名(形态名)」。如果角色只有一个形态或没有特定形态要求，可以只写角色名

请开始创作，直接输出剧本内容。
"""
    
    def _build_page_details_prompt(
        self,
        chapter_script: str,
        page_number: int,
        timeline_data: Dict,
        manga_style: str = ""  # 不再使用此参数
    ) -> str:
        """生成每页剧情细化的提示词"""
        # 从timeline_data中提取角色名称（只需要名字，不需要外貌描述）
        characters_info = ""
        if timeline_data and "characters" in timeline_data:
            characters_list = []
            for char in timeline_data["characters"]:
                char_text = f"- {char['name']}"
                if char.get('aliases'):
                    char_text += f"（别名：{', '.join(char['aliases'][:2])}）"
                characters_list.append(char_text)
            characters_info = "\n".join(characters_list)
        
        return f"""请将以下剧本中的第{page_number}页内容，细化为结构化的剧情数据。

# 角色列表
{characters_info}

# 全话脚本

{chapter_script}

# 任务

请针对**第{page_number}页**，提取以下信息，输出为JSON格式：

{{
  "page_number": {page_number},
  "characters": ["角色名1", "角色名2"],
  "character_forms": [
    {{"character": "角色名1", "form_id": "形态ID"}},
    {{"character": "角色名2", "form_id": "形态ID"}}
  ],
  "description": "详细的画面描述，包括：角色的动作、表情、姿势、位置关系，以及建议的镜头角度（俯视/仰视/平视/特写等）",
  "dialogues": [
    {{"character": "角色名", "text": "对话内容"}}
  ],
  "mood": "情绪氛围（如：紧张、温馨、悲伤、压抑、冲击性）"
}}

## 要求

1. **只提取剧本内容**：不要添加剧本中没有的信息
2. **角色名准确**：使用角色列表中的标准名称（不含形态后缀）
3. **形态提取**：从剧本的「人物」行提取每个角色的形态，格式为 `角色名(形态名)`。将形态名作为 `form_id`。如果剧本中没有标注形态，则省略 `character_forms` 中该角色的条目
4. **画面描述详细**：
   - ✅ 描述角色的动作和表情（如：回头看、皱眉、握紧拳头）
   - ✅ 描述角色的位置关系（如：站在门口、躺在地上、背对背）
   - ✅ 建议镜头角度（如：俯视角度表现压迫感、特写表情）
   - ❌ 不要描述外貌特征（发色、眼睛颜色等）
   - ❌ 不要描述服装样式
5. **对话原样保留**：保持剧本中的对话内容

请直接输出JSON数据。
"""
    
    def _build_image_prompt_prompt(
        self,
        page_content: PageContent,
        character_descriptions: Dict[str, str]
    ) -> str:
        """生成生图提示词的LLM提示
        
        根据角色是否有参考图，动态调整描述规则：
        - 有参考图的角色：不描述外貌/服装（由参考图决定）
        - 无参考图的角色：需要生成外貌/服装描述
        """
        # 分类角色：有/无参考图
        chars_with_desc = []
        chars_without_desc = []
        
        for char_name in page_content.characters:
            if char_name in character_descriptions and character_descriptions[char_name]:
                chars_with_desc.append(char_name)
            else:
                chars_without_desc.append(char_name)
        
        chars_list = ", ".join(page_content.characters) if page_content.characters else "无"
        
        dialogues_text = ""
        if page_content.dialogues:
            dialogues_text = "\n".join([
                f"- {d['character']}：「{d['text']}」"
                for d in page_content.dialogues
            ])
        
        # 根据角色参考图情况动态生成规则
        if chars_with_desc and chars_without_desc:
            # 部分角色有参考图
            appearance_rule = f"""**画面内容**：
- ✅ **所有角色**：描述角色的动作、表情、姿势、位置关系
- ✅ **所有角色**：描述镜头聚焦点（如：聚焦于角色的眼睛、手部特写）
- ✅ **所有角色**：描述背景中可见的物体或场景元素
- ❌ **有参考图的角色**（{', '.join(chars_with_desc)}）：不要描述外貌特征、发色、服装（由参考图决定）
- ✅ **无参考图的角色**（{', '.join(chars_without_desc)}）：需要简要描述外貌特征（发色、体型、穿着风格等）
- ❌ 不要描述绘画风格、光影效果、色调（由画风参考图决定）"""
        elif chars_with_desc:
            # 所有角色都有参考图
            appearance_rule = """**画面内容**：
- ✅ 描述：角色的动作、表情、姿势、位置关系
- ✅ 描述：镜头聚焦点（如：聚焦于角色的眼睛、手部特写）
- ✅ 描述：背景中可见的物体或场景元素
- ❌ 不要描述：角色的发色、眼睛颜色、服装样式（这些由角色参考图决定）
- ❌ 不要描述：绘画风格、光影效果、色调（这些由画风参考图决定）"""
        else:
            # 没有任何角色参考图
            appearance_rule = """**画面内容**：
- ✅ 描述：角色的动作、表情、姿势、位置关系
- ✅ 描述：镜头聚焦点（如：聚焦于角色的眼睛、手部特写）
- ✅ 描述：背景中可见的物体或场景元素
- ✅ 描述：角色的外貌特征（发色、发型、体型、大致穿着风格），但保持简洁
- ❌ 不要描述：绘画风格、光影效果、色调（这些由画风参考图决定）"""

        return f"""请基于以下剧情信息，生成一个详细的漫画绘图提示词。

# 剧情信息

- 出场角色：{chars_list}
- 画面描述：{page_content.description}
- 情绪氛围：{page_content.mood}
- 对话：
{dialogues_text if dialogues_text else "（无对话）"}

# 输出格式

一页漫画。

**画面描述**：
- **分格1**（[大小]，[镜头角度]）**：[具体画面内容，角色的动作、表情、位置关系]
- **分格2**（[大小]，[镜头角度]）**：[具体画面内容]
- **分格3**（[大小]，[镜头角度]）**：[具体画面内容]
- [根据剧情需要继续添加分格]
- **页面底部**：[如有旁白或特殊文字，在这里说明]

**对话气泡**（简体中文）：
- 分格X中，[位置描述]，内容：「对话内容」
- 分格X中，[位置描述]，内容：「对话内容」

# 分格描述规范

**分格大小**：大格（占半页以上）、中格、小格、长条格、特写小图
**镜头角度**：俯视角度、仰视角度、平视角度、斜角、远景、中景、特写
{appearance_rule}

# 重要规则

1. **分格要详细**：每个分格都要说明大小、镜头角度、具体画面内容
2. **动作要具体**：描述角色在做什么，表情如何，位置在哪
3. **对话用简体中文**：所有气泡内容必须是简体中文
4. **不输出\"场景\"字段**：场景信息融入分格描述即可

请直接输出提示词。
"""

    
    def _extract_chapter_title(self, script_text: str) -> str:
        """从脚本文本中提取章节标题"""
        # 匹配 【第XX话 - 标题】 格式
        match = re.search(r'【第\d+话\s*[-–—]\s*(.+?)】', script_text)
        if match:
            return match.group(1).strip()
        
        # 匹配其他格式
        match = re.search(r'第\d+话[:：]\s*(.+)', script_text)
        if match:
            return match.group(1).strip()
        
        return "续写章节"
    
    def _parse_json_response(self, response: str) -> Dict:
        """解析 JSON 响应"""
        # 清理响应文本
        text = response.strip()
        
        # 移除 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # 尝试解析 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 对象
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        
        logger.warning(f"无法解析 JSON 响应: {text[:200]}...")
        return {}
