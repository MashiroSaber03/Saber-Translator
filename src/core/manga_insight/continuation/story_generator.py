"""
Manga Insight 剧情生成器

负责三层递进的剧情生成：
1. 全话脚本生成
2. 每页剧情细化
3. 生图提示词生成
"""

import logging
import os
import re
from typing import Dict, List, Any

from ..config_utils import load_insight_config, create_chat_client
from ..storage import AnalysisStorage
from ..utils.json_parser import parse_llm_json
from .models import ChapterScript, PageContent, ContinuationCharacters
from .character_manager import CharacterManager
from .reference_tokens import (
    build_original_reference_candidates,
    list_original_manga_page_paths,
    resolve_reference_tokens,
)

logger = logging.getLogger("MangaInsight.Continuation.StoryGenerator")
HierarchicalSummaryGenerator = None


class StoryGenerator:
    """剧情生成器"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.config = load_insight_config()
        self.storage = AnalysisStorage(book_id)
        # 续写脚本生成改用 VLM，可以看到原作图片
        from ..vlm_client import VLMClient
        self.vlm_client = VLMClient(self.config.vlm, self.config.prompts)
        # 保留 ChatClient 用于其他文本生成任务（使用工厂函数处理 use_same_as_vlm 配置）
        self.chat_client = create_chat_client(self.config)
        self.char_manager = CharacterManager(book_id)
    
    async def prepare_continuation_data(self) -> Dict[str, Any]:
        """
        准备续写所需的数据，必要时自动补齐故事概要。
        
        Returns:
            Dict: {"ready": bool, "message": str}
        """
        result = {
            "ready": False,
            "message": ""
        }
        
        try:
            story_summary = await self._ensure_story_summary()
        except Exception as exc:
            logger.warning(f"自动生成故事概要失败: {exc}")
            result["message"] = f"故事概要自动生成失败：{exc}"
            return result

        if not story_summary or not story_summary.get("content"):
            result["message"] = "故事概要自动生成失败，请检查分析配置后重试"
            return result

        # 2. 检查时间线是否存在
        timeline_data = await self.storage.load_timeline()
        
        if not timeline_data or not timeline_data.get("events"):
            result["message"] = "时间线数据不存在或为空，请先完成漫画分析"
            return result
        
        result["ready"] = True
        result["message"] = "数据准备完成"
        return result

    async def _ensure_story_summary(self) -> Dict[str, Any]:
        """确保故事概要存在，缺失时自动生成。"""
        story_summary = await self.storage.load_template_overview("story_summary")
        if story_summary and story_summary.get("content"):
            return story_summary

        logger.info("故事概要缺失，开始自动生成")
        llm_client = None
        try:
            summary_generator_cls = HierarchicalSummaryGenerator
            if summary_generator_cls is None:
                from ..features.hierarchical_summary import HierarchicalSummaryGenerator as summary_generator_cls

            llm_client = create_chat_client(self.config)
            generator = summary_generator_cls(
                book_id=self.book_id,
                storage=self.storage,
                llm_client=llm_client,
                prompts_config=self.config.prompts,
            )
            result = await generator.generate_with_template("story_summary", skip_cache=True)
            if not result or not result.get("content"):
                raise ValueError("未生成有效内容")
            await self.storage.save_template_overview("story_summary", result)
            return result
        finally:
            if llm_client:
                await llm_client.close()
    
    async def generate_chapter_script(
        self,
        user_direction: str = "",
        page_count: int = 15,
        custom_reference_tokens: List[str] = None,
        reference_image_count: int = 5,
    ) -> ChapterScript:
        """
        生成全话脚本（第一层）- 使用 VLM + 原作图片

        Args:
            user_direction: 用户指定的续写方向
            page_count: 续写页数
            custom_reference_tokens: 自定义参考图 token 列表（可选）
                如果提供，使用这些引用替代自动选择的最后N张
                如果为 None 或空列表，使用默认的自动选择逻辑
            reference_image_count: 自动选择参考图时使用的原作图片数量

        Returns:
            ChapterScript: 生成的脚本
        """
        reference_image_count = max(1, int(reference_image_count or 5))

        # 获取必要数据
        story_summary = await self._ensure_story_summary()
        timeline_data = await self.storage.load_timeline()

        if not story_summary or not story_summary.get("content"):
            raise ValueError("故事概要不存在，请先调用 prepare_continuation_data")

        if not timeline_data:
            raise ValueError("时间线数据不存在")

        # 获取最近几页的分析结果作为参考（文本信息）
        reference_pages = await self._get_recent_page_analyses(5)

        # 获取原漫画图片作为视觉参考
        if custom_reference_tokens and len(custom_reference_tokens) > 0:
            all_original_paths = await self._get_all_manga_image_paths()
            original_candidates = build_original_reference_candidates(all_original_paths)
            resolved_refs = resolve_reference_tokens(custom_reference_tokens, original_candidates)
            resolved_paths = [ref["path"] for ref in resolved_refs if ref.get("path")]
            original_images = await self._load_images_from_paths(resolved_paths)
            if not original_images:
                logger.warning("自定义参考图当前均不可用，回退到自动选择")
                original_images = await self._get_recent_manga_images(reference_image_count)
            else:
                logger.info(f"使用自定义参考图: {len(original_images)} 张")
        else:
            # 使用默认的自动选择逻辑（最后 N 张）
            original_images = await self._get_recent_manga_images(reference_image_count)
            logger.info(f"使用自动选择的参考图: {len(original_images)} 张")

        # 构建提示词（针对 VLM 优化）
        prompt = self._build_chapter_script_prompt(
            manga_summary=story_summary.get("content", ""),
            timeline_data=timeline_data,
            reference_pages=reference_pages,
            user_direction=user_direction,
            page_count=page_count,
            has_visual_reference=len(original_images) > 0  # 传递是否有视觉参考的标志
        )

        # 调用 VLM（附带图片）
        logger.info(f"正在生成全话脚本，页数: {page_count}，视觉参考: {len(original_images)} 张原作图片")

        if original_images:
            # 使用 VLM 的多模态能力
            response = await self._call_vlm_with_images(original_images, prompt)
        else:
            # 降级：无图片时使用纯文本 LLM
            logger.warning("未找到原作图片，降级为纯文本生成")
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
        character_forms = self._normalize_character_forms(page_data.get("character_forms", []))
        
        return PageContent(
            page_number=page_data.get("page_number", page_number),
            characters=page_data.get("characters", []),
            character_forms=character_forms,
            description=page_data.get("description", ""),
            dialogues=page_data.get("dialogues", [])
        )
    
    async def generate_image_prompt(
        self,
        page_content: PageContent
    ) -> str:
        """
        生成生图提示词（第三层）
        
        Args:
            page_content: 页面内容
            
        Returns:
            str: 生图提示词
        """
        prompt = self._build_image_prompt_prompt(page_content)
        
        logger.info(f"正在生成第 {page_content.page_number} 页的生图提示词")
        response = await self.chat_client.generate(prompt)
        
        return response.strip()
    
    async def generate_all_image_prompts(
        self,
        pages: List[PageContent]
    ) -> List[PageContent]:
        """
        批量生成所有页面的生图提示词
        
        Args:
            pages: 页面内容列表
            
        Returns:
            List[PageContent]: 更新后的页面内容列表
        """
        for page in pages:
            try:
                image_prompt = await self.generate_image_prompt(page)
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
    
    async def _get_recent_manga_images(self, count: int = 5) -> List[bytes]:
        """
        获取原漫画最后 N 张图片（用于 VLM 视觉参考）

        Args:
            count: 需要获取的图片数量

        Returns:
            List[bytes]: 图片字节数据列表（最后几页）
        """
        images = []

        try:
            all_image_paths = await self._get_all_manga_image_paths()
            recent_paths = all_image_paths[-count:] if len(all_image_paths) > count else all_image_paths

            # 读取图片字节数据
            for img_path in recent_paths:
                try:
                    with open(img_path, "rb") as f:
                        images.append(f.read())
                except Exception as e:
                    logger.warning(f"读取图片失败: {img_path}, {e}")

            logger.info(f"成功加载 {len(images)} 张原作图片用于脚本生成")
            return images

        except Exception as e:
            logger.error(f"获取原漫画图片失败: {e}")
            return images

    async def _get_all_manga_image_paths(self) -> List[str]:
        """获取原漫画全部图片路径。"""
        image_paths = list_original_manga_page_paths(self.book_id)
        if not image_paths:
            logger.warning(f"未找到可用原作图片: {self.book_id}")
        return image_paths

    async def _load_images_from_paths(self, image_paths: List[str]) -> List[bytes]:
        """
        从指定路径列表加载图片字节数据

        Args:
            image_paths: 图片路径列表

        Returns:
            List[bytes]: 图片字节数据列表
        """
        images = []

        for img_path in image_paths:
            if not img_path:
                continue

            # 规范化路径
            img_path = os.path.normpath(img_path)

            if not os.path.exists(img_path):
                logger.warning(f"参考图路径不存在，跳过: {img_path}")
                continue

            try:
                with open(img_path, "rb") as f:
                    images.append(f.read())
                logger.debug(f"已加载参考图: {img_path}")
            except Exception as e:
                logger.warning(f"读取参考图失败: {img_path}, {e}")

        logger.info(f"从自定义路径加载了 {len(images)} 张参考图")
        return images
    
    def _build_chapter_script_prompt(
        self,
        manga_summary: str,
        timeline_data: Dict,
        reference_pages: List[Dict] = None,
        user_direction: str = "",
        page_count: int = 15,
        has_visual_reference: bool = False
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
        
        # 【新增】针对 VLM 的视觉参考说明
        visual_reference_text = ""
        if has_visual_reference:
            visual_reference_text = """

## 视觉参考（原作最后几页图片）

我已为你提供了原作最后几页的实际画面。请仔细观察这些图片，但注意以下事项：

### 图片筛选
**注意**：提供的图片中可能包含与剧情无关的内容，请主动识别并忽略：
- ❌ **宣传图/广告页**：推广其他作品、杂志、周边商品的图片
- ❌ **预告页**：下一话预告、角色设定图、特典插画等
- ❌ **版权页/致谢页**：包含大量文字说明、作者后记、出版信息等
- ❌ **封面/扉页**：单独的封面图或章节标题页
- ✅ **只关注正式的剧情页**：包含分格、对话、叙事推进的页面

### 观察要点（仅针对有效剧情页）
- **叙事节奏**：注意原作每页的信息量、分格数量、剧情推进速度
- **分镜风格**：大格、小格、全景、特写的使用比例和频率
- **对话密度**：每页的对话气泡数量和对话长度
- **情绪氛围**：最后几页的氛围是紧张、轻松、悬疑还是温馨
- **场景连贯性**：故事最后停在什么场景，角色最后的状态和位置

**重要**：
1. 如果最后几张图都不是剧情页，请忽略它们，只根据文字信息（时间线、故事概要）进行续写
2. 续写的剧本需在视觉叙事风格上自然衔接有效的剧情页，保持原作的独特节奏感
"""
        
        return f"""你是一位经验丰富的漫画编剧。请基于以下漫画的背景信息，续写下一话的剧情脚本。

# 漫画背景信息

## 故事概要
{manga_summary}

## 主要角色
{characters_info}

## 时间线（最近的剧情发展）
{timeline_text}
{reference_pages_text}
{visual_reference_text}
{direction_text}

# 任务要求

请为这部漫画创作续写的一话剧情脚本，共{page_count}页。

## 输出格式

【第XX话 - 标题】

第1页：
场景：[具体场景描述]
人物：[角色名（若"主要角色"中有对应的形态名，则标注形态名）]
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
7. **风格继承**：参考原作页面分析{' 和提供的原作图片' if has_visual_reference else ''}，保持相似的叙事节奏和分镜习惯
8. **形态明确**：如果角色有多个形态，每个出场角色需标注使用的形态，格式为「角色名(形态名)」。如果角色没有特定形态要求则只需只写角色名，不要编造形态名。

请开始创作，直接输出剧本内容。
"""
    
    def _build_page_details_prompt(
        self,
        chapter_script: str,
        page_number: int,
        timeline_data: Dict
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
        
        form_registry = self._build_character_form_registry()

        return f"""请将以下剧本中的第{page_number}页内容，细化为结构化的剧情数据。

# 角色列表
{characters_info}

# 角色形态映射
{form_registry}

# 全话脚本

{chapter_script}

# 任务

请针对**第{page_number}页**，提取以下信息，输出为JSON格式：

{{
  "page_number": {page_number},
  "characters": ["角色名1", "角色名2"],
  "character_forms": [
    {{"character": "角色名1", "form_id": "稳定形态ID", "form_name": "形态显示名"}},
    {{"character": "角色名2", "form_id": "稳定形态ID", "form_name": "形态显示名"}}
  ],
  "description": "详细的画面描述，包括：角色的动作、表情、姿势、位置关系，以及建议的镜头角度（俯视/仰视/平视/特写等）",
  "dialogues": [
    {{"character": "角色名", "text": "对话内容"}}
  ]
}}

## 要求

1. **只提取剧本内容**：不要添加剧本中没有的信息
2. **形态提取**：如果剧本的「人物」行中出现 `角色名(形态名)`，请根据上方的“角色形态映射”返回对应的稳定 `form_id`，并同时填写 `form_name`。如果剧本没有标注形态，则省略该角色的 `character_forms` 条目，严禁编造新形态。
3. **对话原样保留**：保持剧本中的对话内容

请直接输出JSON数据。
"""
    
    def _build_image_prompt_prompt(
        self,
        page_content: PageContent
    ) -> str:
        """生成生图提示词的LLM提示"""
        chars_list = ", ".join(page_content.characters) if page_content.characters else "无"

        forms_text = ""
        if page_content.character_forms:
            normalized_forms = []
            for form in page_content.character_forms:
                character_name = str(form.get("character") or "").strip()
                form_name = str(form.get("form_name") or form.get("form_id") or "").strip()
                if character_name and form_name:
                    normalized_forms.append(f"- {character_name}：{form_name}")
            if normalized_forms:
                forms_text = "\n- 角色形态：\n" + "\n".join(normalized_forms)

        dialogues_text = ""
        if page_content.dialogues:
            dialogues_text = "\n".join([
                f"- {d['character']}：「{d['text']}」"
                for d in page_content.dialogues
            ])

        return f"""请基于以下剧情信息，生成一个“简洁优先”的漫画生图提示词。

目标：告诉绘图模型“这一页必须发生什么”，不要把提示词写成详细分镜脚本。

# 剧情信息

- 出场角色：{chars_list}
{forms_text}
- 核心画面：{page_content.description}
- 必须保留对白：
{dialogues_text if dialogues_text else "（无对话）"}

# 输出要求

请直接输出一个 5 行以内、以内容约束为主的简洁提示词，使用以下结构：

出场角色：...
核心动作/情绪：...
场景：...
关键对白：...
风格约束：...

# 重要规则

1. **简洁优先**：只保留角色、核心动作/情绪、场景一句话、关键对白、风格约束
2. **不要按“分格1/分格2”展开**，也不要写成详细分镜脚本；不要写镜头角度、景深、灯光、材质等过细视觉说明，除非剧情不可缺少
3. **不要扩写新剧情**：只提炼这一页必须发生的内容，不要补充输入里没有的新事件
4. **风格约束必须明确写出**：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。
5. **负向约束必须写出**：禁止写实插画、电影感光影、3D感、厚涂感、霓虹夜景氛围、景深虚化、镜头语言、电影级光影、写实城市背景、概念插画风。
6. **对话用简体中文**：所有对白与文字保持简体中文

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
        result = parse_llm_json(response)
        if not isinstance(result, dict) or not result:
            logger.warning(f"无法解析 JSON 响应: {response[:200]}...")
            raise ValueError("LLM 未返回有效的页面详情 JSON")
        return result

    def _build_character_form_registry(self) -> str:
        """为页面细化提示词构建角色形态 ID 映射。"""
        characters = self._load_continuation_characters()
        if not characters or not characters.characters:
            return "（暂无角色形态信息）"

        lines: List[str] = []
        for character in characters.characters:
            if character.enabled is False:
                continue
            if not character.forms:
                lines.append(f"- {character.name}: （未配置形态）")
                continue

            for form in character.forms:
                if form.enabled is False:
                    continue
                lines.append(
                    f"- {character.name}: form_id={form.form_id}, form_name={form.form_name}"
                )

        return "\n".join(lines) if lines else "（暂无角色形态信息）"

    def _load_continuation_characters(self) -> ContinuationCharacters | None:
        try:
            return self.char_manager.load_characters()
        except Exception as exc:
            logger.warning(f"加载续写角色失败: {exc}")
            return None

    def _normalize_character_forms(self, raw_forms: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_forms, list):
            return []

        characters = self._load_continuation_characters()
        normalized: List[Dict[str, str]] = []
        for item in raw_forms:
            if not isinstance(item, dict):
                continue

            character_name = str(item.get("character") or "").strip()
            raw_form_id = str(item.get("form_id") or "").strip()
            raw_form_name = str(item.get("form_name") or "").strip()
            if not character_name or not (raw_form_id or raw_form_name):
                continue

            resolved_form_id = raw_form_id
            resolved_form_name = raw_form_name

            if characters:
                character = characters.get_character(character_name)
                if character:
                    form = character.resolve_form(raw_form_id, raw_form_name)
                    if form:
                        resolved_form_id = form.form_id
                        resolved_form_name = form.form_name

            entry = {"character": character_name}
            if resolved_form_id:
                entry["form_id"] = resolved_form_id
            if resolved_form_name:
                entry["form_name"] = resolved_form_name
            normalized.append(entry)

        return normalized

    async def _call_vlm_with_images(self, images: List[bytes], prompt: str) -> str:
        """
        调用 VLM 生成脚本（附带图片）
        
        Args:
            images: 图片数据列表
            prompt: 文本提示词
            
        Returns:
            str: VLM 生成的脚本文本
        """
        logger.info(f"调用 VLM 生成脚本，图片数量: {len(images)}")
        
        # 调用 VLM 的核心方法（包含重试和错误处理）
        response = await self.vlm_client._call_vlm(images, prompt)
        
        return response


