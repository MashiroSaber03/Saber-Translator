import asyncio
import types
import unittest
from unittest import mock


class MangaInsightStructuredJsonAdoptionTests(unittest.TestCase):
    def test_qa_decompose_question_uses_generate_json(self) -> None:
        from src.core.manga_insight.qa import MangaQA

        qa = MangaQA.__new__(MangaQA)
        qa.config = types.SimpleNamespace(
            prompts=types.SimpleNamespace(
                question_decompose="请拆解问题：{question}",
            )
        )
        qa.chat_client = types.SimpleNamespace(
            generate_json=mock.AsyncMock(
                return_value={
                    "sub_questions": [
                        "先确认主角当前的直接目标是什么",
                        "再确认阻碍目标的关键冲突是什么",
                    ]
                }
            )
        )

        result = asyncio.run(qa._decompose_question("主角现在为什么这么做？"))

        self.assertEqual(
            result,
            [
                "先确认主角当前的直接目标是什么",
                "再确认阻碍目标的关键冲突是什么",
            ],
        )

    def test_qa_decompose_question_returns_empty_list_when_structured_call_fails(self) -> None:
        from src.core.manga_insight.qa import MangaQA

        qa = MangaQA.__new__(MangaQA)
        qa.config = types.SimpleNamespace(
            prompts=types.SimpleNamespace(
                question_decompose="请拆解问题：{question}",
            )
        )
        qa.chat_client = types.SimpleNamespace(
            generate_json=mock.AsyncMock(side_effect=ValueError("JSON 解析失败"))
        )

        result = asyncio.run(qa._decompose_question("主角现在为什么这么做？"))

        self.assertEqual(result, [])

    def test_timeline_synthesis_uses_generate_json(self) -> None:
        from src.core.manga_insight.features.timeline_enhanced import EnhancedTimelineBuilder

        fake_llm = types.SimpleNamespace(
            generate_json=mock.AsyncMock(
                return_value={
                    "events": [{"id": "evt-1", "summary": "主角做出关键决定"}],
                    "characters": [{"name": "主角"}],
                }
            ),
            close=mock.AsyncMock(return_value=None),
        )

        builder = EnhancedTimelineBuilder.__new__(EnhancedTimelineBuilder)
        builder.llm = fake_llm

        result = asyncio.run(builder._synthesize_timeline("分析数据"))

        self.assertIsNotNone(result)
        self.assertEqual(result["events"][0]["id"], "evt-1")
        self.assertIsNone(builder.llm)

    def test_timeline_synthesis_returns_none_when_structured_call_fails(self) -> None:
        from src.core.manga_insight.features.timeline_enhanced import EnhancedTimelineBuilder

        fake_llm = types.SimpleNamespace(
            generate_json=mock.AsyncMock(side_effect=ValueError("JSON 解析失败")),
            close=mock.AsyncMock(return_value=None),
        )

        builder = EnhancedTimelineBuilder.__new__(EnhancedTimelineBuilder)
        builder.llm = fake_llm

        result = asyncio.run(builder._synthesize_timeline("分析数据"))

        self.assertIsNone(result)
        self.assertIsNone(builder.llm)

    def test_summary_generator_uses_raw_text_fallback_when_structured_retries_exhaust(self) -> None:
        from src.core.manga_insight.summary_generator import SummaryGenerator
        from src.shared.openai_execution import OpenAICompatibleBusinessRetriesExhaustedError

        fake_client = types.SimpleNamespace(
            generate_json=mock.AsyncMock(
                side_effect=OpenAICompatibleBusinessRetriesExhaustedError(
                    "漫画分析对话 业务重试耗尽",
                    last_raw_content="原始章节总结文本",
                )
            ),
            close=mock.AsyncMock(return_value=None),
        )

        config = types.SimpleNamespace(
            prompts=types.SimpleNamespace(
                chapter_summary="",
                analysis_system="",
            )
        )
        storage = types.SimpleNamespace()

        generator = SummaryGenerator("book-demo", config, storage)

        with mock.patch("src.core.manga_insight.summary_generator.create_chat_client", return_value=fake_client):
            result = asyncio.run(
                generator.generate_chapter_from_segments(
                    "chapter-1",
                    {"title": "第一章"},
                    [
                        {
                            "page_range": {"start": 1, "end": 2},
                            "summary": "片段总结",
                            "key_events": ["事件1"],
                        }
                    ],
                )
            )

        self.assertEqual(result["summary"], "原始章节总结文本")

    def test_segment_summary_uses_raw_text_fallback_when_structured_retries_exhaust(self) -> None:
        from src.core.manga_insight.summary_generator import SummaryGenerator
        from src.shared.openai_execution import OpenAICompatibleBusinessRetriesExhaustedError

        fake_client = types.SimpleNamespace(
            generate_json=mock.AsyncMock(
                side_effect=OpenAICompatibleBusinessRetriesExhaustedError(
                    "漫画分析对话 业务重试耗尽",
                    last_raw_content="原始段落总结文本",
                )
            ),
            close=mock.AsyncMock(return_value=None),
        )

        config = types.SimpleNamespace(
            prompts=types.SimpleNamespace(
                segment_summary="",
                analysis_system="",
            )
        )
        saved_payloads = []

        class _StorageStub:
            async def load_segment_summary(self, _segment_id):
                return None

            async def save_segment_summary(self, _segment_id, payload):
                saved_payloads.append(payload)

        generator = SummaryGenerator("book-demo", config, _StorageStub())

        with mock.patch("src.core.manga_insight.summary_generator.create_chat_client", return_value=fake_client):
            result = asyncio.run(
                generator.generate_segment_summary(
                    "segment-1",
                    [
                        {
                            "page_range": {"start": 1, "end": 2},
                            "batch_summary": "片段批次摘要",
                            "key_events": ["事件1"],
                        }
                    ],
                )
            )

        self.assertEqual(result["summary"], "原始段落总结文本")
        self.assertEqual(saved_payloads[0]["summary"], "原始段落总结文本")
