import types
import unittest
from unittest import mock

import httpx
from openai import APIConnectionError, APIStatusError

from src.core.web_import.firecrawl_tools import (
    FIRECRAWL_TOOLS,
    execute_firecrawl_tool_sync,
)


class WebImportAgentRuntimeTests(unittest.TestCase):
    def _build_agent(self, **agent_overrides):
        from src.core.web_import.agent import MangaScraperAgent

        return MangaScraperAgent(
            firecrawl_api_key="firecrawl-key",
            provider="custom",
            api_key="test-key",
            base_url="https://example.com/v1",
            model_name="gpt-test",
            use_stream=agent_overrides.get("useStream", False),
            force_json=True,
            max_retries=agent_overrides.get("maxRetries", 3),
            timeout=120,
            prompt="Extract manga pages as JSON.",
            max_iterations=10,
            bypass_proxy=False,
        )

    def test_call_llm_retries_transient_errors_up_to_configured_retries(self) -> None:
        agent = self._build_agent(maxRetries=2, useStream=False)
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        failing_error = APIConnectionError(request=request)
        final_message = types.SimpleNamespace(content="ok", tool_calls=None)

        create_mock = mock.Mock(
            side_effect=[
                failing_error,
                failing_error,
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=final_message)]),
            ]
        )
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )

        with mock.patch.object(agent, "_wait_before_retry"):
            result = agent._call_llm([{"role": "user", "content": "hello"}])

        self.assertIs(result, final_message)
        self.assertEqual(create_mock.call_count, 3)

    def test_agent_constructor_rejects_invalid_current_settings(self) -> None:
        invalid = (
            {"maxRetries": -1},
            {"maxRetries": True},
            {"useStream": "false"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(
                (TypeError, ValueError)
            ):
                self._build_agent(**overrides)

    def test_agent_constructor_requires_custom_base_url(self) -> None:
        from src.core.web_import.agent import MangaScraperAgent

        with self.assertRaisesRegex(ValueError, "Base URL"):
            MangaScraperAgent(
                firecrawl_api_key="firecrawl-key",
                provider="custom",
                api_key="test-key",
                base_url="",
                model_name="gpt-test",
                use_stream=False,
                force_json=True,
                max_retries=0,
                timeout=120,
                prompt="Extract manga pages as JSON.",
                max_iterations=10,
                bypass_proxy=False,
            )

    def test_parse_result_requires_sequential_http_pages(self) -> None:
        agent = self._build_agent()

        wrong_order = agent._parse_result(
            '{"comic_title":"Comic","chapter_title":"Chapter",'
            '"pages":[{"page_number":2,"image_url":"https://example.com/2.jpg"}],'
            '"total_pages":1}',
            "https://example.com/chapter",
        )
        wrong_scheme = agent._parse_result(
            '{"comic_title":"Comic","chapter_title":"Chapter",'
            '"pages":[{"page_number":1,"image_url":"file:///tmp/1.jpg"}],'
            '"total_pages":1}',
            "https://example.com/chapter",
        )

        self.assertFalse(wrong_order.success)
        self.assertFalse(wrong_scheme.success)

    def test_parse_result_requires_raw_non_empty_json_result(self) -> None:
        agent = self._build_agent()
        fenced = agent._parse_result(
            '```json\n{"comic_title":"Comic","chapter_title":"Chapter",'
            '"pages":[{"page_number":1,"image_url":"https://example.com/1.jpg"}],'
            '"total_pages":1}\n```',
            "https://example.com/chapter",
        )
        empty = agent._parse_result(
            '{"comic_title":"Comic","chapter_title":"Chapter",'
            '"pages":[],"total_pages":0}',
            "https://example.com/chapter",
        )

        self.assertFalse(fenced.success)
        self.assertFalse(empty.success)

    def test_call_llm_does_not_retry_non_retryable_status(self) -> None:
        agent = self._build_agent(maxRetries=3, useStream=False)
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        create_mock = mock.Mock(
            side_effect=APIStatusError(
                "unauthorized",
                response=response,
                body=None,
            )
        )
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )

        with self.assertRaises(APIStatusError):
            agent._call_llm([{"role": "user", "content": "hello"}])

        self.assertEqual(create_mock.call_count, 1)

    def test_call_llm_does_not_retry_memory_failure(self) -> None:
        agent = self._build_agent(maxRetries=3, useStream=False)
        create_mock = mock.Mock(side_effect=MemoryError("native allocation failed"))
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )

        with self.assertRaisesRegex(MemoryError, "allocation failed"):
            agent._call_llm([{"role": "user", "content": "hello"}])
        self.assertEqual(create_mock.call_count, 1)

    def test_call_llm_stream_mode_can_reconstruct_tool_calls(self) -> None:
        agent = self._build_agent(useStream=True)

        chunks = [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(
                            content="done",
                            tool_calls=[
                                types.SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=types.SimpleNamespace(name="search", arguments='{"query":"hello"}'),
                                )
                            ],
                        )
                    )
                ]
            )
        ]
        create_mock = mock.Mock(return_value=iter(chunks))
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )

        with self.assertLogs("saber.user", level="DEBUG") as captured:
            message = agent._call_llm(
                [{"role": "user", "content": "hello"}]
            )

        self.assertIsNotNone(message)
        self.assertEqual(message.content, "done")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].id, "call_1")
        self.assertEqual(message.tool_calls[0].function.name, "search")
        self.assertEqual(message.tool_calls[0].function.arguments, '{"query":"hello"}')
        rendered = "\n".join(captured.output)
        self.assertIn("网页导入助手请求内容", rendered)
        self.assertIn("网页导入助手开始流式返回", rendered)
        self.assertIn("done", rendered)
        self.assertIn("返回 1 个工具调用", rendered)

    def test_extract_stops_before_provider_call_when_control_is_requested(self) -> None:
        from src.core.web_import.agent import WebImportAgentControlRequested

        agent = self._build_agent(useStream=False)
        create_mock = mock.Mock()
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )

        with self.assertRaises(WebImportAgentControlRequested):
            agent.extract(
                "https://example.com/chapter",
                should_stop=lambda: True,
            )

        create_mock.assert_not_called()

    def test_extract_requires_an_absolute_http_source_url(self) -> None:
        agent = self._build_agent(useStream=False)
        create_mock = mock.Mock()
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )

        for source_url in ("", "chapter/1", "file:///tmp/chapter.html"):
            with self.subTest(source_url=source_url), self.assertRaises(ValueError):
                agent.extract(source_url)

        create_mock.assert_not_called()

    def test_extract_preserves_one_assistant_turn_for_multiple_tool_calls(self) -> None:
        agent = self._build_agent(useStream=False)
        calls = [
            types.SimpleNamespace(
                id="call_1",
                function=types.SimpleNamespace(
                    name="firecrawl_scrape",
                    arguments='{"url":"https://example.com/1"}',
                ),
            ),
            types.SimpleNamespace(
                id="call_2",
                function=types.SimpleNamespace(
                    name="firecrawl_scrape",
                    arguments='{"url":"https://example.com/2"}',
                ),
            ),
        ]
        final = types.SimpleNamespace(
            content=(
                '{"comic_title":"Comic","chapter_title":"Chapter",'
                '"pages":[{"page_number":1,'
                '"image_url":"https://example.com/1.jpg"}],'
                '"total_pages":1}'
            ),
            tool_calls=None,
        )
        messages_seen: list[list[dict[str, object]]] = []

        def call_llm(messages, **_kwargs):
            messages_seen.append([dict(message) for message in messages])
            if len(messages_seen) == 1:
                return types.SimpleNamespace(content=None, tool_calls=calls)
            return final

        with mock.patch.object(agent, "_call_llm", side_effect=call_llm), mock.patch(
            "src.core.web_import.agent.execute_firecrawl_tool_sync",
            return_value={"success": True, "data": {"html": "x" * 60_000}},
        ):
            result = agent.extract("https://example.com/chapter")

        self.assertTrue(result.success)
        follow_up = messages_seen[1]
        self.assertEqual([message["role"] for message in follow_up[-3:]], [
            "assistant",
            "tool",
            "tool",
        ])
        self.assertEqual(len(follow_up[-3]["tool_calls"]), 2)
        self.assertGreater(len(follow_up[-2]["content"]), 50_000)

    def test_retry_wait_observes_control_request(self) -> None:
        from src.core.web_import.agent import WebImportAgentControlRequested

        agent = self._build_agent(maxRetries=3, useStream=False)
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        create_mock = mock.Mock(side_effect=APIConnectionError(request=request))
        agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        )
        checks = iter((False, True))

        with self.assertRaises(WebImportAgentControlRequested):
            agent._call_llm(
                [{"role": "user", "content": "hello"}],
                should_stop=lambda: next(checks, True),
            )

        self.assertEqual(create_mock.call_count, 1)

    def test_close_releases_the_openai_client(self) -> None:
        agent = self._build_agent()
        close_mock = mock.Mock()
        agent.client = types.SimpleNamespace(close=close_mock)

        agent.close()

        close_mock.assert_called_once_with()

    def test_firecrawl_exposes_only_the_current_scrape_tool(self) -> None:
        self.assertEqual(
            [tool["function"]["name"] for tool in FIRECRAWL_TOOLS],
            ["firecrawl_scrape"],
        )
        self.assertFalse(
            FIRECRAWL_TOOLS[0]["function"]["parameters"][
                "additionalProperties"
            ]
        )

    def test_firecrawl_returns_only_the_canonical_success_payload(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "success": True,
            "data": {"html": "<img src='page.jpg'>"},
        }
        client = mock.MagicMock()
        client.__enter__.return_value.post.return_value = response

        with mock.patch(
            "src.core.web_import.firecrawl_tools.httpx.Client",
            return_value=client,
        ) as client_ctor:
            result = execute_firecrawl_tool_sync(
                "firecrawl_scrape",
                {
                    "url": "https://example.com/chapter",
                    "formats": ["html"],
                    "wait_for": 250,
                },
                "firecrawl-key",
                timeout=30,
                bypass_proxy=True,
            )

        self.assertEqual(
            result,
            {"success": True, "data": {"html": "<img src='page.jpg'>"}},
        )
        client_ctor.assert_called_once_with(timeout=30.0, trust_env=False)
        request = client.__enter__.return_value.post.call_args
        self.assertEqual(request.args[0], "https://api.firecrawl.dev/v2/scrape")
        self.assertEqual(
            request.kwargs["json"],
            {
                "url": "https://example.com/chapter",
                "formats": ["html"],
                "waitFor": 250,
            },
        )

    def test_firecrawl_rejects_unknown_arguments_before_network_io(self) -> None:
        with mock.patch(
            "src.core.web_import.firecrawl_tools.httpx.Client"
        ) as client_ctor:
            result = execute_firecrawl_tool_sync(
                "firecrawl_scrape",
                {
                    "url": "https://example.com/chapter",
                    "retiredOption": True,
                },
                "firecrawl-key",
                timeout=30,
                bypass_proxy=False,
            )

        self.assertEqual(
            result,
            {"error": "Firecrawl 工具参数包含未知字段"},
        )
        client_ctor.assert_not_called()

    def test_firecrawl_does_not_leak_provider_error_bodies(self) -> None:
        request = httpx.Request("POST", "https://api.firecrawl.dev/v2/scrape")
        response = httpx.Response(
            401,
            request=request,
            text='{"error":"api_key=secret-value"}',
        )
        client = mock.MagicMock()
        client.__enter__.return_value.post.return_value = response

        with mock.patch(
            "src.core.web_import.firecrawl_tools.httpx.Client",
            return_value=client,
        ):
            result = execute_firecrawl_tool_sync(
                "firecrawl_scrape",
                {"url": "https://example.com/chapter"},
                "firecrawl-key",
                timeout=30,
                bypass_proxy=False,
            )

        self.assertEqual(result, {"error": "Firecrawl HTTP 401"})
        self.assertNotIn("secret-value", str(result))

    def test_firecrawl_memory_failure_is_not_a_tool_result(self) -> None:
        with mock.patch(
            "src.core.web_import.firecrawl_tools.httpx.Client",
            side_effect=MemoryError("native allocation failed"),
        ), self.assertRaisesRegex(MemoryError, "allocation failed"):
            execute_firecrawl_tool_sync(
                "firecrawl_scrape",
                {"url": "https://example.com/chapter"},
                "firecrawl-key",
                timeout=30,
                bypass_proxy=False,
            )

if __name__ == "__main__":
    unittest.main()
