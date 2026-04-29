import os
import sys
import types
import unittest
from unittest import mock


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub


class WebImportAgentRuntimeTests(unittest.TestCase):
    def _build_agent(self, **agent_overrides):
        from src.core.web_import.agent import MangaScraperAgent

        return MangaScraperAgent(
            {
                "agent": {
                    "provider": "custom",
                    "apiKey": "test-key",
                    "customBaseUrl": "https://example.com/v1",
                    "modelName": "gpt-test",
                    **agent_overrides,
                }
            }
        )

    def test_call_llm_retries_transient_errors_up_to_max_retries(self) -> None:
        agent = self._build_agent(maxRetries=3, useStream=False)
        failing_error = RuntimeError("API 错误 500: temporary failure")
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

        try:
            result = agent._call_llm([{"role": "user", "content": "hello"}])
        except Exception:
            result = None

        self.assertIs(result, final_message)
        self.assertEqual(create_mock.call_count, 3)

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

        try:
            message = agent._call_llm([{"role": "user", "content": "hello"}])
        except Exception:
            message = None

        self.assertIsNotNone(message)
        self.assertEqual(message.content, "done")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].id, "call_1")
        self.assertEqual(message.tool_calls[0].function.name, "search")
        self.assertEqual(message.tool_calls[0].function.arguments, '{"query":"hello"}')


if __name__ == "__main__":
    unittest.main()
