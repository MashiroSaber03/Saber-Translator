"""Exercise both public transports against a real, deliberately slow HTTP server."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.shared.ai_transport import (
    AIRequestDeadlineExceeded,
    AsyncOpenAICompatibleTransport,
    OpenAICompatibleChatTransport,
    UnifiedChatRequest,
)
from src.shared.openai_execution import build_openai_compatible_runtime_options, resolve_openai_compatible_invocation
from src.shared.openai_options import OpenAICompatibleExecutionOptions, OpenAICompatibleOptions


def test_shared_default_is_300_seconds_and_explicit_timeouts_are_preserved():
    for stream in (False, True):
        options = OpenAICompatibleOptions(execution=OpenAICompatibleExecutionOptions(use_stream=stream))
        assert resolve_openai_compatible_invocation("custom", "chat", options).timeout == 300
        assert resolve_openai_compatible_invocation(
            "custom", "chat", options, build_openai_compatible_runtime_options(timeout=45),
        ).timeout == 45


@pytest.fixture
def model_server():
    state = {"attempts": 0, "mode": "slow", "disconnected": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["attempts"] += 1
            attempt = state["attempts"]
            stream = body.get("stream", False)
            if state["mode"] == "rate_limit" and attempt == 1:
                self.send_response(429)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.end_headers()
            try:
                slow = state["mode"] == "slow" or (state["mode"] == "retry" and attempt == 1)
                if state["mode"] == "partial":
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"part"}}]}\n\n')
                    self.wfile.flush()
                    slow = True
                if slow:
                    # Bytes arrive frequently enough to avoid HTTP read inactivity
                    # timeouts, but must not extend the complete attempt deadline.
                    for _ in range(30):
                        self.wfile.write(b": heartbeat\n\n" if stream else b" ")
                        self.wfile.flush()
                        time.sleep(0.02)
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
                    if stream else b'{"choices":[{"message":{"content":"ok"}}]}'
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                state["disconnected"].set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", state
    server.shutdown()
    server.server_close()
    thread.join()


def invoke(url, *, asynchronous, stream, retries=0, before_request=None):
    request = UnifiedChatRequest(
        provider="custom", api_key="", model="test", base_url=url,
        messages=[{"role": "user", "content": "test"}],
        openai_options=OpenAICompatibleOptions(execution=OpenAICompatibleExecutionOptions(
            use_stream=stream, transport_retries=retries,
        )),
        runtime_options=build_openai_compatible_runtime_options(timeout=0.15),
    )
    if asynchronous:
        async def prepare():
            if before_request:
                await asyncio.sleep(0.2)
        return asyncio.run(AsyncOpenAICompatibleTransport().complete(request, before_request=prepare))
    return OpenAICompatibleChatTransport().complete(request, before_request=before_request)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_incoming_bytes_cannot_extend_attempt_deadline(model_server, asynchronous, stream):
    url, state = model_server
    with pytest.raises(AIRequestDeadlineExceeded):
        invoke(url, asynchronous=asynchronous, stream=stream)
    assert state["attempts"] == 1
    assert state["disconnected"].wait(2), "timed-out connection was not closed"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_retry_receives_a_fresh_deadline(model_server, monkeypatch, asynchronous, stream):
    url, state = model_server
    state["mode"] = "retry"
    monkeypatch.setattr("src.shared.ai_transport._calculate_backoff", lambda *_: 0)
    assert invoke(url, asynchronous=asynchronous, stream=stream, retries=1) == "ok"
    assert state["attempts"] == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_partial_stream_is_not_replayed(model_server, asynchronous):
    url, state = model_server
    state["mode"] = "partial"
    with pytest.raises(AIRequestDeadlineExceeded):
        invoke(url, asynchronous=asynchronous, stream=True, retries=2)
    assert state["attempts"] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_pre_request_wait_is_outside_deadline(model_server, asynchronous):
    url, state = model_server
    state["mode"] = "success"
    assert invoke(url, asynchronous=asynchronous, stream=False,
                  before_request=lambda: time.sleep(0.2)) == "ok"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_http_retry_wait_is_outside_stream_deadline(model_server, monkeypatch, asynchronous):
    url, state = model_server
    state["mode"] = "rate_limit"
    monkeypatch.setattr("src.shared.ai_transport._calculate_backoff", lambda *_: 0.2)
    assert invoke(url, asynchronous=asynchronous, stream=True, retries=1) == "ok"
    assert state["attempts"] == 2


def test_cancelling_a_live_request_closes_connection_without_retry(model_server):
    url, state = model_server

    async def run():
        request = UnifiedChatRequest(
            provider="custom", api_key="", model="test", base_url=url,
            messages=[{"role": "user", "content": "test"}],
            openai_options=OpenAICompatibleOptions(execution=OpenAICompatibleExecutionOptions(
                use_stream=True, transport_retries=3,
            )),
        )
        task = asyncio.create_task(AsyncOpenAICompatibleTransport().complete(request))
        try:
            async with asyncio.timeout(2):
                while not state["attempts"]:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert state["attempts"] == 1
    assert state["disconnected"].wait(2)
