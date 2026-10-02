"""Model parsing, persistence and API contracts must accept the same data."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
from sqlalchemy import select, update

from tests_backend.v2.test_stage5_insight import (
    FakeInsightAlgorithms, _run_job, insight_platform,
)
from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.insight.continuation import DefaultContinuationAlgorithms
from src.backend_v2.insight.derived import (
    InsightDerivedRepository, ProviderDerivedAlgorithms, _normalized_timeline_result,
)
from src.backend_v2.insight.worker import ProviderInsightAlgorithms
from src.backend_v2.insight.repository import InsightRepository
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.schema import timeline_versions, timeline_characters, timeline_events, provider_settings
from src.backend_v2.studio.service import DefaultStudioAlgorithms
from src.backend_v2.translation.pipeline import CoreTranslationAlgorithms
from src.shared.openai_execution import OpenAICompatibleBusinessRetriesExhaustedError


def _config(platform):
    config = SettingsResolver(platform["engine"]).resolve_insight(
        book_id=platform["book"]["id"], scope="full",
    )
    for section in ("chat", "vlm"):
        config[section]["openai_options"]["execution"]["business_retries"] = 1
    return config


@pytest.fixture
def responses(monkeypatch):
    values, calls = [], []

    def complete(_self, request, **kwargs):
        calls.append(request)
        result = values.pop(0)
        if isinstance(result, Exception):
            raise result
        return json.dumps(result, ensure_ascii=False)

    async def async_complete(*args, **kwargs):
        return complete(*args, **kwargs)

    async def sleep(_seconds):
        pass

    monkeypatch.setattr("src.shared.ai_transport.AsyncOpenAICompatibleTransport.complete", async_complete)
    monkeypatch.setattr("src.shared.ai_transport.OpenAICompatibleChatTransport.complete", complete)
    monkeypatch.setattr("src.shared.openai_execution.asyncio.sleep", sleep)
    monkeypatch.setattr("src.shared.openai_execution.time.sleep", lambda _: None)
    return values, calls


@pytest.mark.parametrize("kind", ["overview", "compressed", "script", "page", "summary", "terms"])
@pytest.mark.parametrize("exhaust", [False, True])
def test_result_field_validation_uses_configured_retries(insight_platform, responses, kind, exhaust):
    config = _config(insight_platform)
    values, calls = responses
    valid = {
        "overview": {"title": "Title", "content": "Story"},
        "compressed": {"summary": {"events": ["Story"]}},
        "script": {"script": "Story"},
        "page": {"storyText": "Story", "continuityText": "", "dialogueText": "", "characters": [], "finalPrompt": "Draw story"},
        "summary": {"summary": "Story"},
        "terms": {"terms": [{"source": " A ", "target": " B "}]},
    }[kind]
    bad = {"terms": [{"original": "A", "translation": "B"}]} if kind == "terms" else {}
    values.extend([bad, bad if exhaust else valid])
    pages = [{"pageId": "p", "pageNumber": 1, "analysis": {"summary": "Story"}}]

    def generate():
        if kind == "overview":
            return ProviderDerivedAlgorithms().build_overview(pages, template="story_summary", config=config)
        if kind == "compressed":
            return ProviderDerivedAlgorithms().build_compressed_context(pages, config=config)
        if kind == "script":
            return DefaultContinuationAlgorithms().generate_script(context={}, config=config)
        if kind == "page":
            return DefaultContinuationAlgorithms().generate_page(ordinal=1, script="Story", previous=None, config=config)
        if kind == "summary":
            return DefaultStudioAlgorithms().summarize([], config=config)
        return CoreTranslationAlgorithms().extract_terms(["A"], config["chat"], prompt="Return source and target")

    if exhaust:
        with pytest.raises(OpenAICompatibleBusinessRetriesExhaustedError):
            generate()
    else:
        result = generate()
        if kind == "terms":
            assert result["candidates"] == [{"source": "A", "target": "B", "note": "", "matchMode": "text"}]
        elif kind == "page":
            assert result == {**valid, "status": "ready"}
        elif kind == "script":
            assert result == "Story"
        else:
            assert result == valid
    assert len(calls) == 2


def _page(number):
    return {"page_number": number, "page_summary": f"Page {number}", "key_events": [], "continuity_notes": "", "warnings": []}


@pytest.mark.parametrize("outcome", ["recovered", "exhausted", "transport-error", "memory-error"])
def test_page_retry_preserves_successes(insight_platform, responses, outcome):
    values, calls = responses
    first = _page(1)
    values.append({"pages": [first, {"page_number": 2}]})
    values.append({
        "recovered": {"pages": [_page(2)]},
        "exhausted": {"pages": [{"page_number": 1}, {"page_number": 2}]},
        "transport-error": RuntimeError("provider failed"),
        "memory-error": MemoryError("allocation failed"),
    }[outcome])
    image = BytesIO()
    with Image.new("RGB", (2, 2), "white") as im:
        im.save(image, format="PNG")

    def generate():
        return ProviderInsightAlgorithms().analyze_batch(
            [image.getvalue()] * 2, page_numbers=[1, 2], previous_batches=[], config=_config(insight_platform),
        )

    if outcome == "memory-error":
        with pytest.raises(MemoryError):
            generate()
    else:
        result = generate()
        assert result["pages"] == ([first, _page(2)] if outcome == "recovered" else [first])
    assert len(calls) == 2


def _timeline():
    return {
        "content": {"story_summary": "Story"},
        "events": [{"summary": "Event", "page_numbers": [1]}],
        "characters": [{"name": "A", "description": "Description", "first_page": 1, "key_moments": []}],
    }


@pytest.mark.parametrize("fault", ["unknown-page", "duplicate-page", "mismatched-pages", "reserved-event-id", "reserved-character-id", "unknown-character-page", "importance-type", "arc-type"])
def test_timeline_retries_unpublishable_results(insight_platform, responses, fault):
    values, calls = responses
    valid = _timeline()
    bad = deepcopy(valid)
    if fault == "unknown-page":
        bad["events"][0]["page_numbers"] = [999]
    elif fault == "duplicate-page":
        bad["events"][0]["page_numbers"] = [1, 1]
    elif fault == "mismatched-pages":
        bad["events"][0].update(page_ids=["p2"], page_numbers=[1])
    elif fault == "reserved-event-id":
        bad["events"][0]["eventId"] = "model-owned"
    elif fault == "reserved-character-id":
        bad["characters"][0]["characterId"] = "model-owned"
    elif fault == "unknown-character-page":
        bad["characters"][0]["first_page"] = 999
    elif fault == "importance-type":
        bad["events"][0]["importance"] = {"level": "high"}
    else:
        bad["characters"][0]["arc"] = ["wrong type"]
    values.extend([bad, valid])
    result = ProviderDerivedAlgorithms().build_timeline([
        {"pageIds": ["p1", "p2"], "pageNumbers": [1, 2], "analysis": {"summary": "Story"}},
    ], config=_config(insight_platform))
    assert len(calls) == 2
    assert result["mode"] == "enhanced"
    assert result["events"][0]["page_ids"] == ["p1"]


def test_compressed_mode_retains_actual_source_pages(insight_platform, responses):
    values, calls = responses
    valid = _timeline()
    valid["events"][0]["page_numbers"] = [237]
    valid["characters"] = []
    values.extend([{}, {}, valid])
    result = ProviderDerivedAlgorithms().build_timeline([{
        "pageIds": ["p237", "p239"], "pageNumbers": [237, 239],
        "analysis": {"compressed_context": {"summary": {"events": ["Story"]}}},
    }], config=_config(insight_platform))
    assert result["mode"] == "compressed"
    assert result["events"][0]["page_ids"] == ["p237"]
    prompt = calls[-1].messages[-1]["content"]
    assert 'page_numbers=[237,239]' in prompt and 'page_ids=["p237","p239"]' in prompt


def test_structured_compressed_context_does_not_block_simple_fallback(insight_platform, responses):
    values, calls = responses
    values.extend([{}] * 4)
    result = ProviderDerivedAlgorithms().build_timeline([{
        "pageId": "p1", "pageNumber": 1,
        "analysis": {"compressed_context": {"summary": {"text": "Structured story"}},
                     "key_events": [{"summary": "Known event"}]},
    }], config=_config(insight_platform))
    assert len(calls) == 4
    assert result["mode"] == "simple"
    assert result["content"]["story_summary"] == "Known event"


@pytest.mark.parametrize("arc", [None, ""])
def test_optional_timeline_values_round_trip_and_match_api_schema(insight_platform, arc):
    import jsonschema
    import yaml

    p = insight_platform
    book_id = p["book"]["id"]
    InsightAnalysisCommandService(p["engine"]).create_analysis_job(
        command={"bookId": book_id, "scope": "full"}, idempotency_key="contract-round-trip",
    )
    assert _run_job(p, FakeInsightAlgorithms()) == "completed"
    repo = InsightDerivedRepository(p["engine"])
    frozen = repo.snapshot(book_id=book_id)
    raw = _timeline()
    raw["content"].update(plot_arcs=None, plot_threads=None)
    raw["characters"][0].update(arc=arc, related_page_numbers=None, key_moments=[{"summary": "Moment", "page": None}])
    raw["events"][0]["importance"] = None
    normalized = _normalized_timeline_result(raw, mode="enhanced", fallback_reason=None)
    with p["engine"].begin() as connection:
        publication = repo.publish_timeline(connection=connection, frozen=frozen, result=normalized)
    timeline_id = publication["timelineVersionId"]
    # Also verify rows already saved by the old validator remain readable.
    with p["engine"].begin() as connection:
        connection.execute(update(timeline_versions).where(timeline_versions.c.id == timeline_id).values(
            content_json=json.dumps({**raw["content"], **{k: normalized["content"][k] for k in ("requested_mode", "actual_mode", "fallback_reason", "degraded")}}),
        ))
        connection.execute(update(timeline_characters).where(timeline_characters.c.timeline_version_id == timeline_id).values(payload_json=json.dumps(raw["characters"][0])))
        event = json.loads(connection.execute(select(timeline_events.c.payload_json).where(timeline_events.c.timeline_version_id == timeline_id)).scalar_one())
        event["importance"] = None
        connection.execute(update(timeline_events).where(timeline_events.c.timeline_version_id == timeline_id).values(payload_json=json.dumps(event)))
    response = repo.get_timeline(book_id=book_id)
    assert "plot_arcs" not in response["content"] and "plot_threads" not in response["content"]
    assert "arc" not in response["characters"][0]
    assert "page" not in response["characters"][0]["key_moments"][0]
    assert "importance" not in response["events"][0]
    document = yaml.safe_load((Path(__file__).parents[2] / "openapi/v2.yaml").read_text(encoding="utf-8"))
    jsonschema.validate(response, {"$ref": "#/components/schemas/InsightTimeline", "components": document["components"]})

    characters = repo.list_timeline_characters(book_id=book_id)["characters"]
    assert characters == response["characters"]
    character = repo.get_timeline_character(book_id=book_id, character_id=characters[0]["characterId"])["character"]
    assert character == characters[0]
    jsonschema.validate(character, {"$ref": "#/components/schemas/InsightTimelineCharacter", "components": document["components"]})

    raw["content"].update(
        plot_arcs=[{"id": "a", "name": "A", "description": "D", "page_range": {"start": 1, "end": 1}, "mood": None, "event_ids": None}],
        plot_threads=[{"id": "t", "name": "T", "type": "clue", "status": "open", "description": None, "introduced_at": None, "resolved_at": None}],
    )
    normalized = _normalized_timeline_result(raw, mode="enhanced", fallback_reason=None)
    with p["engine"].begin() as connection:
        repo.publish_timeline(connection=connection, frozen=frozen, result=normalized)
    response = repo.get_timeline(book_id=book_id)
    jsonschema.validate(response, {"$ref": "#/components/schemas/InsightTimeline", "components": document["components"]})


@pytest.mark.parametrize("recover", [False, True])
def test_partial_page_retry_over_http_publishes_global_results(insight_platform, recover):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    calls = []
    first = _page(1)
    first["key_events"] = [{"summary": "Known event", "importance": "normal"}]
    replies = [
        {"pages": [first, {"page_number": 2}]},
        {"pages": [_page(2)] if recover else [{"page_number": 2}]},
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            raw = json.dumps(replies.pop(0))
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": raw}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    p = insight_platform
    try:
        with p["engine"].begin() as connection:
            payload = json.loads(connection.execute(select(provider_settings.c.payload_json).where(
                provider_settings.c.domain == "insight_vlm", provider_settings.c.provider == "ollama",
            )).scalar_one())
            payload["customBaseUrl"] = f"http://127.0.0.1:{server.server_port}/v1"
            payload["openaiOptions"]["execution"]["business_retries"] = 1
            connection.execute(update(provider_settings).where(
                provider_settings.c.domain == "insight_vlm", provider_settings.c.provider == "ollama",
            ).values(payload_json=json.dumps(payload)))
        book_id = p["book"]["id"]
        accepted = InsightAnalysisCommandService(p["engine"]).create_analysis_job(
            command={"bookId": book_id, "scope": "full"}, idempotency_key="http-partial-batch",
        )
        expected_status = "completed" if recover else "completed_with_errors"
        assert _run_job(p, ProviderInsightAlgorithms()) == expected_status
        assert len(calls) == 2
        repository = InsightRepository(p["engine"])
        run = repository.get_run(accepted["runId"])
        assert run["successCount"] == (2 if recover else 1)
        assert repository.page_detail(page_id=p["page_ids"][0])["analysis"]["page_summary"] == first["page_summary"]
        derived = InsightDerivedRepository(p["engine"])
        assert derived.get_artifact(book_id=book_id, kind="compressed_context", template="default")["status"] == ("ready" if recover else "degraded")
        assert derived.get_timeline(book_id=book_id)["status"] == ("ready" if recover else "degraded")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
