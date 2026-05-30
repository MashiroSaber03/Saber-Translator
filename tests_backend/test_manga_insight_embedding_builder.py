import asyncio
import sys
import types
import unittest


if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    openai_stub.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_stub


class _FakeStorage:
    def __init__(self, batches=None, batch_data=None, pages=None, page_analysis=None):
        self._batches = batches or []
        self._batch_data = batch_data or {}
        self._pages = pages or []
        self._page_analysis = page_analysis or {}

    async def list_batches(self):
        return list(self._batches)

    async def load_batch_analysis(self, start_page, end_page):
        return self._batch_data.get((start_page, end_page))

    async def list_pages(self):
        return list(self._pages)

    async def load_page_analysis(self, page_num):
        return self._page_analysis.get(page_num)


class _BlockingStorage(_FakeStorage):
    def __init__(self, *args, started_event: asyncio.Event, release_event: asyncio.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self._started_event = started_event
        self._release_event = release_event

    async def list_batches(self):
        self._started_event.set()
        await self._release_event.wait()
        return await super().list_batches()


class _FakeEmbeddingClient:
    async def embed(self, text):
        return [float(len(text or ""))]


class _FakeVectorStore:
    def __init__(self, add_page_ok=True, add_event_ok=True):
        self.calls = []
        self.add_page_ok = add_page_ok
        self.add_event_ok = add_event_ok

    def is_available(self):
        return True

    async def delete_all(self):
        self.calls.append("delete_all")
        return True

    async def delete_all_pages(self):
        self.calls.append("delete_all_pages")
        return True

    async def delete_all_events(self):
        self.calls.append("delete_all_events")
        return True

    async def add_page_embedding(self, page_num, embedding, metadata):
        self.calls.append(("add_page", page_num, metadata.get("parent_batch")))
        return self.add_page_ok

    async def add_event_embedding(self, event_id, embedding, metadata):
        self.calls.append(("add_event", event_id, metadata.get("parent_batch")))
        return self.add_event_ok


class EmbeddingBuilderResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_embeddings_reports_progress_for_batches_and_fallback_pages(self):
        from src.core.manga_insight.embedding_builder import EmbeddingBuilder

        events = []
        storage = _FakeStorage(
            batches=[{"start_page": 1, "end_page": 1}],
            batch_data={
                (1, 1): {
                    "pages": [{"page_number": 1, "page_summary": "第一页摘要"}],
                    "key_events": [],
                }
            },
            pages=[1, 2],
            page_analysis={2: {"page_summary": "第二页摘要"}},
        )
        builder = EmbeddingBuilder("book-progress", storage, _FakeEmbeddingClient(), _FakeVectorStore())

        await builder.build_embeddings(progress_callback=lambda current, total, phase, message: events.append((current, total, phase, message)))

        self.assertTrue(any(phase == "embedding_batches" for _, _, phase, _ in events))
        self.assertTrue(any(phase == "embedding_fallback_pages" for _, _, phase, _ in events))
        self.assertTrue(any(current == total for current, total, _, _ in events if total > 0))

    async def test_build_embeddings_rejects_reentrant_rebuild_for_same_book(self):
        from src.core.manga_insight.embedding_builder import EmbeddingBuilder

        started = asyncio.Event()
        release = asyncio.Event()
        blocking_storage = _BlockingStorage(
            batches=[{"start_page": 1, "end_page": 1}],
            batch_data={
                (1, 1): {
                    "pages": [{"page_number": 1, "page_summary": "第一页摘要"}],
                    "key_events": [],
                }
            },
            started_event=started,
            release_event=release,
        )

        builder1 = EmbeddingBuilder("book-lock", blocking_storage, _FakeEmbeddingClient(), _FakeVectorStore())
        builder2 = EmbeddingBuilder(
            "book-lock",
            _FakeStorage(
                batches=[{"start_page": 1, "end_page": 1}],
                batch_data={
                    (1, 1): {
                        "pages": [{"page_number": 1, "page_summary": "第一页摘要"}],
                        "key_events": [],
                    }
                },
            ),
            _FakeEmbeddingClient(),
            _FakeVectorStore(),
        )

        first_task = asyncio.create_task(builder1.build_embeddings())
        await started.wait()

        second_result = await builder2.build_embeddings()
        release.set()
        first_result = await first_task

        self.assertTrue(first_result["success"])
        self.assertFalse(second_result["success"])
        self.assertEqual(second_result["error_code"], "EMBEDDING_REBUILD_IN_PROGRESS")
        self.assertEqual(second_result["status_code"], 409)

    async def test_build_embeddings_recreates_collections_before_rebuilding(self):
        from src.core.manga_insight.embedding_builder import EmbeddingBuilder

        storage = _FakeStorage(
            batches=[{"start_page": 1, "end_page": 2}],
            batch_data={
                (1, 2): {
                    "pages": [
                        {"page_number": 1, "page_summary": "第一页摘要"},
                        {"page_number": 2, "page_summary": "第二页摘要"},
                    ],
                    "key_events": ["事件一"],
                }
            },
            pages=[1, 2],
        )
        vector_store = _FakeVectorStore()

        builder = EmbeddingBuilder("book-1", storage, _FakeEmbeddingClient(), vector_store)
        result = await builder.build_embeddings()

        self.assertTrue(result["success"])
        self.assertEqual(vector_store.calls[0], "delete_all")
        self.assertNotIn("delete_all_pages", vector_store.calls)
        self.assertNotIn("delete_all_events", vector_store.calls)

    async def test_build_embeddings_from_pages_recreates_collections(self):
        from src.core.manga_insight.embedding_builder import EmbeddingBuilder

        storage = _FakeStorage(
            batches=[],
            pages=[7],
            page_analysis={7: {"page_summary": "单页摘要"}},
        )
        vector_store = _FakeVectorStore()

        builder = EmbeddingBuilder("book-1", storage, _FakeEmbeddingClient(), vector_store)
        result = await builder.build_embeddings()

        self.assertTrue(result["success"])
        self.assertEqual(vector_store.calls[0], "delete_all")
        self.assertNotIn("delete_all_pages", vector_store.calls)

    async def test_build_embeddings_counts_only_successful_writes(self):
        from src.core.manga_insight.embedding_builder import EmbeddingBuilder

        storage = _FakeStorage(
            batches=[{"start_page": 1, "end_page": 1}],
            batch_data={
                (1, 1): {
                    "pages": [{"page_number": 1, "page_summary": "第一页摘要"}],
                    "key_events": ["事件一"],
                }
            },
            pages=[1],
        )
        vector_store = _FakeVectorStore(add_page_ok=False, add_event_ok=False)

        builder = EmbeddingBuilder("book-1", storage, _FakeEmbeddingClient(), vector_store)
        result = await builder.build_embeddings()

        self.assertEqual(result["pages_count"], 0)
        self.assertEqual(result["events_count"], 0)
        self.assertEqual(result["total_count"], 0)


if __name__ == "__main__":
    unittest.main()
