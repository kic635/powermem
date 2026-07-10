"""OceanBase-focused coverage for embedding-failure text fallback (Issue #1094).

Follow-up from PR #1052: when query embeddings are unavailable, search must fall
back to text retrieval on OceanBase-capable backends and must not invent hits on
vector-only backends.

Adapter tests use a call-recording spy only (no parallel search logic).
Store tests construct OceanBaseVectorStore via __init__ with mocked client/table
boundaries, then exercise production search / _vector_search / _fulltext_search.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from powermem.storage.adapter import StorageAdapter
from powermem.storage.base import OutputData


class _FakeRow:
    def __init__(self, mapping):
        self._mapping = mapping


class _FakeResult:
    returns_rows = True

    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def begin(self):
        return _FakeTransaction()

    def execute(self, stmt):
        self.executed.append(stmt)
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows
        self.last_connection = None

    def connect(self):
        self.last_connection = _FakeConnection(self._rows)
        return self.last_connection


class RecordingOceanBaseStore:
    """Records search() kwargs only; never re-implements OceanBase logic."""

    # StorageAdapter._supports_text_search_without_vector() keys off __module__.
    __module__ = "powermem.storage.oceanbase.oceanbase"
    collection_name = "memories"
    hybrid_search = True
    text_field = "document"

    def __init__(self, *, hybrid_search: bool = True):
        self.hybrid_search = hybrid_search
        self.search_calls: list[dict] = []
        self._fixed_results = [
            OutputData(
                id=1,
                score=0.85,
                payload={"data": "recorded store result", "metadata": {}},
            )
        ]

    def search(
        self,
        query,
        vectors,
        limit=5,
        filters=None,
        sparse_embedding=None,
        threshold=None,
        retrieval_mode="auto",
        fusion="rrf",
        vector_weight=None,
        fts_weight=None,
        rrf_k=60,
        candidate_limit=None,
        include_explanation=False,
    ):
        self.search_calls.append(
            {
                "query": query,
                "vectors": vectors,
                "limit": limit,
                "filters": filters,
                "sparse_embedding": sparse_embedding,
                "threshold": threshold,
                "retrieval_mode": retrieval_mode,
                "fusion": fusion,
                "vector_weight": vector_weight,
                "fts_weight": fts_weight,
                "rrf_k": rrf_k,
                "candidate_limit": candidate_limit,
                "include_explanation": include_explanation,
            }
        )
        return list(self._fixed_results)


def _fts_rows(*scores: float):
    return [_FakeRow({"score": score}) for score in scores]


def _fts_hit(score: float = 0.85, text: str = "needle memory") -> OutputData:
    return OutputData(
        id=1,
        score=score,
        payload={
            "data": text,
            "_fts_score": score,
            "_fts_quality_score": 1.0,
        },
    )


@pytest.fixture
def recording_store():
    return RecordingOceanBaseStore(hybrid_search=True)


@pytest.fixture
def vector_only_recording_store():
    return RecordingOceanBaseStore(hybrid_search=False)


@pytest.fixture
def recording_adapter(recording_store):
    return StorageAdapter(recording_store)


@pytest.fixture
def oceanbase_store(monkeypatch):
    pytest.importorskip("pyobvector")
    from powermem.storage.oceanbase.models import create_memory_model
    from powermem.storage.oceanbase.oceanbase import OceanBaseVectorStore

    mock_obvector = MagicMock()
    mock_obvector.check_table_exists.return_value = True
    mock_obvector.engine = MagicMock()
    mock_obvector.metadata_obj = MagicMock()
    mock_obvector.ann_search.return_value = iter([])

    def fake_create_client(self, **kwargs):
        self.obvector = mock_obvector

    def fake_create_col(self):
        self.model_class = create_memory_model(
            "memories",
            3,
            include_sparse=False,
        )
        self.table = self.model_class.__table__

    monkeypatch.setattr(OceanBaseVectorStore, "_create_client", fake_create_client)
    monkeypatch.setattr(OceanBaseVectorStore, "_create_col", fake_create_col)
    monkeypatch.setattr(
        OceanBaseVectorStore,
        "_configure_vector_index_settings",
        lambda self: None,
    )
    monkeypatch.setattr(
        OceanBaseVectorStore,
        "_check_and_create_fulltext_index",
        lambda self: None,
    )

    store = OceanBaseVectorStore(
        collection_name="memories",
        embedding_model_dims=3,
        host="127.0.0.1",
        port="2881",
        user="root",
        password="pwd",
        db_name="test",
        hybrid_search=True,
        enable_native_hybrid=False,
        auto_configure_vector_index=False,
        create_vector_index=False,
    )
    store._mock_obvector = mock_obvector
    return store


def _configure_fts_engine_store(store, rows):
    engine = _FakeEngine(rows)
    store.obvector = SimpleNamespace(engine=engine, metadata_obj=MagicMock())
    store.collection_name = "memories"
    store.fulltext_field = "fulltext_content"
    store._generate_where_clause = lambda filters, table=None: []
    store._get_standard_select_columns = lambda: []
    store._parse_row_to_dict = lambda row, **kwargs: {
        "vector_id": 1,
        "text_content": "needle memory",
        "metadata": {"user_id": "u1", "metadata": {}},
        "score_or_distance": row["score"],
    }
    return engine


def _use_embedded_hybrid_path(store):
    """Force the sequential embedded hybrid branch (no ThreadPoolExecutor)."""
    store.connection_args = {**store.connection_args, "host": None}


# ------------------------------------------------------------------ #
# StorageAdapter → OceanBase spy (routing only)
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("retrieval_mode", ["auto", "fts", "hybrid"])
def test_adapter_forwards_no_vector_fallback_to_oceanbase_store(
    recording_adapter,
    recording_store,
    retrieval_mode,
):
    results = recording_adapter.search_memories(
        query_embedding=None,
        query="offline fallback phrase",
        retrieval_mode=retrieval_mode,
        limit=5,
    )

    assert len(results) == 1
    call = recording_store.search_calls[-1]
    assert call["vectors"] is None
    assert call["retrieval_mode"] == retrieval_mode
    assert call["query"] == "offline fallback phrase"
    assert call["limit"] == 5


def test_adapter_auto_mode_without_embedding_or_query_returns_empty(
    recording_adapter,
    recording_store,
):
    results = recording_adapter.search_memories(
        query_embedding=None,
        query=None,
        retrieval_mode="auto",
        limit=5,
    )

    assert results == []
    assert recording_store.search_calls == []


def test_adapter_hybrid_mode_without_embedding_or_query_returns_empty(
    recording_adapter,
    recording_store,
):
    results = recording_adapter.search_memories(
        query_embedding=None,
        query="   ",
        retrieval_mode="hybrid",
        limit=5,
    )

    assert results == []
    assert recording_store.search_calls == []


def test_adapter_vector_mode_without_embedding_skips_store(
    recording_adapter,
    recording_store,
):
    results = recording_adapter.search_memories(
        query_embedding=None,
        query="should not search",
        retrieval_mode="vector",
        limit=5,
    )

    assert results == []
    assert recording_store.search_calls == []


def test_adapter_vector_only_oceanbase_backend_blocks_text_fallback(
    vector_only_recording_store,
):
    adapter = StorageAdapter(vector_only_recording_store)

    results = adapter.search_memories(
        query_embedding=None,
        query="no vector-only false positives",
        retrieval_mode="auto",
        limit=5,
    )

    assert results == []
    assert vector_only_recording_store.search_calls == []


def test_adapter_forwards_fallback_controls_to_oceanbase_store(
    recording_adapter,
    recording_store,
):
    sparse = {12: 0.75}
    recording_adapter._generate_sparse_embedding = MagicMock(return_value=sparse)

    recording_adapter.search_memories(
        query_embedding=None,
        query="controlled fallback",
        retrieval_mode="hybrid",
        fusion="weighted",
        vector_weight=0.25,
        fts_weight=0.75,
        rrf_k=24,
        candidate_limit=40,
        threshold=0.5,
        filters={"scope": "personal"},
        user_id="u1",
        limit=5,
        include_explanation=True,
    )

    call = recording_store.search_calls[-1]
    assert call["sparse_embedding"] == sparse
    assert call["fusion"] == "weighted"
    assert call["vector_weight"] == 0.25
    assert call["fts_weight"] == 0.75
    assert call["rrf_k"] == 24
    assert call["candidate_limit"] == 40
    assert call["threshold"] == 0.5
    assert call["include_explanation"] is True
    assert call["limit"] == 40
    assert call["filters"] == {"user_id": "u1", "scope": "personal"}


# ------------------------------------------------------------------ #
# OceanBaseVectorStore production paths
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("vectors", [None, []])
def test_vector_search_returns_empty_without_valid_vectors(oceanbase_store, vectors):
    """Core safety belt: missing embeddings must not call ann_search."""
    oceanbase_store._mock_obvector.ann_search.reset_mock()

    results = oceanbase_store._vector_search("needle", vectors, limit=5)

    assert results == []
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


def test_fulltext_search_builds_match_sql_and_normalizes_quality_score(
    oceanbase_store,
):
    engine = _configure_fts_engine_store(oceanbase_store, _fts_rows(0.85))

    results = oceanbase_store._fulltext_search("needle memory", limit=5)

    assert len(results) == 1
    assert results[0].payload["_fts_score"] == pytest.approx(0.85)
    assert results[0].payload["_quality_score"] == 1.0
    executed = str(engine.last_connection.executed[0])
    assert "MATCH(fulltext_content)" in executed
    assert "AGAINST" in executed


def test_search_fts_mode_threshold_uses_normalized_quality_score(oceanbase_store):
    """FTS search() thresholds against _quality_score (always 1.0 from _fulltext_search)."""
    _configure_fts_engine_store(oceanbase_store, _fts_rows(0.4, 0.85))

    kept = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="fts",
        threshold=0.5,
        limit=5,
    )
    filtered = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="fts",
        threshold=1.5,
        limit=5,
    )

    assert len(kept) == 2
    assert all(result.payload["_quality_score"] == 1.0 for result in kept)
    assert filtered == []


def test_search_fts_mode_forwards_filters_and_candidate_limit(oceanbase_store):
    seen = {}

    def capture_fulltext(query, limit=5, filters=None):
        seen["query"] = query
        seen["limit"] = limit
        seen["filters"] = filters
        return [_fts_hit()]

    oceanbase_store._fulltext_search = capture_fulltext

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="fts",
        filters={"user_id": "u1"},
        candidate_limit=40,
        limit=5,
    )

    assert len(results) == 1
    assert seen == {
        "query": "needle",
        "limit": 40,
        "filters": {"user_id": "u1"},
    }


def test_search_vector_mode_without_vectors_returns_empty(oceanbase_store):
    fts = MagicMock(return_value=[_fts_hit()])
    oceanbase_store._fulltext_search = fts

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="vector",
        limit=5,
    )

    assert results == []
    fts.assert_not_called()
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


def test_search_auto_mode_without_vectors_on_vector_only_backend_returns_empty(
    oceanbase_store,
):
    oceanbase_store.hybrid_search = False
    fts = MagicMock()
    oceanbase_store._fulltext_search = fts

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="auto",
        limit=5,
    )

    assert results == []
    fts.assert_not_called()
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


def test_search_hybrid_mode_without_vectors_or_query_returns_empty(oceanbase_store):
    fts = MagicMock()
    oceanbase_store._fulltext_search = fts

    results = oceanbase_store.search(
        "",
        vectors=None,
        retrieval_mode="hybrid",
        limit=5,
    )

    assert results == []
    fts.assert_not_called()
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


@pytest.mark.parametrize("retrieval_mode", ["auto", "hybrid"])
def test_search_without_vectors_runs_real_vector_branch_and_fts(
    oceanbase_store,
    retrieval_mode,
):
    """Embedding failure path: real _vector_search returns []; FTS still yields hits."""
    fts = MagicMock(return_value=[_fts_hit()])
    oceanbase_store._fulltext_search = fts
    _use_embedded_hybrid_path(oceanbase_store)
    oceanbase_store._mock_obvector.ann_search.reset_mock()

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode=retrieval_mode,
        threshold=0.5,
        limit=5,
    )

    assert len(results) == 1
    assert results[0].payload["_quality_score"] == 1.0
    fts.assert_called_once()
    # Real _vector_search ran and refused to call ANN with None vectors.
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


def test_search_hybrid_without_vectors_uses_candidate_limit(oceanbase_store):
    seen_limits = []

    def capture_fulltext(query, limit=5, filters=None):
        seen_limits.append(limit)
        return [_fts_hit()]

    oceanbase_store._fulltext_search = capture_fulltext
    _use_embedded_hybrid_path(oceanbase_store)

    oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="hybrid",
        candidate_limit=40,
        limit=5,
    )

    assert seen_limits == [40]


def test_search_hybrid_weighted_fusion_without_vectors_uses_fts_only(
    oceanbase_store,
):
    """With no query vector, weighted hybrid still returns FTS-only fused results."""
    oceanbase_store._fulltext_search = MagicMock(return_value=[_fts_hit(score=0.85)])
    _use_embedded_hybrid_path(oceanbase_store)
    oceanbase_store._mock_obvector.ann_search.reset_mock()

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="hybrid",
        fusion="weighted",
        vector_weight=0.25,
        fts_weight=0.75,
        limit=5,
    )

    assert len(results) == 1
    assert results[0].payload["_fusion_info"]["fusion_method"] == "weighted"
    oceanbase_store._fulltext_search.assert_called_once()
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


def test_search_hybrid_without_vectors_on_remote_path_still_queries_fts(
    oceanbase_store,
):
    """Remote OceanBase path uses ThreadPoolExecutor; vector=None must still FTS."""
    oceanbase_store._fulltext_search = MagicMock(return_value=[_fts_hit()])
    # Keep host set so _hybrid_search takes the remote/parallel branch.
    assert oceanbase_store.connection_args.get("host")
    oceanbase_store._mock_obvector.ann_search.reset_mock()

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="hybrid",
        limit=5,
    )

    assert len(results) == 1
    oceanbase_store._fulltext_search.assert_called_once()
    oceanbase_store._mock_obvector.ann_search.assert_not_called()


def test_search_native_hybrid_without_vectors_falls_back_to_application_hybrid(
    oceanbase_store,
    monkeypatch,
):
    """Native hybrid requires a query vector; without one it must fall back to FTS."""
    from powermem.utils import oceanbase_util

    oceanbase_store.enable_native_hybrid = True
    oceanbase_store._fulltext_search = MagicMock(return_value=[_fts_hit(text="application hybrid hit")])
    _use_embedded_hybrid_path(oceanbase_store)

    monkeypatch.setattr(
        oceanbase_util.OceanBaseUtil,
        "check_filters_all_in_columns",
        lambda filters, model_class: True,
    )

    results = oceanbase_store.search(
        "needle",
        vectors=None,
        retrieval_mode="hybrid",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].payload["data"] == "application hybrid hit"
    oceanbase_store._fulltext_search.assert_called_once()
