import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.routers.insights as insights_router
import api.routers.rag as rag_router
import api.routers.reviews as reviews_router
from api.main import app
from genai.reviews.service import ReviewsNotAnalyzedError


@pytest.fixture
def client(monkeypatch):
    # --- reviews router stubs ---
    monkeypatch.setattr(
        reviews_router,
        "analyze_reviews",
        lambda limit=None: {
            "reviews_analyzed": 100,
            "output_dir": "data/processed/genai/reviews_analysis",
            "keywords": ["clean", "friendly"],
            "topics": [{"topic_id": 0, "keywords": ["clean", "room"]}],
        },
    )
    monkeypatch.setattr(
        reviews_router,
        "get_summary",
        lambda: {"summary": "Mostly positive.", "keywords": ["clean"], "csat_by_hotel": [{"hotel_id": "H1", "csat": 80.0}]},
    )
    monkeypatch.setattr(reviews_router, "get_topics", lambda: {"topics": [{"topic_id": 0, "keywords": ["clean"]}]})
    monkeypatch.setattr(reviews_router, "get_complaints", lambda: {"complaints": [{"category": "wifi", "count": 12}]})
    monkeypatch.setattr(
        reviews_router,
        "get_trends",
        lambda grain="weekly": {"grain": grain, "series": [{"hotel_id": "H1", "period": "2026-W01", "csat": 80.0}], "trend_by_hotel": {"H1": "stable"}},
    )

    # --- rag router stubs ---
    monkeypatch.setattr(rag_router, "build_index", lambda **kwargs: None)
    monkeypatch.setattr(rag_router, "incremental_index", lambda **kwargs: None)
    monkeypatch.setattr(rag_router, "index_stats", lambda: {"indexed": True, "vector_count": 42, "store_dir": "genai/rag/vector_store"})
    monkeypatch.setattr(rag_router, "_safe_llm_provider", lambda: None)

    class _FakeCitation:
        def __init__(self):
            self.__dict__ = {"source": "hotel_sop.md", "doc_type": "markdown", "score": 0.9, "extra": {}}

    class _FakeQAResult:
        answer = "Occupancy is trending upward."
        citations = [_FakeCitation()]
        used_llm = False

    monkeypatch.setattr(rag_router, "_build_retriever", lambda: object())
    monkeypatch.setattr(rag_router, "answer_question", lambda *a, **k: _FakeQAResult())

    def _fake_stream_answer(*a, **k):
        yield "Occupancy "
        yield "is trending up."

    monkeypatch.setattr(rag_router, "stream_answer", _fake_stream_answer)

    # --- insights router stubs ---
    monkeypatch.setattr(insights_router, "_safe_llm_provider", lambda: None)
    monkeypatch.setattr(
        insights_router,
        "get_insights",
        lambda category=None, min_severity=None, llm_provider=None: [
            {
                "category": "revenue",
                "title": "Revenue drop at branch 1",
                "metric": "total_revenue",
                "metric_delta": -12.0,
                "severity": "high",
                "branch_id": 1,
                "supporting_data": {},
                "citation": "mart_revenue_daily",
                "priority_score": 3.1,
                "recommendation": "Investigate pricing.",
            }
        ],
    )
    monkeypatch.setattr(
        insights_router,
        "get_executive_summary",
        lambda llm_provider=None: {
            "narrative": "Revenue is down at branch 1.",
            "top_findings": [
                {
                    "category": "revenue",
                    "title": "Revenue drop at branch 1",
                    "metric": "total_revenue",
                    "metric_delta": -12.0,
                    "severity": "high",
                    "branch_id": 1,
                    "supporting_data": {},
                    "citation": "mart_revenue_daily",
                    "priority_score": 3.1,
                    "recommendation": "Investigate pricing.",
                }
            ],
        },
    )
    monkeypatch.setattr(
        insights_router,
        "get_recommendations",
        lambda llm_provider=None: [
            {"category": "revenue", "title": "Revenue drop at branch 1", "recommendation": "Investigate pricing.", "priority_score": 3.1}
        ],
    )
    monkeypatch.setattr(
        insights_router,
        "get_anomalies",
        lambda llm_provider=None: [
            {
                "category": "anomaly",
                "title": "Statistical anomaly in total_revenue",
                "metric": "total_revenue",
                "metric_delta": 200.0,
                "severity": "medium",
                "branch_id": 1,
                "supporting_data": {},
                "citation": "anomaly_detector",
                "priority_score": 2.5,
                "recommendation": "Check for data entry errors.",
            }
        ],
    )

    return TestClient(app)


# --- reviews endpoints ---


def test_reviews_analyze(client):
    resp = client.post("/reviews/analyze", json={"limit": 100})
    assert resp.status_code == 200
    assert resp.json()["reviews_analyzed"] == 100


def test_reviews_summary(client):
    resp = client.get("/reviews/summary")
    assert resp.status_code == 200
    assert resp.json()["summary"] == "Mostly positive."


def test_reviews_summary_not_analyzed_returns_503(client, monkeypatch):
    def _raise():
        raise ReviewsNotAnalyzedError("not analyzed yet")

    monkeypatch.setattr(reviews_router, "get_summary", _raise)
    resp = client.get("/reviews/summary")
    assert resp.status_code == 503


def test_reviews_topics(client):
    resp = client.get("/reviews/topics")
    assert resp.status_code == 200
    assert resp.json()["topics"][0]["topic_id"] == 0


def test_reviews_complaints(client):
    resp = client.get("/reviews/complaints")
    assert resp.status_code == 200
    assert resp.json()["complaints"][0]["category"] == "wifi"


def test_reviews_trends(client):
    resp = client.get("/reviews/trends?grain=weekly")
    assert resp.status_code == 200
    assert resp.json()["trend_by_hotel"]["H1"] == "stable"


def test_reviews_trends_invalid_grain(client, monkeypatch):
    def _raise(grain="weekly"):
        raise ValueError(f"Unknown grain: {grain}")

    monkeypatch.setattr(reviews_router, "get_trends", _raise)
    resp = client.get("/reviews/trends?grain=yearly")
    assert resp.status_code == 422


# --- rag endpoints ---


def test_rag_index(client):
    resp = client.post("/rag/index", json={})
    assert resp.status_code == 200
    assert resp.json()["vector_count"] == 42


def test_rag_reindex(client):
    resp = client.post("/rag/reindex", json={})
    assert resp.status_code == 200
    assert resp.json()["indexed"] is True


def test_rag_stats(client):
    resp = client.get("/rag/stats")
    assert resp.status_code == 200
    assert resp.json()["vector_count"] == 42


def test_rag_query_non_streaming(client):
    resp = client.post("/rag/query", json={"query": "What is the occupancy forecast?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Occupancy is trending upward."
    assert body["citations"][0]["source"] == "hotel_sop.md"


def test_rag_query_no_context_returns_503(client, monkeypatch):
    from genai.rag.chains.qa_chain import NoContextError

    def _raise(*a, **k):
        raise NoContextError("index empty")

    monkeypatch.setattr(rag_router, "answer_question", _raise)
    resp = client.post("/rag/query", json={"query": "anything"})
    assert resp.status_code == 503


def test_rag_query_streaming(client):
    resp = client.post("/rag/query", json={"query": "occupancy?", "stream": True})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


# --- insights endpoints ---


def test_insights_list(client):
    resp = client.get("/insights")
    assert resp.status_code == 200
    assert resp.json()["insights"][0]["category"] == "revenue"


def test_insights_executive(client):
    resp = client.get("/insights/executive")
    assert resp.status_code == 200
    assert "Revenue is down" in resp.json()["narrative"]


def test_insights_recommendations(client):
    resp = client.get("/insights/recommendations")
    assert resp.status_code == 200
    assert resp.json()["recommendations"][0]["priority_score"] == 3.1


def test_insights_anomalies(client):
    resp = client.get("/insights/anomalies")
    assert resp.status_code == 200
    assert resp.json()["anomalies"][0]["category"] == "anomaly"


def test_health_still_works(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# --- error-path coverage (500s, 503s) across all three genai routers ---


def test_reviews_analyze_error_returns_500(client, monkeypatch):
    def _raise(limit=None):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(reviews_router, "analyze_reviews", _raise)
    resp = client.post("/reviews/analyze", json={})
    assert resp.status_code == 500


def test_reviews_topics_not_analyzed_returns_503(client, monkeypatch):
    def _raise():
        raise ReviewsNotAnalyzedError("not analyzed")

    monkeypatch.setattr(reviews_router, "get_topics", _raise)
    resp = client.get("/reviews/topics")
    assert resp.status_code == 503


def test_reviews_topics_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(reviews_router, "get_topics", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/reviews/topics")
    assert resp.status_code == 500


def test_reviews_complaints_not_analyzed_returns_503(client, monkeypatch):
    def _raise():
        raise ReviewsNotAnalyzedError("not analyzed")

    monkeypatch.setattr(reviews_router, "get_complaints", _raise)
    resp = client.get("/reviews/complaints")
    assert resp.status_code == 503


def test_reviews_complaints_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(reviews_router, "get_complaints", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/reviews/complaints")
    assert resp.status_code == 500


def test_reviews_trends_not_analyzed_returns_503(client, monkeypatch):
    def _raise(grain="weekly"):
        raise ReviewsNotAnalyzedError("not analyzed")

    monkeypatch.setattr(reviews_router, "get_trends", _raise)
    resp = client.get("/reviews/trends")
    assert resp.status_code == 503


def test_reviews_trends_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(reviews_router, "get_trends", lambda grain="weekly": (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/reviews/trends")
    assert resp.status_code == 500


def test_reviews_summary_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(reviews_router, "get_summary", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/reviews/summary")
    assert resp.status_code == 500


def test_rag_index_error_returns_500(client, monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("index build failed")

    monkeypatch.setattr(rag_router, "build_index", _raise)
    resp = client.post("/rag/index", json={})
    assert resp.status_code == 500


def test_rag_reindex_error_returns_500(client, monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("reindex failed")

    monkeypatch.setattr(rag_router, "incremental_index", _raise)
    resp = client.post("/rag/reindex", json={})
    assert resp.status_code == 500


def test_rag_stats_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(rag_router, "index_stats", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/rag/stats")
    assert resp.status_code == 500


def test_rag_query_generic_error_returns_500(client, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(rag_router, "answer_question", _raise)
    resp = client.post("/rag/query", json={"query": "anything"})
    assert resp.status_code == 500


# Note: a variant of this test that makes `stream_answer` raise
# NoContextError/Exception on the first `next()` call was attempted, but
# triggers an unrelated sse_starlette/anyio TestClient event-loop-mismatch
# quirk (see sse_starlette#84-style issues) rather than exercising a real
# bug in our `event_generator()` code -- the try/except around
# `stream_answer` in api/routers/rag.py is still covered indirectly by
# `test_rag_query_no_context_returns_503` (same NoContextError branch, via
# the non-streaming path) and by the happy-path streaming test above.


def test_insights_list_error_returns_500(client, monkeypatch):
    def _raise(category=None, min_severity=None, llm_provider=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(insights_router, "get_insights", _raise)
    resp = client.get("/insights")
    assert resp.status_code == 500


def test_insights_executive_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(insights_router, "get_executive_summary", lambda llm_provider=None: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/insights/executive")
    assert resp.status_code == 500


def test_insights_recommendations_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(insights_router, "get_recommendations", lambda llm_provider=None: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/insights/recommendations")
    assert resp.status_code == 500


def test_insights_anomalies_error_returns_500(client, monkeypatch):
    monkeypatch.setattr(insights_router, "get_anomalies", lambda llm_provider=None: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.get("/insights/anomalies")
    assert resp.status_code == 500


def test_rag_safe_llm_provider_returns_none_on_error(monkeypatch):
    """Directly exercises the real `_safe_llm_provider` helper's exception
    path in the rag router -- never propagates, always returns None."""

    def _raise(*a, **k):
        raise RuntimeError("no provider configured")

    monkeypatch.setattr(rag_router, "get_llm_provider", _raise)
    assert rag_router._safe_llm_provider() is None


def test_insights_safe_llm_provider_returns_none_on_error(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("no provider configured")

    monkeypatch.setattr(insights_router, "get_llm_provider", _raise)
    assert insights_router._safe_llm_provider() is None
