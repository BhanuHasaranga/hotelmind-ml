"""Tests for genai/reviews/service.py -- the precompute-then-query pattern."""

from __future__ import annotations

import pytest

from genai.config.genai_settings import genai_settings
from genai.reviews import service as reviews_service
from genai.reviews.service import ReviewsNotAnalyzedError


@pytest.fixture
def isolated_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(type(genai_settings), "REVIEWS_OUTPUT_DIR", "unused", raising=False)
    monkeypatch.setattr(genai_settings, "REVIEWS_OUTPUT_DIR", str(tmp_path / "reviews_out"))
    return tmp_path / "reviews_out"


def test_get_summary_before_analyze_raises(isolated_output_dir):
    with pytest.raises(ReviewsNotAnalyzedError):
        reviews_service.get_summary()


def test_get_topics_before_analyze_raises(isolated_output_dir):
    with pytest.raises(ReviewsNotAnalyzedError):
        reviews_service.get_topics()


def test_get_complaints_before_analyze_raises(isolated_output_dir):
    with pytest.raises(ReviewsNotAnalyzedError):
        reviews_service.get_complaints()


def test_get_trends_before_analyze_raises(isolated_output_dir):
    with pytest.raises(ReviewsNotAnalyzedError):
        reviews_service.get_trends()


def test_get_trends_unknown_grain_raises_value_error(isolated_output_dir):
    with pytest.raises(ValueError):
        reviews_service.get_trends(grain="yearly")


def test_analyze_reviews_persists_and_get_endpoints_read_back(isolated_output_dir):
    manifest = reviews_service.analyze_reviews(limit=300)
    assert manifest["reviews_analyzed"] == 300
    assert isinstance(manifest["keywords"], list)

    summary = reviews_service.get_summary()
    assert "summary" in summary
    assert "csat_by_hotel" in summary

    topics = reviews_service.get_topics()
    assert "topics" in topics

    complaints = reviews_service.get_complaints()
    assert "complaints" in complaints

    trends = reviews_service.get_trends(grain="daily")
    assert trends["grain"] == "daily"
    assert "series" in trends
