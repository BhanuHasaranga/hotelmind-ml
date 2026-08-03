import pandas as pd
import pytest

from genai.reviews.pipeline import (
    aggregate_csat,
    classify_sentiment,
    compute_csat,
    detect_complaints,
    detect_emotion,
    detect_trend,
    extract_keywords_tfidf,
    extract_topics,
    run_pipeline,
    summarize_reviews,
    summarize_reviews_rule_based,
)
from genai.reviews.synthetic_reviews import generate_reviews


def test_generate_reviews_schema_and_size():
    df = generate_reviews(n=500, seed=1)
    assert len(df) >= 490  # allow for rounding of language splits
    assert {"review", "rating", "date", "hotel_id", "language"}.issubset(df.columns)
    assert df["rating"].between(1, 5).all()
    assert set(df["language"].unique()).issubset({"en", "si", "ta"})


def test_generate_reviews_deterministic_with_seed():
    df1 = generate_reviews(n=200, seed=7)
    df2 = generate_reviews(n=200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_classify_sentiment_positive():
    assert classify_sentiment("The room was spotless and staff were fantastic and helpful") == "positive"


def test_classify_sentiment_negative():
    assert classify_sentiment("The room was not clean and staff were rude") == "negative"


def test_detect_emotion_maps_from_sentiment():
    assert detect_emotion("Excellent and fantastic stay", "positive") in {"happy", "excited"}
    assert detect_emotion("Broken air conditioning, faulty lock", "negative") == "angry"


def test_detect_complaints_multi_label():
    text = "The room was not clean and the WiFi kept disconnecting"
    complaints = detect_complaints(text)
    assert "cleanliness" in complaints
    assert "wifi" in complaints


def test_detect_complaints_no_hits_on_unambiguous_positive_text():
    # Avoid words like "staff"/"everything" that also appear in negative
    # phrase banks for other categories (lexicon overlap is expected for a
    # keyword-based classifier); use vocabulary unique to positive phrasing.
    complaints = detect_complaints("Breakfast buffet was excellent with fresh options, WiFi fast and reliable")
    assert complaints == []


def test_extract_keywords_tfidf_returns_terms():
    texts = ["clean room great staff", "clean room great location", "dirty room bad staff"]
    keywords = extract_keywords_tfidf(texts, top_n=5)
    assert len(keywords) > 0


def test_extract_keywords_tfidf_empty_input():
    assert extract_keywords_tfidf([]) == []


def test_extract_topics_returns_topics_for_enough_docs():
    texts = [f"clean room staff friendly location {i}" for i in range(20)]
    topics = extract_topics(texts, n_topics=3)
    assert len(topics) == 3
    for topic in topics:
        assert "keywords" in topic


def test_extract_topics_too_few_docs_returns_empty():
    assert extract_topics(["one review"], n_topics=5) == []


def test_summarize_reviews_rule_based():
    summary = summarize_reviews_rule_based(["a", "b", "c"], ["positive", "negative", "neutral"])
    assert "3 reviews" in summary


def test_summarize_reviews_rule_based_empty():
    assert summarize_reviews_rule_based([], []) == "No reviews available to summarize."


def test_summarize_reviews_falls_back_without_llm():
    summary = summarize_reviews(["good stay"], ["positive"], llm_provider=None)
    assert "1 reviews" in summary


def test_compute_csat():
    ratings = pd.Series([5, 5, 5, 5, 5])
    assert compute_csat(ratings) == 100.0
    ratings = pd.Series([1, 1, 1])
    assert compute_csat(ratings) == 0.0


def test_compute_csat_empty():
    assert compute_csat(pd.Series([], dtype=float)) == 0.0


def test_aggregate_csat_hotel_grain():
    df = pd.DataFrame(
        {
            "hotel_id": ["H1", "H1", "H2"],
            "rating": [5, 4, 2],
            "date": ["2026-01-01", "2026-01-02", "2026-01-01"],
        }
    )
    result = aggregate_csat(df, "hotel")
    assert set(result["hotel_id"]) == {"H1", "H2"}


def test_aggregate_csat_invalid_grain():
    df = pd.DataFrame({"hotel_id": ["H1"], "rating": [5], "date": ["2026-01-01"]})
    with pytest.raises(ValueError):
        aggregate_csat(df, "yearly")


def test_detect_trend_stable_with_insufficient_data():
    assert detect_trend(pd.Series([50, 51])) == "stable"


def test_detect_trend_improving():
    values = pd.Series([50, 51, 50, 70, 71, 72])
    assert detect_trend(values, window=3) == "improving"


def test_detect_trend_declining():
    values = pd.Series([80, 81, 80, 50, 49, 48])
    assert detect_trend(values, window=3) == "declining"


def test_run_pipeline_end_to_end():
    df = generate_reviews(n=300, seed=3)
    result = run_pipeline(df)
    assert "annotated" in result
    assert "sentiment" in result["annotated"].columns
    assert "keywords" in result
    assert "csat_daily" in result
    assert "trend_by_hotel" in result
