"""Guest review analysis pipeline.

Stages: sentiment classification, emotion detection, complaint-category
detection, unsupervised topic extraction (LDA — chosen over BERTopic to
avoid a heavy transformer-based topic dependency), keyword extraction
(TF-IDF, with KeyBERT as an optional enrichment when available), LLM-based
review summarization (batched/map-reduce), guest satisfaction score (CSAT,
0-100) at hotel/branch/daily/weekly/monthly grain, and trend detection via
rolling-window comparison.

Sentiment/emotion/complaint detection here are lexicon-based (no heavy NLP
model download required) since the synthetic reviews are template-generated
from known phrase banks — this keeps the pipeline fast, dependency-light,
and fully deterministic for testing, while remaining swappable for a
transformer classifier later without changing the pipeline's public API.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from genai.reviews.synthetic_reviews import COMPLAINT_CATEGORIES, _NEGATIVE_PHRASES, _POSITIVE_PHRASES
from src.utils.logging import get_logger

logger = get_logger(__name__)

_POSITIVE_WORDS = {
    "excellent", "fantastic", "great", "friendly", "clean", "spotless", "delicious",
    "perfect", "wonderful", "helpful", "reasonable", "quiet", "peaceful", "reliable",
    "modern", "recommend", "exceeded", "warm", "responsive", "professional", "easy",
    "secure", "beautifully", "loved",
}
_NEGATIVE_WORDS = {
    "not clean", "mold", "rude", "unhelpful", "overpriced", "bland", "expensive",
    "overcharged", "far", "noisy", "inconvenient", "broken", "faulty", "leaked",
    "disconnecting", "slow", "unusable", "disappointing", "no parking", "surprise",
    "full", "hard", "terrible", "worse",
}

_STOPWORDS = {
    "the", "was", "were", "we", "our", "and", "a", "to", "of", "for", "with", "in", "at",
    "is", "it", "on", "this", "that", "very", "so", "but", "had", "have", "has",
}


def _words(phrase: str) -> set[str]:
    return set(re.findall(r"[a-z]+", phrase.lower())) - _STOPWORDS


def _phrase_set_words(phrases: list[str]) -> set[str]:
    return set().union(*(_words(p) for p in phrases)) if phrases else set()


# Negative-only vocabulary per category: words that appear in the negative
# phrase bank but NOT in the positive phrase bank for the same category, so
# shared nouns (e.g. "food", "wifi", "staff") don't trigger a false
# complaint on purely positive text.
_NEGATIVE_KEYWORDS = {
    category: _phrase_set_words(_NEGATIVE_PHRASES[category]) - _phrase_set_words(_POSITIVE_PHRASES[category])
    for category in COMPLAINT_CATEGORIES
}

_CATEGORY_KEYWORDS = {
    category: _NEGATIVE_KEYWORDS[category] | set().union(*(_words(p) for p in _POSITIVE_PHRASES[category]))
    for category in COMPLAINT_CATEGORIES
}


def classify_sentiment(text: str) -> str:
    """Lexicon-based sentiment classifier. English-only heuristic; non-English
    text (Sinhala/Tamil) falls back to 'neutral' since the lexicon is
    English, matching the pipeline's documented scope."""
    lowered = text.lower()
    pos_hits = sum(1 for w in _POSITIVE_WORDS if w in lowered)
    neg_hits = sum(1 for w in _NEGATIVE_WORDS if w in lowered)
    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


def detect_emotion(text: str, sentiment: str) -> str:
    lowered = text.lower()
    if sentiment == "negative":
        if any(w in lowered for w in ("rude", "broken", "faulty", "mold")):
            return "angry"
        return "frustrated"
    if sentiment == "positive":
        if any(w in lowered for w in ("excellent", "exceeded", "loved", "fantastic")):
            return "excited"
        return "happy"
    return "satisfied"


def detect_complaints(text: str) -> list[str]:
    """Multi-label complaint-category detection using per-category keyword
    overlap. Only fires for text containing negative-toned keywords."""
    words = set(re.findall(r"[a-z]+", text.lower())) - _STOPWORDS
    hits = []
    for category, neg_keywords in _NEGATIVE_KEYWORDS.items():
        if words & neg_keywords:
            hits.append(category)
    return hits


def extract_keywords_tfidf(texts: list[str], top_n: int = 10) -> list[str]:
    if not texts:
        return []
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(scores)[::-1][:top_n]
    return terms[order].tolist()


def extract_keywords_keybert(texts: list[str], top_n: int = 10) -> list[str]:
    """Optional KeyBERT-based enrichment. Falls back to TF-IDF keywords if
    the `keybert` package or its embedding backend is unavailable (e.g. no
    internet access to download the sentence-transformer model)."""
    try:
        from keybert import KeyBERT

        joined = " ".join(texts)[:20000]
        model = KeyBERT()
        pairs = model.extract_keywords(joined, top_n=top_n, stop_words="english")
        return [kw for kw, _ in pairs]
    except Exception as exc:
        logger.warning("KeyBERT unavailable, falling back to TF-IDF keywords: %s", exc)
        return extract_keywords_tfidf(texts, top_n=top_n)


def extract_topics(texts: list[str], n_topics: int = 5, n_words: int = 6) -> list[dict]:
    """Unsupervised topic extraction via sklearn LDA (lightweight vs.
    BERTopic, avoids a heavy dependency chain)."""
    if len(texts) < n_topics:
        return []
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return []
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(matrix)
    terms = np.array(vectorizer.get_feature_names_out())
    topics = []
    for idx, component in enumerate(lda.components_):
        top_indices = component.argsort()[::-1][:n_words]
        topics.append({"topic_id": idx, "keywords": terms[top_indices].tolist()})
    return topics


def summarize_reviews_rule_based(texts: list[str], sentiments: list[str]) -> str:
    """Deterministic, dependency-free fallback summary used when no LLM
    provider is available/configured."""
    n = len(texts)
    if n == 0:
        return "No reviews available to summarize."
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    return (
        f"Analyzed {n} reviews: {pos} positive ({pos / n:.0%}), "
        f"{neu} neutral ({neu / n:.0%}), {neg} negative ({neg / n:.0%})."
    )


def summarize_reviews(texts: list[str], sentiments: list[str], llm_provider=None) -> str:
    """LLM-based summarization with map-reduce batching for large volumes.
    Falls back to a rule-based summary if no LLM provider is supplied or the
    provider is not live-available (no API key / unreachable)."""
    if llm_provider is None or not llm_provider.is_available():
        return summarize_reviews_rule_based(texts, sentiments)

    from genai.prompts.loader import load_prompt

    system_prompt = load_prompt("review_analyst").text
    batch_size = 200
    batch_summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        joined = "\n".join(f"- {t}" for t in batch[:50])
        prompt = f"Summarize the key themes in these guest reviews:\n{joined}"
        try:
            result = llm_provider.generate(prompt, system=system_prompt)
            batch_summaries.append(result.text)
        except Exception as exc:
            logger.warning("LLM summarization batch failed, using rule-based fallback: %s", exc)
            return summarize_reviews_rule_based(texts, sentiments)

    if len(batch_summaries) == 1:
        return batch_summaries[0]

    reduce_prompt = "Combine these partial summaries into one cohesive summary:\n" + "\n".join(batch_summaries)
    try:
        final = llm_provider.generate(reduce_prompt, system=system_prompt)
        return final.text
    except Exception as exc:
        logger.warning("LLM reduce step failed, concatenating batch summaries: %s", exc)
        return " ".join(batch_summaries)


def compute_csat(ratings: pd.Series) -> float:
    """0-100 satisfaction score derived from 1-5 star ratings."""
    if ratings.empty:
        return 0.0
    return round(float((ratings.mean() - 1) / 4 * 100), 2)


def aggregate_csat(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Aggregate CSAT at hotel/branch/daily/weekly/monthly grain.

    `grain` in {"hotel", "daily", "weekly", "monthly"}. `df` must have
    `hotel_id`, `rating`, `date` (as datetime-parseable string) columns.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])

    if grain == "hotel":
        grouped = work.groupby("hotel_id")["rating"].apply(compute_csat).reset_index(name="csat")
        return grouped

    if grain == "daily":
        key = work["date"].dt.date.astype(str)
    elif grain == "weekly":
        key = work["date"].dt.to_period("W").astype(str)
    elif grain == "monthly":
        key = work["date"].dt.to_period("M").astype(str)
    else:
        raise ValueError(f"Unknown grain: {grain}")

    work["_period"] = key
    grouped = work.groupby(["hotel_id", "_period"])["rating"].apply(compute_csat).reset_index(name="csat")
    grouped = grouped.rename(columns={"_period": "period"})
    return grouped


def detect_trend(csat_series: pd.Series, window: int = 3) -> str:
    """Rolling-window comparison: compares the mean of the most recent
    `window` periods to the mean of the `window` periods before that."""
    values = csat_series.dropna().to_numpy()
    if len(values) < window * 2:
        return "stable"
    recent = values[-window:].mean()
    prior = values[-2 * window : -window].mean()
    delta = recent - prior
    if delta > 2.0:
        return "improving"
    if delta < -2.0:
        return "declining"
    return "stable"


def run_pipeline(df: pd.DataFrame, llm_provider=None) -> dict:
    """Run the full analysis pipeline over a reviews DataFrame with columns
    `review, rating, date, hotel_id, language`. Returns a dict of DataFrames
    /values ready for persistence by `service.py`."""
    work = df.copy()
    work["sentiment"] = work["review"].apply(classify_sentiment)
    work["emotion"] = work.apply(lambda r: detect_emotion(r["review"], r["sentiment"]), axis=1)
    work["complaints"] = work["review"].apply(detect_complaints)

    english_texts = work.loc[work["language"] == "en", "review"].tolist()
    keywords = extract_keywords_tfidf(english_texts, top_n=15)
    topics = extract_topics(english_texts, n_topics=5)
    summary = summarize_reviews(english_texts, work["sentiment"].tolist(), llm_provider=llm_provider)

    complaints_exploded = work.explode("complaints").dropna(subset=["complaints"])
    complaint_counts = (
        complaints_exploded.groupby("complaints").size().reset_index(name="count").sort_values("count", ascending=False)
        if not complaints_exploded.empty
        else pd.DataFrame(columns=["complaints", "count"])
    )
    complaint_counts = complaint_counts.rename(columns={"complaints": "category"})

    csat_daily = aggregate_csat(work, "daily")
    csat_weekly = aggregate_csat(work, "weekly")
    csat_monthly = aggregate_csat(work, "monthly")
    csat_hotel = aggregate_csat(work, "hotel")

    trend_by_hotel = {}
    for hotel_id, group in csat_weekly.groupby("hotel_id"):
        trend_by_hotel[hotel_id] = detect_trend(group.sort_values("period")["csat"])

    return {
        "annotated": work,
        "keywords": keywords,
        "topics": topics,
        "summary": summary,
        "complaint_counts": complaint_counts,
        "csat_daily": csat_daily,
        "csat_weekly": csat_weekly,
        "csat_monthly": csat_monthly,
        "csat_hotel": csat_hotel,
        "trend_by_hotel": trend_by_hotel,
    }
