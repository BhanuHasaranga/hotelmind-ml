"""Synthetic guest review generator.

No real guest review data exists anywhere in this project. This generator
produces a large, seeded, clearly-labeled synthetic dataset (columns:
review, rating, date, hotel_id, language) so the review-analysis pipeline
(Module 1) has realistic input to operate on. Follows the precedent of
`src/pipelines/synthetic_data.py` (seeded, deterministic, documented as
fabricated) rather than pure noise: ratings are correlated with sentiment,
and reviews are template-generated from sentiment-conditioned phrase banks
per complaint category.

Output: `hotelmind-ml/data/raw/guest_reviews_synthetic.csv`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from genai.config.genai_settings import genai_settings
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

COMPLAINT_CATEGORIES = [
    "cleanliness",
    "food",
    "staff",
    "price",
    "location",
    "noise",
    "maintenance",
    "wifi",
    "parking",
    "other",
]

HOTEL_IDS = [f"HTL-{i:03d}" for i in range(1, 13)]

# Sentiment-conditioned phrase banks per complaint category. Positive phrases
# praise the category; negative phrases complain about it.
_POSITIVE_PHRASES = {
    "cleanliness": [
        "The room was spotless and the housekeeping team did a fantastic job.",
        "Everything felt clean and well maintained from the lobby to the bathroom.",
    ],
    "food": [
        "Breakfast buffet was excellent with a great variety of fresh options.",
        "The restaurant food was delicious and beautifully presented.",
    ],
    "staff": [
        "The staff were incredibly friendly and went out of their way to help us.",
        "Reception was warm, professional, and very responsive to our requests.",
    ],
    "price": [
        "Great value for money considering the quality of the room and service.",
        "Pricing was very reasonable for what we got, would book again.",
    ],
    "location": [
        "The location was perfect, walking distance to everything we wanted to see.",
        "Loved how central and convenient the hotel's location was.",
    ],
    "noise": [
        "Very quiet room, we slept great every night of our stay.",
        "Surprisingly peaceful for a hotel in the city center.",
    ],
    "maintenance": [
        "Everything in the room worked perfectly, no issues at all.",
        "The facilities were modern and well kept, nothing was broken.",
    ],
    "wifi": [
        "WiFi was fast and reliable throughout the entire property.",
        "Had no trouble working remotely thanks to the strong internet connection.",
    ],
    "parking": [
        "Parking was easy, secure, and included in the rate.",
        "Plenty of parking space available whenever we needed it.",
    ],
    "other": [
        "Overall a wonderful stay, we will definitely be back.",
        "Exceeded our expectations in every way, highly recommended.",
    ],
}

_NEGATIVE_PHRASES = {
    "cleanliness": [
        "The room was not clean when we checked in, dust everywhere.",
        "Bathroom had visible mold and the sheets looked used.",
    ],
    "food": [
        "Breakfast was repetitive and often cold by the time we got to it.",
        "The restaurant food was overpriced and quite bland.",
    ],
    "staff": [
        "Staff were rude and unhelpful whenever we asked for assistance.",
        "Reception took forever to respond and seemed indifferent to complaints.",
    ],
    "price": [
        "Way too expensive for the quality of the room we received.",
        "Felt overcharged compared to similar hotels nearby.",
    ],
    "location": [
        "The location was far from everything, hard to get around without a car.",
        "Noisy neighborhood and inconvenient distance from the city center.",
    ],
    "noise": [
        "Extremely noisy at night, could hear traffic and neighboring rooms clearly.",
        "Thin walls meant we barely slept due to the noise.",
    ],
    "maintenance": [
        "Air conditioning was broken for our entire stay and nobody fixed it.",
        "The shower leaked and the door lock was faulty.",
    ],
    "wifi": [
        "WiFi kept disconnecting and was unusable for video calls.",
        "Internet was painfully slow throughout the property.",
    ],
    "parking": [
        "No parking available, we had to park several blocks away.",
        "Parking fee was an unpleasant surprise and the lot was always full.",
    ],
    "other": [
        "Overall disappointing experience, would not stay here again.",
        "Fell well short of what we expected for this price range.",
    ],
}

_NEUTRAL_PHRASES = {
    category: [f"The {category} was okay, nothing special but nothing terrible either."]
    for category in COMPLAINT_CATEGORIES
}

_EMOTIONS_BY_SENTIMENT = {
    "positive": ["happy", "satisfied", "excited"],
    "neutral": ["satisfied"],
    "negative": ["angry", "frustrated"],
}

# Small sample of Sinhala/Tamil reviews, sentiment-labeled by construction.
_SINHALA_REVIEWS = [
    ("හෝටලයේ සේවාව ඉතා හොඳයි. නැවතත් පැමිණීමට බලාපොරොත්තු වෙනවා.", "positive"),
    ("කාමරය පිරිසිදු නැත. සේවකයින්ගේ ආචාරශීලීත්වය අඩුයි.", "negative"),
    ("හෝටලයේ ආහාර සාමාන්‍යයි, විශේෂ කිසිවක් නැත.", "neutral"),
    ("ස්ථානය ඉතා පහසුයි, නගරයට ආසන්නයි.", "positive"),
    ("වයි-ෆයි සම්බන්ධතාවය දුර්වලයි, වැඩ කිරීමට අපහසුයි.", "negative"),
]

_TAMIL_REVIEWS = [
    ("இந்த ஹோட்டலின் சேவை மிகவும் சிறப்பாக இருந்தது.", "positive"),
    ("அறை சுத்தமாக இல்லை, மோசமான அனுபவம்.", "negative"),
    ("உணவு சராசரியாக இருந்தது, ஒன்றும் சிறப்பில்லை.", "neutral"),
    ("இடம் மிகவும் வசதியானது, நகர மையத்திற்கு அருகில் உள்ளது.", "positive"),
    ("பணியாளர்கள் மிகவும் அன்பாக நடந்துகொண்டனர்.", "positive"),
]


def _pick_sentiment(rng: np.random.Generator) -> str:
    """Ratings are correlated with sentiment: sample rating first, derive
    sentiment from it, so the correlation is baked in rather than applied
    after the fact."""
    return rng.choice(["positive", "neutral", "negative"], p=[0.62, 0.18, 0.20])


def _rating_for_sentiment(sentiment: str, rng: np.random.Generator) -> int:
    if sentiment == "positive":
        return int(rng.choice([4, 5], p=[0.35, 0.65]))
    if sentiment == "neutral":
        return int(rng.choice([3, 4], p=[0.7, 0.3]))
    return int(rng.choice([1, 2], p=[0.55, 0.45]))


def _phrase_bank(sentiment: str) -> dict:
    return {
        "positive": _POSITIVE_PHRASES,
        "neutral": _NEUTRAL_PHRASES,
        "negative": _NEGATIVE_PHRASES,
    }[sentiment]


def _generate_english_review(rng: np.random.Generator) -> dict:
    sentiment = _pick_sentiment(rng)
    rating = _rating_for_sentiment(sentiment, rng)
    n_categories = int(rng.integers(1, 3))
    categories = list(rng.choice(COMPLAINT_CATEGORIES, size=n_categories, replace=False))
    bank = _phrase_bank(sentiment)
    sentences = [rng.choice(bank[c]) for c in categories]
    rng.shuffle(sentences)
    review_text = " ".join(sentences)
    emotion = rng.choice(_EMOTIONS_BY_SENTIMENT[sentiment])
    return {
        "review": review_text,
        "rating": rating,
        "language": "en",
        "_sentiment": sentiment,
        "_emotion": emotion,
        "_categories": ",".join(categories) if sentiment == "negative" else "",
    }


def _generate_regional_review(rng: np.random.Generator, bank: list[tuple[str, str]], language: str) -> dict:
    text, sentiment = bank[rng.integers(0, len(bank))]
    rating = _rating_for_sentiment(sentiment, rng)
    emotion = rng.choice(_EMOTIONS_BY_SENTIMENT[sentiment])
    return {
        "review": text,
        "rating": rating,
        "language": language,
        "_sentiment": sentiment,
        "_emotion": emotion,
        "_categories": "",
    }


def generate_reviews(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    n = n or genai_settings.SYNTHETIC_REVIEW_COUNT
    seed = seed if seed is not None else genai_settings.SYNTHETIC_REVIEW_SEED
    rng = np.random.default_rng(seed)

    # English-majority (92%), with smaller Sinhala (5%) / Tamil (3%) samples.
    n_si = max(1, int(n * 0.05))
    n_ta = max(1, int(n * 0.03))
    n_en = n - n_si - n_ta

    rows = []
    for _ in range(n_en):
        rows.append(_generate_english_review(rng))
    for _ in range(n_si):
        rows.append(_generate_regional_review(rng, _SINHALA_REVIEWS, "si"))
    for _ in range(n_ta):
        rows.append(_generate_regional_review(rng, _TAMIL_REVIEWS, "ta"))

    rng.shuffle(rows)

    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=730)
    dates = start_date + pd.to_timedelta(rng.integers(0, 730, size=len(rows)), unit="D")
    hotel_ids = rng.choice(HOTEL_IDS, size=len(rows))

    df = pd.DataFrame(rows)
    df["date"] = dates
    df["hotel_id"] = hotel_ids
    df["date"] = df["date"].dt.date.astype(str)

    df = df[["review", "rating", "date", "hotel_id", "language", "_sentiment", "_emotion", "_categories"]]
    return df.reset_index(drop=True)


def ensure_synthetic_reviews(force: bool = False) -> "pd.io.common.FilePath":
    path = settings.data_raw_dir_path / genai_settings.REVIEWS_RAW_FILENAME
    if path.exists() and not force:
        logger.info("Synthetic reviews already exist at %s", path)
        return path
    df = generate_reviews()
    settings.data_raw_dir_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Generated %d synthetic reviews -> %s", len(df), path)
    return path


if __name__ == "__main__":
    ensure_synthetic_reviews(force=True)
