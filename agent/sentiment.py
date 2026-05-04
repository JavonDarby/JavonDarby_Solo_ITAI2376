"""
Sentiment analysis module using the ProsusAI/FinBERT deep learning model.

FinBERT is a BERT-based transformer fine-tuned on financial news corpora. It
classifies text into three categories -- Positive, Negative, and Neutral --
each with a confidence score. This module loads the model once at import time
and exposes a single function for headline-level sentiment analysis.

This module handles classification only. All reasoning, risk assessment, and
natural-language generation are performed by the LLM agent brain, which is
a separate component.
"""

from transformers import pipeline

# Load FinBERT once at module level so the model weights stay in memory
# across multiple calls. First import triggers a download (~250 MB).
_finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def analyze_sentiment(headlines: list[str]) -> dict:
    """
    Run FinBERT sentiment classification on a list of financial headlines.

    Each headline is independently classified as Positive, Negative, or
    Neutral. The function returns the dominant label across all headlines,
    the average confidence score, and a per-label count breakdown.

    Args:
        headlines: A list of plain-text financial news headlines.

    Returns:
        A dict with the following keys:
            dominant_sentiment (str): The label with the highest count.
            average_confidence (float): Mean confidence across all headlines.
            breakdown (dict): Counts for each label (positive, negative, neutral).
            headlines_analyzed (int): Number of headlines that were processed.
    """
    # Run batch inference through the FinBERT pipeline
    results = _finbert(headlines)

    # Accumulate counts and confidence for each sentiment label
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    confidence_sum = 0.0

    for r in results:
        label = r["label"].lower()
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        confidence_sum += r["score"]

    # Identify the label with the highest count
    dominant = max(sentiment_counts, key=sentiment_counts.get)
    avg_confidence = confidence_sum / len(results)

    return {
        "dominant_sentiment": dominant.capitalize(),
        "average_confidence": round(avg_confidence, 4),
        "breakdown": sentiment_counts,
        "headlines_analyzed": len(headlines),
    }
