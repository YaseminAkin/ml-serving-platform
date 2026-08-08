"""Sentiment analysis inference using a HuggingFace pipeline."""

import time

from transformers import pipeline

_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_classifier = pipeline("sentiment-analysis", model=_MODEL_NAME)


def predict(text: str) -> dict:
    """Run sentiment analysis on `text` and return label, score, and latency."""
    start = time.perf_counter()
    result = _classifier(text)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "label": result["label"],
        "score": result["score"],
        "latency_ms": latency_ms,
    }
