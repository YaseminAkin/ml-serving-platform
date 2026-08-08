"""Tests for gateway.models.model.predict."""

from gateway.models.model import predict


def test_positive_sentiment_returns_positive_label() -> None:
    result = predict("I absolutely love this product, it works great!")

    assert result["label"] == "POSITIVE"
    assert result["score"] > 0.5


def test_negative_sentiment_returns_negative_label() -> None:
    result = predict("This is terrible, I hate how badly it performs.")

    assert result["label"] == "NEGATIVE"
    assert result["score"] > 0.5


def test_result_contains_exactly_required_keys() -> None:
    result = predict("The weather is fine today.")

    assert set(result.keys()) == {"label", "score", "latency_ms"}


def test_score_is_float_between_zero_and_one() -> None:
    result = predict("The weather is fine today.")

    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0


def test_latency_ms_is_positive_float() -> None:
    result = predict("The weather is fine today.")

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] > 0
