"""Prediction scoring methods used by the eval worker."""

import json
import logging
import os

import httpx
from dotenv import load_dotenv

from db.models import Prediction

logger = logging.getLogger(__name__)

load_dotenv()

CONFIDENCE_METHOD = "confidence"
LLM_JUDGE_METHOD = "llm_judge"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini")

_JUDGE_PROMPT = (
    "You are grading a sentiment classifier. Given a piece of text and the "
    'label a model assigned it, respond with strict JSON {{"score": x}} '
    "where x is a number between 0 and 1 — 1 means the label is clearly "
    "correct, 0 means clearly wrong.\n\n"
    "Text: {text}\n"
    "Predicted label: {label}"
)


def score_confidence(prediction: Prediction) -> float:
    """Use the model's own confidence score as the primary eval metric."""
    return prediction.output["score"]


async def score_llm_judge(prediction: Prediction) -> float | None:
    """Ask an LLM to judge whether the predicted label fits the input text.

    Returns None (skip) if OPENAI_API_KEY isn't set, or if the call fails —
    eval scoring is best-effort and must never break the worker loop.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt = _JUDGE_PROMPT.format(
        text=prediction.text, label=prediction.output["label"]
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                OPENAI_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            score = float(json.loads(content)["score"])
    except Exception:
        logger.exception("LLM-judge scoring failed for prediction id=%s", prediction.id)
        return None

    return max(0.0, min(1.0, score))
