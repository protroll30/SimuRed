"""Detect Gemini / LiteLLM error strings stored in dataset ``output`` fields."""

from __future__ import annotations

_BAD_LLM_MARKERS = (
    "resource_exhausted",
    "quota exceeded",
    "generativelanguage.googleapis.com",
    "generate_requests_per_model",
    "error: ",  # llm_client f"Error: {e}"
    "ratelimiterror",
    "geminiexception",
    "you exceeded your current quota",
    "exceeded its spending cap",
    "ai.google.dev/gemini-api/docs/rate-limits",
)


def is_bad_llm_output(text: str | None) -> bool:
    if text is None or not str(text).strip():
        return True
    low = str(text).lower()
    return any(m in low for m in _BAD_LLM_MARKERS)


def is_hard_quota_error(text: str | None) -> bool:
    """Daily/model caps; retrying after a few seconds will not help."""
    if text is None or not str(text).strip():
        return False
    low = str(text).lower()
    return (
        "resource_exhausted" in low
        or "quota exceeded" in low
        or "generate_requests_per_model" in low
        or "exceeded its spending cap" in low
    )
