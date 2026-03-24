"""Detect Gemini / LiteLLM error strings stored in dataset ``output`` fields."""

from __future__ import annotations

_BAD_LLM_MARKERS = (
    "resource_exhausted",
    "quota exceeded",
    "generativelanguage.googleapis.com",
    "generate_requests_per_model",
    "error: ",  # llm_client f"Error: {e}"
)


def is_bad_llm_output(text: str | None) -> bool:
    if text is None or not str(text).strip():
        return True
    low = str(text).lower()
    return any(m in low for m in _BAD_LLM_MARKERS)
