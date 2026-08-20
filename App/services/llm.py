"""
services/llm.py — Groq LLM wrapper with retry + backoff

Retries on:
  - 429 Rate limit
  - 500/502/503/504 server errors
  - Connection errors / timeouts
"""

import logging
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

_GROQ_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


def _is_groq_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _GROQ_RETRYABLE):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(_is_groq_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_response(prompt: str) -> str:
    """
    Generate a response from Groq LLM.
    Retries up to 4 times on rate-limit or server errors with exponential backoff.
    """
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""

    except _GROQ_RETRYABLE:
        raise  # tenacity will catch and retry

    except APIStatusError as e:
        if e.status_code in (500, 502, 503, 504):
            raise  # retryable
        return f"Groq API Error ({e.status_code}): {e.message}"

    except Exception as e:
        return f"Groq Error: {e}"