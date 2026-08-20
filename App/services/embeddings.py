"""
services/embeddings.py — Jina AI Embeddings with Retry + Batching

Changes from original:
  - Batched embedding calls (100 chunks per request) to prevent OOM / timeout on large PDFs
  - Tenacity retry with exponential backoff on rate-limit (429) and server errors (5xx)
  - Retry on query embedding too (used during every chat request)
"""

import time
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
import logging

from config import JINA_API_KEY

logger = logging.getLogger(__name__)

API_URL = "https://api.jina.ai/v1/embeddings"
JINA_BATCH_SIZE = 100   # Jina max recommended per request
MAX_RETRIES = 4
MIN_WAIT = 2            # seconds
MAX_WAIT = 30           # seconds


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429 rate-limit or any 5xx server error."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=MIN_WAIT, max=MAX_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _embed_batch(texts: list[str], task: str) -> list:
    """
    Embed a single batch of texts (≤ JINA_BATCH_SIZE) with retry.
    """
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "model": "jina-embeddings-v3",
            "task": task,
            "input": texts,
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def _embed(texts: list[str], task: str) -> list:
    """
    Embed any number of texts, automatically batching into chunks of JINA_BATCH_SIZE.
    """
    all_embeddings = []
    total_batches = (len(texts) - 1) // JINA_BATCH_SIZE + 1

    for i in range(0, len(texts), JINA_BATCH_SIZE):
        batch = texts[i : i + JINA_BATCH_SIZE]
        batch_num = i // JINA_BATCH_SIZE + 1
        print(f"  [Jina] Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)...")
        embeddings = _embed_batch(batch, task)
        all_embeddings.extend(embeddings)

    return all_embeddings


def generate_document_embeddings(chunks: list[str]) -> list:
    """Embed document chunks for storage. Batched + retried."""
    print(f"Generating document embeddings using Jina AI ({len(chunks)} chunks)...")
    embeddings = _embed(chunks, "retrieval.passage")
    print(f"Embeddings returned: {len(embeddings)}")
    return embeddings


def generate_query_embedding(question: str) -> list:
    """Embed a single query for search. Retried."""
    print("Generating query embedding using Jina AI...")
    # _embed handles batching; single text returns list of 1 embedding
    return _embed([question], "retrieval.query")[0]
