import time
from pathlib import Path

import numpy as np
from diskcache import Cache
from google import genai
from google.genai.errors import APIError
from google.genai.types import (
    ContentEmbedding,
    EmbedContentConfig,
    EmbedContentResponse,
)

MODEL = "gemini-embedding-001"
BATCH_SIZE = 100
RETRYABLE_STATUS_CODES = {429, 503}
MAX_SLEEP = 60

# Values explicitly mentioned in
# https://ai.google.dev/gemini-api/docs/embeddings
VALID_DIMENSIONALITIES = {128, 256, 512, 768, 1536, 2048, 3072}

CACHE = Cache(Path(__file__).resolve().parent / ".cache")


def _validate_dimensionality(dimensionality: int) -> None:
    """Raise ValueError if dimensionality is not a supported value."""
    if dimensionality not in VALID_DIMENSIONALITIES:
        sorted_dims = sorted(VALID_DIMENSIONALITIES)
        raise ValueError(
            f"Invalid dimensionality {dimensionality}. Must be one of {sorted_dims}."
        )


def _normalize_embeddings(
    embeddings: list[ContentEmbedding],
) -> list[list[float]]:
    """Normalize embedding objects to unit vectors."""
    values = np.array([e.values for e in embeddings])
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / norms).tolist()


def _embed_with_retry(
    client: genai.Client,
    model: str,
    contents: list[str],
    *,
    dimensionality: int,
) -> EmbedContentResponse:
    """Call embed_content with exponential backoff on retryable errors."""
    _validate_dimensionality(dimensionality)
    attempt = 0
    while True:
        try:
            return client.models.embed_content(
                model=model,
                contents=contents,
                config=EmbedContentConfig(output_dimensionality=dimensionality),
            )
        except APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES:
                raise
            sleep_time = min(2**attempt, MAX_SLEEP)
            time.sleep(sleep_time)
            attempt += 1


def get_embeddings(
    texts: list[str],
    *,
    dimensionality: int,
) -> list[list[float]]:
    """Get embeddings for texts, using cache when possible.

    Checks the cache for each text. Uncached texts are fetched from the
    Gemini API in batches, normalized, and written back to the cache.
    """
    _validate_dimensionality(dimensionality)

    results: dict[int, list[float]] = {}
    uncached_indices: list[int] = []

    for i, text in enumerate(texts):
        cached = CACHE.get((text, dimensionality))
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)

    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]
        client = genai.Client()
        all_embeddings: list[ContentEmbedding] = []
        for batch_start in range(0, len(uncached_texts), BATCH_SIZE):
            batch = uncached_texts[batch_start : batch_start + BATCH_SIZE]
            response = _embed_with_retry(
                client, MODEL, batch, dimensionality=dimensionality
            )
            assert response.embeddings is not None, "API returned no embeddings."
            all_embeddings.extend(response.embeddings)

        normalized = _normalize_embeddings(all_embeddings)
        for idx, embedding in zip(uncached_indices, normalized):
            results[idx] = embedding
            CACHE.set((texts[idx], dimensionality), embedding)

    return [results[i] for i in range(len(texts))]
