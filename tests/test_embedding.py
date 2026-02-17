from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import diskcache
import numpy as np
import pytest
from google.genai.errors import ClientError, ServerError
from google.genai.types import EmbedContentConfig

import embedding
from embedding import (
    _embed_with_retry,
    _normalize_embeddings,
    get_embeddings,
)


def _make_embedding(seed: int, dimension: int) -> MagicMock:
    """Create a mock embedding object with deterministic random values.

    Generates a full 3072-dim unit vector then truncates to the requested
    dimension, mimicking the API's truncation behavior. The result is
    intentionally not unit-norm so normalization tests are meaningful.
    """
    rng = np.random.default_rng(seed)
    full = rng.standard_normal(3072)
    full /= np.linalg.norm(full)
    emb = MagicMock()
    emb.values = full[:dimension].tolist()
    return emb


def _fake_embed(
    *,
    model: str,
    contents: list[str],
    config: EmbedContentConfig,
) -> MagicMock:
    """Side effect for mock embed_content that returns sized fake embeddings."""
    response = MagicMock()
    response.embeddings = [
        _make_embedding(i, config.output_dimensionality) for i in range(len(contents))
    ]
    return response


class TestNormalizeEmbeddings:
    def test_normalizes_to_unit_vectors(self) -> None:
        """Non-unit embedding objects are normalized to unit norm."""
        embeddings = [
            MagicMock(values=[3.0, 4.0]),
            MagicMock(values=[0.0, 5.0]),
            MagicMock(values=[6.0, 8.0]),
        ]
        result = _normalize_embeddings(embeddings)
        for vec in result:
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_direction(self) -> None:
        """Normalized vector points in the same direction as the original."""
        original = np.array([3.0, 4.0])
        embeddings = [MagicMock(values=original.tolist())]
        result = _normalize_embeddings(embeddings)
        expected = original / np.linalg.norm(original)
        np.testing.assert_allclose(result[0], expected, atol=1e-6)


class TestEmbedWithRetry:
    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def mock_sleep(self) -> Generator[MagicMock, None, None]:
        with patch("embedding.time.sleep") as sleep:
            yield sleep

    @pytest.mark.parametrize(
        ("error_cls", "code"),
        [
            (ClientError, 429),
            (ServerError, 503),
        ],
    )
    def test_retries_on_retryable_error(
        self,
        mock_client: MagicMock,
        mock_sleep: MagicMock,
        error_cls: type,
        code: int,
    ) -> None:
        """Retries on 429/503 and returns the successful response."""
        error = error_cls(code=code, response_json={})
        success = MagicMock()
        mock_client.models.embed_content.side_effect = [
            error,
            error,
            success,
        ]

        result = _embed_with_retry(mock_client, "model", ["a", "b"], dimension=768)

        assert result is success
        assert mock_sleep.call_args_list == [call(1), call(2)]

    def test_non_retryable_error_raises_immediately(
        self,
        mock_client: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """A 400 error propagates immediately with no retry."""
        error = ClientError(code=400, response_json={})
        mock_client.models.embed_content.side_effect = error

        with pytest.raises(ClientError):
            _embed_with_retry(mock_client, "model", ["a"], dimension=768)

        mock_sleep.assert_not_called()

    def test_sleep_capped_at_max(
        self,
        mock_client: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Sleep durations follow 1, 2, 4, 8, 16, 32, 60, 60."""
        error = ClientError(code=429, response_json={})
        success = MagicMock()
        mock_client.models.embed_content.side_effect = [error] * 8 + [success]

        result = _embed_with_retry(mock_client, "model", ["a"], dimension=768)

        assert result is success
        assert mock_sleep.call_args_list == [
            call(1),
            call(2),
            call(4),
            call(8),
            call(16),
            call(32),
            call(60),
            call(60),
        ]

    def test_valid_dimension_passed_to_config(
        self,
        mock_client: MagicMock,
    ) -> None:
        """A valid dimension is forwarded in EmbedContentConfig."""
        _embed_with_retry(mock_client, "model", ["a"], dimension=256)

        mock_client.models.embed_content.assert_called_once_with(
            model="model",
            contents=["a"],
            config=EmbedContentConfig(output_dimensionality=256),
        )

    @pytest.mark.parametrize("dimension", [100, 500, 1024])
    def test_invalid_dimension_raises(
        self,
        mock_client: MagicMock,
        dimension: int,
    ) -> None:
        """Invalid dimensions raise ValueError before calling the API."""
        with pytest.raises(ValueError, match="dimension"):
            _embed_with_retry(
                mock_client,
                "model",
                ["a"],
                dimension=dimension,
            )

        mock_client.models.embed_content.assert_not_called()


class TestGetEmbeddings:
    @pytest.fixture()
    def cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Generator[diskcache.Cache, None, None]:
        c = diskcache.Cache(tmp_path / "cache")
        monkeypatch.setattr(embedding, "CACHE", c)
        yield c
        c.close()

    @pytest.fixture()
    def mock_genai_client(self) -> Generator[MagicMock, None, None]:
        """Patch genai.Client; embed_content returns sized fake embeddings."""
        with patch("embedding.genai.Client") as mock_cls:
            client = mock_cls.return_value
            client.models.embed_content.side_effect = _fake_embed
            yield mock_cls

    def test_uncached_texts_call_api(
        self,
        cache: diskcache.Cache,
        mock_genai_client: MagicMock,
    ) -> None:
        """All uncached texts are fetched from the API."""
        result = get_embeddings(["dog", "cat"], dimension=768)

        assert len(result) == 2
        client = mock_genai_client.return_value
        client.models.embed_content.assert_called_once()

    def test_cached_texts_skip_api(
        self,
        cache: diskcache.Cache,
        mock_genai_client: MagicMock,
    ) -> None:
        """Fully cached texts bypass the API entirely."""
        cache.set(("dog", 768), [1.0, 0.0])
        cache.set(("cat", 768), [0.0, 1.0])

        result = get_embeddings(["dog", "cat"], dimension=768)

        client = mock_genai_client.return_value
        client.models.embed_content.assert_not_called()
        assert result == [[1.0, 0.0], [0.0, 1.0]]

    def test_mixed_fetches_only_uncached(
        self,
        cache: diskcache.Cache,
        mock_genai_client: MagicMock,
    ) -> None:
        """Only uncached texts are sent to the API."""
        cache.set(("dog", 768), [1.0, 0.0])

        result = get_embeddings(["dog", "cat"], dimension=768)

        client = mock_genai_client.return_value
        call_args = client.models.embed_content.call_args
        assert call_args.kwargs["contents"] == ["cat"]
        assert len(result) == 2
        assert result[0] == [1.0, 0.0]

    def test_writes_to_cache(
        self,
        cache: diskcache.Cache,
        mock_genai_client: MagicMock,
    ) -> None:
        """Fetched embeddings are written back to the cache."""
        get_embeddings(["dog"], dimension=768)

        cached = cache.get(("dog", 768))
        assert cached is not None
        assert np.linalg.norm(cached) == pytest.approx(1.0, abs=1e-6)

    def test_results_are_normalized(
        self,
        cache: diskcache.Cache,
        mock_genai_client: MagicMock,
    ) -> None:
        """Fetched embeddings are normalized to unit norm."""
        result = get_embeddings(["dog", "cat"], dimension=768)

        for vec in result:
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("dimension", [100, 500, 1024])
    def test_invalid_dimension_raises(
        self,
        cache: diskcache.Cache,
        dimension: int,
    ) -> None:
        """Invalid dimensions raise ValueError before any work."""
        with pytest.raises(ValueError, match="dimension"):
            get_embeddings(["dog"], dimension=dimension)

    def test_batches_large_requests(
        self,
        cache: diskcache.Cache,
    ) -> None:
        """Requests with >100 texts are split into batches of 100."""

        def _reject_oversized_batch(
            *,
            model: str,
            contents: list[str],
            config: EmbedContentConfig,
        ) -> MagicMock:
            if len(contents) > 100:
                raise ClientError(
                    code=400,
                    response_json={
                        "error": {
                            "message": "contents must have 100 or fewer items",
                            "status": "INVALID_ARGUMENT",
                        }
                    },
                )
            return _fake_embed(
                model=model,
                contents=contents,
                config=config,
            )

        with patch("embedding.genai.Client") as mock_cls:
            client = mock_cls.return_value
            client.models.embed_content.side_effect = _reject_oversized_batch

            texts = [f"concept_{i}" for i in range(150)]
            result = get_embeddings(texts, dimension=768)

            calls = client.models.embed_content.call_args_list
            assert len(calls) == 2
            assert len(calls[0].kwargs["contents"]) == 100
            assert len(calls[1].kwargs["contents"]) == 50
            assert len(result) == 150
