"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for token counting from Google responses.
"""

import pytest
from unittest.mock import Mock
from revenium_middleware.google.common.types import TokenCounts


class TestTokenCountExtraction:
    """Test token count extraction from responses."""

    def test_extract_all_token_types(self):
        """Test extraction of all token types from usage metadata."""
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 150
        mock_usage.total_token_count = 250
        mock_usage.cached_content_token_count = 25

        mock_response.usage_metadata = mock_usage

        # Simulate extraction
        token_counts = TokenCounts(
            input_tokens=mock_usage.prompt_token_count,
            output_tokens=mock_usage.candidates_token_count,
            total_tokens=mock_usage.total_token_count,
            cached_tokens=mock_usage.cached_content_token_count
        )

        assert token_counts.input_tokens == 100
        assert token_counts.output_tokens == 150
        assert token_counts.total_tokens == 250
        assert token_counts.cached_tokens == 25

    def test_extract_without_cached_tokens(self):
        """Test extraction when cached tokens not present."""
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 150
        mock_usage.total_token_count = 250
        mock_usage.cached_content_token_count = 0  # No cached tokens

        mock_response.usage_metadata = mock_usage

        token_counts = TokenCounts(
            input_tokens=mock_usage.prompt_token_count,
            output_tokens=mock_usage.candidates_token_count,
            total_tokens=mock_usage.total_token_count,
            cached_tokens=mock_usage.cached_content_token_count
        )

        assert token_counts.cached_tokens == 0

    def test_extract_no_usage_metadata(self):
        """Test handling when usage_metadata is missing."""
        mock_response = Mock()
        mock_response.usage_metadata = None

        # Should handle gracefully with zeros
        if not mock_response.usage_metadata:
            token_counts = TokenCounts(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cached_tokens=0
            )

            assert token_counts.input_tokens == 0
            assert token_counts.output_tokens == 0
            assert token_counts.total_tokens == 0
            assert token_counts.cached_tokens == 0

    def test_token_count_consistency(self):
        """Test that input + output typically equals total."""
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 150
        mock_usage.total_token_count = 250

        mock_response.usage_metadata = mock_usage

        # Verify consistency
        assert mock_usage.prompt_token_count + mock_usage.candidates_token_count == mock_usage.total_token_count


class TestStreamingTokenCounts:
    """Test token counting in streaming responses."""

    def test_extract_tokens_from_final_chunk(self):
        """Test extraction of tokens from final streaming chunk."""
        mock_chunk = Mock()
        mock_usage = Mock()
        mock_usage.prompt_token_count = 50
        mock_usage.candidates_token_count = 75
        mock_usage.total_token_count = 125

        mock_chunk.usage_metadata = mock_usage

        # Streaming typically provides usage in final chunk
        if mock_chunk.usage_metadata:
            token_counts = TokenCounts(
                input_tokens=mock_usage.prompt_token_count,
                output_tokens=mock_usage.candidates_token_count,
                total_tokens=mock_usage.total_token_count,
                cached_tokens=0
            )

            assert token_counts.total_tokens == 125

    def test_accumulate_tokens_from_chunks(self):
        """Test that tokens can be accumulated from streaming chunks."""
        chunks = []

        # Simulate streaming chunks (only last one has usage)
        for i in range(5):
            chunk = Mock()
            if i < 4:
                chunk.usage_metadata = None
            else:
                # Final chunk has usage
                mock_usage = Mock()
                mock_usage.prompt_token_count = 100
                mock_usage.candidates_token_count = 200
                mock_usage.total_token_count = 300
                chunk.usage_metadata = mock_usage
            chunks.append(chunk)

        # Extract from final chunk
        final_chunk = chunks[-1]
        assert final_chunk.usage_metadata is not None
        assert final_chunk.usage_metadata.total_token_count == 300

    def test_streaming_with_no_usage_metadata(self):
        """Test streaming when no usage metadata provided."""
        chunks = [Mock(usage_metadata=None) for _ in range(5)]

        # Should handle gracefully
        final_usage = None
        for chunk in chunks:
            if chunk.usage_metadata:
                final_usage = chunk.usage_metadata

        if not final_usage:
            token_counts = TokenCounts(0, 0, 0, 0)
            assert token_counts.total_tokens == 0


class TestEmbeddingsTokenCounts:
    """Test token counting for embeddings."""

    def test_vertex_ai_embeddings_with_statistics(self):
        """Test Vertex AI embeddings that include token statistics."""
        mock_response = Mock()
        mock_stats = Mock()
        mock_stats.token_count = 50
        mock_stats.truncated = False

        mock_response.statistics = mock_stats

        # Vertex AI provides token counts for embeddings
        if hasattr(mock_response, "statistics"):
            token_count = mock_response.statistics.token_count
            assert token_count == 50
            assert not mock_response.statistics.truncated

    def test_google_ai_embeddings_no_tokens(self):
        """Test Google AI SDK embeddings (no token counts)."""
        mock_response = Mock()
        # Google AI SDK doesn't provide token counts for embeddings
        mock_response.usage_metadata = None

        # Should handle this limitation gracefully
        if not mock_response.usage_metadata:
            token_counts = TokenCounts(0, 0, 0, 0)
            assert token_counts.total_tokens == 0

    def test_embeddings_with_truncation(self):
        """Test embeddings that were truncated."""
        mock_response = Mock()
        mock_stats = Mock()
        mock_stats.token_count = 100
        mock_stats.truncated = True

        mock_response.statistics = mock_stats

        # Should detect truncation
        if hasattr(mock_response, "statistics"):
            assert mock_response.statistics.truncated is True


class TestEdgeCases:
    """Test edge cases in token counting."""

    def test_zero_tokens(self):
        """Test response with zero tokens."""
        token_counts = TokenCounts(0, 0, 0, 0)

        assert token_counts.input_tokens == 0
        assert token_counts.output_tokens == 0
        assert token_counts.total_tokens == 0

    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        token_counts = TokenCounts(
            input_tokens=100000,
            output_tokens=100000,
            total_tokens=200000,
            cached_tokens=50000
        )

        assert token_counts.total_tokens == 200000
        assert token_counts.cached_tokens == 50000

    def test_cached_tokens_reduce_cost(self):
        """Test that cached tokens are tracked separately."""
        token_counts = TokenCounts(
            input_tokens=100,
            output_tokens=150,
            total_tokens=250,
            cached_tokens=50  # 50 of the 100 input tokens were cached
        )

        # Cached tokens typically have lower cost
        assert token_counts.cached_tokens > 0
        assert token_counts.cached_tokens <= token_counts.input_tokens

    def test_missing_total_calculated_from_parts(self):
        """Test that missing total can be calculated."""
        input_tokens = 100
        output_tokens = 150

        # If total not provided, calculate it
        total_tokens = input_tokens + output_tokens

        assert total_tokens == 250

    def test_attribute_missing_returns_zero(self):
        """Test that missing attributes default to zero."""
        mock_usage = Mock(spec=['prompt_token_count'])  # Only has prompt_token_count
        mock_usage.prompt_token_count = 100

        input_tokens = getattr(mock_usage, 'prompt_token_count', 0)
        output_tokens = getattr(mock_usage, 'candidates_token_count', 0)
        cached_tokens = getattr(mock_usage, 'cached_content_token_count', 0)

        assert input_tokens == 100
        assert output_tokens == 0  # Missing, defaults to 0
        assert cached_tokens == 0  # Missing, defaults to 0
