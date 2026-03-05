"""
Tests for provider detection logic.
"""

import pytest
from unittest.mock import MagicMock

from revenium_middleware.anthropic.provider import (
    Provider,
    detect_provider,
    get_provider_metadata,
    is_bedrock_provider,
    get_or_detect_provider
)


class TestProviderDetection:
    """Test provider detection functionality."""

    def test_detect_bedrock_via_boto3_client(self):
        """Test detection of Bedrock via boto3 client metadata."""
        # Create mock boto3 bedrock-runtime client
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        
        result = detect_provider(client=mock_client)
        assert result == Provider.BEDROCK

    def test_detect_bedrock_via_base_url(self):
        """Test detection of Bedrock via base_url containing amazonaws.com."""
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        
        result = detect_provider(base_url=base_url)
        assert result == Provider.BEDROCK

    def test_detect_bedrock_via_client_base_url(self):
        """Test detection of Bedrock via client.base_url."""
        mock_client = MagicMock()
        mock_client.base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        
        result = detect_provider(client=mock_client)
        assert result == Provider.BEDROCK

    def test_detect_anthropic_default(self):
        """Test default detection returns Anthropic."""
        result = detect_provider()
        assert result == Provider.ANTHROPIC

    def test_detect_anthropic_with_regular_client(self):
        """Test detection with regular Anthropic client."""
        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com"
        
        result = detect_provider(client=mock_client)
        assert result == Provider.ANTHROPIC

    def test_detect_provider_handles_missing_meta(self):
        """Test detection handles clients without meta attribute gracefully."""
        mock_client = MagicMock()
        del mock_client.meta  # Remove meta attribute
        
        result = detect_provider(client=mock_client)
        assert result == Provider.ANTHROPIC

    def test_detect_provider_handles_missing_service_model(self):
        """Test detection handles clients without service_model gracefully."""
        mock_client = MagicMock()
        mock_client.meta = MagicMock()
        del mock_client.meta.service_model  # Remove service_model
        
        result = detect_provider(client=mock_client)
        assert result == Provider.ANTHROPIC


class TestProviderMetadata:
    """Test provider metadata functionality."""

    def test_get_bedrock_metadata(self):
        """Test metadata for Bedrock provider."""
        metadata = get_provider_metadata(Provider.BEDROCK)

        assert metadata["provider"] == "AWS"
        assert metadata["model_source"] == "ANTHROPIC"

    def test_get_anthropic_metadata(self):
        """Test metadata for Anthropic provider."""
        metadata = get_provider_metadata(Provider.ANTHROPIC)
        
        assert metadata["provider"] == "ANTHROPIC"
        assert metadata["model_source"] == "ANTHROPIC"

    def test_is_bedrock_provider_true(self):
        """Test is_bedrock_provider returns True for Bedrock."""
        assert is_bedrock_provider(Provider.BEDROCK) is True

    def test_is_bedrock_provider_false(self):
        """Test is_bedrock_provider returns False for Anthropic."""
        assert is_bedrock_provider(Provider.ANTHROPIC) is False


class TestProviderCaching:
    """Test provider caching functionality."""

    def test_get_or_detect_provider_caches_result(self):
        """Test that provider detection is cached."""
        # First call should detect and cache
        result1 = get_or_detect_provider()
        
        # Second call should return cached result
        result2 = get_or_detect_provider()
        
        assert result1 == result2
        assert result1 == Provider.ANTHROPIC  # Default

    def test_force_redetect_bypasses_cache(self):
        """Test that force_redetect bypasses the cache."""
        # First call
        get_or_detect_provider()
        
        # Force redetection with different parameters
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        
        result = get_or_detect_provider(client=mock_client, force_redetect=True)
        assert result == Provider.BEDROCK
