"""
Tests for provider detection functionality.
"""
import pytest
from unittest.mock import MagicMock
from revenium_middleware.perplexity.provider import (
    Provider,
    detect_provider,
    get_provider_metadata,
    is_perplexity_provider
)


@pytest.mark.unit
class TestProviderDetection:
    """Test provider detection logic."""
    
    def test_detect_perplexity_via_base_url(self):
        """Test Perplexity detection via base URL."""
        provider = detect_provider(base_url="https://api.perplexity.ai")
        assert provider == Provider.PERPLEXITY
    
    def test_detect_perplexity_via_client_base_url(self):
        """Test Perplexity detection via client.base_url."""
        mock_client = MagicMock()
        mock_client.base_url = "https://api.perplexity.ai"
        
        provider = detect_provider(client=mock_client)
        assert provider == Provider.PERPLEXITY
    
    def test_detect_perplexity_default(self):
        """Test default to Perplexity when no URL provided."""
        provider = detect_provider()
        assert provider == Provider.PERPLEXITY
    
    def test_detect_perplexity_case_insensitive(self):
        """Test case-insensitive URL matching."""
        provider = detect_provider(base_url="https://api.PERPLEXITY.ai")
        assert provider == Provider.PERPLEXITY


@pytest.mark.unit
class TestProviderMetadata:
    """Test provider metadata generation."""
    
    def test_get_perplexity_metadata(self):
        """Test metadata for Perplexity provider."""
        metadata = get_provider_metadata(Provider.PERPLEXITY)
        
        assert metadata["provider"] == "PERPLEXITY"
        assert metadata["model_source"] == "PERPLEXITY"
    
    def test_is_perplexity_provider_true(self):
        """Test is_perplexity_provider returns True for Perplexity."""
        assert is_perplexity_provider(Provider.PERPLEXITY) is True
    
    def test_is_perplexity_provider_false(self):
        """Test is_perplexity_provider returns False for non-Perplexity."""
        assert is_perplexity_provider(Provider.OPENAI) is False

