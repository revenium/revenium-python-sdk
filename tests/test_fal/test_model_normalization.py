"""
Tests for fal.ai model name normalization.

Verifies that model names are correctly normalized to the LiteLLM format:
  fal_ai/{fal_endpoint_id}
"""

import pytest
from revenium_middleware.fal.trace_fields import normalize_model_name


class TestNormalizeModelName:
    """Tests for the normalize_model_name function."""

    def test_already_correct_format(self):
        """Model already in fal_ai/fal-ai/... format should be returned as-is."""
        assert normalize_model_name("fal_ai/fal-ai/flux/dev") == "fal_ai/fal-ai/flux/dev"

    def test_already_correct_format_deep_path(self):
        """Deeply nested correct format."""
        assert (
            normalize_model_name("fal_ai/fal-ai/flux/dev/image-to-image")
            == "fal_ai/fal-ai/flux/dev/image-to-image"
        )

    def test_fal_ai_prefix_missing_inner_segment(self):
        """fal_ai/flux/dev -> fal_ai/fal-ai/flux/dev"""
        assert normalize_model_name("fal_ai/flux/dev") == "fal_ai/fal-ai/flux/dev"

    def test_fal_endpoint_prefix_only(self):
        """fal-ai/flux/dev -> fal_ai/fal-ai/flux/dev"""
        assert normalize_model_name("fal-ai/flux/dev") == "fal_ai/fal-ai/flux/dev"

    def test_bare_model_name(self):
        """flux/dev -> fal_ai/fal-ai/flux/dev"""
        assert normalize_model_name("flux/dev") == "fal_ai/fal-ai/flux/dev"

    def test_bare_model_single_segment(self):
        """flux -> fal_ai/fal-ai/flux"""
        assert normalize_model_name("flux") == "fal_ai/fal-ai/flux"

    def test_third_party_vendor_correct_format(self):
        """Third-party vendor already in correct format."""
        assert (
            normalize_model_name("fal_ai/bria/text-to-image/3.2")
            == "fal_ai/fal-ai/bria/text-to-image/3.2"
        )

    def test_third_party_with_fal_ai_prefix(self):
        """fal-ai/bria/... -> fal_ai/fal-ai/bria/..."""
        assert (
            normalize_model_name("fal-ai/bria/text-to-image/3.2")
            == "fal_ai/fal-ai/bria/text-to-image/3.2"
        )

    def test_third_party_already_correct(self):
        """fal_ai/fal-ai/bria/... should be returned as-is."""
        assert (
            normalize_model_name("fal_ai/fal-ai/bria/text-to-image/3.2")
            == "fal_ai/fal-ai/bria/text-to-image/3.2"
        )

    def test_minimax_vendor(self):
        """minimax third-party vendor normalization."""
        assert (
            normalize_model_name("fal-ai/minimax/video-01")
            == "fal_ai/fal-ai/minimax/video-01"
        )

    def test_xai_vendor(self):
        """xai third-party vendor normalization."""
        assert (
            normalize_model_name("fal-ai/xai/grok-2")
            == "fal_ai/fal-ai/xai/grok-2"
        )

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        assert normalize_model_name("") == ""

    def test_none_passthrough(self):
        """None should be returned as-is."""
        assert normalize_model_name(None) is None
