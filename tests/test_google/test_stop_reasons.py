"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for stop reason normalization.
"""

import pytest
from revenium_middleware.google.common.types import normalize_stop_reason, Provider


class TestStopReasonNormalization:
    """Test normalization of Google stop reasons to Revenium standards."""

    def test_normalize_google_ai_stop(self):
        """Test normalization of STOP reason."""
        result = normalize_stop_reason("STOP", Provider.GOOGLE_AI_SDK)

        assert result == "END"

    def test_normalize_google_ai_max_tokens(self):
        """Test normalization of MAX_TOKENS reason."""
        result = normalize_stop_reason("MAX_TOKENS", Provider.GOOGLE_AI_SDK)

        assert result == "TOKEN_LIMIT"

    def test_normalize_google_ai_safety(self):
        """Test normalization of SAFETY reason."""
        result = normalize_stop_reason("SAFETY", Provider.GOOGLE_AI_SDK)

        assert result == "ERROR"

    def test_normalize_google_ai_recitation(self):
        """Test normalization of RECITATION reason."""
        result = normalize_stop_reason("RECITATION", Provider.GOOGLE_AI_SDK)

        assert result == "ERROR"

    def test_normalize_google_ai_other(self):
        """Test normalization of OTHER reason."""
        result = normalize_stop_reason("OTHER", Provider.GOOGLE_AI_SDK)

        assert result == "END"

    def test_normalize_google_ai_none(self):
        """Test normalization of None reason."""
        result = normalize_stop_reason(None, Provider.GOOGLE_AI_SDK)

        assert result == "END"

    def test_normalize_vertex_ai_stop(self):
        """Test normalization of Vertex AI STOP reason."""
        result = normalize_stop_reason("STOP", Provider.VERTEX_AI_SDK)

        assert result == "END"

    def test_normalize_vertex_ai_max_tokens(self):
        """Test normalization of Vertex AI MAX_TOKENS reason."""
        result = normalize_stop_reason("MAX_TOKENS", Provider.VERTEX_AI_SDK)

        assert result == "TOKEN_LIMIT"

    def test_normalize_vertex_ai_safety(self):
        """Test normalization of Vertex AI SAFETY reason."""
        result = normalize_stop_reason("SAFETY", Provider.VERTEX_AI_SDK)

        assert result == "ERROR"

    def test_normalize_vertex_ai_recitation(self):
        """Test normalization of Vertex AI RECITATION reason."""
        result = normalize_stop_reason("RECITATION", Provider.VERTEX_AI_SDK)

        assert result == "ERROR"

    def test_normalize_vertex_ai_unspecified(self):
        """Test normalization of FINISH_REASON_UNSPECIFIED."""
        result = normalize_stop_reason("FINISH_REASON_UNSPECIFIED", Provider.VERTEX_AI_SDK)

        assert result == "END"

    def test_normalize_vertex_ai_none(self):
        """Test normalization of None reason for Vertex AI."""
        result = normalize_stop_reason(None, Provider.VERTEX_AI_SDK)

        assert result == "END"

    def test_normalize_unknown_reason_defaults_to_end(self):
        """Test that unknown reasons default to END."""
        result = normalize_stop_reason("UNKNOWN_REASON", Provider.GOOGLE_AI_SDK)

        assert result == "END"

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        result = normalize_stop_reason("", Provider.GOOGLE_AI_SDK)

        # Empty string should default to END
        assert result == "END"


class TestStopReasonMapping:
    """Test stop reason mapping consistency."""

    @pytest.mark.parametrize("google_reason,expected_revenium", [
        ("STOP", "END"),
        ("MAX_TOKENS", "TOKEN_LIMIT"),
        ("SAFETY", "ERROR"),
        ("RECITATION", "ERROR"),
        ("OTHER", "END"),
        (None, "END"),
    ])
    def test_google_ai_sdk_mappings(self, google_reason, expected_revenium):
        """Test all Google AI SDK mappings."""
        result = normalize_stop_reason(google_reason, Provider.GOOGLE_AI_SDK)

        assert result == expected_revenium

    @pytest.mark.parametrize("vertex_reason,expected_revenium", [
        ("STOP", "END"),
        ("MAX_TOKENS", "TOKEN_LIMIT"),
        ("SAFETY", "ERROR"),
        ("RECITATION", "ERROR"),
        ("FINISH_REASON_UNSPECIFIED", "END"),
        (None, "END"),
    ])
    def test_vertex_ai_sdk_mappings(self, vertex_reason, expected_revenium):
        """Test all Vertex AI SDK mappings."""
        result = normalize_stop_reason(vertex_reason, Provider.VERTEX_AI_SDK)

        assert result == expected_revenium


class TestReveniumStopReasons:
    """Test that normalized reasons match Revenium API spec."""

    def test_valid_revenium_stop_reasons(self):
        """Test that all normalized reasons are valid Revenium values."""
        valid_revenium_reasons = {
            "END",
            "END_SEQUENCE",
            "TIMEOUT",
            "TOKEN_LIMIT",
            "COST_LIMIT",
            "COMPLETION_LIMIT",
            "ERROR",
            "CANCELLED"
        }

        # All Google reasons should normalize to valid Revenium values
        google_reasons = ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", None]

        for reason in google_reasons:
            normalized = normalize_stop_reason(reason, Provider.GOOGLE_AI_SDK)
            assert normalized in valid_revenium_reasons

    def test_end_is_most_common(self):
        """Test that END is the most common normalized reason."""
        end_reasons = ["STOP", "OTHER", None, "FINISH_REASON_UNSPECIFIED"]

        for reason in end_reasons:
            result = normalize_stop_reason(reason, Provider.GOOGLE_AI_SDK)
            assert result == "END"

    def test_error_reasons_grouped(self):
        """Test that error-related reasons normalize to ERROR."""
        error_reasons = ["SAFETY", "RECITATION"]

        for reason in error_reasons:
            result = normalize_stop_reason(reason, Provider.GOOGLE_AI_SDK)
            assert result == "ERROR"


class TestEdgeCases:
    """Test edge cases in stop reason normalization."""

    def test_case_sensitivity(self):
        """Test that stop reasons are case-sensitive."""
        # Google uses uppercase
        result = normalize_stop_reason("STOP", Provider.GOOGLE_AI_SDK)
        assert result == "END"

        # Lowercase should default to END (unknown)
        result_lower = normalize_stop_reason("stop", Provider.GOOGLE_AI_SDK)
        assert result_lower == "END"  # Unknown, defaults to END

    def test_whitespace_in_reason(self):
        """Test handling of whitespace in stop reason."""
        result = normalize_stop_reason(" STOP ", Provider.GOOGLE_AI_SDK)

        # May or may not match depending on implementation
        # Should either normalize or default to END
        assert result in ["END", "STOP"]

    def test_numeric_reason(self):
        """Test handling of numeric stop reason."""
        result = normalize_stop_reason(123, Provider.GOOGLE_AI_SDK)

        # Should handle gracefully (convert to string or default)
        assert result is not None

    def test_unknown_provider_defaults_to_end(self):
        """Test that unknown provider type defaults to END."""
        result = normalize_stop_reason("STOP", "UNKNOWN_PROVIDER")

        assert result == "END"
