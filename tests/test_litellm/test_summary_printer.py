"""
Tests for the summary printer module.

Tests cover:
- Config parsing (parse_print_summary_value, get_print_summary_config)
- JSON format output
- Human format output
- Fetch metrics functionality
- Integration tests
"""

import json
import os
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from revenium_middleware.litellm.client.config import (
    parse_print_summary_value,
    get_print_summary_config,
    get_team_id,
    get_base_url,
    DEFAULT_BASE_URL,
)
from revenium_middleware.litellm.client.summary_printer import (
    CompletionMetrics,
    fetch_completion_metrics,
    format_and_print_json_summary,
    format_and_print_human_summary,
    print_usage_summary,
)


# =============================================================================
# Config Parsing Tests
# =============================================================================

class TestParseConfigValue:
    """Test parse_print_summary_value function."""

    def test_parse_true_values(self):
        """Test values that should return 'human' format."""
        for value in ['true', 'True', 'TRUE', '1', 'yes', 'on', 'enabled', 'human', 'HUMAN']:
            assert parse_print_summary_value(value) == "human", f"Failed for value: {value}"

    def test_parse_json_value(self):
        """Test values that should return 'json' format."""
        assert parse_print_summary_value('json') == "json"
        assert parse_print_summary_value('JSON') == "json"
        assert parse_print_summary_value('Json') == "json"

    def test_parse_false_values(self):
        """Test values that should return False (disabled)."""
        for value in ['false', 'False', 'FALSE', '0', 'no', 'off', 'disabled', None, '']:
            assert parse_print_summary_value(value) is False, f"Failed for value: {value}"

    def test_parse_invalid_value(self):
        """Test invalid values should return False with warning."""
        assert parse_print_summary_value('invalid') is False
        assert parse_print_summary_value('maybe') is False


class TestGetPrintSummaryConfig:
    """Test get_print_summary_config function."""

    def test_get_config_human(self):
        """Test getting human config from environment."""
        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'human'}):
            assert get_print_summary_config() == "human"

    def test_get_config_json(self):
        """Test getting json config from environment."""
        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'json'}):
            assert get_print_summary_config() == "json"

    def test_get_config_disabled(self):
        """Test getting disabled config from environment."""
        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'false'}):
            assert get_print_summary_config() is False

    def test_get_config_not_set(self):
        """Test default when environment variable not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the var if it exists
            os.environ.pop('REVENIUM_PRINT_SUMMARY', None)
            assert get_print_summary_config() is False


class TestGetTeamId:
    """Test get_team_id function."""

    def test_get_team_id_set(self):
        """Test getting team ID when set."""
        with patch.dict(os.environ, {'REVENIUM_TEAM_ID': 'test-team-123'}):
            assert get_team_id() == 'test-team-123'

    def test_get_team_id_not_set(self):
        """Test getting team ID when not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('REVENIUM_TEAM_ID', None)
            assert get_team_id() is None


class TestGetBaseUrl:
    """Test get_base_url function."""

    def test_get_base_url_set(self):
        """Test getting base URL when set."""
        with patch.dict(os.environ, {'REVENIUM_METERING_BASE_URL': 'https://custom.api.com'}):
            assert get_base_url() == 'https://custom.api.com'

    def test_get_base_url_default(self):
        """Test getting default base URL."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('REVENIUM_METERING_BASE_URL', None)
            assert get_base_url() == DEFAULT_BASE_URL


# =============================================================================
# JSON Format Tests
# =============================================================================

class TestJsonFormat:
    """Test JSON format output."""

    def test_json_format_with_cost(self, capsys):
        """Test JSON output with cost available."""
        format_and_print_json_summary(
            model="gpt-4o-mini",
            provider="LITELLM",
            duration_seconds=1.234,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            cost=0.000045,
            cost_status="available",
            trace_id="test-trace-123",
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())

        assert output["model"] == "gpt-4o-mini"
        assert output["provider"] == "LITELLM"
        assert output["durationSeconds"] == 1.234
        assert output["inputTokenCount"] == 150
        assert output["outputTokenCount"] == 250
        assert output["totalTokenCount"] == 400
        assert output["cost"] == 0.000045
        assert output["costStatus"] == "available"
        assert output["traceId"] == "test-trace-123"

    def test_json_format_without_cost(self, capsys):
        """Test JSON output without cost."""
        format_and_print_json_summary(
            model="gpt-4o-mini",
            provider="LITELLM",
            duration_seconds=1.5,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            cost_status="unavailable",
            trace_id=None,
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())

        assert "cost" not in output
        assert output["costStatus"] == "unavailable"
        assert "traceId" not in output


# =============================================================================
# Human Format Tests
# =============================================================================

class TestHumanFormat:
    """Test human-readable format output."""

    def test_human_format_with_cost(self, capsys):
        """Test human output with cost available."""
        format_and_print_human_summary(
            model="gpt-4o-mini",
            provider="LITELLM",
            duration_seconds=1.234,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            cost=0.000045,
            cost_status="available",
            trace_id="test-trace-123",
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "REVENIUM USAGE SUMMARY" in output
        assert "Model: gpt-4o-mini" in output
        assert "Provider: LITELLM" in output
        assert "Duration: 1.23s" in output
        assert "Input Tokens:  150" in output
        assert "Output Tokens: 250" in output
        assert "Total Tokens:  400" in output
        assert "Cost: $0.000045" in output
        assert "Trace ID: test-trace-123" in output

    def test_human_format_without_cost_no_team_id(self, capsys):
        """Test human output without cost and no team ID."""
        format_and_print_human_summary(
            model="gpt-4o-mini",
            provider="LITELLM",
            duration_seconds=1.5,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            cost_status="unavailable",
            trace_id=None,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "Add REVENIUM_TEAM_ID to see pricing" in output
        assert "Trace ID:" not in output

    def test_human_format_pending_cost(self, capsys):
        """Test human output with pending cost status."""
        format_and_print_human_summary(
            model="gpt-4o-mini",
            provider="LITELLM",
            duration_seconds=1.0,
            input_token_count=50,
            output_token_count=50,
            total_token_count=100,
            cost=None,
            cost_status="pending",
            trace_id=None,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "Pending (aggregating... check Revenium dashboard)" in output

    def test_human_format_null_token_counts(self, capsys):
        """Test human output with null token counts."""
        format_and_print_human_summary(
            model="gpt-4o-mini",
            provider="LITELLM",
            duration_seconds=1.0,
            input_token_count=None,
            output_token_count=None,
            total_token_count=None,
            cost=None,
            cost_status="unavailable",
            trace_id=None,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "Input Tokens:  N/A" in output
        assert "Output Tokens: N/A" in output
        assert "Total Tokens:  N/A" in output


# =============================================================================
# Fetch Metrics Tests
# =============================================================================

class TestFetchMetrics:
    """Test fetch_completion_metrics function."""

    def test_fetch_metrics_no_team_id(self):
        """Test fetch returns metrics with None cost when no team ID."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('REVENIUM_TEAM_ID', None)
            metrics = fetch_completion_metrics("test-tx-123", "test-api-key")
            assert metrics is not None
            assert metrics.total_cost is None

    @patch('urllib.request.urlopen')
    def test_fetch_metrics_success(self, mock_urlopen):
        """Test successful fetch of metrics."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"totalCost": 0.00123}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(os.environ, {'REVENIUM_TEAM_ID': 'test-team'}):
            metrics = fetch_completion_metrics("test-tx-123", "test-api-key")
            assert metrics is not None
            assert metrics.total_cost == 0.00123

    @patch('urllib.request.urlopen')
    def test_fetch_metrics_retry_on_failure(self, mock_urlopen):
        """Test retry logic when fetch fails."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        with patch.dict(os.environ, {'REVENIUM_TEAM_ID': 'test-team'}):
            with patch('time.sleep'):  # Skip actual sleep
                metrics = fetch_completion_metrics("test-tx-123", "test-api-key")
                # Should have been called 3 times (retry attempts)
                assert mock_urlopen.call_count == 3
                # Should return CompletionMetrics with None cost (consistent type)
                assert metrics is not None
                assert metrics.total_cost is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestPrintUsageSummary:
    """Test print_usage_summary integration."""

    def test_disabled_mode_no_output(self, capsys):
        """Test that disabled mode produces no output."""
        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'false'}):
            print_usage_summary(
                model="gpt-4o-mini",
                provider="LITELLM",
                request_duration=1234.0,
                input_token_count=100,
                output_token_count=200,
                total_token_count=300,
                transaction_id="test-tx-123",
                trace_id="test-trace",
                revenium_api_key="test-key",
            )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_json_mode_output(self, capsys):
        """Test JSON mode produces valid JSON output."""
        with patch.dict(os.environ, {
            'REVENIUM_PRINT_SUMMARY': 'json',
            'REVENIUM_METERING_API_KEY': 'test-key',
        }):
            # Remove team ID to avoid fetch attempt
            os.environ.pop('REVENIUM_TEAM_ID', None)

            print_usage_summary(
                model="gpt-4o-mini",
                provider="LITELLM",
                request_duration=1234.0,
                input_token_count=100,
                output_token_count=200,
                total_token_count=300,
                transaction_id="test-tx-123",
                trace_id="test-trace",
                revenium_api_key="test-key",
            )

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output["model"] == "gpt-4o-mini"
        assert output["durationSeconds"] == 1.234

    def test_human_mode_output(self, capsys):
        """Test human mode produces formatted output."""
        with patch.dict(os.environ, {
            'REVENIUM_PRINT_SUMMARY': 'human',
            'REVENIUM_METERING_API_KEY': 'test-key',
        }):
            os.environ.pop('REVENIUM_TEAM_ID', None)

            print_usage_summary(
                model="gpt-4o-mini",
                provider="LITELLM",
                request_duration=1234.0,
                input_token_count=100,
                output_token_count=200,
                total_token_count=300,
                transaction_id="test-tx-123",
                trace_id="test-trace",
                revenium_api_key="test-key",
            )

        captured = capsys.readouterr()
        assert "REVENIUM USAGE SUMMARY" in captured.out
        assert "Model: gpt-4o-mini" in captured.out

    def test_fire_and_forget_no_exception(self):
        """Test that exceptions are caught and don't propagate."""
        # This should not raise even with invalid config
        with patch('revenium_middleware.litellm.client.summary_printer.get_print_summary_config',
                   side_effect=Exception("Test error")):
            # Should not raise
            print_usage_summary(
                model="gpt-4o-mini",
                provider="LITELLM",
                request_duration=1234.0,
                input_token_count=100,
                output_token_count=200,
                total_token_count=300,
                transaction_id="test-tx-123",
                trace_id="test-trace",
                revenium_api_key="test-key",
            )

