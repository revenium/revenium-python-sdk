"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for the summary printer module.

This module tests:
- Config parsing for REVENIUM_PRINT_SUMMARY
- JSON format output
- Human-readable format output
- Fetch metrics functionality
- Integration tests
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from revenium_middleware.google.common import trace_fields
from revenium_middleware.google.common.summary_printer import (
    CompletionMetrics,
    fetch_completion_metrics,
    format_and_print_json_summary,
    format_and_print_human_summary,
    print_usage_summary,
)


class TestConfigParsing:
    """Tests for config parsing functions."""

    def test_parse_print_summary_value_true(self):
        """Test parsing 'true' value."""
        assert trace_fields.parse_print_summary_value("true") == "human"
        assert trace_fields.parse_print_summary_value("TRUE") == "human"
        assert trace_fields.parse_print_summary_value("1") == "human"
        assert trace_fields.parse_print_summary_value("yes") == "human"
        assert trace_fields.parse_print_summary_value("on") == "human"

    def test_parse_print_summary_value_false(self):
        """Test parsing 'false' value."""
        assert trace_fields.parse_print_summary_value("false") is False
        assert trace_fields.parse_print_summary_value("FALSE") is False
        assert trace_fields.parse_print_summary_value("0") is False
        assert trace_fields.parse_print_summary_value("no") is False
        assert trace_fields.parse_print_summary_value("off") is False
        assert trace_fields.parse_print_summary_value("") is False
        assert trace_fields.parse_print_summary_value(None) is False

    def test_parse_print_summary_value_human(self):
        """Test parsing 'human' value."""
        assert trace_fields.parse_print_summary_value("human") == "human"
        assert trace_fields.parse_print_summary_value("HUMAN") == "human"

    def test_parse_print_summary_value_json(self):
        """Test parsing 'json' value."""
        assert trace_fields.parse_print_summary_value("json") == "json"
        assert trace_fields.parse_print_summary_value("JSON") == "json"

    def test_get_print_summary_config(self, monkeypatch):
        """Test get_print_summary_config function."""
        monkeypatch.setenv("REVENIUM_PRINT_SUMMARY", "json")
        assert trace_fields.get_print_summary_config() == "json"

        monkeypatch.setenv("REVENIUM_PRINT_SUMMARY", "human")
        assert trace_fields.get_print_summary_config() == "human"

        monkeypatch.delenv("REVENIUM_PRINT_SUMMARY", raising=False)
        assert trace_fields.get_print_summary_config() is False


class TestJsonFormat:
    """Tests for JSON format output."""

    def test_json_format_with_cost(self, capsys):
        """Test JSON output with cost available."""
        format_and_print_json_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            duration_seconds=1.234,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            cost=0.000045,
            trace_id="abc-123",
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())

        assert output["model"] == "gemini-2.0-flash-001"
        assert output["provider"] == "Google"
        assert output["durationSeconds"] == 1.234
        assert output["inputTokenCount"] == 150
        assert output["outputTokenCount"] == 250
        assert output["totalTokenCount"] == 400
        assert output["cost"] == 0.000045
        assert output["costStatus"] == "available"
        assert output["traceId"] == "abc-123"

    def test_json_format_without_cost(self, capsys):
        """Test JSON output without cost."""
        format_and_print_json_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            duration_seconds=1.0,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            trace_id=None,
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())

        assert output["costStatus"] == "unavailable"
        assert "cost" not in output
        assert "traceId" not in output


class TestHumanFormat:
    """Tests for human-readable format output."""

    def test_human_format_with_cost(self, capsys):
        """Test human-readable output with cost available."""
        format_and_print_human_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            duration_seconds=1.23,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            cost=0.000045,
            trace_id="abc-123",
            team_id="team-123",
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "REVENIUM USAGE SUMMARY" in output
        assert "gemini-2.0-flash-001" in output
        assert "Google" in output
        assert "1.23s" in output
        assert "150" in output
        assert "250" in output
        assert "400" in output
        assert "$0.000045" in output
        assert "abc-123" in output

    def test_human_format_without_team_id(self, capsys):
        """Test human-readable output without team ID."""
        format_and_print_human_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            duration_seconds=1.0,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            trace_id=None,
            team_id=None,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "Add REVENIUM_TEAM_ID to see pricing" in output

    def test_human_format_pending_cost(self, capsys):
        """Test human-readable output with pending cost (team ID set)."""
        format_and_print_human_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            duration_seconds=1.0,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            trace_id=None,
            team_id="team-123",
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "Pending" in output

    def test_human_format_null_tokens(self, capsys):
        """Test human-readable output with null token counts."""
        format_and_print_human_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            duration_seconds=1.0,
            input_token_count=None,
            output_token_count=None,
            total_token_count=None,
            cost=None,
            trace_id=None,
            team_id=None,
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should show 0 for null tokens
        assert "Input Tokens:  0" in output
        assert "Output Tokens: 0" in output
        assert "Total Tokens:  0" in output


class TestFetchMetrics:
    """Tests for fetch_completion_metrics function."""

    def test_fetch_metrics_no_team_id(self, monkeypatch):
        """Test fetch metrics returns None when no team ID."""
        monkeypatch.delenv("REVENIUM_TEAM_ID", raising=False)

        result = fetch_completion_metrics("txn-123", "api-key")

        assert result is not None
        assert result.total_cost is None

    @patch("revenium_middleware.google.common.summary_printer.requests.get")
    def test_fetch_metrics_success(self, mock_get, monkeypatch):
        """Test successful metrics fetch."""
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-123")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"totalCost": 0.000045}
        mock_get.return_value = mock_response

        result = fetch_completion_metrics("txn-123", "api-key")

        assert result is not None
        assert result.total_cost == 0.000045

    @patch("revenium_middleware.google.common.summary_printer.requests.get")
    def test_fetch_metrics_retry(self, mock_get, monkeypatch):
        """Test metrics fetch with retry."""
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-123")

        # First call fails, second succeeds
        mock_fail = Mock()
        mock_fail.status_code = 500

        mock_success = Mock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"totalCost": 0.00005}

        mock_get.side_effect = [mock_fail, mock_success]

        result = fetch_completion_metrics("txn-123", "api-key")

        assert result is not None
        assert result.total_cost == 0.00005
        assert mock_get.call_count == 2

    @patch("revenium_middleware.google.common.summary_printer.requests.get")
    def test_fetch_metrics_failure(self, mock_get, monkeypatch):
        """Test metrics fetch failure after all retries."""
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-123")

        mock_fail = Mock()
        mock_fail.status_code = 500
        mock_get.return_value = mock_fail

        result = fetch_completion_metrics("txn-123", "api-key")

        assert result is None
        assert mock_get.call_count == 3  # 3 retry attempts


class TestPrintUsageSummary:
    """Integration tests for print_usage_summary."""

    def test_disabled_mode(self, monkeypatch, capsys):
        """Test that nothing is printed when disabled."""
        monkeypatch.setenv("REVENIUM_PRINT_SUMMARY", "false")

        print_usage_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            request_duration=1000,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            transaction_id="txn-123",
            trace_id="trace-123",
            revenium_api_key="api-key",
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    @patch("revenium_middleware.google.common.summary_printer.fetch_completion_metrics")
    def test_json_format_integration(self, mock_fetch, monkeypatch, capsys):
        """Test JSON format end-to-end."""
        monkeypatch.setenv("REVENIUM_PRINT_SUMMARY", "json")
        mock_fetch.return_value = CompletionMetrics(total_cost=0.00005)

        print_usage_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            request_duration=1230,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            transaction_id="txn-123",
            trace_id="trace-123",
            revenium_api_key="api-key",
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())

        assert output["model"] == "gemini-2.0-flash-001"
        assert output["durationSeconds"] == 1.23
        assert output["cost"] == 0.00005

    @patch("revenium_middleware.google.common.summary_printer.fetch_completion_metrics")
    def test_human_format_integration(self, mock_fetch, monkeypatch, capsys):
        """Test human format end-to-end."""
        monkeypatch.setenv("REVENIUM_PRINT_SUMMARY", "human")
        mock_fetch.return_value = CompletionMetrics(total_cost=0.00005)

        print_usage_summary(
            model="gemini-2.0-flash-001",
            provider="Google",
            request_duration=1230,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            transaction_id="txn-123",
            trace_id="trace-123",
            revenium_api_key="api-key",
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "REVENIUM USAGE SUMMARY" in output
        assert "gemini-2.0-flash-001" in output
        assert "$0.000050" in output

    def test_fire_and_forget(self, monkeypatch, capsys):
        """Test that exceptions don't propagate."""
        monkeypatch.setenv("REVENIUM_PRINT_SUMMARY", "json")

        # This should not raise even with invalid data
        with patch(
            "revenium_middleware.google.common.summary_printer.fetch_completion_metrics"
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Test error")

            # Should not raise
            print_usage_summary(
                model="gemini-2.0-flash-001",
                provider="Google",
                request_duration=1000,
                input_token_count=100,
                output_token_count=200,
                total_token_count=300,
                transaction_id="txn-123",
                trace_id="trace-123",
                revenium_api_key="api-key",
            )

