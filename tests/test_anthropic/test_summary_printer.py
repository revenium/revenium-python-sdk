"""
Tests for the summary printer module.

This module tests the terminal summary output functionality including:
- Configuration parsing
- JSON format output
- Human-readable format output
- Cost fetching with retry logic
- Integration tests
"""

import json
import os
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

from revenium_middleware.anthropic.config import (
    parse_print_summary_value,
    get_print_summary_config,
    get_team_id,
    get_base_url,
)
from revenium_middleware.anthropic.summary_printer import (
    CompletionMetrics,
    fetch_completion_metrics,
    format_and_print_json_summary,
    format_and_print_human_summary,
    print_usage_summary,
)


class TestConfigParsing(unittest.TestCase):
    """Tests for configuration parsing functions."""

    def test_parse_print_summary_value_true(self):
        """Test parsing 'true' value."""
        self.assertEqual(parse_print_summary_value("true"), "human")
        self.assertEqual(parse_print_summary_value("TRUE"), "human")
        self.assertEqual(parse_print_summary_value("True"), "human")

    def test_parse_print_summary_value_false(self):
        """Test parsing 'false' value."""
        self.assertFalse(parse_print_summary_value("false"))
        self.assertFalse(parse_print_summary_value("FALSE"))
        self.assertFalse(parse_print_summary_value(None))
        self.assertFalse(parse_print_summary_value(""))

    def test_parse_print_summary_value_human(self):
        """Test parsing 'human' value."""
        self.assertEqual(parse_print_summary_value("human"), "human")
        self.assertEqual(parse_print_summary_value("HUMAN"), "human")

    def test_parse_print_summary_value_json(self):
        """Test parsing 'json' value."""
        self.assertEqual(parse_print_summary_value("json"), "json")
        self.assertEqual(parse_print_summary_value("JSON"), "json")

    @patch.dict(os.environ, {"REVENIUM_PRINT_SUMMARY": "json"})
    def test_get_print_summary_config(self):
        """Test getting print summary config from environment."""
        self.assertEqual(get_print_summary_config(), "json")

    @patch.dict(os.environ, {}, clear=True)
    def test_get_print_summary_config_default(self):
        """Test default print summary config when not set."""
        # Remove the key if it exists
        os.environ.pop("REVENIUM_PRINT_SUMMARY", None)
        self.assertFalse(get_print_summary_config())


class TestJsonFormat(unittest.TestCase):
    """Tests for JSON format output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_and_print_json_summary_with_cost(self, mock_stdout):
        """Test JSON output with cost available."""
        format_and_print_json_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            duration_seconds=1.234,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            cost=0.000045,
            cost_status="available",
            trace_id="test-trace-123",
        )

        output = mock_stdout.getvalue().strip()
        data = json.loads(output)

        self.assertEqual(data["model"], "claude-3-haiku-20240307")
        self.assertEqual(data["provider"], "ANTHROPIC")
        self.assertEqual(data["durationSeconds"], 1.234)
        self.assertEqual(data["inputTokenCount"], 150)
        self.assertEqual(data["outputTokenCount"], 250)
        self.assertEqual(data["totalTokenCount"], 400)
        self.assertEqual(data["cost"], 0.000045)
        self.assertEqual(data["costStatus"], "available")
        self.assertEqual(data["traceId"], "test-trace-123")

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_and_print_json_summary_without_cost(self, mock_stdout):
        """Test JSON output without cost."""
        format_and_print_json_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            duration_seconds=1.0,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            cost_status="Add REVENIUM_TEAM_ID to see pricing",
        )

        output = mock_stdout.getvalue().strip()
        data = json.loads(output)

        self.assertIsNone(data["cost"])
        self.assertEqual(data["costStatus"], "Add REVENIUM_TEAM_ID to see pricing")
        self.assertNotIn("traceId", data)


class TestHumanFormat(unittest.TestCase):
    """Tests for human-readable format output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_and_print_human_summary_with_cost(self, mock_stdout):
        """Test human-readable output with cost available."""
        format_and_print_human_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            duration_seconds=1.23,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            cost=0.000045,
            cost_status="available",
            trace_id="test-trace-123",
        )

        output = mock_stdout.getvalue()

        self.assertIn("REVENIUM USAGE SUMMARY", output)
        self.assertIn("Model: claude-3-haiku-20240307", output)
        self.assertIn("Provider: ANTHROPIC", output)
        self.assertIn("Duration: 1.23s", output)
        self.assertIn("Input Tokens:  150", output)
        self.assertIn("Output Tokens: 250", output)
        self.assertIn("Total Tokens:  400", output)
        self.assertIn("Cost: $0.000045", output)
        self.assertIn("Trace ID: test-trace-123", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_and_print_human_summary_without_team_id(self, mock_stdout):
        """Test human-readable output without team ID."""
        format_and_print_human_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            duration_seconds=1.0,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            cost_status="Add REVENIUM_TEAM_ID to see pricing",
        )

        output = mock_stdout.getvalue()
        self.assertIn("Cost: Add REVENIUM_TEAM_ID to see pricing", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_and_print_human_summary_pending_cost(self, mock_stdout):
        """Test human-readable output with pending cost."""
        format_and_print_human_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            duration_seconds=1.0,
            input_token_count=100,
            output_token_count=200,
            total_token_count=300,
            cost=None,
            cost_status="Pending (aggregating... check Revenium dashboard)",
        )

        output = mock_stdout.getvalue()
        self.assertIn("Cost: Pending (aggregating... check Revenium dashboard)", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_format_and_print_human_summary_null_tokens(self, mock_stdout):
        """Test human-readable output with null token counts."""
        format_and_print_human_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            duration_seconds=1.0,
            input_token_count=None,
            output_token_count=None,
            total_token_count=None,
            cost=None,
            cost_status="unavailable",
        )

        output = mock_stdout.getvalue()
        self.assertIn("Input Tokens:  N/A", output)
        self.assertIn("Output Tokens: N/A", output)
        self.assertIn("Total Tokens:  N/A", output)


class TestFetchMetrics(unittest.TestCase):
    """Tests for fetching completion metrics."""

    @patch.dict(os.environ, {"REVENIUM_TEAM_ID": "test-team-id"})
    @patch("urllib.request.urlopen")
    def test_fetch_completion_metrics_success(self, mock_urlopen):
        """Test successful metrics fetch."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"totalCost": 0.000045}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        metrics = fetch_completion_metrics("txn-123", "api-key-123")

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_cost, 0.000045)

    @patch.dict(os.environ, {"REVENIUM_TEAM_ID": "test-team-id"})
    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_fetch_completion_metrics_retry(self, mock_sleep, mock_urlopen):
        """Test retry logic on failure."""
        import urllib.error

        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"totalCost": 0.00005}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection failed"),
            urllib.error.URLError("Connection failed"),
            mock_response,
        ]

        metrics = fetch_completion_metrics("txn-123", "api-key-123")

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_cost, 0.00005)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch.dict(os.environ, {"REVENIUM_TEAM_ID": "test-team-id"})
    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_fetch_completion_metrics_failure(self, mock_sleep, mock_urlopen):
        """Test failure after all retries."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        metrics = fetch_completion_metrics("txn-123", "api-key-123")

        self.assertIsNone(metrics)
        self.assertEqual(mock_sleep.call_count, 2)  # 3 attempts, 2 sleeps

    @patch.dict(os.environ, {}, clear=True)
    def test_fetch_completion_metrics_no_team_id(self):
        """Test fetch returns None when no team ID is set."""
        os.environ.pop("REVENIUM_TEAM_ID", None)
        metrics = fetch_completion_metrics("txn-123", "api-key-123")
        self.assertIsNone(metrics)


class TestPrintUsageSummary(unittest.TestCase):
    """Integration tests for print_usage_summary."""

    @patch.dict(os.environ, {"REVENIUM_PRINT_SUMMARY": "json"})
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_usage_summary_json_format(self, mock_stdout):
        """Test print_usage_summary with JSON format."""
        print_usage_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            request_duration=1230.0,  # milliseconds
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            transaction_id="txn-123",
            trace_id="trace-123",
        )

        output = mock_stdout.getvalue().strip()
        data = json.loads(output)

        self.assertEqual(data["model"], "claude-3-haiku-20240307")
        self.assertEqual(data["durationSeconds"], 1.23)

    @patch.dict(os.environ, {"REVENIUM_PRINT_SUMMARY": "human"})
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_usage_summary_human_format(self, mock_stdout):
        """Test print_usage_summary with human format."""
        print_usage_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            request_duration=1230.0,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            transaction_id="txn-123",
        )

        output = mock_stdout.getvalue()
        self.assertIn("REVENIUM USAGE SUMMARY", output)
        self.assertIn("Model: claude-3-haiku-20240307", output)

    @patch.dict(os.environ, {"REVENIUM_PRINT_SUMMARY": "false"})
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_usage_summary_disabled(self, mock_stdout):
        """Test print_usage_summary when disabled."""
        print_usage_summary(
            model="claude-3-haiku-20240307",
            provider="ANTHROPIC",
            request_duration=1230.0,
            input_token_count=150,
            output_token_count=250,
            total_token_count=400,
            transaction_id="txn-123",
        )

        output = mock_stdout.getvalue()
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()

