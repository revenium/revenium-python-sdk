import os
import json
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest

from revenium_middleware.openai.summary_printer import (
    print_usage_summary,
    format_and_print_json_summary,
    format_and_print_human_summary,
    fetch_completion_metrics,
    CompletionMetrics,
)
from revenium_middleware.openai.config import (
    parse_print_summary_value,
    get_print_summary_config,
)


class TestSummaryPrinter:
    """Tests for the summary printer module."""

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing."""
        with patch.dict(os.environ, {
            'REVENIUM_PRINT_SUMMARY': 'true',
            'REVENIUM_TEAM_ID': 'test-team-id',
            'REVENIUM_METERING_BASE_URL': 'https://api.revenium.ai'
        }):
            yield

    def test_parse_print_summary_value_true(self):
        """Test parsing 'true' value."""
        # 'true' maps to 'human' format
        assert parse_print_summary_value('true') == 'human'
        assert parse_print_summary_value('TRUE') == 'human'
        assert parse_print_summary_value('1') == 'human'
        assert parse_print_summary_value('yes') == 'human'
        assert parse_print_summary_value('on') == 'human'

    def test_parse_print_summary_value_false(self):
        """Test parsing 'false' value."""
        assert parse_print_summary_value('false') is False
        assert parse_print_summary_value('FALSE') is False
        assert parse_print_summary_value('0') is False
        assert parse_print_summary_value(None) is False

    def test_parse_print_summary_value_human(self):
        """Test parsing 'human' value."""
        assert parse_print_summary_value('human') == 'human'
        assert parse_print_summary_value('HUMAN') == 'human'

    def test_parse_print_summary_value_json(self):
        """Test parsing 'json' value."""
        assert parse_print_summary_value('json') == 'json'
        assert parse_print_summary_value('JSON') == 'json'

    def test_get_print_summary_config(self):
        """Test getting print summary config from environment."""
        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'json'}):
            assert get_print_summary_config() == 'json'

        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'true'}):
            assert get_print_summary_config() == 'human'

        with patch.dict(os.environ, {}, clear=True):
            assert get_print_summary_config() is False

    @patch('sys.stdout', new_callable=StringIO)
    def test_format_and_print_json_summary_with_cost(self, mock_stdout):
        """Test JSON format output with cost."""
        metrics = CompletionMetrics(total_cost=0.000045)
        format_and_print_json_summary(
            model='gpt-4o-mini',
            provider='OPENAI',
            duration_seconds=1.234,
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
            trace_id='trace-456',
            metrics=metrics,
        )

        output = mock_stdout.getvalue()
        data = json.loads(output.strip())

        assert data['model'] == 'gpt-4o-mini'
        assert data['provider'] == 'OPENAI'
        assert data['durationSeconds'] == 1.23
        assert data['inputTokenCount'] == 100
        assert data['outputTokenCount'] == 50
        assert data['totalTokenCount'] == 150
        assert data['traceId'] == 'trace-456'
        assert data['cost'] == 0.000045

    @patch('sys.stdout', new_callable=StringIO)
    @patch.dict(os.environ, {}, clear=True)
    def test_format_and_print_json_summary_without_cost(self, mock_stdout):
        """Test JSON format output without cost."""
        format_and_print_json_summary(
            model='gpt-4o-mini',
            provider='OPENAI',
            duration_seconds=1.234,
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
            trace_id='trace-456',
            metrics=None,
        )

        output = mock_stdout.getvalue()
        data = json.loads(output.strip())

        assert data['cost'] is None
        assert data['costStatus'] == 'unavailable'

    @patch('sys.stdout', new_callable=StringIO)
    def test_format_and_print_human_summary_with_cost(self, mock_stdout):
        """Test human-readable format output with cost."""
        metrics = CompletionMetrics(total_cost=0.000045)
        format_and_print_human_summary(
            model='gpt-4o-mini',
            provider='OPENAI',
            duration_seconds=1.234,
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
            trace_id='trace-456',
            metrics=metrics,
        )

        output = mock_stdout.getvalue()

        assert 'REVENIUM USAGE SUMMARY' in output
        assert 'Model: gpt-4o-mini' in output
        assert 'Provider: OPENAI' in output
        assert 'Duration: 1.23s' in output
        assert '100' in output
        assert '50' in output
        assert '150' in output
        assert 'Cost: $0.000045' in output
        assert 'Trace ID: trace-456' in output

    @patch('sys.stdout', new_callable=StringIO)
    @patch.dict(os.environ, {}, clear=True)
    def test_format_and_print_human_summary_without_cost(self, mock_stdout):
        """Test human-readable format output without cost."""
        format_and_print_human_summary(
            model='gpt-4o-mini',
            provider='OPENAI',
            duration_seconds=1.234,
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
            trace_id='trace-456',
            metrics=None,
        )

        output = mock_stdout.getvalue()

        assert 'REVENIUM USAGE SUMMARY' in output
        assert 'Add REVENIUM_TEAM_ID to see pricing' in output

    @patch('sys.stdout', new_callable=StringIO)
    @patch.dict(os.environ, {'REVENIUM_TEAM_ID': 'team-123'})
    def test_format_and_print_human_summary_pending_cost(self, mock_stdout):
        """Test human-readable format output with pending cost."""
        format_and_print_human_summary(
            model='gpt-4o-mini',
            provider='OPENAI',
            duration_seconds=1.234,
            input_token_count=100,
            output_token_count=50,
            total_token_count=150,
            trace_id='trace-456',
            metrics=None,
        )

        output = mock_stdout.getvalue()

        assert 'REVENIUM USAGE SUMMARY' in output
        assert 'Cost: Pending' in output

    @patch('sys.stdout', new_callable=StringIO)
    def test_format_and_print_human_summary_null_tokens(self, mock_stdout):
        """Test human-readable format output with null token counts."""
        metrics = CompletionMetrics(total_cost=0.000045)
        format_and_print_human_summary(
            model='gpt-4o-mini',
            provider='OPENAI',
            duration_seconds=1.234,
            input_token_count=None,
            output_token_count=None,
            total_token_count=None,
            trace_id=None,
            metrics=metrics,
        )

        output = mock_stdout.getvalue()

        assert 'REVENIUM USAGE SUMMARY' in output
        # Null tokens should show as 0
        assert '0' in output

    @patch('revenium_middleware.openai.summary_printer.urlopen')
    @patch.dict(os.environ, {
        'REVENIUM_TEAM_ID': 'team-456',
        'REVENIUM_METERING_BASE_URL': 'https://api.revenium.ai'
    })
    def test_fetch_completion_metrics_success(self, mock_urlopen):
        """Test successful fetch of completion metrics."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            '_embedded': {
                'aICompletionMetricResourceList': [
                    {'totalCost': 0.000045}
                ]
            }
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        metrics = fetch_completion_metrics(
            transaction_id='txn-123',
            revenium_api_key='test-key',
        )

        assert metrics is not None
        assert metrics.total_cost == 0.000045

    @patch('revenium_middleware.openai.summary_printer.urlopen')
    @patch.dict(os.environ, {
        'REVENIUM_TEAM_ID': 'team-456',
        'REVENIUM_METERING_BASE_URL': 'https://api.revenium.ai'
    })
    def test_fetch_completion_metrics_retry(self, mock_urlopen):
        """Test retry logic when fetching metrics."""
        # First two calls return empty, third succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status = 200
        mock_response_fail.read.return_value = json.dumps({
            '_embedded': {
                'aICompletionMetricResourceList': []
            }
        }).encode('utf-8')
        mock_response_fail.__enter__.return_value = mock_response_fail

        mock_response_success = MagicMock()
        mock_response_success.status = 200
        mock_response_success.read.return_value = json.dumps({
            '_embedded': {
                'aICompletionMetricResourceList': [
                    {'totalCost': 0.000045}
                ]
            }
        }).encode('utf-8')
        mock_response_success.__enter__.return_value = mock_response_success

        mock_urlopen.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success,
        ]

        with patch('time.sleep'):  # Skip actual sleep
            metrics = fetch_completion_metrics(
                transaction_id='txn-123',
                revenium_api_key='test-key',
            )

        assert metrics is not None
        assert metrics.total_cost == 0.000045
        assert mock_urlopen.call_count == 3

    @patch('revenium_middleware.openai.summary_printer.urlopen')
    @patch.dict(os.environ, {
        'REVENIUM_TEAM_ID': 'team-456',
        'REVENIUM_METERING_BASE_URL': 'https://api.revenium.ai'
    })
    def test_fetch_completion_metrics_failure(self, mock_urlopen):
        """Test failure to fetch metrics after retries."""
        mock_urlopen.side_effect = Exception('Network error')

        with patch('time.sleep'):  # Skip actual sleep
            metrics = fetch_completion_metrics(
                transaction_id='txn-123',
                revenium_api_key='test-key',
            )

        assert metrics is None

    @patch('revenium_middleware.openai.summary_printer.fetch_completion_metrics')
    @patch('revenium_middleware.openai.summary_printer.format_and_print_json_summary')
    def test_print_usage_summary_json_format(self, mock_print_json, mock_fetch):
        """Test print_usage_summary with JSON format."""
        mock_fetch.return_value = CompletionMetrics(total_cost=0.000045)

        with patch.dict(os.environ, {
            'REVENIUM_PRINT_SUMMARY': 'json',
            'REVENIUM_TEAM_ID': 'team-123'
        }):
            print_usage_summary(
                model='gpt-4o-mini',
                provider='OPENAI',
                request_duration=1234,
                input_token_count=100,
                output_token_count=50,
                total_token_count=150,
                transaction_id='txn-123',
                trace_id='trace-456',
                revenium_api_key='test-key',
            )

        mock_print_json.assert_called_once()

    @patch('revenium_middleware.openai.summary_printer.fetch_completion_metrics')
    @patch('revenium_middleware.openai.summary_printer.format_and_print_human_summary')
    def test_print_usage_summary_human_format(self, mock_print_human, mock_fetch):
        """Test print_usage_summary with human format."""
        mock_fetch.return_value = CompletionMetrics(total_cost=0.000045)

        with patch.dict(os.environ, {
            'REVENIUM_PRINT_SUMMARY': 'true',
            'REVENIUM_TEAM_ID': 'team-123'
        }):
            print_usage_summary(
                model='gpt-4o-mini',
                provider='OPENAI',
                request_duration=1234,
                input_token_count=100,
                output_token_count=50,
                total_token_count=150,
                transaction_id='txn-123',
                trace_id='trace-456',
                revenium_api_key='test-key',
            )

        mock_print_human.assert_called_once()

    @patch('revenium_middleware.openai.summary_printer.fetch_completion_metrics')
    @patch('revenium_middleware.openai.summary_printer.format_and_print_json_summary')
    def test_print_usage_summary_disabled(self, mock_print_json, mock_fetch):
        """Test print_usage_summary when disabled."""
        with patch.dict(os.environ, {'REVENIUM_PRINT_SUMMARY': 'false'}):
            print_usage_summary(
                model='gpt-4o-mini',
                provider='OPENAI',
                request_duration=1234,
                input_token_count=100,
                output_token_count=50,
                total_token_count=150,
                transaction_id='txn-123',
                trace_id='trace-456',
                revenium_api_key='test-key',
            )

        mock_print_json.assert_not_called()
        mock_fetch.assert_not_called()


