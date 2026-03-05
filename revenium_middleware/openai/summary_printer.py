"""
Terminal summary printer for Revenium OpenAI middleware.

This module provides functionality to print cost/metrics summaries to the terminal
after each API request. Supports both human-readable and JSON output formats.

Fetches cost data from Revenium's traces API and formats for console display.
"""

import json
import logging
import time
from typing import Optional, Dict, Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .config import (
    Config,
    SummaryFormat,
    get_print_summary_config,
    get_team_id,
    get_base_url,
)

logger = logging.getLogger("revenium_middleware.summary_printer")


class CompletionMetrics:
    """Completion metrics from Revenium API."""

    def __init__(self, total_cost: Optional[float] = None):
        self.total_cost = total_cost


def fetch_completion_metrics(
    transaction_id: str,
    revenium_api_key: str,
) -> Optional[CompletionMetrics]:
    """
    Fetch metrics from Revenium completions API.

    Args:
        transaction_id: The transaction ID to fetch metrics for
        revenium_api_key: Revenium API key for authentication

    Returns:
        CompletionMetrics if successful, None otherwise
    """
    team_id = get_team_id()
    if not team_id:
        logger.debug(
            "Team ID not configured, skipping cost retrieval for summary"
        )
        return None

    base_url = get_base_url().rstrip("/")
    # Note: profitstream API uses a different path structure than the metering API
    url = f"{base_url}/profitstream/v2/api/sources/metrics/ai/completions"
    params = {
        "teamId": team_id,
        "transactionId": transaction_id,
    }
    url_with_params = f"{url}?{urlencode(params)}"

    logger.debug(f"Fetching completion metrics from {url_with_params}")

    max_retries = Config.SUMMARY_RETRY_ATTEMPTS
    retry_delay = Config.SUMMARY_RETRY_DELAY

    for attempt in range(max_retries):
        try:
            request = Request(url_with_params)
            request.add_header("Authorization", revenium_api_key)
            request.add_header("Content-Type", "application/json")

            with urlopen(request, timeout=Config.SUMMARY_API_TIMEOUT) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    embedded = data.get("_embedded", {})
                    metrics_list = embedded.get("aICompletionMetricResourceList", [])

                    if metrics_list:
                        first_metric = metrics_list[0]
                        total_cost = first_metric.get("totalCost")
                        logger.debug(f"Retrieved cost: {total_cost}")
                        return CompletionMetrics(total_cost=total_cost)

                    logger.debug(
                        f"No metrics found yet (attempt {attempt + 1}/{max_retries})"
                    )

            if attempt < max_retries - 1:
                logger.debug(
                    f"Waiting for metrics to aggregate "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)

        except (HTTPError, URLError) as e:
            logger.debug(
                f"Failed to fetch trace metrics: {e} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            logger.debug(
                f"Unexpected error fetching metrics: {e} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return None


def format_and_print_json_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    trace_id: Optional[str],
    metrics: Optional[CompletionMetrics],
) -> None:
    """
    Format and print summary in JSON format.

    Args:
        model: Model name
        provider: Provider name
        duration_seconds: Request duration in seconds
        input_token_count: Input token count
        output_token_count: Output token count
        total_token_count: Total token count
        trace_id: Trace ID
        metrics: Optional completion metrics from API
    """
    team_id = get_team_id()

    summary: Dict[str, Any] = {
        "model": model,
        "provider": provider,
        "durationSeconds": round(duration_seconds, 2),
        "inputTokenCount": input_token_count,
        "outputTokenCount": output_token_count,
        "totalTokenCount": total_token_count,
        "cost": metrics.total_cost if metrics and metrics.total_cost is not None else None,
    }

    # Add cost status if cost is null
    if summary["cost"] is None:
        summary["costStatus"] = "pending" if team_id else "unavailable"

    # Add trace ID if present
    if trace_id:
        summary["traceId"] = trace_id

    print(json.dumps(summary))


def format_and_print_human_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    trace_id: Optional[str],
    metrics: Optional[CompletionMetrics],
) -> None:
    """
    Format and print summary in human-readable format.

    Args:
        model: Model name
        provider: Provider name
        duration_seconds: Request duration in seconds
        input_token_count: Input token count
        output_token_count: Output token count
        total_token_count: Total token count
        trace_id: Trace ID
        metrics: Optional completion metrics from API
    """
    team_id = get_team_id()

    print("=" * 60)
    print("REVENIUM USAGE SUMMARY")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Duration: {duration_seconds:.2f}s")

    print("\nToken Usage:")
    print(f"  Input Tokens:  {(input_token_count or 0):,}")
    print(f"  Output Tokens: {(output_token_count or 0):,}")
    print(f"  Total Tokens:  {(total_token_count or 0):,}")

    if metrics and metrics.total_cost is not None:
        print(f"\nCost: ${metrics.total_cost:.6f}")
    else:
        if team_id:
            print(
                "\nCost: Pending (aggregating... check Revenium dashboard)"
            )
        else:
            print(
                "\nCost: Add REVENIUM_TEAM_ID to see pricing "
                "(find your team ID in the Revenium web app)"
            )

    if trace_id:
        print(f"\nTrace ID: {trace_id}")

    print("=" * 60 + "\n")


def print_usage_summary(
    model: str,
    provider: str,
    request_duration: int,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    transaction_id: Optional[str],
    trace_id: Optional[str],
    revenium_api_key: str,
) -> None:
    """
    Print usage summary to console (fire-and-forget).

    This function is called after tracking is complete to optionally display
    a formatted cost/metrics summary in the terminal.

    Args:
        model: Model name
        provider: Provider name
        request_duration: Request duration in milliseconds
        input_token_count: Input token count
        output_token_count: Output token count
        total_token_count: Total token count
        transaction_id: Transaction ID for fetching metrics
        trace_id: Trace ID for display
        revenium_api_key: Revenium API key for authentication
    """
    print_summary = get_print_summary_config()
    if not print_summary:
        return

    # Determine format
    format_type: SummaryFormat = "human" if print_summary is True else print_summary

    duration_seconds = request_duration / 1000.0

    # Fetch metrics if team_id and transaction_id are available
    metrics: Optional[CompletionMetrics] = None
    team_id = get_team_id()
    if team_id and transaction_id:
        try:
            metrics = fetch_completion_metrics(transaction_id, revenium_api_key)
        except Exception as e:
            logger.debug(f"Failed to fetch metrics: {e}")

    # Print summary in the appropriate format
    try:
        if format_type == "json":
            format_and_print_json_summary(
                model=model,
                provider=provider,
                duration_seconds=duration_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                total_token_count=total_token_count,
                trace_id=trace_id,
                metrics=metrics,
            )
        else:
            format_and_print_human_summary(
                model=model,
                provider=provider,
                duration_seconds=duration_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                total_token_count=total_token_count,
                trace_id=trace_id,
                metrics=metrics,
            )
    except Exception as e:
        logger.debug(f"Failed to format and print summary: {e}")

