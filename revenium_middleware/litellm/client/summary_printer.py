"""
Terminal summary output module for Revenium LiteLLM middleware.

This module provides functionality to print usage summaries after API requests
in either human-readable or JSON format.
"""

import json
import logging
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional

from .config import (
    get_print_summary_config,
    get_team_id,
    get_base_url,
    SUMMARY_RETRY_ATTEMPTS,
    SUMMARY_RETRY_DELAY,
    SUMMARY_API_TIMEOUT,
)

logger = logging.getLogger(__name__)


class CompletionMetrics:
    """Data class to hold cost information from Revenium API."""

    def __init__(self, total_cost: Optional[float] = None):
        self.total_cost = total_cost


def fetch_completion_metrics(
    transaction_id: str,
    revenium_api_key: str,
) -> CompletionMetrics:
    """
    Fetch cost data from Revenium profitstream API.

    Args:
        transaction_id: The transaction ID to fetch metrics for
        revenium_api_key: The Revenium API key

    Returns:
        CompletionMetrics with cost data (total_cost=None if fetch fails)
    """
    team_id = get_team_id()
    if not team_id:
        logger.debug("No team ID configured, cannot fetch cost metrics")
        return CompletionMetrics(total_cost=None)

    base_url = get_base_url()
    # Use urlencode to properly encode query parameters
    params = urllib.parse.urlencode({
        'teamId': team_id,
        'transactionId': transaction_id
    })
    url = (
        f"{base_url}/profitstream/v2/api/sources/metrics/ai/completions"
        f"?{params}"
    )

    headers = {
        "Authorization": f"Bearer {revenium_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(SUMMARY_RETRY_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(
                request, timeout=SUMMARY_API_TIMEOUT
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
                # Check both camelCase and snake_case
                # Don't use 'or' to avoid treating 0.0 as falsy
                total_cost = data.get("totalCost")
                if total_cost is None:
                    total_cost = data.get("total_cost")
                return CompletionMetrics(total_cost=total_cost)
        except urllib.error.HTTPError as e:
            logger.debug(f"HTTP error fetching metrics (attempt {attempt + 1}): {e}")
        except urllib.error.URLError as e:
            logger.debug(f"URL error fetching metrics (attempt {attempt + 1}): {e}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.debug(f"Error fetching metrics (attempt {attempt + 1}): {e}")

        if attempt < SUMMARY_RETRY_ATTEMPTS - 1:
            time.sleep(SUMMARY_RETRY_DELAY)

    # Return consistent type - CompletionMetrics with None cost
    return CompletionMetrics(total_cost=None)


def format_and_print_json_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    cost: Optional[float],
    cost_status: str,
    trace_id: Optional[str],
) -> None:
    """
    Print single-line JSON output with usage summary.

    Args:
        model: The model name
        provider: The provider name
        duration_seconds: Request duration in seconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total tokens
        cost: Cost value or None
        cost_status: Status string for cost ('available', 'pending', 'unavailable')
        trace_id: Optional trace ID
    """
    output = {
        "model": model,
        "provider": provider,
        "durationSeconds": round(duration_seconds, 3),
        "inputTokenCount": input_token_count,
        "outputTokenCount": output_token_count,
        "totalTokenCount": total_token_count,
    }

    if cost is not None:
        output["cost"] = cost
    output["costStatus"] = cost_status

    if trace_id:
        output["traceId"] = trace_id

    print(json.dumps(output, separators=(",", ":")))


def format_and_print_human_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    cost: Optional[float],
    cost_status: str,
    trace_id: Optional[str],
    unavailable_reason: Optional[str] = None,
) -> None:
    """
    Print professional human-readable output with usage summary.

    Args:
        model: The model name
        provider: The provider name
        duration_seconds: Request duration in seconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total tokens
        cost: Cost value or None
        cost_status: Status string for cost
        trace_id: Optional trace ID
        unavailable_reason: Reason cost is unavailable
    """
    separator = "=" * 60

    lines = [
        separator,
        "REVENIUM USAGE SUMMARY",
        separator,
        f"Model: {model}",
        f"Provider: {provider}",
        f"Duration: {duration_seconds:.2f}s",
        "",
        "Token Usage:",
    ]

    # Handle null token counts gracefully
    input_str = str(input_token_count) if input_token_count is not None else "N/A"
    output_str = str(output_token_count) if output_token_count is not None else "N/A"
    total_str = str(total_token_count) if total_token_count is not None else "N/A"

    lines.extend([
        f"  Input Tokens:  {input_str}",
        f"  Output Tokens: {output_str}",
        f"  Total Tokens:  {total_str}",
        "",
    ])

    # Format cost based on status
    if cost is not None:
        lines.append(f"Cost: ${cost:.6f}")
    elif cost_status == "pending":
        lines.append("Cost: Pending (aggregating... check Revenium dashboard)")
    else:
        # Provide specific message based on what's missing
        if unavailable_reason == "api_key_missing":
            lines.append("Cost: Add REVENIUM_METERING_API_KEY to see pricing")
        elif unavailable_reason == "team_id_missing":
            lines.append("Cost: Add REVENIUM_TEAM_ID to see pricing")
        else:
            lines.append("Cost: Add REVENIUM_TEAM_ID to see pricing")

    if trace_id:
        lines.extend(["", f"Trace ID: {trace_id}"])

    lines.append(separator)

    print("\n".join(lines))


def print_usage_summary(
    model: str,
    provider: str,
    request_duration: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    transaction_id: str,
    trace_id: Optional[str],
    revenium_api_key: Optional[str],
) -> None:
    """
    Main entry point for printing usage summary.

    This function is fire-and-forget - it will never raise exceptions to the caller.

    Args:
        model: The model name
        provider: The provider name
        request_duration: Request duration in milliseconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total tokens
        transaction_id: The transaction ID
        trace_id: Optional trace ID
        revenium_api_key: The Revenium API key (optional)
    """
    try:
        summary_config = get_print_summary_config()

        # If disabled, return immediately
        if summary_config is False:
            return

        # Convert duration from milliseconds to seconds
        duration_seconds = request_duration / 1000.0

        # Determine cost status and fetch metrics if team_id is available
        team_id = get_team_id()
        cost: Optional[float] = None
        cost_status: str = "unavailable"
        # Track reason for unavailability for better user messaging
        unavailable_reason: Optional[str] = None

        if team_id and revenium_api_key:
            metrics = fetch_completion_metrics(transaction_id, revenium_api_key)
            if metrics and metrics.total_cost is not None:
                cost = metrics.total_cost
                cost_status = "available"
            else:
                cost_status = "pending"
        else:
            cost_status = "unavailable"
            if not revenium_api_key:
                unavailable_reason = "api_key_missing"
            elif not team_id:
                unavailable_reason = "team_id_missing"

        # Print in the appropriate format
        if summary_config == "json":
            format_and_print_json_summary(
                model=model,
                provider=provider,
                duration_seconds=duration_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                total_token_count=total_token_count,
                cost=cost,
                cost_status=cost_status,
                trace_id=trace_id,
            )
        else:  # human format
            format_and_print_human_summary(
                model=model,
                provider=provider,
                duration_seconds=duration_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                total_token_count=total_token_count,
                cost=cost,
                cost_status=cost_status,
                trace_id=trace_id,
                unavailable_reason=unavailable_reason,
            )

    except Exception as e:
        # Fire-and-forget: log but never propagate exceptions
        logger.debug(f"Failed to print summary: {e}")


__all__ = [
    'CompletionMetrics',
    'fetch_completion_metrics',
    'format_and_print_json_summary',
    'format_and_print_human_summary',
    'print_usage_summary',
]

