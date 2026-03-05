"""
Summary printer for terminal output of API usage metrics.

This module provides functionality to display cost and metrics information
after each API request in human-readable or JSON format.
"""

import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import requests

from . import trace_fields

logger = logging.getLogger("revenium_middleware.extension")


@dataclass
class CompletionMetrics:
    """Data class to hold cost information from the Revenium API."""

    total_cost: Optional[float] = None


def fetch_completion_metrics(
    transaction_id: str,
    revenium_api_key: str,
) -> Optional[CompletionMetrics]:
    """
    Fetch cost data from Revenium profitstream API.

    Implements retry logic (3 attempts with 1-second delay).

    Args:
        transaction_id: The transaction ID to fetch metrics for
        revenium_api_key: The Revenium API key for authentication

    Returns:
        CompletionMetrics with cost data, or None if fetch fails
    """
    team_id = trace_fields.get_team_id()
    if not team_id:
        logger.debug("No REVENIUM_TEAM_ID set, skipping cost fetch")
        return CompletionMetrics(total_cost=None)

    base_url = trace_fields.get_base_url()
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

    for attempt in range(trace_fields.SUMMARY_RETRY_ATTEMPTS):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=trace_fields.SUMMARY_API_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                # Check both camelCase and snake_case
                # Don't use 'or' to avoid treating 0.0 as falsy
                total_cost = data.get("totalCost")
                if total_cost is None:
                    total_cost = data.get("total_cost")
                return CompletionMetrics(total_cost=total_cost)
            else:
                logger.debug(
                    f"Metrics API returned status {response.status_code} "
                    f"on attempt {attempt + 1}"
                )

        except requests.exceptions.RequestException as e:
            logger.debug(
                f"Failed to fetch metrics on attempt {attempt + 1}: {e}"
            )

        # Wait before retry (except on last attempt)
        if attempt < trace_fields.SUMMARY_RETRY_ATTEMPTS - 1:
            time.sleep(trace_fields.SUMMARY_RETRY_DELAY)

    logger.debug(
        f"Failed to fetch metrics after {trace_fields.SUMMARY_RETRY_ATTEMPTS} attempts"
    )
    return None


def format_and_print_json_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    cost: Optional[float],
    trace_id: Optional[str],
) -> None:
    """
    Print single-line JSON output for machine-readable summary.

    Args:
        model: Model name used
        provider: Provider name (e.g., "GOOGLE")
        duration_seconds: Request duration in seconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total token count
        cost: Cost in dollars (or None if unavailable)
        trace_id: Trace ID for the request
    """
    summary = {
        "model": model,
        "provider": provider,
        "durationSeconds": round(duration_seconds, 3),
        "inputTokenCount": input_token_count,
        "outputTokenCount": output_token_count,
        "totalTokenCount": total_token_count,
    }

    if cost is not None:
        summary["cost"] = cost
        summary["costStatus"] = "available"
    else:
        summary["costStatus"] = "unavailable"

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
    cost: Optional[float],
    trace_id: Optional[str],
    team_id: Optional[str],
) -> None:
    """
    Print professional human-readable output.

    NO EMOJIS - professional format only.

    Args:
        model: Model name used
        provider: Provider name (e.g., "GOOGLE")
        duration_seconds: Request duration in seconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total token count
        cost: Cost in dollars (or None if unavailable)
        trace_id: Trace ID for the request
        team_id: Team ID (used for cost status message)
    """
    separator = "=" * 60

    print(separator)
    print("REVENIUM USAGE SUMMARY")
    print(separator)
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Duration: {duration_seconds:.2f}s")
    print()
    print("Token Usage:")
    print(f"  Input Tokens:  {input_token_count or 0}")
    print(f"  Output Tokens: {output_token_count or 0}")
    print(f"  Total Tokens:  {total_token_count or 0}")
    print()

    if cost is not None:
        print(f"Cost: ${cost:.6f}")
    elif team_id:
        print("Cost: Pending (aggregating... check Revenium dashboard)")
    else:
        print("Cost: Add REVENIUM_TEAM_ID to see pricing")

    if trace_id:
        print()
        print(f"Trace ID: {trace_id}")

    print(separator)


def print_usage_summary(
    model: str,
    provider: str,
    request_duration: int,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    transaction_id: str,
    trace_id: Optional[str],
    revenium_api_key: str,
) -> None:
    """
    Main entry point for printing usage summary.

    Fire-and-forget: Wrapped in try/except to never fail the main API call.

    Args:
        model: Model name used
        provider: Provider name (e.g., "Google")
        request_duration: Request duration in milliseconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total token count
        transaction_id: Transaction ID for fetching metrics
        trace_id: Trace ID for the request
        revenium_api_key: Revenium API key for authentication
    """
    try:
        # Check if summary output is enabled
        summary_format = trace_fields.get_print_summary_config()
        if summary_format is False:
            return

        # Convert duration from milliseconds to seconds
        duration_seconds = request_duration / 1000.0

        # Attempt to fetch cost metrics
        metrics = fetch_completion_metrics(transaction_id, revenium_api_key)
        cost = metrics.total_cost if metrics else None
        team_id = trace_fields.get_team_id()

        if summary_format == "json":
            format_and_print_json_summary(
                model=model,
                provider=provider,
                duration_seconds=duration_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                total_token_count=total_token_count,
                cost=cost,
                trace_id=trace_id,
            )
        else:  # "human"
            format_and_print_human_summary(
                model=model,
                provider=provider,
                duration_seconds=duration_seconds,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                total_token_count=total_token_count,
                cost=cost,
                trace_id=trace_id,
                team_id=team_id,
            )

    except Exception as e:
        # Fire-and-forget: Never fail the main operation
        logger.debug(f"Failed to print summary: {e}")

