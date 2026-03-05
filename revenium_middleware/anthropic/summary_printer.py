"""
Summary printer module for Revenium Anthropic middleware.

This module provides terminal summary output functionality that displays
cost and metrics information after each API request in either JSON or
human-readable format.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
import urllib.request
import urllib.error

from .config import (
    Config,
    get_base_url,
    get_print_summary_config,
    get_team_id,
)

logger = logging.getLogger("revenium_middleware.summary_printer")


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

    Implements retry logic (3 attempts with 2-second delay).

    Args:
        transaction_id: The transaction ID to fetch metrics for
        revenium_api_key: The Revenium API key for authentication

    Returns:
        CompletionMetrics if successful, None if fetch fails or cost not available
    """
    team_id = get_team_id()
    if not team_id:
        logger.debug("No team ID configured, skipping cost fetch")
        return None

    base_url = get_base_url()
    url = (
        f"{base_url}/profitstream/v2/api/sources/metrics/ai/completions"
        f"?teamId={team_id}&transactionId={transaction_id}"
    )

    headers = {
        "Authorization": f"Bearer {revenium_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(Config.SUMMARY_RETRY_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(
                request, timeout=Config.SUMMARY_API_TIMEOUT
            ) as response:
                data = json.loads(response.read().decode("utf-8"))

                # Extract cost from response
                total_cost = data.get("totalCost")
                if total_cost is not None:
                    return CompletionMetrics(total_cost=float(total_cost))

                # Cost might be in a nested structure
                if isinstance(data, dict):
                    # Try common field names
                    for field in ["cost", "total_cost", "totalCost"]:
                        if field in data and data[field] is not None:
                            return CompletionMetrics(total_cost=float(data[field]))

                return CompletionMetrics(total_cost=None)

        except urllib.error.HTTPError as e:
            logger.debug(f"HTTP error fetching metrics (attempt {attempt + 1}): {e}")
        except urllib.error.URLError as e:
            logger.debug(f"URL error fetching metrics (attempt {attempt + 1}): {e}")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.debug(f"Error fetching metrics (attempt {attempt + 1}): {e}")

        # Wait before retry (except on last attempt)
        if attempt < Config.SUMMARY_RETRY_ATTEMPTS - 1:
            time.sleep(Config.SUMMARY_RETRY_DELAY)

    return None


def format_and_print_json_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    cost: Optional[float],
    cost_status: str,
    trace_id: Optional[str] = None,
) -> None:
    """
    Print single-line JSON output summary.

    Args:
        model: The model name used
        provider: The provider name (e.g., ANTHROPIC, BEDROCK)
        duration_seconds: Request duration in seconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total token count
        cost: Total cost if available
        cost_status: Status of cost retrieval
        trace_id: Optional trace ID
    """
    summary = {
        "model": model,
        "provider": provider,
        "durationSeconds": round(duration_seconds, 3),
        "inputTokenCount": input_token_count,
        "outputTokenCount": output_token_count,
        "totalTokenCount": total_token_count,
        "cost": cost,
        "costStatus": cost_status,
    }

    if trace_id:
        summary["traceId"] = trace_id

    print(json.dumps(summary, separators=(",", ":")))


def format_and_print_human_summary(
    model: str,
    provider: str,
    duration_seconds: float,
    input_token_count: Optional[int],
    output_token_count: Optional[int],
    total_token_count: Optional[int],
    cost: Optional[float],
    cost_status: str,
    trace_id: Optional[str] = None,
) -> None:
    """
    Print professional human-readable output summary.

    NO EMOJIS - professional format only.

    Args:
        model: The model name used
        provider: The provider name (e.g., ANTHROPIC, BEDROCK)
        duration_seconds: Request duration in seconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total token count
        cost: Total cost if available
        cost_status: Status of cost retrieval
        trace_id: Optional trace ID
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
        f"  Input Tokens:  {input_token_count if input_token_count is not None else 'N/A'}",
        f"  Output Tokens: {output_token_count if output_token_count is not None else 'N/A'}",
        f"  Total Tokens:  {total_token_count if total_token_count is not None else 'N/A'}",
        "",
    ]

    # Add cost information
    if cost is not None:
        lines.append(f"Cost: ${cost:.6f}")
    else:
        lines.append(f"Cost: {cost_status}")

    # Add trace ID if available
    if trace_id:
        lines.append("")
        lines.append(f"Trace ID: {trace_id}")

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
    trace_id: Optional[str] = None,
    revenium_api_key: Optional[str] = None,
) -> None:
    """
    Main entry point for printing usage summary.

    This is a fire-and-forget function - it never raises exceptions to the caller.

    Args:
        model: The model name used
        provider: The provider name (e.g., ANTHROPIC, BEDROCK)
        request_duration: Request duration in milliseconds
        input_token_count: Number of input tokens
        output_token_count: Number of output tokens
        total_token_count: Total token count
        transaction_id: The transaction ID for fetching cost
        trace_id: Optional trace ID
        revenium_api_key: The Revenium API key for fetching cost
    """
    try:
        # Check if summary printing is enabled
        summary_format = get_print_summary_config()
        if summary_format is False:
            return

        # Convert duration from milliseconds to seconds
        duration_seconds = request_duration / 1000.0

        # Determine cost status and fetch cost if possible
        cost: Optional[float] = None
        cost_status: str = "unavailable"
        team_id = get_team_id()

        if revenium_api_key and team_id:
            metrics = fetch_completion_metrics(transaction_id, revenium_api_key)
            if metrics and metrics.total_cost is not None:
                cost = metrics.total_cost
                cost_status = "available"
            else:
                cost_status = "Pending (aggregating... check Revenium dashboard)"
        elif not team_id:
            cost_status = "Add REVENIUM_TEAM_ID to see pricing"

        # Print in the appropriate format
        if summary_format == "json":
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
        else:
            # Default to human format
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
            )

    except Exception as e:
        # Fire-and-forget: never fail the main API call
        logger.debug(f"Failed to print summary: {e}")
