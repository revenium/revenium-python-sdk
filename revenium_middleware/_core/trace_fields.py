"""
Shared trace visualization field capture and validation.

This module provides the canonical implementations of trace field functions
shared across all provider middlewares. Provider-specific trace_fields modules
import from here and re-export, keeping only provider-specific functions.
"""

import os
import re
import logging
from typing import Optional, Dict, Any

from .config import Config

logger = logging.getLogger(__name__)

# Re-export environment variable names from core config
ENV_REVENIUM_ENVIRONMENT = Config.ENV_REVENIUM_ENVIRONMENT
ENV_ENVIRONMENT = Config.ENV_ENVIRONMENT
ENV_DEPLOYMENT_ENV = Config.ENV_DEPLOYMENT_ENV

ENV_REVENIUM_REGION = Config.ENV_REVENIUM_REGION
ENV_AWS_REGION = "AWS_REGION"
ENV_AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
ENV_AZURE_REGION = "AZURE_REGION"
ENV_GCP_REGION = "GCP_REGION"
ENV_GOOGLE_CLOUD_REGION = "GOOGLE_CLOUD_REGION"

ENV_REVENIUM_CREDENTIAL_ALIAS = Config.ENV_REVENIUM_CREDENTIAL_ALIAS
ENV_REVENIUM_TRACE_TYPE = Config.ENV_REVENIUM_TRACE_TYPE
ENV_REVENIUM_TRACE_NAME = Config.ENV_REVENIUM_TRACE_NAME
ENV_REVENIUM_PARENT_TRANSACTION_ID = Config.ENV_REVENIUM_PARENT_TRANSACTION_ID
ENV_REVENIUM_TRANSACTION_NAME = Config.ENV_REVENIUM_TRANSACTION_NAME
ENV_REVENIUM_RETRY_NUMBER = Config.ENV_REVENIUM_RETRY_NUMBER
ENV_REVENIUM_TICKET_ID = Config.ENV_REVENIUM_TICKET_ID

# Validation constants
TRACE_TYPE_MAX_LENGTH = 128
TRACE_NAME_MAX_LENGTH = 256
TICKET_ID_MAX_LENGTH = 256
TRACE_TYPE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def get_environment() -> Optional[str]:
    """
    Get deployment environment from environment variables.

    Checks in order:
    1. REVENIUM_ENVIRONMENT
    2. ENVIRONMENT
    3. DEPLOYMENT_ENV

    Returns:
        Environment name (e.g., 'production', 'staging') or None
    """
    return (
        os.getenv(ENV_REVENIUM_ENVIRONMENT) or
        os.getenv(ENV_ENVIRONMENT) or
        os.getenv(ENV_DEPLOYMENT_ENV)
    )


def get_region() -> Optional[str]:
    """
    Get cloud region from environment variables.

    Checks in order:
    1. REVENIUM_REGION
    2. AWS_REGION or AWS_DEFAULT_REGION
    3. AZURE_REGION
    4. GCP_REGION or GOOGLE_CLOUD_REGION

    Returns:
        Region name (e.g., 'us-east-1', 'eastus') or None
    """
    # Try Revenium-specific env var first
    region = os.getenv(ENV_REVENIUM_REGION)
    if region:
        return region

    # Try AWS region
    region = os.getenv(ENV_AWS_REGION) or os.getenv(ENV_AWS_DEFAULT_REGION)
    if region:
        return region

    # Try Azure region
    region = os.getenv(ENV_AZURE_REGION)
    if region:
        return region

    # Try GCP region
    region = os.getenv(ENV_GCP_REGION) or os.getenv(ENV_GOOGLE_CLOUD_REGION)
    if region:
        return region

    return None


def get_credential_alias() -> Optional[str]:
    """
    Get credential alias from environment variables.

    Returns:
        Credential alias (e.g., 'prod-api-key', 'staging-key') or None
    """
    return os.getenv(ENV_REVENIUM_CREDENTIAL_ALIAS)


def get_trace_type() -> Optional[str]:
    """
    Get and validate trace type from environment variables.

    Returns:
        Validated trace type or None if invalid/not set
    """
    trace_type = os.getenv(ENV_REVENIUM_TRACE_TYPE)
    if trace_type:
        return validate_trace_type(trace_type)
    return None


def get_trace_name() -> Optional[str]:
    """
    Get and validate trace name from environment variables.

    Returns:
        Validated trace name (truncated if needed) or None if not set
    """
    trace_name = os.getenv(ENV_REVENIUM_TRACE_NAME)
    if trace_name:
        return validate_trace_name(trace_name)
    return None


def get_parent_transaction_id() -> Optional[str]:
    """
    Get parent transaction ID from environment variables.

    Returns:
        Parent transaction ID or None
    """
    return os.getenv(ENV_REVENIUM_PARENT_TRANSACTION_ID)


def get_transaction_name(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get transaction name with fallback to task_type.

    Checks in order:
    1. REVENIUM_TRANSACTION_NAME env var
    2. transactionName from usage_metadata
    3. task_type from usage_metadata (fallback)

    Args:
        usage_metadata: Optional metadata dictionary

    Returns:
        Transaction name or None
    """
    # First priority: env var
    transaction_name = os.getenv(ENV_REVENIUM_TRANSACTION_NAME)
    if transaction_name:
        return transaction_name

    # Second priority: usage_metadata
    if usage_metadata:
        transaction_name = (
            usage_metadata.get('transactionName') or
            usage_metadata.get('transaction_name')
        )
        if transaction_name:
            return transaction_name

        # Third priority: fallback to task_type
        task_type = (
            usage_metadata.get('task_type') or
            usage_metadata.get('taskType')
        )
        if task_type:
            return task_type

    return None


def get_ticket_id(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get ticket ID from usage metadata or environment variables.

    Checks in order:
    1. ticketId / ticket_id from usage_metadata
    2. REVENIUM_TICKET_ID env var

    Args:
        usage_metadata: Optional metadata dictionary

    Returns:
        Validated ticket ID (truncated if needed) or None if not set
    """
    ticket_id = None
    if usage_metadata:
        ticket_id = (
            usage_metadata.get('ticketId') or
            usage_metadata.get('ticket_id')
        )
    if not ticket_id:
        ticket_id = os.getenv(ENV_REVENIUM_TICKET_ID)
    if ticket_id:
        return validate_ticket_id(ticket_id)
    return None


def get_retry_number() -> int:
    """
    Get retry number from environment variables.

    Returns:
        Retry number (0 for first attempt, 1+ for retries)
    """
    try:
        return int(os.getenv(ENV_REVENIUM_RETRY_NUMBER, '0'))
    except ValueError:
        logger.warning(
            "Invalid REVENIUM_RETRY_NUMBER value, defaulting to 0"
        )
        return 0


def validate_trace_type(trace_type: str) -> Optional[str]:
    """
    Validate trace type format and length.

    Rules:
    - Only alphanumeric characters, hyphens, and underscores
    - Maximum 128 characters

    Args:
        trace_type: Trace type to validate

    Returns:
        Valid trace type or None if invalid
    """
    if not trace_type:
        return None

    # Check length
    if len(trace_type) > TRACE_TYPE_MAX_LENGTH:
        logger.warning(
            f"traceType exceeds maximum length of "
            f"{TRACE_TYPE_MAX_LENGTH} characters: '{trace_type}'. "
            f"Field will be omitted."
        )
        return None

    # Check format
    if not TRACE_TYPE_PATTERN.match(trace_type):
        logger.warning(
            f"traceType contains invalid characters "
            f"(only alphanumeric, hyphens, and underscores allowed): "
            f"'{trace_type}'. Field will be omitted."
        )
        return None

    return trace_type


def validate_trace_name(trace_name: str) -> Optional[str]:
    """
    Validate trace name length and truncate if needed.

    Rules:
    - Maximum 256 characters
    - Truncates with warning if too long

    Args:
        trace_name: Trace name to validate

    Returns:
        Valid trace name (truncated if needed) or None if empty
    """
    if not trace_name:
        return None

    # Check length and truncate if needed
    if len(trace_name) > TRACE_NAME_MAX_LENGTH:
        logger.warning(
            f"traceName exceeds maximum length of "
            f"{TRACE_NAME_MAX_LENGTH} characters. "
            f"Truncating from {len(trace_name)} to "
            f"{TRACE_NAME_MAX_LENGTH} characters."
        )
        return trace_name[:TRACE_NAME_MAX_LENGTH]

    return trace_name


def validate_ticket_id(ticket_id: str) -> Optional[str]:
    """
    Validate ticket ID length and truncate if needed.

    Rules:
    - Maximum 256 characters
    - Truncates with warning if too long

    Args:
        ticket_id: Ticket ID to validate

    Returns:
        Valid ticket ID (truncated if needed) or None if empty
    """
    if not ticket_id:
        return None

    if len(ticket_id) > TICKET_ID_MAX_LENGTH:
        logger.warning(
            f"ticketId exceeds maximum length of "
            f"{TICKET_ID_MAX_LENGTH} characters. "
            f"Truncating from {len(ticket_id)} to "
            f"{TICKET_ID_MAX_LENGTH} characters."
        )
        return ticket_id[:TICKET_ID_MAX_LENGTH]

    return ticket_id
