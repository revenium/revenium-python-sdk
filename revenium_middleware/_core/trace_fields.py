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
from .fields import extract_agent_version_field

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
# Mirrors ReveniumAttributes.MAX_AGENT_VERSION_LENGTH on the metering
# service, which caps agentVersion on all four /v2/ai ingest bodies.
AGENT_VERSION_MAX_LENGTH = 64
TRACE_TYPE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def _per_call(usage_metadata: Optional[Dict[str, Any]], *aliases: str) -> Any:
    """The first set value of ``aliases`` in the call's metadata, or None."""
    if not usage_metadata:
        return None
    for alias in aliases:
        value = usage_metadata.get(alias)
        if value:
            return value
    return None


def get_environment(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get deployment environment, the call's own value first.

    Checks in order:
    1. environment from usage_metadata
    2. REVENIUM_ENVIRONMENT
    3. ENVIRONMENT
    4. DEPLOYMENT_ENV

    Returns:
        Environment name (e.g., 'production', 'staging') or None
    """
    return (
        _per_call(usage_metadata, 'environment') or
        os.getenv(ENV_REVENIUM_ENVIRONMENT) or
        os.getenv(ENV_ENVIRONMENT) or
        os.getenv(ENV_DEPLOYMENT_ENV)
    )


def get_region(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get cloud region, the call's own value first.

    Checks in order:
    1. region from usage_metadata
    2. REVENIUM_REGION
    3. AWS_REGION or AWS_DEFAULT_REGION
    4. AZURE_REGION
    5. GCP_REGION or GOOGLE_CLOUD_REGION

    Returns:
        Region name (e.g., 'us-east-1', 'eastus') or None
    """
    region = _per_call(usage_metadata, 'region') or os.getenv(ENV_REVENIUM_REGION)
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


def get_credential_alias(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get credential alias, the call's own value first, then the environment.

    Returns:
        Credential alias (e.g., 'prod-api-key', 'staging-key') or None
    """
    return (
        _per_call(usage_metadata, 'credentialAlias', 'credential_alias') or
        os.getenv(ENV_REVENIUM_CREDENTIAL_ALIAS)
    )


def get_trace_type(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get and validate trace type, the call's own value first, then the environment.

    An invalid per-call value is omitted rather than replaced by the
    environment's.

    Returns:
        Validated trace type or None if invalid/not set
    """
    trace_type = _per_call(usage_metadata, 'traceType', 'trace_type') or os.getenv(ENV_REVENIUM_TRACE_TYPE)
    if trace_type:
        return validate_trace_type(trace_type)
    return None


def get_trace_name(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get and validate trace name, the call's own value first, then the environment.

    Returns:
        Validated trace name (truncated if needed) or None if not set
    """
    trace_name = _per_call(usage_metadata, 'traceName', 'trace_name') or os.getenv(ENV_REVENIUM_TRACE_NAME)
    if trace_name:
        return validate_trace_name(trace_name)
    return None


def get_parent_transaction_id(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get parent transaction ID, the call's own value first, then the environment.

    Returns:
        Parent transaction ID or None
    """
    return (
        _per_call(usage_metadata, 'parentTransactionId', 'parent_transaction_id') or
        os.getenv(ENV_REVENIUM_PARENT_TRANSACTION_ID)
    )


def get_transaction_name(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get transaction name with fallback to task_type.

    Checks in order:
    1. transactionName / transaction_name from usage_metadata
    2. REVENIUM_TRANSACTION_NAME env var
    3. task_type from usage_metadata (fallback)

    Args:
        usage_metadata: Optional metadata dictionary

    Returns:
        Transaction name or None
    """
    return (
        _per_call(usage_metadata, 'transactionName', 'transaction_name') or
        os.getenv(ENV_REVENIUM_TRANSACTION_NAME) or
        _per_call(usage_metadata, 'task_type', 'taskType')
    )


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


def get_agent_version(
    usage_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Get the AI agent's own version from usage metadata.

    Reads ``agent_version`` then its ``agentVersion`` alias through
    ``fields.AGENT_VERSION_FIELD_MAP``, the single declaration of the accepted
    aliases. Unlike ticketId there is no environment-variable fallback: the
    agent version is per-call attribution the middleware cannot infer.

    This is not ``agenticJobVersion`` (the agentic job definition's version).
    The two are separate wire fields and must not be conflated.

    Args:
        usage_metadata: Optional metadata dictionary

    Returns:
        Validated agent version (truncated if needed) or None if not set
    """
    if not usage_metadata:
        return None
    agent_version = extract_agent_version_field(usage_metadata).get('agent_version')
    if agent_version:
        return validate_agent_version(agent_version)
    return None


def _per_call_retry_number(usage_metadata: Optional[Dict[str, Any]]) -> Optional[int]:
    if not usage_metadata:
        return None
    value = usage_metadata.get('retryNumber')
    if value is None:
        value = usage_metadata.get('retry_number')
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    logger.warning("Invalid retry number in usage_metadata, using REVENIUM_RETRY_NUMBER")
    return None


def get_retry_number(usage_metadata: Optional[Dict[str, Any]] = None) -> int:
    """
    Get retry number, the call's own value first, then the environment.

    Returns:
        Retry number (0 for first attempt, 1+ for retries)
    """
    per_call = _per_call_retry_number(usage_metadata)
    if per_call is not None:
        return per_call
    try:
        return int(os.getenv(ENV_REVENIUM_RETRY_NUMBER, '0'))
    except ValueError:
        logger.warning(
            "Invalid REVENIUM_RETRY_NUMBER value, defaulting to 0"
        )
        return 0


def _warn_not_a_string(name: str, value: Any) -> None:
    logger.warning(
        f"{name} must be a string, got {type(value).__name__}. Field will be omitted."
    )


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
    if not isinstance(trace_type, str):
        _warn_not_a_string("traceType", trace_type)
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
    if not isinstance(trace_name, str):
        _warn_not_a_string("traceName", trace_name)
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


def validate_ticket_id(ticket_id: Any) -> Optional[str]:
    """
    Validate ticket ID length and truncate if needed.

    Rules:
    - Non-string values are dropped, never coerced (as validate_agent_version)
    - Maximum 256 characters
    - Truncates with warning if too long

    Args:
        ticket_id: Caller-supplied ticket ID, straight from ``usage_metadata``

    Returns:
        Valid ticket ID (truncated if needed) or None if empty or non-string
    """
    if ticket_id is None:
        return None

    if not isinstance(ticket_id, str):
        logger.debug(
            "Ignoring non-string ticketId of type %s; "
            "ticket attribution dropped for this event.",
            type(ticket_id).__name__,
        )
        return None

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


def validate_agent_version(agent_version: Any) -> Optional[str]:
    """
    Validate agent version length and truncate if needed.

    Rules:
    - Non-string values are dropped, never coerced
    - Maximum AGENT_VERSION_MAX_LENGTH characters
    - Truncates with warning if too long

    Args:
        agent_version: Caller-supplied agent version. Typed as Any rather
            than str because it arrives straight from ``usage_metadata``,
            which the caller controls and the SDK does not validate upstream.

    Returns:
        Valid agent version (truncated if needed) or None if empty,
        non-string, or otherwise unusable
    """
    if agent_version is None:
        return None

    if not isinstance(agent_version, str):
        # Drop, never coerce: str(123) would invent attribution the caller
        # never wrote. Dropping keeps the metering event and its cost
        # flowing, which matters here because this runs after the provider
        # call has already returned (and again on stream finalisation) --
        # a TypeError at this point would surface as the failure of a call
        # that actually succeeded.
        logger.debug(
            "Ignoring non-string agentVersion of type %s; "
            "agent version attribution dropped for this event.",
            type(agent_version).__name__,
        )
        return None

    if not agent_version:
        return None

    if len(agent_version) > AGENT_VERSION_MAX_LENGTH:
        logger.warning(
            f"agentVersion exceeds maximum length of "
            f"{AGENT_VERSION_MAX_LENGTH} characters. "
            f"Truncating from {len(agent_version)} to "
            f"{AGENT_VERSION_MAX_LENGTH} characters."
        )
        return agent_version[:AGENT_VERSION_MAX_LENGTH]

    return agent_version
