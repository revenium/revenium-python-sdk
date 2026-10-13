"""
Shared subscriber extraction from usage metadata.

This module provides the canonical implementation of subscriber extraction
shared across all provider middlewares.
"""

import logging

logger = logging.getLogger("revenium_middleware")


def extract_subscriber_from_metadata(usage_metadata: dict) -> dict:
    """
    Extract subscriber from usage_metadata.

    Supports both nested and flat patterns:
    - Nested: usage_metadata["subscriber"]["id"], ["email"], ["credential"]
    - Flat: usage_metadata["subscriber_id"], ["subscriber_email"], etc.

    Args:
        usage_metadata: Dictionary containing usage metadata

    Returns:
        Dictionary containing subscriber information (id, email, credential)
    """
    subscriber = {}

    # Pattern 1: Nested subscriber object (OpenAI, Ollama, Anthropic non-Bedrock, LiteLLM)
    if "subscriber" in usage_metadata and isinstance(usage_metadata["subscriber"], dict):
        nested = usage_metadata["subscriber"]

        if nested.get("id"):
            subscriber["id"] = nested["id"]
        if nested.get("email"):
            subscriber["email"] = nested["email"]
        if nested.get("credential") and isinstance(nested["credential"], dict):
            subscriber["credential"] = {
                "name": nested["credential"].get("name"),
                "value": nested["credential"].get("value"),
            }

    # Pattern 2: Flat field pattern (Anthropic Bedrock legacy)
    if not subscriber:
        if usage_metadata.get("subscriber_id"):
            subscriber["id"] = usage_metadata["subscriber_id"]
        if usage_metadata.get("subscriber_email"):
            subscriber["email"] = usage_metadata["subscriber_email"]
        if usage_metadata.get("subscriber_credential_name") or usage_metadata.get("subscriber_credential"):
            subscriber["credential"] = {
                "name": usage_metadata.get("subscriber_credential_name"),
                "value": usage_metadata.get("subscriber_credential"),
            }

    return subscriber


def subscriber_with_credential_alias(subscriber, credential_alias):
    """
    Return the subscriber to send, carrying the credential alias as its credential name.

    Revenium stores a credential's name only at ``subscriber.credential.name``,
    so the alias travels there. Revenium looks a credential up by that name
    before it looks at the subscriber id or email, so the alias is never sent
    on a subscriber that carries an id or email: each subscriber keeps its own
    credential. A credential the caller named or gave a value for wins over the
    alias. The caller's dicts are never modified.

    Args:
        subscriber: The subscriber as given (a dict, NOT_GIVEN or None)
        credential_alias: The alias; anything other than a non-blank str is ignored

    Returns:
        The subscriber unchanged when there is no usable alias, the subscriber
        has a non-blank id or email, the credential is already named or valued,
        or the caller's subscriber or credential is not a dict; otherwise a copy
        whose credential name is the stripped alias.
    """
    if not isinstance(credential_alias, str) or not credential_alias.strip():
        return subscriber
    alias = credential_alias.strip()

    # Imported here: the metering client imports this module while its own package is initialising.
    from revenium_middleware._metering._types import NotGiven

    if subscriber is None or isinstance(subscriber, NotGiven):
        return {"credential": {"name": alias}}
    if not isinstance(subscriber, dict):
        logger.debug("Credential alias not applied: subscriber is not a dict")
        return subscriber
    if _has_subscriber_identity(subscriber):
        return subscriber

    credential = subscriber.get("credential")
    if credential is None:
        credential = {}
    if not isinstance(credential, dict):
        logger.debug("Credential alias not applied: subscriber credential is not a dict")
        return subscriber
    if credential.get("name") or _is_non_empty_str(credential.get("value")):
        return subscriber

    return {**subscriber, "credential": {**credential, "name": alias}}


def _is_non_empty_str(value) -> bool:
    return isinstance(value, str) and value != ""


def _has_subscriber_identity(subscriber: dict) -> bool:
    return any(_is_identity(subscriber.get(key)) for key in ("id", "email"))


def _is_identity(value) -> bool:
    if isinstance(value, str):
        return value.strip() != ""
    return value is not None
