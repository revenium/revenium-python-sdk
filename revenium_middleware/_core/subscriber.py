"""
Shared subscriber extraction from usage metadata.

This module provides the canonical implementation of subscriber extraction
shared across all provider middlewares.
"""


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
        if usage_metadata.get("subscriber_credential_name"):
            subscriber["credential"] = {
                "name": usage_metadata["subscriber_credential_name"],
                "value": usage_metadata.get("subscriber_credential"),
            }

    return subscriber
