"""Shared usage_metadata helpers for the Griptape drivers."""

# Keys that would corrupt the middleware's authentication state if they
# reached a provider API call.
_AUTH_KEYS = frozenset({"revenium_api_key", "revenium_api_base_url"})


def strip_revenium_auth_keys(metadata):
    """Return a copy of metadata without Revenium auth keys."""
    return {k: v for k, v in metadata.items() if k not in _AUTH_KEYS}
