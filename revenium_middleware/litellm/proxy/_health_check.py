"""Recognise LiteLLM's own health-check calls so they are never metered.

``litellm.ahealth_check`` sends a real test call to a deployment and marks it in
``litellm_metadata``: the tag, and the service account's key hash and team id,
all equal to one marker. A caller can write the tag (``x-litellm-tags``, body
``metadata.tags``) but LiteLLM overwrites the key and team fields with the real
key's, so only a call that carries all three marks, and no ``proxy_server_request``,
is a probe.

Deliberately dependency-free and Python 3.8-compatible, like ``_metering_owner``.
"""

import logging
import threading
from collections.abc import Mapping

logger = logging.getLogger("revenium_middleware.extension")

LITELLM_HEALTH_CHECK_MARKER = "litellm-internal-health-check"

_announce_lock = threading.Lock()
_announced = False


def is_litellm_health_check(container):
    # Runs inside the caller's LiteLLM call, so it must never raise.
    try:
        litellm_metadata = container.get("litellm_metadata")
        if not isinstance(litellm_metadata, Mapping):
            return False
        tags = litellm_metadata.get("tags")
        return (
            isinstance(tags, (list, tuple))
            and LITELLM_HEALTH_CHECK_MARKER in tags
            and litellm_metadata.get("user_api_key_hash") == LITELLM_HEALTH_CHECK_MARKER
            and litellm_metadata.get("user_api_key_team_id") == LITELLM_HEALTH_CHECK_MARKER
            and not isinstance(container.get("proxy_server_request"), Mapping)
        )
    except Exception:
        return False


def note_health_check_skipped(where):
    global _announced
    with _announce_lock:
        first = not _announced
        _announced = True
    if first:
        logger.info("LiteLLM health-check calls are not metered")
    logger.debug("Skipped metering a LiteLLM health-check call (%s)", where)
