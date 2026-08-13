"""
Revenium LiteLLM Client Middleware

When you install and import this library, it will automatically hook
litellm.completion using wrapt, and log token usage after each request.

New in v0.2.0:
- Context-based metadata injection via metadata_context
- Type-safe validation with UsageMetadata (requires pydantic)
- Decorator-based metadata injection (8 decorators available)
- Framework integrations (CrewAI integration available)

Basic Usage:
    >>> import revenium_middleware.litellm.client.middleware
    >>> import litellm
    >>> response = litellm.completion(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello"}],
    ...     usage_metadata={"agent": "my-agent"}
    ... )

Context-Based Usage:
    >>> from revenium_middleware.litellm.client import metadata_context
    >>> with metadata_context.set(agent="Lead Analyst", task_type="research"):
    ...     response = litellm.completion(...)  # Metadata auto-injected
"""
import logging

logger = logging.getLogger(__name__)

# Import components that don't require litellm SDK
from .context import metadata_context, MetadataContext
from .validation import UsageMetadata, Subscriber, SubscriberCredential, PYDANTIC_AVAILABLE
from .decorators import (
    track_agent,
    track_task,
    track_job,
    track_trace,
    track_organization,
    track_subscription,
    track_product,
    track_subscriber,
    track_quality
)
from .hooks import (
    register_metadata_hook,
    unregister_metadata_hook,
    clear_metadata_hooks,
    get_registered_hooks
)

# Conditionally import middleware (requires litellm SDK)
try:
    import litellm as _litellm  # noqa: F401
    from .middleware import completion_wrapper
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("LiteLLM", e, required_packages=("litellm",))
    completion_wrapper = None  # type: ignore

__version__ = "0.2.0"

__all__ = [
    'completion_wrapper',
    'metadata_context',
    'MetadataContext',
    'UsageMetadata',
    'Subscriber',
    'SubscriberCredential',
    'PYDANTIC_AVAILABLE',
    'track_agent',
    'track_task',
    'track_job',
    'track_trace',
    'track_organization',
    'track_subscription',
    'track_product',
    'track_subscriber',
    'track_quality',
    'register_metadata_hook',
    'unregister_metadata_hook',
    'clear_metadata_hooks',
    'get_registered_hooks',
]
