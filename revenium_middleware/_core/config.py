"""
Shared configuration constants and settings for Revenium middleware.

This module is the single source of truth for configuration values shared
across all provider middlewares. Provider-specific config files extend
this via class inheritance and re-export symbols for backward compatibility.
"""

import logging
import os
import urllib.parse
import warnings
from typing import Callable, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

_METER_PATH_SEGMENT = "/meter"


Number = TypeVar("Number", int, float)


def read_env_number(
    name: str,
    default: Number,
    parse: Callable[[str], Number],
    is_valid: Optional[Callable[[Number], bool]] = None,
) -> Number:
    """Parse a numeric env var; unset uses ``default``, malformed or invalid warns and uses it.

    ``is_valid`` is optional; without it any value ``parse`` accepts is used.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = parse(raw)
    except (ValueError, OverflowError):
        value = None
    if value is None or (is_valid is not None and not is_valid(value)):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    return value


def _is_positive(value: int) -> bool:
    return value > 0


# A smaller limit cannot hold the 14-character "...[TRUNCATED]" marker.
MIN_PROMPT_LENGTH = 200
DEFAULT_MAX_PROMPT_LENGTH = 50_000


def read_prompt_length(name: str, default: int) -> int:
    """A prompt capture limit from ``name``; a positive value below ``MIN_PROMPT_LENGTH`` is raised to it."""
    length = read_env_number(name, default, int, _is_positive)
    if length < MIN_PROMPT_LENGTH:
        logger.warning("%s=%d is below the minimum; using %d", name, length, MIN_PROMPT_LENGTH)
        return MIN_PROMPT_LENGTH
    return length


class Config:
    """Shared configuration constants for all Revenium middleware providers."""

    # Threading and async timeouts
    THREAD_JOIN_TIMEOUT: float = 5.0
    API_REQUEST_TIMEOUT: float = 30.0
    BACKGROUND_THREAD_TIMEOUT: float = 5.0

    # Cache settings
    PROVIDER_CACHE_TTL: int = 3600  # 1 hour
    MODEL_CACHE_TTL: int = 3600     # 1 hour

    # Logging and security
    MAX_LOG_STRING_LENGTH: int = 100
    MAX_SANITIZATION_DEPTH: int = 3

    # Stream processing
    STREAM_CHUNK_BUFFER_SIZE: int = 1000
    STREAM_TIMEOUT: float = 30.0

    # Retry and resilience
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 1.5
    INITIAL_RETRY_DELAY: float = 0.1

    # API key validation
    VALID_API_KEY_PREFIXES: tuple = ("hak_", "rev_")

    # Core environment variable names
    ENV_REVENIUM_API_KEY: str = "REVENIUM_METERING_API_KEY"
    ENV_LOG_LEVEL: str = "REVENIUM_LOG_LEVEL"

    # Trace visualization environment variables
    ENV_REVENIUM_ENVIRONMENT: str = "REVENIUM_ENVIRONMENT"
    ENV_ENVIRONMENT: str = "ENVIRONMENT"
    ENV_DEPLOYMENT_ENV: str = "DEPLOYMENT_ENV"
    ENV_REVENIUM_REGION: str = "REVENIUM_REGION"
    ENV_REVENIUM_CREDENTIAL_ALIAS: str = "REVENIUM_CREDENTIAL_ALIAS"
    ENV_REVENIUM_TRACE_TYPE: str = "REVENIUM_TRACE_TYPE"
    ENV_REVENIUM_TRACE_NAME: str = "REVENIUM_TRACE_NAME"
    ENV_REVENIUM_PARENT_TRANSACTION_ID: str = "REVENIUM_PARENT_TRANSACTION_ID"
    ENV_REVENIUM_TRANSACTION_NAME: str = "REVENIUM_TRANSACTION_NAME"
    ENV_REVENIUM_RETRY_NUMBER: str = "REVENIUM_RETRY_NUMBER"
    ENV_REVENIUM_TICKET_ID: str = "REVENIUM_TICKET_ID"

    # Agentic job field environment variables (BACK-777 fallbacks)
    ENV_REVENIUM_AGENTIC_JOB_ID: str = "REVENIUM_AGENTIC_JOB_ID"
    ENV_REVENIUM_AGENTIC_JOB_NAME: str = "REVENIUM_AGENTIC_JOB_NAME"
    ENV_REVENIUM_AGENTIC_JOB_TYPE: str = "REVENIUM_AGENTIC_JOB_TYPE"
    ENV_REVENIUM_AGENTIC_JOB_VERSION: str = "REVENIUM_AGENTIC_JOB_VERSION"
    # Skill attribution environment variables
    ENV_REVENIUM_SKILL_INVOCATION_TRIGGER: str = "REVENIUM_SKILL_INVOCATION_TRIGGER"
    ENV_REVENIUM_SKILL_KIND: str = "REVENIUM_SKILL_KIND"
    ENV_REVENIUM_SKILL_MARKETPLACE_NAME: str = "REVENIUM_SKILL_MARKETPLACE_NAME"
    ENV_REVENIUM_SKILL_NAME: str = "REVENIUM_SKILL_NAME"
    ENV_REVENIUM_SKILL_PLUGIN_NAME: str = "REVENIUM_SKILL_PLUGIN_NAME"
    ENV_REVENIUM_SKILL_SOURCE: str = "REVENIUM_SKILL_SOURCE"
    ENV_REVENIUM_WRITE_API_KEY: str = "REVENIUM_WRITE_API_KEY"
    ENV_REVENIUM_OUTCOME_API_KEY: str = "REVENIUM_OUTCOME_API_KEY"
    ENV_REVENIUM_PROFITSTREAM_BASE_URL: str = "REVENIUM_PROFITSTREAM_BASE_URL"

    # Prompt capture settings
    ENV_REVENIUM_CAPTURE_PROMPTS: str = "REVENIUM_CAPTURE_PROMPTS"
    ENV_REVENIUM_MAX_PROMPT_SIZE: str = "REVENIUM_MAX_PROMPT_SIZE"
    CAPTURE_PROMPTS: bool = False
    MAX_PROMPT_LENGTH: int = read_prompt_length(ENV_REVENIUM_MAX_PROMPT_SIZE, DEFAULT_MAX_PROMPT_LENGTH)

    # Exception text in failed tool events and auto-failed job outcomes
    ENV_REVENIUM_CAPTURE_ERROR_MESSAGES: str = "REVENIUM_CAPTURE_ERROR_MESSAGES"

    # Revenium API targeting
    ENV_REVENIUM_TEAM_ID: str = "REVENIUM_TEAM_ID"
    ENV_REVENIUM_BASE_URL: str = "REVENIUM_METERING_BASE_URL"
    DEFAULT_BASE_URL: str = "https://api.revenium.ai"

    # Enforcement / circuit breaker — see _core/enforcement.py.
    # Taxonomy locked in BACK-1154 docs; keep in sync across SDKs.
    ENV_REVENIUM_BYPASS: str = "REVENIUM_BYPASS"
    ENV_CIRCUIT_BREAKER_ENABLED: str = "REVENIUM_CIRCUIT_BREAKER_ENABLED"
    ENV_REVENIUM_ENFORCEMENT_BASE_URL: str = "REVENIUM_ENFORCEMENT_BASE_URL"
    ENV_REVENIUM_CB_POLL_INTERVAL_SECONDS: str = "REVENIUM_CB_POLL_INTERVAL_SECONDS"
    ENV_REVENIUM_CB_FAIL_MODE: str = "REVENIUM_CB_FAIL_MODE"
    ENV_REVENIUM_CACHE_DIR: str = "REVENIUM_CACHE_DIR"

    # Shared per-call id for the LiteLLM proxy -- see
    # revenium_middleware.litellm.proxy.guardrail. On by default.
    ENV_REVENIUM_LITELLM_SHARED_CALL_ID: str = "REVENIUM_LITELLM_SHARED_CALL_ID"
    # The LiteLLM proxy's organization follows the calling key's team rather
    # than the caller's organization header. Off by default.
    ENV_REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM: str = "REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM"

    # Process shutdown -- see _core/metering.py and _core/shutdown_signals.py.
    ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS: str = "REVENIUM_SHUTDOWN_TIMEOUT_SECONDS"
    ENV_REVENIUM_INSTALL_SIGNAL_HANDLERS: str = "REVENIUM_INSTALL_SIGNAL_HANDLERS"


class SecurityConfig:
    """Security-related configuration shared across all providers."""

    SENSITIVE_FIELDS: Set[str] = {
        'api_key', 'subscriber_credential', 'subscriber_email', 'messages',
        'input', 'content', 'prompt', 'text', 'data', 'authorization',
        'x-api-key', 'bearer', 'token', 'password', 'secret', 'key',
        'credential', 'auth', 'private',
        'system_prompt', 'input_messages', 'output_response'
    }

    SENSITIVE_PATTERNS: Set[str] = {
        'sk-', 'pk-', 'Bearer ', 'Basic ', 'Token '
    }


_FLAG_TRUE_VALUES = ("1", "true", "yes", "on")
_FLAG_FALSE_VALUES = ("0", "false", "no", "off")

_warned_env_settings: Set[str] = set()


def _env_setting(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def _warn_once_unrecognised(name: str, raw: str, fallback: str) -> None:
    if name in _warned_env_settings:
        return
    _warned_env_settings.add(name)
    logger.warning("%s=%r is not a recognised value; using %s", name, raw, fallback)


def env_flag_enabled(name: str) -> bool:
    """Whether an opt-in environment flag is set to "1", "true", "yes" or "on".

    Case and surrounding whitespace are ignored. Blank or unset is off; any
    value that is neither on nor "0", "false", "no", "off" is off and warned
    about once.
    """
    value = _env_setting(name)
    if value in _FLAG_TRUE_VALUES:
        return True
    if value and value not in _FLAG_FALSE_VALUES:
        _warn_once_unrecognised(name, os.environ.get(name, ""), "false")
    return False


def env_choice(name: str, choices: Tuple[str, ...], default: str) -> str:
    """The value of a mode setting among ``choices``, ignoring case and surrounding whitespace.

    Blank or unset gives ``default``; any other value gives ``default`` and is warned about once.
    """
    value = _env_setting(name)
    if value in choices:
        return value
    if value:
        _warn_once_unrecognised(name, os.environ.get(name, ""), default)
    return default


def is_capture_prompts_enabled() -> bool:
    return env_flag_enabled(Config.ENV_REVENIUM_CAPTURE_PROMPTS)


def is_capture_error_messages_enabled() -> bool:
    """Whether failed tool events and auto-failed job outcomes send the exception message, not just its class."""
    return env_flag_enabled(Config.ENV_REVENIUM_CAPTURE_ERROR_MESSAGES)


def is_shared_call_id_enabled() -> bool:
    """Whether the LiteLLM proxy integration mints a shared per-call id.

    On by default; set the variable to "false" to opt out. When on,
    ``ReveniumGuardrail`` mints one id per proxied Anthropic messages request,
    returns it to the client as the ``request-id`` response header and submits
    it as the metered row's ``transaction_id``, so a Claude Code call seen by
    both its own telemetry and the proxy is stored once rather than twice.

    Read per call rather than once at import: a module-level global reading
    ``os.environ`` at import is what ``docs/conventions/PYTHON_CONVENTIONS.md``
    forbids under Secrets and Config, and an import-time read cannot be
    exercised by a test that sets the environment.

    A blank value counts as unset, because a deployment template that passes
    an unset host variable through hands the process an empty string.

    Returns:
        True when the environment variable is unset, blank, or one of "true",
        "1", "yes", "on", ignoring case and surrounding whitespace.
    """
    value = (shared_call_id_setting() or "").strip().lower()
    return not value or value in ("true", "1", "yes", "on")


def shared_call_id_setting() -> Optional[str]:
    """The raw ``REVENIUM_LITELLM_SHARED_CALL_ID`` value, or None when unset."""
    return os.getenv(Config.ENV_REVENIUM_LITELLM_SHARED_CALL_ID)


def is_organization_pinned_to_key_team() -> bool:
    """Whether the LiteLLM proxy takes the organization from the calling key's team.

    Off by default. When on, a call made with a virtual key that belongs to a
    LiteLLM team is checked against budgets and recorded under that team's
    alias, whatever ``x-revenium-organization-name`` or
    ``x-revenium-organization-id`` header the caller sends. A key with no team
    alias (unset or blank) keeps the header. Read per call, like
    ``is_shared_call_id_enabled``.
    """
    return env_flag_enabled(Config.ENV_REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM)


def organization_pin_setting_unrecognised() -> bool:
    """Whether the pin setting holds a non-blank value that reads neither on nor off.

    Such a value, for example a quoted ``"TRUE"`` or a typo, leaves the pin
    off, which an operator who set it would otherwise never notice.
    """
    value = _env_setting(Config.ENV_REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM)
    if not value or is_organization_pinned_to_key_team():
        return False
    return value not in _FLAG_FALSE_VALUES


def is_selective_metering_enabled() -> bool:
    """
    Check if selective metering is enabled.

    When enabled, only functions decorated with @revenium_meter will be metered.
    When disabled (default), all API calls are metered automatically.

    The setting is controlled by the REVENIUM_SELECTIVE_METERING environment variable.
    Accepted values for enabled: "true", "1", "yes", "on" (case-insensitive,
    surrounding whitespace ignored)
    """
    return env_flag_enabled("REVENIUM_SELECTIVE_METERING")


def get_config_value(key: str, default: any = None) -> any:
    """Get configuration value from environment or use default."""
    return os.getenv(key, default)


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return logging.getLogger("revenium_middleware").getEffectiveLevel() <= logging.DEBUG


def get_timeout_config() -> dict:
    """Get base timeout-related configuration."""
    return {
        'thread_join': Config.THREAD_JOIN_TIMEOUT,
        'api_request': Config.API_REQUEST_TIMEOUT,
        'background_thread': Config.BACKGROUND_THREAD_TIMEOUT,
        'stream': Config.STREAM_TIMEOUT,
    }


def get_team_id() -> Optional[str]:
    """Get Revenium team ID from environment."""
    return get_config_value(Config.ENV_REVENIUM_TEAM_ID)


def get_base_url() -> str:
    """Get Revenium base URL from environment (defaults to https://api.revenium.ai)."""
    return get_config_value(Config.ENV_REVENIUM_BASE_URL, Config.DEFAULT_BASE_URL)


def revenium_address_from_usage_address(usage_address: Optional[str]) -> str:
    """The Revenium address a usage address sits under: the same URL without its ``/meter`` segment.

    Customers set the usage address with or without a trailing ``/meter``, and
    the metering client accepts both. A query string or fragment is dropped;
    everything else is kept, so a reverse proxy's path prefix survives. A
    value that is empty or not an http(s) URL yields ``Config.DEFAULT_BASE_URL``.
    """
    parts = urllib.parse.urlsplit((usage_address or "").strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return Config.DEFAULT_BASE_URL
    path = parts.path.rstrip("/")
    if path.lower().endswith(_METER_PATH_SEGMENT):
        path = path[: -len(_METER_PATH_SEGMENT)].rstrip("/")
    return urllib.parse.urlunsplit(parts._replace(path=path, query="", fragment=""))


def get_revenium_address() -> str:
    """The Revenium address usage records go to, without the ``/meter`` segment."""
    return revenium_address_from_usage_address(os.getenv(Config.ENV_REVENIUM_BASE_URL))


def get_profitstream_base_url(
    explicit: Optional[str] = None, *, metering_base_url: Optional[str] = None
) -> str:
    """Resolve the address for outcome, job and job-type economics calls.

    Highest first: ``explicit``; ``REVENIUM_PROFITSTREAM_BASE_URL``; the usage
    address (``metering_base_url`` when given, else
    ``REVENIUM_METERING_BASE_URL``) without its ``/meter`` segment;
    ``Config.DEFAULT_BASE_URL``. An explicit or environment outcome address is
    returned as given, so its own context path survives.
    """
    if explicit:
        return explicit
    configured = os.getenv(Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL)
    if configured and configured.strip():
        return configured
    return revenium_address_from_usage_address(
        metering_base_url or os.getenv(Config.ENV_REVENIUM_BASE_URL)
    )


def _env_key(name: str) -> str:
    return (os.getenv(name) or "").strip()


def resolve_write_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the write-scope key from the supported precedence chain, ignoring surrounding whitespace."""
    explicit_key = (api_key or "").strip()
    if explicit_key:
        return explicit_key

    write_key = _env_key(Config.ENV_REVENIUM_WRITE_API_KEY)
    if write_key:
        return write_key

    outcome_key = _env_key(Config.ENV_REVENIUM_OUTCOME_API_KEY)
    if outcome_key:
        warnings.warn(
            "REVENIUM_OUTCOME_API_KEY is deprecated; set REVENIUM_WRITE_API_KEY instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return outcome_key

    return _env_key(Config.ENV_REVENIUM_API_KEY) or None


def validate_api_key(api_key: str) -> None:
    """Raise ``ValueError`` if ``api_key`` lacks a Revenium prefix or holds a character an HTTP header cannot carry.

    The caller is responsible for any None/empty check and for trimming; this
    validates format only. The message never contains the key.
    """
    if not api_key.startswith(Config.VALID_API_KEY_PREFIXES):
        raise ValueError(
            'Invalid Revenium API key format. '
            'Revenium API keys should start with "hak_" or "rev_".'
        )
    for position, character in enumerate(api_key, start=1):
        if not "!" <= character <= "~":
            raise ValueError(
                f"Invalid Revenium API key format: the key contains U+{ord(character):04X} at position {position}; "
                "a key holds only visible ASCII characters."
            )
