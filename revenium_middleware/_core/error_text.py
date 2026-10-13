"""The exception text the SDK reports to Revenium for a failed tool call, job or model call."""

from revenium_middleware._core.config import is_capture_error_messages_enabled, is_capture_prompts_enabled
from revenium_middleware._core.log_sanitize import redact_error_text

__all__ = ["reportable_exception_text"]

DEFAULT_MAX_ERROR_TEXT_LENGTH = 2048
_MAX_CHAINED_EXCEPTIONS = 10


def _message_of(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return ""


def _is_http_status(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599


def _is_status_code_error(exc: BaseException) -> bool:
    """OpenAI, Anthropic and LiteLLM status errors."""
    return _is_http_status(getattr(exc, "status_code", None)) and (hasattr(exc, "response") or hasattr(exc, "body"))


def _is_google_genai_error(exc: BaseException) -> bool:
    """google-genai's ``APIError`` and its ``ClientError`` and ``ServerError``."""
    return _is_http_status(getattr(exc, "code", None)) and hasattr(exc, "response")


def _is_ollama_response_error(exc: BaseException) -> bool:
    """A ``ResponseError`` broken off mid-stream carries ``status_code`` -1, so any integer counts."""
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and any(
        cls.__name__ == "ResponseError" and cls.__module__.partition(".")[0] == "ollama"
        for cls in type(exc).__mro__)


def _is_http_status_error(exc: BaseException) -> bool:
    return _is_status_code_error(exc) or _is_google_genai_error(exc) or _is_ollama_response_error(exc)


def _is_provider_http_error(exc: BaseException) -> bool:
    """Whether ``exc``, or an exception it was raised from or while handling, is a provider's HTTP error.

    That is an OpenAI, Anthropic, LiteLLM, google-genai or Ollama error
    carrying the provider's status. A chain too long to check
    whole counts as one, since an unchecked link could quote the prompt.
    """
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        if len(seen) >= _MAX_CHAINED_EXCEPTIONS:
            return True
        seen.add(id(current))
        try:
            if _is_http_status_error(current):
                return True
        except Exception:
            pass
        pending.extend((current.__cause__, current.__context__))
    return False


def reportable_exception_text(exc: BaseException, max_len: int = DEFAULT_MAX_ERROR_TEXT_LENGTH) -> str:
    """The exception's class name, plus its redacted message when REVENIUM_CAPTURE_ERROR_MESSAGES is on.

    The message is opt-in because customer code routinely puts request URLs
    (with their query-string keys), personal data and failed values in it.
    A provider's HTTP error can quote the request, prompt included, so its
    message is sent only when REVENIUM_CAPTURE_PROMPTS is on as well.
    The result never exceeds ``max_len`` characters.
    """
    class_name = type(exc).__name__
    if not is_capture_error_messages_enabled():
        return class_name[:max_len]
    if not is_capture_prompts_enabled() and _is_provider_http_error(exc):
        return class_name[:max_len]
    message = _message_of(exc)
    if not message:
        return class_name[:max_len]
    return redact_error_text(f"{class_name}: {message}", max_len)
