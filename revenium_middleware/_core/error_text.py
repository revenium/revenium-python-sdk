"""The exception text the SDK reports to Revenium for a failed tool call, job or model call."""

from typing import Callable, Optional

from revenium_middleware._core.config import is_capture_error_messages_enabled, is_capture_prompts_enabled
from revenium_middleware._core.log_sanitize import redact_error_text

__all__ = ["QuotesRequest", "is_http_exception", "reportable_exception_text"]

QuotesRequest = Callable[[BaseException], bool]

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


def is_http_exception(exc: BaseException) -> bool:
    """A web framework's ``HTTPException``: an HTTP status with a ``detail``."""
    return _is_http_status(getattr(exc, "status_code", None)) and hasattr(exc, "detail")


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


def _quotes_the_request(exc: BaseException, quotes_request: Optional[QuotesRequest]) -> bool:
    return _is_http_status_error(exc) or (quotes_request is not None and quotes_request(exc))


def _chain_can_quote_the_request(exc: BaseException, quotes_request: Optional[QuotesRequest] = None) -> bool:
    """Whether ``exc``, or an exception it was raised from or while handling, can quote the request.

    That is an OpenAI, Anthropic, LiteLLM, google-genai or Ollama error
    carrying the provider's status, or one ``quotes_request`` names. A chain
    too long to check whole counts as one, since an unchecked link could
    quote the prompt.
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
            if _quotes_the_request(current, quotes_request):
                return True
        except Exception:
            pass
        pending.extend((current.__cause__, current.__context__))
    return False


def reportable_exception_text(exc: BaseException, max_len: int = DEFAULT_MAX_ERROR_TEXT_LENGTH,
                              quotes_request: Optional[QuotesRequest] = None) -> str:
    """The exception's class name, plus its redacted message when REVENIUM_CAPTURE_ERROR_MESSAGES is on.

    The message is opt-in because customer code routinely puts request URLs
    (with their query-string keys), personal data and failed values in it.
    A provider's HTTP error can quote the request, prompt included, so its
    message is sent only when REVENIUM_CAPTURE_PROMPTS is on as well; so is
    the message of any exception ``quotes_request`` names, for a caller that
    knows more of its errors can quote the request.
    The result never exceeds ``max_len`` characters.
    """
    class_name = type(exc).__name__
    if not is_capture_error_messages_enabled():
        return class_name[:max_len]
    if not is_capture_prompts_enabled() and _chain_can_quote_the_request(exc, quotes_request):
        return class_name[:max_len]
    message = _message_of(exc)
    if not message:
        return class_name[:max_len]
    return redact_error_text(f"{class_name}: {message}", max_len)
