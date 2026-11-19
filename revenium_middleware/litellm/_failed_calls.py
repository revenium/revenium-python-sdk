"""What the record of a failed LiteLLM call says about the failure, in client mode and on the proxy alike."""

from typing import Dict, Optional

import httpx
import openai

from revenium_middleware._core.error_text import QuotesRequest, is_http_exception
from revenium_middleware._core.failed_calls import CallFailure, failure_of, raised_from_any

__all__ = ["litellm_call_failure", "proxy_failure_record_fields"]

_NO_SERVER_ANSWERED = (openai.APIConnectionError, httpx.TransportError)


def litellm_call_failure(error: BaseException, outer: Optional[BaseException] = None,
                         quotes_request: Optional[QuotesRequest] = None) -> Optional[CallFailure]:
    """``failure_of`` for an exception LiteLLM raised.

    LiteLLM gives a connection error a status of 500 and a timeout one of
    408 that no server answered, so neither names an HTTP status. On its
    ``openai/`` and ``anthropic/`` providers it raises a refused connection as
    an ``InternalServerError``, with the connection error only in its chain.
    """
    if raised_from_any(error, outer, _NO_SERVER_ANSWERED):
        return failure_of(error, outer, status=None, quotes_request=quotes_request)
    return failure_of(error, outer, quotes_request=quotes_request)


def proxy_failure_record_fields(error: object) -> Dict[str, object]:
    """The error reason and error code a failed proxied call's record carries; none when ``error`` is no exception.

    The proxy refuses a request with an ``HTTPException``, and a content
    guardrail's detail can quote the phrase it matched in the request, so its
    message is held back like a provider's HTTP error.
    """
    if not isinstance(error, BaseException):
        return {}
    failure = litellm_call_failure(error, quotes_request=is_http_exception)
    return failure.record_fields() if failure else {}
