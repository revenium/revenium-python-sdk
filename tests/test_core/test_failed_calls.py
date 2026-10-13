"""The error reason of a failed call's record names the HTTP status only where the provider gave one (BACK-4335),
and the same status is the record's error code (BACK-4406)."""
import httpx
import pytest

from revenium_middleware._core.failed_calls import failure_of

openai = pytest.importorskip("openai")
genai_errors = pytest.importorskip("google.genai.errors")


def test_an_openai_stream_error_with_a_numeric_code_keeps_its_reason():
    """An SSE error event's body code is not the HTTP status of a response that arrived as 200."""
    error = openai.APIError("boom", httpx.Request("POST", "https://api.example.test"), body={"code": 502})
    failure = failure_of(error)
    assert failure.error_reason == "APIError"
    assert failure.error_code is None


def test_a_google_error_given_its_status_names_it():
    error = genai_errors.ClientError(429, {"error": {"code": 429, "message": "m", "status": "RESOURCE_EXHAUSTED"}})
    assert failure_of(error, status=error.code).error_reason == "ClientError (HTTP 429)"


def test_a_failure_with_an_http_status_carries_it_as_its_error_code():
    error = genai_errors.ClientError(429, {"error": {"code": 429, "message": "m", "status": "RESOURCE_EXHAUSTED"}})
    failure = failure_of(error, status=error.code)
    assert failure.error_code == 429
    assert failure.error_reason == "ClientError (HTTP 429)"


@pytest.mark.parametrize("value", [None, "429", True, -1, 99, 600, 429.0])
def test_a_value_that_is_not_an_http_status_names_none(value):
    error = genai_errors.ClientError(429, {"error": {"code": 429, "message": "m", "status": "RESOURCE_EXHAUSTED"}})
    failure = failure_of(error, status=value)
    assert failure.error_reason == "ClientError"
    assert failure.error_code is None


def test_an_anthropic_error_raised_inside_a_200_stream_has_no_error_code():
    """A mid-stream ``error`` event is raised as a status error on the stream's 200 response, which no
    provider answered as an error."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    response = httpx2.Response(200, request=httpx2.Request("POST", "https://api.anthropic.test/v1/messages"))
    error = anthropic.APIStatusError("Overloaded", response=response,
                                     body={"type": "error", "error": {"type": "overloaded_error"}})
    assert failure_of(error).error_code is None
