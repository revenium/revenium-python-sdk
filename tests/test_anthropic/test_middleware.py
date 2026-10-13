import asyncio
import datetime
import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from freezegun import freeze_time

from revenium_middleware import shutdown_event
from revenium_middleware.anthropic.middleware import create_wrapper, _extract_organization_and_product_names


def _middleware_fn(obj):
    """Resolve the plain middleware function across wrapt versions.

    Under wrapt <2.2 the module attribute left behind by
    @wrapt.patch_function_wrapper is the patched FunctionWrapper — calling it
    directly would invoke the real anthropic SDK method. The plain middleware
    function is stored on its `_self_wrapper` attribute. Under wrapt >=2.2 the
    module attribute is already the plain function.
    """
    return getattr(obj, "_self_wrapper", obj)


create_wrapper = _middleware_fn(create_wrapper)


def _run_metering_synchronously(mock_run_async):
    """Make a patched revenium_middleware.run_async_in_thread execute the
    metering coroutine synchronously in the test thread (instead of a
    background thread) and return a fake thread handle, so the test can
    assert on the metering call deterministically."""
    mock_thread = MagicMock()

    def _run(coroutine):
        asyncio.run(coroutine)
        return mock_thread

    mock_run_async.side_effect = _run
    return mock_thread


class TestMiddleware:
    @pytest.fixture
    def reset_state(self):
        """Fixture to reset global state before each test."""
        shutdown_event.clear()
        yield
        # Cleanup after test
        shutdown_event.clear()

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create a mock Anthropic response object."""
        mock_response = MagicMock()
        mock_response.id = "test-response-id"
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"

        # Set up usage attributes (Anthropic naming)
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_creation_input_tokens = 3
        mock_response.usage.cache_read_input_tokens = 7

        return mock_response

    @pytest.fixture
    def test_kwargs(self):
        """Common test kwargs for Anthropic API calls."""
        return {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage_metadata": {
                "trace_id": "test-trace",
                "task_id": "test-task",
                "task_type": "test-type",
                "organizationName": "AcmeCorp",
                "subscription_id": "test-sub",
                "productName": "customer-chatbot",
                "agent": "test-agent"
            }
        }

    @freeze_time("2023-01-01T12:00:00Z")
    @patch("revenium_middleware.anthropic.middleware.submit_ai_event")
    @patch("revenium_middleware.run_async_in_thread")
    def test_create_wrapper_basic(self, mock_run_async, mock_submit_ai_event, reset_state, mock_anthropic_response,
                                  test_kwargs):
        """Test the basic functionality of create_wrapper: the wrapped call is
        forwarded without usage_metadata and exactly one metering event is
        submitted with values derived from the response and metadata."""
        mock_wrapped = MagicMock(return_value=mock_anthropic_response)
        _run_metering_synchronously(mock_run_async)

        # Call the wrapper - the plain middleware function expects:
        # wrapped, instance, args, kwargs. instance=None keeps provider
        # detection deterministic (defaults to ANTHROPIC).
        result = create_wrapper(mock_wrapped, None, (), test_kwargs.copy())

        # The wrapped call returns unchanged and usage_metadata is stripped
        assert result is mock_anthropic_response
        mock_wrapped.assert_called_once_with(**{k: v for k, v in test_kwargs.items() if k != "usage_metadata"})

        # Metering happened exactly once, via submit_ai_event
        mock_run_async.assert_called_once()
        mock_submit_ai_event.assert_called_once()
        operation, payload = mock_submit_ai_event.call_args[0]
        assert operation == "completion"

        # Values derived from the response
        assert payload["provider"] == "ANTHROPIC"
        assert payload["model_source"] == "ANTHROPIC"
        assert payload["model"] == "claude-3-5-sonnet-20241022"
        assert payload["input_token_count"] == 100
        assert payload["output_token_count"] == 50
        assert payload["total_token_count"] == 150
        assert payload["cache_creation_token_count"] == 3
        assert payload["cache_read_token_count"] == 7
        assert payload["stop_reason"] == "END"  # "end_turn" maps to END
        assert payload["transaction_id"] == "test-response-id"
        assert payload["cost_type"] == "AI"
        assert payload["is_streamed"] is False
        assert payload["middleware_source"] == "PYTHON"

        # Timing under freeze_time: request and response coincide
        assert payload["request_time"] == "2023-01-01T12:00:00Z"
        assert payload["response_time"] == "2023-01-01T12:00:00Z"
        assert payload["completion_start_time"] == "2023-01-01T12:00:00Z"
        assert payload["request_duration"] == 0

        # usage_metadata fields flow through to the metering payload
        assert payload["trace_id"] == "test-trace"
        assert payload["task_type"] == "test-type"
        assert payload["subscription_id"] == "test-sub"
        assert payload["agent"] == "test-agent"
        assert payload["organization_name"] == "AcmeCorp"
        assert payload["product_name"] == "customer-chatbot"

    @freeze_time("2023-01-01T12:00:00Z")
    @patch("revenium_middleware.anthropic.middleware.submit_ai_event")
    @patch("revenium_middleware.run_async_in_thread")
    def test_create_wrapper_no_metadata(self, mock_run_async, mock_submit_ai_event, reset_state,
                                        mock_anthropic_response):
        """Test create_wrapper with no usage_metadata provided: the call still
        succeeds and is metered, with metadata-derived fields left unset."""
        mock_wrapped = MagicMock(return_value=mock_anthropic_response)
        _run_metering_synchronously(mock_run_async)

        # Test data without usage_metadata
        test_kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-3-5-sonnet-20241022"
        }

        # Call the wrapper
        result = create_wrapper(mock_wrapped, None, (), test_kwargs.copy())

        # Assertions
        assert result is mock_anthropic_response
        mock_wrapped.assert_called_once_with(**test_kwargs)
        mock_run_async.assert_called_once()

        mock_submit_ai_event.assert_called_once()
        operation, payload = mock_submit_ai_event.call_args[0]
        assert operation == "completion"
        assert payload["input_token_count"] == 100
        assert payload["output_token_count"] == 50
        assert payload["transaction_id"] == "test-response-id"
        assert payload["trace_id"] is None
        assert payload["subscription_id"] is None
        assert payload["organization_name"] is None
        assert payload["product_name"] is None
        assert payload["subscriber"] is None

    @freeze_time("2023-01-01T12:00:00Z")
    @patch("revenium_middleware.anthropic.middleware.submit_ai_event")
    @patch("revenium_middleware.run_async_in_thread")
    def test_create_wrapper_during_shutdown(self, mock_run_async, mock_submit_ai_event, reset_state,
                                            mock_anthropic_response, test_kwargs, caplog):
        """Test create_wrapper behavior during shutdown: the wrapped call still
        returns normally but metering is skipped entirely."""
        caplog.set_level(logging.WARNING)

        # Set up mocks
        mock_wrapped = MagicMock(return_value=mock_anthropic_response)

        # Set shutdown event
        shutdown_event.set()

        # Call the wrapper
        result = create_wrapper(mock_wrapped, None, (), test_kwargs.copy())

        # The caller is unaffected
        assert result is mock_anthropic_response
        mock_wrapped.assert_called_once()

        # No metering thread is started and nothing is submitted
        mock_run_async.assert_not_called()
        mock_submit_ai_event.assert_not_called()

        # The skip is logged as a warning
        assert "Skipping async operation during shutdown" in caplog.text

    @freeze_time("2023-01-01T12:00:00Z")
    @patch("revenium_middleware.anthropic.middleware.submit_ai_event")
    @patch("revenium_middleware.run_async_in_thread")
    def test_create_wrapper_metering_exception(self, mock_run_async, mock_submit_ai_event, reset_state,
                                               mock_anthropic_response, test_kwargs, caplog):
        """Test create_wrapper when the metering call raises an exception: the
        exception is caught inside the metering coroutine and logged, and the
        wrapped response is still returned to the caller."""
        caplog.set_level(logging.WARNING)

        # Set up mocks
        mock_wrapped = MagicMock(return_value=mock_anthropic_response)
        _run_metering_synchronously(mock_run_async)

        # Make the metering submission raise an exception
        mock_submit_ai_event.side_effect = Exception("Test error")

        # Call the wrapper - must not raise
        result = create_wrapper(mock_wrapped, None, (), test_kwargs.copy())

        # The caller still gets the wrapped response
        assert result is mock_anthropic_response
        mock_wrapped.assert_called_once()
        mock_run_async.assert_called_once()
        mock_submit_ai_event.assert_called_once()

        # The exception is caught inside the metering coroutine and logged
        assert "Error in metering call: Test error" in caplog.text

    @freeze_time("2023-01-01T12:00:00Z")
    @patch("revenium_middleware.anthropic.middleware.submit_ai_event")
    @patch("revenium_middleware.run_async_in_thread")
    def test_create_wrapper_no_choices(self, mock_run_async, mock_submit_ai_event, reset_state,
                                       mock_anthropic_response, test_kwargs):
        """Test create_wrapper when the response carries no usage data
        (Anthropic responses have no `choices`; the analogous degenerate
        response today is usage=None): the response is returned unchanged
        and metering is skipped."""
        # Set up mocks
        mock_wrapped = MagicMock(return_value=mock_anthropic_response)
        _run_metering_synchronously(mock_run_async)

        # Response without usage data
        mock_anthropic_response.usage = None

        # Call the wrapper
        result = create_wrapper(mock_wrapped, None, (), test_kwargs.copy())

        # The response is returned early, without any metering
        assert result is mock_anthropic_response
        mock_wrapped.assert_called_once()
        mock_run_async.assert_not_called()
        mock_submit_ai_event.assert_not_called()

    @freeze_time("2023-01-01T12:00:00Z")
    @patch("revenium_middleware.anthropic.middleware.datetime")
    @patch("revenium_middleware.anthropic.middleware.submit_ai_event")
    @patch("revenium_middleware.run_async_in_thread")
    def test_create_wrapper_request_duration(self, mock_run_async, mock_submit_ai_event, mock_datetime,
                                             reset_state, mock_anthropic_response, test_kwargs):
        """Test create_wrapper calculates request duration correctly as the
        elapsed wall-clock milliseconds between request and response."""
        # Set up mocks
        mock_wrapped = MagicMock(return_value=mock_anthropic_response)
        _run_metering_synchronously(mock_run_async)

        # Mock datetime to simulate 1 second elapsing between the request
        # timestamp and the response timestamp
        request_time = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        response_time = datetime.datetime(2023, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)

        mock_datetime.timezone = datetime.timezone
        mock_datetime.datetime.now.side_effect = [request_time, response_time]

        # Call the wrapper
        result = create_wrapper(mock_wrapped, None, (), test_kwargs.copy())

        # Assertions
        assert result is mock_anthropic_response
        mock_wrapped.assert_called_once()
        mock_run_async.assert_called_once()

        # Request duration should be 1000ms (1 second)
        mock_submit_ai_event.assert_called_once()
        payload = mock_submit_ai_event.call_args[0][1]
        assert payload["request_time"] == "2023-01-01T12:00:00Z"
        assert payload["response_time"] == "2023-01-01T12:00:01Z"
        assert payload["request_duration"] == 1000
        assert payload["time_to_first_token"] == 1000



class TestExtractOrganizationAndProductNames:
    """Tests for _extract_organization_and_product_names backward compatibility logic."""

    def test_snake_case_has_highest_precedence(self):
        usage_metadata = {
            "organizationName": "CamelCaseOrg",
            "organization_name": "snake_case_org",
            "organizationId": "old_camel_id",
            "organization_id": "old_snake_id",
            "productName": "CamelCaseProduct",
            "product_name": "snake_case_product",
            "productId": "old_camel_prod_id",
            "product_id": "old_snake_prod_id",
        }
        org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
        assert org_name == "snake_case_org"
        assert prod_name == "snake_case_product"

    def test_snake_case_new_names_take_precedence_over_deprecated(self):
        """organization_name and product_name take precedence over deprecated fields."""
        usage_metadata = {
            "organization_name": "snake_case_org",
            "organizationId": "old_camel_id",
            "organization_id": "old_snake_id",
            "product_name": "snake_case_product",
            "productId": "old_camel_prod_id",
            "product_id": "old_snake_prod_id",
        }
        org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
        assert org_name == "snake_case_org"
        assert prod_name == "snake_case_product"

    def test_deprecated_camel_case_works_with_warning(self, caplog):
        """organizationId and productId still work but log deprecation warning."""
        with caplog.at_level(logging.WARNING, logger="revenium_middleware._core.fields"):
            usage_metadata = {
                "organizationId": "old_camel_id",
                "productId": "old_camel_prod_id",
            }
            org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
            assert org_name == "old_camel_id"
            assert prod_name == "old_camel_prod_id"
            assert "'organizationId' and 'organization_id' are deprecated" in caplog.text
            assert "'productId' and 'product_id' are deprecated" in caplog.text

    def test_deprecated_snake_case_works_with_warning(self, caplog):
        """organization_id and product_id still work but log deprecation warning."""
        with caplog.at_level(logging.WARNING, logger="revenium_middleware._core.fields"):
            usage_metadata = {
                "organization_id": "old_snake_id",
                "product_id": "old_snake_prod_id",
            }
            org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
            assert org_name == "old_snake_id"
            assert prod_name == "old_snake_prod_id"
            assert "'organizationId' and 'organization_id' are deprecated" in caplog.text
            assert "'productId' and 'product_id' are deprecated" in caplog.text

    def test_deprecated_snake_case_takes_precedence_over_camel_case(self):
        usage_metadata = {
            "organizationId": "old_camel_id",
            "organization_id": "old_snake_id",
            "productId": "old_camel_prod_id",
            "product_id": "old_snake_prod_id",
        }
        org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
        assert org_name == "old_snake_id"
        assert prod_name == "old_snake_prod_id"

    def test_empty_metadata_returns_none(self):
        """Empty metadata returns None for both fields."""
        usage_metadata = {}
        org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
        assert org_name is None
        assert prod_name is None

    def test_partial_metadata_organization_only(self):
        """Only organization field present works correctly."""
        usage_metadata = {"organizationName": "TestOrg"}
        org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
        assert org_name == "TestOrg"
        assert prod_name is None

    def test_partial_metadata_product_only(self):
        """Only product field present works correctly."""
        usage_metadata = {"productName": "TestProduct"}
        org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
        assert org_name is None
        assert prod_name == "TestProduct"

    def test_mixed_new_and_deprecated_fields(self, caplog):
        """New organizationName with deprecated productId works correctly."""
        with caplog.at_level(logging.WARNING, logger="revenium_middleware._core.fields"):
            usage_metadata = {
                "organizationName": "NewOrg",
                "productId": "old_product_id",
            }
            org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
            assert org_name == "NewOrg"
            assert prod_name == "old_product_id"
            assert "'organizationId' and 'organization_id' are deprecated" not in caplog.text
            assert "'productId' and 'product_id' are deprecated" in caplog.text

    def test_no_deprecation_warning_when_new_fields_used(self, caplog):
        """No deprecation warning when only new field names are used."""
        with caplog.at_level(logging.WARNING, logger="revenium_middleware._core.fields"):
            usage_metadata = {
                "organizationName": "TestOrg",
                "productName": "TestProduct",
            }
            org_name, prod_name = _extract_organization_and_product_names(usage_metadata)
            assert org_name == "TestOrg"
            assert prod_name == "TestProduct"
            assert "deprecated" not in caplog.text.lower()


def _run_metering_on_this_thread(coro_func, *args, **kwargs):
    """Run the metering coroutine on the caller's thread so a per-thread cache would be reused."""
    asyncio.run(coro_func(*args, **kwargs))
    return MagicMock()


def _anthropic_message(model="claude-sonnet-4-6"):
    return SimpleNamespace(
        id="msg_key_after_import",
        model=model,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
                              cache_read_input_tokens=0),
    )


def _fake_metering_client():
    fake = MagicMock()
    fake.ai.create_completion.return_value = MagicMock(status_code=201)
    return fake


class TestKeySetAfterImport:
    """BACK-4014: Anthropic calls are metered once the key reaches the environment after import."""

    MODEL = "claude-sonnet-4-6"

    @pytest.fixture(autouse=True)
    def key_set_after_import(self, monkeypatch):
        import revenium_middleware
        from revenium_middleware import _core
        from revenium_middleware._core import metering, metering_submission
        from revenium_middleware.anthropic import middleware

        monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
        monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)
        for module in (metering, metering_submission, _core, revenium_middleware):
            monkeypatch.setattr(module, "client", None, raising=False)
        monkeypatch.setattr(metering, "_last_failed_key", None)
        monkeypatch.setattr(middleware, "_thread_local", threading.local(), raising=False)
        shutdown_event.clear()

        self.first = _fake_metering_client()
        monkeypatch.setattr(metering, "_build_metering_client", lambda key, url: self.first)
        monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_set_after_import")
        self.monkeypatch = monkeypatch
        self.metering = metering
        yield
        shutdown_event.clear()

    def _create(self):
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": self.MODEL,
            "usage_metadata": {"trace_id": "trace-key-after-import"},
        }
        create_wrapper(MagicMock(return_value=_anthropic_message(self.MODEL)), None, (), request)

    @patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
           side_effect=_run_metering_on_this_thread)
    def test_messages_create_is_metered_when_key_is_set_after_import(self, _runner):
        self._create()

        assert self.first.ai.create_completion.call_count == 1
        assert self.first.ai.create_completion.call_args.kwargs["model"] == self.MODEL

    @patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
           side_effect=_run_metering_on_this_thread)
    def test_bedrock_call_is_metered_when_key_is_set_after_import(self, _runner):
        from revenium_middleware.anthropic.middleware import _create_bedrock_metering_call

        response = SimpleNamespace(
            id="bedrock-key-after-import",
            model="anthropic.claude-3-haiku-20240307-v1:0",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        _create_bedrock_metering_call(
            response, {"trace_id": "trace-bedrock-key-after-import"},
            "2026-10-08T00:00:00Z", "2026-10-08T00:00:01Z", 1000.0,
            {"model": response.model, "messages": [{"role": "user", "content": "Hello"}]},
        )

        assert self.first.ai.create_completion.call_count == 1
        assert self.first.ai.create_completion.call_args.kwargs["model"] == response.model

    @patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
           side_effect=_run_metering_on_this_thread)
    def test_messages_create_uses_the_new_client_after_initialize_metering_on_the_same_thread(self, _runner):
        self._create()
        second = _fake_metering_client()
        self.monkeypatch.setattr(self.metering, "_build_metering_client", lambda key, url: second)
        assert self.metering.initialize_metering(api_key="hak_second_dummy") is True

        self._create()

        assert self.first.ai.create_completion.call_count == 1
        assert second.ai.create_completion.call_count == 1


_PASS_THROUGH_WRAPPERS = (
    "create_wrapper",
    "async_create_wrapper",
    "stream_wrapper",
    "async_stream_wrapper",
    "_parse_wrapper",
    "_beta_create_wrapper",
    "_beta_async_create_wrapper",
    "_beta_stream_wrapper",
)
_COROUTINE_WRAPPERS = {"async_create_wrapper", "_beta_async_create_wrapper"}


def _call_through(wrapper, wrapped, awaited, kwargs):
    """Drive a wrapper that returns a coroutine the way the SDK caller would: by awaiting it."""
    if not awaited:
        return wrapper(wrapped, None, (), kwargs)

    async def provider(*args, **provider_kwargs):
        return wrapped(*args, **provider_kwargs)

    return asyncio.run(wrapper(provider, None, (), kwargs))


@pytest.mark.parametrize("usage_metadata", [{"organizationName": "AcmeCorp"}, None], ids=["metadata", "none"])
@pytest.mark.parametrize("name", _PASS_THROUGH_WRAPPERS)
def test_selective_pass_through_drops_usage_metadata(name, usage_metadata, monkeypatch):
    """Selective metering on, outside @revenium_meter: the call is not
    metered, and the Revenium-only keyword never reaches the Anthropic SDK,
    whose signature rejects it (BACK-3991)."""
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seen = []

    def wrapped(*_args, **kwargs):
        seen.append(kwargs)
        return "provider response"

    wrapper = _middleware_fn(getattr(anthropic_middleware, name))
    kwargs = {"model": "claude-sonnet-4-6", "max_tokens": 5, "usage_metadata": usage_metadata}

    assert _call_through(wrapper, wrapped, name in _COROUTINE_WRAPPERS, kwargs) == "provider response"
    assert seen == [{"model": "claude-sonnet-4-6", "max_tokens": 5}]


def _message_without_cache_counts():
    message = _anthropic_message()
    message.usage.cache_creation_input_tokens = None
    message.usage.cache_read_input_tokens = None
    return message


def _meter_message_without_cache_counts(mw):
    mw._meter_message(_message_without_cache_counts(), {}, {"model": "claude-sonnet-4-6"},
                      "2026-10-11T00:00:00Z", datetime.datetime.now(datetime.timezone.utc), "ANTHROPIC")


def _meter_message_stream_without_cache_counts(mw):
    now = datetime.datetime.now(datetime.timezone.utc)
    mw._meter_message_stream(_message_without_cache_counts(), {}, {"model": "claude-sonnet-4-6"},
                             "2026-10-11T00:00:00Z", now, now, "ANTHROPIC")


@pytest.mark.parametrize("meter", [_meter_message_without_cache_counts, _meter_message_stream_without_cache_counts],
                         ids=["_meter_message", "_meter_message_stream"])
def test_debug_line_with_no_cache_counts_logs_cleanly(meter, caplog):
    """A response that reports no cache counts logs its debug token line
    without a logging error (BACK-4452 N16)."""
    from revenium_middleware.anthropic import middleware as mw

    caplog.set_level(logging.DEBUG, logger="revenium_middleware")
    with patch.object(mw, "_get_thread_safe_client", return_value=MagicMock()), \
            patch.object(mw, "_safe_run_async_in_thread", MagicMock()):
        meter(mw)

    assert any("cache_read_input_tokens: None" in record.getMessage() for record in caplog.records)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.fixture
def no_metering_key(monkeypatch):
    """No key in the environment and no metering client built yet."""
    import revenium_middleware
    from revenium_middleware import _core
    from revenium_middleware._core import metering, metering_submission
    from revenium_middleware.anthropic import middleware

    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)
    for module in (metering, metering_submission, _core, revenium_middleware):
        monkeypatch.setattr(module, "client", None, raising=False)
    monkeypatch.setattr(metering, "_last_failed_key", None)
    monkeypatch.setattr(middleware, "_thread_local", threading.local(), raising=False)
    monkeypatch.setattr(metering, "_new_client", lambda key, url: _fake_metering_client())
    shutdown_event.clear()
    yield metering
    shutdown_event.clear()


class TestAnthropicHealthCheck:
    """anthropic.is_initialized() and initialize() report whether calls will
    be metered, wherever the key came from (BACK-4452 N13)."""

    @staticmethod
    def _health():
        from revenium_middleware import anthropic as anthropic_integration
        return anthropic_integration.is_initialized(), anthropic_integration.initialize()

    def test_is_initialized_sees_a_key_set_in_code(self, no_metering_key):
        assert no_metering_key.initialize_metering(api_key="hak_dummy_health_check") is True

        assert self._health() == (True, True)

    def test_is_initialized_is_false_with_no_key_anywhere(self, no_metering_key):
        assert self._health() == (False, False)

    def test_is_initialized_is_false_for_a_blank_environment_key(self, no_metering_key, monkeypatch):
        monkeypatch.setenv("REVENIUM_METERING_API_KEY", "   ")

        assert self._health() == (False, False)

    def test_is_initialized_is_false_after_the_key_set_in_code_is_cleared(self, no_metering_key):
        assert no_metering_key.initialize_metering(api_key="hak_dummy_health_check") is True
        assert no_metering_key.initialize_metering(api_key="") is False

        assert self._health() == (False, False)

    def test_is_initialized_is_false_for_a_malformed_environment_key(self, no_metering_key, monkeypatch):
        monkeypatch.setenv("REVENIUM_METERING_API_KEY", "sk-not-revenium")

        assert self._health() == (False, False)


def _meter_raw_stream_with_usage(mw):
    from revenium_middleware.anthropic.stream_create import StreamUsageState
    state = StreamUsageState()
    state.saw_message_start = True
    state.message_id = "msg_no_key_raw_stream"
    state.model = "claude-sonnet-4-6"
    state.input_tokens = 10
    state.output_tokens = 5
    mw._meter_raw_stream(state, {}, {"model": "claude-sonnet-4-6"}, "2026-10-11T00:00:00Z",
                         datetime.datetime.now(datetime.timezone.utc), "ANTHROPIC")


def _meter_message_with_usage(mw):
    mw._meter_message(_anthropic_message(), {}, {"model": "claude-sonnet-4-6"},
                      "2026-10-11T00:00:00Z", datetime.datetime.now(datetime.timezone.utc), "ANTHROPIC")


def _meter_message_stream_with_usage(mw):
    now = datetime.datetime.now(datetime.timezone.utc)
    mw._meter_message_stream(_anthropic_message(), {}, {"model": "claude-sonnet-4-6"},
                             "2026-10-11T00:00:00Z", now, now, "ANTHROPIC")


def _create_bedrock_metering_call_with_usage(mw):
    response = SimpleNamespace(id="bedrock-no-key", model="anthropic.claude-3-haiku-20240307-v1:0",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15))
    mw._create_bedrock_metering_call(response, {}, "2026-10-11T00:00:00Z", "2026-10-11T00:00:01Z", 1000.0,
                                     {"model": response.model, "messages": [{"role": "user", "content": "Hello"}]})


def _bedrock_stream_send_metering_data(mw):
    from revenium_middleware.anthropic.bedrock_adapter import BedrockStreamWrapper
    wrapper = BedrockStreamWrapper(model="claude-3-haiku-20240307", payload={}, region="us-east-1",
                                   messages=[{"role": "user", "content": "Hello"}])
    wrapper.final_message = SimpleNamespace(id="bedrock-stream-no-key", model="claude-3-haiku-20240307",
                                            usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    wrapper._call_context.run(wrapper._send_metering_data, 1.0)


class TestNoKeyStaysQuiet:
    """With no metering key, Anthropic calls return before building a record
    and log nothing per call, like the other providers (BACK-4452 N14)."""

    @pytest.fixture(autouse=True)
    def quiet_setup(self, no_metering_key, caplog):
        from revenium_middleware.anthropic import middleware as mw
        caplog.set_level(logging.DEBUG, logger="revenium_middleware")
        self.caplog = caplog
        self.mw = mw
        with patch.object(mw, "_safe_run_async_in_thread", MagicMock()) as runner:
            self.runner = runner
            yield

    def _warnings(self):
        return [r for r in self.caplog.records if r.levelno >= logging.WARNING]

    def test_messages_create_with_no_key_builds_no_record_and_logs_nothing(self):
        message = _anthropic_message()
        request = {"messages": [{"role": "user", "content": "Hello"}], "model": "claude-sonnet-4-6"}

        for _ in range(3):
            assert create_wrapper(MagicMock(return_value=message), None, (), dict(request)) is message

        assert self.runner.call_count == 0
        assert self._warnings() == []

    @pytest.mark.parametrize("meter", [
        _meter_message_with_usage,
        _meter_message_stream_with_usage,
        _meter_raw_stream_with_usage,
        _create_bedrock_metering_call_with_usage,
        _bedrock_stream_send_metering_data,
    ], ids=["_meter_message", "_meter_message_stream", "_meter_raw_stream",
            "_create_bedrock_metering_call", "BedrockStreamWrapper._send_metering_data"])
    def test_every_anthropic_meter_path_returns_before_building_the_record(self, meter):
        with patch.object(self.mw, "extract_prompt_data_if_enabled",
                             return_value=(None, None, None, False)) as prompt_capture, \
                patch("revenium_middleware.anthropic.trace_fields.detect_vision_content") as vision_detection:
            meter(self.mw)

        assert self.runner.call_count == 0
        assert prompt_capture.call_count == 0
        assert vision_detection.call_count == 0
        assert self._warnings() == []
