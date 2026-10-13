"""A blank organization or product header counts as absent on the proxy (BACK-4030).

A client whose ORG variable is unset still sends ``x-revenium-organization-name``,
just empty. That must fall back exactly as a missing header does, to the
deprecated header and then the virtual key's team, instead of storing the call
under an empty organization. Product falls back the same way.

This file drives the deprecated ``MiddlewareHandler`` callback, which imports
litellm but not fastapi, so it runs in every CI job. The guardrail half of the
same chain is covered in ``test_proxy_guardrail_metering.py``.
"""
import asyncio
import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as proxy_mw  # noqa: E402

NOW = datetime.datetime.now(datetime.timezone.utc)
TEAM = "team-a"
ORG_HEADER = "x-revenium-organization-name"
OLD_ORG_HEADER = "x-revenium-organization-id"
PRODUCT_HEADER = "x-revenium-product-name"
OLD_PRODUCT_HEADER = "x-revenium-product-id"


def run_inline(coro):
    """Execute the metering coroutine synchronously so asserts see the call."""
    asyncio.run(coro)
    return SimpleNamespace(name="inline-metering")


def run_hook(coro):
    """Drive MiddlewareHandler's async_log_*_event coroutine to completion."""
    try:
        coro.send(None)
    except StopIteration:
        pass


class SubscriptableResponse:
    """Stand-in for LiteLLM's ModelResponse: ``response_obj["usage"]`` plus ``.id``."""

    def __init__(self):
        self.id = "txn-blank-attribution"
        self._usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)

    def __getitem__(self, key):
        if key == "usage":
            return self._usage
        raise KeyError(key)


def request_kwargs(headers, team_alias=TEAM):
    return {
        "model": "gpt-4o-mini",
        "litellm_params": {
            "metadata": {"headers": headers, "user_api_key_team_alias": team_alias},
        },
    }


def submitted_args(mock_submit):
    assert mock_submit.call_count == 1
    return mock_submit.call_args[0][1]


@pytest.fixture(autouse=True)
def _no_guardrail_owns_metering():
    """A guardrail registered by another test would make the callback skip metering."""
    _metering_owner.reset_metering_owner()
    yield
    _metering_owner.reset_metering_owner()


@patch.object(proxy_mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(proxy_mw, "get_client", return_value=object())
@patch.object(proxy_mw, "submit_ai_event")
class TestBlankAttributionHeaders:
    @staticmethod
    def _succeed(headers, team_alias=TEAM):
        run_hook(proxy_mw.proxy_handler_instance.async_log_success_event(
            request_kwargs(headers, team_alias), SubscriptableResponse(), NOW, NOW
        ))

    @pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
    def test_a_blank_organization_header_falls_back_to_the_key_team(
        self, mock_submit, _get_client, _run, blank
    ):
        self._succeed({ORG_HEADER: blank})

        assert submitted_args(mock_submit)["organization_name"] == TEAM

    def test_a_blank_organization_header_falls_back_to_the_deprecated_header(
        self, mock_submit, _get_client, _run
    ):
        self._succeed({ORG_HEADER: "", OLD_ORG_HEADER: "acme-old"})

        assert submitted_args(mock_submit)["organization_name"] == "acme-old"

    def test_a_blank_deprecated_organization_header_is_ignored(
        self, mock_submit, _get_client, _run, caplog
    ):
        caplog.set_level("WARNING", logger="revenium_middleware.extension")

        self._succeed({OLD_ORG_HEADER: "  "})

        assert submitted_args(mock_submit)["organization_name"] == TEAM
        assert not [r for r in caplog.records if OLD_ORG_HEADER in r.getMessage()]

    def test_a_whitespace_team_alias_sends_no_organization(
        self, mock_submit, _get_client, _run
    ):
        self._succeed({}, team_alias="  ")

        assert submitted_args(mock_submit)["organization_name"] is None

    @pytest.mark.parametrize("blank", ["", "  "], ids=["empty", "whitespace"])
    def test_a_blank_product_header_falls_back_to_the_deprecated_product_id(
        self, mock_submit, _get_client, _run, blank
    ):
        self._succeed({PRODUCT_HEADER: blank, OLD_PRODUCT_HEADER: "p1"})

        assert submitted_args(mock_submit)["product_name"] == "p1"

    def test_a_blank_product_header_with_nothing_else_sends_no_product(
        self, mock_submit, _get_client, _run
    ):
        self._succeed({PRODUCT_HEADER: ""})

        assert submitted_args(mock_submit)["product_name"] is None

    def test_the_failure_path_falls_back_too(self, mock_submit, _get_client, _run):
        run_hook(proxy_mw.proxy_handler_instance.async_log_failure_event(
            request_kwargs({ORG_HEADER: "", PRODUCT_HEADER: " ", OLD_PRODUCT_HEADER: "p1"}),
            SubscriptableResponse(), NOW, NOW,
        ))

        args = submitted_args(mock_submit)
        assert args["organization_name"] == TEAM
        assert args["product_name"] == "p1"

    @pytest.mark.parametrize("organization, product", [
        ("AcmeCorp", "Checkout"),
        (" AcmeCorp ", " Checkout "),
    ], ids=["plain", "padded"])
    def test_filled_headers_are_unchanged(
        self, mock_submit, _get_client, _run, organization, product
    ):
        self._succeed({ORG_HEADER: organization, PRODUCT_HEADER: product})

        args = submitted_args(mock_submit)
        assert args["organization_name"] == organization
        assert args["product_name"] == product


PIN_SETTING = "REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM"


@patch.object(proxy_mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(proxy_mw, "get_client", return_value=object())
@patch.object(proxy_mw, "submit_ai_event")
class TestOrganizationPinnedToTheKeyTeam:
    """With the operator's pin on, the key's team beats a caller's organization header (BACK-4363)."""

    @staticmethod
    def _succeed(kwargs):
        run_hook(proxy_mw.proxy_handler_instance.async_log_success_event(
            kwargs, SubscriptableResponse(), NOW, NOW
        ))

    @pytest.mark.parametrize("header", [ORG_HEADER, OLD_ORG_HEADER])
    def test_the_callback_record_names_the_pinned_key_team(
        self, mock_submit, _get_client, _run, monkeypatch, header
    ):
        monkeypatch.setenv(PIN_SETTING, "true")

        self._succeed(request_kwargs({header: "SomeoneElse"}))

        assert submitted_args(mock_submit)["organization_name"] == TEAM

    def test_the_callback_failure_record_names_the_pinned_key_team(
        self, mock_submit, _get_client, _run, monkeypatch
    ):
        monkeypatch.setenv(PIN_SETTING, "true")

        run_hook(proxy_mw.proxy_handler_instance.async_log_failure_event(
            request_kwargs({ORG_HEADER: "SomeoneElse"}), SubscriptableResponse(), NOW, NOW,
        ))

        assert submitted_args(mock_submit)["organization_name"] == TEAM

    def test_without_the_setting_the_callback_keeps_the_header(
        self, mock_submit, _get_client, _run, monkeypatch
    ):
        monkeypatch.delenv(PIN_SETTING, raising=False)

        self._succeed(request_kwargs({ORG_HEADER: "SomeoneElse"}))

        assert submitted_args(mock_submit)["organization_name"] == "SomeoneElse"

    def test_a_blank_team_alias_does_not_pin_the_callback(
        self, mock_submit, _get_client, _run, monkeypatch
    ):
        monkeypatch.setenv(PIN_SETTING, "true")

        self._succeed(request_kwargs({ORG_HEADER: "SomeoneElse"}, team_alias="  "))

        assert submitted_args(mock_submit)["organization_name"] == "SomeoneElse"

    def test_the_callback_pins_the_key_team_on_the_messages_route(
        self, mock_submit, _get_client, _run, monkeypatch
    ):
        """On /v1/messages LiteLLM writes the key's fields to litellm_metadata, not metadata."""
        monkeypatch.setenv(PIN_SETTING, "true")

        self._succeed({
            "model": "claude-sonnet-4-5",
            "litellm_params": {
                "litellm_metadata": {
                    "headers": {ORG_HEADER: "SomeoneElse"},
                    "user_api_key_team_alias": TEAM,
                },
                "metadata": {"user_id": "x"},
            },
        })

        assert submitted_args(mock_submit)["organization_name"] == TEAM

    def test_the_callback_ignores_a_team_alias_planted_in_the_messages_body(
        self, mock_submit, _get_client, _run, monkeypatch
    ):
        """On /v1/messages ``metadata`` is the caller's own body, never the key's team."""
        monkeypatch.setenv(PIN_SETTING, "true")

        self._succeed({
            "model": "claude-sonnet-4-5",
            "litellm_params": {
                "litellm_metadata": {
                    "headers": {ORG_HEADER: "SomeoneElse"},
                    "user_api_key_team_alias": TEAM,
                },
                "metadata": {"user_id": "x", "user_api_key_team_alias": "Planted"},
            },
        })

        assert submitted_args(mock_submit)["organization_name"] == TEAM


@pytest.mark.parametrize("value, pinned", [
    ("1", True),
    ("yes", True),
    ("on", True),
    ("TRUE", True),
    ("true", True),
    (" TRUE\n", True),
    ("false", False),
    ("0", False),
    ("", False),
    (None, False),
], ids=["1", "yes", "on", "TRUE", "true", "padded-true", "false", "0", "empty", "unset"])
def test_the_pin_setting_values(monkeypatch, value, pinned):
    from revenium_middleware._core.config import is_organization_pinned_to_key_team

    if value is None:
        monkeypatch.delenv(PIN_SETTING, raising=False)
    else:
        monkeypatch.setenv(PIN_SETTING, value)

    assert is_organization_pinned_to_key_team() is pinned


@pytest.mark.parametrize("value, warned", [
    (" TRUE\n", False),
    ("enabled", True),
    ("true", False),
    ("false", False),
    ("off", False),
    (" off\n", False),
    ("", False),
    (None, False),
], ids=["padded-true", "unknown-word", "true", "false", "off", "padded-off", "empty", "unset"])
def test_an_unrecognised_pin_value_leaves_the_pin_off_and_warns_at_startup(
    monkeypatch, caplog, value, warned
):
    from revenium_middleware._core.config import is_organization_pinned_to_key_team

    if value is None:
        monkeypatch.delenv(PIN_SETTING, raising=False)
    else:
        monkeypatch.setenv(PIN_SETTING, value)

    with caplog.at_level("INFO", logger="revenium_middleware.extension"):
        proxy_mw.log_organization_pin_setting()

    warnings = [r for r in caplog.records if r.levelname == "WARNING" and PIN_SETTING in r.getMessage()]
    assert bool(warnings) is warned
    if warned:
        assert is_organization_pinned_to_key_team() is False
