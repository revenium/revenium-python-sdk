"""Tests for the validate_api_key helper in revenium_middleware._core.config."""

import logging
import warnings

import pytest

from revenium_middleware._core import config as config_module
from revenium_middleware._core.config import (
    Config,
    env_flag_enabled,
    get_profitstream_base_url,
    is_capture_error_messages_enabled,
    is_capture_prompts_enabled,
    is_selective_metering_enabled,
    is_shared_call_id_enabled,
    resolve_write_api_key,
    validate_api_key,
)
from revenium_middleware._core.enforcement import (
    _fail_mode_is_closed,
    is_bypass_enabled,
    is_circuit_breaker_enabled,
)


class TestValidateApiKeyPrefixes:
    """Cover both accepted Revenium prefixes."""

    def test_accepts_hak_prefix(self):
        validate_api_key("hak_abc123def456")

    def test_accepts_rev_mk_prefix(self):
        validate_api_key("rev_mk_3By1Ra6_abc123")

    def test_accepts_rev_sk_prefix(self):
        validate_api_key("rev_sk_3By1Ra6_abc123")


class TestValidateApiKeyRejections:
    """Reject anything that does not start with hak_ or rev_."""

    def test_rejects_unknown_prefix(self):
        with pytest.raises(ValueError, match='should start with "hak_" or "rev_"'):
            validate_api_key("invalid_key")

    def test_rejects_uppercase_hak(self):
        with pytest.raises(ValueError):
            validate_api_key("HAK_abc")

    def test_rejects_uppercase_rev(self):
        with pytest.raises(ValueError):
            validate_api_key("REV_abc")

    @pytest.mark.parametrize(
        "bad_key",
        [
            "rev-key-abc123",        # hyphen, not underscore
            "sk-test-123456",        # OpenAI
            "AIzaSyABC123-456_789",  # Google
            "pplx-abc123",           # Perplexity
            "sk-ant-abc123",         # Anthropic
        ],
    )
    def test_rejects_other_provider_prefixes(self, bad_key):
        with pytest.raises(ValueError, match='should start with "hak_" or "rev_"'):
            validate_api_key(bad_key)


class TestValidApiKeyPrefixesConstant:
    """Constant must remain stable; downstream tests/scripts reference these strings."""

    def test_prefixes_are_exactly_hak_and_rev(self):
        assert Config.VALID_API_KEY_PREFIXES == ("hak_", "rev_")

    def test_write_env_var_name_is_exact(self):
        assert Config.ENV_REVENIUM_WRITE_API_KEY == "REVENIUM_WRITE_API_KEY"


class TestResolveWriteApiKey:
    def test_prefers_explicit_then_write_then_outcome_then_metering(self, monkeypatch):
        monkeypatch.setenv(Config.ENV_REVENIUM_WRITE_API_KEY, "rev_sk_TENANT_write")
        monkeypatch.setenv(Config.ENV_REVENIUM_OUTCOME_API_KEY, "rev_sk_TENANT_legacy")
        monkeypatch.setenv(Config.ENV_REVENIUM_API_KEY, "rev_mk_TENANT_metering")

        assert resolve_write_api_key("rev_sk_TENANT_explicit") == "rev_sk_TENANT_explicit"
        assert resolve_write_api_key() == "rev_sk_TENANT_write"

        monkeypatch.delenv(Config.ENV_REVENIUM_WRITE_API_KEY)
        with pytest.warns(DeprecationWarning, match="REVENIUM_WRITE_API_KEY"):
            assert resolve_write_api_key() == "rev_sk_TENANT_legacy"

        monkeypatch.delenv(Config.ENV_REVENIUM_OUTCOME_API_KEY)
        assert resolve_write_api_key() == "rev_mk_TENANT_metering"

    def test_both_write_and_outcome_set_uses_write_without_deprecation(self, monkeypatch):
        monkeypatch.setenv(Config.ENV_REVENIUM_WRITE_API_KEY, "rev_sk_TENANT_write")
        monkeypatch.setenv(Config.ENV_REVENIUM_OUTCOME_API_KEY, "rev_sk_TENANT_legacy")

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            assert resolve_write_api_key() == "rev_sk_TENANT_write"

        assert recorded == []

    def test_a_padded_key_from_any_source_is_trimmed(self, monkeypatch):
        for name in (Config.ENV_REVENIUM_WRITE_API_KEY, Config.ENV_REVENIUM_OUTCOME_API_KEY,
                     Config.ENV_REVENIUM_API_KEY):
            monkeypatch.delenv(name, raising=False)

        assert resolve_write_api_key("rev_sk_TENANT_explicit\n") == "rev_sk_TENANT_explicit"

        monkeypatch.setenv(Config.ENV_REVENIUM_API_KEY, "rev_mk_TENANT_metering \n")
        assert resolve_write_api_key() == "rev_mk_TENANT_metering"

        monkeypatch.setenv(Config.ENV_REVENIUM_OUTCOME_API_KEY, "rev_sk_TENANT_legacy\r\n")
        with pytest.warns(DeprecationWarning):
            assert resolve_write_api_key() == "rev_sk_TENANT_legacy"

        monkeypatch.setenv(Config.ENV_REVENIUM_WRITE_API_KEY, " rev_sk_TENANT_write\n")
        assert resolve_write_api_key() == "rev_sk_TENANT_write"

        monkeypatch.setenv(Config.ENV_REVENIUM_WRITE_API_KEY, " \n")
        monkeypatch.delenv(Config.ENV_REVENIUM_OUTCOME_API_KEY)
        assert resolve_write_api_key("  ") == "rev_mk_TENANT_metering"

        monkeypatch.delenv(Config.ENV_REVENIUM_API_KEY)
        assert resolve_write_api_key() is None


USAGE_ENV = Config.ENV_REVENIUM_BASE_URL
OUTCOME_ENV = Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL


@pytest.mark.parametrize("explicit, outcome_env, usage_env, expected", [
    ("https://explicit.example.test", "https://outcomes.example.test", "https://usage.example.test",
     "https://explicit.example.test"),
    (None, "https://outcomes.example.test", "https://usage.example.test", "https://outcomes.example.test"),
    (None, "https://outcomes.example.test/ctx/", "https://usage.example.test",
     "https://outcomes.example.test/ctx/"),
    (None, None, "https://usage.example.test", "https://usage.example.test"),
    (None, None, "https://usage.example.test/meter", "https://usage.example.test"),
    (None, None, "https://usage.example.test/Meter", "https://usage.example.test"),
    (None, None, "https://usage.example.test/meter/", "https://usage.example.test"),
    (None, None, "https://usage.example.test:8443/meter", "https://usage.example.test:8443"),
    (None, None, "https://proxy.corp.test/revenium/meter", "https://proxy.corp.test/revenium"),
    (None, None, "https://proxy.corp.test/revenium", "https://proxy.corp.test/revenium"),
    (None, None, "  https://usage.example.test/meter  ", "https://usage.example.test"),
    (None, None, "https://meter", "https://meter"),
    ("", "", "https://usage.example.test", "https://usage.example.test"),
    (None, "   ", "https://usage.example.test", "https://usage.example.test"),
    (None, None, "", Config.DEFAULT_BASE_URL),
    (None, None, "usage.example.test/meter", Config.DEFAULT_BASE_URL),
    (None, None, None, Config.DEFAULT_BASE_URL),
], ids=[
    "explicit-wins", "outcome-env-over-usage-env", "outcome-context-path-intact",
    "usage-env-plain", "usage-env-with-meter", "usage-env-with-Meter", "usage-env-trailing-slash",
    "usage-env-with-port", "usage-env-path-prefix-kept", "usage-env-prefix-without-meter",
    "usage-env-whitespace", "host-named-meter", "empty-explicit-and-outcome-env",
    "blank-outcome-env", "empty-usage-env", "usage-env-without-scheme", "nothing-set",
])
def test_profitstream_address_precedence(monkeypatch, explicit, outcome_env, usage_env, expected):
    for name, value in ((OUTCOME_ENV, outcome_env), (USAGE_ENV, usage_env)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert get_profitstream_base_url(explicit) == expected


def test_profitstream_address_follows_a_usage_address_passed_in_code(monkeypatch):
    monkeypatch.delenv(OUTCOME_ENV, raising=False)
    monkeypatch.setenv(USAGE_ENV, "https://usage.example.test")

    assert get_profitstream_base_url(
        metering_base_url="https://self-hosted.example.test/meter"
    ) == "https://self-hosted.example.test"


SHARED_CALL_ID_ENV = Config.ENV_REVENIUM_LITELLM_SHARED_CALL_ID


class TestSharedCallIdFlag:
    """BACK-4019: the shared call id is on unless the operator turns it off.

    A deployment template that passes the variable through empty must not
    switch it off; only a real opt-out value does.
    """

    @pytest.mark.parametrize("value", ["", " ", "\t", " true", "TRUE "],
                             ids=["empty", "space", "tab", "leading-space-true", "upper-trailing-space"])
    def test_a_blank_or_padded_value_leaves_the_shared_call_id_on(self, monkeypatch, value):
        monkeypatch.setenv(SHARED_CALL_ID_ENV, value)

        assert is_shared_call_id_enabled() is True

    @pytest.mark.parametrize("value", ["false", " false ", "FALSE", "0"])
    def test_false_still_turns_the_shared_call_id_off(self, monkeypatch, value):
        monkeypatch.setenv(SHARED_CALL_ID_ENV, value)

        assert is_shared_call_id_enabled() is False

    def test_an_unset_flag_is_on(self, monkeypatch):
        monkeypatch.delenv(SHARED_CALL_ID_ENV, raising=False)

        assert is_shared_call_id_enabled() is True


FLAG_READERS = [
    (Config.ENV_CIRCUIT_BREAKER_ENABLED, is_circuit_breaker_enabled),
    (Config.ENV_REVENIUM_BYPASS, is_bypass_enabled),
    (Config.ENV_REVENIUM_CAPTURE_PROMPTS, is_capture_prompts_enabled),
    (Config.ENV_REVENIUM_CAPTURE_ERROR_MESSAGES, is_capture_error_messages_enabled),
    ("REVENIUM_SELECTIVE_METERING", is_selective_metering_enabled),
    (Config.ENV_REVENIUM_INSTALL_SIGNAL_HANDLERS,
     lambda: env_flag_enabled(Config.ENV_REVENIUM_INSTALL_SIGNAL_HANDLERS)),
    ("REVENIUM_VERBOSE_STARTUP", lambda: env_flag_enabled("REVENIUM_VERBOSE_STARTUP")),
]


def _warnings(caplog):
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


class TestEnvironmentFlags:
    """An on/off or mode setting ignores surrounding whitespace and names a value it does not recognise."""

    @pytest.fixture(autouse=True)
    def no_setting_warned_yet(self, monkeypatch, caplog):
        monkeypatch.setattr(config_module, "_warned_env_settings", set(), raising=False)
        caplog.set_level(logging.WARNING)

    @pytest.mark.parametrize("value", ["true ", "true\r", " TRUE\n"], ids=["space", "carriage-return", "upper-newline"])
    @pytest.mark.parametrize("setting, reader", FLAG_READERS, ids=[setting for setting, _ in FLAG_READERS])
    def test_a_padded_true_turns_each_flag_on(self, monkeypatch, setting, reader, value):
        monkeypatch.setenv(setting, value)

        assert reader() is True

    @pytest.mark.parametrize("value", ["false ", "0\r", "", "  "], ids=["false-space", "zero-cr", "empty", "blank"])
    @pytest.mark.parametrize("setting, reader", FLAG_READERS, ids=[setting for setting, _ in FLAG_READERS])
    def test_a_padded_false_or_blank_stays_off_without_a_warning(self, monkeypatch, caplog, setting, reader, value):
        monkeypatch.setenv(setting, value)

        assert reader() is False
        assert _warnings(caplog) == []

    def test_an_unrecognised_flag_value_warns_once_naming_the_setting(self, monkeypatch, caplog):
        monkeypatch.setenv(Config.ENV_CIRCUIT_BREAKER_ENABLED, "enabled")

        assert [is_circuit_breaker_enabled() for _call in range(3)] == [False, False, False]
        warned = _warnings(caplog)
        assert len(warned) == 1
        assert all(part in warned[0] for part in ("REVENIUM_CIRCUIT_BREAKER_ENABLED", "'enabled'", "false"))

    def test_an_unrecognised_fail_mode_warns_once_and_stays_open(self, monkeypatch, caplog):
        monkeypatch.setenv(Config.ENV_REVENIUM_CB_FAIL_MODE, "fail-closed")

        assert [_fail_mode_is_closed() for _call in range(3)] == [False, False, False]
        warned = _warnings(caplog)
        assert len(warned) == 1
        assert all(part in warned[0] for part in ("REVENIUM_CB_FAIL_MODE", "'fail-closed'", "open"))
