"""Tests for the validate_api_key helper in revenium_middleware._core.config."""

import warnings

import pytest

from revenium_middleware._core.config import Config, resolve_write_api_key, validate_api_key


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
