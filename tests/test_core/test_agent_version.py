"""agentVersion resolution, length capping and separation from agenticJobVersion.

The wire field is the AI agent's own version. ``agenticJobVersion`` -- the
agentic job definition's version -- is a different field that already lived in
_core/fields.py, so these tests pin the two apart as much as they pin the new
field's behaviour.
"""
import inspect
import os
from unittest.mock import patch

import pytest

from revenium_middleware._core.fields import (
    AGENT_VERSION_FIELD_MAP,
    AGENTIC_JOB_FIELD_MAP,
    extract_agent_version_field,
    extract_agentic_job_fields,
)
from revenium_middleware._core.trace_fields import (
    AGENT_VERSION_MAX_LENGTH,
    get_agent_version,
    get_ticket_id,
    validate_agent_version,
)
from revenium_middleware._core.context import (
    clear_injected_metadata,
    merge_metadata,
    set_injected_metadata,
)


class TestAgentVersionFieldMap:
    def test_registers_both_wire_aliases(self):
        assert AGENT_VERSION_FIELD_MAP["agent_version"] == (
            "agent_version",
            "agentVersion",
        )

    def test_is_not_folded_into_the_agentic_job_map(self):
        """Sharing a key with agenticJobVersion would overwrite job attribution."""
        assert "agent_version" not in AGENTIC_JOB_FIELD_MAP
        assert "agentVersion" not in AGENTIC_JOB_FIELD_MAP
        assert "agenticJobVersion" not in AGENT_VERSION_FIELD_MAP

    def test_snake_case_alias(self):
        assert extract_agent_version_field({"agent_version": "1.4.2"}) == {
            "agent_version": "1.4.2"
        }

    def test_camel_case_alias(self):
        assert extract_agent_version_field({"agentVersion": "1.4.2"}) == {
            "agent_version": "1.4.2"
        }

    def test_snake_case_takes_precedence(self):
        source = {"agent_version": "snake", "agentVersion": "camel"}
        assert extract_agent_version_field(source) == {"agent_version": "snake"}

    def test_absent_field_omitted_not_none(self):
        """A None would reach the wire as an explicit null; NotGiven must survive."""
        assert extract_agent_version_field({"trace_id": "unrelated"}) == {}
        assert extract_agent_version_field({}) == {}
        assert extract_agent_version_field(None) == {}

    def test_does_not_read_the_agentic_job_version(self):
        assert extract_agent_version_field({"agentic_job_version": "job-9"}) == {}
        assert extract_agent_version_field({"agenticJobVersion": "job-9"}) == {}

    def test_both_versions_resolve_independently(self):
        source = {"agent_version": "agent-1", "agentic_job_version": "job-9"}
        assert extract_agent_version_field(source) == {"agent_version": "agent-1"}
        assert extract_agentic_job_fields(source)["agenticJobVersion"] == "job-9"


class TestValidateAgentVersion:
    def test_cap_matches_the_backend(self):
        """MAX_AGENT_VERSION_LENGTH in ReveniumAttributes.kt on the metering service."""
        assert AGENT_VERSION_MAX_LENGTH == 64

    def test_value_at_the_limit_passes_through(self):
        value = "v" * AGENT_VERSION_MAX_LENGTH
        assert validate_agent_version(value) == value

    def test_over_long_value_is_truncated_not_rejected(self, caplog):
        """Same disposition as an over-long ticketId: cap and warn, never drop."""
        value = "v" * (AGENT_VERSION_MAX_LENGTH + 50)
        with caplog.at_level("WARNING"):
            result = validate_agent_version(value)
        assert result == "v" * AGENT_VERSION_MAX_LENGTH
        assert "agentVersion" in caplog.text

    def test_empty_value_is_none(self):
        assert validate_agent_version("") is None

    def test_capping_applies_through_the_reader(self):
        value = "v" * (AGENT_VERSION_MAX_LENGTH + 50)
        assert get_agent_version({"agent_version": value}) == (
            "v" * AGENT_VERSION_MAX_LENGTH
        )


class TestGetAgentVersion:
    def test_reads_both_aliases(self):
        assert get_agent_version({"agent_version": "1.4.2"}) == "1.4.2"
        assert get_agent_version({"agentVersion": "1.4.2"}) == "1.4.2"

    def test_none_when_unset(self):
        assert get_agent_version({}) is None
        assert get_agent_version(None) is None
        assert get_agent_version() is None

    def test_no_env_var_fallback(self):
        """The agent version is per-call attribution, not a process property."""
        with patch.dict(os.environ, {"REVENIUM_AGENT_VERSION": "env-1"}, clear=False):
            assert get_agent_version({}) is None


class TestTypedClientSurface:
    """All four /v2/ai paths accept the field, together.

    BACK-2556 shipped a wire field on completions only and it was silently
    dropped on audio, image and video. These assertions fail loudly if a
    future edit reintroduces that asymmetry.
    """

    @pytest.mark.parametrize("media", ["completion", "audio", "image", "video"])
    def test_params_typeddict_declares_the_alias(self, media):
        import importlib

        module = importlib.import_module(
            f"revenium_middleware._metering.types.ai_create_{media}_params"
        )
        source = inspect.getsource(module)
        assert 'agent_version: Annotated[str, PropertyInfo(alias="agentVersion")]' in source

    @pytest.mark.parametrize(
        "resource_name", ["AIResource", "AsyncAIResource"]
    )
    @pytest.mark.parametrize("media", ["completion", "audio", "image", "video"])
    def test_create_method_accepts_the_keyword(self, resource_name, media):
        from revenium_middleware._metering.resources import ai as ai_module

        resource = getattr(ai_module, resource_name)
        method = getattr(resource, f"create_{media}")
        assert "agent_version" in inspect.signature(method).parameters


class TestNonStringAgentVersionIsDropped:
    """A malformed agent version must never break the metering event.

    ``validate_agent_version`` runs after the provider call has already
    returned (and again on stream finalisation), so a TypeError from len()
    or slicing would surface as the failure of a call that succeeded.
    """

    @pytest.mark.parametrize("value", [123, 12.5, True, [1, 2], {"a": 1}, ("x",), object()])
    def test_truthy_non_strings_are_dropped_not_coerced(self, value):
        assert validate_agent_version(value) is None
        assert get_agent_version({"agent_version": value}) is None

    @pytest.mark.parametrize("value", [0, 0.0, [], {}, ""])
    def test_falsy_non_strings_are_dropped(self, value):
        assert validate_agent_version(value) is None
        assert get_agent_version({"agent_version": value}) is None

    def test_value_is_never_stringified(self):
        """str(123) would invent attribution the caller never wrote."""
        assert get_agent_version({"agent_version": 123}) != "123"

    def test_camel_case_alias_is_guarded_too(self):
        assert get_agent_version({"agentVersion": [1, 2]}) is None

    def test_a_valid_version_still_resolves(self):
        assert get_agent_version({"agent_version": "1.4.2"}) == "1.4.2"


class TestScopedVersusDirectMetadataPrecedence:
    """Direct-call metadata outranks scoped metadata whatever the spelling.

    ``merge_metadata`` merges by literal key, so before this fix a scoped
    ``agent_version`` survived alongside a direct ``agentVersion`` and the
    alias precedence downstream resolved the scoped value -- inverting the
    precedence merge_metadata documents.
    """

    def _merge(self, scoped, direct):
        set_injected_metadata(scoped)
        try:
            return merge_metadata(direct)
        finally:
            clear_injected_metadata()

    def test_direct_camel_case_beats_scoped_snake_case(self):
        merged = self._merge({"agent_version": "scoped"}, {"agentVersion": "direct"})
        assert get_agent_version(merged) == "direct"

    def test_direct_snake_case_beats_scoped_camel_case(self):
        merged = self._merge({"agentVersion": "scoped"}, {"agent_version": "direct"})
        assert get_agent_version(merged) == "direct"

    def test_the_losing_spelling_is_removed_not_just_outranked(self):
        merged = self._merge({"agent_version": "scoped"}, {"agentVersion": "direct"})
        assert merged == {"agentVersion": "direct"}

    def test_ticket_id_shared_the_bug_and_is_fixed_with_it(self):
        """get_ticket_id reads ticketId first, so the same merge left a
        scoped camelCase value outranking a direct snake_case one."""
        merged = self._merge({"ticketId": "scoped"}, {"ticket_id": "direct"})
        assert get_ticket_id(merged) == "direct"
        merged = self._merge({"ticket_id": "scoped"}, {"ticketId": "direct"})
        assert get_ticket_id(merged) == "direct"

    def test_scoped_fields_the_direct_call_omits_survive(self):
        merged = self._merge(
            {"agent_version": "scoped", "trace_id": "scoped-trace"},
            {"agentVersion": "direct"},
        )
        assert merged["trace_id"] == "scoped-trace"
        assert get_agent_version(merged) == "direct"

    def test_scoped_only_metadata_is_untouched(self):
        merged = self._merge({"agent_version": "scoped"}, {})
        assert get_agent_version(merged) == "scoped"

    def test_non_string_keys_do_not_break_the_merge(self):
        merged = self._merge({1: "scoped"}, {"agent_version": "direct"})
        assert merged == {1: "scoped", "agent_version": "direct"}
