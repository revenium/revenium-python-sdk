import logging
import os
from unittest.mock import patch

import pytest

from revenium_middleware._core.fields import (
    extract_agentic_job_fields,
    extract_common_metadata,
    extract_field_with_fallback,
    extract_org_and_product,
    extract_with_aliases,
    merge_extra_body,
)
from revenium_middleware._core.context import (
    get_agentic_job_fields,
    set_agentic_job_fields,
    _agentic_job_context,
)


class TestExtractFieldWithFallback:
    def _call(self, source, **overrides):
        defaults = {
            "new_snake": "organization_name",
            "new_camel": "organizationName",
            "old_snake": "organization_id",
            "old_camel": "organizationId",
            "field_label": "organization",
        }
        defaults.update(overrides)
        return extract_field_with_fallback(source, **defaults)

    def test_snake_case_has_highest_precedence(self):
        source = {
            "organization_name": "snake",
            "organizationName": "camel",
            "organization_id": "old_snake",
            "organizationId": "old_camel",
        }
        assert self._call(source) == "snake"

    def test_camel_case_is_second_precedence(self):
        source = {
            "organizationName": "camel",
            "organization_id": "old_snake",
            "organizationId": "old_camel",
        }
        assert self._call(source) == "camel"

    def test_deprecated_snake_is_third_precedence(self):
        source = {
            "organization_id": "old_snake",
            "organizationId": "old_camel",
        }
        assert self._call(source) == "old_snake"

    def test_deprecated_camel_is_last_precedence(self):
        source = {"organizationId": "old_camel"}
        assert self._call(source) == "old_camel"

    def test_returns_none_when_no_field_present(self):
        assert self._call({}) is None

    def test_none_values_fall_through(self):
        source = {
            "organization_name": None,
            "organizationName": None,
            "organization_id": "old_snake",
        }
        assert self._call(source) == "old_snake"

    @pytest.mark.parametrize("falsy_value", [0, False, ""])
    def test_falsy_values_are_preserved(self, falsy_value):
        source = {
            "organization_name": falsy_value,
            "organizationName": "should_not_reach",
        }
        assert self._call(source) == falsy_value

    def test_deprecation_warning_when_only_deprecated_fields(self, caplog):
        source = {"organizationId": "old"}
        with caplog.at_level(logging.WARNING):
            self._call(source)
        assert "deprecated" in caplog.text.lower()

    def test_no_deprecation_warning_when_new_fields_present(self, caplog):
        source = {"organization_name": "new", "organizationId": "old"}
        with caplog.at_level(logging.WARNING):
            self._call(source)
        assert "deprecated" not in caplog.text.lower()

    def test_logger_warning_dedups_per_field_pair(self, caplog):
        """logger.warning has no built-in dedup; the module gates it on a (old, new) pair
        so high-volume callers using deprecated aliases don't get a log flood per call."""
        source = {"organizationId": "old"}
        with caplog.at_level(logging.WARNING):
            self._call(source)
            self._call(source)
            self._call(source)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "deprecated" in r.getMessage().lower()]
        assert len(warning_records) == 1

    def test_logger_warning_dedup_is_per_pair_not_global(self, caplog):
        """Different deprecated-field pairs each get one warning; the dedup is keyed by pair."""
        with caplog.at_level(logging.WARNING):
            self._call({"organizationId": "org"})
            self._call({"productId": "prod"}, new_snake="product_name", new_camel="productName",
                       old_snake="product_id", old_camel="productId", field_label="product")
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "deprecated" in r.getMessage().lower()]
        assert len(warning_records) == 2

    def test_logger_warning_dedup_is_thread_safe(self, caplog):
        """Many threads racing on the same first-call must produce at most one logger.warning.
        Without a lock around the check-then-add, concurrent threads could all see "not present"
        and emit duplicate startup lines before the set converges."""
        import threading

        source = {"organizationId": "old"}
        thread_count = 32
        barrier = threading.Barrier(thread_count)

        def worker():
            barrier.wait()
            self._call(source)

        with caplog.at_level(logging.WARNING):
            threads = [threading.Thread(target=worker) for _ in range(thread_count)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "deprecated" in r.getMessage().lower()]
        assert len(warning_records) == 1


class TestExtractWithAliases:
    def test_snake_case_preferred(self):
        source = {"trace_id": "snake", "traceId": "camel"}
        assert extract_with_aliases(source, "trace_id", "traceId") == "snake"

    def test_camel_case_fallback(self):
        source = {"traceId": "camel"}
        assert extract_with_aliases(source, "trace_id", "traceId") == "camel"

    def test_returns_none_when_absent(self):
        assert extract_with_aliases({}, "trace_id", "traceId") is None


class TestExtractOrgAndProduct:
    def test_returns_tuple(self):
        source = {"organization_name": "org", "product_name": "prod"}
        assert extract_org_and_product(source) == ("org", "prod")

    def test_camel_case_aliases(self):
        source = {"organizationName": "org", "productName": "prod"}
        assert extract_org_and_product(source) == ("org", "prod")

    def test_deprecated_snake_aliases(self):
        source = {"organization_id": "org", "product_id": "prod"}
        assert extract_org_and_product(source) == ("org", "prod")

    def test_deprecated_camel_aliases(self):
        source = {"organizationId": "org", "productId": "prod"}
        assert extract_org_and_product(source) == ("org", "prod")


class TestExtractCommonMetadata:
    def test_returns_all_five_fields(self):
        source = {
            "trace_id": "t1",
            "task_type": "tt",
            "subscription_id": "s1",
            "agent": "a1",
            "response_quality_score": 0.9,
        }
        result = extract_common_metadata(source)
        assert result == {
            "trace_id": "t1",
            "task_type": "tt",
            "subscription_id": "s1",
            "agent": "a1",
            "response_quality_score": 0.9,
        }

    def test_trace_id_from_camel(self):
        assert extract_common_metadata({"traceId": "t"})["trace_id"] == "t"

    def test_task_type_from_camel(self):
        assert extract_common_metadata({"taskType": "tt"})["task_type"] == "tt"

    def test_subscription_id_from_camel(self):
        assert extract_common_metadata({"subscriptionId": "s"})["subscription_id"] == "s"

    def test_response_quality_score_from_camel(self):
        assert extract_common_metadata({"responseQualityScore": 0.5})["response_quality_score"] == 0.5

    def test_agent_has_no_camel_variant(self):
        source = {"agent": "a1"}
        assert extract_common_metadata(source)["agent"] == "a1"

    def test_missing_fields_return_none(self):
        result = extract_common_metadata({})
        for value in result.values():
            assert value is None


class TestExtractAgenticJobFields:
    def test_snake_case_aliases(self):
        source = {
            "agentic_job_id": "id1",
            "agentic_job_name": "name1",
            "agentic_job_type": "type1",
            "agentic_job_version": "v1",
        }
        result = extract_agentic_job_fields(source)
        assert result == {
            "agenticJobId": "id1",
            "agenticJobName": "name1",
            "agenticJobType": "type1",
            "agenticJobVersion": "v1",
        }

    def test_camel_case_aliases(self):
        source = {
            "agenticJobId": "id1",
            "agenticJobName": "name1",
            "agenticJobType": "type1",
            "agenticJobVersion": "v1",
        }
        result = extract_agentic_job_fields(source)
        assert result == {
            "agenticJobId": "id1",
            "agenticJobName": "name1",
            "agenticJobType": "type1",
            "agenticJobVersion": "v1",
        }

    def test_all_four_fields_extracted(self):
        source = {"agentic_job_id": "id", "agenticJobName": "n", "agentic_job_type": "t", "agenticJobVersion": "v"}
        result = extract_agentic_job_fields(source)
        assert len(result) == 4

    def test_missing_fields_omitted(self):
        source = {"agentic_job_id": "id1"}
        result = extract_agentic_job_fields(source)
        assert result == {"agenticJobId": "id1"}
        assert "agenticJobName" not in result

    def test_empty_dict_when_no_agentic_fields(self):
        assert extract_agentic_job_fields({}) == {}

    def test_mixed_snake_and_camel(self):
        source = {"agentic_job_id": "snake_id", "agenticJobName": "camel_name"}
        result = extract_agentic_job_fields(source)
        assert result == {"agenticJobId": "snake_id", "agenticJobName": "camel_name"}


class TestMergeExtraBody:
    def test_returns_none_when_both_empty(self):
        assert merge_extra_body(None, {}) is None
        assert merge_extra_body({}, {}) is None

    def test_preserves_existing_extra_body_keys(self):
        existing = {"key1": "val1", "key2": "val2"}
        result = merge_extra_body(existing, {"key3": "val3"})
        assert result["key1"] == "val1"
        assert result["key2"] == "val2"
        assert result["key3"] == "val3"

    def test_adds_agentic_fields(self):
        result = merge_extra_body(None, {"agenticJobId": "id1"})
        assert result == {"agenticJobId": "id1"}

    def test_existing_keys_not_overwritten(self):
        existing = {"agenticJobId": "original"}
        result = merge_extra_body(existing, {"agenticJobId": "new"})
        assert result["agenticJobId"] == "original"

    def test_none_existing_with_agentic_fields(self):
        result = merge_extra_body(None, {"agenticJobId": "id", "agenticJobName": "name"})
        assert result == {"agenticJobId": "id", "agenticJobName": "name"}

    def test_does_not_mutate_original(self):
        existing = {"key": "val"}
        merge_extra_body(existing, {"new_key": "new_val"})
        assert "new_key" not in existing


class TestAgenticJobFieldResolution:
    """Per-field precedence: explicit metadata > contextvar > env var (BACK-777 Part 1)."""

    def setup_method(self):
        self._token = None

    def teardown_method(self):
        if self._token is not None:
            _agentic_job_context.reset(self._token)

    def test_explicit_metadata_wins_over_context_and_env(self):
        self._token = set_agentic_job_fields(job_id="ctx-job")
        with patch.dict(os.environ, {"REVENIUM_AGENTIC_JOB_ID": "env-job"}):
            result = extract_agentic_job_fields({"agentic_job_id": "meta-job"})
        assert result["agenticJobId"] == "meta-job"

    def test_camel_case_metadata_wins_too(self):
        self._token = set_agentic_job_fields(job_id="ctx-job")
        result = extract_agentic_job_fields({"agenticJobId": "meta-job"})
        assert result["agenticJobId"] == "meta-job"

    def test_contextvar_fallback_when_metadata_missing(self):
        self._token = set_agentic_job_fields(job_id="ctx-job", type="loan_processing")
        with patch.dict(os.environ, {"REVENIUM_AGENTIC_JOB_ID": "env-job"}):
            result = extract_agentic_job_fields({})
        assert result["agenticJobId"] == "ctx-job"
        assert result["agenticJobType"] == "loan_processing"

    def test_env_fallback_when_metadata_and_context_missing(self):
        env = {
            "REVENIUM_AGENTIC_JOB_ID": "env-job",
            "REVENIUM_AGENTIC_JOB_NAME": "Env Job",
            "REVENIUM_AGENTIC_JOB_TYPE": "env_type",
            "REVENIUM_AGENTIC_JOB_VERSION": "9.9.9",
        }
        with patch.dict(os.environ, env):
            result = extract_agentic_job_fields({})
        assert result == {
            "agenticJobId": "env-job",
            "agenticJobName": "Env Job",
            "agenticJobType": "env_type",
            "agenticJobVersion": "9.9.9",
        }

    def test_per_field_independence(self):
        # id from metadata, name from context, type from env — each field resolves alone
        self._token = set_agentic_job_fields(name="Ctx Name")
        with patch.dict(os.environ, {"REVENIUM_AGENTIC_JOB_TYPE": "env_type"}):
            result = extract_agentic_job_fields({"agentic_job_id": "meta-job"})
        assert result == {
            "agenticJobId": "meta-job",
            "agenticJobName": "Ctx Name",
            "agenticJobType": "env_type",
        }

    def test_empty_env_var_treated_as_unset(self):
        with patch.dict(os.environ, {"REVENIUM_AGENTIC_JOB_ID": ""}):
            assert extract_agentic_job_fields({}) == {}

    def test_empty_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert extract_agentic_job_fields({}) == {}

    def test_token_reset_restores_previous_context(self):
        outer = set_agentic_job_fields(job_id="outer")
        try:
            inner = set_agentic_job_fields(job_id="inner")
            assert get_agentic_job_fields() == {"agenticJobId": "inner"}
            _agentic_job_context.reset(inner)
            assert get_agentic_job_fields() == {"agenticJobId": "outer"}
        finally:
            _agentic_job_context.reset(outer)

    def test_set_agentic_job_fields_requires_a_field(self):
        import pytest
        with pytest.raises(ValueError):
            set_agentic_job_fields()


class TestExtractSkillFields:
    """Skill attribution resolver: usage_metadata snake > camel alias > env."""

    _ALL_SNAKE = {
        "skill_invocation_trigger": "manual",
        "skill_kind": "workflow",
        "skill_marketplace_name": "acme-marketplace",
        "skill_name": "summarize-docs",
        "skill_plugin_name": "docs-plugin",
        "skill_source": "marketplace",
    }

    def _extract(self, source):
        from revenium_middleware._core.fields import extract_skill_fields

        return extract_skill_fields(source)

    def test_snake_case_keys(self):
        assert self._extract(dict(self._ALL_SNAKE)) == self._ALL_SNAKE

    def test_camel_case_aliases(self):
        source = {
            "skillInvocationTrigger": "manual",
            "skillKind": "workflow",
            "skillMarketplaceName": "acme-marketplace",
            "skillName": "summarize-docs",
            "skillPluginName": "docs-plugin",
            "skillSource": "marketplace",
        }
        assert self._extract(source) == self._ALL_SNAKE

    def test_snake_takes_precedence_over_camel(self):
        source = {"skill_name": "snake-wins", "skillName": "camel-loses"}
        assert self._extract(source) == {"skill_name": "snake-wins"}

    def test_env_var_fallback(self):
        env = {
            "REVENIUM_SKILL_INVOCATION_TRIGGER": "manual",
            "REVENIUM_SKILL_KIND": "workflow",
            "REVENIUM_SKILL_MARKETPLACE_NAME": "acme-marketplace",
            "REVENIUM_SKILL_NAME": "summarize-docs",
            "REVENIUM_SKILL_PLUGIN_NAME": "docs-plugin",
            "REVENIUM_SKILL_SOURCE": "marketplace",
        }
        with patch.dict(os.environ, env, clear=False):
            assert self._extract({}) == self._ALL_SNAKE

    def test_metadata_beats_env(self):
        with patch.dict(os.environ, {"REVENIUM_SKILL_NAME": "env-name"}):
            assert self._extract({"skillName": "meta-name"}) == {"skill_name": "meta-name"}

    def test_fields_resolve_independently(self):
        with patch.dict(os.environ, {"REVENIUM_SKILL_SOURCE": "env-source"}):
            result = self._extract({"skill_name": "meta-name"})
        assert result == {"skill_name": "meta-name", "skill_source": "env-source"}

    def test_missing_fields_omitted(self):
        result = self._extract({"skill_name": "only-name"})
        assert result == {"skill_name": "only-name"}
        assert "skill_kind" not in result

    def test_empty_dict_when_no_skill_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            assert self._extract({}) == {}
