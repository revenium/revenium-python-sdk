import logging

import pytest

from revenium_middleware._core.fields import (
    extract_agentic_job_fields,
    extract_common_metadata,
    extract_field_with_fallback,
    extract_org_and_product,
    extract_with_aliases,
    merge_extra_body,
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
