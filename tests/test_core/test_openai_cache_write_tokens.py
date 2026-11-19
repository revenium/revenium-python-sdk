"""Reading OpenAI's cache-write count, and taking it out of the prompt (BACK-3998)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai._models import BaseModel
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_usage import ResponseUsage

from revenium_middleware._core.cache_tokens import (
    extract_cache_creation_ttl_counts,
    extract_cache_tokens,
    openai_cache_write_tokens,
    without_openai_cache_writes,
)

WRITTEN = 4580


class DetailsWithoutTheField(BaseModel):
    """An older openai release's prompt details: no declared write field."""

    cached_tokens: int = 0


class UsageWithoutTheField(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: DetailsWithoutTheField


@pytest.mark.parametrize(
    "usage",
    [
        CompletionUsage(prompt_tokens=4583, completion_tokens=5, total_tokens=4588,
                        prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": WRITTEN}),
        ResponseUsage(input_tokens=4583, output_tokens=5, total_tokens=4588,
                      input_tokens_details={"cached_tokens": 0, "cache_write_tokens": WRITTEN},
                      output_tokens_details={"reasoning_tokens": 0}),
        {"prompt_tokens": 4583, "prompt_tokens_details": {"cache_write_tokens": WRITTEN}},
        {"input_tokens": 4583, "input_tokens_details": {"cache_write_tokens": WRITTEN}},
        SimpleNamespace(prompt_tokens=4583, prompt_tokens_details=SimpleNamespace(cache_write_tokens=WRITTEN)),
        UsageWithoutTheField.construct(
            prompt_tokens=4583, completion_tokens=5, total_tokens=4588,
            prompt_tokens_details=DetailsWithoutTheField.construct(cached_tokens=0, cache_write_tokens=WRITTEN),
        ),
    ],
    ids=["chat", "responses", "chat_dict", "responses_dict", "attribute_object", "older_openai_extra_key"],
)
def test_the_write_is_read_from_every_openai_shape(usage):
    assert openai_cache_write_tokens(usage) == WRITTEN


def test_an_older_openai_release_keeps_the_write_as_an_extra_key():
    usage = UsageWithoutTheField.model_validate({
        "prompt_tokens": 4583, "completion_tokens": 5, "total_tokens": 4588,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": WRITTEN},
    })

    assert openai_cache_write_tokens(usage) == WRITTEN


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens_details": None},
        {"prompt_tokens_details": {"cached_tokens": 3}},
        {"prompt_tokens_details": {"cache_write_tokens": None}},
        {"prompt_tokens_details": {"cache_write_tokens": True}},
        {"prompt_tokens_details": {"cache_write_tokens": "4580"}},
        {"prompt_tokens_details": {"cache_write_tokens": -5}},
        {"prompt_tokens_details": MagicMock()},
        MagicMock(),
    ],
    ids=["none", "empty", "null_details", "no_field", "null_field", "bool", "string", "negative",
         "mock_details", "mock_usage"],
)
def test_anything_that_is_not_a_positive_count_means_no_write(usage):
    assert openai_cache_write_tokens(usage) == 0


def test_an_anthropic_write_mirrored_by_litellm_is_not_an_openai_write():
    litellm_utils = pytest.importorskip("litellm.types.utils")
    usage = litellm_utils.Usage(prompt_tokens=100, completion_tokens=5, total_tokens=105,
                                cache_creation_input_tokens=40, cache_read_input_tokens=30)
    assert usage.prompt_tokens_details.cache_write_tokens == 40

    assert openai_cache_write_tokens(usage) == 0
    assert without_openai_cache_writes(100, usage) == 100
    assert extract_cache_tokens(usage).cache_creation_tokens == 40


def test_an_openai_write_through_litellm_is_read():
    litellm_utils = pytest.importorskip("litellm.types.utils")
    usage = litellm_utils.Usage(prompt_tokens=4583, completion_tokens=5, total_tokens=4588,
                                prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": WRITTEN})

    assert extract_cache_tokens(usage).cache_creation_tokens == WRITTEN
    assert without_openai_cache_writes(4583, usage) == 3


@pytest.mark.parametrize(
    "prompt, written, expected",
    [(4583, WRITTEN, 3), (100, 150, 0), (0, WRITTEN, 0), (None, WRITTEN, None), (11, 0, 11)],
    ids=["write_removed", "clamped_at_zero", "zero_prompt", "missing_prompt", "no_write"],
)
def test_the_write_leaves_the_prompt_never_below_zero(prompt, written, expected):
    usage = {"prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": written}}

    assert without_openai_cache_writes(prompt, usage) == expected


def test_cache_reads_stay_in_the_prompt():
    usage = {"prompt_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 1500}}

    assert without_openai_cache_writes(6000, usage) == 4500


TWELVE_HOUR_SHARE = 4000


def _responses_usage_with_write_details(write_details):
    return ResponseUsage(
        input_tokens=5000, output_tokens=5, total_tokens=5005,
        input_tokens_details={"cached_tokens": 0, "cache_write_tokens": WRITTEN,
                              "cache_write_tokens_details": write_details},
        output_tokens_details={"reasoning_tokens": 0},
    )


@pytest.mark.parametrize(
    "usage",
    [
        _responses_usage_with_write_details({"ttl_12h_tokens": TWELVE_HOUR_SHARE, "ttl_30m_tokens": 580}),
        {"input_tokens_details": {"cache_write_tokens": WRITTEN, "cache_write_tokens_details": {
            "ttl_12h_tokens": TWELVE_HOUR_SHARE, "ttl_30m_tokens": 580}}},
        SimpleNamespace(input_tokens_details=SimpleNamespace(
            cache_write_tokens=WRITTEN,
            cache_write_tokens_details=SimpleNamespace(ttl_12h_tokens=TWELVE_HOUR_SHARE, ttl_30m_tokens=580))),
    ],
    ids=["responses", "responses_dict", "attribute_object"],
)
def test_the_12h_share_of_a_responses_write_is_its_own_bucket(usage):
    assert extract_cache_creation_ttl_counts(usage) == {"cache_creation12h_token_count": TWELVE_HOUR_SHARE}
    assert extract_cache_tokens(usage).cache_creation_tokens == WRITTEN


def test_a_reported_12h_share_of_zero_is_forwarded():
    usage = _responses_usage_with_write_details({"ttl_12h_tokens": 0, "ttl_30m_tokens": WRITTEN})

    assert extract_cache_creation_ttl_counts(usage) == {"cache_creation12h_token_count": 0}


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens_details": {"cache_write_tokens": WRITTEN}},
        {"input_tokens_details": {"cache_write_tokens": WRITTEN, "cache_write_tokens_details": None}},
        {"input_tokens_details": {"cache_write_tokens": WRITTEN,
                                  "cache_write_tokens_details": {"ttl_30m_tokens": WRITTEN}}},
        {"input_tokens_details": {"cache_write_tokens": WRITTEN,
                                  "cache_write_tokens_details": {"ttl_12h_tokens": None}}},
        {"input_tokens_details": {"cache_write_tokens": WRITTEN,
                                  "cache_write_tokens_details": {"ttl_12h_tokens": True}}},
        {"input_tokens_details": {"cache_write_tokens": WRITTEN,
                                  "cache_write_tokens_details": {"ttl_12h_tokens": "4000"}}},
        {"input_tokens_details": {"cache_write_tokens": WRITTEN, "cache_write_tokens_details": MagicMock()}},
        MagicMock(),
    ],
    ids=["no_details", "null_details", "no_12h_field", "null_12h", "bool", "string", "mock_details", "mock_usage"],
)
def test_no_12h_bucket_without_a_reported_12h_count(usage):
    assert "cache_creation12h_token_count" not in extract_cache_creation_ttl_counts(usage)


def _write_with_12h_share(written, twelve_hour):
    return {"input_tokens_details": {"cache_write_tokens": written,
                                     "cache_write_tokens_details": {"ttl_12h_tokens": twelve_hour}}}


def test_a_write_made_wholly_at_12h_is_all_one_bucket():
    usage = _write_with_12h_share(WRITTEN, WRITTEN)

    assert extract_cache_creation_ttl_counts(usage) == {"cache_creation12h_token_count": WRITTEN}


@pytest.mark.parametrize(
    "usage, aggregate",
    [
        (_write_with_12h_share(WRITTEN, WRITTEN + 1), WRITTEN),
        (_responses_usage_with_write_details({"ttl_12h_tokens": WRITTEN + 1, "ttl_30m_tokens": 0}), WRITTEN),
        (_write_with_12h_share(WRITTEN, -1), WRITTEN),
        (_write_with_12h_share(None, TWELVE_HOUR_SHARE), 0),
        (_write_with_12h_share(0, 0), 0),
        (_write_with_12h_share("4580", TWELVE_HOUR_SHARE), 0),
        ({"input_tokens_details": {"cache_write_tokens_details": {"ttl_12h_tokens": TWELVE_HOUR_SHARE}}}, 0),
    ],
    ids=["larger_than_the_write", "typed_larger_than_the_write", "negative", "null_write", "no_write",
         "string_write", "write_absent"],
)
def test_a_12h_share_that_does_not_fit_inside_the_write_is_dropped_and_the_write_kept(usage, aggregate):
    assert extract_cache_creation_ttl_counts(usage) == {}
    assert extract_cache_tokens(usage).cache_creation_tokens == aggregate


def test_a_12h_share_beside_an_anthropic_shaped_write_is_not_read():
    usage = {
        "cache_creation_input_tokens": 40,
        "input_tokens_details": {"cache_write_tokens": 40, "cache_write_tokens_details": {"ttl_12h_tokens": 40}},
    }

    assert extract_cache_creation_ttl_counts(usage) == {}


def test_the_typed_openai_12h_share_is_read_where_the_installed_openai_declares_it():
    response_usage = pytest.importorskip("openai.types.responses.response_usage")
    details_type = getattr(response_usage, "InputTokensDetailsCacheWriteTokensDetails", None)
    if details_type is None:
        pytest.skip("the installed openai release does not declare the cache-write TTL split")
    usage = _responses_usage_with_write_details(details_type(ttl_12h_tokens=TWELVE_HOUR_SHARE, ttl_30m_tokens=580))

    assert extract_cache_creation_ttl_counts(usage) == {"cache_creation12h_token_count": TWELVE_HOUR_SHARE}
