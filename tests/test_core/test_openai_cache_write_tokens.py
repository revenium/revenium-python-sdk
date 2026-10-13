"""Reading OpenAI's cache-write count, and taking it out of the prompt (BACK-3998)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai._models import BaseModel
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_usage import ResponseUsage

from revenium_middleware._core.cache_tokens import (
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
