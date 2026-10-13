"""The error-text cleaner and the captured-prompt masker hide the same secret shapes."""

import time

import pytest

from revenium_middleware._core.log_sanitize import redact_error_text
from revenium_middleware._core.prompt_extraction import sanitize_credentials
from revenium_middleware._core.secret_shapes import SECRET_LABEL_ENDINGS
from secret_samples import CARD_GROUPS, PROMPT_REDACTED, SAMPLES


@pytest.mark.parametrize("sample", SAMPLES.values(), ids=SAMPLES.keys())
def test_error_cleaner_and_prompt_masker_hide_the_same_shapes(sample):
    assert redact_error_text(f"failed with {sample.value} today", 2048) == "failed with [REDACTED] today"
    assert sanitize_credentials(f"use {sample.value} today") == f"use {sample.prompt_masked} today"


def test_every_shared_shape_has_a_sample():
    from revenium_middleware._core.secret_shapes import SECRET_SHAPES

    sampled = {sample.shape for sample in SAMPLES.values()} - {"json_web_token"}
    assert sampled == {shape.name for shape in SECRET_SHAPES}


def test_both_maskers_use_the_one_shared_shape_list():
    from revenium_middleware._core import log_sanitize, prompt_extraction, secret_shapes

    assert log_sanitize.SECRET_SHAPES is secret_shapes.SECRET_SHAPES
    assert prompt_extraction.SECRET_SHAPES is secret_shapes.SECRET_SHAPES


CARD = "".join(CARD_GROUPS)
SPACED_CARD = " ".join(CARD_GROUPS)
CARD_IN_SENTENCES = {
    "plain_before_period": (f"Please use {CARD}.", "Please use {}."),
    "spaced_before_period": (f"{SPACED_CARD}.", "{}."),
    "plain_before_cvv": (f"card {CARD} 123 ok", "card {} 123 ok"),
    "plain_before_expiry": (f"card {CARD} 12/27 ok", "card {} 12/27 ok"),
    "spaced_before_expiry": (f"card {SPACED_CARD} 12/27 ok", "card {} 12/27 ok"),
    "spaced_before_cvv": (f"card {SPACED_CARD} 123 ok", "card {} 123 ok"),
}


@pytest.mark.parametrize("text, expected", CARD_IN_SENTENCES.values(), ids=CARD_IN_SENTENCES.keys())
def test_card_followed_by_a_period_expiry_or_cvv_is_hidden_by_both_maskers(text, expected):
    """BACK-4359: a sentence period, an expiry or a CVV after a card left the card readable."""
    assert redact_error_text(text, 2048) == expected.format("[REDACTED]")
    assert sanitize_credentials(text) == expected.format(PROMPT_REDACTED)


@pytest.mark.parametrize("text", [
    f"id 1.{CARD} kept",
    f"value {CARD}.5 kept",
    f"groups {SPACED_CARD} 1111 kept",
    f"groups 1111 {SPACED_CARD} kept",
])
def test_card_digits_inside_a_longer_number_are_kept_by_both_maskers(text):
    """BACK-4359: a decimal part or a further four-digit group still means a longer number, not a card."""
    assert redact_error_text(text, 2048) == text
    assert sanitize_credentials(text) == text


def test_url_password_with_an_empty_user_name_is_hidden_by_both_maskers():
    """BACK-4368: redis://:password@host kept its password."""
    secret = "FAKE" + "s3cretpass"
    text = f"connect redis://:{secret}@cache.internal:6379/0 failed"

    assert secret not in redact_error_text(text, 2048)
    assert sanitize_credentials(text) == f"connect redis://:{PROMPT_REDACTED}@cache.internal:6379/0 failed"


@pytest.mark.parametrize("control, escape", [("\n", "\\n"), ("\t", "\\t"), ("\r", "\\r")])
@pytest.mark.parametrize("sample", ["openai", "openai_project", "aws_access_key_id", "github_personal"])
def test_key_after_a_serialized_escape_is_hidden_by_both_maskers(control, escape, sample):
    """BACK-4359: in JSON-escaped or repr text a key on its own line follows a literal backslash escape."""
    value = SAMPLES[sample].value
    error_repr = repr(ValueError(f"bad config:{control}{value}"))

    assert escape + value in error_repr
    assert value not in redact_error_text(error_repr, 2048)
    assert sanitize_credentials(f"config:{escape}{value} done") == f"config:{escape}{SAMPLES[sample].prompt_masked} done"


@pytest.mark.parametrize("escape", ["\\n", "\\r", "\\t"])
def test_card_after_a_serialized_escape_is_hidden_by_both_maskers(escape):
    """BACK-4359: the letter of a literal backslash escape is not a card digit or letter run before the card."""
    assert CARD not in redact_error_text(f"config:{escape}{CARD} done", 2048)
    assert sanitize_credentials(f"config:{escape}{CARD} done") == f"config:{escape}{PROMPT_REDACTED} done"


BASIC_VALUE = "dXNlcjpwYXNzd29yZA=="


@pytest.mark.parametrize("escape", ["\\n", "\\r", "\\t"])
def test_basic_credentials_after_a_serialized_escape_are_hidden_by_both_maskers(escape):
    """BACK-4359: a Basic credential on its own line follows a literal backslash escape."""
    text = f"config:{escape}Basic {BASIC_VALUE} done"

    assert BASIC_VALUE not in redact_error_text(text, 2048)
    assert BASIC_VALUE not in sanitize_credentials(text)


@pytest.mark.parametrize("ending", SECRET_LABEL_ENDINGS)
def test_every_prompt_masker_label_ending_is_redacted_in_error_text(ending):
    """BACK-4359: the error cleaner draws its label names from the prompt masker's label endings."""
    secret = "FAKE" + "hunter2hunter2"
    for label in (ending, f"db_{ending}", ending.upper()):
        assert secret not in redact_error_text(f"rejected {label}={secret} today", 2048), label
        assert secret not in redact_error_text(f'rejected {{"{label}": "{secret}"}}', 2048), label


LONG_TEXT_LENGTH = 50000
LONG_TEXTS = {
    "pem_headers": ("-----BEGIN " + "PRIVATE KEY-----") * (LONG_TEXT_LENGTH // 27),
    "pem_body_without_end": "-----BEGIN " + "PRIVATE KEY-----\n" + "A" * LONG_TEXT_LENGTH,
    "pem_partial_headers": "-----BEGIN RSA " * (LONG_TEXT_LENGTH // 15),
    "escaped_line_starts": "\\nsk-" * (LONG_TEXT_LENGTH // 5),
    "card_groups_with_periods": "4111 1111 1111 1111." * (LONG_TEXT_LENGTH // 20),
    "url_starts_without_user": "://:" * (LONG_TEXT_LENGTH // 4),
}


@pytest.mark.parametrize("text", LONG_TEXTS.values(), ids=LONG_TEXTS.keys())
def test_shared_shapes_are_fast_on_long_text(text):
    from revenium_middleware._core.secret_shapes import SECRET_SHAPES, URL_USERINFO_PASSWORD, mask_secret_shapes

    started = time.perf_counter()
    mask_secret_shapes(text, SECRET_SHAPES, lambda kept: "")
    URL_USERINFO_PASSWORD.sub("", text)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, elapsed
