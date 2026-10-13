import base64
import contextvars
import datetime
import json
import logging
import math
import threading
import time
import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pydantic
import pytest

from revenium_middleware import idempotency_key
from revenium_middleware._core import metering_buffer
from revenium_middleware._core.metering_buffer import MeteringBuffer, delivery_deferred_to_buffer
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._core.prompt_extraction import sanitize_credentials
from revenium_middleware._core.wire_guards import stored_subscriber_value
from revenium_middleware._metering import ReveniumMetering


@pytest.fixture
def mock_client():
    """Patch the `client` import that submit_ai_event reads."""
    mock = MagicMock()
    mock.ai.create_completion = MagicMock(return_value="completion-result")
    mock.ai.create_image = MagicMock(return_value="image-result")
    mock.ai.create_video = MagicMock(return_value="video-result")
    mock.ai.create_audio = MagicMock(return_value="audio-result")
    with patch("revenium_middleware._core.metering_submission.current_client", lambda: mock):
        yield mock


def _extract_key(mock_method):
    """Pull the Idempotency-Key from the most recent call's kwargs."""
    kwargs = mock_method.call_args.kwargs
    return kwargs["extra_headers"]["Idempotency-Key"]


def test_submit_completion_auto_generates_uuid_v4(mock_client):
    submit_ai_event("completion", {"some_field": "value"})
    key = _extract_key(mock_client.ai.create_completion)
    parsed = uuid.UUID(key)
    assert parsed.version == 4


def test_submit_completion_uses_explicit_override(mock_client):
    submit_ai_event("completion", {"x": 1}, idempotency_key="my-explicit-key")
    assert _extract_key(mock_client.ai.create_completion) == "my-explicit-key"


def test_submit_completion_uses_contextvar_when_no_explicit(mock_client):
    with idempotency_key("ctx-key"):
        submit_ai_event("completion", {"x": 1})
    assert _extract_key(mock_client.ai.create_completion) == "ctx-key"


def test_explicit_override_wins_over_contextvar(mock_client):
    with idempotency_key("ctx-key"):
        submit_ai_event("completion", {"x": 1}, idempotency_key="explicit")
    assert _extract_key(mock_client.ai.create_completion) == "explicit"


def test_each_call_generates_distinct_uuid(mock_client):
    keys = set()
    for _ in range(3):
        submit_ai_event("completion", {"x": 1})
        keys.add(_extract_key(mock_client.ai.create_completion))
    assert len(keys) == 3


def test_image_operation_routes_to_create_image(mock_client):
    submit_ai_event("image", {"x": 1})
    mock_client.ai.create_image.assert_called_once()
    mock_client.ai.create_completion.assert_not_called()


def test_video_operation_routes_to_create_video(mock_client):
    submit_ai_event("video", {"x": 1})
    mock_client.ai.create_video.assert_called_once()
    mock_client.ai.create_completion.assert_not_called()


def test_audio_operation_routes_to_create_audio(mock_client):
    submit_ai_event("audio", {"x": 1})
    mock_client.ai.create_audio.assert_called_once()
    mock_client.ai.create_completion.assert_not_called()


def test_unknown_operation_raises_value_error(mock_client):
    with pytest.raises(ValueError, match="Unknown AI metering operation"):
        submit_ai_event("invalid", {})


def test_returns_none_when_client_unconfigured():
    with patch("revenium_middleware._core.metering_submission.current_client", lambda: None):
        result = submit_ai_event("completion", {"x": 1})
    assert result is None


def test_preserves_existing_extra_headers(mock_client):
    submit_ai_event(
        "completion",
        {"extra_headers": {"X-Custom": "v"}, "x": 1},
    )
    call_kwargs = mock_client.ai.create_completion.call_args.kwargs
    assert call_kwargs["extra_headers"]["X-Custom"] == "v"
    assert "Idempotency-Key" in call_kwargs["extra_headers"]


def test_preserves_existing_extra_body(mock_client):
    submit_ai_event(
        "completion",
        {"extra_body": {"agentic": "fields"}, "x": 1},
    )
    call_kwargs = mock_client.ai.create_completion.call_args.kwargs
    assert call_kwargs["extra_body"] == {"agentic": "fields"}


def test_args_dict_not_mutated(mock_client):
    args = {"extra_headers": {"X": "Y"}, "x": 1}
    snapshot = {**args, "extra_headers": dict(args["extra_headers"])}
    submit_ai_event("completion", args)
    assert args == snapshot


def test_empty_string_explicit_key_raises_value_error(mock_client):
    """Empty string explicit override is a likely caller bug; raise ValueError instead of silently breaking metering."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        submit_ai_event("completion", {}, idempotency_key="")
    mock_client.ai.create_completion.assert_not_called()


def _is_uuid4(value):
    try:
        return uuid.UUID(value).version == 4
    except ValueError:
        return False


def test_an_unsendable_key_set_directly_on_the_context_is_replaced_and_the_record_sent(mock_client, caplog):
    from revenium_middleware._core.context import _idempotency_key_context

    token = _idempotency_key_context.set("order-1006\n")
    try:
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            submit_ai_event("completion", {"x": 1})
    finally:
        _idempotency_key_context.reset(token)

    mock_client.ai.create_completion.assert_called_once()
    assert _is_uuid4(_extract_key(mock_client.ai.create_completion))
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "\n" not in warnings[0]
    assert "order-1006" not in warnings[0]


def test_extra_headers_explicitly_none_in_args(mock_client):
    """args containing extra_headers=None must be treated as no headers (not raise)."""
    submit_ai_event("completion", {"extra_headers": None, "x": 1})
    call_kwargs = mock_client.ai.create_completion.call_args.kwargs
    assert "Idempotency-Key" in call_kwargs["extra_headers"]


def test_raises_if_idempotency_key_passed_in_extra_headers(mock_client):
    """Idempotency-Key in extra_headers is a footgun — the wrapper owns this header. Fail fast."""
    with pytest.raises(ValueError, match="Pass Idempotency-Key via the idempotency_key parameter"):
        submit_ai_event(
            "completion",
            {"extra_headers": {"Idempotency-Key": "would-be-silently-overwritten"}},
        )
    mock_client.ai.create_completion.assert_not_called()


@pytest.mark.parametrize("header_name", ["idempotency-key", "IDEMPOTENCY-KEY", "Idempotency-key", "idempotency-Key"])
def test_idempotency_key_header_guard_is_case_insensitive(mock_client, header_name):
    """HTTP libraries often normalize header names; the guard must reject any casing variant
    of Idempotency-Key in extra_headers, not just the canonical one."""
    with pytest.raises(ValueError, match="Pass Idempotency-Key via the idempotency_key parameter"):
        submit_ai_event(
            "completion",
            {"extra_headers": {header_name: "would-be-silently-shadowed"}},
        )
    mock_client.ai.create_completion.assert_not_called()


def _sent_keys(mock_method):
    return [call.kwargs["extra_headers"]["Idempotency-Key"] for call in mock_method.call_args_list]


def _run_block(mock_client, key, calls):
    """Meter ``calls`` model calls inside one idempotency block and return the keys sent."""
    sent_before = mock_client.ai.create_completion.call_count
    with idempotency_key(key):
        for _ in range(calls):
            submit_ai_event("completion", {"x": 1})
    return _sent_keys(mock_client.ai.create_completion)[sent_before:]


def test_each_call_in_an_idempotency_block_gets_its_own_key(mock_client):
    assert _run_block(mock_client, "order-42", 3) == ["order-42", "order-42:2", "order-42:3"]


def test_rerunning_a_block_reuses_the_same_keys_in_order(mock_client):
    first_run = _run_block(mock_client, "order-42", 2)
    second_run = _run_block(mock_client, "order-42", 2)

    assert first_run == second_run
    assert len(set(first_run)) == 2


def test_calls_from_copied_contexts_in_one_block_get_distinct_keys(mock_client):
    with idempotency_key("order-42"):
        workers = [
            threading.Thread(
                target=contextvars.copy_context().run,
                args=(submit_ai_event, "completion", {"x": 1}),
            )
            for _ in range(8)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

    keys = _sent_keys(mock_client.ai.create_completion)
    assert len(keys) == 8
    assert len(set(keys)) == 8


def test_explicit_key_is_sent_verbatim_inside_a_block(mock_client):
    with idempotency_key("order-42"):
        submit_ai_event("completion", {"x": 1})
        submit_ai_event("completion", {"x": 1}, idempotency_key="caller-owned")
        submit_ai_event("completion", {"x": 1}, idempotency_key="caller-owned")

    assert _sent_keys(mock_client.ai.create_completion)[1:] == ["caller-owned", "caller-owned"]


def test_derived_key_stays_within_255_characters(mock_client):
    longest_key = "k" * 255

    first_run = _run_block(mock_client, longest_key, 2)
    second_run = _run_block(mock_client, longest_key, 2)

    assert first_run[0] == longest_key
    assert len(first_run[1]) <= 255
    assert first_run[1] != first_run[0]
    assert first_run == second_run


def test_derived_key_stays_within_255_characters_at_the_tenth_call(mock_client):
    # ":2" to ":9" still fit after a 253-character key; ":10" does not.
    block_key = "k" * 253

    first_run = _run_block(mock_client, block_key, 10)
    second_run = _run_block(mock_client, block_key, 10)

    assert first_run[1] == block_key + ":2"
    assert all(len(key) <= 255 for key in first_run)
    assert len(set(first_run)) == 10
    assert first_run == second_run


def test_block_key_set_to_none_falls_back_to_a_fresh_key_per_call(mock_client):
    from revenium_middleware._core.context import _idempotency_key_context, set_idempotency_key

    token = set_idempotency_key(None)
    try:
        submit_ai_event("completion", {"x": 1})
        submit_ai_event("completion", {"x": 1})
    finally:
        _idempotency_key_context.reset(token)
    fresh_keys = _sent_keys(mock_client.ai.create_completion)
    assert len(set(fresh_keys)) == 2
    assert all(uuid.UUID(key).version == 4 for key in fresh_keys)

    token = _idempotency_key_context.set("plain")
    try:
        submit_ai_event("completion", {"x": 1})
    finally:
        _idempotency_key_context.reset(token)
    assert _sent_keys(mock_client.ai.create_completion)[-1] == "plain"


# BACK-4237: captured prompt text is sent with credential shapes masked.
# Every secret below is fake; the "FAKE" marker must never reach the backend.
FAKE_BEARER = "FAKEBEARERTOKENFAKEBEARERTOKEN0002"
FAKE_PROJECT_KEY = "sk-proj-FAKE" + "0" * 48
FAKE_ANTHROPIC_KEY = "sk-ant-FAKE" + "3" * 24
FAKE_TOOL_KEY = "FAKEtoolkey" + "1" * 20
FAKE_ACCESS_TOKEN = "FAKEaccesstoken" + "2" * 20
FAKE_ARGUMENT_KEY = "FAKEargkey" + "4" * 20


def _prompt_args_with_secrets():
    messages = [
        {"role": "user", "content": f"My OpenAI key is {FAKE_PROJECT_KEY} and password: FAKEpassw0rd!"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "deploy",
             "input": {"api_key": FAKE_TOOL_KEY, "access_token": FAKE_ACCESS_TOKEN}},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "deploy", "arguments": json.dumps({"api_key": FAKE_ARGUMENT_KEY})}},
        ]},
    ]
    return {
        "model": "gpt-4o-mini",
        "system_prompt": f"Call the tools with Authorization: Bearer {FAKE_BEARER}",
        "input_messages": json.dumps(messages, ensure_ascii=False),
        "output_response": json.dumps([{"type": "text", "text": f"Noted, your other key is {FAKE_ANTHROPIC_KEY}"}]),
    }


PROMPT_FIELDS = ("system_prompt", "input_messages", "output_response")


def _sent(mock_client):
    return mock_client.ai.create_completion.call_args.kwargs


def test_prompt_fields_are_sent_with_credentials_masked(mock_client):
    args = _prompt_args_with_secrets()
    snapshot = dict(args)

    submit_ai_event("completion", args)

    sent = _sent(mock_client)
    for field in PROMPT_FIELDS:
        assert "FAKE" not in sent[field], (field, sent[field])
        assert "***REDACTED***" in sent[field], (field, sent[field])
    assert json.loads(sent["input_messages"])
    assert json.loads(sent["output_response"])
    assert sent["model"] == "gpt-4o-mini"
    assert args == snapshot


def test_prompt_fields_without_credentials_are_sent_unchanged(mock_client):
    args = {
        "system_prompt": "You are the bearer of bad news. Explain the password requirements.",
        "input_messages": json.dumps([{"role": "user", "content": "What is a token? Keep the secret recipe."}]),
        "output_response": "A token is a unit of text; sk-short stays as written.",
    }

    submit_ai_event("completion", args)

    sent = _sent(mock_client)
    assert {field: sent[field] for field in PROMPT_FIELDS} == args


def test_masker_failure_still_sends_record_without_prompt_fields(mock_client, monkeypatch):
    import revenium_middleware._core.metering_submission as submission

    def broken(text):
        raise RuntimeError("masker bug")

    monkeypatch.setattr(submission, "sanitize_credentials", broken, raising=False)
    args = _prompt_args_with_secrets()

    submit_ai_event("completion", args)

    mock_client.ai.create_completion.assert_called_once()
    sent = _sent(mock_client)
    assert [field for field in PROMPT_FIELDS if field in sent] == []
    assert sent["model"] == "gpt-4o-mini"


def test_jwt_pattern_is_fast_on_long_input(mock_client):
    args = {"system_prompt": "eyJ" * 16667, "input_messages": None, "output_response": None}

    started = time.perf_counter()
    submit_ai_event("completion", args)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, elapsed
    assert _sent(mock_client)["system_prompt"] == args["system_prompt"]


# BACK-4313: a JSON Web Token anywhere in captured prompt text is stored as ***REDACTED_JWT***.
def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


JWT_SEGMENTS = (_b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()),
                _b64url(json.dumps({"sub": "user-42", "exp": 1999999999}).encode()),
                _b64url(b"fixed test signature, not a secret"))
FAKE_JWT = ".".join(JWT_SEGMENTS)


def _masked_sentence(sent):
    assert sent == "my session token is ***REDACTED_JWT*** please check it"


def _masked_message_list(sent):
    assert json.loads(sent)[0]["content"] == "here: ***REDACTED_JWT***"


def _masked_content_block(sent):
    assert json.loads(sent)[0]["content"][0]["text"] == "saved ***REDACTED_JWT*** earlier"


def _masked_cookie(sent):
    assert sent == "Cookie: session=***REDACTED_JWT***; theme=dark"


JWT_PROMPTS = {
    "sentence": ("system_prompt", f"my session token is {FAKE_JWT} please check it", _masked_sentence),
    "json_message_list": ("input_messages", json.dumps([{"role": "user", "content": f"here: {FAKE_JWT}"}]),
                          _masked_message_list),
    "nested_content_block": ("input_messages", json.dumps(
        [{"role": "user", "content": [{"type": "text", "text": f"saved {FAKE_JWT} earlier"}]}]),
        _masked_content_block),
    "cookie_header": ("output_response", f"Cookie: session={FAKE_JWT}; theme=dark", _masked_cookie),
}


@pytest.mark.parametrize("field, text, assert_masked", JWT_PROMPTS.values(), ids=JWT_PROMPTS.keys())
def test_a_json_web_token_in_captured_prompt_text_is_masked(mock_client, field, text, assert_masked):
    submit_ai_event("completion", {"model": "gpt-4o-mini", field: text})

    sent = _sent(mock_client)[field]
    assert "***REDACTED_JWT***" in sent
    assert not any(segment in sent for segment in JWT_SEGMENTS), sent
    assert_masked(sent)


def _tool_use_messages(tool_input):
    return json.dumps([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "tu_1", "name": "login", "input": tool_input},
    ]}], ensure_ascii=False)


def test_password_with_escaped_quote_is_masked_whole_and_json_stays_valid(mock_client):
    password = 'abc"FAKEdefghijkl'
    args = {"input_messages": _tool_use_messages({"password": password})}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert "FAKE" not in sent, sent
    assert "defghijkl" not in sent, sent
    tool_input = json.loads(sent)[0]["content"][0]["input"]
    assert tool_input == {"password": "***REDACTED***"}


@pytest.mark.parametrize("tool_input", [
    {"require_password": None},
    {"is_secret": True},
])
def test_non_string_values_after_secret_like_keys_keep_json_valid(mock_client, tool_input):
    args = {"input_messages": _tool_use_messages(tool_input)}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert json.loads(sent)[0]["content"][0]["input"] == tool_input


def test_unquoted_value_after_password_key_does_not_swallow_json_structure(mock_client):
    args = {"system_prompt": 'Config was {"password": 12345678, "other": 1}'}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["system_prompt"]
    assert ", \"other\": 1}" in sent, sent


@pytest.mark.parametrize("tool_input, expected", [
    ({"password": 12345678, "other": 1}, {"password": "***REDACTED***", "other": 1}),
    ({"api_key": 12345678901234567890}, {"api_key": "***REDACTED***"}),
    ({"password": ["FAKEhunter2hunter2"]}, {"password": ["***REDACTED***"]}),
])
def test_non_string_secrets_in_json_are_masked_and_json_stays_valid(mock_client, tool_input, expected):
    args = {"input_messages": _tool_use_messages(tool_input)}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert json.loads(sent)[0]["content"][0]["input"] == expected


@pytest.mark.parametrize("text", [
    "password=FAKE1,xyzwvu99",
    "secret: FAKE@ss{word}123",
    "password: FAKE]defghijk",
])
def test_text_password_containing_json_punctuation_is_masked_whole(mock_client, text):
    args = {"system_prompt": f"Log in with {text} please"}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["system_prompt"]
    assert "FAKE" not in sent, sent
    assert sent.endswith(" please"), sent


def test_credentials_in_json_object_keys_are_masked_and_json_stays_valid(mock_client):
    key = "sk-FAKEabcdefghijklmnopqrst"
    args = {"output_response": json.dumps({key: "active", "note": "ok"})}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["output_response"]
    assert "FAKE" not in sent, sent
    assert json.loads(sent) == {"sk-***REDACTED***": "active", "note": "ok"}


@pytest.mark.parametrize("text", [
    '{"password": "FAKE horse battery staple"}',
    '{"password":"FAKE words make a secret"}',
])
def test_json_password_containing_spaces_is_masked_whole(text):
    assert json.loads(sanitize_credentials(text)) == {"password": "***REDACTED***"}


@pytest.mark.parametrize("text", [
    '{"secretary": "Margaret Thompson"}',
    '{"password_hint": "FAKEhint-firstpet"}',
    '{"tokenizer_name": "FAKEtokenizer0123456789abcdef"}',
    '{"password_policy_url": "https://example.com/FAKEpolicy"}',
])
def test_json_keys_only_containing_a_secret_word_are_not_masked(text):
    assert sanitize_credentials(text) == text


@pytest.mark.parametrize("key", ["client_secret", "db_password"])
def test_json_keys_ending_with_a_secret_word_are_masked(key):
    assert json.loads(sanitize_credentials(json.dumps({key: "FAKEabcdefgh12"}))) == {key: "***REDACTED***"}


@pytest.mark.parametrize("key, masked", [("db_password", True), ("password_hint", False)])
def test_json_and_text_forms_agree_on_which_keys_are_secret(key, masked):
    value = "FAKEabcdefgh12"
    from_json = json.loads(sanitize_credentials(json.dumps({key: value})))[key]
    from_text = sanitize_credentials(f"{key}: {value}")

    assert (from_json == "***REDACTED***") is masked
    assert (from_text == f"{key}: ***REDACTED***") is masked


FAKE_KEY_LABEL_VALUE = "FAKE" + "abcdefgh12" * 2


@pytest.mark.parametrize("key", [
    "author_key", "capital_key", "therapist_key", "accessibility_key", "client_cache_key",
])
def test_key_labels_whose_word_before_key_is_not_a_qualifier_are_not_masked(key):
    """BACK-4368: a qualifier word buried inside another word (author, capital) no longer makes a key secret."""
    as_json = json.dumps({key: FAKE_KEY_LABEL_VALUE})
    as_text = f"{key}: {FAKE_KEY_LABEL_VALUE}"

    assert sanitize_credentials(as_json) == as_json
    assert sanitize_credentials(as_text) == as_text


@pytest.mark.parametrize("key", [
    "AWS_SECRET_ACCESS_KEY", "SecretAccessKey", "awsSecretAccessKey", "AWSSECRETACCESSKEY",
    "client_key", "api_key", "apiKey", "x-api-key", "accesskey", "private_key", "oauth_key",
])
def test_key_labels_whose_word_before_key_is_a_qualifier_are_masked_in_both_forms(key):
    """BACK-4368: credential key labels stay masked in JSON and in text."""
    from_json = json.loads(sanitize_credentials(json.dumps({key: FAKE_KEY_LABEL_VALUE})))[key]
    from_text = sanitize_credentials(f"{key}: {FAKE_KEY_LABEL_VALUE}")

    assert from_json == "***REDACTED***"
    assert from_text == f"{key}: ***REDACTED***"


def test_truncated_json_password_with_escaped_quote_is_masked_whole(mock_client):
    messages = _tool_use_messages({"password": 'abc"FAKEdefghijkl', "note": "x" * 40})
    args = {"input_messages": messages[:-20] + "...[TRUNCATED]"}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert "FAKE" not in sent, sent
    assert "defghijkl" not in sent, sent


@pytest.mark.parametrize("key", [
    "sk-" + "FAKE" + "5" * 20,
    "sk-proj-" + "FAKE" + "0" * 48,
    "AKIA" + "FAKE" * 4,
    "ghp_" + "FAKE" + "g1" * 18,
])
def test_truncated_input_messages_mask_a_key_on_its_own_line(mock_client, key):
    """BACK-4359: past the cap the JSON cannot be parsed, and a key after a literal \\n escape still leaked."""
    messages = json.dumps([{"role": "user", "content": "config:\n" + key + "\nnote " + "x" * 40}])
    args = {"input_messages": messages[:-20] + "...[TRUNCATED]"}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert "FAKE" not in sent, sent
    assert "\\n" + key[:3] in sent, sent


@pytest.mark.parametrize("secret", ["4111111111111111", "Basic dXNlcjpwYXNzd29yZA=="])
def test_truncated_input_messages_mask_a_card_or_basic_credential_on_its_own_line(mock_client, secret):
    """BACK-4359: past the cap a card or Basic credential after a literal \\n escape still leaked."""
    messages = json.dumps([{"role": "user", "content": "config:\n" + secret + "\nnote " + "x" * 40}])
    args = {"input_messages": messages[:-20] + "...[TRUNCATED]"}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert "4111111111111111" not in sent, sent
    assert "dXNlcjpwYXNzd29yZA==" not in sent, sent


@pytest.mark.parametrize("label", ["apiKey", "api_key", "password"])
def test_truncated_input_messages_mask_a_labelled_value_on_its_own_line(mock_client, label):
    """BACK-4368: the letter of a literal \\n escape is not read as the start of the label."""
    messages = json.dumps([{"role": "user", "content": f"config:\n{label}: {FAKE_KEY_LABEL_VALUE}\nnote " + "x" * 40}])
    args = {"input_messages": messages[:-20] + "...[TRUNCATED]"}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["input_messages"]
    assert "FAKE" not in sent, sent


def test_escaped_json_password_in_text_keeps_its_closing_quote(mock_client):
    args = {"system_prompt": 'Config {\\"password\\": \\"FAKEhunter22\\", \\"x\\": 1}'}

    submit_ai_event("completion", args)

    sent = _sent(mock_client)["system_prompt"]
    assert sent == 'Config {\\"password\\": \\"***REDACTED***\\", \\"x\\": 1}', sent


def test_text_password_scan_is_fast_on_a_long_run_of_backslashes():
    text = "password: " + "\\" * 100001

    started = time.perf_counter()
    sanitize_credentials(text)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, elapsed


# A record holding a value JSON cannot write (BACK-4373). The client above is a
# MagicMock that never writes JSON, so these tests send through the real
# metering client onto an httpx MockTransport, which writes the body exactly as
# a live send would.

SAMPLE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")


class JsonWritingMeter:
    """The real metering client on a MockTransport that records every body it is sent.

    ``statuses`` scripts the answers, one per request; once it runs out every
    request is answered 201.
    """

    def __init__(self):
        self.bodies = []
        self.statuses = []
        self.client = ReveniumMetering(
            api_key="hak_test_dummy_key_for_ci_testing_only",
            base_url="http://revenium.test/meter/",
            http_client=httpx.Client(transport=httpx.MockTransport(self._answer)),
            max_retries=0,
        )

    def _answer(self, request):
        status = self.statuses.pop(0) if self.statuses else 201
        if status == 201:
            self.bodies.append(json.loads(request.content))
            return httpx.Response(201, json={"id": "evt-1", "resourceType": "metered-event"})
        return httpx.Response(status, json={"error": "unavailable"})


@pytest.fixture
def isolated_buffer(monkeypatch):
    buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    return buf


@pytest.fixture
def json_writing_client(isolated_buffer):
    meter = JsonWritingMeter()
    with patch("revenium_middleware._core.metering_submission.current_client", lambda: meter.client):
        yield meter
    meter.client.close()


@pytest.fixture(autouse=True)
def forget_json_safe_warnings():
    try:
        from revenium_middleware._core import json_safe
    except ImportError:
        yield
        return
    json_safe._warn_once.cache_clear()
    json_safe._warn_once_sent_as_given.cache_clear()
    yield
    json_safe._warn_once.cache_clear()
    json_safe._warn_once_sent_as_given.cache_clear()


def _completion_args(**extra):
    args = {
        "completion_start_time": "2026-10-11T00:00:00Z",
        "cost_type": "AI",
        "input_token_count": 5,
        "is_streamed": False,
        "model": "gpt-4o-mini",
        "output_token_count": 3,
        "provider": "OPENAI",
        "request_duration": 100,
        "request_time": "2026-10-11T00:00:00Z",
        "response_time": "2026-10-11T00:00:01Z",
        "stop_reason": "END",
        "total_token_count": 8,
        "transaction_id": "txn-json-safe",
    }
    args.update(extra)
    return args


def _self_referencing_dict():
    looped = {}
    looped["a"] = looped
    looped["b"] = looped
    return looped


def _is_negative_zero(value):
    return value == 0 and math.copysign(1.0, value) < 0


UNWRITABLE_VALUE_CASES = [
    pytest.param({"subscriber": {"id": SAMPLE_UUID}},
                 lambda body: body["subscriber"]["id"] == str(SAMPLE_UUID), id="uuid-subscriber-id"),
    pytest.param({"organization_name": SAMPLE_UUID},
                 lambda body: body["organizationName"] == str(SAMPLE_UUID), id="uuid-organization-name"),
    pytest.param({"response_quality_score": Decimal("0.9")},
                 lambda body: body["responseQualityScore"] == 0.9 and isinstance(body["responseQualityScore"], float),
                 id="decimal-quality-score"),
    pytest.param({"response_quality_score": Decimal("sNaN")},
                 lambda body: "responseQualityScore" not in body, id="decimal-snan-quality-score"),
    pytest.param({"response_quality_score": float("nan")},
                 lambda body: "responseQualityScore" not in body, id="nan-quality-score"),
    pytest.param({"response_quality_score": float("inf")},
                 lambda body: "responseQualityScore" not in body, id="infinite-quality-score"),
    pytest.param({"trace_id": b"abc"}, lambda body: body["traceId"] == "abc", id="bytes-trace-id"),
    pytest.param({"trace_id": datetime.datetime(2026, 10, 11, 12, 30, tzinfo=datetime.timezone.utc)},
                 lambda body: body["traceId"] == "2026-10-11T12:30:00+00:00", id="datetime-trace-id"),
    pytest.param({"subscriber": {"id": datetime.date(2026, 10, 11)}},
                 lambda body: body["subscriber"]["id"] == "2026-10-11", id="date-in-subscriber"),
    pytest.param({"trace_id": "Jos\ud83d"}, lambda body: body["traceId"] == "Jos\ufffd", id="half-emoji-trace-id"),
    pytest.param({"subscriber": {"id": "user\ud83d"}},
                 lambda body: body["subscriber"]["id"] == "user\ufffd", id="half-emoji-subscriber-id"),
    pytest.param({"extra_body": {"tags": _self_referencing_dict()}},
                 lambda body: body["tags"] == {}, id="self-reference-in-extra-body"),
    pytest.param({"extra_body": {"tags": {"letters": {"b", "a"}, "pair": ("x", 1)}}},
                 lambda body: sorted(body["tags"]["letters"]) == ["a", "b"] and body["tags"]["pair"] == ["x", 1],
                 id="set-and-tuple-in-extra-body"),
    pytest.param({"extra_body": {"tags": {1: "a"}}},
                 lambda body: body["tags"] == {"1": "a"}, id="non-text-dict-key"),
    pytest.param({"extra_body": {"tags": {"zero": -0.0, "big": 10 ** 30}}},
                 lambda body: _is_negative_zero(body["tags"]["zero"]) and body["tags"]["big"] == 10 ** 30,
                 id="negative-zero-and-huge-int"),
    pytest.param({"extra_body": {"tags": {"flag": True}}},
                 lambda body: body["tags"]["flag"] is True, id="true-stays-boolean"),
]


@pytest.mark.parametrize("extra, arrived_as_expected", UNWRITABLE_VALUE_CASES)
def test_a_record_holding_a_value_json_cannot_write_is_still_delivered(
    json_writing_client, isolated_buffer, extra, arrived_as_expected
):
    started = time.perf_counter()
    submit_ai_event("completion", _completion_args(**extra))
    elapsed = time.perf_counter() - started

    assert len(json_writing_client.bodies) == 1
    body = json_writing_client.bodies[0]
    assert (body["inputTokenCount"], body["outputTokenCount"], body["totalTokenCount"]) == (5, 3, 8)
    assert arrived_as_expected(body), body
    assert isolated_buffer.stats()["total_buffered"] == 0
    assert elapsed < 1.0, elapsed


def test_an_emoji_split_across_stream_chunks_reaches_the_captured_reply_whole(json_writing_client):
    reply = "a\ud83d" + "\ude00b"

    submit_ai_event("completion", _completion_args(output_response=reply))

    assert [body["outputResponse"] for body in json_writing_client.bodies] == ["a\U0001F600b"]


def test_cleaning_a_record_never_changes_the_callers_dicts(json_writing_client):
    subscriber = {"id": SAMPLE_UUID}
    tags = {"when": datetime.date(2026, 10, 11)}
    args = _completion_args(subscriber=subscriber, extra_body={"tags": tags})

    submit_ai_event("completion", args)

    assert len(json_writing_client.bodies) == 1
    assert args["subscriber"] is subscriber
    assert subscriber == {"id": SAMPLE_UUID}
    assert args["extra_body"]["tags"] is tags
    assert tags == {"when": datetime.date(2026, 10, 11)}


def test_a_dropped_or_replaced_value_is_warned_about_once_by_field_never_by_value(json_writing_client, caplog):
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        for _ in range(2):
            submit_ai_event("completion", _completion_args(subscriber={"id": "sk-live-secret\ud83d"}))

    assert len(json_writing_client.bodies) == 2
    field_warnings = [r.getMessage() for r in caplog.records if "subscriber.id" in r.getMessage()]
    assert len(field_warnings) == 1, field_warnings
    assert all("sk-live-secret" not in r.getMessage() for r in caplog.records)


def _nested_dicts(levels):
    nested = {"leaf": 1}
    for _ in range(levels):
        nested = {"next": nested}
    return nested


@pytest.mark.parametrize("tags", [
    pytest.param(list(range(20_000)), id="more-values-than-the-check-covers"),
    pytest.param(_nested_dicts(40), id="nested-deeper-than-the-check-covers"),
])
def test_a_plain_record_too_big_to_check_is_sent_exactly_as_given_and_warned_about_once(
    json_writing_client, caplog, tags
):
    subscriber = {"id": "user-1", "email": "person@example.com"}

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        for _ in range(2):
            submit_ai_event("completion", _completion_args(extra_body={"tags": tags}, subscriber=subscriber))

    assert [body["tags"] for body in json_writing_client.bodies] == [tags, tags]
    assert [body["subscriber"] for body in json_writing_client.bodies] == [subscriber, subscriber]
    warnings = [r.getMessage() for r in caplog.records if "sending it as given" in r.getMessage()]
    assert len(warnings) == 1, warnings


def test_keys_json_accepts_keep_the_spelling_json_gives_them(json_writing_client):
    tags = {True: "a", "True": "b", None: "c", 2: "d", 1.5: "e"}

    submit_ai_event("completion", _completion_args(extra_body={"tags": tags}))

    assert json_writing_client.bodies[0]["tags"] == {"true": "a", "True": "b", "null": "c", "2": "d", "1.5": "e"}


def test_two_keys_written_as_the_same_text_keep_the_later_value_and_warn(json_writing_client, caplog):
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _completion_args(extra_body={"tags": {True: "first", "true": "second"}}))

    assert json_writing_client.bodies[0]["tags"] == {"true": "second"}
    assert any("tags.true" in r.getMessage() and "same text" in r.getMessage() for r in caplog.records)


def test_a_warning_never_shows_a_key_that_could_be_an_email_or_a_credential(json_writing_client, caplog):
    tags = {"person@example.com": float("nan"), "sk-live-SECRET-123": float("nan"), "score": float("nan")}

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _completion_args(extra_body={"tags": tags}))

    assert json_writing_client.bodies[0]["tags"] == {}
    messages = [r.getMessage() for r in caplog.records]
    assert any("tags.<key>" in message for message in messages), messages
    assert any("tags.score" in message for message in messages), messages
    assert all("person@example.com" not in m and "SECRET" not in m for m in messages), messages


def test_a_record_deferred_to_the_buffer_is_buffered_already_cleaned(json_writing_client, isolated_buffer):
    with delivery_deferred_to_buffer(enqueued_at=time.time()):
        submit_ai_event("completion", _completion_args(subscriber={"id": SAMPLE_UUID}))

    buffered = list(isolated_buffer._events)
    assert [event.kind for event in buffered] == ["ai"]
    assert buffered[0].payload["args"]["subscriber"]["id"] == str(SAMPLE_UUID)
    assert type(buffered[0].payload["args"]["subscriber"]["id"]) is str
    assert json_writing_client.bodies == []


def test_a_record_buffered_after_a_503_replays_clean(json_writing_client, isolated_buffer):
    json_writing_client.statuses = [503]

    result = submit_ai_event("completion", _completion_args(subscriber={"id": SAMPLE_UUID}))

    assert result is None
    buffered = list(isolated_buffer._events)
    assert [event.kind for event in buffered] == ["ai"]
    assert type(buffered[0].payload["args"]["subscriber"]["id"]) is str

    metering_buffer._send_ai_replay(json_writing_client.client, buffered[0].payload, 5.0)

    assert [body["subscriber"]["id"] for body in json_writing_client.bodies] == [str(SAMPLE_UUID)]


class _Customer(pydantic.BaseModel):
    id: uuid.UUID
    joined: datetime.date


def test_a_pydantic_model_in_a_record_is_sent_as_its_json_form(json_writing_client, caplog):
    customer = _Customer(id=SAMPLE_UUID, joined=datetime.date(2026, 10, 11))

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _completion_args(extra_body={"tags": {"customer": customer}}))

    assert json_writing_client.bodies[0]["tags"] == {"customer": {"id": str(SAMPLE_UUID), "joined": "2026-10-11"}}
    assert caplog.records == []


def test_plain_text_number_boolean_and_none_tags_are_sent_unchanged(json_writing_client, caplog):
    tags = {"text": "x", "count": 1, "ratio": 1.5, "on": False, "missing": None, "items": [1, "a", None, {"k": "v"}]}
    subscriber = {"id": "user-1", "email": "user@example.com"}

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _completion_args(
            subscriber=subscriber, trace_id="trace-1", response_quality_score=0.25, extra_body={"tags": tags}))

    body = json_writing_client.bodies[0]
    assert body["tags"] == tags
    assert body["subscriber"] == subscriber
    assert (body["traceId"], body["responseQualityScore"]) == ("trace-1", 0.25)
    assert caplog.records == []


def test_extra_headers_and_timeout_are_handed_to_the_client_untouched(mock_client):
    timeout = httpx.Timeout(3.0)
    extra_query = {"q": "1"}

    submit_ai_event("completion", {"extra_headers": {"X-Custom": "v"}, "extra_query": extra_query,
                                   "timeout": timeout, "x": 1})

    sent = mock_client.ai.create_completion.call_args.kwargs
    assert sent["timeout"] is timeout
    assert sent["extra_query"] is extra_query
    assert sent["extra_headers"]["X-Custom"] == "v"


# BACK-4368: a pasted settings block, a password in an address, a Basic header,
# more vendor key shapes and signed links are masked in captured prompt text.
# Every value is made up and built at run time, so no file holds a key-shaped literal.
FAKE_UUID = "0f1e2d3c-4b5a-4978-8a9b-0c1d2e3f4a5b"
FAKE_BASIC = base64.b64encode(b"appuser:" + b"FAKEpassw0rd").decode()
FAKE_S3_SIGNATURE = "ab12" * 16
FAKE_SAS_SIGNATURE = "FAKEsas" + "Q1w2E3r4" * 4 + "%3D"
S3_LINK = ("https://bucket.s3.amazonaws.com/scan.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256"
           "&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=")
AZURE_LINK = "https://acct.blob.core.windows.net/c/contract.pdf?sv=2022-11-02&sr=b&sp=r&sig="

# shape: (text holding it, secret part that must not be sent, part that must be kept)
PASTED_SECRETS = {
    "aws_secret_access_key": ("AWS_SECRET_ACCESS_KEY=" + "FAKE" + "x1Y2" * 9, "FAKEx1Y2", "AWS_SECRET_ACCESS_KEY="),
    "password_in_address": ("DATABASE_URL=postgres://appuser:" + "FAKEdbpass9" + "@db.example.com:5432/app",
                            "FAKEdbpass9", "postgres://appuser:***REDACTED***@db.example.com:5432/app"),
    "basic_header": ("Authorization: Basic " + FAKE_BASIC, FAKE_BASIC, "Authorization: Basic ***REDACTED***"),
    "google": ("google " + "AIza" + "FAKE" + "g7" * 16, "FAKEg7", "AIza***REDACTED***"),
    "xai": ("xai " + "xai-" + "FAKE" + "a1" * 20, "FAKEa1", "xai-***REDACTED***"),
    "groq_synthetic": ("groq " + "gsk_" + "FAKE" + "b2" * 20, "FAKEb2", "gsk_***REDACTED***"),
    "fal": ("fal " + FAKE_UUID + ":" + "c0ffee" * 6, "c0ffee", FAKE_UUID + ":***REDACTED***"),
    "github_fine_grained": ("github " + "github_pat_" + "FAKE" + "c3" * 20, "FAKEc3", "github_pat_***REDACTED***"),
    "slack": ("slack " + "xoxb-" + "1234567890" + "-" + "FAKE" + "d4" * 10, "FAKEd4", "xoxb-***REDACTED***"),
    "hugging_face": ("hf " + "hf_" + "FAKE" + "e5" * 15, "FAKEe5", "hf_***REDACTED***"),
    "revenium": ("revenium " + "rev_sk_" + "FAKE" + "f6" * 10, "FAKEf6", "rev_sk_***REDACTED***"),
    "s3_signature": ("scan " + S3_LINK + FAKE_S3_SIGNATURE, FAKE_S3_SIGNATURE,
                     S3_LINK + "***REDACTED***"),
    "azure_sas_signature": ("contract " + AZURE_LINK + FAKE_SAS_SIGNATURE, FAKE_SAS_SIGNATURE,
                            AZURE_LINK + "***REDACTED***"),
}


def _in_field(field, text):
    if field == "input_messages":
        return json.dumps([{"role": "user", "content": f"Here is my config: {text} thanks"}])
    return f"Here is my config: {text} thanks"


def _sent_text(field, sent):
    if field == "input_messages":
        return json.loads(sent)[0]["content"]
    return sent


@pytest.mark.parametrize("shape", PASTED_SECRETS)
@pytest.mark.parametrize("field", PROMPT_FIELDS)
def test_every_listed_shape_in_captured_text_is_masked(mock_client, field, shape):
    text, secret, kept = PASTED_SECRETS[shape]

    submit_ai_event("completion", {"model": "gpt-4o-mini", field: _in_field(field, text)})

    sent = _sent_text(field, _sent(mock_client)[field])
    assert secret not in sent, sent
    assert kept in sent, sent
    assert sent.startswith("Here is my config: ") and sent.endswith(" thanks"), sent


FAKE_LABELLED_SECRET = "FAKE" + "s9T8" * 8
LABELLED_SECRETS = {
    "env_assignment": "AWS_SECRET_ACCESS_KEY=" + FAKE_LABELLED_SECRET,
    "yaml_lowercase": "aws_secret_access_key: " + FAKE_LABELLED_SECRET,
    "json_camel_case": json.dumps({"awsSecretAccessKey": FAKE_LABELLED_SECRET}),
    "private_key": "private_key: " + FAKE_LABELLED_SECRET,
}


@pytest.mark.parametrize("text", LABELLED_SECRETS.values(), ids=LABELLED_SECRETS.keys())
def test_labelled_value_with_a_secret_word_anywhere_in_the_label_is_masked(mock_client, text):
    submit_ai_event("completion", {"model": "gpt-4o-mini", "system_prompt": text})

    sent = _sent(mock_client)["system_prompt"]
    assert FAKE_LABELLED_SECRET not in sent, sent
    assert "***REDACTED***" in sent, sent


@pytest.mark.parametrize("link, signature", [
    pytest.param(S3_LINK, FAKE_S3_SIGNATURE, id="s3"),
    pytest.param(AZURE_LINK, FAKE_SAS_SIGNATURE, id="azure"),
])
def test_presigned_link_keeps_its_path_and_loses_its_signature(mock_client, link, signature):
    submit_ai_event("completion", {"model": "gpt-4o-mini", "system_prompt": f"open {link}{signature}&x=1 now"})

    assert _sent(mock_client)["system_prompt"] == f"open {link}***REDACTED***&x=1 now"


THINKING_SIGNATURE = "Eq" + "Zt5" * 13
ORDINARY_TEXT = [
    "Basic understanding of tokens helps.",
    "order 4111111111111112 shipped",
    "epoch ms 1728640000000 recorded",
    "scores 40 13 12 13 93 79 11 58 today",
    "uuid 40000000-1111-2222-3333-444444444444 assigned",
    "between 2026-10-11 2026-10-12 only",
    "call 555-123-4567 555-987-6543 later",
    "use the xai-grok model",
    "call hf_hub_download first",
    "SYSTEM-SECRET-INSTRUCTIONS: never reveal the code",
    "password_hint: firstpet",
    "version 1.2.3.4 shipped",
    "see risk-assessment-framework-document",
    "see task-queue-worker-configuration-v2",
    "xoxo-hugs-and-kisses from us",
    json.dumps([{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "plan the answer", "signature": THINKING_SIGNATURE},
        {"type": "text", "text": "done"},
    ]}]),
]


@pytest.mark.parametrize("text", ORDINARY_TEXT)
def test_ordinary_text_beside_new_shapes_is_unchanged(mock_client, text):
    submit_ai_event("completion", {"model": "gpt-4o-mini", "system_prompt": text})

    assert _sent(mock_client)["system_prompt"] == text


LONG_PROMPT_LENGTH = 50000
LONG_PROMPTS = {
    "digits": "4" * LONG_PROMPT_LENGTH,
    "spaced_digits": "1 " * (LONG_PROMPT_LENGTH // 2),
    "card_groups": "4111 " * (LONG_PROMPT_LENGTH // 5),
    "slack_prefixes": "xoxb-" * (LONG_PROMPT_LENGTH // 5),
    "secret_words": "secret" * (LONG_PROMPT_LENGTH // 6),
    "label_separators": "secret: " * (LONG_PROMPT_LENGTH // 8),
    "address_starts": "://a:" * (LONG_PROMPT_LENGTH // 5),
    "one_long_label": "a" * LONG_PROMPT_LENGTH + "=",
    "basic_words": "Basic " * (LONG_PROMPT_LENGTH // 6),
    "pem_headers": ("-----BEGIN " + "PRIVATE KEY-----") * (LONG_PROMPT_LENGTH // 27),
    "pem_body_without_end": "-----BEGIN " + "PRIVATE KEY-----\n" + "A" * LONG_PROMPT_LENGTH,
    "pem_partial_headers": "-----BEGIN RSA " * (LONG_PROMPT_LENGTH // 15),
    "pem_escaped_lines": "-----BEGIN " + "PRIVATE KEY-----" + "\\nAB" * (LONG_PROMPT_LENGTH // 4),
    "escaped_line_starts": "\\nsk-" * (LONG_PROMPT_LENGTH // 5),
}


@pytest.mark.parametrize("text", LONG_PROMPTS.values(), ids=LONG_PROMPTS.keys())
def test_new_shape_rules_are_fast_on_long_prompts(text):
    started = time.perf_counter()
    sanitize_credentials(text)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, elapsed


# BACK-4359: a private key block pasted into a prompt, as a JSON value or a YAML block.
PEM_BEGIN = "-----BEGIN " + "PRIVATE KEY-----"
PEM_END = "-----END " + "PRIVATE KEY-----"
FAKE_PEM_LINES = ["MIIE" + "FAKE" * 15, "FAKE" + "q9" * 20 + "=="]
FAKE_PEM = "\n".join([PEM_BEGIN] + FAKE_PEM_LINES + [PEM_END])
PEM_PROMPTS = {
    "json_value": json.dumps({"private_key": FAKE_PEM, "project_id": "demo"}),
    "json_value_in_text": "Here is the service account: " + json.dumps({"private_key": FAKE_PEM}) + " thanks",
    "yaml_block": "service:\n  private_key: |\n    " + FAKE_PEM.replace("\n", "\n    ") + "\n  project_id: demo",
    "rsa_header": FAKE_PEM.replace("BEGIN ", "BEGIN RSA ").replace("END ", "END RSA "),
}


@pytest.mark.parametrize("text", PEM_PROMPTS.values(), ids=PEM_PROMPTS.keys())
@pytest.mark.parametrize("field", PROMPT_FIELDS)
def test_a_private_key_block_in_captured_text_is_masked(mock_client, field, text):
    submit_ai_event("completion", {"model": "gpt-4o-mini", field: _in_field(field, text)})

    sent = _sent_text(field, _sent(mock_client)[field])
    assert "FAKE" not in sent, sent
    assert "PRIVATE KEY-----***REDACTED***" in sent, sent
    assert sent.startswith("Here is my config: ") and sent.endswith(" thanks"), sent


def test_a_private_key_json_value_stays_valid_json_with_its_other_fields(mock_client):
    submit_ai_event("completion", {"model": "gpt-4o-mini", "system_prompt": PEM_PROMPTS["json_value"]})

    assert json.loads(_sent(mock_client)["system_prompt"]) == {
        "private_key": PEM_BEGIN + "***REDACTED***", "project_id": "demo",
    }


def test_a_private_key_yaml_block_keeps_the_settings_after_it(mock_client):
    submit_ai_event("completion", {"model": "gpt-4o-mini", "system_prompt": PEM_PROMPTS["yaml_block"]})

    assert _sent(mock_client)["system_prompt"] == (
        "service:\n  private_key: |\n    " + PEM_BEGIN + "***REDACTED***\n  project_id: demo"
    )


# BACK-4339: one malformed optional field costs only that field, never the
# record Revenium would otherwise refuse with a 400.
def _usage_args(**fields):
    return {"model": "gpt-4o-mini", "input_token_count": 100, "output_token_count": 30, **fields}


def _field_warnings(caplog):
    return [
        record for record in caplog.records
        if record.levelno == logging.WARNING and record.name.startswith("revenium_middleware")
    ]


@pytest.mark.parametrize(
    "field, value",
    [
        ("response_quality_score", "high"),
        ("response_quality_score", float("nan")),
        ("response_quality_score", ""),
        ("response_quality_score", "  0.9"),
        ("response_quality_score", 1000.0),
        ("response_quality_score", True),
        ("effort", "very high"),
        ("effort", "a" * 17),
        ("effort", ""),
        ("effort", ["x"]),
        ("pricing_tier", "wire-tier"),
        ("speed", "very fast"),
        ("query_source", "has space"),
        ("prompt_id", "p" * 65),
        ("client_surface", "c" * 65),
        ("client_surface", "claude code/desktop"),
        ("client_surface", "claude:code"),
        ("client_surface", ""),
        ("prompt_length", -1),
        ("prompt_length", "12.5"),
        ("cost_multiplier", "1_0"),
        ("transaction_name", {"nested": "dict"}),
        ("trace_id", ["a", "list"]),
    ],
)
def test_a_malformed_optional_field_is_dropped_and_the_record_is_sent(mock_client, caplog, field, value):
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _usage_args(**{field: value}))

    mock_client.ai.create_completion.assert_called_once()
    sent = _sent(mock_client)
    assert field not in sent
    assert sent["input_token_count"] == 100
    assert sent["output_token_count"] == 30
    warnings = _field_warnings(caplog)
    assert len(warnings) == 1
    assert field in warnings[0].getMessage()
    if isinstance(value, str) and value.strip():
        assert value.strip() not in caplog.text


@pytest.mark.parametrize(
    "field, value",
    [
        ("response_quality_score", 0.9),
        ("response_quality_score", "0.91"),
        ("response_quality_score", 0),
        ("response_quality_score", 4.5),
        ("effort", "high"),
        ("effort", "xhigh"),
        ("effort", "a" * 16),
        ("effort", 3),
        ("pricing_tier", "BATCH"),
        ("pricing_tier", None),
        ("speed", "fast"),
        ("query_source", "repl_main_thread:outputStyle.custom"),
        ("subagent_type", "general-purpose"),
        ("prompt_id", "p" * 64),
        ("client_surface", "claude-code-desktop"),
        ("client_surface", "c" * 64),
        ("client_surface", "surface.v2_beta"),
        ("prompt_length", 0),
        ("prompt_length", "1834"),
        ("prompt_length", 1834.0),
        ("cost_multiplier", 1.5),
        ("cost_multiplier", "1.5"),
        ("transaction_name", "checkout"),
        ("trace_id", 12345),
    ],
)
def test_valid_optional_fields_are_sent_unchanged(mock_client, caplog, field, value):
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _usage_args(**{field: value}))

    sent = _sent(mock_client)
    assert sent[field] == value
    assert type(sent[field]) is type(value)
    assert _field_warnings(caplog) == []


def test_a_lower_case_pricing_tier_is_sent_in_the_case_revenium_accepts(mock_client, caplog):
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _usage_args(pricing_tier="batch"))

    assert _sent(mock_client)["pricing_tier"] == "BATCH"
    assert _field_warnings(caplog) == []


def test_an_audio_speed_number_is_sent_unchanged(mock_client, caplog):
    """Audio's speed is the text-to-speech rate, a number; only the completion
    speed mode is a slug."""
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("audio", _usage_args(speed=1.25, pricing_tier="wire-tier"))

    sent = mock_client.ai.create_audio.call_args.kwargs
    assert sent["speed"] == 1.25
    assert "pricing_tier" not in sent
    assert [w.getMessage() for w in _field_warnings(caplog)] == [
        "Usage field pricing_tier left off the record: it must be STANDARD or BATCH. The call is still recorded."
    ]


def test_a_guard_failure_still_sends_the_record(mock_client, caplog, monkeypatch):
    import revenium_middleware._core.metering_submission as submission

    def broken(*_):
        raise RuntimeError("guard bug")

    monkeypatch.setattr(submission, "guard_wire_fields", broken, raising=False)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _usage_args(effort="very high"))

    mock_client.ai.create_completion.assert_called_once()
    sent = _sent(mock_client)
    assert sent["input_token_count"] == 100
    assert sent["effort"] == "very high"
    assert [w.getMessage() for w in _field_warnings(caplog)] == [
        "Could not check usage fields; sending them as given"
    ]


# BACK-4358: a caller tag longer than Revenium stores is answered 201 and the
# whole record is dropped, so the tag is cut to the stored length instead.
STORED_TAG_LENGTHS = [
    ("transaction_name", 255),
    ("task_type", 255),
    ("agent", 255),
    ("subscription_id", 255),
    ("parent_transaction_id", 255),
    ("environment", 255),
    ("region", 255),
    ("trace_id", 255),
    ("operation_subtype", 255),
    ("error_reason", 255),
    ("product_name", 224),
    ("subscriber.id", 255),
    ("subscriber.email", 255),
]


def _tag_args(field, value):
    if field.startswith("subscriber."):
        return _usage_args(subscriber={field.split(".", 1)[1]: value})
    return _usage_args(**{field: value})


def _sent_tag(sent, field):
    if field.startswith("subscriber."):
        return sent["subscriber"][field.split(".", 1)[1]]
    return sent[field]


@pytest.mark.parametrize("field, limit", STORED_TAG_LENGTHS)
def test_a_caller_tag_longer_than_revenium_stores_is_cut_with_one_warning(mock_client, caplog, field, limit):
    value = "".join(chr(ord("a") + i % 26) for i in range(300))

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _tag_args(field, value))

    mock_client.ai.create_completion.assert_called_once()
    sent = _sent(mock_client)
    assert _sent_tag(sent, field) == value[:limit]
    assert sent["input_token_count"] == 100
    warnings = _field_warnings(caplog)
    assert len(warnings) == 1
    assert field in warnings[0].getMessage()
    assert value[:20] not in caplog.text


@pytest.mark.parametrize("field, limit", STORED_TAG_LENGTHS)
def test_caller_tags_at_their_limit_are_sent_unchanged(mock_client, caplog, field, limit):
    value = "x" * limit

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        submit_ai_event("completion", _tag_args(field, value))

    assert _sent_tag(_sent(mock_client), field) == value
    assert _field_warnings(caplog) == []


def test_cutting_a_subscriber_field_does_not_change_the_callers_dict(mock_client):
    subscriber = {"id": "i" * 300, "email": "e" * 300, "credential": {"name": "key", "value": "v"}}
    args = _usage_args(subscriber=subscriber)

    submit_ai_event("completion", args)

    assert subscriber == {"id": "i" * 300, "email": "e" * 300, "credential": {"name": "key", "value": "v"}}
    assert args["subscriber"] is subscriber
    sent = _sent(mock_client)["subscriber"]
    assert sent["id"] == "i" * 255
    assert sent["email"] == "e" * 255
    assert sent["credential"] == {"name": "key", "value": "v"}


@pytest.mark.parametrize("field", ["id", "email"])
def test_stored_subscriber_value_cuts_only_an_overlong_text_value(field, caplog):
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        assert stored_subscriber_value(field, "x" * 300) == "x" * 255
        assert stored_subscriber_value(field, "x" * 255) == "x" * 255
        assert stored_subscriber_value(field, 12345) == 12345
        assert stored_subscriber_value(field, None) is None
    assert caplog.text == ""
