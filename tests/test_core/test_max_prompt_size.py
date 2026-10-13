"""REVENIUM_MAX_PROMPT_SIZE limits captured prompts in every integration, as in the Go and Node SDKs (BACK-4413).

The limit is read once at import, so each case runs in a fresh interpreter.
"""
import json
import os
import subprocess
import sys

import pytest

PROMPT_LENGTH = 5_000

CAPTURED_LENGTHS = f"""
from revenium_middleware._core.prompt_extraction import extract_prompts_from_request as extract_openai
from revenium_middleware.google.prompt_extractor import extract_prompts_from_request as extract_google

prompt = "A" * {PROMPT_LENGTH}
openai = extract_openai({{"messages": [{{"role": "system", "content": prompt}}, {{"role": "user", "content": "hi"}}]}})
google = extract_google({{"system_instruction": prompt, "contents": "hi"}})
print(len(openai["systemPrompt"]), len(google["systemPrompt"]))
"""


def _captured_lengths(settings):
    env = {name: value for name, value in os.environ.items()
           if name not in ("REVENIUM_MAX_PROMPT_SIZE", "REVENIUM_MAX_PROMPT_LENGTH")}
    env.update(settings, REVENIUM_CAPTURE_PROMPTS="true")
    result = subprocess.run([sys.executable, "-c", CAPTURED_LENGTHS], env=env,
                            capture_output=True, text=True, check=True, timeout=120)
    openai_length, google_length = result.stdout.split()[-2:]
    return int(openai_length), int(google_length)


@pytest.mark.parametrize("settings, expected", [
    pytest.param({"REVENIUM_MAX_PROMPT_SIZE": "2000"}, (2000, 2000), id="size_only"),
    pytest.param({"REVENIUM_MAX_PROMPT_LENGTH": "1500"}, (PROMPT_LENGTH, 1500), id="gemini_length_only"),
    pytest.param({"REVENIUM_MAX_PROMPT_SIZE": "2000", "REVENIUM_MAX_PROMPT_LENGTH": "1500"}, (2000, 2000),
                 id="size_wins_over_length"),
    pytest.param({"REVENIUM_MAX_PROMPT_SIZE": "", "REVENIUM_MAX_PROMPT_LENGTH": "1500"}, (PROMPT_LENGTH, 1500),
                 id="empty_size_keeps_length"),
    pytest.param({"REVENIUM_MAX_PROMPT_SIZE": "10"}, (200, 200), id="size_below_the_minimum"),
    pytest.param({"REVENIUM_MAX_PROMPT_LENGTH": "10"}, (PROMPT_LENGTH, 200), id="gemini_length_below_the_minimum"),
])
def test_the_prompt_size_setting_limits_captured_prompts(settings, expected):
    assert _captured_lengths(settings) == expected


GEMINI_DEFAULTS = """
import json
from types import SimpleNamespace

from revenium_middleware.google.config import Config
from revenium_middleware.google.prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content,
)

def parses(text):
    try:
        return isinstance(json.loads(text), list)
    except ValueError:
        return False


text = "A" * {length}
request = extract_prompts_from_request({{"system_instruction": text, "contents": text}})
part = SimpleNamespace(text=text)
reply_response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
reply = extract_response_content(reply_response)
streamed = extract_streaming_response_content(text)
input_messages = request["inputMessages"]
print(json.dumps({{
    "limit": Config.MAX_PROMPT_LENGTH,
    "system": [len(request["systemPrompt"]), request["systemPrompt"].endswith("...[TRUNCATED]")],
    "input_parses": parses(input_messages),
    "input_length": len(input_messages),
    "input_flag": request["promptsTruncated"],
    "reply": [len(reply["outputResponse"]), reply["promptsTruncated"]],
    "streamed": [len(streamed["outputResponse"]), streamed["outputResponse"].endswith("...[TRUNCATED]"),
                 streamed["promptsTruncated"]],
}}))
"""

LIMITS_SEEN = """
import json

from revenium_middleware.anthropic.prompt_extractor import Config as anthropic
from revenium_middleware.google.prompt_extractor import Config as google
from revenium_middleware._core.prompt_extraction import Config as core

print(json.dumps([core.MAX_PROMPT_LENGTH, anthropic.MAX_PROMPT_LENGTH, google.MAX_PROMPT_LENGTH]))
"""

MESSAGE_LISTS = """
import json

from revenium_middleware.anthropic.prompt_extractor import extract_prompts_from_request as extract_anthropic
from revenium_middleware.google.prompt_extractor import extract_prompts_from_request as extract_google
from revenium_middleware._core.prompt_extraction import extract_prompts_from_request as extract_openai


def parses(text):
    try:
        return isinstance(json.loads(text), list)
    except ValueError:
        return False


text = "A" * 5_000
messages = [{"role": "user", "content": text}]
openai = extract_openai({"messages": messages})
anthropic = extract_anthropic({"messages": messages})
google = extract_google({"contents": text})
print(json.dumps([
    [len(result["inputMessages"]), parses(result["inputMessages"]), result["promptsTruncated"]]
    for result in (openai, anthropic, google)
]))
"""


def _run_script(script, settings):
    env = {name: value for name, value in os.environ.items()
           if name not in ("REVENIUM_MAX_PROMPT_SIZE", "REVENIUM_MAX_PROMPT_LENGTH")}
    env.update(settings, REVENIUM_CAPTURE_PROMPTS="true")
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_with_no_setting_gemini_keeps_30000_characters_whole():
    seen = _run_script(GEMINI_DEFAULTS.format(length=30_000), {})

    assert seen["limit"] == 50_000
    assert seen["system"] == [30_000, False]
    assert seen["input_parses"] is True
    assert seen["input_length"] > 30_000
    assert seen["input_flag"] is False
    assert seen["reply"] == [30_000, False]
    assert seen["streamed"] == [30_000, False, False]


def test_with_no_setting_gemini_cuts_60000_characters_to_50000():
    seen = _run_script(GEMINI_DEFAULTS.format(length=60_000), {})

    assert seen["system"] == [50_000, True]
    assert seen["reply"] == [50_000, True]
    assert seen["streamed"] == [50_000, True, True]
    assert seen["input_parses"] is True
    assert 49_000 < seen["input_length"] <= 50_000
    assert seen["input_flag"] is True


def test_with_no_setting_every_integration_has_the_same_limit():
    assert _run_script(LIMITS_SEEN, {}) == [50_000, 50_000, 50_000]


def test_the_prompt_size_setting_limits_message_lists_everywhere():
    seen = _run_script(MESSAGE_LISTS, {"REVENIUM_MAX_PROMPT_SIZE": "2000"})

    for length, parses, flagged in seen:
        assert parses is True
        assert 1_900 < length <= 2_000
        assert flagged is True
