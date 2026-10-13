"""LiteLLM calls the SDK does not record never pass Revenium metadata on to the provider (BACK-4405).

LiteLLM forwards keywords it does not know inside the provider request, so an
unwrapped call given ``usage_metadata`` sent the end user's email and any
credential value to the image, audio, video or search provider. The requests
here reach a loopback stub, so the assertions are on the bytes the provider
would have received.
"""
import asyncio
import io
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402

import revenium_middleware  # noqa: E402
import revenium_middleware.litellm.client.middleware  # noqa: E402,F401
from revenium_middleware import _core  # noqa: E402
from revenium_middleware._core import metering, metering_pool, metering_submission  # noqa: E402

from . import fake_openai_upstream  # noqa: E402

CANARY_EMAIL = b"canary.person@example.test"
CANARY_CREDENTIAL = b"CANARYVALUE"
USAGE_METADATA = {
    "subscriber": {"email": CANARY_EMAIL.decode(), "credential": {"name": "n", "value": CANARY_CREDENTIAL.decode()}},
    "organization_name": "ORGCANARY",
}
IMAGE_REPLY = b'{"created": 1, "data": [{"b64_json": "aGk="}]}'


class _RecordingHandler(fake_openai_upstream._Handler):
    requests = []

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        type(self).requests.append((self.path, self.rfile.read(length)))
        reply = b'{"text": "hi"}' if self.path.endswith("/audio/transcriptions") else IMAGE_REPLY
        self._send("application/json", reply)


@pytest.fixture
def provider():
    _RecordingHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake_openai_upstream.api_base(server), _RecordingHandler.requests
    server.shutdown()
    server.server_close()


@pytest.fixture
def revenium(monkeypatch):
    """A recording Revenium client: any usage record the SDK queues lands on it."""
    recorder = MagicMock(name="revenium")
    for module in (metering, metering_submission, _core, revenium_middleware):
        monkeypatch.setattr(module, "client", recorder, raising=False)
    return recorder


def revenium_records(recorder):
    assert metering_pool.drain(5) == 0
    return [name for name, _args, _kwargs in recorder.ai.method_calls if name.startswith("create_")]


def png():
    image = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    image.name = "a.png"
    return image


def assert_one_request_without_metadata(requests):
    assert len(requests) == 1
    _path, body = requests[0]
    assert b"usage_metadata" not in body
    assert CANARY_EMAIL not in body
    assert CANARY_CREDENTIAL not in body
    return body


@pytest.mark.parametrize("kind", ["sync", "async"])
def test_image_generation_does_not_send_usage_metadata_to_the_provider(kind, provider, revenium):
    api_base, requests = provider
    arguments = dict(model="openai/gpt-image-1", prompt="a fox", api_base=api_base, api_key="sk-test",
                     usage_metadata=USAGE_METADATA)

    if kind == "sync":
        litellm.image_generation(**arguments)
    else:
        asyncio.run(litellm.aimage_generation(**arguments))

    assert requests[0][0] == "/v1/images/generations"
    body = assert_one_request_without_metadata(requests)
    assert b'"prompt": "a fox"' in body or b'"prompt":"a fox"' in body
    assert revenium_records(revenium) == []


def _run(result):
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


def _call_and_ignore_the_stub_reply(call):
    """The stub answers every path with an image reply, which some calls cannot parse after sending."""
    try:
        _run(call())
    except Exception:
        pass


OTHER_UNRECORDED_CALLS = {
    "image_edit": lambda base: litellm.image_edit(
        model="openai/gpt-image-1", image=png(), prompt="a fox", api_base=base, api_key="sk-test",
        usage_metadata=USAGE_METADATA),
    "aimage_edit": lambda base: litellm.aimage_edit(
        model="openai/gpt-image-1", image=png(), prompt="a fox", api_base=base, api_key="sk-test",
        usage_metadata=USAGE_METADATA),
    "transcription": lambda base: litellm.transcription(
        model="openai/whisper-1", file=("a.wav", b"RIFF0000WAVE"), api_base=base, api_key="sk-test",
        usage_metadata=USAGE_METADATA),
    "atranscription": lambda base: litellm.atranscription(
        model="openai/whisper-1", file=("a.wav", b"RIFF0000WAVE"), api_base=base, api_key="sk-test",
        usage_metadata=USAGE_METADATA),
    "video_generation": lambda base: litellm.video_generation(
        model="openai/sora-2", prompt="a fox", api_base=base, api_key="sk-test", usage_metadata=USAGE_METADATA),
    "avideo_generation": lambda base: litellm.avideo_generation(
        model="openai/sora-2", prompt="a fox", api_base=base, api_key="sk-test", usage_metadata=USAGE_METADATA),
    "search": lambda base: litellm.search(
        query="q", search_provider="tavily", api_base=base, api_key="sk-test", usage_metadata=USAGE_METADATA),
    "asearch": lambda base: litellm.asearch(
        query="q", search_provider="tavily", api_base=base, api_key="sk-test", usage_metadata=USAGE_METADATA),
}


@pytest.mark.parametrize("name", sorted(OTHER_UNRECORDED_CALLS))
def test_other_unrecorded_litellm_calls_do_not_send_usage_metadata(name, provider):
    if not hasattr(litellm, name):
        pytest.skip(f"litellm {name} is not available in this LiteLLM version")
    api_base, requests = provider

    _call_and_ignore_the_stub_reply(lambda: OTHER_UNRECORDED_CALLS[name](api_base))

    assert_one_request_without_metadata(requests)
