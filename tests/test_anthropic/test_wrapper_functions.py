"""The Anthropic wrapper names are the wrapper functions on every wrapt version (BACK-4235).

The wrappers were installed with ``@wrapt.patch_function_wrapper``, which on
wrapt below 2.2 binds the module name to the patched SDK method rather than to
the wrapper. The parse() and beta delegates call those names as wrappers, so on
those wrapt releases ``client.messages.parse()`` and the beta create and stream
calls raised ``TypeError`` in the customer's code.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("anthropic")

from anthropic.types import Message, TextBlock, Usage  # noqa: E402

from revenium_middleware.anthropic import middleware as mw  # noqa: E402


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper", "stream_wrapper", "async_stream_wrapper"])
def test_anthropic_wrapper_names_are_plain_functions(name):
    assert not hasattr(getattr(mw, name), "_self_wrapper")


def test_the_parse_delegate_calls_the_wrapped_method_once():
    message = Message(id="msg_parse", content=[TextBlock(type="text", text="hi")], model="claude-haiku-4-5",
                      role="assistant", stop_reason="end_turn", stop_sequence=None, type="message",
                      usage=Usage(input_tokens=3, output_tokens=5))
    wrapped = MagicMock(return_value=message)
    kwargs = {"model": "claude-haiku-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

    assert mw._parse_wrapper(wrapped, SimpleNamespace(_client=None), (), kwargs) is message
    wrapped.assert_called_once()
