"""A LangChain call that is cancelled, or a batch that fails part way, leaves
no run behind in the Revenium callback, and answered calls are still recorded
exactly as before (BACK-4379).

langchain-core sends no end hook to a run whose ``agenerate`` is cancelled
while it waits for the model, nor to the other runs of a sync ``generate``
batch once one message list raises. Stand-in chat models report a provider no
Revenium transport wrap meters, so every record comes from the callback.
"""
import asyncio
import tracemalloc
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain_core")

from langchain_core.callbacks import AsyncCallbackHandler  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from revenium_middleware._core.call_ownership import (  # noqa: E402
    _open_publications, _published_in_context, reset_claimed_response_ids,
)
from revenium_middleware.openai.langchain import _abandoned_runs, wrap  # noqa: E402
from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler  # noqa: E402
from entry_points import calls  # noqa: E402

INPUT_TOKENS = 13
OUTPUT_TOKENS = 5
FAILING_PROMPT = "the provider refuses this one"
PROMPT_SIZE = 20_000
CANCEL_AFTER = 0.001


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
    reset_claimed_response_ids()
    publications = _published_in_context.set(())
    yield
    _published_in_context.reset(publications)
    reset_claimed_response_ids()


@pytest.fixture
def records():
    recording_client = MagicMock()

    def recorded():
        calls.wait_for_metering()
        return [entry.kwargs for entry in recording_client.ai.create_completion.call_args_list]

    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recorded


def _answer():
    message = AIMessage(content="hi", response_metadata={"model_name": "command-r"},
                        usage_metadata={"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                                        "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS})
    return ChatResult(generations=[ChatGeneration(message=message)])


class StandIn(BaseChatModel):
    """A callback-only chat model: no Revenium transport wrap sees its calls.
    It refuses the message list whose last message is ``FAILING_PROMPT``."""

    ls_provider: str = "cohere"
    model: str = "command-r"
    delay: float = 0.0

    @property
    def _llm_type(self):
        return "cohere-stand-in"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": self.ls_provider, "ls_model_name": self.model, "ls_model_type": "chat"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if messages[-1].content == FAILING_PROMPT:
            raise RuntimeError("provider refused the call")
        return _answer()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._generate(messages)


class Hanging(StandIn):
    """A model that never answers, as a provider call cut off by a timeout."""

    started: Optional[Any] = None

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.started is not None:
            self.started.set()
        await asyncio.Event().wait()


class SlowUserHandler(AsyncCallbackHandler):
    """A caller's own inline handler that does I/O when a call ends."""

    run_inline = True
    ends_started = 0

    async def on_llm_end(self, response, **kwargs):
        self.ends_started += 1
        await asyncio.sleep(0.05)


class HoldingUserHandler(SlowUserHandler):
    """A caller's own inline handler whose end hook waits until ``released``."""

    released: Optional[asyncio.Event] = None

    async def on_llm_end(self, response, **kwargs):
        self.ends_started += 1
        await self.released.wait()


async def _cancelled(call, timeout=CANCEL_AFTER):
    try:
        await asyncio.wait_for(call, timeout)
    except asyncio.TimeoutError:
        return
    raise AssertionError("the call was expected to be cut off")


async def _cancelled_while_its_end_hook_runs(call, user_handler):
    ends_before = user_handler.ends_started
    task = asyncio.create_task(call)
    while user_handler.ends_started == ends_before:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _batch(*prompts):
    return [[HumanMessage(content=prompt)] for prompt in prompts]


def _active_runs(handler):
    return handler.get_stats()["active_runs"]


def test_two_thousand_cancelled_ainvoke_calls_leave_no_run_and_no_memory_behind():
    handler = UnifiedReveniumCallbackHandler()
    model = Hanging(callbacks=[handler])

    async def main():
        tracemalloc.start()
        try:
            for call in range(2000):
                await _cancelled(model.ainvoke(f"{call:06d}" + "x" * PROMPT_SIZE))
                if call == 199:
                    after_two_hundred = tracemalloc.get_traced_memory()[0]
            grown = tracemalloc.get_traced_memory()[0] - after_two_hundred
        finally:
            tracemalloc.stop()
        return grown, _open_publications()

    grown, open_publications = asyncio.run(main())

    assert _active_runs(handler) == 0
    assert open_publications == ()
    assert grown < 0.1 * 1800 * PROMPT_SIZE


def test_a_cancelled_task_leaves_no_run_behind():
    handler = UnifiedReveniumCallbackHandler()

    async def main():
        for _ in range(50):
            model = Hanging(callbacks=[handler], started=asyncio.Event())
            task = asyncio.create_task(model.ainvoke("hi"))
            await model.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(main())

    assert _active_runs(handler) == 0


def test_a_cancelled_batched_agenerate_leaves_no_run_behind():
    handler = UnifiedReveniumCallbackHandler()
    model = Hanging(callbacks=[handler])

    asyncio.run(_cancelled(model.agenerate(_batch("one", "two", "three"))))

    assert _active_runs(handler) == 0


def test_a_cancelled_call_on_two_handlers_is_forgotten_by_both():
    model = wrap(Hanging())
    wrapped_handler = model.callbacks[-1]
    config_handler = UnifiedReveniumCallbackHandler()

    async def main():
        for _ in range(5):
            await _cancelled(model.ainvoke("hi", config={"callbacks": [config_handler]}))

    asyncio.run(main())

    assert (_active_runs(wrapped_handler), _active_runs(config_handler)) == (0, 0)


def test_sync_generate_with_one_failing_prompt_leaves_no_run_behind(records):
    handler = UnifiedReveniumCallbackHandler()
    model = StandIn(callbacks=[handler])

    with pytest.raises(RuntimeError):
        model.generate(_batch("one", FAILING_PROMPT, "three"))

    assert _active_runs(handler) == 0
    assert records() == []


def test_cancelled_calls_send_no_record(records):
    handler = UnifiedReveniumCallbackHandler()
    model = Hanging(callbacks=[handler])

    async def main():
        for _ in range(20):
            await _cancelled(model.ainvoke("hi"))

    asyncio.run(main())

    assert records() == []


def test_answered_and_failed_calls_are_recorded_as_before(records):
    handler = UnifiedReveniumCallbackHandler()
    model = StandIn(callbacks=[handler])

    async def answered_async():
        for _ in range(5):
            await model.ainvoke("hi")

    asyncio.run(answered_async())
    for _ in range(5):
        model.invoke("hi")
    assert [(record["input_token_count"], record["output_token_count"]) for record in records()] == [
        (INPUT_TOKENS, OUTPUT_TOKENS)] * 10
    assert _active_runs(handler) == 0

    with pytest.raises(RuntimeError):
        model.invoke(FAILING_PROMPT)
    with pytest.raises(RuntimeError):
        asyncio.run(model.ainvoke(FAILING_PROMPT))
    assert len(records()) == 10
    assert _active_runs(handler) == 0

    with pytest.raises(RuntimeError):
        asyncio.run(model.agenerate(_batch("one", FAILING_PROMPT, "three")))
    assert len(records()) == 12
    assert _active_runs(handler) == 0


def test_an_answered_call_cancelled_while_its_end_hook_is_pending_is_still_recorded(records):
    handler = UnifiedReveniumCallbackHandler()
    slow = SlowUserHandler()
    model = StandIn(callbacks=[slow, handler])

    async def main():
        for _ in range(20):
            await _cancelled_while_its_end_hook_runs(model.ainvoke("hi"), slow)
        await asyncio.sleep(0.3)

    asyncio.run(main())

    assert len(records()) == 20
    assert _active_runs(handler) == 0


def test_cancelling_one_call_leaves_a_concurrent_call_on_the_same_handler_recorded(records):
    handler = UnifiedReveniumCallbackHandler()
    slow = StandIn(callbacks=[handler], delay=0.05)
    hanging = Hanging(callbacks=[handler])

    async def main():
        survivor = asyncio.create_task(slow.ainvoke("survivor"))
        for _ in range(10):
            await _cancelled(hanging.ainvoke("victim"), timeout=0.01)
        await survivor

    asyncio.run(main())

    assert len(records()) == 1


def test_a_failing_nested_model_call_does_not_forget_the_outer_call(records):
    handler = UnifiedReveniumCallbackHandler()
    inner = StandIn(callbacks=[handler])

    class Outer(StandIn):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            with pytest.raises(RuntimeError):
                await inner.ainvoke(FAILING_PROMPT)
            return _answer()

    asyncio.run(Outer(callbacks=[handler]).ainvoke("outer"))

    assert len(records()) == 1
    assert _active_runs(handler) == 0


def test_the_abandoned_run_memory_is_bounded():
    handler = UnifiedReveniumCallbackHandler()
    model = Hanging(callbacks=[handler])
    kept = []

    async def main():
        for call in range(300):
            await _cancelled(model.ainvoke(f"prompt {call}"))
            kept.append(len(handler._abandoned_runs_with_no_end_on_its_way))

    asyncio.run(main())

    assert max(kept) == 256
    assert kept[-1] == 256
    assert all(run_info["prompts"] is None for run_info in handler._abandoned_runs_with_no_end_on_its_way.values())
    assert handler._runs_awaiting_their_end == {}
    assert _active_runs(handler) == 0


def test_answered_calls_whose_end_hooks_outlast_many_cancellations_are_all_recorded(records):
    """More answered calls with a pending end hook than the abandoned-run
    limit, then as many never-answered calls, lose no record (BACK-4379)."""
    handler = UnifiedReveniumCallbackHandler()
    holding = HoldingUserHandler()
    answered = StandIn(callbacks=[holding, handler])
    hanging = Hanging(callbacks=[handler])
    answered_calls = 300

    async def main():
        holding.released = asyncio.Event()
        for call in range(answered_calls):
            await _cancelled_while_its_end_hook_runs(answered.ainvoke(f"answered {call}"), holding)
        for call in range(300):
            await _cancelled(hanging.ainvoke(f"never answered {call}"))
        holding.released.set()
        await asyncio.sleep(0.3)

    asyncio.run(main())

    assert len(records()) == answered_calls
    assert handler._runs_awaiting_their_end == {}
    assert _abandoned_runs._runs_with_an_end_on_its_way == set()
    assert _active_runs(handler) == 0
