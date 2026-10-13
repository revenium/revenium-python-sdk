"""``@revenium_meter`` scope survives nesting (BACK-3991), and decorator
metadata layers across nesting and reaches the record (BACK-4013).

An inner decorated function used to clear the "inside a metered function"
flag when it returned, so the outer function's remaining calls were treated
as undecorated: skipped under selective metering, or failing on the
``usage_metadata`` keyword.

Both decorators also scope the body of a decorated generator or async generator
function, one step at a time, so a streaming endpoint's calls are tagged and
metered while the caller handling each item stays outside the scope (BACK-4228).
"""
import asyncio
import contextvars
import functools
import inspect
import types

import pytest

import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches
from revenium_middleware._core.context import (
    get_function_metadata, get_injected_metadata, injected_metadata_scope, is_inside_decorated_function,
    merge_metadata, set_injected_metadata,
)
from revenium_middleware._core.decorators import revenium_meter, revenium_metadata
from entry_points import calls


def _scope():
    metadata = get_function_metadata()
    return is_inside_decorated_function(), metadata["function_name"] if metadata else None


def test_nested_revenium_meter_restores_the_outer_scope():
    seen = {}

    @revenium_meter()
    def inner():
        seen["inner"] = _scope()

    @revenium_meter()
    def outer():
        inner()
        seen["outer_after_inner"] = _scope()

    outer()

    assert seen["inner"] == (True, "inner")
    assert seen["outer_after_inner"] == (True, "outer")
    assert _scope() == (False, None)


def test_nested_async_revenium_meter_restores_the_outer_scope():
    seen = {}

    @revenium_meter()
    async def inner():
        seen["inner"] = _scope()

    @revenium_meter()
    async def outer():
        await inner()
        seen["outer_after_inner"] = _scope()

    asyncio.run(outer())

    assert seen["inner"] == (True, "inner")
    assert seen["outer_after_inner"] == (True, "outer")
    assert _scope() == (False, None)


def test_outer_scope_survives_an_inner_function_that_raises():
    seen = {}

    @revenium_meter()
    def inner():
        raise ValueError("inner failed")

    @revenium_meter()
    def outer():
        with pytest.raises(ValueError):
            inner()
        seen["outer_after_inner"] = _scope()

    outer()

    assert seen["outer_after_inner"] == (True, "outer")
    assert _scope() == (False, None)


def test_a_task_started_after_the_inner_returns_is_still_in_the_outer_scope():
    seen = {}

    @revenium_meter()
    async def inner():
        return None

    async def child():
        seen["task"] = _scope()

    @revenium_meter()
    async def outer():
        await inner()
        await asyncio.create_task(child())

    asyncio.run(outer())

    assert seen["task"] == (True, "outer")


def test_the_outermost_return_resets_the_scope():
    @revenium_meter()
    def inner():
        return None

    @revenium_meter()
    def outer():
        inner()

    outer()

    assert is_inside_decorated_function() is False
    assert get_function_metadata() is None


@pytest.fixture
def records(mock_revenium_client, monkeypatch):
    monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)

    def recorded():
        calls.wait_for_metering()
        return [entry.kwargs for entry in mock_revenium_client.ai.create_completion.call_args_list]

    return recorded


def _chat():
    calls._openai().chat.completions.create(**calls._chat_kwargs())
    calls.wait_for_metering()


async def _async_chat():
    await calls._async_openai().chat.completions.create(**calls._chat_kwargs())
    calls.wait_for_metering()


def _tags(record):
    return record.get("organization_name"), record.get("trace_id")


def test_nested_revenium_metadata_keeps_outer_tags_inside_and_after_the_inner_call(records):
    @revenium_metadata(trace_id="inner-trace")
    def inner():
        _chat()

    @revenium_metadata(organizationName="AcmeCorp", environment="production")
    def outer():
        _chat()
        inner()
        _chat()

    outer()
    _chat()

    assert [_tags(record) for record in records()] == [
        ("AcmeCorp", None), ("AcmeCorp", "inner-trace"), ("AcmeCorp", None), (None, None),
    ]
    assert get_injected_metadata() is None


def test_nested_async_revenium_metadata_keeps_outer_tags(records):
    @revenium_metadata(trace_id="inner-trace")
    async def inner():
        await _async_chat()

    @revenium_metadata(organizationName="AcmeCorp", environment="production")
    async def outer():
        await _async_chat()
        await inner()
        await _async_chat()

    asyncio.run(outer())

    assert [_tags(record) for record in records()] == [
        ("AcmeCorp", None), ("AcmeCorp", "inner-trace"), ("AcmeCorp", None),
    ]
    assert get_injected_metadata() is None


def test_outer_tags_survive_an_inner_function_that_raises(records):
    @revenium_metadata(trace_id="inner-trace")
    def inner():
        raise ValueError("inner failed")

    @revenium_metadata(organizationName="AcmeCorp")
    def outer():
        with pytest.raises(ValueError):
            inner()
        _chat()

    outer()

    assert [_tags(record) for record in records()] == [("AcmeCorp", None)]


def test_concurrent_tasks_each_record_only_their_own_tags(records):
    @revenium_metadata(organizationName="OrgA")
    async def task_a():
        await asyncio.sleep(0)
        await _async_chat()

    @revenium_metadata(organizationName="OrgB")
    async def task_b():
        await asyncio.sleep(0)
        await _async_chat()

    async def both():
        await asyncio.gather(task_a(), task_b())

    asyncio.run(both())

    assert sorted(record.get("organization_name") for record in records()) == ["OrgA", "OrgB"]


@pytest.mark.parametrize("selective", ["true", "false"])
def test_revenium_meter_metadata_reaches_the_record(records, monkeypatch, selective):
    seen = {}

    @revenium_meter(metadata={"task_type": "chat_analysis", "organizationName": "MeterOrg"})
    def analyze():
        seen["merged"] = merge_metadata()
        _chat()

    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", selective)
    analyze()

    [record] = records()
    assert record.get("task_type") == "chat_analysis"
    assert record.get("organization_name") == "MeterOrg"
    assert "function_name" not in seen["merged"]
    assert "is_decorated" not in seen["merged"]


def test_operation_type_is_ignored_with_one_warning_per_process(records, monkeypatch, caplog):
    from revenium_middleware._core import decorators

    monkeypatch.setattr(decorators, "_operation_type_warned", False, raising=False)

    with caplog.at_level("WARNING", logger="revenium_middleware"):
        @revenium_meter(operation_type="EMBED")
        def first():
            seen["first"] = get_function_metadata()
            _chat()

        @revenium_meter(operation_type="EMBED")
        def second():
            _chat()

        seen = {}
        first()
        second()

    warnings = [record for record in caplog.records if "operation_type" in record.getMessage()]
    assert len(warnings) == 1
    assert "EMBED" not in warnings[0].getMessage()
    assert "operation_type" not in seen["first"]
    assert [record.get("operation_type") for record in records()] == ["CHAT", "CHAT"]


def _decorated_chat_generator(decorator, kind, streamed):
    if kind == "sync":
        @decorator
        def chat_generator():
            client = calls._openai()
            if streamed:
                yield from client.chat.completions.create(**calls._chat_kwargs(), stream=True)
            else:
                yield client.chat.completions.create(**calls._chat_kwargs())
        return chat_generator

    @decorator
    async def chat_async_generator():
        client = calls._async_openai()
        if streamed:
            async for chunk in await client.chat.completions.create(**calls._chat_kwargs(), stream=True):
                yield chunk
        else:
            yield await client.chat.completions.create(**calls._chat_kwargs())
    return chat_async_generator


def _consume(generator_function, kind):
    if kind == "sync":
        return list(generator_function())

    async def drain():
        return [item async for item in generator_function()]
    return asyncio.run(drain())


STREAMED = pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "not_streamed"])
KIND = pytest.mark.parametrize("kind", ["sync", "async"])


@STREAMED
def test_revenium_metadata_tags_calls_inside_a_sync_generator(records, streamed):
    generator_function = _decorated_chat_generator(revenium_metadata(organizationName="GenOrg"), "sync", streamed)

    assert _consume(generator_function, "sync")
    assert [_tags(record) for record in records()] == [("GenOrg", None)]
    assert get_injected_metadata() is None


@STREAMED
def test_revenium_metadata_tags_calls_inside_an_async_generator(records, streamed):
    generator_function = _decorated_chat_generator(revenium_metadata(organizationName="GenOrg"), "async", streamed)

    assert _consume(generator_function, "async")
    assert [_tags(record) for record in records()] == [("GenOrg", None)]
    assert get_injected_metadata() is None


@KIND
@STREAMED
def test_revenium_meter_meters_calls_inside_a_generator_under_selective_metering(
        records, monkeypatch, kind, streamed):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    generator_function = _decorated_chat_generator(revenium_meter(metadata={"task_type": "gen_task"}), kind, streamed)

    _consume(generator_function, kind)

    assert [record.get("task_type") for record in records()] == ["gen_task"]


@pytest.mark.parametrize("meter_outermost", [True, False], ids=["meter_outer", "metadata_outer"])
def test_stacked_meter_and_metadata_on_a_generator_meter_and_tag_the_call(records, monkeypatch, meter_outermost):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    meter, metadata = revenium_meter(metadata={"task_type": "gen_task"}), revenium_metadata(organizationName="StackOrg")
    outer, inner = (meter, metadata) if meter_outermost else (metadata, meter)

    _consume(_decorated_chat_generator(lambda func: outer(inner(func)), "sync", streamed=True), "sync")

    assert [(record.get("task_type"), *_tags(record)) for record in records()] == [("gen_task", "StackOrg", None)]


def test_decorated_generators_stay_generator_functions():
    def generator():
        yield 1

    async def async_generator():
        yield 1

    def plain():
        return 1

    for decorator in (revenium_meter(), revenium_metadata(organizationName="GenOrg")):
        assert inspect.isgeneratorfunction(decorator(generator))
        assert inspect.isasyncgenfunction(decorator(async_generator))
        decorated_plain = decorator(plain)
        assert not inspect.isgeneratorfunction(decorated_plain)
        assert not inspect.isasyncgenfunction(decorated_plain)


@types.coroutine
def _legacy_coroutine(value):
    yield
    return value


@pytest.mark.parametrize("wrap", [
    lambda func: func,
    lambda func: functools.partial(func),
    lambda func: functools.partial(functools.partial(func)),
], ids=["function", "partial", "nested-partial"])
def test_a_decorated_legacy_coroutine_stays_awaitable(wrap):
    decorated = revenium_metadata(trace_id="T")(wrap(_legacy_coroutine))

    async def drive():
        return await decorated("value")

    assert asyncio.run(drive()) == "value"


def _observed_scope():
    return get_injected_metadata(), is_inside_decorated_function()


def _caller_scope_is_clear():
    return get_injected_metadata() is None and is_inside_decorated_function() is False


def _scope_recording_generator(kind, seen):
    decorator = revenium_meter(metadata={"organizationName": "GenOrg"})
    if kind == "sync":
        @decorator
        def steps():
            for item in range(3):
                seen.append(_observed_scope())
                yield item
        return steps

    @decorator
    async def async_steps():
        for item in range(3):
            seen.append(_observed_scope())
            yield item
    return async_steps


TAGGED = ({"organizationName": "GenOrg"}, True)
TAGGED_WITH_CALLER_TRACE = ({"organizationName": "GenOrg", "trace_id": "T2"}, True)


@KIND
def test_a_generator_scope_does_not_leak_to_the_caller_between_items(kind):
    seen, caller = [], []
    steps = _scope_recording_generator(kind, seen)

    if kind == "sync":
        generator = steps()
        next(generator)
        caller.append(_caller_scope_is_clear())
        with injected_metadata_scope({"trace_id": "T2"}):
            next(generator)
        caller.append(_caller_scope_is_clear())
        next(generator)
        caller.append(_caller_scope_is_clear())
        generator.close()
    else:
        async def drive():
            generator = steps()
            await generator.__anext__()
            caller.append(_caller_scope_is_clear())
            with injected_metadata_scope({"trace_id": "T2"}):
                await generator.__anext__()
            caller.append(_caller_scope_is_clear())
            await generator.__anext__()
            caller.append(_caller_scope_is_clear())
            await generator.aclose()
        asyncio.run(drive())

    assert seen == [TAGGED, TAGGED_WITH_CALLER_TRACE, TAGGED]
    assert caller == [True, True, True]


@KIND
def test_each_step_in_a_fresh_context_keeps_the_scope(kind):
    seen, caller = [], []
    steps = _scope_recording_generator(kind, seen)

    if kind == "sync":
        generator = steps()
        for _ in range(2):
            contextvars.copy_context().run(next, generator)
            caller.append(_caller_scope_is_clear())
        contextvars.copy_context().run(generator.close)
    else:
        async def drive():
            generator = steps()

            async def step():
                return await generator.__anext__()

            async def close():
                await generator.aclose()

            for _ in range(2):
                await asyncio.create_task(step())
                caller.append(_caller_scope_is_clear())
            await asyncio.create_task(close())
        asyncio.run(drive())

    assert seen == [TAGGED, TAGGED]
    assert caller == [True, True]
    assert _caller_scope_is_clear()


def test_close_send_and_throw_reach_the_inner_generator():
    seen = []

    def organization():
        return (get_injected_metadata() or {}).get("organizationName")

    @revenium_metadata(organizationName="GenOrg")
    def conversation():
        try:
            received = yield "first"
            seen.append(("sent", received, organization()))
            try:
                yield "second"
            except KeyError:
                seen.append(("thrown", organization()))
            yield "third"
            yield "unreached"
        finally:
            seen.append(("finally", organization()))

    generator = conversation()
    assert next(generator) == "first"
    assert generator.send("hello") == "second"
    assert generator.throw(KeyError("boom")) == "third"
    generator.close()

    assert seen == [("sent", "hello", "GenOrg"), ("thrown", "GenOrg"), ("finally", "GenOrg")]
    assert get_injected_metadata() is None

    @revenium_metadata(organizationName="GenOrg")
    def returns_a_value():
        yield organization()
        return "done"

    returning = returns_a_value()
    assert next(returning) == "GenOrg"
    with pytest.raises(StopIteration) as stop:
        next(returning)
    assert stop.value.value == "done"


def _inner_tag_generator(kind, seen):
    decorator = revenium_meter(metadata={"organizationName": "O"})
    if kind == "sync":
        @decorator
        def steps():
            with injected_metadata_scope({"trace_id": "inner"}):
                seen.append(get_injected_metadata())
                yield 1
                seen.append(get_injected_metadata())
                yield 2
            seen.append(get_injected_metadata())
            yield 3
        return steps

    @decorator
    async def async_steps():
        with injected_metadata_scope({"trace_id": "inner"}):
            seen.append(get_injected_metadata())
            yield 1
            seen.append(get_injected_metadata())
            yield 2
        seen.append(get_injected_metadata())
        yield 3
    return async_steps


@KIND
def test_a_tag_the_body_sets_persists_across_yields_without_leaking(kind):
    seen, caller = [], []
    steps = _inner_tag_generator(kind, seen)

    if kind == "sync":
        generator = steps()
        for _ in range(3):
            next(generator)
            caller.append(_caller_scope_is_clear())
        generator.close()
    else:
        async def drive():
            generator = steps()
            for _ in range(3):
                await generator.__anext__()
                caller.append(_caller_scope_is_clear())
            await generator.aclose()
        asyncio.run(drive())

    tagged = {"organizationName": "O", "trace_id": "inner"}
    assert seen == [tagged, tagged, {"organizationName": "O"}]
    assert caller == [True, True, True]


def _inner_scope_generator(kind, seen):
    decorator = revenium_meter(metadata={"organizationName": "O"})

    def trace():
        return (get_injected_metadata() or {}).get("trace_id")

    if kind == "sync":
        @decorator
        def steps():
            with injected_metadata_scope({"trace_id": "inner"}):
                yield 1
                seen.append(trace())
                yield 2
            yield 3
            seen.append(trace())
            yield 4
        return steps

    @decorator
    async def async_steps():
        with injected_metadata_scope({"trace_id": "inner"}):
            yield 1
            seen.append(trace())
            yield 2
        yield 3
        seen.append(trace())
        yield 4
    return async_steps


@KIND
def test_a_restored_caller_trace_is_not_carried_after_an_inner_scope_exits(kind):
    seen = []
    steps = _inner_scope_generator(kind, seen)
    callers = ["A", "B", "B", "B"]

    if kind == "sync":
        generator = steps()
        for trace_id in callers:
            with injected_metadata_scope({"trace_id": trace_id}):
                next(generator)
        generator.close()
    else:
        async def drive():
            generator = steps()
            for trace_id in callers:
                with injected_metadata_scope({"trace_id": trace_id}):
                    await generator.__anext__()
            await generator.aclose()
        asyncio.run(drive())

    assert seen == ["inner", "B"]


def _drive(steps, kind, callers):
    """Advance ``steps()`` once under each caller's metadata, then close it;
    return the metadata each caller saw right after its step."""
    seen_by_caller = []
    if kind == "sync":
        generator = steps()
        for caller in callers:
            with injected_metadata_scope(caller):
                next(generator)
                seen_by_caller.append(get_injected_metadata())
        generator.close()
        return seen_by_caller

    async def drive():
        generator = steps()
        for caller in callers:
            with injected_metadata_scope(caller):
                await generator.__anext__()
                seen_by_caller.append(get_injected_metadata())
        await generator.aclose()
    asyncio.run(drive())
    return seen_by_caller


def _trace():
    return (get_injected_metadata() or {}).get("trace_id")


def _reentered_trace_generator(kind):
    decorator = revenium_metadata(organizationName="O")
    if kind == "sync":
        @decorator
        def steps():
            yield 1
            with injected_metadata_scope({"trace_id": "A"}):
                yield 2
                _chat()
                yield 3
        return steps

    @decorator
    async def async_steps():
        yield 1
        with injected_metadata_scope({"trace_id": "A"}):
            yield 2
            await _async_chat()
            yield 3
    return async_steps


@KIND
def test_a_body_tag_equal_to_an_earlier_caller_value_is_still_carried(kind, records):
    traces = [{"trace_id": "A"}, {"trace_id": "B"}, {"trace_id": "B"}]

    seen_by_caller = _drive(_reentered_trace_generator(kind), kind, traces)

    assert [_tags(record) for record in records()] == [("O", "A")]
    assert seen_by_caller == traces


def _nested_scopes_generator(kind):
    decorator = revenium_metadata(organizationName="O")
    if kind == "sync":
        @decorator
        def steps():
            with injected_metadata_scope({"trace_id": "X"}):
                yield 1
                with injected_metadata_scope({"trace_id": "Y"}):
                    yield 2
                _chat()
                yield 3
        return steps

    @decorator
    async def async_steps():
        with injected_metadata_scope({"trace_id": "X"}):
            yield 1
            with injected_metadata_scope({"trace_id": "Y"}):
                yield 2
            await _async_chat()
            yield 3
    return async_steps


@KIND
def test_leaving_a_nested_body_scope_in_a_later_step_restores_the_outer_body_scope(kind, records):
    seen_by_caller = _drive(_nested_scopes_generator(kind), kind, [None, None, None])

    assert [_tags(record) for record in records()] == [("O", "X")]
    assert seen_by_caller == [None, None, None]


def _scopes_adding_different_keys_generator(kind):
    decorator = revenium_metadata(organizationName="O")
    if kind == "sync":
        @decorator
        def steps():
            with injected_metadata_scope({"trace_id": "X"}):
                yield 1
                with injected_metadata_scope({"task_type": "Y"}):
                    yield 2
                    _chat()
                    yield 3
        return steps

    @decorator
    async def async_steps():
        with injected_metadata_scope({"trace_id": "X"}):
            yield 1
            with injected_metadata_scope({"task_type": "Y"}):
                yield 2
                await _async_chat()
                yield 3
    return async_steps


@KIND
def test_a_later_body_scope_adding_another_key_keeps_the_earlier_body_tag(kind, records):
    seen_by_caller = _drive(_scopes_adding_different_keys_generator(kind), kind, [None, None, None])

    assert [(*_tags(record), record.get("task_type")) for record in records()] == [("O", "X", "Y")]
    assert seen_by_caller == [None, None, None]


def _set_metadata_generator(kind):
    decorator = revenium_metadata(organizationName="O")
    if kind == "sync":
        @decorator
        def steps():
            set_injected_metadata({"trace_id": "S"})
            yield 1
            _chat()
            yield 2
        return steps

    @decorator
    async def async_steps():
        set_injected_metadata({"trace_id": "S"})
        yield 1
        await _async_chat()
        yield 2
    return async_steps


@KIND
def test_metadata_the_body_sets_directly_persists_without_reaching_the_caller(kind, records):
    traces = [{"trace_id": "C1"}, {"trace_id": "C2"}]

    seen_by_caller = _drive(_set_metadata_generator(kind), kind, traces)

    assert [_tags(record) for record in records()] == [("O", "S")]
    assert seen_by_caller == traces


@KIND
def test_a_body_without_any_metadata_sees_none(kind):
    seen = []
    if kind == "sync":
        @revenium_meter()
        def steps():
            while True:
                seen.append(get_injected_metadata())
                yield
    else:
        @revenium_meter()
        async def steps():
            while True:
                seen.append(get_injected_metadata())
                yield

    seen_by_caller = _drive(steps, kind, [None, None, None])

    assert seen == [None, None, None]
    assert seen_by_caller == [None, None, None]


@KIND
def test_a_long_stream_keeps_the_body_tag_under_changing_callers(kind):
    seen = set()
    if kind == "sync":
        @revenium_metadata(organizationName="O")
        def steps():
            with injected_metadata_scope({"trace_id": "body"}):
                while True:
                    seen.add(_trace())
                    yield
    else:
        @revenium_metadata(organizationName="O")
        async def steps():
            with injected_metadata_scope({"trace_id": "body"}):
                while True:
                    seen.add(_trace())
                    yield
    traces = [{"trace_id": f"caller-{index}"} for index in range(5000)]

    seen_by_caller = _drive(steps, kind, traces)

    assert seen == {"body"}
    assert seen_by_caller == traces


@KIND
def test_a_tag_the_body_sets_is_active_when_the_generator_is_closed(kind):
    seen = []

    if kind == "sync":
        @revenium_metadata(organizationName="O")
        def steps():
            with injected_metadata_scope({"trace_id": "inner"}):
                try:
                    yield 1
                finally:
                    seen.append(get_injected_metadata())

        generator = steps()
        next(generator)
        generator.close()
    else:
        @revenium_metadata(organizationName="O")
        async def steps():
            with injected_metadata_scope({"trace_id": "inner"}):
                try:
                    yield 1
                finally:
                    seen.append(get_injected_metadata())

        async def drive():
            generator = steps()
            await generator.__anext__()
            await generator.aclose()
        asyncio.run(drive())

    assert seen == [{"organizationName": "O", "trace_id": "inner"}]
    assert _caller_scope_is_clear()


def test_close_send_and_throw_reach_the_inner_async_generator():
    seen = []

    def organization():
        return (get_injected_metadata() or {}).get("organizationName")

    @revenium_metadata(organizationName="GenOrg")
    async def conversation():
        try:
            received = yield "first"
            seen.append(("sent", received, organization()))
            try:
                yield "second"
            except KeyError:
                seen.append(("thrown", organization()))
            yield "third"
            yield "fourth"
            yield "unreached"
        finally:
            seen.append(("finally", organization()))

    async def drive():
        generator = conversation()
        assert await generator.__anext__() == "first"
        assert await generator.asend("hello") == "second"
        assert await generator.athrow(KeyError("boom")) == "third"
        with pytest.raises(ValueError):
            await generator.athrow(ValueError("uncaught"))
        await generator.aclose()

        closing = conversation()
        await closing.__anext__()
        await closing.aclose()

    asyncio.run(drive())

    assert seen == [
        ("sent", "hello", "GenOrg"), ("thrown", "GenOrg"), ("finally", "GenOrg"), ("finally", "GenOrg"),
    ]
    assert get_injected_metadata() is None


def test_async_send_throw_and_close_see_a_tag_the_body_sets():
    seen = []

    @revenium_metadata(organizationName="O")
    async def conversation():
        with injected_metadata_scope({"trace_id": "inner"}):
            try:
                received = yield "first"
                seen.append(("sent", received, _trace()))
                try:
                    yield "second"
                except KeyError:
                    seen.append(("caught", _trace()))
                yield "third"
            finally:
                seen.append(("finally", _trace()))

    async def drive():
        generator = conversation()
        await generator.__anext__()
        await generator.asend("hello")
        await generator.athrow(KeyError("boom"))
        with pytest.raises(ValueError):
            await generator.athrow(ValueError("uncaught"))

        closing = conversation()
        await closing.__anext__()
        await closing.aclose()

    asyncio.run(drive())

    assert seen == [("sent", "hello", "inner"), ("caught", "inner"), ("finally", "inner"), ("finally", "inner")]
    assert _caller_scope_is_clear()


DECORATOR_TAGS = {
    "organization_name": "acme", "product_name": "chatbot", "agent": "support-agent", "subscriber": {"id": "u-1"},
}
BLANK = pytest.mark.parametrize("blank", [None, ""], ids=["none", "empty"])


def _chat_inside_decorator(usage_metadata, decorator_tags=DECORATOR_TAGS):
    @revenium_metadata(**decorator_tags)
    def handle():
        calls._openai().chat.completions.create(**calls._chat_kwargs(), usage_metadata=usage_metadata)

    handle()


@BLANK
@pytest.mark.parametrize("field", list(DECORATOR_TAGS))
def test_an_empty_per_call_field_keeps_the_decorators_value(records, field, blank):
    """BACK-4322: a None or "" per-call value never erases what the decorator set."""
    _chat_inside_decorator({"trace_id": "t-1", field: blank})

    [record] = records()
    assert record.get(field) == DECORATOR_TAGS[field]
    assert record.get("trace_id") == "t-1"


@BLANK
def test_an_empty_per_call_field_in_the_other_spelling_keeps_the_decorators_value(records, blank):
    _chat_inside_decorator({"trace_id": "t-1", "organizationName": blank})

    [record] = records()
    assert record.get("organization_name") == "acme"


@BLANK
def test_an_empty_per_call_name_keeps_a_decorator_value_set_under_the_id_name(records, blank):
    _chat_inside_decorator({"organization_name": blank}, decorator_tags={"organization_id": "acme"})

    [record] = records()
    assert record.get("organization_name") == "acme"


@BLANK
def test_an_empty_per_call_name_beside_a_per_call_id_records_the_per_call_id(records, blank):
    from revenium_middleware._core.fields import extract_organization_name, extract_product_name

    _chat_inside_decorator({
        "organization_name": blank, "organization_id": "other",
        "product_name": blank, "product_id": "other-product",
    })

    [record] = records()
    assert (extract_organization_name(record), extract_product_name(record)) == ("other", "other-product")


@pytest.mark.parametrize("override", [
    {"organization_name": None, "organization_id": "other"},
    {"organization_id": "other"},
], ids=["beside-a-blank-name", "alone"])
def test_a_per_call_id_beats_a_name_set_underneath(override):
    from revenium_middleware._core.context import overlay_metadata
    from revenium_middleware._core.fields import extract_organization_name

    assert extract_organization_name(overlay_metadata({"organization_name": "acme"}, override)) == "other"


def test_a_real_per_call_value_still_overrides_the_decorator(records):
    _chat_inside_decorator({"organization_name": "per-call", "agent": "per-call-agent"})

    [record] = records()
    assert (record.get("organization_name"), record.get("agent")) == ("per-call", "per-call-agent")


def test_a_blank_per_call_value_with_nothing_underneath_passes_through_unchanged():
    from revenium_middleware._core.context import overlay_metadata

    assert overlay_metadata({"trace_id": "t"}, {"organization_name": None}) == {
        "trace_id": "t", "organization_name": None,
    }


def test_zero_false_and_empty_containers_are_real_per_call_values():
    from revenium_middleware._core.context import overlay_metadata

    base = {"response_quality_score": 0.9, "is_test": True, "subscriber": {"id": "u-1"}, "tags": ["a"]}
    override = {"response_quality_score": 0, "is_test": False, "subscriber": {}, "tags": []}

    assert overlay_metadata(base, override) == override


def _wraps_third_party(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def _bare_third_party(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


THIRD_PARTY = pytest.mark.parametrize(
    "third_party", [_wraps_third_party, _bare_third_party], ids=["functools_wraps", "bare_closure"],
)
SELECTIVE = pytest.mark.parametrize("selective", ["true", "false"])


def _meter_and_tag(func):
    return revenium_meter(metadata={"task_type": "chat"})(revenium_metadata(organization_name="AcmeCorp")(func))


def _chat_steps():
    yield calls._openai().chat.completions.create(**calls._chat_kwargs())


async def _async_chat_steps():
    yield await calls._async_openai().chat.completions.create(**calls._chat_kwargs())


def _meter_and_tag_records(records):
    return [(record.get("task_type"), record.get("organization_name")) for record in records()]


@SELECTIVE
@THIRD_PARTY
def test_a_generator_behind_a_third_party_decorator_is_metered_and_tagged(records, monkeypatch, selective, third_party):
    """BACK-4325: a retry, tracing or caching decorator under ours keeps the stream's calls in scope."""
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", selective)

    @_meter_and_tag
    @third_party
    def stream():
        yield from _chat_steps()

    assert len(list(stream())) == 1
    assert _meter_and_tag_records(records) == [("chat", "AcmeCorp")]


@SELECTIVE
def test_a_function_returning_a_generator_is_metered_and_tagged(records, monkeypatch, selective):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", selective)

    @_meter_and_tag
    def handler():
        return _chat_steps()

    assert len(list(handler())) == 1
    assert _meter_and_tag_records(records) == [("chat", "AcmeCorp")]


@SELECTIVE
def test_an_async_function_returning_a_generator_is_metered_and_tagged(records, monkeypatch, selective):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", selective)

    @_meter_and_tag
    async def handler():
        return _chat_steps()

    assert len(list(asyncio.run(handler()))) == 1
    assert _meter_and_tag_records(records) == [("chat", "AcmeCorp")]


@SELECTIVE
@pytest.mark.parametrize("handler_kind", ["plain", "async"])
def test_a_function_returning_an_async_generator_is_metered_and_tagged(records, monkeypatch, selective, handler_kind):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", selective)
    if handler_kind == "plain":
        @_meter_and_tag
        def handler():
            return _async_chat_steps()
    else:
        @_meter_and_tag
        async def handler():
            return _async_chat_steps()

    async def drain():
        generator = handler() if handler_kind == "plain" else await handler()
        return [item async for item in generator]

    assert len(asyncio.run(drain())) == 1
    assert _meter_and_tag_records(records) == [("chat", "AcmeCorp")]


def test_the_caller_is_outside_the_scope_between_items_of_a_returned_generator():
    seen, caller = [], []

    def steps():
        for item in range(3):
            seen.append(_observed_scope())
            yield item

    @revenium_meter(metadata={"organizationName": "GenOrg"})
    def handler():
        return steps()

    for _ in handler():
        caller.append(_caller_scope_is_clear())

    assert seen == [TAGGED] * 3
    assert caller == [True] * 3
    assert _caller_scope_is_clear()


@pytest.mark.parametrize("decorator", [revenium_meter(), revenium_metadata(organizationName="O")], ids=["meter", "metadata"])
def test_a_non_generator_result_is_returned_unchanged(decorator):
    sentinel = object()

    @decorator
    def plain():
        return sentinel

    @decorator
    async def coroutine():
        return sentinel

    assert plain() is sentinel
    assert asyncio.run(coroutine()) is sentinel


@KIND
def test_a_returned_generator_that_was_already_started_is_returned_unchanged(kind):
    started = {}

    if kind == "sync":
        def consumer():
            received = yield "ready"
            yield received

        @revenium_metadata(organizationName="O")
        def primed():
            generator = consumer()
            next(generator)
            started["generator"] = generator
            return generator

        result = primed()
        assert result is started["generator"]
        assert result.send(5) == 5
        return

    async def async_consumer():
        received = yield "ready"
        yield received

    @revenium_metadata(organizationName="O")
    async def async_primed():
        generator = async_consumer()
        await generator.__anext__()
        started["generator"] = generator
        return generator

    async def drive():
        result = await async_primed()
        assert result is started["generator"]
        return await result.asend(5)

    assert asyncio.run(drive()) == 5


def test_a_returned_legacy_coroutine_stays_awaitable():
    @revenium_metadata(trace_id="T")
    def handler():
        return _legacy_coroutine("value")

    async def drive():
        return await handler()

    assert asyncio.run(drive()) == "value"
