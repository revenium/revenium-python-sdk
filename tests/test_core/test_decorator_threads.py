"""AI calls a decorated function hands to worker threads keep its metering
mark and tags when the thread's target is wrapped with ``carry_context``
(BACK-4361).

``ThreadPoolExecutor``, ``threading.Thread`` and ``loop.run_in_executor`` start
their target in an empty context, so a bare worker's calls were skipped under
selective metering and recorded without the decorator's tags with it off.
"""
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import revenium_middleware.anthropic  # noqa: F401  installs the Anthropic patches
import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches
from revenium_middleware import carry_context
from revenium_middleware._core import selective_metering
from revenium_middleware._core.context import get_injected_metadata, idempotency_key, set_injected_metadata
from revenium_middleware.litellm.client import middleware as litellm_middleware
from revenium_middleware._core.decorators import revenium_meter, revenium_metadata
from entry_points import calls

TAGS = ("PremiumTier", "premium-feature", "Pro")
NO_TAGS = (None, None, None)


def _openai_chat(_=None):
    calls._openai().chat.completions.create(**calls._chat_kwargs())


def _anthropic_message(_=None):
    calls._anthropic().messages.create(**calls._ANTHROPIC_KWARGS)


PROVIDERS = {"openai": _openai_chat, "anthropic": _anthropic_message}


@pytest.fixture
def records(mock_revenium_client):
    def recorded():
        calls.wait_for_metering()
        return [entry.kwargs for entry in mock_revenium_client.ai.create_completion.call_args_list]

    return recorded


@pytest.fixture
def selective(monkeypatch):
    def set_selective(enabled):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true" if enabled else "false")

    return set_selective


@pytest.fixture
def skip_lines(monkeypatch, caplog):
    monkeypatch.setattr(selective_metering, "_unmetered_call_reported", False)
    caplog.set_level("INFO", logger="revenium_middleware")

    def lines():
        return [record for record in caplog.records if "carry_context" in record.getMessage()]

    return lines


def _tags(record):
    return record.get("organization_name"), record.get("task_type"), record.get("product_name")


def _pool_map(call):
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(carry_context(call), range(2)))


def _pool_submit(call):
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(carry_context(call), index) for index in range(2)]:
            future.result()


def _threads(call):
    threads = [threading.Thread(target=carry_context(call), args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@revenium_meter()
@revenium_metadata(task_type="premium-feature", organizationName="PremiumTier", productName="Pro")
def _fan_out(launch, call):
    launch(call)


@revenium_meter()
@revenium_metadata(task_type="premium-feature", organizationName="PremiumTier", productName="Pro")
async def _fan_out_async(call):
    loop = asyncio.get_running_loop()
    await asyncio.gather(*(loop.run_in_executor(None, carry_context(call), index) for index in range(2)))


@revenium_meter()
@revenium_metadata(task_type="premium-feature", organizationName="PremiumTier", productName="Pro")
def _fan_out_bare(call):
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(call, range(2)))


LAUNCHERS = {
    "pool.map": lambda call: _fan_out(_pool_map, call),
    "pool.submit": lambda call: _fan_out(_pool_submit, call),
    "threading.Thread": lambda call: _fan_out(_threads, call),
    "run_in_executor": lambda call: asyncio.run(_fan_out_async(call)),
}


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("selective_on", [True, False], ids=["selective-on", "selective-off"])
@pytest.mark.parametrize("provider", PROVIDERS)
def test_calls_a_decorated_function_runs_on_worker_threads_are_metered_and_tagged(
    records, selective, provider, selective_on, launcher
):
    selective(selective_on)

    LAUNCHERS[launcher](PROVIDERS[provider])

    assert [_tags(record) for record in records()] == [TAGS, TAGS]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_skipped_call_on_a_bare_worker_thread_logs_one_line(records, selective, skip_lines, provider):
    selective(True)

    _fan_out_bare(PROVIDERS[provider])
    _fan_out_bare(PROVIDERS[provider])

    assert records() == []
    assert len(skip_lines()) == 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_no_skip_line_when_every_call_is_metered(records, selective, skip_lines, provider):
    selective(True)

    _fan_out(_pool_map, PROVIDERS[provider])

    assert len(records()) == 2
    assert skip_lines() == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_bare_worker_thread_with_selective_metering_off_is_recorded_without_tags(records, selective, provider):
    selective(False)

    _fan_out_bare(PROVIDERS[provider])

    assert [_tags(record) for record in records()] == [NO_TAGS, NO_TAGS]


def test_carry_context_outside_a_decorated_function_meters_nothing(records, selective):
    selective(True)

    _pool_map(_openai_chat)

    assert records() == []


def test_each_carried_call_runs_in_its_own_copy():
    both_started = threading.Barrier(2)
    both_tagged = threading.Barrier(2)

    def worker(name):
        both_started.wait(timeout=5)
        inherited = get_injected_metadata()
        set_injected_metadata({"organizationName": name})
        both_tagged.wait(timeout=5)
        return inherited, get_injected_metadata()

    @revenium_metadata(organizationName="Caller")
    def fan_out():
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(carry_context(worker), ["one", "two"]))
        return results, get_injected_metadata()

    results, caller_after = fan_out()

    assert results == [
        ({"organizationName": "Caller"}, {"organizationName": "one"}),
        ({"organizationName": "Caller"}, {"organizationName": "two"}),
    ]
    assert caller_after == {"organizationName": "Caller"}


def test_carried_calls_draw_distinct_idempotency_keys(records, selective):
    selective(False)

    @revenium_meter()
    def fan_out():
        with idempotency_key("blk"):
            _pool_map(_openai_chat)

    fan_out()

    assert {record["extra_headers"]["Idempotency-Key"] for record in records()} == {"blk", "blk:2"}


def test_nested_decorators_carry_both_tags(records, selective):
    selective(True)

    @revenium_meter()
    @revenium_metadata(task_type="premium-feature")
    def inner():
        _pool_map(_openai_chat)

    @revenium_metadata(organizationName="PremiumTier")
    def outer():
        inner()

    outer()

    assert [_tags(record)[:2] for record in records()] == [TAGS[:2], TAGS[:2]]


def test_simultaneous_first_skips_log_exactly_one_line(skip_lines):
    count = 16
    barrier = threading.Barrier(count)

    def first_skip():
        barrier.wait()
        selective_metering.reported_if_unmetered(True)

    threads = [threading.Thread(target=first_skip) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(skip_lines()) == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable")
def test_a_forked_child_reports_its_own_first_skip(monkeypatch):
    monkeypatch.setattr(selective_metering, "_unmetered_call_reported", True)

    pid = os.fork()
    if pid == 0:
        os._exit(0 if selective_metering._unmetered_call_reported is False else 1)
    _, status = os.waitpid(pid, 0)

    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.parametrize("proxy_meters, lines", [(True, 0), (False, 1)])
def test_litellm_skip_line_is_only_logged_when_the_proxy_does_not_meter(
    monkeypatch, selective, skip_lines, proxy_meters, lines
):
    selective(True)
    monkeypatch.setattr(litellm_middleware, "proxy_meters_request", lambda kwargs: proxy_meters)

    assert litellm_middleware._selectively_unmetered(litellm_middleware.proxy_meters_request, {}) is True
    assert len(skip_lines()) == lines
