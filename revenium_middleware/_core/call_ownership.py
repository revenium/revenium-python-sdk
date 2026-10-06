"""Exactly-once ownership of a model call's usage record.

Two layers can meter the same LangChain model call: the provider transport
wrap (the OpenAI and Anthropic client patches) and the LangChain callback.
The transport record is the richer one (the provider's own id, the dated
model, the real provider, ``is_streamed``), so the transport signals and the
callback reads:

* A transport wrap calls ``claim_call_for_transport(provider, response_id)``
  once it has dispatched a call's record, or once it has taken over a stream
  that it will meter when the stream ends. The claim bumps the counter for
  that provider in the current context and, when the provider response id is
  known, remembers that id.
* The callback takes ``transport_claim_mark()`` when the model call starts and
  asks ``claimed_by_transport(mark, providers, response_ids)`` before sending
  its own record. ``providers`` is every transport the model class can
  plausibly call through. The answer is yes when one of those providers'
  counters in the callback's context has moved past the mark (any provider's,
  when the callback could not tell which provider serves the model), or when one of
  the response ids LangChain kept on the result was claimed. Asking never
  changes the answer, so every handler attached to the same run gets the same
  one.
* A streamed model call inside ``ainvoke`` leaves no id on the result, so the
  callback also asks (without ids) from ``on_llm_new_token``, a coroutine hook
  that langchain-core awaits in the model call's own task.

The signal cannot run the other way: during ``ainvoke``/``astream``
langchain-core runs a sync handler method on a copy of the caller's context
(``run_in_executor(copy_context().run, ...)``), so nothing the callback writes
reaches the transport, while what the transport writes in the caller's task
is visible to the callback's later ``on_llm_end``.

The counters cover every call whose transport runs in that caller context:
sync calls in a thread and ``stream``/``astream``. ``agenerate`` (behind
``ainvoke``) runs the model call in a child task created by
``asyncio.gather``, so the counter moves only in the child; there the
response id, which LangChain keeps on non-streamed results, carries the
signal instead. Concurrent calls in different threads or tasks do not see
each other's counters, and their response ids differ.

The id ledger is bounded and evicts its oldest ids. That is safe because an
id only has to outlive the gap between the transport dispatching the record
and the same run's ``on_llm_end``, which langchain-core dispatches as soon as
the model call returns; the capacity is far above the number of calls a
process completes in that gap.

Known limit: a callback-only call and a transport-metered call to the same
provider, interleaved in one thread or task, make the callback-only call look
claimed, and its callback record is skipped. That needs a call the
provider's transport wrap let through unmetered (selective metering, for
example) iterated alongside one it metered. A model LangChain labels Ollama
is checked against the OpenAI transport too, since OpenAI-compatible Ollama
endpoints go through the OpenAI client, so an OpenAI call interleaved with a
callback-only Ollama call in the same task also hides the Ollama record.
Otherwise calls to different providers do not interfere, and neither do
calls in different threads or tasks.
"""
import threading
from collections import OrderedDict
from contextvars import ContextVar
from typing import AbstractSet, Iterable, Mapping, Optional

OPENAI = "openai"
ANTHROPIC = "anthropic"
OLLAMA = "ollama"

_CLAIMED_ID_CAPACITY = 65536

_EMPTY: Mapping[str, int] = {}

# Replaced, never mutated: a child task shares the parent's mapping object
# until one of them sets a new one, so an in-place update would leak into
# sibling tasks.
_claims_in_context: ContextVar[Mapping[str, int]] = ContextVar("revenium_transport_claims", default=_EMPTY)


class _ClaimedResponseIds:
    """Bounded set of claimed provider response ids, oldest evicted first."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._ids: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, response_id: str) -> None:
        with self._lock:
            self._ids[response_id] = None
            self._ids.move_to_end(response_id)
            while len(self._ids) > self._capacity:
                self._ids.popitem(last=False)

    def contains_any(self, response_ids: Iterable[str]) -> bool:
        with self._lock:
            return any(response_id in self._ids for response_id in response_ids)

    def clear(self) -> None:
        with self._lock:
            self._ids.clear()


_claimed_response_ids = _ClaimedResponseIds(_CLAIMED_ID_CAPACITY)


def claim_call_for_transport(provider: str, response_id: Optional[str] = None) -> None:
    """Record that the ``provider`` transport wrap has metered, or will meter, this call."""
    claims = _claims_in_context.get()
    _claims_in_context.set({**claims, provider: claims.get(provider, 0) + 1})
    if response_id:
        _claimed_response_ids.add(str(response_id))


def transport_claim_mark() -> Mapping[str, int]:
    """The claim counters in the current context, to compare against later."""
    return _claims_in_context.get()


def claimed_by_transport(mark: Mapping[str, int], providers: Optional[AbstractSet[str]] = None,
                         response_ids: Iterable[str] = ()) -> bool:
    """Whether a transport wrap claimed the call that started at ``mark``.

    ``providers`` limits the counter check to those providers' transports;
    None means the provider is unknown and any transport's claim counts.
    """
    if _claimed_response_ids.contains_any(response_ids):
        return True
    claims = _claims_in_context.get()
    if providers is None:
        return claims != mark
    return any(claims.get(provider, 0) != mark.get(provider, 0) for provider in providers)


def reset_claimed_response_ids() -> None:
    """Forget every remembered response id (for tests)."""
    _claimed_response_ids.clear()
