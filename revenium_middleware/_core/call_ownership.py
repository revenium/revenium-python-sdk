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

Because the transport's record is the one kept, it has to carry the
attribution the caller gave the callback (subscriber, organization, trace,
task). The callback calls ``publish_callback_metadata(metadata, providers,
call_id)`` when the model call starts, with LangChain's run id as
``call_id``; a transport wrap takes it with ``with_callback_metadata(provider,
call_metadata)``, where the call's own metadata wins field by field. That
direction only works because the callback runs inline (``run_inline``):
otherwise, during ``ainvoke``/``astream``, langchain-core runs a sync handler
method on a copy of the caller's context (``run_in_executor(copy_context().run,
...)``) and nothing the callback writes reaches the transport. A publication
is seen in the context it was made in and in tasks started from it, which is
where the model call runs.

A publication belongs to the one model call it was made for, and a stream's
consumer loop runs in the same context as the stream, so a provider-wide
offer would hand a stream's attribution to every unrelated client call made
between its chunks. Each publication is therefore offered to one transport
call only. LangChain makes exactly one transport call per model call (one
``create``, ``parse`` or ``stream`` on the OpenAI, Anthropic or Ollama
client, the client's retries running below the wrap), so the first transport
call of a provider in scope takes the latest open model call and closes every
publication made for it, one per attached handler. The publication also
closes when the callback withdraws it: on the model call's first token,
which langchain-core dispatches before the chunk reaches the caller, and when
the call ends or fails. That covers a model call whose transport never takes
it (a client the SDK does not meter, or one selective metering lets through).
Closing marks the publication itself, so a call that ends in another task (an
abandoned ``astream`` closed by the event loop's finalizer, or the child task
``agenerate`` runs the transport in) still closes it in the context it
started in.

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
from typing import Any, AbstractSet, Dict, Iterable, Mapping, Optional, Tuple

from revenium_middleware._core.context import overlay_metadata

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


class CallbackMetadata:
    """Attribution a LangChain callback published for one model call."""

    __slots__ = ("metadata", "providers", "call_id", "closed")

    def __init__(self, metadata: Mapping[str, Any], providers: Optional[AbstractSet[str]], call_id: Any):
        self.metadata = dict(metadata)
        self.providers = providers
        self.call_id = call_id
        self.closed = False

    def serves(self, provider: str) -> bool:
        return not self.closed and (self.providers is None or provider in self.providers)

    def belongs_with(self, other: "CallbackMetadata") -> bool:
        return self is other or (self.call_id is not None and self.call_id == other.call_id)

    def withdraw(self) -> None:
        self.closed = True
        _published_in_context.set(_open_publications())


# Replaced, never mutated, for the same reason as ``_claims_in_context``.
_published_in_context: ContextVar[Tuple[CallbackMetadata, ...]] = ContextVar(
    "revenium_callback_metadata", default=()
)


def _open_publications() -> Tuple[CallbackMetadata, ...]:
    return tuple(publication for publication in _published_in_context.get() if not publication.closed)


def publish_callback_metadata(metadata: Mapping[str, Any], providers: Optional[AbstractSet[str]] = None,
                              call_id: Any = None) -> CallbackMetadata:
    """Offer ``metadata`` to the next transport call in ``providers`` (any,
    when None) for the model call ``call_id`` starting in this context."""
    publication = CallbackMetadata(metadata, providers, call_id)
    _published_in_context.set(_open_publications() + (publication,))
    return publication


def take_callback_metadata(provider: str) -> Dict[str, Any]:
    """The metadata published for the latest open model call that
    ``provider``'s transport serves, or an empty dict, closing every
    publication made for that call. Among the call's publications the latest
    non-empty one wins, so a handler attached without metadata never hides
    one attached with it."""
    publications = _open_publications()
    serving = [publication for publication in publications if publication.serves(provider)]
    if not serving:
        return {}
    model_call = [publication for publication in publications if publication.belongs_with(serving[-1])]
    for publication in model_call:
        publication.closed = True
    _published_in_context.set(_open_publications())
    return next((dict(publication.metadata) for publication in reversed(model_call) if publication.metadata), {})


def with_callback_metadata(provider: str, call_metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``call_metadata`` laid over the callback's metadata for this call.

    Takes the callback's publication, so a transport wrap calls it exactly once
    per transport call and reuses the result for the budget check and the
    record; a second call in the same context gets no callback metadata.
    """
    return overlay_metadata(take_callback_metadata(provider), dict(call_metadata or {}))


def reset_claimed_response_ids() -> None:
    """Forget every remembered response id (for tests)."""
    _claimed_response_ids.clear()
