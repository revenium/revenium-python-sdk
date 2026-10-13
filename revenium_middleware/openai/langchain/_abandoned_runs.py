"""Forgets the LangChain runs a chat model's ``generate`` or ``agenerate``
started when that call raises (BACK-4379).

langchain-core sends no end hook to a run whose ``agenerate`` is cancelled
while it waits (a timeout, a dropped request, a stopped agent), nor to the
other runs of a sync ``generate`` batch once one message list raises. The
callback registers a ``forget`` for each run it starts inside one of these
calls; when the call raises anything, every registered ``forget`` runs and the
exception goes on unchanged. On success nothing runs and nothing is logged.

langchain-core delivers an async run's end hook in a shielded task, so a call
cancelled while that hook is under way is still answered. Such runs are marked
from the moment their end hook starts until it has reached every handler.
"""
import asyncio
import contextvars
from typing import Any, Callable, List, Optional, Set

from revenium_middleware._core.patch_registry import register_patch, unregister_patch

CLEANUP_KEY_PREFIX = "openai.langchain:"
Forget = Callable[[], None]

_forgets_of_this_generate: "contextvars.ContextVar[Optional[List[Forget]]]" = contextvars.ContextVar(
    "revenium_langchain_forgets_of_this_generate", default=None)


def forget_if_its_generate_fails(forget: Forget) -> None:
    """Run ``forget`` if the ``generate`` or ``agenerate`` this run started in
    raises; outside one, do nothing."""
    forgets = _forgets_of_this_generate.get()
    if forgets is not None:
        forgets.append(forget)


def _forget_all(forgets: List[Forget]) -> None:
    for forget in forgets:
        try:
            forget()
        except Exception:  # the caller's own exception is the one that must reach it
            continue


def _generate_forgetting_its_runs_on_failure(wrapped, instance, args, kwargs):
    forgets: List[Forget] = []
    token = _forgets_of_this_generate.set(forgets)
    try:
        return wrapped(*args, **kwargs)
    except BaseException:
        _forget_all(forgets)
        raise
    finally:
        _forgets_of_this_generate.reset(token)


async def _agenerate_forgetting_its_runs_on_failure(wrapped, instance, args, kwargs):
    forgets: List[Forget] = []
    token = _forgets_of_this_generate.set(forgets)
    try:
        return await wrapped(*args, **kwargs)
    except BaseException:
        _forget_all(forgets)
        raise
    finally:
        _forgets_of_this_generate.reset(token)


_runs_with_an_end_on_its_way: Set[Any] = set()


def end_is_on_its_way(run_id: Any) -> bool:
    """Whether langchain-core has started delivering this run's end hook and
    has not yet finished delivering it to every handler."""
    return run_id in _runs_with_an_end_on_its_way


async def _on_llm_end_marking_its_run_until_delivered(wrapped, instance, args, kwargs):
    run_id = instance.run_id
    _runs_with_an_end_on_its_way.add(run_id)
    delivery = asyncio.ensure_future(wrapped(*args, **kwargs))
    delivery.add_done_callback(lambda _: _runs_with_an_end_on_its_way.discard(run_id))
    return await asyncio.shield(delivery)


_LANGCHAIN_CHAT_MODELS = "langchain_core.language_models.chat_models"
_LANGCHAIN_CALLBACK_MANAGER = "langchain_core.callbacks.manager"
_TARGETS = (
    (_LANGCHAIN_CHAT_MODELS, "BaseChatModel.generate", _generate_forgetting_its_runs_on_failure),
    (_LANGCHAIN_CHAT_MODELS, "BaseChatModel.agenerate", _agenerate_forgetting_its_runs_on_failure),
    (_LANGCHAIN_CALLBACK_MANAGER, "AsyncCallbackManagerForLLMRun.on_llm_end",
     _on_llm_end_marking_its_run_until_delivered),
)


def install_abandoned_run_cleanup() -> None:
    """Wrap langchain-core's chat ``generate`` and ``agenerate``, and its async
    end hook delivery, when it is installed."""
    try:
        import wrapt
    except ImportError:  # the callback reports the missing wrapt, with its install command, when built
        return
    for module, attribute, wrapper in _TARGETS:
        key = f"{CLEANUP_KEY_PREFIX}{module}.{attribute}"
        if not register_patch(key):
            continue
        try:
            wrapt.wrap_function_wrapper(module, attribute, wrapper)
        except (ImportError, AttributeError):
            unregister_patch(key)
