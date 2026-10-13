"""Pass-through wrappers for the raw-stream form of client.messages.create(stream=True).

The Anthropic SDK returns a Stream/AsyncStream of raw events for this call form
(unlike client.messages.stream(), which returns a context-manager helper). These
wrappers relay every event unchanged, accumulate usage from the event stream,
and fire a metering callback exactly once when the stream ends -- including
early termination (break, close, garbage collection), so partial usage is
still metered.

Also holds the event relay that feeds a StreamUsageState from the private
``_iterator`` of a client.messages.stream() helper, which is how that helper's
advisor and compaction passes are read.
"""

import inspect
import logging

from revenium_middleware._core.cache_tokens import extract_cache_creation_ttl_counts
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.failed_calls import exception_in_flight, failure_of

logger = logging.getLogger("revenium_middleware.anthropic")


class StreamUsageState:
    """Usage accumulated from raw stream events.

    id / model and the first pass's usage arrive on message_start; message_delta
    events carry cumulative usage for the whole turn (a turn with server tools or
    an advisor runs several executor passes), the turn's usage iterations and the
    stop_reason. The per-TTL cache-creation breakdown arrives on message_start,
    and stays empty when the response carries no split. A message_delta that
    changes the cache-creation total replaces the breakdown with its own, or
    drops it, since the message_start split no longer sums to the total.
    """

    _CUMULATIVE_DELTA_FIELDS = (
        "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "iterations",
    )

    def __init__(self):
        self.message_id = None
        self.model = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation_ttl_counts = {}
        self.iterations = None
        self.stop_reason = None
        self.first_event_time_dt = None
        self.saw_message_start = False
        self.accumulated_text = []
        self.failure = None
        self._outer = exception_in_flight()
        self._call_context = CallContext()

    def record_failure(self, error):
        """Mark the stream as broken off by ``error``, so it is recorded as an error."""
        self.failure = self._call_context.run(failure_of, error, self._outer)

    def ingest(self, event, keep_text=True):
        try:
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                import datetime
                self.first_event_time_dt = datetime.datetime.now(datetime.timezone.utc)
                message = getattr(event, "message", None)
                self.message_id = getattr(message, "id", None)
                self.model = getattr(message, "model", None)
                usage = getattr(message, "usage", None)
                if usage is not None:
                    self.input_tokens = getattr(usage, "input_tokens", 0) or 0
                    self.output_tokens = getattr(usage, "output_tokens", 0) or 0
                    self.cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
                    self.cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                    self.cache_creation_ttl_counts = extract_cache_creation_ttl_counts(usage)
                    self.iterations = getattr(usage, "iterations", None)
                self.saw_message_start = True
            elif event_type == "message_delta":
                self._overwrite_from_delta(getattr(event, "usage", None))
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
                if stop_reason:
                    self.stop_reason = stop_reason
            elif keep_text and event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta is not None else None
                if text:
                    self.accumulated_text.append(text)
        except Exception as e:
            logger.debug("Error ingesting stream event for metering: %s", e)

    def _overwrite_from_delta(self, usage):
        if usage is None:
            return
        delta_cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        if delta_cache_creation is not None and delta_cache_creation != self.cache_creation_input_tokens:
            self.cache_creation_ttl_counts = extract_cache_creation_ttl_counts(usage)
        for field in self._CUMULATIVE_DELTA_FIELDS:
            value = getattr(usage, field, None)
            if value is not None:
                setattr(self, field, value)


def meter_stream_helper_events(message_stream, state):
    """Feed ``state`` every event a client.messages.stream() helper yields.

    text_stream, get_final_message(), get_final_text() and until_done() all read
    the helper's one event iterator, so relaying that iterator sees the events
    whichever of them the caller uses. The helper keeps the response text
    itself, so the relay reads usage only.
    """
    def relay(events):
        try:
            for event in events:
                state.ingest(event, keep_text=False)
                yield event
        except Exception as error:
            state.record_failure(error)
            raise

    _relay_stream_helper_events(message_stream, inspect.isgenerator, relay)


def meter_async_stream_helper_events(message_stream, state):
    async def relay(events):
        try:
            async for event in events:
                state.ingest(event, keep_text=False)
                yield event
        except Exception as error:
            state.record_failure(error)
            raise

    _relay_stream_helper_events(message_stream, inspect.isasyncgen, relay)


def _relay_stream_helper_events(message_stream, is_relayable, relay):
    """Swap the helper's private event iterator for ``relay`` of it, or leave the stream alone.

    ``_iterator`` is not public anthropic API. If it is missing, is not the
    generator the relay expects, or cannot be replaced, the caller's stream is
    untouched and the call is metered from the helper's accumulated message.
    """
    try:
        events = getattr(message_stream, "_iterator", None)
        if is_relayable(events):
            message_stream._iterator = relay(events)
    except Exception as e:
        logger.debug("Stream helper events are not relayed for metering: %s", e)


class RawStreamMeteringWrapper:
    """Wraps a sync anthropic.Stream; meters once on any form of termination."""

    def __init__(self, stream, state, finalize):
        self._stream = stream
        self._state = state
        self._finalize_cb = finalize
        self._finalized = False

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        try:
            self._finalize_cb(self._state)
        except Exception as e:
            logger.warning("Error finalizing stream metering: %s", e)

    def __iter__(self):
        try:
            for event in self._stream:
                self._state.ingest(event)
                yield event
        except Exception as error:
            self._state.record_failure(error)
            raise
        finally:
            # Runs on normal exhaustion AND on GeneratorExit from an
            # abandoned/broken-out-of loop -- partial usage still meters.
            self._finalize()

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._stream.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._finalize()

    def close(self):
        try:
            self._stream.close()
        finally:
            self._finalize()

    def __del__(self):
        try:
            self._finalize()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class AsyncRawStreamMeteringWrapper:
    """Wraps an anthropic.AsyncStream; meters once on any form of termination.

    The finalize callback is synchronous (it only schedules the fire-and-forget
    metering thread), so it is safe to call from __del__ and aclose alike.
    """

    def __init__(self, stream, state, finalize):
        self._stream = stream
        self._state = state
        self._finalize_cb = finalize
        self._finalized = False

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        try:
            self._finalize_cb(self._state)
        except Exception as e:
            logger.warning("Error finalizing async stream metering: %s", e)

    async def __aiter__(self):
        try:
            async for event in self._stream:
                self._state.ingest(event)
                yield event
        except Exception as error:
            self._state.record_failure(error)
            raise
        finally:
            self._finalize()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._finalize()

    async def aclose(self):
        try:
            await self._stream.aclose()
        finally:
            self._finalize()

    def __del__(self):
        try:
            self._finalize()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)
