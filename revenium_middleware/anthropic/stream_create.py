"""Pass-through wrappers for the raw-stream form of client.messages.create(stream=True).

The Anthropic SDK returns a Stream/AsyncStream of raw events for this call form
(unlike client.messages.stream(), which returns a context-manager helper). These
wrappers relay every event unchanged, accumulate usage from the event stream,
and fire a metering callback exactly once when the stream ends -- including
early termination (break, close, garbage collection), so partial usage is
still metered.
"""

import logging

from revenium_middleware._core.cache_tokens import extract_cache_creation_ttl_counts

logger = logging.getLogger("revenium_middleware.anthropic")


class StreamUsageState:
    """Usage accumulated from raw stream events.

    input tokens / id / model arrive on message_start; output tokens (cumulative)
    and stop_reason arrive on message_delta events. The per-TTL cache-creation
    breakdown also arrives on message_start, and stays empty when the response
    carries no split.
    """

    def __init__(self):
        self.message_id = None
        self.model = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation_ttl_counts = {}
        self.stop_reason = None
        self.first_event_time_dt = None
        self.saw_message_start = False
        self.accumulated_text = []

    def ingest(self, event):
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
                self.saw_message_start = True
            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
                if output_tokens is not None:
                    self.output_tokens = output_tokens
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
                if stop_reason:
                    self.stop_reason = stop_reason
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", None) if delta is not None else None
                if text:
                    self.accumulated_text.append(text)
        except Exception as e:
            logger.debug("Error ingesting stream event for metering: %s", e)


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
