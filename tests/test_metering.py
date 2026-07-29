import pytest
import logging
import asyncio
from unittest.mock import patch, AsyncMock

from revenium_middleware._core.metering import (
    active_threads,
    shutdown_event,
    handle_exit,
    run_async_in_thread
)


class TestMetering:
    @pytest.fixture
    def reset_state(self):
        """Fixture to reset global state before each test."""
        # Clear any active threads
        active_threads.clear()
        shutdown_event.clear()
        yield
        # Cleanup: make sure threads get stopped
        handle_exit()

    def test_run_async_in_thread_when_shutdown(self, reset_state, caplog):
        """If shutdown_event is set, run_async_in_thread should log a warning and return None."""
        shutdown_event.set()
        with caplog.at_level(logging.WARNING):
            thread = run_async_in_thread(asyncio.sleep(0.01))
            assert thread is None
            assert "Not starting new metering thread during shutdown" in caplog.text

    def test_run_async_in_thread_normal(self, reset_state):
        """Check that run_async_in_thread starts a thread and runs the coroutine."""
        coro_mock = AsyncMock()
        thread = run_async_in_thread(coro_mock)
        assert thread is not None
        thread.join(timeout=1.0)
        # Ensure the thread is removed from active_threads once done
        assert thread not in active_threads

    def test_metering_thread_error_handling(self, reset_state, caplog):
        """Ensure MeteringThread logs a warning if an exception occurs."""

        async def fail_coro():
            raise ValueError("Test error")

        with caplog.at_level(logging.WARNING):
            thread = run_async_in_thread(fail_coro())
            thread.join(timeout=1.0)
            # Check for the error message (format may vary with formatter)
            assert "Error in metering thread" in caplog.text
            assert "Test error" in caplog.text

    @patch("signal.signal")
    def test_handle_exit(self, mock_signal, reset_state, caplog):
        """handle_exit should set the shutdown_event and wait for threads to complete."""

        # Create a long-running coroutine
        async def long_run():
            await asyncio.sleep(0.5)

        thread = run_async_in_thread(long_run())
        assert thread is not None
        # Give thread time to start
        import time
        time.sleep(0.1)

        # Capture DEBUG level logs since shutdown messages are DEBUG
        # Need to capture from the specific logger
        with caplog.at_level(logging.DEBUG, logger='revenium_middleware'):
            handle_exit()
            assert shutdown_event.is_set()
            # Thread should have been joined
            assert "Shutdown initiated" in caplog.text or "SHUTDOWN" in caplog.text
            assert "Shutdown complete" in caplog.text or "COMPLETE" in caplog.text


class TestCreateCompletionTicketId:
    """Regression tests for ticket_id support in create_completion (FRONT-1545)."""

    def test_create_completion_accepts_ticket_id(self):
        """create_completion must accept ticket_id and send it as ticketId in the body."""
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-07-21T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="gpt-test",
                output_token_count=5,
                provider="OPENAI",
                request_duration=100,
                request_time="2026-07-21T00:00:00Z",
                response_time="2026-07-21T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-123",
                ticket_id="JIRA-123",
            )

        body = mock_post.call_args.kwargs["body"]
        assert body["ticketId"] == "JIRA-123"
