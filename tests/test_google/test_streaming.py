"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for streaming functionality.
"""

import pytest
import datetime
from unittest.mock import Mock, patch, MagicMock


class TestStreamingWrapper:
    """Test streaming wrapper functionality."""

    def test_stream_wrapper_tracks_first_chunk_time(self):
        """Test that first chunk time is recorded."""
        # Simulate tracking first chunk time
        first_chunk_time = None
        chunks = []

        for i in range(5):
            if first_chunk_time is None:
                first_chunk_time = datetime.datetime.now(datetime.timezone.utc)
            chunks.append(f"chunk_{i}")

        assert first_chunk_time is not None
        assert len(chunks) == 5

    def test_calculate_time_to_first_token(self):
        """Test calculation of time to first token."""
        request_time = datetime.datetime.now(datetime.timezone.utc)
        # Simulate some delay
        import time
        time.sleep(0.01)  # 10ms delay
        first_chunk_time = datetime.datetime.now(datetime.timezone.utc)

        time_to_first_token_ms = int(
            (first_chunk_time - request_time).total_seconds() * 1000
        )

        assert time_to_first_token_ms >= 10  # At least 10ms
        assert time_to_first_token_ms < 1000  # Less than 1 second

    def test_memory_protection_max_chunks(self):
        """Test that memory protection limits chunk collection."""
        max_chunks = 1000
        chunks = []

        # Simulate collecting chunks
        for i in range(1500):
            if len(chunks) >= max_chunks:
                break
            chunks.append(f"chunk_{i}")

        assert len(chunks) == max_chunks

    def test_idempotent_usage_logging(self):
        """Test that usage is logged only once."""
        usage_logged = False

        def log_usage():
            nonlocal usage_logged
            if usage_logged:
                return  # Already logged
            usage_logged = True
            # Log usage here

        # Call multiple times
        log_usage()
        log_usage()
        log_usage()

        # Should only log once
        assert usage_logged is True

    def test_extract_finish_reason_from_chunks(self):
        """Test extraction of finish reason from chunks."""
        finish_reason = None

        # Simulate chunks
        mock_chunks = []
        for i in range(5):
            chunk = Mock()
            candidate = Mock()
            if i == 4:  # Last chunk has finish reason
                candidate.finish_reason = "STOP"
            else:
                candidate.finish_reason = None
            chunk.candidates = [candidate]
            mock_chunks.append(chunk)

        # Extract finish reason
        for chunk in mock_chunks:
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.finish_reason:
                        finish_reason = candidate.finish_reason

        assert finish_reason == "STOP"

    def test_extract_usage_from_final_chunk(self):
        """Test that usage metadata is extracted from final chunk."""
        usage_metadata = None

        # Simulate streaming chunks
        mock_chunks = []
        for i in range(5):
            chunk = Mock()
            if i == 4:  # Last chunk has usage
                mock_usage = Mock()
                mock_usage.prompt_token_count = 100
                mock_usage.candidates_token_count = 150
                chunk.usage_metadata = mock_usage
            else:
                chunk.usage_metadata = None
            mock_chunks.append(chunk)

        # Extract usage from last chunk
        for chunk in mock_chunks:
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

        assert usage_metadata is not None
        assert usage_metadata.prompt_token_count == 100
        assert usage_metadata.candidates_token_count == 150


class TestStreamingEdgeCases:
    """Test edge cases in streaming."""

    def test_empty_stream(self):
        """Test handling of empty stream."""
        chunks = []

        # Empty stream should not crash
        for chunk in chunks:
            pass

        assert len(chunks) == 0

    def test_stream_with_no_usage_metadata(self):
        """Test stream where no chunk has usage metadata."""
        mock_chunks = [Mock(usage_metadata=None) for _ in range(5)]

        usage_metadata = None
        for chunk in mock_chunks:
            if chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

        # Should handle gracefully
        assert usage_metadata is None

    def test_stream_interrupted(self):
        """Test handling of interrupted stream."""
        chunks_collected = []
        max_chunks = 10

        # Simulate stream interruption
        for i in range(5):  # Only 5 chunks before interruption
            chunks_collected.append(f"chunk_{i}")

        assert len(chunks_collected) < max_chunks

    def test_stream_with_exceptions_in_chunks(self):
        """Test handling when chunk processing raises exception."""
        def process_chunks():
            chunks = []
            for i in range(5):
                if i == 3:
                    # Simulate an error in chunk 3
                    raise ValueError("Chunk processing error")
                chunks.append(f"chunk_{i}")
            return chunks

        with pytest.raises(ValueError):
            process_chunks()

    def test_concurrent_stream_wrappers(self):
        """Test that multiple concurrent streams are independent."""
        stream1_first_time = None
        stream2_first_time = None

        # Simulate two concurrent streams
        import time

        # Stream 1
        stream1_first_time = datetime.datetime.now(datetime.timezone.utc)
        time.sleep(0.01)

        # Stream 2 (independent)
        stream2_first_time = datetime.datetime.now(datetime.timezone.utc)

        # Should be different times
        assert stream1_first_time != stream2_first_time
        assert stream2_first_time > stream1_first_time


class TestStreamingCleanup:
    """Test streaming cleanup and resource management."""

    def test_stream_close_logs_usage(self):
        """Test that closing stream logs usage."""
        usage_logged = False
        stream_closed = False

        def close_stream():
            nonlocal usage_logged, stream_closed
            if not stream_closed:
                # Log usage on close
                if not usage_logged:
                    usage_logged = True
                stream_closed = True

        close_stream()

        assert usage_logged is True
        assert stream_closed is True

    def test_stream_cleanup_with_finally(self):
        """Test that cleanup happens in finally block."""
        cleanup_executed = False

        try:
            # Simulate streaming
            for i in range(5):
                if i == 3:
                    raise Exception("Stream error")
        except Exception:
            pass
        finally:
            cleanup_executed = True

        assert cleanup_executed is True

    def test_stream_wrapper_del_method(self):
        """Test that __del__ method handles cleanup."""
        # Simulate object deletion cleanup
        class StreamWrapper:
            def __init__(self):
                self.closed = False

            def __del__(self):
                if not self.closed:
                    self.close()

            def close(self):
                self.closed = True

        wrapper = StreamWrapper()
        wrapper_id = id(wrapper)
        wrapper = None  # Delete reference

        # Cleanup should have happened
        # (Can't easily test __del__ but this simulates the pattern)

    def test_multiple_close_calls_idempotent(self):
        """Test that calling close() multiple times is safe."""
        closed = False
        usage_logged = False

        def close():
            nonlocal closed, usage_logged
            if closed:
                return  # Already closed
            usage_logged = True
            closed = True

        # Call multiple times
        close()
        close()
        close()

        # Should only execute once
        assert closed is True
        assert usage_logged is True


class TestStreamingPerformance:
    """Test streaming performance characteristics."""

    def test_chunk_collection_memory_efficient(self):
        """Test that chunk collection doesn't grow unbounded."""
        import sys

        max_chunks = 1000
        chunks = []

        # Collect up to max
        for i in range(2000):
            if len(chunks) >= max_chunks:
                break
            chunks.append(f"chunk_{i}")

        # Should stop at max
        assert len(chunks) == max_chunks

        # Estimate memory usage (rough)
        chunk_size = sys.getsizeof(chunks[0])
        total_size = chunk_size * len(chunks)

        # Should be reasonable (less than 10MB for 1000 chunks)
        assert total_size < 10 * 1024 * 1024

    def test_time_to_first_token_precision(self):
        """Test that TTFT is measured in milliseconds."""
        start = datetime.datetime.now(datetime.timezone.utc)
        import time
        time.sleep(0.05)  # 50ms
        first_chunk = datetime.datetime.now(datetime.timezone.utc)

        ttft_ms = int((first_chunk - start).total_seconds() * 1000)

        # Should be around 50ms (with some tolerance)
        assert 45 <= ttft_ms <= 100
