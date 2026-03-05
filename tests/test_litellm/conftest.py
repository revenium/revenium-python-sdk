"""
Pytest configuration and fixtures for revenium-python-sdk LiteLLM tests.
"""

import pytest
import asyncio
import warnings


@pytest.fixture(autouse=True)
def suppress_litellm_warnings():
    """Suppress known LiteLLM warnings that are not related to our code."""
    with warnings.catch_warnings():
        # Suppress the RuntimeWarning about un-awaited coroutine at shutdown
        # This is a known LiteLLM issue with async client cleanup
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'close_litellm_async_clients' was never awaited",
            category=RuntimeWarning
        )
        yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_litellm_async_clients():
    """Clean up LiteLLM async clients at the end of the test session."""
    yield
    # Attempt to properly close LiteLLM async clients
    try:
        import litellm
        from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

        # Get or create event loop for cleanup
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the cleanup coroutine
        if not loop.is_closed():
            loop.run_until_complete(close_litellm_async_clients())
    except (ImportError, AttributeError):
        # LiteLLM version may not have this function
        pass
    except Exception:
        # Best effort cleanup - don't fail tests if cleanup fails
        pass
