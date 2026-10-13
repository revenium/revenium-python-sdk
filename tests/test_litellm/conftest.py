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


@pytest.fixture(autouse=True)
def keep_metering_ownership_per_test():
    """Leave the process-wide metering owners as each test found them.

    A guardrail registers itself as the owner and a proxy config lists the
    callback in LiteLLM's callback lists; either one makes the client wrapper
    stand aside for proxied calls. A test that leaves one behind would silently
    change what every later test meters, while a reset would undo a
    module-scoped proxy's own registration, so the state is put back exactly.
    """
    try:
        import litellm
        from revenium_middleware.litellm.proxy import _metering_owner
    except ImportError:
        yield
        return
    owner = _metering_owner._owner
    callback_lists = {
        name: list(getattr(litellm, name)) for name in _metering_owner.LITELLM_CALLBACK_LISTS
        if isinstance(getattr(litellm, name, None), list)
    }
    yield
    if owner is None:
        _metering_owner.reset_metering_owner()
    else:
        _metering_owner.register_metering_guardrail(owner)
    for name, saved in callback_lists.items():
        getattr(litellm, name)[:] = saved


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
