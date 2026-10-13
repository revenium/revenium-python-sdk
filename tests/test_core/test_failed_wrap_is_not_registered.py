"""A hook that fails to install is reported as not installed (BACK-4235).

The patch registry recorded a method as wrapped before the wrap was tried and
kept it when the wrap raised. The SDK then reported a method it never wrapped,
the LangChain callback skipped its own cap check because it believed the
Anthropic hook would make it, and CI's install check passed on a hook that was
not there. Each case runs in a fresh interpreter whose wrapt fails one target,
the way a provider SDK release that moved or renamed the method does.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PRELUDE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo_root!r})
    import wrapt

    FAILING = ({module!r}, {attribute!r})
    real_wrap, real_patch = wrapt.wrap_function_wrapper, wrapt.patch_function_wrapper

    def _target(module, attribute):
        return (module if isinstance(module, str) else module.__name__, attribute)

    def _missing():
        return ModuleNotFoundError(f"No module named {{FAILING[0]!r}} (simulated)")

    def wrap_function_wrapper(module, attribute, wrapper, *args, **kwargs):
        if _target(module, attribute) == FAILING:
            raise _missing()
        return real_wrap(module, attribute, wrapper, *args, **kwargs)

    def patch_function_wrapper(module, attribute, *args, **kwargs):
        if _target(module, attribute) == FAILING:
            def fail(wrapper):
                raise _missing()
            return fail
        return real_patch(module, attribute, *args, **kwargs)

    wrapt.wrap_function_wrapper, wrapt.patch_function_wrapper = wrap_function_wrapper, patch_function_wrapper

    import {integration}
    from revenium_middleware._core.patch_registry import is_patched
    from tests._wrap_inspect import wrapped_by_revenium
    KEY = {key!r}
""")

REPORT_REGISTRATION = textwrap.dedent("""
    print(json.dumps({"is_patched": is_patched(KEY), "wrapped": wrapped_by_revenium(KEY)}))
""")

CHECK_LANGCHAIN_CAPS = textwrap.dedent("""
    import os, uuid
    os.environ["REVENIUM_CIRCUIT_BREAKER_ENABLED"] = "true"
    from revenium_middleware._core import enforcement
    from revenium_middleware._core.exceptions import BudgetExceededError
    from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler

    enforcement._load_cache_from_disk = lambda: None
    enforcement._ensure_poller_running = lambda: None
    enforcement._cached_rules = [{
        "ruleId": 4235, "name": "Spending cap", "metricType": "TOTAL_COST", "threshold": 1.0,
        "currentValue": 2.5, "windowType": "DAILY", "windowEnd": "2999-01-01T00:00:00Z",
        "action": "BLOCK", "breached": True, "shadowMode": False,
    }]
    enforcement._cache_timestamp = float("inf")
    enforcement._cache_initialized = True

    serialized = {"id": ["langchain", "chat_models", "anthropic", "ChatAnthropic"],
                  "kwargs": {"model": "claude-haiku-4-5"}}
    try:
        UnifiedReveniumCallbackHandler().on_chat_model_start(
            serialized, [[]], run_id=uuid.uuid4(), invocation_params={"model": "claude-haiku-4-5"})
        refused = False
    except BudgetExceededError:
        refused = True
    print(json.dumps({"refused": refused}))
""")

HOOKS = {
    "anthropic": ("revenium_middleware.anthropic", "anthropic.resources.messages.messages", "Messages.create"),
    "openai": ("revenium_middleware.openai", "openai.resources.chat.completions", "Completions.create"),
    "google_genai": ("revenium_middleware.google.google_ai", "google.genai.models", "Models.generate_content"),
}


def run_child(hook, body):
    integration, module, attribute = HOOKS[hook]
    pytest.importorskip(module)
    source = PRELUDE.format(repo_root=str(REPO_ROOT), module=module, attribute=attribute,
                            integration=integration, key=f"{module}.{attribute}") + body
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=120,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("hook", sorted(HOOKS))
def test_a_hook_that_fails_to_install_is_not_reported_as_wrapped(hook):
    assert run_child(hook, REPORT_REGISTRATION) == {"is_patched": False, "wrapped": False}


def test_langchain_checks_caps_itself_when_the_anthropic_hook_failed():
    pytest.importorskip("langchain_core")

    assert run_child("anthropic", CHECK_LANGCHAIN_CAPS) == {"refused": True}


FAL_ALIAS_REBIND_FAILS = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {repo_root!r})
    import fal_client

    fal_client.sync_client = object()
    try:
        import revenium_middleware.fal.middleware  # noqa: F401
        raised = False
    except AttributeError:
        raised = True
    from revenium_middleware._core.patch_registry import is_patched
    print(json.dumps({{"raised": raised, "is_patched": is_patched("fal_client.run")}}))
""")


def test_a_fal_alias_that_fails_to_rebind_is_not_reported_as_wrapped():
    pytest.importorskip("fal_client")
    source = FAL_ALIAS_REBIND_FAILS.format(repo_root=str(REPO_ROOT))
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=120,
                            cwd=str(REPO_ROOT))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"raised": True, "is_patched": False}
