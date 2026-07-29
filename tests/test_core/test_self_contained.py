"""BACK-2151 guards: the SDK is self-contained, public surface unchanged."""
import pathlib

import revenium_middleware as rm


EXPECTED_ALL = {
    "client", "get_buffer_stats", "get_client", "initialize_metering",
    "run_async_in_thread", "shutdown_event",
    "revenium_meter", "revenium_metadata", "track_usage",
    "is_inside_decorated_function", "get_function_metadata",
    "set_decorated_context", "clear_decorated_context",
    "get_injected_metadata", "set_injected_metadata",
    "clear_injected_metadata", "merge_metadata",
    "idempotency_key", "get_idempotency_key", "set_idempotency_key",
    "is_selective_metering_enabled",
    "meter_tool", "report_tool_call", "configure",
    "AgenticOutcomeClient", "AgenticOutcomeSettings",
    # BACK-777: public job-context surface
    "JobContext", "OutcomeReportingError", "OutcomeAlreadyReportedError",
    # BACK-777 Phase 3: amendments + history
    "get_outcome_history", "JobOutcomeAmendment",
    "OutcomeNotReportedError", "OutcomeAmendConflictError",
}


def test_public_all_unchanged():
    assert set(rm.__all__) == EXPECTED_ALL


def test_tool_registry_resolves_to_vendored_module():
    for name in ("meter_tool", "report_tool_call", "configure"):
        sym = getattr(rm, name)
        assert sym.__module__.startswith("revenium_middleware._metering"), (name, sym.__module__)


def test_metering_client_is_vendored():
    from revenium_middleware._metering import ReveniumMetering
    c = ReveniumMetering(api_key="rev_mk_demo")
    assert hasattr(c.ai, "create_completion")
    assert hasattr(c.ai, "create_image")
    assert hasattr(c.ai, "create_video")
    assert hasattr(c.ai, "create_audio")
    assert type(c).__module__.startswith("revenium_middleware._metering")


def test_no_external_revenium_metering_dependency():
    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    # crude but sufficient: the dependency name must not appear in the deps list
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    data = tomllib.loads(text)
    deps = data["project"]["dependencies"]
    assert not any("revenium_metering" in d or "revenium-metering" in d for d in deps), deps


def test_outcome_client_uses_vendored_metering():
    from revenium_middleware import AgenticOutcomeClient, AgenticOutcomeSettings
    client = AgenticOutcomeClient(AgenticOutcomeSettings(api_key="rev_sk_t_demo"))
    assert type(client._metering()).__module__.startswith("revenium_middleware._metering")
