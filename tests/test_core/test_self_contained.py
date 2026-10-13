"""BACK-2151 guards: the SDK is self-contained, public surface unchanged."""
import pathlib

import pytest

import revenium_middleware as rm


EXPECTED_ALL = {
    "client", "get_buffer_stats", "get_client", "initialize_metering",
    "run_async_in_thread", "shutdown_event",
    "revenium_meter", "revenium_metadata", "track_usage",
    "is_inside_decorated_function", "get_function_metadata",
    "set_decorated_context", "clear_decorated_context",
    "get_injected_metadata", "set_injected_metadata",
    "clear_injected_metadata", "merge_metadata",
    "idempotency_key", "get_idempotency_key", "set_idempotency_key", "carry_context",
    "is_selective_metering_enabled",
    "meter_tool", "report_tool_call", "configure",
    "AgenticOutcomeClient", "AgenticOutcomeSettings",
    # BACK-777: public job-context surface
    "JobContext", "OutcomeReportingError", "OutcomeAlreadyReportedError",
    # BACK-777 Phase 3: amendments + history
    "get_outcome_history", "JobOutcomeAmendment",
    "OutcomeNotReportedError", "OutcomeAmendConflictError",
    # BACK-4033: job reads
    "list_jobs", "get_job", "get_job_types", "get_job_roi",
    "get_job_transactions", "get_conversion_funnel",
    # BACK-2746: job-type economics, baselines and period facts
    "OutcomeMetricProvenance", "PeriodFactEntry", "Baseline",
    "BaselineProvenance", "JobTypeEconomics", "JobTypeMetricDirection",
    "JobTypeMonetizationCategory", "JobTypeMonetizationBasis",
    "report_period_facts", "get_job_type_economics",
    "upsert_job_type_economics", "create_baseline", "list_baselines",
    # Metering error visibility (BACK-778)
    "MeteringErrorEvent", "MeteringStatus", "get_metering_status",
    "on_metering_error", "remove_metering_error_callback", "reset_metering_status",
    # Spending-cap refusal (BACK-4190)
    "BudgetExceededError", "ReveniumCostLimitExceeded",
    # Per-thread or per-task metering key (BACK-4257)
    "metering_key",
    # Flush a worker process's usage before it ends (BACK-4353)
    "flush_metering",
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


def test_the_langchain_extra_brings_what_the_callback_imports():
    """BACK-4011: the LangChain callback meters through the OpenAI middleware,
    which imports wrapt, so the README's ``[langchain]`` install must bring it."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    installed = project["dependencies"] + project["optional-dependencies"]["langchain"]
    assert any(requirement.startswith("wrapt") for requirement in installed), installed


CURRENT_GOOGLE_VERTEX_SDK = "2.4.0"
LATEST_RELEASE_BEFORE_A_MAJOR_CAP = "2.999.0"


def _project():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    return tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def _keeps_the_current_google_sdk(requirement_lines):
    from packaging.requirements import Requirement

    requirements = [Requirement(line) for line in requirement_lines]
    aiplatform = [req for req in requirements if req.name == "google-cloud-aiplatform"]
    return (not any(req.name == "vertexai" for req in requirements)
            and len(aiplatform) == 1
            and aiplatform[0].specifier.contains(CURRENT_GOOGLE_VERTEX_SDK)
            and aiplatform[0].specifier.contains(LATEST_RELEASE_BEFORE_A_MAJOR_CAP))


def test_the_google_vertex_extra_installs_the_current_google_sdk():
    """BACK-4024: the PyPI ``vertexai`` package is a frozen 2024 metapackage that pins
    google-cloud-aiplatform 1.71.1, so ``[google-vertex]`` rolled customers back two years."""
    assert _keeps_the_current_google_sdk(_project()["optional-dependencies"]["google-vertex"])


@pytest.mark.parametrize("extra", [
    ["vertexai>=1.0.0"],
    ["google-cloud-aiplatform==1.71.1"],
    ["google-cloud-aiplatform>=1.71.1,<2"],
    ["google-cloud-aiplatform>=1.71.1", "vertexai"],
])
def test_a_vertex_extra_that_pins_an_old_google_sdk_fails_the_check(extra):
    assert not _keeps_the_current_google_sdk(extra)


_EXTRA_BY_README_HEADING_WITH_A_DOTENV_EXAMPLE = {
    "### 2. Import and Use": "openai",
    "### OpenAI": "openai",
    "### Anthropic": "anthropic",
    "#### Google AI SDK": "google-genai",
    "#### Vertex AI SDK (deprecated by Google)": "google-vertex",
    "### Ollama": "ollama",
    "#### Client Mode": "litellm",
    "#### Using OpenAI SDK": "perplexity-openai",
}


def _headings_of_readme_examples_that_load_a_dotenv_file():
    headings = []
    heading = None
    block = None
    for line in pathlib.Path("README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            if block is not None:
                if any("from dotenv import" in text for text in block):
                    headings.append(heading)
                block = None
            else:
                block = []
        elif block is not None:
            block.append(line)
        elif line.startswith("#"):
            heading = line.strip()
    return headings


def test_every_extra_whose_readme_example_loads_a_dotenv_file_installs_python_dotenv():
    """BACK-4411: an example that opens with ``from dotenv import load_dotenv``
    must run after installing only the extra the README names for it."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    extras = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    headings = _headings_of_readme_examples_that_load_a_dotenv_file()

    unmapped = [heading for heading in headings if heading not in _EXTRA_BY_README_HEADING_WITH_A_DOTENV_EXAMPLE]
    assert unmapped == [], f"add these README headings to the extra table: {unmapped}"
    without_dotenv = sorted({
        extra for heading, extra in _EXTRA_BY_README_HEADING_WITH_A_DOTENV_EXAMPLE.items()
        if heading in headings
        and not any(requirement.startswith("python-dotenv") for requirement in extras[extra])
    })
    assert without_dotenv == []
