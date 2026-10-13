"""The README's list of calls the SDK does not meter matches the coverage inventory.

tests/entry_points/resource_inventory.yaml records every billable call path the
SDK knowingly leaves unmetered, in groups marked ``disclose: true``. A customer
never reads that file, so README.md carries the same list between two markers,
and this test fails when the two disagree. It reads only the two files, so it
runs in every environment, whichever provider SDKs are installed.
"""
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINTS = REPO_ROOT / "tests" / "entry_points"
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
INVENTORY = yaml.safe_load((ENTRY_POINTS / "resource_inventory.yaml").read_text())["providers"]
MANIFEST_ROWS = yaml.safe_load((ENTRY_POINTS / "manifest.yaml").read_text())["rows"]
BLOCK_START = "<!-- not-metered:start -->"
BLOCK_END = "<!-- not-metered:end -->"
BILLABLE_REASONS = (
    "billable model usage the SDK does not meter",
    "asynchronous job (batch, fine-tuning, evaluation run, async completion)",
    "hosted agent, session or workflow",
    "hosted tool call (web search, page fetch, code sandbox)",
)
RAW_RESPONSE_SEGMENTS = (".with_raw_response.", ".with_streaming_response.")
# The caller roots the README folds together: the sync and async clients and the
# module-level functions reach the same provider call.
README_PROVIDERS = {
    "openai": ("OpenAI", ("OpenAI.", "AsyncOpenAI.")),
    "anthropic": ("Anthropic", ("Anthropic.", "AsyncAnthropic.")),
    "perplexity": ("Perplexity", ("Perplexity.", "AsyncPerplexity.")),
    "google-genai": ("Google Gemini", ("Client.aio.chats.create().", "Client.chats.create().", "Client.aio.", "Client.")),
    "ollama": ("Ollama", ("Client.", "AsyncClient.", "ollama.")),
    "litellm": ("LiteLLM", ("litellm.",)),
    "fal": ("fal.ai", ("SyncClient.", "AsyncClient.", "SyncRequestHandle.", "AsyncRequestHandle.", "fal_client.")),
    "vertex": ("Google Vertex AI", ()),
    "bedrock": ("AWS Bedrock", ("bedrock-runtime.",)),
    "langchain": ("LangChain", ()),
    "griptape": ("Griptape", ()),
}
BULLET = re.compile(r"^- \*\*(?P<name>[^*]+)\*\*(?: \([^)]*\))?: (?P<calls>.+)$")


def _caller_name(provider, method):
    for root in README_PROVIDERS[provider][1]:
        if method.startswith(root):
            return method[len(root):]
    return method


def _unsupported_rows():
    return {row["id"] for row in MANIFEST_ROWS if row.get("status") == "unsupported"}


def _executed_but_unmetered(entry):
    unsupported = _unsupported_rows()
    return {method for method, rows in entry["registered"].items() if set(rows) <= unsupported}


def _disclosed_methods(entry):
    grouped = {method for group in entry["allowed"] if group.get("disclose") for method in group["methods"]}
    return grouped | _executed_but_unmetered(entry)


def inventory_list():
    listed = {}
    for provider, entry in INVENTORY.items():
        names = {_caller_name(provider, method) for method in _disclosed_methods(entry)
                 if not any(segment in method for segment in RAW_RESPONSE_SEGMENTS)}
        if names:
            listed[README_PROVIDERS[provider][0]] = names
    return listed


def readme_list():
    assert BLOCK_START in README and BLOCK_END in README, (
        f"README.md has no {BLOCK_START} ... {BLOCK_END} block listing the calls the SDK does not meter"
    )
    block = README.split(BLOCK_START, 1)[1].split(BLOCK_END, 1)[0]
    listed = {}
    for line in block.splitlines():
        match = BULLET.match(line.strip())
        if match:
            listed[match["name"]] = set(re.findall(r"`([^`]+)`", match["calls"]))
    return listed


def test_every_inventory_provider_has_a_readme_name():
    assert set(INVENTORY) <= set(README_PROVIDERS)


def test_the_readme_lists_every_call_the_inventory_discloses():
    in_inventory, in_readme = inventory_list(), readme_list()
    missing = {name: sorted(calls - in_readme.get(name, set())) for name, calls in in_inventory.items()}
    extra = {name: sorted(calls - in_inventory.get(name, set())) for name, calls in in_readme.items()}
    assert ({name: calls for name, calls in missing.items() if calls},
            {name: calls for name, calls in extra.items() if calls}) == ({}, {}), (
        "README.md's not-metered list and tests/entry_points/resource_inventory.yaml disagree. "
        "First dict: disclosed in the inventory, missing from the README. "
        "Second dict: in the README, not disclosed in the inventory."
    )


def test_every_disclosed_group_is_a_billable_reason():
    for provider, entry in INVENTORY.items():
        for group in entry["allowed"]:
            billable = group["reason"].startswith(BILLABLE_REASONS)
            assert bool(group.get("disclose")) == billable, (provider, group["reason"])


def test_the_readme_names_gemini_interactions_and_litellm_messages():
    listed = readme_list()
    assert "interactions.create" in listed.get("Google Gemini", set())
    assert {"anthropic_messages", "anthropic_interface.messages.create"} <= listed.get("LiteLLM", set())


def test_the_readme_lists_litellm_responses_while_its_rows_are_unsupported():
    assert {"responses", "aresponses"} <= inventory_list()["LiteLLM"]


def test_the_readme_names_vertex_jobs_context_caches_and_bedrock():
    expected = {
        "Google Gemini": {"caches.create", "caches.update"},
        "Google Vertex AI": {"batch_prediction.BatchPredictionJob.submit", "tuning.sft.train",
                             "preview.caching.CachedContent.create"},
        "AWS Bedrock": {"invoke_model", "converse"},
    }
    in_readme, in_inventory = readme_list(), inventory_list()
    for name, calls in expected.items():
        assert calls <= in_readme.get(name, set()), (name, sorted(calls - in_readme.get(name, set())))
        assert calls <= in_inventory.get(name, set()), (name, sorted(calls - in_inventory.get(name, set())))


def test_the_readme_lists_langchain_embeddings_from_other_vendors():
    expected = {"Embeddings.embed_documents", "Embeddings.embed_query",
                "Embeddings.aembed_documents", "Embeddings.aembed_query"}
    assert expected <= readme_list().get("LangChain", set())
    assert expected <= inventory_list().get("LangChain", set())


def test_the_readme_no_longer_offers_wrap_for_embeddings_models():
    assert "(or embeddings model)" not in README
    assert "**Embeddings models:**" in README
    assert "LangChain embeddings models run no callbacks: `OpenAIEmbeddings` and `AzureOpenAIEmbeddings` are checked" in README
