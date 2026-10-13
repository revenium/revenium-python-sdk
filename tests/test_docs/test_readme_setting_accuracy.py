"""README sentences about settings, keys and cap refusals that must match what the SDK does."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
AUDIT_NOTES_PATH = REPO_ROOT / "docs" / "conventions" / "sdk-audit-notes.md"


def _section(text, heading):
    """Lines from the exact heading line to the next heading of the same or higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    start = lines.index(heading)
    collected = [lines[start]]
    in_fence = False
    for line in lines[start + 1:]:
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            marks = len(line) - len(line.lstrip("#"))
            if marks <= level:
                break
        collected.append(line)
    return "\n".join(collected)


def test_the_readme_never_advertises_the_removed_terminal_summary():
    assert "Terminal Summary" not in README


def test_the_audit_notes_say_the_summary_printer_pointers_are_historical():
    if not AUDIT_NOTES_PATH.exists():
        pytest.skip("docs/ is internal-only; the public export copies tests/ without it")
    section = _section(AUDIT_NOTES_PATH.read_text(encoding="utf-8"), "## Audit methodology")
    flat = " ".join(section.split())
    assert "summary_printer.py" in flat
    assert "0.1.6" in flat
    assert "historical" in flat
    assert "remains supported" not in flat


def test_the_key_rule_says_a_tool_event_keeps_the_key_from_when_it_started():
    section = _section(README, "#### Metering Several Customers in One Process")
    assert "when the tool finished" not in section
    flat = " ".join(section.split())
    assert "`@meter_tool` tool events" in flat
    other_records = flat.split("Every other record")[1].split("one started while metering was off")[0]
    assert "@meter_tool" in other_records
    assert "keeps the key that was current when it started" in other_records
    assert "fails with a provider error is also filed under the key from when it started" in flat
    assert "wrap(ChatOllama(...))" in flat


def test_the_key_rule_names_the_non_streamed_calls_that_keep_the_key_from_when_they_returned():
    section = _section(README, "#### Metering Several Customers in One Process")
    assert "Bedrock fast path" not in section
    assert "Every other record keeps the key that was current when it was" not in section
    flat = " ".join(section.split())
    sentences = [s for s in flat.split(". ") if "when the call returns" in s]
    assert sentences
    for name in ("embeddings", "LiteLLM", "Ollama", "Perplexity"):
        assert name in sentences[0]


def test_the_exception_contract_says_batch_completion_returns_refusals_in_its_list():
    section = _section(README, "### Exception Contract")
    assert "batch_completion" in section
    assert "isinstance(r, BudgetExceededError)" in section
    assert "always reaches your" not in section


def test_every_bedrock_disable_entry_says_it_leaves_the_boto3_transport_and_langchain_bedrock_alone():
    entries = [
        line for line in README.splitlines()
        if "REVENIUM_BEDROCK_DISABLE" in line
        and (line.startswith("|") or line.startswith("**Stop Bedrock detection"))
    ]
    assert len(entries) == 3
    for entry in entries:
        assert "REVENIUM_BEDROCK_TRANSPORT" in entry
        assert "ChatBedrockConverse" in entry
