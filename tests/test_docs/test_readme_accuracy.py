"""README and CHANGELOG sentences that state what the SDK records, masks or supports."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_LINES = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
CHANGELOG_LINES = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()


def _readme_section(heading):
    start = README_LINES.index(heading)
    end = next(
        (i for i in range(start + 1, len(README_LINES)) if README_LINES[i].startswith(("## ", "### "))),
        len(README_LINES),
    )
    return "\n".join(README_LINES[start:end])


def _newest_changelog_section():
    start = next(i for i, line in enumerate(CHANGELOG_LINES) if line.startswith("## ["))
    end = next(i for i in range(start + 1, len(CHANGELOG_LINES)) if CHANGELOG_LINES[i].startswith("## ["))
    return "\n".join(CHANGELOG_LINES[start:end])


def test_the_background_response_footnote_names_the_streaming_response_retrieve():
    footnote = next(line for line in README_LINES if line.startswith("[^openai-background]:"))
    assert "with_streaming_response.retrieve" in footnote


def test_the_newest_changelog_section_never_says_anthropic_bedrock_beta_calls_are_unmetered():
    assert "neither capped nor metered" not in _newest_changelog_section()


def test_the_anthropic_note_names_the_beta_interface_and_parse_as_metered():
    section = _readme_section("### Anthropic")
    assert "beta.messages" in section
    assert "messages.parse" in section
    assert "Other Anthropic SDK features work normally but aren't metered" not in section


def test_prompt_capture_says_which_json_web_tokens_are_masked():
    section = _readme_section("### What Gets Captured")
    sentence = next((line for line in section.splitlines() if "unsigned" in line), "")
    assert "unsigned" in sentence
    assert "encrypted" in sentence
    assert "signed JSON Web Tokens" in sentence
    assert "three parts" in sentence
    assert "***REDACTED_JWT***" in sentence


def _python_blocks(text):
    blocks, current = [], None
    for line in text.splitlines():
        if line.startswith("```python"):
            current = []
        elif line.startswith("```") and current is not None:
            blocks.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    return blocks


def test_the_google_section_marks_the_vertex_ai_sdk_deprecated_and_links_googles_notice():
    section = _readme_section("### Google AI (Gemini / Vertex AI)")
    assert "https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk" in section
    for module in ("generative_models", "language_models", "vision_models"):
        assert f"`{module}`" in section
    assert any("vertexai=True" in block for block in _python_blocks(section))
    rows = [line for line in section.splitlines() if line.startswith("|") and "Vertex AI SDK" in line]
    assert rows
    assert [row for row in rows if "deprecated" not in row.lower()] == []
