"""The README's LangChain section names only credential routes that meter AI calls (BACK-4246).

It offered ``revenium_middleware.configure()`` for the key. configure() sets the
key for tool events only, so a customer who followed it got no usage records.
"""
import re
from pathlib import Path

README = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")


def _langchain_section():
    start = README.index("### LangChain")
    return README[start:README.index("\n### ", start + 1)]


def _credentials_sentence():
    sentences = re.findall(r"Credentials come from[^\n]*(?:\n[^\n]+)*", _langchain_section())
    assert len(sentences) == 1, sentences
    return sentences[0]


def test_langchain_section_names_only_routes_that_meter_ai_calls():
    sentence = _credentials_sentence()

    assert "initialize_metering(" in sentence
    routes = sentence.split(". ")[0]
    assert "configure()" not in routes
