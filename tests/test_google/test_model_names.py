"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
The model name a Gemini or Vertex AI record carries.

The Vertex AI classes run in the vertex environment (``.[google-vertex]``),
where vertexai is installed; the Gemini response classes need no provider SDK.
"""

import importlib.util
from types import SimpleNamespace

import pytest

from revenium_middleware.google.common.utils import base_model_name, extract_model_name


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


vertexai_missing = not _module_available("vertexai")


def vertex_middleware():
    from revenium_middleware.google.vertex_ai import middleware
    return middleware


@pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
class TestVertexModelPath:
    @pytest.mark.parametrize("resource_name", [
        "publishers/google/models/gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "google/models/gemini-1.5-pro",
        "gemini-1.5-pro",
    ])
    def test_google_path_prefixes_are_stripped(self, resource_name):
        assert vertex_middleware()._strip_model_path(resource_name) == "gemini-1.5-pro"


@pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
class TestVertexModelObjectName:
    def test_the_private_model_name_wins_and_loses_its_path(self):
        model = SimpleNamespace(_model_name="publishers/google/models/gemini-2.0-flash", model="ignored")

        assert vertex_middleware()._model_name_of(model) == "gemini-2.0-flash"

    def test_empty_and_non_string_attributes_are_skipped(self):
        model = SimpleNamespace(_model_name="", model_name=None, _model_id=42, model_id="text-embedding-005")

        assert vertex_middleware()._model_name_of(model) == "text-embedding-005"

    @pytest.mark.parametrize("representation", [
        "<GenerativeModel model_name='gemini-1.5-flash'>",
        "GenerativeModel(models/gemini-1.5-flash)",
    ])
    def test_the_representation_is_the_last_resort(self, representation):
        class Model:
            def __str__(self):
                return representation

        assert vertex_middleware()._model_name_of(Model()) == "gemini-1.5-flash"

    def test_an_object_with_no_model_name_gives_none(self):
        assert vertex_middleware()._model_name_of(SimpleNamespace()) is None


class TestGeminiResponseModelName:
    def test_model_is_preferred_over_model_version(self):
        response = SimpleNamespace(model="gemini-2.5-flash", model_version="gemini-2.5-flash-001")

        assert extract_model_name(response) == "gemini-2.5-flash"

    def test_model_version_is_used_when_it_is_the_only_name(self):
        response = SimpleNamespace(model_version="gemini-2.5-flash-lite")

        assert extract_model_name(response) == "gemini-2.5-flash-lite"

    def test_the_fallback_is_used_when_the_response_names_no_model(self):
        assert extract_model_name(SimpleNamespace(model_version=None), "gemini-2.0-flash") == "gemini-2.0-flash"

    def test_no_name_and_no_fallback_gives_unknown_model(self):
        assert extract_model_name(SimpleNamespace()) == "unknown-model"

    def test_a_missing_response_uses_the_fallback(self):
        assert extract_model_name(None, "text-embedding-004") == "text-embedding-004"


class TestBaseModelName:
    @pytest.mark.parametrize("name", [
        "veo-3.0-generate-001",
        "models/veo-3.0-generate-001",
        "publishers/google/models/veo-3.0-generate-001",
        "projects/p/locations/l/publishers/google/models/veo-3.0-generate-001",
        "google/veo-3.0-generate-001",
    ])
    def test_one_google_path_prefix_is_removed(self, name):
        assert base_model_name(name) == "veo-3.0-generate-001"

    @pytest.mark.parametrize("name", [
        "tunedModels/x",
        "projects/p/locations/l/endpoints/123",
        "publishers/anthropic/models/claude-x",
        None,
        7,
    ])
    def test_names_that_are_not_a_google_model_path_are_unchanged(self, name):
        assert base_model_name(name) == name
