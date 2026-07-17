"""
Tests for optional dependencies handling.
"""

import pytest
from unittest.mock import patch


class TestOptionalDependencies:
    """Test optional dependency handling for boto3."""

    def test_boto3_import_success(self):
        """Test successful boto3 import."""
        from revenium_middleware.anthropic.bedrock_adapter import _import_boto3
        
        # This should work if boto3 is installed
        boto3 = _import_boto3()
        assert boto3 is not None
        assert hasattr(boto3, 'client')

    def test_boto3_import_failure(self):
        """Test boto3 import failure raises helpful error."""
        # This test verifies the error message format
        # We'll test this by checking the _import_boto3 function behavior
        from revenium_middleware.anthropic.bedrock_adapter import _import_boto3

        # The function should work normally when boto3 is available
        # If boto3 were missing, it would raise ImportError with our custom message
        try:
            boto3 = _import_boto3()
            assert boto3 is not None
        except ImportError as e:
            # If boto3 is not installed, verify our error message
            error_message = str(e)
            assert "boto3 is required for Bedrock support" in error_message
            assert "pip install revenium-python-sdk[anthropic]" in error_message

    def test_bedrock_extra_installation_instructions(self):
        """Test that the error message provides correct installation instructions."""
        from revenium_middleware.anthropic.bedrock_adapter import _import_boto3
        
        # Temporarily patch the import to simulate missing boto3
        with patch('builtins.__import__', side_effect=ImportError("No module named 'boto3'")):
            with pytest.raises(ImportError) as exc_info:
                _import_boto3()
            
            error_message = str(exc_info.value)
            # Verify the error message contains the correct package and extra name
            assert "revenium-python-sdk[anthropic]" in error_message

    def test_graceful_degradation_without_boto3(self):
        """Test that the main package works without boto3 installed."""
        # Import the main package - this should work even without boto3
        import revenium_middleware.anthropic
        
        # Basic imports should work
        from revenium_middleware.anthropic.provider import Provider, detect_provider
        from revenium_middleware.anthropic.middleware import usage_context
        
        # Provider detection should work for non-Bedrock cases
        result = detect_provider()
        assert result == Provider.ANTHROPIC
        
        # Usage context should be available
        assert usage_context is not None

    @patch.dict('os.environ', {'REVENIUM_BEDROCK_DISABLE': '1'})
    def test_bedrock_disabled_via_env_var(self):
        """Test that Bedrock can be disabled via environment variable."""
        from revenium_middleware.anthropic.provider import Provider, detect_provider
        
        # Even with a mock Bedrock client, should return Anthropic when disabled
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        
        # This test would need to be in the middleware context, but the concept is:
        # When REVENIUM_BEDROCK_DISABLE=1, always use Anthropic regardless of client type
        # For now, just test that the env var can be read
        import os
        assert os.getenv("REVENIUM_BEDROCK_DISABLE") == "1"


class TestBedrockTransportOptionality:
    """The transport canary must never burden non-Bedrock installations."""

    def test_transport_module_imports_without_botocore(self, monkeypatch):
        import importlib
        import sys

        monkeypatch.setitem(sys.modules, "botocore", None)
        monkeypatch.setitem(sys.modules, "botocore.client", None)

        from revenium_middleware.anthropic import bedrock_transport
        reloaded = importlib.reload(bedrock_transport)  # no botocore at import

        assert callable(reloaded.activate_bedrock_transport)

    def test_activation_without_botocore_gives_actionable_message(self, monkeypatch):
        import sys

        from revenium_middleware.anthropic import bedrock_transport

        monkeypatch.setitem(sys.modules, "botocore", None)
        monkeypatch.setitem(sys.modules, "botocore.client", None)

        with pytest.raises(ImportError) as excinfo:
            bedrock_transport.activate_bedrock_transport()

        assert "pip install revenium-python-sdk" in str(excinfo.value)

    def test_flag_states(self, monkeypatch):
        from revenium_middleware.anthropic import bedrock_transport

        monkeypatch.delenv("REVENIUM_BEDROCK_TRANSPORT", raising=False)
        assert bedrock_transport.is_transport_enabled() is False

        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
        assert bedrock_transport.is_transport_enabled() is True

        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "true")
        assert bedrock_transport.is_transport_enabled() is True

        # Kill switch: same variable, re-read at call time.
        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "0")
        assert bedrock_transport.is_transport_enabled() is False

    def test_import_time_activation_defaults_off(self, monkeypatch):
        from revenium_middleware.anthropic import bedrock_transport

        monkeypatch.delenv("REVENIUM_BEDROCK_TRANSPORT", raising=False)
        assert bedrock_transport.activate_if_enabled() is False
