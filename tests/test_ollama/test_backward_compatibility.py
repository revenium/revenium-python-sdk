"""
Tests for backward compatibility with legacy field names.

This module tests that the deprecated field names (organizationId, organization_id,
productId, product_id) still work correctly and trigger appropriate deprecation warnings.
"""

import pytest
from unittest.mock import patch
import ollama
from ollama._client import Client
import time


@pytest.mark.unit
class TestLegacyFieldNames:
    """Test backward compatibility with legacy field names."""

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_organization_id_camelcase(self, mock_client, mock_ollama_chat_response):
        """Test that organizationId (camelCase) still works and triggers deprecation warning."""
        mock_client.ai.create_completion.return_value = {"status": "success"}

        # Mock the actual ollama chat call to return the fixture
        # Mock the underlying _request method to avoid actual Ollama connection
        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware.ollama.middleware.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationId': 'legacy-org-camel',  # Legacy camelCase
                        'productName': 'test-product'
                    }
                )

                # Wait for background metering thread to complete
                time.sleep(0.2)

                # Verify deprecation warning was logged
                assert mock_logger.warning.called
                # Check all warning calls for the deprecation message
                warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
                deprecation_warnings = [msg for msg in warning_messages if 'deprecated' in msg.lower()]
                assert len(deprecation_warnings) > 0, f"No deprecation warning found in: {warning_messages}"
                warning_message = deprecation_warnings[0]
                assert 'organizationId' in warning_message or 'organization_id' in warning_message

                # Verify the field was passed to create_completion
                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['organization_name'] == 'legacy-org-camel'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_organization_id_snakecase(self, mock_client, mock_ollama_chat_response):
        """Test that organization_id (snake_case) still works and triggers deprecation warning."""
        mock_client.ai.create_completion.return_value = {"status": "success"}

        # Mock the actual ollama chat call to return the fixture
        # Mock the underlying _request method to avoid actual Ollama connection
        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware.ollama.middleware.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organization_id': 'legacy-org-snake',  # Legacy snake_case
                        'productName': 'test-product'
                    }
                )

                # Wait for background metering thread to complete
                time.sleep(0.2)

                # Verify deprecation warning was logged
                assert mock_logger.warning.called
                # Check all warning calls for the deprecation message
                warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
                deprecation_warnings = [msg for msg in warning_messages if 'deprecated' in msg.lower()]
                assert len(deprecation_warnings) > 0, f"No deprecation warning found in: {warning_messages}"
                warning_message = deprecation_warnings[0]
                assert 'organizationId' in warning_message or 'organization_id' in warning_message

                # Verify the field was passed to create_completion
                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['organization_name'] == 'legacy-org-snake'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_product_id_camelcase(self, mock_client, mock_ollama_chat_response):
        """Test that productId (camelCase) still works and triggers deprecation warning."""
        mock_client.ai.create_completion.return_value = {"status": "success"}

        # Mock the actual ollama chat call to return the fixture
        # Mock the underlying _request method to avoid actual Ollama connection
        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware.ollama.middleware.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationName': 'test-org',
                        'productId': 'legacy-product-camel'  # Legacy camelCase
                    }
                )

                # Wait for background metering thread to complete
                time.sleep(0.2)

                # Verify deprecation warning was logged
                assert mock_logger.warning.called
                # Check all warning calls for the deprecation message
                warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
                deprecation_warnings = [msg for msg in warning_messages if 'deprecated' in msg.lower()]
                assert len(deprecation_warnings) > 0, f"No deprecation warning found in: {warning_messages}"
                warning_message = deprecation_warnings[0]
                assert 'productId' in warning_message or 'product_id' in warning_message

                # Verify the field was passed to create_completion
                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['product_name'] == 'legacy-product-camel'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_product_id_snakecase(self, mock_client, mock_ollama_chat_response):
        """Test that product_id (snake_case) still works and triggers deprecation warning."""
        mock_client.ai.create_completion.return_value = {"status": "success"}

        # Mock the actual ollama chat call to return the fixture
        # Mock the underlying _request method to avoid actual Ollama connection
        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware.ollama.middleware.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationName': 'test-org',
                        'product_id': 'legacy-product-snake'  # Legacy snake_case
                    }
                )

                # Wait for background metering thread to complete
                time.sleep(0.2)

                # Verify deprecation warning was logged
                assert mock_logger.warning.called
                # Check all warning calls for the deprecation message
                warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
                deprecation_warnings = [msg for msg in warning_messages if 'deprecated' in msg.lower()]
                assert len(deprecation_warnings) > 0, f"No deprecation warning found in: {warning_messages}"
                warning_message = deprecation_warnings[0]
                assert 'productId' in warning_message or 'product_id' in warning_message

                # Verify the field was passed to create_completion
                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['product_name'] == 'legacy-product-snake'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_new_fields_take_precedence_over_legacy(self, mock_client, mock_ollama_chat_response):
        """Test that new field names take precedence over legacy names."""
        mock_client.ai.create_completion.return_value = {"status": "success"}

        # Mock the actual ollama chat call to return the fixture
        # Mock the underlying _request method to avoid actual Ollama connection
        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware.ollama.middleware.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationName': 'new-org',
                        'organizationId': 'old-org',  # Should be ignored
                        'productName': 'new-product',
                        'productId': 'old-product'  # Should be ignored
                    }
                )

                # Wait for background metering thread to complete
                time.sleep(0.2)

                # Verify NO deprecation warning when new fields are present
                # There might be other warnings, but no deprecation warnings
                if mock_logger.warning.called:
                    warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
                    deprecation_warnings = [msg for msg in warning_messages if 'deprecated' in msg.lower()]
                    assert len(deprecation_warnings) == 0, f"Unexpected deprecation warning: {deprecation_warnings}"

                # Verify new field values were used
                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['organization_name'] == 'new-org'
                assert call_args['product_name'] == 'new-product'

