import pytest
from unittest.mock import patch
import ollama
from ollama._client import Client
import time


def _find_deprecation_warning(mock_logger, field_name):
    for call in mock_logger.warning.call_args_list:
        args = call[0]
        if len(args) >= 3 and "deprecated" in args[0].lower():
            formatted = args[0] % args[1:]
            if field_name in formatted:
                return formatted
    return None


@pytest.mark.unit
class TestLegacyFieldNames:

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_organization_id_camelcase(self, mock_client, mock_ollama_chat_response):
        mock_client.ai.create_completion.return_value = {"status": "success"}

        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware._core.fields.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationId': 'legacy-org-camel',
                        'productName': 'test-product'
                    }
                )

                time.sleep(0.2)

                assert _find_deprecation_warning(mock_logger, 'organizationId')

                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['organization_name'] == 'legacy-org-camel'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_organization_id_snakecase(self, mock_client, mock_ollama_chat_response):
        mock_client.ai.create_completion.return_value = {"status": "success"}

        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware._core.fields.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organization_id': 'legacy-org-snake',
                        'productName': 'test-product'
                    }
                )

                time.sleep(0.2)

                assert _find_deprecation_warning(mock_logger, 'organization_id')

                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['organization_name'] == 'legacy-org-snake'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_product_id_camelcase(self, mock_client, mock_ollama_chat_response):
        mock_client.ai.create_completion.return_value = {"status": "success"}

        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware._core.fields.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationName': 'test-org',
                        'productId': 'legacy-product-camel'
                    }
                )

                time.sleep(0.2)

                assert _find_deprecation_warning(mock_logger, 'productId')

                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['product_name'] == 'legacy-product-camel'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_legacy_product_id_snakecase(self, mock_client, mock_ollama_chat_response):
        mock_client.ai.create_completion.return_value = {"status": "success"}

        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware._core.fields.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationName': 'test-org',
                        'product_id': 'legacy-product-snake'
                    }
                )

                time.sleep(0.2)

                assert _find_deprecation_warning(mock_logger, 'product_id')

                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['product_name'] == 'legacy-product-snake'

    @patch('revenium_middleware.ollama.middleware.client')
    def test_new_fields_take_precedence_over_legacy(self, mock_client, mock_ollama_chat_response):
        mock_client.ai.create_completion.return_value = {"status": "success"}

        with patch.object(Client, '_request', return_value=mock_ollama_chat_response):
            with patch('revenium_middleware._core.fields.logger') as mock_logger:
                ollama.chat(
                    model='llama2',
                    messages=[{'role': 'user', 'content': 'test'}],
                    usage_metadata={
                        'organizationName': 'new-org',
                        'organizationId': 'old-org',
                        'productName': 'new-product',
                        'productId': 'old-product'
                    }
                )

                time.sleep(0.2)

                assert not _find_deprecation_warning(mock_logger, 'organizationId')
                assert not _find_deprecation_warning(mock_logger, 'productId')

                assert mock_client.ai.create_completion.called
                call_args = mock_client.ai.create_completion.call_args[1]
                assert call_args['organization_name'] == 'new-org'
                assert call_args['product_name'] == 'new-product'
