"""
Azure OpenAI configuration management.

This module handles Azure-specific environment variables and configuration
according to F2 requirements. Configuration is loaded lazily only when
Azure mode is detected to avoid performance impact for non-Azure users.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("revenium_middleware.extension")


class AzureConfig:
    """
    Azure OpenAI configuration manager.
    
    Handles Azure-specific environment variables and provides
    configuration validation and header generation.
    """
    
    def __init__(self):
        """Initialize Azure configuration from environment variables."""
        self.endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        self.deployment = os.environ.get('AZURE_OPENAI_DEPLOYMENT')
        self.api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-10-21')
        self.api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        
        # Future AAD support
        self.tenant_id = os.environ.get('AZURE_OPENAI_TENANT_ID')
        self.resource_group = os.environ.get('AZURE_OPENAI_RESOURCE_GROUP')
        
        logger.debug(f"Azure config initialized - endpoint: {self.endpoint}, "
                    f"deployment: {self.deployment}, api_version: {self.api_version}")
    
    def is_valid(self) -> bool:
        """
        Check if the minimum required configuration is present.
        
        Returns:
            True if configuration is valid for Azure OpenAI usage
        """
        # Minimum requirement is endpoint
        is_valid = bool(self.endpoint)
        
        if not is_valid:
            logger.debug("Azure config invalid - missing AZURE_OPENAI_ENDPOINT")
        
        return is_valid
    
    def get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """
        Generate appropriate headers for Azure OpenAI requests.
        
        Args:
            api_key: Optional API key override
            
        Returns:
            Dictionary of headers for Azure OpenAI requests
        """
        headers = {}
        
        # Use provided key or fall back to config
        key_to_use = api_key or self.api_key
        
        if key_to_use:
            # Azure OpenAI uses 'api-key' header instead of 'Authorization'
            headers['api-key'] = key_to_use
            logger.debug("Added api-key header for Azure authentication")
        else:
            logger.warning("No API key available for Azure OpenAI authentication")
        
        return headers
    
    def get_base_url(self) -> Optional[str]:
        """
        Get the base URL for Azure OpenAI API calls.
        
        Returns:
            Base URL or None if not configured
        """
        if self.endpoint:
            # Ensure endpoint ends with /openai for API calls
            base_url = self.endpoint.rstrip('/')
            if not base_url.endswith('/openai'):
                base_url += '/openai'
            return base_url
        return None
    
    def get_deployment_url(self, endpoint_path: str) -> Optional[str]:
        """
        Construct full deployment URL for Azure OpenAI API calls.
        
        Args:
            endpoint_path: API endpoint path (e.g., '/chat/completions')
            
        Returns:
            Full URL with deployment and API version
        """
        if not self.is_valid() or not self.deployment:
            return None
        
        base_url = self.get_base_url()
        if not base_url:
            return None
        
        # Construct: https://resource.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21
        url = f"{base_url}/deployments/{self.deployment}{endpoint_path}?api-version={self.api_version}"
        logger.debug(f"Constructed Azure deployment URL: {url}")
        return url
    
    def validate_deployment(self) -> bool:
        """
        Validate that deployment name is configured.
        
        Returns:
            True if deployment is configured
        """
        if not self.deployment:
            logger.warning("AZURE_OPENAI_DEPLOYMENT not configured - this may cause API calls to fail")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary (excluding sensitive data).
        
        Returns:
            Configuration dictionary for logging/debugging
        """
        return {
            'endpoint': self.endpoint,
            'deployment': self.deployment,
            'api_version': self.api_version,
            'tenant_id': self.tenant_id,
            'resource_group': self.resource_group,
            'has_api_key': bool(self.api_key),
            'is_valid': self.is_valid()
        }


# Global configuration instance - lazy loaded
_azure_config: Optional[AzureConfig] = None


def get_azure_config() -> AzureConfig:
    """
    Get or create Azure configuration instance.
    
    Lazy loading ensures non-Azure users don't pay any performance cost.
    
    Returns:
        AzureConfig instance
    """
    global _azure_config
    
    if _azure_config is None:
        _azure_config = AzureConfig()
        logger.debug("Azure configuration loaded")
    
    return _azure_config


def reset_azure_config():
    """Reset Azure configuration cache. Useful for testing."""
    global _azure_config
    _azure_config = None
