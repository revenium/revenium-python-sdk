"""
Azure OpenAI model name resolution.

This module maps Azure deployment names to LiteLLM-compatible model names
for accurate pricing. Uses heuristic pattern matching with optional API fallback.

The resolution is fire-and-forget - failures cannot break primary AI calls.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any
import re

from .config import Config
from .exceptions import ModelResolutionError, handle_exception_safely

logger = logging.getLogger("revenium_middleware.extension")

# Global model name cache - deployment_name -> litellm_model_name
_azure_model_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()


def resolve_azure_model_name(deployment_name: str, base_url: Optional[str] = None, 
                           headers: Optional[Dict[str, str]] = None, 
                           use_api_fallback: bool = True) -> str:
    """
    Resolve Azure deployment name to LiteLLM-compatible model name.
    
    This function is fire-and-forget - it will never raise exceptions that
    could break the primary AI call.
    
    Args:
        deployment_name: Azure deployment name from API response
        base_url: Azure endpoint URL (for API fallback)
        headers: Request headers (for API fallback) 
        use_api_fallback: Whether to attempt API resolution after heuristics
        
    Returns:
        LiteLLM model name or original deployment_name if no match found
    """
    logger.debug(f"Resolving Azure model name for deployment: {deployment_name}")
    
    try:
        # 1. Check in-memory cache first
        with _cache_lock:
            if deployment_name in _azure_model_cache:
                cached_name = _azure_model_cache[deployment_name]
                logger.debug(f"Found cached mapping: {deployment_name} -> {cached_name}")
                return cached_name
        
        # 2. Try heuristic matching first (fast, reliable for common cases)
        heuristic_name = _heuristic_model_mapping(deployment_name)
        if heuristic_name:
            logger.debug(f"Heuristic mapping successful: {deployment_name} -> {heuristic_name}")
            with _cache_lock:
                _azure_model_cache[deployment_name] = heuristic_name
            return heuristic_name
        
        # 3. Log warning - heuristics failed
        logger.warning(f"No heuristic match found for Azure deployment '{deployment_name}'. "
                      f"This may result in pricing lookup failures on Revenium backend.")
        
        # 4. Try Azure API fallback if enabled
        if use_api_fallback and base_url and headers:
            logger.debug(f"Attempting Azure API resolution for deployment: {deployment_name}")
            _async_resolve_model_name(deployment_name, base_url, headers)
        
        # 5. Return deployment name as fallback
        logger.warning(f"Using deployment name '{deployment_name}' as model name. "
                      f"Verify this matches LiteLLM pricing tables to ensure accurate cost calculation.")
        return deployment_name
        
    except Exception as e:
        # CRITICAL: Never let this break the main AI call
        logger.warning(f"Azure model resolution failed for '{deployment_name}': {str(e)}. "
                      f"Using deployment name as fallback.")
        return deployment_name


def _heuristic_model_mapping(deployment_name: str) -> Optional[str]:
    """
    Fast pattern matching for common Azure deployment naming patterns.
    
    Based on LiteLLM model names: azure_ai/gpt-4, azure_ai/gpt-35-turbo, etc.
    
    Args:
        deployment_name: Azure deployment name
        
    Returns:
        LiteLLM model name if pattern matched, None otherwise
    """
    # Normalize deployment name for pattern matching
    name_lower = deployment_name.lower().replace('_', '-').replace('.', '-')
    
    # GPT-4o family patterns (highest priority - newest models)
    if re.search(r'gpt-?4o', name_lower) or re.search(r'o4', name_lower):
        if 'mini' in name_lower:
            return 'gpt-4o-mini'
        return 'gpt-4o'
    
    # GPT-4 family patterns
    if re.search(r'gpt-?4', name_lower):
        # Check for specific variants first
        if any(variant in name_lower for variant in ['turbo', '1106', '0125', '0613']):
            return 'gpt-4-turbo'
        if any(variant in name_lower for variant in ['vision', 'v', 'preview']):
            return 'gpt-4-vision-preview'
        if '32k' in name_lower:
            return 'gpt-4-32k'
        # Default GPT-4
        return 'gpt-4'
    
    # GPT-3.5 family patterns  
    if any(pattern in name_lower for pattern in ['gpt-35', 'gpt-3-5', 'gpt3-5', 'gpt35']):
        if '16k' in name_lower:
            return 'gpt-3.5-turbo-16k'
        if 'instruct' in name_lower:
            return 'gpt-3.5-turbo-instruct'
        return 'gpt-3.5-turbo'
    
    # Embedding model patterns
    if 'embedding' in name_lower or 'embed' in name_lower:
        if 'ada-002' in name_lower or 'ada002' in name_lower:
            return 'text-embedding-ada-002'
        if '3-large' in name_lower or 'large' in name_lower:
            return 'text-embedding-3-large'
        if '3-small' in name_lower or 'small' in name_lower:
            return 'text-embedding-3-small'
    
    # DALL-E patterns
    if 'dall' in name_lower or 'dalle' in name_lower:
        if '3' in name_lower:
            return 'dall-e-3'
        if '2' in name_lower:
            return 'dall-e-2'
    
    # Whisper patterns
    if 'whisper' in name_lower:
        return 'whisper-1'
    
    # TTS patterns
    if 'tts' in name_lower:
        if 'hd' in name_lower:
            return 'tts-1-hd'
        return 'tts-1'
    
    # No pattern matched
    logger.debug(f"No heuristic pattern matched for deployment: {deployment_name}")
    return None


def _async_resolve_model_name(deployment_name: str, base_url: str, headers: Dict[str, str]):
    """
    Background Azure API resolution - never blocks main thread.
    Updates cache if successful, silent failure otherwise.
    
    Args:
        deployment_name: Azure deployment name
        base_url: Azure endpoint URL
        headers: Request headers for authentication
    """
    def background_resolve():
        try:
            # Import here to avoid dependency issues for non-Azure users
            import requests
            
            # Construct Azure model info endpoint
            # Format: https://resource.openai.azure.com/openai/deployments/{deployment}?api-version=2024-10-21
            api_url = f"{base_url.rstrip('/')}/openai/deployments/{deployment_name}?api-version={Config.AZURE_API_VERSION_DEFAULT}"
            
            logger.debug(f"Attempting Azure API model resolution: {api_url}")
            
            # Make API call with timeout
            response = requests.get(api_url, headers=headers, timeout=Config.AZURE_MODEL_RESOLUTION_TIMEOUT)
            
            if response.status_code == 200:
                model_info = response.json()
                actual_model = model_info.get('id', deployment_name)
                
                # Update cache
                with _cache_lock:
                    _azure_model_cache[deployment_name] = actual_model
                
                logger.debug(f"Azure API resolved: {deployment_name} -> {actual_model}")
            else:
                logger.debug(f"Azure API resolution failed for {deployment_name}: HTTP {response.status_code}")
                
        except Exception as e:
            # Silent failure - heuristic result already used
            logger.debug(f"Background Azure API resolution failed for {deployment_name}: {str(e)}")
    
    # Fire and forget - daemon thread won't block shutdown
    thread = threading.Thread(target=background_resolve, daemon=True)
    thread.start()


def get_model_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the model name cache.
    
    Returns:
        Dictionary with cache statistics
    """
    with _cache_lock:
        return {
            'cache_size': len(_azure_model_cache),
            'cached_models': dict(_azure_model_cache),
            'cache_hit_rate': 'N/A'  # Could be implemented with counters
        }


def clear_model_cache():
    """Clear the model name cache. Useful for testing."""
    with _cache_lock:
        _azure_model_cache.clear()
    logger.debug("Azure model cache cleared")
