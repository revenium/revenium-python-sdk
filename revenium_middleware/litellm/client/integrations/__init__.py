"""
Framework integrations for Revenium LiteLLM middleware.

This package provides pre-built integrations for popular AI frameworks,
eliminating the need for custom monkey-patching code.

Available integrations:
- CrewAI: Multi-agent AI framework integration

Example:
    >>> from revenium_middleware.litellm.client.integrations.crewai import ReveniumCrewWrapper
    >>> 
    >>> # Wrap your Crew with automatic metadata tracking
    >>> crew = ReveniumCrewWrapper(
    ...     agents=[...],
    ...     tasks=[...],
    ...     organization_id="AcmeCorp",
    ...     subscription_id="82764738",
    ...     product_id="Platinum"
    ... )
    >>> 
    >>> result = crew.kickoff()
"""

__all__ = []

