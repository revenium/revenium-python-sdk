"""
Pydantic models for validating usage metadata in Revenium LiteLLM middleware.

This module provides type-safe validation for metadata fields using Pydantic.
It helps catch errors early in development and provides IDE autocomplete support.

Example:
    >>> from revenium_middleware.litellm.client import UsageMetadata
    >>> 
    >>> # Valid metadata
    >>> metadata = UsageMetadata(
    ...     organization_id="AcmeCorp",
    ...     subscription_id="82764738",
    ...     product_id="Platinum",
    ...     trace_id="abc-123",
    ...     agent="Lead Analyst",
    ...     task_type="research"
    ... )
    >>> 
    >>> # Type errors caught immediately
    >>> try:
    ...     bad_metadata = UsageMetadata(organization_id=12345)  # Wrong type
    ... except ValidationError as e:
    ...     print(e)
"""

try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    from typing import Optional, Dict, Any
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Pydantic is optional - provide fallback
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore


if PYDANTIC_AVAILABLE:
    class SubscriberCredential(BaseModel):
        """
        Credential information for a subscriber.

        Attributes:
            name: An alias for an API key used by one or more users
            value: The key value associated with the subscriber (e.g., an API key)
        """
        model_config = ConfigDict(extra="forbid")

        name: Optional[str] = Field(
            None,
            description="An alias for an API key used by one or more users"
        )
        value: Optional[str] = Field(
            None,
            description="The key value associated with the subscriber (e.g., an API key)"
        )


    class Subscriber(BaseModel):
        """
        Subscriber information for tracking individual users.

        Attributes:
            id: The ID of the subscriber from non-Revenium systems
            email: The email address of the subscriber
            credential: Credential information (API key, etc.)
        """
        model_config = ConfigDict(extra="forbid")

        id: Optional[str] = Field(
            None,
            description="The ID of the subscriber from non-Revenium systems"
        )
        email: Optional[str] = Field(
            None,
            description="The email address of the subscriber"
        )
        credential: Optional[SubscriberCredential] = Field(
            None,
            description="Credential information for the subscriber"
        )


    class UsageMetadata(BaseModel):
        """
        Complete usage metadata for Revenium metering.

        All fields are optional. Adding them enables more detailed reporting
        and analytics in Revenium.

        Attributes:
            trace_id: Unique identifier for a conversation or session
            task_type: Classification of the AI operation by type of work
            subscriber: Object containing subscriber information
            organization_id: Customer or department ID from non-Revenium systems
            subscription_id: Reference to a billing plan in non-Revenium systems
            product_id: Your product or feature making the AI call
            agent: Identifier for the specific AI agent
            response_quality_score: The quality of the AI response (0.0 to 1.0)

        Example:
            >>> metadata = UsageMetadata(
            ...     organization_id="AcmeCorp",
            ...     subscription_id="82764738",
            ...     product_id="Platinum",
            ...     trace_id="abc-123",
            ...     agent="Lead Analyst",
            ...     task_type="market_research",
            ...     response_quality_score=0.95
            ... )
            >>> metadata_dict = metadata.model_dump(exclude_none=True)
        """
        model_config = ConfigDict(extra="allow")  # Allow additional custom fields

        trace_id: Optional[str] = Field(
            None,
            description="Unique identifier for a conversation or session"
        )
        task_type: Optional[str] = Field(
            None,
            description="Classification of the AI operation by type of work"
        )
        subscriber: Optional[Subscriber] = Field(
            None,
            description="Object containing subscriber information"
        )
        organization_id: Optional[str] = Field(
            None,
            description="Customer or department ID from non-Revenium systems"
        )
        subscription_id: Optional[str] = Field(
            None,
            description="Reference to a billing plan in non-Revenium systems"
        )
        product_id: Optional[str] = Field(
            None,
            description="Your product or feature making the AI call"
        )
        agent: Optional[str] = Field(
            None,
            description="Identifier for the specific AI agent"
        )
        response_quality_score: Optional[float] = Field(
            None,
            ge=0.0,
            le=1.0,
            description="The quality of the AI response (0.0 to 1.0)"
        )

        @field_validator('response_quality_score')
        @classmethod
        def validate_quality_score(cls, v: Optional[float]) -> Optional[float]:
            """Validate that quality score is between 0 and 1."""
            if v is not None and (v < 0.0 or v > 1.0):
                raise ValueError('response_quality_score must be between 0.0 and 1.0')
            return v

        def to_dict(self) -> Dict[str, Any]:
            """
            Convert to dictionary, excluding None values.
            
            Returns:
                Dict with only non-None values
            """
            return self.model_dump(exclude_none=True)

else:
    # Fallback when Pydantic is not installed
    class UsageMetadata:  # type: ignore
        """
        Fallback UsageMetadata class when Pydantic is not installed.
        
        This provides basic functionality without validation. For full
        type safety and validation, install pydantic:
        
            pip install "revenium-python-sdk[litellm]"
        """
        
        def __init__(self, **kwargs):
            """Initialize with any keyword arguments."""
            self.__dict__.update(kwargs)
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary, excluding None values."""
            return {k: v for k, v in self.__dict__.items() if v is not None}
    
    class Subscriber:  # type: ignore
        """Fallback Subscriber class when Pydantic is not installed."""
        
        def __init__(self, **kwargs):
            """Initialize with any keyword arguments."""
            self.__dict__.update(kwargs)
    
    class SubscriberCredential:  # type: ignore
        """Fallback SubscriberCredential class when Pydantic is not installed."""
        
        def __init__(self, **kwargs):
            """Initialize with any keyword arguments."""
            self.__dict__.update(kwargs)


__all__ = [
    'UsageMetadata',
    'Subscriber', 
    'SubscriberCredential',
    'PYDANTIC_AVAILABLE'
]

