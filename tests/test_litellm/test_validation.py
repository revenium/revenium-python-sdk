"""
Tests for Pydantic validation models.

Tests cover:
- Valid metadata creation
- Invalid data type handling
- Field validation (e.g., quality score range)
- Nested models (Subscriber, SubscriberCredential)
- to_dict() method
- Fallback behavior when Pydantic not installed
"""

import pytest
from revenium_middleware.litellm.client.validation import (
    UsageMetadata,
    Subscriber,
    SubscriberCredential,
    PYDANTIC_AVAILABLE
)


# Only run Pydantic tests if it's available
pytestmark = pytest.mark.skipif(
    not PYDANTIC_AVAILABLE,
    reason="Pydantic not installed"
)


class TestUsageMetadata:
    """Test suite for UsageMetadata model."""
    
    def test_create_empty(self):
        """Test creating empty metadata."""
        metadata = UsageMetadata()
        assert metadata.to_dict() == {}
    
    def test_create_with_basic_fields(self):
        """Test creating metadata with basic fields."""
        metadata = UsageMetadata(
            agent="Test Agent",
            task_type="research",
            trace_id="abc-123"
        )
        
        result = metadata.to_dict()
        assert result == {
            "agent": "Test Agent",
            "task_type": "research",
            "trace_id": "abc-123"
        }
    
    def test_create_with_all_fields(self):
        """Test creating metadata with all fields."""
        metadata = UsageMetadata(
            organization_id="AcmeCorp",
            subscription_id="82764738",
            product_id="Platinum",
            trace_id="abc-123",
            agent="Lead Analyst",
            task_type="market_research",
            response_quality_score=0.95
        )
        
        result = metadata.to_dict()
        assert result["organization_id"] == "AcmeCorp"
        assert result["subscription_id"] == "82764738"
        assert result["product_id"] == "Platinum"
        assert result["trace_id"] == "abc-123"
        assert result["agent"] == "Lead Analyst"
        assert result["task_type"] == "market_research"
        assert result["response_quality_score"] == 0.95
    
    def test_quality_score_valid_range(self):
        """Test that valid quality scores are accepted."""
        # Test boundary values
        metadata1 = UsageMetadata(response_quality_score=0.0)
        assert metadata1.response_quality_score == 0.0
        
        metadata2 = UsageMetadata(response_quality_score=1.0)
        assert metadata2.response_quality_score == 1.0
        
        metadata3 = UsageMetadata(response_quality_score=0.5)
        assert metadata3.response_quality_score == 0.5
    
    def test_quality_score_invalid_too_low(self):
        """Test that quality score below 0 is rejected."""
        if PYDANTIC_AVAILABLE:
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                UsageMetadata(response_quality_score=-0.1)
    
    def test_quality_score_invalid_too_high(self):
        """Test that quality score above 1 is rejected."""
        if PYDANTIC_AVAILABLE:
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                UsageMetadata(response_quality_score=1.1)
    
    def test_to_dict_excludes_none(self):
        """Test that to_dict() excludes None values."""
        metadata = UsageMetadata(
            agent="Test Agent",
            task_type=None,  # Explicitly set to None
            trace_id="abc-123"
        )
        
        result = metadata.to_dict()
        assert "task_type" not in result
        assert result == {
            "agent": "Test Agent",
            "trace_id": "abc-123"
        }
    
    def test_extra_fields_allowed(self):
        """Test that extra custom fields are allowed."""
        metadata = UsageMetadata(
            agent="Test Agent",
            custom_field="custom_value",
            another_custom="another_value"
        )
        
        result = metadata.to_dict()
        assert result["agent"] == "Test Agent"
        assert result["custom_field"] == "custom_value"
        assert result["another_custom"] == "another_value"


class TestSubscriber:
    """Test suite for Subscriber model."""
    
    def test_create_empty(self):
        """Test creating empty subscriber."""
        subscriber = Subscriber()
        assert subscriber.id is None
        assert subscriber.email is None
        assert subscriber.credential is None
    
    def test_create_with_id_and_email(self):
        """Test creating subscriber with ID and email."""
        subscriber = Subscriber(
            id="user-123",
            email="user@example.com"
        )
        
        assert subscriber.id == "user-123"
        assert subscriber.email == "user@example.com"
    
    def test_create_with_credential(self):
        """Test creating subscriber with credential."""
        credential = SubscriberCredential(
            name="api_key_alias",
            value="sk-abc123"
        )
        subscriber = Subscriber(
            id="user-123",
            credential=credential
        )
        
        assert subscriber.id == "user-123"
        assert subscriber.credential.name == "api_key_alias"
        assert subscriber.credential.value == "sk-abc123"
    
    def test_no_extra_fields(self):
        """Test that extra fields are forbidden in Subscriber."""
        if PYDANTIC_AVAILABLE:
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                Subscriber(
                    id="user-123",
                    invalid_field="should_fail"
                )


class TestSubscriberCredential:
    """Test suite for SubscriberCredential model."""
    
    def test_create_empty(self):
        """Test creating empty credential."""
        credential = SubscriberCredential()
        assert credential.name is None
        assert credential.value is None
    
    def test_create_with_name_and_value(self):
        """Test creating credential with name and value."""
        credential = SubscriberCredential(
            name="api_key_alias",
            value="sk-abc123"
        )
        
        assert credential.name == "api_key_alias"
        assert credential.value == "sk-abc123"
    
    def test_no_extra_fields(self):
        """Test that extra fields are forbidden in SubscriberCredential."""
        if PYDANTIC_AVAILABLE:
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                SubscriberCredential(
                    name="api_key",
                    value="sk-123",
                    invalid_field="should_fail"
                )


class TestNestedModels:
    """Test suite for nested model relationships."""
    
    def test_usage_metadata_with_subscriber(self):
        """Test UsageMetadata with nested Subscriber."""
        subscriber = Subscriber(
            id="user-123",
            email="user@example.com"
        )
        
        metadata = UsageMetadata(
            agent="Test Agent",
            subscriber=subscriber
        )
        
        result = metadata.to_dict()
        assert result["agent"] == "Test Agent"
        assert result["subscriber"]["id"] == "user-123"
        assert result["subscriber"]["email"] == "user@example.com"
    
    def test_usage_metadata_with_full_subscriber(self):
        """Test UsageMetadata with fully populated Subscriber."""
        credential = SubscriberCredential(
            name="api_key_alias",
            value="sk-abc123"
        )
        subscriber = Subscriber(
            id="user-123",
            email="user@example.com",
            credential=credential
        )
        
        metadata = UsageMetadata(
            organization_id="AcmeCorp",
            agent="Test Agent",
            subscriber=subscriber
        )
        
        result = metadata.to_dict()
        assert result["organization_id"] == "AcmeCorp"
        assert result["agent"] == "Test Agent"
        assert result["subscriber"]["id"] == "user-123"
        assert result["subscriber"]["email"] == "user@example.com"
        assert result["subscriber"]["credential"]["name"] == "api_key_alias"
        assert result["subscriber"]["credential"]["value"] == "sk-abc123"


class TestFallbackBehavior:
    """Test fallback behavior when Pydantic is not available."""
    
    @pytest.mark.skipif(PYDANTIC_AVAILABLE, reason="Only test fallback when Pydantic not available")
    def test_fallback_usage_metadata(self):
        """Test that fallback UsageMetadata works without Pydantic."""
        metadata = UsageMetadata(
            agent="Test Agent",
            task_type="research"
        )
        
        result = metadata.to_dict()
        assert result["agent"] == "Test Agent"
        assert result["task_type"] == "research"
    
    @pytest.mark.skipif(PYDANTIC_AVAILABLE, reason="Only test fallback when Pydantic not available")
    def test_fallback_no_validation(self):
        """Test that fallback doesn't validate (accepts any values)."""
        # This should not raise an error even with invalid quality score
        metadata = UsageMetadata(
            response_quality_score=5.0  # Invalid, but no validation
        )
        
        result = metadata.to_dict()
        assert result["response_quality_score"] == 5.0

