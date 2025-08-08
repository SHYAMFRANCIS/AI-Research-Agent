"""
Custom exception classes for the AI Research Assistant application.

This module defines a hierarchical structure of custom exceptions that provide
specific error handling for different types of failures that can occur in the
research assistant workflow.

Exception Hierarchy:
- AIResearchAssistantError (base)
  ├── ApiQuotaError
  ├── NetworkError
  │   ├── ConnectionTimeoutError
  │   ├── ReadTimeoutError
  │   └── DNSResolutionError
  ├── ParsingError
  │   ├── JSONParsingError
  │   ├── ResponseStructureError
  │   └── DataValidationError
  ├── InputValidationError
  │   ├── RequiredFieldError
  │   ├── InvalidFormatError
  │   └── BusinessRuleError
  └── SystemError
      ├── DatabaseError
      ├── FileSystemError
      └── ThirdPartyServiceError
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta


class AIResearchAssistantError(Exception):
    """
    Base exception class for all AI Research Assistant errors.
    
    Attributes:
        message (str): Human-readable error message
        error_code (str): Unique error code for tracking
        context (Dict[str, Any]): Additional context data
        timestamp (datetime): When the error occurred
        user_message (str): User-friendly message
        recovery_suggestions (list): List of recovery actions
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[list] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
        self.user_message = user_message or "An unexpected error occurred. Please try again."
        self.recovery_suggestions = recovery_suggestions or ["Try again", "Contact support if the issue persists"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "recovery_suggestions": self.recovery_suggestions
        }


# =============================================================================
# API QUOTA AND RATE LIMITING ERRORS
# =============================================================================

class ApiQuotaError(AIResearchAssistantError):
    """Raised when API quota or rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        quota_type: str = "requests",
        quota_limit: Optional[int] = None,
        quota_reset_time: Optional[datetime] = None,
        retry_after_seconds: Optional[int] = None,
        **kwargs
    ):
        self.quota_type = quota_type
        self.quota_limit = quota_limit
        self.quota_reset_time = quota_reset_time
        self.retry_after_seconds = retry_after_seconds
        
        # Generate user-friendly message
        if quota_reset_time:
            reset_str = quota_reset_time.strftime("%I:%M %p %Z")
            user_msg = f"You've reached your {quota_type} limit. Quota resets at {reset_str}."
        elif retry_after_seconds:
            user_msg = f"Rate limit exceeded. Please wait {retry_after_seconds} seconds before trying again."
        else:
            user_msg = f"You've reached your {quota_type} limit. Please try again later."
        
        recovery_suggestions = [
            "Wait for quota reset",
            "Consider upgrading your plan for higher limits",
            "Optimize your request frequency"
        ]
        
        if retry_after_seconds and retry_after_seconds < 300:  # Less than 5 minutes
            recovery_suggestions.insert(0, f"Wait {retry_after_seconds} seconds and retry automatically")
        
        super().__init__(
            message=message,
            error_code="API_QUOTA_EXCEEDED",
            context={
                "quota_type": quota_type,
                "quota_limit": quota_limit,
                "quota_reset_time": quota_reset_time.isoformat() if quota_reset_time else None,
                "retry_after_seconds": retry_after_seconds
            },
            user_message=user_msg,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


# =============================================================================
# NETWORK AND CONNECTIVITY ERRORS
# =============================================================================

class NetworkError(AIResearchAssistantError):
    """Base class for network-related errors."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        self.endpoint = endpoint
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            context={"endpoint": endpoint},
            user_message="Network error occurred. Please check your connection and try again.",
            recovery_suggestions=[
                "Check your internet connection",
                "Try again in a few moments",
                "Contact support if the issue persists"
            ],
            **kwargs
        )


class ConnectionTimeoutError(NetworkError):
    """Raised when connection cannot be established within timeout period."""
    
    def __init__(self, message: str, timeout_seconds: Optional[int] = None, **kwargs):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=message,
            error_code="CONNECTION_TIMEOUT",
            user_message="Unable to connect to the service. Please check your internet connection and try again.",
            recovery_suggestions=[
                "Check your internet connection",
                "Try again with a stronger network connection",
                "Contact your network administrator if the issue persists"
            ],
            **kwargs
        )
        if timeout_seconds:
            self.context["timeout_seconds"] = timeout_seconds


class ReadTimeoutError(NetworkError):
    """Raised when a connection is established but response takes too long."""
    
    def __init__(self, message: str, timeout_seconds: Optional[int] = None, **kwargs):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=message,
            error_code="READ_TIMEOUT",
            user_message="The request is taking longer than expected. Please try again.",
            recovery_suggestions=[
                "Try again - the service may be temporarily slow",
                "Check your internet connection stability",
                "Try again in a few minutes"
            ],
            **kwargs
        )
        if timeout_seconds:
            self.context["timeout_seconds"] = timeout_seconds


class DNSResolutionError(NetworkError):
    """Raised when DNS resolution fails."""
    
    def __init__(self, message: str, hostname: Optional[str] = None, **kwargs):
        self.hostname = hostname
        super().__init__(
            message=message,
            error_code="DNS_RESOLUTION_ERROR",
            user_message="Unable to reach the service. Please check your network connection.",
            recovery_suggestions=[
                "Check your internet connection",
                "Try using a different DNS server",
                "Contact your network administrator"
            ],
            **kwargs
        )
        if hostname:
            self.context["hostname"] = hostname


# =============================================================================
# DATA PARSING AND PROCESSING ERRORS
# =============================================================================

class ParsingError(AIResearchAssistantError):
    """Base class for data parsing and processing errors."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        self.data_source = data_source
        super().__init__(
            message=message,
            error_code="PARSING_ERROR",
            context={"data_source": data_source},
            user_message="Unable to process the response data. Please try again.",
            recovery_suggestions=[
                "Try again - this may be a temporary issue",
                "Refresh the page/application",
                "Contact support if the issue continues"
            ],
            **kwargs
        )


class JSONParsingError(ParsingError):
    """Raised when JSON parsing fails."""
    
    def __init__(self, message: str, json_content: Optional[str] = None, **kwargs):
        self.json_content = json_content
        super().__init__(
            message=message,
            error_code="JSON_PARSING_ERROR",
            user_message="Received invalid data format from the service. Please try again.",
            **kwargs
        )
        # Only store first 200 chars of JSON content for logging
        if json_content:
            self.context["json_content_preview"] = json_content[:200] + "..." if len(json_content) > 200 else json_content


class ResponseStructureError(ParsingError):
    """Raised when response structure doesn't match expected format."""
    
    def __init__(
        self,
        message: str,
        expected_fields: Optional[list] = None,
        missing_fields: Optional[list] = None,
        **kwargs
    ):
        self.expected_fields = expected_fields or []
        self.missing_fields = missing_fields or []
        
        super().__init__(
            message=message,
            error_code="RESPONSE_STRUCTURE_ERROR",
            user_message="The service returned incomplete information. Please try again.",
            **kwargs
        )
        
        self.context.update({
            "expected_fields": self.expected_fields,
            "missing_fields": self.missing_fields
        })


class DataValidationError(ParsingError):
    """Raised when data fails validation checks."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            user_message="Unexpected data format received. Please try again.",
            **kwargs
        )
        
        self.context.update({
            "field_name": field_name,
            "expected_type": expected_type,
            "actual_value": str(actual_value) if actual_value is not None else None
        })


# =============================================================================
# INPUT VALIDATION ERRORS
# =============================================================================

class InputValidationError(AIResearchAssistantError):
    """Base class for input validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        self.field_name = field_name
        super().__init__(
            message=message,
            error_code="INPUT_VALIDATION_ERROR",
            context={"field_name": field_name},
            user_message="Please check your input and try again.",
            recovery_suggestions=[
                "Review your input for any errors",
                "Check the required format",
                "Try again with valid data"
            ],
            **kwargs
        )


class RequiredFieldError(InputValidationError):
    """Raised when a required field is missing."""
    
    def __init__(self, field_name: str, **kwargs):
        message = f"Required field '{field_name}' is missing"
        user_message = f"The field '{field_name}' is required. Please provide a value."
        
        super().__init__(
            message=message,
            field_name=field_name,
            error_code="REQUIRED_FIELD_ERROR",
            user_message=user_message,
            recovery_suggestions=[
                f"Please enter a value for '{field_name}'",
                "All required fields must be filled out"
            ],
            **kwargs
        )


class InvalidFormatError(InputValidationError):
    """Raised when input doesn't match expected format."""
    
    def __init__(
        self,
        field_name: str,
        expected_format: str,
        provided_value: Optional[str] = None,
        **kwargs
    ):
        self.expected_format = expected_format
        self.provided_value = provided_value
        
        message = f"Field '{field_name}' has invalid format. Expected: {expected_format}"
        user_message = f"Please enter a valid {expected_format} for '{field_name}'."
        
        super().__init__(
            message=message,
            field_name=field_name,
            error_code="INVALID_FORMAT_ERROR",
            user_message=user_message,
            recovery_suggestions=[
                f"Use the format: {expected_format}",
                "Check the example format provided",
                "Correct the format and try again"
            ],
            **kwargs
        )
        
        self.context.update({
            "expected_format": expected_format,
            "provided_value": provided_value
        })


class BusinessRuleError(InputValidationError):
    """Raised when input violates business rules."""
    
    def __init__(self, message: str, rule_name: Optional[str] = None, **kwargs):
        self.rule_name = rule_name
        super().__init__(
            message=message,
            error_code="BUSINESS_RULE_ERROR",
            user_message="This input conflicts with existing data. Please choose a different value.",
            recovery_suggestions=[
                "Try a different value",
                "Check for conflicts with existing data",
                "Contact support for assistance"
            ],
            **kwargs
        )
        if rule_name:
            self.context["rule_name"] = rule_name


# =============================================================================
# SYSTEM AND INFRASTRUCTURE ERRORS
# =============================================================================

class SystemError(AIResearchAssistantError):
    """Base class for system and infrastructure errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="SYSTEM_ERROR",
            user_message="We're experiencing technical difficulties. Please try again later.",
            recovery_suggestions=[
                "Try again in a few minutes",
                "Contact support if the issue persists",
                "Check our status page for updates"
            ],
            **kwargs
        )


class DatabaseError(SystemError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        self.operation = operation
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            user_message="Unable to save or retrieve data. Please try again.",
            recovery_suggestions=[
                "Try again in a moment",
                "Check if the data was saved",
                "Contact support if the issue continues"
            ],
            **kwargs
        )
        if operation:
            self.context["operation"] = operation


class FileSystemError(SystemError):
    """Raised when file system operations fail."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.file_path = file_path
        self.operation = operation
        
        super().__init__(
            message=message,
            error_code="FILESYSTEM_ERROR",
            user_message="Unable to save or access files. Please try again.",
            recovery_suggestions=[
                "Check if you have sufficient permissions",
                "Try saving to a different location",
                "Contact support if the issue persists"
            ],
            **kwargs
        )
        
        self.context.update({
            "file_path": file_path,
            "operation": operation
        })


class ThirdPartyServiceError(SystemError):
    """Raised when third-party services fail."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        self.service_name = service_name
        self.status_code = status_code
        
        super().__init__(
            message=message,
            error_code="THIRD_PARTY_SERVICE_ERROR",
            user_message=f"External service {service_name or 'connection'} is temporarily unavailable.",
            recovery_suggestions=[
                "Try again in a few minutes",
                "The external service may be temporarily down",
                "Contact support if the issue persists"
            ],
            **kwargs
        )
        
        self.context.update({
            "service_name": service_name,
            "status_code": status_code
        })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_with_context(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None
) -> AIResearchAssistantError:
    """
    Convert a generic exception to a custom AIResearchAssistantError with context.
    
    Args:
        exception: The original exception
        context: Additional context data
        user_message: User-friendly message override
        
    Returns:
        AIResearchAssistantError: Wrapped exception with enhanced context
    """
    return AIResearchAssistantError(
        message=str(exception),
        error_code="WRAPPED_EXCEPTION",
        context={
            "original_exception_type": type(exception).__name__,
            "original_exception_args": exception.args,
            **(context or {})
        },
        user_message=user_message
    )


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is likely transient and worth retrying.
    
    Args:
        error: The exception to check
        
    Returns:
        bool: True if the error is likely transient
    """
    transient_error_types = (
        ConnectionTimeoutError,
        ReadTimeoutError,
        ApiQuotaError,
        NetworkError,
        ThirdPartyServiceError
    )
    
    return isinstance(error, transient_error_types)


def get_retry_delay(error: Exception, attempt: int, max_delay: int = 300) -> int:
    """
    Calculate retry delay based on error type and attempt number.
    
    Args:
        error: The exception that occurred
        attempt: Current attempt number (starting from 1)
        max_delay: Maximum delay in seconds
        
    Returns:
        int: Delay in seconds before next retry
    """
    if isinstance(error, ApiQuotaError) and error.retry_after_seconds:
        return min(error.retry_after_seconds, max_delay)
    
    # Exponential backoff with jitter
    base_delay = min(2 ** (attempt - 1), max_delay)
    jitter = base_delay * 0.1  # 10% jitter
    import random
    return int(base_delay + random.uniform(-jitter, jitter))
