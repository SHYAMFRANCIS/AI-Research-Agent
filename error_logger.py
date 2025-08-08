"""
Structured logging system for the AI Research Assistant application.

This module provides consistent error logging, formatting, and tracking capabilities
with structured JSON output, correlation IDs, and integration with monitoring systems.
"""

import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from contextlib import contextmanager

from exceptions import AIResearchAssistantError


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data["correlation_id"] = record.correlation_id
        
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        # Handle exception information
        if record.exc_info and self.include_traceback:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Handle custom AIResearchAssistantError
        if hasattr(record, 'error_data') and isinstance(record.error_data, dict):
            log_data["error"] = record.error_data
        
        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text',
                          'correlation_id', 'user_id', 'request_id', 'error_data']:
                extra_fields[key] = value
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class ErrorLogger:
    """
    Centralized error logging with structured output and correlation tracking.
    """
    
    def __init__(
        self,
        name: str = "ai_research_assistant",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        structured_format: bool = True
    ):
        """
        Initialize error logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            console_output: Whether to output to console
            structured_format: Whether to use structured JSON format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatter
        if structured_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Error metrics
        self.error_counts = {}
        self.session_id = str(uuid.uuid4())
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        return str(uuid.uuid4())
    
    def _create_log_record(
        self,
        level: str,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a structured log record."""
        
        log_data = {
            "level": level,
            "log_message": message,  # Renamed to avoid conflict with logging's 'message'
            "correlation_id": correlation_id or self._generate_correlation_id(),
            "session_id": self.session_id,
            "context": context or {}
        }
        
        if user_id:
            log_data["user_id"] = user_id
        
        if request_id:
            log_data["request_id"] = request_id
        
        if exception:
            error_type = type(exception).__name__
            
            # Track error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Handle AIResearchAssistantError specially
            if isinstance(exception, AIResearchAssistantError):
                log_data["error"] = exception.to_dict()
            else:
                log_data["error"] = {
                    "type": error_type,
                    "message": str(exception),
                    "traceback": traceback.format_exception(type(exception), exception, exception.__traceback__)
                }
        
        return log_data
    
    def debug(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Log debug message."""
        extra = self._create_log_record("DEBUG", message, context=context, correlation_id=correlation_id)
        extra.update(kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Log info message."""
        extra = self._create_log_record("INFO", message, context=context, correlation_id=correlation_id)
        extra.update(kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(
        self,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Log warning message."""
        extra = self._create_log_record("WARNING", message, exception, context, correlation_id)
        extra.update(kwargs)
        
        if exception:
            extra["error_data"] = extra.get("error", {})
        
        self.logger.warning(message, extra=extra, exc_info=exception is not None)
    
    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """Log error message."""
        extra = self._create_log_record(
            "ERROR", message, exception, context, correlation_id, user_id, request_id
        )
        extra.update(kwargs)
        
        if exception:
            extra["error_data"] = extra.get("error", {})
        
        self.logger.error(message, extra=extra, exc_info=exception is not None)
    
    def critical(
        self,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """Log critical message."""
        extra = self._create_log_record(
            "CRITICAL", message, exception, context, correlation_id, user_id, request_id
        )
        extra.update(kwargs)
        
        if exception:
            extra["error_data"] = extra.get("error", {})
        
        self.logger.critical(message, extra=extra, exc_info=exception is not None)
    
    def log_api_call(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: Optional[int] = None,
        duration: Optional[float] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Log API call information."""
        context = {
            "service": service,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": duration * 1000 if duration else None
        }
        context.update(kwargs)
        
        message = f"API call to {service} {method} {endpoint}"
        if status_code:
            message += f" returned {status_code}"
        if duration:
            message += f" in {duration:.3f}s"
        
        level = "INFO"
        if status_code and status_code >= 400:
            level = "WARNING"
        if status_code and status_code >= 500:
            level = "ERROR"
        
        extra = self._create_log_record(level, message, context=context, correlation_id=correlation_id)
        if request_id:
            extra["request_id"] = request_id
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def log_user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Log user action."""
        log_context = {
            "action": action,
            "success": success,
            **(context or {})
        }
        log_context.update(kwargs)
        
        message = f"User action: {action} - {'SUCCESS' if success else 'FAILED'}"
        
        extra = self._create_log_record("INFO", message, context=log_context, correlation_id=correlation_id)
        if user_id:
            extra["user_id"] = user_id
        
        self.logger.info(message, extra=extra)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        return {
            "session_id": self.session_id,
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": self.error_counts.copy(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Context manager for correlation tracking
@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for tracking operations with correlation IDs."""
    corr_id = correlation_id or str(uuid.uuid4())
    
    # Store in thread-local or asyncio context
    import threading
    if not hasattr(threading.current_thread(), 'correlation_id'):
        threading.current_thread().correlation_id = corr_id
    
    try:
        yield corr_id
    finally:
        if hasattr(threading.current_thread(), 'correlation_id'):
            delattr(threading.current_thread(), 'correlation_id')


class UserFriendlyErrorMapper:
    """
    Maps internal errors to user-friendly messages based on the error handling matrix.
    """
    
    ERROR_MESSAGES = {
        # API Quota Errors
        "API_QUOTA_EXCEEDED": {
            "title": "Usage Limit Reached",
            "message": "You've reached your current usage limit.",
            "suggestions": [
                "Wait for your quota to reset",
                "Consider upgrading for higher limits",
                "Try again later"
            ]
        },
        
        # Network Errors
        "NETWORK_ERROR": {
            "title": "Connection Issue",
            "message": "Unable to connect to the service.",
            "suggestions": [
                "Check your internet connection",
                "Try again in a few moments",
                "Contact support if the issue persists"
            ]
        },
        
        "CONNECTION_TIMEOUT": {
            "title": "Connection Timeout",
            "message": "Unable to connect to the service. Please check your internet connection.",
            "suggestions": [
                "Check your internet connection",
                "Try again with a stronger network connection",
                "Contact your network administrator if needed"
            ]
        },
        
        "READ_TIMEOUT": {
            "title": "Request Timeout",
            "message": "The request is taking longer than expected.",
            "suggestions": [
                "Try again - the service may be temporarily slow",
                "Check your internet connection stability",
                "Wait a few minutes and try again"
            ]
        },
        
        # Parsing Errors
        "PARSING_ERROR": {
            "title": "Data Processing Error",
            "message": "Unable to process the response data.",
            "suggestions": [
                "Try again - this may be a temporary issue",
                "Refresh the application",
                "Contact support if the issue continues"
            ]
        },
        
        "JSON_PARSING_ERROR": {
            "title": "Invalid Data Format",
            "message": "Received invalid data format from the service.",
            "suggestions": [
                "Try again - this may be temporary",
                "The service may be experiencing issues",
                "Contact support if this continues"
            ]
        },
        
        # Input Validation Errors
        "INPUT_VALIDATION_ERROR": {
            "title": "Invalid Input",
            "message": "Please check your input and try again.",
            "suggestions": [
                "Review your input for any errors",
                "Check the required format",
                "Try again with valid data"
            ]
        },
        
        "REQUIRED_FIELD_ERROR": {
            "title": "Required Field Missing",
            "message": "A required field is missing.",
            "suggestions": [
                "Please fill out all required fields",
                "Check for any empty required fields",
                "Try again after completing the form"
            ]
        },
        
        # System Errors
        "SYSTEM_ERROR": {
            "title": "Technical Difficulty",
            "message": "We're experiencing technical difficulties.",
            "suggestions": [
                "Try again in a few minutes",
                "Contact support if the issue persists",
                "Check our status page for updates"
            ]
        },
        
        "FILESYSTEM_ERROR": {
            "title": "File System Error",
            "message": "Unable to save or access files.",
            "suggestions": [
                "Check if you have sufficient permissions",
                "Try saving to a different location",
                "Contact support if the issue persists"
            ]
        },
        
        # Default fallback
        "UNKNOWN_ERROR": {
            "title": "Unexpected Error",
            "message": "An unexpected error occurred.",
            "suggestions": [
                "Try again",
                "Contact support if the issue persists",
                "Check our help documentation"
            ]
        }
    }
    
    @classmethod
    def get_user_friendly_error(
        self,
        error: Union[AIResearchAssistantError, Exception],
        include_technical_details: bool = False
    ) -> Dict[str, Any]:
        """
        Convert an internal error to a user-friendly format.
        
        Args:
            error: The error to convert
            include_technical_details: Whether to include technical details
            
        Returns:
            Dict with user-friendly error information
        """
        if isinstance(error, AIResearchAssistantError):
            error_code = error.error_code
            user_message = error.user_message
            recovery_suggestions = error.recovery_suggestions
            context = error.context if include_technical_details else {}
        else:
            error_code = "UNKNOWN_ERROR"
            user_message = "An unexpected error occurred. Please try again."
            recovery_suggestions = ["Try again", "Contact support if the issue persists"]
            context = {"original_error": str(error)} if include_technical_details else {}
        
        # Get predefined messages or use defaults
        predefined = self.ERROR_MESSAGES.get(error_code, self.ERROR_MESSAGES["UNKNOWN_ERROR"])
        
        result = {
            "error_id": str(uuid.uuid4()),
            "title": predefined["title"],
            "message": user_message,
            "suggestions": recovery_suggestions,
            "error_code": error_code,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if include_technical_details:
            result["technical_details"] = {
                "error_type": type(error).__name__,
                "context": context,
                "original_message": str(error)
            }
        
        return result


# Global logger instance
error_logger = ErrorLogger()

# Convenience functions
def log_error(
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Convenience function for logging errors."""
    error_logger.error(message, exception, context, **kwargs)

def log_warning(
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Convenience function for logging warnings."""
    error_logger.warning(message, exception, context, **kwargs)

def log_info(message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
    """Convenience function for logging info."""
    error_logger.info(message, context, **kwargs)

def log_api_call(service: str, endpoint: str, **kwargs):
    """Convenience function for logging API calls."""
    error_logger.log_api_call(service, endpoint, **kwargs)

def get_user_friendly_error(error: Exception, include_technical_details: bool = False) -> Dict[str, Any]:
    """Convenience function for getting user-friendly error messages."""
    return UserFriendlyErrorMapper.get_user_friendly_error(error, include_technical_details)
