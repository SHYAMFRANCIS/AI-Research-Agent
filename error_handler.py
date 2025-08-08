"""
Central error handler that integrates all error handling components.

This module provides the main error handling interface that combines custom exceptions,
retry logic, logging, and user-friendly error mapping. It demonstrates where try/catch
blocks should be placed and how to handle different types of errors appropriately.
"""

import asyncio
import functools
import json
import time
from typing import Callable, Optional, Dict, Any, Union, TypeVar, Type
from datetime import datetime

# Import our custom error handling components
from exceptions import (
    AIResearchAssistantError,
    ApiQuotaError,
    NetworkError,
    ConnectionTimeoutError,
    ReadTimeoutError,
    ParsingError,
    JSONParsingError,
    ResponseStructureError,
    InputValidationError,
    RequiredFieldError,
    SystemError,
    FileSystemError,
    ThirdPartyServiceError,
    is_transient_error,
    create_error_with_context
)
from retry_handler import (
    retry,
    aggressive_retry,
    conservative_retry,
    network_retry,
    RetryConfig,
    CircuitBreaker,
    create_circuit_breaker
)
from error_logger import (
    error_logger,
    correlation_context,
    get_user_friendly_error,
    log_error,
    log_warning,
    log_info,
    log_api_call
)

# Import standard library exceptions for mapping
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
from requests.exceptions import ConnectionError, ReadTimeout, Timeout
from json import JSONDecodeError
import pydantic


T = TypeVar('T')


class ErrorHandler:
    """
    Central error handler that provides a unified interface for error handling.
    
    This class demonstrates where different types of error handling should be applied
    and provides convenience methods for common error scenarios.
    """
    
    def __init__(self, service_name: str = "ai_research_assistant"):
        """
        Initialize error handler.
        
        Args:
            service_name: Name of the service for logging context
        """
        self.service_name = service_name
        self.circuit_breakers = {}
        self.error_counts = {}
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = create_circuit_breaker(name)
        return self.circuit_breakers[name]
    
    def handle_api_error(
        self,
        func: Callable[..., T],
        service_name: str,
        *args,
        **kwargs
    ) -> T:
        """
        Handle API calls with comprehensive error handling.
        
        This method should be used to wrap all external API calls.
        It includes retry logic, circuit breaker protection, and proper error mapping.
        
        Args:
            func: The API function to call
            service_name: Name of the API service for logging
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the API call
            
        Raises:
            AIResearchAssistantError: Mapped and enhanced error
        """
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        @retry(circuit_breaker=circuit_breaker)
        def wrapped_call():
            with correlation_context() as correlation_id:
                start_time = time.time()
                
                try:
                    log_info(
                        f"Starting API call to {service_name}",
                        context={"service": service_name, "function": func.__name__},
                        correlation_id=correlation_id
                    )
                    
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    log_api_call(
                        service=service_name,
                        endpoint=func.__name__,
                        method="CALL",
                        status_code=200,
                        duration=duration,
                        correlation_id=correlation_id
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    mapped_error = self._map_exception_to_custom_error(e, service_name)
                    
                    # Log the error with context
                    log_error(
                        f"API call to {service_name} failed",
                        exception=mapped_error,
                        context={
                            "service": service_name,
                            "function": func.__name__,
                            "duration": duration,
                            "original_error": str(e)
                        },
                        correlation_id=correlation_id
                    )
                    
                    # Track error counts
                    error_type = type(mapped_error).__name__
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
                    raise mapped_error
        
        return wrapped_call()
    
    def handle_data_processing(
        self,
        func: Callable[..., T],
        data_source: str,
        *args,
        **kwargs
    ) -> T:
        """
        Handle data processing operations with proper error handling.
        
        This should be used for parsing, validation, and transformation operations.
        
        Args:
            func: The processing function to call
            data_source: Description of the data source
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Processed data
            
        Raises:
            AIResearchAssistantError: Mapped parsing or validation error
        """
        with correlation_context() as correlation_id:
            try:
                log_info(
                    f"Processing data from {data_source}",
                    context={"data_source": data_source, "function": func.__name__},
                    correlation_id=correlation_id
                )
                
                result = func(*args, **kwargs)
                
                log_info(
                    f"Successfully processed data from {data_source}",
                    correlation_id=correlation_id
                )
                
                return result
                
            except Exception as e:
                mapped_error = self._map_exception_to_custom_error(e, data_source)
                
                log_error(
                    f"Failed to process data from {data_source}",
                    exception=mapped_error,
                    context={
                        "data_source": data_source,
                        "function": func.__name__,
                        "original_error": str(e)
                    },
                    correlation_id=correlation_id
                )
                
                raise mapped_error
    
    def handle_user_input(
        self,
        func: Callable[..., T],
        input_name: str,
        *args,
        **kwargs
    ) -> T:
        """
        Handle user input validation with proper error handling.
        
        Args:
            func: The validation function to call
            input_name: Name/description of the input being validated
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Validated input
            
        Raises:
            AIResearchAssistantError: Input validation error
        """
        with correlation_context() as correlation_id:
            try:
                log_info(
                    f"Validating user input: {input_name}",
                    context={"input_name": input_name},
                    correlation_id=correlation_id
                )
                
                result = func(*args, **kwargs)
                
                return result
                
            except Exception as e:
                mapped_error = self._map_exception_to_custom_error(e, input_name)
                
                # Input validation errors are usually not logged as errors
                # since they're expected user behavior
                log_warning(
                    f"User input validation failed: {input_name}",
                    exception=mapped_error,
                    context={
                        "input_name": input_name,
                        "function": func.__name__
                    },
                    correlation_id=correlation_id
                )
                
                raise mapped_error
    
    def handle_file_operation(
        self,
        func: Callable[..., T],
        file_path: str,
        operation: str,
        *args,
        **kwargs
    ) -> T:
        """
        Handle file operations with proper error handling.
        
        Args:
            func: The file operation function to call
            file_path: Path of the file being operated on
            operation: Description of the operation (read, write, delete, etc.)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of file operation
            
        Raises:
            AIResearchAssistantError: File system error
        """
        with correlation_context() as correlation_id:
            try:
                log_info(
                    f"File {operation}: {file_path}",
                    context={
                        "file_path": file_path,
                        "operation": operation,
                        "function": func.__name__
                    },
                    correlation_id=correlation_id
                )
                
                result = func(*args, **kwargs)
                
                log_info(
                    f"File {operation} completed successfully: {file_path}",
                    correlation_id=correlation_id
                )
                
                return result
                
            except Exception as e:
                mapped_error = self._map_exception_to_custom_error(e, file_path, operation)
                
                log_error(
                    f"File {operation} failed: {file_path}",
                    exception=mapped_error,
                    context={
                        "file_path": file_path,
                        "operation": operation,
                        "function": func.__name__
                    },
                    correlation_id=correlation_id
                )
                
                raise mapped_error
    
    def _map_exception_to_custom_error(
        self,
        exception: Exception,
        context_info: str,
        operation: Optional[str] = None
    ) -> AIResearchAssistantError:
        """
        Map standard exceptions to our custom exception hierarchy.
        
        This is where we define how standard library and third-party exceptions
        get converted to our structured error types.
        """
        context = {
            "context_info": context_info,
            "original_exception_type": type(exception).__name__
        }
        
        if operation:
            context["operation"] = operation
        
        # Google API Core exceptions (for Gemini API)
        if isinstance(exception, ResourceExhausted):
            return ApiQuotaError(
                message=str(exception),
                quota_type="requests",
                context=context
            )
        
        if isinstance(exception, DeadlineExceeded):
            return ReadTimeoutError(
                message=str(exception),
                context=context
            )
        
        # Network-related exceptions
        if isinstance(exception, ConnectionError):
            return ConnectionTimeoutError(
                message=str(exception),
                context=context
            )
        
        if isinstance(exception, (ReadTimeout, Timeout)):
            return ReadTimeoutError(
                message=str(exception),
                context=context
            )
        
        # JSON parsing errors
        if isinstance(exception, JSONDecodeError):
            return JSONParsingError(
                message=f"Failed to parse JSON: {str(exception)}",
                json_content=getattr(exception, 'doc', None),
                context=context
            )
        
        # Pydantic validation errors
        if hasattr(pydantic, 'ValidationError') and isinstance(exception, pydantic.ValidationError):
            return ResponseStructureError(
                message=f"Response validation failed: {str(exception)}",
                missing_fields=[error['loc'] for error in exception.errors()],
                context=context
            )
        
        # File system errors
        if isinstance(exception, (FileNotFoundError, PermissionError, OSError)):
            return FileSystemError(
                message=str(exception),
                file_path=context_info,
                operation=operation,
                context=context
            )
        
        # Value errors (often input validation)
        if isinstance(exception, ValueError):
            return InputValidationError(
                message=str(exception),
                context=context
            )
        
        # If it's already one of our custom exceptions, just add context
        if isinstance(exception, AIResearchAssistantError):
            exception.context.update(context)
            return exception
        
        # Default fallback - wrap as system error
        return SystemError(
            message=f"Unexpected error: {str(exception)}",
            context=context
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors handled by this instance."""
        circuit_breaker_status = {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "service_name": self.service_name,
            "error_counts": self.error_counts.copy(),
            "circuit_breakers": circuit_breaker_status,
            "logger_summary": error_logger.get_error_summary(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global error handler instance
global_error_handler = ErrorHandler()


# Decorator versions for easy use
def handle_api_errors(service_name: str):
    """Decorator for handling API errors."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return global_error_handler.handle_api_error(func, service_name, *args, **kwargs)
        return wrapper
    return decorator


def handle_data_processing_errors(data_source: str):
    """Decorator for handling data processing errors."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return global_error_handler.handle_data_processing(func, data_source, *args, **kwargs)
        return wrapper
    return decorator


def handle_user_input_errors(input_name: str):
    """Decorator for handling user input validation errors."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return global_error_handler.handle_user_input(func, input_name, *args, **kwargs)
        return wrapper
    return decorator


def handle_file_errors(file_path: str, operation: str):
    """Decorator for handling file operation errors."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return global_error_handler.handle_file_operation(func, file_path, operation, *args, **kwargs)
        return wrapper
    return decorator


# Context manager for error handling
class error_context:
    """
    Context manager that provides structured error handling for code blocks.
    
    Usage:
        with error_context("user_query_processing") as ctx:
            # Your code here
            result = process_user_query(query)
    """
    
    def __init__(
        self,
        operation_name: str,
        error_handler: Optional[ErrorHandler] = None,
        reraise: bool = True
    ):
        """
        Initialize error context.
        
        Args:
            operation_name: Name of the operation for logging
            error_handler: Custom error handler (uses global if None)
            reraise: Whether to reraise exceptions after handling
        """
        self.operation_name = operation_name
        self.error_handler = error_handler or global_error_handler
        self.reraise = reraise
        self.correlation_id = None
        self.start_time = None
    
    def __enter__(self):
        self.correlation_id = error_logger._generate_correlation_id()
        self.start_time = time.time()
        
        log_info(
            f"Starting operation: {self.operation_name}",
            correlation_id=self.correlation_id
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_val is None:
            # Success
            log_info(
                f"Operation completed successfully: {self.operation_name}",
                context={"duration": duration},
                correlation_id=self.correlation_id
            )
            return False
        
        # Error occurred
        mapped_error = self.error_handler._map_exception_to_custom_error(
            exc_val,
            self.operation_name
        )
        
        log_error(
            f"Operation failed: {self.operation_name}",
            exception=mapped_error,
            context={
                "duration": duration,
                "original_error": str(exc_val)
            },
            correlation_id=self.correlation_id
        )
        
        if self.reraise:
            # Replace the original exception with our mapped one
            raise mapped_error from exc_val
        
        return True  # Suppress the exception


# Convenience functions for common patterns
def safe_api_call(func: Callable[..., T], service_name: str, *args, **kwargs) -> Optional[T]:
    """
    Safely call an API function and return None if it fails.
    
    This is useful for non-critical API calls where you want to continue
    execution even if the call fails.
    """
    try:
        return global_error_handler.handle_api_error(func, service_name, *args, **kwargs)
    except AIResearchAssistantError as e:
        log_warning(
            f"Non-critical API call failed: {service_name}",
            exception=e
        )
        return None


def safe_file_operation(func: Callable[..., T], file_path: str, operation: str, *args, **kwargs) -> Optional[T]:
    """
    Safely perform a file operation and return None if it fails.
    """
    try:
        return global_error_handler.handle_file_operation(func, file_path, operation, *args, **kwargs)
    except AIResearchAssistantError as e:
        log_warning(
            f"Non-critical file operation failed: {operation} on {file_path}",
            exception=e
        )
        return None


def display_user_error(error: Exception) -> Dict[str, Any]:
    """
    Convert any error to a user-friendly format for display.
    
    This should be used at the UI boundary to show errors to users.
    """
    return get_user_friendly_error(error, include_technical_details=False)


# Exception handler for uncaught exceptions
def setup_global_exception_handler():
    """
    Set up a global exception handler to catch any unhandled exceptions.
    
    This should be called early in your application startup.
    """
    import sys
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupt to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the uncaught exception
        error_logger.critical(
            "Uncaught exception occurred",
            exception=exc_value,
            context={
                "exc_type": exc_type.__name__,
                "traceback": exc_traceback
            }
        )
        
        # Display user-friendly error
        user_error = get_user_friendly_error(exc_value)
        print(f"\n❌ {user_error['title']}")
        print(f"{user_error['message']}")
        print("\n💡 Suggestions:")
        for suggestion in user_error['suggestions']:
            print(f"  • {suggestion}")
        print(f"\nError ID: {user_error['error_id']} (for support)")
    
    sys.excepthook = handle_exception
