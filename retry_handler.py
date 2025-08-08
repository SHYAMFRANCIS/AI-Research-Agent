"""
Retry handler with exponential backoff for handling transient errors.

This module provides decorators and utilities for implementing robust retry logic
with exponential backoff, jitter, and circuit breaker patterns for the AI Research
Assistant application.
"""

import asyncio
import functools
import logging
import time
import random
from typing import Callable, Optional, Tuple, Type, Union, Any, Dict
from datetime import datetime, timedelta

from exceptions import (
    AIResearchAssistantError,
    ApiQuotaError,
    NetworkError,
    ConnectionTimeoutError,
    ReadTimeoutError,
    ThirdPartyServiceError,
    is_transient_error,
    get_retry_delay
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retry_on: Optional[Tuple[Type[Exception], ...]] = None,
        stop_on: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
            jitter_factor: Factor for jitter calculation (0.0 to 1.0)
            retry_on: Tuple of exception types that should trigger retries
            stop_on: Tuple of exception types that should stop retries immediately
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        
        # Default transient error types if not specified
        if retry_on is None:
            self.retry_on = (
                ConnectionTimeoutError,
                ReadTimeoutError,
                NetworkError,
                ThirdPartyServiceError,
                ApiQuotaError
            )
        else:
            self.retry_on = retry_on
            
        self.stop_on = stop_on or ()
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if we should retry based on the exception and attempt number.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-indexed)
            
        Returns:
            bool: True if we should retry
        """
        # Check if we've exceeded max attempts
        if attempt >= self.max_attempts:
            return False
        
        # Check if this exception type should stop retries immediately
        if self.stop_on and isinstance(exception, self.stop_on):
            return False
        
        # Check if this exception type should trigger retries
        if isinstance(exception, self.retry_on):
            return True
        
        # For AIResearchAssistantError, check if it's transient
        if isinstance(exception, AIResearchAssistantError):
            return is_transient_error(exception)
        
        return False
    
    def get_delay(self, exception: Exception, attempt: int) -> float:
        """
        Calculate delay before next retry attempt.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-indexed)
            
        Returns:
            float: Delay in seconds
        """
        # Handle API quota errors with specific retry-after
        if isinstance(exception, ApiQuotaError) and exception.retry_after_seconds:
            base_delay = min(exception.retry_after_seconds, self.max_delay)
        else:
            # Exponential backoff
            base_delay = min(
                self.base_delay * (self.exponential_base ** (attempt - 1)),
                self.max_delay
            )
        
        # Add jitter if enabled
        if self.jitter and base_delay > 0:
            jitter_amount = base_delay * self.jitter_factor
            jitter_offset = random.uniform(-jitter_amount, jitter_amount)
            base_delay = max(0, base_delay + jitter_offset)
        
        return base_delay


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is open, requests fail fast
    - HALF_OPEN: Testing if service has recovered
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying to close circuit
            success_threshold: Number of successes needed to close circuit from half-open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = asyncio.Lock() if hasattr(asyncio, '_get_running_loop') else None
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self.state == self.CLOSED:
            return True
        elif self.state == self.OPEN:
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.recovery_timeout):
                self.state = self.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == self.HALF_OPEN:
            return True
        return False
    
    def record_success(self):
        """Record a successful operation."""
        if self.state == self.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = self.CLOSED
                self.failure_count = 0
        elif self.state == self.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
        elif (self.state == self.CLOSED and 
              self.failure_count >= self.failure_threshold):
            self.state = self.OPEN


class RetryHandler:
    """Main retry handler class."""
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        logger_name: Optional[str] = None
    ):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration
            circuit_breaker: Optional circuit breaker
            logger_name: Name for the logger
        """
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        self.logger = logging.getLogger(logger_name or __name__)
    
    def retry(
        self,
        func: Optional[Callable] = None,
        *,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Decorator for adding retry logic to functions.
        
        Args:
            func: Function to decorate
            config: Override retry configuration
            circuit_breaker: Override circuit breaker
            
        Returns:
            Decorated function with retry logic
        """
        def decorator(f: Callable) -> Callable:
            if asyncio.iscoroutinefunction(f):
                return self._async_retry_wrapper(f, config, circuit_breaker)
            else:
                return self._sync_retry_wrapper(f, config, circuit_breaker)
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _sync_retry_wrapper(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ) -> Callable:
        """Create synchronous retry wrapper."""
        retry_config = config or self.config
        cb = circuit_breaker or self.circuit_breaker
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            last_exception = None
            
            while attempt <= retry_config.max_attempts:
                # Check circuit breaker
                if cb and not cb.can_execute():
                    raise AIResearchAssistantError(
                        "Service temporarily unavailable (circuit breaker open)",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        user_message="Service is temporarily unavailable. Please try again later.",
                        context={"circuit_breaker_state": cb.state}
                    )
                
                try:
                    self.logger.debug(f"Attempting {func.__name__} (attempt {attempt})")
                    result = func(*args, **kwargs)
                    
                    # Record success
                    if cb:
                        cb.record_success()
                    
                    if attempt > 1:
                        self.logger.info(f"{func.__name__} succeeded on attempt {attempt}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Record failure in circuit breaker
                    if cb:
                        cb.record_failure()
                    
                    # Check if we should retry
                    if not retry_config.should_retry(e, attempt):
                        self.logger.warning(
                            f"{func.__name__} failed with non-retryable error on attempt {attempt}: {e}"
                        )
                        raise
                    
                    # Calculate delay for next attempt
                    if attempt < retry_config.max_attempts:
                        delay = retry_config.get_delay(e, attempt)
                        self.logger.warning(
                            f"{func.__name__} failed on attempt {attempt}: {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                    
                    attempt += 1
            
            # All attempts exhausted
            self.logger.error(f"{func.__name__} failed after {retry_config.max_attempts} attempts")
            raise last_exception
        
        return wrapper
    
    def _async_retry_wrapper(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ) -> Callable:
        """Create asynchronous retry wrapper."""
        retry_config = config or self.config
        cb = circuit_breaker or self.circuit_breaker
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempt = 1
            last_exception = None
            
            while attempt <= retry_config.max_attempts:
                # Check circuit breaker
                if cb and not cb.can_execute():
                    raise AIResearchAssistantError(
                        "Service temporarily unavailable (circuit breaker open)",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        user_message="Service is temporarily unavailable. Please try again later.",
                        context={"circuit_breaker_state": cb.state}
                    )
                
                try:
                    self.logger.debug(f"Attempting {func.__name__} (attempt {attempt})")
                    result = await func(*args, **kwargs)
                    
                    # Record success
                    if cb:
                        cb.record_success()
                    
                    if attempt > 1:
                        self.logger.info(f"{func.__name__} succeeded on attempt {attempt}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Record failure in circuit breaker
                    if cb:
                        cb.record_failure()
                    
                    # Check if we should retry
                    if not retry_config.should_retry(e, attempt):
                        self.logger.warning(
                            f"{func.__name__} failed with non-retryable error on attempt {attempt}: {e}"
                        )
                        raise
                    
                    # Calculate delay for next attempt
                    if attempt < retry_config.max_attempts:
                        delay = retry_config.get_delay(e, attempt)
                        self.logger.warning(
                            f"{func.__name__} failed on attempt {attempt}: {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    
                    attempt += 1
            
            # All attempts exhausted
            self.logger.error(f"{func.__name__} failed after {retry_config.max_attempts} attempts")
            raise last_exception
        
        return async_wrapper


# =============================================================================
# PREDEFINED RETRY CONFIGURATIONS
# =============================================================================

# Default retry configuration
DEFAULT_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

# Aggressive retry for critical operations
AGGRESSIVE_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=60.0,
    exponential_base=1.5,
    jitter=True
)

# Conservative retry for rate-limited APIs
CONSERVATIVE_RETRY = RetryConfig(
    max_attempts=2,
    base_delay=5.0,
    max_delay=120.0,
    exponential_base=3.0,
    jitter=True,
    retry_on=(ApiQuotaError, NetworkError)
)

# Network-specific retry configuration
NETWORK_RETRY = RetryConfig(
    max_attempts=4,
    base_delay=2.0,
    max_delay=45.0,
    exponential_base=2.0,
    jitter=True,
    retry_on=(ConnectionTimeoutError, ReadTimeoutError, NetworkError)
)

# Global retry handler instances
default_retry_handler = RetryHandler(DEFAULT_RETRY)
aggressive_retry_handler = RetryHandler(AGGRESSIVE_RETRY)
conservative_retry_handler = RetryHandler(CONSERVATIVE_RETRY)
network_retry_handler = RetryHandler(NETWORK_RETRY)

# Convenience decorators
retry = default_retry_handler.retry
aggressive_retry = aggressive_retry_handler.retry
conservative_retry = conservative_retry_handler.retry
network_retry = network_retry_handler.retry


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0
) -> CircuitBreaker:
    """
    Create a named circuit breaker with logging.
    
    Args:
        name: Name for the circuit breaker (used in logging)
        failure_threshold: Number of failures before opening
        recovery_timeout: Time to wait before testing recovery
        
    Returns:
        CircuitBreaker: Configured circuit breaker instance
    """
    cb = CircuitBreaker(failure_threshold, recovery_timeout)
    logger.info(f"Created circuit breaker '{name}' with threshold={failure_threshold}, timeout={recovery_timeout}s")
    return cb


def retry_with_context(
    context: Dict[str, Any],
    config: Optional[RetryConfig] = None
):
    """
    Create a retry decorator that adds context to exceptions.
    
    Args:
        context: Context to add to exceptions
        config: Retry configuration
        
    Returns:
        Decorator function
    """
    retry_config = config or DEFAULT_RETRY
    handler = RetryHandler(retry_config)
    
    def decorator(func: Callable) -> Callable:
        @handler.retry
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, AIResearchAssistantError):
                    e.context.update(context)
                    raise
                else:
                    # Wrap generic exceptions with context
                    from exceptions import create_error_with_context
                    raise create_error_with_context(e, context)
        
        return wrapper
    
    return decorator


# =============================================================================
# MONITORING AND METRICS
# =============================================================================

class RetryMetrics:
    """Collect and track retry metrics."""
    
    def __init__(self):
        self.metrics = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'circuit_breaker_trips': 0,
            'total_delay_time': 0.0,
            'error_counts': {}
        }
        self.start_time = datetime.now()
    
    def record_attempt(self, function_name: str, attempt: int, success: bool, delay: float = 0.0):
        """Record a retry attempt."""
        self.metrics['total_attempts'] += 1
        self.metrics['total_delay_time'] += delay
        
        if attempt > 1:  # This was a retry
            if success:
                self.metrics['successful_retries'] += 1
            else:
                self.metrics['failed_retries'] += 1
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.metrics['error_counts'][error_type] = self.metrics['error_counts'].get(error_type, 0) + 1
    
    def record_circuit_breaker_trip(self):
        """Record a circuit breaker trip."""
        self.metrics['circuit_breaker_trips'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        runtime = (datetime.now() - self.start_time).total_seconds()
        return {
            **self.metrics,
            'runtime_seconds': runtime,
            'retry_success_rate': (
                self.metrics['successful_retries'] / 
                max(1, self.metrics['successful_retries'] + self.metrics['failed_retries'])
            ),
            'average_delay': (
                self.metrics['total_delay_time'] / 
                max(1, self.metrics['total_attempts'])
            )
        }

# Global metrics instance
retry_metrics = RetryMetrics()
