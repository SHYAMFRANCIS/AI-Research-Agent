#!/usr/bin/env python3
"""
Comprehensive test suite for the AI Research Assistant error handling system.

This test suite covers:
- Unit tests for individual error handling components
- Integration tests for the complete error handling workflow
- Mock-based simulation of quota limits, network failures, malformed responses
- Verification of proper error messaging without uncaught exceptions
- Circuit breaker and retry mechanism testing
- Error correlation and logging verification

Test Coverage:
1. Exception creation and hierarchy
2. Error mapping and transformation
3. Retry mechanisms with exponential backoff
4. Circuit breaker functionality
5. User-friendly error message generation
6. API quota handling and recovery
7. Network failure simulation
8. JSON parsing error handling
9. Error context management
10. Logging and correlation tracking
"""

import asyncio
import json
import logging
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from unittest import TestCase
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the error handling modules
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
    get_retry_delay,
    create_error_with_context
)

from error_handler import (
    ErrorHandler,
    global_error_handler,
    handle_api_errors,
    handle_data_processing_errors,
    handle_user_input_errors,
    error_context,
    display_user_error,
    safe_api_call,
    setup_global_exception_handler
)

from retry_handler import (
    RetryConfig,
    CircuitBreaker,
    RetryHandler,
    retry,
    aggressive_retry,
    conservative_retry,
    network_retry,
    create_circuit_breaker,
    RetryMetrics
)

from error_logger import (
    ErrorLogger,
    StructuredFormatter,
    UserFriendlyErrorMapper,
    error_logger,
    correlation_context,
    log_error,
    log_warning,
    log_info,
    get_user_friendly_error
)


class TestCustomExceptions(TestCase):
    """Test custom exception classes and their behavior."""
    
    def test_base_exception_creation(self):
        """Test base AIResearchAssistantError creation and properties."""
        error = AIResearchAssistantError(
            message="Test error",
            error_code="TEST_ERROR",
            context={"test": "value"},
            user_message="User friendly message"
        )
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.context["test"], "value")
        self.assertEqual(error.user_message, "User friendly message")
        self.assertIsInstance(error.timestamp, datetime)
        
        # Test to_dict method
        error_dict = error.to_dict()
        self.assertEqual(error_dict["error_type"], "AIResearchAssistantError")
        self.assertEqual(error_dict["error_code"], "TEST_ERROR")
        self.assertEqual(error_dict["message"], "Test error")
    
    def test_api_quota_error_with_retry_after(self):
        """Test ApiQuotaError with retry-after seconds."""
        retry_time = datetime.now() + timedelta(hours=1)
        error = ApiQuotaError(
            message="Quota exceeded",
            quota_type="requests",
            quota_limit=1000,
            quota_reset_time=retry_time,
            retry_after_seconds=3600
        )
        
        self.assertIn("quota", error.user_message.lower())
        self.assertEqual(error.retry_after_seconds, 3600)
        self.assertEqual(error.quota_type, "requests")
        self.assertIn("upgrade", error.recovery_suggestions[1])
    
    def test_network_error_hierarchy(self):
        """Test network error inheritance and specific types."""
        base_error = NetworkError("Connection failed", endpoint="https://api.test.com")
        timeout_error = ConnectionTimeoutError("Timeout", timeout_seconds=30)
        read_error = ReadTimeoutError("Read timeout", timeout_seconds=60)
        
        # Test inheritance
        self.assertIsInstance(timeout_error, NetworkError)
        self.assertIsInstance(read_error, NetworkError)
        
        # Test specific properties
        self.assertEqual(base_error.endpoint, "https://api.test.com")
        self.assertEqual(timeout_error.timeout_seconds, 30)
        self.assertEqual(read_error.timeout_seconds, 60)
    
    def test_parsing_error_types(self):
        """Test different parsing error types."""
        json_error = JSONParsingError(
            "Invalid JSON",
            json_content='{"invalid": json}'
        )
        structure_error = ResponseStructureError(
            "Missing fields",
            expected_fields=["name", "value"],
            missing_fields=["value"]
        )
        
        self.assertIn("json_content_preview", json_error.context)
        self.assertEqual(structure_error.missing_fields, ["value"])
    
    def test_transient_error_detection(self):
        """Test is_transient_error function."""
        transient_errors = [
            NetworkError("Network issue"),
            ApiQuotaError("Quota exceeded"),
            ThirdPartyServiceError("Service down")
        ]
        
        non_transient_errors = [
            InputValidationError("Invalid input"),
            RequiredFieldError("email")
        ]
        
        for error in transient_errors:
            self.assertTrue(is_transient_error(error))
        
        for error in non_transient_errors:
            self.assertFalse(is_transient_error(error))


class TestRetryMechanism(TestCase):
    """Test retry mechanisms and configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delays for testing
            max_delay=1.0,
            jitter=False  # Disable jitter for predictable tests
        )
    
    def test_retry_config_should_retry(self):
        """Test retry decision logic."""
        # Test with retryable error
        network_error = NetworkError("Connection failed")
        self.assertTrue(self.retry_config.should_retry(network_error, 1))
        self.assertTrue(self.retry_config.should_retry(network_error, 2))
        self.assertFalse(self.retry_config.should_retry(network_error, 3))
        
        # Test with non-retryable error
        input_error = InputValidationError("Invalid input")
        self.assertFalse(self.retry_config.should_retry(input_error, 1))
    
    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        error = NetworkError("Test error")
        
        # Test exponential backoff
        delay1 = self.retry_config.get_delay(error, 1)
        delay2 = self.retry_config.get_delay(error, 2)
        delay3 = self.retry_config.get_delay(error, 3)
        
        self.assertLess(delay1, delay2)
        self.assertLess(delay2, delay3)
    
    def test_api_quota_specific_delay(self):
        """Test API quota error with specific retry delay."""
        quota_error = ApiQuotaError(
            "Quota exceeded",
            retry_after_seconds=60
        )
        
        delay = self.retry_config.get_delay(quota_error, 1)
        self.assertLessEqual(delay, 60)  # Should respect max_delay
    
    def test_retry_decorator_success_after_failures(self):
        """Test retry decorator with function that succeeds after failures."""
        attempt_count = 0
        
        @retry(config=self.retry_config)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        result = flaky_function()
        self.assertEqual(attempt_count, 3)
        self.assertEqual(result, "Success on attempt 3")
    
    def test_retry_decorator_exhausted_attempts(self):
        """Test retry decorator when all attempts are exhausted."""
        attempt_count = 0
        
        @retry(config=self.retry_config)
        def always_failing_function():
            nonlocal attempt_count
            attempt_count += 1
            raise NetworkError(f"Attempt {attempt_count} failed")
        
        with self.assertRaises(NetworkError):
            always_failing_function()
        
        self.assertEqual(attempt_count, self.retry_config.max_attempts)
    
    def test_retry_decorator_non_retryable_error(self):
        """Test retry decorator with non-retryable error."""
        attempt_count = 0
        
        @retry(config=self.retry_config)
        def validation_error_function():
            nonlocal attempt_count
            attempt_count += 1
            raise InputValidationError("Invalid input")
        
        with self.assertRaises(InputValidationError):
            validation_error_function()
        
        # Should only attempt once for non-retryable errors
        self.assertEqual(attempt_count, 1)


class TestCircuitBreaker(TestCase):
    """Test circuit breaker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            success_threshold=2
        )
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
        self.assertTrue(self.circuit_breaker.can_execute())
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after failure threshold."""
        # Record failures
        for i in range(3):
            self.circuit_breaker.record_failure()
            if i < 2:
                self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
        
        # After 3 failures, should be open
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.OPEN)
        self.assertFalse(self.circuit_breaker.can_execute())
    
    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker transitions to half-open after timeout."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.record_failure()
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.OPEN)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should allow execution (half-open state)
        self.assertTrue(self.circuit_breaker.can_execute())
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.HALF_OPEN)
    
    def test_circuit_breaker_closes_after_successes(self):
        """Test circuit breaker closes after successful operations in half-open."""
        # Open the circuit and wait
        for i in range(3):
            self.circuit_breaker.record_failure()
        time.sleep(0.2)
        
        # Execute to get to half-open
        self.circuit_breaker.can_execute()
        
        # Record successes
        self.circuit_breaker.record_success()
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.HALF_OPEN)
        
        self.circuit_breaker.record_success()
        self.assertEqual(self.circuit_breaker.state, CircuitBreaker.CLOSED)
    
    def test_retry_with_circuit_breaker(self):
        """Test retry mechanism with circuit breaker integration."""
        attempt_count = 0
        
        config = RetryConfig(max_attempts=5, base_delay=0.01)
        handler = RetryHandler(config, self.circuit_breaker)
        
        @handler.retry
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            raise NetworkError(f"Attempt {attempt_count}")
        
        with self.assertRaises(NetworkError):
            failing_function()
        
        # Should stop before max attempts due to circuit breaker
        self.assertLess(attempt_count, 5)


class TestErrorHandlerIntegration(TestCase):
    """Test error handler integration and mapping."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler("test_service")
    
    def test_api_error_handling(self):
        """Test API error handling with mapping."""
        def failing_api_call():
            raise ConnectionError("Connection refused")
        
        with self.assertRaises(ConnectionTimeoutError):
            self.error_handler.handle_api_error(failing_api_call, "test_api")
    
    def test_data_processing_error_handling(self):
        """Test data processing error handling."""
        def failing_parser():
            raise json.JSONDecodeError("Invalid JSON", "", 0)
        
        with self.assertRaises(JSONParsingError):
            self.error_handler.handle_data_processing(failing_parser, "test_data")
    
    def test_user_input_validation(self):
        """Test user input validation error handling."""
        def invalid_input_handler():
            raise ValueError("Invalid value")
        
        with self.assertRaises(InputValidationError):
            self.error_handler.handle_user_input(invalid_input_handler, "test_input")
    
    def test_file_operation_error_handling(self):
        """Test file operation error handling."""
        def failing_file_op():
            raise FileNotFoundError("File not found")
        
        with self.assertRaises(FileSystemError):
            self.error_handler.handle_file_operation(
                failing_file_op, "/path/to/file", "read"
            )
    
    @patch('error_logger.log_error')
    def test_error_context_manager(self, mock_log_error):
        """Test error context manager functionality."""
        with self.assertRaises(SystemError):
            with error_context("test_operation"):
                raise ValueError("Test error")
        
        # Verify logging was called
        mock_log_error.assert_called()


class TestMockApiFailures(TestCase):
    """Test mock API failures and error handling."""
    
    def setUp(self):
        """Set up mock objects and test fixtures."""
        self.mock_api = Mock()
        self.error_handler = ErrorHandler("mock_api_service")
    
    @patch('time.sleep')  # Speed up tests by mocking sleep
    def test_quota_limit_simulation(self, mock_sleep):
        """Simulate API quota limit exceeded."""
        # Configure mock to raise quota error
        self.mock_api.side_effect = Exception("Quota exceeded")
        
        def quota_limited_call():
            # Simulate quota error detection
            try:
                self.mock_api()
            except Exception as e:
                if "quota" in str(e).lower():
                    raise ApiQuotaError(
                        str(e),
                        quota_type="requests",
                        retry_after_seconds=60
                    )
                raise
        
        with self.assertRaises(ApiQuotaError) as context:
            self.error_handler.handle_api_error(quota_limited_call, "quota_api")
        
        error = context.exception
        self.assertEqual(error.retry_after_seconds, 60)
        self.assertIn("quota", error.user_message.lower())
    
    @patch('time.sleep')
    def test_network_failure_simulation(self, mock_sleep):
        """Simulate various network failures."""
        network_errors = [
            ConnectionError("Connection refused"),
            Exception("Timeout"),
            Exception("DNS resolution failed")
        ]
        
        for network_error in network_errors:
            with self.subTest(error=network_error):
                self.mock_api.side_effect = network_error
                
                def network_call():
                    try:
                        self.mock_api()
                    except ConnectionError:
                        raise ConnectionTimeoutError("Connection failed")
                    except Exception as e:
                        if "timeout" in str(e).lower():
                            raise ReadTimeoutError(str(e))
                        raise NetworkError(str(e))
                
                with self.assertRaises(NetworkError):
                    self.error_handler.handle_api_error(network_call, "network_api")
    
    def test_malformed_response_simulation(self):
        """Simulate malformed JSON responses."""
        malformed_responses = [
            '{"invalid": json}',  # Missing quotes
            '{"incomplete":',     # Incomplete JSON
            'Not JSON at all',    # Plain text
            '',                   # Empty response
            None                  # Null response
        ]
        
        for response in malformed_responses:
            with self.subTest(response=response):
                def malformed_parser():
                    if response is None:
                        raise ValueError("No response data")
                    return json.loads(response)
                
                with self.assertRaises((JSONParsingError, ResponseStructureError)):
                    self.error_handler.handle_data_processing(
                        malformed_parser, "malformed_response"
                    )
    
    def test_generic_exception_handling(self):
        """Test handling of generic exceptions not in hierarchy."""
        generic_errors = [
            RuntimeError("Runtime issue"),
            MemoryError("Out of memory"),
            KeyError("Missing key"),
            AttributeError("Missing attribute")
        ]
        
        for error in generic_errors:
            with self.subTest(error=error):
                def generic_failing_function():
                    raise error
                
                with self.assertRaises(SystemError):
                    self.error_handler.handle_api_error(
                        generic_failing_function, "generic_api"
                    )


class TestAsyncErrorHandling(TestCase):
    """Test asynchronous error handling patterns."""
    
    def setUp(self):
        """Set up async test fixtures."""
        self.config = RetryConfig(max_attempts=3, base_delay=0.01)
        self.handler = RetryHandler(self.config)
    
    async def test_async_retry_success(self):
        """Test async retry decorator with eventual success."""
        attempt_count = 0
        
        @self.handler.retry
        async def async_flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError(f"Async attempt {attempt_count} failed")
            return f"Async success on attempt {attempt_count}"
        
        result = await async_flaky_function()
        self.assertEqual(result, "Async success on attempt 3")
        self.assertEqual(attempt_count, 3)
    
    async def test_async_retry_failure(self):
        """Test async retry decorator with all attempts failing."""
        attempt_count = 0
        
        @self.handler.retry
        async def async_failing_function():
            nonlocal attempt_count
            attempt_count += 1
            raise NetworkError(f"Async attempt {attempt_count} failed")
        
        with self.assertRaises(NetworkError):
            await async_failing_function()
        
        self.assertEqual(attempt_count, self.config.max_attempts)
    
    def test_run_async_tests(self):
        """Run async tests using asyncio."""
        asyncio.run(self.test_async_retry_success())
        asyncio.run(self.test_async_retry_failure())


class TestErrorLogging(TestCase):
    """Test error logging and correlation."""
    
    def setUp(self):
        """Set up logging test fixtures."""
        # Create logger with in-memory handler for testing
        self.test_logger = ErrorLogger(
            name="test_logger",
            console_output=False
        )
        
        # Mock handler to capture log records
        self.mock_handler = Mock()
        self.test_logger.logger.addHandler(self.mock_handler)
    
    def test_structured_error_logging(self):
        """Test structured error logging with context."""
        test_error = NetworkError("Test network error")
        context = {"operation": "test_api_call", "attempt": 1}
        
        self.test_logger.error(
            "Test error message",
            exception=test_error,
            context=context,
            correlation_id="test-correlation-123"
        )
        
        # Verify logging was called
        self.assertTrue(self.mock_handler.handle.called)
        log_record = self.mock_handler.handle.call_args[0][0]
        
        # Verify log record structure
        self.assertEqual(log_record.levelname, "ERROR")
        self.assertEqual(log_record.correlation_id, "test-correlation-123")
    
    def test_correlation_context(self):
        """Test correlation context manager."""
        with correlation_context("test-correlation") as corr_id:
            self.assertEqual(corr_id, "test-correlation")
            
            # Test that correlation ID is maintained
            self.test_logger.info("Test message", correlation_id=corr_id)
    
    def test_api_call_logging(self):
        """Test API call logging functionality."""
        self.test_logger.log_api_call(
            service="test_service",
            endpoint="/api/test",
            method="POST",
            status_code=200,
            duration=0.5
        )
        
        self.assertTrue(self.mock_handler.handle.called)
        log_record = self.mock_handler.handle.call_args[0][0]
        self.assertEqual(log_record.levelname, "INFO")
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Log some errors
        self.test_logger.error("Error 1", exception=NetworkError("Network issue"))
        self.test_logger.error("Error 2", exception=ApiQuotaError("Quota exceeded"))
        self.test_logger.error("Error 3", exception=NetworkError("Another network issue"))
        
        summary = self.test_logger.get_error_summary()
        
        self.assertIn("session_id", summary)
        self.assertIn("total_errors", summary)
        self.assertIn("error_breakdown", summary)
        self.assertEqual(summary["total_errors"], 3)
        self.assertEqual(summary["error_breakdown"]["NetworkError"], 2)
        self.assertEqual(summary["error_breakdown"]["ApiQuotaError"], 1)


class TestUserFriendlyMessages(TestCase):
    """Test user-friendly error message generation."""
    
    def test_api_quota_user_message(self):
        """Test API quota error user message generation."""
        quota_error = ApiQuotaError(
            "Quota exceeded",
            quota_type="requests",
            retry_after_seconds=300
        )
        
        user_error = display_user_error(quota_error)
        
        self.assertIn("title", user_error)
        self.assertIn("message", user_error)
        self.assertIn("suggestions", user_error)
        self.assertIn("error_id", user_error)
        
        # Verify user-friendly content
        self.assertIn("quota", user_error["message"].lower())
        self.assertTrue(len(user_error["suggestions"]) > 0)
    
    def test_network_error_user_message(self):
        """Test network error user message generation."""
        network_error = ConnectionTimeoutError(
            "Connection timeout",
            timeout_seconds=30
        )
        
        user_error = display_user_error(network_error)
        
        self.assertEqual(user_error["title"], "Connection Timeout")
        self.assertIn("connection", user_error["message"].lower())
        self.assertIn("Check your internet connection", user_error["suggestions"][0])
    
    def test_parsing_error_user_message(self):
        """Test parsing error user message generation."""
        parsing_error = JSONParsingError(
            "Invalid JSON format",
            json_content='{"invalid": json}'
        )
        
        user_error = display_user_error(parsing_error)
        
        self.assertIn("Invalid Data Format", user_error["title"])
        self.assertIn("Try again", user_error["suggestions"][0])
    
    def test_generic_error_fallback(self):
        """Test generic error fallback messaging."""
        generic_error = RuntimeError("Some generic error")
        
        user_error = display_user_error(generic_error)
        
        self.assertIn("Unexpected Error", user_error["title"])
        self.assertIn("Try again", user_error["suggestions"])


class TestIntegrationScenarios(TestCase):
    """Test complete integration scenarios."""
    
    @patch('time.sleep')
    def test_complete_api_workflow_with_retries(self, mock_sleep):
        """Test complete API workflow with retry and recovery."""
        attempt_count = 0
        
        def mock_api_call():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count == 1:
                raise ConnectionError("Connection refused")
            elif attempt_count == 2:
                raise Exception("Timeout")
            else:
                return {"status": "success", "data": "response"}
        
        # Use retry decorator
        @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        def wrapped_api_call():
            try:
                return mock_api_call()
            except ConnectionError:
                raise ConnectionTimeoutError("Connection failed")
            except Exception as e:
                if "timeout" in str(e).lower():
                    raise ReadTimeoutError(str(e))
                raise NetworkError(str(e))
        
        # Should succeed after retries
        result = wrapped_api_call()
        self.assertEqual(result["status"], "success")
        self.assertEqual(attempt_count, 3)
    
    def test_error_context_with_correlation(self):
        """Test error context manager with correlation tracking."""
        with self.assertRaises(NetworkError):
            with error_context("test_workflow") as ctx:
                # Simulate some operations
                with correlation_context() as corr_id:
                    log_info("Starting operation", correlation_id=corr_id)
                    
                    # Simulate failure
                    raise Exception("Simulated network failure")
    
    @patch('error_logger.log_error')
    def test_safe_api_call_fallback(self, mock_log_error):
        """Test safe API call with graceful fallback."""
        def failing_api():
            raise NetworkError("Service unavailable")
        
        result = safe_api_call(failing_api, "test_service")
        self.assertIsNone(result)  # Should return None on failure
        mock_log_error.assert_called()  # Should log the error
    
    def test_no_uncaught_exceptions(self):
        """Test that no uncaught exceptions leak through error handling."""
        test_cases = [
            lambda: ApiQuotaError("Quota exceeded"),
            lambda: NetworkError("Network failure"),
            lambda: JSONParsingError("Invalid JSON"),
            lambda: InputValidationError("Invalid input"),
            lambda: SystemError("System failure")
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                try:
                    error = test_case()
                    # Verify error can be handled without uncaught exceptions
                    user_error = display_user_error(error)
                    self.assertIn("title", user_error)
                    self.assertIn("message", user_error)
                except Exception as e:
                    self.fail(f"Uncaught exception: {e}")


class TestErrorMetrics(TestCase):
    """Test error metrics and monitoring."""
    
    def test_retry_metrics_collection(self):
        """Test retry metrics collection and reporting."""
        metrics = RetryMetrics()
        
        # Simulate some retry attempts
        metrics.record_attempt("test_function", 1, False, 0.1)
        metrics.record_attempt("test_function", 2, False, 0.2)
        metrics.record_attempt("test_function", 3, True, 0.0)
        
        metrics.record_error("NetworkError")
        metrics.record_error("NetworkError")
        metrics.record_error("ApiQuotaError")
        
        summary = metrics.get_summary()
        
        self.assertEqual(summary["total_attempts"], 3)
        self.assertEqual(summary["successful_retries"], 1)
        self.assertEqual(summary["failed_retries"], 2)
        self.assertEqual(summary["error_counts"]["NetworkError"], 2)
        self.assertIn("retry_success_rate", summary)
        self.assertIn("average_delay", summary)


if __name__ == "__main__":
    # Set up test environment
    logging.basicConfig(level=logging.WARNING)  # Reduce test noise
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCustomExceptions,
        TestRetryMechanism,
        TestCircuitBreaker,
        TestErrorHandlerIntegration,
        TestMockApiFailures,
        TestAsyncErrorHandling,
        TestErrorLogging,
        TestUserFriendlyMessages,
        TestIntegrationScenarios,
        TestErrorMetrics
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    print("🚀 Running Comprehensive Error Handling Test Suite")
    print("=" * 60)
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  • Tests run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ {len(result.failures)} test failures:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\n🔥 {len(result.errors)} test errors:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.splitlines()[-1]}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed! Error handling system is robust.")
    
    print("\n🎯 Test Coverage Areas:")
    print("  ✓ Exception creation and hierarchy")
    print("  ✓ Error mapping and transformation")
    print("  ✓ Retry mechanisms with exponential backoff")
    print("  ✓ Circuit breaker functionality")
    print("  ✓ User-friendly error message generation")
    print("  ✓ API quota handling and recovery")
    print("  ✓ Network failure simulation")
    print("  ✓ JSON parsing error handling")
    print("  ✓ Error context management")
    print("  ✓ Logging and correlation tracking")
    print("  ✓ Integration scenarios")
    print("  ✓ Async error handling")
    print("  ✓ Metrics and monitoring")
