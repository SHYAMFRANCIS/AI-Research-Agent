#!/usr/bin/env python3
"""
Integration tests for real-world failure scenarios and edge cases.

This module focuses on testing complex integration scenarios that might occur
in production, including cascading failures, partial service degradation,
and recovery patterns.
"""

import asyncio
import json
import logging
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from unittest import TestCase
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the error handling modules
from exceptions import *
from error_handler import *
from retry_handler import *
from error_logger import *


class MockGoogleAPIClient:
    """Mock Google Gemini API client for testing quota and network issues."""
    
    def __init__(self):
        self.call_count = 0
        self.quota_calls = 0
        self.quota_limit = 50  # Simulate daily quota limit
        self.network_failures = 0
        self.max_network_failures = 3
    
    def generate_content(self, prompt: str):
        """Mock content generation with various failure modes."""
        self.call_count += 1
        self.quota_calls += 1
        
        # Simulate quota exhaustion
        if self.quota_calls > self.quota_limit:
            raise Exception("Quota exceeded. Please wait until tomorrow.")
        
        # Simulate intermittent network failures
        if self.network_failures < self.max_network_failures and self.call_count % 3 == 0:
            self.network_failures += 1
            raise ConnectionError("Connection timed out")
        
        # Simulate malformed responses occasionally
        if self.call_count % 7 == 0:
            return "This is not JSON {invalid response"
        
        # Simulate empty responses
        if self.call_count % 11 == 0:
            return ""
        
        # Normal successful response
        return json.dumps({
            "topic": f"Generated content for: {prompt[:50]}...",
            "summary": f"This is a mock response for the query about {prompt[:30]}",
            "sources": [f"https://example.com/source{self.call_count}"],
            "tools_used": ["mock_search", "mock_generation"]
        })


class MockLangChainAgent:
    """Mock LangChain agent for testing complex workflow failures."""
    
    def __init__(self):
        self.execution_count = 0
        self.tool_failures = 0
    
    def invoke(self, query_data: Dict[str, Any]):
        """Mock agent invocation with various failure scenarios."""
        self.execution_count += 1
        query = query_data.get("query", "")
        
        # Simulate tool execution failures
        if "search" in query.lower() and self.tool_failures < 2:
            self.tool_failures += 1
            raise Exception("Search tool unavailable - external API down")
        
        # Simulate parsing issues with agent output
        if "complex" in query.lower():
            return {
                "output": "Here's a complex response that doesn't follow the expected JSON format..."
            }
        
        # Simulate structured response
        if len(query) > 100:
            raise ValueError("Query too long - please shorten your request")
        
        # Simulate successful execution
        return {
            "output": json.dumps({
                "topic": f"Research on: {query[:30]}",
                "summary": f"Based on the query '{query}', here is the research summary...",
                "sources": ["https://example.com/research1", "https://example.com/research2"],
                "tools_used": ["search_tool", "wiki_tool"]
            })
        }


class TestRealWorldScenarios(TestCase):
    """Test real-world failure scenarios and recovery patterns."""
    
    def setUp(self):
        """Set up test environment with mock services."""
        self.mock_gemini = MockGoogleAPIClient()
        self.mock_langchain = MockLangChainAgent()
        self.error_handler = ErrorHandler("integration_test")
        
        # Set up retry configuration for testing
        self.test_retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            max_delay=0.1,
            jitter=False
        )
    
    @patch('time.sleep')
    def test_quota_exhaustion_recovery(self, mock_sleep):
        """Test quota exhaustion and recovery workflow."""
        
        def quota_aware_api_call():
            try:
                response = self.mock_gemini.generate_content("Test query")
                return response
            except Exception as e:
                if "quota exceeded" in str(e).lower():
                    # Calculate retry time (simulate daily reset)
                    tomorrow = datetime.now() + timedelta(days=1)
                    tomorrow_reset = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
                    seconds_until_reset = (tomorrow_reset - datetime.now()).total_seconds()
                    
                    raise ApiQuotaError(
                        str(e),
                        quota_type="daily_requests",
                        quota_reset_time=tomorrow_reset,
                        retry_after_seconds=min(seconds_until_reset, 86400)  # Cap at 24 hours
                    )
                raise
        
        # First, exhaust the quota
        for _ in range(self.mock_gemini.quota_limit + 1):
            try:
                self.error_handler.handle_api_error(quota_aware_api_call, "gemini_api")
            except ApiQuotaError:
                break
        
        # Now test quota exceeded error
        with self.assertRaises(ApiQuotaError) as context:
            self.error_handler.handle_api_error(quota_aware_api_call, "gemini_api")
        
        error = context.exception
        self.assertEqual(error.quota_type, "daily_requests")
        self.assertIsNotNone(error.quota_reset_time)
        self.assertIn("quota", error.user_message.lower())
        
        # Test user-friendly error display
        user_error = display_user_error(error)
        self.assertIn("Usage Limit Reached", user_error["title"])
        self.assertTrue(len(user_error["suggestions"]) >= 3)
    
    @patch('time.sleep')
    def test_cascading_network_failures(self, mock_sleep):
        """Test cascading failures across multiple service calls."""
        
        @retry(config=self.test_retry_config)
        def network_dependent_operation():
            # Simulate a complex operation that depends on multiple network calls
            try:
                # First API call - might fail
                response1 = self.mock_gemini.generate_content("First query")
                
                # Second API call - using result from first
                agent_input = {"query": f"Elaborate on: {response1[:50]}"}
                response2 = self.mock_langchain.invoke(agent_input)
                
                return {"step1": response1, "step2": response2}
                
            except ConnectionError as e:
                raise ConnectionTimeoutError(str(e))
            except Exception as e:
                if "unavailable" in str(e).lower():
                    raise ThirdPartyServiceError(str(e), service_name="search_service")
                raise
        
        # This should eventually succeed after retries handle network issues
        result = network_dependent_operation()
        self.assertIn("step1", result)
        self.assertIn("step2", result)
        
        # Verify that retries happened (network failures should have occurred)
        self.assertGreater(self.mock_gemini.network_failures, 0)
    
    def test_malformed_response_parsing_chain(self):
        """Test handling of malformed responses through the parsing chain."""
        
        malformed_responses = [
            "Not JSON at all",
            '{"incomplete": ',
            '{"valid": "json", but: "missing quotes"}',
            "",
            None
        ]
        
        for i, response in enumerate(malformed_responses):
            with self.subTest(response=response, index=i):
                
                def parse_response():
                    if response is None:
                        raise ValueError("No response received from API")
                    
                    if not response or not response.strip():
                        raise ResponseStructureError(
                            "Empty response received",
                            expected_fields=["topic", "summary", "sources"]
                        )
                    
                    try:
                        data = json.loads(response)
                        
                        # Validate required fields
                        required_fields = ["topic", "summary", "sources"]
                        missing_fields = [field for field in required_fields if field not in data]
                        
                        if missing_fields:
                            raise ResponseStructureError(
                                "Missing required fields in response",
                                expected_fields=required_fields,
                                missing_fields=missing_fields
                            )
                        
                        return data
                        
                    except json.JSONDecodeError as e:
                        raise JSONParsingError(
                            "Failed to parse JSON response",
                            json_content=response[:200] if response else ""
                        )
                
                # Test that all malformed responses are handled gracefully
                with self.assertRaises((JSONParsingError, ResponseStructureError, InputValidationError)):
                    self.error_handler.handle_data_processing(parse_response, f"response_{i}")
    
    @patch('error_logger.log_error')
    @patch('error_logger.log_warning')
    def test_error_correlation_across_operations(self, mock_log_warning, mock_log_error):
        """Test error correlation across multiple related operations."""
        
        with correlation_context("workflow-123") as correlation_id:
            
            # Step 1: User input validation
            try:
                with error_context("user_input_validation"):
                    if len("") == 0:  # Simulate empty input
                        raise RequiredFieldError("query")
            except RequiredFieldError:
                pass  # Expected
            
            # Step 2: API call with network issues
            try:
                with error_context("api_call"):
                    @handle_api_errors("gemini")
                    def api_call():
                        if True:  # Simulate network failure
                            raise ConnectionError("Network unreachable")
                        return {"data": "success"}
                    
                    api_call()
            except ConnectionTimeoutError:
                pass  # Expected
            
            # Step 3: Response parsing
            try:
                with error_context("response_parsing"):
                    @handle_data_processing_errors("api_response")
                    def parse_response():
                        raise json.JSONDecodeError("Invalid JSON", "", 0)
                    
                    parse_response()
            except JSONParsingError:
                pass  # Expected
        
        # Verify that all errors were logged with proper correlation
        self.assertTrue(mock_log_error.called)
        
        # Check that correlation ID was used in logging
        logged_calls = mock_log_error.call_args_list
        correlation_ids = [
            call.kwargs.get('correlation_id') for call in logged_calls
            if 'correlation_id' in call.kwargs
        ]
        self.assertTrue(any(corr_id == correlation_id for corr_id in correlation_ids))
    
    def test_circuit_breaker_prevents_cascading_failures(self):
        """Test circuit breaker preventing cascading failures."""
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            success_threshold=1
        )
        
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
        handler = RetryHandler(retry_config, circuit_breaker)
        
        failure_count = 0
        
        @handler.retry
        def unreliable_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:  # Fail first 5 attempts
                raise NetworkError(f"Service failure {failure_count}")
            return f"Success after {failure_count} attempts"
        
        # First few calls should fail and open the circuit
        with self.assertRaises(NetworkError):
            unreliable_service()
        
        # Circuit should now be open, preventing further calls
        with self.assertRaises(AIResearchAssistantError) as context:
            unreliable_service()
        
        error = context.exception
        self.assertEqual(error.error_code, "CIRCUIT_BREAKER_OPEN")
        self.assertIn("temporarily unavailable", error.user_message.lower())
        
        # Wait for recovery timeout
        time.sleep(0.1)
        
        # Circuit should allow one test call (half-open)
        # This should still fail, reopening the circuit
        with self.assertRaises(NetworkError):
            unreliable_service()
    
    @patch('time.sleep')
    def test_progressive_degradation_handling(self, mock_sleep):
        """Test handling of progressive service degradation."""
        
        class DegradingService:
            def __init__(self):
                self.degradation_level = 0
                self.call_count = 0
            
            def make_request(self, request_type="normal"):
                self.call_count += 1
                
                # Simulate increasing degradation
                if self.call_count > 5:
                    self.degradation_level = 1  # Slow responses
                if self.call_count > 10:
                    self.degradation_level = 2  # Partial failures
                if self.call_count > 15:
                    self.degradation_level = 3  # Complete failure
                
                if self.degradation_level == 1:
                    time.sleep(0.01)  # Simulate slow response
                    if self.call_count % 4 == 0:  # 25% failure rate
                        raise ReadTimeoutError("Request timeout", timeout_seconds=30)
                
                elif self.degradation_level == 2:
                    if self.call_count % 2 == 0:  # 50% failure rate
                        raise NetworkError("Intermittent network failure")
                
                elif self.degradation_level == 3:
                    raise ThirdPartyServiceError("Service completely unavailable", service_name="degrading_service")
                
                return f"Success response {self.call_count}"
        
        service = DegradingService()
        success_count = 0
        error_count = 0
        
        # Adaptive retry configuration that adjusts to degradation
        adaptive_config = RetryConfig(
            max_attempts=4,
            base_delay=0.01,
            max_delay=0.1,
            exponential_base=1.5  # Gentler backoff
        )
        
        @retry(config=adaptive_config)
        def adaptive_service_call():
            return service.make_request()
        
        # Make multiple calls to observe degradation pattern
        for i in range(20):
            try:
                result = adaptive_service_call()
                success_count += 1
            except AIResearchAssistantError as e:
                error_count += 1
                
                # Log different error types for analysis
                if isinstance(e, ReadTimeoutError):
                    log_warning(f"Service slowdown detected on call {i}")
                elif isinstance(e, NetworkError):
                    log_warning(f"Service degradation detected on call {i}")
                elif isinstance(e, ThirdPartyServiceError):
                    log_error(f"Service failure detected on call {i}")
        
        # Verify progressive degradation was handled
        self.assertGreater(error_count, 0)
        self.assertGreater(service.call_count, 15)  # Should have triggered full degradation
        self.assertEqual(service.degradation_level, 3)  # Should reach complete failure
    
    def test_error_recovery_patterns(self):
        """Test different error recovery patterns and strategies."""
        
        class RecoverableService:
            def __init__(self):
                self.failure_mode = "network"
                self.call_count = 0
                self.recovery_threshold = 3
            
            def call_service(self, operation="test"):
                self.call_count += 1
                
                # Simulate recovery after certain number of attempts
                if self.call_count >= self.recovery_threshold:
                    return f"Recovered! Operation {operation} successful"
                
                # Different failure modes
                if self.failure_mode == "network":
                    raise ConnectionTimeoutError(f"Network failure {self.call_count}")
                elif self.failure_mode == "quota":
                    raise ApiQuotaError(
                        f"Quota exceeded {self.call_count}",
                        retry_after_seconds=1
                    )
                elif self.failure_mode == "parsing":
                    raise JSONParsingError(f"Parse error {self.call_count}")
                else:
                    raise SystemError(f"System error {self.call_count}")
        
        service = RecoverableService()
        
        # Test network failure recovery
        service.failure_mode = "network"
        service.call_count = 0
        
        @retry(config=RetryConfig(max_attempts=5, base_delay=0.01))
        def network_recovery_call():
            return service.call_service("network_test")
        
        result = network_recovery_call()
        self.assertIn("Recovered!", result)
        self.assertEqual(service.call_count, service.recovery_threshold)
        
        # Test quota recovery (should not retry immediately)
        service.failure_mode = "quota"
        service.call_count = 0
        service.recovery_threshold = 1  # Should fail on first attempt
        
        @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        def quota_recovery_call():
            return service.call_service("quota_test")
        
        with self.assertRaises(ApiQuotaError):
            quota_recovery_call()
        
        self.assertEqual(service.call_count, 1)  # Should only try once for quota errors
    
    def test_complex_workflow_error_handling(self):
        """Test error handling in a complex multi-step workflow."""
        
        workflow_state = {
            "step": 0,
            "data": None,
            "errors": [],
            "retries": 0
        }
        
        def complex_workflow(user_query: str):
            """Simulate a complex research workflow with multiple failure points."""
            
            # Step 1: Input validation
            workflow_state["step"] = 1
            if not user_query or len(user_query.strip()) == 0:
                raise RequiredFieldError("query")
            
            if len(user_query) > 500:
                raise InputValidationError(
                    "Query too long",
                    field_name="query",
                    user_message="Please shorten your query to under 500 characters"
                )
            
            # Step 2: API call to Gemini
            workflow_state["step"] = 2
            try:
                gemini_response = self.mock_gemini.generate_content(user_query)
                if not gemini_response:
                    raise ResponseStructureError("Empty response from Gemini")
            except Exception as e:
                if "quota" in str(e).lower():
                    raise ApiQuotaError(str(e))
                elif "connection" in str(e).lower():
                    raise ConnectionTimeoutError(str(e))
                else:
                    raise ThirdPartyServiceError(str(e), service_name="gemini")
            
            # Step 3: Parse Gemini response
            workflow_state["step"] = 3
            try:
                if gemini_response.startswith("{"):
                    parsed_data = json.loads(gemini_response)
                else:
                    # Handle non-JSON response
                    parsed_data = {
                        "topic": "Generated Content",
                        "summary": gemini_response,
                        "sources": [],
                        "tools_used": ["gemini"]
                    }
            except json.JSONDecodeError:
                raise JSONParsingError("Invalid JSON from Gemini", json_content=gemini_response)
            
            # Step 4: LangChain agent processing
            workflow_state["step"] = 4
            try:
                agent_response = self.mock_langchain.invoke({"query": user_query})
                agent_output = agent_response.get("output", "")
                
                if agent_output.startswith("{"):
                    agent_data = json.loads(agent_output)
                else:
                    # Handle non-JSON agent output
                    agent_data = {
                        "topic": parsed_data.get("topic", "Research"),
                        "summary": agent_output,
                        "sources": [],
                        "tools_used": ["langchain_agent"]
                    }
            except Exception as e:
                if "tool unavailable" in str(e).lower():
                    raise ThirdPartyServiceError(str(e), service_name="search_tool")
                else:
                    raise ParsingError(str(e), data_source="langchain_agent")
            
            # Step 5: Merge and validate final result
            workflow_state["step"] = 5
            final_result = {
                "topic": agent_data.get("topic", parsed_data.get("topic")),
                "summary": agent_data.get("summary", parsed_data.get("summary")),
                "sources": list(set(
                    parsed_data.get("sources", []) + agent_data.get("sources", [])
                )),
                "tools_used": list(set(
                    parsed_data.get("tools_used", []) + agent_data.get("tools_used", [])
                )),
                "workflow_steps": workflow_state["step"]
            }
            
            workflow_state["data"] = final_result
            return final_result
        
        # Test successful workflow
        try:
            result = complex_workflow("What is artificial intelligence?")
            self.assertIn("topic", result)
            self.assertIn("summary", result)
            self.assertEqual(workflow_state["step"], 5)
        except Exception as e:
            # Even if individual components fail, we should get structured errors
            self.assertIsInstance(e, AIResearchAssistantError)
            user_error = display_user_error(e)
            self.assertIn("title", user_error)
            self.assertIn("message", user_error)
        
        # Test workflow with various failure scenarios
        test_cases = [
            ("", RequiredFieldError),  # Empty query
            ("x" * 600, InputValidationError),  # Too long query
            ("search query", None)  # May succeed or fail depending on mock state
        ]
        
        for query, expected_error in test_cases:
            workflow_state = {"step": 0, "data": None, "errors": [], "retries": 0}
            
            with self.subTest(query=query[:20], expected_error=expected_error):
                if expected_error:
                    with self.assertRaises(expected_error):
                        complex_workflow(query)
                else:
                    try:
                        result = complex_workflow(query)
                        # Verify result structure even for complex scenarios
                        self.assertIn("workflow_steps", result)
                    except AIResearchAssistantError as e:
                        # Should still produce user-friendly errors
                        user_error = display_user_error(e)
                        self.assertIn("suggestions", user_error)


class TestEdgeCasesAndRaceConditions(TestCase):
    """Test edge cases and potential race conditions in error handling."""
    
    def test_concurrent_circuit_breaker_state_changes(self):
        """Test circuit breaker behavior under concurrent access."""
        
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        results = []
        
        def concurrent_operation(operation_id: int):
            """Simulate concurrent operations that might fail."""
            try:
                if operation_id % 2 == 0:  # Every other operation fails
                    circuit_breaker.record_failure()
                    raise NetworkError(f"Operation {operation_id} failed")
                else:
                    circuit_breaker.record_success()
                    return f"Operation {operation_id} succeeded"
            except Exception as e:
                return f"Operation {operation_id} error: {str(e)}"
        
        # Simulate concurrent operations
        import threading
        threads = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda i=i: results.append(concurrent_operation(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 10)
        
        # Circuit breaker should have changed state due to failures
        self.assertGreaterEqual(circuit_breaker.failure_count, 1)
    
    def test_memory_usage_during_error_cascades(self):
        """Test that error handling doesn't cause memory leaks during cascades."""
        
        import gc
        import sys
        
        # Get initial object counts
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create many errors and handle them
        for i in range(100):
            try:
                error = NetworkError(f"Error {i}", endpoint=f"endpoint_{i}")
                error.context = {"large_data": "x" * 1000}  # Add some bulk to the error
                
                # Process error through handling chain
                user_error = display_user_error(error)
                
                # Simulate error logging
                log_error(f"Test error {i}", exception=error)
                
            except Exception:
                pass  # Ignore any exceptions during this test
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage shouldn't grow significantly
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000, "Memory usage grew too much during error handling")
    
    def test_error_handling_with_corrupted_state(self):
        """Test error handling when internal state is corrupted."""
        
        # Simulate corrupted error logger
        corrupted_logger = ErrorLogger("corrupted_test")
        corrupted_logger.error_counts = None  # Corrupt internal state
        
        # Should handle gracefully
        try:
            corrupted_logger.error("Test error", exception=NetworkError("Test"))
        except Exception as e:
            self.fail(f"Error handling failed with corrupted state: {e}")
        
        # Simulate corrupted circuit breaker
        corrupted_cb = CircuitBreaker()
        corrupted_cb.state = "invalid_state"  # Invalid state
        
        # Should handle gracefully
        try:
            can_execute = corrupted_cb.can_execute()
            self.assertIsInstance(can_execute, bool)
        except Exception as e:
            self.fail(f"Circuit breaker failed with corrupted state: {e}")
    
    def test_unicode_and_special_characters_in_errors(self):
        """Test error handling with unicode and special characters."""
        
        special_test_cases = [
            "Query with émojis 🚀🔥💻",
            "中文查询测试",
            "Query with\nnewlines\tand\ttabs",
            "Query with \"quotes\" and 'apostrophes'",
            "Query with special chars: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        ]
        
        for test_case in special_test_cases:
            with self.subTest(test_case=test_case[:20]):
                # Test that error creation handles special characters
                error = InputValidationError(
                    f"Invalid input: {test_case}",
                    field_name="query"
                )
                
                # Test that error display handles special characters
                user_error = display_user_error(error)
                self.assertIn("title", user_error)
                self.assertIn("message", user_error)
                
                # Test that logging handles special characters
                try:
                    log_error(f"Test error with special chars: {test_case}", exception=error)
                except Exception as e:
                    self.fail(f"Logging failed with special characters: {e}")
    
    def test_extremely_large_error_contexts(self):
        """Test handling of errors with very large context data."""
        
        # Create error with large context
        large_context = {
            "large_text": "x" * 10000,  # 10KB of text
            "large_list": list(range(1000)),  # 1000 integers
            "nested_data": {
                "level1": {
                    "level2": {
                        "level3": "deeply nested data" * 100
                    }
                }
            }
        }
        
        error = SystemError("Error with large context", context=large_context)
        
        # Should handle large contexts gracefully
        try:
            user_error = display_user_error(error)
            self.assertIn("title", user_error)
            
            # Context should be handled without causing issues
            error_dict = error.to_dict()
            self.assertIn("context", error_dict)
            
        except Exception as e:
            self.fail(f"Failed to handle error with large context: {e}")


if __name__ == "__main__":
    # Set up test environment
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestRealWorldScenarios,
        TestEdgeCasesAndRaceConditions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    print("🌐 Running Integration Scenarios Test Suite")
    print("=" * 60)
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Summary:")
    print(f"  • Tests run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    
    if result.failures or result.errors:
        print(f"  • Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    else:
        print("  • Success rate: 100.0%")
    
    if result.failures:
        print(f"\n❌ {len(result.failures)} test failures:")
        for test, traceback in result.failures:
            print(f"  • {test}")
    
    if result.errors:
        print(f"\n🔥 {len(result.errors)} test errors:")
        for test, traceback in result.errors:
            print(f"  • {test}")
    
    if not result.failures and not result.errors:
        print("\n✅ All integration tests passed!")
        print("🛡️  Error handling system is production-ready!")
    
    print("\n🎯 Integration Test Coverage:")
    print("  ✓ Real-world API quota scenarios")
    print("  ✓ Cascading network failures")
    print("  ✓ Complex parsing error chains")
    print("  ✓ Error correlation across operations")
    print("  ✓ Circuit breaker failure prevention")
    print("  ✓ Progressive service degradation")
    print("  ✓ Multi-step workflow error handling")
    print("  ✓ Concurrent access edge cases")
    print("  ✓ Memory usage under error cascades")
    print("  ✓ Unicode and special character handling")
    print("  ✓ Large error context handling")
