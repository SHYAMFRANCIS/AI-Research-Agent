#!/usr/bin/env python3
"""
Test script to demonstrate the structured error handling system.

This script shows how the error handling components work together
and provides examples of different error scenarios.
"""

import time
import json
from typing import Dict, Any

# Import our error handling components
from exceptions import (
    AIResearchAssistantError,
    ApiQuotaError,
    NetworkError,
    ConnectionTimeoutError,
    ParsingError,
    JSONParsingError,
    InputValidationError,
    RequiredFieldError
)
from error_handler import (
    handle_api_errors,
    handle_data_processing_errors,
    handle_user_input_errors,
    error_context,
    display_user_error,
    safe_api_call
)
from error_logger import log_info, log_error
from retry_handler import retry, aggressive_retry


def demo_custom_exceptions():
    """Demonstrate custom exception creation and handling."""
    print("\n🔧 Demo: Custom Exception Classes")
    print("=" * 40)
    
    # Create different types of errors
    errors = [
        ApiQuotaError(
            "Quota exceeded",
            quota_type="requests",
            quota_limit=1000,
            retry_after_seconds=3600
        ),
        NetworkError(
            "Connection failed",
            endpoint="https://api.example.com"
        ),
        JSONParsingError(
            "Invalid JSON format",
            json_content='{"invalid": json}'
        ),
        RequiredFieldError("email")
    ]
    
    for error in errors:
        user_friendly = display_user_error(error)
        print(f"\n📋 {error.__class__.__name__}:")
        print(f"  Title: {user_friendly['title']}")
        print(f"  Message: {user_friendly['message']}")
        print(f"  Suggestions: {', '.join(user_friendly['suggestions'][:2])}...")
        print(f"  Error Code: {error.error_code}")


def demo_retry_mechanism():
    """Demonstrate retry mechanism with different failure scenarios."""
    print("\n🔄 Demo: Retry Mechanism")
    print("=" * 40)
    
    attempt_count = 0
    
    @retry
    def flaky_api_call(success_on_attempt: int = 3):
        """Simulate a flaky API that succeeds on the nth attempt."""
        nonlocal attempt_count
        attempt_count += 1
        
        print(f"  Attempt {attempt_count}: Calling flaky API...")
        
        if attempt_count < success_on_attempt:
            raise NetworkError(f"Connection failed (attempt {attempt_count})")
        
        return {"status": "success", "data": "API response"}
    
    try:
        print("\n🎯 Testing API that succeeds on 3rd attempt:")
        attempt_count = 0
        result = flaky_api_call(3)
        print(f"  ✅ Success: {result}")
    except AIResearchAssistantError as e:
        print(f"  ❌ Failed: {e.user_message}")
    
    try:
        print("\n🎯 Testing API that always fails:")
        attempt_count = 0
        result = flaky_api_call(10)  # Will never succeed within retry limit
        print(f"  ✅ Success: {result}")
    except AIResearchAssistantError as e:
        print(f"  ❌ Failed after retries: {e.user_message}")


@handle_api_errors("demo_service")
def demo_api_function(should_fail: bool = False):
    """Demo function that can succeed or fail."""
    if should_fail:
        raise ApiQuotaError("Demo quota exceeded", retry_after_seconds=60)
    return {"message": "API call successful"}


@handle_data_processing_errors("demo_data")
def demo_data_processing(data: str):
    """Demo function for data processing."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise JSONParsingError("Failed to parse JSON", json_content=data)


@handle_user_input_errors("demo_input")
def demo_input_validation(email: str):
    """Demo function for input validation."""
    if not email:
        raise RequiredFieldError("email")
    if "@" not in email:
        raise InputValidationError("Invalid email format", field_name="email")
    return {"email": email, "valid": True}


def demo_decorators():
    """Demonstrate decorator-based error handling."""
    print("\n🎨 Demo: Decorator-Based Error Handling")
    print("=" * 40)
    
    # Test successful API call
    try:
        result = demo_api_function(should_fail=False)
        print(f"✅ API Success: {result}")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"❌ API Failed: {user_error['message']}")
    
    # Test failing API call
    try:
        result = demo_api_function(should_fail=True)
        print(f"✅ API Success: {result}")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"❌ API Failed: {user_error['message']}")
    
    # Test data processing
    try:
        result = demo_data_processing('{"valid": "json"}')
        print(f"✅ Parsing Success: {result}")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"❌ Parsing Failed: {user_error['message']}")
    
    try:
        result = demo_data_processing('{invalid: json}')
        print(f"✅ Parsing Success: {result}")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"❌ Parsing Failed: {user_error['message']}")
    
    # Test input validation
    try:
        result = demo_input_validation("test@example.com")
        print(f"✅ Validation Success: {result}")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"❌ Validation Failed: {user_error['message']}")
    
    try:
        result = demo_input_validation("")
        print(f"✅ Validation Success: {result}")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"❌ Validation Failed: {user_error['message']}")


def demo_context_manager():
    """Demonstrate context manager error handling."""
    print("\n🏗️  Demo: Context Manager Error Handling")
    print("=" * 40)
    
    # Successful operation
    try:
        with error_context("demo_successful_operation") as ctx:
            print("  Executing successful operation...")
            time.sleep(0.1)  # Simulate some work
            result = {"status": "completed"}
        print(f"  ✅ Operation completed successfully")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"  ❌ Operation failed: {user_error['message']}")
    
    # Failing operation
    try:
        with error_context("demo_failing_operation") as ctx:
            print("  Executing failing operation...")
            time.sleep(0.1)  # Simulate some work
            raise ValueError("Something went wrong in the operation")
    except AIResearchAssistantError as e:
        user_error = display_user_error(e)
        print(f"  ❌ Operation failed: {user_error['message']}")


def demo_safe_operations():
    """Demonstrate safe operation utilities."""
    print("\n🛡️  Demo: Safe Operations")
    print("=" * 40)
    
    def failing_api():
        raise NetworkError("Service unavailable")
    
    def successful_api():
        return {"data": "success"}
    
    # Test safe API call that fails
    result = safe_api_call(failing_api, "demo_service")
    if result is None:
        print("  🔶 Safe API call failed gracefully (returned None)")
    else:
        print(f"  ✅ Safe API call succeeded: {result}")
    
    # Test safe API call that succeeds
    result = safe_api_call(successful_api, "demo_service")
    if result is None:
        print("  🔶 Safe API call failed gracefully (returned None)")
    else:
        print(f"  ✅ Safe API call succeeded: {result}")


def demo_error_correlation():
    """Demonstrate error correlation and logging."""
    print("\n🔗 Demo: Error Correlation and Logging")
    print("=" * 40)
    
    try:
        with error_context("demo_correlation_workflow") as ctx:
            log_info("Starting correlated workflow", correlation_id=ctx.correlation_id)
            
            # Simulate multiple steps
            for step in range(1, 4):
                log_info(f"Executing step {step}", correlation_id=ctx.correlation_id)
                if step == 3:
                    raise NetworkError("Network failure in step 3")
                time.sleep(0.05)
            
    except AIResearchAssistantError as e:
        print(f"  ❌ Workflow failed with correlation: {e.error_code}")
        print(f"      Correlation ID can be found in logs for debugging")


def main():
    """Run all error handling demos."""
    print("🚀 AI Research Assistant - Error Handling Demo")
    print("=" * 50)
    print("This script demonstrates the structured error handling system.")
    
    try:
        demo_custom_exceptions()
        demo_retry_mechanism()
        demo_decorators()
        demo_context_manager()
        demo_safe_operations()
        demo_error_correlation()
        
        print("\n✅ All demos completed successfully!")
        print("\n📊 Check the logs for detailed error tracking information.")
        
    except Exception as e:
        user_error = display_user_error(e)
        print(f"\n❌ Demo failed: {user_error['title']}")
        print(f"   {user_error['message']}")
        
        if 'suggestions' in user_error:
            print("\n💡 Suggestions:")
            for suggestion in user_error['suggestions']:
                print(f"   • {suggestion}")


if __name__ == "__main__":
    main()
