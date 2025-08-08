from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from flask import Flask, request, jsonify
from flask_cors import CORS # Add this import

# Import our structured error handling components
from error_handler import (
    global_error_handler,
    handle_api_errors,
    handle_data_processing_errors,
    error_context,
    display_user_error,
    setup_global_exception_handler
)
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
    SystemError,
    ThirdPartyServiceError
)
from error_logger import log_info, log_error, log_warning, error_logger
from retry_handler import (
    retry,
    aggressive_retry,
    conservative_retry,
    network_retry,
    RetryConfig,
    create_circuit_breaker
)
import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any



# =============================================================================
# ENHANCED ERROR HANDLING CONFIGURATION
# =============================================================================

# Circuit breakers for different services
gemini_circuit_breaker = create_circuit_breaker(
    name="gemini_api",
    failure_threshold=3,  # Open after 3 consecutive failures
    recovery_timeout=60.0  # Wait 60 seconds before testing recovery
)

langchain_circuit_breaker = create_circuit_breaker(
    name="langchain_agent",
    failure_threshold=2,
    recovery_timeout=30.0
)

# Custom retry configurations for different operations
api_retry_config = RetryConfig(
    max_attempts=4,
    base_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0,
    jitter=True,
    retry_on=(ApiQuotaError, NetworkError, ConnectionTimeoutError, ReadTimeoutError)
)

critical_retry_config = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=1.8,
    jitter=True
)

# Set up global exception handling for uncaught exceptions
setup_global_exception_handler()

load_dotenv()

class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# =============================================================================
# ENHANCED ERROR HANDLING WRAPPER FUNCTIONS
# =============================================================================

def handle_quota_exceeded_error(e: ApiQuotaError) -> str:
    """Generate actionable message for quota exceeded errors."""
    if e.retry_after_seconds:
        minutes = e.retry_after_seconds // 60
        seconds = e.retry_after_seconds % 60
        if minutes > 0:
            time_str = f"{minutes} minute(s)" + (f" and {seconds} second(s)" if seconds > 0 else "")
        else:
            time_str = f"{seconds} second(s)"
        return f"Quota exceeded—retry in {time_str} or upgrade plan"
    elif e.quota_reset_time:
        reset_str = e.quota_reset_time.strftime("%I:%M %p %Z")
        return f"Daily quota exceeded—resets at {reset_str} or upgrade plan"
    else:
        return "API quota exceeded—try again later or upgrade plan"

def handle_network_error(e: NetworkError) -> str:
    """Generate actionable message for network errors."""
    if isinstance(e, ConnectionTimeoutError):
        return "Connection timeout—check network and retry in 30s"
    elif isinstance(e, ReadTimeoutError):
        return "Request timeout—service slow, retry in 60s"
    else:
        return "Network error—check connection and retry"

def safe_api_call_with_recovery(func, service_name: str, *args, **kwargs):
    """Enhanced API call wrapper with recovery suggestions."""
    try:
        return global_error_handler.handle_api_error(func, service_name, *args, **kwargs)
    except ApiQuotaError as e:
        error_msg = handle_quota_exceeded_error(e)
        log_warning(f"API quota exceeded for {service_name}: {error_msg}")
        raise ApiQuotaError(
            message=str(e),
            quota_type=e.quota_type,
            retry_after_seconds=e.retry_after_seconds,
            user_message=error_msg,
            context=e.context
        )
    except NetworkError as e:
        error_msg = handle_network_error(e)
        log_warning(f"Network error for {service_name}: {error_msg}")
        raise NetworkError(
            message=str(e),
            endpoint=service_name,
            user_message=error_msg,
            context=e.context
        )
    except Exception as e:
        log_error(f"Unexpected error in {service_name}", exception=e)
        raise

def robust_parse_with_fallback(parser_func, data, data_source: str):
    """Enhanced parsing with multiple fallback strategies."""
    try:
        return global_error_handler.handle_data_processing(parser_func, data_source, data)
    except JSONParsingError as e:
        log_warning(f"JSON parsing failed for {data_source}, attempting cleanup")
        # Try to clean common JSON issues
        if isinstance(data, str):
            cleaned_data = data.strip()
            # Remove markdown code blocks
            if cleaned_data.startswith("```"):
                lines = cleaned_data.split('\n')
                cleaned_data = '\n'.join(lines[1:-1]) if len(lines) > 2 else cleaned_data
            try:
                return parser_func(cleaned_data)
            except Exception:
                pass
        
        raise JSONParsingError(
            message=f"Failed to parse JSON from {data_source} after cleanup attempts",
            json_content=str(data)[:200],
            user_message="Received malformed data—service may be experiencing issues. Try again.",
            context={"data_source": data_source, "attempted_cleanup": True}
        )
    except ResponseStructureError as e:
        log_warning(f"Response structure error for {data_source}")
        raise ResponseStructureError(
            message=str(e),
            missing_fields=e.missing_fields,
            user_message="Response missing expected information—try rephrasing your query",
            context={"data_source": data_source}
        )

# Using Gemini 2.0 Flash (better free tier limits: 200 requests/day vs 50)
# Enhanced LLM initialization with circuit breaker
@retry(config=api_retry_config, circuit_breaker=gemini_circuit_breaker)
def create_llm_with_retry():
    """Create LLM instance with enhanced error handling and retry logic."""
    try:
        log_info("Initializing Gemini 2.0 Flash model")
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    except Exception as e:
        log_error("Failed to initialize Gemini model", exception=e)
        if "api key" in str(e).lower() or "authentication" in str(e).lower():
            raise ThirdPartyServiceError(
                message=str(e),
                service_name="Gemini API",
                user_message="API authentication failed—check your API key in .env file",
                context={"service": "gemini", "error_type": "authentication"}
            )
        elif "quota" in str(e).lower() or "limit" in str(e).lower():
            raise ApiQuotaError(
                message=str(e),
                quota_type="requests",
                user_message="API quota exceeded—wait or upgrade plan",
                context={"service": "gemini"}
            )
        else:
            raise ThirdPartyServiceError(
                message=str(e),
                service_name="Gemini API",
                user_message="AI service temporarily unavailable—try again in a few minutes",
                context={"service": "gemini"}
            )

def create_llm():
    """Wrapper function for LLM creation with comprehensive error handling."""
    return create_llm_with_retry()

# Initialize components with error handling
try:
    llm = create_llm()
    parser = PydanticOutputParser(pydantic_object=ResponseModel)
    log_info("Successfully initialized LLM and parser")
except AIResearchAssistantError as e:
    print(f"❌ Failed to initialize AI components: {e.user_message}")
    exit(1)

prompt = ChatPromptTemplate.from_messages(
     [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

# Enhanced agent creation with retry logic
@retry(config=critical_retry_config, circuit_breaker=langchain_circuit_breaker)
def create_agent_with_retry():
    """Create agent components with enhanced error handling."""
    try:
        log_info("Creating LangChain agent components")
        agent = create_tool_calling_agent(
            llm=llm,
            prompt=prompt,
            tools=tools,
        )
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    except Exception as e:
        log_error("Failed to create agent components", exception=e)
        if "llm" in str(e).lower() or "model" in str(e).lower():
            raise ThirdPartyServiceError(
                message=str(e),
                service_name="LangChain Agent",
                user_message="Failed to create AI agent—model service may be unavailable",
                context={"component": "langchain_agent"}
            )
        else:
            raise SystemError(
                message=str(e),
                user_message="System error during agent setup—please try again",
                context={"component": "langchain_agent"}
            )

try:
    agent_executor = create_agent_with_retry()
    log_info("Successfully created agent executor")
except AIResearchAssistantError as e:
    print(f"❌ Failed to create agent: {e.user_message}")
    if isinstance(e, ThirdPartyServiceError):
        print("\n🔧 Troubleshooting:")
        print("  • Check your API key configuration")
        print("  • Verify internet connectivity")
        print("  • Try restarting the application")
    exit(1)

# Get user input with validation
@handle_data_processing_errors("user_input")
def get_user_query():
    query = input("What can i help you research? ")
    if not query or not query.strip():
        raise InputValidationError(
            "Query cannot be empty",
            field_name="query",
            user_message="Please enter a research topic or question."
        )
    return query.strip()

try:
    query = get_user_query()
except AIResearchAssistantError as e:
    user_error = display_user_error(e)
    print(f"\n❌ {user_error['title']}: {user_error['message']}")
    exit(1)

# =============================================================================
# ENHANCED MAIN EXECUTION FUNCTIONS WITH RETRY LOGIC
# =============================================================================

@retry(config=critical_retry_config, circuit_breaker=langchain_circuit_breaker)
def execute_research_query_with_retry(query: str):
    """Execute research query with comprehensive retry logic and error handling."""
    try:
        log_info(f"Executing research query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Validate query length and content
        if len(query) > 1000:
            raise InputValidationError(
                "Query too long",
                field_name="query",
                user_message="Query is too long—please shorten to under 1000 characters"
            )
        
        result = agent_executor.invoke({"query": query})
        log_info("Research query executed successfully")
        return result
        
    except Exception as e:
        log_error("Research query execution failed", exception=e)
        
        # Handle specific error types with actionable messages
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            raise ApiQuotaError(
                message=str(e),
                quota_type="requests",
                user_message="API quota exceeded—wait 60s or upgrade plan",
                retry_after_seconds=60,
                context={"operation": "research_query"}
            )
        elif "timeout" in str(e).lower():
            raise ReadTimeoutError(
                message=str(e),
                timeout_seconds=30,
                user_message="Request timeout—service slow, retry in 30s",
                context={"operation": "research_query"}
            )
        elif "connection" in str(e).lower():
            raise ConnectionTimeoutError(
                message=str(e),
                timeout_seconds=15,
                user_message="Connection failed—check network and retry",
                context={"operation": "research_query"}
            )
        else:
            raise ThirdPartyServiceError(
                message=str(e),
                service_name="AI Research Agent",
                user_message="Research service temporarily unavailable—try again in 2 minutes",
                context={"operation": "research_query"}
            )

@retry(config=RetryConfig(max_attempts=3, base_delay=0.5, jitter=True))
def parse_agent_response_with_fallback(raw_response):
    """Enhanced response parsing with multiple fallback strategies."""
    try:
        log_info("Parsing agent response")
        
        # Validate response structure
        if not isinstance(raw_response, dict):
            raise ResponseStructureError(
                message="Invalid response type",
                expected_fields=["output"],
                user_message="Invalid response format—try rephrasing your query"
            )
        
        output_text = raw_response.get("output")
        if not output_text:
            raise ResponseStructureError(
                message="No output in response",
                missing_fields=["output"],
                user_message="Empty response received—try rephrasing your query"
            )
        
        if not isinstance(output_text, str):
            output_text = str(output_text)
        
        # Progressive cleanup strategies
        cleaned_text = clean_response_text(output_text)
        
        try:
            structured_response = parser.parse(cleaned_text)
            log_info("Successfully parsed structured response")
            return structured_response
            
        except Exception as parse_error:
            log_warning(f"Primary parsing failed, attempting fallback: {parse_error}")
            
            # Fallback parsing strategies
            fallback_response = attempt_fallback_parsing(cleaned_text, output_text)
            if fallback_response:
                return fallback_response
            
            # Final fallback - return raw text in structured format
            log_warning("All parsing strategies failed, using raw text fallback")
            return create_fallback_response(output_text, str(parse_error))
            
    except AIResearchAssistantError:
        raise  # Re-raise our custom errors
    except Exception as e:
        log_error("Unexpected error during response parsing", exception=e)
        raise ParsingError(
            message=f"Response parsing failed: {str(e)}",
            data_source="agent_response",
            user_message="Unable to process response—try again or rephrase query",
            context={"original_error": str(e)}
        )

def clean_response_text(text: str) -> str:
    """Clean and normalize response text for parsing."""
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text.replace("```json\n", "").replace("\n```", "")
    elif text.startswith("```"):
        text = text.replace("```", "")
    
    # Remove common prefixes/suffixes
    text = text.strip()
    
    # Handle common formatting issues
    if text.startswith("Here's") or text.startswith("Here is"):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                text = '\n'.join(lines[i:])
                break
    
    return text.strip()

def attempt_fallback_parsing(cleaned_text: str, original_text: str) -> Optional[Dict]:
    """Attempt various fallback parsing strategies."""
    strategies = [
        lambda t: extract_json_from_text(t),
        lambda t: parse_loose_json(t),
        lambda t: extract_structured_info(t)
    ]
    
    for strategy in strategies:
        try:
            result = strategy(cleaned_text)
            if result:
                log_info(f"Fallback parsing succeeded with strategy: {strategy.__name__}")
                return result
        except Exception as e:
            log_warning(f"Fallback strategy {strategy.__name__} failed: {e}")
            continue
    
    return None

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from text that may contain extra content."""
    import re
    import json
    
    # Look for JSON-like structures
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and any(key in data for key in ['topic', 'summary']):
                return data
        except json.JSONDecodeError:
            continue
    
    return None

def parse_loose_json(text: str) -> Optional[Dict]:
    """Attempt to parse JSON with common formatting fixes."""
    import json
    import re
    
    # Common fixes
    fixes = [
        lambda t: t.replace("'", '"'),  # Single quotes to double quotes
        lambda t: re.sub(r'(\w+):', r'"\1":', t),  # Unquoted keys
        lambda t: re.sub(r': *([^"\[\{][^,\n\}\]]*)', r': "\1"', t),  # Unquoted values
    ]
    
    for fix in fixes:
        try:
            fixed_text = fix(text)
            data = json.loads(fixed_text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, Exception):
            continue
    
    return None

def extract_structured_info(text: str) -> Optional[Dict]:
    """Extract structured information from free text."""
    # Simple extraction based on common patterns
    result = {
        "topic": "Research Query",
        "summary": text[:500] + "..." if len(text) > 500 else text,
        "sources": [],
        "tools_used": ["text_extraction"]
    }
    
    # Look for topic indicators
    topic_patterns = [r"Topic:?\s*(.+)", r"Subject:?\s*(.+)", r"About:?\s*(.+)"]
    for pattern in topic_patterns:
        import re
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["topic"] = match.group(1).strip()[:100]
            break
    
    # Look for URLs or sources
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    if urls:
        result["sources"] = urls[:5]  # Limit to 5 sources
    
    return result

def create_fallback_response(text: str, error_msg: str) -> Dict:
    """Create a fallback response when all parsing fails."""
    return {
        "topic": "Research Query (Parsing Failed)",
        "summary": f"Response received but couldn't parse structured format.\n\nOriginal response:\n{text[:800]}{'...' if len(text) > 800 else ''}",
        "sources": [],
        "tools_used": ["fallback_parser"],
        "parsing_error": error_msg,
        "note": "The AI provided a response, but it couldn't be parsed into the expected format. The content is shown above."
    }

# Main wrapper functions that integrate the enhanced retry logic
def execute_research_query(query: str):
    """Main wrapper for research query execution with comprehensive error handling."""
    return execute_research_query_with_retry(query)

def parse_agent_response(raw_response):
    """Main wrapper for response parsing with enhanced fallback strategies."""
    return parse_agent_response_with_fallback(raw_response)

# Execute the main research workflow
print("\n🔍 Starting research process...")
print("=" * 50)

try:
    with error_context("research_workflow") as ctx:
        # Step 1: Execute research query
        log_info(f"Processing research query: {query[:100]}{'...' if len(query) > 100 else ''}")
        raw_response = execute_research_query(query)
        
        # Step 2: Parse and structure response
        structured_response = parse_agent_response(raw_response)
        
        # Step 3: Display results
        print("\n✅ Research completed successfully!")
        print("=" * 50)
        
        if isinstance(structured_response, dict):
            print(f"📋 Topic: {structured_response.get('topic', 'Unknown')}")
            print(f"\n📝 Summary:")
            print(structured_response.get('summary', 'No summary available'))
            
            sources = structured_response.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            
            tools = structured_response.get('tools_used', [])
            if tools:
                print(f"\n🛠️ Tools used: {', '.join(tools)}")
        else:
            # Pydantic model
            print(f"📋 Topic: {structured_response.topic}")
            print(f"\n📝 Summary:")
            print(structured_response.summary)
            
            if structured_response.sources:
                print(f"\n📚 Sources ({len(structured_response.sources)}):")
                for i, source in enumerate(structured_response.sources, 1):
                    print(f"  {i}. {source}")
            
            if structured_response.tools_used:
                print(f"\n🛠️ Tools used: {', '.join(structured_response.tools_used)}")
        
        log_info("Research workflow completed successfully")

except AIResearchAssistantError as e:
    # Handle our custom errors with user-friendly messages
    user_error = display_user_error(e)
    
    print(f"\n❌ {user_error['title']}")
    print("=" * 50)
    print(f"🔴 {user_error['message']}")
    
    if user_error['suggestions']:
        print("\n💡 Suggestions:")
        for i, suggestion in enumerate(user_error['suggestions'], 1):
            print(f"  {i}. {suggestion}")
    
    print(f"\n🆔 Error ID: {user_error['error_id']}")
    
    # Log the error for monitoring
    log_error("Research workflow failed", exception=e)
    
    # Handle specific error types with enhanced actionable context
    if isinstance(e, ApiQuotaError):
        print("\n⏰ Quota Information:")
        print("  • Daily quotas reset at midnight Pacific time")
        print("  • Gemini 2.0 Flash: 200 requests/day (free tier)")
        print("  • Consider upgrading at: https://aistudio.google.com/apikey")
        if e.retry_after_seconds:
            wait_time = e.retry_after_seconds
            if wait_time < 300:  # Less than 5 minutes
                print(f"\n⚡ Quick Action: Wait {wait_time} seconds and run the program again")
            else:
                minutes = wait_time // 60
                print(f"\n⏳ Estimated wait time: {minutes} minute(s) before retry")
    
    elif isinstance(e, NetworkError):
        print("\n🌐 Network Troubleshooting:")
        if isinstance(e, ConnectionTimeoutError):
            print("  • Check your internet connection stability")
            print("  • Try connecting to a different network (mobile hotspot)")
            print("  • Disable VPN temporarily if using one")
            print("  • Check firewall/antivirus settings")
        elif isinstance(e, ReadTimeoutError):
            print("  • Service is responding slowly - try again in 30-60 seconds")
            print("  • Check if there are service outages")
            print("  • Try with a shorter, simpler query")
        else:
            print("  • Check your internet connection")
            print("  • Try connecting to a different network")
            print("  • Verify firewall settings")
    
    elif isinstance(e, ParsingError) or isinstance(e, ResponseStructureError):
        print("\n🔧 Response Processing Help:")
        print("  • Try rephrasing your query to be more specific")
        print("  • Avoid very complex or multi-part questions")
        print("  • The AI may have responded in an unexpected format")
        print("  • Try asking for simpler information first")
    
    elif isinstance(e, ThirdPartyServiceError):
        print("\n🛠️ Service Recovery:")
        print("  • Wait 2-3 minutes and try again")
        print("  • Check Google AI Studio status")
        print("  • Verify your API key is valid and active")
        print("  • Try restarting the application")
    
    elif isinstance(e, SystemError):
        print("\n⚙️ System Troubleshooting:")
        print("  • Restart the application")
        print("  • Check available disk space")
        print("  • Verify Python environment is working correctly")
        print("  • Try running with administrator privileges")
    
    # Circuit breaker status
    if hasattr(e, 'context') and 'circuit_breaker_state' in e.context:
        cb_state = e.context['circuit_breaker_state']
        if cb_state == 'open':
            print("\n🔒 Circuit Breaker Status: OPEN")
            print("  • Service is temporarily blocked due to repeated failures")
            print("  • Wait 60 seconds for automatic recovery")
            print("  • This is a protective measure to prevent further issues")

except Exception as e:
    # Fallback for any unexpected errors
    user_error = display_user_error(e)
    print(f"\n❌ {user_error['title']}")
    print("=" * 50)
    print(f"🔴 {user_error['message']}")
    print(f"\n🆔 Error ID: {user_error['error_id']}")
    
    log_error("Unexpected error in main workflow", exception=e)

finally:
    # Display error summary for debugging
    error_summary = global_error_handler.get_error_summary()
    if error_summary['error_counts']:
        print("\n📊 Session Error Summary:")
        print("=" * 30)
        for error_type, count in error_summary['error_counts'].items():
            print(f"  • {error_type}: {count}")
        print(f"\n⏱️ Session ID: {error_summary['logger_summary']['session_id']}")

app = Flask(__name__)
CORS(app) # Add this line to enable CORS for all origins
def parse_agent_response(raw_response):
    """Main wrapper for response parsing with enhanced fallback strategies."""
    return parse_agent_response_with_fallback(raw_response)

# Execute the main research workflow
print("\n🔍 Starting research process...")
print("=" * 50)

try:
    with error_context("research_workflow") as ctx:
        # Step 1: Execute research query
        log_info(f"Processing research query: {query[:100]}{'...' if len(query) > 100 else ''}")
        raw_response = execute_research_query(query)
        
        # Step 2: Parse and structure response
        structured_response = parse_agent_response(raw_response)
        
        # Step 3: Display results
        print("\n✅ Research completed successfully!")
        print("=" * 50)
        
        if isinstance(structured_response, dict):
            print(f"📋 Topic: {structured_response.get('topic', 'Unknown')}")
            print(f"\n📝 Summary:")
            print(structured_response.get('summary', 'No summary available'))
            
            sources = structured_response.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            
            tools = structured_response.get('tools_used', [])
            if tools:
                print(f"\n🛠️ Tools used: {', '.join(tools)}")
        else:
            # Pydantic model
            print(f"📋 Topic: {structured_response.topic}")
            print(f"\n📝 Summary:")
            print(structured_response.summary)
            
            if structured_response.sources:
                print(f"\n📚 Sources ({len(structured_response.sources)}):")
                for i, source in enumerate(structured_response.sources, 1):
                    print(f"  {i}. {source}")
            
            if structured_response.tools_used:
                print(f"\n🛠️ Tools used: {', '.join(structured_response.tools_used)}")
        
        log_info("Research workflow completed successfully")

except AIResearchAssistantError as e:
    # Handle our custom errors with user-friendly messages
    user_error = display_user_error(e)
    
    print(f"\n❌ {user_error['title']}")
    print("=" * 50)
    print(f"🔴 {user_error['message']}")
    
    if user_error['suggestions']:
        print("\n💡 Suggestions:")
        for i, suggestion in enumerate(user_error['suggestions'], 1):
            print(f"  {i}. {suggestion}")
    
    print(f"\n🆔 Error ID: {user_error['error_id']}")
    
    # Log the error for monitoring
    log_error("Research workflow failed", exception=e)
    
    # Handle specific error types with enhanced actionable context
    if isinstance(e, ApiQuotaError):
        print("\n⏰ Quota Information:")
        print("  • Daily quotas reset at midnight Pacific time")
        print("  • Gemini 2.0 Flash: 200 requests/day (free tier)")
        print("  • Consider upgrading at: https://aistudio.google.com/apikey")
        if e.retry_after_seconds:
            wait_time = e.retry_after_seconds
            if wait_time < 300:  # Less than 5 minutes
                print(f"\n⚡ Quick Action: Wait {wait_time} seconds and run the program again")
            else:
                minutes = wait_time // 60
                print(f"\n⏳ Estimated wait time: {minutes} minute(s) before retry")
    
    elif isinstance(e, NetworkError):
        print("\n🌐 Network Troubleshooting:")
        if isinstance(e, ConnectionTimeoutError):
            print("  • Check your internet connection stability")
            print("  • Try connecting to a different network (mobile hotspot)")
            print("  • Disable VPN temporarily if using one")
            print("  • Check firewall/antivirus settings")
        elif isinstance(e, ReadTimeoutError):
            print("  • Service is responding slowly - try again in 30-60 seconds")
            print("  • Check if there are service outages")
            print("  • Try with a shorter, simpler query")
        else:
            print("  • Check your internet connection")
            print("  • Try connecting to a different network")
            print("  • Verify firewall settings")
    
    elif isinstance(e, ParsingError) or isinstance(e, ResponseStructureError):
        print("\n🔧 Response Processing Help:")
        print("  • Try rephrasing your query to be more specific")
        print("  • Avoid very complex or multi-part questions")
        print("  • The AI may have responded in an unexpected format")
        print("  • Try asking for simpler information first")
    
    elif isinstance(e, ThirdPartyServiceError):
        print("\n🛠️ Service Recovery:")
        print("  • Wait 2-3 minutes and try again")
        print("  • Check Google AI Studio status")
        print("  • Verify your API key is valid and active")
        print("  • Try restarting the application")
    
    elif isinstance(e, SystemError):
        print("\n⚙️ System Troubleshooting:")
        print("  • Restart the application")
        print("  • Check available disk space")
        print("  • Verify Python environment is working correctly")
        print("  • Try running with administrator privileges")
    
    # Circuit breaker status
    if hasattr(e, 'context') and 'circuit_breaker_state' in e.context:
        cb_state = e.context['circuit_breaker_state']
        if cb_state == 'open':
            print("\n🔒 Circuit Breaker Status: OPEN")
            print("  • Service is temporarily blocked due to repeated failures")
            print("  • Wait 60 seconds for automatic recovery")
            print("  • This is a protective measure to prevent further issues")

except Exception as e:
    # Fallback for any unexpected errors
    user_error = display_user_error(e)
    print(f"\n❌ {user_error['title']}")
    print("=" * 50)
    print(f"🔴 {user_error['message']}")
    print(f"\n🆔 Error ID: {user_error['error_id']}")
    
    log_error("Unexpected error in main workflow", exception=e)

finally:
    # Display error summary for debugging
    error_summary = global_error_handler.get_error_summary()
    if error_summary['error_counts']:
        print("\n📊 Session Error Summary:")
        print("=" * 30)
        for error_type, count in error_summary['error_counts'].items():
            print(f"  • {error_type}: {count}")
        print(f"\n⏱️ Session ID: {error_summary['logger_summary']['session_id']}")

        
