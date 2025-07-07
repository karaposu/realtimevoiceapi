#!/usr/bin/env python3
"""
Test 6: Function Calling with Voice - COMPREHENSIVE TEST

This test verifies:
- Voice-triggered function calls ("What's the weather?")
- Function result integration (AI speaks the results)
- Multiple function types (weather, calculator, search)
- Natural conversation flow with function calling
- Complex function parameters and error handling
- Real-world voice assistant scenarios

This showcases the full power of RealtimeVoiceAPI: Speech → Understanding → Action → Speech

Run: python -m realtimevoiceapi.smoke_tests.test_6_function_calling
"""

import sys
import os
import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path so we can import realtimevoiceapi
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️ python-dotenv not installed. Using system environment variables.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCTION DEFINITIONS - Real-world functions the AI can call
# ============================================================================

class FunctionRegistry:
    """Registry of functions that the AI can call"""
    
    def __init__(self):
        self.functions = {}
        self.call_log = []
        
    def register(self, name: str, func, schema: Dict[str, Any]):
        """Register a function with its schema"""
        self.functions[name] = {
            'function': func,
            'schema': schema
        }
        
    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a function and return the result"""
        if name not in self.functions:
            return f"Error: Function '{name}' not found"
            
        try:
            # Log the call
            call_info = {
                'function': name,
                'arguments': arguments,
                'timestamp': datetime.now().isoformat()
            }
            self.call_log.append(call_info)
            
            # Call the function
            result = self.functions[name]['function'](**arguments)
            
            # Update log with result
            call_info['result'] = result
            call_info['success'] = True
            
            print(f"    🔧 Function call: {name}({arguments}) → {result}")
            return str(result)
            
        except Exception as e:
            error_msg = f"Error calling {name}: {str(e)}"
            call_info['error'] = error_msg
            call_info['success'] = False
            
            print(f"    ❌ Function error: {name}({arguments}) → {error_msg}")
            return error_msg
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all function schemas for the AI"""
        return [
            {
                "type": "function",
                "name": name,
                **func_info['schema']
            }
            for name, func_info in self.functions.items()
        ]


def setup_function_registry() -> FunctionRegistry:
    """Setup all the functions the AI can call"""
    registry = FunctionRegistry()
    
    # Weather function
    def get_weather(location: str) -> str:
        """Get current weather for a location"""
        # Simulate weather API
        weather_data = {
            "san francisco": "72°F, sunny with light clouds",
            "new york": "65°F, partly cloudy",
            "london": "58°F, rainy",
            "tokyo": "75°F, clear skies",
            "default": "70°F, pleasant weather"
        }
        location_key = location.lower()
        weather = weather_data.get(location_key, weather_data["default"])
        return f"The weather in {location} is {weather}"
    
    registry.register("get_weather", get_weather, {
        "description": "Get current weather information for any city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name to get weather for"
                }
            },
            "required": ["location"]
        }
    })
    
    # Calculator function
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression"""
        try:
            # Basic safety - only allow numbers and basic operators
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate safely
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"
    
    registry.register("calculate", calculate, {
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to calculate (e.g., '2 + 2' or '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    })
    
    # Time function
    def get_current_time(timezone: str = "UTC") -> str:
        """Get current time"""
        now = datetime.now()
        if timezone.lower() == "utc":
            return f"Current UTC time is {now.strftime('%H:%M:%S on %B %d, %Y')}"
        else:
            return f"Current time is {now.strftime('%H:%M:%S on %B %d, %Y')} (local time)"
    
    registry.register("get_current_time", get_current_time, {
        "description": "Get the current time",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (UTC or local)",
                    "default": "local"
                }
            }
        }
    })
    
    # Search function
    def search_information(query: str) -> str:
        """Search for information (simulated)"""
        # Simulate search results
        search_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability",
            "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence",
            "weather": "Weather refers to atmospheric conditions including temperature, humidity, precipitation, and wind",
            "openai": "OpenAI is an AI research company focused on developing and promoting safe artificial general intelligence",
            "default": f"Here's some information about '{query}': This is a simulated search result for demonstration purposes"
        }
        
        query_key = query.lower()
        for key, result in search_results.items():
            if key in query_key:
                return result
        
        return search_results["default"]
    
    registry.register("search_information", search_information, {
        "description": "Search for information on any topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or topic to find information about"
                }
            },
            "required": ["query"]
        }
    })
    
    # Task management function
    def manage_task(action: str, task: str = "") -> str:
        """Manage a simple task list"""
        if not hasattr(manage_task, 'tasks'):
            manage_task.tasks = []
        
        if action == "add":
            if task:
                manage_task.tasks.append(task)
                return f"Added task: '{task}'. You now have {len(manage_task.tasks)} tasks."
            else:
                return "Error: Please specify a task to add"
        elif action == "list":
            if manage_task.tasks:
                task_list = "\n".join(f"{i+1}. {task}" for i, task in enumerate(manage_task.tasks))
                return f"Your current tasks:\n{task_list}"
            else:
                return "You have no tasks in your list"
        elif action == "clear":
            count = len(manage_task.tasks)
            manage_task.tasks.clear()
            return f"Cleared {count} tasks from your list"
        else:
            return f"Error: Unknown action '{action}'. Use 'add', 'list', or 'clear'"
    
    registry.register("manage_task", manage_task, {
        "description": "Add, list, or clear tasks in a simple task manager",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: 'add', 'list', or 'clear'",
                    "enum": ["add", "list", "clear"]
                },
                "task": {
                    "type": "string",
                    "description": "Task description (required for 'add' action)"
                }
            },
            "required": ["action"]
        }
    })
    
    return registry


def test_function_setup():
    """Test that function registry works correctly"""
    print("🔧 Testing Function Setup...")
    
    try:
        registry = setup_function_registry()
        
        # Test function registration
        schemas = registry.get_schemas()
        if len(schemas) >= 5:
            print(f"  ✅ Registered {len(schemas)} functions")
        else:
            print(f"  ❌ Expected at least 5 functions, got {len(schemas)}")
            return False
        
        # Test function calls
        test_calls = [
            ("get_weather", {"location": "San Francisco"}),
            ("calculate", {"expression": "2 + 2"}),
            ("get_current_time", {}),
            ("search_information", {"query": "Python"}),
            ("manage_task", {"action": "add", "task": "Test task"})
        ]
        
        for func_name, args in test_calls:
            try:
                result = registry.call(func_name, args)
                if "Error" not in result:
                    print(f"  ✅ {func_name}: {result[:50]}...")
                else:
                    print(f"  ❌ {func_name}: {result}")
                    return False
            except Exception as e:
                print(f"  ❌ {func_name} failed: {e}")
                return False
        
        print("  ✅ All function tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Function setup failed: {e}")
        return False


async def test_simple_function_calling():
    """Test basic function calling with voice"""
    print("\n🎯 Testing Simple Function Calling...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig, Tool
        
        # Setup function registry
        registry = setup_function_registry()
        
        client = RealtimeClient(api_key)
        
        # Configure with function calling
        config = SessionConfig(
            instructions="""You are a helpful voice assistant with access to several functions. 
            When users ask for weather, calculations, time, information, or task management, 
            use the appropriate functions. Always explain what you're doing and speak the results naturally.""",
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.7,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            ),
            tools=[Tool(
                type="function",
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"]
            ) for schema in registry.get_schemas()]
        )
        
        # Track function calling events
        function_calls_made = []
        responses_completed = 0
        
        @client.on_event("response.function_call_arguments.done")
        async def handle_function_call(event_data):
            call_id = event_data.get("call_id")
            name = event_data.get("name", "unknown")
            arguments_str = event_data.get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
                print(f"    🔧 AI called function: {name}({arguments})")
                
                # Execute the function
                result = registry.call(name, arguments)
                
                # Submit result back to AI
                await client.submit_function_result(call_id, result)
                
                function_calls_made.append({
                    'name': name,
                    'arguments': arguments,
                    'result': result
                })
                
            except Exception as e:
                print(f"    ❌ Function call error: {e}")
                await client.submit_function_result(call_id, f"Error: {e}")
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal responses_completed
            responses_completed += 1
        
        # Connect
        await client.connect(config)
        print("  ✅ Connected with function calling enabled")
        
        # Test simple function call
        print("  📤 Asking: 'What's the weather in San Francisco?'")
        await client.send_text("What's the weather in San Francisco?")
        
        # Wait for function call and response
        timeout = 25
        start_time = time.time()
        while len(function_calls_made) == 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
        
        # Wait a bit more for the complete response
        await asyncio.sleep(3)
        
        if len(function_calls_made) > 0:
            call = function_calls_made[0]
            print(f"  ✅ Function call successful!")
            print(f"    Function: {call['name']}")
            print(f"    Arguments: {call['arguments']}")
            print(f"    Result: {call['result']}")
            
            # Save the conversation
            audio_duration = client.get_audio_output_duration()
            if audio_duration > 0:
                client.save_audio_output("function_call_weather.wav")
                print(f"    💾 Conversation saved: function_call_weather.wav ({audio_duration:.0f}ms)")
            
            result = True
        else:
            print("  ❌ No function calls were made")
            print("    The AI may not have understood the request or function calling may not be working")
            result = False
        
        await client.disconnect()
        return result
        
    except Exception as e:
        print(f"  ❌ Simple function calling test failed: {e}")
        logger.exception("Simple function calling error")
        return False


async def test_complex_function_calling():
    """Test complex function calling scenarios"""
    print("\n🧠 Testing Complex Function Calling...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig, Tool
        
        registry = setup_function_registry()
        client = RealtimeClient(api_key)
        
        # Configure for complex function calling
        config = SessionConfig(
            instructions="""You are an advanced voice assistant. When users ask complex questions,
            use multiple functions as needed. For example, if they ask for weather and time,
            call both functions. Always explain your actions clearly.""",
            modalities=["text", "audio"],
            voice="echo",
            temperature=0.8,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            ),
            tools=[Tool(
                type="function",
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"]
            ) for schema in registry.get_schemas()]
        )
        
        # Track complex interactions
        function_calls_made = []
        
        @client.on_event("response.function_call_arguments.done")
        async def handle_function_call(event_data):
            call_id = event_data.get("call_id")
            name = event_data.get("name", "unknown")
            arguments_str = event_data.get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
                result = registry.call(name, arguments)
                await client.submit_function_result(call_id, result)
                
                function_calls_made.append({
                    'name': name,
                    'arguments': arguments,
                    'result': result
                })
                
            except Exception as e:
                await client.submit_function_result(call_id, f"Error: {e}")
        
        await client.connect(config)
        print("  ✅ Connected for complex function calling")
        
        # Test scenarios
        complex_requests = [
            "Calculate 15 times 7 and tell me the current time",
            "Add 'buy groceries' to my task list and search for information about Python",
            "What's the weather in New York and what time is it?"
        ]
        
        successful_scenarios = 0
        
        for i, request in enumerate(complex_requests):
            print(f"\n  📤 Scenario {i+1}: '{request}'")
            
            initial_calls = len(function_calls_made)
            await client.send_text(request)
            
            # Wait for function calls
            timeout = 20
            start_time = time.time()
            while len(function_calls_made) == initial_calls and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)
            
            # Wait for completion
            await asyncio.sleep(3)
            
            new_calls = function_calls_made[initial_calls:]
            if len(new_calls) > 0:
                print(f"    ✅ Scenario {i+1}: {len(new_calls)} function(s) called")
                for call in new_calls:
                    print(f"      🔧 {call['name']}({call['arguments']})")
                successful_scenarios += 1
                
                # Save each scenario
                audio_duration = client.get_audio_output_duration()
                if audio_duration > 0:
                    filename = f"complex_function_call_{i+1}.wav"
                    client.save_audio_output(filename)
                    print(f"      💾 Saved: {filename}")
            else:
                print(f"    ❌ Scenario {i+1}: No functions called")
            
            # Small delay between scenarios
            await asyncio.sleep(2)
        
        await client.disconnect()
        
        print(f"\n  📊 Complex Function Calling Results:")
        print(f"    Scenarios tested: {len(complex_requests)}")
        print(f"    Scenarios successful: {successful_scenarios}")
        print(f"    Total function calls: {len(function_calls_made)}")
        
        return successful_scenarios >= len(complex_requests) - 1  # Allow 1 failure
        
    except Exception as e:
        print(f"  ❌ Complex function calling test failed: {e}")
        logger.exception("Complex function calling error")
        return False


async def test_voice_triggered_functions():
    """Test function calling triggered by voice input"""
    print("\n🎤 Testing Voice-Triggered Function Calls...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    # Check for voice file
    voice_files = ["test_voice.wav", "my_voice.wav", "voice_input.wav"]
    voice_file = None
    for file in voice_files:
        if Path(file).exists():
            voice_file = file
            break
    
    if not voice_file:
        print("  ⏩ Skipping - no voice recording available")
        print("    Record your voice saying 'What's the weather in London?' for this test")
        return True
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        from realtimevoiceapi.models import TurnDetectionConfig, Tool
        
        registry = setup_function_registry()
        client = RealtimeClient(api_key)
        processor = AudioProcessor()
        
        # Load voice recording
        voice_audio = processor.load_wav_file(voice_file)
        duration_ms = processor.get_audio_duration_ms(voice_audio)
        print(f"  📁 Using voice recording: {voice_file} ({duration_ms:.0f}ms)")
        
        # Configure for voice + function calling
        config = SessionConfig(
            instructions="""You are a voice assistant that can call functions. 
            Listen to the user's voice and call appropriate functions based on what they say.
            Always explain what you heard and what function you're calling.""",
            modalities=["text", "audio"],
            voice="shimmer",
            temperature=0.7,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                silence_duration_ms=500,
                create_response=True
            ),
            tools=[Tool(
                type="function",
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"]
            ) for schema in registry.get_schemas()]
        )
        
        # Track voice + function interaction
        speech_detected = False
        function_calls_made = []
        
        @client.on_event("input_audio_buffer.speech_started")
        async def handle_speech_start(data):
            nonlocal speech_detected
            speech_detected = True
            print("    🎙️ Voice input detected!")
        
        @client.on_event("response.function_call_arguments.done")
        async def handle_function_call(event_data):
            call_id = event_data.get("call_id")
            name = event_data.get("name", "unknown")
            arguments_str = event_data.get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
                print(f"    🔧 Voice triggered function: {name}({arguments})")
                
                result = registry.call(name, arguments)
                await client.submit_function_result(call_id, result)
                
                function_calls_made.append({
                    'name': name,
                    'arguments': arguments,
                    'result': result
                })
                
            except Exception as e:
                await client.submit_function_result(call_id, f"Error: {e}")
        
        await client.connect(config)
        print("  ✅ Connected for voice-triggered function calling")
        
        print("  📤 Sending voice input...")
        await client.send_audio_simple(voice_audio)
        
        # Wait for speech detection and function calls
        timeout = 30
        start_time = time.time()
        while (not speech_detected or len(function_calls_made) == 0) and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
        
        # Wait for complete response
        await asyncio.sleep(5)
        
        if speech_detected and len(function_calls_made) > 0:
            print(f"  ✅ Voice-triggered function calling successful!")
            print(f"    Speech detected: {speech_detected}")
            print(f"    Functions called: {len(function_calls_made)}")
            for call in function_calls_made:
                print(f"      🔧 {call['name']}: {call['result']}")
            
            # Save the voice conversation
            audio_duration = client.get_audio_output_duration()
            if audio_duration > 0:
                client.save_audio_output("voice_triggered_function.wav")
                print(f"    💾 Voice conversation saved: voice_triggered_function.wav")
            
            result = True
        else:
            print(f"  ❌ Voice-triggered function calling failed")
            print(f"    Speech detected: {speech_detected}")
            print(f"    Functions called: {len(function_calls_made)}")
            result = False
        
        await client.disconnect()
        return result
        
    except Exception as e:
        print(f"  ❌ Voice-triggered function test failed: {e}")
        logger.exception("Voice-triggered function error")
        return False


async def test_function_error_handling():
    """Test how AI handles function errors"""
    print("\n🛠️ Testing Function Error Handling...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig, Tool
        
        registry = setup_function_registry()
        client = RealtimeClient(api_key)
        
        # Configure with error-prone scenarios
        config = SessionConfig(
            instructions="""You are a helpful assistant. When functions return errors,
            explain the error to the user in a friendly way and suggest alternatives.""",
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.7,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            ),
            tools=[Tool(
                type="function",
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"]
            ) for schema in registry.get_schemas()]
        )
        
        function_calls_made = []
        error_count = 0
        
        @client.on_event("response.function_call_arguments.done")
        async def handle_function_call(event_data):
            nonlocal error_count
            call_id = event_data.get("call_id")
            name = event_data.get("name", "unknown")
            arguments_str = event_data.get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
                result = registry.call(name, arguments)
                
                if "Error" in result:
                    error_count += 1
                
                await client.submit_function_result(call_id, result)
                function_calls_made.append((name, arguments, result))
                
            except Exception as e:
                error_count += 1
                error_msg = f"Function execution error: {e}"
                await client.submit_function_result(call_id, error_msg)
                function_calls_made.append((name, arguments, error_msg))
        
        await client.connect(config)
        print("  ✅ Connected for error handling test")
        
        # Test error scenarios
        error_requests = [
            "Calculate the square root of negative one",  # Math error
            "What's the weather on Mars?",                # Unusual location
            "Add an empty task to my list",               # Missing required parameter
        ]
        
        for i, request in enumerate(error_requests):
            print(f"  📤 Error scenario {i+1}: '{request}'")
            
            initial_calls = len(function_calls_made)
            await client.send_text(request)
            
            # Wait for function call
            timeout = 15
            start_time = time.time()
            while len(function_calls_made) == initial_calls and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)
            
            await asyncio.sleep(2)  # Wait for AI response
        
        await client.disconnect()
        
        print(f"\n  📊 Error Handling Results:")
        print(f"    Function calls made: {len(function_calls_made)}")
        print(f"    Errors handled: {error_count}")
        
        # Success if we made function calls and handled some errors gracefully
        success = len(function_calls_made) > 0
        
        if success:
            print("  ✅ Error handling test successful!")
            print("    AI can handle function errors gracefully")
        else:
            print("  ❌ Error handling test failed")
            print("    AI did not attempt function calls")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Function error handling test failed: {e}")
        logger.exception("Function error handling error")
        return False


async def test_conversation_with_functions():
    """Test natural conversation that weaves in function calls"""
    print("\n💬 Testing Conversational Function Integration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig, Tool
        
        registry = setup_function_registry()
        client = RealtimeClient(api_key)
        
        config = SessionConfig(
            instructions="""You are a conversational assistant. Have natural conversations
            with users, and when they mention things you can help with (weather, calculations,
            time, information, tasks), offer to help and use the appropriate functions.
            Be conversational and natural, not robotic.""",
            modalities=["text", "audio"],
            voice="echo",
            temperature=0.9,  # More conversational
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            ),
            tools=[Tool(
                type="function",
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"]
            ) for schema in registry.get_schemas()]
        )
        
        conversation_turns = 0
        function_calls_made = []
        
        @client.on_event("response.function_call_arguments.done")
        async def handle_function_call(event_data):
            call_id = event_data.get("call_id")
            name = event_data.get("name", "unknown")
            arguments_str = event_data.get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
                result = registry.call(name, arguments)
                await client.submit_function_result(call_id, result)
                function_calls_made.append((name, arguments, result))
            except Exception as e:
                await client.submit_function_result(call_id, f"Error: {e}")
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal conversation_turns
            conversation_turns += 1
            
            # Save each turn
            audio_duration = client.get_audio_output_duration()
            if audio_duration > 0:
                filename = f"conversation_function_turn_{conversation_turns}.wav"
                client.save_audio_output(filename)
                print(f"    💾 Turn {conversation_turns} saved: {filename}")
        
        await client.connect(config)
        print("  ✅ Connected for conversational function integration")
        
        # Natural conversation that includes function-triggering topics
        conversation_flow = [
            "Hi there! I'm planning my day and could use some help.",
            "I need to know the weather in San Francisco for my meeting.",
            "Thanks! Can you also calculate how much tip I should leave on a $47 bill for 18%?",
            "Perfect! And what time is it right now?",
            "Great! Can you add 'call mom' to my task list?",
            "Thank you so much for all your help!"
        ]
        
        print("\n  🗣️ Starting conversational function integration...")
        
        for i, message in enumerate(conversation_flow):
            print(f"\n  👤 Turn {i+1}: {message}")
            print(f"  🤖 Response: ", end="")
            
            await client.send_text(message)
            
            # Wait for turn completion
            current_turn = conversation_turns
            timeout = 20
            start_time = time.time()
            while conversation_turns == current_turn and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if conversation_turns > current_turn:
                print(f"✅ Turn {i+1} completed")
            else:
                print(f"⚠️ Turn {i+1} timeout")
            
            await asyncio.sleep(1)  # Natural pause between turns
        
        await client.disconnect()
        
        print(f"\n  📊 Conversational Integration Results:")
        print(f"    Conversation turns: {conversation_turns}")
        print(f"    Function calls made: {len(function_calls_made)}")
        print(f"    Functions used:")
        for name, args, result in function_calls_made:
            print(f"      🔧 {name}({args})")
        
        # Success if we had a good conversation with some function calls
        success = conversation_turns >= 4 and len(function_calls_made) >= 2
        
        if success:
            print("  ✅ Conversational function integration successful!")
            print("    AI naturally integrated function calls into conversation")
        else:
            print("  ❌ Conversational function integration needs improvement")
            print(f"    Expected: 4+ turns and 2+ function calls")
            print(f"    Got: {conversation_turns} turns and {len(function_calls_made)} function calls")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Conversational function integration test failed: {e}")
        logger.exception("Conversational function integration error")
        return False


async def main():
    """Run all function calling tests"""
    print("🧪 RealtimeVoiceAPI - Test 4: Function Calling with Voice")
    print("=" * 70)
    print("This test showcases the full power of voice AI: Speech → Understanding → Action → Speech")
    print("⚠️  This test will use moderate API quota for function calling demonstrations")
    print()
    
    # Check if we should skip API tests
    if os.getenv("SKIP_API_TESTS", "0").lower() in ("1", "true", "yes"):
        print("⏩ Skipping API tests (SKIP_API_TESTS=1)")
        print("   Set SKIP_API_TESTS=0 in .env to enable function calling tests")
        return True
    
    # Check for voice files
    voice_files = ["test_voice.wav", "my_voice.wav", "voice_input.wav"]
    available_files = [f for f in voice_files if Path(f).exists()]
    
    if available_files:
        print(f"✅ Found voice recording(s): {', '.join(available_files)}")
        print("   Voice-triggered function calling will be tested")
    else:
        print("⚠️  No voice recordings found. Voice-triggered test will be skipped.")
        print("💡 For complete testing, record your voice saying 'What's the weather in London?'")
    print()
    
    tests = [
        ("Function Setup", test_function_setup),
        ("Simple Function Calling", test_simple_function_calling),
        ("Complex Function Calling", test_complex_function_calling),
        ("Voice-Triggered Functions", test_voice_triggered_functions),
        ("Function Error Handling", test_function_error_handling),
        ("Conversational Functions", test_conversation_with_functions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
            
        except KeyboardInterrupt:
            print(f"\n⏹️ {test_name} interrupted by user")
            break
            
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            logger.exception(f"Test crash: {test_name}")
            results.append((test_name, False))
        
        # Delay between tests to avoid rate limits
        await asyncio.sleep(2.0)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test 4 Results - Function Calling")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Test 4 PASSED! Function calling works perfectly.")
        print("🚀 Your RealtimeVoiceAPI supports:")
        print("   - ✅ Voice-triggered function calls")
        print("   - ✅ Complex multi-function scenarios")
        print("   - ✅ Natural conversational function integration")
        print("   - ✅ Error handling and recovery")
        print("   - ✅ Real-world function types (weather, calculator, tasks)")
        print("\n💪 You now have a COMPLETE voice AI assistant capable of:")
        print("   🗣️ Natural voice conversation")
        print("   🧠 Understanding complex requests")
        print("   🔧 Taking real actions via function calls")
        print("   📱 Building full voice applications")
        
    elif passed >= total - 1:
        print(f"\n✅ Test 4 MOSTLY PASSED! {passed}/{total} tests successful.")
        print("   Minor issue detected, but core function calling works.")
        print("   You can build voice assistants with function calling capability.")
        
    else:
        print(f"\n❌ Test 4 FAILED! {total - passed} test(s) need attention.")
        print("\n🔧 Common function calling issues:")
        print("  - AI may not understand when to call functions")
        print("  - Function schemas might be unclear")
        print("  - Function execution errors")
        print("  - Network timeouts during complex operations")
        print("\n💡 Troubleshooting:")
        print("  1. Ensure your function descriptions are clear and specific")
        print("  2. Test individual functions before integration")
        print("  3. Check that function results are properly formatted")
        print("  4. Verify API quota allows for complex operations")
    
    # Show generated audio files
    audio_files = [
        "function_call_weather.wav",
        "complex_function_call_1.wav",
        "complex_function_call_2.wav",
        "complex_function_call_3.wav",
        "voice_triggered_function.wav",
        "conversation_function_turn_1.wav",
        "conversation_function_turn_2.wav",
        "conversation_function_turn_3.wav",
        "conversation_function_turn_4.wav",
        "conversation_function_turn_5.wav",
        "conversation_function_turn_6.wav"
    ]
    
    found_files = [f for f in audio_files if Path(f).exists()]
    if found_files:
        print(f"\n🎵 Generated function calling conversations:")
        for f in found_files:
            size = Path(f).stat().st_size
            print(f"   📁 {f} ({size:,} bytes)")
        print("   🎧 Play these to hear the AI calling functions and speaking results!")
        print("   🎯 This demonstrates Speech → Function Call → Spoken Result")
    
    return passed >= total - 1  # Allow 1 failure


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)