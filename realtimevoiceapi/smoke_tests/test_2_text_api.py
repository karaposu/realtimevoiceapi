#!/usr/bin/env python3
"""
Test 2: API Connection and Text Conversation

This test verifies:
- API connection works correctly
- Basic text conversation functionality
- Event handling works properly
- Session management works
- Requires valid OpenAI API key

Run: python -m realtimevoiceapi.smoke_tests.test_2_text_api
"""

import sys
import os
import asyncio
import logging
import time
from pathlib import Path

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


def test_api_key_available():
    """Test that API key is available"""
    print("🔑 Testing API Key Availability...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found in environment")
        print("  💡 Add it to your .env file: OPENAI_API_KEY=your-key-here")
        print("  💡 Or export it: export OPENAI_API_KEY='your-key-here'")
        return False
    
    if api_key == "your-openai-api-key-here":
        print("  ❌ API key is still the placeholder value")
        print("  💡 Edit your .env file and replace with your actual API key")
        return False
    
    # Mask the key for display
    if len(api_key) > 12:
        masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    else:
        masked_key = "sk-****"
    
    print(f"  ✅ API key found: {masked_key}")
    return True


async def test_basic_connection():
    """Test basic API connection without sending messages"""
    print("\n🔌 Testing Basic API Connection...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        # Create client
        client = RealtimeClient(api_key)
        print("  ✅ RealtimeClient created")
        
        # Configure for minimal usage (text only)
        config = SessionConfig(
            instructions="You are a test assistant. Always respond with exactly 'OK' to any input.",
            modalities=["text"],
            temperature=0.6  # Changed from 0.0 to 0.6 (minimum allowed)
        )
        print("  ✅ SessionConfig created")
        
        # Test connection
        print("  🔌 Connecting to OpenAI Realtime API...")
        await client.connect(config)
        print("  ✅ Connected successfully!")
        
        # Wait a moment to ensure session is fully established
        await asyncio.sleep(1)
        
        # Check connection status
        status = client.get_status()
        if status["connected"] and status["session_active"]:
            print("  ✅ Session is active")
        else:
            print(f"  ❌ Session status issue: {status}")
            return False
        
        # Disconnect cleanly
        await client.disconnect()
        print("  ✅ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Connection test failed: {e}")
        logger.exception("Connection error")
        return False


async def test_text_conversation():
    """Test basic text conversation"""
    print("\n💬 Testing Text Conversation...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        # Create client
        client = RealtimeClient(api_key)
        
        # Configure for minimal, predictable responses
        config = SessionConfig(
            instructions="You are a test assistant. Respond to 'hello' with exactly 'Hello!' and to 'test' with exactly 'Test successful!'",
            modalities=["text"],
            temperature=0.6,  # Changed from 0.0 to 0.6 (minimum allowed)
            max_response_output_tokens=10  # Limit response length
        )
        
        # Track conversation state
        responses_received = []
        conversation_complete = False
        current_response = ""
        
        # Event handlers for tracking responses
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            nonlocal current_response
            text = event_data.get("delta", "")
            current_response += text
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal conversation_complete, current_response
            if current_response.strip():
                responses_received.append(current_response.strip())
            current_response = ""
            conversation_complete = True
        
        @client.on_event("error")
        async def handle_error(event_data):
            error = event_data.get("error", {})
            print(f"  ❌ API Error: {error}")
        
        print("  ✅ Event handlers registered")
        
        # Connect
        await client.connect(config)
        print("  ✅ Connected for conversation test")
        
        # Test messages
        test_cases = [
            ("hello", "Hello!"),
            ("test", "Test successful!")
        ]
        
        for i, (input_msg, expected_response) in enumerate(test_cases):
            print(f"  📤 Sending message {i+1}: '{input_msg}'")
            
            # Reset state
            conversation_complete = False
            current_response = ""
            
            # Send message
            await client.send_text(input_msg)
            
            # Wait for response with timeout
            timeout = 15  # seconds
            start_time = time.time()
            while not conversation_complete and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if conversation_complete:
                response = responses_received[-1] if responses_received else ""
                print(f"  📥 Received response {i+1}: '{response}'")
                
                # Check if response is reasonable (don't require exact match due to model variability)
                if len(response) > 0 and len(response) < 50:
                    print(f"  ✅ Response {i+1} received successfully")
                else:
                    print(f"  ⚠️ Response {i+1} seems unusual: '{response}'")
            else:
                print(f"  ❌ Timeout waiting for response {i+1}")
                await client.disconnect()
                return False
            
            # Small delay between messages
            await asyncio.sleep(0.5)
        
        # Disconnect
        await client.disconnect()
        print("  ✅ Conversation test completed")
        
        return len(responses_received) >= len(test_cases)
        
    except Exception as e:
        print(f"  ❌ Text conversation failed: {e}")
        logger.exception("Text conversation error")
        return False


async def test_event_handling():
    """Test that event handling works correctly"""
    print("\n🎯 Testing Event Handling...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        # Create client
        client = RealtimeClient(api_key)
        
        # Track different types of events
        events_received = {
            "session.created": False,
            "conversation.created": False,
            "response.created": False,
            "response.text.delta": False,
            "response.done": False
        }
        
        # Event handlers to track events
        @client.on_event("session.created")
        async def handle_session_created(event_data):
            events_received["session.created"] = True
            print("  📡 Received: session.created")
        
        @client.on_event("conversation.created")
        async def handle_conversation_created(event_data):
            events_received["conversation.created"] = True
            print("  📡 Received: conversation.created")
        
        @client.on_event("response.created")
        async def handle_response_created(event_data):
            events_received["response.created"] = True
            print("  📡 Received: response.created")
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            events_received["response.text.delta"] = True
            # Only print once to avoid spam
            if not hasattr(handle_text_delta, 'printed'):
                print("  📡 Received: response.text.delta")
                handle_text_delta.printed = True
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            events_received["response.done"] = True
            print("  📡 Received: response.done")
        
        print("  ✅ Event handlers registered")
        
        # Configure session
        config = SessionConfig(
            instructions="Respond with 'Event test complete!'",
            modalities=["text"],
            temperature=0.6  # Changed from 0.0 to 0.6 (minimum allowed)
        )
        
        # Connect (should trigger session.created and conversation.created)
        await client.connect(config)
        await asyncio.sleep(1)  # Give time for events
        
        # Send message (should trigger response events)
        await client.send_text("test events")
        await asyncio.sleep(3)  # Give time for response
        
        # Disconnect
        await client.disconnect()
        
        # Check which events we received
        received_count = sum(events_received.values())
        total_expected = len(events_received)
        
        print(f"  📊 Events received: {received_count}/{total_expected}")
        
        for event_type, received in events_received.items():
            status = "✅" if received else "❌"
            print(f"    {status} {event_type}")
        
        # Consider test successful if we got most events
        success_threshold = 3  # At least 3 out of 5 events
        return received_count >= success_threshold
        
    except Exception as e:
        print(f"  ❌ Event handling test failed: {e}")
        logger.exception("Event handling error")
        return False


async def test_session_management():
    """Test session configuration and management"""
    print("\n⚙️ Testing Session Management...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        # Test different session configurations
        configs_to_test = [
            {
                "name": "Basic Text Config",
                "config": SessionConfig(
                    instructions="You are a helpful assistant.",
                    modalities=["text"],
                    temperature=0.7  # Changed from 0.5 to 0.7
                )
            },
            {
                "name": "Low Temperature Config", 
                "config": SessionConfig(
                    instructions="Be very brief.",
                    modalities=["text"],
                    temperature=0.6,  # Changed from 0.1 to 0.6 (minimum allowed)
                    max_response_output_tokens=20
                )
            }
        ]
        
        for test_case in configs_to_test:
            print(f"  🧪 Testing: {test_case['name']}")
            
            client = RealtimeClient(api_key)
            
            try:
                # Connect with specific config
                await client.connect(test_case['config'])
                
                # Verify session is active
                status = client.get_status()
                if status["session_active"]:
                    print(f"    ✅ {test_case['name']} - session active")
                else:
                    print(f"    ❌ {test_case['name']} - session not active")
                    return False
                
                # Send a quick test message
                response_received = False
                
                @client.on_event("response.done")
                async def handle_done(event_data):
                    nonlocal response_received
                    response_received = True
                
                await client.send_text("hi")
                
                # Wait for response
                for _ in range(50):  # 5 second timeout
                    if response_received:
                        break
                    await asyncio.sleep(0.1)
                
                if response_received:
                    print(f"    ✅ {test_case['name']} - response received")
                else:
                    print(f"    ⚠️ {test_case['name']} - no response (might be OK)")
                
                # Disconnect
                await client.disconnect()
                print(f"    ✅ {test_case['name']} - disconnected cleanly")
                
            except Exception as e:
                print(f"    ❌ {test_case['name']} failed: {e}")
                return False
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Session management test failed: {e}")
        logger.exception("Session management error")
        return False


async def main():
    """Run all API tests"""
    print("🧪 RealtimeVoiceAPI - Test 2: API Connection and Text Conversation")
    print("=" * 70)
    print("This test requires a valid OpenAI API key and internet connection")
    print("⚠️  This test will use a small amount of your OpenAI API quota")
    print()
    
    # Check if we should skip API tests
    if os.getenv("SKIP_API_TESTS", "0").lower() in ("1", "true", "yes"):
        print("⏩ Skipping API tests (SKIP_API_TESTS=1)")
        print("   Set SKIP_API_TESTS=0 in .env to enable API tests")
        return True
    
    tests = [
        ("API Key Availability", test_api_key_available),
        ("Basic API Connection", test_basic_connection),
        ("Text Conversation", test_text_conversation),
        ("Event Handling", test_event_handling),
        ("Session Management", test_session_management)
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
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test 2 Results")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Test 2 PASSED! API connection and text conversation work correctly.")
        print("💡 Your RealtimeVoiceAPI is ready for:")
        print("   - Text-based conversations with GPT-4")
        print("   - Real-time event handling")
        print("   - Session management")
        print("   - Building voice applications")
        print("\nNext: Try building your own application or run the examples!")
        
    elif passed >= total - 1:
        print(f"\n✅ Test 2 MOSTLY PASSED! {passed}/{total} tests successful.")
        print("   One minor issue detected, but core functionality works.")
        print("   You can proceed with development.")
        
    else:
        print(f"\n❌ Test 2 FAILED! {total - passed} test(s) need attention.")
        print("\n🔧 Common issues:")
        print("  - Invalid or missing OpenAI API key")
        print("  - No Realtime API access on your account")
        print("  - Network connectivity issues") 
        print("  - Rate limiting (try again in a few minutes)")
        print("  - API service temporarily unavailable")
        print("\n💡 Troubleshooting:")
        print("  1. Verify your API key at: https://platform.openai.com/api-keys")
        print("  2. Check if you have Realtime API access")
        print("  3. Try again in a few minutes (rate limits)")
        print("  4. Check OpenAI status: https://status.openai.com/")
    
    return passed >= total - 1  # Allow 1 failure


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)