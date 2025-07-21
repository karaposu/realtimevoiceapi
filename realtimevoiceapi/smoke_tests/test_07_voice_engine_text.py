"""
Test 07: VoiceEngine Text Interactions
Tests text-based interactions with REAL OpenAI API connections.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
from realtimevoiceapi.voice_engine import VoiceEngine
from realtimevoiceapi.core.stream_protocol import StreamEventType

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

async def test_text_conversation():
    """Test text conversation with real API"""
    print("\n=== Test 1: Text Conversation ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track responses
        text_responses = []
        audio_chunks = []
        errors = []
        response_complete = False
        
        # Setup handlers
        engine.on_text_response = lambda text: text_responses.append(text)
        engine.on_audio_response = lambda audio: audio_chunks.append(audio)
        engine.on_error = lambda error: errors.append(error)
        engine.on_response_done = lambda: globals().update({'response_complete': True})
        
        # Connect to API
        await engine.connect()
        print("✓ Connected to OpenAI API")
        
        # Send a text message
        test_message = "Hello! Please respond with a short greeting."
        await engine.send_text(test_message)
        print(f"✓ Sent text: '{test_message}'")
        
        # Wait for response (with timeout)
        start_time = time.time()
        timeout = 10  # 10 seconds timeout
        
        while not response_complete and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        
        # Check results
        print(f"✓ Response received:")
        print(f"  Text responses: {len(text_responses)}")
        print(f"  Audio chunks: {len(audio_chunks)}")
        print(f"  Errors: {len(errors)}")
        
        if text_responses:
            print(f"  AI said: '{text_responses[0][:100]}...'")
        
        if audio_chunks:
            total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
            print(f"  Total audio bytes: {total_audio_bytes}")
        
        # Disconnect
        await engine.disconnect()
        print("✓ Disconnected from API")
        
        # Success if we got either text or audio response
        return len(text_responses) > 0 or len(audio_chunks) > 0
        
    except Exception as e:
        print(f"✗ Text conversation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_messages():
    """Test sending multiple messages"""
    print("\n=== Test 2: Multiple Messages ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track all responses
        all_responses = []
        
        engine.on_text_response = lambda text: all_responses.append(('text', text))
        engine.on_audio_response = lambda audio: all_responses.append(('audio', len(audio)))
        
        # Connect
        await engine.connect()
        print("✓ Connected to API")
        
        # Send multiple messages
        messages = [
            "What is 2 + 2?",
            "What color is the sky?",
            "Say goodbye!"
        ]
        
        for msg in messages:
            response_count_before = len(all_responses)
            
            await engine.send_text(msg)
            print(f"✓ Sent: '{msg}'")
            
            # Wait for response
            await asyncio.sleep(3)  # Give time for response
            
            response_count_after = len(all_responses)
            print(f"  Received {response_count_after - response_count_before} responses")
        
        # Disconnect
        await engine.disconnect()
        print("✓ Disconnected")
        
        print(f"✓ Total responses: {len(all_responses)}")
        
        return len(all_responses) >= len(messages)
        
    except Exception as e:
        print(f"✗ Multiple messages test failed: {e}")
        return False

async def test_interruption():
    """Test interrupting a response"""
    print("\n=== Test 3: Response Interruption ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track state
        response_started = False
        response_interrupted = False
        
        def on_audio(audio):
            nonlocal response_started
            response_started = True
        
        engine.on_audio_response = on_audio
        
        # Connect
        await engine.connect()
        print("✓ Connected to API")
        
        # Send a message that will generate a long response
        await engine.send_text("Count slowly from 1 to 20, pausing between each number.")
        print("✓ Sent message for long response")
        
        # Wait for response to start
        await asyncio.sleep(1)
        
        if response_started:
            # Interrupt
            await engine.interrupt()
            response_interrupted = True
            print("✓ Interrupted response")
        else:
            print("⚠ Response didn't start in time to interrupt")
        
        # Disconnect
        await engine.disconnect()
        
        return True  # Test passes if no errors
        
    except Exception as e:
        print(f"✗ Interruption test failed: {e}")
        return False

async def test_metrics_after_conversation():
    """Test metrics collection after real conversation"""
    print("\n=== Test 4: Metrics After Conversation ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Connect and have a conversation
        await engine.connect()
        
        # Send a few messages
        await engine.send_text("Hello!")
        await asyncio.sleep(2)
        
        await engine.send_text("What's 5 + 5?")
        await asyncio.sleep(2)
        
        # Get metrics (not async)
        metrics = engine.get_metrics()
        print("✓ Retrieved metrics:")
        print(f"  Total interactions: {metrics.get('total_interactions', 0)}")
        print(f"  Audio chunks sent: {metrics.get('audio_chunks_sent', 0)}")
        print(f"  Audio chunks received: {metrics.get('audio_chunks_received', 0)}")
        print(f"  Is connected: {metrics.get('is_connected', False)}")
        
        # Get usage
        usage = await engine.get_usage()
        print(f"✓ Usage: {usage}")
        
        # Estimate cost
        cost = await engine.estimate_cost()
        print(f"✓ Estimated cost: {cost}")
        
        # Disconnect
        await engine.disconnect()
        
        # Accept the test as passing if we got metrics and usage
        return metrics is not None and usage is not None
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("VoiceEngine Text Interaction Tests")
    print("=" * 60)
    
    results = []
    results.append(("Text Conversation", await test_text_conversation()))
    results.append(("Multiple Messages", await test_multiple_messages()))
    results.append(("Interruption", await test_interruption()))
    results.append(("Metrics After Conversation", await test_metrics_after_conversation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)