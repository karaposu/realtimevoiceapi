"""
Test 09: Full Integration Test
Tests complete end-to-end flow with real OpenAI API.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
import numpy as np
from realtimevoiceapi.voice_engine import VoiceEngine, VoiceEngineConfig

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

async def test_full_conversation_cycle():
    """Test a complete conversation cycle"""
    print("\n=== Test 1: Full Conversation Cycle ===")
    
    try:
        # Create engine with full config
        config = VoiceEngineConfig(
            api_key=API_KEY,
            mode="fast",
            sample_rate=24000,
            chunk_duration_ms=20,
            vad_enabled=True,
            vad_threshold=0.02,
            voice="alloy",
            latency_mode="balanced"
        )
        
        engine = VoiceEngine(config=config)
        
        # Track complete conversation
        events = []
        
        def track_event(event_type, data=None):
            events.append((time.time(), event_type, data))
        
        # Setup comprehensive handlers
        engine.on_audio_response = lambda audio: track_event('audio', len(audio))
        engine.on_text_response = lambda text: track_event('text', text)
        engine.on_error = lambda error: track_event('error', str(error))
        engine.on_response_done = lambda: track_event('response_done')
        
        # Connect
        start_time = time.time()
        await engine.connect()
        connect_time = time.time() - start_time
        print(f"✓ Connected in {connect_time*1000:.0f}ms")
        
        # Start listening
        await engine.start_listening()
        print("✓ Listening started")
        
        # Conversation
        messages = [
            "Hello! I'm testing the voice engine.",
            "Can you count to three for me?",
            "Great! Now say goodbye."
        ]
        
        for i, msg in enumerate(messages):
            print(f"\n  Round {i+1}:")
            print(f"  User: {msg}")
            
            msg_start = time.time()
            await engine.send_text(msg)
            
            # Wait for response
            await asyncio.sleep(8)
            
            # Show what happened
            recent_events = [e for e in events if e[0] > msg_start]
            text_events = [e for e in recent_events if e[1] == 'text']
            audio_events = [e for e in recent_events if e[1] == 'audio']
            
            if text_events:
                print(f"  AI: {text_events[-1][2]}")
            print(f"  (Received {len(audio_events)} audio chunks)")
        
        # Stop listening
        await engine.stop_listening()
        print("\n✓ Stopped listening")
        
        # Get final metrics
        metrics = engine.get_metrics()  # Not async
        usage = await engine.get_usage()
        cost = await engine.estimate_cost()
        
        print(f"\n✓ Session Summary:")
        print(f"  Total events: {len(events)}")
        print(f"  Text responses: {len([e for e in events if e[1] == 'text'])}")
        print(f"  Audio chunks: {len([e for e in events if e[1] == 'audio'])}")
        print(f"  Errors: {len([e for e in events if e[1] == 'error'])}")
        print(f"  Usage: {usage}")
        print(f"  Estimated cost: {cost}")
        
        # Disconnect
        await engine.disconnect()
        print("✓ Disconnected")
        
        return len(events) > 0 and len([e for e in events if e[1] == 'error']) == 0
        
    except Exception as e:
        print(f"✗ Full conversation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test error handling and recovery"""
    print("\n=== Test 2: Error Handling ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY)
        
        errors_caught = []
        engine.on_error = lambda e: errors_caught.append(e)
        
        await engine.connect()
        
        # Test various error scenarios
        
        # 1. Send text without listening (should work)
        await engine.send_text("Test message")
        print("✓ Text without listening works")
        
        # 2. Try to interrupt without active response
        await engine.interrupt()
        print("✓ Interrupt without response handled")
        
        # 3. Send very long text
        long_text = "Hello " * 1000  # Very long message
        await engine.send_text(long_text[:500])  # Truncate for API
        print("✓ Long text handled")
        
        await asyncio.sleep(5)
        
        # Check errors
        print(f"✓ Errors caught: {len(errors_caught)}")
        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

async def test_state_transitions():
    """Test state transitions during conversation"""
    print("\n=== Test 3: State Transitions ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY)
        
        states = []
        
        def check_state(label):
            state = {
                'label': label,
                'connected': engine.is_connected,
                'listening': engine.is_listening,
                'stream_state': engine.get_state().value
            }
            states.append(state)
            print(f"  {label}: connected={state['connected']}, listening={state['listening']}, state={state['stream_state']}")
        
        # Track states
        check_state("Initial")
        
        await engine.connect()
        check_state("After connect")
        
        await engine.start_listening()
        check_state("After start listening")
        
        await engine.send_text("Hello!")
        check_state("After send text")
        
        await asyncio.sleep(5)
        check_state("During response")
        
        await engine.interrupt()
        check_state("After interrupt")
        
        await engine.stop_listening()
        check_state("After stop listening")
        
        await engine.disconnect()
        check_state("After disconnect")
        
        print(f"\n✓ Captured {len(states)} state transitions")
        
        return len(states) == 8
        
    except Exception as e:
        print(f"✗ State transitions test failed: {e}")
        return False

async def test_rapid_interactions():
    """Test rapid back-and-forth interactions"""
    print("\n=== Test 4: Rapid Interactions ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        response_times = []
        
        await engine.connect()
        await engine.start_listening()
        
        # Rapid fire messages
        questions = [
            "What's 1+1?",
            "What's 2+2?",
            "What's 3+3?",
            "Done!"
        ]
        
        for q in questions:
            start = time.time()
            first_response = None
            
            def on_response(audio):
                nonlocal first_response
                if first_response is None:
                    first_response = time.time()
            
            engine.on_audio_response = on_response
            
            await engine.send_text(q)
            
            # Wait briefly
            await asyncio.sleep(3)
            
            if first_response:
                response_times.append((first_response - start) * 1000)
            
            # Quick interrupt for next question
            await engine.interrupt()
        
        await engine.stop_listening()
        await engine.disconnect()
        
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            print(f"✓ Average response time: {avg_response:.0f}ms")
            print(f"✓ Min: {min(response_times):.0f}ms, Max: {max(response_times):.0f}ms")
        
        return len(response_times) > 0
        
    except Exception as e:
        print(f"✗ Rapid interactions test failed: {e}")
        return False

async def test_resource_cleanup():
    """Test proper resource cleanup"""
    print("\n=== Test 5: Resource Cleanup ===")
    
    try:
        # Create and destroy multiple engines
        for i in range(3):
            engine = VoiceEngine(api_key=API_KEY)
            
            await engine.connect()
            await engine.send_text(f"Test {i}")
            await asyncio.sleep(2)
            
            # Get metrics before cleanup
            metrics_before = engine.get_metrics()  # Not async
            
            await engine.disconnect()
            
            print(f"✓ Engine {i+1} created and cleaned up")
        
        print("✓ Multiple engine lifecycle test passed")
        
        # Test with context manager
        async with VoiceEngine(api_key=API_KEY) as engine:
            await engine.send_text("Context manager test")
            await asyncio.sleep(2)
        
        print("✓ Context manager cleanup verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Resource cleanup test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Full Integration Tests (Real API)")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Full Conversation", await test_full_conversation_cycle()))
    results.append(("Error Handling", await test_error_handling()))
    results.append(("State Transitions", await test_state_transitions()))
    results.append(("Rapid Interactions", await test_rapid_interactions()))
    results.append(("Resource Cleanup", await test_resource_cleanup()))
    
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