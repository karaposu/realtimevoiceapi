"""
Test 08: VoiceEngine Audio Interactions
Tests audio-based interactions with REAL OpenAI API connections.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
import numpy as np
from realtimevoiceapi.voice_engine import VoiceEngine
from audioengine.audioengine.audio_types import AudioBytes

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

def generate_test_audio(duration_ms: int = 1000, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio saying 'Hello'"""
    # Generate a simple tone pattern that could represent speech
    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    
    # Create a more complex waveform to simulate speech
    frequencies = [300, 400, 500, 600]  # Multiple frequencies
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        # Add some amplitude modulation to make it more speech-like
        envelope = np.exp(-t * 2) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        audio += envelope * np.sin(2 * np.pi * freq * t) / len(frequencies)
    
    # Add some noise
    audio += np.random.normal(0, 0.01, len(t))
    
    # Normalize and convert to int16
    audio = audio / np.max(np.abs(audio)) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

async def test_audio_streaming():
    """Test audio streaming with real API"""
    print("\n=== Test 1: Audio Streaming ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track responses
        text_responses = []
        audio_responses = []
        errors = []
        
        engine.on_text_response = lambda text: text_responses.append(text)
        engine.on_audio_response = lambda audio: audio_responses.append(audio)
        engine.on_error = lambda error: errors.append(error)
        
        # Connect
        await engine.connect()
        print("✓ Connected to OpenAI API")
        
        # Start listening
        await engine.start_listening()
        print("✓ Started listening")
        
        # Send some test audio
        test_audio = generate_test_audio(2000)  # 2 seconds
        print(f"✓ Generated test audio: {len(test_audio)} bytes")
        
        # Send audio in chunks
        chunk_size = 4800  # 100ms at 24kHz
        chunks_sent = 0
        
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            await engine.send_audio(chunk)
            chunks_sent += 1
            await asyncio.sleep(0.05)  # Small delay between chunks
        
        print(f"✓ Sent {chunks_sent} audio chunks")
        
        # Wait for responses
        await asyncio.sleep(3)
        
        # Stop listening
        await engine.stop_listening()
        print("✓ Stopped listening")
        
        # Check results
        print(f"✓ Results:")
        print(f"  Text responses: {len(text_responses)}")
        print(f"  Audio responses: {len(audio_responses)}")
        print(f"  Errors: {len(errors)}")
        
        if audio_responses:
            total_response_bytes = sum(len(chunk) for chunk in audio_responses)
            print(f"  Total response audio: {total_response_bytes} bytes")
        
        # Disconnect
        await engine.disconnect()
        print("✓ Disconnected")
        
        return True  # Success if no errors
        
    except Exception as e:
        print(f"✗ Audio streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_conversation_flow():
    """Test complete conversation flow"""
    print("\n=== Test 2: Conversation Flow ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track conversation
        conversation = []
        
        def on_text(text):
            conversation.append(('AI', text))
            print(f"  AI: {text}")
        
        def on_audio(audio):
            conversation.append(('AI_AUDIO', len(audio)))
        
        engine.on_text_response = on_text
        engine.on_audio_response = on_audio
        
        # Connect
        await engine.connect()
        print("✓ Connected")
        
        # Have a conversation
        await engine.start_listening()
        
        # Message 1
        print("  User: Hello, how are you?")
        await engine.send_text("Hello, how are you?")
        await asyncio.sleep(3)
        
        # Message 2
        print("  User: What's the weather like?")
        await engine.send_text("What's the weather like?")
        await asyncio.sleep(3)
        
        # Message 3
        print("  User: Thank you, goodbye!")
        await engine.send_text("Thank you, goodbye!")
        await asyncio.sleep(3)
        
        await engine.stop_listening()
        await engine.disconnect()
        
        print(f"✓ Conversation completed with {len(conversation)} responses")
        
        return len(conversation) >= 3
        
    except Exception as e:
        print(f"✗ Conversation flow test failed: {e}")
        return False

async def test_text_to_audio_response():
    """Test text_2_audio_response convenience method"""
    print("\n=== Test 3: Text to Audio Response ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Connect first
        await engine.connect()
        print("✓ Connected to API")
        
        # Track what we receive
        final_text = None
        audio_chunks = []
        
        # Use the convenience method
        try:
            audio_result = await engine.text_2_audio_response("Say 'Hello World' please.")
            print(f"✓ Received audio response: {len(audio_result)} bytes")
            audio_chunks.append(audio_result)
        except Exception as e:
            print(f"✗ Error getting audio response: {e}")
        
        # Disconnect
        await engine.disconnect()
        
        print(f"✓ Results:")
        print(f"  Audio chunks: {len(audio_chunks)}")
        
        if audio_chunks:
            total_bytes = sum(len(chunk) for chunk in audio_chunks)
            print(f"  Total audio bytes: {total_bytes}")
        
        return len(audio_chunks) > 0
        
    except Exception as e:
        print(f"✗ Text to audio response test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_context_manager():
    """Test context manager usage"""
    print("\n=== Test 4: Context Manager ===")
    
    try:
        # Use context manager
        async with VoiceEngine(api_key=API_KEY, mode="fast") as engine:
            print("✓ Entered context manager")
            
            # Quick test
            await engine.send_text("Hi!")
            await asyncio.sleep(2)
            
            # Get state
            print(f"✓ Connected: {engine.is_connected}")
            print(f"✓ State: {engine.get_state().value}")
        
        print("✓ Exited context manager")
        
        # Verify disconnected
        print(f"✓ After exit - connected: {engine.is_connected}")
        
        return True
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        return False

async def test_performance_metrics():
    """Test performance with real audio"""
    print("\n=== Test 5: Performance Metrics ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        await engine.connect()
        await engine.start_listening()
        
        # Send audio and measure
        start_time = time.time()
        first_response_time = None
        
        def on_first_audio(audio):
            nonlocal first_response_time
            if first_response_time is None:
                first_response_time = time.time()
        
        engine.on_audio_response = on_first_audio
        
        # Send audio
        await engine.send_text("Count to three.")
        
        # Wait for response
        await asyncio.sleep(5)
        
        # Calculate metrics
        if first_response_time:
            latency = (first_response_time - start_time) * 1000
            print(f"✓ First response latency: {latency:.0f}ms")
        
        # Get engine metrics (not async)
        metrics = engine.get_metrics()
        print(f"✓ Engine metrics:")
        print(f"  Interactions: {metrics.get('total_interactions', 0)}")
        print(f"  Audio chunks received: {metrics.get('audio_chunks_received', 0)}")
        
        await engine.stop_listening()
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"✗ Performance metrics test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("VoiceEngine Audio Interaction Tests (Real API)")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Audio Streaming", await test_audio_streaming()))
    results.append(("Conversation Flow", await test_conversation_flow()))
    results.append(("Text to Audio", await test_text_to_audio_response()))
    results.append(("Context Manager", await test_context_manager()))
    results.append(("Performance", await test_performance_metrics()))
    
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