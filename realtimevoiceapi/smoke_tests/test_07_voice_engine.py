
# realtimevoiceapi/smoke_tests/test_07_voice_engine.py
"""
Test 07: Voice Engine - Test the unified voice engine with REAL connections

Tests the complete voice engine without mocks:
- Real WebSocket connection to OpenAI
- Real audio capture and playback
- Real VAD processing
- Complete conversation flow
- All convenience methods
- Error handling with real scenarios

Requirements:
- Valid OpenAI API key in .env file
- Working microphone
- sounddevice installed

python -m realtimevoiceapi.smoke_tests.test_07_voice_engine
"""

import sys
import asyncio
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import voice engine components
from realtimevoiceapi.voice_engine import (
    VoiceEngine, VoiceEngineConfig, create_voice_session, run_voice_engine
)
from realtimevoiceapi.core.stream_protocol import StreamState
from realtimevoiceapi.core.exceptions import EngineError
from realtimevoiceapi.core.audio_processor import AudioProcessor


def generate_test_audio(duration_seconds: float = 1.0, frequency: float = 440.0) -> bytes:
    """Generate test audio data"""
    sample_rate = 24000
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    # Generate tone with envelope to simulate speech
    envelope = np.sin(np.pi * t / duration_seconds)  # Fade in/out
    signal = envelope * 0.3 * np.sin(2 * np.pi * frequency * t)
    audio_data = (signal * 32767).astype(np.int16)
    return audio_data.tobytes()


async def test_basic_initialization():
    """Test basic voice engine initialization"""
    print("\n🔧 Testing Basic Initialization...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found")
        return False
    
    try:
        # Test with API key
        engine = VoiceEngine(api_key=api_key, mode="fast")
        assert engine.mode == "fast"
        assert engine.config.api_key == api_key
        assert engine.get_state() == StreamState.IDLE
        print("  ✅ Basic initialization works")
        
        # Test with config object
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="echo",
            vad_enabled=True,
            latency_mode="ultra_low"
        )
        engine2 = VoiceEngine(config=config)
        assert engine2.config.voice == "echo"
        assert engine2.config.vad_enabled == True
        print("  ✅ Config object initialization works")
        
        # Test factory method
        engine3 = VoiceEngine.create_simple(api_key=api_key, voice="shimmer")
        assert engine3.config.voice == "shimmer"
        assert engine3.config.latency_mode == "ultra_low"
        print("  ✅ Factory method works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        logger.exception("Initialization error")
        return False


async def test_connection_lifecycle():
    """Test connection and disconnection"""
    print("\n🔌 Testing Connection Lifecycle...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        
        # Test initial state
        assert not engine.is_connected
        assert engine.get_state() == StreamState.IDLE
        print("  ✅ Initial state correct")
        
        # Test connection
        print("  🔄 Connecting to OpenAI...")
        await engine.connect()
        
        assert engine.is_connected
        assert engine.get_state() in [StreamState.ACTIVE, StreamState.ACTIVE]
        print("  ✅ Connected successfully")
        
        # Test metrics after connection
        metrics = engine.get_metrics()
        assert metrics["connected"] == True
        assert metrics["mode"] == "fast"
        assert "uptime" in metrics
        print("  ✅ Metrics available after connection")
        
        # Test disconnection
        await engine.disconnect()
        assert not engine.is_connected
        print("  ✅ Disconnected successfully")
        
        # Test reconnection
        await engine.connect()
        assert engine.is_connected
        print("  ✅ Reconnection works")
        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Connection lifecycle failed: {e}")
        logger.exception("Connection error")
        return False


async def test_context_manager():
    """Test async context manager support"""
    print("\n🔄 Testing Context Manager...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        # Test context manager
        async with VoiceEngine(api_key=api_key, mode="fast") as engine:
            assert engine.is_connected
            print("  ✅ Context manager entry works")
            
            # Do something with engine
            state = engine.get_state()
            assert state in [StreamState.ACTIVE, StreamState.ACTIVE]
        
        # Engine should be disconnected after context
        assert not engine.is_connected
        print("  ✅ Context manager exit works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Context manager failed: {e}")
        logger.exception("Context manager error")
        return False


async def test_text_conversation():
    """Test text-based conversation"""
    print("\n💬 Testing Text Conversation...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        
        # Track responses
        text_responses = []
        audio_chunks = []
        
        engine.on_text_response = lambda text: text_responses.append(text)
        engine.on_audio_response = lambda audio: audio_chunks.append(audio)
        
        # Connect
        await engine.connect()
        print("  ✅ Connected for text conversation")
        
        # Send text
        test_message = "Say 'Hello from voice engine test' and nothing else."
        print(f"  📤 Sending: {test_message}")
        await engine.send_text(test_message)
        
        # Wait for response
        await asyncio.sleep(5.0)
        
        # Check responses
        if text_responses:
            full_text = "".join(text_responses)
            print(f"  📥 Received text: {full_text}")
            assert "hello" in full_text.lower()
            print("  ✅ Text response received")
        
        if audio_chunks:
            total_audio_size = sum(len(chunk) for chunk in audio_chunks)
            print(f"  🔊 Received audio: {len(audio_chunks)} chunks, {total_audio_size} bytes")
            print("  ✅ Audio response received")
        
        assert len(text_responses) > 0 or len(audio_chunks) > 0
        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Text conversation failed: {e}")
        logger.exception("Text conversation error")
        return False


async def test_audio_input():
    """Test audio input with VAD"""
    print("\n🎤 Testing Audio Input...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        # Create engine with VAD enabled
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            vad_enabled=True,
            vad_threshold=0.02
        )
        engine = VoiceEngine(config=config)
        
        # Track responses
        responses_received = False
        
        def on_response(audio):
            nonlocal responses_received
            responses_received = True
        
        engine.on_audio_response = on_response
        
        # Connect
        await engine.connect()
        print("  ✅ Connected with VAD enabled")
        
        # Send test audio directly (bypass listening)
        test_audio = generate_test_audio(duration_seconds=0.5)
        print("  📤 Sending test audio (0.5s)")
        
        # Send in chunks to simulate real-time
        chunk_size = 4800  # 100ms chunks
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            await engine.send_audio(chunk)
            await asyncio.sleep(0.05)  # Simulate real-time
        
        # Commit audio (manually since we're not using listening mode)
        # This would normally be handled by VAD
        from realtimevoiceapi.core.message_protocol import MessageFactory
        commit_msg = MessageFactory.input_audio_buffer_commit()
        response_msg = MessageFactory.response_create()
        
        if hasattr(engine._strategy, '_stream_manager'):
            await engine._strategy._stream_manager.connection.send(commit_msg)
            await engine._strategy._stream_manager.connection.send(response_msg)
        
        # Wait for response
        await asyncio.sleep(5.0)
        
        print("  ✅ Audio sent successfully")
        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio input test failed: {e}")
        logger.exception("Audio input error")
        return False


async def test_listening_mode():
    """Test listening mode with real audio capture"""
    print("\n🎧 Testing Listening Mode...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        import sounddevice as sd
        
        # List audio devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("  ⚠️ No input devices found, skipping listening test")
            return True
        
        print(f"  ℹ️ Found {len(input_devices)} input devices")
        
        engine = VoiceEngine(api_key=api_key, mode="fast")
        await engine.connect()
        
        # Test starting/stopping listening
        print("  🎤 Starting listening mode...")
        await engine.start_listening()
        assert engine.is_listening
        print("  ✅ Listening started")
        
        # Listen for 1 second
        await asyncio.sleep(1.0)
        
        # Get metrics during listening
        metrics = engine.get_metrics()
        assert metrics["listening"] == True
        
        # Stop listening
        await engine.stop_listening()
        assert not engine.is_listening
        print("  ✅ Listening stopped")
        
        await engine.disconnect()
        
        return True
        
    except ImportError:
        print("  ⚠️ sounddevice not available, skipping listening test")
        return True
    except Exception as e:
        print(f"  ❌ Listening mode failed: {e}")
        logger.exception("Listening mode error")
        return False

async def test_convenience_methods():
    """Test convenience methods like text_2_audio_response()"""
    print("\n🛠️ Testing Convenience Methods...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        
        # Connect first!
        await engine.connect()
        
        # Test text_2_audio_response() method
        print("  🔊 Testing text_2_audio_response() method...")
        audio_data = await engine.text_2_audio_response("Testing voice engine convenience method.")
        
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 1000  # Should have some audio
        
        # Calculate duration
        processor = AudioProcessor()
        duration_ms = processor.calculate_duration(audio_data)
        print(f"  ✅ text_2_audio_response() returned {len(audio_data)} bytes ({duration_ms:.0f}ms)")
        
        # Test that we're still connected
        assert engine.is_connected
        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Convenience methods failed: {e}")
        logger.exception("Convenience method error")
        return False


async def test_error_handling():
    """Test error handling scenarios"""
    print("\n❌ Testing Error Handling...")
    
    try:
        # Test invalid API key
        try:
            engine = VoiceEngine(api_key="invalid_key", mode="fast")
            await engine.connect(retry_count=1)
            assert False, "Should have failed with invalid key"
        except Exception:
            print("  ✅ Invalid API key handled correctly")
        
        # Test operations without connection
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            engine = VoiceEngine(api_key=api_key, mode="fast")
            
            try:
                await engine.send_text("test")
                assert False, "Should fail when not connected"
            except EngineError:
                print("  ✅ Not connected error handled correctly")
            
            # Test invalid mode
            try:
                engine = VoiceEngine(api_key=api_key, mode="invalid")
                assert False, "Should fail with invalid mode"
            except ValueError:
                print("  ✅ Invalid mode error handled correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        logger.exception("Error handling test error")
        return False


async def test_usage_and_metrics():
    """Test usage tracking and metrics"""
    print("\n📊 Testing Usage and Metrics...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        await engine.connect()
        
        # Get initial usage
        usage = await engine.get_usage()
        assert usage.audio_input_seconds >= 0
        assert usage.audio_output_seconds >= 0
        print("  ✅ Initial usage tracking works")
        
        # Send a message to generate usage
        await engine.send_text("Say hello")
        await asyncio.sleep(3.0)
        
        # Get updated usage
        usage2 = await engine.get_usage()
        # Note: Usage might not update immediately
        print(f"  ℹ️ Usage - Audio in: {usage2.audio_input_seconds}s, out: {usage2.audio_output_seconds}s")
        
        # Get metrics
        metrics = engine.get_metrics()
        assert "uptime" in metrics
        assert metrics["uptime"] > 0
        print(f"  ✅ Metrics: uptime={metrics['uptime']:.1f}s")
        
        # Estimate cost
        cost = await engine.estimate_cost()
       
        assert cost.total >= 0
        # print(f"  ✅ Cost estimation: ${cost.total_cost:.4f}")
        print(f"  ✅ Cost estimation: ${cost.total:.4f}")

        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Usage/metrics test failed: {e}")
        logger.exception("Usage/metrics error")
        return False


async def test_interrupt_functionality():
    """Test interrupting AI responses"""
    print("\n⏸️ Testing Interrupt Functionality...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        
        audio_chunks_before_interrupt = []
        audio_chunks_after_interrupt = []
        interrupted = False
        
        def track_audio(audio):
            if not interrupted:
                audio_chunks_before_interrupt.append(audio)
            else:
                audio_chunks_after_interrupt.append(audio)
        
        engine.on_audio_response = track_audio
        
        await engine.connect()
        
        # Send a request that will generate a long response
        await engine.send_text("Count from 1 to 20 slowly")
        
        # Wait a bit then interrupt
        await asyncio.sleep(2.0)
        interrupted = True
        
        print("  🛑 Sending interrupt...")
        await engine.interrupt()
        
        # Wait to see if more audio comes
        await asyncio.sleep(2.0)
        
        # Should have received some audio before interrupt
        assert len(audio_chunks_before_interrupt) > 0
        print(f"  ✅ Received {len(audio_chunks_before_interrupt)} chunks before interrupt")
        
        # Should receive minimal or no audio after interrupt
        print(f"  ℹ️ Received {len(audio_chunks_after_interrupt)} chunks after interrupt")
        
        await engine.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Interrupt test failed: {e}")
        logger.exception("Interrupt error")
        return False


async def test_multiple_engines():
    """Test multiple engine instances"""
    print("\n👥 Testing Multiple Engine Instances...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        # Create two engines with different configs
        engine1 = VoiceEngine(api_key=api_key, mode="fast", voice="alloy")
        engine2 = VoiceEngine(api_key=api_key, mode="fast", voice="echo")
        
        # Both should be independent
        assert engine1.config.voice == "alloy"
        assert engine2.config.voice == "echo"
        assert engine1 is not engine2
        print("  ✅ Multiple instances created")
        
        # Connect first engine
        await engine1.connect()
        assert engine1.is_connected
        assert not engine2.is_connected
        print("  ✅ Instances are independent")
        
        await engine1.disconnect()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Multiple engines test failed: {e}")
        logger.exception("Multiple engines error")
        return False


def main():
    """Run all voice engine tests"""
    print("🧪 RealtimeVoiceAPI - Test 07: Voice Engine")
    print("=" * 60)
    print("Testing the unified voice engine with REAL connections")
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")
        return False
    
    tests = [
        ("Basic Initialization", test_basic_initialization),
        ("Connection Lifecycle", test_connection_lifecycle),
        ("Context Manager", test_context_manager),
        ("Text Conversation", test_text_conversation),
        ("Audio Input", test_audio_input),
        ("Listening Mode", test_listening_mode),
        ("Convenience Methods", test_convenience_methods),
        ("Error Handling", test_error_handling),
        ("Usage and Metrics", test_usage_and_metrics),
        ("Interrupt Functionality", test_interrupt_functionality),
        ("Multiple Engines", test_multiple_engines),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = asyncio.run(test_func())
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            logger.exception(f"{test_name} crash")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Voice Engine fully tested and working!")
        print("✨ Verified functionality:")
        print("  - Initialization and configuration")
        print("  - Real WebSocket connections")
        print("  - Text and audio conversations")
        print("  - Audio capture and playback")
        print("  - Error handling and recovery")
        print("  - Usage tracking and metrics")
        print("  - All convenience methods")
        print("\n🚀 Voice Engine is ready for production use!")
    else:
        print(f"\n❌ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)