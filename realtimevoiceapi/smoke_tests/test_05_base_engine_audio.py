"""
Test 05: BaseEngine Audio Integration
Tests BaseEngine's audio functionality through AudioEngine.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import logging
import numpy as np
from realtimevoiceapi.base_engine import BaseEngine
from realtimevoiceapi.strategies.base_strategy import EngineConfig
from realtimevoiceapi.core.stream_protocol import StreamEvent, StreamEventType
from audioengine.audioengine.audio_types import AudioBytes

# Set up logging
logging.basicConfig(level=logging.INFO)

def generate_test_audio(duration_ms: int = 100, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio (sine wave)"""
    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767 * 0.5).astype(np.int16)
    return audio_int16.tobytes()

async def test_audio_processing_loop():
    """Test audio processing loop setup"""
    print("\n=== Test 1: Audio Processing Loop ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Initialize strategy
        config = EngineConfig(api_key="test-key", provider="openai")
        await engine.initialize_strategy(config)
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Start audio processing (without actual connection)
        # This will fail to get audio but tests the setup
        print("✓ Audio processing setup completed")
        
        # Check state
        print(f"✓ Audio engine exists: {engine.components.audio_engine is not None}")
        print(f"✓ Is listening: {engine.is_listening}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_playback():
    """Test audio playback through AudioEngine"""
    print("\n=== Test 2: Audio Playback ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Initialize strategy first
        config = EngineConfig(api_key="test-key", provider="openai")
        await engine.initialize_strategy(config)
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Test direct playback
        test_audio = generate_test_audio(100)
        engine.play_audio(test_audio)
        print("✓ Direct audio playback called")
        
        # Simulate audio event for playback
        audio_event = StreamEvent(
            type=StreamEventType.AUDIO_OUTPUT_CHUNK,
            stream_id="test-stream",
            timestamp=0,
            data={"audio": test_audio}
        )
        
        # Setup handlers to test audio routing
        audio_chunks_received = []
        
        def audio_handler(event):
            audio_chunks_received.append(event.data.get("audio"))
        
        engine.setup_event_handlers({
            StreamEventType.AUDIO_OUTPUT_CHUNK: audio_handler
        })
        
        # Trigger the wrapped handler
        wrapped_handler = engine.event_registry.get_handler(StreamEventType.AUDIO_OUTPUT_CHUNK)
        if wrapped_handler:
            wrapped_handler(audio_event)
            print("✓ Audio event processed through handler")
        
        # Check metrics
        print(f"✓ Audio chunks received by handler: {len(audio_chunks_received)}")
        print(f"✓ Response audio started: {engine.state.response_audio_started}")
        
        return True
    except Exception as e:
        print(f"✗ Audio playback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_interruption():
    """Test audio interruption"""
    print("\n=== Test 3: Audio Interruption ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Queue some audio
        for i in range(5):
            test_audio = generate_test_audio(20)
            if engine.components.audio_engine:
                engine.components.audio_engine.queue_playback(test_audio)
        
        print("✓ Queued audio for playback")
        
        # Test interruption (without actual strategy connection)
        try:
            await engine.interrupt()
        except Exception as e:
            # Expected since strategy isn't connected
            print(f"✓ Interruption attempted (expected error: {e})")
        
        # Direct interruption through audio engine
        if engine.components.audio_engine:
            engine.components.audio_engine.interrupt_playback(force=True)
            print("✓ Direct audio interruption completed")
        
        return True
    except Exception as e:
        print(f"✗ Audio interruption test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vad_integration():
    """Test VAD integration"""
    print("\n=== Test 4: VAD Integration ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio with VAD
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Test VAD processing
        quiet_audio = (np.zeros(2400, dtype=np.int16) + np.random.randint(-50, 50, 2400)).tobytes()
        loud_audio = generate_test_audio(100)
        
        if engine.components.audio_engine:
            vad_quiet = engine.components.audio_engine.process_vad_chunk(quiet_audio)
            vad_loud = engine.components.audio_engine.process_vad_chunk(loud_audio)
            
            print(f"✓ VAD processing:")
            print(f"  Quiet audio: {vad_quiet}")
            print(f"  Loud audio: {vad_loud}")
        
        return True
    except Exception as e:
        print(f"✗ VAD integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_metrics():
    """Test audio metrics through BaseEngine"""
    print("\n=== Test 5: Audio Metrics ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Process some audio to generate metrics
        if engine.components.audio_engine:
            for i in range(10):
                test_audio = generate_test_audio(20)
                engine.components.audio_engine.process_audio(test_audio)
                engine.components.audio_engine.queue_playback(test_audio)
        
        # Get metrics
        metrics = engine.get_metrics()
        print("✓ Retrieved metrics through BaseEngine:")
        
        if 'audio_engine' in metrics:
            ae_metrics = metrics['audio_engine']
            print(f"  Total chunks: {ae_metrics.get('total_chunks', 0)}")
            print(f"  Avg latency: {ae_metrics.get('avg_latency_ms', 0):.2f}ms")
            print(f"  Is playing: {ae_metrics.get('is_playing', False)}")
            print(f"  Mode: {ae_metrics.get('mode', 'unknown')}")
            
            # Check component metrics
            if 'audio_manager' in ae_metrics:
                print(f"✓ Audio manager metrics available")
            if 'buffered_player' in ae_metrics:
                print(f"✓ Buffered player metrics available")
        
        return True
    except Exception as e:
        print(f"✗ Audio metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_callbacks():
    """Test audio callbacks"""
    print("\n=== Test 6: Audio Callbacks ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Track callback state
        playback_complete_called = False
        chunks_played = []
        
        # Override callback methods
        original_complete = engine._on_audio_playback_complete
        original_chunks = engine._on_chunks_played
        
        def test_complete():
            nonlocal playback_complete_called
            playback_complete_called = True
            original_complete()
        
        def test_chunks(num):
            chunks_played.append(num)
            original_chunks(num)
        
        engine._on_audio_playback_complete = test_complete
        engine._on_chunks_played = test_chunks
        
        print("✓ Audio callbacks configured")
        
        # Note: Actual callback testing requires real audio playback
        print("✓ Callbacks ready (full testing requires audio hardware)")
        
        return True
    except Exception as e:
        print(f"✗ Audio callbacks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_stream_ended_handling():
    """Test stream ended event handling"""
    print("\n=== Test 7: Stream Ended Handling ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Track stream ended
        stream_ended_called = False
        
        def stream_ended_handler(event):
            nonlocal stream_ended_called
            stream_ended_called = True
            print(f"  Stream ended event received: {event.type}")
        
        # Setup handlers
        engine.setup_event_handlers({
            StreamEventType.STREAM_ENDED: stream_ended_handler
        })
        
        # Simulate stream ended
        engine.state.response_audio_started = True
        ended_event = StreamEvent(
            type=StreamEventType.STREAM_ENDED,
            stream_id="test-stream",
            timestamp=0,
            data={}
        )
        
        # Trigger wrapped handler
        wrapped_handler = engine.event_registry.get_handler(StreamEventType.STREAM_ENDED)
        if wrapped_handler:
            wrapped_handler(ended_event)
            print("✓ Stream ended event processed")
        
        print(f"✓ Audio marked complete: {engine.state.response_audio_complete}")
        
        return True
    except Exception as e:
        print(f"✗ Stream ended handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_cleanup():
    """Test audio cleanup through BaseEngine"""
    print("\n=== Test 8: Audio Cleanup ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Add some data
        if engine.components.audio_engine:
            for i in range(5):
                engine.components.audio_engine.process_audio(generate_test_audio(20))
                engine.components.audio_engine.queue_playback(generate_test_audio(20))
        
        # Cleanup
        await engine.cleanup()
        print("✓ Cleanup completed")
        
        # Verify cleanup
        print(f"✓ Audio engine cleaned up: {engine.components.audio_engine is None}")
        print(f"✓ State reset: listening={engine.state.is_listening}, connected={engine.state.is_connected}")
        
        return True
    except Exception as e:
        print(f"✗ Audio cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("BaseEngine Audio Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Audio Processing Loop", await test_audio_processing_loop()))
    results.append(("Audio Playback", await test_audio_playback()))
    results.append(("Audio Interruption", await test_audio_interruption()))
    results.append(("VAD Integration", await test_vad_integration()))
    results.append(("Audio Metrics", await test_audio_metrics()))
    results.append(("Audio Callbacks", await test_audio_callbacks()))
    results.append(("Stream Ended", await test_stream_ended_handling()))
    results.append(("Audio Cleanup", await test_audio_cleanup()))
    
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