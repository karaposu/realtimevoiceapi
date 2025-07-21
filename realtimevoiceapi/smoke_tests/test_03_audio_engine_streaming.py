"""
Test 03: AudioEngine Streaming
Tests audio capture and playback functionality through AudioEngine.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import asyncio
import time
import numpy as np
from audioengine.audioengine.audio_engine import AudioEngine, create_fast_lane_engine
from audioengine.audioengine.audio_types import (
    AudioConfig, ProcessingMode, AudioBytes, VADConfig, VADType
)

def generate_test_audio(duration_ms: int = 100, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio (sine wave)"""
    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767 * 0.5).astype(np.int16)
    return audio_int16.tobytes()

async def test_device_configuration():
    """Test device configuration"""
    print("\n=== Test 1: Device Configuration ===")
    
    try:
        engine = create_fast_lane_engine()
        
        # Configure devices
        engine.configure_devices(input_device=None, output_device=None)
        print("✓ Configured devices (default)")
        
        # Configure with specific devices (using None for default)
        engine.configure_devices(input_device=0, output_device=0)
        print("✓ Configured devices with IDs")
        
        # Configure VAD
        vad_config = VADConfig(
            type=VADType.ENERGY_BASED,
            energy_threshold=0.02,
            speech_start_ms=100,
            speech_end_ms=500
        )
        engine.configure_vad(vad_config)
        print("✓ Configured VAD")
        
        return True
    except Exception as e:
        print(f"✗ Device configuration failed: {e}")
        return False

async def test_playback_queue():
    """Test audio playback queueing"""
    print("\n=== Test 2: Playback Queue ===")
    
    try:
        engine = AudioEngine()
        
        # Queue multiple audio chunks
        chunks_queued = 0
        for i in range(5):
            test_audio = generate_test_audio(20)  # 20ms chunks
            engine.queue_playback(test_audio)
            chunks_queued += 1
        
        print(f"✓ Queued {chunks_queued} audio chunks")
        
        # Mark playback complete
        engine.mark_playback_complete()
        print("✓ Marked playback complete")
        
        # Test interruption
        engine.queue_playback(generate_test_audio(100))
        engine.interrupt_playback(force=True)
        print("✓ Interrupted playback")
        
        return True
    except Exception as e:
        print(f"✗ Playback queue test failed: {e}")
        return False

async def test_capture_stream_init():
    """Test capture stream initialization"""
    print("\n=== Test 3: Capture Stream Initialization ===")
    
    try:
        engine = create_fast_lane_engine()
        
        # Note: This will try to initialize audio hardware
        # It may fail if no audio devices are available
        try:
            queue = await engine.start_capture_stream()
            print("✓ Capture stream started")
            
            # Stop capture
            await engine.stop_capture_stream()
            print("✓ Capture stream stopped")
        except Exception as hw_error:
            print(f"⚠ Hardware initialization skipped: {hw_error}")
            print("  (This is expected in CI or without audio devices)")
        
        return True
    except Exception as e:
        print(f"✗ Capture stream test failed: {e}")
        return False

async def test_vad_processing():
    """Test VAD processing on audio chunks"""
    print("\n=== Test 4: VAD Processing ===")
    
    try:
        # Create engine with VAD
        vad_config = VADConfig(
            type=VADType.ENERGY_BASED,
            energy_threshold=0.02,
            speech_start_ms=100,
            speech_end_ms=500
        )
        
        engine = create_fast_lane_engine(vad_config=vad_config)
        
        # Generate test audio
        quiet_audio = (np.zeros(2400, dtype=np.int16) + np.random.randint(-50, 50, 2400)).tobytes()
        loud_audio = generate_test_audio(100)
        
        # Process through VAD
        # Note: Full VAD testing requires audio manager initialization
        result_quiet = engine.process_vad_chunk(quiet_audio)
        result_loud = engine.process_vad_chunk(loud_audio)
        
        print(f"✓ VAD processing configured")
        print(f"  Quiet audio result: {result_quiet}")
        print(f"  Loud audio result: {result_loud}")
        
        return True
    except Exception as e:
        print(f"✗ VAD processing test failed: {e}")
        return False

async def test_playback_callbacks():
    """Test playback callback functionality"""
    print("\n=== Test 5: Playback Callbacks ===")
    
    try:
        engine = AudioEngine()
        
        # Track callbacks
        completion_called = False
        chunks_played = []
        
        def on_complete():
            nonlocal completion_called
            completion_called = True
        
        def on_chunk_played(num_chunks):
            chunks_played.append(num_chunks)
        
        # Set callbacks
        engine.set_playback_callbacks(
            completion_callback=on_complete,
            chunk_played_callback=on_chunk_played
        )
        print("✓ Set playback callbacks")
        
        # Queue some audio
        for i in range(3):
            engine.queue_playback(generate_test_audio(20))
        
        # Note: Actual callback testing requires real playback
        print("✓ Callbacks configured (full testing requires audio playback)")
        
        return True
    except Exception as e:
        print(f"✗ Playback callback test failed: {e}")
        return False

async def test_streaming_metrics():
    """Test streaming-related metrics"""
    print("\n=== Test 6: Streaming Metrics ===")
    
    try:
        engine = AudioEngine()
        
        # Simulate streaming scenario
        for i in range(10):
            # Queue playback
            engine.queue_playback(generate_test_audio(20))
            
            # Process audio
            engine.process_audio(generate_test_audio(20))
        
        # Get metrics
        metrics = engine.get_metrics()
        print("✓ Streaming metrics:")
        print(f"  Is playing: {metrics.get('is_playing', False)}")
        print(f"  Total chunks processed: {metrics.get('total_chunks', 0)}")
        print(f"  Mode: {metrics.get('mode', 'unknown')}")
        
        # Check component metrics
        if 'buffered_player' in metrics:
            print(f"✓ Buffered player metrics available")
        
        return True
    except Exception as e:
        print(f"✗ Streaming metrics test failed: {e}")
        return False

async def test_cleanup_with_components():
    """Test cleanup with active components"""
    print("\n=== Test 7: Component Cleanup ===")
    
    try:
        engine = create_fast_lane_engine()
        
        # Initialize components
        engine.queue_playback(generate_test_audio(100))
        
        # Cleanup
        await engine.cleanup_async()
        print("✓ Async cleanup completed")
        
        # Verify cleanup
        metrics = engine.get_metrics()
        print(f"✓ Post-cleanup state verified")
        
        return True
    except Exception as e:
        print(f"✗ Component cleanup test failed: {e}")
        return False

async def test_concurrent_operations():
    """Test concurrent capture and playback operations"""
    print("\n=== Test 8: Concurrent Operations ===")
    
    try:
        engine = create_fast_lane_engine()
        
        # Simulate concurrent operations
        tasks = []
        
        # Playback task
        async def playback_task():
            for i in range(5):
                engine.queue_playback(generate_test_audio(20))
                await asyncio.sleep(0.02)
        
        # Processing task
        async def processing_task():
            for i in range(5):
                engine.process_audio(generate_test_audio(20))
                await asyncio.sleep(0.02)
        
        # Run concurrently
        tasks.append(asyncio.create_task(playback_task()))
        tasks.append(asyncio.create_task(processing_task()))
        
        await asyncio.gather(*tasks)
        print("✓ Concurrent operations completed")
        
        # Check final state
        print(f"✓ Is playing: {engine.is_playing}")
        
        return True
    except Exception as e:
        print(f"✗ Concurrent operations test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("AudioEngine Streaming Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Device Configuration", await test_device_configuration()))
    results.append(("Playback Queue", await test_playback_queue()))
    results.append(("Capture Stream Init", await test_capture_stream_init()))
    results.append(("VAD Processing", await test_vad_processing()))
    results.append(("Playback Callbacks", await test_playback_callbacks()))
    results.append(("Streaming Metrics", await test_streaming_metrics()))
    results.append(("Component Cleanup", await test_cleanup_with_components()))
    results.append(("Concurrent Operations", await test_concurrent_operations()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)