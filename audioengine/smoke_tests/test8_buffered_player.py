"""
Smoke tests for BufferedAudioPlayer - streaming audio playback.

Tests buffered playback, completion tracking, and metrics collection.

python -m smoke_tests.test8_buffered_player
"""

import sys
import time
import threading
import traceback
import math
import struct
from typing import List, Optional, Dict, Any, Tuple

# Import audio components
try:
    from audioengine.buffered_audio_player import BufferedAudioPlayer
    from audioengine.audio_types import AudioConfig, AudioBytes
    from audioengine.exceptions import AudioError
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure audioengine is in PYTHONPATH")
    sys.exit(1)


def print_test_header(test_name: str):
    """Print a test section header"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


def print_result(test_desc: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_desc}")
    if details:
        print(f"     | {details}")


def generate_test_audio(duration_ms: int, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio (sine wave)"""
    samples = int(sample_rate * duration_ms / 1000)
    frequency = 440  # A4 note
    
    audio_data = bytearray()
    for i in range(samples):
        # Generate sine wave
        t = i / sample_rate
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
        # Pack as 16-bit little-endian
        audio_data.extend(struct.pack('<h', value))
    
    return bytes(audio_data)


def test_basic_playback() -> Tuple[int, int]:
    """Test basic buffer accumulation and playback"""
    print_test_header("Basic Buffered Playback")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create player
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000, channels=1)
        player = BufferedAudioPlayer(config)
        
        passed = (not player.is_playing and 
                 player.chunks_received == 0 and
                 len(player.buffer) == 0)
        print_result("Player creation", passed, 
                    "Initial state correct")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Player creation", False, str(e))
    
    # Test 2: Buffer accumulation before playback
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        # Add first chunk - shouldn't start playing yet (min_buffer_chunks=2)
        chunk1 = generate_test_audio(20)
        player.play(chunk1)
        
        time.sleep(0.1)  # Give time for any playback to start
        
        passed = (player.chunks_received == 1 and
                 len(player.buffer) == 1 and
                 not player.is_playing)
        
        print_result("Buffer accumulation", passed,
                    "Waits for minimum chunks before playing")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Buffer accumulation", False, str(e))
    
    # Test 3: Playback starts after threshold
    tests_total += 1
    try:
        # Add second chunk - should start playing
        chunk2 = generate_test_audio(20)
        player.play(chunk2)
        
        time.sleep(0.1)  # Give time for playback to start
        
        passed = (player.chunks_received == 2 and
                 player.is_playing)
        
        print_result("Playback triggering", passed,
                    "Starts after 2 chunks")
        if passed:
            tests_passed += 1
            
        # Cleanup
        player.stop()
    except Exception as e:
        print_result("Playback triggering", False, str(e))
    
    # Test 4: Complete playback cycle
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        # Add several chunks
        for i in range(3):
            chunk = generate_test_audio(20)
            player.play(chunk)
        
        # Mark complete
        player.mark_complete()
        
        # Wait for playback to finish
        start_time = time.time()
        while player.is_playing and time.time() - start_time < 2.0:
            time.sleep(0.05)
        
        passed = (not player.is_playing and
                 player.chunks_played == 3 and
                 player.is_complete)
        
        print_result("Complete playback cycle", passed,
                    f"Played {player.chunks_played} chunks")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Complete playback cycle", False, str(e))
    
    return tests_passed, tests_total


def test_completion_callback() -> Tuple[int, int]:
    """Test playback completion detection"""
    print_test_header("Completion Callback")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic completion callback
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = BufferedAudioPlayer(config)
        
        # Track completion
        completion_called = threading.Event()
        
        def on_complete():
            completion_called.set()
        
        player.set_completion_callback(on_complete)
        
        # Play audio
        for i in range(3):
            player.play(generate_test_audio(20))
        player.mark_complete()
        
        # Wait for completion
        passed = completion_called.wait(timeout=2.0)
        
        print_result("Completion callback", passed,
                    "Callback triggered on completion")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Completion callback", False, str(e))
    
    # Test 2: Chunk played callback
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        chunks_played_count = 0
        
        def on_chunk_played(num_chunks):
            nonlocal chunks_played_count
            chunks_played_count += num_chunks
        
        player.set_chunk_played_callback(on_chunk_played)
        
        # Play chunks
        for i in range(5):
            player.play(generate_test_audio(20))
        player.mark_complete()
        
        # Wait for playback
        start_time = time.time()
        while player.is_playing and time.time() - start_time < 2.0:
            time.sleep(0.05)
        
        passed = chunks_played_count == 5
        print_result("Chunk played callback", passed,
                    f"Reported {chunks_played_count} chunks played")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Chunk played callback", False, str(e))
    
    return tests_passed, tests_total


def test_stop_control() -> Tuple[int, int]:
    """Test stop functionality"""
    print_test_header("Stop Control")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Normal stop
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = BufferedAudioPlayer(config)
        
        # Start playback
        for i in range(5):
            player.play(generate_test_audio(20))
        
        time.sleep(0.1)  # Let playback start
        player.stop()
        
        passed = (not player.is_playing and
                 len(player.buffer) == 0)
        
        print_result("Normal stop", passed,
                    "Playback stopped and buffer cleared")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Normal stop", False, str(e))
    
    # Test 2: Force stop
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        # Start playback
        for i in range(10):
            player.play(generate_test_audio(50))
        
        time.sleep(0.1)
        player.stop(force=True)
        
        passed = not player.is_playing
        print_result("Force stop", passed,
                    "Force stop successful")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Force stop", False, str(e))
    
    # Test 3: Stop without playback
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        player.stop()  # Stop without playing
        
        passed = True
        print_result("Stop without playback", passed,
                    "No error on empty stop")
        tests_passed += 1
    except Exception as e:
        print_result("Stop without playback", False, str(e))
    
    return tests_passed, tests_total


def test_batching_behavior() -> Tuple[int, int]:
    """Test chunk batching for efficiency"""
    print_test_header("Batching Behavior")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Batch playback
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = BufferedAudioPlayer(config)
        
        chunks_played_events = []
        
        def on_chunk_played(num_chunks):
            chunks_played_events.append(num_chunks)
        
        player.set_chunk_played_callback(on_chunk_played)
        
        # Add 10 chunks rapidly
        for i in range(10):
            player.play(generate_test_audio(20))
        
        player.mark_complete()
        
        # Wait for completion
        start_time = time.time()
        while player.is_playing and time.time() - start_time < 3.0:
            time.sleep(0.05)
        
        # Verify batching occurred
        total_chunks = sum(chunks_played_events)
        max_batch = max(chunks_played_events) if chunks_played_events else 0
        
        passed = (total_chunks == 10 and max_batch > 1)
        print_result("Batch playback", passed,
                    f"Max batch size: {max_batch}, Events: {len(chunks_played_events)}")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Batch playback", False, str(e))
    
    return tests_passed, tests_total


def test_metrics() -> Tuple[int, int]:
    """Test metrics collection"""
    print_test_header("Metrics and Statistics")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic metrics
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = BufferedAudioPlayer(config)
        
        # Initial metrics
        metrics = player.get_metrics()
        
        passed = (metrics["chunks_received"] == 0 and
                 metrics["chunks_played"] == 0 and
                 not metrics["is_playing"])
        
        print_result("Initial metrics", passed,
                    "All counters start at zero")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Initial metrics", False, str(e))
    
    # Test 2: Playback metrics
    tests_total += 1
    try:
        # Play some audio
        for i in range(5):
            player.play(generate_test_audio(20))
        
        player.mark_complete()
        
        # Wait for completion
        start_time = time.time()
        while player.is_playing and time.time() - start_time < 2.0:
            time.sleep(0.05)
        
        # Final metrics
        metrics = player.get_metrics()
        
        passed = (metrics["chunks_received"] == 5 and
                 metrics["chunks_played"] == 5 and
                 metrics["is_complete"])
        
        print_result("Playback metrics", passed,
                    f"Received: {metrics['chunks_received']}, "
                    f"Played: {metrics['chunks_played']}")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Playback metrics", False, str(e))
    
    # Test 3: Latency tracking
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        # Play with timing
        player.play(generate_test_audio(20))
        player.play(generate_test_audio(20))
        
        # Wait for playback to start
        time.sleep(0.2)
        
        metrics = player.get_metrics()
        
        if "initial_latency_ms" in metrics and metrics["initial_latency_ms"] is not None:
            latency = metrics["initial_latency_ms"]
            passed = 0 < latency < 500
            details = f"Initial latency: {latency:.1f}ms"
        else:
            passed = True  # Latency tracking is optional
            details = "Latency tracking not available"
        
        print_result("Latency tracking", passed, details)
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Latency tracking", False, str(e))
    
    return tests_passed, tests_total


def test_edge_cases() -> Tuple[int, int]:
    """Test edge cases and error conditions"""
    print_test_header("Edge Cases")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Empty audio chunk
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = BufferedAudioPlayer(config)
        
        player.play(b"")  # Empty chunk
        passed = True
        print_result("Empty chunk handling", passed,
                    "Handles empty chunk gracefully")
        tests_passed += 1
        
        player.stop()
    except Exception as e:
        print_result("Empty chunk handling", False, str(e))
    
    # Test 2: Very large chunk
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        large_chunk = generate_test_audio(5000)  # 5 second chunk
        player.play(large_chunk)
        
        passed = player.chunks_received == 1
        print_result("Large chunk handling", passed,
                    f"5s chunk handled ({len(large_chunk)} bytes)")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Large chunk handling", False, str(e))
    
    # Test 3: Rapid start/stop
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        for i in range(3):
            player.play(generate_test_audio(20))
            player.play(generate_test_audio(20))
            time.sleep(0.05)
            player.stop()
            time.sleep(0.05)
        
        passed = True
        print_result("Rapid start/stop", passed,
                    "Handled 3 rapid cycles")
        tests_passed += 1
    except Exception as e:
        print_result("Rapid start/stop", False, str(e))
    
    # Test 4: Play after complete
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        player.mark_complete()
        
        player.play(generate_test_audio(20))
        passed = player.chunks_received == 1
        
        print_result("Play after complete", passed,
                    "Can add chunks after marking complete")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Play after complete", False, str(e))
    
    return tests_passed, tests_total


def test_buffer_properties() -> Tuple[int, int]:
    """Test buffer duration and state properties"""
    print_test_header("Buffer Properties")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Buffer duration calculation
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = BufferedAudioPlayer(config)
        
        # Add 20ms chunk
        chunk_20ms = generate_test_audio(20, 24000)
        player.play(chunk_20ms)
        
        duration = player.buffer_duration_ms
        passed = 15 < duration < 25
        
        print_result("Buffer duration", passed,
                    f"20ms chunk = {duration:.1f}ms buffer")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Buffer duration", False, str(e))
    
    # Test 2: Active playing state
    tests_total += 1
    try:
        player = BufferedAudioPlayer(config)
        
        # Not playing initially
        passed1 = not player.is_actively_playing
        
        # Add chunks and start playing
        for i in range(3):
            player.play(generate_test_audio(20))
        
        time.sleep(0.1)  # Let playback start
        passed2 = player.is_actively_playing
        
        player.mark_complete()
        while player.is_playing:
            time.sleep(0.05)
        
        # Not playing after completion
        passed3 = not player.is_actively_playing
        
        passed = passed1 and passed2 and passed3
        print_result("Active playing state", passed,
                    "State tracking correct throughout lifecycle")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Active playing state", False, str(e))
    
    return tests_passed, tests_total


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("BUFFERED AUDIO PLAYER SMOKE TESTS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_basic_playback,
        test_completion_callback,
        test_stop_control,
        test_batching_behavior,
        test_metrics,
        test_edge_cases,
        test_buffer_properties
    ]
    
    for test_func in test_functions:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n❌ FATAL ERROR in {test_func.__name__}:")
            traceback.print_exc()
            total_tests += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())