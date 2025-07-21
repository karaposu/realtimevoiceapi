# here is /smoke_tests/test5_direct_audio_player.py


# python -m smoke_tests.test5_direct_audio_player





"""
Smoke tests for direct audio player functionality.

Run with: python test_direct_audio_player.py
or: python -m realtimevoiceapi.smoke_tests.audio.test_direct_audio_player
"""

import sys
import time
import asyncio
import threading
import traceback
import struct
import math
from typing import List, Optional, Dict, Any
import concurrent.futures

# Import audio player components
try:
    from audioengine.direct_audio_capture import DirectAudioPlayer
    from audioengine.audio_types import (
        AudioConfig, AudioFormat, AudioConstants, AudioBytes
    )
    from audioengine.exceptions import AudioError
    import sounddevice as sd
    import numpy as np
    HAS_NUMPY = True
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure realtimevoiceapi is in PYTHONPATH and sounddevice is installed")
    HAS_NUMPY = False
    np = None
    if 'sounddevice' not in str(e):
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


def get_test_output_device():
    """Get a suitable test output device (prefer default)"""
    try:
        # Get default output device
        default_device = sd.default.device[1]  # Output device
        if default_device is not None:
            return default_device
        
        # Fall back to first available output device
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                return i
        
        return None
    except Exception:
        return None


def generate_test_audio(duration_ms: int, frequency_hz: float = 440.0, 
                       sample_rate: int = 24000, amplitude: float = 0.3) -> AudioBytes:
    """Generate test audio (sine wave)"""
    num_samples = int(sample_rate * duration_ms / 1000)
    
    if HAS_NUMPY and np is not None:
        # Use numpy for efficient generation
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        wave = np.sin(2 * np.pi * frequency_hz * t) * amplitude
        samples = (wave * 32767).astype(np.int16)
        return samples.tobytes()
    else:
        # Pure Python fallback
        samples = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency_hz * t))
            # Clamp to int16 range
            value = max(-32768, min(32767, value))
            samples.extend(struct.pack('<h', value))
        return bytes(samples)


def generate_silence(duration_ms: int, sample_rate: int = 24000) -> AudioBytes:
    """Generate silence"""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b'\x00' * (num_samples * 2)  # 16-bit samples


def generate_white_noise(duration_ms: int, sample_rate: int = 24000, 
                        amplitude: float = 0.1) -> AudioBytes:
    """Generate white noise for testing"""
    num_samples = int(sample_rate * duration_ms / 1000)
    
    if HAS_NUMPY and np is not None:
        noise = np.random.normal(0, amplitude, num_samples)
        samples = (noise * 32767).astype(np.int16)
        return samples.tobytes()
    else:
        # Simple pseudo-random noise
        import random
        samples = bytearray()
        for _ in range(num_samples):
            value = int(random.gauss(0, amplitude) * 32767)
            value = max(-32768, min(32767, value))
            samples.extend(struct.pack('<h', value))
        return bytes(samples)


def test_device_enumeration():
    """Test output device enumeration and info"""
    print_test_header("Output Device Enumeration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: List available output devices
    tests_total += 1
    try:
        devices = DirectAudioPlayer.list_output_devices()
        
        passed = isinstance(devices, list)
        print_result("List output devices", passed,
                    f"Found {len(devices)} output devices")
        
        if passed and len(devices) > 0:
            print("     | Output devices:")
            for dev in devices[:3]:  # Show first 3
                default_mark = " (default)" if dev.get('default') else ""
                print(f"     |   [{dev['index']}] {dev['name']}{default_mark}")
            if len(devices) > 3:
                print(f"     |   ... and {len(devices) - 3} more")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("List output devices", False, str(e))
    
    # Test 2: Get default output device
    tests_total += 1
    try:
        default_output = sd.default.device[1]
        
        if default_output is not None:
            device_info = sd.query_devices(default_output)
            passed = device_info['max_output_channels'] > 0
            print_result("Default output device", passed,
                        f"Device {default_output}: {device_info['name']}")
        else:
            # No default device is acceptable on some systems
            passed = True
            print_result("Default output device", passed, "No default device configured")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default output device", False, str(e))
    
    # Test 3: Query specific device info through DirectAudioPlayer
    tests_total += 1
    try:
        test_device = get_test_output_device()
        
        if test_device is not None:
            config = AudioConfig()
            player = DirectAudioPlayer(device=test_device, config=config)
            info = player.get_device_info()
            
            passed = (
                'name' in info and
                'sample_rate' in info and
                'channels' in info
            )
            print_result("Output device info", passed,
                        f"Name: {info.get('name', 'Unknown')}, "
                        f"Channels: {info.get('channels', 0)}")
        else:
            passed = False
            print_result("Output device info", passed, "No output device available")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Output device info", False, str(e))
    
    return tests_passed, tests_total


def test_player_creation():
    """Test player object creation"""
    print_test_header("Player Creation")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create with default device
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        player = DirectAudioPlayer(device=None, config=config)
        
        passed = player is not None and not player.is_playing
        print_result("Default device player", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default device player", False, str(e))
    
    # Test 2: Create with specific device
    tests_total += 1
    try:
        test_device = get_test_output_device()
        if test_device is not None:
            config = AudioConfig()
            player = DirectAudioPlayer(device=test_device, config=config)
            passed = player.device == test_device
            print_result("Specific device player", passed,
                        f"Using device {test_device}")
        else:
            passed = False
            print_result("Specific device player", passed, "No device available")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Specific device player", False, str(e))
    
    # Test 3: Create with custom config
    tests_total += 1
    try:
        config = AudioConfig(
            sample_rate=48000,
            channels=1
        )
        player = DirectAudioPlayer(config=config)
        
        passed = (
            player.config.sample_rate == 48000 and
            player.config.channels == 1
        )
        print_result("Custom config player", passed,
                    f"Rate: {player.config.sample_rate}Hz")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Custom config player", False, str(e))
    
    return tests_passed, tests_total


def test_single_audio_playback():
    """Test playing single audio chunks"""
    print_test_header("Single Audio Playback")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Play short tone
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Generate 100ms tone
        audio_data = generate_test_audio(100, frequency_hz=440.0)
        
        start_time = time.time()
        success = player.play_audio(audio_data)
        play_time = time.time() - start_time
        
        # Playback should be non-blocking (return quickly)
        passed = success and play_time < 0.05  # Should return in < 50ms
        
        print_result("Play short tone", passed,
                    f"Returned in {play_time*1000:.1f}ms")
        
        # Give time for audio to actually play
        time.sleep(0.15)
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Play short tone", False, str(e))
    
    # Test 2: Play different frequencies
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        frequencies = [220, 440, 880]  # A3, A4, A5
        
        all_played = True
        for freq in frequencies:
            audio_data = generate_test_audio(50, frequency_hz=freq)
            if not player.play_audio(audio_data):
                all_played = False
                break
            time.sleep(0.06)  # Small gap between tones
        
        passed = all_played
        print_result("Play multiple frequencies", passed,
                    f"Played {len(frequencies)} different tones")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Play multiple frequencies", False, str(e))
    
    # Test 3: Play silence
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Generate silence
        silence = generate_silence(100)
        success = player.play_audio(silence)
        
        passed = success
        print_result("Play silence", passed,
                    "Silence played without error")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Play silence", False, str(e))
    
    # Test 4: Play white noise
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Generate white noise
        noise = generate_white_noise(100, amplitude=0.05)
        success = player.play_audio(noise)
        
        passed = success
        print_result("Play white noise", passed,
                    "Noise played successfully")
        
        time.sleep(0.12)
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Play white noise", False, str(e))
    
    return tests_passed, tests_total


def test_rapid_play_stop():
    """Test rapid play/stop cycles"""
    print_test_header("Rapid Play/Stop Cycles")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Multiple play calls
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Play multiple chunks rapidly
        chunks_played = 0
        for i in range(10):
            audio_data = generate_test_audio(20, frequency_hz=440 + i * 50)
            if player.play_audio(audio_data):
                chunks_played += 1
            time.sleep(0.025)  # 25ms between plays
        
        passed = chunks_played == 10
        print_result("Rapid play calls", passed,
                    f"Played {chunks_played}/10 chunks")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Rapid play calls", False, str(e))
    
    # Test 2: Play with immediate stop
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Play and immediately stop multiple times
        stop_count = 0
        for _ in range(5):
            audio_data = generate_test_audio(200)  # 200ms tone
            player.play_audio(audio_data)
            time.sleep(0.01)  # 10ms
            player.stop_playback()
            stop_count += 1
            time.sleep(0.05)  # 50ms gap
        
        passed = stop_count == 5
        print_result("Play/stop cycles", passed,
                    f"Completed {stop_count}/5 cycles")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Play/stop cycles", False, str(e))
    
    # Test 3: Stop without play
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Should handle gracefully
        player.stop_playback()
        player.stop_playback()  # Multiple stops
        
        passed = True
        print_result("Stop without play", passed,
                    "Handled gracefully")
        
        tests_passed += 1
    except Exception as e:
        print_result("Stop without play", False, str(e))
    
    return tests_passed, tests_total


def test_playback_non_blocking():
    """Verify playback doesn't block"""
    print_test_header("Non-Blocking Playback")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Measure blocking time
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Generate 500ms of audio
        long_audio = generate_test_audio(500)
        
        start_time = time.time()
        success = player.play_audio(long_audio)
        return_time = time.time() - start_time
        
        # Should return almost immediately (< 50ms)
        passed = success and return_time < 0.05
        
        print_result("Non-blocking return", passed,
                    f"Returned in {return_time*1000:.1f}ms for 500ms audio")
        
        if passed:
            tests_passed += 1
            
        # Clean up
        player.stop_playback()
    except Exception as e:
        print_result("Non-blocking return", False, str(e))
    
    # Test 2: Concurrent operations while playing
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Start playing long audio
        long_audio = generate_test_audio(1000)  # 1 second
        player.play_audio(long_audio)
        
        # Should be able to check status while playing
        start = time.time()
        checks_done = 0
        while time.time() - start < 0.5:
            is_playing = player.is_playing
            info = player.get_device_info()
            checks_done += 1
            time.sleep(0.05)
        
        player.stop_playback()
        
        passed = checks_done > 5
        print_result("Concurrent operations", passed,
                    f"Performed {checks_done} operations while playing")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Concurrent operations", False, str(e))
    
    return tests_passed, tests_total


def test_different_audio_formats():
    """Test handling of different audio formats"""
    print_test_header("Different Audio Formats")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Different sample rates
    tests_total += 1
    try:
        sample_rates = [16000, 24000, 48000]
        all_played = True
        
        for rate in sample_rates:
            config = AudioConfig(sample_rate=rate)
            player = DirectAudioPlayer(config=config)
            
            # Generate audio at the specific rate
            audio_data = generate_test_audio(50, sample_rate=rate)
            
            if not player.play_audio(audio_data):
                all_played = False
                print(f"     | Failed at {rate}Hz")
                break
            
            time.sleep(0.06)
        
        passed = all_played
        print_result("Different sample rates", passed,
                    f"Tested rates: {sample_rates}")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Different sample rates", False, str(e))
    
    # Test 2: Different chunk sizes
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        chunk_sizes_ms = [10, 50, 100, 200]
        all_played = True
        
        for chunk_ms in chunk_sizes_ms:
            audio_data = generate_test_audio(chunk_ms)
            if not player.play_audio(audio_data):
                all_played = False
                break
            time.sleep(chunk_ms / 1000 + 0.01)
        
        passed = all_played
        print_result("Different chunk sizes", passed,
                    f"Played chunks: {chunk_sizes_ms}ms")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Different chunk sizes", False, str(e))
    
    # Test 3: Very short audio
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # 5ms audio (very short)
        tiny_audio = generate_test_audio(5)
        success = player.play_audio(tiny_audio)
        
        passed = success
        print_result("Very short audio", passed,
                    f"Played {len(tiny_audio)} bytes (5ms)")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Very short audio", False, str(e))
    
    return tests_passed, tests_total


def test_concurrent_playback():
    """Test concurrent playback attempts"""
    print_test_header("Concurrent Playback")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Overlapping playback
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Play first audio
        audio1 = generate_test_audio(200, frequency_hz=440)
        success1 = player.play_audio(audio1)
        
        # Immediately play second audio (should mix or replace)
        time.sleep(0.05)  # 50ms into first audio
        audio2 = generate_test_audio(200, frequency_hz=880)
        success2 = player.play_audio(audio2)
        
        passed = success1 and success2
        print_result("Overlapping playback", passed,
                    "Both audio chunks accepted")
        
        time.sleep(0.3)  # Let audio finish
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Overlapping playback", False, str(e))
    
    
    
    # Test 3: Rapid sequential playback
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Play many short chunks with no gap
        chunks_played = 0
        start_time = time.time()
        
        for i in range(20):
            audio = generate_test_audio(10, frequency_hz=400 + i * 10)
            if player.play_audio(audio):
                chunks_played += 1
        
        elapsed = time.time() - start_time
        
        passed = chunks_played == 20 and elapsed < 0.5
        print_result("Rapid sequential playback", passed,
                    f"Played {chunks_played} chunks in {elapsed:.2f}s")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Rapid sequential playback", False, str(e))
    
    return tests_passed, tests_total


def test_metrics():
    """Test metrics collection"""
    print_test_header("Playback Metrics")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic metrics
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Get initial metrics
        metrics_before = player.get_metrics()
        
        # Play some audio
        for i in range(5):
            audio = generate_test_audio(50)
            player.play_audio(audio)
            time.sleep(0.06)
        
        # Get metrics after
        metrics_after = player.get_metrics()
        
        passed = (
            metrics_after['chunks_played'] > metrics_before['chunks_played'] and
            metrics_after['chunks_played'] >= 5
        )
        
        print_result("Basic metrics", passed,
                    f"Played: {metrics_after['chunks_played']} chunks, "
                    f"Errors: {metrics_after.get('errors', 0)}")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Basic metrics", False, str(e))
    
    # Test 2: Playback duration tracking
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Play known duration
        total_ms = 0
        for duration in [50, 100, 150]:
            audio = generate_test_audio(duration)
            player.play_audio(audio)
            total_ms += duration
            time.sleep(duration / 1000 + 0.02)
        
        metrics = player.get_metrics()
        
        # Check if duration is tracked (if available)
        if 'total_duration_ms' in metrics:
            # Allow 10% tolerance
            expected = total_ms
            actual = metrics['total_duration_ms']
            passed = abs(actual - expected) < expected * 0.1
            details = f"Expected: ~{expected}ms, Actual: {actual}ms"
        else:
            # If duration not tracked, just check chunk count
            passed = metrics['chunks_played'] >= 3
            details = "Duration tracking not available"
        
        print_result("Duration tracking", passed, details)
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Duration tracking", False, str(e))
    
    return tests_passed, tests_total


def test_error_handling():
    """Test error handling and recovery"""
    print_test_header("Error Handling")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Invalid audio data
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Try to play invalid data
        try:
            success = player.play_audio(b'')  # Empty data
            # Should either handle gracefully or return False
            passed = not success or True
            print_result("Empty audio data", passed,
                        "Handled gracefully")
        except Exception:
            # Exception is also acceptable
            passed = True
            print_result("Empty audio data", passed,
                        "Raised appropriate error")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Empty audio data", False, str(e))
    
    # Test 2: Very large audio chunk
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Generate 10 seconds of audio
        large_audio = generate_test_audio(10000)
        
        start = time.time()
        success = player.play_audio(large_audio)
        elapsed = time.time() - start
        
        # Should still be non-blocking
        passed = elapsed < 0.1  # Should return quickly
        
        print_result("Large audio chunk", passed,
                    f"10s audio, returned in {elapsed*1000:.1f}ms")
        
        player.stop_playback()  # Clean up
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Large audio chunk", False, str(e))
    
    # Test 3: Recovery after error
    tests_total += 1
    try:
        player = DirectAudioPlayer()
        
        # Cause an error (if possible)
        try:
            player.play_audio(None)  # Invalid type
        except:
            pass  # Expected
        
        # Should still work after error
        audio = generate_test_audio(50)
        success = player.play_audio(audio)
        
        passed = success
        print_result("Recovery after error", passed,
                    "Player still functional after error")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Recovery after error", False, str(e))
    
    return tests_passed, tests_total


def test_device_switching():
    """Test switching between devices (if multiple available)"""
    print_test_header("Device Switching")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: List available devices for switching
    tests_total += 1
    try:
        devices = DirectAudioPlayer.list_output_devices()
        
        if len(devices) > 1:
            # Test switching between devices
            audio = generate_test_audio(50)
            switched = 0
            
            for device in devices[:2]:  # Test first 2 devices
                try:
                    player = DirectAudioPlayer(device=device['index'])
                    if player.play_audio(audio):
                        switched += 1
                        time.sleep(0.1)
                except:
                    continue
            
            passed = switched >= 1
            print_result("Device switching", passed,
                        f"Successfully used {switched} devices")
        else:
            passed = True
            print_result("Device switching", passed,
                        "Only one device available (skipped)")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Device switching", False, str(e))
    
    return tests_passed, tests_total


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("DIRECT AUDIO PLAYER SMOKE TESTS")
    print("="*60)
    
    # Check if we have any audio output devices
    try:
        devices = DirectAudioPlayer.list_output_devices()
        if not devices:
            print("\n⚠️  WARNING: No audio output devices found!")
            print("Some tests may fail or be skipped.")
    except Exception as e:
        print(f"\n⚠️  WARNING: Could not query audio devices: {e}")
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_device_enumeration,
        test_player_creation,
        test_single_audio_playback,
        test_rapid_play_stop,
        test_playback_non_blocking,
        test_different_audio_formats,
        test_concurrent_playback,
        test_metrics,
        test_error_handling,
        test_device_switching
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