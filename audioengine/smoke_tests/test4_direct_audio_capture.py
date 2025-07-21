# here is realtimevoiceapi/smoke_tests/audio/test_direct_audio_capture.py
"""
Smoke tests for direct audio capture functionality.

Run with: python test_direct_audio_capture.py

python -m realtimevoiceapi.smoke_tests.audio.test_direct_audio_capture
"""
"""
Smoke tests for direct audio capture functionality.

Run with: python -m smoke_tests.test_direct_audio_capture
"""

import sys
import time
import asyncio
import threading
import traceback
import queue
from typing import List, Optional, Dict, Any

# Import audio capture components
try:
    from audioengine.direct_audio_capture import DirectAudioCapture
    from audioengine.audio_types import (
        AudioConfig, AudioFormat, AudioConstants
    )
    from audioengine.exceptions import AudioError
    import sounddevice as sd
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure realtimevoiceapi is in PYTHONPATH and sounddevice is installed")
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


def get_test_device():
    """Get a suitable test device (prefer default)"""
    try:
        # Get default input device
        default_device = sd.default.device[0]  # Input device
        if default_device is not None:
            return default_device
        
        # Fall back to first available device
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                return i
        
        return None
    except Exception:
        return None


def run_async(coro):
    """Helper to run async code in sync context"""
    if sys.platform == 'win32':
        # Windows requires specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Clean up pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def test_device_enumeration():
    """Test device enumeration and info"""
    print_test_header("Device Enumeration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: List available devices using static method
    tests_total += 1
    try:
        devices = DirectAudioCapture.list_devices()
        
        passed = isinstance(devices, list)
        print_result("List devices", passed,
                    f"Found {len(devices)} input devices")
        
        if passed and len(devices) > 0:
            print("     | Input devices:")
            for dev in devices[:3]:  # Show first 3
                default_mark = " (default)" if dev.get('default') else ""
                print(f"     |   [{dev['index']}] {dev['name']}{default_mark}")
            if len(devices) > 3:
                print(f"     |   ... and {len(devices) - 3} more")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("List devices", False, str(e))
    
    # Test 2: Get default device
    tests_total += 1
    try:
        default_input = sd.default.device[0]
        
        if default_input is not None:
            device_info = sd.query_devices(default_input)
            passed = device_info['max_input_channels'] > 0
            print_result("Default device", passed,
                        f"Device {default_input}: {device_info['name']}")
        else:
            # No default device is acceptable on some systems
            passed = True
            print_result("Default device", passed, "No default device configured")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default device", False, str(e))
    
    # Test 3: Query specific device info through DirectAudioCapture
    tests_total += 1
    try:
        test_device = get_test_device()
        
        if test_device is not None:
            config = AudioConfig()
            capture = DirectAudioCapture(device=test_device, config=config)
            info = capture.get_device_info()
            
            passed = (
                'name' in info and
                'sample_rate' in info and
                'channels' in info
            )
            print_result("Device info", passed,
                        f"Name: {info.get('name', 'Unknown')}, "
                        f"Channels: {info.get('channels', 0)}")
        else:
            passed = False
            print_result("Device info", passed, "No input device available")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Device info", False, str(e))
    
    return tests_passed, tests_total


def test_capture_creation():
    """Test capture object creation"""
    print_test_header("Capture Creation")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create with default device
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000, chunk_duration_ms=100)
        capture = DirectAudioCapture(device=None, config=config)
        
        passed = capture is not None and not capture.is_capturing
        print_result("Default device capture", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default device capture", False, str(e))
    
    # Test 2: Create with specific device
    tests_total += 1
    try:
        test_device = get_test_device()
        if test_device is not None:
            config = AudioConfig()
            capture = DirectAudioCapture(device=test_device, config=config)
            passed = capture.device == test_device
            print_result("Specific device capture", passed,
                        f"Using device {test_device}")
        else:
            passed = False
            print_result("Specific device capture", passed, "No device available")
        
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Specific device capture", False, str(e))
    
    # Test 3: Create with custom config
    tests_total += 1
    try:
        config = AudioConfig(
            sample_rate=16000,
            chunk_duration_ms=50,
            channels=1
        )
        capture = DirectAudioCapture(config=config)
        
        passed = (
            capture.config.sample_rate == 16000 and
            capture.config.chunk_duration_ms == 50
        )
        print_result("Custom config capture", passed,
                    f"Rate: {capture.config.sample_rate}Hz, "
                    f"Chunk: {capture.config.chunk_duration_ms}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Custom config capture", False, str(e))
    
    return tests_passed, tests_total


def test_capture_start_stop():
    """Test capture start/stop cycles"""
    print_test_header("Capture Start/Stop")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic start/stop
    tests_total += 1
    async def test_basic_start_stop():
        try:
            config = AudioConfig(chunk_duration_ms=50)
            capture = DirectAudioCapture(config=config)
            
            # Start capture
            audio_queue = await capture.start_async_capture()
            started = capture.is_capturing
            
            # Brief capture
            await asyncio.sleep(0.1)
            
            # Stop capture
            capture.stop_capture()
            stopped = not capture.is_capturing
            
            return started and stopped, f"Started: {started}, Stopped: {stopped}"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_basic_start_stop())
    print_result("Basic start/stop", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 2: Multiple start/stop cycles
    tests_total += 1
    async def test_multiple_cycles():
        try:
            capture = DirectAudioCapture()
            cycles_passed = 0
            
            for i in range(3):
                q = await capture.start_async_capture()
                if capture.is_capturing:
                    await asyncio.sleep(0.05)
                    capture.stop_capture()
                    if not capture.is_capturing:
                        cycles_passed += 1
            
            return cycles_passed == 3, f"Completed {cycles_passed}/3 cycles"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_multiple_cycles())
    print_result("Multiple cycles", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 3: Stop without start
    tests_total += 1
    try:
        capture = DirectAudioCapture()
        # Should not error
        capture.stop_capture()
        passed = not capture.is_capturing
        print_result("Stop without start", passed,
                    "Handled gracefully")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Stop without start", False, str(e))
    
    return tests_passed, tests_total


def test_audio_chunk_generation():
    """Test audio chunk generation timing"""
    print_test_header("Audio Chunk Generation")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Verify chunk timing
    tests_total += 1
    async def test_chunk_timing():
        try:
            chunk_ms = 50
            config = AudioConfig(chunk_duration_ms=chunk_ms)
            capture = DirectAudioCapture(config=config)
            
            audio_queue = await capture.start_async_capture()
            chunk_times = []
            last_time = time.time()
            
            # Collect 5 chunks
            for _ in range(5):
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                    current_time = time.time()
                    chunk_times.append((current_time - last_time) * 1000)
                    last_time = current_time
                except asyncio.TimeoutError:
                    break
            
            capture.stop_capture()
            
            if len(chunk_times) >= 3:
                # Skip first chunk (startup time)
                avg_interval = sum(chunk_times[1:]) / len(chunk_times[1:])
                # Allow 30% tolerance for async overhead
                passed = abs(avg_interval - chunk_ms) < chunk_ms * 0.3
                return passed, f"Target: {chunk_ms}ms, Actual avg: {avg_interval:.1f}ms"
            else:
                return False, f"Only got {len(chunk_times)} chunks"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_chunk_timing())
    print_result("Chunk timing", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 2: Verify chunk size
    tests_total += 1
    async def test_chunk_size():
        try:
            config = AudioConfig(
                sample_rate=24000,
                chunk_duration_ms=20,
                channels=1
            )
            capture = DirectAudioCapture(config=config)
            
            expected_size = config.chunk_size_bytes(20)
            audio_queue = await capture.start_async_capture()
            
            # Get a chunk
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                actual_size = len(chunk)
                
                # Allow small variation due to buffer alignment
                passed = abs(actual_size - expected_size) < 100
                capture.stop_capture()
                return passed, f"Expected: {expected_size} bytes, Got: {actual_size} bytes"
            except asyncio.TimeoutError:
                capture.stop_capture()
                return False, "No chunks received"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_chunk_size())
    print_result("Chunk size", passed, details)
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def test_queue_operations():
    """Test queue operations"""
    print_test_header("Queue Operations")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Async queue get
    tests_total += 1
    async def test_async_get():
        try:
            capture = DirectAudioCapture()
            audio_queue = await capture.start_async_capture()
            
            # Should get chunk within reasonable time
            start = time.time()
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            elapsed = time.time() - start
            
            capture.stop_capture()
            
            return (chunk is not None and len(chunk) > 0, 
                   f"Got {len(chunk) if chunk else 0} bytes in {elapsed:.3f}s")
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_async_get())
    print_result("Async queue get", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 2: Queue size monitoring
    tests_total += 1
    async def test_queue_monitoring():
        try:
            capture = DirectAudioCapture(config=AudioConfig(chunk_duration_ms=10))
            audio_queue = await capture.start_async_capture()
            
            # Let queue fill up
            await asyncio.sleep(0.1)
            queue_size = audio_queue.qsize()
            
            # Drain some chunks
            drained = 0
            while not audio_queue.empty() and drained < 3:
                await audio_queue.get()
                drained += 1
            
            new_size = audio_queue.qsize()
            
            capture.stop_capture()
            
            return (queue_size > 0 and new_size < queue_size,
                   f"Initial size: {queue_size}, After drain: {new_size}")
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_queue_monitoring())
    print_result("Queue monitoring", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 3: Continuous streaming
    tests_total += 1
    async def test_continuous_stream():
        try:
            capture = DirectAudioCapture(config=AudioConfig(chunk_duration_ms=20))
            audio_queue = await capture.start_async_capture()
            
            chunks_received = 0
            total_bytes = 0
            start_time = time.time()
            
            # Stream for 0.5 seconds
            while time.time() - start_time < 0.5:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    chunks_received += 1
                    total_bytes += len(chunk)
                except asyncio.TimeoutError:
                    continue
            
            capture.stop_capture()
            
            # Should get roughly 25 chunks in 0.5s with 20ms chunks
            expected_chunks = 25
            passed = chunks_received >= expected_chunks * 0.7  # Allow 30% tolerance
            
            return passed, f"Got {chunks_received} chunks ({total_bytes} bytes)"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_continuous_stream())
    print_result("Continuous streaming", passed, details)
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def test_different_chunk_sizes():
    """Test capture with different chunk sizes"""
    print_test_header("Different Chunk Sizes")
    
    tests_passed = 0
    tests_total = 0
    
    chunk_sizes = [10, 20, 50, 100, 200]
    
    async def test_chunk_size_async(chunk_ms):
        try:
            config = AudioConfig(chunk_duration_ms=chunk_ms)
            capture = DirectAudioCapture(config=config)
            
            audio_queue = await capture.start_async_capture()
            
            # Get one chunk
            try:
                chunk = await asyncio.wait_for(
                    audio_queue.get(), 
                    timeout=chunk_ms/1000 * 3  # 3x chunk duration
                )
                expected_size = config.chunk_size_bytes(chunk_ms)
                actual_size = len(chunk)
                
                capture.stop_capture()
                
                # Allow 10% tolerance
                passed = abs(actual_size - expected_size) < expected_size * 0.1
                return passed, f"Expected: ~{expected_size}, Got: {actual_size}"
            except asyncio.TimeoutError:
                capture.stop_capture()
                return False, "No chunk received"
        except Exception as e:
            return False, str(e)
    
    for chunk_ms in chunk_sizes:
        tests_total += 1
        passed, details = run_async(test_chunk_size_async(chunk_ms))
        print_result(f"{chunk_ms}ms chunks", passed, details)
        if passed:
            tests_passed += 1
    
    return tests_passed, tests_total


def test_metrics():
    """Test metrics collection"""
    print_test_header("Metrics Collection")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic metrics
    tests_total += 1
    async def test_basic_metrics():
        try:
            capture = DirectAudioCapture()
            
            # Get initial metrics
            metrics_before = capture.get_metrics()
            
            # Capture some audio
            queue = await capture.start_async_capture()
            
            # Collect chunks for 200ms
            chunks_collected = 0
            start = time.time()
            while time.time() - start < 0.2:
                try:
                    await asyncio.wait_for(queue.get(), timeout=0.05)
                    chunks_collected += 1
                except asyncio.TimeoutError:
                    continue
            
            # Get metrics after capture
            metrics_after = capture.get_metrics()
            
            capture.stop_capture()
            
            passed = (
                metrics_after['chunks_captured'] > metrics_before['chunks_captured'] and
                metrics_after['chunks_captured'] >= chunks_collected
            )
            
            return passed, (f"Captured: {metrics_after['chunks_captured']}, "
                          f"Dropped: {metrics_after['chunks_dropped']}")
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_basic_metrics())
    print_result("Basic metrics", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 2: Duration tracking
    tests_total += 1
    async def test_duration_tracking():
        try:
            capture = DirectAudioCapture()
            queue = await capture.start_async_capture()
            
            # Capture for known duration
            await asyncio.sleep(0.5)
            
            metrics = capture.get_metrics()
            capture.stop_capture()
            
            # Check duration is reasonable (allow 20% tolerance)
            duration = metrics['duration_seconds']
            passed = 0.4 <= duration <= 0.6
            
            return passed, f"Duration: {duration:.2f}s"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_duration_tracking())
    print_result("Duration tracking", passed, details)
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def test_error_handling():
    """Test graceful error handling"""
    print_test_header("Error Handling")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Invalid device
    tests_total += 1
    try:
        # Use clearly invalid device ID
        invalid_device = 9999
        config = AudioConfig()
        
        try:
            capture = DirectAudioCapture(device=invalid_device, config=config)
            passed = False  # Should have raised error
        except (AudioError, sd.PortAudioError):
            passed = True  # Expected error
        
        print_result("Invalid device", passed,
                    "Raised appropriate error")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Invalid device", False, str(e))
    
    # Test 2: Double start
    tests_total += 1
    async def test_double_start():
        try:
            capture = DirectAudioCapture()
            
            # Start once
            await capture.start_async_capture()
            
            # Try to start again
            try:
                await capture.start_async_capture()
                capture.stop_capture()
                return False, "Should have raised error"
            except AudioError:
                capture.stop_capture()
                return True, "Correctly prevented double start"
        except Exception as e:
            return False, str(e)
    
    passed, details = run_async(test_double_start())
    print_result("Double start prevention", passed, details)
    if passed:
        tests_passed += 1
    
    # Test 3: Resource cleanup after error
    tests_total += 1
    try:
        capture = DirectAudioCapture()
        
        # Multiple stops should be safe
        capture.stop_capture()
        capture.stop_capture()
        
        passed = True
        print_result("Multiple stops", passed,
                    "Handled gracefully")
        tests_passed += 1
    except Exception as e:
        print_result("Multiple stops", False, str(e))
    
    return tests_passed, tests_total


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("DIRECT AUDIO CAPTURE SMOKE TESTS")
    print("="*60)
    
    # Check if we have any audio devices
    try:
        devices = DirectAudioCapture.list_devices()
        if not devices:
            print("\n⚠️  WARNING: No audio input devices found!")
            print("Some tests may fail or be skipped.")
    except Exception as e:
        print(f"\n⚠️  WARNING: Could not query audio devices: {e}")
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_device_enumeration,
        test_capture_creation,
        test_capture_start_stop,
        test_audio_chunk_generation,
        test_queue_operations,
        test_different_chunk_sizes,
        test_metrics,
        test_error_handling
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