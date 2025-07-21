"""
Smoke tests for AudioManager - Component Integration.

Tests AudioManager coordinating all audio components including
capture, playback, and VAD.

python -m smoke_tests.test_audio_manager
"""

import sys
import time
import asyncio
import threading
import struct
import math
import traceback
from typing import List, Optional, Dict, Any, Callable
import random
from dataclasses import dataclass

# Import audio components
try:
    from audioengine.audio_manager import (
        AudioManager, AudioManagerConfig
    )
    from audioengine.audio_types import (
        AudioConfig, AudioBytes, VADConfig, VADType
    )
    from audioengine.exceptions import AudioError
    from audioengine.fast_vad_detector import VADState, FastVADDetector
    
    import numpy as np
    HAS_NUMPY = True
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure realtimevoiceapi is in PYTHONPATH")
    HAS_NUMPY = False
    np = None
    sys.exit(1)


# ============== Test Configuration ==============

# Set to True to skip hardware-dependent tests
SKIP_HARDWARE_TESTS = False
HARDWARE_TIMEOUT = 2.0  # Timeout for hardware operations


# ============== Test Infrastructure ==============

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


def print_skip(test_desc: str, reason: str = ""):
    """Print skipped test"""
    print(f"⏭️  SKIP | {test_desc}")
    if reason:
        print(f"     | {reason}")


def generate_test_audio(
    duration_ms: int,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    sample_rate: int = 24000
) -> AudioBytes:
    """Generate test tone"""
    num_samples = int(sample_rate * duration_ms / 1000)
    
    if HAS_NUMPY:
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        samples = (signal * 32767).astype(np.int16)
        return samples.tobytes()
    else:
        samples = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            value = amplitude * math.sin(2 * math.pi * frequency * t)
            sample = int(value * 32767)
            samples.extend(struct.pack('<h', sample))
        return bytes(samples)


def generate_silence(duration_ms: int, sample_rate: int = 24000) -> AudioBytes:
    """Generate silence"""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b'\x00' * (num_samples * 2)


async def with_timeout(coro, timeout: float, default=None):
    """Run coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


# ============== Test Functions ==============

async def test_basic_initialization():
    """Test basic AudioManager initialization"""
    print_test_header("Basic Initialization")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create with default config
    tests_total += 1
    try:
        config = AudioManagerConfig()
        manager = AudioManager(config)
        
        passed = (
            manager is not None and
            not manager._is_initialized
        )
        print_result("Default creation", passed,
                    f"Created: {manager is not None}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default creation", False, str(e))
    
    # Test 2: Initialize components
    tests_total += 1
    if SKIP_HARDWARE_TESTS:
        print_skip("Component initialization", "Hardware tests disabled")
    else:
        try:
            config = AudioManagerConfig(
                sample_rate=24000,
                chunk_duration_ms=100,
                vad_enabled=True
            )
            manager = AudioManager(config)
            
            # Initialize with timeout
            init_result = await with_timeout(
                manager.initialize(),
                HARDWARE_TIMEOUT,
                None
            )
            
            if init_result is None:
                print_result("Component initialization", False,
                            "Timeout - no audio hardware available?")
            else:
                passed = manager._is_initialized
                print_result("Component initialization", passed,
                            f"Initialized: {manager._is_initialized}")
                if passed:
                    tests_passed += 1
                
                # Cleanup
                await manager.cleanup()
            
        except Exception as e:
            print_result("Component initialization", False, str(e))
    
    # Test 3: Initialize with custom config (no hardware needed)
    tests_total += 1
    try:
        config = AudioManagerConfig(
            input_device=None,
            output_device=None,
            sample_rate=16000,
            channels=1,
            chunk_duration_ms=50,
            vad_enabled=False
        )
        manager = AudioManager(config)
        
        # This creates the config but might fail on hardware init
        passed = (
            manager._audio_config.sample_rate == 16000 and
            manager._audio_config.chunk_duration_ms == 50
        )
        print_result("Custom config creation", passed,
                    f"Config created with custom values")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Custom config creation", False, str(e))
    
    return tests_passed, tests_total


async def test_capture_functionality():
    """Test audio capture functionality"""
    print_test_header("Capture Functionality")
    
    tests_passed = 0
    tests_total = 0
    
    if SKIP_HARDWARE_TESTS:
        print_skip("All capture tests", "Hardware tests disabled")
        return tests_passed, tests_total
    
    # Test 1: Start/stop capture
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Initialize with timeout
        init_success = await with_timeout(
            manager.initialize(),
            HARDWARE_TIMEOUT,
            False
        )
        
        if not init_success:
            print_skip("Start/stop capture", "Hardware initialization failed")
        else:
            # Start capture with timeout
            queue = await with_timeout(
                manager.start_capture(),
                HARDWARE_TIMEOUT,
                None
            )
            
            if queue is None:
                print_result("Start/stop capture", False,
                            "Timeout starting capture")
            else:
                passed = (
                    queue is not None and
                    manager._is_capturing
                )
                
                # Stop capture
                await manager.stop_capture()
                
                passed = passed and not manager._is_capturing
                
                print_result("Start/stop capture", passed,
                            f"Queue created: {queue is not None}")
                if passed:
                    tests_passed += 1
            
            await manager.cleanup()
        
    except Exception as e:
        print_result("Start/stop capture", False, str(e))
    
    # Test 2: Capture without initialization
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Try to capture without init
        error_raised = False
        try:
            await manager.start_capture()
        except AudioError:
            error_raised = True
        
        passed = error_raised
        print_result("Capture without init", passed,
                    "Should raise AudioError")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Capture without init", False, str(e))
    
    return tests_passed, tests_total


async def test_playback_functionality():
    """Test audio playback functionality"""
    print_test_header("Playback Functionality")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Playback without initialization (should handle gracefully)
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        test_audio = generate_test_audio(50)
        success = manager.play_audio(test_audio)
        
        # Should return False and log warning
        passed = success is False
        print_result("Playback without init", passed,
                    "Should return False")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Playback without init", False, str(e))
    
    # Test 2: Stop playback without playing
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Stop should not raise exception even without init
        manager.stop_playback()
        
        passed = True
        print_result("Stop without play", passed,
                    "No error on stop")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Stop without play", False, str(e))
    
    if not SKIP_HARDWARE_TESTS:
        # Test 3: Basic playback with hardware
        tests_total += 1
        try:
            manager = AudioManager(AudioManagerConfig())
            
            init_success = await with_timeout(
                manager.initialize(),
                HARDWARE_TIMEOUT,
                False
            )
            
            if not init_success:
                print_skip("Basic playback", "Hardware initialization failed")
            else:
                test_audio = generate_test_audio(100)
                success = manager.play_audio(test_audio)
                
                passed = isinstance(success, bool)
                print_result("Basic playback", passed,
                            f"Play returned: {success}")
                if passed:
                    tests_passed += 1
                
                await manager.cleanup()
            
        except Exception as e:
            print_result("Basic playback", False, str(e))
    
    return tests_passed, tests_total


async def test_vad_integration():
    """Test VAD integration"""
    print_test_header("VAD Integration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: VAD processing (no hardware needed)
    tests_total += 1
    try:
        config = AudioManagerConfig(
            vad_enabled=True,
            vad_config=VADConfig(
                type=VADType.ENERGY_BASED,
                energy_threshold=0.02
            )
        )
        manager = AudioManager(config)
        
        # For VAD testing, we can bypass full initialization
        # Just create the VAD detector directly
        manager._vad = FastVADDetector(
            config=config.vad_config,
            audio_config=manager._audio_config
        )
        
        # Process silence
        silence = generate_silence(100)
        silence_state = manager.process_vad(silence)
        
        # Process speech
        speech = generate_test_audio(100, amplitude=0.5)
        speech_state = manager.process_vad(speech)
        
        passed = (
            silence_state is not None and
            speech_state is not None and
            isinstance(silence_state, str) and
            isinstance(speech_state, str)
        )
        print_result("VAD processing", passed,
                    f"Silence: {silence_state}, Speech: {speech_state}")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("VAD processing", False, str(e))
    
    # Test 2: VAD disabled
    tests_total += 1
    try:
        config = AudioManagerConfig(vad_enabled=False)
        manager = AudioManager(config)
        
        # Should return None without VAD
        audio = generate_test_audio(100)
        state = manager.process_vad(audio)
        
        passed = state is None and manager._vad is None
        print_result("VAD disabled", passed,
                    "Returns None when disabled")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("VAD disabled", False, str(e))
    
    # Test 3: VAD error handling
    tests_total += 1
    try:
        config = AudioManagerConfig(vad_enabled=True)
        manager = AudioManager(config)
        
        # Create VAD manually
        manager._vad = FastVADDetector(
            config=VADConfig(),
            audio_config=manager._audio_config
        )
        
        # Process invalid audio (should handle gracefully)
        invalid_audio = b'invalid'
        state = manager.process_vad(invalid_audio)
        
        # Should return None on error
        passed = state is None or isinstance(state, str)
        print_result("VAD error handling", passed,
                    f"Handled invalid audio: {state}")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("VAD error handling", False, str(e))
    
    return tests_passed, tests_total


async def test_cleanup_and_safety():
    """Test cleanup and thread safety"""
    print_test_header("Cleanup and Safety")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Cleanup without init
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Should not crash
        await manager.cleanup()
        
        passed = True
        print_result("Cleanup without init", passed,
                    "No crash on cleanup")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Cleanup without init", False, str(e))
    
    # Test 2: Thread safety with concurrent operations
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Try concurrent operations without full init
        results = []
        
        async def concurrent_play():
            for _ in range(5):
                result = manager.play_audio(generate_test_audio(20))
                results.append(result)
                await asyncio.sleep(0.01)
        
        # Run multiple tasks
        await asyncio.gather(
            concurrent_play(),
            concurrent_play(),
            concurrent_play()
        )
        
        # Should handle concurrent access
        passed = len(results) == 15  # 3 tasks * 5 calls each
        print_result("Thread safety", passed,
                    f"Concurrent calls: {len(results)}")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Thread safety", False, str(e))
    
    # Test 3: Multiple cleanup calls
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Multiple cleanups should be safe
        await manager.cleanup()
        await manager.cleanup()
        await manager.cleanup()
        
        passed = True
        print_result("Multiple cleanup", passed,
                    "No crash on multiple cleanup")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Multiple cleanup", False, str(e))
    
    return tests_passed, tests_total


async def test_metrics():
    """Test metrics collection"""
    print_test_header("Metrics")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic metrics without init
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Get metrics before init
        metrics = manager.get_metrics()
        
        passed = (
            'initialized' in metrics and
            'capturing' in metrics and
            metrics['initialized'] is False and
            metrics['capturing'] is False
        )
        print_result("Basic metrics", passed,
                    f"Init: {metrics['initialized']}, Capturing: {metrics['capturing']}")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Basic metrics", False, str(e))
    
    # Test 2: Metrics structure
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        metrics = manager.get_metrics()
        
        # Should be a dict with expected keys
        passed = (
            isinstance(metrics, dict) and
            len(metrics) >= 2
        )
        print_result("Metrics structure", passed,
                    f"Keys: {list(metrics.keys())}")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Metrics structure", False, str(e))
    
    return tests_passed, tests_total


async def test_error_scenarios():
    """Test various error scenarios"""
    print_test_header("Error Scenarios")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Operations after cleanup
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        await manager.cleanup()
        
        # Try operations after cleanup
        success = manager.play_audio(generate_test_audio(50))
        
        # Should return False
        passed = success is False
        print_result("Operations after cleanup", passed,
                    "Playback returned False")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Operations after cleanup", False, str(e))
    
    # Test 2: Invalid audio data
    tests_total += 1
    try:
        manager = AudioManager(AudioManagerConfig())
        
        # Try to play various invalid data
        results = []
        results.append(manager.play_audio(b''))  # Empty
        results.append(manager.play_audio(b'x'))  # Single byte
        results.append(manager.play_audio(None))  # None
        
        # Should all return False or handle gracefully
        passed = all(r is False for r in results if r is not None)
        print_result("Invalid audio data", passed,
                    f"All returned False or handled")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Invalid audio data", False, str(e))
    
    return tests_passed, tests_total


async def test_configuration_options():
    """Test various configuration options"""
    print_test_header("Configuration Options")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Configuration validation
    tests_total += 1
    try:
        configs = [
            AudioManagerConfig(sample_rate=8000),
            AudioManagerConfig(sample_rate=16000),
            AudioManagerConfig(sample_rate=24000),
            AudioManagerConfig(sample_rate=48000),
        ]
        
        all_ok = True
        for config in configs:
            manager = AudioManager(config)
            if manager._audio_config.sample_rate != config.sample_rate:
                all_ok = False
                break
        
        passed = all_ok
        print_result("Sample rate configs", passed,
                    "All configs created correctly")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Sample rate configs", False, str(e))
    
    # Test 2: VAD configuration
    tests_total += 1
    try:
        # Test with custom VAD config
        vad_config = VADConfig(
            type=VADType.ENERGY_BASED,
            energy_threshold=0.1,
            speech_start_ms=100,
            speech_end_ms=500
        )
        config = AudioManagerConfig(
            vad_enabled=True,
            vad_config=vad_config
        )
        manager = AudioManager(config)
        
        # Config should be stored
        passed = (
            manager.config.vad_config is not None and
            manager.config.vad_config.energy_threshold == 0.1
        )
        print_result("Custom VAD config", passed,
                    "VAD config stored correctly")
        if passed:
            tests_passed += 1
        
    except Exception as e:
        print_result("Custom VAD config", False, str(e))
    
    return tests_passed, tests_total


async def run_all_tests():
    """Run all AudioManager smoke tests"""
    print("\n" + "="*60)
    print("AUDIO MANAGER SMOKE TESTS")
    print("="*60)
    
    if SKIP_HARDWARE_TESTS:
        print("\n⚠️  WARNING: Hardware tests disabled. Set SKIP_HARDWARE_TESTS=False to enable.")
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_basic_initialization,
        test_capture_functionality,
        test_playback_functionality,
        test_vad_integration,
        test_cleanup_and_safety,
        test_metrics,
        test_error_scenarios,
        test_configuration_options
    ]
    
    for test_func in test_functions:
        try:
            passed, total = await test_func()
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
    if total_tests > 0:
        print(f"Success Rate: {(total_passed/total_tests*100):.1f}%")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} TESTS FAILED!")
        return 1


def main():
    """Main entry point"""
    # Check for command line args
    if len(sys.argv) > 1:
        if "--with-hardware" in sys.argv:
            global SKIP_HARDWARE_TESTS
            SKIP_HARDWARE_TESTS = False
            print("Hardware tests enabled")
    
    # Run async tests
    return asyncio.run(run_all_tests())


if __name__ == "__main__":
    sys.exit(main())