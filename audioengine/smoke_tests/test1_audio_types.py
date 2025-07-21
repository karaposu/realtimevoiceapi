# here is realtimevoiceapi/smoke_tests/audio/test_audio_types.py


"""
Test Audio Types and Configurations

Tests core audio type definitions, configurations, and calculations.
Ensures type safety and proper validation of audio parameters.

python -m smoke_tests.test_audio_types
"""

import sys
import traceback
from typing import List, Tuple

# Import the audio types
try:
    from audioengine.audio_types import (
        AudioConfig, AudioFormat, AudioQuality, ProcessingMode,
        VADType, VADConfig, AudioMetadata, BufferConfig,
        AudioConstants, get_optimal_chunk_size, validate_audio_config
    )
except ImportError:
    print("ERROR: Could not import audio types. Make sure realtimevoiceapi is in PYTHONPATH")
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


def test_audio_config_creation():
    """Test AudioConfig creation and defaults"""
    print_test_header("AudioConfig Creation")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Default configuration
    tests_total += 1
    try:
        config = AudioConfig()
        passed = (
            config.sample_rate == 24000 and
            config.channels == 1 and
            config.bit_depth == 16 and
            config.format == AudioFormat.PCM16
        )
        print_result("Default AudioConfig creation", passed, 
                    f"sample_rate={config.sample_rate}, channels={config.channels}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default AudioConfig creation", False, str(e))
    
    # Test 2: Custom configuration
    tests_total += 1
    try:
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            chunk_duration_ms=50
        )
        passed = (
            config.sample_rate == 48000 and
            config.channels == 2 and
            config.chunk_duration_ms == 50
        )
        print_result("Custom AudioConfig creation", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Custom AudioConfig creation", False, str(e))
    
    # Test 3: Computed properties
    tests_total += 1
    try:
        config = AudioConfig()
        frame_size = config.frame_size
        bytes_per_second = config.bytes_per_second
        expected_frame_size = 1 * 2  # mono * 16-bit
        expected_bps = 24000 * 2
        
        passed = (
            frame_size == expected_frame_size and
            bytes_per_second == expected_bps
        )
        print_result("Computed properties", passed,
                    f"frame_size={frame_size}, bytes_per_second={bytes_per_second}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Computed properties", False, str(e))
    
    # Test 4: Chunk size calculations
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        chunk_100ms = config.chunk_size_bytes(100)
        expected = int(100 * 24000 * 2 / 1000)  # 100ms * sample_rate * 2 bytes
        
        passed = chunk_100ms == expected
        print_result("Chunk size calculation", passed,
                    f"100ms chunk = {chunk_100ms} bytes (expected {expected})")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Chunk size calculation", False, str(e))
    
    # Test 5: Duration from bytes
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        duration = config.duration_from_bytes(4800)
        expected = 100.0  # 4800 bytes = 100ms at 24kHz mono 16-bit
        
        passed = abs(duration - expected) < 0.1
        print_result("Duration from bytes", passed,
                    f"4800 bytes = {duration:.1f}ms (expected {expected}ms)")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Duration from bytes", False, str(e))
    
    return tests_passed, tests_total


def test_audio_formats():
    """Test AudioFormat enum functionality"""
    print_test_header("AudioFormat Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Format properties
    tests_total += 1
    try:
        pcm16 = AudioFormat.PCM16
        g711_ulaw = AudioFormat.G711_ULAW
        
        passed = (
            pcm16.bytes_per_sample == 2 and
            g711_ulaw.bytes_per_sample == 1 and
            not pcm16.requires_compression and
            g711_ulaw.requires_compression
        )
        print_result("Format properties", passed,
                    f"PCM16: {pcm16.bytes_per_sample} bytes, "
                    f"G711: {g711_ulaw.bytes_per_sample} bytes")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Format properties", False, str(e))
    
    # Test 2: API format conversion
    tests_total += 1
    try:
        format_str = AudioFormat.PCM16.to_api_format()
        passed = format_str == "pcm16"
        print_result("API format conversion", passed, f"'{format_str}'")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("API format conversion", False, str(e))
    
    return tests_passed, tests_total


def test_processing_modes():
    """Test ProcessingMode configurations"""
    print_test_header("ProcessingMode Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Mode properties
    tests_total += 1
    try:
        realtime = ProcessingMode.REALTIME
        quality = ProcessingMode.QUALITY
        
        passed = (
            realtime.buffer_size_ms == 10 and
            quality.buffer_size_ms == 200 and
            not realtime.allows_numpy and
            quality.allows_numpy and
            realtime.max_latency_ms == 20.0 and
            quality.max_latency_ms == 500.0
        )
        print_result("Mode properties", passed,
                    f"Realtime buffer: {realtime.buffer_size_ms}ms, "
                    f"Quality buffer: {quality.buffer_size_ms}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Mode properties", False, str(e))
    
    return tests_passed, tests_total


def test_vad_config():
    """Test VAD configuration"""
    print_test_header("VAD Configuration Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Default VAD config
    tests_total += 1
    try:
        vad = VADConfig()
        passed = (
            vad.type == VADType.ENERGY_BASED and
            0 <= vad.energy_threshold <= 1 and
            vad.speech_start_ms == 100 and
            vad.speech_end_ms == 500
        )
        print_result("Default VAD config", passed,
                    f"type={vad.type.value}, threshold={vad.energy_threshold}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default VAD config", False, str(e))
    
    # Test 2: VAD type properties
    tests_total += 1
    try:
        energy_vad = VADType.ENERGY_BASED
        server_vad = VADType.SERVER_VAD
        
        passed = (
            energy_vad.is_local and
            not server_vad.is_local and
            server_vad.is_api_based
        )
        print_result("VAD type properties", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("VAD type properties", False, str(e))
    
    # Test 3: VAD config to API dict
    tests_total += 1
    try:
        vad = VADConfig(type=VADType.SERVER_VAD, energy_threshold=0.5)
        api_dict = vad.to_api_dict()
        
        passed = (
            api_dict.get("type") == "server_vad" and
            "threshold" in api_dict and
            api_dict.get("create_response") == True
        )
        print_result("VAD to API dict", passed, f"keys: {list(api_dict.keys())}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("VAD to API dict", False, str(e))
    
    return tests_passed, tests_total


def test_buffer_config():
    """Test BufferConfig"""
    print_test_header("BufferConfig Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Default buffer config
    tests_total += 1
    try:
        buffer = BufferConfig()
        passed = (
            buffer.max_size_bytes == 1024 * 1024 and
            buffer.overflow_strategy == "drop_oldest" and
            buffer.use_circular and
            buffer.validate()
        )
        print_result("Default buffer config", passed,
                    f"max_size={buffer.max_size_bytes}, "
                    f"strategy={buffer.overflow_strategy}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Default buffer config", False, str(e))
    
    # Test 2: Buffer validation
    tests_total += 1
    try:
        invalid_buffer = BufferConfig(max_size_bytes=-1)
        passed = not invalid_buffer.validate()
        print_result("Buffer validation", passed, "Correctly rejected invalid config")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Buffer validation", False, str(e))
    
    return tests_passed, tests_total


def test_audio_constants():
    """Test audio constants"""
    print_test_header("Audio Constants Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: OpenAI specific constants
    tests_total += 1
    try:
        passed = (
            AudioConstants.OPENAI_SAMPLE_RATE == 24000 and
            AudioConstants.OPENAI_CHANNELS == 1 and
            AudioConstants.OPENAI_FORMAT == AudioFormat.PCM16
        )
        print_result("OpenAI constants", passed,
                    f"sample_rate={AudioConstants.OPENAI_SAMPLE_RATE}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("OpenAI constants", False, str(e))
    
    # Test 2: Limits
    tests_total += 1
    try:
        passed = (
            AudioConstants.MAX_AUDIO_SIZE_BYTES == 25 * 1024 * 1024 and
            AudioConstants.MIN_CHUNK_DURATION_MS == 10 and
            AudioConstants.MAX_CHUNK_DURATION_MS == 1000
        )
        print_result("Audio limits", passed,
                    f"max_size={AudioConstants.MAX_AUDIO_SIZE_BYTES / 1024 / 1024}MB")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Audio limits", False, str(e))
    
    # Test 3: Fast lane constants
    tests_total += 1
    try:
        passed = (
            AudioConstants.FAST_LANE_CHUNK_MS == 20 and
            AudioConstants.FAST_LANE_BUFFER_COUNT == 10 and
            AudioConstants.FAST_LANE_MAX_LATENCY_MS == 50
        )
        print_result("Fast lane constants", passed,
                    f"chunk={AudioConstants.FAST_LANE_CHUNK_MS}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Fast lane constants", False, str(e))
    
    return tests_passed, tests_total


def test_utility_functions():
    """Test utility functions"""
    print_test_header("Utility Functions Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Get optimal chunk size
    tests_total += 1
    try:
        realtime_chunk = get_optimal_chunk_size(ProcessingMode.REALTIME)
        quality_chunk = get_optimal_chunk_size(ProcessingMode.QUALITY)
        
        # Expected: 20ms for realtime, 200ms for quality at 24kHz
        expected_realtime = 24000 * 2 * 20 // 1000  # 960 bytes
        expected_quality = 24000 * 2 * 200 // 1000   # 9600 bytes
        
        passed = (
            realtime_chunk == expected_realtime and
            quality_chunk == expected_quality
        )
        print_result("Optimal chunk size", passed,
                    f"realtime={realtime_chunk}, quality={quality_chunk}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Optimal chunk size", False, str(e))
    
    # Test 2: Validate audio config
    tests_total += 1
    try:
        valid_config = AudioConfig(sample_rate=24000)
        invalid_config = AudioConfig(sample_rate=12345)  # Unsupported rate
        
        valid_ok, valid_msg = validate_audio_config(valid_config)
        invalid_ok, invalid_msg = validate_audio_config(invalid_config)
        
        passed = valid_ok and not invalid_ok and invalid_msg is not None
        print_result("Config validation", passed,
                    f"invalid_msg='{invalid_msg}'")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Config validation", False, str(e))
    
    return tests_passed, tests_total


def test_audio_quality():
    """Test AudioQuality presets"""
    print_test_header("AudioQuality Presets Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Quality presets
    tests_total += 1
    try:
        low = AudioQuality.LOW.to_config()
        standard = AudioQuality.STANDARD.to_config()
        high = AudioQuality.HIGH.to_config()
        
        passed = (
            low.sample_rate == 16000 and
            standard.sample_rate == 24000 and
            high.sample_rate == 48000
        )
        print_result("Quality presets", passed,
                    f"low={low.sample_rate}Hz, standard={standard.sample_rate}Hz, "
                    f"high={high.sample_rate}Hz")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Quality presets", False, str(e))
    
    return tests_passed, tests_total


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("AUDIO TYPES SMOKE TESTS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_audio_config_creation,
        test_audio_formats,
        test_processing_modes,
        test_vad_config,
        test_buffer_config,
        test_audio_constants,
        test_utility_functions,
        test_audio_quality
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