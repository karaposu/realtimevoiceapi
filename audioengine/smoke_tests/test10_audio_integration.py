"""
Integration tests for complete audio system.

Tests complex scenarios with all components working together.

python -m smoke_tests.test10_audio_integration
"""

import sys
import time
import threading
import traceback
import math
import struct
import gc
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

# Import from audioengine
try:
    from audioengine.audio_types import AudioConfig, ProcessingMode, VADConfig, AudioBytes, VADType
    from audioengine.audio_engine import AudioEngine, create_fast_lane_engine, create_adaptive_engine
    from audioengine.audio_manager import AudioManager, AudioManagerConfig
    from audioengine.buffered_audio_player import BufferedAudioPlayer
    from audioengine.fast_vad_detector import FastVADDetector
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


def generate_test_audio(duration_ms: int, frequency: float = 440.0, 
                       sample_rate: int = 24000, amplitude: float = 0.3) -> AudioBytes:
    """Generate test audio (sine wave)"""
    num_samples = int(sample_rate * duration_ms / 1000)
    audio_data = bytearray()
    
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        audio_data.extend(struct.pack('<h', value))
    
    return bytes(audio_data)


def generate_silence(duration_ms: int, sample_rate: int = 24000) -> AudioBytes:
    """Generate silence"""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b'\x00' * (num_samples * 2)


@dataclass
class IntegrationMetrics:
    """Metrics for integration testing"""
    chunks_processed: int = 0
    vad_speech_detected: int = 0
    vad_silence_detected: int = 0
    playback_completed: int = 0
    mode_switches: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_duration(self) -> float:
        return time.time() - self.start_time


def test_engine_with_audio_manager() -> Tuple[int, int]:
    """Test AudioEngine integrated with AudioManager"""
    print_test_header("AudioEngine + AudioManager Integration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create integrated system
    tests_total += 1
    try:
        # Create audio manager with VAD
        config = AudioManagerConfig(
            sample_rate=24000,
            vad_enabled=True,
            vad_config=VADConfig(
                type=VADType.ENERGY_BASED,
                energy_threshold=0.02
            )
        )
        manager = AudioManager(config)
        
        # For testing, manually create the VAD since initialize is async
        from audioengine.fast_vad_detector import FastVADDetector
        manager._vad = FastVADDetector(
            config=config.vad_config,
            audio_config=AudioConfig(sample_rate=config.sample_rate)
        )
        
        # Get audio config from manager config
        audio_config = AudioConfig(sample_rate=config.sample_rate)
        
        # Create audio engine
        engine = AudioEngine(
            mode=ProcessingMode.BALANCED,
            config=audio_config
        )
        
        passed = manager is not None and engine is not None
        print_result("System creation", passed,
                    "AudioManager + AudioEngine created")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("System creation", False, str(e))
    
    # Test 2: Process audio through both components
    tests_total += 1
    try:
        metrics = IntegrationMetrics()
        
        # Process test audio
        speech = generate_test_audio(100, amplitude=0.5)
        silence = generate_silence(100)
        
        # Process through engine
        processed_speech = engine.process_audio(speech)
        processed_silence = engine.process_audio(silence)
        
        # Check VAD through manager
        vad_speech = manager.process_vad(processed_speech)
        vad_silence = manager.process_vad(processed_silence)
        
        passed = (processed_speech is not None and
                 processed_silence is not None and
                 vad_speech is not None and
                 vad_silence is not None)
        
        print_result("Audio processing chain", passed,
                    f"Speech: {vad_speech}, Silence: {vad_silence}")
        if passed:
            tests_passed += 1
            
    except Exception as e:
        print_result("Audio processing chain", False, str(e))
    
    # Test 3: Mode switching
    tests_total += 1
    try:
        # Switch engine to fast lane
        engine.optimize_for_latency()
        metrics.mode_switches += 1
        
        # Process in realtime mode
        audio = generate_test_audio(20)
        start = time.time()
        processed = engine.process_audio(audio)
        latency = (time.time() - start) * 1000
        
        passed = processed is not None and latency < 5  # Should be very fast
        print_result("Fast lane processing", passed,
                    f"Processing latency: {latency:.2f}ms")
        if passed:
            tests_passed += 1
            
    except Exception as e:
        print_result("Fast lane processing", False, str(e))
    
    return tests_passed, tests_total


def test_buffered_playback_integration() -> Tuple[int, int]:
    """Test BufferedAudioPlayer with AudioEngine"""
    print_test_header("BufferedAudioPlayer + AudioEngine Integration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Process and play audio
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        engine = AudioEngine(mode=ProcessingMode.QUALITY, config=config)
        player = BufferedAudioPlayer(config)
        
        # Track playback completion
        playback_completed = threading.Event()
        
        def on_complete():
            playback_completed.set()
        
        player.set_completion_callback(on_complete)
        
        # Generate and process audio chunks
        for i in range(5):
            # Generate audio with different frequencies
            freq = 440 + i * 100
            audio = generate_test_audio(50, frequency=freq)
            
            # Process through engine
            processed = engine.process_audio(audio)
            
            # Play processed audio
            player.play(processed)
        
        player.mark_complete()
        
        # Wait for playback
        passed = playback_completed.wait(timeout=3.0)
        
        print_result("Process and playback chain", passed,
                    f"Played {player.chunks_played} processed chunks")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Process and playback chain", False, str(e))
    
    # Test 2: Adaptive processing with playback
    tests_total += 1
    try:
        # Create adaptive engine
        engine = create_adaptive_engine(
            sample_rate=config.sample_rate,
            latency_target_ms=30.0
        )
        player = BufferedAudioPlayer(config)
        
        # Process with varying latencies
        modes_used = set()
        
        for i in range(10):
            audio = generate_test_audio(20)
            
            # Simulate varying processing load
            if i % 3 == 0:
                time.sleep(0.01)  # Add delay
            
            processed = engine.process_audio(audio)
            player.play(processed)
            
            # Track mode
            modes_used.add(engine.mode)
        
        passed = len(modes_used) > 1  # Should have switched modes
        print_result("Adaptive processing", passed,
                    f"Used modes: {modes_used}")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Adaptive processing", False, str(e))
    
    return tests_passed, tests_total


def test_vad_integration() -> Tuple[int, int]:
    """Test VAD integration with other components"""
    print_test_header("VAD Integration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: VAD with AudioEngine
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        engine = AudioEngine(mode=ProcessingMode.BALANCED, config=config)
        
        vad_config = VADConfig(
            type=VADType.ENERGY_BASED,
            energy_threshold=0.02,
            speech_start_ms=50,
            speech_end_ms=200
        )
        vad = FastVADDetector(
            config=vad_config,
            audio_config=config
        )
        
        # Track speech events
        speech_events = []
        
        def on_speech_start():
            speech_events.append('start')
        
        def on_speech_end():
            speech_events.append('end')
        
        vad.on_speech_start = on_speech_start
        vad.on_speech_end = on_speech_end
        
        # Process alternating speech and silence
        for i in range(4):
            if i % 2 == 0:
                audio = generate_test_audio(100, amplitude=0.5)
            else:
                audio = generate_silence(250)
            
            # Process through engine then VAD
            processed = engine.process_audio(audio)
            vad_state = vad.process_chunk(processed)
        
        passed = len(speech_events) >= 2  # Should have start/end events
        print_result("VAD event detection", passed,
                    f"Events: {speech_events}")
        if passed:
            tests_passed += 1
            
    except Exception as e:
        print_result("VAD event detection", False, str(e))
    
    # Test 2: VAD prebuffer retrieval
    tests_total += 1
    try:
        # Create VAD with prebuffer
        vad_config = VADConfig(
            type=VADType.ENERGY_BASED,
            pre_buffer_ms=200  # Use correct parameter name
        )
        vad = FastVADDetector(config=vad_config, audio_config=config)
        
        # Feed silence then speech
        for i in range(5):
            silence = generate_silence(50)
            vad.process_chunk(silence)
        
        # Now speech
        speech = generate_test_audio(50, amplitude=0.5)
        state = vad.process_chunk(speech)
        
        # Get prebuffer
        prebuffer = vad.get_pre_buffer()
        
        passed = prebuffer is not None and len(prebuffer) > 0
        print_result("VAD prebuffer", passed,
                    f"Retrieved {len(prebuffer) if prebuffer else 0} bytes")
        if passed:
            tests_passed += 1
            
    except Exception as e:
        print_result("VAD prebuffer", False, str(e))
    
    return tests_passed, tests_total


def test_multi_component_scenario() -> Tuple[int, int]:
    """Test realistic scenario with multiple components"""
    print_test_header("Multi-Component Scenario")
    
    tests_passed = 0
    tests_total = 0
    
    # Test: Simulated voice conversation
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        metrics = IntegrationMetrics()
        
        # Create components
        engine = create_fast_lane_engine(sample_rate=config.sample_rate)
        player = BufferedAudioPlayer(config)
        vad = FastVADDetector(
            config=VADConfig(type=VADType.ENERGY_BASED),
            audio_config=config
        )
        
        # Simulate conversation
        conversation_chunks = [
            ("user_speech", generate_test_audio(200, frequency=300, amplitude=0.4)),
            ("silence", generate_silence(100)),
            ("user_speech", generate_test_audio(150, frequency=350, amplitude=0.5)),
            ("silence", generate_silence(300)),
            ("ai_response", generate_test_audio(300, frequency=500, amplitude=0.3)),
            ("silence", generate_silence(200)),
        ]
        
        for chunk_type, audio in conversation_chunks:
            # Process through engine
            processed = engine.process_audio(audio)
            metrics.chunks_processed += 1
            
            # Check VAD
            vad_state = vad.process_chunk(processed)
            # vad_state is a string like 'speech', 'silence', etc.
            if vad_state in ['speech', 'speech_starting', 'speech_continuing']:
                metrics.vad_speech_detected += 1
            else:
                metrics.vad_silence_detected += 1
            
            # Play AI responses
            if chunk_type == "ai_response":
                player.play(processed)
        
        # Mark playback complete
        player.mark_complete()
        
        # Brief wait for playback
        time.sleep(0.5)
        
        passed = (metrics.chunks_processed == len(conversation_chunks) and
                 metrics.vad_speech_detected >= 2 and
                 metrics.vad_silence_detected >= 2)
        
        print_result("Conversation simulation", passed,
                    f"Processed: {metrics.chunks_processed}, "
                    f"Speech: {metrics.vad_speech_detected}, "
                    f"Silence: {metrics.vad_silence_detected}")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Conversation simulation", False, str(e))
    
    return tests_passed, tests_total


def test_resource_cleanup() -> Tuple[int, int]:
    """Test proper resource cleanup"""
    print_test_header("Resource Cleanup")
    
    tests_passed = 0
    tests_total = 0
    
    # Test: Multiple create/destroy cycles
    tests_total += 1
    try:
        # Force garbage collection for baseline
        gc.collect()
        
        # Run multiple cycles
        for i in range(5):
            config = AudioConfig(sample_rate=24000)
            
            # Create components
            engine = AudioEngine(mode=ProcessingMode.BALANCED, config=config)
            player = BufferedAudioPlayer(config)
            manager = AudioManager(AudioManagerConfig(sample_rate=config.sample_rate))
            
            # Use them briefly
            audio = generate_test_audio(20)
            processed = engine.process_audio(audio)
            player.play(processed)
            manager.process_vad(audio)
            
            # Cleanup
            player.stop()
            del engine, player, manager
            
            if i % 2 == 0:
                gc.collect()
        
        # Final collection
        gc.collect()
        
        passed = True  # If we got here without crashes
        print_result("Resource lifecycle", passed,
                    "Completed 5 create/destroy cycles")
        tests_passed += 1
    except Exception as e:
        print_result("Resource lifecycle", False, str(e))
    
    return tests_passed, tests_total


def test_error_scenarios() -> Tuple[int, int]:
    """Test error handling across components"""
    print_test_header("Error Scenarios")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Invalid audio data
    tests_total += 1
    try:
        config = AudioConfig(sample_rate=24000)
        engine = AudioEngine(mode=ProcessingMode.BALANCED, config=config)
        player = BufferedAudioPlayer(config)
        
        # Try to process invalid data
        errors = 0
        
        # Empty data
        try:
            engine.process_audio(b'')
        except:
            errors += 1
        
        # Odd byte count
        try:
            engine.process_audio(b'x')
        except:
            errors += 1
        
        # None
        try:
            engine.process_audio(None)
        except:
            errors += 1
        
        passed = errors >= 2  # Should catch at least some errors
        print_result("Invalid data handling", passed,
                    f"Caught {errors}/3 error cases")
        if passed:
            tests_passed += 1
            
        player.stop()
    except Exception as e:
        print_result("Invalid data handling", False, str(e))
    
    # Test 2: Component interaction with errors
    tests_total += 1
    try:
        engine = AudioEngine(mode=ProcessingMode.BALANCED, config=config)
        vad = FastVADDetector(audio_config=config)
        
        # Process very short audio that might cause issues
        tiny_audio = generate_test_audio(1)  # 1ms - very short
        
        handled_gracefully = True
        try:
            processed = engine.process_audio(tiny_audio)
            vad_state = vad.process_chunk(processed if processed else tiny_audio)
        except Exception as e:
            # Some errors are acceptable for edge cases
            handled_gracefully = "too short" in str(e).lower()
        
        print_result("Edge case handling", handled_gracefully,
                    "Handled 1ms audio chunk")
        if handled_gracefully:
            tests_passed += 1
            
    except Exception as e:
        print_result("Edge case handling", False, str(e))
    
    return tests_passed, tests_total


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("AUDIO ENGINE INTEGRATION TESTS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_engine_with_audio_manager,
        test_buffered_playback_integration,
        test_vad_integration,
        test_multi_component_scenario,
        test_resource_cleanup,
        test_error_scenarios
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