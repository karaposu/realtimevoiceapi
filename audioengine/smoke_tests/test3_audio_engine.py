# here is audioengine/smoke_tests/test_audio_engine.py

# to run python -m smoke_tests.test_audio_engine




"""
Smoke tests for audio engine functionality.

Run with: python test_audio_engine.py
"""

import sys
import time
import struct
import traceback
import signal
import math
import threading
import asyncio
from typing import List, Tuple, Optional

# Import audio engine
try:
    from audioengine.audio_engine import (
        AudioEngine, ProcessingMetrics, ProcessingStrategy,
        create_fast_lane_engine, create_big_lane_engine, create_adaptive_engine
    )
    from audioengine.audio_types import (
        AudioConfig, ProcessingMode, BufferConfig, AudioConstants
    )
    from audioengine.exceptions import AudioError
except ImportError:
    print("ERROR: Could not import audio engine. Make sure realtimevoiceapi is in PYTHONPATH")
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


def timeout_handler(signum, frame):
    """Handle test timeout"""
    raise TimeoutError("Test timed out")


def run_with_timeout(func, timeout_seconds=5):
    """Run a function with timeout"""
    if sys.platform != 'win32':
        # Use signal alarm on Unix
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            return func()
        finally:
            signal.alarm(0)
    else:
        # On Windows, just run without timeout
        return func()


def create_test_audio(duration_ms: int = 100, sample_rate: int = 24000, frequency: float = 440) -> bytes:
    """
    Create test audio data.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate
        frequency: Frequency for sine wave. 0 = silence, 440 = A4 note
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    
    if frequency == 0:
        # Create silence
        return bytes(num_samples * 2)  # 16-bit samples
    else:
        # Create sine wave
        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            value = int(10000 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', value))
        return b''.join(samples)


def test_engine_creation():
    """Test engine creation in different modes"""
    print_test_header("Engine Creation")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create REALTIME engine
    tests_total += 1
    try:
        engine = AudioEngine(mode=ProcessingMode.REALTIME)
        passed = (
            engine.mode == ProcessingMode.REALTIME and
            engine.buffer_pool is not None  # Should have buffer pool
        )
        print_result("REALTIME engine creation", passed,
                    f"mode={engine.mode.value}, has_pool={engine.buffer_pool is not None}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("REALTIME engine creation", False, str(e))
    
    # Test 2: Create QUALITY engine
    tests_total += 1
    try:
        engine = AudioEngine(mode=ProcessingMode.QUALITY)
        passed = (
            engine.mode == ProcessingMode.QUALITY and
            hasattr(engine, '_stream_buffer')
        )
        print_result("QUALITY engine creation", passed,
                    f"mode={engine.mode.value}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("QUALITY engine creation", False, str(e))
    
    # Test 3: Create BALANCED engine
    tests_total += 1
    try:
        engine = AudioEngine(mode=ProcessingMode.BALANCED)
        passed = engine.mode == ProcessingMode.BALANCED
        print_result("BALANCED engine creation", passed,
                    f"mode={engine.mode.value}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("BALANCED engine creation", False, str(e))
    
    # Test 4: Factory functions
    tests_total += 1
    try:
        fast_engine = create_fast_lane_engine(chunk_duration_ms=20)
        big_engine = create_big_lane_engine(enable_enhancement=True)
        adaptive_engine = create_adaptive_engine(latency_target_ms=50)
        
        passed = (
            fast_engine.mode == ProcessingMode.REALTIME and
            big_engine.mode == ProcessingMode.QUALITY and
            adaptive_engine.mode == ProcessingMode.BALANCED
        )
        print_result("Factory functions", passed,
                    f"Created {3} different engine types")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Factory functions", False, str(e))
    
    return tests_passed, tests_total


def test_basic_processing():
    """Test basic audio processing in each mode"""
    print_test_header("Basic Processing")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: REALTIME processing
    tests_total += 1
    try:
        engine = create_fast_lane_engine()
        test_audio = create_test_audio(20)  # 20ms chunk
        
        start = time.perf_counter()
        result = engine.process_audio(test_audio)
        elapsed = (time.perf_counter() - start) * 1000
        
        passed = (
            result is not None and
            len(result) == len(test_audio) and
            elapsed < 5.0  # Should process in < 5ms
        )
        print_result("REALTIME processing", passed,
                    f"processed {len(test_audio)} bytes in {elapsed:.2f}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("REALTIME processing", False, str(e))
    
    # Test 2: QUALITY processing
    tests_total += 1
    try:
        engine = create_big_lane_engine()
        test_audio = create_test_audio(100)  # 100ms chunk
        
        result = engine.process_audio(test_audio)
        passed = (
            result is not None and
            len(result) == len(test_audio)
        )
        print_result("QUALITY processing", passed,
                    f"processed {len(test_audio)} bytes")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("QUALITY processing", False, str(e))
    
    # Test 3: Empty audio handling
    tests_total += 1
    try:
        engine = AudioEngine()
        result = engine.process_audio(b'')
        passed = result == b''
        print_result("Empty audio handling", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Empty audio handling", False, str(e))
    
    return tests_passed, tests_total


def test_realtime_performance():
    """Test realtime performance constraints"""
    print_test_header("Realtime Performance")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Single chunk processing latency
    tests_total += 1
    try:
        engine = create_fast_lane_engine(chunk_duration_ms=20)
        chunk = create_test_audio(20)  # 20ms chunk
        
        # Warm up
        for _ in range(5):
            engine.process_audio(chunk)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            engine.process_audio(chunk)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        passed = avg_latency < 5.0 and max_latency < 10.0
        print_result("Fast lane latency", passed,
                    f"avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Fast lane latency", False, str(e))
    
    # Test 2: Verify processing time < chunk duration
    tests_total += 1
    try:
        engine = create_fast_lane_engine(chunk_duration_ms=20)
        chunk = create_test_audio(20)  # 20ms chunk
        
        # Process continuously for 1 second worth of audio
        chunks_to_process = 50  # 50 * 20ms = 1 second
        
        start = time.perf_counter()
        for _ in range(chunks_to_process):
            engine.process_audio(chunk)
        total_time = time.perf_counter() - start
        
        # Should process 1 second of audio in less than 1 second
        passed = total_time < 1.0
        print_result("Realtime throughput", passed,
                    f"Processed 1s of audio in {total_time:.3f}s")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Realtime throughput", False, str(e))
    
    # Test 3: Sustained processing
    tests_total += 1
    try:
        engine = create_fast_lane_engine(chunk_duration_ms=20)
        chunk = create_test_audio(20)
        
        # Process for 2 seconds
        sustained_chunks = 100
        start = time.perf_counter()
        
        for i in range(sustained_chunks):
            result = engine.process_audio(chunk)
            if result is None or len(result) != len(chunk):
                passed = False
                break
        else:
            elapsed = time.perf_counter() - start
            passed = elapsed < 2.0  # Should complete in < 2 seconds
        
        print_result("Sustained processing", passed,
                    f"Processed {sustained_chunks} chunks in {elapsed:.2f}s")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Sustained processing", False, str(e))
    
    return tests_passed, tests_total


def test_metrics_collection():
    """Test metrics collection and reporting"""
    print_test_header("Metrics Collection")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic metrics
    tests_total += 1
    try:
        engine = AudioEngine()
        
        # Process some audio
        for i in range(10):
            chunk = create_test_audio(20)
            engine.process_audio(chunk)
        
        metrics = engine.get_metrics()
        
        passed = (
            'total_chunks' in metrics and
            metrics['total_chunks'] == 10 and
            'avg_latency_ms' in metrics and
            metrics['avg_latency_ms'] > 0
        )
        print_result("Basic metrics", passed,
                    f"chunks={metrics.get('total_chunks')}, "
                    f"avg_latency={metrics.get('avg_latency_ms', 0):.2f}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Basic metrics", False, str(e))
    
    # Test 2: Buffer pool metrics
    tests_total += 1
    try:
        engine = create_fast_lane_engine()
        
        # Process to trigger buffer pool usage
        for _ in range(5):
            engine.process_audio(create_test_audio(20))
        
        metrics = engine.get_metrics()
        
        passed = (
            'buffer_pool' in metrics and
            'pool_size' in metrics['buffer_pool']
        )
        print_result("Buffer pool metrics", passed,
                    f"pool_size={metrics.get('buffer_pool', {}).get('pool_size', 0)}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Buffer pool metrics", False, str(e))
    
    # Test 3: Performance report
    tests_total += 1
    try:
        engine = AudioEngine()
        
        # Process some data
        chunk_size = 20 * 48  # 20ms at 24kHz * 2 bytes
        for _ in range(20):
            engine.process_audio(create_test_audio(20))
        
        report = engine.get_performance_report()
        
        passed = (
            'performance' in report and
            'realtime_capable' in report['performance']
        )
        print_result("Performance report", passed,
                    f"realtime_capable={report.get('performance', {}).get('realtime_capable')}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Performance report", False, str(e))
    
    return tests_passed, tests_total


def test_adaptive_mode():
    """Test adaptive mode switching logic"""
    print_test_header("Adaptive Mode")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Create adaptive engine
    tests_total += 1
    try:
        engine = create_adaptive_engine(latency_target_ms=30.0)
        
        passed = (
            engine.mode == ProcessingMode.BALANCED and
            engine._adaptive_threshold == 30.0
        )
        print_result("Adaptive engine creation", passed,
                    f"threshold={engine._adaptive_threshold}ms")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Adaptive engine creation", False, str(e))
    
    # Test 2: Mode optimization
    tests_total += 1
    try:
        engine = AudioEngine(mode=ProcessingMode.BALANCED)
        
        # Switch to latency optimized
        engine.optimize_for_latency()
        latency_mode = engine.mode
        
        # Switch to quality optimized
        engine.optimize_for_quality()
        quality_mode = engine.mode
        
        passed = (
            latency_mode == ProcessingMode.REALTIME and
            quality_mode == ProcessingMode.QUALITY
        )
        print_result("Mode optimization", passed,
                    f"latency={latency_mode.value}, quality={quality_mode.value}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Mode optimization", False, str(e))
    
    return tests_passed, tests_total


def test_buffer_pool_integration():
    """Test buffer pool integration for fast lane"""
    print_test_header("Buffer Pool Integration")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Buffer pool usage in realtime mode
    tests_total += 1
    try:
        config = AudioConfig(pre_allocate_buffers=True)
        buffer_config = BufferConfig(
            pre_allocate=True,
            pool_size=10,
            pool_buffer_size=4800  # 100ms at 24kHz
        )
        
        engine = AudioEngine(
            mode=ProcessingMode.REALTIME,
            config=config,
            buffer_config=buffer_config
        )
        
        # Process small chunks that fit in buffer pool
        small_chunk = create_test_audio(20)  # Should use pool
        
        for _ in range(5):
            engine.process_audio(small_chunk)
        
        metrics = engine.get_metrics()
        
        passed = (
            engine.buffer_pool is not None and
            metrics.get('buffer_pool_hit_rate', 0) > 0
        )
        print_result("Buffer pool usage", passed,
                    f"hit_rate={metrics.get('buffer_pool_hit_rate', 0):.2f}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Buffer pool usage", False, str(e))
    
    # Test 2: Large chunk handling (exceeds pool buffer)
    tests_total += 1
    try:
        engine = create_fast_lane_engine()
        
        # Create chunk larger than typical pool buffer
        large_chunk = create_test_audio(200)  # 200ms
        
        result = engine.process_audio(large_chunk)
        
        passed = (
            result is not None and
            len(result) == len(large_chunk)
        )
        print_result("Large chunk handling", passed,
                    f"Processed {len(large_chunk)} bytes")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Large chunk handling", False, str(e))
    
    return tests_passed, tests_total


def test_concurrent_processing():
    """Test concurrent chunk processing without blocking"""
    print_test_header("Concurrent Processing")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Thread-safe processing
    tests_total += 1
    try:
        engine = create_fast_lane_engine()
        chunk = create_test_audio(20)
        results = []
        errors = []
        
        def process_chunk(engine, chunk, results, errors):
            try:
                result = engine.process_audio(chunk)
                results.append(len(result))
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=process_chunk,
                args=(engine, chunk, results, errors)
            )
            threads.append(t)
        
        # Start all threads
        start = time.perf_counter()
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=1.0)
        
        elapsed = time.perf_counter() - start
        
        passed = (
            len(errors) == 0 and
            len(results) == 10 and
            all(r == len(chunk) for r in results) and
            elapsed < 0.5  # Should complete quickly
        )
        print_result("Thread-safe processing", passed,
                    f"Processed {len(results)} chunks in {elapsed:.3f}s")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Thread-safe processing", False, str(e))
    
    # Test 2: Non-blocking operation
    tests_total += 1
    try:
        engine = create_fast_lane_engine()
        
        # Process many chunks rapidly
        chunk = create_test_audio(20)
        start = time.perf_counter()
        
        # This should not block
        for _ in range(50):
            result = engine.process_audio(chunk)
            if result is None:
                passed = False
                break
        else:
            elapsed = time.perf_counter() - start
            # 50 chunks * 20ms = 1 second of audio
            # Should process much faster than realtime
            passed = elapsed < 0.5
        
        print_result("Non-blocking operation", passed,
                    f"50 chunks in {elapsed:.3f}s")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Non-blocking operation", False, str(e))
    
    return tests_passed, tests_total


def test_resource_cleanup():
    """Test resource cleanup and memory management"""
    print_test_header("Resource Cleanup")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Basic cleanup
    tests_total += 1
    try:
        engine = create_fast_lane_engine()
        
        # Process some data
        for _ in range(10):
            engine.process_audio(create_test_audio(20))
        
        # Clean up
        engine.cleanup()
        
        passed = (
            engine.buffer_pool is None and
            len(engine._pre_processors) == 0 and
            len(engine._post_processors) == 0
        )
        print_result("Basic cleanup", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Basic cleanup", False, str(e))
    
    # Test 2: Metrics reset
    tests_total += 1
    try:
        engine = AudioEngine()
        
        # Generate some metrics
        for _ in range(5):
            engine.process_audio(create_test_audio(20))
        
        # Reset metrics
        engine.reset_metrics()
        metrics = engine.get_metrics()
        
        passed = (
            metrics['total_chunks'] == 0 and
            metrics['total_bytes'] == 0
        )
        print_result("Metrics reset", passed)
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Metrics reset", False, str(e))
    
    # Test 3: Multiple cleanup calls
    tests_total += 1
    try:
        engine = create_big_lane_engine()
        
        # Multiple cleanup should not error
        engine.cleanup()
        engine.cleanup()
        
        passed = True
        print_result("Multiple cleanup calls", passed)
        tests_passed += 1
    except Exception as e:
        print_result("Multiple cleanup calls", False, str(e))
    
    return tests_passed, tests_total


def test_pipeline_processing():
    """Test pipeline with pre/post processors"""
    print_test_header("Pipeline Processing")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Pre/post processor chain
    tests_total += 1
    try:
        engine = AudioEngine()
        
        # Add simple processors
        pre_called = []
        post_called = []
        
        def pre_processor(audio):
            pre_called.append(True)
            return audio
        
        def post_processor(audio):
            post_called.append(True)
            return audio
        
        engine.add_pre_processor(pre_processor)
        engine.add_post_processor(post_processor)
        
        # Process through pipeline
        chunk = create_test_audio(20)
        result = engine.process_with_pipeline(chunk)
        
        passed = (
            len(pre_called) == 1 and
            len(post_called) == 1 and
            result is not None
        )
        print_result("Pre/post processor chain", passed,
                    f"pre={len(pre_called)}, post={len(post_called)}")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Pre/post processor chain", False, str(e))
    
    # Test 2: Stream processing
    tests_total += 1
    try:
        engine = AudioEngine()
        
        # Create multiple chunks
        chunks = [create_test_audio(20) for _ in range(5)]
        callback_count = 0
        
        def chunk_callback(processed, index):
            nonlocal callback_count
            callback_count += 1
        
        # Process stream
        result = engine.process_stream(chunks, chunk_callback)
        
        passed = (
            callback_count == 5 and
            len(result) == sum(len(c) for c in chunks)
        )
        print_result("Stream processing", passed,
                    f"Processed {callback_count} chunks")
        if passed:
            tests_passed += 1
    except Exception as e:
        print_result("Stream processing", False, str(e))
    
    return tests_passed, tests_total


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("AUDIO ENGINE SMOKE TESTS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    test_functions = [
        test_engine_creation,
        test_basic_processing,
        test_realtime_performance,
        test_metrics_collection,
        test_adaptive_mode,
        test_buffer_pool_integration,
        test_concurrent_processing,
        test_resource_cleanup,
        test_pipeline_processing
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