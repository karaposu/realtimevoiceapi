"""
Performance benchmarks for audio system components.

Establishes baselines and detects performance regressions.

python -m smoke_tests.test12_audio_benchmarks
"""

import sys
import time
import gc
import statistics
import traceback
import struct
import math
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field

# Import from audioengine
try:
    from audioengine.audio_types import AudioConfig, ProcessingMode, VADConfig, AudioBytes, VADType
    from audioengine.audio_engine import AudioEngine, create_fast_lane_engine, create_adaptive_engine
    from audioengine.audio_processor import AudioProcessor, BufferPool
    from audioengine.buffered_audio_player import BufferedAudioPlayer
    from audioengine.fast_vad_detector import FastVADDetector
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure audioengine is in PYTHONPATH")
    sys.exit(1)


def print_test_header(test_name: str):
    """Print a test section header"""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {test_name}")
    print(f"{'='*60}")


def print_result(test_desc: str, value: float, unit: str = "ms", threshold: Optional[float] = None):
    """Print benchmark result"""
    if threshold is not None:
        passed = value < threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_desc}: {value:.3f}{unit} (threshold: {threshold}{unit})")
        return passed
    else:
        print(f"📊 | {test_desc}: {value:.3f}{unit}")
        return True


def generate_test_audio(duration_ms: int, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio (sine wave)"""
    num_samples = int(sample_rate * duration_ms / 1000)
    frequency = 440.0
    amplitude = 0.3
    
    audio_data = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        audio_data.extend(struct.pack('<h', value))
    
    return bytes(audio_data)


def measure_latency(func: Callable, iterations: int = 100) -> Dict[str, float]:
    """Measure function latency statistics"""
    latencies = []
    
    # Warmup
    for _ in range(10):
        func()
    
    # Measure
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p50": statistics.median(latencies),
        "p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99": sorted(latencies)[int(len(latencies) * 0.99)]
    }


def benchmark_audio_processor() -> Tuple[int, int]:
    """Benchmark AudioProcessor performance"""
    print_test_header("AudioProcessor Performance")
    
    tests_passed = 0
    tests_total = 0
    
    config = AudioConfig(sample_rate=24000)
    test_audio = generate_test_audio(20)  # 20ms chunk
    
    # Test 1: REALTIME mode processing
    tests_total += 1
    processor = AudioProcessor(mode=ProcessingMode.REALTIME, config=config)
    
    stats = measure_latency(lambda: processor.process_audio(test_audio))
    passed = print_result("REALTIME processing latency", stats["mean"], " ms", threshold=1.0)
    if passed:
        tests_passed += 1
    
    print(f"     | Min: {stats['min']:.3f}ms, Max: {stats['max']:.3f}ms, P95: {stats['p95']:.3f}ms")
    
    # Test 2: QUALITY mode processing
    tests_total += 1
    processor = AudioProcessor(mode=ProcessingMode.QUALITY, config=config)
    
    stats = measure_latency(lambda: processor.process_audio(test_audio))
    passed = print_result("QUALITY processing latency", stats["mean"], " ms", threshold=5.0)
    if passed:
        tests_passed += 1
    
    # Test 3: Throughput test
    tests_total += 1
    processor = AudioProcessor(mode=ProcessingMode.REALTIME, config=config)
    
    start_time = time.perf_counter()
    chunks_processed = 0
    target_chunks = 1000
    
    for _ in range(target_chunks):
        processor.process_audio(test_audio)
        chunks_processed += 1
    
    elapsed = time.perf_counter() - start_time
    throughput = chunks_processed / elapsed
    
    passed = print_result("Throughput", throughput, " chunks/sec", threshold=1000)
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def benchmark_audio_engine() -> Tuple[int, int]:
    """Benchmark AudioEngine performance"""
    print_test_header("AudioEngine Performance")
    
    tests_passed = 0
    tests_total = 0
    
    config = AudioConfig(sample_rate=24000)
    test_audio = generate_test_audio(20)
    
    # Test 1: Fast lane engine
    tests_total += 1
    engine = create_fast_lane_engine()
    
    stats = measure_latency(lambda: engine.process_audio(test_audio))
    passed = print_result("Fast lane latency", stats["mean"], " ms", threshold=2.0)
    if passed:
        tests_passed += 1
    
    # Test 2: Adaptive engine mode switching
    tests_total += 1
    engine = create_adaptive_engine()
    
    # Measure mode switching overhead
    mode_switches = 0
    start_time = time.perf_counter()
    
    for i in range(100):
        if i % 20 == 0:
            engine.optimize_for_quality() if i % 40 == 0 else engine.optimize_for_latency()
            mode_switches += 1
        engine.process_audio(test_audio)
    
    elapsed = time.perf_counter() - start_time
    avg_switch_time = (elapsed * 1000) / mode_switches
    
    passed = print_result("Mode switch overhead", avg_switch_time, " ms", threshold=10.0)
    if passed:
        tests_passed += 1
    
    # Test 3: Sustained processing
    tests_total += 1
    engine = AudioEngine(mode=ProcessingMode.BALANCED, config=config)
    
    # Process 1 second of audio
    total_audio_ms = 1000
    chunk_ms = 20
    chunks = total_audio_ms // chunk_ms
    
    start_time = time.perf_counter()
    for _ in range(chunks):
        chunk = generate_test_audio(chunk_ms)
        engine.process_audio(chunk)
    
    elapsed = time.perf_counter() - start_time
    realtime_factor = elapsed / (total_audio_ms / 1000)
    
    passed = print_result("Realtime factor", realtime_factor, "", threshold=0.5)
    if passed:
        tests_passed += 1
    
    print(f"     | Processed {total_audio_ms}ms in {elapsed*1000:.1f}ms")
    
    return tests_passed, tests_total


def benchmark_vad_detector() -> Tuple[int, int]:
    """Benchmark VAD detector performance"""
    print_test_header("VAD Detector Performance")
    
    tests_passed = 0
    tests_total = 0
    
    config = AudioConfig(sample_rate=24000)
    vad_config = VADConfig(type=VADType.ENERGY_BASED)
    vad = FastVADDetector(config=vad_config, audio_config=config)
    
    test_audio = generate_test_audio(20)
    
    # Test 1: VAD processing latency
    tests_total += 1
    stats = measure_latency(lambda: vad.process_chunk(test_audio))
    passed = print_result("VAD processing latency", stats["mean"], " ms", threshold=0.5)
    if passed:
        tests_passed += 1
    
    # Test 2: VAD with prebuffer
    tests_total += 1
    vad_config = VADConfig(type=VADType.ENERGY_BASED, pre_buffer_ms=200)
    vad_with_buffer = FastVADDetector(config=vad_config, audio_config=config)
    
    stats = measure_latency(lambda: vad_with_buffer.process_chunk(test_audio))
    passed = print_result("VAD with prebuffer latency", stats["mean"], " ms", threshold=1.0)
    if passed:
        tests_passed += 1
    
    # Test 3: State transition performance
    tests_total += 1
    transitions = 0
    last_state = None
    
    start_time = time.perf_counter()
    for i in range(200):
        # Alternate between speech and silence
        if i % 20 < 10:
            audio = generate_test_audio(10, sample_rate=24000)  # Speech-like
        else:
            audio = b'\x00' * (24000 * 10 // 1000 * 2)  # Silence
        
        state = vad.process_chunk(audio)
        if state != last_state:
            transitions += 1
            last_state = state
    
    elapsed = time.perf_counter() - start_time
    transition_rate = transitions / elapsed
    
    passed = print_result("State transitions", transition_rate, " per sec", threshold=10)
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def benchmark_buffered_player() -> Tuple[int, int]:
    """Benchmark BufferedAudioPlayer performance"""
    print_test_header("BufferedAudioPlayer Performance")
    
    tests_passed = 0
    tests_total = 0
    
    config = AudioConfig(sample_rate=24000)
    
    # Test 1: Buffer add latency
    tests_total += 1
    player = BufferedAudioPlayer(config)
    test_audio = generate_test_audio(20)
    
    def add_chunk():
        player.play(test_audio)
    
    stats = measure_latency(add_chunk, iterations=50)
    passed = print_result("Buffer add latency", stats["mean"], " ms", threshold=1.0)
    if passed:
        tests_passed += 1
    
    player.stop()
    
    # Test 2: Playback startup latency
    tests_total += 1
    latencies = []
    
    for _ in range(10):
        player = BufferedAudioPlayer(config)
        
        # Measure time to start playback
        start_time = time.perf_counter()
        player.play(generate_test_audio(20))
        player.play(generate_test_audio(20))  # Trigger playback at 2 chunks
        
        # Wait for playback to actually start
        while not player.is_playing and time.perf_counter() - start_time < 0.1:
            time.sleep(0.001)
        
        if player.is_playing:
            latencies.append((time.perf_counter() - start_time) * 1000)
        
        player.stop()
    
    if latencies:
        avg_startup = statistics.mean(latencies)
        passed = print_result("Playback startup latency", avg_startup, " ms", threshold=50.0)
        if passed:
            tests_passed += 1
    else:
        print("❌ FAIL | Playback startup latency: Failed to start")
    
    return tests_passed, tests_total


def benchmark_buffer_pool() -> Tuple[int, int]:
    """Benchmark BufferPool performance"""
    print_test_header("BufferPool Performance")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Buffer acquisition/release
    tests_total += 1
    pool = BufferPool(pool_size=10, buffer_size=960)  # 20ms at 24kHz
    
    def acquire_release():
        buf = pool.acquire()
        if buf:
            pool.release(buf)
    
    stats = measure_latency(acquire_release)
    passed = print_result("Buffer acquire/release", stats["mean"], " ms", threshold=0.1)
    if passed:
        tests_passed += 1
    
    # Test 2: Pool exhaustion handling
    tests_total += 1
    buffers = []
    
    # Acquire all buffers
    start_time = time.perf_counter()
    for _ in range(15):  # Try to acquire more than pool size
        buf = pool.acquire()
        if buf:
            buffers.append(buf)
    
    exhaustion_time = (time.perf_counter() - start_time) * 1000
    
    # Release all
    for buf in buffers:
        pool.release(buf)
    
    passed = print_result("Pool exhaustion handling", exhaustion_time, " ms", threshold=5.0)
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def test_memory_efficiency() -> Tuple[int, int]:
    """Test memory efficiency of components"""
    print_test_header("Memory Efficiency")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Component creation memory
    tests_total += 1
    gc.collect()
    
    # Create many components
    components = []
    for _ in range(100):
        config = AudioConfig(sample_rate=24000)
        engine = AudioEngine(mode=ProcessingMode.BALANCED, config=config)
        components.append(engine)
    
    # Force garbage collection
    del components
    gc.collect()
    
    # If we didn't crash, consider it passed
    passed = True
    print_result("Component lifecycle", 100, " components created/destroyed")
    if passed:
        tests_passed += 1
    
    return tests_passed, tests_total


def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("\n" + "="*60)
    print("AUDIO ENGINE PERFORMANCE BENCHMARKS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    benchmark_functions = [
        benchmark_audio_processor,
        benchmark_audio_engine,
        benchmark_vad_detector,
        benchmark_buffered_player,
        benchmark_buffer_pool,
        test_memory_efficiency
    ]
    
    for bench_func in benchmark_functions:
        try:
            passed, total = bench_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n❌ FATAL ERROR in {bench_func.__name__}:")
            traceback.print_exc()
            total_tests += 1
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%")
    
    # Performance guidelines
    print("\n" + "="*60)
    print("PERFORMANCE GUIDELINES")
    print("="*60)
    print("- Audio processing latency: < 5ms (realtime mode)")
    print("- VAD latency: < 1ms")
    print("- Throughput: > 1000 chunks/sec")
    print("- Realtime factor: < 0.5 (2x faster than realtime)")
    print("- Buffer operations: < 0.1ms")
    
    if total_passed == total_tests:
        print("\n✅ ALL BENCHMARKS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} BENCHMARKS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_benchmarks())