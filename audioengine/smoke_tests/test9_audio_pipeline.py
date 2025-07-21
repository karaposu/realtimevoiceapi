#!/usr/bin/env python3
"""
Smoke test for Audio Pipeline
Tests end-to-end audio streaming pipeline
"""


# NOTE: not sure if this should be included in audio engine....maybe not 

import time
import threading
import queue
import statistics
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from collections import deque
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtimevoiceapi.core.audio_types import AudioConfig, ProcessingMode
from realtimevoiceapi.core.audio_engine import AudioEngine, create_fast_lane_engine


# Test utilities
class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_true(self, condition, message):
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  ✗ {message}")
    
    def assert_equal(self, actual, expected, message):
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(f"{message} (expected {expected}, got {actual})")
            print(f"  ✗ {message} (expected {expected}, got {actual})")
    
    def assert_less_than(self, actual, threshold, message):
        if actual < threshold:
            self.passed += 1
            print(f"  ✓ {message} ({actual:.2f} < {threshold})")
        else:
            self.failed += 1
            self.errors.append(f"{message} ({actual:.2f} >= {threshold})")
            print(f"  ✗ {message} ({actual:.2f} >= {threshold})")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'='*60}\n")
        return self.failed == 0


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance"""
    chunks_processed: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    jitter_measurements: List[float] = field(default_factory=list)
    queue_depths: List[int] = field(default_factory=list)
    errors: int = 0
    backpressure_events: int = 0
    start_time: float = field(default_factory=time.time)
    
    def add_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)
        self.total_latency_ms += latency_ms
        self.chunks_processed += 1
    
    def add_jitter(self, jitter_ms: float):
        self.jitter_measurements.append(jitter_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "chunks_processed": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "max_latency_ms": 0,
                "jitter_ms": 0,
                "throughput_chunks_per_sec": 0
            }
        
        elapsed = time.time() - self.start_time
        return {
            "chunks_processed": self.chunks_processed,
            "avg_latency_ms": statistics.mean(self.latencies),
            "p95_latency_ms": sorted(self.latencies)[int(len(self.latencies) * 0.95)],
            "max_latency_ms": max(self.latencies),
            "jitter_ms": statistics.stdev(self.jitter_measurements) if len(self.jitter_measurements) > 1 else 0,
            "throughput_chunks_per_sec": self.chunks_processed / elapsed if elapsed > 0 else 0,
            "errors": self.errors,
            "backpressure_events": self.backpressure_events,
            "avg_queue_depth": statistics.mean(self.queue_depths) if self.queue_depths else 0,
            "max_queue_depth": max(self.queue_depths) if self.queue_depths else 0
        }


class MockAudioCapture:
    """Mock audio capture that generates chunks at realtime rate"""
    
    def __init__(self, chunk_duration_ms: int = 20, sample_rate: int = 24000):
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000) * 2  # 16-bit
        self.is_capturing = False
        self.chunk_interval = chunk_duration_ms / 1000.0
        self.capture_thread = None
        self.output_queue = queue.Queue(maxsize=100)
        self.chunks_generated = 0
        
    def start_capture(self):
        """Start generating audio chunks"""
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def stop_capture(self):
        """Stop capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
    
    def _capture_loop(self):
        """Generate chunks at realtime rate"""
        next_chunk_time = time.time()
        
        while self.is_capturing:
            # Generate chunk
            chunk = self._generate_chunk()
            timestamp = time.time()
            
            try:
                self.output_queue.put((chunk, timestamp), timeout=0.001)
                self.chunks_generated += 1
            except queue.Full:
                pass  # Drop chunk if queue full
            
            # Sleep until next chunk time
            next_chunk_time += self.chunk_interval
            sleep_time = next_chunk_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _generate_chunk(self) -> bytes:
        """Generate test audio chunk"""
        # Simple tone for testing
        import math
        samples = []
        frequency = 440  # A4
        
        num_samples = self.chunk_size // 2
        for i in range(num_samples):
            t = (self.chunks_generated * num_samples + i) / self.sample_rate
            value = int(16383 * math.sin(2 * math.pi * frequency * t))
            samples.extend(value.to_bytes(2, byteorder='little', signed=True))
        
        return bytes(samples)


class MockNetworkSender:
    """Mock network sender with configurable behavior"""
    
    def __init__(self, latency_ms: float = 5.0, jitter_ms: float = 2.0):
        self.base_latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.chunks_sent = 0
        self.bytes_sent = 0
        self.last_send_time = None
        self.send_times = deque(maxlen=100)
        self.error_rate = 0.0  # Probability of error
        
    def send(self, audio_data: bytes) -> bool:
        """Simulate network send with latency"""
        # Simulate variable latency
        latency = self.base_latency_ms + random.uniform(-self.jitter_ms, self.jitter_ms)
        time.sleep(latency / 1000.0)
        
        # Simulate errors
        if random.random() < self.error_rate:
            raise Exception("Simulated network error")
        
        self.chunks_sent += 1
        self.bytes_sent += len(audio_data)
        
        current_time = time.time()
        if self.last_send_time:
            self.send_times.append(current_time - self.last_send_time)
        self.last_send_time = current_time
        
        return True
    
    def get_stats(self):
        """Get sender statistics"""
        return {
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "avg_interval_ms": statistics.mean(self.send_times) * 1000 if self.send_times else 0
        }


class AudioPipeline:
    """Complete audio pipeline: capture → process → send"""
    
    def __init__(
        self,
        capture: MockAudioCapture,
        engine: AudioEngine,
        sender: MockNetworkSender,
        max_queue_depth: int = 50
    ):
        self.capture = capture
        self.engine = engine
        self.sender = sender
        self.max_queue_depth = max_queue_depth
        
        # Queues
        self.process_queue = queue.Queue(maxsize=max_queue_depth)
        self.send_queue = queue.Queue(maxsize=max_queue_depth)
        
        # State
        self.is_running = False
        self.threads = []
        
        # Metrics
        self.metrics = PipelineMetrics()
        self.last_chunk_time = None
        
    def start(self):
        """Start the pipeline"""
        self.is_running = True
        self.metrics = PipelineMetrics()
        
        # Start capture
        self.capture.start_capture()
        
        # Start pipeline threads
        threads = [
            threading.Thread(target=self._capture_thread, name="PipelineCapture"),
            threading.Thread(target=self._process_thread, name="PipelineProcess"),
            threading.Thread(target=self._send_thread, name="PipelineSend"),
            threading.Thread(target=self._monitor_thread, name="PipelineMonitor")
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
            self.threads.append(t)
    
    def stop(self):
        """Stop the pipeline"""
        self.is_running = False
        self.capture.stop_capture()
        
        # Wait for threads
        for t in self.threads:
            t.join(timeout=1.0)
        
        self.threads.clear()
    
    def _capture_thread(self):
        """Move chunks from capture to process queue"""
        while self.is_running:
            try:
                # Get from capture
                chunk, timestamp = self.capture.output_queue.get(timeout=0.1)
                
                # Track jitter
                if self.last_chunk_time:
                    actual_interval = timestamp - self.last_chunk_time
                    expected_interval = self.capture.chunk_duration_ms / 1000.0
                    jitter = abs(actual_interval - expected_interval) * 1000
                    self.metrics.add_jitter(jitter)
                
                self.last_chunk_time = timestamp
                
                # Put in process queue
                try:
                    self.process_queue.put((chunk, timestamp), timeout=0.001)
                except queue.Full:
                    self.metrics.backpressure_events += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.metrics.errors += 1
    
    def _process_thread(self):
        """Process audio chunks"""
        while self.is_running:
            try:
                # Get chunk to process
                chunk, capture_time = self.process_queue.get(timeout=0.1)
                
                # Process through engine
                process_start = time.time()
                processed = self.engine.process_audio(chunk)
                process_time = time.time() - process_start
                
                # Track latency
                self.metrics.add_latency(process_time * 1000)
                
                # Send to output queue
                try:
                    self.send_queue.put((processed, capture_time), timeout=0.001)
                except queue.Full:
                    self.metrics.backpressure_events += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.metrics.errors += 1
    
    def _send_thread(self):
        """Send processed chunks"""
        while self.is_running:
            try:
                # Get chunk to send
                chunk, capture_time = self.send_queue.get(timeout=0.1)
                
                # Send
                self.sender.send(chunk)
                
                # Track end-to-end latency
                total_latency = (time.time() - capture_time) * 1000
                
            except queue.Empty:
                continue
            except Exception as e:
                self.metrics.errors += 1
    
    def _monitor_thread(self):
        """Monitor queue depths"""
        while self.is_running:
            # Record queue depths
            self.metrics.queue_depths.append(self.process_queue.qsize())
            self.metrics.queue_depths.append(self.send_queue.qsize())
            
            # Keep only recent measurements
            if len(self.metrics.queue_depths) > 1000:
                self.metrics.queue_depths = self.metrics.queue_depths[-1000:]
            
            time.sleep(0.1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        stats = self.metrics.get_stats()
        stats.update({
            "capture_chunks": self.capture.chunks_generated,
            "sender_stats": self.sender.get_stats(),
            "engine_metrics": self.engine.get_metrics()
        })
        return stats


def test_pipeline_creation():
    """Test pipeline creation and configuration"""
    print("\n1. Testing Pipeline Creation and Configuration")
    print("-" * 40)
    
    result = TestResult()
    
    # Create components
    capture = MockAudioCapture(chunk_duration_ms=20)
    engine = create_fast_lane_engine(sample_rate=24000, chunk_duration_ms=20)
    sender = MockNetworkSender(latency_ms=5.0)
    
    # Create pipeline
    pipeline = AudioPipeline(capture, engine, sender)
    
    result.assert_equal(pipeline.is_running, False, "Pipeline initially not running")
    result.assert_equal(pipeline.process_queue.maxsize, 50, "Process queue configured")
    result.assert_equal(pipeline.send_queue.maxsize, 50, "Send queue configured")
    
    # Test configuration
    result.assert_equal(engine.mode, ProcessingMode.REALTIME, "Engine in realtime mode")
    result.assert_equal(capture.chunk_duration_ms, 20, "Capture chunk duration set")
    
    return result


def test_continuous_streaming():
    """Test continuous streaming flow"""
    print("\n2. Testing Continuous Streaming")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline
    capture = MockAudioCapture(chunk_duration_ms=20)
    engine = create_fast_lane_engine()
    sender = MockNetworkSender(latency_ms=2.0)
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Start pipeline
    pipeline.start()
    result.assert_true(pipeline.is_running, "Pipeline started")
    
    # Let it run for 2 seconds
    time.sleep(2.0)
    
    # Check metrics
    metrics = pipeline.get_metrics()
    
    # Should process ~100 chunks in 2 seconds (50 per second)
    expected_chunks = 90  # Allow some margin
    result.assert_true(
        metrics["chunks_processed"] > expected_chunks,
        f"Processed enough chunks: {metrics['chunks_processed']} > {expected_chunks}"
    )
    
    # Check sender
    result.assert_true(
        metrics["sender_stats"]["chunks_sent"] > expected_chunks,
        f"Sent enough chunks: {metrics['sender_stats']['chunks_sent']}"
    )
    
    # Check latency
    result.assert_less_than(
        metrics["avg_latency_ms"],
        5.0,
        "Average processing latency < 5ms"
    )
    
    # Stop pipeline
    pipeline.stop()
    result.assert_equal(pipeline.is_running, False, "Pipeline stopped")
    
    return result


def test_backpressure_handling():
    """Test backpressure and queue management"""
    print("\n3. Testing Backpressure Handling")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline with slow sender
    capture = MockAudioCapture(chunk_duration_ms=20)
    engine = create_fast_lane_engine()
    sender = MockNetworkSender(latency_ms=50.0)  # Very slow sender
    pipeline = AudioPipeline(capture, engine, sender, max_queue_depth=10)
    
    # Start pipeline
    pipeline.start()
    
    # Run for 1 second
    time.sleep(1.0)
    
    # Check for backpressure
    metrics = pipeline.get_metrics()
    
    result.assert_true(
        metrics["backpressure_events"] > 0,
        f"Backpressure detected: {metrics['backpressure_events']} events"
    )
    
    # Check queue depths stayed bounded
    result.assert_less_than(
        metrics["max_queue_depth"],
        15,  # Should stay near max_queue_depth of 10
        "Queue depth bounded"
    )
    
    # Stop pipeline
    pipeline.stop()
    
    # Test with fast sender (no backpressure)
    sender_fast = MockNetworkSender(latency_ms=0.5)
    pipeline_fast = AudioPipeline(capture, engine, sender_fast)
    
    pipeline_fast.start()
    time.sleep(1.0)
    
    metrics_fast = pipeline_fast.get_metrics()
    result.assert_equal(
        metrics_fast["backpressure_events"],
        0,
        "No backpressure with fast sender"
    )
    
    pipeline_fast.stop()
    
    return result


def test_pipeline_state_management():
    """Test pipeline state transitions"""
    print("\n4. Testing Pipeline State Management")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline
    capture = MockAudioCapture()
    engine = create_fast_lane_engine()
    sender = MockNetworkSender()
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Test start/stop cycles
    for i in range(3):
        # Start
        pipeline.start()
        result.assert_true(pipeline.is_running, f"Cycle {i+1}: Pipeline running")
        result.assert_true(capture.is_capturing, f"Cycle {i+1}: Capture active")
        
        # Process some chunks
        time.sleep(0.2)
        
        # Stop
        pipeline.stop()
        result.assert_equal(pipeline.is_running, False, f"Cycle {i+1}: Pipeline stopped")
        result.assert_equal(capture.is_capturing, False, f"Cycle {i+1}: Capture stopped")
        
        # Brief pause
        time.sleep(0.1)
    
    # Test metrics reset
    pipeline.start()
    time.sleep(0.5)
    metrics1 = pipeline.get_metrics()
    chunks1 = metrics1["chunks_processed"]
    
    pipeline.stop()
    pipeline.start()  # Restart should reset metrics
    time.sleep(0.1)
    
    metrics2 = pipeline.get_metrics()
    result.assert_true(
        metrics2["chunks_processed"] < chunks1,
        "Metrics reset on restart"
    )
    
    pipeline.stop()
    
    return result


def test_error_propagation():
    """Test error handling and recovery"""
    print("\n5. Testing Error Propagation and Recovery")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline with error-prone sender
    capture = MockAudioCapture()
    engine = create_fast_lane_engine()
    sender = MockNetworkSender()
    sender.error_rate = 0.1  # 10% error rate
    
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Start pipeline
    pipeline.start()
    
    # Run for 2 seconds
    time.sleep(2.0)
    
    # Check error handling
    metrics = pipeline.get_metrics()
    
    result.assert_true(
        metrics["errors"] > 0,
        f"Errors detected and counted: {metrics['errors']}"
    )
    
    # Pipeline should still be running despite errors
    result.assert_true(pipeline.is_running, "Pipeline continues despite errors")
    
    # Check that we still processed chunks
    result.assert_true(
        metrics["chunks_processed"] > 50,
        f"Still processing despite errors: {metrics['chunks_processed']} chunks"
    )
    
    pipeline.stop()
    
    return result


def test_sustained_performance():
    """Test performance over sustained period"""
    print("\n6. Testing Sustained Streaming Performance")
    print("-" * 40)
    
    result = TestResult()
    
    # Create optimized pipeline
    capture = MockAudioCapture(chunk_duration_ms=20)
    engine = create_fast_lane_engine()
    sender = MockNetworkSender(latency_ms=1.0, jitter_ms=0.5)
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Run for 10 seconds
    print("  Running 10-second performance test...")
    pipeline.start()
    
    # Monitor performance over time
    performance_samples = []
    for i in range(10):
        time.sleep(1.0)
        metrics = pipeline.get_metrics()
        performance_samples.append({
            "latency": metrics["avg_latency_ms"],
            "throughput": metrics["throughput_chunks_per_sec"],
            "queue_depth": metrics["avg_queue_depth"]
        })
        print(f"    Second {i+1}: {metrics['chunks_processed']} chunks, "
              f"avg latency: {metrics['avg_latency_ms']:.2f}ms")
    
    pipeline.stop()
    
    # Analyze sustained performance
    final_metrics = pipeline.get_metrics()
    
    # Check throughput (should be ~50 chunks/sec for 20ms chunks)
    expected_throughput = 48  # Allow some margin
    result.assert_true(
        final_metrics["throughput_chunks_per_sec"] > expected_throughput,
        f"Sustained throughput: {final_metrics['throughput_chunks_per_sec']:.1f} chunks/sec"
    )
    
    # Check latency stayed low
    result.assert_less_than(
        final_metrics["avg_latency_ms"],
        2.0,
        "Sustained low latency"
    )
    
    result.assert_less_than(
        final_metrics["p95_latency_ms"],
        5.0,
        "P95 latency acceptable"
    )
    
    # Check queue depths stayed bounded
    result.assert_less_than(
        final_metrics["avg_queue_depth"],
        5.0,
        "Average queue depth stayed low"
    )
    
    return result


def test_realtime_throughput():
    """Test pipeline maintains realtime throughput"""
    print("\n7. Testing Realtime Throughput Maintenance")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline
    capture = MockAudioCapture(chunk_duration_ms=20)
    engine = create_fast_lane_engine()
    sender = MockNetworkSender(latency_ms=0.5)
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Measure precise timing
    pipeline.start()
    
    # Wait for steady state
    time.sleep(0.5)
    
    # Measure over 5 seconds
    start_time = time.time()
    start_metrics = pipeline.get_metrics()
    
    time.sleep(5.0)
    
    end_time = time.time()
    end_metrics = pipeline.get_metrics()
    
    pipeline.stop()
    
    # Calculate actual throughput
    elapsed = end_time - start_time
    chunks_processed = end_metrics["chunks_processed"] - start_metrics["chunks_processed"]
    actual_throughput = chunks_processed / elapsed
    
    # Expected: 50 chunks/sec for 20ms chunks
    expected_throughput = 1000.0 / capture.chunk_duration_ms
    throughput_ratio = actual_throughput / expected_throughput
    
    result.assert_true(
        0.95 < throughput_ratio < 1.05,
        f"Realtime throughput ratio: {throughput_ratio:.3f} "
        f"({actual_throughput:.1f}/{expected_throughput:.1f} chunks/sec)"
    )
    
    return result


def test_bounded_queue_depths():
    """Test that queue depths stay bounded"""
    print("\n8. Testing Bounded Queue Depths")
    print("-" * 40)
    
    result = TestResult()
    
    # Test different scenarios
    scenarios = [
        ("Balanced", 5.0, 2.0, 50),      # Normal operation
        ("Slow sender", 25.0, 5.0, 20),   # Slow network
        ("Fast sender", 0.1, 0.0, 100),   # Very fast
    ]
    
    for name, latency, jitter, max_depth in scenarios:
        print(f"\n  Testing {name} scenario...")
        
        capture = MockAudioCapture(chunk_duration_ms=20)
        engine = create_fast_lane_engine()
        sender = MockNetworkSender(latency_ms=latency, jitter_ms=jitter)
        pipeline = AudioPipeline(capture, engine, sender, max_queue_depth=max_depth)
        
        pipeline.start()
        time.sleep(3.0)
        pipeline.stop()
        
        metrics = pipeline.get_metrics()
        
        # Queue should not exceed max_depth + small margin
        result.assert_less_than(
            metrics["max_queue_depth"],
            max_depth + 5,
            f"{name}: Max queue depth bounded"
        )
        
        # Average should be much less than max
        result.assert_less_than(
            metrics["avg_queue_depth"],
            max_depth * 0.5,
            f"{name}: Average queue depth reasonable"
        )
    
    return result


def test_jitter_measurements():
    """Test jitter in chunk delivery timing"""
    print("\n9. Testing Chunk Delivery Jitter")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline with minimal processing
    capture = MockAudioCapture(chunk_duration_ms=20)
    engine = create_fast_lane_engine()
    sender = MockNetworkSender(latency_ms=1.0, jitter_ms=0.1)  # Low jitter
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Run pipeline
    pipeline.start()
    time.sleep(5.0)
    pipeline.stop()
    
    metrics = pipeline.get_metrics()
    
    # Check jitter is low
    result.assert_less_than(
        metrics["jitter_ms"],
        2.0,
        "Chunk arrival jitter low"
    )
    
    # Test with high jitter sender
    sender_jittery = MockNetworkSender(latency_ms=10.0, jitter_ms=5.0)
    pipeline_jittery = AudioPipeline(capture, engine, sender_jittery)
    
    pipeline_jittery.start()
    time.sleep(3.0)
    pipeline_jittery.stop()
    
    metrics_jittery = pipeline_jittery.get_metrics()
    
    # Jitter should be higher but pipeline should handle it
    result.assert_true(
        metrics_jittery["jitter_ms"] > metrics["jitter_ms"],
        f"Higher jitter detected: {metrics_jittery['jitter_ms']:.2f}ms"
    )
    
    # But throughput should still be maintained
    result.assert_true(
        metrics_jittery["throughput_chunks_per_sec"] > 45,
        f"Throughput maintained despite jitter: {metrics_jittery['throughput_chunks_per_sec']:.1f}"
    )
    
    return result


def test_pipeline_recovery():
    """Test pipeline recovery from various conditions"""
    print("\n10. Testing Pipeline Recovery")
    print("-" * 40)
    
    result = TestResult()
    
    # Create pipeline
    capture = MockAudioCapture()
    engine = create_fast_lane_engine()
    sender = MockNetworkSender()
    pipeline = AudioPipeline(capture, engine, sender)
    
    # Test recovery from queue overflow
    pipeline.max_queue_depth = 5  # Very small queue
    pipeline.start()
    
    # Create temporary slowdown
    original_latency = sender.base_latency_ms
    sender.base_latency_ms = 100.0  # Very slow
    
    time.sleep(0.5)  # Let queues fill
    
    # Restore normal speed
    sender.base_latency_ms = original_latency
    
    time.sleep(2.0)  # Let it recover
    
    metrics = pipeline.get_metrics()
    
    # Should have experienced backpressure
    result.assert_true(
        metrics["backpressure_events"] > 0,
        "Experienced backpressure during slowdown"
    )
    
    # But should have recovered
    current_queue_depth = pipeline.process_queue.qsize()
    result.assert_less_than(
        current_queue_depth,
        3,
        f"Queue recovered to normal depth: {current_queue_depth}"
    )
    
    pipeline.stop()
    
    return result


def main():
    """Run all tests"""
    print("=" * 60)
    print("Audio Pipeline Smoke Tests")
    print("=" * 60)
    
    all_results = []
    
    # Run all tests
    tests = [
        test_pipeline_creation,
        test_continuous_streaming,
        test_backpressure_handling,
        test_pipeline_state_management,
        test_error_propagation,
        test_sustained_performance,
        test_realtime_throughput,
        test_bounded_queue_depths,
        test_jitter_measurements,
        test_pipeline_recovery
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            
            # Create failed result
            result = TestResult()
            result.failed = 1
            result.errors.append(f"Test crashed: {e}")
            all_results.append(result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed
    
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_failed > 0:
        print(f"\n❌ {total_failed} tests failed")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())