#!/usr/bin/env python3
"""
Test realtime voice streaming scenarios for AI conversations
Focuses on bidirectional streaming, interrupts, and network conditions
"""

import time
import threading
import asyncio
import queue
import statistics
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from collections import deque
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtimevoiceapi.core.audio_types import AudioConfig, ProcessingMode
from realtimevoiceapi.core.audio_engine import AudioEngine, create_fast_lane_engine
from realtimevoiceapi.audio.buffered_audio_player import BufferedAudioPlayer


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
    
    def assert_less_than(self, actual, threshold, message):
        if actual < threshold:
            self.passed += 1
            print(f"  ✓ {message} ({actual:.2f} < {threshold})")
        else:
            self.failed += 1
            self.errors.append(f"{message} ({actual:.2f} >= {threshold})")
            print(f"  ✗ {message} ({actual:.2f} >= {threshold})")
    
    def assert_equal(self, actual, expected, message):
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(f"{message} (expected {expected}, got {actual})")
            print(f"  ✗ {message} (expected {expected}, got {actual})")
    
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
class StreamingMetrics:
    """Metrics for voice streaming performance"""
    # Timing
    chunk_processing_times: List[float] = field(default_factory=list)
    network_times: List[float] = field(default_factory=list)
    end_to_end_latencies: List[float] = field(default_factory=list)
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Counts
    chunks_sent: int = 0
    chunks_received: int = 0
    chunks_played: int = 0
    interrupts_triggered: int = 0
    audio_dropouts: int = 0
    
    # Interrupt timing
    interrupt_latencies: List[float] = field(default_factory=list)
    audio_flush_times: List[float] = field(default_factory=list)
    
    # Network
    packet_loss_events: int = 0
    jitter_buffer_underruns: int = 0
    jitter_buffer_overruns: int = 0
    
    def add_chunk_timing(self, process_time: float, network_time: float):
        self.chunk_processing_times.append(process_time)
        self.network_times.append(network_time)
        self.end_to_end_latencies.append(process_time + network_time)
        self.latency_samples.append(process_time + network_time)
    
    def add_interrupt_timing(self, interrupt_latency: float, flush_time: float):
        self.interrupt_latencies.append(interrupt_latency)
        self.audio_flush_times.append(flush_time)
        self.interrupts_triggered += 1
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.chunk_processing_times:
            return {"error": "No data collected"}
        
        return {
            "chunks": {
                "sent": self.chunks_sent,
                "received": self.chunks_received,
                "played": self.chunks_played
            },
            "timing": {
                "avg_processing_ms": statistics.mean(self.chunk_processing_times) * 1000,
                "avg_network_ms": statistics.mean(self.network_times) * 1000,
                "avg_e2e_ms": statistics.mean(self.end_to_end_latencies) * 1000,
                "max_e2e_ms": max(self.end_to_end_latencies) * 1000,
                "p95_e2e_ms": sorted(self.end_to_end_latencies)[int(len(self.end_to_end_latencies) * 0.95)] * 1000
            },
            "interrupts": {
                "count": self.interrupts_triggered,
                "avg_latency_ms": statistics.mean(self.interrupt_latencies) if self.interrupt_latencies else 0,
                "avg_flush_time_ms": statistics.mean(self.audio_flush_times) if self.audio_flush_times else 0
            },
            "quality": {
                "dropouts": self.audio_dropouts,
                "packet_loss": self.packet_loss_events,
                "jitter_underruns": self.jitter_buffer_underruns,
                "jitter_overruns": self.jitter_buffer_overruns
            }
        }
    
    def check_latency_stability(self) -> Tuple[bool, float]:
        """Check if latency is stable (not accumulating)"""
        if len(self.latency_samples) < 100:
            return True, 0.0
        
        # Split samples into first half and second half
        mid = len(self.latency_samples) // 2
        first_half = list(self.latency_samples)[:mid]
        second_half = list(self.latency_samples)[mid:]
        
        # Compare averages
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        # Latency should not increase by more than 10%
        increase = (avg_second - avg_first) / avg_first
        is_stable = increase < 0.1
        
        return is_stable, increase * 100


class MockNetworkTransport:
    """Simulates network transport with configurable behavior"""
    
    def __init__(self, base_latency_ms: float = 30.0, jitter_ms: float = 10.0):
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.packet_loss_rate = 0.0
        self.bandwidth_limit_kbps = 1000  # 1 Mbps default
        
        # Queues
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        
        # State
        self.is_running = False
        self.transport_thread = None
        
        # Metrics
        self.bytes_sent = 0
        self.packets_dropped = 0
    
    def start(self):
        """Start network transport simulation"""
        self.is_running = True
        self.transport_thread = threading.Thread(target=self._transport_loop)
        self.transport_thread.daemon = True
        self.transport_thread.start()
    
    def stop(self):
        """Stop transport"""
        self.is_running = False
        if self.transport_thread:
            self.transport_thread.join(timeout=1.0)
    
    def send(self, data: bytes, timestamp: float) -> bool:
        """Send data through network"""
        try:
            self.send_queue.put((data, timestamp), timeout=0.01)
            return True
        except queue.Full:
            return False
    
    def receive(self, timeout: float = 0.01) -> Optional[Tuple[bytes, float]]:
        """Receive data from network"""
        try:
            return self.receive_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _transport_loop(self):
        """Simulate network transport"""
        while self.is_running:
            try:
                # Get data to send
                data, timestamp = self.send_queue.get(timeout=0.01)
                
                # Simulate packet loss
                if random.random() < self.packet_loss_rate:
                    self.packets_dropped += 1
                    continue
                
                # Simulate network latency
                latency = self.base_latency_ms + random.uniform(-self.jitter_ms, self.jitter_ms)
                latency_seconds = latency / 1000.0
                
                # Simulate bandwidth limit
                bandwidth_delay = (len(data) * 8) / (self.bandwidth_limit_kbps * 1000)
                
                total_delay = latency_seconds + bandwidth_delay
                
                # Schedule delivery
                def deliver():
                    self.receive_queue.put((data, timestamp + total_delay))
                    self.bytes_sent += len(data)
                
                timer = threading.Timer(total_delay, deliver)
                timer.daemon = True
                timer.start()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transport error: {e}")


class VoiceStreamingSystem:
    """Complete voice streaming system for AI conversations"""
    
    def __init__(self, sample_rate: int = 24000, chunk_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000) * 2  # 16-bit
        
        # Components
        self.audio_engine = create_fast_lane_engine(sample_rate, chunk_duration_ms)
        self.network = MockNetworkTransport()
        self.ai_player = BufferedAudioPlayer(AudioConfig(sample_rate=sample_rate))
        
        # State
        self.is_streaming = False
        self.is_ai_speaking = False
        self.user_is_speaking = False
        
        # Threads
        self.capture_thread = None
        self.network_send_thread = None
        self.network_receive_thread = None
        self.ai_playback_thread = None
        
        # Metrics
        self.metrics = StreamingMetrics()
        
        # Interrupt handling
        self.interrupt_event = threading.Event()
        self.last_interrupt_time = None
        
        # Audio generation
        self.audio_generator_phase = 0
    
    def start(self):
        """Start voice streaming system"""
        self.is_streaming = True
        self.metrics = StreamingMetrics()
        
        # Start network
        self.network.start()
        
        # Start threads
        threads = [
            ("Capture", self._capture_loop),
            ("NetworkSend", self._network_send_loop),
            ("NetworkReceive", self._network_receive_loop),
            ("AIPlayback", self._ai_playback_loop)
        ]
        
        for name, target in threads:
            thread = threading.Thread(target=target, name=f"Voice{name}")
            thread.daemon = True
            thread.start()
    
    def stop(self):
        """Stop voice streaming"""
        self.is_streaming = False
        self.network.stop()
        
        # Stop AI playback
        if self.ai_player:
            self.ai_player.stop(force=True)
    
    def trigger_interrupt(self) -> float:
        """Trigger user interrupt of AI speech"""
        interrupt_start = time.time()
        
        # Set interrupt flag
        self.interrupt_event.set()
        self.user_is_speaking = True
        
        # Stop AI playback immediately
        if self.is_ai_speaking and self.ai_player:
            self.ai_player.stop(force=True)
            flush_time = time.time() - interrupt_start
            
            # Record interrupt timing
            if self.last_interrupt_time:
                interrupt_latency = interrupt_start - self.last_interrupt_time
                self.metrics.add_interrupt_timing(interrupt_latency, flush_time)
            
            self.is_ai_speaking = False
        
        self.last_interrupt_time = interrupt_start
        return time.time() - interrupt_start
    
    def simulate_ai_response(self, duration_seconds: float = 2.0):
        """Simulate AI starting to speak"""
        self.is_ai_speaking = True
        self.user_is_speaking = False
        self.interrupt_event.clear()
        
        # Generate AI audio chunks
        num_chunks = int(duration_seconds * 1000 / self.chunk_duration_ms)
        
        for i in range(num_chunks):
            if self.interrupt_event.is_set():
                break
            
            # Generate audio chunk
            chunk = self._generate_audio_chunk(frequency=660)  # Higher pitch for AI
            
            # Send through network
            self.network.send(chunk, time.time())
            
            # Simulate real-time generation
            time.sleep(self.chunk_duration_ms / 1000.0)
    
    def _capture_loop(self):
        """Simulate audio capture"""
        chunk_interval = self.chunk_duration_ms / 1000.0
        next_chunk_time = time.time()
        
        while self.is_streaming:
            # Generate user audio
            if self.user_is_speaking:
                chunk = self._generate_audio_chunk(frequency=440)  # User voice
                
                # Process through engine
                start_time = time.time()
                processed = self.audio_engine.process_audio(chunk)
                process_time = time.time() - start_time
                
                # Send to network
                self.network.send(processed, start_time)
                self.metrics.chunks_sent += 1
                
                # Track processing time
                if process_time > chunk_interval:
                    self.metrics.audio_dropouts += 1
            
            # Maintain real-time rate
            next_chunk_time += chunk_interval
            sleep_time = next_chunk_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're falling behind
                next_chunk_time = time.time()
    
    def _network_send_loop(self):
        """Handle outgoing network traffic"""
        while self.is_streaming:
            # This is handled by capture loop directly
            time.sleep(0.1)
    
    def _network_receive_loop(self):
        """Handle incoming AI audio"""
        while self.is_streaming:
            # Receive from network
            result = self.network.receive(timeout=0.01)
            if result:
                data, timestamp = result
                
                # Track network delay
                network_delay = time.time() - timestamp
                
                # Play AI audio if not interrupted
                if not self.interrupt_event.is_set() and self.is_ai_speaking:
                    self.ai_player.play(data)
                    self.metrics.chunks_received += 1
                    
                    # Track timing
                    self.metrics.add_chunk_timing(0, network_delay)
    
    def _ai_playback_loop(self):
        """Monitor AI playback"""
        while self.is_streaming:
            if self.ai_player and self.ai_player.is_playing:
                self.is_ai_speaking = True
                self.metrics.chunks_played = self.ai_player.chunks_played
            else:
                self.is_ai_speaking = False
            
            time.sleep(0.05)
    
    def _generate_audio_chunk(self, frequency: int = 440) -> bytes:
        """Generate test audio chunk"""
        import math
        samples = []
        num_samples = self.chunk_size // 2
        
        for i in range(num_samples):
            t = (self.audio_generator_phase + i) / self.sample_rate
            value = int(16383 * math.sin(2 * math.pi * frequency * t))
            samples.extend(value.to_bytes(2, byteorder='little', signed=True))
        
        self.audio_generator_phase += num_samples
        return bytes(samples)


async def test_realtime_throughput():
    """Test system maintains realtime throughput without delay accumulation"""
    print("\n1. Testing Realtime Throughput (60 seconds)")
    print("-" * 40)
    
    result = TestResult()
    
    # Create system
    system = VoiceStreamingSystem(sample_rate=24000, chunk_duration_ms=20)
    
    # Configure ideal network
    system.network.base_latency_ms = 20
    system.network.jitter_ms = 5
    
    # Start streaming
    system.start()
    system.user_is_speaking = True  # Continuous user speech
    
    print("  Running 60-second throughput test...")
    
    # Monitor for 60 seconds
    start_time = time.time()
    checkpoints = []
    
    for i in range(12):  # Check every 5 seconds
        await asyncio.sleep(5.0)
        
        # Get metrics
        stats = system.metrics.get_stats()
        elapsed = time.time() - start_time
        expected_chunks = int(elapsed * 1000 / system.chunk_duration_ms)
        
        # Check throughput
        sent_ratio = system.metrics.chunks_sent / expected_chunks if expected_chunks > 0 else 0
        
        checkpoints.append({
            "time": elapsed,
            "sent": system.metrics.chunks_sent,
            "expected": expected_chunks,
            "ratio": sent_ratio,
            "avg_latency": stats["timing"]["avg_e2e_ms"]
        })
        
        print(f"    {elapsed:.0f}s: {system.metrics.chunks_sent}/{expected_chunks} chunks, "
              f"ratio: {sent_ratio:.2%}, latency: {stats['timing']['avg_e2e_ms']:.1f}ms")
    
    system.stop()
    
    # Analyze results
    final_stats = system.metrics.get_stats()
    
    # Check throughput maintained
    final_ratio = checkpoints[-1]["ratio"]
    result.assert_true(
        0.95 < final_ratio < 1.05,
        f"Maintained realtime throughput: {final_ratio:.1%}"
    )
    
    # Check latency didn't accumulate
    is_stable, increase = system.metrics.check_latency_stability()
    result.assert_true(
        is_stable,
        f"Latency remained stable: {increase:.1f}% change"
    )
    
    # Check chunk timing
    chunk_duration_seconds = system.chunk_duration_ms / 1000.0
    avg_total_time = final_stats["timing"]["avg_processing_ms"] / 1000 + final_stats["timing"]["avg_network_ms"] / 1000
    
    result.assert_less_than(
        avg_total_time,
        chunk_duration_seconds,
        f"Processing + network time < chunk duration"
    )
    
    # No dropouts
    result.assert_equal(
        system.metrics.audio_dropouts,
        0,
        "No audio dropouts"
    )
    
    return result


async def test_bidirectional_streaming():
    """Test simultaneous capture and playback"""
    print("\n2. Testing Bidirectional Streaming (Full Duplex)")
    print("-" * 40)
    
    result = TestResult()
    
    system = VoiceStreamingSystem()
    system.start()
    
    # Test patterns
    print("  Testing alternating speech patterns...")
    
    # Pattern 1: User speaks, then AI responds
    system.user_is_speaking = True
    await asyncio.sleep(2.0)
    
    system.user_is_speaking = False
    system.simulate_ai_response(2.0)
    await asyncio.sleep(2.5)
    
    # Pattern 2: Overlapping speech (both speaking)
    print("  Testing overlapping speech...")
    
    # Start AI speaking
    ai_thread = threading.Thread(target=lambda: system.simulate_ai_response(3.0))
    ai_thread.start()
    
    # User starts speaking after 1 second
    await asyncio.sleep(1.0)
    system.user_is_speaking = True
    
    # Both speaking for 2 seconds
    await asyncio.sleep(2.0)
    system.user_is_speaking = False
    
    ai_thread.join()
    
    # Check metrics
    stats = system.metrics.get_stats()
    
    # Both directions should have processed chunks
    result.assert_true(
        stats["chunks"]["sent"] > 100,
        f"User audio sent: {stats['chunks']['sent']} chunks"
    )
    
    result.assert_true(
        stats["chunks"]["received"] > 100,
        f"AI audio received: {stats['chunks']['received']} chunks"
    )
    
    # No quality issues
    result.assert_equal(
        stats["quality"]["dropouts"],
        0,
        "No audio dropouts during full duplex"
    )
    
    # Latency should stay low even during full duplex
    result.assert_less_than(
        stats["timing"]["avg_e2e_ms"],
        50.0,
        "Low latency maintained during full duplex"
    )
    
    system.stop()
    return result


async def test_interrupt_scenarios():
    """Test various interrupt scenarios"""
    print("\n3. Testing Interrupt Scenarios")
    print("-" * 40)
    
    result = TestResult()
    
    system = VoiceStreamingSystem()
    system.start()
    
    # Test 1: Mid-response interrupt
    print("  Testing mid-response interrupt...")
    
    # Start AI speaking
    ai_thread = threading.Thread(target=lambda: system.simulate_ai_response(3.0))
    ai_thread.start()
    
    # Wait for AI to start
    await asyncio.sleep(0.5)
    result.assert_true(system.is_ai_speaking, "AI is speaking")
    
    # User interrupts
    interrupt_time = system.trigger_interrupt()
    
    # Check interrupt was fast
    result.assert_less_than(
        interrupt_time * 1000,
        50.0,
        "Interrupt completed in < 50ms"
    )
    
    # Verify AI stopped
    await asyncio.sleep(0.1)
    result.assert_equal(system.is_ai_speaking, False, "AI stopped speaking")
    
    ai_thread.join()
    
    # Test 2: Rapid interrupts
    print("  Testing rapid successive interrupts...")
    
    interrupt_times = []
    for i in range(5):
        # Start AI
        ai_thread = threading.Thread(target=lambda: system.simulate_ai_response(1.0))
        ai_thread.start()
        
        # Quick interrupt
        await asyncio.sleep(0.2)
        interrupt_time = system.trigger_interrupt()
        interrupt_times.append(interrupt_time)
        
        ai_thread.join()
        await asyncio.sleep(0.1)
    
    # All interrupts should be fast
    avg_interrupt_time = statistics.mean(interrupt_times) * 1000
    result.assert_less_than(
        avg_interrupt_time,
        50.0,
        f"Average interrupt time for rapid interrupts"
    )
    
    # Test 3: Interrupt with queue flushing
    print("  Testing interrupt with buffer flushing...")
    
    # Fill AI playback buffer
    for i in range(20):
        chunk = system._generate_audio_chunk(660)
        system.ai_player.play(chunk)
    
    initial_buffer_size = len(system.ai_player.buffer)
    result.assert_true(
        initial_buffer_size > 10,
        f"AI buffer filled: {initial_buffer_size} chunks"
    )
    
    # Interrupt and check buffer cleared
    system.trigger_interrupt()
    
    result.assert_equal(
        len(system.ai_player.buffer),
        0,
        "AI buffer cleared on interrupt"
    )
    
    # Check interrupt metrics
    stats = system.metrics.get_stats()
    result.assert_true(
        stats["interrupts"]["count"] >= 6,
        f"All interrupts tracked: {stats['interrupts']['count']}"
    )
    
    system.stop()
    return result


async def test_network_simulation():
    """Test behavior under various network conditions"""
    print("\n4. Testing Network Simulation")
    print("-" * 40)
    
    result = TestResult()
    
    system = VoiceStreamingSystem()
    
    # Test 1: Variable latency
    print("  Testing variable network latency...")
    
    system.network.base_latency_ms = 100
    system.network.jitter_ms = 50  # High jitter
    
    system.start()
    system.user_is_speaking = True
    
    # Run for 5 seconds
    await asyncio.sleep(5.0)
    
    stats = system.metrics.get_stats()
    
    # System should handle high latency
    result.assert_true(
        stats["timing"]["avg_network_ms"] > 50,
        f"Network latency simulated: {stats['timing']['avg_network_ms']:.1f}ms"
    )
    
    # But still maintain streaming
    result.assert_true(
        stats["chunks"]["sent"] > 200,
        f"Streaming continued with high latency: {stats['chunks']['sent']} chunks"
    )
    
    system.stop()
    
    # Test 2: Packet loss
    print("  Testing packet loss...")
    
    system = VoiceStreamingSystem()
    system.network.packet_loss_rate = 0.05  # 5% loss
    
    system.start()
    
    # Simulate AI speaking with packet loss
    ai_thread = threading.Thread(target=lambda: system.simulate_ai_response(3.0))
    ai_thread.start()
    
    await asyncio.sleep(3.5)
    ai_thread.join()
    
    # Check packet loss handling
    expected_chunks = int(3.0 * 1000 / system.chunk_duration_ms)
    received_ratio = system.metrics.chunks_received / expected_chunks
    
    result.assert_true(
        0.90 < received_ratio < 0.97,  # Should lose ~5%
        f"Packet loss handled: {(1-received_ratio)*100:.1f}% loss"
    )
    
    system.stop()
    
    # Test 3: Bandwidth limitation
    print("  Testing bandwidth constraints...")
    
    system = VoiceStreamingSystem()
    system.network.bandwidth_limit_kbps = 128  # Low bandwidth
    
    system.start()
    system.user_is_speaking = True
    
    await asyncio.sleep(3.0)
    
    stats = system.metrics.get_stats()
    
    # Should still work but with higher latency
    result.assert_true(
        stats["chunks"]["sent"] > 100,
        f"Streaming works with limited bandwidth: {stats['chunks']['sent']} chunks"
    )
    
    system.stop()
    
    # Test 4: Network recovery
    print("  Testing network condition changes...")
    
    system = VoiceStreamingSystem()
    system.start()
    system.user_is_speaking = True
    
    # Start with good network
    system.network.base_latency_ms = 20
    await asyncio.sleep(2.0)
    
    # Degrade network
    system.network.base_latency_ms = 200
    system.network.packet_loss_rate = 0.1
    await asyncio.sleep(2.0)
    
    # Recover network
    system.network.base_latency_ms = 30
    system.network.packet_loss_rate = 0.0
    await asyncio.sleep(2.0)
    
    # System should handle transitions
    final_stats = system.metrics.get_stats()
    
    result.assert_true(
        final_stats["chunks"]["sent"] > 250,
        "System maintained streaming through network changes"
    )
    
    system.stop()
    return result


async def test_echo_cancellation_performance():
    """Test echo cancellation doesn't introduce delays"""
    print("\n5. Testing Echo Cancellation Performance")
    print("-" * 40)
    
    result = TestResult()
    
    # Test with and without echo cancellation
    configs = [
        ("Without EC", False),
        ("With EC", True)
    ]
    
    latencies = {}
    
    for name, enable_ec in configs:
        print(f"  Testing {name}...")
        
        system = VoiceStreamingSystem()
        
        # Simulate echo cancellation processing
        if enable_ec:
            original_process = system.audio_engine.process_audio
            
            def process_with_ec(audio_bytes):
                # Add 1ms for EC processing
                time.sleep(0.001)
                return original_process(audio_bytes)
            
            system.audio_engine.process_audio = process_with_ec
        
        system.start()
        
        # Full duplex operation
        system.user_is_speaking = True
        
        # AI also speaking (creates echo scenario)
        ai_thread = threading.Thread(target=lambda: system.simulate_ai_response(3.0))
        ai_thread.start()
        
        await asyncio.sleep(3.0)
        ai_thread.join()
        
        stats = system.metrics.get_stats()
        latencies[name] = stats["timing"]["avg_e2e_ms"]
        
        system.stop()
    
    # Compare latencies
    latency_increase = latencies["With EC"] - latencies["Without EC"]
    
    result.assert_less_than(
        latency_increase,
        5.0,
        f"Echo cancellation adds < 5ms latency"
    )
    
    # Both should meet realtime requirements
    for name, latency in latencies.items():
        result.assert_less_than(
            latency,
            50.0,
            f"{name}: Meets realtime requirement"
        )
    
    return result


async def test_conversation_patterns():
    """Test realistic conversation patterns"""
    print("\n6. Testing Realistic Conversation Patterns")
    print("-" * 40)
    
    result = TestResult()
    
    system = VoiceStreamingSystem()
    system.start()
    
    # Simulate a conversation
    conversation_script = [
        ("user", 2.0, "User asks question"),
        ("ai", 3.0, "AI provides answer"),
        ("user", 0.5, "User interrupts"),
        ("pause", 0.5, "Brief pause"),
        ("user", 1.5, "User clarifies"),
        ("ai", 2.0, "AI responds"),
        ("both", 1.0, "Overlapping speech"),
        ("user", 2.0, "User continues alone")
    ]
    
    print("  Running conversation simulation...")
    
    for speaker, duration, description in conversation_script:
        print(f"    {description} ({duration}s)")
        
        if speaker == "user":
            system.user_is_speaking = True
            await asyncio.sleep(duration)
            system.user_is_speaking = False
            
        elif speaker == "ai":
            ai_thread = threading.Thread(
                target=lambda d=duration: system.simulate_ai_response(d)
            )
            ai_thread.start()
            await asyncio.sleep(duration)
            ai_thread.join()
            
        elif speaker == "both":
            system.user_is_speaking = True
            ai_thread = threading.Thread(
                target=lambda d=duration: system.simulate_ai_response(d)
            )
            ai_thread.start()
            await asyncio.sleep(duration)
            system.user_is_speaking = False
            ai_thread.join()
            
        elif speaker == "pause":
            await asyncio.sleep(duration)
    
    # Analyze conversation metrics
    stats = system.metrics.get_stats()
    
    # Should handle all patterns smoothly
    result.assert_true(
        stats["chunks"]["sent"] > 200,
        f"User speech processed: {stats['chunks']['sent']} chunks"
    )
    
    result.assert_true(
        stats["chunks"]["received"] > 150,
        f"AI speech received: {stats['chunks']['received']} chunks"
    )
    
    result.assert_true(
        stats["interrupts"]["count"] >= 1,
        f"Interrupts handled: {stats['interrupts']['count']}"
    )
    
    # Quality maintained throughout
    result.assert_equal(
        stats["quality"]["dropouts"],
        0,
        "No dropouts during conversation"
    )
    
    result.assert_less_than(
        stats["timing"]["avg_e2e_ms"],
        50.0,
        "Low latency throughout conversation"
    )
    
    system.stop()
    return result


async def test_stress_conditions():
    """Test system under stress conditions"""
    print("\n7. Testing Stress Conditions")
    print("-" * 40)
    
    result = TestResult()
    
    # Create stressed system
    system = VoiceStreamingSystem(chunk_duration_ms=10)  # Smaller chunks = more stress
    
    # Add processing overhead
    original_process = system.audio_engine.process_audio
    
    def stressed_process(audio_bytes):
        # Simulate heavy processing
        time.sleep(0.008)  # 8ms processing for 10ms chunks
        return original_process(audio_bytes)
    
    system.audio_engine.process_audio = stressed_process
    
    # Difficult network
    system.network.base_latency_ms = 80
    system.network.jitter_ms = 40
    system.network.packet_loss_rate = 0.02
    
    system.start()
    
    # Heavy usage pattern
    print("  Running stress test...")
    
    # Continuous user speech
    system.user_is_speaking = True
    
    # Multiple AI responses
    for i in range(5):
        ai_thread = threading.Thread(target=lambda: system.simulate_ai_response(1.0))
        ai_thread.start()
        await asyncio.sleep(0.8)  # Overlapping
        system.trigger_interrupt()  # Interrupt each one
        ai_thread.join()
    
    system.user_is_speaking = False
    
    # Check system survived
    stats = system.metrics.get_stats()
    
    # Should degrade gracefully
    result.assert_true(
        stats["chunks"]["sent"] > 300,
        f"System continued under stress: {stats['chunks']['sent']} chunks sent"
    )
    
    # Some dropouts acceptable under stress
    dropout_rate = stats["quality"]["dropouts"] / stats["chunks"]["sent"] if stats["chunks"]["sent"] > 0 else 0
    result.assert_less_than(
        dropout_rate,
        0.05,
        f"Dropout rate < 5%"
    )
    
    # Interrupts still fast
    if stats["interrupts"]["avg_latency_ms"] > 0:
        result.assert_less_than(
            stats["interrupts"]["avg_latency_ms"],
            100.0,
            "Interrupts still responsive under stress"
        )
    
    system.stop()
    return result


async def main():
    """Run all voice streaming tests"""
    print("=" * 60)
    print("Realtime Voice Streaming Tests")
    print("=" * 60)
    
    all_results = []
    
    # Run all tests
    tests = [
        test_realtime_throughput,
        test_bidirectional_streaming,
        test_interrupt_scenarios,
        test_network_simulation,
        test_echo_cancellation_performance,
        test_conversation_patterns,
        test_stress_conditions
    ]
    
    for test_func in tests:
        try:
            result = await test_func()
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
        print("\n✅ All voice streaming tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)