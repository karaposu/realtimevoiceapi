"""
Audio Engine - Central Processing Orchestrator

Manages audio processing paths for fast/big lane optimization.
Coordinates between AudioProcessor, BufferPool, and processing pipelines.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from .audio_types import (
    AudioBytes, AudioConfig, ProcessingMode, AudioFormat,
    AudioConstants, AudioMetadata, BufferConfig,AudioErrorType

)
from .audio_processor import AudioProcessor, BufferPool, AudioStreamBuffer
from .exceptions import AudioError

@dataclass
class ProcessingMetrics:
    """Metrics for audio processing performance"""
    total_chunks: int = 0
    total_bytes: int = 0
    total_processing_time: float = 0.0
    
    # Latency tracking
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Buffer pool stats
    buffer_pool_hits: int = 0
    buffer_pool_misses: int = 0
    
    # Processing mode stats
    realtime_chunks: int = 0
    quality_chunks: int = 0
    
    def update(self, bytes_processed: int, processing_time: float):
        """Update metrics with new processing data"""
        self.total_chunks += 1
        self.total_bytes += bytes_processed
        self.total_processing_time += processing_time
        
        # Calculate latency in ms
        latency_ms = processing_time * 1000
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        
        # Update rolling average
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.total_chunks - 1) + latency_ms) / 
            self.total_chunks
        )
    
    def record_error(self, error: Exception):
        """Record processing error"""
        self.error_count += 1
        self.last_error = str(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_chunks": self.total_chunks,
            "total_bytes": self.total_bytes,
            "total_processing_time_s": self.total_processing_time,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.total_chunks > 0 else 0,
            "max_latency_ms": self.max_latency_ms,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "buffer_pool_hit_rate": (
                self.buffer_pool_hits / (self.buffer_pool_hits + self.buffer_pool_misses)
                if (self.buffer_pool_hits + self.buffer_pool_misses) > 0 else 0
            ),
            "realtime_percentage": (
                self.realtime_chunks / self.total_chunks * 100
                if self.total_chunks > 0 else 0
            )
        }


class ProcessingStrategy(Enum):
    """Processing strategies for different scenarios"""
    ZERO_COPY = "zero_copy"          # Minimal processing, pass-through
    FAST_LANE = "fast_lane"          # Realtime with minimal features
    BALANCED = "balanced"            # Balance of speed and quality
    QUALITY = "quality"              # Full processing pipeline
    ADAPTIVE = "adaptive"            # Dynamically adjust based on load


class AudioEngine:
    """
    Central audio processing engine.
    
    Coordinates processing based on mode and performance requirements.
    Manages resources and provides adaptive optimization.
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.BALANCED,
        config: AudioConfig = None,
        buffer_config: BufferConfig = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize audio engine.
        
        Args:
            mode: Processing mode (REALTIME, QUALITY, BALANCED)
            config: Audio configuration
            buffer_config: Buffer pool configuration
            logger: Optional logger
        """
        self.mode = mode
        self.config = config or AudioConfig()
        self.buffer_config = buffer_config or BufferConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Create processor
        self.processor = AudioProcessor(self.config, mode)
        
        # Initialize buffer pool for realtime mode
        self.buffer_pool: Optional[BufferPool] = None
        if mode == ProcessingMode.REALTIME and self.buffer_config.pre_allocate:
            self._init_buffer_pool()
        
        # Processing history for adaptive optimization
        self._processing_history = deque(maxlen=100)
        self._adaptive_threshold = 10.0  # ms threshold for switching strategies
        
        # Metrics
        self.metrics = ProcessingMetrics()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Processing pipeline components
        self._pre_processors: List[Callable] = []
        self._post_processors: List[Callable] = []
        
        # Stream buffer for big lane
        self._stream_buffer: Optional[AudioStreamBuffer] = None
        if mode == ProcessingMode.QUALITY:
            self._stream_buffer = AudioStreamBuffer(
                config=self.buffer_config,
                audio_config=self.config,
                mode=mode
            )
        
        self.logger.info(f"AudioEngine initialized in {mode.value} mode")
    
    def _init_buffer_pool(self):
        """Initialize buffer pool for zero-copy operations"""
        pool_size = self.buffer_config.pool_size
        buffer_size = self.buffer_config.pool_buffer_size
        
        self.buffer_pool = BufferPool(pool_size, buffer_size)
        self.logger.debug(f"Buffer pool initialized: {pool_size} buffers of {buffer_size} bytes")
    
    # ============== Main Processing Interface ==============
    
    def process_audio(self, audio_bytes: AudioBytes) -> AudioBytes:
        """
        Process audio through appropriate path.
        
        Routes to fast/balanced/quality processing based on mode.
        
        Args:
            audio_bytes: Input audio data
            
        Returns:
            Processed audio bytes
        """
        start_time = time.perf_counter()
        
        try:
            # Quick validation
            if not audio_bytes:
                return audio_bytes
            
            # Choose processing strategy
            if self.mode == ProcessingMode.REALTIME:
                result = self._process_realtime(audio_bytes)
            elif self.mode == ProcessingMode.QUALITY:
                result = self._process_quality(audio_bytes)
            else:  # BALANCED
                result = self._process_balanced(audio_bytes)
            
            # Update metrics
            processing_time = time.perf_counter() - start_time
            with self._lock:
                self.metrics.update(len(audio_bytes), processing_time)
                self._processing_history.append(processing_time)
            
            return result
            
        except Exception as e:
            self.metrics.record_error(e)
            self.logger.error(f"Processing error: {e}")
            raise
    
    # ============== Processing Strategies ==============
    
    def _process_realtime(self, audio_bytes: AudioBytes) -> AudioBytes:
        """
        Fast lane processing - minimal latency.
        
        Uses buffer pool for zero-copy when possible.
        """
        self.metrics.realtime_chunks += 1
        
        # Try zero-copy with buffer pool
        if self.buffer_pool and len(audio_bytes) <= self.buffer_config.pool_buffer_size:
            buffer = self.buffer_pool.acquire()
            if buffer:
                try:
                    # Process into pre-allocated buffer
                    self.metrics.buffer_pool_hits += 1
                    return self._process_with_buffer(audio_bytes, buffer)
                finally:
                    self.buffer_pool.release(buffer)
            else:
                self.metrics.buffer_pool_misses += 1
        
        # Fallback to minimal processing
        return self.processor.process_realtime(audio_bytes)
    
    def _process_balanced(self, audio_bytes: AudioBytes) -> AudioBytes:
        """
        Balanced processing - adaptive based on load.
        
        Switches between fast and quality based on performance.
        """
        # Check recent processing times
        if self._should_use_fast_path():
            return self._process_realtime(audio_bytes)
        else:
            # Use lighter quality processing
            return self._process_quality_light(audio_bytes)
    
    def _process_quality(self, audio_bytes: AudioBytes) -> AudioBytes:
        """
        Quality processing - full pipeline.
        
        Includes enhancement, normalization, and analysis.
        """
        self.metrics.quality_chunks += 1
        
        # Full quality processing with metadata
        processed, metadata = self.processor.process_quality(
            audio_bytes,
            enhance=True,
            normalize=True
        )
        
        # Store metadata if needed
        if metadata.is_speech:
            if metadata.signal_to_noise_ratio is not None:
                self.logger.debug(f"Speech detected: SNR={metadata.signal_to_noise_ratio:.1f}dB")
            else:
                self.logger.debug("Speech detected: SNR=N/A")
                
        return processed
    
    def _process_quality_light(self, audio_bytes: AudioBytes) -> AudioBytes:
        """Lighter quality processing for balanced mode"""
        # Skip enhancement for faster processing
        processed, _ = self.processor.process_quality(
            audio_bytes,
            enhance=False,
            normalize=True
        )
        return processed
    
    def _process_with_buffer(
        self,
        audio_bytes: AudioBytes,
        buffer: bytearray
    ) -> AudioBytes:
        """Process using pre-allocated buffer for zero-copy"""
        # Copy input to buffer
        buffer[:len(audio_bytes)] = audio_bytes
        
        # Process in-place (simplified - real implementation would modify buffer)
        # For now, just validate and return
        if self.processor._validators['min_size'](len(audio_bytes)):
            return bytes(buffer[:len(audio_bytes)])
        else:
            raise AudioError("Invalid audio size", AudioErrorType.VALIDATION_ERROR)
    
    # ============== Adaptive Optimization ==============
    
    def _should_use_fast_path(self) -> bool:
        """Determine if fast path should be used based on recent performance"""
        if not self._processing_history:
            return True
        
        # Calculate recent average latency
        recent_avg = sum(self._processing_history) / len(self._processing_history) * 1000
        
        # Use fast path if recent latency is high
        return recent_avg > self._adaptive_threshold
    
    def set_adaptive_threshold(self, threshold_ms: float):
        """Set threshold for adaptive processing switching"""
        self._adaptive_threshold = threshold_ms
        self.logger.info(f"Adaptive threshold set to {threshold_ms}ms")
    
    # ============== Pipeline Management ==============
    
    def add_pre_processor(self, processor: Callable[[AudioBytes], AudioBytes]):
        """Add pre-processing stage"""
        self._pre_processors.append(processor)
    
    def add_post_processor(self, processor: Callable[[AudioBytes], AudioBytes]):
        """Add post-processing stage"""
        self._post_processors.append(processor)
    
    def process_with_pipeline(self, audio_bytes: AudioBytes) -> AudioBytes:
        """Process through full pipeline with pre/post processors"""
        # Pre-processing
        for processor in self._pre_processors:
            audio_bytes = processor(audio_bytes)
        
        # Main processing
        audio_bytes = self.process_audio(audio_bytes)
        
        # Post-processing
        for processor in self._post_processors:
            audio_bytes = processor(audio_bytes)
        
        return audio_bytes
    
    # ============== Streaming Support ==============
    
    def process_stream(
        self,
        audio_chunks: List[AudioBytes],
        chunk_callback: Optional[Callable[[AudioBytes, int], None]] = None
    ) -> AudioBytes:
        """
        Process multiple chunks as a stream.
        
        Args:
            audio_chunks: List of audio chunks
            chunk_callback: Optional callback for each processed chunk
            
        Returns:
            Combined processed audio
        """
        processed_chunks = []
        
        for i, chunk in enumerate(audio_chunks):
            processed = self.process_audio(chunk)
            processed_chunks.append(processed)
            
            if chunk_callback:
                chunk_callback(processed, i)
        
        # Combine chunks
        return b''.join(processed_chunks)
    
    def add_to_stream_buffer(self, audio_bytes: AudioBytes) -> Optional[AudioBytes]:
        """
        Add audio to stream buffer (for quality mode).
        
        Returns processed audio when buffer is ready.
        """
        if not self._stream_buffer:
            # No buffering in realtime mode
            return self.process_audio(audio_bytes)
        
        # Add to buffer
        self._stream_buffer.add_audio(audio_bytes)
        
        # Check if we have enough for processing
        chunk_size = self.config.chunk_size_bytes(200)  # 200ms chunks for quality
        
        if self._stream_buffer.get_available_bytes() >= chunk_size:
            chunk = self._stream_buffer.get_chunk(chunk_size)
            if chunk:
                return self.process_audio(chunk)
        
        return None
    
    # ============== Resource Management ==============
    
    def optimize_for_latency(self):
        """Optimize settings for minimum latency"""
        self.mode = ProcessingMode.REALTIME
        self.processor.mode = ProcessingMode.REALTIME
        
        # Ensure buffer pool is available
        if not self.buffer_pool:
            self._init_buffer_pool()
        
        # Clear any heavy processors
        self._pre_processors.clear()
        self._post_processors.clear()
        
        self.logger.info("Optimized for minimum latency")
    
    def optimize_for_quality(self):
        """Optimize settings for best quality"""
        self.mode = ProcessingMode.QUALITY
        self.processor.mode = ProcessingMode.QUALITY
        
        # Initialize stream buffer if needed
        if not self._stream_buffer:
            self._stream_buffer = AudioStreamBuffer(
                config=self.buffer_config,
                audio_config=self.config,
                mode=ProcessingMode.QUALITY
            )
        
        self.logger.info("Optimized for quality")
    
    # ============== Metrics and Monitoring ==============
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engine metrics"""
        with self._lock:
            metrics = self.metrics.to_dict()
        
        # Add engine-specific metrics
        metrics.update({
            "mode": self.mode.value,
            "has_buffer_pool": self.buffer_pool is not None,
            "pre_processors": len(self._pre_processors),
            "post_processors": len(self._post_processors),
            "adaptive_threshold_ms": self._adaptive_threshold
        })
        
        # Add buffer pool stats if available
        if self.buffer_pool:
            metrics["buffer_pool"] = {
                "pool_size": self.buffer_pool.pool_size,
                "buffer_size": self.buffer_pool.buffer_size,
                "available": len(self.buffer_pool.available),
                "in_use": len(self.buffer_pool.in_use)
            }
        
        # Add stream buffer stats if available
        if self._stream_buffer:
            metrics["stream_buffer"] = self._stream_buffer.get_stats()
        
        return metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance analysis"""
        metrics = self.get_metrics()
        
        # Add performance analysis
        if self.metrics.total_chunks > 0:
            # Calculate throughput
            if self.metrics.total_processing_time > 0:
                throughput_mbps = (
                    self.metrics.total_bytes / 1024 / 1024 / 
                    self.metrics.total_processing_time
                )
            else:
                throughput_mbps = 0
            
            # Determine if meeting realtime requirements
            realtime_capable = self.metrics.avg_latency_ms < 20  # 20ms threshold
            
            metrics["performance"] = {
                "throughput_mbps": throughput_mbps,
                "realtime_capable": realtime_capable,
                "processing_efficiency": (
                    1.0 - (self.metrics.avg_latency_ms / 100)  # Efficiency score
                    if self.metrics.avg_latency_ms < 100 else 0
                )
            }
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics = ProcessingMetrics()
            self._processing_history.clear()
        
        self.logger.info("Metrics reset")
    
    # ============== Cleanup ==============
    
    def cleanup(self):
        """Clean up resources"""
        # Clear processors
        self._pre_processors.clear()
        self._post_processors.clear()
        
        # Clear buffer pool
        if self.buffer_pool:
            self.buffer_pool = None
        
        # Clear stream buffer
        if self._stream_buffer:
            self._stream_buffer.clear()
            self._stream_buffer = None
        
        # Reset metrics
        self.reset_metrics()
        
        self.logger.info("AudioEngine cleaned up")


# ============== Factory Functions ==============

def create_fast_lane_engine(
    sample_rate: int = 24000,
    chunk_duration_ms: int = 20
) -> AudioEngine:
    """Create engine optimized for fast lane"""
    config = AudioConfig(
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        use_numpy=False,  # Disable numpy for speed
        pre_allocate_buffers=True
    )
    
    buffer_config = BufferConfig(
        pre_allocate=True,
        use_circular=True,
        zero_copy=True,
        pool_size=20,  # More buffers for fast lane
        pool_buffer_size=sample_rate // 10  # 100ms max
    )
    
    engine = AudioEngine(
        mode=ProcessingMode.REALTIME,
        config=config,
        buffer_config=buffer_config
    )
    
    return engine


def create_big_lane_engine(
    sample_rate: int = 24000,
    enable_enhancement: bool = True
) -> AudioEngine:
    """Create engine optimized for big lane"""
    config = AudioConfig(
        sample_rate=sample_rate,
        chunk_duration_ms=200,  # Larger chunks for quality
        use_numpy=True,
        pre_allocate_buffers=False
    )
    
    buffer_config = BufferConfig(
        pre_allocate=False,
        use_circular=False,
        max_size_bytes=10 * 1024 * 1024,  # 10MB buffer
        overflow_strategy="drop_oldest"
    )
    
    engine = AudioEngine(
        mode=ProcessingMode.QUALITY,
        config=config,
        buffer_config=buffer_config
    )
    
    # Add enhancement if requested
    if enable_enhancement:
        # Could add enhancement processors here
        pass
    
    return engine


def create_adaptive_engine(
    sample_rate: int = 24000,
    latency_target_ms: float = 50.0
) -> AudioEngine:
    """Create adaptive engine that switches modes based on load"""
    config = AudioConfig(
        sample_rate=sample_rate,
        chunk_duration_ms=50,  # Medium chunks
        use_numpy=True,
        pre_allocate_buffers=True
    )
    
    engine = AudioEngine(
        mode=ProcessingMode.BALANCED,
        config=config
    )
    
    # Set adaptive threshold
    engine.set_adaptive_threshold(latency_target_ms)
    
    return engine