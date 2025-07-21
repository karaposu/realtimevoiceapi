# core/audio_processor.py

"""
Unified Audio Processing Core

Consolidated audio processing with optimized fast/big lane paths.
Zero-copy operations for realtime performance.
"""

import base64
import struct
import logging
import time
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
import wave
import threading

# Conditional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .audio_types import (
    AudioFormat, AudioConfig, AudioMetadata, AudioErrorType,
    AudioBytes, SampleRate, DurationMs, AudioConstants,
    ProcessingMode, BufferConfig, VADConfig, VADType,
    get_optimal_chunk_size
)
from .exceptions import AudioError


# ============== Buffer Pool for Zero-Copy Operations ==============

class BufferPool:
    """Pre-allocated buffer pool for zero-copy operations"""
    
    def __init__(self, pool_size: int = 10, buffer_size: int = 48000):
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.buffers = [bytearray(buffer_size) for _ in range(pool_size)]
        self.available = list(range(pool_size))
        self.in_use = set()
        self.lock = threading.Lock()
        
    def acquire(self) -> Optional[bytearray]:
        """Get a buffer from pool"""
        with self.lock:
            if not self.available:
                return None
            idx = self.available.pop()
            self.in_use.add(idx)
            return self.buffers[idx]
    
    def release(self, buffer: bytearray) -> None:
        """Return buffer to pool"""
        with self.lock:
            for idx, buf in enumerate(self.buffers):
                if buf is buffer:
                    if idx in self.in_use:
                        self.in_use.remove(idx)
                        self.available.append(idx)
                    break


# ============== Main Audio Processor ==============

class AudioProcessor:
    """
    Unified audio processor with fast/big lane optimization.
    
    Features:
    - Zero-copy fast lane processing
    - Quality processing for big lane
    - Automatic format detection and conversion
    - Integrated VAD support
    - Buffer pooling for low latency
    """
    
    def __init__(
        self,
        config: AudioConfig = None,
        mode: ProcessingMode = ProcessingMode.BALANCED,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or AudioConfig()
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Check numpy availability
        self.has_numpy = HAS_NUMPY
        if not self.has_numpy and mode == ProcessingMode.QUALITY:
            self.logger.warning("NumPy not available - quality mode limited")
        
        # Buffer pool for fast lane
        self.buffer_pool: Optional[BufferPool] = None
        if mode == ProcessingMode.REALTIME and config.pre_allocate_buffers:
            self.buffer_pool = BufferPool(
                pool_size=AudioConstants.FAST_LANE_BUFFER_COUNT,
                buffer_size=config.chunk_size_bytes(1000)  # 1 second max
            )
        
        # Pre-compile validators for fast lane
        self._validators = self._compile_validators()
        
    def _compile_validators(self) -> Dict[str, Callable]:
        """Pre-compile validation functions for speed"""
        return {
            'min_size': lambda size: size >= self.config.frame_size,
            'alignment': lambda size: size % self.config.frame_size == 0,
            'max_size': lambda size: size <= AudioConstants.MAX_AUDIO_SIZE_BYTES
        }
    
    # ============== Fast Lane Processing (Minimal Latency) ==============
    
    def process_realtime(self, audio_bytes: AudioBytes) -> AudioBytes:
        """
        Fast lane processing - minimal latency, no allocations.
        
        Returns:
            Processed audio (may be same buffer for zero-copy)
        """
        # Quick validation
        if not self._validators['min_size'](len(audio_bytes)):
            raise AudioError("Audio too small", AudioErrorType.VALIDATION_ERROR)
        
        if not self._validators['alignment'](len(audio_bytes)):
            raise AudioError("Audio not aligned to frame boundary", AudioErrorType.FORMAT_ERROR)
        
        # For realtime mode, return as-is (zero processing overhead)
        if self.mode == ProcessingMode.REALTIME:
            return audio_bytes
        
        # Minimal processing only
        return self._apply_fast_processing(audio_bytes)
    
    def _apply_fast_processing(self, audio_bytes: AudioBytes) -> AudioBytes:
        """Apply minimal processing suitable for fast lane"""
        # Only do in-place operations if needed
        if self.config.format != AudioFormat.PCM16:
            # Format conversion needed - use buffer pool if available
            if self.buffer_pool:
                buffer = self.buffer_pool.acquire()
                if buffer and len(buffer) >= len(audio_bytes):
                    # Convert in-place using pre-allocated buffer
                    return self._convert_format_fast(audio_bytes, buffer)
            
            # Fallback to allocation
            return self._convert_format(audio_bytes, self.config.format)
        
        return audio_bytes
    
    # ============== Big Lane Processing (Quality) ==============
    
    def process_quality(
        self,
        audio_bytes: AudioBytes,
        enhance: bool = True,
        normalize: bool = True
    ) -> Tuple[AudioBytes, AudioMetadata]:
        """
        Big lane processing - higher quality, more features.
        
        Returns:
            Tuple of (processed_audio, metadata)
        """
        if not self.has_numpy:
            # Fallback to basic processing
            metadata = self._analyze_basic(audio_bytes)
            return audio_bytes, metadata
        
        # Full quality processing pipeline
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Analyze first
        metadata = self._analyze_quality(audio_array)
        
        # Enhance if requested
        if enhance:
            audio_array = self._enhance_audio(audio_array, metadata)
        
        # Normalize if requested
        if normalize:
            audio_array = self._normalize_audio(audio_array)
        
        # Update metadata after processing
        metadata = self._analyze_quality(audio_array)
        
        return audio_array.tobytes(), metadata
    
    def _enhance_audio(self, audio_array: np.ndarray, metadata: AudioMetadata) -> np.ndarray:
        """Apply audio enhancement"""
        if metadata.signal_to_noise_ratio and metadata.signal_to_noise_ratio < 10:
            # Apply simple noise reduction
            audio_array = self._reduce_noise_simple(audio_array)
        
        return audio_array
    
    def _reduce_noise_simple(self, audio_array: np.ndarray) -> np.ndarray:
        """Simple noise reduction using spectral subtraction"""
        # This is a placeholder - real implementation would use FFT
        # For now, just apply a gentle high-pass filter effect
        if len(audio_array) > 1:
            diff = np.diff(audio_array)
            return np.concatenate([[audio_array[0]], audio_array[1:] - diff * 0.1])
        return audio_array
    
    # ============== Core Operations ==============
    
    def validate_format(
        self,
        audio_bytes: AudioBytes,
        expected_format: AudioFormat = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate audio data format"""
        if not audio_bytes:
            return False, "Audio data is empty"
        
        expected_format = expected_format or self.config.format
        
        # Basic validation
        if expected_format == AudioFormat.PCM16:
            if len(audio_bytes) % 2 != 0:
                return False, "PCM16 data must have even number of bytes"
        
        # Check duration
        duration_ms = self.calculate_duration(audio_bytes)
        if duration_ms < AudioConstants.MIN_CHUNK_DURATION_MS:
            return False, f"Audio too short: {duration_ms:.1f}ms"
        if duration_ms > AudioConstants.MAX_DURATION_MS:
            return False, f"Audio too long: {duration_ms:.1f}ms"
        
        # Advanced validation if numpy available
        if self.has_numpy and self.mode != ProcessingMode.REALTIME:
            return self._validate_advanced(audio_bytes)
        
        return True, None
    
    def _validate_advanced(self, audio_bytes: AudioBytes) -> Tuple[bool, Optional[str]]:
        """Advanced validation with numpy"""
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Check for silence
            if np.max(np.abs(samples)) < 100:
                return False, "Audio appears to be silent"
            
            # Check for clipping
            clipping_threshold = int(32767 * AudioConstants.CLIPPING_THRESHOLD)
            if np.any(np.abs(samples) > clipping_threshold):
                return False, "Audio may be clipping"
            
            # Check for DC offset
            dc_offset = np.mean(samples)
            if abs(dc_offset) > 1000:
                return False, f"Audio has DC offset: {dc_offset:.0f}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def calculate_duration(self, audio_bytes: AudioBytes) -> DurationMs:
        """Calculate audio duration in milliseconds"""
        if not audio_bytes:
            return 0.0
        
        num_samples = len(audio_bytes) // self.config.frame_size
        duration_seconds = num_samples / self.config.sample_rate
        return duration_seconds * 1000
    
    # ============== Chunking Operations ==============
    
    def chunk_audio(
        self,
        audio_bytes: AudioBytes,
        chunk_duration_ms: int = None,
        align_to_frames: bool = True
    ) -> List[AudioBytes]:
        """Split audio into chunks for streaming"""
        if chunk_duration_ms is None:
            chunk_duration_ms = self.config.chunk_duration_ms
        
        # Validate chunk duration
        chunk_duration_ms = max(
            AudioConstants.MIN_CHUNK_DURATION_MS,
            min(chunk_duration_ms, AudioConstants.MAX_CHUNK_DURATION_MS)
        )
        
        chunk_size = self.config.chunk_size_bytes(chunk_duration_ms)
        
        # Ensure alignment to frame boundaries
        if align_to_frames:
            chunk_size = (chunk_size // self.config.frame_size) * self.config.frame_size
        
        # Fast chunking for realtime mode
        if self.mode == ProcessingMode.REALTIME:
            return self._chunk_fast(audio_bytes, chunk_size)
        
        # Quality chunking with overlap prevention
        return self._chunk_quality(audio_bytes, chunk_size)
    
    def _chunk_fast(self, audio_bytes: AudioBytes, chunk_size: int) -> List[AudioBytes]:
        """Fast chunking without allocations"""
        chunks = []
        for i in range(0, len(audio_bytes), chunk_size):
            # Use memoryview for zero-copy slicing
            chunk = memoryview(audio_bytes)[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(bytes(chunk))
        return chunks
    
    def _chunk_quality(self, audio_bytes: AudioBytes, chunk_size: int) -> List[AudioBytes]:
        """Quality chunking with crossfade option"""
        chunks = []
        offset = 0
        
        while offset < len(audio_bytes):
            end = min(offset + chunk_size, len(audio_bytes))
            chunk = audio_bytes[offset:end]
            
            # Only add non-empty chunks
            if len(chunk) > 0:
                chunks.append(chunk)
            
            offset = end
        
        return chunks
    
    # ============== Format Conversion ==============
    
    def ensure_mono(self, audio_bytes: AudioBytes, channels: int) -> AudioBytes:
        """Convert multi-channel audio to mono"""
        if channels == 1:
            return audio_bytes
        
        if not self.has_numpy:
            # Simple channel averaging without numpy
            return self._ensure_mono_simple(audio_bytes, channels)
        
        # Numpy version for quality
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        multi_channel = samples.reshape(-1, channels)
        
        # Average channels to create mono
        mono = np.mean(multi_channel, axis=1, dtype=np.int16)
        
        return mono.tobytes()
    
    def _ensure_mono_simple(self, audio_bytes: AudioBytes, channels: int) -> AudioBytes:
        """Simple mono conversion without numpy"""
        result = bytearray()
        sample_size = 2  # 16-bit
        
        for i in range(0, len(audio_bytes), sample_size * channels):
            # Average all channels
            total = 0
            for c in range(channels):
                offset = i + c * sample_size
                if offset + sample_size <= len(audio_bytes):
                    sample = struct.unpack('<h', audio_bytes[offset:offset + sample_size])[0]
                    total += sample
            
            avg_sample = int(total / channels)
            result.extend(struct.pack('<h', avg_sample))
        
        return bytes(result)
    
    def resample(
        self,
        audio_bytes: AudioBytes,
        from_rate: SampleRate,
        to_rate: SampleRate
    ) -> AudioBytes:
        """Resample audio to different sample rate"""
        if from_rate == to_rate:
            return audio_bytes
        
        if not self.has_numpy:
            raise AudioError(
                "NumPy required for resampling",
                AudioErrorType.UNSUPPORTED_OPERATION
            )
        
        # Convert to numpy
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Simple linear interpolation resampling
        ratio = to_rate / from_rate
        new_length = int(len(samples) * ratio)
        
        old_indices = np.linspace(0, len(samples) - 1, new_length)
        new_samples = np.interp(old_indices, np.arange(len(samples)), samples)
        
        return new_samples.astype(np.int16).tobytes()
    
    # ============== Quality Analysis ==============
    
    def analyze_audio(
        self,
        audio_bytes: AudioBytes,
        detailed: bool = True
    ) -> AudioMetadata:
        """Analyze audio and extract metadata"""
        if not self.has_numpy or self.mode == ProcessingMode.REALTIME:
            # Basic analysis only
            return self._analyze_basic(audio_bytes)
        
        # Full quality analysis
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        return self._analyze_quality(samples)
    
    def _analyze_basic(self, audio_bytes: AudioBytes) -> AudioMetadata:
        """Basic analysis without numpy"""
        metadata = AudioMetadata(
            format=self.config.format,
            duration_ms=self.calculate_duration(audio_bytes),
            size_bytes=len(audio_bytes),
            processing_mode=self.mode
        )
        
        # Sample-based peak detection (first 1000 samples)
        if len(audio_bytes) >= 2:
            max_val = 0
            sample_count = min(1000, len(audio_bytes) // 2)
            
            for i in range(0, sample_count * 2, 2):
                sample = abs(struct.unpack('<h', audio_bytes[i:i+2])[0])
                max_val = max(max_val, sample)
            
            metadata.peak_amplitude = max_val / 32767.0
            metadata.is_speech = metadata.peak_amplitude > self.config.min_amplitude
        
        return metadata
    
    def _analyze_quality(self, samples: Union[np.ndarray, AudioBytes]) -> AudioMetadata:
        """Quality analysis with numpy"""
        if isinstance(samples, bytes):
            samples = np.frombuffer(samples, dtype=np.int16)
        
        # Normalize to float
        float_samples = samples.astype(np.float32) / 32768.0
        
        metadata = AudioMetadata(
            format=self.config.format,
            duration_ms=len(samples) / self.config.sample_rate * 1000,
            size_bytes=len(samples) * 2,
            processing_mode=self.mode
        )
        
        # Calculate metrics
        metadata.peak_amplitude = float(np.max(np.abs(float_samples)))
        metadata.rms_amplitude = float(np.sqrt(np.mean(float_samples ** 2)))
        
        # Estimate SNR (simple version)
        if len(float_samples) > self.config.sample_rate:
            # Use first 100ms as "noise" estimate
            noise_samples = float_samples[:int(self.config.sample_rate * 0.1)]
            noise_power = np.mean(noise_samples ** 2)
            signal_power = np.mean(float_samples ** 2)
            
            if noise_power > 0:
                metadata.signal_to_noise_ratio = float(
                    10 * np.log10(signal_power / noise_power)
                )
        
        # Simple speech detection
        metadata.is_speech = (
            metadata.rms_amplitude > self.config.min_amplitude and
            metadata.peak_amplitude < self.config.max_amplitude
        )
        
        return metadata
    
    def _normalize_audio(
        self,
        audio_array: np.ndarray,
        target_level: float = 0.8
    ) -> np.ndarray:
        """Normalize audio level"""
        peak = np.max(np.abs(audio_array))
        if peak == 0:
            return audio_array
        
        scale_factor = (target_level * 32767) / peak
        return (audio_array * scale_factor).astype(np.int16)
    
    # ============== Encoding/Decoding ==============
    
    @staticmethod
    def bytes_to_base64(audio_bytes: AudioBytes) -> str:
        """Convert audio bytes to base64 string"""
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    @staticmethod
    def base64_to_bytes(audio_b64: str) -> AudioBytes:
        """Convert base64 string to audio bytes"""
        try:
            return base64.b64decode(audio_b64)
        except Exception as e:
            raise AudioError(f"Failed to decode base64 audio: {e}")
    
    # ============== I/O Operations ==============
    
    def load_wav_file(self, file_path: Union[str, Path]) -> AudioBytes:
        """Load WAV file and convert to required format"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise AudioError(f"File not found: {file_path}")
        
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # Read parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert if necessary
                audio_data = frames
                
                # Convert to mono
                if channels > 1:
                    audio_data = self.ensure_mono(audio_data, channels)
                
                # Resample if needed
                if framerate != self.config.sample_rate:
                    audio_data = self.resample(
                        audio_data, framerate, self.config.sample_rate
                    )
                
                # Handle sample width conversion
                if sample_width != 2:  # Not 16-bit
                    audio_data = self._convert_sample_width(
                        audio_data, sample_width, 2
                    )
                
                return audio_data
                
        except Exception as e:
            raise AudioError(f"Failed to load WAV file: {e}")
    
    def save_wav_file(
        self,
        audio_bytes: AudioBytes,
        file_path: Union[str, Path]
    ) -> None:
        """Save audio data as WAV file"""
        file_path = Path(file_path)
        
        try:
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            self.logger.info(f"Saved {len(audio_bytes)} bytes to {file_path}")
            
        except Exception as e:
            raise AudioError(f"Failed to save WAV file: {e}")
    
    def _convert_sample_width(
        self,
        audio_data: AudioBytes,
        from_width: int,
        to_width: int
    ) -> AudioBytes:
        """Convert between different sample widths"""
        if from_width == to_width:
            return audio_data
        
        if not self.has_numpy:
            # Simple conversion for 8-bit to 16-bit
            if from_width == 1 and to_width == 2:
                result = bytearray()
                for byte in audio_data:
                    # Convert unsigned 8-bit to signed 16-bit
                    sample = (byte - 128) * 256
                    result.extend(struct.pack('<h', sample))
                return bytes(result)
            
            raise AudioError(
                f"Sample width conversion {from_width}->{to_width} requires NumPy"
            )
        
        # Numpy conversions
        if from_width == 1:  # 8-bit to 16-bit
            audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.int16)
            audio_array = (audio_array - 128) * 256
        elif from_width == 3:  # 24-bit to 16-bit
            # Complex conversion - keeping it simple
            raise AudioError("24-bit conversion not implemented")
        elif from_width == 4:  # 32-bit to 16-bit
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
            audio_array = (audio_array >> 16).astype(np.int16)
        else:
            raise AudioError(f"Unsupported sample width: {from_width}")
        
        return audio_array.tobytes()
    
    def _convert_format(self, audio_data: AudioBytes, target_format: AudioFormat) -> AudioBytes:
        """Convert between audio formats"""
        if target_format == AudioFormat.PCM16:
            return audio_data
        
        # Implement G.711 conversions if needed
        raise AudioError(
            f"Format conversion to {target_format} not implemented",
            AudioErrorType.UNSUPPORTED_OPERATION
        )
    
    def _convert_format_fast(
        self,
        audio_data: AudioBytes,
        buffer: bytearray
    ) -> AudioBytes:
        """Fast in-place format conversion"""
        # This is a placeholder for fast conversion
        # Real implementation would do in-place conversion
        return audio_data


# ============== Stream Buffer ==============

class AudioStreamBuffer:
    """
    Optimized buffer for audio streaming.
    
    Uses circular buffer for realtime mode,
    dynamic buffer for quality mode.
    """
    
    def __init__(
        self,
        config: BufferConfig = None,
        audio_config: AudioConfig = None,
        mode: ProcessingMode = ProcessingMode.BALANCED,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or BufferConfig()
        self.audio_config = audio_config or AudioConfig()
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Choose buffer implementation based on mode
        if mode == ProcessingMode.REALTIME and self.config.use_circular:
            self._init_circular_buffer()
        else:
            self._init_dynamic_buffer()
        
        # Metrics
        self.total_bytes_added = 0
        self.total_bytes_consumed = 0
        self.overflow_count = 0
        self.underflow_count = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _init_circular_buffer(self):
        """Initialize circular buffer for realtime mode"""
        self.buffer_size = self.config.max_size_bytes
        self.buffer = bytearray(self.buffer_size)
        self.write_pos = 0
        self.read_pos = 0
        self.available = 0
        self.is_circular = True
    
    def _init_dynamic_buffer(self):
        """Initialize dynamic buffer for quality mode"""
        self.buffer = bytearray()
        self.is_circular = False
    
    def add_audio(self, audio_bytes: AudioBytes) -> bool:
        """Add audio to buffer"""
        with self.lock:
            if self.is_circular:
                return self._add_circular(audio_bytes)
            else:
                return self._add_dynamic(audio_bytes)
    
    def _add_circular(self, audio_bytes: AudioBytes) -> bool:
        """Add to circular buffer (fast lane)"""
        bytes_to_add = len(audio_bytes)
        
        # Check space
        free_space = self.buffer_size - self.available
        if bytes_to_add > free_space:
            # Handle overflow
            if self.config.overflow_strategy == "error":
                raise AudioError("Buffer overflow", AudioErrorType.BUFFER_OVERFLOW)
            elif self.config.overflow_strategy == "drop_newest":
                self.overflow_count += 1
                return False
            else:  # drop_oldest
                # Advance read position
                overflow = bytes_to_add - free_space
                self.read_pos = (self.read_pos + overflow) % self.buffer_size
                self.available -= overflow
                self.overflow_count += 1
        
        # Copy data (handle wrap-around)
        remaining = bytes_to_add
        src_offset = 0
        
        while remaining > 0:
            chunk_size = min(remaining, self.buffer_size - self.write_pos)
            
            # Use memoryview for efficient copying
            dst = memoryview(self.buffer)[self.write_pos:self.write_pos + chunk_size]
            src = memoryview(audio_bytes)[src_offset:src_offset + chunk_size]
            dst[:] = src
            
            self.write_pos = (self.write_pos + chunk_size) % self.buffer_size
            self.available += chunk_size
            remaining -= chunk_size
            src_offset += chunk_size
        
        self.total_bytes_added += bytes_to_add
        return True
    
    def _add_dynamic(self, audio_bytes: AudioBytes) -> bool:
        """Add to dynamic buffer (big lane)"""
        # Check size limits
        new_size = len(self.buffer) + len(audio_bytes)
        if new_size > self.config.max_size_bytes:
            if self.config.overflow_strategy == "error":
                raise AudioError("Buffer overflow", AudioErrorType.BUFFER_OVERFLOW)
            elif self.config.overflow_strategy == "drop_oldest":
                # Remove old data
                overflow = new_size - self.config.max_size_bytes
                self.buffer = self.buffer[overflow:]
                self.overflow_count += 1
            else:  # drop_newest
                self.overflow_count += 1
                return False
        
        self.buffer.extend(audio_bytes)
        self.total_bytes_added += len(audio_bytes)
        return True
    
    def get_chunk(self, chunk_size: int) -> Optional[AudioBytes]:
        """Get chunk from buffer"""
        with self.lock:
            if self.is_circular:
                return self._get_circular(chunk_size)
            else:
                return self._get_dynamic(chunk_size)
    
    def _get_circular(self, chunk_size: int) -> Optional[AudioBytes]:
        """Get from circular buffer"""
        if self.available < chunk_size:
            if self.config.underflow_strategy == "error":
                raise AudioError("Buffer underflow", AudioErrorType.BUFFER_UNDERFLOW)
            self.underflow_count += 1
            return None
        
        # Read data (handle wrap-around)
        if self.config.zero_copy and chunk_size <= self.buffer_size - self.read_pos:
            # Can return view directly
            result = bytes(memoryview(self.buffer)[self.read_pos:self.read_pos + chunk_size])
            self.read_pos = (self.read_pos + chunk_size) % self.buffer_size
            self.available -= chunk_size
        else:
            # Need to copy (wrap-around case)
            result = bytearray(chunk_size)
            remaining = chunk_size
            dst_offset = 0
            
            while remaining > 0:
                chunk = min(remaining, self.buffer_size - self.read_pos)
                
                src = memoryview(self.buffer)[self.read_pos:self.read_pos + chunk]
                dst = memoryview(result)[dst_offset:dst_offset + chunk]
                dst[:] = src
                
                self.read_pos = (self.read_pos + chunk) % self.buffer_size
                self.available -= chunk
                remaining -= chunk
                dst_offset += chunk
            
            result = bytes(result)
        
        self.total_bytes_consumed += chunk_size
        return result
    
    def _get_dynamic(self, chunk_size: int) -> Optional[AudioBytes]:
        """Get from dynamic buffer"""
        if len(self.buffer) < chunk_size:
            if self.config.underflow_strategy == "error":
                raise AudioError("Buffer underflow", AudioErrorType.BUFFER_UNDERFLOW)
            self.underflow_count += 1
            return None
        
        chunk = bytes(self.buffer[:chunk_size])
        self.buffer = self.buffer[chunk_size:]
        self.total_bytes_consumed += chunk_size
        
        return chunk
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            if self.is_circular:
                self.write_pos = 0
                self.read_pos = 0
                self.available = 0
            else:
                self.buffer.clear()
    
    def get_available_bytes(self) -> int:
        """Get number of bytes available"""
        with self.lock:
            if self.is_circular:
                return self.available
            else:
                return len(self.buffer)
    
    def get_duration_ms(self) -> float:
        """Get duration of buffered audio"""
        available = self.get_available_bytes()
        return self.audio_config.duration_from_bytes(available)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            available = self.get_available_bytes()
            
            stats = {
                "available_bytes": available,
                "duration_ms": self.get_duration_ms(),
                "total_added": self.total_bytes_added,
                "total_consumed": self.total_bytes_consumed,
                "overflow_count": self.overflow_count,
                "underflow_count": self.underflow_count,
                "utilization": available / self.config.max_size_bytes,
                "mode": self.mode.value,
                "is_circular": self.is_circular
            }
            
            if self.is_circular:
                stats.update({
                    "write_pos": self.write_pos,
                    "read_pos": self.read_pos,
                    "buffer_size": self.buffer_size
                })
            
            return stats


# ============== Convenience Functions ==============

def create_processor(mode: str = "balanced") -> AudioProcessor:
    """Create audio processor with mode"""
    processing_mode = ProcessingMode[mode.upper()]
    config = AudioConfig()
    
    # Optimize config for mode
    if processing_mode == ProcessingMode.REALTIME:
        config.chunk_duration_ms = AudioConstants.FAST_LANE_CHUNK_MS
        config.use_numpy = False
        config.pre_allocate_buffers = True
    elif processing_mode == ProcessingMode.QUALITY:
        config.chunk_duration_ms = 200
        config.use_numpy = True
        config.pre_allocate_buffers = False
    
    return AudioProcessor(config, processing_mode)


def validate_realtime_audio(audio_bytes: AudioBytes) -> Tuple[bool, Optional[str]]:
    """Quick validation for realtime audio"""
    processor = create_processor("realtime")
    return processor.validate_format(audio_bytes, AudioFormat.PCM16)


def chunk_for_streaming(
    audio_bytes: AudioBytes,
    chunk_ms: int = 100,
    mode: str = "balanced"
) -> List[AudioBytes]:
    """Chunk audio for streaming"""
    processor = create_processor(mode)
    return processor.chunk_audio(audio_bytes, chunk_ms)