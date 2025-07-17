"""
Audio Processing Core

Main audio processing functionality. Can be used by both fast and big lanes
with different configurations.


Fast lane uses circular pre-allocated buffers
Big lane uses dynamic buffers
Both are handled by the same class with different configs
"""

import base64
import wave
import struct
import logging
from typing import Optional, Union, List, Tuple, BinaryIO
from pathlib import Path
import numpy as np

from .audio_types import (
    AudioFormat, AudioConfig, AudioMetadata, AudioErrorType,
    AudioBytes, SampleRate, DurationMs, AudioConstants,
    ProcessingMode, BufferConfig
)
from .exceptions import AudioError


class AudioProcessor:
    """
    Core audio processing functionality.
    
    Stateless operations that can be used by both fast and big lanes.
    Fast lane uses directly, big lane may wrap with additional features.
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
        
        # Check if numpy is available for advanced operations
        self.has_numpy = self._check_numpy()
    
    def _check_numpy(self) -> bool:
        """Check if numpy is available"""
        try:
            import numpy as np
            return True
        except ImportError:
            self.logger.warning("NumPy not available - some features disabled")
            return False
    
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
    
    # ============== Validation ==============
    
    def validate_format(
        self,
        audio_bytes: AudioBytes,
        expected_format: AudioFormat = AudioFormat.PCM16
    ) -> Tuple[bool, Optional[str]]:
        """Validate audio data format"""
        
        if not audio_bytes:
            return False, "Audio data is empty"
        
        if expected_format == AudioFormat.PCM16:
            # Check even number of bytes for 16-bit samples
            if len(audio_bytes) % 2 != 0:
                return False, "PCM16 data must have even number of bytes"
            
            # Check reasonable duration
            duration_ms = self.calculate_duration(audio_bytes)
            if duration_ms < 10:
                return False, f"Audio too short: {duration_ms:.1f}ms"
            if duration_ms > AudioConstants.MAX_DURATION_MS:
                return False, f"Audio too long: {duration_ms:.1f}ms"
        
        return True, None
    
    def validate_realtime_format(self, audio_bytes: AudioBytes) -> Tuple[bool, str]:
        """Validate audio meets realtime API requirements"""
        
        # Basic validation
        is_valid, error = self.validate_format(audio_bytes)
        if not is_valid:
            return False, error
        
        # Check sample values if numpy available
        if self.has_numpy:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Check for silence
            if np.max(np.abs(samples)) < 100:
                return False, "Audio appears to be silent"
            
            # Check for clipping
            if np.any(np.abs(samples) > 32000):
                return False, "Audio may be clipping"
        
        return True, "Valid realtime format"
    
    # ============== Duration Calculations ==============
    
    def calculate_duration(self, audio_bytes: AudioBytes) -> DurationMs:
        """Calculate audio duration in milliseconds"""
        if not audio_bytes:
            return 0.0
        
        num_samples = len(audio_bytes) // self.config.frame_size
        duration_seconds = num_samples / self.config.sample_rate
        return duration_seconds * 1000
    
    def calculate_bytes_needed(self, duration_ms: DurationMs) -> int:
        """Calculate bytes needed for given duration"""
        return self.config.chunk_size_bytes(int(duration_ms))
    
    # ============== Chunking ==============
    
    def chunk_audio(
        self,
        audio_bytes: AudioBytes,
        chunk_duration_ms: int = None
    ) -> List[AudioBytes]:
        """Split audio into chunks"""
        
        if chunk_duration_ms is None:
            chunk_duration_ms = self.config.chunk_duration_ms
        
        # Clamp to valid range
        chunk_duration_ms = max(
            self.config.min_chunk_ms,
            min(chunk_duration_ms, self.config.max_chunk_ms)
        )
        
        chunk_size = self.config.chunk_size_bytes(chunk_duration_ms)
        
        # Ensure alignment to frame boundaries
        chunk_size = (chunk_size // self.config.frame_size) * self.config.frame_size
        
        chunks = []
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if len(chunk) > 0:  # Skip empty chunks
                chunks.append(chunk)
        
        self.logger.debug(
            f"Split {len(audio_bytes)} bytes into {len(chunks)} chunks "
            f"of ~{chunk_duration_ms}ms each"
        )
        
        return chunks
    
    # ============== Format Conversion ==============
    
    def ensure_mono(self, audio_bytes: AudioBytes, channels: int) -> AudioBytes:
        """Convert multi-channel audio to mono"""
        
        if channels == 1:
            return audio_bytes
        
        if not self.has_numpy:
            raise AudioError("NumPy required for channel conversion")
        
        # Convert to numpy array
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        multi_channel = samples.reshape(-1, channels)
        
        # Average channels to create mono
        mono = np.mean(multi_channel, axis=1, dtype=np.int16)
        
        return mono.tobytes()
    
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
            raise AudioError("NumPy required for resampling")
        
        # Convert to numpy
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Simple linear interpolation resampling
        # For production, consider using scipy.signal.resample
        ratio = to_rate / from_rate
        new_length = int(len(samples) * ratio)
        
        old_indices = np.linspace(0, len(samples) - 1, new_length)
        new_samples = np.interp(old_indices, np.arange(len(samples)), samples)
        
        return new_samples.astype(np.int16).tobytes()
    
    # ============== Quality Analysis ==============
    
    def analyze_quality(self, audio_bytes: AudioBytes) -> AudioMetadata:
        """Analyze audio quality and extract metadata"""
        
        metadata = AudioMetadata(
            format=AudioFormat.PCM16,
            duration_ms=self.calculate_duration(audio_bytes),
            size_bytes=len(audio_bytes)
        )
        
        if not self.has_numpy or not audio_bytes:
            return metadata
        
        # Convert to float for analysis
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0  # Normalize to [-1, 1]
        
        # Calculate metrics
        metadata.peak_amplitude = float(np.max(np.abs(samples)))
        metadata.rms_amplitude = float(np.sqrt(np.mean(samples ** 2)))
        
        # Simple speech detection
        metadata.is_speech = (
            metadata.rms_amplitude > self.config.min_amplitude and
            metadata.peak_amplitude < self.config.max_amplitude
        )
        
        return metadata
    
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
            raise AudioError("NumPy required for sample width conversion")
        
        # Implementation would go here - keeping it simple for now
        # This would handle 8-bit, 24-bit, 32-bit conversions
        raise NotImplementedError("Sample width conversion not implemented")


# ============== Audio Buffer (Integrated) ==============

class AudioStreamBuffer:
    """
    Buffer for managing audio streams.
    
    Integrated here rather than separate file since it's tightly
    coupled with audio processing.
    """
    
    def __init__(
        self,
        config: BufferConfig = None,
        audio_config: AudioConfig = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or BufferConfig()
        self.audio_config = audio_config or AudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Buffer storage
        if self.config.use_circular:
            # For fast lane - pre-allocated circular buffer
            self._create_circular_buffer()
        else:
            # For big lane - dynamic bytearray
            self.buffer = bytearray()
        
        # Metrics
        self.total_bytes_added = 0
        self.total_bytes_consumed = 0
        self.overflow_count = 0
    
    def _create_circular_buffer(self):
        """Create pre-allocated circular buffer"""
        self.buffer_size = self.config.max_size_bytes
        self.buffer = bytearray(self.buffer_size)
        self.write_pos = 0
        self.read_pos = 0
        self.available = 0
    
    def add_audio(self, audio_bytes: AudioBytes) -> bool:
        """Add audio to buffer"""
        
        if self.config.use_circular:
            return self._add_circular(audio_bytes)
        else:
            return self._add_dynamic(audio_bytes)
    
    def _add_circular(self, audio_bytes: AudioBytes) -> bool:
        """Add to circular buffer (fast lane)"""
        
        bytes_to_add = len(audio_bytes)
        
        if bytes_to_add > self.buffer_size - self.available:
            # Handle overflow
            if self.config.overflow_strategy == "error":
                raise AudioError("Buffer overflow")
            elif self.config.overflow_strategy == "drop_newest":
                return False
            else:  # drop_oldest
                # Advance read position
                overflow = bytes_to_add - (self.buffer_size - self.available)
                self.read_pos = (self.read_pos + overflow) % self.buffer_size
                self.available = self.buffer_size - bytes_to_add
                self.overflow_count += 1
        
        # Copy data in chunks if it wraps around
        remaining = bytes_to_add
        src_offset = 0
        
        while remaining > 0:
            chunk_size = min(remaining, self.buffer_size - self.write_pos)
            self.buffer[self.write_pos:self.write_pos + chunk_size] = \
                audio_bytes[src_offset:src_offset + chunk_size]
            
            self.write_pos = (self.write_pos + chunk_size) % self.buffer_size
            self.available += chunk_size
            remaining -= chunk_size
            src_offset += chunk_size
        
        self.total_bytes_added += bytes_to_add
        return True
    
    def _add_dynamic(self, audio_bytes: AudioBytes) -> bool:
        """Add to dynamic buffer (big lane)"""
        
        # Check size limits
        if len(self.buffer) + len(audio_bytes) > self.config.max_size_bytes:
            if self.config.overflow_strategy == "error":
                raise AudioError("Buffer overflow")
            elif self.config.overflow_strategy == "drop_oldest":
                # Remove old data
                overflow = len(self.buffer) + len(audio_bytes) - self.config.max_size_bytes
                self.buffer = self.buffer[overflow:]
                self.overflow_count += 1
            else:  # drop_newest
                return False
        
        self.buffer.extend(audio_bytes)
        self.total_bytes_added += len(audio_bytes)
        return True
    
    def get_chunk(self, chunk_size: int) -> Optional[AudioBytes]:
        """Get chunk from buffer"""
        
        if self.config.use_circular:
            return self._get_circular(chunk_size)
        else:
            return self._get_dynamic(chunk_size)
    
    def _get_circular(self, chunk_size: int) -> Optional[AudioBytes]:
        """Get from circular buffer"""
        
        if self.available < chunk_size:
            return None
        
        # Read data, handling wrap-around
        result = bytearray(chunk_size)
        remaining = chunk_size
        dst_offset = 0
        
        while remaining > 0:
            chunk = min(remaining, self.buffer_size - self.read_pos)
            result[dst_offset:dst_offset + chunk] = \
                self.buffer[self.read_pos:self.read_pos + chunk]
            
            self.read_pos = (self.read_pos + chunk) % self.buffer_size
            self.available -= chunk
            remaining -= chunk
            dst_offset += chunk
        
        self.total_bytes_consumed += chunk_size
        return bytes(result)
    
    def _get_dynamic(self, chunk_size: int) -> Optional[AudioBytes]:
        """Get from dynamic buffer"""
        
        if len(self.buffer) < chunk_size:
            return None
        
        chunk = bytes(self.buffer[:chunk_size])
        self.buffer = self.buffer[chunk_size:]
        self.total_bytes_consumed += chunk_size
        
        return chunk
    
    def clear(self):
        """Clear buffer"""
        if self.config.use_circular:
            self.write_pos = 0
            self.read_pos = 0
            self.available = 0
        else:
            self.buffer.clear()
    
    def get_available_bytes(self) -> int:
        """Get number of bytes available"""
        if self.config.use_circular:
            return self.available
        else:
            return len(self.buffer)
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            "available_bytes": self.get_available_bytes(),
            "total_added": self.total_bytes_added,
            "total_consumed": self.total_bytes_consumed,
            "overflow_count": self.overflow_count,
            "utilization": self.get_available_bytes() / self.config.max_size_bytes
        }