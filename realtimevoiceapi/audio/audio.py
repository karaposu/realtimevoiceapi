
# here is realtimevoiceapi/audio/audio.py

# to run python -m realtimevoiceapi.audio

"""
Audio processing utilities for RealtimeVoiceAPI

This module provides comprehensive audio processing capabilities for the OpenAI Realtime API,
including format conversion, validation, chunking, and real-time streaming support.

Supported formats:
- PCM16: 16-bit PCM at 24kHz sample rate, mono, little-endian
- G711_ULAW: G.711 μ-law encoding
- G711_ALAW: G.711 A-law encoding

Requirements:
- numpy: For audio data manipulation
- wave: For WAV file handling
- struct: For binary data packing/unpacking
- pydub: For advanced audio format support (optional)
"""

import base64
import io
import wave
import struct
import logging
import math
from typing import Optional, Union, List, Tuple, BinaryIO
from enum import Enum
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    AudioSegment = None

from ..core.exceptions import AudioError


class AudioFormat(Enum):
    """Supported audio formats for OpenAI Realtime API"""
    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"


class AudioConfig:
    """Audio configuration constants for OpenAI Realtime API"""
    
    # OpenAI Realtime API specifications
    SAMPLE_RATE = 24000      # 24kHz sample rate
    CHANNELS = 1             # Mono audio
    SAMPLE_WIDTH = 2         # 16-bit samples (2 bytes)
    FRAME_SIZE = CHANNELS * SAMPLE_WIDTH  # Bytes per frame
    
    # Chunk sizes for streaming
    DEFAULT_CHUNK_MS = 100   # 100ms chunks
    MIN_CHUNK_MS = 10        # Minimum chunk size
    MAX_CHUNK_MS = 1000      # Maximum chunk size
    
    # Audio quality thresholds
    MIN_AMPLITUDE = 0.01     # Minimum amplitude for speech detection
    MAX_AMPLITUDE = 0.95     # Maximum amplitude before clipping


class AudioValidator:
    """Validates audio data against OpenAI Realtime API requirements"""
    
    @staticmethod
    def validate_wav_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate WAV file format compatibility
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # Check channels
                if wav_file.getnchannels() != AudioConfig.CHANNELS:
                    return False, f"Expected {AudioConfig.CHANNELS} channel(s), got {wav_file.getnchannels()}"
                
                # Check sample width
                if wav_file.getsampwidth() != AudioConfig.SAMPLE_WIDTH:
                    return False, f"Expected {AudioConfig.SAMPLE_WIDTH} byte sample width, got {wav_file.getsampwidth()}"
                
                # Check sample rate
                if wav_file.getframerate() != AudioConfig.SAMPLE_RATE:
                    return False, f"Expected {AudioConfig.SAMPLE_RATE}Hz sample rate, got {wav_file.getframerate()}"
                
                # Check if file has frames
                if wav_file.getnframes() == 0:
                    return False, "WAV file contains no audio data"
                
                return True, "Valid WAV file"
                
        except Exception as e:
            return False, f"Failed to validate WAV file: {e}"
    
    @staticmethod
    def validate_audio_data(audio_data: bytes, format_type: AudioFormat) -> Tuple[bool, str]:
        """
        Validate raw audio data
        
        Args:
            audio_data: Raw audio bytes
            format_type: Expected audio format
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not audio_data:
            return False, "Audio data is empty"
        
        if format_type == AudioFormat.PCM16:
            # Check if data length is valid for PCM16
            if len(audio_data) % AudioConfig.FRAME_SIZE != 0:
                return False, f"PCM16 data length must be multiple of {AudioConfig.FRAME_SIZE} bytes"
            
            # Check for reasonable duration (not too short or too long)
            duration_ms = len(audio_data) / (AudioConfig.SAMPLE_RATE * AudioConfig.FRAME_SIZE) * 1000
            if duration_ms < 10:  # Less than 10ms
                return False, f"Audio too short: {duration_ms:.1f}ms"
            if duration_ms > 300000:  # More than 5 minutes
                return False, f"Audio too long: {duration_ms:.1f}ms"
        
        return True, "Valid audio data"


class AudioConverter:
    """Converts between different audio formats"""
    
    @staticmethod
    def resample_audio(audio_data: bytes, source_rate: int, target_rate: int) -> bytes:
        """
        Resample audio data to target sample rate
        
        Args:
            audio_data: Raw PCM16 audio data
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if not HAS_NUMPY:
            raise AudioError("NumPy required for audio resampling")
        
        if source_rate == target_rate:
            return audio_data
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate resampling ratio
        ratio = target_rate / source_rate
        new_length = int(len(audio_array) * ratio)
        
        # Simple linear interpolation resampling
        old_indices = np.linspace(0, len(audio_array) - 1, new_length)
        new_audio = np.interp(old_indices, np.arange(len(audio_array)), audio_array)
        
        # Convert back to int16 and bytes
        return new_audio.astype(np.int16).tobytes()
    
    @staticmethod
    def convert_to_mono(audio_data: bytes, channels: int) -> bytes:
        """
        Convert multi-channel audio to mono
        
        Args:
            audio_data: Raw PCM16 audio data
            channels: Number of source channels
            
        Returns:
            Mono audio data
        """
        if channels == 1:
            return audio_data
        
        if not HAS_NUMPY:
            raise AudioError("NumPy required for channel conversion")
        
        # Convert to numpy array and reshape
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        multi_channel = audio_array.reshape(-1, channels)
        
        # Average all channels to create mono
        mono_audio = np.mean(multi_channel, axis=1, dtype=np.int16)
        
        return mono_audio.tobytes()
    
    @staticmethod
    def normalize_volume(audio_data: bytes, target_level: float = 0.8) -> bytes:
        """
        Normalize audio volume to target level
        
        Args:
            audio_data: Raw PCM16 audio data
            target_level: Target volume level (0.0 to 1.0)
            
        Returns:
            Volume-normalized audio data
        """
        if not HAS_NUMPY:
            raise AudioError("NumPy required for volume normalization")
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Calculate current peak
        peak = np.max(np.abs(audio_array))
        if peak == 0:
            return audio_data  # Silent audio
        
        # Calculate scaling factor
        target_peak = target_level * 32767  # Max value for int16
        scale_factor = target_peak / peak
        
        # Apply scaling and convert back
        normalized = (audio_array * scale_factor).astype(np.int16)
        
        return normalized.tobytes()


class AudioProcessor:
    """Main audio processing class for RealtimeVoiceAPI"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validator = AudioValidator()
        self.converter = AudioConverter()
    
    # Base64 encoding/decoding
    @staticmethod
    def bytes_to_base64(audio_bytes: bytes) -> str:
        """Convert audio bytes to base64 string for API transmission"""
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    @staticmethod
    def base64_to_bytes(audio_b64: str) -> bytes:
        """Convert base64 string to audio bytes"""
        try:
            return base64.b64decode(audio_b64)
        except Exception as e:
            raise AudioError(f"Failed to decode base64 audio: {e}")
    
    # WAV file operations
    def load_wav_file(self, file_path: Union[str, Path]) -> bytes:
        """
        Load WAV file and return raw audio bytes
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            Raw PCM16 audio bytes compatible with Realtime API
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AudioError(f"Audio file not found: {file_path}")
        
        # Validate file format first
        is_valid, error_msg = self.validator.validate_wav_file(file_path)
        
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # If file doesn't match exact requirements, try to convert
                if not is_valid:
                    self.logger.warning(f"WAV file format mismatch: {error_msg}")
                    return self._convert_wav_to_realtime_format(wav_file)
                
                # File is already in correct format
                audio_data = wav_file.readframes(wav_file.getnframes())
                self.logger.info(f"Loaded WAV file: {file_path} ({len(audio_data)} bytes)")
                return audio_data
                
        except Exception as e:
            raise AudioError(f"Failed to load WAV file {file_path}: {e}")
    
    def _convert_wav_to_realtime_format(self, wav_file: wave.Wave_read) -> bytes:
        """Convert WAV file to Realtime API format"""
        audio_data = wav_file.readframes(wav_file.getnframes())
        
        # Convert sample rate if needed
        if wav_file.getframerate() != AudioConfig.SAMPLE_RATE:
            audio_data = self.converter.resample_audio(
                audio_data, 
                wav_file.getframerate(), 
                AudioConfig.SAMPLE_RATE
            )
        
        # Convert to mono if needed
        if wav_file.getnchannels() != AudioConfig.CHANNELS:
            audio_data = self.converter.convert_to_mono(
                audio_data, 
                wav_file.getnchannels()
            )
        
        # Handle sample width conversion if needed
        if wav_file.getsampwidth() != AudioConfig.SAMPLE_WIDTH:
            audio_data = self._convert_sample_width(
                audio_data, 
                wav_file.getsampwidth(), 
                AudioConfig.SAMPLE_WIDTH
            )
        
        return audio_data
    
    def _convert_sample_width(self, audio_data: bytes, source_width: int, target_width: int) -> bytes:
        """Convert between different sample widths"""
        if source_width == target_width:
            return audio_data
        
        if not HAS_NUMPY:
            raise AudioError("NumPy required for sample width conversion")
        
        # Convert based on source width
        if source_width == 1:  # 8-bit to 16-bit
            audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.int16)
            audio_array = (audio_array - 128) * 256  # Convert unsigned 8-bit to signed 16-bit
        elif source_width == 3:  # 24-bit to 16-bit
            # 24-bit is more complex, read as bytes and convert
            audio_array = []
            for i in range(0, len(audio_data), 3):
                if i + 2 < len(audio_data):
                    # Read 24-bit sample (little-endian)
                    sample = int.from_bytes(audio_data[i:i+3], byteorder='little', signed=True)
                    # Convert to 16-bit by shifting
                    sample_16 = sample >> 8
                    audio_array.append(sample_16)
            audio_array = np.array(audio_array, dtype=np.int16)
        elif source_width == 4:  # 32-bit to 16-bit
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
            audio_array = (audio_array >> 16).astype(np.int16)  # Convert 32-bit to 16-bit
        else:
            raise AudioError(f"Unsupported sample width conversion: {source_width} -> {target_width}")
        
        return audio_array.tobytes()
    
    def save_wav_file(self, audio_bytes: bytes, file_path: Union[str, Path], 
                      format_type: AudioFormat = AudioFormat.PCM16):
        """
        Save raw audio bytes as WAV file
        
        Args:
            audio_bytes: Raw audio data
            file_path: Output file path
            format_type: Audio format of the input data
        """
        file_path = Path(file_path)
        
        # Validate audio data
        is_valid, error_msg = self.validator.validate_audio_data(audio_bytes, format_type)
        if not is_valid:
            raise AudioError(f"Invalid audio data: {error_msg}")
        
        try:
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(AudioConfig.CHANNELS)
                wav_file.setsampwidth(AudioConfig.SAMPLE_WIDTH)
                wav_file.setframerate(AudioConfig.SAMPLE_RATE)
                wav_file.writeframes(audio_bytes)
            
            self.logger.info(f"Saved WAV file: {file_path} ({len(audio_bytes)} bytes)")
            
        except Exception as e:
            raise AudioError(f"Failed to save WAV file {file_path}: {e}")
    
    # Audio chunking for streaming
    def chunk_audio(self, audio_bytes: bytes, chunk_size_ms: int = AudioConfig.DEFAULT_CHUNK_MS) -> List[bytes]:
        """
        Split audio into chunks for streaming
        
        Args:
            audio_bytes: Raw audio data
            chunk_size_ms: Chunk size in milliseconds
            
        Returns:
            List of audio chunks
        """
        if chunk_size_ms < AudioConfig.MIN_CHUNK_MS:
            chunk_size_ms = AudioConfig.MIN_CHUNK_MS
        elif chunk_size_ms > AudioConfig.MAX_CHUNK_MS:
            chunk_size_ms = AudioConfig.MAX_CHUNK_MS
        
        # Calculate bytes per chunk
        bytes_per_ms = (AudioConfig.SAMPLE_RATE * AudioConfig.FRAME_SIZE) // 1000
        chunk_size_bytes = chunk_size_ms * bytes_per_ms
        
        # Ensure chunk size is aligned to frame boundaries
        chunk_size_bytes = (chunk_size_bytes // AudioConfig.FRAME_SIZE) * AudioConfig.FRAME_SIZE
        
        chunks = []
        for i in range(0, len(audio_bytes), chunk_size_bytes):
            chunk = audio_bytes[i:i + chunk_size_bytes]
            if len(chunk) > 0:  # Skip empty chunks
                chunks.append(chunk)
        
        self.logger.debug(f"Split audio into {len(chunks)} chunks of ~{chunk_size_ms}ms each")
        return chunks
    
    def get_audio_duration_ms(self, audio_bytes: bytes) -> float:
        """
        Calculate audio duration in milliseconds
        
        Args:
            audio_bytes: Raw PCM16 audio data
            
        Returns:
            Duration in milliseconds
        """
        if not audio_bytes:
            return 0.0
        
        num_frames = len(audio_bytes) // AudioConfig.FRAME_SIZE
        duration_seconds = num_frames / AudioConfig.SAMPLE_RATE
        return duration_seconds * 1000
    
    def get_audio_info(self, audio_bytes: bytes) -> dict:
        """
        Get comprehensive audio information
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Dictionary with audio metadata
        """
        info = {
            'size_bytes': len(audio_bytes),
            'duration_ms': self.get_audio_duration_ms(audio_bytes),
            'sample_rate': AudioConfig.SAMPLE_RATE,
            'channels': AudioConfig.CHANNELS,
            'sample_width': AudioConfig.SAMPLE_WIDTH,
            'format': 'PCM16'
        }
        
        if HAS_NUMPY and audio_bytes:
            # Calculate additional statistics
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            info.update({
                'num_samples': len(audio_array),
                'peak_amplitude': float(np.max(np.abs(audio_array)) / 32767),
                'rms_amplitude': float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)) / 32767),
                'is_silent': np.max(np.abs(audio_array)) < 100  # Very low threshold for silence
            })
        
        return info
    
    

    def validate_realtime_api_format(self, audio_bytes: bytes) -> Tuple[bool, str]:
        """
        Validate audio format against exact OpenAI Realtime API requirements
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check basic requirements
            if not audio_bytes:
                return False, "Audio data is empty"
            
            # Must be even number of bytes (16-bit samples)
            if len(audio_bytes) % 2 != 0:
                return False, "Audio must have even number of bytes for 16-bit samples"
            
            # Check duration (minimum 100ms as per API)
            duration_ms = self.get_audio_duration_ms(audio_bytes)
            if duration_ms < 100:
                return False, f"Audio too short: {duration_ms:.1f}ms (minimum 100ms required)"
            
            # Check maximum size (15 MiB as per API)
            max_size = 15 * 1024 * 1024  # 15 MiB
            if len(audio_bytes) > max_size:
                return False, f"Audio too large: {len(audio_bytes)} bytes (maximum {max_size} bytes)"
            
            # Validate sample format by checking byte order and amplitude
            if HAS_NUMPY:
                import struct
                
                # Check first few samples for reasonable values
                num_samples_to_check = min(10, len(audio_bytes) // 2)
                samples = struct.unpack(f'<{num_samples_to_check}h', audio_bytes[:num_samples_to_check * 2])
                
                # Check for reasonable amplitude range
                max_amplitude = max(abs(s) for s in samples)
                if max_amplitude == 0:
                    return False, "Audio appears to be silent (all zero samples)"
                
                if max_amplitude > 32767:
                    return False, f"Audio amplitude too high: {max_amplitude} (maximum 32767 for 16-bit)"
                
                # Check for endianness issues (very large values might indicate wrong endianness)
                if max_amplitude > 30000 and all(abs(s) > 25000 for s in samples):
                    return False, "Audio might have endianness issues (all samples very high amplitude)"
            
            return True, "Valid Realtime API format"
            
        except Exception as e:
            return False, f"Format validation error: {e}"


    def ensure_little_endian_pcm16(self, audio_bytes: bytes) -> bytes:
        """
        Ensure audio is in little-endian 16-bit PCM format
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Audio data in correct format
        """
        if not HAS_NUMPY:
            # If no NumPy, assume format is correct
            return audio_bytes
        
        try:
            import numpy as np
            
            # Convert to numpy array assuming little-endian
            audio_array = np.frombuffer(audio_bytes, dtype='<i2')  # little-endian 16-bit signed
            
            # Check if this seems reasonable
            max_val = np.max(np.abs(audio_array))
            
            if max_val > 32767:
                # Try big-endian interpretation
                audio_array = np.frombuffer(audio_bytes, dtype='>i2')  # big-endian 16-bit signed
                max_val_be = np.max(np.abs(audio_array))
                
                if max_val_be < max_val:
                    # Big-endian interpretation gives more reasonable values
                    self.logger.warning("Converting audio from big-endian to little-endian")
                    # Convert to little-endian
                    audio_array = audio_array.astype('<i2')
                    return audio_array.tobytes()
            
            # Return as little-endian
            return audio_array.astype('<i2').tobytes()
            
        except Exception as e:
            self.logger.warning(f"Endianness conversion failed: {e}, returning original data")
            return audio_bytes
    
    # Audio quality analysis
    def analyze_audio_quality(self, audio_bytes: bytes) -> dict:
        """
        Analyze audio quality and detect potential issues
        
        Args:
            audio_bytes: Raw PCM16 audio data
            
        Returns:
            Dictionary with quality analysis results
        """
        if not HAS_NUMPY:
            return {'error': 'NumPy required for audio analysis'}
        
        if not audio_bytes:
            return {'error': 'No audio data provided'}
        
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
        
        analysis = {
            'duration_ms': self.get_audio_duration_ms(audio_bytes),
            'peak_level': float(np.max(np.abs(audio_array))),
            'rms_level': float(np.sqrt(np.mean(audio_array ** 2))),
            'dynamic_range': 0.0,
            'clipping_detected': False,
            'silence_detected': False,
            'quality_score': 1.0,
            'warnings': []
        }
        
        # Check for clipping
        if analysis['peak_level'] > AudioConfig.MAX_AMPLITUDE:
            analysis['clipping_detected'] = True
            analysis['warnings'].append('Audio clipping detected')
            analysis['quality_score'] *= 0.7
        
        # Check for silence
        if analysis['rms_level'] < AudioConfig.MIN_AMPLITUDE:
            analysis['silence_detected'] = True
            analysis['warnings'].append('Very low audio level detected')
            analysis['quality_score'] *= 0.8
        
        # Calculate dynamic range
        if len(audio_array) > AudioConfig.SAMPLE_RATE:  # At least 1 second of audio
            # Calculate RMS in 100ms windows
            window_size = AudioConfig.SAMPLE_RATE // 10  # 100ms
            windows = [audio_array[i:i+window_size] for i in range(0, len(audio_array), window_size)]
            rms_values = [np.sqrt(np.mean(window ** 2)) for window in windows if len(window) == window_size]
            
            if rms_values:
                max_rms = np.max(rms_values)
                min_rms = np.min([rms for rms in rms_values if rms > 0.001])  # Ignore near-silence
                analysis['dynamic_range'] = float(max_rms / min_rms) if min_rms > 0 else 0.0
        
        # Overall quality assessment
        if len(analysis['warnings']) == 0:
            analysis['quality_score'] = 1.0
        
        return analysis
    
    # Format conversion utilities
    def convert_from_any_format(self, file_path: Union[str, Path]) -> bytes:
        """
        Convert any supported audio format to Realtime API format
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Raw PCM16 audio bytes
        """
        if not HAS_PYDUB:
            # Fall back to WAV-only support
            return self.load_wav_file(file_path)
        
        try:
            # Use pydub for format detection and conversion
            audio = AudioSegment.from_file(str(file_path))
            
            # Convert to required format
            audio = audio.set_frame_rate(AudioConfig.SAMPLE_RATE)
            audio = audio.set_channels(AudioConfig.CHANNELS)
            audio = audio.set_sample_width(AudioConfig.SAMPLE_WIDTH)
            
            # Get raw audio data
            audio_bytes = audio.raw_data
            
            self.logger.info(f"Converted audio file: {file_path} -> {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            raise AudioError(f"Failed to convert audio file {file_path}: {e}")
    
    # Real-time streaming utilities
    def create_audio_stream_buffer(self, max_duration_ms: int = 5000) -> 'AudioStreamBuffer':
        """
        Create a buffer for real-time audio streaming
        
        Args:
            max_duration_ms: Maximum buffer duration in milliseconds
            
        Returns:
            AudioStreamBuffer instance
        """
        return AudioStreamBuffer(max_duration_ms, self.logger)


class AudioStreamBuffer:
    """Buffer for managing real-time audio streaming"""
    
    def __init__(self, max_duration_ms: int = 5000, logger: Optional[logging.Logger] = None):
        self.max_duration_ms = max_duration_ms
        self.logger = logger or logging.getLogger(__name__)
        
        # Calculate max buffer size in bytes
        bytes_per_ms = (AudioConfig.SAMPLE_RATE * AudioConfig.FRAME_SIZE) // 1000
        self.max_buffer_size = max_duration_ms * bytes_per_ms
        
        # Internal buffer
        self.buffer = bytearray()
        self.total_bytes_added = 0
        self.total_bytes_consumed = 0
    
    def add_audio(self, audio_bytes: bytes) -> bool:
        """
        Add audio data to buffer
        
        Args:
            audio_bytes: Raw audio data to add
            
        Returns:
            True if added successfully, False if buffer would overflow
        """
        if len(self.buffer) + len(audio_bytes) > self.max_buffer_size:
            # Buffer would overflow, remove old data
            overflow = len(self.buffer) + len(audio_bytes) - self.max_buffer_size
            self.buffer = self.buffer[overflow:]
            self.total_bytes_consumed += overflow
            self.logger.warning(f"Audio buffer overflow, removed {overflow} bytes")
        
        self.buffer.extend(audio_bytes)
        self.total_bytes_added += len(audio_bytes)
        return True
    
    def get_chunk(self, chunk_size_ms: int = AudioConfig.DEFAULT_CHUNK_MS) -> Optional[bytes]:
        """
        Get audio chunk from buffer
        
        Args:
            chunk_size_ms: Desired chunk size in milliseconds
            
        Returns:
            Audio chunk or None if not enough data
        """
        bytes_per_ms = (AudioConfig.SAMPLE_RATE * AudioConfig.FRAME_SIZE) // 1000
        chunk_size_bytes = chunk_size_ms * bytes_per_ms
        
        if len(self.buffer) < chunk_size_bytes:
            return None
        
        # Extract chunk
        chunk = bytes(self.buffer[:chunk_size_bytes])
        self.buffer = self.buffer[chunk_size_bytes:]
        self.total_bytes_consumed += len(chunk)
        
        return chunk
    
    def get_available_duration_ms(self) -> float:
        """Get duration of audio currently in buffer"""
        bytes_per_ms = (AudioConfig.SAMPLE_RATE * AudioConfig.FRAME_SIZE) // 1000
        return len(self.buffer) / bytes_per_ms
    
    def clear(self):
        """Clear all audio from buffer"""
        cleared_bytes = len(self.buffer)
        self.buffer.clear()
        self.total_bytes_consumed += cleared_bytes
        self.logger.debug(f"Cleared {cleared_bytes} bytes from audio buffer")
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            'current_size_bytes': len(self.buffer),
            'current_duration_ms': self.get_available_duration_ms(),
            'max_duration_ms': self.max_duration_ms,
            'total_bytes_added': self.total_bytes_added,
            'total_bytes_consumed': self.total_bytes_consumed,
            'buffer_utilization': len(self.buffer) / self.max_buffer_size
        }


# Convenience functions for quick operations
def load_audio_file(file_path: Union[str, Path]) -> bytes:
    """Quick function to load any audio file"""
    processor = AudioProcessor()
    try:
        return processor.convert_from_any_format(file_path)
    except:
        # Fall back to WAV loading
        return processor.load_wav_file(file_path)


def save_audio_file(audio_bytes: bytes, file_path: Union[str, Path]):
    """Quick function to save audio as WAV"""
    processor = AudioProcessor()
    processor.save_wav_file(audio_bytes, file_path)


def validate_audio_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """Quick function to validate audio file"""
    validator = AudioValidator()
    return validator.validate_wav_file(file_path)


def get_audio_info(audio_bytes: bytes) -> dict:
    """Quick function to get audio information"""
    processor = AudioProcessor()
    return processor.get_audio_info(audio_bytes)




# In audio.py
class LocalVAD:
    """Local voice activity detection for pre-filtering"""
    
    def __init__(self, energy_threshold=0.02, zero_crossing_threshold=0.1):
        self.energy_threshold = energy_threshold
        self.zero_crossing_threshold = zero_crossing_threshold
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Detect if audio chunk contains speech"""
        if not HAS_NUMPY:
            return True  # Pass through if no NumPy
        
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32767
        
        # Energy-based detection
        energy = np.sqrt(np.mean(audio_array ** 2))
        if energy < self.energy_threshold:
            return False
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / (2 * len(audio_array))
        
        return zero_crossings > self.zero_crossing_threshold



# # Add to audio.py
# class AudioQualityMonitor:
#     def __init__(self, sample_rate=24000):
#         self.sample_rate = sample_rate
#         self.noise_floor = None
#         self.signal_history = []
        
#     def analyze_realtime(self, audio_chunk: bytes) -> Dict[str, Any]:
#         """Real-time audio quality analysis"""
#         if not HAS_NUMPY:
#             return {"status": "monitoring_disabled"}
            
#         audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
#         rms = np.sqrt(np.mean(audio_array.astype(float)**2))
        
#         # Adaptive noise floor estimation
#         if self.noise_floor is None:
#             self.noise_floor = rms
#         else:
#             self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            
#         snr = 20 * np.log10(rms / (self.noise_floor + 1e-10))
        
#         return {
#             "rms_level": float(rms),
#             "snr_db": float(snr),
#             "quality_score": min(1.0, max(0.0, (snr + 10) / 30)),
#             "clipping_detected": np.any(np.abs(audio_array) > 32000),
#             "recommendation": self._get_quality_recommendation(snr)
#         }



def main():
    """
    Standalone smoke test for audio processing functionality
    
    Tests all major features to ensure the audio module is working correctly.
    """
    import tempfile
    import os
    from pathlib import Path
    
    print("🎵 RealtimeVoiceAPI Audio Module Smoke Test")
    print("=" * 50)
    
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Test results tracking
    tests_passed = 0
    tests_failed = 0
    
    def run_test(test_name: str, test_func):
        """Helper to run a test and track results"""
        nonlocal tests_passed, tests_failed
        print(f"\n🧪 Testing {test_name}...")
        try:
            test_func()
            print(f"✅ {test_name} - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {e}")
            tests_failed += 1
    
    # Test 1: Create synthetic audio data
    def test_synthetic_audio():
        """Test creating synthetic audio data"""
        if not HAS_NUMPY:
            print("⚠️  NumPy not available, skipping synthetic audio generation")
            return
        
        # Generate 1 second of 440Hz sine wave (A note)
        duration_seconds = 1.0
        frequency = 440.0
        
        sample_count = int(AudioConfig.SAMPLE_RATE * duration_seconds)
        t = np.linspace(0, duration_seconds, sample_count)
        audio_array = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_array = (audio_array * 32767 * 0.5).astype(np.int16)  # 50% volume
        audio_bytes = audio_array.tobytes()
        
        # Validate the generated audio
        processor = AudioProcessor()
        info = processor.get_audio_info(audio_bytes)
        
        assert info['duration_ms'] >= 990 and info['duration_ms'] <= 1010, f"Expected ~1000ms, got {info['duration_ms']}"
        assert info['size_bytes'] == len(audio_bytes), "Size mismatch"
        
        print(f"   Generated {len(audio_bytes)} bytes of 440Hz sine wave")
        print(f"   Duration: {info['duration_ms']:.1f}ms")
        return audio_bytes
    
    # Test 2: Audio validation
    def test_audio_validation():
        """Test audio validation functionality"""
        processor = AudioProcessor()
        
        # Test with empty data
        is_valid, msg = processor.validator.validate_audio_data(b"", AudioFormat.PCM16)
        assert not is_valid, "Empty audio should be invalid"
        
        # Test with valid data (if we have synthetic audio from previous test)
        if 'synthetic_audio' in locals():
            is_valid, msg = processor.validator.validate_audio_data(synthetic_audio, AudioFormat.PCM16)
            assert is_valid, f"Valid audio should pass validation: {msg}"
        
        print("   ✓ Empty audio correctly rejected")
        print("   ✓ Valid audio correctly accepted")
    
    # Test 3: WAV file operations
    def test_wav_operations():
        """Test WAV file saving and loading"""
        processor = AudioProcessor()
        
        # Create test audio if NumPy is available
        if HAS_NUMPY:
            # Generate short test tone
            duration = 0.5  # 500ms
            sample_count = int(AudioConfig.SAMPLE_RATE * duration)
            t = np.linspace(0, duration, sample_count)
            audio_array = np.sin(2 * np.pi * 880 * t)  # 880Hz (A5 note)
            audio_array = (audio_array * 32767 * 0.3).astype(np.int16)
            test_audio = audio_array.tobytes()
        else:
            # Create simple test pattern without NumPy
            # 100ms of alternating high/low samples
            pattern_length = AudioConfig.SAMPLE_RATE // 10  # 100ms
            pattern = []
            for i in range(pattern_length):
                if i % 100 < 50:
                    pattern.extend([10000, -10000])  # High amplitude
                else:
                    pattern.extend([5000, -5000])    # Lower amplitude
            
            # Convert to bytes
            test_audio = b''.join(struct.pack('<h', sample) for sample in pattern)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save audio
            processor.save_wav_file(test_audio, temp_path)
            assert Path(temp_path).exists(), "WAV file was not created"
            
            # Load audio back
            loaded_audio = processor.load_wav_file(temp_path)
            assert len(loaded_audio) == len(test_audio), "Loaded audio size mismatch"
            assert loaded_audio == test_audio, "Loaded audio content mismatch"
            
            # Validate the file
            is_valid, msg = processor.validator.validate_wav_file(temp_path)
            assert is_valid, f"Saved WAV file should be valid: {msg}"
            
            file_size = Path(temp_path).stat().st_size
            print(f"   ✓ Saved and loaded {len(test_audio)} bytes as WAV ({file_size} bytes on disk)")
            
        finally:
            # Cleanup
            if Path(temp_path).exists():
                os.unlink(temp_path)
    
    # Test 4: Audio chunking
    def test_audio_chunking():
        """Test audio chunking for streaming"""
        processor = AudioProcessor()
        
        # Create test audio (2 seconds)
        if HAS_NUMPY:
            duration = 2.0
            sample_count = int(AudioConfig.SAMPLE_RATE * duration)
            audio_array = np.random.randint(-1000, 1000, sample_count, dtype=np.int16)
            test_audio = audio_array.tobytes()
        else:
            # Simple pattern
            sample_count = AudioConfig.SAMPLE_RATE * 2  # 2 seconds
            test_audio = b''.join(struct.pack('<h', i % 1000) for i in range(sample_count))
        
        # Test different chunk sizes
        for chunk_ms in [50, 100, 200]:
            chunks = processor.chunk_audio(test_audio, chunk_ms)
            
            # Calculate expected number of chunks
            total_duration_ms = processor.get_audio_duration_ms(test_audio)
            expected_chunks = int(total_duration_ms / chunk_ms)
            
            assert len(chunks) >= expected_chunks - 1, f"Too few chunks for {chunk_ms}ms"
            assert len(chunks) <= expected_chunks + 1, f"Too many chunks for {chunk_ms}ms"
            
            # Verify chunk sizes
            for i, chunk in enumerate(chunks[:-1]):  # Skip last chunk (may be shorter)
                chunk_duration = processor.get_audio_duration_ms(chunk)
                assert abs(chunk_duration - chunk_ms) < 5, f"Chunk {i} duration mismatch"
            
            print(f"   ✓ {chunk_ms}ms chunks: {len(chunks)} chunks from {total_duration_ms:.1f}ms audio")
    
    # Test 5: Base64 encoding/decoding
    def test_base64_operations():
        """Test base64 encoding and decoding"""
        # Create test data
        test_data = b"Hello, audio world!" + b"\x00\x01\x02\x03\xFF\xFE"
        
        # Test encoding
        encoded = AudioProcessor.bytes_to_base64(test_data)
        assert isinstance(encoded, str), "Encoded data should be string"
        assert len(encoded) > 0, "Encoded string should not be empty"
        
        # Test decoding
        decoded = AudioProcessor.base64_to_bytes(encoded)
        assert decoded == test_data, "Decoded data should match original"
        
        print(f"   ✓ Encoded {len(test_data)} bytes to {len(encoded)} chars")
        print(f"   ✓ Decoded back to {len(decoded)} bytes (match: {decoded == test_data})")
    
    # Test 6: Audio info and duration calculation
    def test_audio_info():
        """Test audio information extraction"""
        processor = AudioProcessor()
        
        # Create known duration audio
        if HAS_NUMPY:
            duration_seconds = 1.5
            sample_count = int(AudioConfig.SAMPLE_RATE * duration_seconds)
            audio_array = np.random.randint(-100, 100, sample_count, dtype=np.int16)
            test_audio = audio_array.tobytes()
        else:
            # 1.5 seconds of simple pattern
            sample_count = int(AudioConfig.SAMPLE_RATE * 1.5)
            test_audio = b''.join(struct.pack('<h', i % 100) for i in range(sample_count))
        
        # Get audio info
        info = processor.get_audio_info(test_audio)
        
        expected_duration = 1500.0  # 1.5 seconds in ms
        assert abs(info['duration_ms'] - expected_duration) < 1, f"Duration mismatch: {info['duration_ms']} vs {expected_duration}"
        assert info['sample_rate'] == AudioConfig.SAMPLE_RATE, "Sample rate mismatch"
        assert info['channels'] == AudioConfig.CHANNELS, "Channels mismatch"
        
        print(f"   ✓ Audio info: {info['duration_ms']:.1f}ms, {info['size_bytes']} bytes")
        if HAS_NUMPY:
            print(f"   ✓ Peak amplitude: {info.get('peak_amplitude', 'N/A'):.3f}")
    
    # Test 7: Audio quality analysis (if NumPy available)
    def test_audio_quality():
        """Test audio quality analysis"""
        if not HAS_NUMPY:
            print("   ⚠️  NumPy not available, skipping quality analysis")
            return
        
        processor = AudioProcessor()
        
        # Create test audio with known characteristics
        duration = 1.0
        sample_count = int(AudioConfig.SAMPLE_RATE * duration)
        
        # Normal audio
        t = np.linspace(0, duration, sample_count)
        normal_audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 50% amplitude
        normal_audio = (normal_audio * 32767).astype(np.int16)
        
        analysis = processor.analyze_audio_quality(normal_audio.tobytes())
        assert analysis['quality_score'] > 0.9, "Normal audio should have high quality score"
        assert not analysis['clipping_detected'], "Normal audio should not have clipping"
        assert not analysis['silence_detected'], "Normal audio should not be silent"
        
        print(f"   ✓ Normal audio quality score: {analysis['quality_score']:.2f}")
        
        # Clipped audio
        clipped_audio = np.ones(sample_count, dtype=np.int16) * 32767  # Maximum amplitude
        analysis = processor.analyze_audio_quality(clipped_audio.tobytes())
        assert analysis['clipping_detected'], "Clipped audio should be detected"
        assert analysis['quality_score'] < 0.8, "Clipped audio should have lower quality score"
        
        print(f"   ✓ Clipped audio detected, quality score: {analysis['quality_score']:.2f}")
        
        # Silent audio
        silent_audio = np.zeros(sample_count, dtype=np.int16)
        analysis = processor.analyze_audio_quality(silent_audio.tobytes())
        assert analysis['silence_detected'], "Silent audio should be detected"
        
        print(f"   ✓ Silent audio detected, quality score: {analysis['quality_score']:.2f}")
    
    # Test 8: Audio stream buffer
    def test_stream_buffer():
        """Test real-time audio streaming buffer"""
        processor = AudioProcessor()
        buffer = processor.create_audio_stream_buffer(max_duration_ms=1000)  # 1 second buffer
        
        # Create test chunks
        chunk_ms = 100
        bytes_per_ms = (AudioConfig.SAMPLE_RATE * AudioConfig.FRAME_SIZE) // 1000
        chunk_size_bytes = chunk_ms * bytes_per_ms
        
        # Create test chunk
        if HAS_NUMPY:
            chunk_samples = chunk_size_bytes // 2  # 16-bit samples
            test_chunk = np.random.randint(-1000, 1000, chunk_samples, dtype=np.int16).tobytes()
        else:
            test_chunk = b''.join(struct.pack('<h', i % 1000) for i in range(chunk_size_bytes // 2))
        
        # Test buffer operations
        assert buffer.get_available_duration_ms() == 0, "New buffer should be empty"
        
        # Add chunks
        for i in range(5):
            success = buffer.add_audio(test_chunk)
            assert success, f"Should be able to add chunk {i}"
        
        duration = buffer.get_available_duration_ms()
        expected_duration = 5 * chunk_ms
        assert abs(duration - expected_duration) < 10, f"Buffer duration mismatch: {duration} vs {expected_duration}"
        
        # Get chunks back
        retrieved_chunks = 0
        while True:
            chunk = buffer.get_chunk(chunk_ms)
            if chunk is None:
                break
            retrieved_chunks += 1
            assert len(chunk) == len(test_chunk), "Retrieved chunk size mismatch"
        
        assert retrieved_chunks == 5, f"Should retrieve 5 chunks, got {retrieved_chunks}"
        
        stats = buffer.get_stats()
        print(f"   ✓ Buffer test: added 5 chunks, retrieved {retrieved_chunks}")
        print(f"   ✓ Buffer stats: {stats['total_bytes_added']} bytes added, {stats['total_bytes_consumed']} consumed")
    
    # Test 9: Check dependencies
    def test_dependencies():
        """Test optional dependencies"""
        print(f"   NumPy available: {'✅' if HAS_NUMPY else '❌'}")
        print(f"   Pydub available: {'✅' if HAS_PYDUB else '❌'}")
        
        if not HAS_NUMPY:
            print("   ⚠️  Install NumPy for advanced audio processing features")
        if not HAS_PYDUB:
            print("   ⚠️  Install pydub for multi-format audio support")
    
    # Run all tests
    synthetic_audio = None
    
    run_test("Dependencies Check", test_dependencies)
    
    def test_synthetic_wrapper():
        nonlocal synthetic_audio
        synthetic_audio = test_synthetic_audio()
    
    run_test("Synthetic Audio Generation", test_synthetic_wrapper)
    run_test("Audio Validation", test_audio_validation)
    run_test("WAV File Operations", test_wav_operations)
    run_test("Audio Chunking", test_audio_chunking)
    run_test("Base64 Operations", test_base64_operations)
    run_test("Audio Info Extraction", test_audio_info)
    run_test("Audio Quality Analysis", test_audio_quality)
    run_test("Stream Buffer", test_stream_buffer)
    
    # Test summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed == 0:
        print("🎉 All tests passed! Audio module is working correctly.")
        print("\n💡 Quick usage examples:")
        print("   from realtimevoiceapi.audio import load_audio_file, AudioProcessor")
        print("   audio_bytes = load_audio_file('my_audio.wav')")
        print("   processor = AudioProcessor()")
        print("   info = processor.get_audio_info(audio_bytes)")
        print("   chunks = processor.chunk_audio(audio_bytes, 100)  # 100ms chunks")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        print("\n🔧 Common issues:")
        print("   - Install missing dependencies: pip install numpy pydub")
        print("   - Check file permissions for temporary file tests")
        print("   - Ensure sufficient memory for audio processing")
        return False


if __name__ == "__main__":
    main()