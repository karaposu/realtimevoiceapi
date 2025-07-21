# core/audio_types.py

"""
Unified Audio Type Definitions and Constants

Single source of truth for all audio types in RealtimeVoiceAPI.
Optimized for zero-overhead type definitions.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, Union, List


# ============== Type Aliases ==============

# For better code readability and type safety
AudioBytes = bytes
SampleRate = int
DurationMs = float
AmplitudeFloat = float  # 0.0 to 1.0
AmplitudeInt16 = int   # -32768 to 32767


# ============== Audio Formats ==============

class AudioFormat(Enum):
    """Supported audio formats for Realtime APIs"""
    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"
    
    @property
    def bytes_per_sample(self) -> int:
        """Get bytes per sample for format"""
        if self == AudioFormat.PCM16:
            return 2
        return 1  # G.711 formats are 8-bit
    
    @property
    def requires_compression(self) -> bool:
        """Check if format uses compression"""
        return self in [AudioFormat.G711_ULAW, AudioFormat.G711_ALAW]
    
    def to_api_format(self) -> str:
        """Convert to API format string"""
        return self.value


# ============== Audio Configuration ==============

@dataclass
class AudioConfig:
    """Standard audio configuration for Realtime APIs"""
    
    # Standard requirements
    sample_rate: int = 24000      # 24kHz standard
    channels: int = 1             # Mono
    bit_depth: int = 16          # 16-bit
    format: AudioFormat = AudioFormat.PCM16
    
    # Chunk settings for streaming
    chunk_duration_ms: int = 100  # Default chunk size
    min_chunk_ms: int = 10       # Minimum chunk
    max_chunk_ms: int = 1000     # Maximum chunk
    
    # Quality thresholds
    min_amplitude: float = 0.01   # Minimum for speech detection
    max_amplitude: float = 0.95   # Maximum before clipping
    
    # Performance settings
    use_numpy: bool = True        # Use numpy if available
    pre_allocate_buffers: bool = True  # Pre-allocate for fast lane
    
    def __post_init__(self):
        """Validate configuration"""
        # Ensure chunk duration is within bounds
        self.chunk_duration_ms = max(
            self.min_chunk_ms,
            min(self.chunk_duration_ms, self.max_chunk_ms)
        )
    
    # Computed properties
   
    
    @property
    def frame_size(self) -> int:
        """Bytes per frame"""
        # Use format's bytes_per_sample if available, otherwise use bit_depth
        bytes_per_sample = self.format.bytes_per_sample if self.format else (self.bit_depth // 8)
        return self.channels * bytes_per_sample

    
    @property
    def bytes_per_second(self) -> int:
        """Bytes per second of audio"""
        return self.sample_rate * self.frame_size
    
    @property
    def bytes_per_ms(self) -> float:
        """Bytes per millisecond"""
        return self.bytes_per_second / 1000
    
    
    def chunk_size_bytes(self, duration_ms: int) -> int:
        """Calculate chunk size in bytes for given duration"""
        # Handle negative durations
        if duration_ms < 0:
            return 0
        return int(duration_ms * self.bytes_per_ms)
    
    def duration_from_bytes(self, num_bytes: int) -> float:
        """Calculate duration in ms from byte count"""
        return num_bytes / self.bytes_per_ms if self.bytes_per_ms > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API"""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "format": self.format.value,
            "chunk_duration_ms": self.chunk_duration_ms
        }


# ============== Audio Quality Levels ==============

class AudioQuality(Enum):
    """Audio quality presets"""
    
    LOW = "low"          # 16kHz, lower bitrate
    STANDARD = "standard" # 24kHz, standard quality
    HIGH = "high"        # 48kHz, high quality
    
    def to_config(self) -> AudioConfig:
        """Convert quality level to audio config"""
        if self == AudioQuality.LOW:
            return AudioConfig(sample_rate=16000)
        elif self == AudioQuality.HIGH:
            return AudioConfig(sample_rate=48000)
        else:
            return AudioConfig()  # Standard


# ============== Processing Modes ==============

class ProcessingMode(Enum):
    """Audio processing modes for different use cases"""
    
    REALTIME = "realtime"       # Minimal latency (fast lane)
    QUALITY = "quality"         # Best quality (big lane)
    BALANCED = "balanced"       # Balance of both
    
    @property
    def buffer_size_ms(self) -> int:
        """Recommended buffer size for mode"""
        if self == ProcessingMode.REALTIME:
            return 10
        elif self == ProcessingMode.QUALITY:
            return 200
        else:
            return 50
    
    @property
    def allows_numpy(self) -> bool:
        """Whether this mode should use numpy processing"""
        return self != ProcessingMode.REALTIME
    
    @property
    def max_latency_ms(self) -> float:
        """Maximum acceptable latency for mode"""
        if self == ProcessingMode.REALTIME:
            return 20.0
        elif self == ProcessingMode.QUALITY:
            return 500.0
        else:
            return 100.0


# ============== VAD Types and Configuration ==============

class VADType(Enum):
    """Voice Activity Detection types"""
    
    NONE = "none"               # No VAD
    ENERGY_BASED = "energy"     # Simple energy threshold
    ZERO_CROSSING = "zcr"       # Zero-crossing rate
    COMBINED = "combined"       # Energy + ZCR
    ML_BASED = "ml"            # Machine learning based
    SERVER_VAD = "server_vad"   # Server-side VAD (API)
    SEMANTIC_VAD = "semantic_vad"  # Semantic VAD (API)
    
    @property
    def is_local(self) -> bool:
        """Check if VAD runs locally"""
        return self not in [VADType.NONE, VADType.SERVER_VAD, VADType.SEMANTIC_VAD]
    
    @property
    def is_api_based(self) -> bool:
        """Check if VAD is API-based"""
        return self in [VADType.SERVER_VAD, VADType.SEMANTIC_VAD]


@dataclass
class VADConfig:
    """VAD configuration parameters"""
    
    type: VADType = VADType.ENERGY_BASED
    energy_threshold: float = 0.02
    zcr_threshold: float = 0.1
    
    # Timing parameters
    speech_start_ms: int = 100      # Time before confirming speech
    speech_end_ms: int = 500        # Silence before ending speech
    pre_buffer_ms: int = 300        # Buffer before speech starts
    
    # Advanced settings
    adaptive: bool = False          # Adaptive threshold
    noise_reduction: bool = False   # Apply noise reduction
    
    # API-specific settings
    create_response: bool = True    # For server/semantic VAD
    
    def __post_init__(self):
        # Validate thresholds
        self.energy_threshold = max(0.0, min(1.0, self.energy_threshold))
        self.zcr_threshold = max(0.0, min(1.0, self.zcr_threshold))
    
    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API format"""
        if self.type == VADType.SEMANTIC_VAD:
            return {
                "type": "semantic_vad",
                "create_response": self.create_response
            }
        elif self.type == VADType.SERVER_VAD:
            return {
                "type": "server_vad",
                "threshold": self.energy_threshold,
                "prefix_padding_ms": self.pre_buffer_ms,
                "silence_duration_ms": self.speech_end_ms,
                "create_response": self.create_response
            }
        else:
            # Local VAD config
            return {
                "type": self.type.value,
                "energy_threshold": self.energy_threshold,
                "zcr_threshold": self.zcr_threshold,
                "speech_start_ms": self.speech_start_ms,
                "speech_end_ms": self.speech_end_ms
            }


# ============== Audio Metadata ==============

@dataclass
class AudioMetadata:
    """Metadata for audio chunks/streams"""
    
    format: AudioFormat
    duration_ms: float
    size_bytes: int
    
    # Optional quality metrics
    peak_amplitude: Optional[float] = None
    rms_amplitude: Optional[float] = None
    is_speech: Optional[bool] = None
    signal_to_noise_ratio: Optional[float] = None
    
    # Timing info
    timestamp: Optional[float] = None
    sequence_number: Optional[int] = None
    
    # Processing flags
    is_final: bool = False
    needs_processing: bool = True
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "format": self.format.value,
            "duration_ms": self.duration_ms,
            "size_bytes": self.size_bytes,
            "is_final": self.is_final
        }
        
        # Add optional fields if present
        optional_fields = [
            "peak_amplitude", "rms_amplitude", "is_speech",
            "signal_to_noise_ratio", "timestamp", "sequence_number"
        ]
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                data[field] = value
        
        return data


# ============== Buffer Configuration ==============

@dataclass
class BufferConfig:
    """Configuration for audio buffers"""
    
    # Size limits
    max_size_bytes: int = 1024 * 1024  # 1MB default
    max_duration_ms: int = 5000        # 5 seconds
    
    # Behavior
    overflow_strategy: Literal["drop_oldest", "drop_newest", "error"] = "drop_oldest"
    underflow_strategy: Literal["silence", "repeat", "error"] = "silence"
    
    # Performance
    pre_allocate: bool = True          # Pre-allocate memory
    use_circular: bool = True          # Use circular buffer
    zero_copy: bool = True             # Enable zero-copy operations
    
    # Pool settings for fast lane
    pool_size: int = 10                # Number of pre-allocated buffers
    pool_buffer_size: int = 48000      # Size of each pooled buffer
    
    # Metrics
    track_metrics: bool = True         # Track buffer statistics
    
    def validate(self) -> bool:
        """Validate buffer configuration"""
        if self.max_size_bytes <= 0:
            return False
        if self.max_duration_ms <= 0:
            return False
        if self.pool_size < 0:
            return False
        return True


# ============== Audio Error Types ==============

class AudioErrorType(Enum):
    """Types of audio processing errors"""
    
    FORMAT_ERROR = "format_error"
    VALIDATION_ERROR = "validation_error"
    CONVERSION_ERROR = "conversion_error"
    BUFFER_OVERFLOW = "buffer_overflow"
    BUFFER_UNDERFLOW = "buffer_underflow"
    QUALITY_ERROR = "quality_error"
    DEVICE_ERROR = "device_error"
    TIMEOUT = "timeout"
    UNSUPPORTED_OPERATION = "unsupported_operation"


# ============== API Types ==============

# Voice types for API
VoiceType = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]

# Modality types
ModalityType = Literal["text", "audio"]

# Turn detection types (API)
TurnDetectionType = Literal["server_vad", "semantic_vad"]


# ============== Common Constants ==============

class AudioConstants:
    """Common audio processing constants"""
    
    # Supported sample rates
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 24000, 44100, 48000]
    
    # OpenAI Realtime API specifics
    OPENAI_SAMPLE_RATE = 24000
    OPENAI_CHANNELS = 1
    OPENAI_FORMAT = AudioFormat.PCM16
    
    # Limits
    MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25MB
    MAX_DURATION_MS = 300000  # 5 minutes
    MIN_CHUNK_DURATION_MS = 10
    MAX_CHUNK_DURATION_MS = 1000
    
    # Processing
    DEFAULT_CHUNK_MS = 100
    SILENCE_THRESHOLD_AMPLITUDE = 0.01
    
    # Fast lane optimizations
    FAST_LANE_CHUNK_MS = 20
    FAST_LANE_BUFFER_COUNT = 10
    FAST_LANE_MAX_LATENCY_MS = 50
    
    # Quality thresholds
    CLIPPING_THRESHOLD = 0.95
    NOISE_FLOOR_THRESHOLD = 0.02


# ============== Utility Functions ==============

def get_optimal_chunk_size(mode: ProcessingMode, sample_rate: int = 24000) -> int:
    """Get optimal chunk size in bytes for processing mode"""
    config = AudioConfig(sample_rate=sample_rate)
    
    if mode == ProcessingMode.REALTIME:
        duration_ms = AudioConstants.FAST_LANE_CHUNK_MS
    elif mode == ProcessingMode.QUALITY:
        duration_ms = 200
    else:
        duration_ms = AudioConstants.DEFAULT_CHUNK_MS
    
    return config.chunk_size_bytes(duration_ms)


def validate_audio_config(config: AudioConfig) -> tuple[bool, Optional[str]]:
    """Validate audio configuration"""
    if config.sample_rate not in AudioConstants.SUPPORTED_SAMPLE_RATES:
        return False, f"Unsupported sample rate: {config.sample_rate}"
    
    if config.channels not in [1, 2]:
        return False, f"Unsupported channel count: {config.channels}"
    
    if config.bit_depth not in [8, 16, 24, 32]:
        return False, f"Unsupported bit depth: {config.bit_depth}"
    
    if config.chunk_duration_ms < AudioConstants.MIN_CHUNK_DURATION_MS:
        return False, f"Chunk duration too small: {config.chunk_duration_ms}ms"
    
    return True, None