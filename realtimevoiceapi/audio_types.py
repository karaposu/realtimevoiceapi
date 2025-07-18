
# here is realtimevoiceapi/audio_types.py


"""
Audio Type Definitions and Constants

Pure data definitions with zero runtime cost. Used by both fast and big lanes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any


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


# ============== Audio Configuration ==============

@dataclass
class AudioConfig:
    """Standard audio configuration for Realtime APIs"""
    
    # Standard requirements
    sample_rate: int = 24000      # 24kHz standard
    channels: int = 1             # Mono
    bit_depth: int = 16          # 16-bit
    
    # Chunk settings for streaming
    chunk_duration_ms: int = 100  # Default chunk size
    min_chunk_ms: int = 10       # Minimum chunk
    max_chunk_ms: int = 1000     # Maximum chunk
    
    # Quality thresholds
    min_amplitude: float = 0.01   # Minimum for speech detection
    max_amplitude: float = 0.95   # Maximum before clipping
    
    # Computed properties
    @property
    def frame_size(self) -> int:
        """Bytes per frame"""
        return self.channels * (self.bit_depth // 8)
    
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
        return int(duration_ms * self.bytes_per_ms)
    
    def duration_from_bytes(self, num_bytes: int) -> float:
        """Calculate duration in ms from byte count"""
        return num_bytes / self.bytes_per_ms


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


# ============== VAD Types ==============

class VADType(Enum):
    """Voice Activity Detection types"""
    
    NONE = "none"               # No VAD
    ENERGY_BASED = "energy"     # Simple energy threshold
    ZERO_CROSSING = "zcr"       # Zero-crossing rate
    COMBINED = "combined"       # Energy + ZCR
    ML_BASED = "ml"            # Machine learning based
    
    @property
    def is_local(self) -> bool:
        """Check if VAD runs locally"""
        return self != VADType.NONE


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
    
    def __post_init__(self):
        # Validate thresholds
        self.energy_threshold = max(0.0, min(1.0, self.energy_threshold))
        self.zcr_threshold = max(0.0, min(1.0, self.zcr_threshold))


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
    
    # Timing info
    timestamp: Optional[float] = None
    sequence_number: Optional[int] = None
    
    # Processing flags
    is_final: bool = False
    needs_processing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "format": self.format.value,
            "duration_ms": self.duration_ms,
            "size_bytes": self.size_bytes,
            "peak_amplitude": self.peak_amplitude,
            "rms_amplitude": self.rms_amplitude,
            "is_speech": self.is_speech,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "is_final": self.is_final
        }


# ============== Audio Error Types ==============

class AudioErrorType(Enum):
    """Types of audio processing errors"""
    
    FORMAT_ERROR = "format_error"
    VALIDATION_ERROR = "validation_error"
    CONVERSION_ERROR = "conversion_error"
    BUFFER_OVERFLOW = "buffer_overflow"
    BUFFER_UNDERFLOW = "buffer_underflow"
    QUALITY_ERROR = "quality_error"
    TIMEOUT = "timeout"


# ============== Buffer Configuration ==============

@dataclass
class BufferConfig:
    """Configuration for audio buffers"""
    
    # Size limits
    max_size_bytes: int = 1024 * 1024  # 1MB default
    max_duration_ms: int = 5000        # 5 seconds
    
    # Behavior
    overflow_strategy: Literal["drop_oldest", "drop_newest", "error"] = "drop_oldest"
    
    # Performance
    pre_allocate: bool = False         # Pre-allocate memory
    use_circular: bool = True          # Use circular buffer
    
    # Metrics
    track_metrics: bool = True         # Track buffer statistics


# ============== Audio Processing Modes ==============

class ProcessingMode(Enum):
    """Audio processing modes for different use cases"""
    
    REALTIME = "realtime"       # Minimal latency
    QUALITY = "quality"         # Best quality
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
    
    # Processing
    DEFAULT_CHUNK_MS = 100
    SILENCE_THRESHOLD_AMPLITUDE = 0.01


# ============== Type Aliases ==============

# For better code readability
AudioBytes = bytes
SampleRate = int
DurationMs = float
AmplitudeFloat = float  # 0.0 to 1.0
AmplitudeInt16 = int   # -32768 to 32767