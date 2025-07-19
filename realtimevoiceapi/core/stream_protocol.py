
# here is realtimevoiceapi/core/stream_protocol.py

"""
Stream Protocol Definitions

Zero-cost abstractions defining contracts for stream handling across
fast and big lane implementations. Uses Python's Protocol for structural
typing - no runtime overhead, just type checking.
"""

from typing import Protocol, AsyncIterator, Optional, Dict, Any, List, Callable, Union
from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass


# ============== Stream Events ==============

class StreamEventType(str, Enum):
    """Common stream event types across all implementations"""
    
    # Stream lifecycle
    STREAM_STARTED = "stream.started"
    STREAM_READY = "stream.ready"
    STREAM_PAUSED = "stream.paused"
    STREAM_RESUMED = "stream.resumed"
    STREAM_ENDED = "stream.ended"
    STREAM_ERROR = "stream.error"
    
    # Audio events
    AUDIO_INPUT_STARTED = "audio.input.started"
    AUDIO_INPUT_CHUNK = "audio.input.chunk"
    AUDIO_INPUT_ENDED = "audio.input.ended"
    AUDIO_OUTPUT_STARTED = "audio.output.started"
    AUDIO_OUTPUT_CHUNK = "audio.output.chunk"
    AUDIO_OUTPUT_ENDED = "audio.output.ended"
    
    # Text events
    TEXT_INPUT_STARTED = "text.input.started"
    TEXT_INPUT_CHUNK = "text.input.chunk"
    TEXT_INPUT_ENDED = "text.input.ended"
    TEXT_OUTPUT_STARTED = "text.output.started"
    TEXT_OUTPUT_CHUNK = "text.output.chunk"
    TEXT_OUTPUT_ENDED = "text.output.ended"
    
    # VAD events
    VAD_SPEECH_STARTED = "vad.speech.started"
    VAD_SPEECH_ENDED = "vad.speech.ended"
    VAD_SOUND_DETECTED = "vad.sound.detected"
    
    # Response events
    RESPONSE_STARTED = "response.started"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_CANCELLED = "response.cancelled"
    
    # Function call events
    FUNCTION_CALL_STARTED = "function.call.started"
    FUNCTION_CALL_COMPLETED = "function.call.completed"
    FUNCTION_CALL_FAILED = "function.call.failed"
    
    # Cost events
    COST_UPDATED = "cost.updated"
    USAGE_TRACKED = "usage.tracked"


@dataclass
class StreamEvent:
    """Base stream event structure"""
    type: StreamEventType
    stream_id: str
    timestamp: float
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# ============== Stream States ==============

class StreamState(str, Enum):
    """Common stream states"""
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    BUFFERING = "buffering"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


# ============== Core Protocols ==============

class IStreamManager(Protocol):
    """
    Protocol for stream management.
    Both fast and big lane must implement this interface.
    """
    
    @property
    @abstractmethod
    def stream_id(self) -> str:
        """Unique identifier for this stream"""
        ...
    
    @property
    @abstractmethod
    def state(self) -> StreamState:
        """Current stream state"""
        ...
    
    @abstractmethod
    async def start(self) -> None:
        """Start the stream"""
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the stream gracefully"""
        ...
    
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data through the stream"""
        ...
    
    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text through the stream"""
        ...
    
    @abstractmethod
    def subscribe_events(
        self, 
        event_types: List[StreamEventType],
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Subscribe to stream events"""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream metrics/statistics"""
        ...


class IAudioHandler(Protocol):
    """
    Protocol for audio handling.
    Defines the contract for audio processing components.
    """
    
    @abstractmethod
    async def process_input_chunk(self, chunk: bytes) -> bytes:
        """Process incoming audio chunk"""
        ...
    
    @abstractmethod
    async def process_output_chunk(self, chunk: bytes) -> bytes:
        """Process outgoing audio chunk"""
        ...
    
    @abstractmethod
    def get_input_format(self) -> Dict[str, Any]:
        """Get expected input audio format"""
        ...
    
    @abstractmethod
    def get_output_format(self) -> Dict[str, Any]:
        """Get output audio format"""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset handler state"""
        ...


class ITextHandler(Protocol):
    """Protocol for text handling"""
    
    @abstractmethod
    async def process_input(self, text: str) -> str:
        """Process incoming text"""
        ...
    
    @abstractmethod
    async def process_output(self, text: str) -> str:
        """Process outgoing text"""
        ...


class IResponseAggregator(Protocol):
    """Protocol for response aggregation"""
    
    @abstractmethod
    async def start_response(self, response_id: str) -> None:
        """Start aggregating a new response"""
        ...
    
    @abstractmethod
    async def add_audio_chunk(self, response_id: str, chunk: bytes) -> None:
        """Add audio chunk to response"""
        ...
    
    @abstractmethod
    async def add_text_chunk(self, response_id: str, text: str) -> None:
        """Add text chunk to response"""
        ...
    
    @abstractmethod
    async def finalize_response(self, response_id: str) -> 'Response':
        """Finalize and return complete response"""
        ...


# ============== Data Structures ==============

@dataclass
class AudioFormat:
    """Audio format specification"""
    sample_rate: int = 24000
    channels: int = 1
    bit_depth: int = 16
    encoding: str = "pcm"
    endianness: str = "little"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "encoding": self.encoding,
            "endianness": self.endianness
        }


@dataclass
class StreamConfig:
    """Base configuration for streams"""
    provider: str
    mode: str  # "audio", "text", "both"
    audio_format: Optional[AudioFormat] = None
    buffer_size_ms: int = 100
    enable_vad: bool = False
    vad_threshold: float = 0.5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Response:
    """Unified response structure"""
    id: str
    text: Optional[str] = None
    audio: Optional[bytes] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    function_calls: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.function_calls is None:
            self.function_calls = []
        if self.metadata is None:
            self.metadata = {}


# ============== Stream Type Definitions ==============

class StreamCapability(str, Enum):
    """Capabilities that streams might support"""
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    TEXT_INPUT = "text_input"
    TEXT_OUTPUT = "text_output"
    VAD = "vad"
    FUNCTION_CALLING = "function_calling"
    STREAMING_RESPONSE = "streaming_response"
    INTERRUPTION = "interruption"
    BACKPRESSURE = "backpressure"


@dataclass
class StreamCapabilities:
    """Declares what a stream implementation supports"""
    supported: List[StreamCapability]
    audio_formats: List[AudioFormat] = None
    max_chunk_size: int = 0
    min_chunk_size: int = 0
    supports_pause_resume: bool = False
    
    def __post_init__(self):
        if self.audio_formats is None:
            self.audio_formats = []
    
    def supports(self, capability: StreamCapability) -> bool:
        return capability in self.supported


# ============== Error Types ==============

class StreamErrorType(str, Enum):
    """Standard stream error types"""
    CONNECTION_ERROR = "connection_error"
    AUDIO_FORMAT_ERROR = "audio_format_error"
    BUFFER_OVERFLOW = "buffer_overflow"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT = "rate_limit"


@dataclass
class StreamError:
    """Standard stream error structure"""
    type: StreamErrorType
    message: str
    stream_id: Optional[str] = None
    timestamp: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True


# ============== Async Iterator Protocols ==============

class AudioStreamIterator(Protocol):
    """Protocol for audio stream iterators"""
    
    @abstractmethod
    def __aiter__(self) -> 'AudioStreamIterator':
        """Return async iterator"""
        ...
    
    @abstractmethod
    async def __anext__(self) -> bytes:
        """Get next audio chunk"""
        ...


class TextStreamIterator(Protocol):
    """Protocol for text stream iterators"""
    
    @abstractmethod
    def __aiter__(self) -> 'TextStreamIterator':
        """Return async iterator"""
        ...
    
    @abstractmethod
    async def __anext__(self) -> str:
        """Get next text chunk"""
        ...

# ============== Factory Protocols ==============

class IStreamFactory(Protocol):
    """Protocol for creating streams"""
    
    @abstractmethod
    async def create_stream(
        self,
        config: StreamConfig
    ) -> IStreamManager:
        """Create a new stream with given configuration"""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> StreamCapabilities:
        """Get factory capabilities"""
        ...


# ============== Metrics Protocol ==============

@dataclass
class StreamMetrics:
    """Standard metrics all streams should track"""
    bytes_sent: int = 0
    bytes_received: int = 0
    chunks_sent: int = 0
    chunks_received: int = 0
    errors_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time  # This should return exactly 1.0 for the test
    
    # @property
    # def duration_seconds(self) -> float:
    #     if self.start_time and self.end_time:
    #         return self.end_time - self.start_time
    #     return 0.0
    
    @property
    def throughput_bps(self) -> float:
        """Bytes per second throughput"""
        if self.duration_seconds > 0:
            return (self.bytes_sent + self.bytes_received) / self.duration_seconds
        return 0.0


# ============== Type Aliases ==============

# Handler types
EventHandler = Callable[[StreamEvent], None]
AsyncEventHandler = Callable[[StreamEvent], Any]  # Returns awaitable

# Data types
AudioChunk = bytes
TextChunk = str

# Stream identifiers
StreamID = str
ResponseID = str