# here is realtimevoiceapi/provider_protocol.py

"""
Provider Protocol Definitions

Defines the contracts that all voice API providers must implement.
Provider-agnostic interfaces for different services (OpenAI, Anthropic, etc).
"""

from typing import Protocol, Dict, Any, List, Optional, AsyncIterator, Tuple
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .stream_protocol import StreamCapabilities, StreamEvent, StreamConfig
from .audio_types import AudioFormat


# ============== Provider Capabilities ==============

class ProviderFeature(str, Enum):
    """Features that providers might support"""
    REALTIME_VOICE = "realtime_voice"
    STREAMING_TEXT = "streaming_text"
    FUNCTION_CALLING = "function_calling"
    VOICE_CLONING = "voice_cloning"
    MULTILINGUAL = "multilingual"
    TRANSCRIPTION = "transcription"
    SERVER_VAD = "server_vad"
    CLIENT_VAD = "client_vad"
    INTERRUPTION = "interruption"
    EMOTION_CONTROL = "emotion_control"
    CUSTOM_VOICES = "custom_voices"


@dataclass
class ProviderCapabilities:
    """What a provider can do"""
    provider_name: str
    features: List[ProviderFeature]
    
    # Audio capabilities
    supported_audio_formats: List[AudioFormat]
    supported_sample_rates: List[int]
    max_audio_duration_ms: int
    min_audio_chunk_ms: int
    
    # Voice options
    available_voices: List[str]
    supports_voice_config: bool
    
    # Language support
    supported_languages: List[str]
    
    # Limits
    max_session_duration_ms: Optional[int] = None
    max_concurrent_streams: int = 1
    rate_limits: Optional[Dict[str, Any]] = None
    
    def supports(self, feature: ProviderFeature) -> bool:
        return feature in self.features


# ============== Cost Models ==============

class CostUnit(str, Enum):
    """Units for billing"""
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_CHARACTER = "per_character"
    PER_TOKEN = "per_token"
    PER_REQUEST = "per_request"


@dataclass
class CostModel:
    """Provider cost structure"""
    # Audio costs
    audio_input_cost: float = 0.0
    audio_input_unit: CostUnit = CostUnit.PER_MINUTE
    
    audio_output_cost: float = 0.0
    audio_output_unit: CostUnit = CostUnit.PER_MINUTE
    
    # Text costs
    text_input_cost: float = 0.0
    text_input_unit: CostUnit = CostUnit.PER_TOKEN
    
    text_output_cost: float = 0.0
    text_output_unit: CostUnit = CostUnit.PER_TOKEN
    
    # Additional costs
    session_cost: float = 0.0  # Per session fee
    function_call_cost: float = 0.0  # Per function call
    
    # Currency
    currency: str = "USD"


@dataclass
class Usage:
    """Track usage for cost calculation"""
    audio_input_seconds: float = 0.0
    audio_output_seconds: float = 0.0
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    function_calls: int = 0
    session_count: int = 0
    
    def add(self, other: 'Usage'):
        """Add another usage to this one"""
        self.audio_input_seconds += other.audio_input_seconds
        self.audio_output_seconds += other.audio_output_seconds
        self.text_input_tokens += other.text_input_tokens
        self.text_output_tokens += other.text_output_tokens
        self.function_calls += other.function_calls
        self.session_count += other.session_count


@dataclass
class Cost:
    """Calculated costs"""
    audio_cost: float = 0.0
    text_cost: float = 0.0
    session_cost: float = 0.0
    function_cost: float = 0.0
    
    @property
    def total(self) -> float:
        return self.audio_cost + self.text_cost + self.session_cost + self.function_cost
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "audio": self.audio_cost,
            "text": self.text_cost,
            "session": self.session_cost,
            "function": self.function_cost,
            "total": self.total
        }


# ============== Provider Configuration ==============

@dataclass
class ProviderConfig:
    """Base configuration for any provider"""
    api_key: str
    endpoint: Optional[str] = None  # Custom endpoint
    timeout: float = 30.0
    max_retries: int = 3
    
    # Provider-specific settings go in metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============== Core Provider Protocol ==============

class IVoiceProvider(Protocol):
    """
    Protocol that all voice providers must implement.
    
    This is the main contract for provider adapters.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')"""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities"""
        ...
    
    @abstractmethod
    def get_cost_model(self) -> CostModel:
        """Get provider cost model"""
        ...
    
    @abstractmethod
    async def create_session(
        self,
        config: ProviderConfig,
        stream_config: StreamConfig
    ) -> 'IProviderSession':
        """Create a new session with the provider"""
        ...
    
    @abstractmethod
    async def validate_config(self, config: ProviderConfig) -> Tuple[bool, Optional[str]]:
        """Validate provider configuration"""
        ...
    
    @abstractmethod
    def estimate_cost(self, usage: Usage) -> Cost:
        """Estimate cost for given usage"""
        ...


class IProviderSession(Protocol):
    """
    Protocol for an active session with a provider.
    
    Represents an active connection/session.
    """
    
    @property
    @abstractmethod
    def session_id(self) -> str:
        """Unique session identifier"""
        ...
    
    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if session is active"""
        ...
    
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio to provider"""
        ...
    
    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text to provider"""
        ...
    
    @abstractmethod
    async def send_function_result(
        self,
        call_id: str,
        result: Any
    ) -> None:
        """Send function call result"""
        ...
    
    @abstractmethod
    def get_event_stream(self) -> AsyncIterator[StreamEvent]:
        """Get stream of events from provider"""
        ...
    
    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt current response"""
        ...
    
    @abstractmethod
    async def end_session(self) -> Usage:
        """End session and return usage stats"""
        ...
    
    @abstractmethod
    def get_usage(self) -> Usage:
        """Get current usage stats"""
        ...


# ============== Provider Registry ==============

class ProviderRegistry:
    """
    Registry of available providers.
    
    Central place to register and discover providers.
    """
    
    def __init__(self):
        self._providers: Dict[str, IVoiceProvider] = {}
        self._default_provider: Optional[str] = None
    
    def register(
        self,
        provider: IVoiceProvider,
        set_as_default: bool = False
    ) -> None:
        """Register a provider"""
        name = provider.name
        self._providers[name] = provider
        
        if set_as_default or not self._default_provider:
            self._default_provider = name
    
    def get(self, name: Optional[str] = None) -> IVoiceProvider:
        """Get provider by name or default"""
        if name is None:
            name = self._default_provider
        
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")
        
        return self._providers[name]
    
    def list_providers(self) -> List[str]:
        """List available provider names"""
        return list(self._providers.keys())
    
    def get_all_capabilities(self) -> Dict[str, ProviderCapabilities]:
        """Get capabilities of all providers"""
        return {
            name: provider.get_capabilities()
            for name, provider in self._providers.items()
        }
    
    def find_providers_with_feature(
        self,
        feature: ProviderFeature
    ) -> List[str]:
        """Find providers that support a specific feature"""
        return [
            name for name, provider in self._providers.items()
            if feature in provider.get_capabilities().features
        ]


# ============== Provider Events ==============

class ProviderEventType(str, Enum):
    """Provider-specific event types"""
    # Connection events
    SESSION_CREATED = "provider.session.created"
    SESSION_ENDED = "provider.session.ended"
    SESSION_ERROR = "provider.session.error"
    
    # Usage events
    USAGE_UPDATED = "provider.usage.updated"
    RATE_LIMIT_WARNING = "provider.rate_limit.warning"
    RATE_LIMIT_EXCEEDED = "provider.rate_limit.exceeded"
    
    # Cost events
    COST_THRESHOLD_WARNING = "provider.cost.threshold_warning"
    COST_UPDATED = "provider.cost.updated"


@dataclass
class ProviderEvent:
    """Provider-specific events"""
    type: ProviderEventType
    provider: str
    session_id: Optional[str]
    data: Dict[str, Any]
    timestamp: float


# ============== Voice Configuration ==============

@dataclass
class VoiceConfig:
    """Provider-agnostic voice configuration"""
    voice_id: str
    
    # Common parameters (may not be supported by all)
    speed: float = 1.0  # 0.5 to 2.0
    pitch: float = 1.0  # 0.5 to 2.0
    volume: float = 1.0  # 0.0 to 1.0
    
    # Emotion/style (provider-specific)
    emotion: Optional[str] = None
    style: Optional[str] = None
    
    # Provider-specific parameters
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============== Transcription Support ==============

@dataclass
class TranscriptionConfig:
    """Configuration for transcription features"""
    enabled: bool = True
    language: Optional[str] = None  # None for auto-detect
    model: Optional[str] = None  # Provider-specific model
    
    # Options
    punctuation: bool = True
    profanity_filter: bool = False
    
    # Provider-specific
    metadata: Dict[str, Any] = None


# ============== Function Calling ==============

@dataclass
class FunctionDefinition:
    """Provider-agnostic function definition"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    
    # Optional metadata
    examples: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FunctionCall:
    """Function call from provider"""
    id: str
    name: str
    arguments: str  # JSON string
    
    def parse_arguments(self) -> Dict[str, Any]:
        """Parse arguments as JSON"""
        import json
        return json.loads(self.arguments)


# ============== Quality Settings ==============

class QualityPreset(str, Enum):
    """Quality presets for providers"""
    ECONOMY = "economy"      # Lowest cost, lower quality
    BALANCED = "balanced"    # Good balance
    PREMIUM = "premium"      # Best quality, higher cost


@dataclass
class QualitySettings:
    """Quality settings for providers"""
    preset: QualityPreset = QualityPreset.BALANCED
    
    # Audio quality
    audio_bitrate: Optional[int] = None
    noise_suppression: bool = True
    echo_cancellation: bool = True
    
    # Response quality
    temperature: float = 0.8
    response_format: str = "natural"  # natural, concise, detailed
    
    # Provider-specific
    metadata: Dict[str, Any] = None