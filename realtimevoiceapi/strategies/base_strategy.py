# here is realtimevoiceapi/strategies/base_strategy.py

"""
Base Strategy Interface

Defines the contract that both fast and big lane strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncIterator, Callable
from dataclasses import dataclass

from ..core.stream_protocol import StreamEvent, StreamEventType, StreamState
from audioengine.audioengine.audio_types import AudioBytes
from ..core.provider_protocol import Usage, Cost


@dataclass
class EngineConfig:
    """Common configuration for all strategies"""
    api_key: str
    provider: str = "openai"
    
    # Audio settings
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    
    # Features
    enable_vad: bool = True
    enable_transcription: bool = False
    enable_functions: bool = False
    
    # Performance
    latency_mode: str = "balanced"  # "ultra_low", "balanced", "quality"
    
    # Additional config
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Base strategy interface for voice engine implementations.
    
    Both fast and big lane strategies implement this interface.
    """
    
    @abstractmethod
    async def initialize(self, config: EngineConfig) -> None:
        """Initialize the strategy with configuration"""
        pass
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to provider"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from provider"""
        pass
    
    @abstractmethod
    async def start_audio_input(self) -> None:
        """Start capturing audio input"""
        pass
    
    @abstractmethod
    async def stop_audio_input(self) -> None:
        """Stop capturing audio input"""
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio data to provider"""
        pass
    
    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text to provider"""
        pass
    
    async def send_recorded_audio(self, audio_data: AudioBytes, 
                                auto_respond: bool = True) -> None:
        """
        Send complete audio recording and optionally trigger response.
        Default implementation chunks the audio and triggers response.
        Strategies can override for more efficient implementation.
        """
        # Default implementation: send audio and trigger response
        await self.send_audio(audio_data)
        if auto_respond:
            await self.trigger_response()
    
    async def trigger_response(self) -> None:
        """
        Explicitly trigger AI response generation.
        Override in strategies that support this.
        """
        # Default: no-op - strategies should override if they support this
        self.logger.warning("trigger_response not implemented for this strategy")
    
    @abstractmethod
    async def get_response_stream(self) -> AsyncIterator[StreamEvent]:
        """Get stream of response events"""
        pass
    
    @abstractmethod
    def set_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Set handler for specific event type"""
        pass
    
    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt current response"""
        pass
    
    @abstractmethod
    def get_state(self) -> StreamState:
        """Get current stream state"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass
    
    @abstractmethod
    def get_usage(self) -> Usage:
        """Get usage statistics"""
        pass
    
    @abstractmethod
    async def estimate_cost(self) -> Cost:
        """Estimate cost of current session"""
        pass