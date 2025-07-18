"""
Unified Voice Engine

Main entry point for the realtime voice API framework.
Automatically selects between fast and big lane implementations based on configuration.


Unified Interface: Single API regardless of fast/big lane
Auto Mode Detection: Automatically chooses optimal implementation
Easy Callbacks: Simple callback-based API for responses
Context Manager: Supports async with for automatic cleanup
Factory Methods: Multiple ways to create engines
Convenience Functions: Helper functions for common use cases
Comprehensive Config: Detailed configuration with sensible defaults
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Literal, Union, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import json

from .stream_protocol import StreamEvent, StreamEventType, StreamState
from .audio_types import AudioBytes, AudioConfig
from .provider_protocol import Usage, Cost
from .strategies.base_strategy import BaseStrategy, EngineConfig
from .strategies.fast_lane_strategy import FastLaneStrategy
# from .strategies.big_lane_strategy import BigLaneStrategy  # TODO: Implement

# For fast lane direct imports
from .fast_lane.direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from .audio_types import VADConfig, VADType
from .fast_lane.fast_vad_detector import FastVADDetector
from .exceptions import EngineError


@dataclass
class VoiceEngineConfig:
    """Configuration for Voice Engine"""
    
    # API Configuration
    api_key: str
    provider: str = "openai"
    
    # Mode selection
    mode: Literal["fast", "big", "auto"] = "auto"
    
    # Audio settings
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 24000
    chunk_duration_ms: int = 100
    
    # Features
    vad_enabled: bool = True
    vad_type: Literal["client", "server"] = "client"
    vad_threshold: float = 0.02
    vad_speech_start_ms: int = 100
    vad_speech_end_ms: int = 500
    
    # Voice settings
    voice: str = "alloy"
    language: Optional[str] = None
    
    # Performance
    latency_mode: Literal["ultra_low", "balanced", "quality"] = "balanced"
    
    # Features (for big lane)
    enable_transcription: bool = False
    enable_functions: bool = False
    enable_multi_provider: bool = False
    
    # Advanced
    log_level: str = "INFO"
    save_audio: bool = False
    audio_save_path: Optional[Path] = None
    
    # Additional provider-specific config
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_engine_config(self) -> EngineConfig:
        """Convert to strategy engine config"""
        return EngineConfig(
            api_key=self.api_key,
            provider=self.provider,
            input_device=self.input_device,
            output_device=self.output_device,
            enable_vad=self.vad_enabled,
            enable_transcription=self.enable_transcription,
            enable_functions=self.enable_functions,
            latency_mode=self.latency_mode,
            metadata={
                **self.metadata,
                "voice": self.voice,
                "language": self.language,
                "vad_type": self.vad_type,
                "vad_threshold": self.vad_threshold,
                "vad_speech_start_ms": self.vad_speech_start_ms,
                "vad_speech_end_ms": self.vad_speech_end_ms,
                "sample_rate": self.sample_rate,
                "chunk_duration_ms": self.chunk_duration_ms,
                "save_audio": self.save_audio,
                "audio_save_path": str(self.audio_save_path) if self.audio_save_path else None
            }
        )


class VoiceEngine:
    """
    Unified Voice Engine - Main entry point for realtime voice API.
    
    Automatically selects optimal implementation based on configuration.
    
    Example:
        ```python
        # Simple usage - auto-selects fast lane
        engine = VoiceEngine(api_key="...")
        await engine.connect()
        await engine.start_listening()
        
        # Handle responses
        engine.on_audio_response = lambda audio: player.play(audio)
        engine.on_text_response = lambda text: print(f"AI: {text}")
        
        # Send text
        await engine.send_text("Hello!")
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[VoiceEngineConfig] = None,
        **kwargs
    ):
        """
        Initialize Voice Engine.
        
        Args:
            api_key: API key for the provider
            config: Full configuration object
            **kwargs: Additional config parameters
        """
        # Handle configuration
        if config:
            self.config = config
        else:
            # Build config from parameters
            if not api_key:
                raise ValueError("API key required")
            
            self.config = VoiceEngineConfig(
                api_key=api_key,
                **kwargs
            )
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Determine mode
        self.mode = self._determine_mode()
        self.logger.info(f"Voice Engine initialized in {self.mode} mode")
        
        # Create strategy
        self._strategy: Optional[BaseStrategy] = None
        self._create_strategy()
        
        # Audio components (for fast lane)
        self.audio_capture: Optional[DirectAudioCapture] = None
        self.audio_player: Optional[DirectAudioPlayer] = None
        self.vad_detector: Optional[FastVADDetector] = None
        
        # Callbacks for easy API
        self.on_audio_response: Optional[Callable[[AudioBytes], None]] = None
        self.on_text_response: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_function_call: Optional[Callable[[Dict[str, Any]], Any]] = None
        
        # State
        self._is_connected = False
        self._is_listening = False
        self._audio_processing_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._session_start_time: Optional[float] = None
    
    def _determine_mode(self) -> Literal["fast", "big"]:
        """Determine which mode to use based on configuration"""
        if self.config.mode != "auto":
            return self.config.mode
        
        # Auto-detect based on features
        # Use fast lane if:
        # - Client-side VAD only
        # - No transcription needed
        # - No function calling
        # - Single provider (OpenAI)
        # - Ultra-low latency required
        
        use_fast_lane = (
            self.config.vad_type == "client" and
            not self.config.enable_transcription and
            not self.config.enable_functions and
            not self.config.enable_multi_provider and
            self.config.provider == "openai" and
            self.config.latency_mode == "ultra_low"
        )
        
        if use_fast_lane:
            self.logger.info(
                "Auto-selected fast lane: client VAD, no advanced features, ultra-low latency"
            )
            return "fast"
        else:
            self.logger.info(
                "Auto-selected big lane: advanced features or non-OpenAI provider"
            )
            return "big"
    
    def _create_strategy(self):
        """Create appropriate strategy implementation"""
        if self.mode == "fast":
            self._strategy = FastLaneStrategy(logger=self.logger)
        else:
            # TODO: Implement big lane strategy
            raise NotImplementedError("Big lane strategy not yet implemented")
            # self._strategy = BigLaneStrategy(logger=self.logger)
    
    async def connect(self) -> None:
        """
        Connect to the voice API provider.
        
        Establishes WebSocket connection and initializes session.
        """
        if self._is_connected:
            self.logger.warning("Already connected")
            return
        
        try:
            # Initialize strategy
            await self._strategy.initialize(self.config.to_engine_config())
            
            # Setup audio components for fast lane
            if self.mode == "fast":
                await self._setup_fast_lane_audio()
            
            # Connect to provider
            await self._strategy.connect()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self._is_connected = True
            self._session_start_time = asyncio.get_event_loop().time()
            
            self.logger.info("Successfully connected to voice API")
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            if self.on_error:
                self.on_error(e)
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from voice API and cleanup resources"""
        if not self._is_connected:
            return
        
        try:
            # Stop listening first
            if self._is_listening:
                await self.stop_listening()
            
            # Disconnect strategy
            await self._strategy.disconnect()
            
            # Cleanup audio components
            if self.audio_player:
                self.audio_player.stop_playback()
            
            self._is_connected = False
            self.logger.info("Disconnected from voice API")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            if self.on_error:
                self.on_error(e)
    
    async def start_listening(self) -> None:
        """
        Start listening for audio input.
        
        Begins audio capture and processing.
        """
        if not self._is_connected:
            raise EngineError("Not connected")
        
        if self._is_listening:
            self.logger.warning("Already listening")
            return
        
        # Start audio input
        await self._strategy.start_audio_input()
        
        # For fast lane with VAD, start processing loop
        if self.mode == "fast" and self.config.vad_enabled:
            self._audio_processing_task = asyncio.create_task(
                self._audio_processing_loop()
            )
        
        self._is_listening = True
        self.logger.info("Started listening for audio input")
    
    async def stop_listening(self) -> None:
        """Stop listening for audio input"""
        if not self._is_listening:
            return
        
        # Stop audio processing
        if self._audio_processing_task:
            self._audio_processing_task.cancel()
            try:
                await self._audio_processing_task
            except asyncio.CancelledError:
                pass
        
        # Stop audio input
        await self._strategy.stop_audio_input()
        
        self._is_listening = False
        self.logger.info("Stopped listening for audio input")
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """
        Send audio data to the API.
        
        Args:
            audio_data: Raw audio bytes in configured format
        """
        if not self._is_connected:
            raise EngineError("Not connected")
        
        await self._strategy.send_audio(audio_data)
    
    async def send_text(self, text: str) -> None:
        """
        Send text message to the API.
        
        Args:
            text: Text message to send
        """
        if not self._is_connected:
            raise EngineError("Not connected")
        
        await self._strategy.send_text(text)
        self.logger.debug(f"Sent text: {text}")
    
    async def interrupt(self) -> None:
        """Interrupt the current AI response"""
        if not self._is_connected:
            raise EngineError("Not connected")
        
        await self._strategy.interrupt()
        self.logger.debug("Interrupted current response")
    
    def get_state(self) -> StreamState:
        """Get current stream state"""
        if self._strategy:
            return self._strategy.get_state()
        return StreamState.IDLE
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            "mode": self.mode,
            "connected": self._is_connected,
            "listening": self._is_listening,
            "uptime": (
                asyncio.get_event_loop().time() - self._session_start_time
                if self._session_start_time else 0
            )
        }
        
        if self._strategy:
            metrics.update(self._strategy.get_metrics())
        
        return metrics
    
    async def get_usage(self) -> Usage:
        """Get usage statistics for current session"""
        if self._strategy:
            return self._strategy.get_usage()
        return Usage()
    
    async def estimate_cost(self) -> Cost:
        """Estimate cost of current session"""
        if self._strategy:
            return await self._strategy.estimate_cost()
        return Cost()
    
    # ============== Easy API Methods ==============
    
    async def transcribe_audio_file(self, file_path: Union[str, Path]) -> str:
        """
        Transcribe an audio file (convenience method).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # This would need big lane implementation
        raise NotImplementedError("File transcription requires big lane mode")
    
    async def speak(self, text: str) -> AudioBytes:
        """
        Convert text to speech (convenience method).
        
        Args:
            text: Text to speak
            
        Returns:
            Audio data
        """
        if not self._is_connected:
            await self.connect()
        
        # Send text and collect audio response
        audio_chunks = []
        
        def collect_audio(event: StreamEvent):
            if event.data and "audio" in event.data:
                audio_chunks.append(event.data["audio"])
        
        # Temporarily set handler
        old_handler = self.on_audio_response
        self._strategy.set_event_handler(
            StreamEventType.AUDIO_OUTPUT_CHUNK,
            collect_audio
        )
        
        try:
            await self.send_text(text)
            
            # Wait for response (with timeout)
            await asyncio.sleep(5.0)  # Simple approach - could be improved
            
            # Combine chunks
            return b"".join(audio_chunks)
            
        finally:
            # Restore handler
            if old_handler:
                self.on_audio_response = old_handler
    
    # ============== Private Methods ==============
    
    async def _setup_fast_lane_audio(self):
        """Setup audio components for fast lane"""
        # Create audio configuration
        audio_config = AudioConfig(
            sample_rate=self.config.sample_rate,
            channels=1,
            bit_depth=16,
            chunk_duration_ms=self.config.chunk_duration_ms
        )
        
        # Setup VAD if enabled
        if self.config.vad_enabled:
            vad_config = VADConfig(
                type=VADType.ENERGY_BASED,
                energy_threshold=self.config.vad_threshold,
                speech_start_ms=self.config.vad_speech_start_ms,
                speech_end_ms=self.config.vad_speech_end_ms
            )
            
            self.vad_detector = FastVADDetector(
                config=vad_config,
                audio_config=audio_config
            )
        
        # Setup audio capture
        self.audio_capture = DirectAudioCapture(
            device=self.config.input_device,
            config=audio_config
        )
        
        # Setup audio player
        self.audio_player = DirectAudioPlayer(
            device=self.config.output_device,
            config=audio_config
        )
    
    def _setup_event_handlers(self):
        """Setup internal event handlers"""
        # Audio output handler
        def handle_audio_output(event: StreamEvent):
            if self.on_audio_response and event.data:
                audio_data = event.data.get("audio")
                if audio_data:
                    self.on_audio_response(audio_data)
                    
                    # Auto-play if player is available
                    if self.audio_player:
                        self.audio_player.play_audio(audio_data)
        
        self._strategy.set_event_handler(
            StreamEventType.AUDIO_OUTPUT_CHUNK,
            handle_audio_output
        )
        
        # Text output handler
        def handle_text_output(event: StreamEvent):
            if self.on_text_response and event.data:
                text = event.data.get("text")
                if text:
                    self.on_text_response(text)
        
        self._strategy.set_event_handler(
            StreamEventType.TEXT_OUTPUT_CHUNK,
            handle_text_output
        )
        
        # Error handler
        def handle_error(event: StreamEvent):
            if self.on_error and event.data:
                error = event.data.get("error")
                if error:
                    self.on_error(Exception(error))
        
        self._strategy.set_event_handler(
            StreamEventType.STREAM_ERROR,
            handle_error
        )
    
    async def _audio_processing_loop(self):
        """Process audio input for fast lane with VAD"""
        if not self.audio_capture:
            return
        
        # Start async audio capture
        audio_queue = await self.audio_capture.start_async_capture()
        
        while self._is_listening:
            try:
                # Get audio chunk
                audio_chunk = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.1
                )
                
                # Process through VAD if enabled
                if self.vad_detector:
                    vad_state = self.vad_detector.process_chunk(audio_chunk)
                    
                    # Only send during speech
                    if vad_state.value in ["speech_starting", "speech"]:
                        await self._strategy.send_audio(audio_chunk)
                else:
                    # No VAD, send everything
                    await self._strategy.send_audio(audio_chunk)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
                if self.on_error:
                    self.on_error(e)
    
    # ============== Context Manager Support ==============
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    # ============== Factory Methods ==============
    
    @classmethod
    def create_simple(
        cls,
        api_key: str,
        voice: str = "alloy"
    ) -> 'VoiceEngine':
        """
        Create a simple voice engine with default settings.
        
        Best for getting started quickly.
        """
        config = VoiceEngineConfig(
            api_key=api_key,
            voice=voice,
            mode="fast",
            latency_mode="ultra_low"
        )
        return cls(config=config)
    
    @classmethod
    def create_advanced(
        cls,
        api_key: str,
        enable_transcription: bool = True,
        enable_functions: bool = True,
        **kwargs
    ) -> 'VoiceEngine':
        """
        Create an advanced voice engine with full features.
        
        Uses big lane implementation.
        """
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="big",
            enable_transcription=enable_transcription,
            enable_functions=enable_functions,
            latency_mode="balanced",
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'VoiceEngine':
        """Create voice engine from configuration file"""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = VoiceEngineConfig(**config_data)
        return cls(config=config)


# ============== Convenience Functions ==============

async def create_voice_session(
    api_key: str,
    **kwargs
) -> VoiceEngine:
    """
    Create and connect a voice session.
    
    Convenience function for quick setup.
    """
    engine = VoiceEngine(api_key=api_key, **kwargs)
    await engine.connect()
    return engine


def run_voice_engine(
    api_key: str,
    on_audio: Optional[Callable[[AudioBytes], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    **kwargs
):
    """
    Run voice engine in a simple event loop.
    
    Good for testing and simple applications.
    """
    async def main():
        engine = VoiceEngine(api_key=api_key, **kwargs)
        
        # Set callbacks
        if on_audio:
            engine.on_audio_response = on_audio
        if on_text:
            engine.on_text_response = on_text
        
        # Connect and start
        await engine.connect()
        await engine.start_listening()
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await engine.disconnect()
    
    asyncio.run(main())