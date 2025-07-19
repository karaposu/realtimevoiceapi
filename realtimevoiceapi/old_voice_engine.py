# here is realtimevoiceapi/voice_engine.py

"""
Unified Voice Engine

Main entry point for the realtime voice API framework.
User explicitly selects between fast and big lane implementations.

Unified Interface: Single API regardless of fast/big lane
Explicit Mode Selection: User chooses optimal implementation
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

from .core.stream_protocol import StreamEvent, StreamEventType, StreamState
from .core.audio_types import AudioBytes, AudioConfig
from .core.provider_protocol import Usage, Cost
from .strategies.base_strategy import BaseStrategy, EngineConfig
from .strategies.fast_lane_strategy import FastLaneStrategy
# from .strategies.big_lane_strategy import BigLaneStrategy  # TODO: Implement

# For fast lane direct imports
from .fast_lane.direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from .core.audio_types import VADConfig, VADType
from .fast_lane.fast_vad_detector import FastVADDetector
from .core.exceptions import EngineError


@dataclass
class VoiceEngineConfig:
    """Configuration for Voice Engine"""
    
    # API Configuration
    api_key: str
    provider: str = "openai"
    
    # Mode selection - MUST be explicitly set
    mode: Literal["fast", "big"] = "fast"  # Default to fast lane
    
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
    
    User must explicitly choose between fast and big lane modes.
    
    Fast Lane:
        - Ultra-low latency (<50ms)
        - Direct audio path
        - Client-side VAD
        - Best for simple voice interactions
        
    Big Lane:
        - Feature-rich processing
        - Audio pipeline with effects
        - Function calling support
        - Best for complex applications
    
    Example:
        ```python
        # Fast lane - low latency
        engine = VoiceEngine(api_key="...", mode="fast")
        await engine.connect()
        await engine.start_listening()
        
        # Big lane - full features
        engine = VoiceEngine(api_key="...", mode="big")
        await engine.connect()
        
        # Handle responses
        engine.on_audio_response = lambda audio: player.play(audio)
        engine.on_text_response = lambda text: print(f"AI: {text}")
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[VoiceEngineConfig] = None,
        mode: Literal["fast", "big"] = "fast",
        **kwargs
    ):
        """
        Initialize Voice Engine.
        
        Args:
            api_key: API key for the provider
            config: Full configuration object
            mode: Explicitly choose "fast" or "big" lane
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
                mode=mode,
                **kwargs
            )
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Use explicit mode
        self.mode = self.config.mode
        self.logger.info(f"Voice Engine initialized in {self.mode} mode")
        
        # Validate mode choice
        if self.mode not in ["fast", "big"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'fast' or 'big'")
        
        # Create strategy
        self._strategy: Optional[BaseStrategy] = None
        self._create_strategy()
        
        # Audio components (for fast lane)
        # TODO: Move these to FastLaneStrategy for better encapsulation
        self.audio_capture: Optional[DirectAudioCapture] = None
        self.audio_player: Optional[DirectAudioPlayer] = None
        self.vad_detector: Optional[FastVADDetector] = None
        
        # Callbacks for easy API
        self.on_audio_response: Optional[Callable[[AudioBytes], None]] = None
        self.on_text_response: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_function_call: Optional[Callable[[Dict[str, Any]], Any]] = None
        self.on_response_done: Optional[Callable[[], None]] = None
        
        # State
        self._is_connected = False
        self._is_listening = False
        self._audio_processing_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._session_start_time: Optional[float] = None
    
    # ============== State Management Properties ==============
    
    @property
    def is_connected(self) -> bool:
        """Check if engine is properly connected"""
        return (
            self._is_connected and 
            self._strategy is not None and 
            self._strategy.get_state() != StreamState.ERROR
        )
    
    @property
    def is_listening(self) -> bool:
        """Check if engine is actively listening"""
        return self._is_listening and self.is_connected
    
    def _ensure_connected(self):
        """Ensure engine is connected before operations"""
        if not self.is_connected:
            raise EngineError("Not connected. Call connect() first.")
    
    # ============== Core Methods ==============
    
    def _create_strategy(self):
        """Create appropriate strategy implementation"""
        if self.mode == "fast":
            self._strategy = FastLaneStrategy(logger=self.logger)
        else:
            # TODO: Implement big lane strategy
            raise NotImplementedError(
                "Big lane strategy not yet implemented. Please use mode='fast' for now."
            )
            # self._strategy = BigLaneStrategy(logger=self.logger)
    
    async def connect(self, retry_count: int = 3) -> None:
        """
        Connect to the voice API provider with retry logic.
        
        Establishes WebSocket connection and initializes session.
        
        Args:
            retry_count: Number of connection attempts
        """
        if self._is_connected:
            self.logger.warning("Already connected")
            return
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1}/{retry_count}")
                    await asyncio.sleep(1.0 * attempt)  # Exponential backoff
                
                await self._do_connect()
                return
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
        
        error = EngineError(f"Failed to connect after {retry_count} attempts")
        if self.on_error:
            self.on_error(error)
        raise error from last_error
    
    async def _do_connect(self) -> None:
        """Internal connection logic"""
        # Initialize strategy only if not already initialized
        if not self._strategy._is_initialized:
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
            
            # Reset strategy initialization state for potential reconnection
            if hasattr(self._strategy, '_is_initialized'):
                self._strategy._is_initialized = False
            
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
        self._ensure_connected()
        
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
        self._ensure_connected()
        await self._strategy.send_audio(audio_data)
    
    async def send_text(self, text: str) -> None:
        """
        Send text message to the API.
        
        Args:
            text: Text message to send
        """
        self._ensure_connected()
        await self._strategy.send_text(text)
        self.logger.debug(f"Sent text: {text}")
    
    async def interrupt(self) -> None:
        """Interrupt the current AI response"""
        self._ensure_connected()
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
    
    # ============== Advanced Event Stream API ==============
    
    async def event_stream(self) -> AsyncIterator[StreamEvent]:
        """
        Stream all events for advanced users.
        
        Yields:
            StreamEvent objects as they occur
            
        Example:
            ```python
            async for event in engine.event_stream():
                if event.type == StreamEventType.AUDIO_OUTPUT_CHUNK:
                    # Process audio chunk
                    pass
            ```
        """
        self._ensure_connected()
        
        event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        
        def queue_event(event: StreamEvent):
            try:
                event_queue.put_nowait(event)
            except asyncio.QueueFull:
                self.logger.warning("Event queue full, dropping event")
        
        # Subscribe to all events
        # Note: This assumes strategy has a universal handler method
        if hasattr(self._strategy, 'set_universal_handler'):
            self._strategy.set_universal_handler(queue_event)
        else:
            # Fallback: subscribe to known events
            for event_type in StreamEventType:
                self._strategy.set_event_handler(event_type, queue_event)
        
        try:
            while self.is_connected:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    continue
        finally:
            # Cleanup handlers
            if hasattr(self._strategy, 'clear_universal_handler'):
                self._strategy.clear_universal_handler()
    
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
        if self.mode != "big":
            raise EngineError("File transcription requires big lane mode")
        raise NotImplementedError("File transcription not yet implemented")
    
    async def speak(self, text: str, timeout: float = 30.0) -> AudioBytes:
        """
        Convert text to speech (convenience method).
        
        Args:
            text: Text to speak
            timeout: Maximum time to wait for response
            
        Returns:
            Audio data
        """
        self._ensure_connected()
        
        audio_future: asyncio.Future[bytes] = asyncio.Future()
        audio_chunks: List[bytes] = []
        
        def collect_audio(audio: AudioBytes):
            audio_chunks.append(audio)
        
        def on_response_done():
            if audio_chunks:
                audio_future.set_result(b"".join(audio_chunks))
            else:
                audio_future.set_exception(EngineError("No audio received"))
        
        # Set temporary handlers
        old_audio_handler = self.on_audio_response
        old_done_handler = self.on_response_done
        
        try:
            self.on_audio_response = collect_audio
            self.on_response_done = on_response_done
            
            # Set internal handlers
            self._setup_event_handlers()
            
            # Send text
            await self.send_text(text)

            try:
                return await asyncio.wait_for(audio_future, timeout=timeout)
            except asyncio.TimeoutError:
                raise EngineError(f"Response timeout after {timeout} seconds")
                
            
            
        finally:
            # Restore handlers
            self.on_audio_response = old_audio_handler
            self.on_response_done = old_done_handler
            self._setup_event_handlers()


    def _handle_response_done(self):
        """Handle response done without event parameter"""
        if self.on_response_done:
            self.on_response_done()
    
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
        """Setup event handlers using mapping"""
        event_mapping = {
            StreamEventType.AUDIO_OUTPUT_CHUNK: self._handle_audio_output,
            StreamEventType.TEXT_OUTPUT_CHUNK: self._handle_text_output,
            StreamEventType.STREAM_ERROR: self._handle_error,
            StreamEventType.STREAM_ENDED: self._handle_stream_ended,
        }
        
        # Add function call handler if it exists
        if hasattr(StreamEventType, 'FUNCTION_CALL'):
            event_mapping[StreamEventType.FUNCTION_CALL] = self._handle_function_call

        for event_type, handler in event_mapping.items():
            self._strategy.set_event_handler(event_type, handler)


        if hasattr(self._strategy, 'stream_manager') and self._strategy.stream_manager:
            if hasattr(self._strategy.stream_manager, 'set_response_done_callback'):
                self._strategy.stream_manager.set_response_done_callback(self._handle_response_done)
                
        
    
    def _handle_audio_output(self, event: StreamEvent):
        """Handle audio output events"""
        if self.on_audio_response and event.data:
            audio_data = event.data.get("audio")
            if audio_data:
                self.on_audio_response(audio_data)
                
                # Auto-play if player is available
                if self.audio_player:
                    self.audio_player.play_audio(audio_data)
    
    def _handle_text_output(self, event: StreamEvent):
        """Handle text output events"""
        if self.on_text_response and event.data:
            text = event.data.get("text")
            if text:
                self.on_text_response(text)
    
    def _handle_error(self, event: StreamEvent):
        """Handle error events"""
        if self.on_error and event.data:
            error_msg = event.data.get("error", "Unknown error")
            error = EngineError(str(error_msg))
            self.on_error(error)
    
    def _handle_function_call(self, event: StreamEvent):
        """Handle function call events"""
        if self.on_function_call and event.data:
            function_data = event.data.get("function")
            if function_data:
                # Call the handler and potentially get a result
                result = self.on_function_call(function_data)
                # TODO: Send function result back to API
    
    def _handle_stream_ended(self, event: StreamEvent):
        """Handle stream ended events"""
        self._handle_response_done()
    
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
        Create a simple voice engine with fast lane.
        
        Best for getting started quickly with low latency.
        """
        config = VoiceEngineConfig(
            api_key=api_key,
            voice=voice,
            mode="fast",  # Explicit mode
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
        Create an advanced voice engine with big lane.
        
        Note: Big lane is not yet implemented.
        """
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="big",  # Explicit mode
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
    mode: Literal["fast", "big"] = "fast",
    **kwargs
) -> VoiceEngine:
    """
    Create and connect a voice session.
    
    Convenience function for quick setup.
    """
    engine = VoiceEngine(api_key=api_key, mode=mode, **kwargs)
    await engine.connect()
    return engine


def run_voice_engine(
    api_key: str,
    mode: Literal["fast", "big"] = "fast",
    on_audio: Optional[Callable[[AudioBytes], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    **kwargs
):
    """
    Run voice engine in a simple event loop.
    
    Good for testing and simple applications.
    """
    async def main():
        engine = VoiceEngine(api_key=api_key, mode=mode, **kwargs)
        
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