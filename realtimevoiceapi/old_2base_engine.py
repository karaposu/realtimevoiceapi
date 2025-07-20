"""
Base Voice Engine Implementation

Contains all the internal implementation details for the voice engine.
This is not meant to be used directly by users - they should use VoiceEngine instead.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List, Literal
from dataclasses import dataclass

from .core.stream_protocol import StreamEvent, StreamEventType, StreamState
from .core.audio_types import AudioBytes, AudioConfig, VADConfig, VADType
from .core.exceptions import EngineError
from .strategies.base_strategy import BaseStrategy, EngineConfig
from .strategies.fast_lane_strategy import FastLaneStrategy
from .audio.audio_manager import AudioManager, AudioManagerConfig


class BaseEngine:
    """
    Base implementation for voice engine.
    
    Handles all the complex internal logic, state management,
    and coordination between components.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize base engine"""
        self.logger = logger or logging.getLogger(__name__)
        
        # Strategy
        self._strategy: Optional[BaseStrategy] = None
        
        # Audio manager (replaces individual components)
        self._audio_manager: Optional[AudioManager] = None
        
        # State
        self._is_connected = False
        self._is_listening = False
        self._audio_processing_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._session_start_time: Optional[float] = None
        
        # Event handlers storage
        self._event_handlers: Dict[StreamEventType, Callable] = {}
        
        # Configuration cache
        self._config: Optional[EngineConfig] = None
        self._mode: Optional[str] = None
        
    # ============== State Properties ==============
    
    @property
    def is_connected(self) -> bool:
        """Check if properly connected"""
        return (
            self._is_connected and 
            self._strategy is not None and 
            self._strategy.get_state() != StreamState.ERROR
        )
    
    @property
    def is_listening(self) -> bool:
        """Check if actively listening"""
        return self._is_listening and self.is_connected
    
    @property
    def strategy(self) -> Optional[BaseStrategy]:
        """Get current strategy"""
        return self._strategy
    
    def get_state(self) -> StreamState:
        """Get current stream state"""
        if self._strategy:
            return self._strategy.get_state()
        return StreamState.IDLE
    
    # ============== Initialization ==============
    
    def create_strategy(self, mode: str) -> BaseStrategy:
        """
        Create appropriate strategy implementation.
        
        Args:
            mode: Either "fast" or "big"
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If mode is invalid
            NotImplementedError: If big lane not implemented
        """
        self._mode = mode
        
        if mode == "fast":
            self._strategy = FastLaneStrategy(logger=self.logger)
        elif mode == "big":
            # TODO: Implement big lane strategy
            raise NotImplementedError(
                "Big lane strategy not yet implemented. Please use mode='fast' for now."
            )
            # self._strategy = BigLaneStrategy(logger=self.logger)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fast' or 'big'")
        
        return self._strategy
    
    async def initialize_strategy(self, config: EngineConfig) -> None:
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Engine configuration
        """
        if not self._strategy:
            raise EngineError("Strategy not created. Call create_strategy first.")
        
        self._config = config
        
        # Initialize strategy only if not already initialized
        if not self._strategy._is_initialized:
            await self._strategy.initialize(config)
    
    async def setup_fast_lane_audio(
        self,
        sample_rate: int,
        chunk_duration_ms: int,
        input_device: Optional[int],
        output_device: Optional[int],
        vad_enabled: bool,
        vad_threshold: float,
        vad_speech_start_ms: int,
        vad_speech_end_ms: int
    ) -> None:
        """Setup audio manager for fast lane"""
        try:
            # Create VAD config if enabled
            vad_config = None
            if vad_enabled:
                vad_config = VADConfig(
                    type=VADType.ENERGY_BASED,
                    energy_threshold=vad_threshold,
                    speech_start_ms=vad_speech_start_ms,
                    speech_end_ms=vad_speech_end_ms
                )
            
            # Create audio manager config
            audio_config = AudioManagerConfig(
                input_device=input_device,
                output_device=output_device,
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration_ms,
                vad_enabled=vad_enabled,
                vad_config=vad_config
            )
            
            # Create and initialize audio manager
            self._audio_manager = AudioManager(audio_config, logger=self.logger)
            await self._audio_manager.initialize()
            
        except Exception as e:
            # Clean up if initialization fails
            self.logger.error(f"Failed to setup audio: {e}")
            self._audio_manager = None
            raise
    
    # ============== Connection Management ==============
    
    async def do_connect(self) -> None:
        """
        Internal connection logic.
        
        Handles strategy connection and event handler setup.
        """
        if not self._strategy:
            raise EngineError("Strategy not initialized")
        
        # Connect to provider
        await self._strategy.connect()
        
        self._is_connected = True
        self._session_start_time = asyncio.get_event_loop().time()
        
        self.logger.info("Successfully connected to voice API")
    
    async def do_disconnect(self) -> None:
        """Internal disconnection logic"""
        if not self._is_connected:
            return
        
        try:
            # Use the comprehensive cleanup
            await self.cleanup()
            self.logger.info("Disconnected from voice API")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            raise
    
    # ============== Audio Processing ==============
    
    async def start_audio_processing(self) -> None:
        """Start audio input processing"""
        if self._is_listening:
            self.logger.warning("Already listening")
            return
        
        # Start audio input through strategy
        await self._strategy.start_audio_input()
        
        # For fast lane with audio manager, start processing loop
        if self._mode == "fast" and self._audio_manager:
            self._audio_processing_task = asyncio.create_task(
                self._audio_processing_loop()
            )
        
        self._is_listening = True
        self.logger.info("Started listening for audio input")
    
    async def stop_audio_processing(self) -> None:
        """Stop audio input processing"""
        if not self._is_listening:
            return
        
        # Mark as not listening first
        self._is_listening = False
        
        # Stop audio processing task
        if self._audio_processing_task:
            self._audio_processing_task.cancel()
            try:
                await self._audio_processing_task
            except asyncio.CancelledError:
                pass
            self._audio_processing_task = None
        
        # Stop audio capture through manager
        if self._audio_manager:
            await self._audio_manager.stop_capture()
        
        # Stop audio input through strategy
        await self._strategy.stop_audio_input()
        
        self.logger.info("Stopped listening for audio input")

    
    async def setup_audio_output_buffering(self, enabled: bool = True):
        """Enable buffered audio output with completion tracking"""
        if enabled and self._audio_manager:
            # Replace the direct player with buffered player
            from .audio.buffered_audio_player import BufferedAudioPlayer
            self._audio_manager._player = BufferedAudioPlayer(
                config=self._audio_manager.config,
                logger=self.logger
            )
            
            # Set up completion callback for response done events
            if hasattr(self._audio_manager._player, 'set_completion_callback'):
                self._audio_manager._player.set_completion_callback(
                    self._on_audio_playback_complete
                )

    # And add a new event handl
    
    async def _audio_processing_loop(self) -> None:
        """Process audio input for fast lane with VAD"""
        if not self._audio_manager:
            self.logger.error("No audio manager available")
            return
        
        # Get audio queue from manager
        try:
            audio_queue = await self._audio_manager.start_capture()
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            return
        
        try:
            while self._is_listening and self._is_connected:
                try:
                    # Get audio chunk
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    
                    # Check if we should still process
                    if not self._is_listening or not self._is_connected:
                        break
                    
                    # Check strategy state
                    if self._strategy and self._strategy.get_state() in [
                        StreamState.ACTIVE, StreamState.STARTING
                    ]:
                        # Process through VAD
                        vad_state = self._audio_manager.process_vad(audio_chunk)
                        
                        # Send if speech or no VAD
                        if not vad_state or vad_state in ["speech_starting", "speech"]:
                            await self._strategy.send_audio(audio_chunk)
                    
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    self.logger.debug("Audio processing cancelled")
                    break
                except Exception as e:
                    # Only log error if we're still supposed to be running
                    if self._is_listening and self._is_connected:
                        self.logger.error(f"Audio processing error: {e}")
                        # Notify error handler if set
                        if StreamEventType.STREAM_ERROR in self._event_handlers:
                            error_event = StreamEvent(
                                type=StreamEventType.STREAM_ERROR,
                                stream_id="unknown",
                                timestamp=time.time(),
                                data={"error": str(e)}
                            )
                            self._event_handlers[StreamEventType.STREAM_ERROR](error_event)
                        
        except Exception as e:
            self.logger.error(f"Fatal audio processing error: {e}")
        finally:
            # Stop capture through manager
            if self._audio_manager:
                await self._audio_manager.stop_capture()
    
    # ============== Event Management ==============
    
    def setup_event_handlers(self, handlers: Dict[StreamEventType, Callable]) -> None:
        """
        Setup all event handlers at once.
        
        Args:
            handlers: Dictionary mapping event types to handler functions
        """
        self._event_handlers = handlers
        
        # Pass handlers to strategy
        if self._strategy:
            for event_type, handler in handlers.items():
                self._strategy.set_event_handler(event_type, handler)
        
        # Special handling for fast lane response done callback
        if (self._mode == "fast" and 
            hasattr(self._strategy, 'stream_manager') and 
            self._strategy.stream_manager):
            
            if hasattr(self._strategy.stream_manager, 'set_response_done_callback'):
                # Create wrapper for response done
                def response_done_wrapper():
                    if StreamEventType.STREAM_ENDED in self._event_handlers:
                        # Create a synthetic event
                        event = StreamEvent(
                            type=StreamEventType.STREAM_ENDED,
                            stream_id=self._strategy.stream_manager.stream_id,
                            timestamp=time.time(),
                            data={}
                        )
                        self._event_handlers[StreamEventType.STREAM_ENDED](event)
                
                self._strategy.stream_manager.set_response_done_callback(response_done_wrapper)
    
    def set_event_handler(self, event_type: StreamEventType, handler: Callable) -> None:
        """Set a single event handler"""
        self._event_handlers[event_type] = handler
        
        if self._strategy:
            self._strategy.set_event_handler(event_type, handler)
    
    # ============== Data Transmission ==============
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio data to the API"""
        if not self._strategy:
            raise EngineError("Not connected")
        
        await self._strategy.send_audio(audio_data)
    
    async def send_text(self, text: str) -> None:
        """Send text message to the API"""
        if not self._strategy:
            raise EngineError("Not connected")
        
        await self._strategy.send_text(text)
        self.logger.debug(f"Sent text: {text}")
    
    async def interrupt(self) -> None:
        """Interrupt the current AI response"""
        if not self._strategy:
            raise EngineError("Not connected")
        
        await self._strategy.interrupt()
        self.logger.debug("Interrupted current response")
    
    # ============== Metrics and Usage ==============
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            "connected": self._is_connected,
            "listening": self._is_listening,
            "uptime": (
                asyncio.get_event_loop().time() - self._session_start_time
                if self._session_start_time else 0
            )
        }
        
        # Strategy metrics
        if self._strategy:
            try:
                strategy_metrics = self._strategy.get_metrics()
                metrics.update(strategy_metrics)
            except Exception as e:
                self.logger.error(f"Error getting strategy metrics: {e}")
        
        # Audio metrics
        if self._audio_manager:
            try:
                metrics["audio"] = self._audio_manager.get_metrics()
            except Exception as e:
                self.logger.error(f"Error getting audio metrics: {e}")
                metrics["audio"] = {"error": str(e)}
        
        return metrics
    
    async def get_usage(self):
        """Get usage statistics"""
        if self._strategy:
            return self._strategy.get_usage()
        
        # Return empty usage if no strategy
        from .core.provider_protocol import Usage
        return Usage()
    
    async def estimate_cost(self):
        """Estimate cost of current session"""
        if self._strategy:
            return await self._strategy.estimate_cost()
        
        # Return zero cost if no strategy
        from .core.provider_protocol import Cost
        return Cost()
    
    # ============== Audio Playback ==============
    
    def play_audio(self, audio_data: AudioBytes) -> None:
        """Play audio through the audio manager"""
        if self._audio_manager:
            self._audio_manager.play_audio(audio_data)
    
    # ============== Cleanup ==============
    
    async def cleanup(self) -> None:
        """Cleanup all resources safely"""
        try:
            # Mark as not listening first
            self._is_listening = False
            
            # Cancel audio processing task
            if self._audio_processing_task and not self._audio_processing_task.done():
                self._audio_processing_task.cancel()
                try:
                    await asyncio.wait_for(self._audio_processing_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._audio_processing_task = None
            
            # Cleanup audio manager
            if self._audio_manager:
                try:
                    await self._audio_manager.cleanup()
                except Exception as e:
                    self.logger.error(f"Audio manager cleanup error: {e}")
                self._audio_manager = None
            
            # Disconnect strategy if connected
            if self._is_connected and self._strategy:
                try:
                    await self._strategy.disconnect()
                except Exception as e:
                    self.logger.error(f"Strategy disconnect error: {e}")
                self._is_connected = False
            
            # Clear references
            self._strategy = None
            self._event_handlers.clear()
            self._session_start_time = None
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            # Don't re-raise to avoid cascading failures