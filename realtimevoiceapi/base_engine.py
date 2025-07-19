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


# For fast lane direct imports
from .fast_lane.direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from .fast_lane.fast_vad_detector import FastVADDetector


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

        self._audio_manager: Optional[AudioManager] = None
        
        # # Audio components (for fast lane)
        # self.audio_capture: Optional[DirectAudioCapture] = None
        # self.audio_player: Optional[DirectAudioPlayer] = None
        # self.vad_detector: Optional[FastVADDetector] = None
        
        # State
        self._is_connected = False
        self._is_listening = False
        self._audio_processing_task: Optional[asyncio.Task] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        
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
        """Setup audio components for fast lane"""
        try:

         

            # Create audio manager config
            audio_config = AudioManagerConfig(
                input_device=input_device,
                output_device=output_device,
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration_ms,
                vad_enabled=vad_enabled,
                vad_config=vad_config
            )
                
            # Setup VAD if enabled
            if vad_enabled:
                vad_config = VADConfig(
                    type=VADType.ENERGY_BASED,
                    energy_threshold=vad_threshold,
                    speech_start_ms=vad_speech_start_ms,
                    speech_end_ms=vad_speech_end_ms
                )
                
                self.vad_detector = FastVADDetector(
                    config=vad_config,
                    audio_config=audio_config
                )
            
            # Setup audio capture
            self.audio_capture = DirectAudioCapture(
                device=input_device,
                config=audio_config
            )
            
            # Setup audio player
            self.audio_player = DirectAudioPlayer(
                device=output_device,
                config=audio_config
            )
            
        except Exception as e:
            # Clean up any partially created resources
            self.logger.error(f"Failed to setup audio: {e}")
            self.audio_capture = None
            self.audio_player = None
            self.vad_detector = None
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
            # Stop listening first
            if self._is_listening:
                await self.stop_audio_processing()
            
            # Disconnect strategy
            if self._strategy:
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
            raise
    
    # ============== Audio Processing ==============
    
    async def start_audio_processing(self) -> None:
        """Start audio input processing"""
        if self._is_listening:
            self.logger.warning("Already listening")
            return
        
        # Start audio input through strategy
        await self._strategy.start_audio_input()
        
        # For fast lane with VAD, start processing loop
        if self._mode == "fast" and self.vad_detector:
            self._audio_processing_task = asyncio.create_task(
                self._audio_processing_loop()
            )
        
        self._is_listening = True
        self.logger.info("Started listening for audio input")
    
    async def stop_audio_processing(self) -> None:
        """Stop audio input processing"""
        if not self._is_listening:
            return
        
        # Stop audio processing task
        if self._audio_processing_task:
            self._audio_processing_task.cancel()
            try:
                await self._audio_processing_task
            except asyncio.CancelledError:
                pass
            self._audio_processing_task = None
        
        # Stop audio input through strategy
        await self._strategy.stop_audio_input()
        
        self._is_listening = False
        self.logger.info("Stopped listening for audio input")



    # In base_engine.py, update _audio_processing_loop:

    async def _audio_processing_loop(self) -> None:
        """
        Process audio input for fast lane with VAD.
        
        This runs in a background task when listening is active.
        """
        if not self.audio_capture:
            self.logger.error("No audio capture available")
            return
        
        # Get audio queue from capture
        self._audio_queue = await self.audio_capture.start_async_capture()
        
        try:
            while self._is_listening and self._is_connected:  # Check both flags
                try:
                    # Get audio chunk
                    audio_chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.1
                    )
                    
                    # Check if we should still process
                    if not self._is_listening or not self._is_connected:
                        break
                    
                    # Check strategy state before sending
                    if self._strategy and self._strategy.get_state() in [
                        StreamState.ACTIVE, StreamState.STARTING
                    ]:
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
            # Ensure capture is stopped
            if self.audio_capture:
                try:
                    self.audio_capture.stop_capture()
                except Exception as e:
                    self.logger.debug(f"Error stopping capture in cleanup: {e}")
    
    
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
        
        if self._strategy:
            try:
                strategy_metrics = self._strategy.get_metrics()
                metrics.update(strategy_metrics)
            except Exception as e:
                self.logger.error(f"Error getting strategy metrics: {e}")
                
        components = {
                "audio_capture": self.audio_capture,
                "audio_player": self.audio_player,
                "vad": self.vad_detector
            }
        

        for name, component in components.items():
            if component and hasattr(component, 'get_metrics'):
                try:
                    component_metrics = component.get_metrics()
                    metrics[name] = component_metrics
                except Exception as e:
                    self.logger.error(f"Error getting {name} metrics: {e}")
                    metrics[name] = {"error": str(e)}
        
        return metrics
        

    async def get_usage(self):
        """Get usage statistics"""
        if self._strategy:
            return self._strategy.get_usage()
        
        # Return empty usage if no strategy
        from ..core.provider_protocol import Usage
        return Usage()
    
    async def estimate_cost(self):
        """Estimate cost of current session"""
        if self._strategy:
            return await self._strategy.estimate_cost()
        
        # Return zero cost if no strategy
        from ..core.provider_protocol import Cost
        return Cost()
    
    # ============== Audio Playback ==============
    
    def play_audio(self, audio_data: AudioBytes) -> None:
        """Play audio through the audio player"""
        if self.audio_player:
            self.audio_player.play_audio(audio_data)
    
    # ============== Cleanup ==============
    
    async def cleanup(self) -> None:
        """Cleanup all resources safely"""
        try:
            # First, stop any audio processing
            if self._audio_processing_task and not self._audio_processing_task.done():
                self._audio_processing_task.cancel()
                try:
                    await asyncio.wait_for(self._audio_processing_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._audio_processing_task = None
            
            # Stop audio capture
            if self.audio_capture and hasattr(self.audio_capture, 'stop_capture'):
                try:
                    self.audio_capture.stop_capture()
                except Exception as e:
                    self.logger.error(f"Error stopping audio capture: {e}")
            
            # Stop audio playback
            if self.audio_player and hasattr(self.audio_player, 'stop_playback'):
                try:
                    self.audio_player.stop_playback()
                except Exception as e:
                    self.logger.error(f"Error stopping audio player: {e}")
            
            # Clear audio queue
            if self._audio_queue:
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._audio_queue = None
            
            # Now disconnect if connected
            if self._is_connected:
                try:
                    if self._strategy:
                        await self._strategy.disconnect()
                except Exception as e:
                    self.logger.error(f"Strategy disconnect error: {e}")
                self._is_connected = False
            
            # Finally, clear all references
            self._strategy = None
            self.audio_capture = None
            self.audio_player = None
            self.vad_detector = None
            self._event_handlers.clear()
            
            # Reset state
            self._is_listening = False
            self._session_start_time = None
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            # Don't re-raise to avoid cascading failures

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