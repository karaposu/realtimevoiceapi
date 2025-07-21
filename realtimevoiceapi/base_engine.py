"""
Base Voice Engine Implementation with Component and Event Handler Organization

Contains all the internal implementation details for the voice engine.
This is not meant to be used directly by users - they should use VoiceEngine instead.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List, Literal
from dataclasses import dataclass, field



from .core.stream_protocol import StreamEvent, StreamEventType, StreamState
from audioengine.audioengine.audio_types import AudioBytes
from .core.exceptions import EngineError
from .strategies.base_strategy import BaseStrategy, EngineConfig
from .strategies.fast_lane_strategy import FastLaneStrategy

# Import AudioEngine from the audioengine package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from audioengine.audioengine.audio_engine import AudioEngine, create_fast_lane_engine
from audioengine.audioengine.audio_types import AudioConfig, VADConfig, VADType, ProcessingMode


@dataclass
class CoreEngineState:
    """Core engine state tracking"""
    
    # Connection state
    is_connected: bool = False
    connection_time: Optional[float] = None
    
    # Audio input state
    is_listening: bool = False
    listening_start_time: Optional[float] = None
    
    # Audio output state
    response_audio_started: bool = False
    response_audio_complete: bool = False
    
    # Session info
    session_start_time: Optional[float] = None
    stream_state: StreamState = StreamState.IDLE
    
    # Metrics
    total_interactions: int = 0
    total_audio_chunks_sent: int = 0
    total_audio_chunks_received: int = 0
    
    @property
    def is_ai_speaking(self) -> bool:
        """Check if AI is currently speaking"""
        return self.response_audio_started and not self.response_audio_complete
    
    @property
    def uptime(self) -> float:
        """Get session uptime in seconds"""
        if self.session_start_time:
            return time.time() - self.session_start_time
        return 0.0
    
    @property
    def is_ready(self) -> bool:
        """Check if ready for new interactions"""
        return (
            self.is_connected and 
            self.stream_state == StreamState.ACTIVE and
            not self.is_ai_speaking
        )
    
    def reset_response_state(self):
        """Reset response-related state"""
        self.response_audio_started = False
        self.response_audio_complete = False
    
    def mark_interaction(self):
        """Mark a new interaction"""
        self.total_interactions += 1
        self.reset_response_state()
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for metrics"""
        return {
            "is_connected": self.is_connected,
            "is_listening": self.is_listening,
            "is_ai_speaking": self.is_ai_speaking,
            "is_ready": self.is_ready,
            "stream_state": self.stream_state.value,
            "uptime": self.uptime,
            "total_interactions": self.total_interactions,
            "audio_chunks_sent": self.total_audio_chunks_sent,
            "audio_chunks_received": self.total_audio_chunks_received
        }


@dataclass
class EngineComponents:
    """Container for all engine components"""
    strategy: Optional[BaseStrategy] = None
    audio_engine: Optional[AudioEngine] = None
    
    # Processing tasks
    audio_processing_task: Optional[asyncio.Task] = None
    
    def cleanup_tasks(self):
        """Cancel all tasks"""
        if self.audio_processing_task and not self.audio_processing_task.done():
            self.audio_processing_task.cancel()
    
    def has_audio(self) -> bool:
        """Check if audio components are available"""
        return self.audio_engine is not None
    
    def clear(self):
        """Clear all component references"""
        self.strategy = None
        self.audio_engine = None
        self.audio_processing_task = None


@dataclass
class EventHandlerRegistry:
    """Manages event handlers with wrapping and state tracking"""
    
    # Handler storage
    user_handlers: Dict[StreamEventType, Callable] = field(default_factory=dict)
    wrapped_handlers: Dict[StreamEventType, Callable] = field(default_factory=dict)
    
    # Handler call counts for metrics
    handler_calls: Dict[StreamEventType, int] = field(default_factory=dict)
    handler_errors: Dict[StreamEventType, int] = field(default_factory=dict)
    
    # Logger reference
    logger: Optional[logging.Logger] = None
    
    def register_handler(self, event_type: StreamEventType, handler: Callable):
        """Register a user handler"""
        self.user_handlers[event_type] = handler
        self.handler_calls[event_type] = 0
        self.handler_errors[event_type] = 0
    
    def wrap_handler(self, event_type: StreamEventType, wrapper: Callable) -> Callable:
        """Wrap a handler with additional functionality"""
        original = self.user_handlers.get(event_type)
        if not original:
            return wrapper
        
        def wrapped_handler(event: StreamEvent):
            # Track calls
            self.handler_calls[event_type] = self.handler_calls.get(event_type, 0) + 1
            
            try:
                # Call wrapper with original
                return wrapper(original, event)
            except Exception as e:
                self.handler_errors[event_type] = self.handler_errors.get(event_type, 0) + 1
                if self.logger:
                    self.logger.error(f"Handler error for {event_type}: {e}")
                raise
        
        self.wrapped_handlers[event_type] = wrapped_handler
        return wrapped_handler
    
    def get_handler(self, event_type: StreamEventType) -> Optional[Callable]:
        """Get wrapped handler if available, otherwise user handler"""
        return self.wrapped_handlers.get(event_type) or self.user_handlers.get(event_type)
    
    def get_all_handlers(self) -> Dict[StreamEventType, Callable]:
        """Get all active handlers (wrapped or user)"""
        handlers = {}
        for event_type in set(self.user_handlers.keys()) | set(self.wrapped_handlers.keys()):
            handler = self.get_handler(event_type)
            if handler:
                handlers[event_type] = handler
        return handlers
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get handler metrics"""
        return {
            "registered_handlers": len(self.user_handlers),
            "handler_calls": dict(self.handler_calls),
            "handler_errors": dict(self.handler_errors)
        }
    
    def clear(self):
        """Clear all handlers"""
        self.user_handlers.clear()
        self.wrapped_handlers.clear()
        self.handler_calls.clear()
        self.handler_errors.clear()


class BaseEngine:
    """
    Base implementation for voice engine.
    
    Handles all the complex internal logic, state management,
    and coordination between components.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize base engine"""
        self.logger = logger or logging.getLogger(__name__)
        
        # Core state
        self.state = CoreEngineState()
        
        # Components
        self.components = EngineComponents()
        
        # Event handler registry
        self.event_registry = EventHandlerRegistry(logger=self.logger)
        
        # Configuration cache
        self._config: Optional[EngineConfig] = None
        self._mode: Optional[str] = None
        
    # ============== State Properties ==============
    
    @property
    def is_connected(self) -> bool:
        """Check if properly connected"""
        return self.state.is_connected and self.components.strategy is not None
    
    @property
    def is_listening(self) -> bool:
        """Check if actively listening"""
        return self.state.is_listening
    
    @property
    def is_ai_speaking(self) -> bool:
        """Check if AI is currently speaking"""
        if self.components.audio_engine:
            metrics = self.components.audio_engine.get_metrics()
            return metrics.get('is_playing', False)
        return self.state.is_ai_speaking
    
    @property
    def strategy(self) -> Optional[BaseStrategy]:
        """Get current strategy"""
        return self.components.strategy
    
    def get_state(self) -> StreamState:
        """Get current stream state"""
        if self.components.strategy:
            self.state.stream_state = self.components.strategy.get_state()
        return self.state.stream_state
    
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
            self.components.strategy = FastLaneStrategy(logger=self.logger)
        elif mode == "big":
            # TODO: Implement big lane strategy
            raise NotImplementedError(
                "Big lane strategy not yet implemented. Please use mode='fast' for now."
            )
            # self.components.strategy = BigLaneStrategy(logger=self.logger)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fast' or 'big'")
        
        return self.components.strategy
    
    async def initialize_strategy(self, config: EngineConfig) -> None:
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Engine configuration
        """
        if not self.components.strategy:
            raise EngineError("Strategy not created. Call create_strategy first.")
        
        self._config = config
        
        # Initialize strategy only if not already initialized
        if not self.components.strategy._is_initialized:
            await self.components.strategy.initialize(config)
    
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
        """Setup AudioEngine for fast lane"""
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
            
            # Create AudioEngine optimized for fast lane with all configurations
            self.components.audio_engine = create_fast_lane_engine(
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration_ms,
                input_device=input_device,
                output_device=output_device,
                vad_config=vad_config
            )
            
            # Set playback callbacks
            self.components.audio_engine.set_playback_callbacks(
                completion_callback=self._on_audio_playback_complete,
                chunk_played_callback=self._on_chunks_played
            )
            
            self.logger.info("AudioEngine initialized for fast lane")
            
        except Exception as e:
            # Clean up if initialization fails
            self.logger.error(f"Failed to setup audio: {e}")
            self.components.audio_engine = None
            raise
    
    # ============== Connection Management ==============
    
    async def do_connect(self) -> None:
        """
        Internal connection logic.
        
        Handles strategy connection and event handler setup.
        """
        if not self.components.strategy:
            raise EngineError("Strategy not initialized")
        
        # Connect to provider
        await self.components.strategy.connect()
        
        # Update state
        self.state.is_connected = True
        self.state.connection_time = time.time()
        self.state.session_start_time = self.state.session_start_time or time.time()
        self.state.stream_state = StreamState.ACTIVE
        
        self.logger.info("Successfully connected to voice API")
    
    async def do_disconnect(self) -> None:
        """Internal disconnection logic"""
        if not self.state.is_connected:
            return
        
        try:
            # Use the comprehensive cleanup
            await self.cleanup()
            
            # Update state
            self.state.is_connected = False
            self.state.stream_state = StreamState.ENDED
            
            self.logger.info("Disconnected from voice API")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            raise
    
    # ============== Audio Processing ==============
    
    async def start_audio_processing(self) -> None:
        """Start audio input processing"""
        if self.state.is_listening:
            self.logger.warning("Already listening")
            return
        
        # Reset audio state
        self.state.reset_response_state()
        
        # Start audio input through strategy
        await self.components.strategy.start_audio_input()
        
        # Update state
        self.state.is_listening = True
        self.state.listening_start_time = time.time()
        
        # For fast lane with audio engine, start processing loop
        if self._mode == "fast" and self.components.audio_engine:
            self.components.audio_processing_task = asyncio.create_task(
                self._audio_processing_loop()
            )
        
        self.logger.info("Started listening for audio input")
    
    async def stop_audio_processing(self) -> None:
        """Stop audio input processing"""
        if not self.state.is_listening:
            return
        
        # Update state first
        self.state.is_listening = False
        
        # Stop audio processing task
        if self.components.audio_processing_task:
            self.components.audio_processing_task.cancel()
            try:
                await self.components.audio_processing_task
            except asyncio.CancelledError:
                pass
            self.components.audio_processing_task = None
        
        # Stop audio capture through audio engine
        if self.components.audio_engine:
            await self.components.audio_engine.stop_capture_stream()
        
        # Stop audio input through strategy
        await self.components.strategy.stop_audio_input()
        
        self.logger.info("Stopped listening for audio input")
    
    async def _audio_processing_loop(self) -> None:
        """Process audio input for fast lane using AudioEngine"""
        if not self.components.audio_engine:
            self.logger.error("No audio engine available")
            return
        
        # Get audio queue from audio engine
        try:
            audio_queue = await self.components.audio_engine.start_capture_stream()
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            return
        
        try:
            while self.state.is_listening and self.state.is_connected:
                try:
                    # Get audio chunk from queue
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    
                    # Check if we should still process
                    if not self.state.is_listening or not self.state.is_connected:
                        break
                    
                    # Check strategy state
                    if self.components.strategy and self.components.strategy.get_state() in [
                        StreamState.ACTIVE, StreamState.STARTING
                    ]:
                        # Process through VAD
                        vad_state = self.components.audio_engine.process_vad_chunk(audio_chunk)
                        
                        # Send if speech or no VAD
                        if not vad_state or vad_state in ["speech_starting", "speech"]:
                            await self.components.strategy.send_audio(audio_chunk)
                            self.state.total_audio_chunks_sent += 1
                    
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    self.logger.debug("Audio processing cancelled")
                    break
                except Exception as e:
                    # Only log error if we're still supposed to be running
                    if self.state.is_listening and self.state.is_connected:
                        self.logger.error(f"Audio processing error: {e}")
                        # Notify error handler if set
                        error_handler = self.event_registry.get_handler(StreamEventType.STREAM_ERROR)
                        if error_handler:
                            error_event = StreamEvent(
                                type=StreamEventType.STREAM_ERROR,
                                stream_id="unknown",
                                timestamp=time.time(),
                                data={"error": str(e)}
                            )
                            error_handler(error_event)
                        
        except Exception as e:
            self.logger.error(f"Fatal audio processing error: {e}")
        finally:
            # Stop capture through audio engine
            if self.components.audio_engine:
                await self.components.audio_engine.stop_capture_stream()
    
    # ============== Event Management ==============
    
    def setup_event_handlers(self, handlers: Dict[StreamEventType, Callable]) -> None:
        """
        Setup all event handlers at once with audio buffering integration.
        
        Args:
            handlers: Dictionary mapping event types to handler functions
        """
        # Register all user handlers
        for event_type, handler in handlers.items():
            self.event_registry.register_handler(event_type, handler)
        
        # Create wrapper for audio handler
        if StreamEventType.AUDIO_OUTPUT_CHUNK in handlers:
            def audio_wrapper(original_handler, event: StreamEvent):
                # Update state
                if not self.state.response_audio_started:
                    self.state.response_audio_started = True
                    self.logger.debug("Response audio started")
                
                self.state.total_audio_chunks_received += 1
                
                # Call original handler
                original_handler(event)
                
                # Play through audio engine if available
                if event.data and "audio" in event.data and self.components.audio_engine:
                    # Queue audio for playback
                    audio_bytes = event.data["audio"]
                    self.components.audio_engine.queue_playback(audio_bytes)
            
            self.event_registry.wrap_handler(StreamEventType.AUDIO_OUTPUT_CHUNK, audio_wrapper)
        
        # Create wrapper for stream ended handler
        if StreamEventType.STREAM_ENDED in handlers:
            def ended_wrapper(original_handler, event: StreamEvent):
                # Mark audio complete for audio engine
                if self.state.response_audio_started and self.components.audio_engine:
                    self.components.audio_engine.mark_playback_complete()
                    self.state.response_audio_complete = True
                    self.logger.debug("Marked audio complete for audio engine")
                
                # Call original
                original_handler(event)
            
            self.event_registry.wrap_handler(StreamEventType.STREAM_ENDED, ended_wrapper)
        
        # Pass all handlers to strategy
        if self.components.strategy:
            all_handlers = self.event_registry.get_all_handlers()
            for event_type, handler in all_handlers.items():
                self.components.strategy.set_event_handler(event_type, handler)
        
        # Special handling for fast lane response done callback
        if (self._mode == "fast" and 
            hasattr(self.components.strategy, 'stream_manager') and 
            self.components.strategy.stream_manager):
            
            if hasattr(self.components.strategy.stream_manager, 'set_response_done_callback'):
                # Create wrapper for response done
                def response_done_wrapper():
                    # Mark audio complete
                    if self.state.response_audio_started and self.components.audio_engine:
                        self.components.audio_engine.mark_playback_complete()
                        self.state.response_audio_complete = True
                    
                    stream_ended_handler = self.event_registry.get_handler(StreamEventType.STREAM_ENDED)
                    if stream_ended_handler:
                        # Create a synthetic event
                        event = StreamEvent(
                            type=StreamEventType.STREAM_ENDED,
                            stream_id=self.components.strategy.stream_manager.stream_id,
                            timestamp=time.time(),
                            data={}
                        )
                        stream_ended_handler(event)
                
                self.components.strategy.stream_manager.set_response_done_callback(response_done_wrapper)
    
    def set_event_handler(self, event_type: StreamEventType, handler: Callable) -> None:
        """Set a single event handler"""
        self.event_registry.register_handler(event_type, handler)
        
        if self.components.strategy:
            # Get the handler (might be wrapped)
            active_handler = self.event_registry.get_handler(event_type)
            if active_handler:
                self.components.strategy.set_event_handler(event_type, active_handler)
    
    # ============== Data Transmission ==============
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio data to the API"""
        if not self.components.strategy:
            raise EngineError("Not connected")
        
        # Mark interaction and reset response state
        self.state.mark_interaction()
        
        # Send audio
        await self.components.strategy.send_audio(audio_data)
        
        # Update metrics
        self.state.total_audio_chunks_sent += 1
    
    async def send_text(self, text: str) -> None:
        """Send text message to the API"""
        if not self.components.strategy:
            raise EngineError("Not connected")
        
        # Mark interaction and reset response state
        self.state.mark_interaction()
        
        await self.components.strategy.send_text(text)
        self.logger.debug(f"Sent text: {text}")
    
    async def interrupt(self) -> None:
        """Interrupt the current AI response"""
        if not self.components.strategy:
            raise EngineError("Not connected")
        
        # Stop audio immediately through audio engine
        if self.components.audio_engine:
            self.components.audio_engine.interrupt_playback(force=True)
            self.logger.debug("Interrupted audio playback")
        
        # Reset audio state
        self.state.reset_response_state()
        
        # Interrupt through strategy
        await self.components.strategy.interrupt()
        self.logger.debug("Interrupted current response")
    
    # ============== Metrics and Usage ==============
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        # Start with state metrics
        metrics = self.state.to_dict()
        
        # Add event handler metrics
        metrics["event_handlers"] = self.event_registry.get_metrics()
        
        # Add component metrics
        if self.components.strategy:
            try:
                strategy_metrics = self.components.strategy.get_metrics()
                metrics["strategy"] = strategy_metrics
            except Exception as e:
                self.logger.error(f"Error getting strategy metrics: {e}")
        
        if self.components.audio_engine:
            try:
                audio_metrics = self.components.audio_engine.get_metrics()
                metrics["audio_engine"] = audio_metrics
            except Exception as e:
                self.logger.error(f"Error getting audio metrics: {e}")
                metrics["audio_engine"] = {"error": str(e)}
        
        return metrics
    
    async def get_usage(self):
        """Get usage statistics"""
        if self.components.strategy:
            return self.components.strategy.get_usage()
        
        # Return empty usage if no strategy
        from .core.provider_protocol import Usage
        return Usage()
    
    async def estimate_cost(self):
        """Estimate cost of current session"""
        if self.components.strategy:
            return await self.components.strategy.estimate_cost()
        
        # Return zero cost if no strategy
        from .core.provider_protocol import Cost
        return Cost()
    
    # ============== Audio Playback ==============
    
    def play_audio(self, audio_data: AudioBytes) -> None:
        """Play audio through the appropriate player"""
        # If we're in a response, audio is already being handled
        if self.state.response_audio_started:
            return
        
        # Otherwise use audio engine for direct playback
        if self.components.audio_engine:
            # Process and play audio
            processed = self.components.audio_engine.process_audio(audio_data)
            # TODO: Add direct playback method to AudioEngine
    
    def _on_audio_playback_complete(self):
        """Called when audio playback completes"""
        self.logger.info("Audio playback fully complete")
        
        # Get final metrics
        if self.components.audio_engine:
            metrics = self.components.audio_engine.get_metrics()
            self.logger.info(f"Final playback metrics: {metrics}")
        
        # Reset state
        self.state.response_audio_started = False
        self.state.response_audio_complete = False
        
        # Notify user's handler if exists
        stream_ended_handler = self.event_registry.user_handlers.get(StreamEventType.STREAM_ENDED)
        if stream_ended_handler:
            # Create event
            event = StreamEvent(
                type=StreamEventType.STREAM_ENDED,
                stream_id=self.components.strategy.stream_id if self.components.strategy else "unknown",
                timestamp=time.time(),
                data={
                    "reason": "audio_playback_complete",
                    "playback_metrics": self.components.audio_engine.get_metrics() if self.components.audio_engine else {}
                }
            )
            
            # Call in asyncio context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._emit_event_async(event, stream_ended_handler))
                else:
                    # Fallback for thread context
                    stream_ended_handler(event)
            except Exception as e:
                self.logger.error(f"Error emitting playback complete event: {e}")
    
    async def _emit_event_async(self, event: StreamEvent, handler: Callable):
        """Emit event in async context"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            self.logger.error(f"Event handler error: {e}")
    
    def _on_chunks_played(self, num_chunks: int):
        """Track chunks being played"""
        self.logger.debug(f"Played {num_chunks} audio chunks")
    
    # ============== Cleanup ==============
    
    async def cleanup(self) -> None:
        """Cleanup all resources safely"""
        try:
            # Update state
            self.state.is_listening = False
            self.state.is_connected = False
            self.state.stream_state = StreamState.ENDED
            
            # Cancel all tasks through components
            self.components.cleanup_tasks()
            
            # Wait for audio processing task if it exists
            if self.components.audio_processing_task:
                try:
                    await asyncio.wait_for(self.components.audio_processing_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Cleanup audio engine
            if self.components.audio_engine:
                try:
                    await self.components.audio_engine.cleanup_async()
                except Exception as e:
                    self.logger.error(f"Audio engine cleanup error: {e}")
            
            # Disconnect strategy if connected
            if self.components.strategy:
                try:
                    await self.components.strategy.disconnect()
                except Exception as e:
                    self.logger.error(f"Strategy disconnect error: {e}")
            
            # Clear all components
            self.components.clear()
            
            # Clear event handlers
            self.event_registry.clear()
            
            # Reset state
            self.state.reset_response_state()
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            # Don't re-raise to avoid cascading failures