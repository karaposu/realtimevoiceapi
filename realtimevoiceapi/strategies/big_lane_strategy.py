"""
Big Lane Strategy Implementation

Full-featured implementation with all abstractions and capabilities.
Supports multiple providers, advanced features, and complex workflows.


Big Lane Strategy:

Full abstraction - Event bus, pipelines, orchestration
Event-driven - Flexible but higher overhead
Queue-based - Handles backpressure
Multi-provider - Provider abstraction
Full features - Transcription, functions, etc
Dynamic allocation - Flexibility over performance

"""

import asyncio
from typing import Optional, Dict, Any, List, AsyncIterator, Callable
import logging
from collections import defaultdict

from .base_strategy import BaseStrategy, EngineConfig
from ..stream_protocol import (
    StreamEvent, StreamEventType, StreamState,
    IStreamManager, StreamConfig, AudioFormat
)
from ..audio_types import AudioBytes, AudioConfig
from ..provider_protocol import (
    Usage, Cost, IVoiceProvider, ProviderRegistry,
    ProviderConfig, IProviderSession
)
from ..websocket_connection import BigLaneConnection
from ..message_protocol import MessageFactory, MessageValidator
from ..session_manager import SessionManager
from ..exceptions import EngineError

# These would be implemented in separate files
from ..big_lane.audio_pipeline import AudioPipeline, AudioProcessor
from ..big_lane.event_bus import EventBus, Event
from ..big_lane.stream_orchestrator import StreamOrchestrator
from ..big_lane.response_aggregator import ResponseAggregator


class BigLaneStrategy(BaseStrategy):
    """
    Big lane implementation with full features.
    
    Supports:
    - Multiple providers
    - Event-driven architecture
    - Audio pipeline processing
    - Advanced features (transcription, functions, etc)
    - Provider failover
    - Complex workflows
    """
    
    def __init__(
        self,
        provider_registry: Optional[ProviderRegistry] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.config: Optional[EngineConfig] = None
        
        # Core components
        self.provider_registry = provider_registry or ProviderRegistry()
        self.session_manager = SessionManager()
        self.event_bus = EventBus()
        
        # Processing components
        self.audio_pipeline: Optional[AudioPipeline] = None
        self.stream_orchestrator: Optional[StreamOrchestrator] = None
        self.response_aggregator: Optional[ResponseAggregator] = None
        
        # Active provider session
        self.provider: Optional[IVoiceProvider] = None
        self.provider_session: Optional[IProviderSession] = None
        self.active_session_id: Optional[str] = None
        
        # Event handlers
        self._event_handlers: Dict[StreamEventType, List[Callable]] = defaultdict(list)
        
        # State
        self._state = StreamState.IDLE
        self._is_initialized = False
        
        # Usage tracking (comprehensive)
        self._usage = Usage()
        self._provider_usage: Dict[str, Usage] = {}
        
        # Audio input queue
        self._audio_input_queue = asyncio.Queue(maxsize=100)
        self._audio_processor_task: Optional[asyncio.Task] = None
    
    async def initialize(self, config: EngineConfig) -> None:
        """Initialize big lane components"""
        if self._is_initialized:
            raise EngineError("Already initialized")
        
        self.config = config
        
        # Initialize event bus
        self.event_bus.start()
        
        # Initialize audio pipeline
        audio_config = AudioConfig(
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            chunk_duration_ms=200 if config.latency_mode == "quality" else 100
        )
        
        self.audio_pipeline = AudioPipeline(
            config=audio_config,
            processors=self._create_audio_processors(config)
        )
        
        # Initialize stream orchestrator
        self.stream_orchestrator = StreamOrchestrator(
            event_bus=self.event_bus,
            logger=self.logger
        )
        
        # Initialize response aggregator
        self.response_aggregator = ResponseAggregator(
            event_bus=self.event_bus
        )
        
        # Get provider
        self.provider = self.provider_registry.get(config.provider)
        
        # Subscribe to internal events
        self._setup_event_subscriptions()
        
        self._is_initialized = True
        self._state = StreamState.IDLE
        
        self.logger.info(f"Big lane strategy initialized with provider: {config.provider}")
    
    async def connect(self) -> None:
        """Connect to provider"""
        if not self._is_initialized:
            raise EngineError("Not initialized")
        
        if self._state == StreamState.ACTIVE:
            return
        
        self._state = StreamState.STARTING
        
        try:
            # Create provider configuration
            provider_config = ProviderConfig(
                api_key=self.config.api_key,
                metadata=self.config.metadata
            )
            
            # Create stream configuration
            stream_config = StreamConfig(
                provider=self.config.provider,
                mode="both",  # audio and text
                audio_format=AudioFormat(),
                enable_vad=self.config.enable_vad,
                metadata={
                    "enable_transcription": self.config.enable_transcription,
                    "enable_functions": self.config.enable_functions
                }
            )
            
            # Create provider session
            self.provider_session = await self.provider.create_session(
                provider_config,
                stream_config
            )
            
            # Register session
            session = self.session_manager.create_session(
                provider=self.config.provider,
                stream_id=self.provider_session.session_id,
                config=stream_config.metadata
            )
            self.active_session_id = session.id
            
            # Start audio processor
            self._audio_processor_task = asyncio.create_task(
                self._audio_processor_loop()
            )
            
            # Start response handler
            asyncio.create_task(self._response_handler_loop())
            
            self._state = StreamState.ACTIVE
            
            # Emit connected event
            await self.event_bus.emit(Event(
                type="engine.connected",
                data={"provider": self.config.provider, "session_id": session.id}
            ))
            
        except Exception as e:
            self._state = StreamState.ERROR
            self.logger.error(f"Connection failed: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from provider"""
        if self._state != StreamState.ACTIVE:
            return
        
        self._state = StreamState.ENDING
        
        try:
            # Stop audio processor
            if self._audio_processor_task:
                self._audio_processor_task.cancel()
                try:
                    await self._audio_processor_task
                except asyncio.CancelledError:
                    pass
            
            # End provider session
            if self.provider_session:
                usage = await self.provider_session.end_session()
                self._usage.add(usage)
                
                # Track provider-specific usage
                self._provider_usage[self.config.provider] = usage
            
            # Update session state
            if self.active_session_id:
                self.session_manager.end_session(self.active_session_id)
            
            # Stop event bus
            await self.event_bus.stop()
            
            self._state = StreamState.ENDED
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            self._state = StreamState.ERROR
    
    async def start_audio_input(self) -> None:
        """Start audio input processing"""
        if self._state != StreamState.ACTIVE:
            raise EngineError("Not connected")
        
        # Big lane uses queue-based audio input
        # Audio capture would be handled by a separate component
        
        await self.event_bus.emit(Event(
            type="audio.input.started",
            data={}
        ))
    
    async def stop_audio_input(self) -> None:
        """Stop audio input processing"""
        # Clear audio queue
        while not self._audio_input_queue.empty():
            try:
                self._audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        await self.event_bus.emit(Event(
            type="audio.input.stopped",
            data={}
        ))
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio through pipeline"""
        if self._state != StreamState.ACTIVE:
            raise EngineError("Not connected")
        
        # Queue audio for processing
        try:
            await asyncio.wait_for(
                self._audio_input_queue.put(audio_data),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("Audio input queue full, dropping chunk")
            
            # Emit queue full event
            await self.event_bus.emit(Event(
                type="audio.queue.full",
                data={"dropped_bytes": len(audio_data)}
            ))
    
    async def send_text(self, text: str) -> None:
        """Send text message"""
        if self._state != StreamState.ACTIVE:
            raise EngineError("Not connected")
        
        # Send through provider session
        await self.provider_session.send_text(text)
        
        # Track usage
        self._usage.text_input_tokens += self._estimate_tokens(text)
        
        # Emit event
        await self.event_bus.emit(Event(
            type="text.sent",
            data={"text": text}
        ))
    
    async def get_response_stream(self) -> AsyncIterator[StreamEvent]:
        """Get response event stream"""
        # Subscribe to response events
        response_queue = asyncio.Queue()
        
        def queue_event(event: Event):
            # Convert internal event to stream event
            stream_event = StreamEvent(
                type=self._map_event_type(event.type),
                stream_id=self.active_session_id or "unknown",
                timestamp=event.timestamp,
                data=event.data
            )
            asyncio.create_task(response_queue.put(stream_event))
        
        # Subscribe to relevant events
        self.event_bus.subscribe("audio.output.*", queue_event)
        self.event_bus.subscribe("text.output.*", queue_event)
        self.event_bus.subscribe("function.call.*", queue_event)
        
        # Yield events as they come
        while self._state == StreamState.ACTIVE:
            try:
                event = await asyncio.wait_for(response_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue
    
    def set_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Set event handler"""
        self._event_handlers[event_type].append(handler)
        
        # Map to internal events
        internal_event = self._map_to_internal_event(event_type)
        
        def wrapper(event: Event):
            stream_event = StreamEvent(
                type=event_type,
                stream_id=self.active_session_id or "unknown",
                timestamp=event.timestamp,
                data=event.data
            )
            handler(stream_event)
        
        self.event_bus.subscribe(internal_event, wrapper)
    
    async def interrupt(self) -> None:
        """Interrupt current response"""
        if self.provider_session:
            await self.provider_session.interrupt()
            
            # Clear response aggregator
            self.response_aggregator.clear_current()
            
            # Emit interrupt event
            await self.event_bus.emit(Event(
                type="response.interrupted",
                data={}
            ))
    
    def get_state(self) -> StreamState:
        """Get current state"""
        return self._state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            "strategy": "big_lane",
            "state": self._state.value,
            "provider": self.config.provider if self.config else None,
            "session_id": self.active_session_id
        }
        
        # Add component metrics
        if self.audio_pipeline:
            metrics["audio_pipeline"] = self.audio_pipeline.get_metrics()
        
        if self.stream_orchestrator:
            metrics["orchestrator"] = self.stream_orchestrator.get_metrics()
        
        if self.event_bus:
            metrics["event_bus"] = self.event_bus.get_metrics()
        
        # Add queue metrics
        metrics["audio_queue_size"] = self._audio_input_queue.qsize()
        
        # Add session metrics
        if self.active_session_id:
            session = self.session_manager.get_session(self.active_session_id)
            if session:
                metrics["session"] = {
                    "duration": time.time() - session.created_at,
                    "audio_seconds": session.audio_seconds_used,
                    "text_tokens": session.text_tokens_used
                }
        
        return metrics
    
    def get_usage(self) -> Usage:
        """Get usage statistics"""
        return self._usage
    
    async def estimate_cost(self) -> Cost:
        """Estimate cost across all providers"""
        total_cost = Cost()
        
        # Get current session usage
        if self.provider_session:
            current_usage = self.provider_session.get_usage()
            current_cost = self.provider.estimate_cost(current_usage)
            
            total_cost.audio_cost += current_cost.audio_cost
            total_cost.text_cost += current_cost.text_cost
            total_cost.function_cost += current_cost.function_cost
        
        # Add historical costs from other providers
        for provider_name, usage in self._provider_usage.items():
            provider = self.provider_registry.get(provider_name)
            cost = provider.estimate_cost(usage)
            
            total_cost.audio_cost += cost.audio_cost
            total_cost.text_cost += cost.text_cost
            total_cost.function_cost += cost.function_cost
        
        return total_cost
    
    # ============== Private Methods ==============
    
    def _create_audio_processors(self, config: EngineConfig) -> List[AudioProcessor]:
        """Create audio processors based on config"""
        processors = []
        
        # Always include validator
        processors.append(AudioValidator())
        
        # Quality enhancement
        if config.latency_mode == "quality":
            processors.append(NoiseReducer())
            processors.append(VolumeNormalizer())
        
        # Echo cancellation
        if config.metadata.get("echo_cancellation"):
            processors.append(EchoCanceller())
        
        # VAD processor
        if config.enable_vad:
            processors.append(VADProcessor(
                threshold=config.metadata.get("vad_threshold", 0.5)
            ))
        
        return processors
    
    async def _audio_processor_loop(self):
        """Process audio through pipeline"""
        while self._state == StreamState.ACTIVE:
            try:
                # Get audio from queue
                audio_data = await self._audio_input_queue.get()
                
                # Process through pipeline
                processed = await self.audio_pipeline.process(audio_data)
                
                # Send to provider if not filtered
                if processed:
                    await self.provider_session.send_audio(processed)
                    
                    # Update usage
                    duration_seconds = len(processed) / 48000
                    self._usage.audio_input_seconds += duration_seconds
                    
                    # Update session
                    self.session_manager.track_usage(
                        self.active_session_id,
                        audio_seconds=duration_seconds
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
                
                # Emit error event
                await self.event_bus.emit(Event(
                    type="audio.processing.error",
                    data={"error": str(e)}
                ))
    
    async def _response_handler_loop(self):
        """Handle responses from provider"""
        try:
            async for event in self.provider_session.get_event_stream():
                # Process provider event
                await self._handle_provider_event(event)
                
        except Exception as e:
            self.logger.error(f"Response handler error: {e}")
            
            # Emit error event
            await self.event_bus.emit(Event(
                type="response.handler.error",
                data={"error": str(e)}
            ))
    
    async def _handle_provider_event(self, event: StreamEvent):
        """Handle event from provider"""
        # Update usage based on event
        if event.type == StreamEventType.AUDIO_OUTPUT_CHUNK:
            audio_data = event.data.get("audio", b"")
            duration = len(audio_data) / 48000
            self._usage.audio_output_seconds += duration
            
        elif event.type == StreamEventType.TEXT_OUTPUT_CHUNK:
            text = event.data.get("text", "")
            self._usage.text_output_tokens += self._estimate_tokens(text)
        
        # Aggregate response
        await self.response_aggregator.add_event(event)
        
        # Emit to event bus
        await self.event_bus.emit(Event(
            type=f"provider.{event.type}",
            data=event.data,
            metadata={"stream_id": event.stream_id}
        ))
    
    def _setup_event_subscriptions(self):
        """Setup internal event subscriptions"""
        # Subscribe to cost events
        self.event_bus.subscribe("cost.*", self._handle_cost_event)
        
        # Subscribe to error events
        self.event_bus.subscribe("*.error", self._handle_error_event)
        
        # Subscribe to audio events for metrics
        self.event_bus.subscribe("audio.*", self._update_audio_metrics)
    
    async def _handle_cost_event(self, event: Event):
        """Handle cost-related events"""
        if event.type == "cost.threshold.warning":
            self.logger.warning(f"Cost threshold warning: {event.data}")
    
    async def _handle_error_event(self, event: Event):
        """Handle error events"""
        self.logger.error(f"Error event: {event.type} - {event.data}")
    
    async def _update_audio_metrics(self, event: Event):
        """Update audio metrics from events"""
        # Track audio flow through the system
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _map_event_type(self, internal_type: str) -> StreamEventType:
        """Map internal event types to stream event types"""
        mapping = {
            "audio.output.chunk": StreamEventType.AUDIO_OUTPUT_CHUNK,
            "text.output.chunk": StreamEventType.TEXT_OUTPUT_CHUNK,
            "function.call.started": StreamEventType.FUNCTION_CALL_STARTED,
            # Add more mappings
        }
        return mapping.get(internal_type, StreamEventType.STREAM_READY)
    
    def _map_to_internal_event(self, stream_type: StreamEventType) -> str:
        """Map stream event types to internal events"""
        mapping = {
            StreamEventType.AUDIO_OUTPUT_CHUNK: "audio.output.chunk",
            StreamEventType.TEXT_OUTPUT_CHUNK: "text.output.chunk",
            StreamEventType.FUNCTION_CALL_STARTED: "function.call.started",
            # Add more mappings
        }
        return mapping.get(stream_type, "unknown")