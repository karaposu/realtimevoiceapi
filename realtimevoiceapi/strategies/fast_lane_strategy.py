# here is realtimevoiceapi/strategies/fast_lane_strategy.py

"""
Fast Lane Strategy Implementation

Minimal overhead implementation for ultra-low latency scenarios.
Uses direct connections and callbacks without abstraction layers.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, AsyncIterator, Callable
import logging

from .base_strategy import BaseStrategy, EngineConfig
from ..core.stream_protocol import StreamEvent, StreamEventType, StreamState
from audioengine.audioengine.audio_types import AudioBytes, AudioConfig, VADConfig, VADType
from ..core.provider_protocol import Usage, Cost
# TODO: These modules need to be created or replaced with AudioEngine equivalents
# from ..fast_lane.direct_audio_capture import DirectAudioCapture
# from ..fast_lane.fast_vad_detector import FastVADDetector,VADState

# Temporary placeholders
class DirectAudioCapture:
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def get_metrics(self):
        return {}
    async def start_async_capture(self):
        # Return an empty queue for the placeholder
        return asyncio.Queue()

class FastVADDetector:
    def __init__(self, *args, **kwargs):
        pass
    def process_chunk(self, *args):
        return None
    def get_metrics(self):
        return {}

class VADState:
    SPEECH_STARTING = "speech_starting"
    SPEECH = "speech"
    SILENCE = "silence"
from ..fast_lane.fast_stream_manager import FastStreamManager, FastStreamConfig
from ..core.exceptions import EngineError


class FastLaneStrategy(BaseStrategy):
    """
    Fast lane implementation with minimal overhead.
    
    Optimized for:
    - Client-side VAD only
    - Single provider (OpenAI)
    - Direct audio path
    - Minimal latency


    Fast Lane Strategy:

Direct connections - No abstraction layers
Callbacks instead of events - Minimal overhead
No queuing - Direct audio path
Single provider - OpenAI only
Minimal features - Just voice I/O


    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config: Optional[EngineConfig] = None
        
        # Core components
        self.audio_capture: Optional[DirectAudioCapture] = None
        self.vad_detector: Optional[FastVADDetector] = None
        self.stream_manager: Optional[FastStreamManager] = None

        self._audio_queue: Optional[asyncio.Queue] = None  # ADD THIS
        self._audio_task: Optional[asyncio.Task] = None    # ADD THIS
        self._error_callback: Optional[Callable] = None    # ADD THIS


        
        # Direct callbacks
        self._event_handlers: Dict[StreamEventType, Callable] = {}
        
        # State
        self._is_initialized = False
        self._is_connected = False
        self._is_capturing = False
        
        # Metrics
        self._start_time = 0
        self._audio_chunks_processed = 0
        
        # Usage tracking (minimal)
        self._usage = Usage()
    
    async def initialize(self, config: EngineConfig) -> None:
        """Initialize fast lane components"""
        if self._is_initialized:
            raise EngineError("Already initialized")
        
        self.config = config
        
        # Validate fast lane requirements
        if config.provider != "openai":
            raise EngineError(f"Fast lane only supports OpenAI, got {config.provider}")
        
        if config.enable_transcription:
            raise EngineError("Fast lane does not support transcription")
        
        if config.enable_functions:
            raise EngineError("Fast lane does not support function calling")
        
        # Create audio config
        audio_config = AudioConfig(
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            chunk_duration_ms=100 if config.latency_mode == "ultra_low" else 200
        )
        
        # Initialize VAD
        if config.enable_vad:
            vad_config = VADConfig(
                type=VADType.ENERGY_BASED,
                energy_threshold=0.02,
                speech_start_ms=100,
                speech_end_ms=500
            )
            
            self.vad_detector = FastVADDetector(
                config=vad_config,
                audio_config=audio_config
            )
        
        # Initialize audio capture
        self.audio_capture = DirectAudioCapture(
            device=config.input_device,
            config=audio_config,
           # callback=self._audio_callback if config.enable_vad else None
        )
        
        # Initialize stream manager
        stream_config = FastStreamConfig(
            websocket_url="wss://api.openai.com/v1/realtime",
            api_key=config.api_key,
            voice=config.metadata.get("voice", "alloy"),
            send_immediately=True
        )
        
        self.stream_manager = FastStreamManager(
            config=stream_config,
            logger=self.logger
        )
        
        # Wire up callbacks
        self._setup_callbacks()
        
        self._is_initialized = True
        self.logger.info("Fast lane strategy initialized")
    
    async def connect(self) -> None:
        """Connect to OpenAI Realtime API"""
        if not self._is_initialized:
            raise EngineError("Not initialized")
        
        if self._is_connected:
            return
        
        self._start_time = time.time()
        
        # Start stream manager
        await self.stream_manager.start()
        
        self._is_connected = True
        self.logger.info("Fast lane connected")
    
    async def disconnect(self) -> None:
        """Disconnect from API"""
        if not self._is_connected:
            return
        
        # Stop audio capture first
        if self._is_capturing:
            await self.stop_audio_input()
        
        # Stop stream
        if self.stream_manager:
            await self.stream_manager.stop()
        
        self._is_connected = False
        self.logger.info("Fast lane disconnected")
    
    async def start_audio_input(self) -> None:
        """Start audio input capture"""
        if not self.audio_capture:
            # Create audio capture
            audio_config = AudioConfig(
                sample_rate=24000,
                channels=1,
                chunk_duration_ms=100
            )
            
            self.audio_capture = DirectAudioCapture(
                device=self.config.input_device,
                config=audio_config,
                logger=self.logger
            )
        
        # Start capture and get the queue
        # self.audio_queue = await self.audio_capture.start_async_capture()
        self._audio_queue = await self.audio_capture.start_async_capture()  # Use self._audio_queue

        self._audio_task = asyncio.create_task(self._process_audio_queue())
        
      
    
    async def _process_audio_queue(self):
        """Process audio queue in background"""
        try:
            # while self.state == StreamState.STREAMING:  # Changed from self.state
            while self._is_connected:  # Use the internal connected flag

                try:
                    # Get audio chunk from queue
                    audio_chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.1
                    )
                    
                    # Send to stream
                    await self.stream_manager.send_audio(audio_chunk)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Audio queue processing error: {e}")
                    if self._error_callback:
                        self._error_callback(e)
        except asyncio.CancelledError:
            self.logger.debug("Audio queue processing cancelled")


    # async def _process_audio_queue(self):
    #     """Process audio from the capture queue"""
    #     while self.state == StreamState.STREAMING:
    #         try:
    #             # Get audio chunk from queue
    #             audio_chunk = await self.audio_queue.get()
                
    #             # Process through VAD if enabled
    #             if self.vad_detector and self.config.enable_vad:
    #                 vad_state = self.vad_detector.process_chunk(audio_chunk)
                    
    #                 # Only send during speech
    #                 if vad_state.value in ["speech_starting", "speech"]:
    #                     await self._send_audio_chunk(audio_chunk)
    #             else:
    #                 # No VAD, send everything
    #                 await self._send_audio_chunk(audio_chunk)
                    
    #         except asyncio.CancelledError:
    #             break
    #         except Exception as e:
    #             self.logger.error(f"Audio processing error: {e}")

    async def _send_audio_chunk(self, audio_chunk: AudioBytes):
        """Send audio chunk to the API"""
        if self.websocket and self.state == StreamState.STREAMING:
            # Convert to base64
            audio_b64 = self.audio_processor.bytes_to_base64(audio_chunk)
            
            # Create message
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            
            # Send to API
            await self.websocket.send_json(message)
    
    async def stop_audio_input(self) -> None:
        """Stop audio capture"""
        if not self._is_capturing:
            return
        
        self.audio_capture.stop_capture()
        self._is_capturing = False

        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None
        
        self.logger.info("Audio capture stopped")
        
       
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio directly to stream"""
        if not self._is_connected:
            raise EngineError("Not connected")
        
        # If VAD is enabled, process through VAD first
        if self.vad_detector:
            vad_state = self.vad_detector.process_chunk(audio_data)
            
            # Only send during speech
            if vad_state in [VADState.SPEECH_STARTING, VADState.SPEECH]:
                await self.stream_manager.send_audio(audio_data)
                self._usage.audio_input_seconds += len(audio_data) / 48000  # 24kHz stereo
        else:
            # No VAD, send everything
            await self.stream_manager.send_audio(audio_data)
            self._usage.audio_input_seconds += len(audio_data) / 48000
        
        self._audio_chunks_processed += 1
    
    async def send_text(self, text: str) -> None:
        """Send text message"""
        if not self._is_connected:
            raise EngineError("Not connected")
        
        await self.stream_manager.send_text(text)
        
        # Rough token estimate
        self._usage.text_input_tokens += len(text.split()) * 1.3
    
    async def send_recorded_audio(self, audio_data: AudioBytes, 
                                auto_respond: bool = True) -> None:
        """
        Send complete audio recording and trigger response.
        
        For FastLaneStrategy, we need to:
        1. Send the audio data via input_audio_buffer.append
        2. Commit the buffer via input_audio_buffer.commit
        3. Trigger response via response.create
        """
        if not self._is_connected:
            raise EngineError("Not connected")
            
        # Send the audio data
        await self.stream_manager.send_audio(audio_data)
        
        if auto_respond:
            # Commit the audio buffer and trigger response
            await self.stream_manager.commit_audio_and_respond()
    
    async def trigger_response(self) -> None:
        """Explicitly trigger AI response generation"""
        if not self._is_connected:
            raise EngineError("Not connected")
            
        if self.stream_manager and self.stream_manager.connection:
            from ..messaging.message_factory import MessageFactory
            await self.stream_manager.connection.send(
                MessageFactory.response_create()
            )
    
    async def get_response_stream(self) -> AsyncIterator[StreamEvent]:
        """Get response event stream"""
        # Fast lane doesn't use async iterators - uses callbacks
        # This is here for interface compatibility
        while self._is_connected:
            await asyncio.sleep(0.1)
            yield  # This won't actually be used
    
    def set_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Set direct event handler"""
        self._event_handlers[event_type] = handler
        
        # Map to stream manager callbacks
        if event_type == StreamEventType.AUDIO_OUTPUT_CHUNK:
            self.stream_manager.set_audio_callback(
                lambda audio: handler(
                    StreamEvent(
                        type=event_type,
                        stream_id=self.stream_manager.stream_id,
                        timestamp=time.time(),
                        data={"audio": audio}
                    )
                )
            )
        elif event_type == StreamEventType.TEXT_OUTPUT_CHUNK:
            self.stream_manager.set_text_callback(
                lambda text: handler(
                    StreamEvent(
                        type=event_type,
                        stream_id=self.stream_manager.stream_id,
                        timestamp=time.time(),
                        data={"text": text}
                    )
                )
            )
        elif event_type == StreamEventType.STREAM_ENDED:
            # Set response done callback
            if hasattr(self.stream_manager, 'set_response_done_callback'):
                self.stream_manager.set_response_done_callback(
                    lambda: handler(
                        StreamEvent(
                            type=event_type,
                            stream_id=self.stream_manager.stream_id,
                            timestamp=time.time(),
                            data={}
                        )
                    )
                )
        elif event_type == StreamEventType.STREAM_ERROR:
            # Set error callback
            self._error_callback = handler
    
    async def interrupt(self) -> None:
        """Interrupt current response"""
        if self.stream_manager:
            # Send cancel message
            await self.stream_manager.connection.send({
                "type": "response.cancel"
            })
    
    def get_state(self) -> StreamState:
        """Get current state"""
        if not self._is_initialized:
            return StreamState.IDLE
        
        if self.stream_manager:
            return self.stream_manager.state
        
        return StreamState.IDLE
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            "strategy": "fast_lane",
            "audio_chunks_processed": self._audio_chunks_processed,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0
        }
        
        # Add component metrics
        if self.audio_capture:
            metrics["audio_capture"] = self.audio_capture.get_metrics()
        
        if self.vad_detector:
            metrics["vad"] = self.vad_detector.get_metrics()
        
        if self.stream_manager:
            metrics["stream"] = self.stream_manager.get_metrics()
        
        return metrics
    
    def get_usage(self) -> Usage:
        """Get usage statistics"""
        return self._usage
    
    async def estimate_cost(self) -> Cost:
        """Estimate cost (simplified for fast lane)"""
        # OpenAI Realtime API pricing (as of 2024)
        # These are example rates - should be configurable
        cost = Cost()
        
        # Audio: $0.06 per minute input, $0.24 per minute output
        cost.audio_cost = (
            (self._usage.audio_input_seconds / 60) * 0.06 +
            (self._usage.audio_output_seconds / 60) * 0.24
        )
        
        # Text: ~$0.015 per 1K input tokens, $0.06 per 1K output tokens
        cost.text_cost = (
            (self._usage.text_input_tokens / 1000) * 0.015 +
            (self._usage.text_output_tokens / 1000) * 0.06
        )
        
        return cost
    
    # ============== Private Methods ==============
    
    def _audio_callback(self, audio_chunk: AudioBytes):
        """
        Direct audio callback from capture.
        
        Runs in audio thread - must be FAST!
        """
        if not self.vad_detector or not self.stream_manager:
            return
        
        # Process through VAD
        vad_state = self.vad_detector.process_chunk(audio_chunk)
        
        # Send if speaking
        if vad_state in [VADState.SPEECH_STARTING, VADState.SPEECH]:
            # Use asyncio to send from audio thread
            asyncio.create_task(
                self.stream_manager.send_audio(audio_chunk)
            )
    
    def _setup_callbacks(self):
        """Wire up VAD callbacks to stream manager"""
        if self.vad_detector:
            # When speech starts, clear any pending audio
            self.vad_detector.on_speech_start = lambda: asyncio.create_task(
                self.stream_manager.connection.send({
                    "type": "input_audio_buffer.clear"
                })
            )
            
            # When speech ends, commit buffer
            self.vad_detector.on_speech_end = lambda: asyncio.create_task(
                self.stream_manager.connection.send({
                    "type": "input_audio_buffer.commit"
                })
            )

    