"""
Fast Stream Manager - Fast Lane Implementation

Minimal overhead stream management for client-side VAD.
Direct integration with WebSocket and audio components.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

from .stream_protocol import (
    IStreamManager, StreamState, StreamEvent, StreamEventType,
    StreamMetrics, AudioFormat
)
from .websocket_connection import FastLaneConnection
from .message_protocol import MessageFactory, ServerMessageType
from .audio_types import AudioBytes, AudioConfig
from .exceptions import StreamError


@dataclass
class FastStreamConfig:
    """Minimal configuration for fast stream"""
    websocket_url: str
    api_key: str
    voice: str = "alloy"
    audio_format: AudioFormat = AudioFormat(
        sample_rate=24000,
        channels=1,
        bit_depth=16
    )
    
    # Performance settings
    send_immediately: bool = True  # No buffering
    event_callbacks: bool = True   # Direct callbacks vs queue


class FastStreamManager(IStreamManager):
    """
    Fast lane implementation of stream manager.
    
    Optimized for minimal latency with client-side VAD.
    No queues, no buffering, direct callbacks.
    """
    
    def __init__(
        self,
        config: FastStreamConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components (minimal)
        self.connection: Optional[FastLaneConnection] = None
        self._stream_id = f"fast_{int(time.time() * 1000)}"
        self._state = StreamState.IDLE
        
        # Direct callbacks (no event queue)
        self._audio_callback: Optional[Callable[[AudioBytes], None]] = None
        self._text_callback: Optional[Callable[[str], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        
        # Minimal metrics
        self._start_time = 0.0
        self._audio_bytes_sent = 0
        self._audio_bytes_received = 0
        
        # Pre-create session config message
        self._session_config = self._create_session_config()
    
    def _create_session_config(self) -> dict:
        """Pre-create session configuration"""
        return MessageFactory.session_update(
            modalities=["text", "audio"],
            voice=self.config.voice,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True
            }
        )
    
    # ============== IStreamManager Implementation ==============
    
    @property
    def stream_id(self) -> str:
        return self._stream_id
    
    @property
    def state(self) -> StreamState:
        return self._state
    
    async def start(self) -> None:
        """Start the stream with minimal setup"""
        if self._state != StreamState.IDLE:
            raise StreamError(f"Cannot start in state {self._state}")
        
        self._state = StreamState.STARTING
        self._start_time = time.time()
        
        try:
            # Create direct WebSocket connection
            self.connection = FastLaneConnection(
                url=f"{self.config.websocket_url}?model=gpt-4o-realtime-preview",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            
            # Set message handler
            self.connection.message_handler = self._handle_message
            
            # Connect
            await self.connection.connect()
            
            # Send session config immediately
            await self.connection.send(self._session_config)
            
            self._state = StreamState.ACTIVE
            
        except Exception as e:
            self._state = StreamState.ERROR
            if self._error_callback:
                self._error_callback(e)
            raise
    
    async def stop(self) -> None:
        """Stop the stream"""
        if self._state not in [StreamState.ACTIVE, StreamState.ERROR]:
            return
        
        self._state = StreamState.ENDING
        
        if self.connection:
            await self.connection.disconnect()
            self.connection = None
        
        self._state = StreamState.ENDED
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio with zero buffering"""
        if self._state != StreamState.ACTIVE:
            raise StreamError(f"Cannot send audio in state {self._state}")
        
        # Create message directly (no intermediate objects)
        message = MessageFactory.input_audio_buffer_append(
            self._encode_audio_fast(audio_data)
        )
        
        # Send immediately
        await self.connection.send(message)
        
        # Update metrics
        self._audio_bytes_sent += len(audio_data)
    
    async def send_text(self, text: str) -> None:
        """Send text message"""
        if self._state != StreamState.ACTIVE:
            raise StreamError(f"Cannot send text in state {self._state}")
        
        # Create and send message
        message = MessageFactory.conversation_item_create(
            item_type="message",
            role="user",
            content=[{"type": "input_text", "text": text}]
        )
        
        await self.connection.send(message)
        
        # Trigger response
        await self.connection.send(MessageFactory.response_create())
    
    def subscribe_events(
        self,
        event_types: list,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Fast lane uses direct callbacks instead"""
        # Map event types to specific callbacks
        if StreamEventType.AUDIO_OUTPUT_CHUNK in event_types:
            self._audio_callback = lambda data: handler(
                StreamEvent(
                    type=StreamEventType.AUDIO_OUTPUT_CHUNK,
                    stream_id=self._stream_id,
                    timestamp=time.time(),
                    data={"audio": data}
                )
            )
        
        if StreamEventType.TEXT_OUTPUT_CHUNK in event_types:
            self._text_callback = lambda text: handler(
                StreamEvent(
                    type=StreamEventType.TEXT_OUTPUT_CHUNK,
                    stream_id=self._stream_id,
                    timestamp=time.time(),
                    data={"text": text}
                )
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get minimal metrics"""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        return {
            "state": self._state.value,
            "uptime_seconds": uptime,
            "audio_bytes_sent": self._audio_bytes_sent,
            "audio_bytes_received": self._audio_bytes_received,
            "throughput_bps": (
                (self._audio_bytes_sent + self._audio_bytes_received) / uptime
                if uptime > 0 else 0
            )
        }
    
    # ============== Fast Direct Callbacks ==============
    
    def set_audio_callback(self, callback: Callable[[AudioBytes], None]):
        """Set direct audio callback (fastest)"""
        self._audio_callback = callback
    
    def set_text_callback(self, callback: Callable[[str], None]):
        """Set direct text callback"""
        self._text_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set error callback"""
        self._error_callback = callback
    
    # ============== Private Methods ==============
    
    def _handle_message(self, message: dict):
        """
        Handle incoming WebSocket message.
        
        Runs in WebSocket thread - must be fast!
        """
        msg_type = message.get("type", "")
        
        # Fast path for audio
        if msg_type == ServerMessageType.RESPONSE_AUDIO_DELTA.value:
            if self._audio_callback and "delta" in message:
                audio_b64 = message["delta"]
                # Decode inline for speed
                audio_bytes = self._decode_audio_fast(audio_b64)
                self._audio_bytes_received += len(audio_bytes)
                self._audio_callback(audio_bytes)
        
        # Text response
        elif msg_type == ServerMessageType.RESPONSE_TEXT_DELTA.value:
            if self._text_callback and "delta" in message:
                self._text_callback(message["delta"])
        
        # Error
        elif msg_type == ServerMessageType.ERROR.value:
            if self._error_callback:
                error = message.get("error", {})
                self._error_callback(
                    StreamError(error.get("message", "Unknown error"))
                )
    
    @staticmethod
    def _encode_audio_fast(audio_bytes: AudioBytes) -> str:
        """Fast base64 encoding"""
        import base64
        return base64.b64encode(audio_bytes).decode('ascii')
    
    @staticmethod  
    def _decode_audio_fast(audio_b64: str) -> AudioBytes:
        """Fast base64 decoding"""
        import base64
        return base64.b64decode(audio_b64.encode('ascii'))


class FastVADStreamManager(FastStreamManager):
    """
    Specialized fast stream manager with integrated VAD.
    
    Combines stream management with VAD for ultimate performance.
    """
    
    def __init__(
        self,
        config: FastStreamConfig,
        vad_detector: 'FastVADDetector',  # From fast_vad_detector.py
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        self.vad = vad_detector
        
        # VAD state
        self._is_speaking = False
        self._speech_start_time = 0
        
        # Wire VAD callbacks
        self.vad.on_speech_start = self._on_speech_start
        self.vad.on_speech_end = self._on_speech_end
    
    async def process_audio_with_vad(self, audio_chunk: AudioBytes) -> None:
        """
        Process audio through VAD before sending.
        
        Only sends when speech is detected.
        """
        # Process through VAD
        vad_state = self.vad.process_chunk(audio_chunk)
        
        # Only send if we're in speech
        if vad_state in [VADState.SPEECH_STARTING, VADState.SPEECH]:
            await self.send_audio(audio_chunk)
    
    def _on_speech_start(self):
        """Called by VAD when speech starts"""
        self._is_speaking = True
        self._speech_start_time = time.time()
        
        # Could send a marker event
        if self.connection and self._state == StreamState.ACTIVE:
            # Clear any pending audio
            asyncio.create_task(
                self.connection.send(
                    MessageFactory.input_audio_buffer_clear()
                )
            )
    
    def _on_speech_end(self):
        """Called by VAD when speech ends"""
        self._is_speaking = False
        
        # Commit audio buffer to trigger response
        if self.connection and self._state == StreamState.ACTIVE:
            asyncio.create_task(
                self.connection.send(
                    MessageFactory.input_audio_buffer_commit()
                )
            )