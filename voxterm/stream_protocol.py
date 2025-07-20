"""
VoxTerm Stream Protocol - Unified real-time streaming abstraction

This abstraction solves the core complexity of real-time voice interaction:
coordinating multiple async streams (audio in, audio out, text, control)
while maintaining low latency and preventing common issues.
"""

import asyncio
from typing import Optional, Callable, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque


class StreamType(Enum):
    """Types of streams in the protocol"""
    AUDIO_IN = "audio_in"
    AUDIO_OUT = "audio_out"
    TEXT_IN = "text_in"
    TEXT_OUT = "text_out"
    TRANSCRIPT = "transcript"
    CONTROL = "control"


class StreamEvent:
    """Base class for stream events"""
    def __init__(self, type: StreamType, data: any, timestamp: float = None):
        self.type = type
        self.data = data
        self.timestamp = timestamp or time.time()


@dataclass
class StreamConfig:
    """Configuration for stream behavior"""
    # Buffering
    audio_buffer_size: int = 10  # Number of chunks to buffer
    text_buffer_size: int = 100  # Number of text events to buffer
    
    # Timing
    audio_timeout: float = 0.1  # Max time to wait for audio chunk
    response_timeout: float = 30.0  # Max time to wait for complete response
    
    # Behavior
    allow_interruption: bool = True
    auto_flush_on_silence: bool = True
    silence_threshold_ms: int = 500


class StreamProtocol:
    """
    Unified protocol for managing real-time streams
    
    This abstraction:
    1. Manages multiple async streams without blocking
    2. Handles backpressure and buffering
    3. Coordinates stream lifecycle (start/stop/interrupt)
    4. Provides clean event-based interface
    5. Maintains real-time performance
    """
    
    def __init__(self, engine: any, config: Optional[StreamConfig] = None):
        self.engine = engine
        self.config = config or StreamConfig()
        
        # Stream queues
        self.queues = {
            StreamType.AUDIO_IN: asyncio.Queue(maxsize=self.config.audio_buffer_size),
            StreamType.AUDIO_OUT: asyncio.Queue(maxsize=self.config.audio_buffer_size),
            StreamType.TEXT_IN: asyncio.Queue(maxsize=self.config.text_buffer_size),
            StreamType.TEXT_OUT: asyncio.Queue(maxsize=self.config.text_buffer_size),
            StreamType.TRANSCRIPT: asyncio.Queue(maxsize=self.config.text_buffer_size),
            StreamType.CONTROL: asyncio.Queue(maxsize=10),
        }
        
        # Stream tasks
        self.tasks = {}
        self.running = False
        
        # Event handlers
        self.handlers = {
            StreamType.AUDIO_OUT: [],
            StreamType.TEXT_OUT: [],
            StreamType.TRANSCRIPT: [],
            StreamType.CONTROL: [],
        }
        
        # State
        self.current_interaction_id = None
        self.interaction_start_time = None
        
    async def start(self):
        """Start the stream protocol"""
        if self.running:
            return
            
        self.running = True
        
        # Connect engine callbacks to our queues
        self._setup_engine_callbacks()
        
        # Start stream processors
        self.tasks['audio_out'] = asyncio.create_task(self._process_audio_out())
        self.tasks['text_out'] = asyncio.create_task(self._process_text_out())
        self.tasks['control'] = asyncio.create_task(self._process_control())
        
    async def stop(self):
        """Stop the stream protocol"""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks.values():
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        # Clear queues
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass
    
    # Stream control methods
    async def start_interaction(self, interaction_type: str = "voice") -> str:
        """Start a new interaction (voice or text)"""
        # End any existing interaction first
        if self.current_interaction_id:
            await self.end_interaction()
            await asyncio.sleep(0.2)  # Brief pause
        
        self.current_interaction_id = f"{interaction_type}_{int(time.time() * 1000)}"
        self.interaction_start_time = time.time()
        
        if interaction_type == "voice":
            await self.engine.start_listening()
        
        # Emit control event
        await self._emit_event(StreamEvent(
            StreamType.CONTROL,
            {"action": "interaction_started", "id": self.current_interaction_id}
        ))
        
        return self.current_interaction_id
    
    async def end_interaction(self):
        """End the current interaction"""
        if not self.current_interaction_id:
            return
            
        # Stop listening if voice - IMPORTANT: Do this first!
        if self.current_interaction_id.startswith("voice"):
            try:
                await self.engine.stop_listening()
                # Give a moment for audio system to settle
                await asyncio.sleep(0.1)
            except Exception as e:
                # Log but don't fail - listening might already be stopped
                pass
        
        # Emit control event
        await self._emit_event(StreamEvent(
            StreamType.CONTROL,
            {"action": "interaction_ended", "id": self.current_interaction_id}
        ))
        
        self.current_interaction_id = None
    
    async def send_text(self, text: str):
        """Send text through the protocol"""
        await self.queues[StreamType.TEXT_IN].put(
            StreamEvent(StreamType.TEXT_IN, text)
        )
        
        # Process immediately
        await self.engine.send_text(text)
    
    async def interrupt(self):
        """Interrupt current interaction"""
        if not self.config.allow_interruption:
            return
            
        # Clear output queues
        for stream_type in [StreamType.AUDIO_OUT, StreamType.TEXT_OUT]:
            queue = self.queues[stream_type]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass
        
        # Interrupt engine
        if hasattr(self.engine, 'interrupt'):
            await self.engine.interrupt()
        
        # Emit control event
        await self._emit_event(StreamEvent(
            StreamType.CONTROL,
            {"action": "interrupted"}
        ))
    
    # Event handling
    def on(self, stream_type: StreamType, handler: Callable):
        """Register an event handler"""
        if stream_type in self.handlers:
            self.handlers[stream_type].append(handler)
    
    async def _emit_event(self, event: StreamEvent):
        """Emit an event to all handlers"""
        handlers = self.handlers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"Handler error: {e}")
    
    # Stream processors
    async def _process_audio_out(self):
        """Process outgoing audio stream"""
        while self.running:
            try:
                # Get audio with timeout
                event = await asyncio.wait_for(
                    self.queues[StreamType.AUDIO_OUT].get(),
                    timeout=self.config.audio_timeout
                )
                
                # Emit to handlers
                await self._emit_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    async def _process_text_out(self):
        """Process outgoing text stream"""
        buffer = []
        last_emit_time = time.time()
        
        while self.running:
            try:
                # Get text with short timeout for responsiveness
                event = await asyncio.wait_for(
                    self.queues[StreamType.TEXT_OUT].get(),
                    timeout=0.05
                )
                
                buffer.append(event.data)
                
                # Emit if buffer is getting full or enough time passed
                if len(buffer) > 10 or (time.time() - last_emit_time) > 0.1:
                    combined_text = "".join(buffer)
                    await self._emit_event(StreamEvent(
                        StreamType.TEXT_OUT,
                        combined_text
                    ))
                    buffer.clear()
                    last_emit_time = time.time()
                
            except asyncio.TimeoutError:
                # Flush buffer if we have data
                if buffer:
                    combined_text = "".join(buffer)
                    await self._emit_event(StreamEvent(
                        StreamType.TEXT_OUT,
                        combined_text
                    ))
                    buffer.clear()
                    last_emit_time = time.time()
            except asyncio.CancelledError:
                break
    
    async def _process_control(self):
        """Process control events"""
        while self.running:
            try:
                event = await self.queues[StreamType.CONTROL].get()
                await self._emit_event(event)
            except asyncio.CancelledError:
                break
    
    # Engine callback setup
    def _setup_engine_callbacks(self):
        """Connect engine callbacks to our queues"""
        original_callbacks = {}
        
        # Audio output
        if hasattr(self.engine, 'on_audio_response'):
            original_callbacks['audio'] = self.engine.on_audio_response
            self.engine.on_audio_response = lambda audio: self._queue_event(
                StreamType.AUDIO_OUT, audio
            )
        
        # Text output
        if hasattr(self.engine, 'on_text_response'):
            original_callbacks['text'] = self.engine.on_text_response
            self.engine.on_text_response = lambda text: self._queue_event(
                StreamType.TEXT_OUT, text
            )
        
        # Transcripts
        if hasattr(self.engine, 'on_transcript'):
            original_callbacks['transcript'] = self.engine.on_transcript
            self.engine.on_transcript = lambda text: self._queue_event(
                StreamType.TRANSCRIPT, text
            )
        
        # Response done
        if hasattr(self.engine, 'on_response_done'):
            original_callbacks['done'] = self.engine.on_response_done
            self.engine.on_response_done = lambda: self._queue_event(
                StreamType.CONTROL, {"action": "response_done"}
            )
        
        self._original_callbacks = original_callbacks
    
    def _queue_event(self, stream_type: StreamType, data: any):
        """Queue an event (called from sync callbacks)"""
        try:
            event = StreamEvent(stream_type, data)
            self.queues[stream_type].put_nowait(event)
        except asyncio.QueueFull:
            # Handle backpressure - drop oldest
            try:
                self.queues[stream_type].get_nowait()
                self.queues[stream_type].put_nowait(event)
            except:
                pass


# High-level interface for common patterns
class StreamSession:
    """High-level session using stream protocol"""
    
    def __init__(self, protocol: StreamProtocol):
        self.protocol = protocol
        self.transcript = []
        self.responses = []
        
        # Register handlers
        self.protocol.on(StreamType.TEXT_OUT, self._on_text)
        self.protocol.on(StreamType.TRANSCRIPT, self._on_transcript)
        self.protocol.on(StreamType.CONTROL, self._on_control)
    
    async def voice_turn(self) -> str:
        """Execute a voice turn and return the response"""
        # Start interaction
        interaction_id = await self.protocol.start_interaction("voice")
        
        # Wait for user to finish speaking (handled by mode)
        
        # End interaction
        await self.protocol.end_interaction()
        
        # Wait for response
        response = await self._wait_for_response()
        
        return response
    
    async def text_turn(self, message: str) -> str:
        """Execute a text turn and return the response"""
        # Start interaction
        interaction_id = await self.protocol.start_interaction("text")
        
        # Send text
        await self.protocol.send_text(message)
        
        # Wait for response
        response = await self._wait_for_response()
        
        # End interaction
        await self.protocol.end_interaction()
        
        return response
    
    async def _wait_for_response(self, timeout: float = 30.0) -> str:
        """Wait for a complete response"""
        response_parts = []
        done_event = asyncio.Event()
        
        def on_text(event):
            response_parts.append(event.data)
        
        def on_control(event):
            if event.data.get("action") == "response_done":
                done_event.set()
        
        # Temporary handlers
        self.protocol.on(StreamType.TEXT_OUT, on_text)
        self.protocol.on(StreamType.CONTROL, on_control)
        
        # Wait for response
        try:
            await asyncio.wait_for(done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        
        return "".join(response_parts)
    
    def _on_text(self, event: StreamEvent):
        """Handle text events"""
        self.responses.append(event.data)
    
    def _on_transcript(self, event: StreamEvent):
        """Handle transcript events"""
        self.transcript.append(event.data)
    
    def _on_control(self, event: StreamEvent):
        """Handle control events"""
        pass