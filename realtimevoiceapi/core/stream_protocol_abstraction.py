"""
High-Level Stream Abstraction - Built on top of existing stream_protocol.py

This provides a simpler, more Pythonic API while using the existing
protocol definitions underneath.
"""

import asyncio
from typing import AsyncIterator, Optional, Any, Dict, List, Union
from contextlib import asynccontextmanager
import time

# Import from existing stream protocol
from realtimevoiceapi.core.stream_protocol import (
    IStreamManager,
    StreamEvent,
    StreamEventType,
    StreamState,
    Response,
    StreamMetrics,
    StreamError,
    StreamErrorType
)


class StreamConversation:
    """
    High-level conversation API built on IStreamManager
    
    Provides a simple async iterator interface while using
    the existing stream protocol underneath.
    """
    
    def __init__(self, stream_manager: IStreamManager):
        self.stream = stream_manager
        self.queue = asyncio.Queue()
        self.active = True
        self._handlers_registered = False
        
        # Track conversation state
        self.responses: List[Response] = []
        self.current_response_chunks: List[str] = []
        self.current_audio_chunks: List[bytes] = []
        self.awaiting_response = False
        
    async def __aenter__(self):
        """Enter conversation context"""
        # Register event handlers
        self._register_handlers()
        
        # Start stream if not already started
        if self.stream.state == StreamState.IDLE:
            await self.stream.start()
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit conversation context"""
        self.active = False
        
        # Note: We don't stop the stream here as it might be reused
        # That's the responsibility of the stream manager owner
        
    def _register_handlers(self):
        """Register event handlers to populate our queue"""
        if self._handlers_registered:
            return
            
        # Text output events
        self.stream.subscribe_events(
            [StreamEventType.TEXT_OUTPUT_CHUNK],
            lambda event: asyncio.create_task(self._handle_text_chunk(event))
        )
        
        # Audio output events
        self.stream.subscribe_events(
            [StreamEventType.AUDIO_OUTPUT_CHUNK],
            lambda event: asyncio.create_task(self._handle_audio_chunk(event))
        )
        
        # Response lifecycle events
        self.stream.subscribe_events(
            [StreamEventType.RESPONSE_STARTED],
            lambda event: asyncio.create_task(self._handle_response_started(event))
        )
        
        self.stream.subscribe_events(
            [StreamEventType.RESPONSE_COMPLETED],
            lambda event: asyncio.create_task(self._handle_response_completed(event))
        )
        
        # Error events
        self.stream.subscribe_events(
            [StreamEventType.STREAM_ERROR],
            lambda event: asyncio.create_task(self._handle_error(event))
        )
        
        self._handlers_registered = True
        
    async def _handle_text_chunk(self, event: StreamEvent):
        """Handle incoming text chunks"""
        if event.data and 'text' in event.data:
            self.current_response_chunks.append(event.data['text'])
            await self.queue.put(('text', event.data['text']))
            
    async def _handle_audio_chunk(self, event: StreamEvent):
        """Handle incoming audio chunks"""
        if event.data and 'audio' in event.data:
            self.current_audio_chunks.append(event.data['audio'])
            await self.queue.put(('audio', event.data['audio']))
            
    async def _handle_response_started(self, event: StreamEvent):
        """Handle response start"""
        self.current_response_chunks = []
        self.current_audio_chunks = []
        await self.queue.put(('response_start', event.data))
        
    async def _handle_response_completed(self, event: StreamEvent):
        """Handle response completion"""
        # Create Response object
        response = Response(
            id=event.data.get('response_id', f'resp_{len(self.responses)}'),
            text=''.join(self.current_response_chunks) if self.current_response_chunks else None,
            audio=b''.join(self.current_audio_chunks) if self.current_audio_chunks else None,
            metadata=event.data
        )
        self.responses.append(response)
        self.awaiting_response = False
        await self.queue.put(('response_complete', response))
        
    async def _handle_error(self, event: StreamEvent):
        """Handle errors"""
        error = StreamError(
            type=StreamErrorType.PROVIDER_ERROR,
            message=event.data.get('message', 'Unknown error'),
            stream_id=self.stream.stream_id,
            details=event.data
        )
        await self.queue.put(('error', error))
        
    # High-level API methods
    async def send_text(self, text: str) -> None:
        """Send text and mark that we're awaiting response"""
        self.awaiting_response = True
        await self.stream.send_text(text)
        
    async def send_audio(self, audio: bytes) -> None:
        """Send audio data"""
        self.awaiting_response = True
        await self.stream.send_audio(audio)
        
    async def interrupt(self) -> None:
        """Interrupt current response if stream supports it"""
        # This would need to be implemented in the stream manager
        if hasattr(self.stream, 'interrupt'):
            await self.stream.interrupt()
            
    def __aiter__(self):
        """Make conversation an async iterator"""
        return self
        
    async def __anext__(self) -> tuple[str, Any]:
        """
        Get next event from conversation
        
        Returns: (event_type, data) tuple where:
            - event_type: 'text', 'audio', 'response_start', 'response_complete', 'error'
            - data: The associated data
        """
        if not self.active and self.queue.empty():
            raise StopAsyncIteration
            
        try:
            return await self.queue.get()
        except asyncio.CancelledError:
            raise StopAsyncIteration
            
    # Convenience methods
    async def get_response(self) -> Response:
        """Send a message and wait for complete response"""
        start_responses = len(self.responses)
        
        # Wait for response
        async for event_type, data in self:
            if event_type == 'response_complete':
                return data
            elif event_type == 'error':
                raise Exception(f"Stream error: {data.message}")
                
        # If we get here, stream ended without response
        raise Exception("Stream ended without response")
        
    async def get_text_response(self, message: str) -> str:
        """Send text and get text response"""
        await self.send_text(message)
        response = await self.get_response()
        return response.text or ""
        
    async def stream_text_response(self, message: str) -> AsyncIterator[str]:
        """Send text and stream response chunks"""
        await self.send_text(message)
        
        async for event_type, data in self:
            if event_type == 'text':
                yield data
            elif event_type == 'response_complete':
                break
            elif event_type == 'error':
                raise Exception(f"Stream error: {data.message}")


class StreamBuilder:
    """
    Fluent interface for building stream interactions
    
    Makes it easy to create common streaming patterns.
    """
    
    def __init__(self, stream_manager: IStreamManager):
        self.stream = stream_manager
        
    @asynccontextmanager
    async def conversation(self):
        """Create a conversation context"""
        async with StreamConversation(self.stream) as conv:
            yield conv
            
    async def chat(self, message: str) -> str:
        """Simple chat interaction"""
        async with self.conversation() as conv:
            return await conv.get_text_response(message)
            
    async def stream_chat(self, message: str) -> AsyncIterator[str]:
        """Streaming chat interaction"""
        async with self.conversation() as conv:
            async for chunk in conv.stream_text_response(message):
                yield chunk
                
    def with_handlers(self, **handlers) -> 'StreamBuilder':
        """
        Add event handlers fluently
        
        Example:
            builder.with_handlers(
                on_text=lambda text: print(text),
                on_error=lambda err: print(f"Error: {err}")
            )
        """
        event_map = {
            'on_text': StreamEventType.TEXT_OUTPUT_CHUNK,
            'on_audio': StreamEventType.AUDIO_OUTPUT_CHUNK,
            'on_error': StreamEventType.STREAM_ERROR,
            'on_start': StreamEventType.RESPONSE_STARTED,
            'on_complete': StreamEventType.RESPONSE_COMPLETED
        }
        
        for handler_name, handler_func in handlers.items():
            if handler_name in event_map:
                self.stream.subscribe_events(
                    [event_map[handler_name]],
                    handler_func
                )
                
        return self


# Convenience functions
def stream_api(stream_manager: IStreamManager) -> StreamBuilder:
    """Create a stream API builder"""
    return StreamBuilder(stream_manager)


# Example usage patterns
async def example_usage():
    """Show how to use the high-level API with existing stream protocol"""
    
    # Assume we have a stream manager that implements IStreamManager
    from realtimevoiceapi.fast_lane.fast_stream_manager import FastStreamManager
    
    stream = FastStreamManager(config)
    await stream.start()
    
    # Simple chat
    api = stream_api(stream)
    response = await api.chat("Hello, how are you?")
    print(response)
    
    # Streaming chat
    async for chunk in api.stream_chat("Tell me a story"):
        print(chunk, end="", flush=True)
    
    # Full conversation control
    async with api.conversation() as conv:
        # Send message
        await conv.send_text("What's the weather like?")
        
        # Process events as they come
        async for event_type, data in conv:
            if event_type == 'text':
                print(f"Chunk: {data}")
            elif event_type == 'response_complete':
                print(f"Complete response: {data.text}")
                break
                
    # With event handlers
    await (api
        .with_handlers(
            on_text=lambda e: print(f"Text: {e.data['text']}"),
            on_error=lambda e: print(f"Error: {e.data}")
        )
        .chat("Hello world")
    )
    
    # Get metrics
    metrics = stream.get_metrics()
    print(f"Sent: {metrics['bytes_sent']} bytes")
    
    await stream.stop()