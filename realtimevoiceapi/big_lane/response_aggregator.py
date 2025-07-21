# here is realtimevoiceapi/big_lane/response_aggregator.py

"""
Response Aggregator - Big Lane Component

Aggregates streaming response chunks into complete responses.
Handles partial data assembly for text, audio, and function calls.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from ..core.stream_protocol import StreamEvent, StreamEventType, Response
from audioengine.audioengine.audio_types import AudioBytes
from ..core.exceptions import StreamError


class ResponseState(Enum):
    """State of response aggregation"""
    IDLE = "idle"
    RECEIVING = "receiving"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class PartialResponse:
    """Tracks partial response data"""
    id: str
    started_at: float = field(default_factory=time.time)
    
    # Accumulated data
    text_chunks: List[str] = field(default_factory=list)
    audio_chunks: List[AudioBytes] = field(default_factory=list)
    function_call_chunks: List[str] = field(default_factory=list)
    
    # Metadata
    item_id: Optional[str] = None
    call_id: Optional[str] = None
    
    # State
    state: ResponseState = ResponseState.RECEIVING
    error: Optional[str] = None
    
    # Metrics
    chunks_received: int = 0
    bytes_received: int = 0
    
    def add_text(self, text: str):
        """Add text chunk"""
        self.text_chunks.append(text)
        self.chunks_received += 1
        self.bytes_received += len(text.encode('utf-8'))
    
    def add_audio(self, audio: AudioBytes):
        """Add audio chunk"""
        self.audio_chunks.append(audio)
        self.chunks_received += 1
        self.bytes_received += len(audio)
    
    def add_function_args(self, args: str):
        """Add function call arguments chunk"""
        self.function_call_chunks.append(args)
        self.chunks_received += 1
        self.bytes_received += len(args.encode('utf-8'))
    
    def get_complete_text(self) -> Optional[str]:
        """Get complete text if any"""
        if self.text_chunks:
            return ''.join(self.text_chunks)
        return None
    
    def get_complete_audio(self) -> Optional[AudioBytes]:
        """Get complete audio if any"""
        if self.audio_chunks:
            return b''.join(self.audio_chunks)
        return None
    
    def get_complete_function_args(self) -> Optional[str]:
        """Get complete function arguments if any"""
        if self.function_call_chunks:
            return ''.join(self.function_call_chunks)
        return None


class ResponseAggregator:
    """
    Aggregates streaming responses into complete responses.
    
    Features:
    - Handles multiple concurrent responses
    - Assembles partial chunks
    - Tracks response metrics
    - Supports cancellation
    """
    
    def __init__(
        self,
        event_bus: Optional['EventBus'] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.event_bus = event_bus
        self.logger = logger or logging.getLogger(__name__)
        
        # Active responses
        self.responses: Dict[str, PartialResponse] = {}
        
        # Callbacks
        self.on_response_complete: Optional[Callable[[Response], None]] = None
        self.on_response_error: Optional[Callable[[str, str], None]] = None
        
        # Metrics
        self.total_responses = 0
        self.completed_responses = 0
        self.failed_responses = 0
        self.total_text_bytes = 0
        self.total_audio_bytes = 0
        
        # Configuration
        self.max_response_duration_ms = 60000  # 1 minute max
        self.cleanup_delay_ms = 5000  # Keep completed responses for 5s
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the aggregator"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Response aggregator started")
    
    async def stop(self):
        """Stop the aggregator"""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Response aggregator stopped")
    
    async def start_response(self, response_id: str, item_id: Optional[str] = None):
        """Start aggregating a new response"""
        if response_id in self.responses:
            self.logger.warning(f"Response {response_id} already exists")
            return
        
        response = PartialResponse(
            id=response_id,
            item_id=item_id
        )
        
        self.responses[response_id] = response
        self.total_responses += 1
        
        self.logger.debug(f"Started aggregating response {response_id}")
        
        # Emit event if bus available
        if self.event_bus:
            await self.event_bus.emit("aggregator.response.started", {
                "response_id": response_id,
                "item_id": item_id
            })
    
    async def add_event(self, event: StreamEvent):
        """Add a stream event to aggregation"""
        # Extract response ID from event
        response_id = event.data.get("response_id") or event.data.get("id")
        if not response_id:
            self.logger.warning(f"Event missing response_id: {event.type}")
            return
        
        # Create response if needed
        if response_id not in self.responses:
            await self.start_response(response_id, event.data.get("item_id"))
        
        response = self.responses[response_id]
        
        # Handle different event types
        if event.type == StreamEventType.TEXT_OUTPUT_CHUNK:
            text = event.data.get("delta", "")
            response.add_text(text)
            
        elif event.type == StreamEventType.AUDIO_OUTPUT_CHUNK:
            audio = event.data.get("audio", b"")
            response.add_audio(audio)
            
        elif event.type == StreamEventType.FUNCTION_CALL_STARTED:
            response.call_id = event.data.get("call_id")
            
        elif event.type == StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA:
            args = event.data.get("delta", "")
            response.add_function_args(args)
            
        elif event.type in [
            StreamEventType.RESPONSE_COMPLETED,
            StreamEventType.TEXT_OUTPUT_ENDED,
            StreamEventType.AUDIO_OUTPUT_ENDED,
            StreamEventType.FUNCTION_CALL_COMPLETED
        ]:
            await self._complete_response(response_id)
            
        elif event.type == StreamEventType.RESPONSE_CANCELLED:
            await self._cancel_response(response_id)
            
        elif event.type == StreamEventType.STREAM_ERROR:
            error = event.data.get("error", "Unknown error")
            await self._error_response(response_id, error)
    
    async def add_audio_chunk(self, response_id: str, chunk: AudioBytes):
        """Add audio chunk to response"""
        if response_id not in self.responses:
            await self.start_response(response_id)
        
        response = self.responses[response_id]
        response.add_audio(chunk)
        self.total_audio_bytes += len(chunk)
    
    async def add_text_chunk(self, response_id: str, text: str):
        """Add text chunk to response"""
        if response_id not in self.responses:
            await self.start_response(response_id)
        
        response = self.responses[response_id]
        response.add_text(text)
        self.total_text_bytes += len(text.encode('utf-8'))
    
    async def finalize_response(self, response_id: str) -> Optional[Response]:
        """Finalize and return complete response"""
        if response_id not in self.responses:
            self.logger.warning(f"Response {response_id} not found")
            return None
        
        partial = self.responses[response_id]
        
        # Create final response
        response = Response(
            id=response_id,
            text=partial.get_complete_text(),
            audio=partial.get_complete_audio(),
            duration_ms=(time.time() - partial.started_at) * 1000,
            tokens_used=len(partial.text_chunks) * 5,  # Rough estimate
            metadata={
                "chunks_received": partial.chunks_received,
                "bytes_received": partial.bytes_received,
                "item_id": partial.item_id
            }
        )
        
        # Add function call if present
        if partial.call_id and partial.function_call_chunks:
            response.function_calls.append({
                "id": partial.call_id,
                "arguments": partial.get_complete_function_args()
            })
        
        partial.state = ResponseState.COMPLETE
        self.completed_responses += 1
        
        # Trigger callback
        if self.on_response_complete:
            self.on_response_complete(response)
        
        # Emit event
        if self.event_bus:
            await self.event_bus.emit("aggregator.response.completed", {
                "response_id": response_id,
                "has_text": response.text is not None,
                "has_audio": response.audio is not None,
                "duration_ms": response.duration_ms
            })
        
        return response
    
    async def _complete_response(self, response_id: str):
        """Mark response as complete"""
        await self.finalize_response(response_id)
    
    async def _cancel_response(self, response_id: str):
        """Cancel response aggregation"""
        if response_id not in self.responses:
            return
        
        response = self.responses[response_id]
        response.state = ResponseState.CANCELLED
        
        self.logger.info(f"Response {response_id} cancelled")
        
        if self.event_bus:
            await self.event_bus.emit("aggregator.response.cancelled", {
                "response_id": response_id
            })
    
    async def _error_response(self, response_id: str, error: str):
        """Mark response as errored"""
        if response_id not in self.responses:
            return
        
        response = self.responses[response_id]
        response.state = ResponseState.ERROR
        response.error = error
        self.failed_responses += 1
        
        self.logger.error(f"Response {response_id} error: {error}")
        
        if self.on_response_error:
            self.on_response_error(response_id, error)
        
        if self.event_bus:
            await self.event_bus.emit("aggregator.response.error", {
                "response_id": response_id,
                "error": error
            })
    
    def clear_current(self):
        """Clear all current responses"""
        active_count = sum(
            1 for r in self.responses.values()
            if r.state == ResponseState.RECEIVING
        )
        
        for response in self.responses.values():
            if response.state == ResponseState.RECEIVING:
                response.state = ResponseState.CANCELLED
        
        self.logger.info(f"Cleared {active_count} active responses")
    
    async def _cleanup_loop(self):
        """Periodically clean up old responses"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                now = time.time()
                to_remove = []
                
                for response_id, response in self.responses.items():
                    # Remove completed/cancelled/errored responses after delay
                    if response.state in [
                        ResponseState.COMPLETE,
                        ResponseState.CANCELLED,
                        ResponseState.ERROR
                    ]:
                        age_ms = (now - response.started_at) * 1000
                        if age_ms > self.cleanup_delay_ms:
                            to_remove.append(response_id)
                    
                    # Timeout active responses
                    elif response.state == ResponseState.RECEIVING:
                        age_ms = (now - response.started_at) * 1000
                        if age_ms > self.max_response_duration_ms:
                            await self._error_response(
                                response_id,
                                "Response timeout"
                            )
                
                # Remove old responses
                for response_id in to_remove:
                    del self.responses[response_id]
                
                if to_remove:
                    self.logger.debug(f"Cleaned up {len(to_remove)} old responses")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def get_active_responses(self) -> List[str]:
        """Get list of active response IDs"""
        return [
            r_id for r_id, r in self.responses.items()
            if r.state == ResponseState.RECEIVING
        ]
    
    def get_response_state(self, response_id: str) -> Optional[ResponseState]:
        """Get state of a response"""
        if response_id in self.responses:
            return self.responses[response_id].state
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregator metrics"""
        active_responses = sum(
            1 for r in self.responses.values()
            if r.state == ResponseState.RECEIVING
        )
        
        return {
            "total_responses": self.total_responses,
            "completed_responses": self.completed_responses,
            "failed_responses": self.failed_responses,
            "active_responses": active_responses,
            "cached_responses": len(self.responses),
            "total_text_bytes": self.total_text_bytes,
            "total_audio_bytes": self.total_audio_bytes,
            "completion_rate": (
                self.completed_responses / self.total_responses
                if self.total_responses > 0 else 0
            )
        }