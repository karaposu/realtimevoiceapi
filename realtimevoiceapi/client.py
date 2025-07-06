# realtimevoiceapi/client.py
"""
Main RealtimeClient for OpenAI Realtime API

This module provides the primary interface for interacting with OpenAI's Realtime API,
including session management, conversation handling, audio processing, and event management.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Callable, Dict, Any, List, Union, Tuple
from pathlib import Path

from .connection import RealtimeConnection
from .session import SessionConfig
from .events import EventDispatcher, RealtimeEvent, EventType
from .audio import AudioProcessor, AudioFormat, AudioConfig
from .models import Tool, FunctionCall, ConversationItem
from .exceptions import (
    RealtimeError, SessionError, AudioError, ConnectionError,
    AuthenticationError, APIError
)


class RealtimeClient:
    """
    Main client for OpenAI Realtime API
    
    Provides a high-level interface for real-time voice conversations with GPT-4,
    including text and audio input/output, function calling, and session management.
    """
    
    def __init__(
        self,
        api_key: str,
        logger: Optional[logging.Logger] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 5
    ):
        """
        Initialize RealtimeClient
        
        Args:
            api_key: OpenAI API key with Realtime API access
            logger: Optional logger instance
            auto_reconnect: Whether to automatically reconnect on connection loss
            max_reconnect_attempts: Maximum number of reconnection attempts
        """
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Core components
        self.connection = RealtimeConnection(api_key, logger)
        self.event_dispatcher = EventDispatcher(logger)
        self.audio_processor = AudioProcessor(logger)
        
        # State management
        self.session_config: Optional[SessionConfig] = None
        self.session_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.is_session_active = False
        self.is_connected = False
        
        # Conversation state
        self.conversation_items: List[ConversationItem] = []
        self.current_response_id: Optional[str] = None
        
        # Audio management
        self.audio_output_buffer = bytearray()
        self.current_response_audio = bytearray()
        self.input_audio_buffer = bytearray()
        
        # Response tracking
        self.pending_responses: Dict[str, Dict[str, Any]] = {}
        self.function_calls: Dict[str, FunctionCall] = {}
        
        # Reconnection state
        self.reconnect_attempts = 0
        self.last_disconnect_time = 0
        
        # Setup internal event handlers
        self._setup_internal_handlers()
        
        # Set connection message handler
        self.connection.set_message_handler(self._handle_message)
    
    def _setup_internal_handlers(self):
        """Setup internal event handlers for session and conversation management"""
        
        # Use direct function calls instead of decorators to avoid the decorator issue
        async def handle_session_created(event_data):
            session = event_data.get("session", {})
            self.session_id = session.get("id")
            self.is_session_active = True
            self.reconnect_attempts = 0  # Reset on successful connection
            self.logger.info(f"Session created: {self.session_id}")
        
        async def handle_session_updated(event_data):
            session = event_data.get("session", {})
            self.logger.info(f"Session updated: {session.get('id', 'unknown')}")
        
        async def handle_conversation_created(event_data):
            conversation = event_data.get("conversation", {})
            self.conversation_id = conversation.get("id")
            self.logger.info(f"Conversation created: {self.conversation_id}")
        
        async def handle_item_created(event_data):
            item_data = event_data.get("item", {})
            item = ConversationItem(
                id=item_data.get("id", ""),
                type=item_data.get("type", ""),
                status=item_data.get("status", ""),
                role=item_data.get("role"),
                content=item_data.get("content", [])
            )
            self.conversation_items.append(item)
            self.logger.debug(f"Conversation item created: {item.id} ({item.type})")
        
        async def handle_error(event_data):
            error = event_data.get("error", {})
            error_type = error.get("type", "unknown")
            error_code = error.get("code", "unknown")
            error_message = error.get("message", "Unknown error")
            
            self.logger.error(f"Realtime API error [{error_type}:{error_code}]: {error_message}")
            
            # Handle specific error types
            if error_type == "invalid_request_error":
                raise APIError(f"Invalid request: {error_message}", error_code, error_type)
            elif error_type == "authentication_error":
                raise AuthenticationError(f"Authentication failed: {error_message}")
            elif error_type == "rate_limit_error":
                self.logger.warning("Rate limit exceeded, waiting...")
                await asyncio.sleep(1)
        
        # Response handling
        async def handle_response_created(event_data):
            response = event_data.get("response", {})
            response_id = response.get("id")
            self.current_response_id = response_id
            self.pending_responses[response_id] = {
                "status": "in_progress",
                "output": [],
                "usage": None,
                "created_at": time.time()
            }
            self.logger.debug(f"Response created: {response_id}")
        
        async def handle_response_done(event_data):
            response = event_data.get("response", {})
            response_id = response.get("id")
            
            if response_id in self.pending_responses:
                self.pending_responses[response_id].update({
                    "status": "completed",
                    "output": response.get("output", []),
                    "usage": response.get("usage"),
                    "completed_at": time.time()
                })
            
            # Move current response audio to output buffer
            if self.current_response_audio:
                self.audio_output_buffer.extend(self.current_response_audio)
                self.current_response_audio.clear()
            
            self.current_response_id = None
            self.logger.debug(f"Response completed: {response_id}")
        
        # Audio handling
        async def handle_audio_delta(event_data):
            audio_b64 = event_data.get("delta", "")
            if audio_b64:
                try:
                    audio_bytes = self.audio_processor.base64_to_bytes(audio_b64)
                    self.current_response_audio.extend(audio_bytes)
                except Exception as e:
                    self.logger.error(f"Failed to decode audio delta: {e}")
        
        async def handle_audio_done(event_data):
            self.logger.debug("Audio response completed")
        
        # Input audio buffer events
        async def handle_audio_committed(event_data):
            item_id = event_data.get("item_id")
            self.logger.debug(f"Audio input committed: {item_id}")
        
        async def handle_audio_cleared(event_data):
            self.input_audio_buffer.clear()
            self.logger.debug("Audio input buffer cleared")
        
        async def handle_speech_started(event_data):
            item_id = event_data.get("item_id")
            audio_start_ms = event_data.get("audio_start_ms", 0)
            self.logger.debug(f"Speech started: {item_id} at {audio_start_ms}ms")
        
        async def handle_speech_stopped(event_data):
            item_id = event_data.get("item_id")
            audio_end_ms = event_data.get("audio_end_ms", 0)
            self.logger.debug(f"Speech stopped: {item_id} at {audio_end_ms}ms")
        
        # Function call handling
        async def handle_function_call_done(event_data):
            call_id = event_data.get("call_id")
            item_id = event_data.get("item_id")
            arguments = event_data.get("arguments", "{}")
            
            # Find the function call in the response items
            if self.current_response_id and self.current_response_id in self.pending_responses:
                response_data = self.pending_responses[self.current_response_id]
                for item in response_data.get("output", []):
                    if item.get("id") == item_id and item.get("type") == "function_call":
                        function_call = FunctionCall(
                            name=item.get("name", ""),
                            arguments=arguments,
                            call_id=call_id
                        )
                        self.function_calls[call_id] = function_call
                        self.logger.info(f"Function call ready: {function_call.name} ({call_id})")
                        break
        
        # Register all handlers using direct calls
        self.event_dispatcher.on(EventType.SESSION_CREATED.value, handle_session_created)
        self.event_dispatcher.on(EventType.SESSION_UPDATED.value, handle_session_updated)
        self.event_dispatcher.on(EventType.CONVERSATION_CREATED.value, handle_conversation_created)
        self.event_dispatcher.on(EventType.CONVERSATION_ITEM_CREATED.value, handle_item_created)
        self.event_dispatcher.on(EventType.ERROR.value, handle_error)
        self.event_dispatcher.on(EventType.RESPONSE_CREATED.value, handle_response_created)
        self.event_dispatcher.on(EventType.RESPONSE_DONE.value, handle_response_done)
        self.event_dispatcher.on(EventType.RESPONSE_AUDIO_DELTA.value, handle_audio_delta)
        self.event_dispatcher.on(EventType.RESPONSE_AUDIO_DONE.value, handle_audio_done)
        self.event_dispatcher.on(EventType.INPUT_AUDIO_BUFFER_COMMITTED.value, handle_audio_committed)
        self.event_dispatcher.on(EventType.INPUT_AUDIO_BUFFER_CLEARED.value, handle_audio_cleared)
        self.event_dispatcher.on(EventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED.value, handle_speech_started)
        self.event_dispatcher.on(EventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED.value, handle_speech_stopped)
        self.event_dispatcher.on(EventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE.value, handle_function_call_done)
    
    async def connect(self, session_config: Optional[SessionConfig] = None) -> bool:
        """
        Connect to Realtime API and configure session
        
        Args:
            session_config: Optional session configuration
            
        Returns:
            True if connection and session setup successful
        """
        try:
            # Connect WebSocket
            self.logger.info("Connecting to OpenAI Realtime API...")
            success = await self.connection.connect()
            
            if not success:
                raise ConnectionError("Failed to establish WebSocket connection")
            
            self.is_connected = True
            self.logger.info("WebSocket connected successfully")
            
            # Configure session if provided
            if session_config:
                await self.configure_session(session_config)
            
            # Wait for session creation
            timeout = 10  # seconds
            start_time = time.time()
            while not self.is_session_active and time.time() - start_time < timeout:
                await asyncio.sleep(0.1)
            
            if not self.is_session_active:
                raise SessionError("Session not created within timeout")
            
            self.logger.info("Realtime API session active")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Realtime API and cleanup"""
        try:
            self.logger.info("Disconnecting from Realtime API...")
            
            # Cancel any pending responses
            if self.current_response_id:
                await self.cancel_response()
            
            # Disconnect WebSocket
            await self.connection.disconnect()
            
            # Reset state
            self.is_connected = False
            self.is_session_active = False
            self.session_id = None
            self.conversation_id = None
            self.conversation_items.clear()
            self.pending_responses.clear()
            self.function_calls.clear()
            
            # Clear audio buffers
            self.audio_output_buffer.clear()
            self.current_response_audio.clear()
            self.input_audio_buffer.clear()
            
            self.logger.info("Disconnected successfully")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            raise
    
    async def configure_session(self, config: SessionConfig):
        """
        Configure the session parameters
        
        Args:
            config: Session configuration object
        """
        if not self.is_connected:
            raise SessionError("Not connected to Realtime API")
        
        self.session_config = config
        
        event = {
            "type": "session.update",
            "session": config.to_dict()
        }
        
        await self.connection.send_event(event)
        self.logger.info("Session configuration sent")
    
    async def _handle_message(self, event_data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            event_type = event_data.get("type", "unknown")
            event = RealtimeEvent(
                event_type=event_type,
                data=event_data,
                event_id=event_data.get("event_id")
            )
            
            await self.event_dispatcher.dispatch(event)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            if self.auto_reconnect:
                await self._attempt_reconnect()
    
    async def _attempt_reconnect(self):
        """Attempt to reconnect to the API"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        self.reconnect_attempts += 1
        delay = min(2 ** self.reconnect_attempts, 30)  # Exponential backoff, max 30s
        
        self.logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {delay}s")
        await asyncio.sleep(delay)
        
        try:
            await self.connect(self.session_config)
            return True
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self.reconnect_attempts} failed: {e}")
            return False
    
    # Event handler registration
    def on_event(self, event_type: str, handler: Callable = None):
        """
        Register event handler - supports both direct call and decorator syntax
        
        Usage:
            # Direct call
            client.on_event("response.text.delta", my_handler)
            
            # Decorator syntax
            @client.on_event("response.text.delta")
            async def my_handler(event_data):
                pass
        """
        if handler is None:
            # Used as decorator
            def decorator(func: Callable):
                self.event_dispatcher.on(event_type, func)
                return func
            return decorator
        else:
            # Direct call
            self.event_dispatcher.on(event_type, handler)
    
    def off_event(self, event_type: str, handler: Callable):
        """Unregister event handler"""
        self.event_dispatcher.off(event_type, handler)
    
    # Conversation methods
    async def send_text(self, text: str, item_id: Optional[str] = None) -> str:
        """
        Send text message to conversation
        
        Args:
            text: Text message to send
            item_id: Optional custom item ID
            
        Returns:
            Item ID of the created message
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        message_id = item_id or f"msg_{uuid.uuid4().hex[:8]}"
        
        # Create conversation item
        event = {
            "type": "conversation.item.create",
            "item": {
                "id": message_id,
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }
        
        await self.connection.send_event(event)
        self.logger.info(f"Sent text message: {text[:50]}...")
        
        # Trigger response
        await self.create_response()
        
        return message_id
    
    # NEW AUDIO METHODS
    async def send_audio_simple(self, audio_bytes: bytes) -> bool:
        """
        Simple audio sending that works with Server VAD.
        This is the recommended method for sending audio.
        
        Args:
            audio_bytes: PCM16 audio data at 24kHz, mono
            
        Returns:
            True if sent successfully
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        try:
            # Validate audio
            is_valid, error_msg = self.audio_processor.validator.validate_audio_data(
                audio_bytes, AudioFormat.PCM16
            )
            if not is_valid:
                raise AudioError(f"Invalid audio: {error_msg}")
            
            # Clear any previous audio
            await self.clear_audio_input()
            await asyncio.sleep(0.3)
            
            # Send audio
            audio_b64 = self.audio_processor.bytes_to_base64(audio_bytes)
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await self.connection.send_event(event)
            
            duration_ms = self.audio_processor.get_audio_duration_ms(audio_bytes)
            self.logger.info(f"Sent {len(audio_bytes)} bytes of audio ({duration_ms:.1f}ms)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send audio: {e}")
            raise
    
    async def send_audio_and_wait_for_response(
        self, 
        audio_bytes: bytes,
        timeout: float = 30.0
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Send audio and wait for both text and audio response.
        
        Args:
            audio_bytes: PCM16 audio data at 24kHz
            timeout: Maximum time to wait for response
            
        Returns:
            Tuple of (response_text, response_audio_bytes)
        """
        # Send audio
        await self.send_audio_simple(audio_bytes)
        
        # Track response
        response_text = ""
        response_done = False
        start_audio_size = len(self.audio_output_buffer)
        
        async def capture_text(data):
            nonlocal response_text
            response_text += data.get("delta", "")
        
        async def capture_done(data):
            nonlocal response_done
            response_done = True
        
        # Register handlers
        self.on_event("response.text.delta", capture_text)
        self.on_event("response.done", capture_done)
        
        try:
            # Wait for response
            start_time = time.time()
            while not response_done and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            # Get audio response
            response_audio = None
            if len(self.audio_output_buffer) > start_audio_size:
                response_audio = bytes(self.audio_output_buffer[start_audio_size:])
            
            return response_text, response_audio
            
        finally:
            # Cleanup handlers
            self.off_event("response.text.delta", capture_text)
            self.off_event("response.done", capture_done)
    
    async def send_audio_file(self, file_path: Union[str, Path]) -> str:
        """
        Send audio file to conversation
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Item ID of the created message
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        try:
            # Load and process audio file
            audio_bytes = self.audio_processor.load_wav_file(file_path)
            return await self.send_audio_bytes(audio_bytes)
            
        except Exception as e:
            raise AudioError(f"Failed to send audio file {file_path}: {e}")
    
    async def send_audio_bytes(self, audio_bytes: bytes, commit: bool = True) -> Optional[str]:
        """
        Send raw audio bytes to input buffer.
        
        NOTE: With Server VAD enabled (required by API), the commit parameter
        is ignored as the server automatically commits when speech ends.
        
        Args:
            audio_bytes: Raw PCM16 audio data
            commit: Ignored (kept for backwards compatibility)
            
        Returns:
            Item ID (estimated) if audio sent successfully
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        try:
            await self.send_audio_simple(audio_bytes)
            return f"msg_{uuid.uuid4().hex[:8]}"
        except Exception as e:
            raise AudioError(f"Failed to send audio bytes: {e}")
    
    async def send_audio_chunks(
        self, 
        audio_bytes: bytes, 
        chunk_size_ms: int = 100,
        real_time: bool = True
    ) -> str:
        """
        Send audio in chunks for streaming.
        
        Args:
            audio_bytes: Raw audio data
            chunk_size_ms: Chunk size in milliseconds
            real_time: Whether to send chunks in real-time with delays
            
        Returns:
            Item ID of the committed audio
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        try:
            # Clear buffer first
            await self.clear_audio_input()
            await asyncio.sleep(0.3)
            
            chunks = self.audio_processor.chunk_audio(audio_bytes, chunk_size_ms)
            
            for i, chunk in enumerate(chunks):
                audio_b64 = self.audio_processor.bytes_to_base64(chunk)
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await self.connection.send_event(event)
                
                if real_time and i < len(chunks) - 1:  # Don't delay after last chunk
                    await asyncio.sleep(chunk_size_ms / 1000)
            
            # With Server VAD, audio is automatically committed
            return f"msg_{uuid.uuid4().hex[:8]}"
            
        except Exception as e:
            raise AudioError(f"Failed to send audio chunks: {e}")
    
    async def commit_audio_input(self) -> str:
        """
        Commit audio input buffer and trigger response.
        
        NOTE: This is usually not needed with Server VAD as it auto-commits.
        Only use this if you have turn_detection.create_response = False.
        
        Returns:
            Item ID of the committed audio message
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        event = {
            "type": "input_audio_buffer.commit"
        }
        await self.connection.send_event(event)
        
        # Trigger response if needed
        if self.session_config and self.session_config.turn_detection:
            td = self.session_config.turn_detection
            if isinstance(td, dict) and not td.get("create_response", True):
                await self.create_response()
            elif hasattr(td, "create_response") and not td.create_response:
                await self.create_response()
        
        # Generate item ID (will be confirmed by server)
        return f"msg_{uuid.uuid4().hex[:8]}"
    
    async def clear_audio_input(self):
        """Clear audio input buffer"""
        if not self.is_session_active:
            raise SessionError("No active session")
        
        event = {
            "type": "input_audio_buffer.clear"
        }
        await self.connection.send_event(event)
        self.logger.debug("Audio input buffer cleared")
    
    async def create_response(
        self, 
        instructions: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[Union[int, str]] = None
    ) -> str:
        """
        Trigger model response
        
        Args:
            instructions: Optional response-specific instructions
            modalities: Optional modalities override
            temperature: Optional temperature override
            max_output_tokens: Optional token limit override
            
        Returns:
            Response ID
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        response_config = {}
        
        if instructions:
            response_config["instructions"] = instructions
        if modalities:
            response_config["modalities"] = modalities
        if temperature is not None:
            response_config["temperature"] = temperature
        if max_output_tokens is not None:
            response_config["max_output_tokens"] = max_output_tokens
        
        event = {
            "type": "response.create",
            "response": response_config
        }
        
        await self.connection.send_event(event)
        
        # Generate response ID (will be replaced by server)
        response_id = f"resp_{uuid.uuid4().hex[:8]}"
        self.logger.debug(f"Response creation requested: {response_id}")
        
        return response_id
    
    async def cancel_response(self, response_id: Optional[str] = None):
        """
        Cancel ongoing response
        
        Args:
            response_id: Optional specific response ID to cancel
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        event = {
            "type": "response.cancel"
        }
        
        if response_id:
            event["response_id"] = response_id
        
        await self.connection.send_event(event)
        self.logger.debug(f"Response cancellation requested: {response_id or 'current'}")
    
    # Function calling
    async def submit_function_result(
        self, 
        call_id: str, 
        result: Any,
        item_id: Optional[str] = None
    ) -> str:
        """
        Submit function call result
        
        Args:
            call_id: Function call ID
            result: Function execution result
            item_id: Optional custom item ID
            
        Returns:
            Item ID of the function result
        """
        if not self.is_session_active:
            raise SessionError("No active session")
        
        result_id = item_id or f"funcres_{uuid.uuid4().hex[:8]}"
        
        # Convert result to string if necessary
        if isinstance(result, (dict, list)):
            import json
            result_str = json.dumps(result)
        else:
            result_str = str(result)
        
        event = {
            "type": "conversation.item.create",
            "item": {
                "id": result_id,
                "type": "function_call_output",
                "call_id": call_id,
                "output": result_str
            }
        }
        
        await self.connection.send_event(event)
        self.logger.info(f"Submitted function result for {call_id}")
        
        # Trigger response to continue conversation
        await self.create_response()
        
        return result_id
    
    # Audio output methods
    def get_audio_output(self, clear_buffer: bool = True) -> bytes:
        """
        Get accumulated audio output
        
        Args:
            clear_buffer: Whether to clear the buffer after reading
            
        Returns:
            Raw PCM16 audio bytes
        """
        output = bytes(self.audio_output_buffer)
        if clear_buffer:
            self.audio_output_buffer.clear()
        return output
    
    def save_audio_output(
        self, 
        file_path: Union[str, Path], 
        clear_buffer: bool = True
    ) -> bool:
        """
        Save accumulated audio output to file
        
        Args:
            file_path: Output file path
            clear_buffer: Whether to clear the buffer after saving
            
        Returns:
            True if audio was saved, False if no audio available
        """
        audio_data = self.get_audio_output(clear_buffer)
        if audio_data:
            try:
                self.audio_processor.save_wav_file(audio_data, file_path)
                self.logger.info(f"Audio output saved: {file_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save audio output: {e}")
                raise AudioError(f"Failed to save audio output: {e}")
        return False
    
    def get_audio_output_duration(self) -> float:
        """
        Get duration of accumulated audio output
        
        Returns:
            Duration in milliseconds
        """
        return self.audio_processor.get_audio_duration_ms(bytes(self.audio_output_buffer))
    
    # Conversation management
    def get_conversation_items(self) -> List[ConversationItem]:
        """Get all conversation items"""
        return self.conversation_items.copy()
    
    def get_conversation_history(self, include_audio: bool = False) -> List[Dict[str, Any]]:
        """
        Get conversation history in a readable format
        
        Args:
            include_audio: Whether to include audio content
            
        Returns:
            List of conversation items as dictionaries
        """
        history = []
        for item in self.conversation_items:
            item_dict = {
                "id": item.id,
                "type": item.type,
                "role": item.role,
                "status": item.status,
                "content": []
            }
            
            for content in item.content:
                if content.get("type") == "input_text":
                    item_dict["content"].append({
                        "type": "text",
                        "text": content.get("text", "")
                    })
                elif content.get("type") == "input_audio" and include_audio:
                    item_dict["content"].append({
                        "type": "audio",
                        "transcript": content.get("transcript", ""),
                        "duration_ms": len(content.get("audio", "")) * 1000 // AudioConfig.SAMPLE_RATE if content.get("audio") else 0
                    })
            
            history.append(item_dict)
        
        return history
    
    # Status and diagnostics
    def get_status(self) -> Dict[str, Any]:
        """Get current client status"""
        return {
            "connected": self.is_connected,
            "session_active": self.is_session_active,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "conversation_items": len(self.conversation_items),
            "pending_responses": len(self.pending_responses),
            "function_calls": len(self.function_calls),
            "audio_output_duration_ms": self.get_audio_output_duration(),
            "input_audio_buffer_size": len(self.input_audio_buffer),
            "reconnect_attempts": self.reconnect_attempts
        }
    
    def get_pending_responses(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending responses"""
        return self.pending_responses.copy()
    
    def get_function_calls(self) -> Dict[str, FunctionCall]:
        """Get all pending function calls"""
        return self.function_calls.copy()
    
    async def wait_for_response(
        self, 
        response_id: Optional[str] = None, 
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a specific response to complete
        
        Args:
            response_id: Response ID to wait for (None for current)
            timeout: Maximum time to wait in seconds
            
        Returns:
            Response data or None if timeout
        """
        target_id = response_id or self.current_response_id
        if not target_id:
            return None
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if target_id in self.pending_responses:
                response = self.pending_responses[target_id]
                if response["status"] == "completed":
                    return response
            
            await asyncio.sleep(0.1)
        
        self.logger.warning(f"Timeout waiting for response: {target_id}")
        return None
    
    # Helper methods for common patterns
    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Use the API to transcribe audio to text.
        
        Args:
            audio_bytes: Audio to transcribe
            
        Returns:
            Transcribed text
        """
        # Temporarily set instructions for transcription
        original_config = self.session_config
        
        transcription_config = SessionConfig(
            instructions="Transcribe the audio exactly as spoken. Do not add any commentary.",
            modalities=["text"],  # Text only for transcription
            temperature=0.3,  # Low temperature for accuracy
        )
        
        await self.configure_session(transcription_config)
        
        try:
            # Send audio and get response
            text, _ = await self.send_audio_and_wait_for_response(audio_bytes, timeout=20)
            
            # Restore original config
            if original_config:
                await self.configure_session(original_config)
            
            return text.strip()
            
        except Exception as e:
            # Restore original config on error
            if original_config:
                await self.configure_session(original_config)
            raise
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()