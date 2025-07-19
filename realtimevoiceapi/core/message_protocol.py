
# here is realtimevoiceapi/core/message_protocol.py


"""
Mmessage Protocol Definition for Realtime Voice API

This module defines all message types, formats, and validation for the
Realtime API protocol. Designed to be thin and shareable between fast
and big lane implementations.

This module provides:

Type Safety: Enums for all message types prevent typos
Message Factory: Clean API for creating messages
Validation: Ensures messages meet protocol requirements
Parser: Extract data from incoming messages
Protocol Info: Constants and validation helpers

Both fast and big lanes can use this without overhead since it's just data structures and simple functions. The fast lane can use MessageFactory directly, while the big lane might wrap it with additional event emission.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Protocol
from dataclasses import dataclass, asdict
import time


# ============== Message Type Definitions ==============

class ClientMessageType(str, Enum):
    """All message types that client can send to server"""
    # Session management
    SESSION_UPDATE = "session.update"
    
    # Audio input
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    
    # Conversation
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    
    # Response
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"


class ServerMessageType(str, Enum):
    """All message types that server can send to client"""
    # Errors
    ERROR = "error"
    
    # Session
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    
    # Conversation
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    
    # Input audio buffer
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    
    # Response
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    
    # Response content
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    
    # Rate limits
    RATE_LIMITS_UPDATED = "rate_limits.updated"


# ============== Message Factories ==============

class MessageFactory:
    """Factory for creating properly formatted messages"""
    
    @staticmethod
    def create_base_message(message_type: Union[ClientMessageType, str]) -> Dict[str, Any]:
        """Create base message with required fields"""
        return {
            "type": message_type.value if hasattr(message_type, 'value') else message_type,
            "event_id": f"evt_{int(time.time() * 1000000)}"
        }
    
    @staticmethod
    def session_update(
        modalities: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        voice: Optional[str] = None,
        input_audio_format: Optional[str] = None,
        output_audio_format: Optional[str] = None,
        input_audio_transcription: Optional[Dict[str, Any]] = None,
        turn_detection: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_response_output_tokens: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """Create session.update message"""
        msg = MessageFactory.create_base_message(ClientMessageType.SESSION_UPDATE)
        session = {}
        
        # Only include non-None values
        if modalities is not None:
            session["modalities"] = modalities
        if instructions is not None:
            session["instructions"] = instructions
        if voice is not None:
            session["voice"] = voice
        if input_audio_format is not None:
            session["input_audio_format"] = input_audio_format
        if output_audio_format is not None:
            session["output_audio_format"] = output_audio_format
        if input_audio_transcription is not None:
            session["input_audio_transcription"] = input_audio_transcription
        if turn_detection is not None:
            session["turn_detection"] = turn_detection
        if tools is not None:
            session["tools"] = tools
        if tool_choice is not None:
            session["tool_choice"] = tool_choice
        if temperature is not None:
            session["temperature"] = temperature
        if max_response_output_tokens is not None:
            session["max_response_output_tokens"] = max_response_output_tokens
            
        msg["session"] = session
        return msg
    
    @staticmethod
    def input_audio_buffer_append(audio_base64: str) -> Dict[str, Any]:
        """Create input_audio_buffer.append message"""
        msg = MessageFactory.create_base_message(ClientMessageType.INPUT_AUDIO_BUFFER_APPEND)
        msg["audio"] = audio_base64
        return msg
    
    @staticmethod
    def input_audio_buffer_commit() -> Dict[str, Any]:
        """Create input_audio_buffer.commit message"""
        return MessageFactory.create_base_message(ClientMessageType.INPUT_AUDIO_BUFFER_COMMIT)
    
    @staticmethod
    def input_audio_buffer_clear() -> Dict[str, Any]:
        """Create input_audio_buffer.clear message"""
        return MessageFactory.create_base_message(ClientMessageType.INPUT_AUDIO_BUFFER_CLEAR)
    
    @staticmethod
    def conversation_item_create(
        item_type: str,
        role: Optional[str] = None,
        content: Optional[List[Dict[str, Any]]] = None,
        call_id: Optional[str] = None,
        output: Optional[str] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create conversation.item.create message"""
        msg = MessageFactory.create_base_message(ClientMessageType.CONVERSATION_ITEM_CREATE)
        
        item = {"type": item_type}
        
        if item_id:
            item["id"] = item_id
            
        if item_type == "message":
            if role:
                item["role"] = role
            if content:
                item["content"] = content
        elif item_type == "function_call_output":
            if call_id:
                item["call_id"] = call_id
            if output:
                item["output"] = output
                
        msg["item"] = item
        return msg
    
    @staticmethod
    def response_create(
        modalities: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        voice: Optional[str] = None,
        output_audio_format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """Create response.create message"""
        msg = MessageFactory.create_base_message(ClientMessageType.RESPONSE_CREATE)
        response = {}
        
        if modalities is not None:
            response["modalities"] = modalities
        if instructions is not None:
            response["instructions"] = instructions
        if voice is not None:
            response["voice"] = voice
        if output_audio_format is not None:
            response["output_audio_format"] = output_audio_format
        if tools is not None:
            response["tools"] = tools
        if tool_choice is not None:
            response["tool_choice"] = tool_choice
        if temperature is not None:
            response["temperature"] = temperature
        if max_output_tokens is not None:
            response["max_output_tokens"] = max_output_tokens
            
        if response:  # Only add if not empty
            msg["response"] = response
            
        return msg
    
    @staticmethod
    def response_cancel() -> Dict[str, Any]:
        """Create response.cancel message"""
        return MessageFactory.create_base_message(ClientMessageType.RESPONSE_CANCEL)


# ============== Message Validation ==============

class MessageValidator:
    """Validates messages against protocol requirements"""
    
    # Required fields for each message type
    REQUIRED_FIELDS = {
        ClientMessageType.SESSION_UPDATE: ["type", "session"],
        ClientMessageType.INPUT_AUDIO_BUFFER_APPEND: ["type", "audio"],
        ClientMessageType.CONVERSATION_ITEM_CREATE: ["type", "item"],
        ClientMessageType.RESPONSE_CREATE: ["type"],
        # Add more as needed
    }
    
    @staticmethod
    def validate_outgoing(message: Dict[str, Any]) -> bool:
        """Validate outgoing message structure"""
        if "type" not in message:
            raise ValueError("Message missing required 'type' field")
            
        msg_type = message["type"]
        
        # Check if it's a known client message type
        try:
            client_type = ClientMessageType(msg_type)
        except ValueError:
            raise ValueError(f"Unknown client message type: {msg_type}")
        
        # Check required fields
        required = MessageValidator.REQUIRED_FIELDS.get(client_type, ["type"])
        for field in required:
            if field not in message:
                raise ValueError(f"Message type {msg_type} missing required field: {field}")
                
        return True
    
    @staticmethod
    def validate_incoming(message: Dict[str, Any]) -> bool:
        """Validate incoming message structure"""
        if "type" not in message:
            raise ValueError("Message missing required 'type' field")
            
        msg_type = message["type"]
        
        # Check if it's a known server message type
        try:
            ServerMessageType(msg_type)
        except ValueError:
            # Not all server messages are in our enum, that's ok
            pass
            
        return True


# ============== Message Parser ==============

class MessageParser:
    """Parse and extract data from messages"""
    
    @staticmethod
    def get_message_type(message: Dict[str, Any]) -> str:
        """Extract message type"""
        return message.get("type", "unknown")
    
    @staticmethod
    def is_error(message: Dict[str, Any]) -> bool:
        """Check if message is an error"""
        return message.get("type") == ServerMessageType.ERROR.value
    
    @staticmethod
    def extract_error(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract error details from error message"""
        if MessageParser.is_error(message):
            return message.get("error", {})
        return None
    
    @staticmethod
    def is_audio_response(message: Dict[str, Any]) -> bool:
        """Check if message contains audio response"""
        return message.get("type") in [
            ServerMessageType.RESPONSE_AUDIO_DELTA.value,
            ServerMessageType.RESPONSE_AUDIO_DONE.value
        ]
    
    @staticmethod
    def extract_audio_delta(message: Dict[str, Any]) -> Optional[str]:
        """Extract audio delta from message"""
        if message.get("type") == ServerMessageType.RESPONSE_AUDIO_DELTA.value:
            return message.get("delta")
        return None
    
    @staticmethod
    def is_text_response(message: Dict[str, Any]) -> bool:
        """Check if message contains text response"""
        return message.get("type") in [
            ServerMessageType.RESPONSE_TEXT_DELTA.value,
            ServerMessageType.RESPONSE_TEXT_DONE.value
        ]
    
    @staticmethod
    def extract_text_delta(message: Dict[str, Any]) -> Optional[str]:
        """Extract text delta from message"""
        if message.get("type") == ServerMessageType.RESPONSE_TEXT_DELTA.value:
            return message.get("delta")
        return None


# ============== Protocol Info ==============

class ProtocolInfo:
    """Protocol metadata and helpers"""
    
    # Audio formats supported by the protocol
    AUDIO_FORMATS = ["pcm16", "g711_ulaw", "g711_alaw"]
    
    # Valid voices
    VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
    
    # Valid modalities
    MODALITIES = ["text", "audio"]
    
    # Turn detection types
    TURN_DETECTION_TYPES = ["server_vad", "semantic_vad"]
    
    @staticmethod
    def is_valid_audio_format(format_str: str) -> bool:
        """Check if audio format is valid"""
        return format_str in ProtocolInfo.AUDIO_FORMATS
    
    @staticmethod
    def is_valid_voice(voice: str) -> bool:
        """Check if voice is valid"""
        return voice in ProtocolInfo.VOICES
    
    @staticmethod
    def is_valid_modality(modality: str) -> bool:
        """Check if modality is valid"""
        return modality in ProtocolInfo.MODALITIES
    
