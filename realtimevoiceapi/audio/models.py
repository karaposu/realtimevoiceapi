# realtimevoiceapi/audio/models.py
"""Type definitions for Realtime Voice API"""

from typing import Union, Literal, Dict, Any, Optional, List
from dataclasses import dataclass, field

# Audio format types
AudioFormatType = Literal["pcm16", "g711_ulaw", "g711_alaw"]

# Modality types
ModalityType = Literal["text", "audio"]

# Voice types - Updated to match API specification
VoiceType = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]

# Tool choice types
ToolChoiceType = Union[Literal["auto", "none", "required"], str]

# Turn detection types - Only server_vad and semantic_vad are supported
TurnDetectionType = Literal["server_vad", "semantic_vad"]


@dataclass
class TurnDetectionConfig:
    """
    Configuration for turn detection.
    
    Note: Different parameters are supported by different VAD types:
    - server_vad: supports threshold, prefix_padding_ms, silence_duration_ms, create_response
    - semantic_vad: only supports type and create_response
    """
    type: TurnDetectionType = "server_vad"
    threshold: Optional[float] = 0.5  # Only for server_vad
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 200
    create_response: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding unsupported parameters based on VAD type"""
        
        if self.type == "semantic_vad":
            # Semantic VAD only supports type and create_response
            return {
                "type": self.type,
                "create_response": self.create_response
            }
        elif self.type == "server_vad":
            # Server VAD supports all parameters
            config = {
                "type": self.type,
                "prefix_padding_ms": self.prefix_padding_ms,
                "silence_duration_ms": self.silence_duration_ms,
                "create_response": self.create_response
            }
            
            # Only include threshold if it's specified
            if self.threshold is not None:
                config["threshold"] = self.threshold
                
            return config
        else:
            # Unknown type, return minimal config
            return {
                "type": self.type,
                "create_response": self.create_response
            }


@dataclass
class TranscriptionConfig:
    """Configuration for input audio transcription"""
    model: str = "whisper-1"
    language: Optional[str] = None
    prompt: str = ""


@dataclass
class Tool:
    """Function tool definition"""
    type: str = "function"
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCall:
    """Function call from the model"""
    name: str
    arguments: str
    call_id: str


@dataclass
class ConversationItem:
    """A conversation item (message, function call, etc.)"""
    id: str
    type: str
    status: str
    role: Optional[str] = None
    content: List[Dict[str, Any]] = field(default_factory=list)