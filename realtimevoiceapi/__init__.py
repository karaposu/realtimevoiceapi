# here is realtimevoiceapi/__init__.py

"""
RealtimeVoiceAPI - Python client for OpenAI's Realtime API

A comprehensive library for real-time voice conversations with GPT-4.
"""

# Import all components in the correct order
from .exceptions import (
    RealtimeError, ConnectionError, AuthenticationError, SessionError,
    AudioError, ConfigurationError, RateLimitError, APIError
)

from .models import (
    AudioFormatType, ModalityType, VoiceType, ToolChoiceType,
    TurnDetectionConfig, TranscriptionConfig, Tool, FunctionCall, ConversationItem
)

from .audio import AudioProcessor, AudioFormat, AudioConfig, load_audio_file, save_audio_file
from .events import EventDispatcher, RealtimeEvent, EventType
from .connection import RealtimeConnection
from .session import SessionConfig
from .client import RealtimeClient

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main exports
__all__ = [
    # Core classes
    "RealtimeClient",
    "SessionConfig", 
    "AudioProcessor",
    "RealtimeConnection",
    "EventDispatcher",
    
    # Events and types
    "RealtimeEvent",
    "EventType",
    
    # Type definitions
    "AudioFormatType",
    "ModalityType", 
    "VoiceType",
    "ToolChoiceType",
    "TurnDetectionConfig",
    "TranscriptionConfig",
    "Tool",
    "FunctionCall",
    "ConversationItem",
    
    # Audio utilities
    "AudioFormat",
    "AudioConfig",
    "load_audio_file",
    "save_audio_file",
    
    # Exceptions
    "RealtimeError",
    "ConnectionError", 
    "AuthenticationError",
    "SessionError",
    "AudioError",
    "ConfigurationError",
    "RateLimitError",
    "APIError"
]

def get_version():
    """Get the package version"""
    return __version__