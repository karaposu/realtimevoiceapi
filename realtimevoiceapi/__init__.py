# here is realtimevoiceapi/__init__.py


"""
RealtimeVoiceAPI - Modern Python framework for OpenAI's Realtime API
"""

__version__ = "0.2.0"

# Core imports for smoke tests
from .voice_engine import VoiceEngine, VoiceEngineConfig
from .core.audio_processor import AudioProcessor
from realtimevoiceapi.session import SessionConfig, SessionPresets
from .core.exceptions import (
    RealtimeError,
    ConnectionError,
    AuthenticationError,
    AudioError,
    StreamError,
    EngineError,
)

# Model imports for smoke tests
from .audio.models import (
    Tool,
    TurnDetectionConfig,
    TranscriptionConfig,
    AudioFormatType,
    ModalityType,
    VoiceType,
)

# Message protocol (used by smoke tests)
from .core.message_protocol import (
    ClientMessageType,
    ServerMessageType,
    MessageFactory,
    MessageValidator,
    ProtocolInfo,
)

# Audio types (used by smoke tests)
from .core.audio_types import (
    AudioFormat,
    AudioConfig,
    VADConfig,
    VADType,
)

# Stream protocol (used by smoke tests)
from .core.stream_protocol import (
    StreamEvent,
    StreamEventType,
    StreamState,
)

# Session manager (used by smoke tests)
from .session.session_manager import SessionManager

__all__ = [
    "__version__",
    "VoiceEngine",
    "VoiceEngineConfig",
    "AudioProcessor",
    "SessionConfig",
    "SessionPresets",
    "RealtimeError",
    "ConnectionError",
    "AuthenticationError",
    "AudioError",
    "StreamError",
    "EngineError",
    "Tool",
    "TurnDetectionConfig",
    "TranscriptionConfig",
    "AudioFormatType",
    "ModalityType",
    "VoiceType",
    "ClientMessageType",
    "ServerMessageType",
    "MessageFactory",
    "MessageValidator",
    "ProtocolInfo",
    "AudioFormat",
    "AudioConfig",
    "VADConfig",
    "VADType",
    "StreamEvent",
    "StreamEventType",
    "StreamState",
    "SessionManager",
]