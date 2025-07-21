# here is realtimevoiceapi/__init__.py

"""
RealtimeVoiceAPI - Modern Python framework for OpenAI's Realtime API
"""

__version__ = "0.2.0"

# Core imports for smoke tests
from .voice_engine import VoiceEngine, VoiceEngineConfig
from audioengine.audioengine.audio_processor import AudioProcessor
from audioengine.audioengine.audio_engine import (
    AudioEngine,
    ProcessingMetrics,
    ProcessingStrategy,
    create_fast_lane_engine,
    create_big_lane_engine,
    create_adaptive_engine,
)
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
# TODO: These models need to be implemented or imported from the correct location
# from .audio.models import (
#     Tool,
#     TurnDetectionConfig,
#     TranscriptionConfig,
#     AudioFormatType,
#     ModalityType,
#     VoiceType,
# )

# Message protocol (used by smoke tests)
from .core.message_protocol import (
    ClientMessageType,
    ServerMessageType,
    MessageFactory,
    MessageValidator,
    ProtocolInfo,
)

# Audio types (used by smoke tests)
from audioengine.audioengine.audio_types import (
    AudioFormat,
    AudioConfig,
    ProcessingMode,
    BufferConfig,
    AudioConstants,
    VADConfig,
    VADType,
)

# Stream protocol (used by smoke tests)
from .core.stream_protocol import (
    StreamEvent,
    StreamEventType,
    StreamState,
)



from audioengine.audioengine.audio_manager import (
    AudioManager,
    AudioManagerConfig,
    # AudioComponentState,
    # create_audio_manager,
)

# Session manager (used by smoke tests)
from .session.session_manager import SessionManager


__all__ = [
    "__version__",
    "VoiceEngine",
    "VoiceEngineConfig",
    "AudioProcessor",
    "AudioEngine",
    "ProcessingMetrics",
    "ProcessingStrategy",
    "create_fast_lane_engine",
    "create_big_lane_engine",
    "create_adaptive_engine",
    "SessionConfig",
    "SessionPresets",
    "RealtimeError",
    "ConnectionError",
    "AuthenticationError",
    "AudioError",
    "StreamError",
    "EngineError",
    # "Tool",
    # "TurnDetectionConfig", 
    # "TranscriptionConfig",
    # "AudioFormatType",
    # "ModalityType",
    # "VoiceType",
    "ClientMessageType",
    "ServerMessageType",
    "MessageFactory",
    "MessageValidator",
    "ProtocolInfo",
    "AudioFormat",
    "AudioConfig",
    "ProcessingMode",
    "BufferConfig",
    "AudioConstants",
    "VADConfig",
    "VADType",
    "StreamEvent",
    "StreamEventType",
    "StreamState",
    "SessionManager",
    "AudioManager",
    "AudioManagerConfig",
    # "AudioComponentState",
    # "create_audio_manager",
]