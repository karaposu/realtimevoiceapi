"""
Audio Engine Module - Comprehensive audio processing for real-time voice applications.
"""

# Core types and configurations
from .audio_types import (
    AudioConfig,
    AudioFormat,
    AudioQuality,
    ProcessingMode,
    VADType,
    VADConfig,
    BufferConfig,
    AudioConstants,
    AudioBytes
)

# Main components
from .audio_engine import AudioEngine
from .audio_processor import AudioProcessor, BufferPool, AudioStreamBuffer
from .audio_manager import AudioManager, AudioManagerConfig

# Audio I/O components
from .direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from .buffered_audio_player import BufferedAudioPlayer

# VAD components
from .fast_vad_detector import FastVADDetector, VADState

# Exceptions
from .exceptions import AudioError

# Interfaces
from .audio_interfaces import (
    AudioComponent,
    AudioPlayerInterface,
    AudioCaptureInterface
)

__all__ = [
    # Core types
    'AudioConfig',
    'AudioFormat', 
    'AudioQuality',
    'ProcessingMode',
    'VADType',
    'VADConfig',
    'BufferConfig',
    'AudioConstants',
    'AudioBytes',
    
    # Main components
    'AudioEngine',
    'AudioProcessor',
    'BufferPool',
    'AudioStreamBuffer',
    'AudioManager',
    'AudioManagerConfig',
    
    # Audio I/O
    'DirectAudioCapture',
    'DirectAudioPlayer',
    'BufferedAudioPlayer',
    
    # VAD
    'FastVADDetector',
    'VADState',
    
    # Exceptions
    'AudioError',
    
    # Interfaces
    'AudioComponent',
    'AudioPlayerInterface',
    'AudioCaptureInterface'
]

__version__ = '0.1.0'