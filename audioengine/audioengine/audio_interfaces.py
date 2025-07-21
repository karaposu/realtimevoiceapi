
#here is realtimevoiceapi/core/audio_interfaces.py

"""Audio component interfaces for consistent behavior"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .audio_types import AudioBytes, AudioConfig


class AudioComponent(ABC):
    """Base interface for all audio components"""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        pass


class AudioCaptureInterface(AudioComponent):
    """Interface for audio capture components"""
    
    @abstractmethod
    async def start_async_capture(self) -> Any:
        """Start capturing audio"""
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing audio"""
        pass


class AudioPlayerInterface(AudioComponent):
    """Interface for audio playback components"""
    
    @abstractmethod
    def play_audio(self, audio_data: AudioBytes) -> bool:
        """Play audio data"""
        pass
    
    @abstractmethod
    def stop_playback(self) -> None:
        """Stop any ongoing playback"""
        pass
    
    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Check if currently playing"""
        pass