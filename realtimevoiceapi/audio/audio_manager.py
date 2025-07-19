"""
Audio Manager - Unified audio interface

Manages all audio components and provides a safe, clean interface
for audio operations. Handles all sounddevice interactions in one place.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import threading
import queue

from ..core.audio_types import AudioBytes, AudioConfig, VADConfig
from ..core.exceptions import AudioError
from ..fast_lane.direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from ..fast_lane.fast_vad_detector import FastVADDetector


@dataclass
class AudioManagerConfig:
    """Configuration for AudioManager"""
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 24000
    channels: int = 1
    chunk_duration_ms: int = 100
    vad_enabled: bool = True
    vad_config: Optional[VADConfig] = None


class AudioManager:
    """
    Unified audio manager that safely handles all audio operations.
    
    This abstraction:
    - Manages audio capture and playback
    - Handles VAD processing
    - Provides thread-safe operations
    - Ensures proper cleanup
    """
    
    def __init__(self, config: AudioManagerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Audio components (created lazily)
        self._capture: Optional[DirectAudioCapture] = None
        self._player: Optional[DirectAudioPlayer] = None
        self._vad: Optional[FastVADDetector] = None
        
        # State
        self._is_initialized = False
        self._is_capturing = False
        self._capture_queue: Optional[asyncio.Queue] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Audio configuration
        self._audio_config = AudioConfig(
            sample_rate=config.sample_rate,
            channels=config.channels,
            chunk_duration_ms=config.chunk_duration_ms
        )
    
    async def initialize(self) -> None:
        """Initialize audio components"""
        with self._lock:
            if self._is_initialized:
                return
            
            try:
                # Create capture
                self._capture = DirectAudioCapture(
                    device=self.config.input_device,
                    config=self._audio_config,
                    logger=self.logger
                )
                
                # Create player
                self._player = DirectAudioPlayer(
                    device=self.config.output_device,
                    config=self._audio_config,
                    logger=self.logger
                )
                
                # Create VAD if enabled
                if self.config.vad_enabled:
                    self._vad = FastVADDetector(
                        config=self.config.vad_config or VADConfig(),
                        audio_config=self._audio_config
                    )
                
                self._is_initialized = True
                self.logger.info("Audio manager initialized")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize audio: {e}")
                await self.cleanup()
                raise AudioError(f"Audio initialization failed: {e}")
    
    async def start_capture(self) -> asyncio.Queue[AudioBytes]:
        """Start audio capture and return queue"""
        with self._lock:
            if not self._is_initialized:
                raise AudioError("Audio manager not initialized")
            
            if self._is_capturing:
                raise AudioError("Already capturing")
            
            self._capture_queue = await self._capture.start_async_capture()
            self._is_capturing = True
            
            return self._capture_queue
    
    async def stop_capture(self) -> None:
        """Stop audio capture"""
        with self._lock:
            if not self._is_capturing:
                return
            
            if self._capture:
                self._capture.stop_capture()
            
            self._is_capturing = False
            self._capture_queue = None
    
    def play_audio(self, audio_data: AudioBytes) -> bool:
        """Play audio safely"""
        with self._lock:
            if not self._is_initialized or not self._player:
                self.logger.warning("Audio player not initialized")
                return False
            
            try:
                return self._player.play_audio(audio_data)
            except Exception as e:
                self.logger.error(f"Playback error: {e}")
                return False
    
    def stop_playback(self) -> None:
        """Stop any ongoing playback"""
        with self._lock:
            if self._player:
                try:
                    self._player.stop_playback()
                except Exception as e:
                    self.logger.debug(f"Stop playback error: {e}")
    
    def process_vad(self, audio_chunk: AudioBytes) -> Optional[str]:
        """Process audio through VAD"""
        if not self._vad:
            return None
        
        try:
            state = self._vad.process_chunk(audio_chunk)
            return state.value
        except Exception as e:
            self.logger.error(f"VAD processing error: {e}")
            return None
    
    async def cleanup(self) -> None:
        """Cleanup all audio resources safely"""
        with self._lock:
            # Stop capture
            if self._is_capturing and self._capture:
                try:
                    self._capture.stop_capture()
                except Exception as e:
                    self.logger.debug(f"Capture stop error: {e}")
            
            # Stop playback
            if self._player:
                try:
                    self._player.stop_playback()
                except Exception as e:
                    self.logger.debug(f"Player stop error: {e}")
            
            # Clear references
            self._capture = None
            self._player = None
            self._vad = None
            self._is_initialized = False
            self._is_capturing = False
            
            self.logger.info("Audio manager cleaned up")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audio metrics"""
        metrics = {
            "initialized": self._is_initialized,
            "capturing": self._is_capturing
        }
        
        with self._lock:
            if self._capture and hasattr(self._capture, 'get_metrics'):
                try:
                    metrics["capture"] = self._capture.get_metrics()
                except Exception as e:
                    metrics["capture"] = {"error": str(e)}
            
            if self._player and hasattr(self._player, 'get_metrics'):
                try:
                    metrics["player"] = self._player.get_metrics()
                except Exception as e:
                    metrics["player"] = {"error": str(e)}
            
            if self._vad and hasattr(self._vad, 'get_metrics'):
                try:
                    metrics["vad"] = self._vad.get_metrics()
                except Exception as e:
                    metrics["vad"] = {"error": str(e)}
        
        return metrics
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            # Use sync cleanup in destructor
            if self._player:
                self._player.stop_playback()
            if self._capture and self._is_capturing:
                self._capture.stop_capture()
        except Exception:
            pass  # Ignore errors in destructor