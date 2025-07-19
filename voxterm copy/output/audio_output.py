"""
VoxTerm Audio Output Management

Manages audio playback, delegating to voice engine when possible.
"""

import asyncio
import threading
import queue
import time
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..core.base import BaseComponent, ComponentState
from ..core.events import Event, EventType, emit_event
from ..core.state import get_state_manager, PlaybackState


@dataclass
class AudioChunk:
    """Audio chunk for playback"""
    data: bytes
    timestamp: float
    duration: float  # Estimated duration in seconds


class AudioOutputManager(BaseComponent):
    """
    Manages audio output for VoxTerm.
    
    Primary mode: Delegates to voice engine's audio playback
    Secondary mode: Can manage its own playback queue
    """
    
    def __init__(self):
        super().__init__("AudioOutputManager")
        
        # Voice engine reference
        self.voice_engine = None
        
        # Playback state
        self._is_playing = False
        self._playback_queue: queue.Queue[AudioChunk] = queue.Queue()
        
        # Metrics
        self.chunks_played = 0
        self.total_duration = 0.0
        self._playback_start_time: Optional[float] = None
        
        # Callbacks
        self._playback_started_callback: Optional[Callable] = None
        self._playback_ended_callback: Optional[Callable] = None
        
        # Volume control
        self._volume = 1.0
        self._is_muted = False
    
    async def initialize(self) -> None:
        """Initialize audio output manager"""
        self.set_state(ComponentState.READY)
        
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={"message": "Audio output manager initialized"}
        ))
    
    async def start(self) -> None:
        """Start audio output manager"""
        self.set_state(ComponentState.RUNNING)
    
    async def stop(self) -> None:
        """Stop audio output manager"""
        # Clear any pending audio
        self.clear_queue()
        
        self.set_state(ComponentState.STOPPED)
    
    def bind_voice_engine(self, voice_engine: Any) -> None:
        """Bind to a voice engine for audio playback"""
        self.voice_engine = voice_engine
    
    def play_audio(self, audio_data: bytes, duration: Optional[float] = None) -> None:
        """
        Play audio data.
        
        When using voice engine, this delegates to the engine.
        """
        if self._is_muted:
            return
        
        # Create audio chunk
        chunk = AudioChunk(
            data=audio_data,
            timestamp=time.time(),
            duration=duration or self._estimate_duration(audio_data)
        )
        
        # Delegate to voice engine if available
        if self.voice_engine and hasattr(self.voice_engine, 'play_audio'):
            try:
                # Apply volume
                if self._volume != 1.0:
                    audio_data = self._apply_volume(audio_data)
                
                self.voice_engine.play_audio(audio_data)
                
                # Update metrics
                self._update_playback_metrics(chunk)
                
                # Emit event
                emit_event(Event(
                    type=EventType.AUDIO_OUTPUT_DATA,
                    source=self.name,
                    data={
                        "size": len(audio_data),
                        "duration": chunk.duration
                    }
                ))
                
            except Exception as e:
                emit_event(Event(
                    type=EventType.ERROR,
                    source=self.name,
                    data={"error": f"Playback error: {e}"}
                ))
        else:
            # Queue for later or handle with own implementation
            self._playback_queue.put(chunk)
    
    def stop_playback(self) -> None:
        """Stop current playback"""
        if self.voice_engine and hasattr(self.voice_engine, 'stop_audio'):
            try:
                self.voice_engine.stop_audio()
            except Exception:
                pass
        
        self._is_playing = False
        
        # Update state
        get_state_manager().update_audio(playback_state=PlaybackState.IDLE)
        
        emit_event(Event(
            type=EventType.AUDIO_OUTPUT_END,
            source=self.name
        ))
    
    def clear_queue(self) -> None:
        """Clear pending audio"""
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
    
    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 to 1.0)"""
        self._volume = max(0.0, min(1.0, volume))
        
        # Update state
        get_state_manager().update_audio(output_volume=self._volume)
        
        # If voice engine supports volume control
        if self.voice_engine and hasattr(self.voice_engine, 'set_output_volume'):
            try:
                self.voice_engine.set_output_volume(self._volume)
            except Exception:
                pass
    
    def set_muted(self, muted: bool) -> None:
        """Set mute state"""
        self._is_muted = muted
        
        if muted:
            self.stop_playback()
        
        emit_event(Event(
            type=EventType.MUTE_TOGGLE,
            source=self.name,
            data={"muted": muted, "type": "output"}
        ))
    
    def get_playback_position(self) -> float:
        """Get current playback position in seconds"""
        if self._playback_start_time and self._is_playing:
            return time.time() - self._playback_start_time
        return 0.0
    
    def _estimate_duration(self, audio_data: bytes) -> float:
        """Estimate audio duration from data size"""
        # Assume 16-bit, mono, 24kHz audio
        # 2 bytes per sample * 24000 samples per second
        bytes_per_second = 2 * 24000
        return len(audio_data) / bytes_per_second
    
    def _apply_volume(self, audio_data: bytes) -> bytes:
        """Apply volume adjustment to audio data"""
        if self._volume == 1.0:
            return audio_data
        
        # Simple volume adjustment for PCM audio
        # This is a basic implementation - real implementation
        # should handle different audio formats properly
        try:
            import numpy as np
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply volume
            audio_array = (audio_array * self._volume).astype(np.int16)
            
            return audio_array.tobytes()
        except Exception:
            # If numpy not available or error, return original
            return audio_data
    
    def _update_playback_metrics(self, chunk: AudioChunk):
        """Update playback metrics"""
        if not self._is_playing:
            self._is_playing = True
            self._playback_start_time = time.time()
            
            # Update state
            get_state_manager().update_audio(playback_state=PlaybackState.PLAYING)
            
            # Emit start event
            emit_event(Event(
                type=EventType.AUDIO_OUTPUT_START,
                source=self.name
            ))
            
            # Call callback
            if self._playback_started_callback:
                threading.Thread(
                    target=self._playback_started_callback,
                    daemon=True
                ).start()
        
        self.chunks_played += 1
        self.total_duration += chunk.duration
    
    def on_playback_complete(self):
        """Called when playback completes"""
        self._is_playing = False
        
        # Update state
        get_state_manager().update_audio(playback_state=PlaybackState.IDLE)
        
        # Emit end event
        emit_event(Event(
            type=EventType.AUDIO_OUTPUT_END,
            source=self.name
        ))
        
        # Call callback
        if self._playback_ended_callback:
            threading.Thread(
                target=self._playback_ended_callback,
                daemon=True
            ).start()
    
    # Callbacks
    def on_playback_started(self, callback: Callable):
        """Set callback for playback started"""
        self._playback_started_callback = callback
    
    def on_playback_ended(self, callback: Callable):
        """Set callback for playback ended"""
        self._playback_ended_callback = callback
    
    # Properties
    @property
    def is_playing(self) -> bool:
        """Check if audio is playing"""
        return self._is_playing
    
    @property
    def is_muted(self) -> bool:
        """Check if muted"""
        return self._is_muted
    
    @property
    def volume(self) -> float:
        """Get current volume"""
        return self._volume
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audio output metrics"""
        avg_chunk_duration = (
            self.total_duration / self.chunks_played
            if self.chunks_played > 0
            else 0
        )
        
        return {
            "chunks_played": self.chunks_played,
            "total_duration": self.total_duration,
            "average_chunk_duration": avg_chunk_duration,
            "is_playing": self._is_playing,
            "is_muted": self._is_muted,
            "volume": self._volume,
            "queue_size": self._playback_queue.qsize()
        }