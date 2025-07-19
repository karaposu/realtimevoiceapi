"""
VoxTerm Audio Input Management

Provides audio input interface that delegates to voice engine.
VoxTerm doesn't process audio - it just manages the UI aspects.
"""

from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import threading
import time

from ..core.base import BaseComponent, ComponentState
from ..core.events import Event, EventType, emit_event, AudioInputEvent
from ..core.state import get_state_manager, RecordingState


class AudioInputSource(Enum):
    """Audio input sources"""
    VOICE_ENGINE = "voice_engine"  # Delegate to voice engine
    SYSTEM = "system"  # Direct system capture (optional)
    FILE = "file"  # File playback (testing)


@dataclass
class AudioDevice:
    """Audio device information"""
    id: str
    name: str
    channels: int
    is_default: bool = False


class AudioInputManager(BaseComponent):
    """
    Manages audio input for VoxTerm.
    
    Primary mode: Delegates to voice engine's audio handling
    Secondary mode: Can optionally manage its own audio (for standalone use)
    
    This class mainly handles:
    - UI state coordination
    - Recording indicators
    - Level monitoring (if provided by voice engine)
    """
    
    def __init__(self):
        super().__init__("AudioInputManager")
        
        # Configuration
        self.source = AudioInputSource.VOICE_ENGINE
        self.voice_engine = None
        
        # State
        self._is_recording = False
        self._is_muted = False
        self._current_level = 0.0
        self._lock = threading.Lock()
        
        # Callbacks
        self._recording_started_callback: Optional[Callable] = None
        self._recording_stopped_callback: Optional[Callable] = None
        self._audio_level_callback: Optional[Callable[[float], None]] = None
        
        # Metrics
        self.total_recordings = 0
        self.total_duration = 0.0
        self._recording_start_time: Optional[float] = None
    
    async def initialize(self) -> None:
        """Initialize audio input manager"""
        self.set_state(ComponentState.READY)
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={"message": "Audio input manager initialized"}
        ))
    
    async def start(self) -> None:
        """Start the audio input manager"""
        self.set_state(ComponentState.RUNNING)
    
    async def stop(self) -> None:
        """Stop the audio input manager"""
        if self._is_recording:
            await self.stop_recording()
        self.set_state(ComponentState.STOPPED)
    
    def bind_voice_engine(self, voice_engine: Any) -> None:
        """
        Bind to a voice engine for audio handling.
        
        The voice engine should have methods:
        - start_listening() -> None
        - stop_listening() -> None
        - get_audio_level() -> float (optional)
        """
        self.voice_engine = voice_engine
        self.source = AudioInputSource.VOICE_ENGINE
    
    async def start_recording(self) -> None:
        """
        Start audio recording.
        
        When using voice engine, this delegates to the engine.
        """
        with self._lock:
            if self._is_recording:
                return
            
            self._is_recording = True
            self._recording_start_time = time.time()
            self.total_recordings += 1
        
        # Update state
        state_manager = get_state_manager()
        state_manager.update_audio(recording_state=RecordingState.RECORDING)
        
        # Emit event
        emit_event(Event(
            type=EventType.AUDIO_INPUT_START,
            source=self.name,
            data={"source": self.source.value}
        ))
        
        # Delegate to voice engine if available
        if self.source == AudioInputSource.VOICE_ENGINE and self.voice_engine:
            try:
                if hasattr(self.voice_engine, 'start_listening'):
                    await self.voice_engine.start_listening()
                else:
                    # Sync method
                    self.voice_engine.start_listening()
            except Exception as e:
                self._handle_error(f"Failed to start recording: {e}")
                self._is_recording = False
                return
        
        # Call callback
        if self._recording_started_callback:
            self._recording_started_callback()
    
    async def stop_recording(self) -> None:
        """Stop audio recording"""
        with self._lock:
            if not self._is_recording:
                return
            
            self._is_recording = False
            
            # Calculate duration
            if self._recording_start_time:
                duration = time.time() - self._recording_start_time
                self.total_duration += duration
                self._recording_start_time = None
        
        # Update state
        state_manager = get_state_manager()
        state_manager.update_audio(recording_state=RecordingState.IDLE)
        
        # Emit event
        emit_event(Event(
            type=EventType.AUDIO_INPUT_END,
            source=self.name,
            data={"duration": duration if 'duration' in locals() else 0}
        ))
        
        # Delegate to voice engine if available
        if self.source == AudioInputSource.VOICE_ENGINE and self.voice_engine:
            try:
                if hasattr(self.voice_engine, 'stop_listening'):
                    await self.voice_engine.stop_listening()
                else:
                    # Sync method
                    self.voice_engine.stop_listening()
            except Exception as e:
                self._handle_error(f"Failed to stop recording: {e}")
        
        # Call callback
        if self._recording_stopped_callback:
            self._recording_stopped_callback()
    
    def toggle_recording(self) -> None:
        """Toggle recording state"""
        if self._is_recording:
            # Create async task for stop
            import asyncio
            asyncio.create_task(self.stop_recording())
        else:
            # Create async task for start
            import asyncio
            asyncio.create_task(self.start_recording())
    
    def set_muted(self, muted: bool) -> None:
        """Set mute state"""
        self._is_muted = muted
        
        # Update state
        state_manager = get_state_manager()
        state_manager.update_audio(is_muted=muted)
        
        # Emit event
        emit_event(Event(
            type=EventType.MUTE_TOGGLE,
            source=self.name,
            data={"muted": muted}
        ))
        
        # If we're recording while muting, we might want to pause
        # But we leave this decision to the voice engine
    
    def toggle_mute(self) -> None:
        """Toggle mute state"""
        self.set_muted(not self._is_muted)
    
    def get_audio_level(self) -> float:
        """
        Get current audio input level (0.0 to 1.0).
        
        Delegates to voice engine if available.
        """
        if self.source == AudioInputSource.VOICE_ENGINE and self.voice_engine:
            if hasattr(self.voice_engine, 'get_audio_level'):
                try:
                    level = self.voice_engine.get_audio_level()
                    self._current_level = level
                    
                    # Call callback if set
                    if self._audio_level_callback:
                        self._audio_level_callback(level)
                    
                    return level
                except Exception:
                    pass
        
        return self._current_level
    
    def list_devices(self) -> List[AudioDevice]:
        """
        List available audio input devices.
        
        Delegates to voice engine if available.
        """
        if self.source == AudioInputSource.VOICE_ENGINE and self.voice_engine:
            if hasattr(self.voice_engine, 'list_input_devices'):
                try:
                    devices = self.voice_engine.list_input_devices()
                    return [
                        AudioDevice(
                            id=str(d.get('id', i)),
                            name=d.get('name', f'Device {i}'),
                            channels=d.get('channels', 1),
                            is_default=d.get('is_default', False)
                        )
                        for i, d in enumerate(devices)
                    ]
                except Exception:
                    pass
        
        return []
    
    def set_device(self, device_id: str) -> bool:
        """
        Set audio input device.
        
        Delegates to voice engine if available.
        """
        if self.source == AudioInputSource.VOICE_ENGINE and self.voice_engine:
            if hasattr(self.voice_engine, 'set_input_device'):
                try:
                    self.voice_engine.set_input_device(device_id)
                    
                    # Update state
                    state_manager = get_state_manager()
                    state_manager.update_audio(input_device=device_id)
                    
                    return True
                except Exception as e:
                    self._handle_error(f"Failed to set device: {e}")
        
        return False
    
    # Callbacks
    def on_recording_started(self, callback: Callable) -> None:
        """Set callback for recording started"""
        self._recording_started_callback = callback
    
    def on_recording_stopped(self, callback: Callable) -> None:
        """Set callback for recording stopped"""
        self._recording_stopped_callback = callback
    
    def on_audio_level(self, callback: Callable[[float], None]) -> None:
        """Set callback for audio level updates"""
        self._audio_level_callback = callback
    
    # Properties
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._is_recording
    
    @property
    def is_muted(self) -> bool:
        """Check if muted"""
        return self._is_muted
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audio input metrics"""
        avg_duration = (
            self.total_duration / self.total_recordings
            if self.total_recordings > 0
            else 0
        )
        
        return {
            "total_recordings": self.total_recordings,
            "total_duration": self.total_duration,
            "average_duration": avg_duration,
            "current_level": self._current_level,
            "is_recording": self._is_recording,
            "is_muted": self._is_muted,
            "source": self.source.value
        }
    
    def _handle_error(self, error_msg: str):
        """Handle errors"""
        emit_event(Event(
            type=EventType.ERROR,
            source=self.name,
            data={"error": error_msg}
        ))