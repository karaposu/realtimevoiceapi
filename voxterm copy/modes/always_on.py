"""
VoxTerm Always-On Mode

Continuous listening with voice activity detection.
"""

import time
import asyncio
from typing import Optional
from enum import Enum

from .base_mode import BaseMode, ModeType
from ..core.events import Event, EventType, emit_event
from ..input.keyboard import KeyEvent


class ListeningState(Enum):
    """Always-on mode states"""
    WAITING = "waiting"          # Waiting for speech
    LISTENING = "listening"      # Detected speech, recording
    PROCESSING = "processing"    # Processing recorded speech
    RESPONDING = "responding"    # AI is responding


class AlwaysOnMode(BaseMode):
    """
    Always-on interaction mode.
    
    Continuously listens for voice input using VAD.
    Automatically detects speech and sends to AI.
    """
    
    def __init__(self):
        super().__init__("AlwaysOn", ModeType.ALWAYS_ON)
        
        # State
        self._listening_state = ListeningState.WAITING
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        
        # Configuration
        self._vad_threshold = 0.5
        self._speech_timeout = 1.5  # Stop after 1.5s of silence
        self._min_speech_duration = 0.3  # Minimum 300ms
        
        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def activate(self) -> None:
        """Activate always-on mode"""
        self.set_active(True)
        
        # Start continuous listening
        if self.audio_input_manager:
            await self.audio_input_manager.start_recording()
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_speech())
        
        # Set initial state
        self._listening_state = ListeningState.WAITING
        
        self.emit_mode_event("activated", {
            "vad_threshold": self._vad_threshold
        })
    
    async def deactivate(self) -> None:
        """Deactivate always-on mode"""
        self.set_active(False)
        
        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop listening
        if self.audio_input_manager:
            await self.audio_input_manager.stop_recording()
        
        self._listening_state = ListeningState.WAITING
        
        self.emit_mode_event("deactivated", {})
    
    def handle_key_event(self, event: KeyEvent) -> bool:
        """Handle keyboard events"""
        if event.action == "press":
            # Interrupt current response
            if event.matches("escape"):
                if self._listening_state == ListeningState.RESPONDING:
                    asyncio.create_task(self._interrupt_response())
                return True
            
            # Pause/resume listening
            elif event.matches("p"):
                asyncio.create_task(self._toggle_pause())
                return True
            
            # Mode switches
            elif event.matches("t"):
                self.emit_mode_event("switch_mode", {"target": "text"})
                return True
            elif event.matches("space"):
                self.emit_mode_event("switch_mode", {"target": "push_to_talk"})
                return True
        
        return False
    
    async def _monitor_speech(self):
        """Monitor for speech activity"""
        while self.is_active:
            try:
                # Get current audio level from voice engine
                if self.voice_engine and hasattr(self.voice_engine, 'get_audio_level'):
                    level = self.voice_engine.get_audio_level()
                    
                    # Check VAD state from voice engine
                    vad_active = False
                    if hasattr(self.voice_engine, 'is_speech_detected'):
                        vad_active = self.voice_engine.is_speech_detected()
                    else:
                        # Simple level-based detection
                        vad_active = level > self._vad_threshold
                    
                    await self._handle_vad_state(vad_active)
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.05)  # 50ms = 20Hz monitoring
                
            except Exception as e:
                self.emit_mode_event("error", {"error": str(e)})
                await asyncio.sleep(0.1)
    
    async def _handle_vad_state(self, speech_detected: bool):
        """Handle VAD state changes"""
        current_time = time.time()
        
        if self._listening_state == ListeningState.WAITING:
            if speech_detected:
                # Start of speech
                self._speech_start_time = current_time
                self._last_speech_time = current_time
                self._listening_state = ListeningState.LISTENING
                
                self.emit_mode_event("speech_detected", {})
                
                # Update UI state
                from ..core.state import get_state_manager
                get_state_manager().update_audio(recording_state="recording")
        
        elif self._listening_state == ListeningState.LISTENING:
            if speech_detected:
                # Continue speech
                self._last_speech_time = current_time
            else:
                # Check for end of speech
                silence_duration = current_time - self._last_speech_time
                
                if silence_duration > self._speech_timeout:
                    # End of speech detected
                    speech_duration = self._last_speech_time - self._speech_start_time
                    
                    if speech_duration >= self._min_speech_duration:
                        # Valid speech, process it
                        await self._process_speech(speech_duration)
                    else:
                        # Too short, ignore
                        self._listening_state = ListeningState.WAITING
                        
                        from ..core.state import get_state_manager
                        get_state_manager().update_audio(recording_state="idle")
                        
                        self.emit_mode_event("speech_ignored", {
                            "reason": "too_short",
                            "duration": speech_duration
                        })
    
    async def _process_speech(self, duration: float):
        """Process detected speech"""
        self._listening_state = ListeningState.PROCESSING
        
        # Update state
        from ..core.state import get_state_manager
        get_state_manager().update_audio(recording_state="processing")
        
        self.emit_mode_event("processing_speech", {
            "duration": duration
        })
        
        # In always-on mode, the voice engine should automatically
        # process the buffered audio when we signal end of speech
        if self.voice_engine and hasattr(self.voice_engine, 'process_buffered_audio'):
            await self.voice_engine.process_buffered_audio()
        
        # Move to responding state
        self._listening_state = ListeningState.RESPONDING
    
    async def _interrupt_response(self):
        """Interrupt current AI response"""
        if self.voice_engine and hasattr(self.voice_engine, 'interrupt'):
            await self.voice_engine.interrupt()
        
        # Return to waiting state
        self._listening_state = ListeningState.WAITING
        
        self.emit_mode_event("response_interrupted", {})
    
    async def _toggle_pause(self):
        """Toggle pause state"""
        if self._listening_state == ListeningState.WAITING:
            # Pause listening
            if self.audio_input_manager:
                await self.audio_input_manager.set_muted(True)
            
            self.emit_mode_event("paused", {})
        else:
            # Resume listening
            if self.audio_input_manager:
                await self.audio_input_manager.set_muted(False)
            
            self.emit_mode_event("resumed", {})
    
    def on_response_complete(self):
        """Called when AI response is complete"""
        self._listening_state = ListeningState.WAITING
        
        from ..core.state import get_state_manager
        get_state_manager().update_audio(recording_state="idle")
    
    def get_status_text(self) -> str:
        """Get status text"""
        state_text = {
            ListeningState.WAITING: "Listening...",
            ListeningState.LISTENING: "Detecting speech...",
            ListeningState.PROCESSING: "Processing...",
            ListeningState.RESPONDING: "AI responding..."
        }
        
        return f"Always-On ({state_text.get(self._listening_state, 'Unknown')})"
    
    def get_help_text(self) -> str:
        """Get help text"""
        return "[P] Pause/Resume | [ESC] Interrupt | Speak naturally"
    
    def set_vad_threshold(self, threshold: float):
        """Set VAD threshold"""
        self._vad_threshold = max(0.0, min(1.0, threshold))
        
        self.emit_mode_event("vad_threshold_changed", {
            "threshold": self._vad_threshold
        })
    
    def get_metrics(self) -> dict:
        """Get mode metrics"""
        return {
            "mode": "always_on",
            "state": self._listening_state.value,
            "vad_threshold": self._vad_threshold,
            "speech_timeout": self._speech_timeout,
            "is_active": self.is_active
        }