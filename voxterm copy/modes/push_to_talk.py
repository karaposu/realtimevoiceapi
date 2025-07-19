"""
VoxTerm Push-to-Talk Mode

Hold a key to record, release to send.
"""

import time
import asyncio
from typing import Optional

from .base_mode import BaseMode, ModeType
from ..core.events import Event, EventType, emit_event
from ..input.keyboard import KeyEvent


class PushToTalkMode(BaseMode):
    """
    Push-to-talk interaction mode.
    
    User holds a key (default: space) to record audio,
    releases to send to the AI.
    """
    
    def __init__(self):
        super().__init__("PushToTalk", ModeType.PUSH_TO_TALK)
        
        # State
        self._is_recording = False
        self._record_start_time: Optional[float] = None
        self._min_record_duration = 0.2  # Minimum 200ms recording
        
        # Key binding
        self._ptt_key = "space"  # Default
    
    async def activate(self) -> None:
        """Activate push-to-talk mode"""
        self.set_active(True)
        
        # Set up keyboard handlers
        if self.keyboard_manager:
            self.keyboard_manager.on_push_to_talk_start(self._on_ptt_start)
            self.keyboard_manager.on_push_to_talk_end(self._on_ptt_end)
        
        self.emit_mode_event("activated", {})
    
    async def deactivate(self) -> None:
        """Deactivate push-to-talk mode"""
        # Stop recording if active
        if self._is_recording:
            await self._stop_recording()
        
        self.set_active(False)
        
        # Clear keyboard handlers
        if self.keyboard_manager:
            self.keyboard_manager.on_push_to_talk_start(None)
            self.keyboard_manager.on_push_to_talk_end(None)
        
        self.emit_mode_event("deactivated", {})
    
    def handle_key_event(self, event: KeyEvent) -> bool:
        """Handle keyboard events"""
        # In PTT mode, most keys are handled by keyboard manager
        # We only handle mode-specific shortcuts here
        
        # Check for mode switch keys
        if event.action == "press":
            if event.matches("t"):  # Switch to text mode
                self.emit_mode_event("switch_mode", {"target": "text"})
                return True
            elif event.matches("a"):  # Switch to always-on
                self.emit_mode_event("switch_mode", {"target": "always_on"})
                return True
        
        return False
    
    def _on_ptt_start(self):
        """Handle push-to-talk start"""
        if self._is_recording:
            return
        
        # Start recording
        asyncio.create_task(self._start_recording())
    
    def _on_ptt_end(self):
        """Handle push-to-talk end"""
        if not self._is_recording:
            return
        
        # Check minimum duration
        if self._record_start_time:
            duration = time.time() - self._record_start_time
            if duration < self._min_record_duration:
                # Too short, cancel
                asyncio.create_task(self._cancel_recording())
                return
        
        # Stop and send
        asyncio.create_task(self._stop_recording())
    
    async def _start_recording(self):
        """Start audio recording"""
        if self.audio_input_manager:
            self._is_recording = True
            self._record_start_time = time.time()
            
            await self.audio_input_manager.start_recording()
            
            # Update state
            from ..core.state import get_state_manager
            get_state_manager().update_audio(recording_state="recording")
            
            self.emit_mode_event("recording_started", {})
    
    async def _stop_recording(self):
        """Stop recording and process"""
        if self.audio_input_manager and self._is_recording:
            self._is_recording = False
            
            duration = time.time() - self._record_start_time
            
            await self.audio_input_manager.stop_recording()
            
            # Update state
            from ..core.state import get_state_manager
            get_state_manager().update_audio(recording_state="processing")
            
            self.emit_mode_event("recording_stopped", {
                "duration": duration
            })
            
            # The audio is sent by the voice engine automatically
            # We just need to trigger the response
            if self.voice_engine and hasattr(self.voice_engine, 'process_audio'):
                await self.voice_engine.process_audio()
    
    async def _cancel_recording(self):
        """Cancel recording without sending"""
        if self.audio_input_manager and self._is_recording:
            self._is_recording = False
            
            await self.audio_input_manager.stop_recording()
            
            # Clear any recorded audio
            if self.voice_engine and hasattr(self.voice_engine, 'clear_audio_buffer'):
                self.voice_engine.clear_audio_buffer()
            
            # Update state
            from ..core.state import get_state_manager
            get_state_manager().update_audio(recording_state="idle")
            
            self.emit_mode_event("recording_cancelled", {
                "reason": "too_short"
            })
    
    def get_status_text(self) -> str:
        """Get status text"""
        if self._is_recording:
            duration = time.time() - self._record_start_time
            return f"Recording... ({duration:.1f}s)"
        else:
            return "Push-to-Talk"
    
    def get_help_text(self) -> str:
        """Get help text"""
        return f"Hold [{self._ptt_key.upper()}] to record, release to send"
    
    def set_ptt_key(self, key: str):
        """Set the push-to-talk key"""
        self._ptt_key = key
        
        # Update keyboard manager binding
        if self.keyboard_manager:
            self.keyboard_manager.update_binding("push_to_talk", key)
    
    def get_metrics(self) -> dict:
        """Get mode metrics"""
        return {
            "mode": "push_to_talk",
            "is_recording": self._is_recording,
            "ptt_key": self._ptt_key,
            "min_duration": self._min_record_duration
        }