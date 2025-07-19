"""
VoxTerm Turn-Based Mode

Structured turn-taking conversation mode.
"""

import time
import asyncio
from typing import Optional
from enum import Enum

from .base_mode import BaseMode, ModeType
from ..core.events import Event, EventType, emit_event
from ..input.keyboard import KeyEvent


class TurnState(Enum):
    """Turn-based conversation states"""
    USER_TURN = "user_turn"
    AI_TURN = "ai_turn"
    WAITING = "waiting"


class TurnBasedMode(BaseMode):
    """
    Turn-based interaction mode.
    
    Enforces explicit turn-taking between user and AI.
    User must signal when they're done speaking.
    """
    
    def __init__(self):
        super().__init__("TurnBased", ModeType.TURN_BASED)
        
        # State
        self._turn_state = TurnState.WAITING
        self._recording = False
        self._turn_start_time: Optional[float] = None
        
        # Configuration
        self._auto_end_silence = 3.0  # Auto-end turn after 3s silence
        self._min_turn_duration = 0.5  # Minimum turn duration
    
    async def activate(self) -> None:
        """Activate turn-based mode"""
        self.set_active(True)
        
        # Start with user turn
        self._turn_state = TurnState.WAITING
        
        self.emit_mode_event("activated", {})
    
    async def deactivate(self) -> None:
        """Deactivate turn-based mode"""
        self.set_active(False)
        
        # Stop any active recording
        if self._recording:
            await self._end_turn()
        
        self._turn_state = TurnState.WAITING
        
        self.emit_mode_event("deactivated", {})
    
    def handle_key_event(self, event: KeyEvent) -> bool:
        """Handle keyboard events"""
        if event.action == "press":
            # Start/end turn
            if event.matches("space"):
                if self._turn_state == TurnState.WAITING:
                    asyncio.create_task(self._start_user_turn())
                elif self._turn_state == TurnState.USER_TURN:
                    asyncio.create_task(self._end_turn())
                return True
            
            # Interrupt AI turn
            elif event.matches("escape"):
                if self._turn_state == TurnState.AI_TURN:
                    asyncio.create_task(self._interrupt_ai())
                return True
            
            # Mode switches
            elif event.matches("t"):
                self.emit_mode_event("switch_mode", {"target": "text"})
                return True
        
        return False
    
    async def _start_user_turn(self):
        """Start user's turn"""
        if self._turn_state != TurnState.WAITING:
            return
        
        self._turn_state = TurnState.USER_TURN
        self._turn_start_time = time.time()
        self._recording = True
        
        # Start recording
        if self.audio_input_manager:
            await self.audio_input_manager.start_recording()
        
        # Update state
        from ..core.state import get_state_manager
        get_state_manager().update_state({
            'conversation.current_turn': 'user',
            'audio.recording_state': 'recording'
        })
        
        self.emit_mode_event("turn_started", {
            "turn": "user"
        })
    
    async def _end_turn(self):
        """End current turn"""
        if self._turn_state != TurnState.USER_TURN or not self._recording:
            return
        
        # Check minimum duration
        duration = time.time() - self._turn_start_time
        if duration < self._min_turn_duration:
            # Too short
            self.emit_mode_event("turn_too_short", {
                "duration": duration
            })
            return
        
        self._recording = False
        
        # Stop recording
        if self.audio_input_manager:
            await self.audio_input_manager.stop_recording()
        
        # Process audio
        self._turn_state = TurnState.AI_TURN
        
        # Update state
        from ..core.state import get_state_manager
        get_state_manager().update_state({
            'conversation.current_turn': 'assistant',
            'audio.recording_state': 'processing'
        })
        
        self.emit_mode_event("turn_ended", {
            "turn": "user",
            "duration": duration
        })
        
        # Send to voice engine
        if self.voice_engine and hasattr(self.voice_engine, 'process_turn'):
            await self.voice_engine.process_turn()
    
    async def _interrupt_ai(self):
        """Interrupt AI's turn"""
        if self._turn_state != TurnState.AI_TURN:
            return
        
        # Interrupt voice engine
        if self.voice_engine and hasattr(self.voice_engine, 'interrupt'):
            await self.voice_engine.interrupt()
        
        # Return to waiting
        self._turn_state = TurnState.WAITING
        
        # Update state
        from ..core.state import get_state_manager
        get_state_manager().update_state({
            'conversation.current_turn': 'user'
        })
        
        self.emit_mode_event("turn_interrupted", {
            "turn": "assistant"
        })
    
    def on_ai_response_complete(self):
        """Called when AI completes its response"""
        if self._turn_state == TurnState.AI_TURN:
            self._turn_state = TurnState.WAITING
            
            # Update state
            from ..core.state import get_state_manager
            get_state_manager().update_state({
                'conversation.current_turn': 'user'
            })
            
            self.emit_mode_event("turn_complete", {
                "turn": "assistant"
            })
    
    def get_status_text(self) -> str:
        """Get status text"""
        status = {
            TurnState.WAITING: "Your turn - Press [SPACE] to speak",
            TurnState.USER_TURN: "Speaking... Press [SPACE] to finish",
            TurnState.AI_TURN: "AI is responding..."
        }
        
        return f"Turn-Based: {status.get(self._turn_state, 'Unknown')}"
    
    def get_help_text(self) -> str:
        """Get help text"""
        if self._turn_state == TurnState.USER_TURN:
            return "[SPACE] End turn | [ESC] Cancel"
        elif self._turn_state == TurnState.AI_TURN:
            return "[ESC] Interrupt AI"
        else:
            return "[SPACE] Start speaking"
    
    def get_metrics(self) -> dict:
        """Get mode metrics"""
        return {
            "mode": "turn_based",
            "turn_state": self._turn_state.value,
            "recording": self._recording,
            "auto_end_silence": self._auto_end_silence
        }