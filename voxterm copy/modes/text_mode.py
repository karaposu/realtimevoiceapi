"""
VoxTerm Text Mode

Text-only interaction without audio.
"""

from typing import Optional
import asyncio

from .base_mode import BaseMode, ModeType
from ..core.events import Event, EventType, emit_event
from ..input.keyboard import KeyEvent
from ..input.text_input import TextCommand


class TextMode(BaseMode):
    """
    Text-only interaction mode.
    
    User types messages and receives text responses.
    No audio input or output.
    """
    
    def __init__(self):
        super().__init__("Text", ModeType.TEXT)
        
        # State
        self._is_typing = False
        self._awaiting_response = False
    
    async def activate(self) -> None:
        """Activate text mode"""
        self.set_active(True)
        
        # Ensure text input is active
        if self.text_input_handler:
            self.text_input_handler.on_message(self._on_text_message)
            self.text_input_handler.on_command(self._on_text_command)
        
        # Disable audio (not async)
        if self.audio_input_manager:
            self.audio_input_manager.set_muted(True)
        
        self.emit_mode_event("activated", {})
    
    async def deactivate(self) -> None:
        """Deactivate text mode"""
        self.set_active(False)
        
        # Clear text handlers
        if self.text_input_handler:
            self.text_input_handler.on_message(None)
            self.text_input_handler.on_command(None)
        
        # Re-enable audio (not async)
        if self.audio_input_manager:
            self.audio_input_manager.set_muted(False)
        
        self.emit_mode_event("deactivated", {})
    
    def handle_key_event(self, event: KeyEvent) -> bool:
        """Handle keyboard events"""
        # In text mode, most input goes to text handler
        # We only handle mode switches and special keys
        
        if event.action == "press":
            # Mode switches
            if event.matches("ctrl+v"):  # Switch to voice
                self.emit_mode_event("switch_mode", {"target": "push_to_talk"})
                return True
            
            # Clear input
            elif event.matches("ctrl+u"):
                if self.text_input_handler:
                    self.text_input_handler.clear_input()
                return True
            
            # Cancel current operation
            elif event.matches("escape"):
                if self._awaiting_response:
                    asyncio.create_task(self._cancel_response())
                return True
        
        return False
    
    def _on_text_message(self, message: str):
        """Handle text message input"""
        if self._awaiting_response:
            # Already processing
            return
        
        # Send to voice engine
        asyncio.create_task(self._send_message(message))
    
    def _on_text_command(self, command: TextCommand):
        """Handle text command"""
        # Commands are handled by text input handler
        # We just track state
        self.emit_mode_event("command_executed", {
            "command": command.command
        })
    
    async def _send_message(self, message: str):
        """Send message to voice engine"""
        self._awaiting_response = True
        
        # Update conversation state
        from ..core.state import get_state_manager
        state_manager = get_state_manager()
        state_manager.add_message("user", message)
        
        # Send to voice engine
        if self.voice_engine and hasattr(self.voice_engine, 'send_text'):
            try:
                await self.voice_engine.send_text(message)
                
                self.emit_mode_event("message_sent", {
                    "message": message
                })
            except Exception as e:
                self._awaiting_response = False
                self.emit_mode_event("error", {
                    "error": str(e)
                })
    
    async def _cancel_response(self):
        """Cancel waiting for response"""
        if self.voice_engine and hasattr(self.voice_engine, 'interrupt'):
            await self.voice_engine.interrupt()
        
        self._awaiting_response = False
        
        self.emit_mode_event("response_cancelled", {})
    
    def on_response_received(self, response: str):
        """Called when response is received"""
        self._awaiting_response = False
        
        # Update conversation state
        from ..core.state import get_state_manager
        state_manager = get_state_manager()
        state_manager.add_message("assistant", response)
        
        self.emit_mode_event("response_received", {
            "response": response
        })
    
    def get_status_text(self) -> str:
        """Get status text"""
        if self._awaiting_response:
            return "Text Mode (waiting for response...)"
        else:
            return "Text Mode"
    
    def get_help_text(self) -> str:
        """Get help text"""
        return "Type message and press [ENTER] | [CTRL+V] Voice mode"
    
    def get_metrics(self) -> dict:
        """Get mode metrics"""
        return {
            "mode": "text",
            "awaiting_response": self._awaiting_response,
            "is_active": self.is_active
        }