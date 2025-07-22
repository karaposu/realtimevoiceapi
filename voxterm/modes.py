"""
Simple mode handlers for VoxTerm

Updated to work with Stream Protocol - modes are now just simple
configuration objects since StreamProtocol handles the complexity.
"""

import asyncio
import time
from typing import Optional, Callable, Any


class PushToTalkMode:
    """Hold key to record, release to send"""
    
    def __init__(self, engine):
        self.engine = engine
        self.is_recording = False
        self.record_start_time = 0
        
    async def on_key_down(self, key: str):
        """Handle key press"""
        if key == "space" and not self.is_recording:
            self.is_recording = True
            self.record_start_time = time.time()
            # Recording will be started by the session manager
            
    async def on_key_up(self, key: str):
        """Handle key release"""
        if key == "space" and self.is_recording:
            self.is_recording = False
            duration = time.time() - self.record_start_time
            
            if duration < 0.2:  # Too short
                return False  # Signal to cancel
            else:
                return True  # Signal to send
                
    def get_help(self) -> str:
        return "Hold [SPACE] to talk, release to send"


class AlwaysOnMode:
    """Continuous listening with VAD"""
    
    def __init__(self, engine):
        self.engine = engine
        self.is_paused = False
        
    async def start(self):
        """Mode started - protocol will handle listening"""
        pass
        
    async def stop(self):
        """Mode stopped - protocol will handle cleanup"""
        pass
            
    async def on_key_down(self, key: str):
        """Handle key press"""
        if key == "p":  # Pause/resume
            self.is_paused = not self.is_paused
            return {"action": "pause" if self.is_paused else "resume"}
                
    async def on_key_up(self, key: str):
        """No action on key release in always-on mode"""
        pass
        
    def get_help(self) -> str:
        return "Always listening | [P] Pause/Resume | 🎧 Best with headphones"


class TextMode:
    """Type messages instead of speaking"""
    
    def __init__(self, engine):
        self.engine = engine
        
    async def on_text_input(self, text: str):
        """Handle typed message - protocol will send it"""
        return text.strip() if text.strip() else None
            
    async def on_key_down(self, key: str):
        """No special keys in text mode"""
        pass
        
    async def on_key_up(self, key: str):
        """No special keys in text mode"""
        pass
        
    def get_help(self) -> str:
        return "Type your message and press [ENTER]"


class TurnBasedMode:
    """Explicit turn-taking"""
    
    def __init__(self, engine):
        self.engine = engine
        self.is_my_turn = True
        self.is_recording = False
        self.audio_buffer = []
        
    async def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.audio_buffer = []
        if hasattr(self.engine, 'start_listening'):
            await self.engine.start_listening()
        
    async def stop_recording_and_send(self):
        """Stop recording and send the audio"""
        self.is_recording = False
        if hasattr(self.engine, 'stop_listening'):
            await self.engine.stop_listening()
        
        # Send recorded audio if we have the method
        if hasattr(self.engine, 'send_recorded_audio') and self.audio_buffer:
            complete_audio = b"".join(self.audio_buffer)
            await self.engine.send_recorded_audio(complete_audio)
            self.audio_buffer = []
            return True
        return False
        
    async def on_key_down(self, key: str):
        """Handle key press - start recording"""
        if key == "space" and not self.is_recording:
            await self.start_recording()
            
    async def on_key_up(self, key: str):
        """Handle key release - stop and send"""
        if key == "space" and self.is_recording:
            return await self.stop_recording_and_send()
        
    def on_response_complete(self):
        """Called when AI finishes responding"""
        self.is_my_turn = True
        
    def get_help(self) -> str:
        return "Press [ENTER] to take your turn"


def create_mode(mode_name: str, engine: Any) -> Optional[object]:
    """Factory function to create mode instances"""
    modes = {
        "push_to_talk": PushToTalkMode,
        "ptt": PushToTalkMode,
        "always_on": AlwaysOnMode,
        "continuous": AlwaysOnMode,
        "text": TextMode,
        "type": TextMode,
        "turn_based": TurnBasedMode,
        "turns": TurnBasedMode,
    }
    
    mode_class = modes.get(mode_name.lower())
    if mode_class:
        return mode_class(engine)
    return None