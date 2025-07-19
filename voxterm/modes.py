"""
Simple mode handlers for VoxTerm

Each mode is just a class with methods for handling input/output.
No complex state management or abstract classes.
"""

import asyncio
import time
from typing import Optional, Callable


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
            
            # Actually start listening on the engine
            try:
                await self.engine.start_listening()
            except Exception as e:
                print(f"\n❌ Failed to start recording: {e}")
                self.is_recording = False
                
    async def on_key_up(self, key: str):
        """Handle key release"""
        if key == "space" and self.is_recording:
            self.is_recording = False
            duration = time.time() - self.record_start_time
            
            if duration < 0.2:  # Too short
                print(" (too short, cancelled)")
                # Clear any audio that was captured
                try:
                    await self.engine.stop_listening()
                    # If engine has a method to clear buffer, use it
                    if hasattr(self.engine, 'clear_audio_buffer'):
                        await self.engine.clear_audio_buffer()
                except:
                    pass
            else:
                print(f" ({duration:.1f}s)", end="", flush=True)
                try:
                    await self.engine.stop_listening()
                except Exception as e:
                    print(f"\n❌ Failed to send: {e}")
                
    def get_help(self) -> str:
        return "Hold [SPACE] to talk, release to send"


class AlwaysOnMode:
    """Continuous listening with VAD"""
    
    def __init__(self, engine):
        self.engine = engine
        self.is_listening = False
        self.is_paused = False
        
    async def start(self):
        """Start continuous listening"""
        print("🎤 Always listening (VAD active)")
        try:
            await self.engine.start_listening()
            self.is_listening = True
        except Exception as e:
            print(f"❌ Failed to start listening: {e}")
        
    async def stop(self):
        """Stop listening"""
        if self.is_listening:
            try:
                await self.engine.stop_listening()
                self.is_listening = False
            except:
                pass
            
    async def on_key_down(self, key: str):
        """Handle key press"""
        if key == "p":  # Pause/resume
            self.is_paused = not self.is_paused
            if self.is_paused:
                print("\n⏸️  Paused")
                try:
                    await self.engine.stop_listening()
                except:
                    pass
            else:
                print("\n▶️  Resumed")
                try:
                    await self.engine.start_listening()
                except Exception as e:
                    print(f"❌ Failed to resume: {e}")
                
    async def on_key_up(self, key: str):
        """No action on key release in always-on mode"""
        pass
        
    def get_help(self) -> str:
        return "Always listening | [P] Pause/Resume"


class TextMode:
    """Type messages instead of speaking"""
    
    def __init__(self, engine):
        self.engine = engine
        self.current_input = ""
        
    async def on_text_input(self, text: str):
        """Handle typed message"""
        if text.strip():
            try:
                # Print AI prefix to show we're waiting for response
                print("🤖 AI: ", end="", flush=True)
                
                # Send text to the engine
                await self.engine.send_text(text)
            except Exception as e:
                print(f"\n❌ Failed to send: {e}")
            
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
        
    async def on_key_down(self, key: str):
        """Handle key press"""
        if key == "space":
            if self.is_my_turn and not self.is_recording:
                # Start turn
                self.is_recording = True
                print("\n🎤 Your turn (press SPACE again to finish)...", end="", flush=True)
                try:
                    await self.engine.start_listening()
                except Exception as e:
                    print(f"\n❌ Failed to start: {e}")
                    self.is_recording = False
            elif self.is_recording:
                # End turn
                self.is_recording = False
                self.is_my_turn = False
                print("\n📤 Sending... AI's turn now")
                try:
                    await self.engine.stop_listening()
                except Exception as e:
                    print(f"\n❌ Failed to send: {e}")
                    self.is_my_turn = True
                
    async def on_key_up(self, key: str):
        """No action on release in turn-based mode"""
        pass
        
    def on_response_complete(self):
        """Called when AI finishes responding"""
        self.is_my_turn = True
        print("\n✅ Your turn again")
        
    def get_help(self) -> str:
        if self.is_recording:
            return "Recording... Press [SPACE] to finish your turn"
        elif self.is_my_turn:
            return "Your turn - Press [SPACE] to speak"
        else:
            return "AI is responding..."


def create_mode(mode_name: str, engine) -> Optional[object]:
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