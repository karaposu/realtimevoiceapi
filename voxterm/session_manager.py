"""
VoxTerm Session Manager - Unified session handling abstraction

This abstraction solves several problems:
1. Consistent callback management across all modes
2. Proper state tracking for responses
3. Clean separation between menu navigation and actual interaction
4. Unified error handling and recovery
5. Mode-agnostic interface for the menu system
"""

import asyncio
from typing import Optional, Any, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .modes import create_mode
from .settings import TerminalSettings


class SessionState(Enum):
    """Session states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class SessionMetrics:
    """Track session metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        return time.time() - self.start_time


class SessionManager:
    """
    Unified session manager that handles all interaction modes consistently
    
    This abstraction:
    - Manages callbacks properly for all modes
    - Tracks response state to prevent double printing
    - Provides consistent interface regardless of mode
    - Handles errors gracefully
    - Manages mode lifecycle (start/stop)
    """
    
    def __init__(self, engine: Any, mode_name: str, settings: Optional[TerminalSettings] = None):
        self.engine = engine
        self.mode_name = mode_name
        self.settings = settings or TerminalSettings()
        
        # Create the mode
        self.mode = create_mode(mode_name, engine)
        if not self.mode:
            raise ValueError(f"Invalid mode: {mode_name}")
        
        # State tracking
        self.state = SessionState.IDLE
        self.running = False
        self.metrics = SessionMetrics()
        
        # Response tracking
        self.current_response = ""
        self.response_in_progress = False
        self.waiting_for_response = False
        
        # Callback management
        self._original_callbacks = {}
        self._callback_handlers = {
            'on_text_response': self._handle_text_response,
            'on_audio_response': self._handle_audio_response,
            'on_response_done': self._handle_response_done,
            'on_error': self._handle_error,
            'on_transcript': self._handle_transcript,
        }
        
        # Mode-specific handlers
        self._mode_handlers = {
            'TextMode': TextModeHandler(),
            'PushToTalkMode': PushToTalkModeHandler(),
            'AlwaysOnMode': AlwaysOnModeHandler(),
            'TurnBasedMode': TurnBasedModeHandler(),
        }
        
    async def start(self):
        """Start the session"""
        if self.running:
            return
            
        self.running = True
        self.state = SessionState.IDLE
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Start the mode if needed
        if hasattr(self.mode, 'start'):
            await self.mode.start()
        
        # Get the appropriate handler
        handler_name = self.mode.__class__.__name__
        self.handler = self._mode_handlers.get(handler_name)
        
        if not self.handler:
            raise ValueError(f"No handler for mode: {handler_name}")
        
        # Initialize the handler
        await self.handler.initialize(self)
    
    async def stop(self):
        """Stop the session"""
        if not self.running:
            return
            
        self.running = False
        
        # Stop the mode
        if hasattr(self.mode, 'stop'):
            await self.mode.stop()
        
        # Cleanup handler
        if hasattr(self, 'handler'):
            await self.handler.cleanup(self)
        
        # Restore callbacks
        self._restore_callbacks()
        
        self.state = SessionState.IDLE
    
    def _setup_callbacks(self):
        """Setup callbacks with proper management"""
        for callback_name, handler in self._callback_handlers.items():
            if hasattr(self.engine, callback_name):
                # Save original
                self._original_callbacks[callback_name] = getattr(self.engine, callback_name)
                # Set our handler
                setattr(self.engine, callback_name, handler)
    
    def _restore_callbacks(self):
        """Restore original callbacks"""
        for callback_name, original in self._original_callbacks.items():
            if hasattr(self.engine, callback_name):
                setattr(self.engine, callback_name, original)
        self._original_callbacks.clear()
    
    # Unified callback handlers
    def _handle_text_response(self, text: str):
        """Handle text responses consistently"""
        if not text or not text.strip():
            return
            
        # First text of a new response
        if not self.response_in_progress:
            self.response_in_progress = True
            self.state = SessionState.RESPONDING
            print("\n🤖 AI: ", end="", flush=True)
        
        print(text, end="", flush=True)
        self.current_response += text
    
    def _handle_audio_response(self, audio: bytes):
        """Handle audio responses"""
        if not self.response_in_progress:
            self.response_in_progress = True
            self.state = SessionState.RESPONDING
            # For voice modes, we might show a visual indicator
            if self.mode_name != "text":
                print("\n🔊 ", end="", flush=True)
    
    def _handle_response_done(self):
        """Handle response completion"""
        if self.response_in_progress:
            print()  # New line
            self.response_in_progress = False
            self.waiting_for_response = False
            self.state = SessionState.IDLE
            self.metrics.messages_received += 1
            
            # Notify mode if it needs to know
            if hasattr(self.mode, 'on_response_complete'):
                self.mode.on_response_complete()
                
            # For turn-based mode, print the turn indicator
            if self.mode_name == 'turn_based':
                print("\n✅ Your turn again")
    
    def _handle_error(self, error: Exception):
        """Handle errors consistently"""
        print(f"\n❌ Error: {error}")
        self.state = SessionState.ERROR
        self.response_in_progress = False
        self.waiting_for_response = False
        self.metrics.errors += 1
    
    def _handle_transcript(self, text: str):
        """Handle transcripts (user speech)"""
        if text and text.strip():
            print(f"\n👤 You: {text}")
    
    # Public interface for modes
    async def send_text(self, text: str):
        """Send text message"""
        if not text.strip():
            return
            
        self.state = SessionState.PROCESSING
        self.waiting_for_response = True
        self.metrics.messages_sent += 1
        
        try:
            await self.engine.send_text(text)
        except Exception as e:
            self._handle_error(e)
    
    async def start_listening(self):
        """Start listening for audio"""
        # Ensure we're not already listening
        if self.state == SessionState.LISTENING:
            print("⚠️  Already listening")
            return
            
        self.state = SessionState.LISTENING
        try:
            await self.engine.start_listening()
        except Exception as e:
            self.state = SessionState.ERROR
            raise
    
    async def stop_listening(self):
        """Stop listening and process audio"""
        # Ensure we're actually listening
        if self.state != SessionState.LISTENING:
            print("⚠️  Not currently listening")
            return
            
        self.state = SessionState.PROCESSING
        self.waiting_for_response = True
        self.metrics.messages_sent += 1
        
        try:
            await self.engine.stop_listening()
        except Exception as e:
            self.state = SessionState.ERROR
            # Still try to clean up state
            self.waiting_for_response = False
            raise
    
    async def run_interactive(self):
        """Run the interactive loop based on mode"""
        await self.handler.run(self)


# Mode-specific handlers that implement the interaction logic
class TextModeHandler:
    """Handler for text mode"""
    
    async def initialize(self, session: SessionManager):
        """Initialize text mode"""
        print("💬 Type your messages:")
        print("   [Q] Return to menu\n")
    
    async def run(self, session: SessionManager):
        """Run text interaction loop"""
        while session.running:
            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, input, "You: ")
                
                if text.lower() == 'q':
                    break
                
                if text.strip():
                    await session.send_text(text)
                    
                    # Wait for response to complete
                    while session.waiting_for_response:
                        await asyncio.sleep(0.1)
                        
            except (KeyboardInterrupt, EOFError):
                break
    
    async def cleanup(self, session: SessionManager):
        """Cleanup text mode"""
        pass


class PushToTalkModeHandler:
    """Handler for push-to-talk mode"""
    
    async def initialize(self, session: SessionManager):
        """Initialize PTT mode"""
        print("🎤 Push-to-Talk Mode")
        print("   Hold [SPACE] to record, release to send")
        print("   [Q] Return to menu\n")
    
    async def run(self, session: SessionManager):
        """Run PTT interaction loop"""
        # This would integrate with keyboard handling
        # For now, simplified version
        print("(Keyboard integration needed)")
        
        while session.running:
            await asyncio.sleep(0.1)
            # Real implementation would handle keyboard events
    
    async def cleanup(self, session: SessionManager):
        """Cleanup PTT mode"""
        pass


class AlwaysOnModeHandler:
    """Handler for always-on mode"""
    
    async def initialize(self, session: SessionManager):
        """Initialize always-on mode"""
        print("🎤 Always Listening")
        print("   Just speak naturally")
        print("   [P] Pause | [Q] Return to menu\n")
        
        # Start listening immediately
        await session.start_listening()
    
    async def run(self, session: SessionManager):
        """Run always-on loop"""
        while session.running:
            await asyncio.sleep(0.1)
            # VAD and response handling happens via callbacks
    
    async def cleanup(self, session: SessionManager):
        """Cleanup always-on mode"""
        if session.state == SessionState.LISTENING:
            await session.engine.stop_listening()


class TurnBasedModeHandler:
    """Handler for turn-based mode"""
    
    def __init__(self):
        self.waiting_for_user_turn = True
        self.ai_is_speaking = False
    
    async def initialize(self, session: SessionManager):
        """Initialize turn-based mode"""
        print("🎯 Turn-Based Conversation")
        print("   Press [ENTER] to start your turn")
        print("   [Q] Return to menu\n")
    
    async def run(self, session: SessionManager):
        """Run turn-based interaction"""
        while session.running:
            try:
                # Only listen when it's user's turn
                if self.waiting_for_user_turn and not self.ai_is_speaking:
                    loop = asyncio.get_event_loop()
                    
                    # Clear any pending input
                    cmd = await loop.run_in_executor(
                        None, input, "Press ENTER to speak (or Q to quit): "
                    )
                    
                    if cmd.lower() == 'q':
                        break
                    
                    print("🎤 Your turn! Press ENTER when done...")
                    
                    # Ensure clean state before starting
                    if session.state == SessionState.LISTENING:
                        print("⚠️  Cleaning up previous listening state...")
                        try:
                            await session.engine.stop_listening()
                        except:
                            pass
                        await asyncio.sleep(0.5)
                    
                    # Reset state
                    session.state = SessionState.IDLE
                    self.waiting_for_user_turn = False
                    
                    try:
                        # Start listening
                        await session.start_listening()
                        
                        # Wait for user to press enter to stop
                        await loop.run_in_executor(None, input, "")
                        
                        print("📤 Processing...")
                        await session.stop_listening()
                        
                    except Exception as e:
                        print(f"\n❌ Error during recording: {e}")
                        # Ensure we're not stuck in listening state
                        session.state = SessionState.IDLE
                        self.waiting_for_user_turn = True
                        continue
                    
                    # Mark that AI will be speaking
                    self.ai_is_speaking = True
                    
                    # Wait for AI response to complete
                    timeout = 30  # 30 second timeout
                    start_time = time.time()
                    
                    while (session.waiting_for_response or session.response_in_progress) and (time.time() - start_time < timeout):
                        await asyncio.sleep(0.1)
                    
                    if time.time() - start_time >= timeout:
                        print("\n⚠️  Response timeout")
                    
                    # Wait a bit more to ensure audio playback is done
                    await asyncio.sleep(1.5)
                    
                    # Now it's user's turn again
                    self.ai_is_speaking = False
                    self.waiting_for_user_turn = True
                    
                else:
                    # Wait while AI is speaking
                    await asyncio.sleep(0.1)
                    
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"\n❌ Turn error: {e}")
                # Reset to safe state
                self.waiting_for_user_turn = True
                self.ai_is_speaking = False
                session.state = SessionState.IDLE
    
    async def cleanup(self, session: SessionManager):
        """Cleanup turn-based mode"""
        if session.state == SessionState.LISTENING:
            try:
                await session.engine.stop_listening()
            except:
                pass
        session.state = SessionState.IDLE


# Factory function for easy creation
def create_session(engine: Any, mode: str, settings: Optional[TerminalSettings] = None) -> SessionManager:
    """Create a session manager for the specified mode"""
    return SessionManager(engine, mode, settings)