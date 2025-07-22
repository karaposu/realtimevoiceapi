"""
VoxTerm Session Manager - Simplified with Stream Protocol

Now uses StreamProtocol for cleaner stream management and timing.
"""

import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from .modes import create_mode
from .settings import TerminalSettings
from .stream_protocol import StreamProtocol, StreamConfig, StreamSession, StreamType, StreamEvent


class SessionState(Enum):
    """Session states"""
    IDLE = "idle"
    ACTIVE = "active"
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
    Simplified session manager using Stream Protocol
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
        
        # Create stream protocol with mode-specific config
        stream_config = self._get_stream_config(mode_name)
        self.protocol = StreamProtocol(engine, stream_config)
        self.session = StreamSession(self.protocol)
        
        # Setup event handlers
        self._setup_handlers()
        
        # Mode-specific handlers
        self._mode_handlers = {
            'TextMode': TextModeHandler(),
            'PushToTalkMode': PushToTalkModeHandler(),
            'AlwaysOnMode': AlwaysOnModeHandler(),
            'TurnBasedMode': TurnBasedModeHandler(),
        }
    
    def _get_stream_config(self, mode_name: str) -> StreamConfig:
        """Get mode-specific stream configuration"""
        config = StreamConfig(
            allow_interruption=True,
            audio_buffer_size=20,
            response_timeout=30.0
        )
        
        # Turn-based mode needs special handling to prevent feedback
        if mode_name in ["turn_based", "turns"]:
            config.auto_flush_on_silence = False  # Don't auto-process silence
            config.silence_threshold_ms = 1000    # Longer silence needed
        
        return config
        
    def _setup_handlers(self):
        """Setup stream event handlers"""
        # Text output
        self.protocol.on(StreamType.TEXT_OUT, self._handle_text_output)
        
        # Audio output (just for indication)
        self.protocol.on(StreamType.AUDIO_OUT, self._handle_audio_output)
        
        # Transcripts
        self.protocol.on(StreamType.TRANSCRIPT, self._handle_transcript)
        
        # Control events
        self.protocol.on(StreamType.CONTROL, self._handle_control)
        
        # Track if we're in a response
        self.response_started = False
        
        # For turn-based mode, we also need direct engine handlers
        if self.mode_name in ["turn_based", "turns"]:
            # Set up direct engine handlers for response tracking
            if hasattr(self.engine, 'on_response_text'):
                self.engine.on_response_text = self._handle_text_output_direct
            if hasattr(self.engine, 'on_response_done'):
                self.engine.on_response_done = self._handle_response_done_direct
        
    def _handle_text_output(self, event: StreamEvent):
        """Handle text output events"""
        if event.data and event.data.strip():
            if not self.response_started:
                print("\n🤖 AI: ", end="", flush=True)
                self.response_started = True
            print(event.data, end="", flush=True)
    
    def _handle_audio_output(self, event: StreamEvent):
        """Handle audio output events"""
        # Just indicate audio is playing
        if not self.response_started:
            print("\n🔊 ", end="", flush=True)
            self.response_started = True
    
    def _handle_transcript(self, event: StreamEvent):
        """Handle transcript events"""
        if event.data and event.data.strip():
            print(f"\n👤 You: {event.data}")
    
    def _handle_control(self, event: StreamEvent):
        """Handle control events"""
        action = event.data.get("action")
        
        if action == "response_done":
            if self.response_started:
                print()  # New line
                self.response_started = False
                self.metrics.messages_received += 1
        elif action == "interaction_started":
            self.metrics.messages_sent += 1
        elif action == "interrupted":
            print("\n[Interrupted]")
    
    def _handle_text_output_direct(self, text: str):
        """Direct handler for text output (turn-based mode)"""
        if text and text.strip():
            print(text, end="", flush=True)
    
    def _handle_response_done_direct(self):
        """Direct handler for response done (turn-based mode)"""
        print()  # New line
        self.metrics.messages_received += 1
    
    async def start(self):
        """Start the session"""
        if self.running:
            return
            
        self.running = True
        self.state = SessionState.ACTIVE
        
        # Start the protocol
        await self.protocol.start()
        
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
        
        # Stop any ongoing interaction
        await self.protocol.end_interaction()
        
        # Stop the mode
        if hasattr(self.mode, 'stop'):
            await self.mode.stop()
        
        # Cleanup handler
        if hasattr(self, 'handler'):
            await self.handler.cleanup(self)
        
        # Stop the protocol
        await self.protocol.stop()
        
        self.state = SessionState.IDLE
    
    async def run_interactive(self):
        """Run the interactive loop based on mode"""
        try:
            await self.handler.run(self)
        except Exception as e:
            self.state = SessionState.ERROR
            self.metrics.errors += 1
            raise


# Mode-specific handlers using Stream Protocol
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
                    # Use stream session for clean interaction
                    await session.session.text_turn(text)
                    
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
        print("   💡 Tip: Use headphones to prevent feedback loops!\n")
        
        # Start continuous listening
        await session.protocol.start_interaction("voice")
    
    async def run(self, session: SessionManager):
        """Run always-on loop"""
        while session.running:
            await asyncio.sleep(0.1)
            # VAD and response handling happens via stream events
    
    async def cleanup(self, session: SessionManager):
        """Cleanup always-on mode"""
        await session.protocol.end_interaction()


class TurnBasedModeHandler:
    """Handler for turn-based mode with proper audio isolation"""
    
    def __init__(self):
        self.response_complete = False
        self.interaction_active = False
        
        # Audio recording parameters (matching VoiceEngine defaults)
        self.sample_rate = 24000
        self.channels = 1
        self.dtype = None  # Will be set to np.int16
        
        # Import dependencies
        try:
            import sounddevice as sd
            import numpy as np
            self.sd = sd
            self.np = np
            self.dtype = np.int16
        except ImportError:
            print("⚠️  sounddevice not found. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice", "numpy"])
            import sounddevice as sd
            import numpy as np
            self.sd = sd
            self.np = np
            self.dtype = np.int16
    
    async def initialize(self, session: SessionManager):
        """Initialize turn-based mode"""
        print("🎯 Turn-Based Conversation")
        print("   Press [ENTER] to start recording")
        print("   Press [ENTER] again to send")
        print("   [Q] Return to menu")
        print("   💡 Using send_recorded_audio for proper turn-based mode")
        print("   💡 Tip: Use headphones for best experience\n")
        
        # Add handler to track when response is complete
        session.protocol.on(StreamType.CONTROL, self._on_control_event)
    
    def _on_control_event(self, event: StreamEvent):
        """Track control events"""
        action = event.data.get("action")
        if action == "response_done":
            self.response_complete = True
        elif action == "interaction_started":
            self.interaction_active = True
        elif action == "interaction_ended":
            self.interaction_active = False
    
    def record_audio(self) -> bytes:
        """Record audio and return PCM data"""
        print("🎤 Recording... Press ENTER when done")
        
        # Start recording in a list to accumulate chunks
        audio_chunks = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"⚠️  Recording status: {status}")
            audio_chunks.append(indata.copy())
        
        # Start recording
        stream = self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=callback
        )
        
        with stream:
            # Wait for Enter key
            input()
        
        # Combine all chunks
        if audio_chunks:
            audio_data = self.np.concatenate(audio_chunks, axis=0)
        else:
            audio_data = self.np.array([], dtype=self.dtype)
        
        # Convert to bytes (PCM16)
        return audio_data.tobytes()
    
    async def run(self, session: SessionManager):
        """Run turn-based interaction with proper turn isolation"""
        
        while session.running:
            try:
                loop = asyncio.get_event_loop()
                
                # Wait for user to start their turn
                cmd = await loop.run_in_executor(
                    None, input, "\nPress ENTER to speak (or 'q' to quit): "
                )
                
                if cmd.lower() in ['q', 'quit']:
                    break
                
                # Record audio
                audio_data = await loop.run_in_executor(None, self.record_audio)
                
                if len(audio_data) > 0:
                    print("📤 Processing...")
                    
                    # Reset response tracking
                    self.response_complete = False
                    print("🤖 AI: ", end="", flush=True)
                    
                    # Use send_recorded_audio for turn-based interaction
                    await session.engine.send_recorded_audio(audio_data)
                    
                    # Wait for the AI response to complete
                    timeout = 30.0
                    start_time = time.time()
                    
                    while not self.response_complete and (time.time() - start_time < timeout):
                        await asyncio.sleep(0.1)
                    
                    if not self.response_complete:
                        print("\n⚠️  Response timeout")
                    
                    # Give AI time to respond
                    await asyncio.sleep(0.5)
                else:
                    print("⚠️  No audio recorded")
                
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)
    
    async def cleanup(self, session: SessionManager):
        """Cleanup turn-based mode"""
        # Make sure any interaction is ended
        if self.interaction_active:
            await session.protocol.end_interaction()


# Factory function for easy creation
def create_session(engine: Any, mode: str, settings: Optional[TerminalSettings] = None) -> SessionManager:
    """Create a session manager for the specified mode"""
    return SessionManager(engine, mode, settings)