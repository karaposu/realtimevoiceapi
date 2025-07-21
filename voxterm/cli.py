"""
VoxTerm CLI - Simple command-line interface for voice chat
"""

import asyncio
import sys
import time
from typing import Optional, Any
from .keyboard import SimpleKeyboard
from .modes import create_mode
from .logger import logger, setup_logging, patch_print, restore_print
from .settings import TerminalSettings


class VoxTermCLI:
    """Minimalist CLI for voice chat"""
    
    def __init__(self, voice_engine: Any, mode: str = "push_to_talk", settings: Optional[TerminalSettings] = None):
        self.engine = voice_engine
        self.mode_name = mode
        self.mode = create_mode(mode, voice_engine)
        self.settings = settings or TerminalSettings()
        
        if not self.mode:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Simple state
        self.running = False
        self.keyboard = None
        self.loop = None  # Will be set when running
        
        # Response tracking
        self.current_ai_response = ""
        self.response_in_progress = False
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """Setup engine callbacks"""
        # Save original callbacks if they exist
        self._orig_on_text = getattr(self.engine, 'on_text_response', None)
        self._orig_on_transcript = getattr(self.engine, 'on_transcript', None)
        self._orig_on_done = getattr(self.engine, 'on_response_done', None)
        self._orig_on_audio = getattr(self.engine, 'on_audio_response', None)
        
        # Set our callbacks
        if hasattr(self.engine, 'on_text_response'):
            self.engine.on_text_response = self._on_ai_text
        
        # VoiceEngine might use 'on_transcript' for AI responses too
        if hasattr(self.engine, 'on_transcript'):
            self.engine.on_transcript = self._on_transcript
        elif hasattr(self.engine, 'on_user_transcript'):
            self.engine.on_user_transcript = self._on_user_transcript
            
        if hasattr(self.engine, 'on_response_done'):
            self.engine.on_response_done = self._on_response_done
            
        # Track if audio is being played (to suppress text in voice mode)
        if hasattr(self.engine, 'on_audio_response'):
            self.engine.on_audio_response = self._on_audio_response
        
    def _on_ai_text(self, text: str):
        """Handle AI text response"""
        if text and text.strip():
            if not self.response_in_progress:
                # First text of a new response
                print("\n🤖 AI: ", end="", flush=True)
                self.response_in_progress = True
                if self.settings.log_to_file:
                    logger.log_api_event("Response started", "text_stream")
            print(text, end="", flush=True)
            self.current_ai_response += text
            
        # Call original if exists
        if self._orig_on_text:
            self._orig_on_text(text)
            
    def _on_user_transcript(self, text: str):
        """Handle user transcript"""
        if text.strip():
            print(f"\n👤 You: {text}")
            if self.settings.log_to_file:
                logger.log_user_action(f"Transcript: {text}")
            
        # Call original if exists
        if self._orig_on_transcript:
            self._orig_on_transcript(text)
            
    def _on_response_done(self):
        """Handle response completion"""
        if self.response_in_progress:
            print()  # New line after response
            if self.settings.log_to_file:
                logger.log_api_event("Response completed")
            self.current_ai_response = ""
            self.response_in_progress = False
            
            # Notify mode if it has handler
            if hasattr(self.mode, 'on_response_complete'):
                self.mode.on_response_complete()
        
        # Call original if exists
        if self._orig_on_done:
            self._orig_on_done()
            
    async def run(self):
        """Main CLI loop"""
        self.running = True
        
        # Setup logging if enabled
        if self.settings.log_to_file:
            setup_logging(self.settings)
            patch_print()
            logger.log_event("SESSION", "VoxTerm CLI started")
            logger.log_event("CONFIG", f"Mode: {self.mode_name}, Voice: {getattr(self.engine.config, 'voice', 'default')}")
        
        # Header
        print("\n🎙️  VoxTerm Voice Chat")
        print("=" * 40)
        print(f"Mode: {self.mode_name}")
        print(f"Voice: {getattr(self.engine.config, 'voice', 'default')}")
        print("=" * 40)
        
        try:
            # Connect engine
            print("Connecting...", end="", flush=True)
            if self.settings.log_to_file:
                logger.log_api_event("Connecting to API")
            await self.engine.connect()
            print(" ✅")
            if self.settings.log_to_file:
                logger.log_api_event("Connected successfully")
            
            # Setup keyboard only if needed
            needs_keyboard = self.mode_name not in ["text", "type", "turn_based", "turns"]
            keyboard_enabled = False
            
            if needs_keyboard:
                self.keyboard = SimpleKeyboard()
                keyboard_enabled = self._setup_keyboard()
                
                # Show macOS warning only if keyboard was enabled
                if keyboard_enabled and sys.platform == "darwin":
                    print("\n⚠️  macOS Note: Keyboard monitoring requires accessibility permissions")
                    print("   If keys don't work, grant Terminal access in System Settings > Privacy")
                    print("   Or use manual commands instead\n")
            
            # Start mode if needed
            if hasattr(self.mode, 'start'):
                await self.mode.start()
            
            # Show help
            print(f"\n📋 {self.mode.get_help()}")
            if needs_keyboard:
                print("    [M] Mute | [H] Help | [Q] Quit")
            print("\n🎯 Ready!\n")
            
            # Run based on mode
            if self.mode_name in ["text", "type"]:
                await self._run_text_mode()
            elif self.mode_name in ["turn_based", "turns"]:
                await self._run_turn_based_mode()
            else:
                await self._run_voice_mode()
                
        except KeyboardInterrupt:
            print("\n\n⚡ Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            await self.cleanup()
            
    def _setup_keyboard(self):
        """Setup keyboard handlers"""
        # Determine if this mode needs keyboard monitoring
        needs_keyboard = self.mode_name not in ["text", "type", "turn_based", "turns"]
        
        if not needs_keyboard:
            return False
            
        # Get the event loop for async calls
        self.loop = asyncio.get_event_loop()
        
        # Common keys
        self.keyboard.on_key('q', self._quit)
        self.keyboard.on_key('h', self._show_help)
        self.keyboard.on_key('m', self._toggle_mute)
        
        # Mode-specific keys (using async wrappers)
        if hasattr(self.mode, 'on_key_down'):
            def on_space_down():
                asyncio.run_coroutine_threadsafe(
                    self.mode.on_key_down("space"), 
                    self.loop
                )
            def on_space_up():
                asyncio.run_coroutine_threadsafe(
                    self.mode.on_key_up("space"),
                    self.loop
                )
                
            self.keyboard.on_space(on_space_down, on_space_up)
            
            # Other keys
            for key in ['p', 'r', 't']:
                def make_handlers(k):
                    return (
                        lambda: asyncio.run_coroutine_threadsafe(
                            self.mode.on_key_down(k), self.loop
                        ),
                        lambda: asyncio.run_coroutine_threadsafe(
                            self.mode.on_key_up(k), self.loop
                        )
                    )
                down, up = make_handlers(key)
                self.keyboard.handler.on_press(key, down)
                self.keyboard.handler.on_release(key, up)
        
        self.keyboard.start()
        return True
        
    async def _run_voice_mode(self):
        """Run voice interaction mode"""
        # Check if we should fall back to manual mode
        if (sys.platform == "darwin" and 
            self.mode_name in ["push_to_talk", "ptt"] and 
            self.keyboard is not None):
            print("💡 If keyboard doesn't work, type commands instead:")
            print("   'r' = record, 's' = stop, 'q' = quit\n")
            
            # Create a task for manual input
            manual_task = asyncio.create_task(self._manual_input_handler())
        
        try:
            while self.running:
                await asyncio.sleep(0.1)
        finally:
            # Cancel manual task if it exists
            if 'manual_task' in locals():
                manual_task.cancel()
                try:
                    await manual_task
                except asyncio.CancelledError:
                    pass
                    
    async def _manual_input_handler(self):
        """Handle manual text input in parallel with keyboard"""
        while self.running:
            try:
                # Get input in executor to not block
                loop = asyncio.get_event_loop()
                cmd = await loop.run_in_executor(None, input, "> ")
                
                cmd = cmd.lower().strip()
                
                if cmd in ['q', 'quit', 'exit']:
                    self.running = False
                    break
                elif cmd in ['r', 'record', 'start']:
                    print("🔴 Recording... (type 's' to stop)")
                    if self.settings.log_to_file:
                        logger.log_user_action("Started recording")
                    await self.mode.on_key_down("space")
                elif cmd in ['s', 'stop', 'send']:
                    print("📤 Sending...")
                    if self.settings.log_to_file:
                        logger.log_user_action("Stopped recording and sending")
                    await self.mode.on_key_up("space")
                elif cmd in ['m', 'mute']:
                    self._toggle_mute()
                elif cmd in ['h', 'help']:
                    print("\nCommands:")
                    print("  r/record - Start recording")
                    print("  s/stop   - Stop recording and send")
                    print("  m/mute   - Toggle mute")
                    print("  q/quit   - Exit")
                    print("  h/help   - Show this help\n")
                    
            except EOFError:
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Input error: {e}")
            
    async def _run_text_mode(self):
        """Run text interaction mode"""
        print("💬 Type your messages:\n")
        
        # Check if engine has send_text method
        if not hasattr(self.engine, 'send_text'):
            print("❌ Error: Voice engine doesn't support text input!")
            print("   This engine might be voice-only.")
            return
        
        while self.running:
            try:
                # Get input in executor to not block
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None, 
                    input, 
                    "You: "
                )
                
                if text.lower() in ['quit', 'exit', 'q']:
                    self.running = False
                    break
                
                if text.strip():  # Only send non-empty messages
                    # Reset response tracking
                    self.current_ai_response = ""
                    self.response_in_progress = False
                    
                    # Don't print "AI:" here - let the callback handle it
                    try:
                        if self.settings.log_to_file:
                            logger.log_user_action(f"Sent text: {text}")
                        await self.engine.send_text(text)
                    except Exception as e:
                        print(f"\n❌ Error sending text: {e}")
                        if self.settings.log_to_file:
                            logger.log_event("ERROR", f"Failed to send text: {e}")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
                
    async def _run_turn_based_mode(self):
        """Run turn-based interaction mode"""
        print("🎯 Turn-based mode - Press ENTER to take your turn\n")
        
        while self.running:
            try:
                # Wait for user to press enter
                loop = asyncio.get_event_loop()
                cmd = await loop.run_in_executor(
                    None, 
                    input, 
                    "Press ENTER to speak (or 'quit' to exit): "
                )
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    self.running = False
                    break
                
                # Start recording
                print("🎤 Your turn! Press ENTER when done...")
                
                # Debug: Check what methods are available
                if self.settings.log_level == LogLevel.DEBUG:
                    print(f"[DEBUG] Session methods: {[m for m in dir(self.session) if not m.startswith('_')]}")
                    print(f"[DEBUG] Mode type: {type(self.mode).__name__}")
                
                # For turn-based, trigger recording start
                if hasattr(self.mode, 'on_key_down'):
                    result = await self.mode.on_key_down("space")
                    if self.settings.log_level == LogLevel.DEBUG:
                        print(f"[DEBUG] on_key_down result: {result}")
                
                # Wait for enter
                await loop.run_in_executor(None, input, "")
                
                # Stop and send
                print("📤 Processing...")
                if hasattr(self.mode, 'on_key_up'):
                    result = await self.mode.on_key_up("space")
                    if self.settings.log_level == LogLevel.DEBUG:
                        print(f"[DEBUG] on_key_up result: {result}")
                
                # Give it a moment to process
                await asyncio.sleep(0.5)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    def _on_transcript(self, text: str):
        """Handle transcript (might be user or AI)"""
        # In text mode, we want to see AI transcripts
        if self.mode_name in ["text", "type"] and text.strip():
            # This might be AI response transcript
            if not self.current_ai_response:
                # Don't print prefix again if we already started
                pass
            print(text, end="", flush=True)
            self.current_ai_response += text
            self.response_in_progress = True
            
        # Call original if exists
        if self._orig_on_transcript:
            self._orig_on_transcript(text)
            
    def _on_audio_response(self, audio: bytes):
        """Handle audio response"""
        # Call original if exists
        if self._orig_on_audio:
            self._orig_on_audio(audio)
            
    def _quit(self):
        """Quit the application"""
        print("\n👋 Quitting...")
        self.running = False
        
    def _show_help(self):
        """Show help"""
        print(f"\n📋 {self.mode.get_help()}")
        print("    [M] Mute | [H] Help | [Q] Quit\n")
        
    def _toggle_mute(self):
        """Toggle mute (if supported)"""
        if hasattr(self.engine, 'toggle_mute'):
            muted = self.engine.toggle_mute()
            print(f"\n🔇 Mute: {'ON' if muted else 'OFF'}\n")
        else:
            print("\n⚠️  Mute not supported\n")
        """Quit the application"""
        print("\n👋 Quitting...")
        self.running = False
        
    def _show_help(self):
        """Show help"""
        print(f"\n📋 {self.mode.get_help()}")
        print("    [M] Mute | [H] Help | [Q] Quit\n")
        
    def _toggle_mute(self):
        """Toggle mute (if supported)"""
        if hasattr(self.engine, 'toggle_mute'):
            muted = self.engine.toggle_mute()
            print(f"\n🔇 Mute: {'ON' if muted else 'OFF'}\n")
        else:
            print("\n⚠️  Mute not supported\n")
            
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.keyboard:
            self.keyboard.stop()
            
        if hasattr(self.mode, 'stop'):
            await self.mode.stop()
            
        if self.engine:
            await self.engine.disconnect()
            
        print("✅ Cleanup complete")
        
        # Clean up logging
        if self.settings.log_to_file:
            logger.log_event("SESSION", "VoxTerm CLI ended")
            restore_print()
            logger.disable()