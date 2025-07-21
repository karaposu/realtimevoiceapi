#!/usr/bin/env python3
"""
Test 07: Fast Lane Interactive Voice Chat with VoiceEngine

Clean implementation using VoiceEngine API with comprehensive logging.
Tests the complete voice chat experience with real OpenAI connection.

Features:
- Push-to-talk with SPACE key
- Interrupt support
- Comprehensive logging
- Clean API usage

python -m realtimevoiceapi.smoke_tests.test_10_fastlane_with_voice_engine

"""



import sys
import asyncio
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import platform
import json
from typing import Optional
import threading

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Import VoiceEngine
from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
from realtimevoiceapi.core.stream_protocol import StreamEventType, StreamEvent

# Create log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"voice_engine_test_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Try keyboard libraries
KEYBOARD_LIB = None
try:
    from pynput import keyboard as pynput_keyboard
    KEYBOARD_LIB = "pynput"
    logger.info("Using pynput for keyboard input")
except ImportError:
    logger.warning("pynput not found. Install with: pip install pynput")


class InteractiveVoiceChat:
    """Interactive voice chat using VoiceEngine"""
    
    def __init__(self, api_key: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create VoiceEngine with configuration
        self.config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            vad_enabled=True,
            vad_threshold=0.02,
            latency_mode="ultra_low",
            log_level="DEBUG"
        )
        
        self.engine = VoiceEngine(config=self.config)
        
        # State tracking
        self.is_recording = False
        self.is_holding_key = False
        self.should_quit = False
        self.interaction_count = 0
        
        # Response tracking
        self.current_user_transcript = ""
        self.current_ai_transcript = ""
        self.response_chunks = 0
        
        # Event loop reference (IMPORTANT!)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Setup callbacks
        self.engine.on_audio_response = self._on_audio_response
        self.engine.on_text_response = self._on_text_response
        self.engine.on_error = self._on_error
        self.engine.on_response_done = self._on_response_done
        
        self.logger.info("InteractiveVoiceChat initialized")
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop reference"""
        self.loop = loop
    
    async def initialize(self):
        """Initialize the voice engine"""
        self.logger.info("Initializing voice engine...")
        print("🎤 Initializing voice chat...")
        
        try:
            # Connect to OpenAI
            await self.engine.connect()
            print("✅ Connected to OpenAI Realtime API")
            
            # The engine handles all the complex setup internally
            self.logger.info("Voice engine ready")
            print("✅ Voice chat ready!")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            raise
    
    def start_recording_threadsafe(self):
        """Start recording from another thread"""
        if self.loop and not self.is_recording:
            asyncio.run_coroutine_threadsafe(
                self.start_recording(),
                self.loop
            )
    
    def stop_recording_threadsafe(self):
        """Stop recording from another thread"""
        if self.loop and self.is_recording:
            asyncio.run_coroutine_threadsafe(
                self.stop_recording(),
                self.loop
            )
    
    async def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return
        
        # Check if we need to interrupt ongoing response
        if self.engine.is_listening and self.response_chunks > 0:
            self.logger.info("Interrupting ongoing response")
            await self.engine.interrupt()
            await asyncio.sleep(0.1)
        
        self.interaction_count += 1
        self.logger.info(f"=== INTERACTION {self.interaction_count} START ===")
        
        # Reset state
        self.is_recording = True
        self.response_chunks = 0
        self.current_user_transcript = ""
        self.current_ai_transcript = ""
        
        # Start listening (engine handles all audio capture internally)
        await self.engine.start_listening()
        print("\n🔴 Recording... (release to stop)")
    
    async def stop_recording(self):
        """Stop recording and process"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.logger.info("Recording stopped")
        
        # Stop listening
        await self.engine.stop_listening()
        print("⏹️  Processing...")
        
        # The engine automatically handles:
        # - VAD processing
        # - Audio buffering  
        # - Sending to API
        # - Response handling
    
    def _on_audio_response(self, audio_data: bytes):
        """Handle audio response"""
        self.response_chunks += 1
        
        if self.response_chunks == 1:
            self.logger.info("First audio chunk received")
            print("🔊 Playing response...")
        
        # Audio is automatically played by the engine
    
    def _on_text_response(self, text: str):
        """Handle text response"""
        if not self.current_ai_transcript:
            print(f"\n🤖 AI: ", end="", flush=True)
        
        self.current_ai_transcript += text
        print(text, end="", flush=True)
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        self.logger.error(f"Error: {error}")
        print(f"\n❌ Error: {error}")
    
    def _on_response_done(self):
        """Handle response completion"""
        if self.current_ai_transcript:
            print()  # New line after text
        
        self.logger.info(f"Response complete: {self.response_chunks} audio chunks")
        print(f"✅ Done ({self.response_chunks} chunks)")
        
        self.logger.info(f"=== INTERACTION {self.interaction_count} END ===")
        print("\n🎯 Ready! (Hold SPACE to talk)")
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up...")
        print("\n👋 Shutting down...")
        
        await self.engine.disconnect()
        
        print("✅ Cleanup complete")
        self.logger.info("Cleanup complete")


async def run_with_keyboard(chat: InteractiveVoiceChat):
    """Run with keyboard control"""
    # Set the event loop reference
    chat.set_event_loop(asyncio.get_running_loop())
    
    if KEYBOARD_LIB == "pynput":
        print("\n🎮 Controls:")
        print("  🎤 Hold SPACE = Record")
        print("  🔴 Release SPACE = Send")
        print("  ❌ Press ESC = Quit")
        print("\n✨ Ready!\n")
        
        stop_event = asyncio.Event()
        
        def on_press(key):
            try:
                if key == pynput_keyboard.Key.space and not chat.is_holding_key:
                    chat.is_holding_key = True
                    # Use threadsafe method
                    chat.start_recording_threadsafe()
                elif key == pynput_keyboard.Key.esc:
                    chat.should_quit = True
                    stop_event.set()
                    return False
            except Exception as e:
                logger.error(f"Key press error: {e}")
        
        def on_release(key):
            try:
                if key == pynput_keyboard.Key.space and chat.is_holding_key:
                    chat.is_holding_key = False
                    # Use threadsafe method
                    chat.stop_recording_threadsafe()
            except Exception as e:
                logger.error(f"Key release error: {e}")
        
        # Start keyboard listener
        listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        listener.start()
        
        try:
            await stop_event.wait()
        finally:
            listener.stop()
    else:
        # Simple input mode
        print("\n📢 Commands:")
        print("  ENTER = Start/stop recording")
        print("  'quit' = Exit")
        print("\n🎯 Ready!\n")
        
        is_recording = False
        
        while not chat.should_quit:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "> "
                )
                
                if command.lower() == 'quit':
                    chat.should_quit = True
                    break
                else:
                    if not is_recording:
                        is_recording = True
                        await chat.start_recording()
                    else:
                        is_recording = False
                        await chat.stop_recording()
                
            except KeyboardInterrupt:
                chat.should_quit = True
                break


async def main_async():
    """Main async function"""
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("   Create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")
        return
    
    # Create chat instance
    chat = InteractiveVoiceChat(api_key)
    
    try:
        # Initialize
        await chat.initialize()
        
        # Get metrics to verify everything is working
        metrics = chat.engine.get_metrics()
        logger.info(f"Initial metrics: {metrics}")
        
        # Run interactive loop
        await run_with_keyboard(chat)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n⚡ Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Get final metrics
        try:
            final_metrics = chat.engine.get_metrics()
            usage = await chat.engine.get_usage()
            cost = await chat.engine.estimate_cost()
            
            logger.info(f"Final metrics: {final_metrics}")
            logger.info(f"Usage: {usage}")
            logger.info(f"Estimated cost: ${cost.total:.4f}")
            
            print(f"\n📊 Session Stats:")
            print(f"  Interactions: {chat.interaction_count}")
            print(f"  Audio seconds: {usage.audio_input_seconds:.1f}s in, {usage.audio_output_seconds:.1f}s out")
            print(f"  Estimated cost: ${cost.total:.4f}")
            
        except Exception as e:
            logger.error(f"Error getting final metrics: {e}")
        
        # Cleanup
        await chat.cleanup()


def main():
    """Main entry point"""
    print("🎙️  RealtimeVoiceAPI - Interactive Voice Chat (VoiceEngine)")
    print("=" * 60)
    print(f"📝 Log file: {log_file}")
    print("=" * 60)
    
    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Check for sounddevice
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ Audio ready ({len(devices)} devices found)")
        logger.info(f"Found {len(devices)} audio devices")
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        return
    except Exception as e:
        print(f"❌ Audio error: {e}")
        logger.error(f"Audio error: {e}")
        return
    
    # Note about macOS permissions
    if platform.system() == "Darwin":
        print("\n⚠️  macOS Users: If you see 'not trusted' warning:")
        print("   Go to System Settings > Privacy & Security > Accessibility")
        print("   Add Terminal/VS Code to allowed apps")
        print()
    
    # Run the async main
    try:
        asyncio.run(main_async())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
    finally:
        print(f"\n📝 Log saved to: {log_file}")


if __name__ == "__main__":
    main()