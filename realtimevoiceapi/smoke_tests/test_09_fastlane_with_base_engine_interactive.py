#!/usr/bin/env python3
"""
Test 09: Interactive Voice Chat using BaseEngine - Toggle Recording

Simplified version with toggle recording (press R to start/stop).
This eliminates push-to-talk complexity to help isolate the feedback issue.

Features:
- Press R once to start recording
- Press R again to stop recording and send
- Clear recording state indicators
- Prevents recording while AI is speaking


python -m realtimevoiceapi.smoke_tests.test_09_fastlane_with_base_engine_interactive
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
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Import BaseEngine and related components
from realtimevoiceapi.base_engine import BaseEngine
from realtimevoiceapi.strategies.base_strategy import EngineConfig
from realtimevoiceapi.core.stream_protocol import StreamEvent, StreamEventType
from realtimevoiceapi.core.audio_types import AudioBytes
from realtimevoiceapi.core.message_protocol import MessageFactory

# Create log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"bbase_engine_toggle_test_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
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


class ToggleRecordingVoiceChat:
    """Voice chat with toggle recording and interrupt support"""
    
    def __init__(self, api_key: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.was_interrupted = False
        # Create BaseEngine instance
        self.base_engine = BaseEngine(logger=self.logger)
        
        # State tracking
        self.is_recording = False
        self.should_quit = False
        self.interaction_count = 0
        self.is_ai_speaking = False
        self.is_processing = False
        
        # Audio tracking
        self.recorded_chunks: List[AudioBytes] = []
        self.audio_capture_task: Optional[asyncio.Task] = None
        
        # Response tracking
        self.response_chunks = 0
        self.current_ai_text = ""
        
        # Event loop reference
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        self.logger.info("ToggleRecordingVoiceChat initialized")
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop reference"""
        self.loop = loop
    
    async def initialize(self):
        """Initialize the base engine"""
        self.logger.info("Initializing BaseEngine...")
        print("🎤 Initializing voice chat...")
        
        try:
            # Create strategy
            self.base_engine.create_strategy("fast")
            print("✅ Created fast lane strategy")
            
            # Create engine configuration
            engine_config = EngineConfig(
                api_key=self.api_key,
                provider="openai",
                input_device=None,
                output_device=None,
                enable_vad=False,  # Disable VAD for this test
                enable_transcription=False,
                enable_functions=False,
                latency_mode="ultra_low",
                metadata={
                    "voice": "alloy",
                    "sample_rate": 24000,
                    "chunk_duration_ms": 100
                }
            )
            
            # Initialize strategy
            await self.base_engine.initialize_strategy(engine_config)
            print("✅ Initialized strategy")
            
            # Setup audio
            await self.base_engine.setup_fast_lane_audio(
                sample_rate=24000,
                chunk_duration_ms=100,
                input_device=None,
                output_device=None,
                vad_enabled=False,  # No VAD for this test
                vad_threshold=0.02,
                vad_speech_start_ms=100,
                vad_speech_end_ms=500
            )
            print("✅ Setup audio components")
            
            # Connect to API
            await self.base_engine.do_connect()
            print("✅ Connected to OpenAI Realtime API")
            
            # Setup event handlers
            self._setup_event_handlers()
            print("✅ Event handlers configured")
            
            self.logger.info("Initialization complete")
            print("✅ Voice chat ready!")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            raise
    
    def _setup_event_handlers(self):
        """Setup event handlers"""
        handlers = {
            StreamEventType.AUDIO_OUTPUT_CHUNK: self._on_audio_chunk,
            StreamEventType.TEXT_OUTPUT_CHUNK: self._on_text_chunk,
            StreamEventType.STREAM_ERROR: self._on_error,
            StreamEventType.STREAM_ENDED: self._on_stream_ended,
        }
        
        # Set handlers for AI speaking state
        if hasattr(self.base_engine._strategy, 'stream_manager'):
            # Track when response ends
            def on_response_done():
                self.is_ai_speaking = False
                self.logger.info("AI finished speaking")
                # Trigger stream ended event
                if StreamEventType.STREAM_ENDED in handlers:
                    event = StreamEvent(
                        type=StreamEventType.STREAM_ENDED,
                        stream_id="unknown",
                        timestamp=time.time(),
                        data={}
                    )
                    handlers[StreamEventType.STREAM_ENDED](event)
            
            # Set callback
            if hasattr(self.base_engine._strategy.stream_manager, 'set_response_done_callback'):
                self.base_engine._strategy.stream_manager.set_response_done_callback(on_response_done)
        
        self.base_engine.setup_event_handlers(handlers)
        self.logger.info("Event handlers setup complete")
    
    def _on_audio_chunk(self, event: StreamEvent):
        """Handle audio output chunks"""
        self.response_chunks += 1  
        if self.response_chunks == 1:
            self.is_ai_speaking = True
            self.is_processing = False  # Clear processing flag when AI starts speaking
            self.logger.info("First audio chunk received")
            print("\n🔊 Playing response...")
        
        
        
        # Play audio
        if event.data and "audio" in event.data:
            self.base_engine.play_audio(event.data["audio"])
    
    def _on_text_chunk(self, event: StreamEvent):
        """Handle text output chunks"""
        if event.data and "text" in event.data:
            text = event.data["text"]
            
            if not self.current_ai_text:
                print(f"\n🤖 AI: ", end="", flush=True)
            
            self.current_ai_text += text
            print(text, end="", flush=True)
    
    def _on_error(self, event: StreamEvent):
        """Handle errors"""
        if hasattr(event, 'data') and event.data:
            error = event.data.get("error", "Unknown error")
        else:
            error = str(event) if event else "Unknown error"
        
        self.logger.error(f"Stream error: {error}")
        print(f"\n❌ Error: {error}")
        self.is_processing = False
        self.is_ai_speaking = False
    
    def _on_stream_ended(self, event: StreamEvent):
        """Handle stream ended"""
        if not self.is_processing and self.response_chunks == 0:
             return
        
        if self.current_ai_text:
            print()  #
        
        if self.was_interrupted:
            self.logger.info(f"Response interrupted after {self.response_chunks} chunks")
            print(f"⚡ Response interrupted ({self.response_chunks} chunks)\n")
        else:
            self.logger.info(f"Response complete: {self.response_chunks} chunks")
            print(f"✅ Response complete ({self.response_chunks} chunks)\n")
        
     
        
        # Reset state
        self.response_chunks = 0
        self.current_ai_text = ""
        self.is_ai_speaking = False
        self.is_processing = False
        self.was_interrupted = False  # Reset the flag
        
        self._show_status()
    
    def _show_status(self):
        """Show current status"""
        if self.is_recording:
            print("🔴 RECORDING - Press R to stop")
        else:
            print("⚪ READY - Press R to record, Q to quit")
    
    def toggle_recording_threadsafe(self):
        """Toggle recording from another thread"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.toggle_recording(),
                self.loop
            )
    
    async def toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_recording:
            await self.stop_recording()
        else:
            await self.start_recording()
    
    async def interrupt_ai(self):
        """Interrupt AI if it's speaking"""
        if self.is_ai_speaking:
            self.logger.info("Interrupting AI...")
            print("\n⚡ Interrupting AI...")
            self.was_interrupted = True

            # Stop audio playback
            if self.base_engine._audio_manager and hasattr(self.base_engine._audio_manager, '_player'):
                # Stop player if it exists
                try:
                    self.base_engine._audio_manager._player.stop_playback()
                except:
                    pass
            
            # Send cancel message
            if hasattr(self.base_engine._strategy, 'stream_manager'):
                try:
                    cancel_msg = MessageFactory.response_cancel()
                    await self.base_engine._strategy.stream_manager.connection.send(cancel_msg)
                    self.logger.debug("Sent response.cancel")
                except Exception as e:
                    self.logger.error(f"Failed to send cancel: {e}")
            
            # Reset state
            self.is_ai_speaking = False
            self.is_processing = False
            self.response_chunks = 0
            self.current_ai_text = ""
            
            # Wait a bit for cancellation to take effect
            await asyncio.sleep(0.2)
    
    async def start_recording(self):
        """Start recording (with interrupt if needed)"""
        # If AI is speaking, interrupt it first
        if self.is_ai_speaking:
            await self.interrupt_ai()
        
        # Check if we're still processing
        if self.is_processing:
            print("⚠️  Still processing previous recording")
            return
        
        self.interaction_count += 1
        self.logger.info(f"=== INTERACTION {self.interaction_count} START ===")
        
        # Clear state
        self.recorded_chunks.clear()
        self.is_recording = True
        self.response_chunks = 0
        self.current_ai_text = ""
        
        print("\n🔴 RECORDING STARTED - Press R to stop")
        
        # Clear audio buffer
        if hasattr(self.base_engine._strategy, 'stream_manager'):
            clear_msg = MessageFactory.input_audio_buffer_clear()
            await self.base_engine._strategy.stream_manager.connection.send(clear_msg)
            self.logger.debug("Cleared audio buffer")
        
        # Start capture task
        self.audio_capture_task = asyncio.create_task(self._capture_audio())
    
    async def _capture_audio(self):
        """Capture audio while recording is active"""
        self.logger.info("Audio capture task started")
        
        try:
            # Start audio capture
            if not self.base_engine._audio_manager:
                self.logger.error("No audio manager")
                return
            
            # Start capture
            audio_queue = await self.base_engine._audio_manager.start_capture()
            self.logger.info("Audio capture started")
            
            chunk_count = 0
            while self.is_recording:
                try:
                    # Get chunk with timeout
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    self.recorded_chunks.append(chunk)
                    chunk_count += 1
                    
                    if chunk_count % 10 == 0:
                        self.logger.debug(f"Captured {chunk_count} chunks")
                        print(".", end="", flush=True)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Capture error: {e}")
                    break
            
        finally:
            # Stop capture
            await self.base_engine._audio_manager.stop_capture()
            self.logger.info(f"Audio capture stopped. Captured {len(self.recorded_chunks)} chunks")
    
    async def stop_recording(self):
        """Stop recording and send audio"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.is_processing = True
        self.logger.info("Recording stopped")
        
        print("\n⏹️  RECORDING STOPPED")
        
        # Wait for capture task to finish
        if self.audio_capture_task:
            try:
                await asyncio.wait_for(self.audio_capture_task, timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.warning("Capture task timeout")
                self.audio_capture_task.cancel()
        
        # Process recorded audio
        if self.recorded_chunks:
            print(f"📤 Sending {len(self.recorded_chunks)} chunks...")
            self.logger.info(f"Sending {len(self.recorded_chunks)} chunks")
            
            # Send all chunks
            for i, chunk in enumerate(self.recorded_chunks):
                await self.base_engine._strategy.send_audio(chunk)
                if i % 10 == 0:
                    await asyncio.sleep(0.01)  # Small delay every 10 chunks
            
            # Commit and create response
            if hasattr(self.base_engine._strategy, 'stream_manager'):
                manager = self.base_engine._strategy.stream_manager
                
                # Commit
                commit_msg = MessageFactory.input_audio_buffer_commit()
                await manager.connection.send(commit_msg)
                self.logger.debug("Committed audio buffer")
                
                # Create response
                response_msg = MessageFactory.response_create()
                await manager.connection.send(response_msg)
                self.logger.debug("Requested response")
                
                print("⏳ Waiting for response...")
        else:
            print("⚠️  No audio recorded")
            self.is_processing = False
            self._show_status()
        
        # Clear recorded chunks
        self.recorded_chunks.clear()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        print("\n👋 Shutting down...")
        
        # Stop recording if active
        if self.is_recording:
            self.is_recording = False
            if self.audio_capture_task:
                self.audio_capture_task.cancel()
        
        # Interrupt AI if speaking
        if self.is_ai_speaking:
            await self.interrupt_ai()
        
        # Cleanup base engine
        await self.base_engine.cleanup()
        
        print("✅ Cleanup complete")


async def run_with_keyboard(chat: ToggleRecordingVoiceChat):
    """Run with keyboard input"""
    chat.set_event_loop(asyncio.get_running_loop())
    
    if KEYBOARD_LIB == "pynput":
        print("\n🎮 Controls:")
        print("  R = Toggle recording (start/stop)")
        print("  R while AI speaks = Interrupt and record")
        print("  Q = Quit")
        print("\n✨ Ready!\n")
        
        chat._show_status()
        
        stop_event = asyncio.Event()
        
        def on_press(key):
            try:
                if hasattr(key, 'char'):
                    if key.char and key.char.lower() == 'r':
                        chat.toggle_recording_threadsafe()
                    elif key.char and key.char.lower() == 'q':
                        chat.should_quit = True
                        stop_event.set()
                        return False
            except Exception as e:
                logger.error(f"Key press error: {e}")
        
        listener = pynput_keyboard.Listener(on_press=on_press)
        listener.start()
        
        try:
            await stop_event.wait()
        finally:
            listener.stop()
    else:
        # Simple input mode
        print("\n📢 Commands:")
        print("  r = Toggle recording")
        print("  q = Quit")
        print("\n✨ Ready!\n")
        
        chat._show_status()
        
        while not chat.should_quit:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, ""
                )
                
                if command.lower() == 'q':
                    chat.should_quit = True
                    break
                elif command.lower() == 'r':
                    await chat.toggle_recording()
                
            except KeyboardInterrupt:
                chat.should_quit = True
                break


async def main_async():
    """Main async function"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        return
    
    chat = ToggleRecordingVoiceChat(api_key)
    
    try:
        await chat.initialize()
        
        # Show initial metrics
        metrics = chat.base_engine.get_metrics()
        logger.info(f"Initial metrics: {json.dumps(metrics, indent=2)}")
        
        await run_with_keyboard(chat)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Final metrics
        try:
            usage = await chat.base_engine.get_usage()
            cost = await chat.base_engine.estimate_cost()
            
            print(f"\n📊 Session Stats:")
            print(f"  Interactions: {chat.interaction_count}")
            print(f"  Audio: {usage.audio_input_seconds:.1f}s in, {usage.audio_output_seconds:.1f}s out")
            print(f"  Cost: ${cost.total:.4f}")
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
        
        await chat.cleanup()


def main():
    """Main entry point"""
    print("🎙️  RealtimeVoiceAPI - Toggle Recording Test (BaseEngine)")
    print("=" * 60)
    print(f"📝 Log file: {log_file}")
    print("=" * 60)
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Import json for metrics
    global json
    import json
    
    # Check sounddevice
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ Audio ready ({len(devices)} devices)")
    except ImportError:
        print("❌ sounddevice not installed")
        return
    
    try:
        asyncio.run(main_async())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        print(f"\n📝 Log saved to: {log_file}")


if __name__ == "__main__":
    main()