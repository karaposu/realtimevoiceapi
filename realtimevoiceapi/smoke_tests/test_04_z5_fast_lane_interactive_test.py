#!/usr/bin/env python3
"""
Test 04.5: Fast Lane Interactive Test - Push-to-Talk with 'T' key

Interactive test with push-to-talk functionality:
- HOLD 'T' to record
- RELEASE 'T' to send
- Press 'Q' to quit

Requirements:
- Valid OpenAI API key in .env file
- sounddevice for audio capture/playback
- pynput (better for macOS) or keyboard library

python -m realtimevoiceapi.smoke_tests.test_04_z5_fast_lane_interactive_test
"""

import sys
import asyncio
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import numpy as np
from typing import List, Optional
import threading
import queue
import platform

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try different keyboard libraries
KEYBOARD_LIB = None
try:
    from pynput import keyboard as pynput_keyboard
    KEYBOARD_LIB = "pynput"
    logger.info("Using pynput for keyboard input (recommended for macOS)")
except ImportError:
    try:
        import keyboard
        KEYBOARD_LIB = "keyboard" 
        logger.info("Using keyboard library")
    except ImportError:
        logger.warning("No keyboard library found. Install with: pip install pynput")

# Import our components
from realtimevoiceapi.fast_lane.direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from realtimevoiceapi.fast_lane.fast_vad_detector import FastVADDetector, VADState
from realtimevoiceapi.fast_lane.fast_stream_manager import FastStreamManager, FastStreamConfig
from realtimevoiceapi.audio_types import AudioConfig, VADConfig
from realtimevoiceapi.stream_protocol import StreamState
from realtimevoiceapi.message_protocol import MessageFactory

try:
    import sounddevice as sd
except ImportError:
    logger.error("sounddevice not found!")


class AudioOutputBuffer:
    """Buffer audio chunks and play them smoothly"""
    
    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer = []
        self.is_playing = False
        self.play_thread = None
        self.stop_flag = threading.Event()
        
    def add_chunk(self, audio_chunk: bytes):
        """Add audio chunk to buffer"""
        self.buffer.append(audio_chunk)
        
        # Start playing if not already
        if not self.is_playing:
            self.start_playback()
    
    def start_playback(self):
        """Start playback thread"""
        if self.is_playing:
            return
            
        self.is_playing = True
        self.stop_flag.clear()
        self.play_thread = threading.Thread(target=self._playback_loop)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def _playback_loop(self):
        """Playback loop that runs in separate thread"""
        try:
            while self.is_playing and not self.stop_flag.is_set():
                if len(self.buffer) > 2:  # Wait for a few chunks to buffer
                    # Combine several chunks
                    chunks_to_play = []
                    for _ in range(min(5, len(self.buffer))):
                        if self.buffer:
                            chunks_to_play.append(self.buffer.pop(0))
                    
                    if chunks_to_play:
                        combined_audio = b''.join(chunks_to_play)
                        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                        
                        try:
                            # Play with blocking to ensure smooth playback
                            sd.play(audio_array, self.sample_rate, blocking=True)
                        except Exception as e:
                            logger.error(f"Playback error: {e}")
                else:
                    time.sleep(0.05)  # Wait for more chunks
                    
                # Check if we're done
                if not self.buffer and self.is_playing:
                    time.sleep(0.5)  # Wait a bit for more chunks
                    if not self.buffer:  # Still empty, we're done
                        self.is_playing = False
        except Exception as e:
            logger.error(f"Playback thread error: {e}")
            self.is_playing = False
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.stop_flag.set()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        self.buffer.clear()


class EnhancedFastStreamManager(FastStreamManager):
    """Enhanced stream manager with better response handling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_done_callback = None
    
    def _handle_message(self, message: dict):
        """Handle incoming WebSocket message with logging"""
        msg_type = message.get("type", "")
        
        # Only log important messages
        if msg_type not in ["response.audio.delta", "response.audio_transcript.delta", 
                           "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"]:
            self.logger.debug(f"Received: {msg_type}")
        
        # Call parent handler
        super()._handle_message(message)
        
        # Handle response completion
        if msg_type == "response.done":
            if self.response_done_callback:
                self.response_done_callback()


class InteractiveVoiceChat:
    """Interactive voice chat using fast lane components"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.is_recording = False
        self.is_processing = False
        self.is_holding_key = False
        self.should_quit = False
        
        # Audio components
        self.audio_config = AudioConfig()
        self.capture = DirectAudioCapture(config=self.audio_config)
        self.audio_buffer = AudioOutputBuffer(
            sample_rate=self.audio_config.sample_rate,
            channels=self.audio_config.channels
        )
        
        # Stream manager
        self.stream_config = FastStreamConfig(
            websocket_url="wss://api.openai.com/v1/realtime",
            api_key=api_key,
            voice="alloy",
            send_immediately=True
        )
        self.stream_manager = EnhancedFastStreamManager(config=self.stream_config)
        
        # Audio buffers
        self.recording_buffer = []
        self.response_count = 0
        self.got_text_response = False
        
        # Async coordination
        self.audio_queue: Optional[asyncio.Queue] = None
        self.event_loop = None
        self.recording_task = None
        
    async def initialize(self):
        """Initialize the voice chat system"""
        print("🎤 Initializing voice chat...")
        
        self.event_loop = asyncio.get_event_loop()
        
        # Connect to OpenAI
        await self.stream_manager.start()
        print("✅ Connected to OpenAI Realtime API")
        
        # Set up callbacks
        self.stream_manager.set_audio_callback(self._on_audio_response)
        self.stream_manager.set_text_callback(self._on_text_response)
        self.stream_manager.set_error_callback(self._on_error)
        self.stream_manager.response_done_callback = self._on_response_done
        
        # Wait for session to be established
        await asyncio.sleep(0.5)
        
        # Update session configuration - DISABLE server VAD auto-response
        session_msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            instructions="You are a helpful voice assistant. Be concise and conversational.",
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": False  # Disable auto-response
            },
            temperature=0.8
        )
        await self.stream_manager.connection.send(session_msg)
        
        # Wait for session update
        await asyncio.sleep(0.5)
        
        print("✅ Voice chat ready!")
        
    async def start_recording_async(self):
        """Start recording audio (async)"""
        if self.is_recording or self.is_processing:
            return
        
        self.is_recording = True
        self.recording_buffer.clear()
        
        print("\n🔴 Recording... (release to send)")
        
        try:
            # Clear input buffer first
            clear_msg = MessageFactory.input_audio_buffer_clear()
            await self.stream_manager.connection.send(clear_msg)
            await asyncio.sleep(0.1)
            
            # Start audio capture
            self.audio_queue = await self.capture.start_async_capture()
            
            # Start processing loop
            self.recording_task = asyncio.create_task(self._process_audio_chunks())
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.is_recording = False
    
    def start_recording(self):
        """Start recording (sync wrapper)"""
        if self.event_loop and not self.is_recording:
            future = asyncio.run_coroutine_threadsafe(
                self.start_recording_async(),
                self.event_loop
            )
            # Don't wait for completion
    
    async def stop_recording_async(self):
        """Stop recording and send to AI (async)"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.capture.stop_capture()
        
        # Cancel recording task
        if self.recording_task:
            self.recording_task.cancel()
            try:
                await self.recording_task
            except asyncio.CancelledError:
                pass
        
        print("⏹️  Processing...")
        
        # Process recorded audio
        if self.recording_buffer:
            await self._send_audio_to_ai()
        else:
            print("⚠️  No audio recorded")
    
    def stop_recording(self):
        """Stop recording (sync wrapper)"""
        if self.event_loop and self.is_recording:
            future = asyncio.run_coroutine_threadsafe(
                self.stop_recording_async(),
                self.event_loop
            )
            # Don't wait for completion
    
    async def _process_audio_chunks(self):
        """Process audio chunks while recording"""
        try:
            while self.is_recording and self.audio_queue:
                try:
                    # Get audio chunk
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    self.recording_buffer.append(chunk)
                            
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    break
        except Exception as e:
            logger.error(f"Recording loop error: {e}")
    
    async def _send_audio_to_ai(self):
        """Send recorded audio to AI"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.response_count = 0
        self.got_text_response = False
        
        print("📤 Sending to AI...")
        
        try:
            # Combine all chunks
            full_audio = b''.join(self.recording_buffer)
            audio_duration_ms = len(full_audio) / (self.audio_config.sample_rate * 2) * 1000
            
            if audio_duration_ms < 100:
                print("⚠️  Audio too short")
                self.is_processing = False
                return
                
            print(f"   Duration: {audio_duration_ms:.0f}ms")
            
            # Send audio in reasonable chunks
            chunk_size = 24000  # 500ms chunks
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i:i+chunk_size]
                await self.stream_manager.send_audio(chunk)
                await asyncio.sleep(0.01)
            
            # Commit the audio buffer
            commit_msg = MessageFactory.input_audio_buffer_commit()
            await self.stream_manager.connection.send(commit_msg)
            
            # Create response
            response_msg = MessageFactory.response_create()
            await self.stream_manager.connection.send(response_msg)
            
            print("⏳ Waiting for response...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.is_processing = False
    
    def _on_audio_response(self, audio_chunk: bytes):
        """Handle audio response from AI"""
        self.response_count += 1
        
        # Log first chunk
        if self.response_count == 1:
            print("🔊 Playing response...")
        
        # Add to buffer for smooth playback
        self.audio_buffer.add_chunk(audio_chunk)
    
    def _on_text_response(self, text: str):
        """Handle text response from AI"""
        if text.strip():
            if not self.got_text_response:
                print(f"\n💬 ", end="", flush=True)
                self.got_text_response = True
            print(text, end="", flush=True)
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        print(f"\n❌ Error: {error}")
        self.is_processing = False
    
    def _on_response_done(self):
        """Handle response completion"""
        if self.got_text_response:
            print()  # New line after text
        
        if self.response_count > 0:
            print(f"✅ Done ({self.response_count} chunks)")
            
        self.is_processing = False
        print("\n🎯 Ready! (Hold SPACE to talk, or type 'quit')")
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n👋 Shutting down...")
        
        if self.is_recording:
            await self.stop_recording_async()
        
        self.audio_buffer.stop()
        
        if self.stream_manager.state == StreamState.ACTIVE:
            await self.stream_manager.stop()
        
        print("✅ Cleanup complete")


async def run_with_simple_input(chat):
    """Simple input mode - type commands"""
    print("\n📢 Commands:")
    print("  SPACE + ENTER = Start/stop recording")
    print("  'quit' = Exit")
    print("\n🎯 Ready!\n")
    
    is_recording = False
    
    while not chat.should_quit:
        try:
            # Non-blocking input check
            loop = asyncio.get_event_loop()
            command = await loop.run_in_executor(None, input, "> ")
            
            if command.lower() == 'quit':
                chat.should_quit = True
                break
            elif command == ' ' or command == '':
                if not is_recording and not chat.is_processing:
                    is_recording = True
                    await chat.start_recording_async()
                elif is_recording:
                    is_recording = False
                    await chat.stop_recording_async()
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.1)
            
        except KeyboardInterrupt:
            chat.should_quit = True
            break
        except Exception as e:
            logger.error(f"Input error: {e}")


async def run_with_pynput(chat):
    """Run with pynput keyboard library"""
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
                chat.start_recording()
            elif key == pynput_keyboard.Key.esc:
                stop_event.set()
                return False  # Stop listener
        except Exception as e:
            logger.error(f"Key press error: {e}")
    
    def on_release(key):
        try:
            if key == pynput_keyboard.Key.space and chat.is_holding_key:
                chat.is_holding_key = False
                chat.stop_recording()
        except Exception as e:
            logger.error(f"Key release error: {e}")
    
    # Start keyboard listener in background thread
    listener = pynput_keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()
    
    try:
        # Wait for quit signal
        await stop_event.wait()
    finally:
        listener.stop()


async def run_interactive_chat():
    """Run the interactive voice chat"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        return
    
    # Create chat instance
    chat = InteractiveVoiceChat(api_key)
    
    try:
        # Initialize
        await chat.initialize()
        
        # Choose input method based on available libraries
        if KEYBOARD_LIB == "pynput":
            await run_with_pynput(chat)
        else:
            # Simple fallback mode
            await run_with_simple_input(chat)
            
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Chat error")
    finally:
        # Cleanup
        await chat.cleanup()


def main():
    """Main entry point"""
    print("🎙️  RealtimeVoiceAPI - Interactive Voice Chat")
    print("=" * 60)
    
    # Check dependencies
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ Audio ready ({len(devices)} devices)")
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        return
    except Exception as e:
        print(f"❌ Audio error: {e}")
        return
    
    if not KEYBOARD_LIB:
        print("💡 Tip: Install pynput for better controls: pip install pynput")
    
    # Run the chat
    try:
        asyncio.run(run_interactive_chat())
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.exception("Fatal error")


if __name__ == "__main__":
    main()