#!/usr/bin/env python3
"""
Test 04.5:


Requirements:
- Valid OpenAI API key in .env file
- sounddevice for audio capture/playback
- keyboard or pynput for key detection

python -m realtimevoiceapi.smoke_tests.test_04_z5_fast_lane_real_test
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import keyboard library
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    logger.warning("keyboard library not found. Install with: pip install keyboard")

# Import our components
from realtimevoiceapi.fast_lane.direct_audio_capture import DirectAudioCapture, DirectAudioPlayer
from realtimevoiceapi.fast_lane.fast_vad_detector import FastVADDetector, VADState
from realtimevoiceapi.fast_lane.fast_stream_manager import FastStreamManager, FastStreamConfig
from realtimevoiceapi.core.audio_types import AudioConfig, VADConfig
from realtimevoiceapi.core.stream_protocol import StreamState
from realtimevoiceapi.core.message_protocol import MessageFactory

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
        self.play_thread.start()
    
    def _playback_loop(self):
        """Playback loop that runs in separate thread"""
        # Combine all chunks into one continuous stream
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
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.stop_flag.set()
        if self.play_thread:
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
        
        # Log non-audio messages for debugging
        if msg_type not in ["response.audio.delta", "response.audio_transcript.delta"]:
            self.logger.info(f"Received: {msg_type}")
        
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
        
        # Audio components
        self.audio_config = AudioConfig()
        self.capture = DirectAudioCapture(config=self.audio_config)
        self.audio_buffer = AudioOutputBuffer(
            sample_rate=self.audio_config.sample_rate,
            channels=self.audio_config.channels
        )
        
        # VAD for speech detection
        self.vad = FastVADDetector(
            config=VADConfig(
                energy_threshold=0.02,
                speech_start_ms=200,
                speech_end_ms=800
            )
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
        
        # Async queue for coordination
        self.audio_queue: Optional[asyncio.Queue] = None
        
    async def initialize(self):
        """Initialize the voice chat system"""
        print("🎤 Initializing voice chat...")
        
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
        
        # Update session configuration - DISABLE server VAD
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
        print("\n📢 Instructions:")
        print("  - Type 'talk' to record for 3 seconds")
        print("  - Type 'test' to send a test message")
        print("  - Type 'quit' to exit")
        print("\n🎯 Ready for commands...\n")
    
    async def start_recording(self):
        """Start recording audio"""
        if self.is_recording or self.is_processing:
            return
        
        self.is_recording = True
        self.recording_buffer.clear()
        self.vad.reset()
        
        print("🔴 Recording... (speak now)")
        
        # Clear input buffer first
        clear_msg = MessageFactory.input_audio_buffer_clear()
        await self.stream_manager.connection.send(clear_msg)
        await asyncio.sleep(0.1)
        
        # Start audio capture
        self.audio_queue = await self.capture.start_async_capture()
        
        # Start processing loop
        asyncio.create_task(self._process_audio_chunks())
    
    async def stop_recording(self):
        """Stop recording and send to AI"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.capture.stop_capture()
        
        print("⏹️  Stopped recording")
        
        # Process recorded audio
        if self.recording_buffer:
            await self._send_audio_to_ai()
        else:
            print("⚠️  No audio recorded")
    
    async def _process_audio_chunks(self):
        """Process audio chunks while recording"""
        speech_detected = False
        
        while self.is_recording and self.audio_queue:
            try:
                # Get audio chunk
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                self.recording_buffer.append(chunk)
                
                # Process through VAD
                vad_state = self.vad.process_chunk(chunk)
                
                if vad_state in [VADState.SPEECH_STARTING, VADState.SPEECH]:
                    speech_detected = True
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                break
    
    async def _send_audio_to_ai(self):
        """Send recorded audio to AI"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.response_count = 0
        self.got_text_response = False
        
        print("📤 Sending audio to AI...")
        
        try:
            # Combine all chunks
            full_audio = b''.join(self.recording_buffer)
            audio_duration_ms = len(full_audio) / (self.audio_config.sample_rate * 2) * 1000
            print(f"   Audio duration: {audio_duration_ms:.1f}ms")
            
            # Send audio in reasonable chunks
            chunk_size = 24000  # 500ms chunks
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i:i+chunk_size]
                await self.stream_manager.send_audio(chunk)
                await asyncio.sleep(0.01)  # Small delay between chunks
            
            # Commit the audio buffer
            commit_msg = MessageFactory.input_audio_buffer_commit()
            await self.stream_manager.connection.send(commit_msg)
            
            # Create response
            response_msg = MessageFactory.response_create()
            await self.stream_manager.connection.send(response_msg)
            
            print("⏳ Waiting for response...")
            
        except Exception as e:
            print(f"❌ Error sending audio: {e}")
            self.is_processing = False
    
    def _on_audio_response(self, audio_chunk: bytes):
        """Handle audio response from AI"""
        self.response_count += 1
        
        # Log first chunk
        if self.response_count == 1:
            print("🔊 Receiving audio response...")
        
        # Add to buffer for smooth playback
        self.audio_buffer.add_chunk(audio_chunk)
    
    def _on_text_response(self, text: str):
        """Handle text response from AI"""
        if text.strip():
            if not self.got_text_response:
                print(f"\n🤖 AI says: ", end="", flush=True)
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
            print(f"✅ Response complete ({self.response_count} audio chunks)")
            # Let audio finish playing
            time.sleep(1.0)
        elif not self.got_text_response:
            print("⚠️  No response received")
            
        self.is_processing = False
        print("\n🎯 Ready for next command...\n")
    
    async def send_test_message(self):
        """Send a test text message"""
        print("📤 Sending test message...")
        self.is_processing = True
        self.response_count = 0
        self.got_text_response = False
        
        await self.stream_manager.send_text("Hello! This is a test. Please respond with a brief greeting.")
        print("⏳ Waiting for response...")
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n👋 Shutting down...")
        
        if self.is_recording:
            await self.stop_recording()
        
        self.audio_buffer.stop()
        
        if self.stream_manager.state == StreamState.ACTIVE:
            await self.stream_manager.stop()
        
        print("✅ Cleanup complete")


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
        
        # Simple command loop
        while True:
            # Get input
            loop = asyncio.get_event_loop()
            command = await loop.run_in_executor(None, input, "Command: ")
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'talk':
                await chat.start_recording()
                await asyncio.sleep(3)  # Record for 3 seconds
                await chat.stop_recording()
                # Wait for response to complete
                while chat.is_processing:
                    await asyncio.sleep(0.1)
            elif command.lower() == 'test':
                await chat.send_test_message()
                # Wait for response
                while chat.is_processing:
                    await asyncio.sleep(0.1)
            elif command.lower() == 'help':
                print("\nCommands:")
                print("  talk - Record audio for 3 seconds")
                print("  test - Send a test text message")
                print("  quit - Exit")
                print("  help - Show this help\n")
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Chat error")
    finally:
        # Cleanup
        await chat.cleanup()


def main():
    """Main entry point"""
    print("🎙️  RealtimeVoiceAPI - Fast Lane Interactive Test")
    print("=" * 60)
    
    # Check dependencies
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ Audio system ready ({len(devices)} devices found)")
    except ImportError:
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        return
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        return
    
    # Run the chat
    asyncio.run(run_interactive_chat())


if __name__ == "__main__":
    main()