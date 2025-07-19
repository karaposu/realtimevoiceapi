#!/usr/bin/env python3
"""
Test 04.5: Fast Lane Interactive Test with Comprehensive Logging

Enhanced version with detailed logging to debug issues:
- All WebSocket messages logged
- Audio buffer states tracked
- Timing information captured
- State transitions logged
- Performance metrics recorded

Logs are saved to: fast_lane_test_TIMESTAMP.log

python -m realtimevoiceapi.smoke_tests.test_04_z5_fast_lane_interactive_test_with_logs
"""

import sys
import asyncio
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import threading
import queue
import platform
import json
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Create log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Setup comprehensive logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"fast_lane_test_{timestamp}.log"

# Configure file handler with detailed format
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Console handler with simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting Fast Lane Interactive Test - Log file: {log_file}")

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
from realtimevoiceapi.core.audio_types import AudioConfig, VADConfig
from realtimevoiceapi.core.stream_protocol import StreamState
from realtimevoiceapi.core.message_protocol import MessageFactory, ClientMessageType
from realtimevoiceapi.connections.websocket_connection import WebSocketConnection

try:
    import sounddevice as sd
    logger.info(f"sounddevice version: {sd.__version__}")
except ImportError:
    logger.error("sounddevice not found!")


class LoggingAudioOutputBuffer:
    """Audio buffer with detailed logging and proper interruption support"""
    
    def __init__(self, sample_rate=24000, channels=1):
        self.logger = logging.getLogger(f"{__name__}.AudioBuffer")
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer = []
        self.is_playing = False
        self.play_thread = None
        self.stop_flag = threading.Event()
        self.chunks_received = 0
        self.chunks_played = 0
        self.total_bytes = 0
        
        self.logger.info(f"AudioBuffer initialized: {sample_rate}Hz, {channels}ch")
        
    def add_chunk(self, audio_chunk: bytes):
        """Add audio chunk to buffer"""
        chunk_size = len(audio_chunk)
        self.chunks_received += 1
        self.total_bytes += chunk_size
        
        self.buffer.append(audio_chunk)
        buffer_size = len(self.buffer)
        
        self.logger.debug(
            f"Audio chunk added: size={chunk_size}, "
            f"chunk_num={self.chunks_received}, "
            f"buffer_size={buffer_size}, "
            f"total_bytes={self.total_bytes}"
        )
        
        # Start playing if not already
        if not self.is_playing:
            self.logger.info("Starting audio playback")
            self.start_playback()
    
    def start_playback(self):
        """Start playback thread"""
        if self.is_playing:
            self.logger.warning("Playback already active")
            return
            
        self.is_playing = True
        self.stop_flag.clear()
        self.play_thread = threading.Thread(target=self._playback_loop, name="AudioPlayback")
        self.play_thread.daemon = True
        self.play_thread.start()
        self.logger.info("Playback thread started")
    
    def _playback_loop(self):
        """Playback loop that runs in separate thread"""
        self.logger.debug("Playback loop started")
        try:
            while self.is_playing and not self.stop_flag.is_set():
                # Check stop flag more frequently
                if self.stop_flag.is_set():
                    self.logger.info("Playback stopped by flag")
                    break
                    
                if len(self.buffer) > 2:  # Wait for a few chunks to buffer
                    # Combine several chunks
                    chunks_to_play = []
                    for _ in range(min(5, len(self.buffer))):
                        if self.buffer and not self.stop_flag.is_set():
                            chunks_to_play.append(self.buffer.pop(0))
                    
                    if chunks_to_play and not self.stop_flag.is_set():
                        combined_size = sum(len(c) for c in chunks_to_play)
                        self.logger.debug(f"Playing {len(chunks_to_play)} chunks, {combined_size} bytes")
                        
                        combined_audio = b''.join(chunks_to_play)
                        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                        
                        try:
                            # Play with blocking but check stop flag
                            if not self.stop_flag.is_set():
                                start_time = time.time()
                                sd.play(audio_array, self.sample_rate, blocking=True)
                                play_duration = time.time() - start_time
                                
                                self.chunks_played += len(chunks_to_play)
                                self.logger.debug(f"Played in {play_duration:.3f}s")
                            else:
                                self.logger.info("Playback interrupted")
                                break
                        except Exception as e:
                            self.logger.error(f"Playback error: {e}", exc_info=True)
                else:
                    if self.stop_flag.is_set():
                        break
                    time.sleep(0.05)  # Wait for more chunks
                    
                # Check if we're done
                if not self.buffer and self.is_playing and not self.stop_flag.is_set():
                    self.logger.debug("Buffer empty, waiting for more chunks...")
                    time.sleep(0.5)  # Wait a bit for more chunks
                    if not self.buffer:  # Still empty, we're done
                        self.logger.info("Playback complete")
                        self.is_playing = False
        except Exception as e:
            self.logger.error(f"Playback thread error: {e}", exc_info=True)
            self.is_playing = False
        finally:
            self.logger.info(f"Playback ended: played {self.chunks_played}/{self.chunks_received} chunks")
    
    def stop(self):
        """Stop playback immediately"""
        self.logger.info("Stopping playback - clearing buffer")
        self.is_playing = False
        self.stop_flag.set()
        
        # Stop any ongoing playback
        try:
            sd.stop()  # Stop sounddevice playback
            self.logger.debug("Stopped sounddevice playback")
        except Exception as e:
            self.logger.debug(f"Error stopping sounddevice: {e}")
        
        # Clear the buffer immediately
        buffer_size = len(self.buffer)
        self.buffer.clear()
        self.logger.info(f"Cleared {buffer_size} chunks from buffer")
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
            if self.play_thread.is_alive():
                self.logger.warning("Playback thread did not stop cleanly")


class LoggingWebSocketConnection(WebSocketConnection):
    """WebSocket connection with message logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msg_logger = logging.getLogger(f"{__name__}.WebSocket")
        self.message_count = {"sent": 0, "received": 0}
        self.message_log = []
        
    async def send(self, data: Any) -> None:
        """Send with logging"""
        self.message_count["sent"] += 1
        msg_num = self.message_count["sent"]
        
        # Log the message
        if isinstance(data, dict):
            msg_type = data.get("type", "unknown")
            
            # Don't log full audio data
            if msg_type == "input_audio_buffer.append" and "audio" in data:
                log_data = {**data, "audio": f"<base64 {len(data['audio'])} chars>"}
            else:
                log_data = data
                
            self.msg_logger.debug(f"SEND #{msg_num} | {msg_type} | {json.dumps(log_data, indent=2)}")
            
            # Track in memory
            self.message_log.append({
                "direction": "sent",
                "num": msg_num,
                "time": time.time(),
                "type": msg_type,
                "size": len(str(data))
            })
        
        # Send the actual message
        await super().send(data)
    
    def _handle_message(self, message: dict):
        """Enhanced message handling with logging"""
        msg_type = message.get("type", "")
        
        # Log non-audio messages
        if msg_type not in ["response.audio.delta", "response.audio_transcript.delta"]:
            self.logger.debug(f"Processing message: {msg_type}")
        
        # Track specific events
        if msg_type == "session.created":
            session_id = message.get("session", {}).get("id", "unknown")
            self.logger.info(f"Session created: {session_id}")
        elif msg_type == "response.created":
            response_id = message.get("response", {}).get("id", "unknown")
            self.logger.info(f"Response started: {response_id}")
            # Track response state - NEW
            if hasattr(self, '_audio_callback'):
                # Find the chat instance through callbacks
                for callback in [self._audio_callback, self._text_callback, self._error_callback]:
                    if callback and hasattr(callback, '__self__'):
                        chat = callback.__self__
                        chat.current_response_id = response_id
                        chat.response_done = False
                        break
        elif msg_type == "response.done":
            self.logger.info("Response completed")
            if self.response_done_callback:
                self.response_done_callback()
        elif msg_type == "error":
            error = message.get("error", {})
            self.logger.error(f"API Error: {json.dumps(error)}")
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # Track user transcript
            transcript = message.get("transcript", "")
            self.logger.info(f"User said: {transcript}")
            print(f"\n👤 You: {transcript}")
        elif msg_type == "response.audio_transcript.done":
            # Track AI transcript
            transcript = message.get("transcript", "")
            self.logger.info(f"AI said: {transcript}")
            print(f"\n🤖 AI: {transcript}")
        
        # Call parent handler
        super()._handle_message(message)

class LoggingFastStreamManager(FastStreamManager):
    """Stream manager with enhanced logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(f"{__name__}.StreamManager")
        self.response_done_callback = None
        self.state_transitions = []
        self._create_connection_override = True
        
    async def start(self) -> None:
        """Start with logging"""
        self.logger.info("Starting stream manager")
        old_state = self._state
        
        # Import what we need
        from realtimevoiceapi.connections.websocket_connection import ConnectionConfig
        from realtimevoiceapi.core.exceptions import StreamError
        
        # Create the connection with proper config
        if self._state != StreamState.IDLE:
            raise StreamError(f"Cannot start in state {self._state}")
        
        self._state = StreamState.STARTING
        self._start_time = time.time()
        
        try:
            # Create WebSocket config
            ws_config = ConnectionConfig(
                url=f"{self.config.websocket_url}?model=gpt-4o-realtime-preview",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                },
                enable_message_queue=False,
                enable_metrics=False,
                auto_reconnect=False
            )
            
            # Create logging connection
            self.connection = LoggingWebSocketConnection(
                config=ws_config,
                logger=self.logger,
                message_handler=self._handle_message
            )
            
            # Connect
            await self.connection.connect()
            
            # Send session config
            await self.connection.send(self._session_config)
            
            self._state = StreamState.ACTIVE
            
        except Exception as e:
            self._state = StreamState.ERROR
            if hasattr(self, '_error_callback') and self._error_callback:
                self._error_callback(e)
            raise
        
        self._log_state_transition(old_state, self._state)
        self.logger.info(f"Stream started: {self._stream_id}")
    
    def _log_state_transition(self, old_state, new_state):
        """Log state transitions"""
        transition = {
            "time": time.time(),
            "from": old_state.value if hasattr(old_state, 'value') else str(old_state),
            "to": new_state.value if hasattr(new_state, 'value') else str(new_state)
        }
        self.state_transitions.append(transition)
        self.logger.info(f"State transition: {transition['from']} -> {transition['to']}")
    
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio with logging"""
        self.logger.debug(f"Sending audio: {len(audio_data)} bytes")
        await super().send_audio(audio_data)
    
    def _handle_message(self, message: dict):
        """Enhanced message handling with logging"""
        msg_type = message.get("type", "")
        
        # Log non-audio messages
        if msg_type not in ["response.audio.delta", "response.audio_transcript.delta"]:
            self.logger.debug(f"Processing message: {msg_type}")
        
        # Track specific events
        if msg_type == "session.created":
            session_id = message.get("session", {}).get("id", "unknown")
            self.logger.info(f"Session created: {session_id}")
        elif msg_type == "response.created":
            response_id = message.get("response", {}).get("id", "unknown")
            self.logger.info(f"Response started: {response_id}")
        elif msg_type == "response.done":
            self.logger.info("Response completed")
            if self.response_done_callback:
                self.response_done_callback()
        elif msg_type == "error":
            error = message.get("error", {})
            self.logger.error(f"API Error: {json.dumps(error)}")
        
        # Call parent handler
        super()._handle_message(message)


class DebugInteractiveVoiceChat:
    """Interactive voice chat with comprehensive debugging"""
   
    
    def __init__(self, api_key: str):
        self.logger = logging.getLogger(f"{__name__}.VoiceChat")
        self.api_key = api_key
        self.is_recording = False
        self.is_processing = False
        self.is_holding_key = False
        self.should_quit = False
        
        # Timing tracking
        self.timings = {
            "recording_start": 0,
            "recording_end": 0,
            "send_start": 0,
            "first_response": 0,
            "response_end": 0
        }
        
        # Audio components
        self.audio_config = AudioConfig()
        self.capture = DirectAudioCapture(config=self.audio_config)
        self.audio_buffer = LoggingAudioOutputBuffer(
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
        self.stream_manager = LoggingFastStreamManager(config=self.stream_config)
        
        # Audio buffers
        self.recording_buffer = []
        self.response_count = 0
        self.got_text_response = False
        
        # Session tracking
        self.session_num = 0
        self.interaction_num = 0
        
        # Async coordination
        self.audio_queue: Optional[asyncio.Queue] = None
        self.event_loop = None
        self.recording_task = None
        
        # Response tracking - NEW
        self.current_response_id = None
        self.response_done = True
        self.current_response_audio = bytearray()
        
        # Transcript tracking - NEW
        self.current_user_transcript = ""
        self.current_ai_transcript = ""
        
        self.logger.info("VoiceChat initialized")

    async def start_recording_async(self):
        """Start recording audio (async)"""
        if self.is_recording:
            self.logger.warning("Already recording")
            return
        
        # Check if audio is still playing or response is being generated
        if self.audio_buffer.is_playing or (self.response_count > 0 and not self.response_done):
            self.logger.info("Interrupting ongoing audio playback/generation")
            
            # Stop audio playback immediately
            self.audio_buffer.stop()
            
            # Only send cancel if response is still being generated
            if self.current_response_id and not self.response_done:
                try:
                    cancel_msg = {
                        "type": "response.cancel",
                        "event_id": f"evt_{int(time.time() * 1000000)}"
                    }
                    await self.stream_manager.connection.send(cancel_msg)
                    self.logger.debug(f"Sent response.cancel for response {self.current_response_id}")
                except Exception as e:
                    self.logger.debug(f"Cancel not needed or failed: {e}")
            
            # Reset all response-related state
            self.is_processing = False
            self.response_count = 0
            self.got_text_response = False
            self.current_response_id = None
            self.response_done = True
            self.current_ai_transcript = ""
            
            # Clear any remaining audio chunks
            self.current_response_audio.clear()
            
            # Wait a bit for cancellation to process
            await asyncio.sleep(0.2)
        
        self.interaction_num += 1
        self.logger.info(f"=== INTERACTION {self.interaction_num} START ===")
        
        # Reset transcripts for new interaction
        self.current_user_transcript = ""
        self.current_ai_transcript = ""
        
        self.is_recording = True
        self.recording_buffer.clear()
        self.timings["recording_start"] = time.time()
        
        self.logger.info("Recording started")
        print("\n🔴 Recording... (release to send)")
        
        try:
            # Clear input buffer first
            self.logger.debug("Clearing input buffer")
            clear_msg = MessageFactory.input_audio_buffer_clear()
            await self.stream_manager.connection.send(clear_msg)
            await asyncio.sleep(0.1)
            
            # Start audio capture
            self.audio_queue = await self.capture.start_async_capture()
            self.logger.info("Audio capture started")
            
            # Start processing loop
            self.recording_task = asyncio.create_task(self._process_audio_chunks())
        except Exception as e:
            self.logger.error(f"Error starting recording: {e}", exc_info=True)
            self.is_recording = False
        
    async def initialize(self):
        """Initialize the voice chat system"""
        self.logger.info("=== INITIALIZATION START ===")
        print("🎤 Initializing voice chat...")
        
        self.event_loop = asyncio.get_event_loop()
        self.session_num += 1
        
        # Connect to OpenAI
        try:
            await self.stream_manager.start()
            self.logger.info("WebSocket connected successfully")
            print("✅ Connected to OpenAI Realtime API")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}", exc_info=True)
            raise
        
        # Set up callbacks
        self.stream_manager.set_audio_callback(self._on_audio_response)
        self.stream_manager.set_text_callback(self._on_text_response)
        self.stream_manager.set_error_callback(self._on_error)
        self.stream_manager.response_done_callback = self._on_response_done
        
        # Wait for session to be established
        await asyncio.sleep(0.5)
        
        # Update session configuration with English language
        session_msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            #instructions="You are a helpful voice assistant. Be concise and conversational. Always respond in English.",
            instructions="""You are a helpful voice assistant. Be concise and conversational. 
    IMPORTANT: You must ALWAYS respond in English, regardless of any other factors. 
    Never respond in any other language unless explicitly asked to translate something.
    Your default language is English.""",
            
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": False  # Disable auto-response
            },
            temperature=0.8,
            input_audio_transcription={  # Force English transcription
                "model": "whisper-1",
                "language": "en"
            }
        )
        
        self.logger.info("Sending session configuration with English language")
        await self.stream_manager.connection.send(session_msg)
        
        # Wait for session update
        await asyncio.sleep(0.5)
        
        self.logger.info("=== INITIALIZATION COMPLETE ===")
        print("✅ Voice chat ready!")
        
    # async def start_recording_async(self):
    #     """Start recording audio (async)"""
    #     if self.is_recording:
    #         self.logger.warning("Already recording")
    #         return
        
    #     # Check if audio is still playing from previous response
    #     if self.audio_buffer.is_playing or (self.response_count > 0 and not self.response_done):
    #     # if self.audio_buffer.is_playing or self.response_count > 0:
    #         self.logger.info("Interrupting ongoing audio playback")
            
    #         # Stop audio playback immediately
    #         self.audio_buffer.stop()
            
    #         # Send response.cancel to stop server-side generation
    #         try:
    #             cancel_msg = {
    #                 "type": "response.cancel",
    #                 "event_id": f"evt_{int(time.time() * 1000000)}"
    #             }
    #             await self.stream_manager.connection.send(cancel_msg)
    #             self.logger.debug("Sent response.cancel")
    #         except Exception as e:
    #             self.logger.error(f"Failed to cancel response: {e}")
            
    #         # Reset all response-related state
    #         self.is_processing = False
    #         self.response_count = 0
    #         self.got_text_response = False
            
    #         # Clear any remaining audio chunks
    #         self.current_response_audio.clear()
            
    #         # Wait a bit for cancellation to process
    #         await asyncio.sleep(0.2)
        
    #     self.interaction_num += 1
    #     self.logger.info(f"=== INTERACTION {self.interaction_num} START ===")
        
    #     self.is_recording = True
    #     self.recording_buffer.clear()
    #     self.timings["recording_start"] = time.time()
        
    #     self.logger.info("Recording started")
    #     print("\n🔴 Recording... (release to send)")
        
    #     try:
    #         # Clear input buffer first
    #         self.logger.debug("Clearing input buffer")
    #         clear_msg = MessageFactory.input_audio_buffer_clear()
    #         await self.stream_manager.connection.send(clear_msg)
    #         await asyncio.sleep(0.1)
            
    #         # Start audio capture
    #         self.audio_queue = await self.capture.start_async_capture()
    #         self.logger.info("Audio capture started")
            
    #         # Start processing loop
    #         self.recording_task = asyncio.create_task(self._process_audio_chunks())
    #     except Exception as e:
    #         self.logger.error(f"Error starting recording: {e}", exc_info=True)
    #         self.is_recording = False
    
    def start_recording(self):
        """Start recording (sync wrapper)"""
        if self.event_loop and not self.is_recording:
            future = asyncio.run_coroutine_threadsafe(
                self.start_recording_async(),
                self.event_loop
            )
    
    async def stop_recording_async(self):
        """Stop recording and send to AI (async)"""
        if not self.is_recording:
            self.logger.warning("Stop recording called but not recording")
            return
        
        self.timings["recording_end"] = time.time()
        recording_duration = self.timings["recording_end"] - self.timings["recording_start"]
        
        self.logger.info(f"Recording stopped after {recording_duration:.2f}s")
        
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
            buffer_size = sum(len(chunk) for chunk in self.recording_buffer)
            self.logger.info(f"Processing {len(self.recording_buffer)} chunks, {buffer_size} bytes")
            await self._send_audio_to_ai()
        else:
            self.logger.warning("No audio recorded")
            print("⚠️  No audio recorded")
    
    def stop_recording(self):
        """Stop recording (sync wrapper)"""
        if self.event_loop and self.is_recording:
            future = asyncio.run_coroutine_threadsafe(
                self.stop_recording_async(),
                self.event_loop
            )
    
    async def _process_audio_chunks(self):
        """Process audio chunks while recording"""
        chunk_count = 0
        self.logger.debug("Audio processing loop started")
        
        try:
            while self.is_recording and self.audio_queue:
                try:
                    # Get audio chunk
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    chunk_count += 1
                    self.recording_buffer.append(chunk)
                    
                    if chunk_count % 10 == 0:
                        self.logger.debug(f"Processed {chunk_count} chunks")
                            
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    self.logger.debug("Audio processing cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Audio processing error: {e}", exc_info=True)
                    break
        except Exception as e:
            self.logger.error(f"Recording loop error: {e}", exc_info=True)
        finally:
            self.logger.debug(f"Audio processing loop ended: {chunk_count} chunks")
    
    async def _send_audio_to_ai(self):
        """Send recorded audio to AI"""
        if self.is_processing:
            self.logger.warning("Already processing")
            return
        
        self.is_processing = True
        self.response_count = 0
        self.got_text_response = False
        self.timings["send_start"] = time.time()
        
        self.logger.info("Sending audio to AI")
        print("📤 Sending to AI...")
        
        try:
            # Combine all chunks
            full_audio = b''.join(self.recording_buffer)
            audio_duration_ms = len(full_audio) / (self.audio_config.sample_rate * 2) * 1000
            
            self.logger.info(f"Audio stats: {len(full_audio)} bytes, {audio_duration_ms:.0f}ms")
            
            if audio_duration_ms < 100:
                self.logger.warning("Audio too short")
                print("⚠️  Audio too short")
                self.is_processing = False
                return
                
            print(f"   Duration: {audio_duration_ms:.0f}ms")
            
            # Send audio in reasonable chunks
            chunk_size = 24000  # 500ms chunks
            chunks_sent = 0
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i:i+chunk_size]
                await self.stream_manager.send_audio(chunk)
                chunks_sent += 1
                await asyncio.sleep(0.01)
            
            self.logger.info(f"Sent {chunks_sent} audio chunks")
            
            # Commit the audio buffer
            self.logger.debug("Committing audio buffer")
            commit_msg = MessageFactory.input_audio_buffer_commit()
            await self.stream_manager.connection.send(commit_msg)
            
            # Create response
            self.logger.debug("Creating response")
            response_msg = MessageFactory.response_create()
            await self.stream_manager.connection.send(response_msg)
            
            print("⏳ Waiting for response...")
            
        except Exception as e:
            self.logger.error(f"Error sending audio: {e}", exc_info=True)
            print(f"❌ Error: {e}")
            self.is_processing = False
    
    def _on_audio_response(self, audio_chunk: bytes):
        """Handle audio response from AI"""
        self.response_count += 1
        
        # Track timing
        if self.response_count == 1:
            self.timings["first_response"] = time.time()
            latency = self.timings["first_response"] - self.timings["send_start"]
            self.logger.info(f"First audio response received: latency={latency:.3f}s")
            print("🔊 Playing response...")
        
        # Add to buffer for smooth playback
        self.audio_buffer.add_chunk(audio_chunk)
    
    def _on_text_response(self, text: str):
        """Handle text response from AI"""
        if text.strip():
            if not self.got_text_response:
                self.logger.info(f"Text response: {text[:50]}...")
                print(f"\n💬 ", end="", flush=True)
                self.got_text_response = True
            print(text, end="", flush=True)
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        self.logger.error(f"Stream error: {error}", exc_info=True)
        print(f"\n❌ Error: {error}")
        self.is_processing = False
    
    def _on_response_done(self):
        """Handle response completion"""
        self.timings["response_end"] = time.time()
        
        # Calculate metrics
        if self.timings["first_response"] > 0:
            response_latency = self.timings["first_response"] - self.timings["send_start"]
            total_time = self.timings["response_end"] - self.timings["recording_start"]
            response_duration = self.timings["response_end"] - self.timings["first_response"]
            
            self.logger.info(
                f"Interaction complete: "
                f"latency={response_latency:.3f}s, "
                f"response_duration={response_duration:.3f}s, "
                f"total_time={total_time:.3f}s, "
                f"chunks={self.response_count}"
            )
        
        if self.got_text_response:
            print()  # New line after text
        
        if self.response_count > 0:
            print(f"✅ Done ({self.response_count} chunks)")
        
        # Reset response tracking
        self.response_count = 0
        self.is_processing = False
        
        # Log message statistics
        if hasattr(self.stream_manager.connection, 'message_count'):
            stats = self.stream_manager.connection.message_count
            self.logger.info(f"Messages: sent={stats['sent']}, received={stats['received']}")
        
        self.logger.info(f"=== INTERACTION {self.interaction_num} END ===")
        print("\n🎯 Ready! (Hold SPACE to talk, or type 'quit')")
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("=== CLEANUP START ===")
        print("\n👋 Shutting down...")
        
        if self.is_recording:
            await self.stop_recording_async()
        
        self.audio_buffer.stop()
        
        if self.stream_manager.state == StreamState.ACTIVE:
            await self.stream_manager.stop()
        
        # Log final statistics
        self.logger.info(f"Session {self.session_num} complete: {self.interaction_num} interactions")
        
        # Save message log if available
        if hasattr(self.stream_manager.connection, 'message_log'):
            msg_log_file = LOG_DIR / f"messages_{timestamp}.json"
            with open(msg_log_file, 'w') as f:
                json.dump(self.stream_manager.connection.message_log, f, indent=2)
            self.logger.info(f"Message log saved: {msg_log_file}")
        
        self.logger.info("=== CLEANUP COMPLETE ===")
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
            logger.error(f"Input error: {e}", exc_info=True)


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
            logger.error(f"Key press error: {e}", exc_info=True)
    
    def on_release(key):
        try:
            if key == pynput_keyboard.Key.space and chat.is_holding_key:
                chat.is_holding_key = False
                chat.stop_recording()
        except Exception as e:
            logger.error(f"Key release error: {e}", exc_info=True)
    
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
        logger.error("OPENAI_API_KEY not found")
        print("❌ OPENAI_API_KEY not found in .env file")
        return
    
    logger.info(f"API key loaded: ...{api_key[-8:]}")
    
    # Create chat instance
    chat = DebugInteractiveVoiceChat(api_key)
    
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
        logger.info("User interrupted")
        print("\n\n⚡ Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        await chat.cleanup()


def main():
    """Main entry point"""
    print("🎙️  RealtimeVoiceAPI - Interactive Voice Chat (Debug Mode)")
    print("=" * 60)
    print(f"📝 Log file: {log_file}")
    print("=" * 60)
    
    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Check dependencies
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        logger.info(f"Audio devices: {len(devices)}")
        
        # Log device details
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                logger.debug(f"Input device {i}: {device['name']} ({device['max_input_channels']}ch)")
        
        print(f"✅ Audio ready ({len(devices)} devices)")
    except ImportError:
        logger.error("sounddevice not installed")
        print("❌ sounddevice not installed. Run: pip install sounddevice")
        return
    except Exception as e:
        logger.error(f"Audio initialization error: {e}", exc_info=True)
        print(f"❌ Audio error: {e}")
        return
    
    if not KEYBOARD_LIB:
        print("💡 Tip: Install pynput for better controls: pip install pynput")
    
    # Run the chat
    try:
        asyncio.run(run_interactive_chat())
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
    finally:
        logger.info("Application shutdown complete")
        print(f"\n📝 Log saved to: {log_file}")


if __name__ == "__main__":
    main()