#!/usr/bin/env python3
"""
Interactive Voice API Terminal - With Interaction Logging

python -m realtimevoiceapi.interactive_test
"""

import asyncio
import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from enum import Enum
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from base_voiceapi_engine import BaseVoiceAPIEngine, VoiceMode, VADType, VoiceResponse
from realtimevoiceapi.audio import AudioConfig, AudioProcessor


class InputMode(Enum):
    """Input format modes"""
    TEXT_ONLY = "text_only"
    VOICE_TURN_BASED = "voice_turn_based"
    VOICE_SEMANTIC_VAD = "voice_semantic_vad"
    VOICE_SERVER_VAD = "voice_server_vad"


class OutputMode(Enum):
    """Response output format"""
    TEXT_ONLY = "text_only"
    VOICE_ONLY = "voice_only"
    BOTH = "both"


class InteractionLogger:
    """Logs all interactions to file"""
    
    def __init__(self, log_dir: Path = Path("voice_logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp}"
        self.log_file = self.log_dir / f"{self.session_id}.json"
        self.text_log_file = self.log_dir / f"{self.session_id}.txt"
        
        # Initialize log data
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "configuration": {},
            "interactions": [],
            "statistics": {
                "total_interactions": 0,
                "total_user_audio_ms": 0,
                "total_assistant_audio_ms": 0,
                "total_user_text_chars": 0,
                "total_assistant_text_chars": 0,
                "vad_events": 0,
                "errors": 0
            }
        }
        
        # Write initial session data
        self._save_session_data()
        
        # Also create human-readable log
        with open(self.text_log_file, 'w') as f:
            f.write(f"Voice API Session Log\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_configuration(self, config: Dict[str, Any]):
        """Log session configuration"""
        self.session_data["configuration"] = {
            "voice": config.get("voice"),
            "input_mode": config.get("input_mode").value if hasattr(config.get("input_mode"), 'value') else str(config.get("input_mode")),
            "output_mode": config.get("output_mode").value if hasattr(config.get("output_mode"), 'value') else str(config.get("output_mode")),
            "temperature": config.get("temperature"),
            "instructions": config.get("instructions")
        }
        self._save_session_data()
        
        # Update text log
        with open(self.text_log_file, 'a') as f:
            f.write(f"Configuration Updated:\n")
            f.write(f"  Voice: {config.get('voice')}\n")
            f.write(f"  Input Mode: {self.session_data['configuration']['input_mode']}\n")
            f.write(f"  Output Mode: {self.session_data['configuration']['output_mode']}\n")
            f.write(f"  Temperature: {config.get('temperature')}\n")
            f.write(f"  Instructions: {config.get('instructions')}\n")
            f.write("-" * 80 + "\n\n")
    
    def log_interaction_start(self, interaction_type: str) -> str:
        """Start logging a new interaction"""
        interaction_id = f"interaction_{int(time.time() * 1000)}"
        interaction = {
            "id": interaction_id,
            "type": interaction_type,
            "start_time": datetime.now().isoformat(),
            "user_input": {},
            "assistant_response": {},
            "events": [],
            "metrics": {}
        }
        self.session_data["interactions"].append(interaction)
        self.session_data["statistics"]["total_interactions"] += 1
        return interaction_id
    
    def log_user_input(self, interaction_id: str, input_type: str, content: Any, metadata: Dict[str, Any] = None):
        """Log user input"""
        interaction = self._get_interaction(interaction_id)
        if interaction:
            interaction["user_input"] = {
                "type": input_type,
                "timestamp": datetime.now().isoformat(),
                "content": content if input_type == "text" else f"[Audio: {metadata.get('duration_ms', 0):.1f}ms]",
                "metadata": metadata or {}
            }
            
            # Update statistics
            if input_type == "text":
                self.session_data["statistics"]["total_user_text_chars"] += len(content)
            elif input_type == "audio" and metadata:
                self.session_data["statistics"]["total_user_audio_ms"] += metadata.get("duration_ms", 0)
            
            self._save_session_data()
            
            # Update text log
            with open(self.text_log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] USER ({input_type}):\n")
                if input_type == "text":
                    f.write(f"  {content}\n")
                else:
                    f.write(f"  [Audio: {metadata.get('duration_ms', 0):.1f}ms]\n")
                f.write("\n")
    
    def log_assistant_response(self, interaction_id: str, text: str = None, audio_duration_ms: float = None, metadata: Dict[str, Any] = None):
        """Log assistant response"""
        interaction = self._get_interaction(interaction_id)
        if interaction:
            response = {
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            if text:
                response["text"] = text
                self.session_data["statistics"]["total_assistant_text_chars"] += len(text)
            
            if audio_duration_ms:
                response["audio_duration_ms"] = audio_duration_ms
                self.session_data["statistics"]["total_assistant_audio_ms"] += audio_duration_ms
            
            interaction["assistant_response"] = response
            interaction["end_time"] = datetime.now().isoformat()
            
            # Calculate interaction duration
            start = datetime.fromisoformat(interaction["start_time"])
            end = datetime.fromisoformat(interaction["end_time"])
            interaction["duration_seconds"] = (end - start).total_seconds()
            
            self._save_session_data()
            
            # Update text log
            with open(self.text_log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ASSISTANT:\n")
                if text:
                    f.write(f"  {text}\n")
                if audio_duration_ms:
                    f.write(f"  [Audio: {audio_duration_ms:.1f}ms]\n")
                f.write(f"  [Total interaction time: {interaction['duration_seconds']:.1f}s]\n")
                f.write("-" * 80 + "\n\n")
    
    def log_event(self, interaction_id: str, event_type: str, details: Dict[str, Any] = None):
        """Log an event during interaction"""
        interaction = self._get_interaction(interaction_id)
        if interaction:
            event = {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                "details": details or {}
            }
            interaction["events"].append(event)
            
            if event_type == "vad_speech_start" or event_type == "vad_speech_stop":
                self.session_data["statistics"]["vad_events"] += 1
            elif event_type == "error":
                self.session_data["statistics"]["errors"] += 1
            
            self._save_session_data()
    
    def log_error(self, error_message: str, interaction_id: str = None):
        """Log an error"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": error_message,
            "interaction_id": interaction_id
        }
        
        if "errors_log" not in self.session_data:
            self.session_data["errors_log"] = []
        
        self.session_data["errors_log"].append(error_entry)
        self.session_data["statistics"]["errors"] += 1
        self._save_session_data()
        
        # Update text log
        with open(self.text_log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR:\n")
            f.write(f"  {error_message}\n\n")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "session_id": self.session_id,
            "duration": (datetime.now() - datetime.fromisoformat(self.session_data["start_time"])).total_seconds(),
            "statistics": self.session_data["statistics"],
            "log_files": {
                "json": str(self.log_file),
                "text": str(self.text_log_file)
            }
        }
    
    def close_session(self):
        """Close the logging session"""
        self.session_data["end_time"] = datetime.now().isoformat()
        start = datetime.fromisoformat(self.session_data["start_time"])
        end = datetime.fromisoformat(self.session_data["end_time"])
        self.session_data["total_duration_seconds"] = (end - start).total_seconds()
        
        self._save_session_data()
        
        # Update text log
        with open(self.text_log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {self.session_data['total_duration_seconds']:.1f} seconds\n")
            f.write(f"\nSession Statistics:\n")
            for key, value in self.session_data["statistics"].items():
                f.write(f"  {key}: {value}\n")
    
    def _get_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get interaction by ID"""
        for interaction in self.session_data["interactions"]:
            if interaction["id"] == interaction_id:
                return interaction
        return None
    
    def _save_session_data(self):
        """Save session data to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)


class MicrophoneStream:
    """Handle continuous microphone streaming for VAD modes"""
    
    def __init__(self, sample_rate=24000, chunk_duration_ms=100):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"\n⚠️ Audio callback error: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def start(self):
        """Start microphone stream"""
        import sounddevice as sd
        self.is_recording = True
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='int16'
        )
        self.stream.start()
    
    def stop(self):
        """Stop microphone stream"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_chunk(self):
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None


class InteractiveVoiceTerminal:
    """Enhanced interactive terminal for Voice API testing with logging"""
    
    def __init__(self):
        self.engine: Optional[BaseVoiceAPIEngine] = None
        self.running = False
        self.mic_stream: Optional[MicrophoneStream] = None
        self.is_mic_muted = False
        self.vad_listening = False
        self.vad_task = None
        self.response_task = None
        self.audio_processor = AudioProcessor()
        self.last_status_update = 0
        
        # Logging
        self.logging_enabled = True  # Default to enabled
        self.interaction_logger: Optional[InteractionLogger] = None
        self.current_interaction_id: Optional[str] = None
        
        # Response tracking
        self.current_response_text = ""
        self.current_audio_duration = 0
        
        self.current_config = {
            "voice": "alloy",
            "input_mode": InputMode.VOICE_SERVER_VAD,
            "output_mode": OutputMode.BOTH,
            "temperature": 0.8,
            "instructions": "You are a helpful assistant. Be conversational and brief.",
            "logging": True
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize interaction logger if enabled
        if self.logging_enabled:
            self.interaction_logger = InteractionLogger()
            self.interaction_logger.log_configuration(self.current_config)
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def update_status_line(self, message: str):
        """Update status line without clearing screen"""
        # Move cursor to status line
        print(f"\r{' ' * 80}\r{message}", end='', flush=True)
    
    def print_header(self):
        """Print application header"""
        self.clear_screen()
        
        # Voice-only mode has minimal UI
        if (self.current_config["input_mode"] != InputMode.TEXT_ONLY and 
            self.current_config["output_mode"] == OutputMode.VOICE_ONLY):
            self.print_voice_only_ui()
            return
        
        # Regular header for other modes
        print("=" * 60)
        print("🎙️  Interactive Voice API Terminal")
        print("=" * 60)
        print("\nCurrent Configuration:")
        print(f"  Voice: {self.current_config['voice']}")
        print(f"  Input Mode: {self.current_config['input_mode'].value}")
        print(f"  Output Mode: {self.current_config['output_mode'].value}")
        print(f"  Temperature: {self.current_config['temperature']}")
        print(f"  Logging: {'✅ Enabled' if self.logging_enabled else '❌ Disabled'}")
        
        if self.logging_enabled and self.interaction_logger:
            print(f"  Log File: {self.interaction_logger.session_id}.json")
        
        # Show actual VAD status
        if self.current_config["input_mode"] in [InputMode.VOICE_SERVER_VAD, InputMode.VOICE_SEMANTIC_VAD]:
            if self.vad_listening:
                vad_status = "🔴 LISTENING" if not self.is_mic_muted else "🔇 MUTED"
            else:
                vad_status = "⚠️ NOT STARTED"
            print(f"  VAD Status: {vad_status}")
        
        print("=" * 60)
        print("\nControls:")
        print("  Q - Show/Change configuration")
        print("  L - Toggle logging")
        print("  V - View session summary")
        
        if self.current_config["input_mode"] == InputMode.TEXT_ONLY:
            print("  R - Send text message")
        elif self.current_config["input_mode"] == InputMode.VOICE_TURN_BASED:
            print("  T - Record and send voice message")
        else:  # VAD modes
            print("  M - Toggle microphone")
            if not self.vad_listening:
                print("  S - Start VAD listening")
        
        print("  X - Exit")
        print("\n" + "-" * 60)
        print("\nStatus: Ready")
        print()  # Extra line for status updates
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio using sounddevice"""
        try:
            import sounddevice as sd
            import numpy as np
            
            self.update_status_line("🔊 Playing audio response...")
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            sd.play(audio_array, samplerate=AudioConfig.SAMPLE_RATE)
            sd.wait()
            self.update_status_line("✅ Audio playback complete")
            
        except Exception as e:
            self.update_status_line(f"⚠️ Audio playback error: {e}")
            self.logger.error(f"Audio playback error: {e}")
            if self.logging_enabled and self.current_interaction_id:
                self.interaction_logger.log_error(f"Audio playback error: {e}", self.current_interaction_id)
    
    async def initialize_engine(self, start_vad=True):
        """Initialize or reinitialize the engine with current config"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        # Stop existing tasks
        await self.stop_vad_listening()
        
        # Disconnect existing engine
        if self.engine and self.engine.is_connected:
            await self.engine.disconnect()
        
        # Get engine configuration
        voice_mode, vad_type, auto_response = self.get_engine_config()
        
        self.engine = BaseVoiceAPIEngine(
            api_key=api_key,
            voice=self.current_config["voice"],
            mode=voice_mode,
            vad_type=vad_type,
            logger=self.logger
        )
        
        # Register response handlers
        self.engine.on_event("response.audio.delta", self._on_audio_delta)
        self.engine.on_event("response.audio.done", self._on_audio_done)
        self.engine.on_event("response.text.delta", self._on_text_delta)
        self.engine.on_event("response.done", self._on_response_done)
        self.engine.on_event("input_audio_buffer.speech_started", self._on_speech_started)
        self.engine.on_event("input_audio_buffer.speech_stopped", self._on_speech_stopped)
        
        try:
            await self.engine.connect(
                instructions=self.current_config["instructions"],
                temperature=self.current_config["temperature"],
                auto_response=auto_response
            )
            
            # Start VAD listening if in VAD mode
            if start_vad and self.current_config["input_mode"] in [InputMode.VOICE_SERVER_VAD, InputMode.VOICE_SEMANTIC_VAD]:
                await asyncio.sleep(0.5)  # Give connection time to stabilize
                await self.start_vad_listening()
            
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            if self.logging_enabled:
                self.interaction_logger.log_error(f"Connection failed: {e}")
            return False
    
    async def _on_speech_started(self, data):
        """Handle speech start event"""
        self.update_status_line("🎤 Speech detected...")
        if self.logging_enabled and self.current_interaction_id:
            self.interaction_logger.log_event(self.current_interaction_id, "vad_speech_start", data)
    
    async def _on_speech_stopped(self, data):
        """Handle speech stop event"""
        self.update_status_line("🤔 Processing...")
        if self.logging_enabled and self.current_interaction_id:
            self.interaction_logger.log_event(self.current_interaction_id, "vad_speech_stop", data)
    
    async def _on_audio_delta(self, data):
        """Handle audio delta event"""
        # Audio is being received
        pass
    
    async def _on_audio_done(self, data):
        """Handle audio done event"""
        self.update_status_line("🔊 Response ready")
    
    async def _on_text_delta(self, data):
        """Handle text delta event"""
        text = data.get("delta", "")
        if text:
            self.current_response_text += text
            if self.current_config["output_mode"] in [OutputMode.TEXT_ONLY, OutputMode.BOTH]:
                # Print text incrementally
                print(text, end='', flush=True)
    
    async def _on_response_done(self, data):
        """Handle response done event"""
        print()  # New line after text
        
        # Check for audio to play
        audio_played = False
        if self.current_config["output_mode"] in [OutputMode.VOICE_ONLY, OutputMode.BOTH]:
            audio_duration = self.engine.client.get_audio_output_duration()
            if audio_duration > 0:
                self.current_audio_duration = audio_duration
                audio_bytes = self.engine.client.get_audio_output(clear_buffer=True)
                if audio_bytes:
                    await asyncio.to_thread(self.play_audio, audio_bytes)
                    audio_played = True
        
        # Log the complete response
        if self.logging_enabled and self.current_interaction_id:
            self.interaction_logger.log_assistant_response(
                self.current_interaction_id,
                text=self.current_response_text if self.current_response_text else None,
                audio_duration_ms=self.current_audio_duration if audio_played else None
            )
        
        # Reset for next interaction
        self.current_response_text = ""
        self.current_audio_duration = 0
        self.current_interaction_id = None
        
        self.update_status_line("Ready")
    
    async def start_vad_listening(self):
        """Start continuous VAD listening"""
        if self.current_config["input_mode"] not in [InputMode.VOICE_SERVER_VAD, InputMode.VOICE_SEMANTIC_VAD]:
            print("VAD listening only available in VAD modes")
            return
        
        if self.vad_listening:
            print("VAD already listening")
            return
        
        try:
            import sounddevice as sd
            
            self.logger.info("Starting VAD listening...")
            
            # Initialize microphone stream
            self.mic_stream = MicrophoneStream()
            self.vad_listening = True
            
            # Start the listening task
            self.vad_task = asyncio.create_task(self.vad_listening_loop())
            
            self.update_status_line("✅ VAD listening started")
            
        except ImportError:
            print("❌ sounddevice required for VAD mode")
            self.vad_listening = False
    
    async def vad_listening_loop(self):
        """Continuous listening loop for VAD mode"""
        if not self.mic_stream:
            return
        
        self.mic_stream.start()
        self.logger.info("Microphone stream started")
        
        chunk_count = 0
        audio_buffer = bytearray()
        
        while self.vad_listening and self.engine and self.engine.is_connected:
            try:
                # Get audio chunk
                chunk = self.mic_stream.get_chunk()
                if chunk is None:
                    await asyncio.sleep(0.01)
                    continue
                
                if self.is_mic_muted:
                    continue
                
                # Convert to bytes
                chunk_bytes = chunk.tobytes()
                audio_buffer.extend(chunk_bytes)
                
                # Start new interaction if needed
                if not self.current_interaction_id and self.logging_enabled:
                    self.current_interaction_id = self.interaction_logger.log_interaction_start("voice_vad")
                
                # Send to API using the simple method
                audio_b64 = self.engine.audio_processor.bytes_to_base64(chunk_bytes)
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await self.engine.client.connection.send_event(event)
                
                chunk_count += 1
                
                # Log audio input periodically (every second)
                if chunk_count % 10 == 0 and self.logging_enabled and self.current_interaction_id:
                    duration_ms = len(audio_buffer) / (AudioConfig.SAMPLE_RATE * 2) * 1000  # 2 bytes per sample
                    self.interaction_logger.log_user_input(
                        self.current_interaction_id,
                        "audio",
                        None,
                        {"duration_ms": duration_ms, "chunks": chunk_count}
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"VAD listening error: {e}")
                if self.logging_enabled:
                    self.interaction_logger.log_error(f"VAD listening error: {e}", self.current_interaction_id)
                self.update_status_line(f"❌ VAD error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("VAD listening loop ended")
        if self.mic_stream:
            self.mic_stream.stop()
    
    async def stop_vad_listening(self):
        """Stop VAD listening properly"""
        if self.vad_listening:
            self.logger.info("Stopping VAD listening...")
            self.vad_listening = False
            
            if self.vad_task:
                self.vad_task.cancel()
                try:
                    await self.vad_task
                except asyncio.CancelledError:
                    pass
                self.vad_task = None
            
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream = None
    
    def toggle_logging(self):
        """Toggle logging on/off"""
        self.logging_enabled = not self.logging_enabled
        self.current_config["logging"] = self.logging_enabled
        
        if self.logging_enabled and not self.interaction_logger:
            # Create new logger
            self.interaction_logger = InteractionLogger()
            self.interaction_logger.log_configuration(self.current_config)
            print(f"\n✅ Logging enabled - Session: {self.interaction_logger.session_id}")
        elif not self.logging_enabled and self.interaction_logger:
            # Close current logger
            self.interaction_logger.close_session()
            print(f"\n❌ Logging disabled - Session saved: {self.interaction_logger.session_id}")
            self.interaction_logger = None
        
        input("\nPress Enter to continue...")
    
    def view_session_summary(self):
        """View current session summary"""
        if not self.interaction_logger:
            print("\n⚠️ Logging is disabled - no session to summarize")
        else:
            summary = self.interaction_logger.get_session_summary()
            print("\n📊 Session Summary")
            print("=" * 50)
            print(f"Session ID: {summary['session_id']}")
            print(f"Duration: {summary['duration']:.1f} seconds")
            print(f"\nStatistics:")
            for key, value in summary['statistics'].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            print(f"\nLog Files:")
            print(f"  JSON: {summary['log_files']['json']}")
            print(f"  Text: {summary['log_files']['text']}")
        
        input("\nPress Enter to continue...")
    
    def toggle_microphone(self):
        """Toggle microphone mute status"""
        self.is_mic_muted = not self.is_mic_muted
        status = "MUTED" if self.is_mic_muted else "LISTENING"
        self.update_status_line(f"🎤 Microphone {status}")
    
    def get_engine_config(self):
        """Convert current config to engine parameters"""
        # Determine VoiceMode
        if self.current_config["input_mode"] == InputMode.TEXT_ONLY:
            if self.current_config["output_mode"] == OutputMode.TEXT_ONLY:
                voice_mode = VoiceMode.TEXT_ONLY
            else:
                voice_mode = VoiceMode.TEXT_AND_VOICE
        else:
            if self.current_config["output_mode"] == OutputMode.TEXT_ONLY:
                voice_mode = VoiceMode.TEXT_ONLY
            elif self.current_config["output_mode"] == OutputMode.VOICE_ONLY:
                voice_mode = VoiceMode.VOICE_ONLY
            else:
                voice_mode = VoiceMode.TEXT_AND_VOICE
        
        # Determine VAD type
        if self.current_config["input_mode"] == InputMode.VOICE_SEMANTIC_VAD:
            vad_type = VADType.SEMANTIC_VAD
        else:
            vad_type = VADType.SERVER_VAD
        
        # Auto response for VAD modes
        auto_response = self.current_config["input_mode"] != InputMode.VOICE_TURN_BASED
        
        return voice_mode, vad_type, auto_response
    
    async def send_text_message(self):
        """Send a text message (for text-only input mode)"""
        if not self.engine or not self.engine.is_connected:
            print("❌ Not connected to Voice API")
            return
        
        # Reset terminal for input
        if sys.platform != 'win32':
            os.system('stty sane')
        
        message = input("\n📝 Enter your message: ").strip()
        if not message:
            return
        
        # Start interaction logging
        interaction_id = None
        if self.logging_enabled:
            interaction_id = self.interaction_logger.log_interaction_start("text")
            self.interaction_logger.log_user_input(interaction_id, "text", message)
        
        print("\n📤 Sending message...")
        self.current_interaction_id = interaction_id
        self.current_response_text = ""
        
        response = await self.engine.send_text(message)
        
        # Response is handled by event handlers
        
        input("\nPress Enter to continue...")
    
    def get_user_input_nonblocking(self) -> str:
        """Get single character input without blocking"""
        try:
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    return msvcrt.getch().decode('utf-8').upper()
            else:
                import termios, tty, select
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        ch = sys.stdin.read(1).upper()
                    else:
                        ch = ''
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
        except:
            return ''
    
    async def run(self):
        """Main application loop"""
        self.running = True
        
        # Initial connection
        if not await self.initialize_engine(start_vad=True):
            print("Failed to initialize. Check your API key.")
            return
        
        while self.running:
            # Don't refresh header too often in VAD mode
            current_time = time.time()
            if current_time - self.last_status_update > 5:  # Refresh every 5 seconds
                self.print_header()
                self.last_status_update = current_time
            
            # Check for user input
            await asyncio.sleep(0.1)
            choice = self.get_user_input_nonblocking()
            
            if choice == 'X':
                print("\n👋 Exiting...")
                self.running = False
            elif choice == 'Q':
                await self.show_configuration()
            elif choice == 'L':
                self.toggle_logging()
            elif choice == 'V':
                self.view_session_summary()
            elif choice == 'R' and self.current_config["input_mode"] == InputMode.TEXT_ONLY:
                await self.send_text_message()
            elif choice == 'M' and self.current_config["input_mode"] in [InputMode.VOICE_SERVER_VAD, InputMode.VOICE_SEMANTIC_VAD]:
                self.toggle_microphone()
            elif choice == 'S' and self.current_config["input_mode"] in [InputMode.VOICE_SERVER_VAD, InputMode.VOICE_SEMANTIC_VAD]:
                if not self.vad_listening:
                    await self.start_vad_listening()
        
        # Cleanup
        await self.stop_vad_listening()
        
        if self.engine and self.engine.is_connected:
            print("\n🔌 Disconnecting...")
            await self.engine.disconnect()
        
        # Close logging session
        if self.interaction_logger:
            self.interaction_logger.close_session()
            print(f"\n📝 Session logs saved:")
            print(f"   JSON: {self.interaction_logger.log_file}")
            print(f"   Text: {self.interaction_logger.text_log_file}")
        
        print("\n✅ Goodbye!")
    
    async def show_configuration(self):
        """Configuration menu"""
        # Stop VAD while in menu
        was_listening = self.vad_listening
        if was_listening:
            await self.stop_vad_listening()
        
        # Your existing configuration menu code...
        # (keeping it minimal for this example)
        
        print("\n🔄 Configuration updated")
        
        # Log configuration change
        if self.logging_enabled:
            self.interaction_logger.log_configuration(self.current_config)
        
        # Restart engine with new config
        await self.initialize_engine(start_vad=was_listening)
        input("\nPress Enter to continue...")


async def main():
    """Main entry point"""
    app = InteractiveVoiceTerminal()
    
    try:
        await app.run()
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
        if app.engine and app.engine.is_connected:
            await app.engine.disconnect()
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    # Check for sounddevice
    try:
        import sounddevice as sd
        print("✅ sounddevice installed - audio enabled")
    except ImportError:
        print("❌ sounddevice required for this application")
        print("   Install with: pip install sounddevice")
        sys.exit(1)
    
    # Run the app
    asyncio.run(main())