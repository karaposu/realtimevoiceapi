#!/usr/bin/env python3
"""
Interactive Voice API Terminal - Fixed VAD Response Handling
"""

import asyncio
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional
import logging
from enum import Enum
import numpy as np

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
    """Enhanced interactive terminal for Voice API testing"""
    
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
        
        self.current_config = {
            "voice": "alloy",
            "input_mode": InputMode.VOICE_SERVER_VAD,
            "output_mode": OutputMode.BOTH,
            "temperature": 0.8,
            "instructions": "You are a helpful assistant. Be conversational and brief."
        }
        
        # Setup logging - more verbose for debugging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
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
            return False
    
    async def _on_speech_started(self, data):
        """Handle speech start event"""
        self.update_status_line("🎤 Speech detected...")
    
    async def _on_speech_stopped(self, data):
        """Handle speech stop event"""
        self.update_status_line("🤔 Processing...")
    
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
        if text and self.current_config["output_mode"] in [OutputMode.TEXT_ONLY, OutputMode.BOTH]:
            # Print text incrementally
            print(text, end='', flush=True)
    
    async def _on_response_done(self, data):
        """Handle response done event"""
        print()  # New line after text
        
        # Check for audio to play
        if self.current_config["output_mode"] in [OutputMode.VOICE_ONLY, OutputMode.BOTH]:
            audio_duration = self.engine.client.get_audio_output_duration()
            if audio_duration > 0:
                audio_bytes = self.engine.client.get_audio_output(clear_buffer=True)
                if audio_bytes:
                    await asyncio.to_thread(self.play_audio, audio_bytes)
        
        self.update_status_line("Ready")
    
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
            
            if self.response_task:
                self.response_task.cancel()
                try:
                    await self.response_task
                except asyncio.CancelledError:
                    pass
                self.response_task = None
            
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream = None
    
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
            
            # Start response monitoring task
            self.response_task = asyncio.create_task(self.response_monitor_loop())
            
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
                
                # Send to API using the simple method
                audio_b64 = self.engine.audio_processor.bytes_to_base64(chunk_bytes)
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await self.engine.client.connection.send_event(event)
                
                chunk_count += 1
                if chunk_count % 10 == 0:  # Update every second
                    self.update_status_line(f"📡 Streaming audio... ({chunk_count/10:.0f}s)")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"VAD listening error: {e}")
                self.update_status_line(f"❌ VAD error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("VAD listening loop ended")
        if self.mic_stream:
            self.mic_stream.stop()
    
    async def response_monitor_loop(self):
        """Monitor for responses in a separate task"""
        self.logger.info("Response monitor started")
        
        while self.vad_listening:
            try:
                # Just wait for events - they're handled by the event handlers
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Response monitor error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("Response monitor ended")
    
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
        
        print("\n✅ Goodbye!")
    
    async def show_configuration(self):
        """Configuration menu - simplified for testing"""
        # Stop VAD while in menu
        was_listening = self.vad_listening
        if was_listening:
            await self.stop_vad_listening()
        
        # Your existing configuration menu code here...
        # (keeping it short for this example)
        
        print("\n🔄 Applying new configuration...")
        await self.initialize_engine(start_vad=was_listening)
        input("\nPress Enter to continue...")
    
    # ... rest of the methods (send_text_message, etc) remain the same ...


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