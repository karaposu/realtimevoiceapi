#!/usr/bin/env python3
"""
Fixed turn-based implementation using external audio recording
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE imports
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Add parent directory to path for imports (same as run.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from realtimevoiceapi import VoiceEngine, VoiceEngineConfig

# Import sounddevice for direct audio recording
try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("❌ sounddevice not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice", "numpy"])
    import sounddevice as sd
    import numpy as np


class TurnBasedChat:
    def __init__(self):
        # Check for API key
      
        
        # Debug: Show what we found
        if api_key:
            print(f"✅ API key found (length: {len(api_key)})")
            # Check if it's a valid format
            if not api_key.startswith("sk-"):
                print("⚠️  API key doesn't start with 'sk-'")
        else:
            print("❌ No API key found!")
            print("\n💡 Set your OpenAI API key:")
            print("   export OPENAI_API_KEY='your-key'")
            print("   or create a .env file with:")
            print("   OPENAI_API_KEY=your-key-here")
            raise ValueError("Missing OPENAI_API_KEY")
            
        self.config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            latency_mode="ultra_low",
            log_level="WARNING",
            vad_enabled=False,  # No VAD for turn-based
            vad_type=None
        )
        self.engine = VoiceEngine(config=self.config)
        
        # Audio recording parameters (matching VoiceEngine defaults)
        self.sample_rate = 24000
        self.channels = 1
        self.dtype = np.int16
        
    def setup_handlers(self):
        """Setup event handlers"""
        # Handle AI responses
        self.engine.on_response_text = self._on_response_text
        self.engine.on_response_done = self._on_response_done
        self.engine.on_error = self._on_error
        
    def _on_response_text(self, text: str):
        """Handle AI text response"""
        print(text, end="", flush=True)
        
    def _on_response_done(self):
        """Handle end of AI response"""
        print("\n")
        
    def _on_error(self, error: Exception):
        """Handle errors"""
        print(f"\n❌ Error: {error}")
        
    def record_audio(self, duration: Optional[float] = None) -> bytes:
        """Record audio and return PCM data"""
        print("🎤 Recording... Press ENTER when done")
        
        # If no duration specified, record until Enter is pressed
        if duration is None:
            # Start recording in a list to accumulate chunks
            audio_chunks = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"⚠️  Recording status: {status}")
                audio_chunks.append(indata.copy())
            
            # Start recording
            stream = sd.InputStream(
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
                audio_data = np.concatenate(audio_chunks, axis=0)
            else:
                audio_data = np.array([], dtype=self.dtype)
        else:
            # Record for specified duration
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype
            )
            sd.wait()
        
        # Convert to bytes (PCM16)
        return audio_data.tobytes()
        
    async def process_turn(self):
        """Record and process a single turn"""
        # Record audio
        audio_data = self.record_audio()
        
        if len(audio_data) > 0:
            print("📤 Processing...")
            # Use send_recorded_audio for turn-based interaction
            await self.engine.send_recorded_audio(audio_data)
        else:
            print("⚠️  No audio recorded")
            
    async def run(self):
        """Run the turn-based chat"""
        self.setup_handlers()
        
        try:
            # Connect
            print("\n🎙️  Turn-Based Voice Chat (Fixed)")
            print("=" * 40)
            print("Connecting...", end="", flush=True)
            await self.engine.connect()
            print(" ✅")
            
            print("\n🎯 Turn-Based Conversation")
            print("   Press [ENTER] to start recording")
            print("   Press [ENTER] again to send")
            print("   Type 'q' to quit")
            print("   💡 Using send_recorded_audio for proper turn-based mode")
            print("   💡 Tip: Use headphones for best experience\n")
            
            while True:
                # Wait for user to start
                cmd = input("\nPress ENTER to speak (or 'q' to quit): ")
                if cmd.lower() in ['q', 'quit']:
                    break
                    
                # Process one turn
                await self.process_turn()
                print("🤖 AI: ", end="", flush=True)
                
                # Give AI time to respond
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⚡ Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if "invalid_api_key" in str(e):
                print("\n⚠️  API Key Issue:")
                print("   1. Make sure your API key has access to the Realtime API")
                print("   2. The Realtime API may require specific permissions")
                print("   3. Check if your key starts with 'sk-' and is valid")
                print(f"   4. Current key starts with: {self.config.api_key[:7] if self.config.api_key else 'None'}...")
            import traceback
            traceback.print_exc()
        finally:
            print("\nDisconnecting...")
            await self.engine.disconnect()
            print("👋 Goodbye!")


async def main():
    chat = TurnBasedChat()
    await chat.run()


if __name__ == "__main__":
    asyncio.run(main())