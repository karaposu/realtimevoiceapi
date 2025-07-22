#!/usr/bin/env python3
"""
Fixed turn-based implementation that properly handles audio recording
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Add parent directory to path for imports (same as run.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtimevoiceapi import VoiceEngine, VoiceEngineConfig

# Load environment variables from .env file
load_dotenv()




class TurnBasedChat:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No API key found!")
            print("\n💡 Set your OpenAI API key:")
            print("   export OPENAI_API_KEY='your-key'")
            print("   or create a .env file")
            raise ValueError("Missing OPENAI_API_KEY")
            
        self.config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            latency_mode="ultra_low",
            log_level="WARNING",
            vad_enabled=False,  # Manual control
            vad_type=None
        )
        self.engine = VoiceEngine(self.config)
        self.audio_buffer: List[bytes] = []
        self.is_recording = False
        
    def setup_handlers(self):
        """Setup event handlers"""
        # Handle AI responses
        self.engine.on_response_text = self._on_response_text
        self.engine.on_response_done = self._on_response_done
        
        # Capture audio chunks during recording
        self.engine.on_audio_chunk = self._on_audio_chunk
        
    def _on_audio_chunk(self, chunk: bytes):
        """Capture audio chunks while recording"""
        if self.is_recording:
            self.audio_buffer.append(chunk)
            
    def _on_response_text(self, text: str):
        """Handle AI text response"""
        print(text, end="", flush=True)
        
    def _on_response_done(self):
        """Handle end of AI response"""
        print("\n")
        
    async def record_turn(self):
        """Record a single turn of audio"""
        # Clear buffer
        self.audio_buffer = []
        self.is_recording = True
        
        # Start listening
        await self.engine.start_listening()
        
        # Wait for user to press enter
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "")
        
        # Stop recording
        self.is_recording = False
        await self.engine.stop_listening()
        
        # Send the complete recording
        if self.audio_buffer:
            print("📤 Processing...")
            complete_audio = b"".join(self.audio_buffer)
            await self.engine.send_recorded_audio(complete_audio)
        else:
            print("⚠️  No audio recorded")
            
    async def run(self):
        """Run the turn-based chat"""
        self.setup_handlers()
        
        try:
            # Connect
            print("\n🎙️  Turn-Based Voice Chat")
            print("=" * 40)
            print("Connecting...", end="", flush=True)
            await self.engine.connect()
            print(" ✅")
            
            print("\n🎯 Turn-Based Conversation")
            print("   Press [ENTER] to start recording")
            print("   Press [ENTER] again to send")
            print("   Type 'q' to quit")
            print("   💡 Tip: Use headphones for best experience\n")
            
            while True:
                # Wait for user to start
                cmd = input("\nPress ENTER to speak (or 'q' to quit): ")
                if cmd.lower() in ['q', 'quit']:
                    break
                    
                print("🎤 Recording... Press ENTER when done")
                await self.record_turn()
                print("🤖 AI: ", end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n\n⚡ Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
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