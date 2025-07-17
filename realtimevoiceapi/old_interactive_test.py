#!/usr/bin/env python3

# to run python -m realtimevoiceapi.interactive_test
"""
Interactive Voice API Terminal Application

A terminal-based app for testing the BaseVoiceAPIEngine interactively.

Controls:
- Q: Change configuration
- R: Send text message
- T: Send voice message
- Y: Send text + voice message
- X: Exit

Run: python -m realtimevoiceapi.interactive_test
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path so we can import realtimevoiceapi
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from base_voiceapi_engine import BaseVoiceAPIEngine, VoiceMode, VADType, VoiceResponse


class InteractiveVoiceTerminal:
    """Interactive terminal application for Voice API testing"""
    
    def __init__(self):
        self.engine: Optional[BaseVoiceAPIEngine] = None
        self.running = False
        self.current_config = {
            "voice": "alloy",
            "mode": VoiceMode.TEXT_AND_VOICE,
            "vad_type": VADType.SERVER_VAD,
            "temperature": 0.8,
            "instructions": "You are a helpful assistant. Be conversational and brief."
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.WARNING,  # Less verbose for interactive use
            format='%(asctime)s - %(message)s'
        )
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print application header"""
        self.clear_screen()
        print("=" * 60)
        print("🎙️  Interactive Voice API Terminal")
        print("=" * 60)
        print(f"Voice: {self.current_config['voice']}")
        print(f"Mode: {self.current_config['mode'].value}")
        print(f"VAD: {self.current_config['vad_type'].value}")
        print(f"Temperature: {self.current_config['temperature']}")
        print("=" * 60)
        print("\nControls:")
        print("  Q - Change configuration")
        print("  R - Send text message")
        print("  T - Send voice message (3 sec recording)")
        print("  Y - Send text + voice message")
        print("  X - Exit")
        print("\n" + "-" * 60 + "\n")
    
    async def initialize_engine(self):
        """Initialize or reinitialize the engine with current config"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        # Disconnect existing engine if any
        if self.engine and self.engine.is_connected:
            print("🔌 Disconnecting existing session...")
            await self.engine.disconnect()
        
        # Create new engine
        print("🔄 Initializing Voice API Engine...")
        self.engine = BaseVoiceAPIEngine(
            api_key=api_key,
            voice=self.current_config["voice"],
            mode=self.current_config["mode"],
            vad_type=self.current_config["vad_type"]
        )
        
        # Connect
        print("🌐 Connecting to OpenAI Realtime API...")
        try:
            await self.engine.connect(
                instructions=self.current_config["instructions"],
                temperature=self.current_config["temperature"]
            )
            print("✅ Connected successfully!\n")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}\n")
            return False
    
    async def change_configuration(self):
        """Interactive configuration change"""
        print("\n🔧 Configuration Menu")
        print("-" * 40)
        
        # Voice selection
        voices = ["alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"]
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"  {i+1}. {voice}")
        
        choice = input("\nSelect voice (1-8) or press Enter to keep current: ").strip()
        if choice and choice.isdigit() and 1 <= int(choice) <= 8:
            self.current_config["voice"] = voices[int(choice)-1]
        
        # Mode selection
        print("\nAvailable modes:")
        print("  1. Text only")
        print("  2. Voice only")
        print("  3. Text and Voice")
        
        choice = input("\nSelect mode (1-3) or press Enter to keep current: ").strip()
        if choice == "1":
            self.current_config["mode"] = VoiceMode.TEXT_ONLY
        elif choice == "2":
            self.current_config["mode"] = VoiceMode.VOICE_ONLY
        elif choice == "3":
            self.current_config["mode"] = VoiceMode.TEXT_AND_VOICE
        
        # Temperature
        temp = input("\nEnter temperature (0.6-1.2) or press Enter to keep current: ").strip()
        if temp:
            try:
                temp_val = float(temp)
                if 0.6 <= temp_val <= 1.2:
                    self.current_config["temperature"] = temp_val
            except ValueError:
                pass
        
        # Instructions
        print(f"\nCurrent instructions: {self.current_config['instructions'][:50]}...")
        new_instructions = input("Enter new instructions or press Enter to keep current: ").strip()
        if new_instructions:
            self.current_config["instructions"] = new_instructions
        
        # Reinitialize with new config
        print("\n🔄 Applying new configuration...")
        await self.initialize_engine()
    
    async def send_text_message(self):
        """Send a text message"""
        if not self.engine or not self.engine.is_connected:
            print("❌ Not connected to Voice API")
            return
        
        message = input("\n📝 Enter your message: ").strip()
        if not message:
            return
        
        print("\n📤 Sending message...")
        start_time = time.time()
        
        response = await self.engine.send_text(message)
        
        if response.success:
            print(f"\n✅ Response received in {response.response_time_ms:.0f}ms")
            if response.text:
                print(f"\n💬 Text: {response.text}")
            if response.audio_bytes:
                print(f"\n🔊 Audio: {response.audio_duration_ms:.0f}ms")
                # Save audio
                filename = f"response_{int(time.time())}.wav"
                self.engine.save_response_audio(response, filename)
                print(f"   Saved to: {filename}")
        else:
            print(f"\n❌ Failed: {response.error}")
        
        input("\nPress Enter to continue...")
    
    async def send_voice_message(self):
        """Record and send a voice message"""
        if not self.engine or not self.engine.is_connected:
            print("❌ Not connected to Voice API")
            return
        
        try:
            import sounddevice as sd
            import numpy as np
            from realtimevoiceapi.audio import AudioConfig
            
            print("\n🎤 Preparing to record...")
            print("   Speak clearly after the countdown")
            
            # Countdown
            for i in range(3, 0, -1):
                print(f"   {i}...")
                await asyncio.sleep(1)
            
            print("   🔴 RECORDING - Speak now!")
            
            # Record 3 seconds
            duration = 3.0
            recording = sd.rec(
                int(duration * AudioConfig.SAMPLE_RATE),
                samplerate=AudioConfig.SAMPLE_RATE,
                channels=AudioConfig.CHANNELS,
                dtype='int16'
            )
            sd.wait()
            
            print("   ✅ Recording complete!")
            
            # Convert to bytes
            audio_bytes = recording.tobytes()
            
            print("\n📤 Sending audio...")
            response = await self.engine.send_audio(audio_bytes)
            
            if response.success:
                print(f"\n✅ Response received in {response.response_time_ms:.0f}ms")
                if response.text:
                    print(f"\n💬 Text: {response.text}")
                if response.audio_bytes:
                    print(f"\n🔊 Audio: {response.audio_duration_ms:.0f}ms")
                    # Save audio
                    filename = f"voice_response_{int(time.time())}.wav"
                    self.engine.save_response_audio(response, filename)
                    print(f"   Saved to: {filename}")
            else:
                print(f"\n❌ Failed: {response.error}")
                
        except ImportError:
            print("\n❌ sounddevice not installed")
            print("   Install with: pip install sounddevice")
        except Exception as e:
            print(f"\n❌ Recording failed: {e}")
        
        input("\nPress Enter to continue...")
    
    async def send_text_and_voice_message(self):
        """Send both text and voice in one interaction"""
        if not self.engine or not self.engine.is_connected:
            print("❌ Not connected to Voice API")
            return
        
        print("\n📝 Text + Voice Message")
        print("-" * 40)
        
        # Get text message
        print("\nYour message (press A then Enter to finish):")
        lines = []
        while True:
            line = input()
            if line.upper() == 'A':
                break
            lines.append(line)
        
        text_message = '\n'.join(lines)
        if not text_message:
            print("❌ No text entered")
            return
        
        print(f"\n📝 Text message: {text_message[:50]}...")
        
        # Record voice
        try:
            import sounddevice as sd
            import numpy as np
            from realtimevoiceapi.audio import AudioConfig
            
            print("\n🎤 Your voice will be recorded in:")
            for i in range(3, 0, -1):
                print(f"   {i}...")
                await asyncio.sleep(1)
            
            print("   🔴 RECORDING - Speak now!")
            
            # Record 3 seconds
            duration = 3.0
            recording = sd.rec(
                int(duration * AudioConfig.SAMPLE_RATE),
                samplerate=AudioConfig.SAMPLE_RATE,
                channels=AudioConfig.CHANNELS,
                dtype='int16'
            )
            sd.wait()
            
            print("   ✅ Recording taken.")
            
            # Convert to bytes
            audio_bytes = recording.tobytes()
            
            print("\n📤 Request sent")
            
            # Send text first
            print("   Sending text...")
            text_response = await self.engine.send_text(text_message, wait_for_response=False)
            
            # Then send audio
            print("   Sending audio...")
            audio_response = await self.engine.send_audio(audio_bytes)
            
            if audio_response.success:
                print(f"\n✅ Response received in {audio_response.response_time_ms:.0f}ms")
                if audio_response.text:
                    print(f"\n💬 Text: {audio_response.text}")
                if audio_response.audio_bytes:
                    print(f"\n🔊 Audio: {audio_response.audio_duration_ms:.0f}ms")
                    # Save audio
                    filename = f"combined_response_{int(time.time())}.wav"
                    self.engine.save_response_audio(audio_response, filename)
                    print(f"   Saved to: {filename}")
            else:
                print(f"\n❌ Failed: {audio_response.error}")
                
        except ImportError:
            print("\n❌ sounddevice not installed")
            print("   Install with: pip install sounddevice")
        except Exception as e:
            print(f"\n❌ Recording failed: {e}")
        
        input("\nPress Enter to continue...")
    
    def get_user_input(self) -> str:
        """Get single character input without pressing Enter (cross-platform)"""
        try:
            # Try Windows-specific method
            if sys.platform == 'win32':
                import msvcrt
                return msvcrt.getch().decode('utf-8').upper()
            else:
                # Unix-based systems
                import termios, tty
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1).upper()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
        except:
            # Fallback to regular input
            return input("\nYour choice: ").strip().upper()
    
    async def run(self):
        """Main application loop"""
        self.running = True
        
        # Initial connection
        if not await self.initialize_engine():
            print("Failed to initialize. Check your API key.")
            return
        
        while self.running:
            self.print_header()
            print("Waiting for command...")
            
            choice = self.get_user_input()
            
            if choice == 'X':
                print("\n👋 Exiting...")
                self.running = False
            elif choice == 'Q':
                await self.change_configuration()
            elif choice == 'R':
                await self.send_text_message()
            elif choice == 'T':
                await self.send_voice_message()
            elif choice == 'Y':
                await self.send_text_and_voice_message()
            else:
                # Invalid choice, just refresh
                continue
        
        # Cleanup
        if self.engine and self.engine.is_connected:
            print("\n🔌 Disconnecting...")
            await self.engine.disconnect()
        
        print("\n✅ Goodbye!")


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
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or add it to your .env file")
        sys.exit(1)
    
    # Run the app
    asyncio.run(main())