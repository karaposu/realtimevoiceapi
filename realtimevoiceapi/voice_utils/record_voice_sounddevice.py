
# here is record_voice_sounddevice.py

#!/usr/bin/env python3
"""
Simple example: Record your voice and use it with RealtimeVoiceAPI
"""

import asyncio
import os
from pathlib import Path

# Check for recording capability
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠️  Install sounddevice for recording: pip install sounddevice soundfile")


def record_my_voice(duration=3):
    """Simple voice recording"""
    if not HAS_AUDIO:
        return None
    
    print("🎤 Quick Voice Recording")
    print(f"Duration: {duration} seconds")
    print("Starting in 3...")
    
    import time
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    print("🔴 RECORDING - Speak now!")
    
    # Record at 24kHz mono (API requirement)
    recording = sd.rec(
        int(duration * 24000),
        samplerate=24000,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    
    print("✅ Done!")
    
    # Save it
    filename = "my_voice.wav"
    sf.write(filename, recording, 24000, subtype='PCM_16')
    
    print(f"💾 Saved as: {filename}")
    return filename


async def test_with_voice():
    """Test the API with your voice recording"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set your OPENAI_API_KEY environment variable")
        return
    
    # Check for voice file
    voice_file = "my_voice.wav"
    if not Path(voice_file).exists():
        print("❌ No voice file found. Recording one now...")
        voice_file = record_my_voice(duration=3)
        if not voice_file:
            return
    
    print(f"\n🎵 Using voice file: {voice_file}")
    
    # Use the API
    from realtimevoiceapi import RealtimeClient, SessionConfig
    from realtimevoiceapi.models import TurnDetectionConfig
    
    client = RealtimeClient(api_key)
    
    # Configure for voice
    config = SessionConfig(
        instructions="You are a helpful assistant. Listen and respond naturally.",
        modalities=["text", "audio"],
        voice="alloy",
        turn_detection=TurnDetectionConfig(
            type="server_vad",  # or "semantic_vad" 
            threshold=0.5,      # Only for server_vad
            create_response=True
        )
    )
    
    try:
        # Connect
        print("\n🔌 Connecting to OpenAI...")
        await client.connect(config)
        print("✅ Connected!")
        
        # Send your voice
        print("\n📤 Sending your voice...")
        text, audio = await client.send_audio_and_wait_for_response(
            client.audio_processor.load_wav_file(voice_file)
        )
        
        # Show results
        print("\n📝 Results:")
        if text:
            print(f"Text response: {text}")
        
        if audio:
            # Save response
            output = "api_response.wav"
            client.audio_processor.save_wav_file(audio, output)
            print(f"Audio saved: {output}")
            
            # Play it back
            if HAS_AUDIO:
                print("\n🔊 Playing response...")
                data, fs = sf.read(output)
                sd.play(data, fs)
                sd.wait()
        
        print("\n✅ Success!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await client.disconnect()


def main():
    """Main menu"""
    print("🎙️ RealtimeVoiceAPI - Simple Voice Test")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Record new voice")
        print("2. Test with existing recording")
        print("3. Exit")
        
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == "1":
            record_my_voice()
        
        elif choice == "2":
            asyncio.run(test_with_voice())
        
        elif choice == "3":
            break
        
        else:
            print("Invalid choice")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()