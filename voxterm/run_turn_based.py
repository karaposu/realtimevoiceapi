#!/usr/bin/env python3
"""
Turn-based voice interaction using send_recorded_audio
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from realtimevoiceapi import VoiceEngine, VoiceEngineConfig


async def run_turn_based():
    """Run turn-based voice interaction"""
    config = VoiceEngineConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="alloy",
        vad_enabled=False  # We handle turn-taking manually
    )
    
    engine = VoiceEngine(config)
    
    # Setup response handlers
    engine.on_response_text = lambda text: print(text, end="", flush=True)
    engine.on_response_done = lambda: print("\n")
    
    try:
        # Connect
        print("🎙️  Turn-Based Voice Chat")
        print("=" * 40)
        print("Connecting...", end="", flush=True)
        await engine.connect()
        print(" ✅")
        
        print("\n🎯 Turn-Based Conversation")
        print("   Press [ENTER] to start your turn")
        print("   [Q] Return to menu")
        print("   💡 Tip: Use headphones for best experience\n")
        
        while True:
            # Wait for user to start
            cmd = input("\nPress ENTER to speak (or Q to quit): ")
            if cmd.lower() in ['q', 'quit']:
                break
                
            # Record audio
            print("🎤 Your turn! Press ENTER when done...")
            
            # Start listening
            await engine.start_listening()
            
            # Capture audio in a buffer
            audio_buffer = []
            recording = True
            
            # Start recording task
            async def capture_audio():
                while recording:
                    # In a real implementation, this would capture from the audio stream
                    # The engine should provide audio chunks through a callback
                    await asyncio.sleep(0.1)
            
            capture_task = asyncio.create_task(capture_audio())
            
            # Wait for user to press enter
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input, "")
            
            # Stop recording
            recording = False
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass
            
            # Stop listening
            await engine.stop_listening()
            
            # For this example, let's use a simple text prompt instead
            # In a real implementation, we'd use the audio_buffer
            print("\n📤 Processing...")
            
            # Since we don't have actual audio capture working yet,
            # let's get text input as a fallback
            text = input("Type your message (or leave empty to skip): ")
            if text:
                print("\n🤖 AI: ", end="", flush=True)
                await engine.send_text(text)
            
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\nDisconnecting...")
        await engine.disconnect()
        print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(run_turn_based())