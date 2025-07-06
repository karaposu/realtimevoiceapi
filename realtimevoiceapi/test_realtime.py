# test_realtime.py
import asyncio
import logging
import os
from realtimevoiceapi import RealtimeClient, SessionConfig

async def test_text_conversation():
    """Test basic text conversation"""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    print("🔗 Starting RealtimeVoiceAPI test...")
    
    # Create client
    client = RealtimeClient(api_key=api_key)
    
    # Configure session for text-only
    config = SessionConfig(
        instructions="You are a helpful assistant. Keep responses short and friendly.",
        modalities=["text"],  # Text only for first test
        voice="alloy"
    )
    
    # Track responses
    full_response = ""
    
    # Setup event handlers
    @client.on_event("response.text.delta")
    async def handle_text_delta(event_data):
        nonlocal full_response
        text = event_data.get("delta", "")
        print(text, end="", flush=True)
        full_response += text
    
    @client.on_event("response.done")
    async def handle_response_done(event_data):
        print(f"\n✅ Response complete!")
        print(f"📊 Usage: {event_data.get('response', {}).get('usage', {})}")
        print("=" * 50)
    
    @client.on_event("error")
    async def handle_error(event_data):
        error = event_data.get("error", {})
        print(f"❌ Error: {error}")
    
    try:
        # Connect
        print("🔌 Connecting to OpenAI Realtime API...")
        success = await client.connect(config)
        
        if not success:
            print("❌ Failed to connect")
            return
        
        print("✅ Connected successfully!")
        
        # Test messages
        test_messages = [
            "Hello! Can you hear me?",
            "What's 2 + 2?",
            "Tell me a very short joke."
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n🗣️ User ({i}/3): {message}")
            print("🤖 Assistant: ", end="")
            
            full_response = ""
            await client.send_text(message)
            
            # Wait for response
            await asyncio.sleep(4)
            
            if not full_response.strip():
                print("⚠️ No response received")
        
        print(f"\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔌 Disconnecting...")
        await client.disconnect()
        print("👋 Disconnected.")

async def test_voice_features():
    """Test voice features (requires audio file)"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("\n🎵 Testing voice features...")
    
    client = RealtimeClient(api_key=api_key)
    
    # Voice configuration
    config = SessionConfig(
        instructions="You are a friendly voice assistant. Speak naturally.",
        modalities=["text", "audio"],
        voice="alloy",
        input_audio_format="pcm16",
        output_audio_format="pcm16"
    )
    
    response_count = 0
    
    @client.on_event("response.text.delta")
    async def handle_text(event_data):
        print(event_data.get("delta", ""), end="")
    
    @client.on_event("response.done")
    async def handle_done(event_data):
        nonlocal response_count
        response_count += 1
        
        # Save audio
        audio_file = f"realtime_response_{response_count}.wav"
        if client.save_audio_output(audio_file):
            print(f"\n🎵 Audio saved: {audio_file}")
        else:
            print(f"\n⚠️ No audio to save")
    
    try:
        await client.connect(config)
        print("✅ Connected for voice test")
        
        # Test with text first
        print("\n🗣️ User: Hello! Please respond with both text and voice.")
        print("🤖 Assistant: ", end="")
        await client.send_text("Hello! Please respond with both text and voice.")
        await asyncio.sleep(5)
        
        # Check if we have an audio input file to test
        test_audio_file = "test_input.wav"
        if os.path.exists(test_audio_file):
            print(f"\n🎵 Sending audio file: {test_audio_file}")
            print("🤖 Assistant: ", end="")
            await client.send_audio_file(test_audio_file)
            await asyncio.sleep(5)
        else:
            print(f"\n💡 To test audio input, create '{test_audio_file}' (PCM16, 24kHz, mono)")
        
        print("\n🎉 Voice test completed!")
        
    except Exception as e:
        print(f"❌ Voice test error: {e}")
        
    finally:
        await client.disconnect()

def main():
    """Run tests"""
    print("🚀 RealtimeVoiceAPI Test Suite")
    print("=" * 40)
    
    # Check requirements
    try:
        import websockets
        import aiohttp
        print("✅ Dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install websockets aiohttp numpy pydub")
        return
    
    # Run tests
    asyncio.run(test_text_conversation())
    
    # Ask if user wants to test voice
    test_voice = input("\n🎵 Test voice features? (y/n): ").lower().startswith('y')
    if test_voice:
        asyncio.run(test_voice_features())
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
