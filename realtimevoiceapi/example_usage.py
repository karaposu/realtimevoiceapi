#!/usr/bin/env python3
"""
RealtimeVoiceAPI example usage - placed in the module directory

This file is intended to be run as: python -m realtimevoiceapi.example_usage
"""

import asyncio
import os
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️ python-dotenv not installed. Using system environment variables.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def audio_processing_example():
    """Example: Audio processing without API"""
    print("\n🎵 Audio Processing Example")
    print("-" * 40)
    
    try:
        from . import AudioProcessor
        
        processor = AudioProcessor()
        
        # Test with NumPy if available
        try:
            import numpy as np
            
            # Generate test audio
            print("Generating test audio...")
            sample_rate = 24000
            duration = 2.0
            
            # Generate a simple melody
            frequencies = [440, 523, 659, 784]  # A, C, E, G
            audio_data = bytearray()
            
            for freq in frequencies:
                t = np.linspace(0, duration/4, int(sample_rate * duration / 4))
                wave = np.sin(2 * np.pi * freq * t) * 0.5
                wave = (wave * 32767).astype(np.int16)
                audio_data.extend(wave.tobytes())
            
            audio_bytes = bytes(audio_data)
            
            # Get audio info
            info = processor.get_audio_info(audio_bytes)
            print(f"📊 Audio info: {info['duration_ms']:.1f}ms, {info['size_bytes']} bytes")
            
            # Analyze quality
            analysis = processor.analyze_audio_quality(audio_bytes)
            print(f"🔍 Quality score: {analysis['quality_score']:.2f}")
            print(f"📈 Peak level: {analysis['peak_level']:.3f}")
            
            # Save as WAV
            processor.save_wav_file(audio_bytes, "test_melody.wav")
            print("💾 Saved as test_melody.wav")
            
            # Test chunking
            chunks = processor.chunk_audio(audio_bytes, 500)  # 500ms chunks
            print(f"🔗 Split into {len(chunks)} chunks of 500ms each")
            
            # Test base64 encoding
            b64_audio = processor.bytes_to_base64(audio_bytes)
            print(f"📝 Base64 encoded: {len(b64_audio)} characters")
            
        except ImportError:
            print("ℹ️ NumPy not available, using simple audio test")
            
            # Simple test without NumPy
            test_data = b"Simple audio test data" * 1000
            b64_data = processor.bytes_to_base64(test_data)
            decoded = processor.base64_to_bytes(b64_data)
            
            print(f"✅ Base64 test: {len(test_data)} -> {len(b64_data)} -> {len(decoded)}")
            print(f"🔍 Round-trip successful: {test_data == decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return False


async def text_conversation_example():
    """Example: Simple text conversation"""
    print("\n🗣️ Text Conversation Example")
    print("-" * 40)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   Add it to your .env file or export it")
        return False
    
    try:
        from . import RealtimeClient, SessionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure session for text only
        config = SessionConfig(
            instructions="You are a helpful assistant. Keep responses concise and friendly.",
            modalities=["text"],
            temperature=0.7
        )
        
        # Track response
        full_response = ""
        response_complete = False
        
        # Setup event handlers with corrected decorator syntax
        @client.on_event("response.text.delta")
        async def handle_text(event_data):
            nonlocal full_response
            text = event_data.get("delta", "")
            full_response += text
            print(text, end="", flush=True)
        
        @client.on_event("response.done")
        async def handle_done(event_data):
            nonlocal response_complete
            response_complete = True
            print("\n")
        
        @client.on_event("error")
        async def handle_error(event_data):
            error = event_data.get("error", {})
            print(f"\n❌ Error: {error}")
        
        # Connect and have conversation
        await client.connect(config)
        
        messages = [
            "Hello! How are you today?",
            "What's 2+2?",
            "Tell me a very short joke.",
            "Goodbye!"
        ]
        
        for message in messages:
            print(f"👤 User: {message}")
            print("🤖 Assistant: ", end="")
            full_response = ""
            response_complete = False
            
            await client.send_text(message)
            
            # Wait for response
            timeout = 10
            elapsed = 0
            while not response_complete and elapsed < timeout:
                await asyncio.sleep(0.1)
                elapsed += 0.1
        
        await client.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Text conversation failed: {e}")
        return False


async def voice_conversation_example():
    """Example: Voice conversation with audio input/output"""
    print("\n🎤 Voice Conversation Example")
    print("-" * 40)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return False
    
    try:
        from . import RealtimeClient, SessionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure session for voice
        config = SessionConfig(
            instructions="You are a friendly voice assistant. Speak naturally and be expressive.",
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.8
        )
        
        # Track responses
        response_count = 0
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            text = event_data.get("delta", "")
            print(text, end="", flush=True)
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal response_count
            response_count += 1
            
            # Save audio output
            audio_file = f"response_{response_count}.wav"
            if client.save_audio_output(audio_file):
                print(f"\n💾 Audio saved: {audio_file}")
            print()
        
        # Connect
        await client.connect(config)
        
        # Send text message first
        print("👤 User: Hi there! Can you introduce yourself?")
        print("🤖 Assistant: ", end="")
        await client.send_text("Hi there! Can you introduce yourself?")
        await asyncio.sleep(5)
        
        # Generate and send audio (synthetic for demo)
        try:
            import numpy as np
            
            sample_rate = 24000
            duration = 2.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_array = np.sin(2 * np.pi * frequency * t) * 0.3
            audio_array = (audio_array * 32767).astype(np.int16)
            test_audio = audio_array.tobytes()
            
            print("👤 User: [Sending audio - 440Hz tone]")
            print("🤖 Assistant: ", end="")
            await client.send_audio_bytes(test_audio)
            await asyncio.sleep(5)
            
        except ImportError:
            print("ℹ️ NumPy not available, skipping audio generation")
        
        await client.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Voice conversation failed: {e}")
        return False


async def main():
    """Run all examples"""
    print("🚀 RealtimeVoiceAPI Usage Examples (Fixed)")
    print("=" * 50)
    
    examples = [
        ("Audio Processing (No API)", audio_processing_example),
        ("Text Conversation", text_conversation_example),
        ("Voice Conversation", voice_conversation_example)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            await example_func()
            print("✅ Example completed successfully")
            
        except KeyboardInterrupt:
            print("\n⏹️ Example interrupted by user")
            break
            
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            logger.exception(f"Example error: {name}")
        
        # Pause between examples
        if name != examples[-1][0]:  # Not the last example
            print("\n⏳ Waiting 2 seconds before next example...")
            await asyncio.sleep(2)
    
    print(f"\n{'='*50}")
    print("🎉 All examples completed!")
    print("\n💡 Next steps:")
    print("  - Modify these examples for your use case")
    print("  - Check the examples/ directory for more detailed demos")
    print("  - Read the documentation for advanced features")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())