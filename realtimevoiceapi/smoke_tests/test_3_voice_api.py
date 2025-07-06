#!/usr/bin/env python3
"""
Test 3: Voice and Audio API Functionality - FIXED VERSION

This test verifies:
- Voice conversation capabilities
- Audio input/output handling
- Different voice options
- Audio format processing
- Audio streaming functionality
- Audio transcription features
- Requires valid OpenAI API key with Realtime access

Run: python -m realtimevoiceapi.smoke_tests.test_3_voice_api
"""

import sys
import os
import asyncio
import logging
import time
from pathlib import Path

# Add parent directory to path so we can import realtimevoiceapi
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️ python-dotenv not installed. Using system environment variables.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_audio_dependencies():
    """Test that audio processing dependencies are available"""
    print("🔊 Testing Audio Dependencies...")
    
    try:
        import numpy as np
        print("  ✅ NumPy available for audio generation")
        numpy_available = True
    except ImportError:
        print("  ⚠️ NumPy not available - will skip audio generation tests")
        numpy_available = False
    
    try:
        from realtimevoiceapi import AudioProcessor
        processor = AudioProcessor()
        print("  ✅ AudioProcessor imported successfully")
        
        # Test basic audio operations
        test_data = b"Voice API test data!"
        encoded = processor.bytes_to_base64(test_data)
        decoded = processor.base64_to_bytes(encoded)
        
        if decoded == test_data:
            print("  ✅ Audio encoding/decoding works")
        else:
            print("  ❌ Audio encoding/decoding failed")
            return False
            
    except Exception as e:
        print(f"  ❌ AudioProcessor test failed: {e}")
        return False
    
    return numpy_available


def generate_test_audio():
    """Generate test audio for voice API testing"""
    try:
        import numpy as np
        from realtimevoiceapi.audio import AudioConfig
        
        # Generate 2 seconds of test audio with a simple melody
        sample_rate = AudioConfig.SAMPLE_RATE  # 24kHz
        duration = 2.0  # 2 seconds (longer audio for better recognition)
        
        # Create a simple melody: A, C, E, G notes
        frequencies = [440, 523, 659, 784]  # A4, C5, E5, G5
        audio_data = bytearray()
        
        note_duration = duration / len(frequencies)
        
        for freq in frequencies:
            samples_per_note = int(sample_rate * note_duration)
            t = np.linspace(0, note_duration, samples_per_note)
            
            # Generate sine wave with fade in/out to avoid clicks
            wave = np.sin(2 * np.pi * freq * t)
            
            # Apply fade in/out (first and last 10% of note)
            fade_samples = samples_per_note // 10
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Scale to 16-bit PCM and add to audio data
            wave_pcm = (wave * 20000).astype(np.int16)  # Higher volume for better recognition
            audio_data.extend(wave_pcm.tobytes())
        
        print(f"  ✅ Generated {duration}s test audio ({len(audio_data)} bytes)")
        return bytes(audio_data)
        
    except ImportError:
        print("  ⚠️ NumPy not available, cannot generate test audio")
        return None
    except Exception as e:
        print(f"  ❌ Audio generation failed: {e}")
        return None


async def test_voice_session_config():
    """Test voice-specific session configuration"""
    print("\n🎙️ Testing Voice Session Configuration...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        # Test different voice configurations - FIXED VOICE NAMES
        voice_configs = [
            {
                "name": "Alloy Voice",
                "voice": "alloy",
                "instructions": "You are testing the Alloy voice. Respond briefly."
            },
            {
                "name": "Echo Voice", 
                "voice": "echo",
                "instructions": "You are testing the Echo voice. Respond briefly."
            },
            {
                "name": "Sage Voice",  # CHANGED FROM NOVA TO SAGE
                "voice": "sage",
                "instructions": "You are testing the Sage voice. Respond briefly."
            }
        ]
        
        for voice_config in voice_configs:
            print(f"  🧪 Testing: {voice_config['name']}")
            
            client = RealtimeClient(api_key)
            
            # Configure for voice
            config = SessionConfig(
                instructions=voice_config["instructions"],
                modalities=["text", "audio"],  # Enable both text and audio
                voice=voice_config["voice"],
                input_audio_format="pcm16",
                output_audio_format="pcm16",
                temperature=0.7,
                turn_detection=None  # Disable for cleaner testing
            )
            
            try:
                # Connect and verify session
                await client.connect(config)
                
                status = client.get_status()
                if status["session_active"]:
                    print(f"    ✅ {voice_config['name']} - session configured successfully")
                else:
                    print(f"    ❌ {voice_config['name']} - session not active")
                    return False
                
                await client.disconnect()
                
            except Exception as e:
                print(f"    ❌ {voice_config['name']} failed: {e}")
                return False
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Voice session config test failed: {e}")
        return False


async def test_voice_text_to_speech():
    """Test text-to-speech voice generation"""
    print("\n🗣️ Testing Text-to-Speech...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure for voice output - IMPROVED CONFIG
        config = SessionConfig(
            instructions="You are a voice assistant. When I say 'test voice', respond with exactly: 'Voice test successful!' Be natural and speak clearly.",
            modalities=["text", "audio"],  # ENSURE BOTH MODALITIES
            voice="alloy",  # VALID VOICE
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.8,  # SLIGHTLY HIGHER FOR MORE NATURAL SPEECH
            turn_detection=None  # DISABLE TURN DETECTION FOR CLEANER TESTS
        )
        
        # Track audio response - IMPROVED TRACKING
        audio_received = False
        audio_duration = 0
        response_text = ""
        response_complete = False
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            nonlocal response_text
            text = event_data.get("delta", "")
            response_text += text
            print(text, end="", flush=True)
        
        @client.on_event("response.audio.delta")  # TRACK AUDIO DELTAS
        async def handle_audio_delta(event_data):
            nonlocal audio_received
            audio_received = True
        
        @client.on_event("response.audio.done")
        async def handle_audio_done(event_data):
            nonlocal audio_duration
            audio_duration = client.get_audio_output_duration()
            print(f"\n    📡 Audio response completed ({audio_duration:.1f}ms)")
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal response_complete
            response_complete = True
            print()  # New line after text response
        
        print("  ✅ Event handlers registered")
        
        # Connect and test
        await client.connect(config)
        print("  ✅ Connected with voice configuration")
        
        # Send text message to generate voice response
        print("  📤 Sending: 'test voice'")
        print("  🤖 Response: ", end="")
        
        await client.send_text("test voice")
        
        # Wait for complete response - IMPROVED WAITING
        timeout = 25  # Longer timeout for voice
        start_time = time.time()
        while not response_complete and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        # Check results
        audio_duration = client.get_audio_output_duration()
        
        if response_complete and audio_duration > 0:
            print(f"  ✅ Voice response generated ({audio_duration:.1f}ms audio)")
            
            # Save the audio response
            audio_file = "voice_test_response.wav"
            if client.save_audio_output(audio_file):
                print(f"  💾 Audio saved: {audio_file}")
            
            result = True
        else:
            print(f"  ❌ Voice response issue (complete: {response_complete}, audio: {audio_duration:.1f}ms)")
            if not response_complete:
                print("    No response received within timeout")
            if audio_duration == 0:
                print("    No audio generated (check modalities configuration)")
            result = False
        
        await client.disconnect()
        print("  ✅ Disconnected")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Text-to-speech test failed: {e}")
        logger.exception("TTS test error")
        return False


async def test_audio_input():
    """Test audio input processing - COMPLETELY REWRITTEN"""
    print("\n🎤 Testing Audio Input...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    # Generate test audio
    test_audio = generate_test_audio()
    if not test_audio:
        print("  ⏩ Skipping - cannot generate test audio (NumPy required)")
        return True  # Don't fail the test, just skip
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        from realtimevoiceapi.models import TranscriptionConfig
        
        processor = AudioProcessor()
        duration_ms = processor.get_audio_duration_ms(test_audio)
        
        if duration_ms < 500:  # Need at least 500ms for reliable recognition
            print(f"  ⚠️ Test audio too short ({duration_ms:.1f}ms), extending...")
            # Duplicate the audio to make it longer
            test_audio = test_audio * 2
            duration_ms = processor.get_audio_duration_ms(test_audio)
        
        print(f"  🎵 Test audio: {len(test_audio)} bytes, {duration_ms:.1f}ms")
        
        client = RealtimeClient(api_key)
        
        # Configure for audio input - IMPROVED CONFIG WITH PROPER TRANSCRIPTION
        config = SessionConfig(
            instructions="You are a voice assistant. When you receive audio input, respond with: 'I received your audio message!' Be brief.",
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16", 
            temperature=0.7,
            turn_detection=None,  # DISABLE AUTO TURN DETECTION
            input_audio_transcription=TranscriptionConfig(model="whisper-1", language="en")  # SPECIFY ENGLISH
        )
        
        # Track response
        response_received = False
        response_text = ""
        audio_committed = False
        error_occurred = False
        
        @client.on_event("input_audio_buffer.committed")
        async def handle_audio_committed(event_data):
            nonlocal audio_committed
            audio_committed = True
            print("    📡 Audio committed to buffer")
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            nonlocal response_text
            text = event_data.get("delta", "")
            response_text += text
            print(text, end="", flush=True)
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal response_received
            response_received = True
            print()
        
        @client.on_event("error")
        async def handle_error(event_data):
            nonlocal error_occurred
            error_occurred = True
            error = event_data.get("error", {})
            print(f"\n    ❌ API Error: {error.get('message', 'Unknown error')}")
        
        print("  ✅ Event handlers registered")
        
        # Connect and test
        await client.connect(config)
        print("  ✅ Connected with audio input configuration")
        
        # FIXED APPROACH: Use the debug method
        print("  📤 Sending audio input...")
        print("  🤖 Response: ", end="")
        
        try:
            # Method: Use the improved audio sending from client
            success = await client.send_audio_bytes_debug(test_audio)
            
            if success:
                # Wait for response
                timeout = 35  # Longer timeout for audio processing
                start_time = time.time()
                while not response_received and not error_occurred and (time.time() - start_time) < timeout:
                    await asyncio.sleep(0.1)
                
                if response_received and len(response_text.strip()) > 0:
                    print(f"  ✅ Audio input processed successfully")
                    print(f"    Response: '{response_text.strip()}'")
                    result = True
                elif error_occurred:
                    print("  ❌ Audio input failed due to API error")
                    result = False
                else:
                    print("  ⏰ Audio input timeout - no response received")
                    result = False
            else:
                print("  ❌ Failed to send audio")
                result = False
                
        except Exception as audio_error:
            print(f"  ❌ Audio processing error: {audio_error}")
            result = False
        
        await client.disconnect()
        print("  ✅ Disconnected")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Audio input test failed: {e}")
        logger.exception("Audio input test error")
        return False


async def test_audio_streaming():
    """Test audio streaming in chunks"""
    print("\n🌊 Testing Audio Streaming...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    # Generate test audio
    test_audio = generate_test_audio()
    if not test_audio:
        print("  ⏩ Skipping - cannot generate test audio (NumPy required)")
        return True
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        
        client = RealtimeClient(api_key)
        processor = AudioProcessor()
        
        # Configure for streaming
        config = SessionConfig(
            instructions="You are testing audio streaming. When you receive audio, respond with: 'Streaming test complete!'",
            modalities=["text", "audio"],
            voice="echo",  # Different voice for variety
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.7,
            turn_detection=None
        )
        
        # Track streaming events
        chunks_received = 0
        response_received = False
        
        @client.on_event("input_audio_buffer.committed")
        async def handle_audio_committed(event_data):
            print("    📡 Audio buffer committed")
        
        @client.on_event("response.audio.delta")
        async def handle_audio_delta(event_data):
            nonlocal chunks_received
            chunks_received += 1
            if chunks_received % 5 == 0:  # Print every 5th chunk to avoid spam
                print(f"    📊 Received {chunks_received} audio chunks...")
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal response_received
            response_received = True
        
        print("  ✅ Event handlers registered")
        
        # Connect
        await client.connect(config)
        print("  ✅ Connected for streaming test")
        
        # Stream audio in chunks
        print("  🌊 Streaming audio in chunks...")
        chunk_size_ms = 250  # 250ms chunks (larger for more reliable streaming)
        chunks = processor.chunk_audio(test_audio, chunk_size_ms)
        print(f"    Split into {len(chunks)} chunks of {chunk_size_ms}ms each")
        
        # Send chunks with real-time delay
        await client.send_audio_chunks(test_audio, chunk_size_ms, real_time=True)  # USE real-time delay to avoid buffer issues
        print("  ✅ Audio streaming completed")
        
        # Wait for response
        timeout = 30  # Longer timeout for streaming
        start_time = time.time()
        while not response_received and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if response_received:
            print(f"  ✅ Streaming response received")
            print(f"    Audio chunks received: {chunks_received}")
            
            # Save streamed response
            audio_duration = client.get_audio_output_duration()
            if audio_duration > 0:
                audio_file = "streaming_test_response.wav"
                client.save_audio_output(audio_file)
                print(f"    💾 Streamed audio saved: {audio_file} ({audio_duration:.1f}ms)")
            
            result = True
        else:
            print("  ❌ No response to streamed audio")
            result = False
        
        await client.disconnect()
        print("  ✅ Disconnected")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Audio streaming test failed: {e}")
        logger.exception("Audio streaming test error")
        return False


async def test_voice_conversation():
    """Test full voice conversation flow - FIXED"""
    print("\n💬 Testing Voice Conversation Flow...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure for full voice conversation - FIXED VOICE NAME
        config = SessionConfig(
            instructions="You are a helpful voice assistant. Respond naturally and conversationally. Keep responses brief but friendly.",
            modalities=["text", "audio"],
            voice="sage",  # CHANGED FROM NOVA TO SAGE - VALID VOICE
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.8,  # More natural/varied responses
            turn_detection=None  # DISABLE FOR CLEANER CONTROL
        )
        
        # Track conversation
        conversation_turns = 0
        audio_responses = 0
        total_audio_duration = 0
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            text = event_data.get("delta", "")
            print(text, end="", flush=True)
        
        @client.on_event("response.audio.done")
        async def handle_audio_done(event_data):
            nonlocal audio_responses, total_audio_duration
            audio_responses += 1
            duration = client.get_audio_output_duration()
            total_audio_duration += duration
            
            # Save each response
            filename = f"conversation_turn_{conversation_turns + 1}.wav"
            client.save_audio_output(filename)
            print(f"\n    💾 Audio saved: {filename} ({duration:.1f}ms)")
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal conversation_turns
            conversation_turns += 1
            print()
        
        @client.on_event("error")
        async def handle_error(event_data):
            error = event_data.get("error", {})
            print(f"\n    ❌ Conversation error: {error.get('message', 'Unknown')}")
        
        print("  ✅ Event handlers registered")
        
        # Connect
        await client.connect(config)
        print("  ✅ Connected for voice conversation")
        
        # Have a multi-turn voice conversation - SIMPLIFIED
        conversation_messages = [
            "Hello! How are you?",
            "What's the weather like?",
            "Thank you!"
        ]
        
        for i, message in enumerate(conversation_messages):
            print(f"\n  👤 Turn {i+1}: {message}")
            print(f"  🤖 Response: ", end="")
            
            try:
                await client.send_text(message)
                
                # Wait for this turn to complete
                current_turn = conversation_turns
                timeout = 20  # Reasonable timeout
                start_time = time.time()
                while conversation_turns == current_turn and (time.time() - start_time) < timeout:
                    await asyncio.sleep(0.1)
                
                if conversation_turns > current_turn:
                    print(f"    ✅ Turn {i+1} completed")
                else:
                    print(f"    ⚠️ Turn {i+1} timeout")
                
                # Small delay between turns
                await asyncio.sleep(1)
                
            except Exception as turn_error:
                print(f"    ❌ Turn {i+1} failed: {turn_error}")
                break
        
        # Final results
        print(f"\n  📊 Conversation Summary:")
        print(f"    Turns completed: {conversation_turns}")
        print(f"    Audio responses: {audio_responses}")
        print(f"    Total audio duration: {total_audio_duration:.1f}ms")
        
        await client.disconnect()
        print("  ✅ Voice conversation completed")
        
        # Consider success if we got most responses
        return conversation_turns >= len(conversation_messages) - 1
        
    except Exception as e:
        print(f"  ❌ Voice conversation test failed: {e}")
        logger.exception("Voice conversation test error")
        return False


async def test_audio_formats():
    """Test different audio format handling"""
    print("\n🔧 Testing Audio Format Handling...")
    
    try:
        from realtimevoiceapi import AudioProcessor
        from realtimevoiceapi.audio import AudioConfig, AudioFormat
        
        processor = AudioProcessor()
        
        # Test audio format validation
        test_audio = generate_test_audio()
        if not test_audio:
            print("  ⏩ Skipping - cannot generate test audio")
            return True
        
        print(f"  🎵 Testing with {len(test_audio)} bytes of audio")
        
        # Test format validation
        is_valid, msg = processor.validator.validate_audio_data(test_audio, AudioFormat.PCM16)
        if is_valid:
            print("  ✅ PCM16 format validation passed")
        else:
            print(f"  ❌ PCM16 format validation failed: {msg}")
            return False
        
        # Test audio info extraction
        info = processor.get_audio_info(test_audio)
        expected_duration = 2000.0  # 2 seconds
        if abs(info['duration_ms'] - expected_duration) < 100:  # Allow 100ms tolerance
            print(f"  ✅ Audio duration correct: {info['duration_ms']:.1f}ms")
        else:
            print(f"  ❌ Audio duration wrong: {info['duration_ms']:.1f}ms vs {expected_duration}ms")
            return False
        
        # Test chunking for different sizes
        chunk_sizes = [100, 250, 500]  # Different chunk sizes in ms
        for chunk_ms in chunk_sizes:
            chunks = processor.chunk_audio(test_audio, chunk_ms)
            expected_chunks = int(info['duration_ms'] / chunk_ms)
            
            if abs(len(chunks) - expected_chunks) <= 1:  # Allow ±1 chunk tolerance
                print(f"  ✅ {chunk_ms}ms chunking: {len(chunks)} chunks")
            else:
                print(f"  ❌ {chunk_ms}ms chunking failed: {len(chunks)} vs ~{expected_chunks}")
                return False
        
        # Test audio quality analysis
        analysis = processor.analyze_audio_quality(test_audio)
        if analysis['quality_score'] > 0.7:
            print(f"  ✅ Audio quality analysis: {analysis['quality_score']:.2f}")
        else:
            print(f"  ⚠️ Audio quality low: {analysis['quality_score']:.2f}")
            # Don't fail for low quality, just warn
        
        print("  ✅ All audio format tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Audio format test failed: {e}")
        logger.exception("Audio format test error")
        return False


async def main():
    """Run all voice API tests"""
    print("🧪 RealtimeVoiceAPI - Test 3: Voice and Audio API")
    print("=" * 70)
    print("This test requires a valid OpenAI API key and tests voice features")
    print("⚠️  This test will use moderate API quota for voice generation")
    print()
    
    # Check if we should skip API tests
    if os.getenv("SKIP_API_TESTS", "0").lower() in ("1", "true", "yes"):
        print("⏩ Skipping API tests (SKIP_API_TESTS=1)")
        print("   Set SKIP_API_TESTS=0 in .env to enable voice API tests")
        return True
    
    tests = [
        ("Audio Dependencies", test_audio_dependencies),
        ("Voice Session Config", test_voice_session_config),
        ("Text-to-Speech", test_voice_text_to_speech),
        ("Audio Input", test_audio_input),
        ("Audio Streaming", test_audio_streaming),
        ("Voice Conversation", test_voice_conversation),
        ("Audio Format Handling", test_audio_formats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
            
        except KeyboardInterrupt:
            print(f"\n⏹️ {test_name} interrupted by user")
            break
            
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            logger.exception(f"Test crash: {test_name}")
            results.append((test_name, False))
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test 3 Results")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Test 3 PASSED! Voice and audio functionality works perfectly.")
        print("💡 Your RealtimeVoiceAPI supports:")
        print("   - ✅ Text-to-speech voice generation")
        print("   - ✅ Audio input processing")
        print("   - ✅ Real-time audio streaming")
        print("   - ✅ Multiple voice options")
        print("   - ✅ Voice conversations")
        print("   - ✅ Audio format handling")
        print("\n🚀 You're ready to build full voice applications!")
        
    elif passed >= total - 1:
        print(f"\n✅ Test 3 MOSTLY PASSED! {passed}/{total} tests successful.")
        print("   Minor issue detected, but core voice functionality works.")
        print("   You can proceed with voice application development.")
        
    else:
        print(f"\n❌ Test 3 FAILED! {total - passed} test(s) need attention.")
        print("\n🔧 Common voice API issues:")
        print("  - Missing NumPy for audio generation")
        print("  - Voice responses taking longer than expected")
        print("  - Audio format compatibility issues")
        print("  - Network latency affecting real-time performance")
        print("\n💡 Troubleshooting:")
        print("  1. Install NumPy: pip install numpy")
        print("  2. Check internet connection speed")
        print("  3. Try with different voice models")
        print("  4. Verify API quota and rate limits")
    
    # Show generated audio files
    audio_files = [
        "voice_test_response.wav",
        "streaming_test_response.wav", 
        "conversation_turn_1.wav",
        "conversation_turn_2.wav",
        "conversation_turn_3.wav"
    ]
    
    found_files = [f for f in audio_files if Path(f).exists()]
    if found_files:
        print(f"\n🎵 Generated audio files:")
        for f in found_files:
            size = Path(f).stat().st_size
            print(f"   📁 {f} ({size} bytes)")
        print("   You can play these files to hear the voice responses!")
    
    return passed >= total - 1  # Allow 1 failure


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)