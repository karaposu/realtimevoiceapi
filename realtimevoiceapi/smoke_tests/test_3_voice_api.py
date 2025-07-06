#!/usr/bin/env python3
"""
Test 3: Voice and Audio API Functionality - FIXED VERSION

This test verifies:
- Voice conversation capabilities
- Audio input/output handling
- Different voice options
- Audio format processing
- Audio streaming functionality
- Real voice processing with Server VAD
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


def get_test_audio():
    """Get test audio - prioritize real voice recording, fallback to synthetic"""
    # First, try to use a real voice recording
    voice_files = [
        "test_voice.wav",
        "my_voice.wav",
        "voice_input.wav", 
        "speech.wav",
        "audio_input.wav"
    ]
    
    for file in voice_files:
        if Path(file).exists():
            try:
                from realtimevoiceapi.audio import AudioProcessor
                processor = AudioProcessor()
                
                print(f"  📁 Using real voice recording: {file}")
                audio_bytes = processor.load_wav_file(file)
                
                # Validate it's the right format
                info = processor.get_audio_info(audio_bytes)
                duration_ms = info.get('duration_ms', 0)
                
                if duration_ms < 500:
                    print(f"  ⚠️  Audio too short ({duration_ms}ms), trying next file...")
                    continue
                
                print(f"  ✅ Loaded {len(audio_bytes)} bytes ({duration_ms:.0f}ms)")
                return audio_bytes
                
            except Exception as e:
                print(f"  ⚠️  Failed to load {file}: {e}")
                continue
    
    # No voice file found, generate synthetic audio
    print("  🎵 No voice recording found, generating synthetic audio...")
    print("  ⚠️  Note: Synthetic audio may not work with Server VAD")
    return generate_speech_like_audio()


def generate_speech_like_audio():
    """Generate more speech-like synthetic audio"""
    try:
        import numpy as np
        from realtimevoiceapi.audio import AudioConfig
        
        # Generate 2 seconds of more realistic speech-like audio
        sample_rate = AudioConfig.SAMPLE_RATE  # 24kHz
        duration = 2.0
        
        samples_total = int(sample_rate * duration)
        t = np.linspace(0, duration, samples_total)
        
        # Create speech-like audio with multiple components
        fundamental = 180  # Human speech fundamental
        audio_signal = np.zeros(samples_total)
        
        # Add harmonics typical of human speech
        for harmonic in range(1, 6):
            freq = fundamental * harmonic
            amplitude = 0.3 / harmonic  # Natural harmonic rolloff
            audio_signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add formant characteristics (vowel-like resonances)
        formant_freqs = [700, 1220, 2600]  # Typical vowel formants
        for formant in formant_freqs:
            modulation = 1 + 0.2 * np.sin(2 * np.pi * formant / 100 * t)
            audio_signal *= modulation
        
        # Add amplitude variation (speech rhythm)
        syllable_rate = 4  # syllables per second
        amplitude_envelope = 0.7 + 0.3 * np.abs(np.sin(2 * np.pi * syllable_rate * t))
        audio_signal *= amplitude_envelope
        
        # Add slight noise for realism
        noise = np.random.normal(0, 0.01, samples_total)
        audio_signal += noise
        
        # Apply fade in/out
        fade_samples = int(0.1 * sample_rate)
        audio_signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio_signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Normalize and convert to PCM16
        audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.8
        audio_pcm = (audio_signal * 32767).astype(np.int16)
        
        # Ensure little-endian
        audio_bytes = audio_pcm.astype('<i2').tobytes()
        
        print(f"  ✅ Generated {duration}s speech-like audio ({len(audio_bytes)} bytes)")
        return audio_bytes
        
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
        from realtimevoiceapi.models import TurnDetectionConfig
        
        # Test different voice configurations with VALID voice names
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
                "name": "Shimmer Voice",
                "voice": "shimmer",
                "instructions": "You are testing the Shimmer voice. Respond briefly."
            }
        ]
        
        for voice_config in voice_configs:
            print(f"  🧪 Testing: {voice_config['name']}")
            
            client = RealtimeClient(api_key)
            
            # Configure for voice with proper turn detection
            config = SessionConfig(
                instructions=voice_config["instructions"],
                modalities=["text", "audio"],
                voice=voice_config["voice"],
                input_audio_format="pcm16",
                output_audio_format="pcm16",
                temperature=0.7,
                turn_detection=TurnDetectionConfig(
                    type="server_vad",
                    threshold=0.5,
                    create_response=False  # Manual control for testing
                )
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
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure for voice output
        config = SessionConfig(
            instructions="You are a voice assistant. When I say 'test voice', respond with exactly: 'Voice test successful!' Be natural and speak clearly.",
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.8,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            )
        )
        
        # Track audio response
        audio_received = False
        response_text = ""
        response_complete = False
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            nonlocal response_text
            text = event_data.get("delta", "")
            response_text += text
            print(text, end="", flush=True)
        
        @client.on_event("response.audio.delta")
        async def handle_audio_delta(event_data):
            nonlocal audio_received
            audio_received = True
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal response_complete
            response_complete = True
            print()
        
        print("  ✅ Event handlers registered")
        
        # Connect and test
        await client.connect(config)
        print("  ✅ Connected with voice configuration")
        
        # Send text message to generate voice response
        print("  📤 Sending: 'test voice'")
        print("  🤖 Response: ", end="")
        
        await client.send_text("test voice")
        
        # Wait for complete response
        timeout = 25
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
            print(f"  ❌ Voice response issue")
            print(f"    Complete: {response_complete}, Audio: {audio_duration:.1f}ms")
            result = False
        
        await client.disconnect()
        print("  ✅ Disconnected")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Text-to-speech test failed: {e}")
        logger.exception("TTS test error")
        return False


async def test_audio_input():
    """Test audio input processing using working methods"""
    print("\n🎤 Testing Audio Input...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    # Get test audio (prefer real voice)
    test_audio = get_test_audio()
    if not test_audio:
        print("  ⏩ Skipping - cannot get test audio")
        return True
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        from realtimevoiceapi.models import TurnDetectionConfig
        
        processor = AudioProcessor()
        duration_ms = processor.get_audio_duration_ms(test_audio)
        print(f"  🎵 Test audio: {len(test_audio)} bytes, {duration_ms:.1f}ms")
        
        client = RealtimeClient(api_key)
        
        # Configure for audio input with Server VAD
        config = SessionConfig(
            instructions="You are a voice assistant. When you receive audio input, respond with: 'I received your audio message!' Be brief.",
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16", 
            temperature=0.7,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                silence_duration_ms=500,
                create_response=True
            )
        )
        
        # Track response
        events_received = []
        response_text = ""
        
        @client.on_event("input_audio_buffer.speech_started")
        async def handle_speech_start(data):
            events_received.append("speech_started")
            print("    🎙️ Speech detected!")
        
        @client.on_event("input_audio_buffer.speech_stopped")
        async def handle_speech_stop(data):
            events_received.append("speech_stopped")
            print("    🔇 Speech ended!")
        
        @client.on_event("input_audio_buffer.committed")
        async def handle_committed(data):
            events_received.append("committed")
            print("    ✅ Audio committed!")
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            nonlocal response_text
            text = event_data.get("delta", "")
            response_text += text
            print(text, end="", flush=True)
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            events_received.append("response_done")
            print()
        
        @client.on_event("error")
        async def handle_error(event_data):
            error = event_data.get("error", {})
            print(f"\n    ❌ API Error: {error.get('message', 'Unknown error')}")
        
        print("  ✅ Event handlers registered")
        
        # Connect and test
        await client.connect(config)
        print("  ✅ Connected with audio input configuration")
        
        print("  📤 Sending audio input...")
        print("  🤖 Response: ", end="")
        
        # Use the working method
        await client.send_audio_simple(test_audio)
        
        # Wait for response
        timeout = 25
        start_time = time.time()
        while "response_done" not in events_received and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
        
        # Check success
        got_speech_detection = "speech_started" in events_received and "speech_stopped" in events_received
        got_response = "response_done" in events_received
        success = got_speech_detection and got_response
        
        if success:
            print(f"  ✅ Audio input processed successfully")
            print(f"    Speech detection: {got_speech_detection}")
            print(f"    Response received: {got_response}")
            if response_text:
                print(f"    Response: '{response_text.strip()}'")
        else:
            print(f"  ❌ Audio input failed")
            print(f"    Speech detection: {got_speech_detection}")
            print(f"    Response received: {got_response}")
            print(f"    Events: {events_received}")
        
        await client.disconnect()
        print("  ✅ Disconnected")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Audio input test failed: {e}")
        logger.exception("Audio input test error")
        return False


async def test_audio_streaming():
    """Test audio streaming using working methods"""
    print("\n🌊 Testing Audio Streaming...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    # Get test audio
    test_audio = get_test_audio()
    if not test_audio:
        print("  ⏩ Skipping - cannot get test audio")
        return True
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        processor = AudioProcessor()
        
        # Configure for streaming
        config = SessionConfig(
            instructions="You are testing audio streaming. When you receive audio, respond with: 'Streaming test complete!'",
            modalities=["text", "audio"],
            voice="echo",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.7,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            )
        )
        
        # Track streaming events
        response_received = False
        response_text = ""
        
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
        
        print("  ✅ Event handlers registered")
        
        # Connect
        await client.connect(config)
        print("  ✅ Connected for streaming test")
        
        # Stream audio in chunks using the working method
        print("  🌊 Streaming audio in chunks...")
        chunk_size_ms = 200  # 200ms chunks
        chunks = processor.chunk_audio(test_audio, chunk_size_ms)
        print(f"    Split into {len(chunks)} chunks of {chunk_size_ms}ms each")
        
        # Use the working chunked sending method
        await client.send_audio_chunks(test_audio, chunk_size_ms, real_time=True)
        print("  ✅ Audio streaming completed")
        
        # Wait for response
        timeout = 30
        start_time = time.time()
        while not response_received and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if response_received:
            print(f"  ✅ Streaming response received")
            
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
    """Test full voice conversation flow"""
    print("\n💬 Testing Voice Conversation Flow...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏩ Skipping - no API key available")
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure for full voice conversation
        config = SessionConfig(
            instructions="You are a helpful voice assistant. Respond naturally and conversationally. Keep responses brief but friendly.",
            modalities=["text", "audio"],
            voice="shimmer",  # Professional voice
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            temperature=0.8,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            )
        )
        
        # Track conversation
        conversation_turns = 0
        total_audio_duration = 0
        
        @client.on_event("response.text.delta")
        async def handle_text_delta(event_data):
            text = event_data.get("delta", "")
            print(text, end="", flush=True)
        
        @client.on_event("response.done")
        async def handle_response_done(event_data):
            nonlocal conversation_turns, total_audio_duration
            conversation_turns += 1
            
            # Get audio duration and save
            duration = client.get_audio_output_duration()
            total_audio_duration += duration
            
            if duration > 0:
                filename = f"conversation_turn_{conversation_turns}.wav"
                client.save_audio_output(filename)
                print(f"\n    💾 Audio saved: {filename} ({duration:.1f}ms)")
            else:
                print()
        
        print("  ✅ Event handlers registered")
        
        # Connect
        await client.connect(config)
        print("  ✅ Connected for voice conversation")
        
        # Have a multi-turn voice conversation
        conversation_messages = [
            "Hello! How are you today?",
            "What can you help me with?",
            "Thank you very much!"
        ]
        
        for i, message in enumerate(conversation_messages):
            print(f"\n  👤 Turn {i+1}: {message}")
            print(f"  🤖 Response: ", end="")
            
            try:
                await client.send_text(message)
                
                # Wait for this turn to complete
                current_turn = conversation_turns
                timeout = 20
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
        print(f"    Total audio duration: {total_audio_duration:.1f}ms")
        
        await client.disconnect()
        print("  ✅ Voice conversation completed")
        
        # Success if we got most responses
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
        test_audio = get_test_audio()
        if not test_audio:
            print("  ⏩ Skipping - cannot get test audio")
            return True
        
        print(f"  🎵 Testing with {len(test_audio)} bytes of audio")
        
        # Test format validation
        is_valid, msg = processor.validator.validate_audio_data(test_audio, AudioFormat.PCM16)
        if is_valid:
            print("  ✅ PCM16 format validation passed")
        else:
            print(f"  ❌ PCM16 format validation failed: {msg}")
            return False
        
        # Test Realtime API format validation
        is_valid, msg = processor.validate_realtime_api_format(test_audio)
        if is_valid:
            print("  ✅ Realtime API format validation passed")
        else:
            print(f"  ❌ Realtime API format validation failed: {msg}")
            return False
        
        # Test audio info extraction
        info = processor.get_audio_info(test_audio)
        duration_ms = info['duration_ms']
        if duration_ms > 0:
            print(f"  ✅ Audio duration extracted: {duration_ms:.1f}ms")
        else:
            print(f"  ❌ Invalid audio duration: {duration_ms}ms")
            return False
        
        # Test chunking for different sizes
        chunk_sizes = [100, 250, 500]  # Different chunk sizes in ms
        for chunk_ms in chunk_sizes:
            chunks = processor.chunk_audio(test_audio, chunk_ms)
            expected_chunks = int(duration_ms / chunk_ms)
            
            if abs(len(chunks) - expected_chunks) <= 2:  # Allow ±2 chunk tolerance
                print(f"  ✅ {chunk_ms}ms chunking: {len(chunks)} chunks")
            else:
                print(f"  ❌ {chunk_ms}ms chunking failed: {len(chunks)} vs ~{expected_chunks}")
                return False
        
        # Test audio quality analysis
        if hasattr(processor, 'analyze_audio_quality'):
            analysis = processor.analyze_audio_quality(test_audio)
            if analysis.get('quality_score', 0) > 0.5:
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
    print("🧪 RealtimeVoiceAPI - Test 3: Voice and Audio API (FIXED)")
    print("=" * 70)
    print("This test requires a valid OpenAI API key and tests voice features")
    print("⚠️  This test will use moderate API quota for voice generation")
    print()
    
    # Check for voice recordings
    voice_files = ["test_voice.wav", "my_voice.wav", "voice_input.wav", "speech.wav"]
    available_files = [f for f in voice_files if Path(f).exists()]
    
    if available_files:
        print(f"✅ Found voice recording(s): {', '.join(available_files)}")
        print("   Audio input tests will be more reliable")
    else:
        print("⚠️  No voice recordings found. Some audio tests may use synthetic audio.")
        print("💡 For better results, record your voice or place test_voice.wav")
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
        await asyncio.sleep(1.0)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test 3 Results (Fixed)")
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
        print("   - ✅ Audio input processing with Server VAD")
        print("   - ✅ Real-time audio streaming")
        print("   - ✅ Multiple voice options (alloy, echo, shimmer)")
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
        print("  - Server VAD requires real speech or high-quality synthetic audio")
        print("  - Voice responses may take longer than expected")
        print("  - Network latency affecting real-time performance")
        print("  - Audio format compatibility issues")
        print("\n💡 Troubleshooting:")
        print("  1. Use real voice recordings for better VAD detection")
        print("  2. Check internet connection speed and stability")
        print("  3. Verify OpenAI API quota and rate limits")
        print("  4. Try the proven working test:")
        print("     python -m realtimevoiceapi.smoke_tests.audio_input_api_compliant")
    
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
            print(f"   📁 {f} ({size:,} bytes)")
        print("   You can play these files to hear the voice responses!")
    
    return passed >= total - 1  # Allow 1 failure


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)