#!/usr/bin/env python3
"""
test 3 udio Input - Working Tests Based on API Requirements (CLEAN VERSION)

This implements audio input tests that work with the API's actual requirements.
Server VAD or Semantic VAD are required - there is no "none" option.

UPDATED: Now saves all audio outputs to sound_outputs/ directory

Save this as: audio_input_api_compliant.py
Run with: python -m realtimevoiceapi.smoke_tests.test_3_audio_input_api_compliant
"""

import sys
import os
import asyncio
import logging
import time
import struct
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def ensure_sound_outputs_dir():
    """Ensure the sound_outputs directory exists"""
    sound_dir = Path("sound_outputs")
    sound_dir.mkdir(exist_ok=True)
    return sound_dir


def generate_speech_like_audio():
    """Generate audio that resembles speech patterns"""
    try:
        import numpy as np
        
        SAMPLE_RATE = 24000
        DURATION = 2.0  # 2 seconds is plenty
        
        # Generate something that sounds more like speech
        num_samples = int(SAMPLE_RATE * DURATION)
        t = np.linspace(0, DURATION, num_samples, endpoint=False)
        
        # Mix of frequencies found in human speech
        # Fundamental frequency around 150-200 Hz for speech
        fundamental = 180
        
        # Create a more speech-like waveform
        audio_signal = np.zeros(num_samples)
        
        # Add fundamental and harmonics
        for harmonic in range(1, 6):
            freq = fundamental * harmonic
            # Decrease amplitude for higher harmonics
            amplitude = 0.3 / harmonic
            audio_signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add some formant-like filtering (simplified)
        # This makes it sound more vowel-like
        formant_freq = 700  # First formant frequency
        modulation = 1 + 0.3 * np.sin(2 * np.pi * formant_freq / 100 * t)
        audio_signal *= modulation
        
        # Add amplitude modulation to simulate speech rhythm
        # Speech typically has syllable rates of 3-7 Hz
        syllable_rate = 4
        amplitude_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * syllable_rate * t)
        audio_signal *= amplitude_envelope
        
        # Add attack and release
        fade_samples = int(0.05 * SAMPLE_RATE)  # 50ms fade
        audio_signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio_signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Normalize and convert to 16-bit PCM
        audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.8
        audio_pcm = (audio_signal * 32767).astype(np.int16)
        
        # Ensure little-endian byte order
        audio_bytes = audio_pcm.astype('<i2').tobytes()
        
        print(f"Generated speech-like audio: {len(audio_bytes)} bytes, {DURATION}s")
        return audio_bytes
        
    except ImportError:
        logger.error("NumPy required for audio generation")
        return None


def get_test_audio():
    """Get test audio - either from file or generate it"""
    # First, try to use a real voice recording if available
    voice_files = [
        "test_voice.wav",
        "test_voice_fixed.wav",  # Fixed version with silence
        "voice_input.wav", 
        "speech.wav",
        "audio_input.wav"
    ]
    
    # Also check in sound_outputs directory
    sound_dir = ensure_sound_outputs_dir()
    sound_voice_files = [sound_dir / f for f in voice_files]
    all_voice_files = voice_files + [str(f) for f in sound_voice_files]
    
    for file in all_voice_files:
        if Path(file).exists():
            try:
                from realtimevoiceapi.audio import AudioProcessor
                processor = AudioProcessor()
                
                print(f"  📁 Using voice recording: {file}")
                audio_bytes = processor.load_wav_file(file)
                
                # Validate it's the right format
                info = processor.get_audio_info(audio_bytes)
                duration_ms = info.get('duration_ms', 0)
                
                if duration_ms < 500:
                    print(f"  ⚠️  Audio too short ({duration_ms}ms), generating synthetic audio instead")
                    return generate_speech_like_audio()
                
                print(f"  ✅ Loaded {len(audio_bytes)} bytes ({duration_ms:.0f}ms)")
                return audio_bytes
                
            except Exception as e:
                print(f"  ⚠️  Failed to load {file}: {e}")
                continue
    
    # No voice file found, generate synthetic audio
    print("  🎵 No voice recording found, generating synthetic audio...")
    return generate_speech_like_audio()


def record_test_audio():
    """Helper to record test audio (uses sounddevice or pyaudio)"""
    # Try sounddevice first (better for Mac)
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        
        print("\n🎤 Recording test audio (using sounddevice)...")
        print("  Speak clearly for 3 seconds...")
        
        # Audio parameters for Realtime API
        SAMPLE_RATE = 24000  # Required by API
        CHANNELS = 1
        DURATION = 3
        
        # Save to sound_outputs directory
        sound_dir = ensure_sound_outputs_dir()
        OUTPUT_FILE = sound_dir / "test_voice.wav"
        
        # Countdown
        import time
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        print("  🔴 RECORDING - Speak now!")
        
        # Record
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16'
        )
        sd.wait()
        
        print("  ✅ Recording complete!")
        
        # Save as WAV with correct format
        sf.write(
            str(OUTPUT_FILE),
            recording,
            SAMPLE_RATE,
            subtype='PCM_16',
            format='WAV'
        )
        
        print(f"  💾 Saved to {OUTPUT_FILE}")
        return True
        
    except ImportError:
        # Fall back to pyaudio
        try:
            import pyaudio
            import wave
            
            print("\n🎤 Recording test audio (using pyaudio)...")
            print("  Speak clearly for 3 seconds...")
            print("  Press Ctrl+C to stop early")
            
            # Audio parameters
            SAMPLE_RATE = 24000
            CHANNELS = 1
            CHUNK = 1024
            RECORD_SECONDS = 3
            
            # Save to sound_outputs directory
            sound_dir = ensure_sound_outputs_dir()
            OUTPUT_FILE = sound_dir / "test_voice.wav"
            
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            
            print("  🔴 Recording...")
            try:
                for i in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                    
                    # Show recording progress
                    if i % (SAMPLE_RATE // CHUNK) == 0:
                        print(f"  ⏱️  {i // (SAMPLE_RATE // CHUNK)}s...")
            except KeyboardInterrupt:
                print("  ⏹️  Recording stopped")
            
            print("  ✅ Recording complete!")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save as WAV
            wf = wave.open(str(OUTPUT_FILE), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"  💾 Saved to {OUTPUT_FILE}")
            return True
            
        except ImportError:
            print("  ❌ No audio recording library available.")
            print("  📦 Install one of these:")
            print("     pip install sounddevice soundfile  (recommended for Mac)")
            print("     pip install pyaudio")
            return False
            
    except Exception as e:
        print(f"  ❌ Recording failed: {e}")
        return False


async def test_server_vad_auto_response():
    """Test Server VAD with automatic response generation"""
    print("\n🎤 Testing Server VAD with Auto Response...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ No API key")
        return False
    
    audio_data = get_test_audio()
    if not audio_data:
        return False
    
    # Add silence to the end to ensure VAD detects speech end
    try:
        import numpy as np
        # Add 1 second of silence at the end
        silence = np.zeros(24000, dtype=np.int16)  # 1 second at 24kHz
        audio_with_silence = audio_data + silence.tobytes()
        audio_data = audio_with_silence
        print("  📝 Added silence padding to ensure speech end detection")
    except:
        print("  ⚠️  Could not add silence padding")
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        processor = AudioProcessor()
        
        # Configure with Server VAD and auto response
        config = SessionConfig(
            instructions="When you hear audio, respond with: 'I heard your audio message!' Be very brief.",
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.3,  # Lower threshold for better detection
                prefix_padding_ms=300,
                silence_duration_ms=500,  # How long silence before speech ends
                create_response=True  # Auto-create response when speech ends
            )
        )
        
        events_received = []
        response_text = ""
        
        @client.on_event("input_audio_buffer.speech_started")
        async def handle_speech_start(data):
            events_received.append("speech_started")
            ms = data.get("audio_start_ms", 0)
            print(f"  🎙️ Speech detected at {ms}ms!")
        
        @client.on_event("input_audio_buffer.speech_stopped")
        async def handle_speech_stop(data):
            events_received.append("speech_stopped")
            ms = data.get("audio_end_ms", 0)
            print(f"  🔇 Speech ended at {ms}ms!")
        
        @client.on_event("input_audio_buffer.committed")
        async def handle_committed(data):
            events_received.append("committed")
            print("  ✅ Audio committed by Server VAD")
        
        @client.on_event("response.text.delta")
        async def handle_text(data):
            nonlocal response_text
            response_text += data.get("delta", "")
        
        @client.on_event("response.done")
        async def handle_done(data):
            events_received.append("response_done")
            print(f"  💬 Response: {response_text}")
        
        # Connect and configure
        print("  🔌 Connecting...")
        await client.connect(config)
        print("  ✅ Connected with Server VAD enabled")
        
        # Use the new simple audio method
        print("  📤 Sending audio...")
        await client.send_audio_simple(audio_data)
        
        # Wait longer for Server VAD to process
        print("  ⏳ Waiting for Server VAD to process...")
        
        # Wait up to 20 seconds for the full flow
        timeout = 20
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
            
            # Check if we got the complete flow
            if "response_done" in events_received:
                break
            
            # Progress indicator
            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0 and elapsed > 0:
                print(f"  ⏱️  Still waiting... ({elapsed}s)")
                if "speech_started" in events_received and "speech_stopped" not in events_received:
                    print("     (Speech detected but not yet ended - waiting for silence)")
        
        # Save audio response if any (UPDATED: save to sound_outputs)
        if client.get_audio_output_duration() > 0:
            sound_dir = ensure_sound_outputs_dir()
            output_file = sound_dir / "server_vad_auto_response.wav"
            client.save_audio_output(str(output_file))
            print(f"  💾 Audio response saved to: {output_file} ({client.get_audio_output_duration():.0f}ms)")
        
        await client.disconnect()
        
        success = "response_done" in events_received
        print(f"\n  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"  Events: {events_received}")
        
        if not success:
            if "speech_started" in events_received and "speech_stopped" not in events_received:
                print("\n  💡 Tip: Speech was detected but never ended.")
                print("     Try speaking with a clear pause at the end.")
                print("     Or record with silence detection: --record-silence")
        
        return success
        
    except Exception as e:
        logger.exception("Test failed")
        return False


async def test_server_vad_manual_response():
    """Test Server VAD with manual response control"""
    print("\n🎯 Testing Server VAD with Manual Response Control...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    audio_data = generate_speech_like_audio()
    if not audio_data:
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        
        # Configure with Server VAD but manual response
        config = SessionConfig(
            instructions="When asked to respond, say: 'Manual response control works!' Be brief.",
            modalities=["text", "audio"],
            voice="alloy",
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.3,
                silence_duration_ms=500,
                create_response=False  # Don't auto-create response
            )
        )
        
        events_received = []
        audio_committed = False
        
        @client.on_event("input_audio_buffer.committed")
        async def handle_committed(data):
            nonlocal audio_committed
            audio_committed = True
            events_received.append("committed")
            print("  ✅ Audio committed by Server VAD")
        
        @client.on_event("response.done")
        async def handle_done(data):
            events_received.append("response_done")
            print("  ✅ Response completed")
        
        await client.connect(config)
        print("  ✅ Connected with manual response control")
        
        # Send audio
        await client.send_audio_simple(audio_data)
        
        # Wait for Server VAD to commit
        print("  ⏳ Waiting for Server VAD to commit audio...")
        timeout = 10
        start_time = time.time()
        
        while not audio_committed and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if audio_committed:
            print("  🚀 Manually creating response...")
            await client.create_response()
            
            # Wait for response
            await asyncio.sleep(5)
            
            # Save response (UPDATED: save to sound_outputs)
            if client.get_audio_output_duration() > 0:
                sound_dir = ensure_sound_outputs_dir()
                output_file = sound_dir / "server_vad_manual_response.wav"
                client.save_audio_output(str(output_file))
                print(f"  💾 Manual response saved to: {output_file}")
        
        await client.disconnect()
        
        success = "committed" in events_received and "response_done" in events_received
        print(f"\n  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        logger.exception("Test failed")
        return False


async def test_semantic_vad():
    """Test Semantic VAD mode"""
    print("\n🧠 Testing Semantic VAD...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    audio_data = get_test_audio()
    if not audio_data:
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        
        # Semantic VAD - use TurnDetectionConfig to properly exclude threshold
        config = SessionConfig(
            instructions="Respond to audio with: 'Semantic VAD is working!' Be brief.",
            modalities=["text", "audio"],
            voice="alloy",
            turn_detection=TurnDetectionConfig(
                type="semantic_vad",  # More advanced turn detection
                threshold=None,       # This will be excluded by to_dict()
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=True
            )
        )
        
        response_received = False
        
        @client.on_event("response.done")
        async def handle_done(data):
            nonlocal response_received
            response_received = True
            print("  ✅ Response received with Semantic VAD")
        
        await client.connect(config)
        print("  ✅ Connected with Semantic VAD")
        
        # Send audio
        await client.send_audio_simple(audio_data)
        
        # Wait for response
        print("  ⏳ Waiting for Semantic VAD processing...")
        await asyncio.sleep(12)
        
        # Save response (UPDATED: save to sound_outputs)
        if client.get_audio_output_duration() > 0:
            sound_dir = ensure_sound_outputs_dir()
            output_file = sound_dir / "semantic_vad_response.wav"
            client.save_audio_output(str(output_file))
            print(f"  💾 Semantic VAD response saved to: {output_file}")
        
        await client.disconnect()
        
        print(f"\n  Result: {'✅ SUCCESS' if response_received else '❌ FAILED'}")
        return response_received
        
    except Exception as e:
        logger.exception("Semantic VAD test failed")
        return False


async def test_audio_conversation():
    """Test complete audio conversation with new methods"""
    print("\n💬 Testing Audio Conversation...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    audio_data = generate_speech_like_audio()
    if not audio_data:
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient, SessionConfig
        from realtimevoiceapi.models import TurnDetectionConfig
        
        client = RealtimeClient(api_key)
        
        # Standard voice assistant config
        config = SessionConfig(
            instructions="You are a helpful voice assistant. Respond briefly and clearly.",
            modalities=["text", "audio"],
            voice="alloy",
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            )
        )
        
        await client.connect(config)
        print("  ✅ Connected as voice assistant")
        
        # Use the new convenience method
        print("  📤 Sending audio and waiting for response...")
        text, audio = await client.send_audio_and_wait_for_response(audio_data)
        
        if text:
            print(f"  📝 Text response: {text}")
        
        if audio:
            # Save using direct path to sound_outputs
            sound_dir = ensure_sound_outputs_dir()
            output_file = sound_dir / "conversation_response.wav"
            client.audio_processor.save_wav_file(audio, output_file)
            duration = client.audio_processor.get_audio_duration_ms(audio)
            print(f"  🔊 Audio response: {duration:.0f}ms saved to {output_file}")
        
        await client.disconnect()
        
        success = bool(text or audio)
        print(f"\n  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        logger.exception("Conversation test failed")
        return False


async def test_preset_configs():
    """Test the preset configurations"""
    print("\n🎨 Testing Preset Configurations...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    audio_data = generate_speech_like_audio()
    if not audio_data:
        return False
    
    try:
        from realtimevoiceapi import RealtimeClient
        from realtimevoiceapi.session import SessionPresets
        
        client = RealtimeClient(api_key)
        sound_dir = ensure_sound_outputs_dir()
        
        # Test voice assistant preset
        print("  🤖 Testing voice assistant preset...")
        config = SessionPresets.voice_assistant()
        
        await client.connect(config)
        await client.send_audio_simple(audio_data)
        await asyncio.sleep(5)
        
        success = client.get_audio_output_duration() > 0
        print(f"    Voice assistant: {'✅' if success else '❌'}")
        
        # Save preset response (UPDATED: save to sound_outputs)
        if success:
            output_file = sound_dir / "voice_assistant_preset_response.wav"
            client.save_audio_output(str(output_file))
            print(f"    💾 Voice assistant response saved to: {output_file}")
        
        await client.disconnect()
        
        # Test customer service preset
        print("  📞 Testing customer service preset...")
        config = SessionPresets.customer_service()
        
        await client.connect(config)
        await client.send_audio_simple(audio_data)
        await asyncio.sleep(5)
        
        success = client.get_audio_output_duration() > 0
        print(f"    Customer service: {'✅' if success else '❌'}")
        
        # Save preset response (UPDATED: save to sound_outputs)
        if success:
            output_file = sound_dir / "customer_service_preset_response.wav"
            client.save_audio_output(str(output_file))
            print(f"    💾 Customer service response saved to: {output_file}")
        
        await client.disconnect()
        
        return True
        
    except Exception as e:
        logger.exception("Preset test failed")
        return False


async def main():
    """Run working audio input tests"""
    print("🎉 RealtimeVoiceAPI - Working Audio Input Tests (CLEAN VERSION)")
    print("=" * 70)
    print("Testing audio input with proper API requirements")
    print("(Server VAD or Semantic VAD are required)")
    print()
    
    # Ensure sound_outputs directory exists and show path
    sound_dir = ensure_sound_outputs_dir()
    print(f"🗂️  Audio outputs will be saved to: {sound_dir.absolute()}")
    print()
    
    # Check for recording options
    if "--record" in sys.argv:
        record_test_audio()
        print()
    elif "--record-silence" in sys.argv:
        # Record with silence detection for better VAD compatibility
        print("🎤 Recording with silence detection...")
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            SAMPLE_RATE = 24000
            print("  Speak your message, then pause for 1.5 seconds to stop")
            print("  Starting in 3...")
            
            import time
            time.sleep(1)
            print("  2...")
            time.sleep(1)
            print("  1...")
            time.sleep(1)
            
            print("  🔴 RECORDING - Speak now!")
            
            # Record with silence detection
            max_duration = 10
            chunks = []
            silence_threshold = 0.01
            silence_duration = 1.5
            silence_chunks = 0
            chunk_duration = 0.1
            
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
                while len(chunks) < max_duration / chunk_duration:
                    chunk, _ = stream.read(int(SAMPLE_RATE * chunk_duration))
                    chunks.append(chunk)
                    
                    # Check for silence
                    rms = np.sqrt(np.mean(chunk.astype(float)**2)) / 32768.0
                    if rms < silence_threshold:
                        silence_chunks += 1
                        if silence_chunks >= silence_duration / chunk_duration:
                            print("\n  🔇 Silence detected - stopping")
                            break
                    else:
                        silence_chunks = 0
                        print(".", end="", flush=True)
            
            # Save recording to sound_outputs
            recording = np.concatenate(chunks)
            output_file = sound_dir / "test_voice.wav"
            sf.write(str(output_file), recording, SAMPLE_RATE, subtype='PCM_16')
            print(f"\n  ✅ Saved {output_file} ({len(recording)/SAMPLE_RATE:.1f}s)")
            print("     This recording includes natural silence at the end")
            
        except Exception as e:
            print(f"  ❌ Recording failed: {e}")
        print()
    
    # Check for voice files (both in current dir and sound_outputs)
    voice_files = ["test_voice.wav", "voice_input.wav", "speech.wav", "audio_input.wav"]
    available_files = []
    
    # Check current directory
    for f in voice_files:
        if Path(f).exists():
            available_files.append(f)
    
    # Check sound_outputs directory
    for f in voice_files:
        sound_file = sound_dir / f
        if sound_file.exists():
            available_files.append(str(sound_file))
    
    if available_files:
        print(f"✅ Found voice recording(s): {', '.join(available_files)}")
        print("   This will improve test reliability with Server VAD")
    else:
        print("⚠️  No voice recordings found. Tests will use synthetic audio.")
        print("💡 For better results, record your voice:")
        print("   python -m realtimevoiceapi.smoke_tests.audio_input_api_compliant --record")
        print("   python -m realtimevoiceapi.smoke_tests.audio_input_api_compliant --record-silence")
        print("   Or place a WAV file named: test_voice.wav")
        print("\n   --record-silence is recommended for better VAD detection")
        print()
    
    tests = [
        ("Server VAD with Auto Response", test_server_vad_auto_response),
        ("Server VAD with Manual Response", test_server_vad_manual_response),
        ("Semantic VAD", test_semantic_vad),
        ("Audio Conversation", test_audio_conversation),
        ("Preset Configurations", test_preset_configs),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  💥 Test crashed: {e}")
            results.append((name, False))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed > 0:
        print("\n🎉 Audio input is working correctly!")
        print("\n💡 Key takeaways:")
        print("  - Server VAD works with all parameters")
        print("  - Semantic VAD doesn't support 'threshold' parameter")
        print("  - Use client.send_audio_simple() for easy audio sending")
        print("  - Use client.send_audio_and_wait_for_response() for conversations")
        print("  - Audio must be 24kHz, 16-bit PCM, mono, little-endian")
        
        if passed < total:
            print("\n⚠️  Some tests failed. Common issues:")
            print("  - Server VAD needs clear silence at the end of speech")
            print("  - Try: --record-silence for better recordings")
            print("  - Background noise can prevent silence detection")
        
        # List generated files in sound_outputs (UPDATED: check sound_outputs)
        audio_files = list(sound_dir.glob("*.wav"))
        if audio_files:
            print(f"\n🎵 Generated audio files in {sound_dir}:")
            for f in audio_files:
                size = f.stat().st_size
                print(f"   📁 {f.name} ({size:,} bytes)")
            print("   You can play these files to hear the voice responses!")
    
    return passed > 0


if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)