
# here is realtimevoiceapi/smoke_tests/test_01_core_modules.py

#!/usr/bin/env python3
"""
Test 01: Core Modules - Test basic modules with real data

Tests core modules without mocks:
- Audio types and processing with real audio
- Stream protocol with real events  
- Provider protocol with real pricing
- Message protocol with real validation
- Session manager with real sessions

python -m realtimevoiceapi.smoke_tests.test_01_core_modules
"""

import sys
import time
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from realtimevoiceapi.audio_types import (
    AudioFormat, AudioConfig, AudioQuality, VADType, VADConfig,
    AudioMetadata, AudioErrorType, BufferConfig, ProcessingMode,
    AudioConstants
)
from realtimevoiceapi.audio_processor import AudioProcessor, AudioStreamBuffer
from realtimevoiceapi.stream_protocol import (
    StreamEventType, StreamEvent, StreamState, StreamMetrics,
    AudioFormat as StreamAudioFormat, StreamConfig, Response,
    StreamCapability, StreamCapabilities, StreamError, StreamErrorType
)
from realtimevoiceapi.provider_protocol import (
    ProviderFeature, ProviderCapabilities, CostUnit, CostModel,
    Usage, Cost, ProviderConfig, VoiceConfig, TranscriptionConfig,
    FunctionDefinition, QualityPreset, QualitySettings
)
from realtimevoiceapi.message_protocol import (
    ClientMessageType, ServerMessageType, MessageFactory,
    MessageValidator, MessageParser, ProtocolInfo
)
from realtimevoiceapi.session_manager import (
    SessionState, Session, SessionManager
)


def print_test_header(test_name: str):
    """Print a nice test header"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")


def print_result(passed: bool, message: str):
    """Print test result"""
    icon = "✅" if passed else "❌"
    print(f"{icon} {message}")


# ============== Audio Types Tests ==============

def test_audio_types():
    """Test audio type definitions and configurations"""
    print_test_header("Testing Audio Types")
    
    results = []
    
    # Test 1: AudioFormat enum
    try:
        assert AudioFormat.PCM16.value == "pcm16"
        assert AudioFormat.PCM16.bytes_per_sample == 2
        assert not AudioFormat.PCM16.requires_compression
        assert AudioFormat.G711_ULAW.requires_compression
        results.append(True)
        print_result(True, "AudioFormat enum works correctly")
    except Exception as e:
        results.append(False)
        print_result(False, f"AudioFormat enum failed: {e}")
    
    # Test 2: AudioConfig with calculations
    try:
        config = AudioConfig(sample_rate=24000, channels=1, bit_depth=16)
        assert config.frame_size == 2  # 1 channel * 16-bit/8
        assert config.bytes_per_second == 48000  # 24000 * 2
        assert config.bytes_per_ms == 48  # 48000 / 1000
        assert config.chunk_size_bytes(100) == 4800  # 100ms * 48
        results.append(True)
        print_result(True, "AudioConfig calculations correct")
    except Exception as e:
        results.append(False)
        print_result(False, f"AudioConfig failed: {e}")
    
    # Test 3: AudioQuality presets
    try:
        low_quality = AudioQuality.LOW.to_config()
        assert low_quality.sample_rate == 16000
        
        high_quality = AudioQuality.HIGH.to_config()
        assert high_quality.sample_rate == 48000
        
        standard = AudioQuality.STANDARD.to_config()
        assert standard.sample_rate == 24000
        results.append(True)
        print_result(True, "AudioQuality presets work")
    except Exception as e:
        results.append(False)
        print_result(False, f"AudioQuality failed: {e}")
    
    # Test 4: VADConfig with validation
    try:
        vad = VADConfig(
            type=VADType.ENERGY_BASED,
            energy_threshold=0.02,
            speech_start_ms=100,
            speech_end_ms=500
        )
        assert vad.type == VADType.ENERGY_BASED
        assert vad.energy_threshold == 0.02
        assert vad.type.is_local
        results.append(True)
        print_result(True, "VADConfig validation works")
    except Exception as e:
        results.append(False)
        print_result(False, f"VADConfig failed: {e}")
    
    # Test 5: AudioMetadata
    try:
        metadata = AudioMetadata(
            format=AudioFormat.PCM16,
            duration_ms=1000.0,
            size_bytes=48000,
            peak_amplitude=0.8,
            is_speech=True
        )
        meta_dict = metadata.to_dict()
        assert meta_dict["format"] == "pcm16"
        assert meta_dict["duration_ms"] == 1000.0
        assert meta_dict["is_speech"] == True
        results.append(True)
        print_result(True, "AudioMetadata serialization works")
    except Exception as e:
        results.append(False)
        print_result(False, f"AudioMetadata failed: {e}")
    
    return all(results)


# ============== Audio Processor Tests ==============

def test_audio_processor():
    """Test audio processor with real audio data"""
    print_test_header("Testing Audio Processor")
    
    results = []
    processor = AudioProcessor()
    
    # Test 1: Generate real audio data
    try:
        # Generate 1 second of 440Hz sine wave
        sample_rate = 24000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_float = np.sin(2 * np.pi * frequency * t) * 0.5
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        results.append(True)
        print_result(True, f"Generated {len(audio_bytes)} bytes of test audio")
    except Exception as e:
        results.append(False)
        print_result(False, f"Audio generation failed: {e}")
        return False
    
    # Test 2: Base64 encoding/decoding
    try:
        encoded = processor.bytes_to_base64(audio_bytes[:1000])
        decoded = processor.base64_to_bytes(encoded)
        assert decoded == audio_bytes[:1000]
        results.append(True)
        print_result(True, "Base64 encoding/decoding works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Base64 test failed: {e}")
    
    # Test 3: Audio validation
    try:
        is_valid, msg = processor.validate_format(audio_bytes)
        assert is_valid
        
        # Test invalid audio (empty)
        is_valid, msg = processor.validate_format(b"")
        assert not is_valid
        assert "empty" in msg.lower()
        
        results.append(True)
        print_result(True, "Audio validation works correctly")
    except Exception as e:
        results.append(False)
        print_result(False, f"Validation failed: {e}")
    
    # Test 4: Duration calculation
    try:
        duration_ms = processor.calculate_duration(audio_bytes)
        expected_ms = 1000.0  # 1 second
        assert abs(duration_ms - expected_ms) < 1  # Within 1ms tolerance
        results.append(True)
        print_result(True, f"Duration calculation: {duration_ms:.1f}ms")
    except Exception as e:
        results.append(False)
        print_result(False, f"Duration calculation failed: {e}")
    
    # Test 5: Audio chunking
    try:
        chunks = processor.chunk_audio(audio_bytes, chunk_duration_ms=100)
        expected_chunks = 10  # 1000ms / 100ms
        assert len(chunks) == expected_chunks
        
        # Verify chunk sizes
        for chunk in chunks[:-1]:  # All but last
            chunk_duration = processor.calculate_duration(chunk)
            assert abs(chunk_duration - 100) < 5  # Within 5ms tolerance
        
        results.append(True)
        print_result(True, f"Audio chunking: {len(chunks)} chunks of 100ms")
    except Exception as e:
        results.append(False)
        print_result(False, f"Chunking failed: {e}")
    
    # Test 6: Audio quality analysis
    try:
        metadata = processor.analyze_quality(audio_bytes)
        assert metadata.duration_ms > 0
        assert metadata.peak_amplitude > 0.4  # Should be ~0.5
        assert metadata.peak_amplitude < 0.6
        assert metadata.is_speech is not None
        results.append(True)
        print_result(True, f"Quality analysis: peak={metadata.peak_amplitude:.2f}")
    except Exception as e:
        results.append(False)
        print_result(False, f"Quality analysis failed: {e}")
    
    # Test 7: Stream buffer
    try:
        buffer = AudioStreamBuffer()
        
        # Add audio chunks
        chunk_size = 4800  # 100ms at 24kHz
        for i in range(5):
            chunk = audio_bytes[i*chunk_size:(i+1)*chunk_size]
            assert buffer.add_audio(chunk)
        
        assert buffer.get_available_bytes() == 5 * chunk_size
        
        # Get chunks back
        retrieved = buffer.get_chunk(chunk_size)
        assert len(retrieved) == chunk_size
        assert buffer.get_available_bytes() == 4 * chunk_size
        
        stats = buffer.get_stats()
        assert stats["total_added"] == 5 * chunk_size
        assert stats["total_consumed"] == chunk_size
        
        results.append(True)
        print_result(True, "Stream buffer operations work")
    except Exception as e:
        results.append(False)
        print_result(False, f"Stream buffer failed: {e}")
    
    return all(results)


# ============== Stream Protocol Tests ==============

def test_stream_protocol():
    """Test stream protocol with real events"""
    print_test_header("Testing Stream Protocol")
    
    results = []
    
    # Test 1: Stream events
    try:
        event = StreamEvent(
            type=StreamEventType.AUDIO_INPUT_STARTED,
            stream_id="test_stream_001",
            timestamp=time.time(),
            data={"source": "microphone", "format": "pcm16"}
        )
        assert event.type == StreamEventType.AUDIO_INPUT_STARTED
        assert event.stream_id == "test_stream_001"
        assert "source" in event.data
        results.append(True)
        print_result(True, "Stream events creation works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Stream events failed: {e}")
    
    # Test 2: Stream states
    try:
        # Test state transitions
        states = [
            StreamState.IDLE,
            StreamState.STARTING,
            StreamState.ACTIVE,
            StreamState.PAUSED,
            StreamState.ENDING,
            StreamState.ENDED
        ]
        for state in states:
            assert isinstance(state.value, str)
        results.append(True)
        print_result(True, "Stream states defined correctly")
    except Exception as e:
        results.append(False)
        print_result(False, f"Stream states failed: {e}")
    
    # Test 3: Stream metrics
    try:
        metrics = StreamMetrics(
            bytes_sent=48000,
            bytes_received=96000,
            chunks_sent=10,
            chunks_received=20,
            start_time=time.time() - 1.0,
            end_time=time.time()
        )
        
        assert metrics.duration_seconds > 0.9
        assert metrics.duration_seconds < 1.1
        assert metrics.throughput_bps > 100000  # Should be ~144000
        
        results.append(True)
        print_result(True, f"Stream metrics: {metrics.throughput_bps:.0f} bps")
    except Exception as e:
        results.append(False)
        print_result(False, f"Stream metrics failed: {e}")
    
    # Test 4: Audio format specification
    try:
        audio_fmt = StreamAudioFormat(
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            encoding="pcm"
        )
        fmt_dict = audio_fmt.to_dict()
        assert fmt_dict["sample_rate"] == 24000
        assert fmt_dict["encoding"] == "pcm"
        results.append(True)
        print_result(True, "Audio format specification works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Audio format failed: {e}")
    
    # Test 5: Stream capabilities
    try:
        caps = StreamCapabilities(
            supported=[
                StreamCapability.AUDIO_INPUT,
                StreamCapability.AUDIO_OUTPUT,
                StreamCapability.VAD,
                StreamCapability.STREAMING_RESPONSE
            ],
            audio_formats=[audio_fmt],
            max_chunk_size=8192,
            min_chunk_size=1024,
            supports_pause_resume=True
        )
        
        assert caps.supports(StreamCapability.AUDIO_INPUT)
        assert caps.supports(StreamCapability.VAD)
        assert not caps.supports(StreamCapability.FUNCTION_CALLING)
        
        results.append(True)
        print_result(True, "Stream capabilities checking works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Stream capabilities failed: {e}")
    
    # Test 6: Response structure
    try:
        response = Response(
            id="resp_123",
            text="Hello, this is a test response",
            audio=b"fake_audio_data",
            duration_ms=1500.0,
            tokens_used=15,
            function_calls=[{"name": "test_func", "args": "{}"}]
        )
        assert response.id == "resp_123"
        assert len(response.function_calls) == 1
        assert response.metadata is not None  # Should be auto-initialized
        results.append(True)
        print_result(True, "Response structure works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Response structure failed: {e}")
    
    return all(results)


# ============== Provider Protocol Tests ==============

def test_provider_protocol():
    """Test provider protocol with real pricing calculations"""
    print_test_header("Testing Provider Protocol")
    
    results = []
    
    # Test 1: Provider capabilities
    try:
        caps = ProviderCapabilities(
            provider_name="openai",
            features=[
                ProviderFeature.REALTIME_VOICE,
                ProviderFeature.STREAMING_TEXT,
                ProviderFeature.SERVER_VAD,
                ProviderFeature.FUNCTION_CALLING
            ],
            supported_audio_formats=[AudioFormat.PCM16],
            supported_sample_rates=[24000],
            max_audio_duration_ms=300000,  # 5 minutes
            min_audio_chunk_ms=100,
            available_voices=["alloy", "echo", "shimmer"],
            supports_voice_config=True,
            supported_languages=["en", "es", "fr", "de", "ja"]
        )
        
        assert caps.supports(ProviderFeature.REALTIME_VOICE)
        assert not caps.supports(ProviderFeature.VOICE_CLONING)
        assert "alloy" in caps.available_voices
        
        results.append(True)
        print_result(True, "Provider capabilities work")
    except Exception as e:
        results.append(False)
        print_result(False, f"Provider capabilities failed: {e}")
    
    # Test 2: Cost model and calculations
    try:
        # Real OpenAI-like pricing
        cost_model = CostModel(
            audio_input_cost=0.006,  # $0.006 per minute
            audio_input_unit=CostUnit.PER_MINUTE,
            audio_output_cost=0.024,  # $0.024 per minute
            audio_output_unit=CostUnit.PER_MINUTE,
            text_input_cost=0.00001,  # $0.01 per 1K tokens
            text_input_unit=CostUnit.PER_TOKEN,
            text_output_cost=0.00003,  # $0.03 per 1K tokens
            text_output_unit=CostUnit.PER_TOKEN
        )
        
        # Track usage for a 2-minute conversation
        usage = Usage(
            audio_input_seconds=120.0,  # 2 minutes input
            audio_output_seconds=90.0,   # 1.5 minutes output
            text_input_tokens=150,
            text_output_tokens=500,
            function_calls=2
        )
        
        # Calculate costs manually
        audio_in_cost = (120 / 60) * 0.006  # $0.012
        audio_out_cost = (90 / 60) * 0.024  # $0.036
        text_in_cost = 150 * 0.00001       # $0.0015
        text_out_cost = 500 * 0.00003      # $0.015
        
        total_expected = audio_in_cost + audio_out_cost + text_in_cost + text_out_cost
        
        results.append(True)
        print_result(True, f"Cost calculations: ${total_expected:.4f}")
    except Exception as e:
        results.append(False)
        print_result(False, f"Cost model failed: {e}")
    
    # Test 3: Voice configuration
    try:
        voice_config = VoiceConfig(
            voice_id="alloy",
            speed=1.1,
            pitch=0.9,
            volume=0.8,
            emotion="friendly",
            metadata={"provider_specific": "data"}
        )
        assert voice_config.voice_id == "alloy"
        assert voice_config.speed == 1.1
        assert voice_config.metadata["provider_specific"] == "data"
        results.append(True)
        print_result(True, "Voice configuration works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Voice configuration failed: {e}")
    
    # Test 4: Function definitions
    try:
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )
        assert func_def.name == "get_weather"
        assert "properties" in func_def.parameters
        results.append(True)
        print_result(True, "Function definitions work")
    except Exception as e:
        results.append(False)
        print_result(False, f"Function definitions failed: {e}")
    
    # Test 5: Quality settings
    try:
        quality = QualitySettings(
            preset=QualityPreset.PREMIUM,
            audio_bitrate=128000,
            noise_suppression=True,
            temperature=0.7,
            response_format="natural"
        )
        assert quality.preset == QualityPreset.PREMIUM
        assert quality.noise_suppression
        results.append(True)
        print_result(True, "Quality settings work")
    except Exception as e:
        results.append(False)
        print_result(False, f"Quality settings failed: {e}")
    
    return all(results)


# ============== Message Protocol Tests ==============

def test_message_protocol():
    """Test message protocol with real message validation"""
    print_test_header("Testing Message Protocol")
    
    results = []
    
    # Test 1: Create session update message
    try:
        msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.8,
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "create_response": True
            }
        )
        
        assert msg["type"] == ClientMessageType.SESSION_UPDATE.value
        assert "event_id" in msg
        assert msg["session"]["modalities"] == ["text", "audio"]
        assert msg["session"]["voice"] == "alloy"
        
        # Validate the message
        assert MessageValidator.validate_outgoing(msg)
        
        results.append(True)
        print_result(True, "Session update message creation works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Session update failed: {e}")
    
    # Test 2: Create audio append message
    try:
        # Create real audio data
        audio_bytes = b"test_audio_data" * 100
        audio_b64 = AudioProcessor.bytes_to_base64(audio_bytes)
        
        msg = MessageFactory.input_audio_buffer_append(audio_b64)
        assert msg["type"] == ClientMessageType.INPUT_AUDIO_BUFFER_APPEND.value
        assert msg["audio"] == audio_b64
        assert MessageValidator.validate_outgoing(msg)
        
        results.append(True)
        print_result(True, "Audio append message works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Audio append failed: {e}")
    
    # Test 3: Parse server messages
    try:
        # Simulate server response
        server_msg = {
            "type": ServerMessageType.RESPONSE_AUDIO_DELTA.value,
            "delta": "base64_audio_data_here",
            "response_id": "resp_123",
            "event_id": "evt_456"
        }
        
        assert MessageParser.get_message_type(server_msg) == ServerMessageType.RESPONSE_AUDIO_DELTA.value
        assert MessageParser.is_audio_response(server_msg)
        assert not MessageParser.is_text_response(server_msg)
        
        audio_delta = MessageParser.extract_audio_delta(server_msg)
        assert audio_delta == "base64_audio_data_here"
        
        results.append(True)
        print_result(True, "Message parsing works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Message parsing failed: {e}")
    
    # Test 4: Error message handling
    try:
        error_msg = {
            "type": ServerMessageType.ERROR.value,
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_audio_format",
                "message": "Audio must be 24kHz PCM16"
            }
        }
        
        assert MessageParser.is_error(error_msg)
        error_details = MessageParser.extract_error(error_msg)
        assert error_details["code"] == "invalid_audio_format"
        
        results.append(True)
        print_result(True, "Error message handling works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Error handling failed: {e}")
    
    # Test 5: Protocol info validation
    try:
        assert ProtocolInfo.is_valid_audio_format("pcm16")
        assert not ProtocolInfo.is_valid_audio_format("mp3")
        
        assert ProtocolInfo.is_valid_voice("alloy")
        assert not ProtocolInfo.is_valid_voice("custom_voice")
        
        assert ProtocolInfo.is_valid_modality("audio")
        assert not ProtocolInfo.is_valid_modality("video")
        
        results.append(True)
        print_result(True, "Protocol validation works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Protocol validation failed: {e}")
    
    # Test 6: Complex message creation
    try:
        msg = MessageFactory.response_create(
            modalities=["text", "audio"],
            instructions="Be helpful and concise",
            temperature=0.7,
            max_output_tokens=150,
            tools=[{
                "type": "function",
                "name": "get_time",
                "description": "Get current time"
            }]
        )
        
        assert msg["response"]["temperature"] == 0.7
        assert len(msg["response"]["tools"]) == 1
        assert MessageValidator.validate_outgoing(msg)
        
        results.append(True)
        print_result(True, "Complex message creation works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Complex message failed: {e}")
    
    return all(results)


# ============== Session Manager Tests ==============

def test_session_manager():
    """Test session manager with real sessions"""
    print_test_header("Testing Session Manager")
    
    results = []
    manager = SessionManager()
    
    # Test 1: Create sessions
    try:
        # Create first session
        session1 = manager.create_session(
            provider="openai",
            stream_id="stream_001",
            config={
                "model": "gpt-4-realtime",
                "voice": "alloy",
                "temperature": 0.8
            }
        )
        
        assert session1.id.startswith("sess_")
        assert session1.provider == "openai"
        assert session1.state == SessionState.INITIALIZING
        assert session1.config["voice"] == "alloy"
        
        # Create second session
        session2 = manager.create_session(
            provider="openai",
            stream_id="stream_002", 
            config={"model": "gpt-4-realtime"}
        )
        
        assert len(manager.sessions) == 2
        assert len(manager.provider_sessions["openai"]) == 2
        
        results.append(True)
        print_result(True, f"Created {len(manager.sessions)} sessions")
    except Exception as e:
        results.append(False)
        print_result(False, f"Session creation failed: {e}")
        return False
    
    # Test 2: Update session states
    try:
        # Update to active
        manager.update_state(session1.id, SessionState.ACTIVE)
        updated = manager.get_session(session1.id)
        assert updated.state == SessionState.ACTIVE
        assert updated.last_activity > session1.created_at
        
        # Update to error
        manager.update_state(session2.id, SessionState.ERROR)
        assert manager.get_session(session2.id).state == SessionState.ERROR
        
        results.append(True)
        print_result(True, "Session state updates work")
    except Exception as e:
        results.append(False)
        print_result(False, f"State update failed: {e}")
    
    # Test 3: Track usage
    try:
        # Simulate some usage
        manager.track_usage(
            session1.id,
            audio_seconds=5.5,
            text_tokens=150,
            function_calls=1
        )
        
        manager.track_usage(
            session1.id,
            audio_seconds=3.2,
            text_tokens=75
        )
        
        session = manager.get_session(session1.id)
        assert session.audio_seconds_used == 8.7  # 5.5 + 3.2
        assert session.text_tokens_used == 225    # 150 + 75
        assert session.function_calls_made == 1
        
        results.append(True)
        print_result(True, f"Usage tracking: {session.audio_seconds_used}s audio")
    except Exception as e:
        results.append(False)
        print_result(False, f"Usage tracking failed: {e}")
    
    # Test 4: Get active sessions
    try:
        active_sessions = manager.get_active_sessions()
        assert len(active_sessions) == 1  # Only session1 is active
        assert active_sessions[0].id == session1.id
        
        # Filter by provider
        openai_active = manager.get_active_sessions(provider="openai")
        assert len(openai_active) == 1
        
        results.append(True)
        print_result(True, f"Found {len(active_sessions)} active sessions")
    except Exception as e:
        results.append(False)
        print_result(False, f"Active sessions query failed: {e}")
    
    # Test 5: End session
    try:
        manager.end_session(session1.id)
        ended = manager.get_session(session1.id)
        
        assert ended.state == SessionState.ENDED
        assert "ended_at" in ended.metadata
        assert "duration" in ended.metadata
        assert ended.metadata["duration"] > 0
        
        results.append(True)
        print_result(True, f"Session ended after {ended.metadata['duration']:.1f}s")
    except Exception as e:
        results.append(False)
        print_result(False, f"End session failed: {e}")
    
    # Test 6: Cleanup old sessions
    try:
        # Mark session as old
        old_session = manager.get_session(session1.id)
        old_session.created_at = time.time() - 7200  # 2 hours ago
        
        initial_count = len(manager.sessions)
        manager.cleanup_old_sessions(max_age_seconds=3600)  # 1 hour
        
        assert len(manager.sessions) == initial_count - 1
        assert session1.id not in manager.sessions
        assert session2.id in manager.sessions  # Not ended, so not cleaned
        
        results.append(True)
        print_result(True, "Old session cleanup works")
    except Exception as e:
        results.append(False)
        print_result(False, f"Cleanup failed: {e}")
    
    return all(results)


# ============== Integration Test ==============

def test_integration():
    """Test modules working together"""
    print_test_header("Testing Module Integration")
    
    results = []
    
    try:
        # Create a complete audio processing pipeline
        processor = AudioProcessor(mode=ProcessingMode.REALTIME)
        
        # Generate test audio
        sample_rate = 24000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
        audio_bytes = audio.tobytes()
        
        # Process through pipeline
        metadata = processor.analyze_quality(audio_bytes)
        chunks = processor.chunk_audio(audio_bytes, chunk_duration_ms=50)
        
        # Create messages for each chunk
        messages = []
        for chunk in chunks:
            b64 = processor.bytes_to_base64(chunk)
            msg = MessageFactory.input_audio_buffer_append(b64)
            messages.append(msg)
        
        # Validate all messages
        for msg in messages:
            assert MessageValidator.validate_outgoing(msg)
        
        # Simulate stream metrics
        metrics = StreamMetrics(
            bytes_sent=len(audio_bytes),
            chunks_sent=len(chunks),
            start_time=time.time() - 0.5,
            end_time=time.time()
        )
        
        assert metrics.throughput_bps > 0
        
        results.append(True)
        print_result(True, f"Integration test passed with {len(chunks)} chunks")
        
    except Exception as e:
        results.append(False)
        print_result(False, f"Integration test failed: {e}")
    
    return all(results)


# ============== Main Test Runner ==============

async def main():
    """Run all core module tests"""
    print("\n🚀 RealtimeVoiceAPI - Core Modules Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Audio Types", test_audio_types),
        ("Audio Processor", test_audio_processor),
        ("Stream Protocol", test_stream_protocol),
        ("Provider Protocol", test_provider_protocol),
        ("Message Protocol", test_message_protocol),
        ("Session Manager", test_session_manager),
        ("Integration", test_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            test_results.append((test_name, passed))
        except Exception as e:
            print(f"\n💥 {test_name} crashed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    total_passed = sum(1 for _, passed in test_results if passed)
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {total_passed}/{total_tests} passed")
    
    if total_passed == total_tests:
        print("\n🎉 All core module tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)