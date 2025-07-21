# here is realtimevoiceapi/smoke_tests/test_01_core_modules.py
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
import logging
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional
import os
from dotenv import load_dotenv
import wave
import numpy as np
import inspect

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_audio_types():
    """Test audio types with real audio data"""
    print("\n🎵 Testing Audio Types Module...")
    
    try:
        from realtimevoiceapi.core.audio_types import (
            AudioFormat, AudioConfig, AudioQuality,
            VADConfig, AudioMetadata, BufferConfig
        )
        
        # Test AudioFormat enum
        assert AudioFormat.PCM16.value == "pcm16"
        assert AudioFormat.G711_ULAW.value == "g711_ulaw"
        print("  ✅ AudioFormat enum works")
        
        # Test AudioConfig with real values
        config = AudioConfig(
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            chunk_duration_ms=100
        )
        
        # Calculate real chunk size
        chunk_size = config.chunk_size_bytes(100)
        expected_size = 24000 * 1 * 2 * 100 // 1000  # rate * channels * bytes_per_sample * duration_ms / 1000
        assert chunk_size == expected_size
        print("  ✅ AudioConfig calculations correct")
        
        # Test AudioQuality enum
        assert hasattr(AudioQuality, 'HIGH')
        assert hasattr(AudioQuality, 'LOW')
        assert hasattr(AudioQuality, 'STANDARD')
        print("  ✅ AudioQuality enum exists")
        
        # Test VADConfig
        vad_config = VADConfig(
            energy_threshold=0.02,
            speech_start_ms=150,
            speech_end_ms=600
        )
        assert vad_config.energy_threshold == 0.02
        print("  ✅ VADConfig validation works")
        
        # Test AudioMetadata - inspect what it actually needs
        sig = inspect.signature(AudioMetadata.__init__)
        metadata_params = list(sig.parameters.keys())
        print(f"  ℹ️ AudioMetadata params: {[p for p in metadata_params if p != 'self']}")
        
        # Create AudioMetadata with correct parameters
        metadata = AudioMetadata(
            duration_ms=1000,
            format=AudioFormat.PCM16,
            size_bytes=48000
        )
        assert metadata.to_dict()["duration_ms"] == 1000
        print("  ✅ AudioMetadata serialization works")
        
        # Test BufferConfig
        buffer = BufferConfig(
            max_size_mb=10,
            max_duration_seconds=60
        )
        assert buffer.max_size_bytes == 10 * 1024 * 1024
        print("  ✅ BufferConfig works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio types test failed: {e}")
        logger.exception("Audio types error")
        return False


def test_stream_protocol():
    """Test stream protocol with real events"""
    print("\n🌊 Testing Stream Protocol Module...")
    
    try:
        from realtimevoiceapi.core.stream_protocol import (
            StreamEventType, StreamState, StreamEvent,
            StreamConfig, StreamCapabilities, StreamMetrics
        )
        from realtimevoiceapi.core.audio_types import AudioFormat
        import time
        
        # Test enums
        assert StreamEventType.STREAM_STARTED.value == "stream.started"
        assert StreamState.IDLE.value == "idle"
        print("  ✅ StreamEventType and StreamState enums work")
        
        # Create real stream event
        event = StreamEvent(
            type=StreamEventType.AUDIO_OUTPUT_CHUNK,
            stream_id="test_stream_123",
            timestamp=time.time(),
            data={"audio": b"real audio data", "sequence": 1}
        )
        
        assert event.stream_id == "test_stream_123"
        assert "audio" in event.data
        print("  ✅ StreamEvent creation works")
        
        # Test AudioFormat value directly
        format = AudioFormat.PCM16
        assert format.value == "pcm16"
        print("  ✅ AudioFormat value access works")
        
        # Create StreamConfig with required parameters
        config = StreamConfig(
            provider="openai",
            mode="realtime",
            audio_format=AudioFormat.PCM16,
            enable_vad=True
        )
        
        assert config.audio_format == AudioFormat.PCM16
        assert config.enable_vad == True
        assert config.provider == "openai"
        print("  ✅ StreamConfig works")
        
        # Test StreamCapabilities
        capabilities = StreamCapabilities(
            audio_input=True,
            audio_output=True,
            text_input=True,
            text_output=True,
            function_calling=False,
            interruptions=True
        )
        
        assert capabilities.supports_audio()
        assert capabilities.supports_text()
        assert not capabilities.supports_functions()
        print("  ✅ StreamCapabilities works")
        
        # Test StreamMetrics with real timing
        metrics = StreamMetrics()
        metrics.start_time = time.time()
        time.sleep(0.1)  # Simulate some processing
        metrics.end_time = time.time()
        
        metrics.audio_chunks_sent = 10
        metrics.audio_chunks_received = 8
        metrics.text_messages_sent = 2
        metrics.text_messages_received = 2
        
        # Check duration calculation
        duration = metrics.duration_seconds
        assert 0.09 < duration < 0.15  # Should be around 0.1s
        print(f"  ✅ Stream metrics duration: {duration:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stream protocol test failed: {e}")
        logger.exception("Stream protocol error")
        return False


def test_provider_protocol():
    """Test provider protocol with real pricing data"""
    print("\n🔌 Testing Provider Protocol Module...")
    
    try:
        from realtimevoiceapi.core.provider_protocol import (
            ProviderFeature, ProviderCapabilities, CostModel,
            Usage, Cost, ProviderConfig, ProviderRegistry,
            VoiceConfig, FunctionDefinition
        )
        from realtimevoiceapi.core.audio_types import AudioFormat
        
        # Test ProviderFeature enum with actual values
        assert ProviderFeature.REALTIME_VOICE.value == "realtime_voice"
        assert ProviderFeature.STREAMING_TEXT.value == "streaming_text"
        assert ProviderFeature.FUNCTION_CALLING.value == "function_calling"
        print("  ✅ ProviderFeature enum works")
        
        # Test ProviderCapabilities with all required parameters
        capabilities = ProviderCapabilities(
            provider_name="openai",
            features={
                ProviderFeature.REALTIME_VOICE,
                ProviderFeature.STREAMING_TEXT,
                ProviderFeature.FUNCTION_CALLING
            },
            supported_audio_formats=[AudioFormat.PCM16, AudioFormat.G711_ULAW],
            supported_sample_rates=[16000, 24000, 48000],
            max_audio_duration_ms=600000,  # 10 minutes
            min_audio_chunk_ms=20,
            available_voices=["alloy", "echo", "fable"],
            supports_voice_config=True,
            supported_languages=["en", "es", "fr"],
            max_session_duration_ms=3600000,  # 1 hour
            max_concurrent_streams=10,
            rate_limits={"requests_per_minute": 60}
        )
        
        assert capabilities.supports(ProviderFeature.REALTIME_VOICE)
        assert "alloy" in capabilities.available_voices
        print("  ✅ ProviderCapabilities works")
        
        # Test CostModel with real OpenAI pricing (as of 2024)
        cost_model = CostModel(
            audio_input_per_minute=0.06,    # $0.06 per minute
            audio_output_per_minute=0.24,   # $0.24 per minute
            text_input_per_1k_tokens=0.015, # $0.015 per 1K tokens
            text_output_per_1k_tokens=0.06, # $0.06 per 1K tokens
            currency="USD"
        )
        
        assert cost_model.audio_input_per_minute == 0.06
        print("  ✅ CostModel works with real pricing")
        
        # Test Usage tracking
        usage = Usage()
        usage.audio_input_seconds = 120  # 2 minutes
        usage.audio_output_seconds = 60  # 1 minute
        usage.text_input_tokens = 500
        usage.text_output_tokens = 1500
        usage.function_calls = 2
        
        # Calculate real cost
        cost = Cost.calculate(usage, cost_model)
        expected_audio_cost = (120/60 * 0.06) + (60/60 * 0.24)  # $0.12 + $0.24
        expected_text_cost = (500/1000 * 0.015) + (1500/1000 * 0.06)  # $0.0075 + $0.09
        
        assert abs(cost.audio_cost - expected_audio_cost) < 0.001
        assert abs(cost.text_cost - expected_text_cost) < 0.001
        print(f"  ✅ Cost calculation works: ${cost.total:.3f}")
        
        # Test ProviderConfig
        config = ProviderConfig(
            api_key="test_key_123",
            base_url="https://api.openai.com/v1",
            api_version="2024-10-01",
            organization_id=None,
            timeout_seconds=30,
            max_retries=3
        )
        
        assert config.api_key == "test_key_123"
        assert config.timeout_seconds == 30
        print("  ✅ ProviderConfig works")
        
        # Test ProviderRegistry
        registry = ProviderRegistry()
        
        # Register a test provider
        registry.register(
            name="openai",
            capabilities=capabilities,
            cost_model=cost_model,
            config_class=ProviderConfig
        )
        
        assert registry.get_provider("openai") is not None
        assert registry.supports("openai", ProviderFeature.REALTIME_VOICE)
        print("  ✅ ProviderRegistry works")
        
        # Test VoiceConfig
        voice_config = VoiceConfig(
            voice_id="alloy",
            speed=1.0,
            pitch=1.0,
            language="en-US",
            style="neutral"
        )
        
        assert voice_config.voice_id == "alloy"
        assert voice_config.to_api_params()["voice"] == "alloy"
        print("  ✅ VoiceConfig works")
        
        # Test FunctionDefinition
        function = FunctionDefinition(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )
        
        assert function.name == "get_weather"
        assert function.validate_params({"location": "New York", "unit": "celsius"})
        print("  ✅ FunctionDefinition works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Provider protocol test failed: {e}")
        logger.exception("Provider protocol error")
        return False


def test_message_protocol():
    """Test message protocol with real message validation"""
    print("\n✉️ Testing Message Protocol Module...")
    
    try:
        from realtimevoiceapi.core.message_protocol import (
            MessageFactory, MessageParser, MessageValidator,
            ClientMessageType, ServerMessageType
        )
        
        # Test message type enums
        assert ClientMessageType.SESSION_UPDATE.value == "session.update"
        assert ServerMessageType.ERROR.value == "error"
        print("  ✅ Message type enums work")
        
        # Test creating real messages
        messages = []
        
        # Session update message
        session_msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            instructions="You are a helpful assistant.",
            temperature=0.8,
            input_audio_format="pcm16",
            output_audio_format="pcm16"
        )
        messages.append(session_msg)
        assert session_msg["type"] == "session.update"
        print("  ✅ Session update message created")
        
        # Audio append message with real audio data
        import base64
        audio_data = b"\x00\x01\x02\x03" * 100  
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        audio_msg = MessageFactory.input_audio_buffer_append(audio_b64)
        messages.append(audio_msg)
        assert audio_msg["type"] == "input_audio_buffer.append"
        assert "audio" in audio_msg
        print("  ✅ Audio append message created")
        
        # Conversation item message
        conv_msg = MessageFactory.conversation_item_create(
            item_type="message",
            role="user",
            content=[{"type": "text", "text": "Hello, how can I help you today?"}]
        )
        messages.append(conv_msg)
        assert conv_msg["type"] == "conversation.item.create"
        print("  ✅ Conversation item message created")
        
        # Response create message
        response_msg = MessageFactory.response_create()
        messages.append(response_msg)
        assert response_msg["type"] == "response.create"
        print("  ✅ Response create message created")
        
        # Validate all messages
        for msg in messages:
            try:
                MessageValidator.validate_outgoing(msg)
            except ValueError as e:
                print(f"  ❌ Message validation failed: {e}")
                return False
        print("  ✅ All messages pass validation")
        
        # Test message parser with real server event
        server_event = {
            "type": "response.audio.delta",
            "response_id": "resp_123",
            "delta": "base64audiodata==",
            "item_id": "item_456"
        }
        
        msg_type = MessageParser.get_message_type(server_event)
        assert msg_type == "response.audio.delta"
        assert MessageParser.is_audio_response(server_event)
        print("  ✅ Message parser works")
        
        # Test error parsing
        error_msg = {
            "type": "error",
            "error": {
                "type": "invalid_request",
                "message": "Invalid audio format"
            }
        }
        assert MessageParser.is_error(error_msg)
        error_details = MessageParser.extract_error(error_msg)
        assert error_details["type"] == "invalid_request"
        print("  ✅ Error parsing works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Message protocol test failed: {e}")
        logger.exception("Message protocol error")
        return False


def test_session_manager():
    """Test session manager with real sessions"""
    print("\n📊 Testing Session Manager Module...")
    
    try:
        from realtimevoiceapi.session_manager import SessionManager, SessionState
        from realtimevoiceapi.voice_engine import VoiceEngineConfig
        import asyncio
        import time
        
        # Create session manager
        manager = SessionManager()
        
        # Create real session config using VoiceEngineConfig
        config = VoiceEngineConfig(
            api_key="test_key",
            voice="alloy",
            mode="fast",
            metadata={
                "temperature": 0.8
            }
        )
        
        # Create session with correct parameters (provider, stream_id, config)
        stream_id = f"stream_{int(time.time())}"
        session = manager.create_session(
            provider="openai",
            stream_id=stream_id,
            config=config.to_engine_config()
        )
        
        assert session.id.startswith("session_")
        assert session.state == SessionState.CREATED
        print("  ✅ Session creation works")
        
        # Get session
        retrieved = manager.get_session(session.id)
        assert retrieved.id == session.id
        print("  ✅ Session retrieval works")
        
        # Update state
        session.state = SessionState.ACTIVE
        session.start_time = time.time()
        assert session.state == SessionState.ACTIVE
        print("  ✅ State update works")
        
        # Track usage
        session.usage.audio_input_seconds = 30
        session.usage.audio_output_seconds = 45
        session.usage.text_input_tokens = 150
        session.usage.text_output_tokens = 500
        
        total_usage = manager.get_total_usage()
        assert total_usage.audio_input_seconds == 30
        assert total_usage.text_output_tokens == 500
        print("  ✅ Usage tracking works")
        
        # Test filtering
        active_sessions = manager.get_sessions(state=SessionState.ACTIVE)
        assert len(active_sessions) == 1
        assert active_sessions[0].id == session.id
        print("  ✅ Session filtering works")
        
        # End session
        session.state = SessionState.ENDED
        session.end_time = time.time()
        
        # Test cleanup of old sessions
        old_stream_id = f"stream_{int(time.time())}_old"
        old_session = manager.create_session(
            provider="openai",
            stream_id=old_stream_id,
            config=config.to_engine_config()
        )
        old_session.state = SessionState.ENDED
        old_session.end_time = time.time() - 3700  # Over an hour ago
        
        # Run cleanup
        asyncio.run(manager.cleanup_old_sessions(max_age_seconds=3600))
        
        # Old session should be removed
        assert manager.get_session(old_session.id) is None
        assert manager.get_session(session.id) is not None  # Recent one kept
        print("  ✅ Session cleanup works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Session manager test failed: {e}")
        logger.exception("Session manager error")
        return False


def main():
    """Run all core module tests"""
    print("🧪 RealtimeVoiceAPI - Test 01: Core Modules")
    print("=" * 60)
    print("Testing basic modules with real data (no mocks)")
    print()
    
    tests = [
        ("Audio Types", test_audio_types),
        ("Stream Protocol", test_stream_protocol),
        ("Provider Protocol", test_provider_protocol),
        ("Message Protocol", test_message_protocol),
        ("Session Manager", test_session_manager),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All core modules working with real data!")
        print("✨ Core modules verified:")
        print("  - Audio types with real calculations")
        print("  - Stream protocol with real events")
        print("  - Provider protocol with real pricing")
        print("  - Message protocol with real validation")
        print("  - Session manager with real sessions")
        print("\nNext: Run test_02_audio_modules.py")
    else:
        print(f"\n❌ {total - passed} core module(s) need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)