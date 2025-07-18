#!/usr/bin/env python3
"""
Test 01: Core Modules - Test basic modules in isolation

Tests:
- audio_types: Type definitions and data structures
- stream_protocol: Protocol definitions  
- provider_protocol: Provider contracts
- message_protocol: Message creation and validation


# python -m realtimevoiceapi.smoke_tests.test_01_core_modules


"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_audio_types():
    """Test audio type definitions"""
    print("\n🎵 Testing Audio Types Module...")
    
    try:
        from realtimevoiceapi.audio_types import (
            AudioFormat, AudioConfig, AudioQuality, 
            VADType, VADConfig, AudioMetadata, BufferConfig
        )
        
        # Test AudioFormat enum
        assert AudioFormat.PCM16.value == "pcm16"
        assert AudioFormat.PCM16.bytes_per_sample == 2
        print("  ✅ AudioFormat enum works")
        
        # Test AudioConfig
        config = AudioConfig()
        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.bit_depth == 16
        
        # Test computed properties
        assert config.frame_size == 2  # 1 channel * 16-bit / 8
        assert config.bytes_per_second == 48000  # 24000 * 2
        assert config.chunk_size_bytes(100) == 4800  # 100ms of audio
        print("  ✅ AudioConfig calculations correct")
        
        # Test AudioQuality presets
        low_quality = AudioQuality.LOW.to_config()
        assert low_quality.sample_rate == 16000
        print("  ✅ AudioQuality presets work")
        
        # Test VADConfig
        vad_config = VADConfig(
            type=VADType.ENERGY_BASED,
            energy_threshold=0.02
        )
        assert vad_config.type == VADType.ENERGY_BASED
        assert 0.0 <= vad_config.energy_threshold <= 1.0
        print("  ✅ VADConfig validation works")
        
        # Test AudioMetadata
        metadata = AudioMetadata(
            format=AudioFormat.PCM16,
            duration_ms=1000.0,
            size_bytes=48000
        )
        metadata_dict = metadata.to_dict()
        assert metadata_dict["format"] == "pcm16"
        assert metadata_dict["duration_ms"] == 1000.0
        print("  ✅ AudioMetadata serialization works")
        
        # Test BufferConfig
        buffer_config = BufferConfig(
            max_size_bytes=1024 * 1024,
            overflow_strategy="drop_oldest"
        )
        assert buffer_config.max_size_bytes == 1024 * 1024
        print("  ✅ BufferConfig works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio types test failed: {e}")
        logger.exception("Audio types error")
        return False


def test_stream_protocol():
    """Test stream protocol definitions"""
    print("\n🌊 Testing Stream Protocol Module...")
    
    try:
        from realtimevoiceapi.stream_protocol import (
            StreamEventType, StreamEvent, StreamState,
            AudioFormat, StreamConfig, StreamCapabilities,
            StreamCapability, StreamMetrics
        )
        
        # Test StreamEventType enum
        assert StreamEventType.STREAM_STARTED.value == "stream.started"
        assert StreamEventType.AUDIO_OUTPUT_CHUNK.value == "audio.output.chunk"
        print("  ✅ StreamEventType enum works")
        
        # Test StreamState enum
        assert StreamState.IDLE.value == "idle"
        assert StreamState.ACTIVE.value == "active"
        print("  ✅ StreamState enum works")
        
        # Test StreamEvent
        event = StreamEvent(
            type=StreamEventType.AUDIO_OUTPUT_CHUNK,
            stream_id="test_stream_123",
            timestamp=1234567890.0,
            data={"audio": b"test_audio_data"}
        )
        assert event.type == StreamEventType.AUDIO_OUTPUT_CHUNK
        assert event.stream_id == "test_stream_123"
        assert event.data["audio"] == b"test_audio_data"
        print("  ✅ StreamEvent creation works")
        
        # Test AudioFormat in protocol
        audio_format = AudioFormat(
            sample_rate=24000,
            channels=1,
            bit_depth=16
        )
        format_dict = audio_format.to_dict()
        assert format_dict["sample_rate"] == 24000
        print("  ✅ AudioFormat works")
        
        # Test StreamConfig
        stream_config = StreamConfig(
            provider="openai",
            mode="audio",
            audio_format=audio_format,
            enable_vad=True
        )
        assert stream_config.provider == "openai"
        assert stream_config.enable_vad == True
        print("  ✅ StreamConfig works")
        
        # Test StreamCapabilities
        capabilities = StreamCapabilities(
            supported=[StreamCapability.AUDIO_INPUT, StreamCapability.VAD],
            audio_formats=[audio_format]
        )
        assert capabilities.supports(StreamCapability.AUDIO_INPUT)
        assert not capabilities.supports(StreamCapability.FUNCTION_CALLING)
        print("  ✅ StreamCapabilities works")
        
        # Test StreamMetrics
        metrics = StreamMetrics(
            bytes_sent=1000,
            bytes_received=2000,
            chunks_sent=10,
            chunks_received=20
        )
        assert metrics.throughput_bps == 0.0  # No duration
        metrics.start_time = 0.0
        metrics.end_time = 1.0
        
        assert abs(metrics.duration_seconds - 1.0) < 0.001
        assert metrics.throughput_bps == 3000.0  # (1000 + 2000) / 1
        print("  ✅ StreamMetrics calculations work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stream protocol test failed: {e}")
        logger.exception("Stream protocol error")
        return False


def test_provider_protocol():
    """Test provider protocol definitions"""
    print("\n🔌 Testing Provider Protocol Module...")
    
    try:
        from realtimevoiceapi.provider_protocol import (
            ProviderFeature, ProviderCapabilities, CostModel, CostUnit,
            Usage, Cost, ProviderConfig, ProviderRegistry,
            VoiceConfig, QualityPreset, FunctionDefinition
        )
        
        # Test ProviderFeature enum
        assert ProviderFeature.REALTIME_VOICE.value == "realtime_voice"
        assert ProviderFeature.SERVER_VAD.value == "server_vad"
        print("  ✅ ProviderFeature enum works")
        
        # Test ProviderCapabilities
        capabilities = ProviderCapabilities(
            provider_name="test_provider",
            features=[ProviderFeature.REALTIME_VOICE, ProviderFeature.CLIENT_VAD],
            supported_audio_formats=["pcm16"],
            supported_sample_rates=[24000],
            max_audio_duration_ms=300000,
            min_audio_chunk_ms=100,
            available_voices=["alloy", "echo"],
            supports_voice_config=True,
            supported_languages=["en", "es", "fr"]
        )
        assert capabilities.supports(ProviderFeature.REALTIME_VOICE)
        assert not capabilities.supports(ProviderFeature.VOICE_CLONING)
        print("  ✅ ProviderCapabilities works")
        
        # Test CostModel
        cost_model = CostModel(
            audio_input_cost=0.06,
            audio_input_unit=CostUnit.PER_MINUTE,
            audio_output_cost=0.24,
            audio_output_unit=CostUnit.PER_MINUTE
        )
        assert cost_model.audio_input_cost == 0.06
        assert cost_model.currency == "USD"
        print("  ✅ CostModel works")
        
        # Test Usage tracking
        usage = Usage(
            audio_input_seconds=60.0,
            audio_output_seconds=30.0,
            text_input_tokens=100,
            text_output_tokens=200
        )
        
        # Test usage addition
        usage2 = Usage(audio_input_seconds=30.0)
        usage.add(usage2)
        assert usage.audio_input_seconds == 90.0
        print("  ✅ Usage tracking works")
        
        # Test Cost calculation
        cost = Cost(
            audio_cost=1.50,
            text_cost=0.50,
            session_cost=0.10
        )
        assert cost.total == 2.10
        cost_dict = cost.to_dict()
        assert cost_dict["total"] == 2.10
        print("  ✅ Cost calculation works")
        
        # Test ProviderConfig
        config = ProviderConfig(
            api_key="test_key_123",
            timeout=30.0,
            max_retries=3
        )
        assert config.api_key == "test_key_123"
        assert config.metadata is not None
        print("  ✅ ProviderConfig works")
        
        # Test ProviderRegistry
        registry = ProviderRegistry()
        
        # Create a mock provider
        class MockProvider:
            @property
            def name(self):
                return "mock_provider"
            
            def get_capabilities(self):
                return capabilities
        
        mock_provider = MockProvider()
        registry.register(mock_provider, set_as_default=True)
        
        assert "mock_provider" in registry.list_providers()
        retrieved = registry.get("mock_provider")
        assert retrieved.name == "mock_provider"
        
        # Test default provider
        default = registry.get()  # No name = default
        assert default.name == "mock_provider"
        print("  ✅ ProviderRegistry works")
        
        # Test VoiceConfig
        voice_config = VoiceConfig(
            voice_id="alloy",
            speed=1.2,
            pitch=0.9
        )
        assert voice_config.voice_id == "alloy"
        assert voice_config.speed == 1.2
        print("  ✅ VoiceConfig works")
        
        # Test FunctionDefinition
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        )
        assert func_def.name == "get_weather"
        print("  ✅ FunctionDefinition works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Provider protocol test failed: {e}")
        logger.exception("Provider protocol error")
        return False


def test_message_protocol():
    """Test message protocol module"""
    print("\n✉️ Testing Message Protocol Module...")
    
    try:
        from realtimevoiceapi.message_protocol import (
            ClientMessageType, ServerMessageType, MessageFactory,
            MessageValidator, MessageParser, ProtocolInfo
        )
        
        # Test message type enums
        assert ClientMessageType.SESSION_UPDATE.value == "session.update"
        assert ServerMessageType.ERROR.value == "error"
        print("  ✅ Message type enums work")
        
        # Test MessageFactory
        msg = MessageFactory.create_base_message(ClientMessageType.SESSION_UPDATE)
        assert msg["type"] == "session.update"
        assert "event_id" in msg
        print("  ✅ Base message creation works")
        
        # Test session update message
        session_msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.8
        )
        assert session_msg["type"] == "session.update"
        assert session_msg["session"]["modalities"] == ["text", "audio"]
        assert session_msg["session"]["voice"] == "alloy"
        assert session_msg["session"]["temperature"] == 0.8
        print("  ✅ Session update message works")
        
        # Test audio message
        audio_msg = MessageFactory.input_audio_buffer_append("base64_audio_data")
        assert audio_msg["type"] == "input_audio_buffer.append"
        assert audio_msg["audio"] == "base64_audio_data"
        print("  ✅ Audio message creation works")
        
        # Test conversation item
        conv_msg = MessageFactory.conversation_item_create(
            item_type="message",
            role="user",
            content=[{"type": "text", "text": "Hello"}]
        )
        assert conv_msg["type"] == "conversation.item.create"
        assert conv_msg["item"]["type"] == "message"
        assert conv_msg["item"]["role"] == "user"
        print("  ✅ Conversation item creation works")
        
        # Test MessageValidator
        valid = MessageValidator.validate_outgoing(session_msg)
        assert valid == True
        print("  ✅ Message validation works")
        
        # Test invalid message
        try:
            MessageValidator.validate_outgoing({"no_type": "invalid"})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "missing required 'type'" in str(e)
            print("  ✅ Invalid message detection works")
        
        # Test MessageParser
        assert MessageParser.get_message_type(audio_msg) == "input_audio_buffer.append"
        assert MessageParser.is_error({"type": "error"}) == True
        assert MessageParser.is_audio_response({
            "type": "response.audio.delta"
        }) == True
        print("  ✅ Message parser works")
        
        # Test ProtocolInfo
        assert ProtocolInfo.is_valid_audio_format("pcm16") == True
        assert ProtocolInfo.is_valid_audio_format("invalid") == False
        assert ProtocolInfo.is_valid_voice("alloy") == True
        assert ProtocolInfo.is_valid_modality("audio") == True
        print("  ✅ Protocol info validation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Message protocol test failed: {e}")
        logger.exception("Message protocol error")
        return False


def test_session_manager():
    """Test session manager module"""
    print("\n📊 Testing Session Manager Module...")
    
    try:
        from realtimevoiceapi.session_manager import (
            SessionManager, Session, SessionState
        )
        
        manager = SessionManager()
        
        # Test session creation
        session = manager.create_session(
            provider="openai",
            stream_id="stream_123",
            config={"test": "config"}
        )
        
        assert session.provider == "openai"
        assert session.stream_id == "stream_123"
        assert session.state == SessionState.INITIALIZING
        assert session.id.startswith("sess_")
        print("  ✅ Session creation works")
        
        # Test session retrieval
        retrieved = manager.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        print("  ✅ Session retrieval works")
        
        # Test state update
        manager.update_state(session.id, SessionState.ACTIVE)
        assert session.state == SessionState.ACTIVE
        assert session.last_activity > 0
        print("  ✅ State update works")
        
        # Test usage tracking
        manager.track_usage(
            session.id,
            audio_seconds=10.5,
            text_tokens=100,
            function_calls=2
        )
        assert session.audio_seconds_used == 10.5
        assert session.text_tokens_used == 100
        assert session.function_calls_made == 2
        print("  ✅ Usage tracking works")
        
        # Test active sessions
        active = manager.get_active_sessions()
        assert len(active) == 1
        assert active[0].id == session.id
        
        # Test provider filtering
        openai_sessions = manager.get_active_sessions(provider="openai")
        assert len(openai_sessions) == 1
        other_sessions = manager.get_active_sessions(provider="anthropic")
        assert len(other_sessions) == 0
        print("  ✅ Session filtering works")
        
        # Test session ending
        manager.end_session(session.id)
        assert session.state == SessionState.ENDED
        assert "ended_at" in session.metadata
        assert "duration" in session.metadata
        print("  ✅ Session ending works")
        
        # Test cleanup
        import time
        time.sleep(0.1)  # Ensure some time passes
        manager.cleanup_old_sessions(max_age_seconds=0.05)
        assert manager.get_session(session.id) is None
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
    print("Testing basic modules in isolation (no external dependencies)")
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
        print("\n🎉 All core modules working correctly!")
        print("Next: Run test_02_audio_modules.py")
    else:
        print(f"\n❌ {total - passed} core module(s) need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)