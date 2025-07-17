#!/usr/bin/env python3
"""
Test 03: Messaging - Test message protocol and WebSocket handling

Tests:
- Message creation and validation
- WebSocket connection (mock)
- Message serialization
- Protocol compliance
"""

import sys
import logging
import asyncio
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_message_creation():
    """Test creating various message types"""
    print("\n✉️ Testing Message Creation...")
    
    try:
        from realtimevoiceapi.message_protocol import MessageFactory, ClientMessageType
        
        # Test all message types
        messages = []
        
        # Session update
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
        messages.append(("session_update", msg))
        
        # Audio append
        msg = MessageFactory.input_audio_buffer_append("base64_audio_data_here")
        messages.append(("audio_append", msg))
        
        # Audio commit
        msg = MessageFactory.input_audio_buffer_commit()
        messages.append(("audio_commit", msg))
        
        # Audio clear
        msg = MessageFactory.input_audio_buffer_clear()
        messages.append(("audio_clear", msg))
        
        # Conversation item (text)
        msg = MessageFactory.conversation_item_create(
            item_type="message",
            role="user",
            content=[{"type": "text", "text": "Hello, AI!"}]
        )
        messages.append(("conversation_text", msg))
        
        # Response create
        msg = MessageFactory.response_create(
            modalities=["text", "audio"],
            temperature=0.7,
            max_output_tokens=150
        )
        messages.append(("response_create", msg))
        
        # Response cancel
        msg = MessageFactory.response_cancel()
        messages.append(("response_cancel", msg))
        
        # Validate all messages
        all_valid = True
        for name, msg in messages:
            assert msg["type"] in [e.value for e in ClientMessageType]
            assert "event_id" in msg
            assert msg["event_id"].startswith("evt_")
            print(f"  ✅ {name}: {msg['type']}")
        
        print(f"  ✅ Created {len(messages)} message types successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Message creation failed: {e}")
        logger.exception("Message creation error")
        return False


def test_message_validation():
    """Test message validation"""
    print("\n✅ Testing Message Validation...")
    
    try:
        from realtimevoiceapi.message_protocol import (
            MessageFactory, MessageValidator, ClientMessageType
        )
        
        # Test valid messages
        valid_messages = [
            MessageFactory.session_update(modalities=["audio"]),
            MessageFactory.input_audio_buffer_append("data"),
            MessageFactory.conversation_item_create("message", role="user"),
            MessageFactory.response_create()
        ]
        
        for msg in valid_messages:
            assert MessageValidator.validate_outgoing(msg) == True
        print("  ✅ Valid messages pass validation")
        
        # Test invalid messages
        invalid_messages = [
            {},  # No type
            {"type": "invalid.type"},  # Unknown type
            {"type": "session.update"},  # Missing required session field
            {"type": "input_audio_buffer.append"},  # Missing audio field
        ]
        
        for msg in invalid_messages:
            try:
                MessageValidator.validate_outgoing(msg)
                assert False, f"Should have failed: {msg}"
            except ValueError:
                pass  # Expected
        print("  ✅ Invalid messages are rejected")
        
        # Test message field validation
        msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.8,
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5
            }
        )
        
        assert msg["session"]["modalities"] == ["text", "audio"]
        assert msg["session"]["voice"] == "alloy"
        assert msg["session"]["temperature"] == 0.8
        assert msg["session"]["turn_detection"]["type"] == "server_vad"
        print("  ✅ Message fields are properly structured")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Message validation failed: {e}")
        logger.exception("Message validation error")
        return False


def test_message_parsing():
    """Test parsing incoming messages"""
    print("\n🔍 Testing Message Parsing...")
    
    try:
        from realtimevoiceapi.message_protocol import MessageParser, ServerMessageType
        
        # Test various server messages
        test_messages = [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request",
                    "message": "Test error"
                }
            },
            {
                "type": "response.audio.delta",
                "delta": "base64_audio_chunk"
            },
            {
                "type": "response.text.delta", 
                "delta": "Hello, "
            },
            {
                "type": "response.done",
                "response": {"id": "resp_123"}
            }
        ]
        
        # Test message type extraction
        for msg in test_messages:
            msg_type = MessageParser.get_message_type(msg)
            assert msg_type == msg["type"]
        print("  ✅ Message type extraction works")
        
        # Test error detection
        assert MessageParser.is_error(test_messages[0]) == True
        assert MessageParser.is_error(test_messages[1]) == False
        error_details = MessageParser.extract_error(test_messages[0])
        assert error_details["message"] == "Test error"
        print("  ✅ Error detection works")
        
        # Test audio response detection
        assert MessageParser.is_audio_response(test_messages[1]) == True
        assert MessageParser.is_audio_response(test_messages[2]) == False
        audio_delta = MessageParser.extract_audio_delta(test_messages[1])
        assert audio_delta == "base64_audio_chunk"
        print("  ✅ Audio response parsing works")
        
        # Test text response detection
        assert MessageParser.is_text_response(test_messages[2]) == True
        assert MessageParser.is_text_response(test_messages[1]) == False
        text_delta = MessageParser.extract_text_delta(test_messages[2])
        assert text_delta == "Hello, "
        print("  ✅ Text response parsing works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Message parsing failed: {e}")
        logger.exception("Message parsing error")
        return False


async def test_websocket_connection_config():
    """Test WebSocket connection configuration"""
    print("\n🔌 Testing WebSocket Connection Configuration...")
    
    try:
        from realtimevoiceapi.websocket_connection import (
            ConnectionConfig, ConnectionState, SerializationFormat,
            JsonSerializer, ConnectionMetrics
        )
        
        # Test connection config
        config = ConnectionConfig(
            url="wss://api.example.com/v1/realtime",
            headers={"Authorization": "Bearer test_key"},
            ping_interval=20.0,
            ping_timeout=10.0,
            auto_reconnect=True,
            reconnect_max_attempts=5,
            serialization_format=SerializationFormat.JSON
        )
        
        assert config.url == "wss://api.example.com/v1/realtime"
        assert config.headers["Authorization"] == "Bearer test_key"
        assert config.auto_reconnect == True
        print("  ✅ Connection config works")
        
        # Test serializer
        serializer = JsonSerializer()
        test_data = {"type": "test", "data": [1, 2, 3]}
        serialized = serializer.serialize(test_data)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == test_data
        print("  ✅ JSON serializer works")
        
        # Test connection states
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTED.value == "connected"
        print("  ✅ Connection states defined")
        
        # Test metrics
        metrics = ConnectionMetrics()
        metrics.on_connect()
        assert metrics.connect_count == 1
        
        metrics.on_message_sent(100)
        assert metrics.messages_sent == 1
        assert metrics.bytes_sent == 100
        
        stats = metrics.get_stats()
        assert stats["connects"] == 1
        assert stats["messages_sent"] == 1
        print("  ✅ Connection metrics work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ WebSocket config test failed: {e}")
        logger.exception("WebSocket config error")
        return False


async def test_fast_vs_big_lane_connections():
    """Test different connection configurations"""
    print("\n🚀 Testing Fast vs Big Lane Connections...")
    
    try:
        from realtimevoiceapi.websocket_connection import (
            FastLaneConnection, BigLaneConnection,
            ConnectionConfig
        )
        
        # Test fast lane connection config
        fast_conn = FastLaneConnection(
            url="wss://api.openai.com/v1/realtime",
            headers={"Authorization": "Bearer test"}
        )
        
        assert fast_conn.config.enable_message_queue == False
        assert fast_conn.config.enable_metrics == False
        assert fast_conn.config.auto_reconnect == False
        print("  ✅ Fast lane connection configured correctly")
        
        # Test big lane connection config
        big_conn = BigLaneConnection(
            url="wss://api.openai.com/v1/realtime",
            headers={"Authorization": "Bearer test"}
        )
        
        assert big_conn.config.enable_message_queue == True
        assert big_conn.config.enable_metrics == True
        assert big_conn.config.auto_reconnect == True
        print("  ✅ Big lane connection configured correctly")
        
        # Verify different behaviors
        assert fast_conn.metrics is None  # No metrics in fast lane
        assert big_conn.metrics is not None  # Metrics in big lane
        
        assert fast_conn.send_queue is None  # No queue in fast lane
        assert big_conn.send_queue is not None  # Queue in big lane
        
        print("  ✅ Fast and big lane have correct feature sets")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Lane comparison test failed: {e}")
        logger.exception("Lane comparison error")
        return False


async def test_message_flow_integration():
    """Test message creation, validation, and serialization flow"""
    print("\n🔄 Testing Message Flow Integration...")
    
    try:
        from realtimevoiceapi.message_protocol import (
            MessageFactory, MessageValidator, ProtocolInfo
        )
        from realtimevoiceapi.websocket_connection import JsonSerializer
        
        # Create a complex session configuration message
        session_config = {
            "modalities": ["text", "audio"],
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True
            },
            "temperature": 0.8,
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }]
        }
        
        # Create message
        msg = MessageFactory.session_update(**session_config)
        print("  ✅ Complex message created")
        
        # Validate
        assert MessageValidator.validate_outgoing(msg) == True
        print("  ✅ Message validates")
        
        # Validate protocol values
        assert ProtocolInfo.is_valid_voice(session_config["voice"])
        assert ProtocolInfo.is_valid_audio_format(session_config["input_audio_format"])
        assert all(ProtocolInfo.is_valid_modality(m) for m in session_config["modalities"])
        print("  ✅ Protocol values valid")
        
        # Serialize for transmission
        serializer = JsonSerializer()
        serialized = serializer.serialize(msg)
        assert isinstance(serialized, str)
        assert len(serialized) > 100  # Should be substantial
        print("  ✅ Message serialized")
        
        # Deserialize (as if received)
        deserialized = serializer.deserialize(serialized)
        assert deserialized["type"] == msg["type"]
        assert deserialized["session"]["voice"] == "alloy"
        print("  ✅ Message deserialized correctly")
        
        # Full round trip maintains structure
        assert json.dumps(msg, sort_keys=True) == json.dumps(deserialized, sort_keys=True)
        print("  ✅ Full round trip preserves message")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Message flow test failed: {e}")
        logger.exception("Message flow error")
        return False


def main():
    """Run all messaging tests"""
    print("🧪 RealtimeVoiceAPI - Test 03: Messaging")
    print("=" * 60)
    print("Testing message protocol and WebSocket configuration")
    print()
    
    tests = [
        ("Message Creation", test_message_creation),
        ("Message Validation", test_message_validation),
        ("Message Parsing", test_message_parsing),
        ("WebSocket Configuration", test_websocket_connection_config),
        ("Fast vs Big Lane Connections", test_fast_vs_big_lane_connections),
        ("Message Flow Integration", test_message_flow_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
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
        print("\n🎉 All messaging modules working correctly!")
        print("Next: Run test_04_fast_lane_units.py")
    else:
        print(f"\n❌ {total - passed} messaging module(s) need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)