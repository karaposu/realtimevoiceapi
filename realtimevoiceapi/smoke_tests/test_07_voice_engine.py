
#!/usr/bin/env python3
"""
Test 07: Voice Engine - Test the main VoiceEngine class

Tests:
- Engine initialization with different configs
- Explicit mode selection (fast vs big)
- Basic operations
- Configuration validation
- Factory methods


python -m realtimevoiceapi.smoke_tests.test_07_voice_engine
"""

import sys
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mock API key for testing
TEST_API_KEY = "test_sk_1234567890"


def test_voice_engine_creation():
    """Test VoiceEngine creation methods"""
    print("\n🎯 Testing VoiceEngine Creation...")
    
    try:
        from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
        
        # Test direct creation with explicit mode
        engine1 = VoiceEngine(api_key=TEST_API_KEY, mode="fast")
        assert engine1.config.api_key == TEST_API_KEY
        assert engine1.mode == "fast"
        print("  ✅ Direct creation works")
        
        # Test with config object
        config = VoiceEngineConfig(
            api_key=TEST_API_KEY,
            mode="fast",
            vad_enabled=True,
            voice="echo"
        )
        engine2 = VoiceEngine(config=config)
        assert engine2.config.api_key == TEST_API_KEY
        assert engine2.config.voice == "echo"
        assert engine2.mode == "fast"
        print("  ✅ Config-based creation works")
        
        # Test factory methods
        simple_engine = VoiceEngine.create_simple(TEST_API_KEY, voice="alloy")
        assert simple_engine.config.mode == "fast"
        assert simple_engine.config.latency_mode == "ultra_low"
        print("  ✅ Simple factory method works")
        
        # Test from config file
        config_data = {
            "api_key": TEST_API_KEY,
            "mode": "fast",
            "vad_enabled": True,
            "voice": "shimmer"
        }
        
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            file_engine = VoiceEngine.from_config_file(temp_path)
            assert file_engine.config.voice == "shimmer"
            print("  ✅ Config file loading works")
        finally:
            os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ VoiceEngine creation failed: {e}")
        logger.exception("VoiceEngine creation error")
        return False


def test_mode_selection():
    """Test explicit mode selection"""
    print("\n🔄 Testing Mode Selection...")
    
    try:
        from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
        
        # Test fast mode
        config1 = VoiceEngineConfig(
            api_key=TEST_API_KEY,
            mode="fast",
            vad_type="client",
            latency_mode="ultra_low"
        )
        engine1 = VoiceEngine(config=config1)
        assert engine1.mode == "fast"
        print("  ✅ Fast mode selection works")
        
        # Test big mode - should fail (not implemented)
        config2 = VoiceEngineConfig(
            api_key=TEST_API_KEY,
            mode="big",
            enable_transcription=True,
            enable_functions=True
        )
        
        try:
            engine2 = VoiceEngine(config=config2)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError as e:
            assert "Big lane strategy not yet implemented" in str(e)
            print("  ✅ Big mode raises NotImplementedError (expected)")
        
        # Test invalid mode
        try:
            engine3 = VoiceEngine(api_key=TEST_API_KEY, mode="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Must be 'fast' or 'big'" in str(e)
            print("  ✅ Invalid mode raises ValueError")
        
        # Test default mode (should be fast)
        engine4 = VoiceEngine(api_key=TEST_API_KEY)
        assert engine4.mode == "fast"
        print("  ✅ Default mode is 'fast'")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Mode selection test failed: {e}")
        logger.exception("Mode selection error")
        return False


def test_configuration_validation():
    """Test configuration validation"""
    print("\n✅ Testing Configuration Validation...")
    
    try:
        from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
        
        # Test valid configuration
        valid_config = VoiceEngineConfig(
            api_key=TEST_API_KEY,
            mode="fast",
            sample_rate=24000,
            chunk_duration_ms=100,
            vad_threshold=0.02
        )
        
        engine = VoiceEngine(config=valid_config)
        assert engine.config.sample_rate == 24000
        print("  ✅ Valid configuration accepted")
        
        # Test config conversion
        engine_config = valid_config.to_engine_config()
        assert engine_config.api_key == TEST_API_KEY
        assert engine_config.metadata["sample_rate"] == 24000
        print("  ✅ Config conversion works")
        
        # Test default values
        default_config = VoiceEngineConfig(api_key=TEST_API_KEY)
        assert default_config.mode == "fast"  # Default is now fast
        assert default_config.vad_enabled == True
        assert default_config.latency_mode == "balanced"
        print("  ✅ Default values applied correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration validation failed: {e}")
        logger.exception("Configuration validation error")
        return False


async def test_engine_lifecycle():
    """Test engine lifecycle methods"""
    print("\n🔄 Testing Engine Lifecycle...")
    
    try:
        from realtimevoiceapi import VoiceEngine
        from realtimevoiceapi.stream_protocol import StreamState
        
        engine = VoiceEngine(api_key=TEST_API_KEY, mode="fast")
        
        # Test initial state
        assert engine._is_connected == False
        assert engine._is_listening == False
        assert engine.get_state() == StreamState.IDLE
        print("  ✅ Initial state correct")
        
        # Test metrics
        metrics = engine.get_metrics()
        assert metrics["mode"] == "fast"
        assert metrics["connected"] == False
        assert metrics["listening"] == False
        print("  ✅ Initial metrics correct")
        
        # Test usage
        usage = await engine.get_usage()
        assert usage.audio_input_seconds == 0
        assert usage.text_input_tokens == 0
        print("  ✅ Usage tracking initialized")
        
        # Test cost estimation
        cost = await engine.estimate_cost()
        assert cost.total == 0
        print("  ✅ Cost estimation works")
        
        # Test context manager support
        assert hasattr(engine, '__aenter__')
        assert hasattr(engine, '__aexit__')
        print("  ✅ Context manager support available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Engine lifecycle test failed: {e}")
        logger.exception("Engine lifecycle error")
        return False


def test_callback_configuration():
    """Test callback configuration"""
    print("\n📞 Testing Callback Configuration...")
    
    try:
        from realtimevoiceapi import VoiceEngine
        
        engine = VoiceEngine(api_key=TEST_API_KEY, mode="fast")
        
        # Test callback assignment
        audio_responses = []
        text_responses = []
        errors = []
        
        engine.on_audio_response = lambda audio: audio_responses.append(audio)
        engine.on_text_response = lambda text: text_responses.append(text)
        engine.on_error = lambda error: errors.append(error)
        
        assert engine.on_audio_response is not None
        assert engine.on_text_response is not None
        assert engine.on_error is not None
        print("  ✅ Callbacks assigned successfully")
        
        # Test direct calls (simulate responses)
        test_audio = b"test_audio_data"
        test_text = "Test response"
        test_error = Exception("Test error")
        
        engine.on_audio_response(test_audio)
        engine.on_text_response(test_text)
        engine.on_error(test_error)
        
        assert len(audio_responses) == 1
        assert audio_responses[0] == test_audio
        assert len(text_responses) == 1
        assert text_responses[0] == test_text
        assert len(errors) == 1
        assert errors[0] == test_error
        print("  ✅ Callbacks execute correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Callback configuration test failed: {e}")
        logger.exception("Callback configuration error")
        return False


def test_audio_component_setup():
    """Test audio component setup for fast lane"""
    print("\n🎵 Testing Audio Component Setup...")
    
    try:
        from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
        
        config = VoiceEngineConfig(
            api_key=TEST_API_KEY,
            mode="fast",
            vad_enabled=True,
            vad_threshold=0.03,
            vad_speech_start_ms=150,
            vad_speech_end_ms=600
        )
        
        engine = VoiceEngine(config=config)
        
        # After initialization, fast lane should be ready
        assert engine._strategy is not None
        print("  ✅ Strategy created")
        
        # Check configuration was passed through
        strategy_config = engine.config.to_engine_config()
        assert strategy_config.enable_vad == True
        assert strategy_config.metadata["vad_threshold"] == 0.03
        assert strategy_config.metadata["vad_speech_start_ms"] == 150
        print("  ✅ Configuration passed to strategy")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio component setup test failed: {e}")
        logger.exception("Audio component setup error")
        return False


def test_convenience_methods():
    """Test convenience methods and functions"""
    print("\n🎯 Testing Convenience Methods...")
    
    try:
        from realtimevoiceapi.voice_engine import (
            create_voice_session,
            run_voice_engine
        )
        from realtimevoiceapi import VoiceEngine
        
        # Test factory method on VoiceEngine class
        simple = VoiceEngine.create_simple(TEST_API_KEY)
        assert simple.config.mode == "fast"
        print("  ✅ VoiceEngine.create_simple works")
        
        # Test advanced engine creation (will fail - not implemented)
        try:
            advanced = VoiceEngine.create_advanced(TEST_API_KEY)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            print("  ✅ VoiceEngine.create_advanced raises NotImplementedError (expected)")
        
        # Test async session creation exists
        assert asyncio.iscoroutinefunction(create_voice_session)
        print("  ✅ create_voice_session is async function")
        
        # Test run_voice_engine exists
        assert callable(run_voice_engine)
        print("  ✅ run_voice_engine is callable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Convenience methods test failed: {e}")
        logger.exception("Convenience methods error")
        return False


def main():
    """Run all voice engine tests"""
    print("🧪 RealtimeVoiceAPI - Test 07: Voice Engine")
    print("=" * 60)
    print("Testing the main VoiceEngine class")
    print()
    
    tests = [
        ("VoiceEngine Creation", test_voice_engine_creation),
        ("Mode Selection", test_mode_selection),
        ("Configuration Validation", test_configuration_validation),
        ("Engine Lifecycle", test_engine_lifecycle),
        ("Callback Configuration", test_callback_configuration),
        ("Audio Component Setup", test_audio_component_setup),
        ("Convenience Methods", test_convenience_methods),
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
        print("\n🎉 All VoiceEngine tests passed!")
        print("✨ VoiceEngine verified:")
        print("  - Multiple creation methods")
        print("  - Explicit mode selection")
        print("  - Configuration handling")
        print("  - Lifecycle management")
        print("  - Callback system")
        print("\n🚀 Your RealtimeVoiceAPI framework is ready to use!")
    else:
        print(f"\n❌ {total - passed} VoiceEngine test(s) failed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)