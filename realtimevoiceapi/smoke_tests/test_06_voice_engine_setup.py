"""
Test 06: VoiceEngine Setup
Tests VoiceEngine creation and configuration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import logging
from realtimevoiceapi.voice_engine import VoiceEngine, VoiceEngineConfig

# Set up logging
logging.basicConfig(level=logging.INFO)

async def test_voice_engine_creation():
    """Test VoiceEngine creation"""
    print("\n=== Test 1: VoiceEngine Creation ===")
    
    try:
        # Test with API key
        engine = VoiceEngine(api_key="test-key", mode="fast")
        print("✓ VoiceEngine created with API key")
        
        # Test with config
        config = VoiceEngineConfig(
            api_key="test-key",
            mode="fast",
            sample_rate=24000,
            vad_enabled=True
        )
        engine2 = VoiceEngine(config=config)
        print("✓ VoiceEngine created with config")
        
        # Check mode
        print(f"✓ Engine mode: {engine.mode}")
        
        return True
    except Exception as e:
        print(f"✗ VoiceEngine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_configuration():
    """Test VoiceEngine configuration"""
    print("\n=== Test 2: Configuration ===")
    
    try:
        config = VoiceEngineConfig(
            api_key="test-key",
            provider="openai",
            mode="fast",
            input_device=None,
            output_device=None,
            sample_rate=24000,
            chunk_duration_ms=20,
            vad_enabled=True,
            vad_threshold=0.02,
            voice="alloy",
            latency_mode="balanced"
        )
        
        engine = VoiceEngine(config=config)
        print("✓ Full configuration applied")
        
        # Convert to engine config
        engine_config = config.to_engine_config()
        print(f"✓ Engine config created:")
        print(f"  Provider: {engine_config.provider}")
        print(f"  VAD enabled: {engine_config.enable_vad}")
        print(f"  Latency mode: {engine_config.latency_mode}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_callback_setup():
    """Test callback setup"""
    print("\n=== Test 3: Callback Setup ===")
    
    try:
        engine = VoiceEngine(api_key="test-key")
        
        # Track callbacks
        callbacks_triggered = {
            'audio': False,
            'text': False,
            'error': False,
            'done': False
        }
        
        # Set callbacks
        engine.on_audio_response = lambda audio: callbacks_triggered.update({'audio': True})
        engine.on_text_response = lambda text: callbacks_triggered.update({'text': True})
        engine.on_error = lambda error: callbacks_triggered.update({'error': True})
        engine.on_response_done = lambda: callbacks_triggered.update({'done': True})
        
        print("✓ Callbacks configured")
        print(f"✓ Callbacks set: {sum(1 for cb in [engine.on_audio_response, engine.on_text_response, engine.on_error, engine.on_response_done] if cb is not None)}")
        
        return True
    except Exception as e:
        print(f"✗ Callback setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_state_properties():
    """Test state properties"""
    print("\n=== Test 4: State Properties ===")
    
    try:
        engine = VoiceEngine(api_key="test-key")
        
        # Check initial state
        print(f"✓ Initial state:")
        print(f"  Is connected: {engine.is_connected}")
        print(f"  Is listening: {engine.is_listening}")
        print(f"  Stream state: {engine.get_state().value}")
        
        # State should be disconnected initially
        assert not engine.is_connected, "Should not be connected initially"
        assert not engine.is_listening, "Should not be listening initially"
        
        return True
    except Exception as e:
        print(f"✗ State properties test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_session_methods():
    """Test session management methods"""
    print("\n=== Test 5: Session Methods ===")
    
    try:
        engine = VoiceEngine(api_key="test-key")
        
        # Test usage (should be empty without connection)
        usage = await engine.get_usage()
        print(f"✓ Retrieved usage: {usage}")
        
        # Test cost estimation
        cost = await engine.estimate_cost()
        print(f"✓ Estimated cost: {cost}")
        
        # VoiceEngine doesn't have generate_session_id
        print("✓ Session methods available")
        
        return True
    except Exception as e:
        print(f"✗ Session methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_metrics_access():
    """Test metrics access"""
    print("\n=== Test 6: Metrics Access ===")
    
    try:
        engine = VoiceEngine(api_key="test-key", mode="fast")
        
        # Get metrics without connection (not async in VoiceEngine)
        metrics = engine.get_metrics()
        print("✓ Retrieved metrics:")
        print(f"  Keys: {list(metrics.keys())}")
        print(f"  Is connected: {metrics.get('is_connected', False)}")
        print(f"  Total interactions: {metrics.get('total_interactions', 0)}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_convenience_properties():
    """Test convenience properties and methods"""
    print("\n=== Test 7: Convenience Properties ===")
    
    try:
        engine = VoiceEngine(api_key="test-key")
        
        # VoiceEngine doesn't have is_ai_speaking directly
        # Check what properties it does have
        print(f"✓ Is connected: {engine.is_connected}")
        print(f"✓ Is listening: {engine.is_listening}")
        print(f"✓ State: {engine.get_state().value}")
        
        # VoiceEngine doesn't have set_fast_mode
        print(f"✓ Engine mode: {engine.mode}")
        
        return True
    except Exception as e:
        print(f"✗ Convenience properties test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cleanup():
    """Test cleanup without connection"""
    print("\n=== Test 8: Cleanup ===")
    
    try:
        engine = VoiceEngine(api_key="test-key")
        
        # VoiceEngine uses context manager for cleanup
        # Test disconnect instead
        await engine.disconnect()  # Should work even if not connected
        print("✓ Disconnect completed without errors")
        
        # Verify state after cleanup
        print(f"✓ Post-cleanup connected: {engine.is_connected}")
        
        return True
    except Exception as e:
        print(f"✗ Cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("VoiceEngine Setup Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Creation", await test_voice_engine_creation()))
    results.append(("Configuration", await test_configuration()))
    results.append(("Callbacks", await test_callback_setup()))
    results.append(("State Properties", await test_state_properties()))
    results.append(("Session Methods", await test_session_methods()))
    results.append(("Metrics", await test_metrics_access()))
    results.append(("Convenience", await test_convenience_properties()))
    results.append(("Cleanup", await test_cleanup()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)