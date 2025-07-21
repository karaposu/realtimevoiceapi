"""
Test 04: BaseEngine Initialization
Tests BaseEngine creation, strategy setup, and basic initialization without connections.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import logging
from realtimevoiceapi.base_engine import BaseEngine
from realtimevoiceapi.strategies.base_strategy import EngineConfig
from realtimevoiceapi.core.stream_protocol import StreamEventType, StreamState

# Set up logging
logging.basicConfig(level=logging.INFO)

async def test_base_engine_creation():
    """Test basic BaseEngine creation"""
    print("\n=== Test 1: BaseEngine Creation ===")
    
    try:
        # Create BaseEngine
        engine = BaseEngine()
        print("✓ BaseEngine created")
        
        # Check initial state
        print(f"✓ Initial state:")
        print(f"  Is connected: {engine.is_connected}")
        print(f"  Is listening: {engine.is_listening}")
        print(f"  Is AI speaking: {engine.is_ai_speaking}")
        print(f"  Stream state: {engine.get_state().value}")
        
        return True
    except Exception as e:
        print(f"✗ BaseEngine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_strategy_creation():
    """Test strategy creation"""
    print("\n=== Test 2: Strategy Creation ===")
    
    try:
        engine = BaseEngine()
        
        # Create fast lane strategy
        strategy = engine.create_strategy("fast")
        print("✓ Fast lane strategy created")
        
        # Verify strategy
        assert engine.strategy is not None, "Strategy should be set"
        print(f"✓ Strategy type: {type(strategy).__name__}")
        
        # Test invalid mode
        try:
            engine2 = BaseEngine()
            engine2.create_strategy("invalid")
            print("✗ Should have raised ValueError for invalid mode")
            return False
        except ValueError:
            print("✓ Correctly rejected invalid mode")
        
        return True
    except Exception as e:
        print(f"✗ Strategy creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_strategy_initialization():
    """Test strategy initialization without connection"""
    print("\n=== Test 3: Strategy Initialization ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Create config
        config = EngineConfig(
            api_key="test-key",
            provider="openai"
        )
        
        # Initialize strategy
        await engine.initialize_strategy(config)
        print("✓ Strategy initialized")
        
        # Check state
        print(f"✓ Post-init state: {engine.get_state().value}")
        
        return True
    except Exception as e:
        print(f"✗ Strategy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_setup():
    """Test audio setup through AudioEngine"""
    print("\n=== Test 4: Audio Setup ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        print("✓ Audio setup completed")
        print(f"✓ Has audio: {engine.components.has_audio()}")
        print(f"✓ Audio engine created: {engine.components.audio_engine is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Audio setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_event_handler_setup():
    """Test event handler registration"""
    print("\n=== Test 5: Event Handler Setup ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Initialize strategy first
        config = EngineConfig(api_key="test-key", provider="openai")
        await engine.initialize_strategy(config)
        
        # Track handler calls
        handler_calls = {}
        
        def create_handler(event_type):
            def handler(event):
                handler_calls[event_type] = handler_calls.get(event_type, 0) + 1
            return handler
        
        # Setup handlers
        handlers = {
            StreamEventType.STREAM_STARTED: create_handler("started"),
            StreamEventType.AUDIO_OUTPUT_CHUNK: create_handler("audio"),
            StreamEventType.STREAM_ENDED: create_handler("ended"),
            StreamEventType.STREAM_ERROR: create_handler("error")
        }
        
        engine.setup_event_handlers(handlers)
        print("✓ Event handlers registered")
        
        # Check registry
        registry_metrics = engine.event_registry.get_metrics()
        print(f"✓ Registered handlers: {registry_metrics['registered_handlers']}")
        
        return True
    except Exception as e:
        print(f"✗ Event handler setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_state_management():
    """Test state management"""
    print("\n=== Test 6: State Management ===")
    
    try:
        engine = BaseEngine()
        
        # Get initial state metrics
        state_dict = engine.state.to_dict()
        print("✓ Initial state retrieved:")
        print(f"  Uptime: {state_dict['uptime']:.2f}s")
        print(f"  Total interactions: {state_dict['total_interactions']}")
        print(f"  Is ready: {state_dict['is_ready']}")
        
        # Mark interaction
        engine.state.mark_interaction()
        print("✓ Marked interaction")
        
        # Update audio chunks
        engine.state.total_audio_chunks_sent = 10
        engine.state.total_audio_chunks_received = 5
        
        # Get updated state
        updated_state = engine.state.to_dict()
        print("✓ Updated state:")
        print(f"  Total interactions: {updated_state['total_interactions']}")
        print(f"  Audio chunks sent: {updated_state['audio_chunks_sent']}")
        print(f"  Audio chunks received: {updated_state['audio_chunks_received']}")
        
        return True
    except Exception as e:
        print(f"✗ State management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_metrics_collection():
    """Test metrics collection without connection"""
    print("\n=== Test 7: Metrics Collection ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup audio for metrics
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Get metrics
        metrics = engine.get_metrics()
        print("✓ Retrieved metrics:")
        print(f"  Has audio engine: {'audio_engine' in metrics}")
        print(f"  Is connected: {metrics['is_connected']}")
        print(f"  Stream state: {metrics['stream_state']}")
        
        # Check audio engine metrics
        if 'audio_engine' in metrics:
            ae_metrics = metrics['audio_engine']
            print(f"✓ Audio engine metrics:")
            print(f"  Mode: {ae_metrics.get('mode', 'unknown')}")
            print(f"  Has buffer pool: {ae_metrics.get('has_buffer_pool', False)}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cleanup():
    """Test cleanup functionality"""
    print("\n=== Test 8: Cleanup ===")
    
    try:
        engine = BaseEngine()
        engine.create_strategy("fast")
        
        # Setup components
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Cleanup
        await engine.cleanup()
        print("✓ Cleanup completed")
        
        # Verify cleanup
        print(f"✓ Post-cleanup state:")
        print(f"  Is connected: {engine.state.is_connected}")
        print(f"  Stream state: {engine.state.stream_state.value}")
        print(f"  Components cleared: {engine.components.audio_engine is None}")
        
        return True
    except Exception as e:
        print(f"✗ Cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("BaseEngine Initialization Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Creation", await test_base_engine_creation()))
    results.append(("Strategy Creation", await test_strategy_creation()))
    results.append(("Strategy Init", await test_strategy_initialization()))
    results.append(("Audio Setup", await test_audio_setup()))
    results.append(("Event Handlers", await test_event_handler_setup()))
    results.append(("State Management", await test_state_management()))
    results.append(("Metrics", await test_metrics_collection()))
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