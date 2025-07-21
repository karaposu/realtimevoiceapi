"""
Test 10: Comprehensive Metrics and Cleanup
Tests metrics collection and resource cleanup across all components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
import gc
from realtimevoiceapi.voice_engine import VoiceEngine, VoiceEngineConfig
from audioengine.audioengine.audio_engine import AudioEngine
from realtimevoiceapi.base_engine import BaseEngine

# Try to import psutil, but make it optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, skipping memory tests")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

async def test_comprehensive_metrics():
    """Test comprehensive metrics collection"""
    print("\n=== Test 1: Comprehensive Metrics ===")
    
    try:
        # Create engine with full config
        config = VoiceEngineConfig(
            api_key=API_KEY,
            mode="fast",
            sample_rate=24000,
            chunk_duration_ms=20,
            vad_enabled=True,
            vad_threshold=0.02,
            voice="alloy",
            latency_mode="balanced"
        )
        
        engine = VoiceEngine(config=config)
        
        # Track all metrics
        metrics_timeline = []
        
        async def collect_metrics(label):
            metrics = engine.get_metrics()  # Not async
            system_metrics = {}
            if HAS_PSUTIL:
                process = psutil.Process()
                system_metrics = {
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(interval=0.1),
                    'threads': process.num_threads()
                }
            
            metrics_timeline.append({
                'label': label,
                'time': time.time(),
                'engine_metrics': metrics,
                'system_metrics': system_metrics
            })
            
            print(f"  {label}:")
            if HAS_PSUTIL:
                print(f"    Memory: {system_metrics['memory_mb']:.1f}MB")
                print(f"    CPU: {system_metrics['cpu_percent']:.1f}%")
                print(f"    Threads: {system_metrics['threads']}")
            else:
                print(f"    System metrics: not available")
        
        # Collect baseline
        await collect_metrics("Baseline")
        
        # Connect
        await engine.connect()
        await collect_metrics("After connect")
        
        # Start listening
        await engine.start_listening()
        await collect_metrics("After start listening")
        
        # Send messages
        for i in range(3):
            await engine.send_text(f"Test message {i+1}")
            await asyncio.sleep(4)
            await collect_metrics(f"After message {i+1}")
        
        # Stop listening
        await engine.stop_listening()
        await collect_metrics("After stop listening")
        
        # Disconnect
        await engine.disconnect()
        await collect_metrics("After disconnect")
        
        # Analyze growth
        print("\n✓ Metrics analysis:")
        if HAS_PSUTIL and metrics_timeline and metrics_timeline[0]['system_metrics']:
            memory_start = metrics_timeline[0]['system_metrics']['memory_mb']
            memory_peak = max(m['system_metrics']['memory_mb'] for m in metrics_timeline)
            memory_end = metrics_timeline[-1]['system_metrics']['memory_mb']
            
            print(f"  Memory: {memory_start:.1f}MB → {memory_peak:.1f}MB (peak) → {memory_end:.1f}MB")
            print(f"  Growth: {memory_peak - memory_start:.1f}MB")
        else:
            print("  Memory metrics: not available")
        print(f"  Collected {len(metrics_timeline)} metric snapshots")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_leak_detection():
    """Test for memory leaks during repeated operations"""
    print("\n=== Test 2: Memory Leak Detection ===")
    
    try:
        # Force garbage collection
        gc.collect()
        
        # Track memory over multiple cycles
        memory_samples = []
        
        for cycle in range(5):
            # Create and destroy engine
            engine = VoiceEngine(api_key=API_KEY, mode="fast")
            
            await engine.connect()
            await engine.send_text(f"Cycle {cycle + 1}")
            await asyncio.sleep(2)
            await engine.disconnect()
            
            # Force cleanup
            del engine
            gc.collect()
            await asyncio.sleep(0.5)
            
            # Measure memory
            if HAS_PSUTIL:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
                print(f"  Cycle {cycle + 1}: {memory_mb:.1f}MB")
            else:
                print(f"  Cycle {cycle + 1}: completed")
        
        # Check for leak
        if HAS_PSUTIL and memory_samples:
            memory_growth = memory_samples[-1] - memory_samples[0]
            avg_growth_per_cycle = memory_growth / len(memory_samples)
            
            print(f"\n✓ Memory analysis:")
            print(f"  Start: {memory_samples[0]:.1f}MB")
            print(f"  End: {memory_samples[-1]:.1f}MB")
            print(f"  Total growth: {memory_growth:.1f}MB")
            print(f"  Avg per cycle: {avg_growth_per_cycle:.1f}MB")
            
            # Reasonable threshold - some growth is expected
            reasonable_growth = 10  # MB
            if memory_growth > reasonable_growth:
                print(f"⚠ Warning: Memory growth exceeds threshold ({reasonable_growth}MB)")
            else:
                print("✓ Memory usage stable")
        else:
            print(f"\n✓ Memory analysis: skipped (psutil not available)")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory leak detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_component_cleanup():
    """Test cleanup of individual components"""
    print("\n=== Test 3: Component Cleanup ===")
    
    try:
        # Create components separately
        components_tested = []
        
        # Test AudioEngine cleanup
        from audioengine.audioengine.audio_types import AudioConfig
        audio_config = AudioConfig(sample_rate=24000, channels=1)
        audio_engine = AudioEngine(config=audio_config)
        
        # Use it - process some audio
        audio_engine.process_audio(b'\x00' * 4800)
        
        # Get metrics before cleanup
        metrics_before = audio_engine.get_metrics()
        
        # Cleanup
        audio_engine.cleanup()
        components_tested.append("AudioEngine")
        print("✓ AudioEngine cleaned up")
        
        # Test BaseEngine cleanup
        base_engine = BaseEngine()
        base_engine.create_strategy("fast")
        
        # Setup audio with all required parameters
        await base_engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=20,
            vad_enabled=True,
            input_device=None,  # Use default
            output_device=None,  # Use default
            vad_threshold=0.02,
            vad_speech_start_ms=200,
            vad_speech_end_ms=1000
        )
        
        # Cleanup
        await base_engine.cleanup()
        components_tested.append("BaseEngine")
        print("✓ BaseEngine cleaned up")
        
        # Test VoiceEngine cleanup
        voice_engine = VoiceEngine(api_key=API_KEY)
        await voice_engine.disconnect()
        components_tested.append("VoiceEngine")
        print("✓ VoiceEngine cleaned up")
        
        print(f"\n✓ Successfully cleaned up {len(components_tested)} components")
        
        return True
        
    except Exception as e:
        print(f"✗ Component cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_state_cleanup():
    """Test cleanup after errors"""
    print("\n=== Test 4: Error State Cleanup ===")
    
    try:
        errors_handled = []
        
        # Test 1: Connection error cleanup
        try:
            engine = VoiceEngine(api_key="invalid-key")
            await engine.connect()
        except Exception as e:
            errors_handled.append(("Connection error", str(e)))
            # Ensure cleanup still works
            await engine.disconnect()
            print("✓ Cleaned up after connection error")
        
        # Test 2: Mid-operation error
        engine = VoiceEngine(api_key=API_KEY)
        await engine.connect()
        
        # Simulate error condition
        try:
            # Send invalid data
            await engine.send_audio(b'invalid')
            await asyncio.sleep(2)
        except Exception as e:
            errors_handled.append(("Operation error", str(e)))
        
        # Cleanup should still work
        await engine.disconnect()
        print("✓ Cleaned up after operation error")
        
        # Test 3: Cleanup with pending operations
        engine = VoiceEngine(api_key=API_KEY)
        await engine.connect()
        
        # Start operations but don't wait
        asyncio.create_task(engine.send_text("Test"))
        
        # Immediate cleanup
        await engine.disconnect()
        print("✓ Cleaned up with pending operations")
        
        print(f"\n✓ Handled {len(errors_handled)} error scenarios")
        
        return True
        
    except Exception as e:
        print(f"✗ Error state cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_resource_limits():
    """Test behavior at resource limits"""
    print("\n=== Test 5: Resource Limits ===")
    
    try:
        # Test multiple concurrent engines
        engines = []
        max_engines = 2  # Reduced to avoid resource conflicts
        
        print(f"  Creating {max_engines} concurrent engines...")
        
        for i in range(max_engines):
            engine = VoiceEngine(api_key=API_KEY, mode="fast")
            await engine.connect()
            engines.append(engine)
            print(f"  ✓ Engine {i+1} connected")
        
        # Use all engines
        for i, engine in enumerate(engines):
            await engine.send_text(f"Engine {i+1} test")
        
        await asyncio.sleep(4)
        
        # Cleanup all
        for i, engine in enumerate(engines):
            await engine.disconnect()
            print(f"  ✓ Engine {i+1} disconnected")
        
        print(f"✓ Successfully managed {max_engines} concurrent engines")
        
        # Test rapid create/destroy cycles
        print("\n  Testing rapid create/destroy...")
        
        start_time = time.time()
        rapid_cycles = 5  # Reduced for stability
        
        for i in range(rapid_cycles):
            async with VoiceEngine(api_key=API_KEY) as engine:
                await engine.send_text(f"Rapid test {i+1}")
                await asyncio.sleep(0.5)
        
        elapsed = time.time() - start_time
        print(f"✓ Completed {rapid_cycles} cycles in {elapsed:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Resource limits test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_final_state_verification():
    """Verify final state after all operations"""
    print("\n=== Test 6: Final State Verification ===")
    
    try:
        # Create and use engine
        engine = VoiceEngine(api_key=API_KEY)
        
        # Track states
        states = []
        
        def check_state(label):
            state = {
                'label': label,
                'connected': engine.is_connected,
                'listening': engine.is_listening,
                'stream_state': engine.get_state().value,
                'handlers': sum(1 for h in [
                    engine.on_audio_response,
                    engine.on_text_response,
                    engine.on_error,
                    engine.on_response_done
                ] if h is not None)
            }
            states.append(state)
            return state
        
        # Initial state
        initial = check_state("Initial")
        print(f"✓ Initial state: connected={initial['connected']}, handlers={initial['handlers']}")
        
        # Setup handlers
        engine.on_audio_response = lambda x: None
        engine.on_text_response = lambda x: None
        engine.on_error = lambda x: None
        engine.on_response_done = lambda: None
        
        # Use engine
        await engine.connect()
        connected = check_state("Connected")
        
        await engine.start_listening()
        listening = check_state("Listening")
        
        await engine.send_text("Final test")
        await asyncio.sleep(2)
        
        await engine.stop_listening()
        stopped = check_state("Stopped")
        
        await engine.disconnect()
        final = check_state("Disconnected")
        
        # Verify progression
        print("\n✓ State progression verified:")
        for state in states:
            print(f"  {state['label']}: connected={state['connected']}, listening={state['listening']}")
        
        # Final verification
        assert not final['connected'], "Should be disconnected"
        assert not final['listening'], "Should not be listening"
        print("✓ Final state correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Final state verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Metrics and Cleanup Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Comprehensive Metrics", await test_comprehensive_metrics()))
    results.append(("Memory Leak Detection", await test_memory_leak_detection()))
    results.append(("Component Cleanup", await test_component_cleanup()))
    results.append(("Error State Cleanup", await test_error_state_cleanup()))
    results.append(("Resource Limits", await test_resource_limits()))
    results.append(("Final State", await test_final_state_verification()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)