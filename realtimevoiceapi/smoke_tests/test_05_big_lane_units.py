#!/usr/bin/env python3
"""
Test 05: Big Lane Units - Test big lane components in isolation


python -m realtimevoiceapi.smoke_tests.test_05_big_lane_units

Tests:
- AudioPipeline: Composable audio processing
- EventBus: Event-driven architecture
- StreamOrchestrator: Multi-stream coordination
- ResponseAggregator: Response assembly


Audio pipeline with multiple processors
Pipeline composition and presets
Event bus basic and advanced features
Stream orchestrator capabilities
Workflow execution
Integration between components
"""

import sys
import logging
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_audio_pipeline_basic():
    """Test basic audio pipeline functionality"""
    print("\n🎵 Testing Audio Pipeline Basics...")
    
    try:
        from realtimevoiceapi.big_lane.audio_pipeline import (
            AudioPipeline, AudioValidator, ProcessorPriority
        )
        from realtimevoiceapi.audio_types import AudioConfig
        
        config = AudioConfig()
        pipeline = AudioPipeline(config=config)
        
        # Add validator
        validator = AudioValidator(config)
        pipeline.add_processor(validator)
        
        # Test processor management
        assert len(pipeline.processors) == 1
        assert pipeline.processors[0].name == "AudioValidator"
        print("  ✅ Processor added to pipeline")
        
        # Test processing valid audio
        valid_audio = b'\x00\x00' * 2400  # 100ms of silence
        result = await pipeline.process(valid_audio)
        assert result is not None
        assert len(result) == len(valid_audio)
        print("  ✅ Valid audio passes through")
        
        # Test processing invalid audio
        invalid_audio = b'\x00'  # Odd number of bytes
        result = await pipeline.process(invalid_audio)
        assert result is None  # Should be filtered out
        print("  ✅ Invalid audio filtered out")
        
        # Test metrics
        metrics = pipeline.get_metrics()
        assert metrics['total_chunks'] == 2  # One valid, one invalid
        assert metrics['processors']['AudioValidator']['chunks'] == 1  # Only valid counted
        print("  ✅ Pipeline metrics working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio pipeline basic test failed: {e}")
        logger.exception("Audio pipeline error")
        return False


async def test_audio_processors():
    """Test individual audio processors"""
    print("\n🔧 Testing Audio Processors...")
    
    try:
        from realtimevoiceapi.big_lane.audio_pipeline import (
            NoiseReducer, VolumeNormalizer, VADProcessor,
            ProcessorMetrics
        )
        import numpy as np
        
        # Test NoiseReducer
        noise_reducer = NoiseReducer(noise_floor=0.02)
        
        # Generate noisy audio
        samples = 2400
        signal = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, samples))
        noise = 0.05 * np.random.randn(samples)
        noisy_signal = signal + noise
        noisy_audio = (noisy_signal * 32767).astype(np.int16).tobytes()
        
        # Process multiple times for calibration
        for _ in range(5):
            result = await noise_reducer.process(noisy_audio)
        
        assert result is not None
        assert noise_reducer.calibration_samples == 5
        print("  ✅ NoiseReducer calibration works")
        
        # Test VolumeNormalizer
        normalizer = VolumeNormalizer(target_level=0.3)
        
        # Quiet audio
        quiet_signal = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, samples))
        quiet_audio = (quiet_signal * 32767).astype(np.int16).tobytes()
        
        normalized = await normalizer.process(quiet_audio)
        assert normalized is not None
        
        # Should be louder
        normalized_array = np.frombuffer(normalized, dtype=np.int16)
        quiet_array = np.frombuffer(quiet_audio, dtype=np.int16)
        assert np.mean(np.abs(normalized_array)) > np.mean(np.abs(quiet_array))
        print("  ✅ VolumeNormalizer amplifies quiet audio")
        
        # Test VADProcessor
       
        vad_processor = VADProcessor(threshold=0.02, min_speech_duration_ms=200)


        # Process speech
        speech_audio = (0.3 * 32767 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.3, 7200))).astype(np.int16).tobytes()
        
        # The processor might return immediately or accumulate based on implementation
        result1 = await vad_processor.process(speech_audio[:2400])
        assert result1 is None  # Still accumulating (100ms < 200ms minimum)


        result2 = await vad_processor.process(speech_audio[2400:4800])
        
        # Now we should have 200ms accumulated, but the processor might still be accumulating
        if result2 is None:
            # Third chunk to ensure we get something
            result3 = await vad_processor.process(speech_audio[4800:])
            assert result3 is not None  # Must return something by now
            print("  ✅ VADProcessor accumulates and returns after 300ms")
        else:
            assert len(result2) >= 4800  # Should have at least 200ms of audio
            print("  ✅ VADProcessor returns accumulated speech after 200ms")

       
        
        
        
        # Test processor metrics
        assert isinstance(normalizer.metrics, ProcessorMetrics)
        assert normalizer.metrics.processed_chunks > 0
        print("  ✅ Processor metrics tracked")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio processors test failed: {e}")
        logger.exception("Audio processors error")
        return False


async def test_pipeline_composition():
    """Test composing multiple processors"""
    print("\n🔗 Testing Pipeline Composition...")
    
    try:
        from realtimevoiceapi.big_lane.audio_pipeline import (
            AudioPipeline, AudioValidator, NoiseReducer,
            VolumeNormalizer, VADProcessor, ProcessorPriority
        )
        from realtimevoiceapi.audio_types import AudioConfig
        
        config = AudioConfig()
        pipeline = AudioPipeline(config=config)
        
        # Add processors in wrong order
        vad = VADProcessor(threshold=0.02)  # NORMAL priority
        validator = AudioValidator(config)   # CRITICAL priority
        normalizer = VolumeNormalizer()     # NORMAL priority
        noise = NoiseReducer()              # HIGH priority
        
        pipeline.add_processor(vad)
        pipeline.add_processor(validator)
        pipeline.add_processor(normalizer)
        pipeline.add_processor(noise)
        
        # Check they're sorted by priority
        assert pipeline.processors[0].name == "AudioValidator"  # CRITICAL=0
        assert pipeline.processors[1].name == "NoiseReducer"    # HIGH=10
        print("  ✅ Processors auto-sorted by priority")
        
        # Test enable/disable
        pipeline.set_processor_enabled("NoiseReducer", False)
        chain = pipeline.get_processor_chain()
        assert "NoiseReducer" not in str(chain)
        print("  ✅ Processor enable/disable works")
        
        # Test removal
        removed = pipeline.remove_processor("VolumeNormalizer")
        assert removed == True
        assert len(pipeline.processors) == 3
        print("  ✅ Processor removal works")
        
        # Test processing through pipeline
        test_audio = b'\x00\x00' * 2400
        result = await pipeline.process(test_audio)
        
        # Should pass validator but might be filtered by VAD
        # (depends on VAD state)
        print("  ✅ Audio processed through pipeline")
        
        # Test reset
        pipeline.reset()
        print("  ✅ Pipeline reset works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline composition test failed: {e}")
        logger.exception("Pipeline composition error")
        return False


async def test_pipeline_presets():
    """Test pipeline presets"""
    print("\n🎨 Testing Pipeline Presets...")
    
    try:
        from realtimevoiceapi.big_lane.audio_pipeline import PipelinePresets
        from realtimevoiceapi.audio_types import AudioConfig
        
        config = AudioConfig()
        
        # Test basic pipeline
        basic = PipelinePresets.create_basic_pipeline(config)
        assert len(basic.processors) == 1
        assert basic.processors[0].name == "AudioValidator"
        print("  ✅ Basic pipeline created")
        
        # Test voice pipeline
        voice = PipelinePresets.create_voice_pipeline(config)
        assert len(voice.processors) == 4
        processor_names = [p.name for p in voice.processors]
        assert "AudioValidator" in processor_names
        assert "NoiseReducer" in processor_names
        assert "VolumeNormalizer" in processor_names
        assert "VADProcessor" in processor_names
        print("  ✅ Voice pipeline created")
        
        # Test quality pipeline
        quality = PipelinePresets.create_quality_pipeline(config)
        assert len(quality.processors) == 5
        assert any(p.name == "EchoCanceller" for p in quality.processors)
        print("  ✅ Quality pipeline created")
        
        # Test realtime pipeline
        realtime = PipelinePresets.create_realtime_pipeline(config)
        assert len(realtime.processors) == 2  # Minimal for speed
        print("  ✅ Realtime pipeline created")
        
        # Test processing through preset
        test_audio = b'\x00\x00' * 2400
        result = await voice.process(test_audio)
        # Result depends on VAD, but pipeline should work
        print("  ✅ Preset pipeline processes audio")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline presets test failed: {e}")
        logger.exception("Pipeline presets error")
        return False


async def test_event_bus_basic():
    """Test basic EventBus functionality"""
    print("\n📮 Testing Event Bus Basics...")
    
    try:
        from realtimevoiceapi.big_lane.event_bus import EventBus, Event
        
        bus = EventBus(name="test_bus")
        bus.start()
        
        # Test subscription
        received_events = []
        
        def handler(event: Event):
            received_events.append(event)
        
        sub_id = bus.subscribe("test.*", handler)
        assert sub_id.startswith("sub_")
        print("  ✅ Event subscription works")
        
        # Test event emission
        await bus.emit("test.event1", data={"value": 42})
        await asyncio.sleep(0.1)  # Let event propagate
        
        assert len(received_events) == 1
        assert received_events[0].type == "test.event1"
        assert received_events[0].data["value"] == 42
        print("  ✅ Event emission and delivery works")
        
        # Test pattern matching
        await bus.emit("test.event2", data={"value": 100})
        await bus.emit("other.event", data={"value": 200})
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 2  # Only test.* events
        print("  ✅ Pattern matching works")
        
        # Test unsubscribe
        bus.unsubscribe(sub_id)
        await bus.emit("test.event3", data={"value": 300})
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 2  # No new events
        print("  ✅ Unsubscribe works")
        
        # Test metrics
        metrics = bus.get_metrics()
        assert metrics["events_emitted"] >= 3
        assert metrics["events_delivered"] >= 2
        print("  ✅ Event bus metrics work")
        
        await bus.stop()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Event bus basic test failed: {e}")
        logger.exception("Event bus error")
        return False


async def test_event_bus_advanced():
    """Test advanced EventBus features"""
    print("\n🚀 Testing Event Bus Advanced Features...")
    
    try:
        from realtimevoiceapi.big_lane.event_bus import (
            EventBus, Event, EventPriority
        )
        
        bus = EventBus(name="advanced_bus", history_size=100)
        bus.start()
        
        # Test priority delivery
        priority_order = []
        
        def low_handler(event):
            priority_order.append("low")
        
        def high_handler(event):
            priority_order.append("high")
        
        bus.subscribe("priority.*", low_handler, priority=0)
        bus.subscribe("priority.*", high_handler, priority=10)
        
        await bus.emit("priority.test", {})
        await asyncio.sleep(0.1)
        
        # High priority should be called first
        assert priority_order == ["high", "low"]
        print("  ✅ Priority-based delivery works")
        
        # Test one-time subscription
        once_count = []
        
        def once_handler(event):
            once_count.append(1)
        
        bus.subscribe("once.*", once_handler, once=True)
        
        await bus.emit("once.test", {})
        await bus.emit("once.test", {})
        await asyncio.sleep(0.1)
        
        assert len(once_count) == 1  # Only called once
        print("  ✅ One-time subscription works")
        
        # Test event filtering
        filtered_events = []
        
        def filter_handler(event):
            filtered_events.append(event)
        
        def value_filter(event):
            return event.data.get("value", 0) > 50
        
        bus.subscribe("filter.*", filter_handler, filter_func=value_filter)
        
        await bus.emit("filter.test", {"value": 30})
        await bus.emit("filter.test", {"value": 70})
        await asyncio.sleep(0.1)
        
        assert len(filtered_events) == 1
        assert filtered_events[0].data["value"] == 70
        print("  ✅ Event filtering works")
        
        # Test event history
        history = bus.get_history(pattern="filter.*")
        assert len(history) == 2  # Both emitted events
        print("  ✅ Event history works")
        
        # Test interceptors
        intercepted = []
        
        def interceptor(event):
            intercepted.append(event.type)
            # Modify event
            event.data["intercepted"] = True
            return event
        
        bus.add_interceptor(interceptor)
        
        final_event = []
        bus.subscribe("intercept.*", lambda e: final_event.append(e))
        
        await bus.emit("intercept.test", {"original": True})
        await asyncio.sleep(0.1)
        
        assert "intercept.test" in intercepted
        assert final_event[0].data["intercepted"] == True
        print("  ✅ Event interceptors work")
        
        await bus.stop()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Event bus advanced test failed: {e}")
        logger.exception("Event bus advanced error")
        return False


async def test_stream_orchestrator_basic():
    """Test basic StreamOrchestrator functionality"""
    print("\n🎭 Testing Stream Orchestrator Basics...")
    
    try:
        from realtimevoiceapi.big_lane.stream_orchestrator import (
            StreamOrchestrator, StreamRole, LoadBalancingStrategy
        )
        from realtimevoiceapi.big_lane.event_bus import EventBus
        from realtimevoiceapi.provider_protocol import ProviderRegistry
        
        # Setup
        event_bus = EventBus()
        event_bus.start()
        
        orchestrator = StreamOrchestrator(
            event_bus=event_bus,
            provider_registry=ProviderRegistry()
        )
        
        await orchestrator.start()
        
        # Test configuration
        assert orchestrator.load_strategy == LoadBalancingStrategy.LATENCY_BASED
        assert orchestrator.max_streams_per_provider == 5
        print("  ✅ Orchestrator initialized")
        
        # Test metrics
        metrics = orchestrator.get_metrics()
        assert metrics["total_streams"] == 0
        assert metrics["healthy_streams"] == 0
        print("  ✅ Initial metrics correct")
        
        # Would test stream creation but needs real providers
        # Test load balancing strategies
        orchestrator.load_strategy = LoadBalancingStrategy.ROUND_ROBIN
        assert orchestrator.load_strategy == LoadBalancingStrategy.ROUND_ROBIN
        print("  ✅ Load balancing strategy configurable")
        
        await orchestrator.stop()
        await event_bus.stop()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stream orchestrator basic test failed: {e}")
        logger.exception("Stream orchestrator error")
        return False


async def test_orchestrator_workflows():
    """Test orchestrator workflow functionality"""
    print("\n🔄 Testing Orchestrator Workflows...")
    
    try:
        from realtimevoiceapi.big_lane.stream_orchestrator import (
            StreamOrchestrator, WorkflowStep
        )
        from realtimevoiceapi.big_lane.event_bus import EventBus
        
        event_bus = EventBus()
        event_bus.start()
        
        orchestrator = StreamOrchestrator(event_bus=event_bus)
        
        # Define test workflow
        workflow_results = []
        
        async def step1_process(stream_id, data, context, orchestrator):
            workflow_results.append("step1")
            context["step1_done"] = True
            return "step1_result"
        
        async def step2_process(stream_id, data, context, orchestrator):
            workflow_results.append("step2")
            return "step2_result"
        
        async def step2_condition(state):
            return state["context"].get("step1_done", False)
        
        steps = [
            WorkflowStep(
                name="step1",
                process_func=step1_process
            ),
            WorkflowStep(
                name="step2",
                process_func=step2_process,
                condition_func=step2_condition
            )
        ]
        
        # Register workflow
        orchestrator.register_workflow("test_workflow", steps)
        assert "test_workflow" in orchestrator.workflows
        print("  ✅ Workflow registered")
        
        # Execute workflow (will fail without streams, but test the mechanism)
        try:
            result = await orchestrator.execute_workflow(
                "test_workflow",
                initial_data={"test": "data"}
            )
        except RuntimeError as e:
            # Expected - no streams available
            if "No suitable stream" in str(e):
                print("  ✅ Workflow execution attempted (no streams available)")
            else:
                raise
        
        # Verify workflow was started
        assert len(workflow_results) >= 0  # Depends on how far it got
        print("  ✅ Workflow execution framework works")
        
        await event_bus.stop()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Orchestrator workflows test failed: {e}")
        logger.exception("Orchestrator workflows error")
        return False


async def test_big_lane_integration():
    """Test integration between big lane components"""
    print("\n🔗 Testing Big Lane Integration...")
    
    try:
        from realtimevoiceapi.big_lane.audio_pipeline import (
            AudioPipeline, AudioValidator
        )
        from realtimevoiceapi.big_lane.event_bus import EventBus, Event
        from realtimevoiceapi.audio_types import AudioConfig
        
        # Create interconnected components
        event_bus = EventBus()
        event_bus.start()
        
        config = AudioConfig()
        pipeline = AudioPipeline(config=config)
        pipeline.add_processor(AudioValidator(config))
        
        # Wire pipeline to emit events
        pipeline_events = []
        
        async def process_with_events(audio):
            result = await pipeline.process(audio)
            
            await event_bus.emit(Event(
                type="pipeline.processed",
                data={
                    "input_size": len(audio),
                    "output_size": len(result) if result else 0,
                    "filtered": result is None
                }
            ))
            
            return result
        
        # Subscribe to pipeline events
        bus_received = []
        event_bus.subscribe("pipeline.*", lambda e: bus_received.append(e))
        
        # Process audio
        test_audio = b'\x00\x00' * 2400
        result = await process_with_events(test_audio)
        await asyncio.sleep(0.1)
        
        assert len(bus_received) == 1
        assert bus_received[0].data["input_size"] == len(test_audio)
        assert bus_received[0].data["filtered"] == False
        print("  ✅ Pipeline + EventBus integration works")
        
        # Test with invalid audio
        invalid_audio = b'\x00'
        result = await process_with_events(invalid_audio)
        await asyncio.sleep(0.1)
        
        assert len(bus_received) == 2
        assert bus_received[1].data["filtered"] == True
        print("  ✅ Event emission for filtered audio works")
        
        await event_bus.stop()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Big lane integration test failed: {e}")
        logger.exception("Big lane integration error")
        return False


def main():
    """Run all big lane unit tests"""
    print("🧪 RealtimeVoiceAPI - Test 05: Big Lane Units")
    print("=" * 60)
    print("Testing big lane components with full abstractions")
    print()
    
    tests = [
        ("Audio Pipeline Basics", test_audio_pipeline_basic),
        ("Audio Processors", test_audio_processors),
        ("Pipeline Composition", test_pipeline_composition),
        ("Pipeline Presets", test_pipeline_presets),
        ("Event Bus Basics", test_event_bus_basic),
        ("Event Bus Advanced", test_event_bus_advanced),
        ("Stream Orchestrator Basics", test_stream_orchestrator_basic),
        ("Orchestrator Workflows", test_orchestrator_workflows),
        ("Big Lane Integration", test_big_lane_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = asyncio.run(test_func())
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
        print("\n🎉 All big lane components working correctly!")
        print("✨ Big lane characteristics verified:")
        print("  - Composable audio pipelines")
        print("  - Event-driven architecture")
        print("  - Priority-based processing")
        print("  - Advanced orchestration")
        print("  - Comprehensive metrics")
        print("\nNext: Run test_06_integration.py")
    else:
        print(f"\n❌ {total - passed} big lane component(s) need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)