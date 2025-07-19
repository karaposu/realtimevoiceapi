#!/usr/bin/env python3
"""
Test 07 Base Engine - Isolated test for BaseEngine

Tests the base engine in isolation to identify issues without
the complexity of the full VoiceEngine wrapper.

Requirements:
- Valid OpenAI API key in .env file
- Working microphone

python -m realtimevoiceapi.smoke_tests.test_07_base_engine
"""

import sys
import asyncio
import os
import time
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import base engine and related components
from realtimevoiceapi.base_engine import BaseEngine
from realtimevoiceapi.strategies.base_strategy import EngineConfig
from realtimevoiceapi.core.stream_protocol import StreamEvent, StreamEventType
from realtimevoiceapi.core.exceptions import EngineError


async def test_base_engine_creation():
    """Test creating BaseEngine"""
    print("\n🔧 Testing BaseEngine Creation...")
    
    try:
        engine = BaseEngine(logger=logger)
        assert engine is not None
        assert not engine.is_connected
        assert not engine.is_listening
        print("  ✅ BaseEngine created successfully")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create BaseEngine: {e}")
        logger.exception("Creation error")
        return False


async def test_strategy_creation():
    """Test strategy creation"""
    print("\n🔧 Testing Strategy Creation...")
    
    try:
        engine = BaseEngine(logger=logger)
        
        # Test fast lane strategy
        strategy = engine.create_strategy("fast")
        assert strategy is not None
        assert engine.strategy == strategy
        print("  ✅ Fast lane strategy created")
        
        # Test invalid mode
        try:
            engine.create_strategy("invalid")
            assert False, "Should have raised error"
        except ValueError:
            print("  ✅ Invalid mode rejected correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy creation failed: {e}")
        logger.exception("Strategy error")
        return False


async def test_audio_setup():
    """Test audio setup without connection"""
    print("\n🎤 Testing Audio Setup...")
    
    try:
        engine = BaseEngine(logger=logger)
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=100,
            input_device=None,
            output_device=None,
            vad_enabled=True,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        # Check audio manager exists
        assert engine._audio_manager is not None
        print("  ✅ Audio manager created")
        
        # Get metrics
        metrics = engine.get_metrics()
        assert "audio" in metrics
        print("  ✅ Audio metrics available")
        
        # Cleanup
        await engine.cleanup()
        print("  ✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio setup failed: {e}")
        logger.exception("Audio setup error")
        return False


async def test_connection_without_audio():
    """Test connection without audio to isolate issues"""
    print("\n🔌 Testing Connection (No Audio)...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found")
        return False
    
    try:
        engine = BaseEngine(logger=logger)
        
        # Create strategy
        engine.create_strategy("fast")
        
        # Initialize strategy
        config = EngineConfig(
            api_key=api_key,
            provider="openai",
            enable_vad=False,  # Disable VAD for this test
            latency_mode="ultra_low"
        )
        
        await engine.initialize_strategy(config)
        print("  ✅ Strategy initialized")
        
        # Connect WITHOUT audio setup
        await engine.do_connect()
        assert engine.is_connected
        print("  ✅ Connected successfully")
        
        # Send a test message
        await engine.send_text("Say hello")
        print("  ✅ Text sent successfully")
        
        # Wait briefly
        await asyncio.sleep(2)
        
        # Disconnect
        await engine.cleanup()
        print("  ✅ Disconnected cleanly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Connection test failed: {e}")
        logger.exception("Connection error")
        return False


async def test_minimal_audio_playback():
    """Test just audio playback to isolate issues"""
    print("\n🔊 Testing Minimal Audio Playback...")
    
    try:
        engine = BaseEngine(logger=logger)
        
        # Setup audio
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=100,
            input_device=None,
            output_device=None,
            vad_enabled=False,  # Disable VAD
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        print("  ✅ Audio setup complete")
        
        # Generate test tone
        import numpy as np
        duration = 0.5
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = (0.1 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        audio_data = tone.tobytes()
        
        # Play audio
        print("  🔊 Playing test tone...")
        engine.play_audio(audio_data)
        
        # Wait for playback
        await asyncio.sleep(1)
        
        # Cleanup
        await engine.cleanup()
        print("  ✅ Audio playback test complete")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio playback failed: {e}")
        logger.exception("Playback error")
        return False


async def test_audio_capture_only():
    """Test just audio capture"""
    print("\n🎤 Testing Audio Capture Only...")
    
    try:
        engine = BaseEngine(logger=logger)
        
        # Setup audio with VAD disabled
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=100,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        
        print("  ✅ Audio setup complete")
        
        # Start capture through audio manager
        if engine._audio_manager:
            queue = await engine._audio_manager.start_capture()
            print("  ✅ Capture started")
            
            # Capture for 1 second
            chunks = 0
            start_time = time.time()
            
            while time.time() - start_time < 1.0:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                    chunks += 1
                except asyncio.TimeoutError:
                    continue
            
            # Stop capture
            await engine._audio_manager.stop_capture()
            print(f"  ✅ Captured {chunks} chunks")
        
        # Cleanup
        await engine.cleanup()
        print("  ✅ Capture test complete")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio capture failed: {e}")
        logger.exception("Capture error")
        return False


async def test_incremental_functionality():
    """Test functionality incrementally"""
    print("\n🔄 Testing Incremental Functionality...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found")
        return False
    
    engine = None
    
    try:
        # Step 1: Create engine
        print("  1️⃣ Creating engine...")
        engine = BaseEngine(logger=logger)
        print("     ✅ Engine created")
        
        # Step 2: Create strategy
        print("  2️⃣ Creating strategy...")
        engine.create_strategy("fast")
        print("     ✅ Strategy created")
        
        # Step 3: Initialize strategy
        print("  3️⃣ Initializing strategy...")
        config = EngineConfig(
            api_key=api_key,
            provider="openai",
            enable_vad=False,
            latency_mode="ultra_low"
        )
        await engine.initialize_strategy(config)
        print("     ✅ Strategy initialized")
        
        # Step 4: Setup audio
        print("  4️⃣ Setting up audio...")
        await engine.setup_fast_lane_audio(
            sample_rate=24000,
            chunk_duration_ms=100,
            input_device=None,
            output_device=None,
            vad_enabled=False,
            vad_threshold=0.02,
            vad_speech_start_ms=100,
            vad_speech_end_ms=500
        )
        print("     ✅ Audio setup complete")
        
        # Step 5: Connect
        print("  5️⃣ Connecting...")
        await engine.do_connect()
        print("     ✅ Connected")
        
        # Step 6: Test text
        print("  6️⃣ Sending text...")
        await engine.send_text("Say the word 'test' and nothing else")
        print("     ✅ Text sent")
        
        # Step 7: Wait for response
        print("  7️⃣ Waiting for response...")
        await asyncio.sleep(3)
        print("     ✅ Response time elapsed")
        
        # Step 8: Cleanup
        print("  8️⃣ Cleaning up...")
        await engine.cleanup()
        print("     ✅ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Incremental test failed at step: {e}")
        logger.exception("Incremental test error")
        
        # Try to cleanup
        if engine:
            try:
                await engine.cleanup()
            except:
                pass
        
        return False


async def test_audio_manager_isolation():
    """Test AudioManager in complete isolation"""
    print("\n🎵 Testing AudioManager in Isolation...")
    
    try:
        from realtimevoiceapi.audio.audio_manager import AudioManager, AudioManagerConfig
        
        # Create config
        config = AudioManagerConfig(
            input_device=None,
            output_device=None,
            sample_rate=24000,
            chunk_duration_ms=100,
            vad_enabled=False
        )
        
        # Create manager
        manager = AudioManager(config, logger=logger)
        print("  ✅ AudioManager created")
        
        # Initialize
        await manager.initialize()
        print("  ✅ AudioManager initialized")
        
        # Get metrics
        metrics = manager.get_metrics()
        print(f"  ✅ Metrics: {metrics}")
        
        # Cleanup
        await manager.cleanup()
        print("  ✅ AudioManager cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AudioManager test failed: {e}")
        logger.exception("AudioManager error")
        return False


def main():
    """Run all base engine tests"""
    print("🧪 RealtimeVoiceAPI - Base Engine Isolated Tests")
    print("=" * 60)
    print("Testing BaseEngine components in isolation")
    print()
    
    tests = [
        ("BaseEngine Creation", test_base_engine_creation),
        ("Strategy Creation", test_strategy_creation),
        ("AudioManager Isolation", test_audio_manager_isolation),
        ("Audio Setup", test_audio_setup),
        ("Connection Without Audio", test_connection_without_audio),
        ("Minimal Audio Playback", test_minimal_audio_playback),
        ("Audio Capture Only", test_audio_capture_only),
        ("Incremental Functionality", test_incremental_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = asyncio.run(test_func())
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            logger.exception(f"{test_name} crash")
            results.append((test_name, False))
        
        # Small delay between tests
        time.sleep(0.5)
    
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
        print("\n🎉 All base engine tests passed!")
        print("The issue might be in the VoiceEngine wrapper")
    else:
        failed_tests = [name for name, result in results if not result]
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        print("Check the logs above to identify where the issue occurs")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)