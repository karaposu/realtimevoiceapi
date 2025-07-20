#!/usr/bin/env python3
"""
Test 07 Voice Engine Extreme - Detailed debugging of VoiceEngine issues

Focuses on finding the exact cause of segmentation fault in VoiceEngine
by testing specific scenarios and method calls with extensive logging.

Requirements:
- Valid OpenAI API key in .env file
- Working microphone

python -m realtimevoiceapi.smoke_tests.test_07_voice_engine_extreme
"""

import sys
import asyncio
import os
import time
import logging
import gc
import traceback
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import voice engine components
from realtimevoiceapi.voice_engine import VoiceEngine, VoiceEngineConfig
from realtimevoiceapi.core.stream_protocol import StreamState
from realtimevoiceapi.core.exceptions import EngineError


class DebugVoiceEngine(VoiceEngine):
    """Voice Engine with extra debugging"""
    
    def __init__(self, *args, **kwargs):
        logger.info("=== DebugVoiceEngine.__init__ START ===")
        super().__init__(*args, **kwargs)
        logger.info("=== DebugVoiceEngine.__init__ END ===")
    
    def _setup_event_handlers(self):
        """Override to add debugging"""
        logger.info("=== _setup_event_handlers START ===")
        try:
            super()._setup_event_handlers()
            logger.info("=== _setup_event_handlers END (success) ===")
        except Exception as e:
            logger.error(f"=== _setup_event_handlers FAILED: {e} ===")
            raise
    
    async def text_2_audio_response(self, text: str, timeout: float = 30.0):
        """Override text_2_audio_response with debugging"""
        logger.info(f"=== text_2_audio_response() START: '{text}' ===")
        try:
            result = await super().text_2_audio_response(text, timeout)
            logger.info(f"=== text_2_audio_response() END: {len(result)} bytes ===")
            return result
        except Exception as e:
            logger.error(f"=== text_2_audio_response() FAILED: {e} ===")
            raise


async def test_minimal_voice_engine():
    """Test minimal VoiceEngine creation"""
    print("\n🔧 Testing Minimal VoiceEngine...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found")
        return False
    
    try:
        # Create engine
        engine = VoiceEngine(api_key=api_key, mode="fast")
        print("  ✅ VoiceEngine created")
        
        # Check initial state
        assert not engine.is_connected
        print("  ✅ Initial state correct")
        
        # Delete engine
        del engine
        gc.collect()
        print("  ✅ Engine deleted cleanly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Minimal test failed: {e}")
        logger.exception("Minimal test error")
        return False


async def test_connect_only():
    """Test just connection"""
    print("\n🔌 Testing Connect Only...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        print("  ✅ Engine created")
        
        # Connect
        await engine.connect()
        print("  ✅ Connected")
        
        # Check state
        assert engine.is_connected
        print("  ✅ State verified")
        
        # Disconnect
        await engine.disconnect()
        print("  ✅ Disconnected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Connect test failed: {e}")
        logger.exception("Connect error")
        return False


async def test_event_handler_setup():
    """Test event handler setup specifically"""
    print("\n🎯 Testing Event Handler Setup...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = DebugVoiceEngine(api_key=api_key, mode="fast")
        print("  ✅ Debug engine created")
        
        # Set callbacks
        call_counts = {"audio": 0, "text": 0, "done": 0}
        
        def on_audio(audio):
            call_counts["audio"] += 1
            logger.info(f"Audio callback called: {len(audio)} bytes")
        
        def on_text(text):
            call_counts["text"] += 1
            logger.info(f"Text callback called: {text}")
        
        def on_done():
            call_counts["done"] += 1
            logger.info("Done callback called")
        
        engine.on_audio_response = on_audio
        engine.on_text_response = on_text
        engine.on_response_done = on_done
        
        print("  ✅ Callbacks set")
        
        # Connect (this triggers _setup_event_handlers)
        await engine.connect()
        print("  ✅ Connected with handlers")
        
        # Send a simple message
        await engine.send_text("Say 'test'")
        print("  ✅ Text sent")
        
        # Wait for response
        await asyncio.sleep(3)
        
        print(f"  📊 Callback counts: {call_counts}")
        
        # Disconnect
        await engine.disconnect()
        print("  ✅ Disconnected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Event handler test failed: {e}")
        logger.exception("Event handler error")
        return False


async def test_speak_isolation():
    """Test text_2_audio_response() method in isolation"""
    print("\n🔊 Testing text_2_audio_response() in Isolation...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    engine = None
    
    try:
        # Create engine
        engine = DebugVoiceEngine(api_key=api_key, mode="fast")
        print("  ✅ Engine created")
        
        # Connect
        await engine.connect()
        print("  ✅ Connected")
        
        # Add delay to ensure connection is stable
        await asyncio.sleep(1)
        print("  ✅ Connection stabilized")
        
        # Try text_2_audio_response with very short timeout first
        try:
            print("  🔊 Testing text_2_audio_response with short timeout...")
            audio = await engine.text_2_audio_response("Hi", timeout=5.0)
            print(f"  ✅ text_2_audio_response successful: {len(audio)} bytes")
        except Exception as e:
            print(f"  ⚠️ text_2_audio_response with short timeout failed: {e}")
        
        # Disconnect and cleanup
        await engine.disconnect()
        print("  ✅ Disconnected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Speak isolation failed: {e}")
        logger.exception("Speak isolation error")
        traceback.print_exc()
        
        # Try to cleanup
        if engine:
            try:
                await engine.disconnect()
            except:
                pass
        
        return False


async def test_callback_manipulation():
    """Test callback manipulation during text_2_audio_response()"""
    print("\n🔄 Testing Callback Manipulation...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        
        # Track original callbacks
        original_audio = lambda x: print(f"Original audio: {len(x)} bytes")
        original_done = lambda: print("Original done")
        
        engine.on_audio_response = original_audio
        engine.on_response_done = original_done
        
        print("  ✅ Original callbacks set")
        
        # Connect
        await engine.connect()
        print("  ✅ Connected")
        
        # Test that callbacks are preserved
        assert engine.on_audio_response == original_audio
        assert engine.on_response_done == original_done
        print("  ✅ Callbacks preserved after connect")
        
        # Now test text_2_audio_response() - it should temporarily replace callbacks
        print("  🔊 Testing text_2_audio_response()...")
        
        # Instead of calling text_2_audio_response, let's manually do what text_2_audio_response does
        audio_chunks = []
        
        def collect_audio(audio):
            audio_chunks.append(audio)
            print(f"  📦 Collected chunk: {len(audio)} bytes")
        
        # Save old callbacks
        old_audio = engine.on_audio_response
        old_done = engine.on_response_done
        
        # Set temporary callbacks
        engine.on_audio_response = collect_audio
        engine.on_response_done = lambda: print("  ✅ Response done")
        
        # Setup handlers
        engine._setup_event_handlers()
        
        # Send text
        await engine.send_text("Say 'hello'")
        
        # Wait for response
        await asyncio.sleep(3)
        
        # Restore callbacks
        engine.on_audio_response = old_audio
        engine.on_response_done = old_done
        engine._setup_event_handlers()
        
        print(f"  ✅ Collected {len(audio_chunks)} audio chunks")
        
        # Disconnect
        await engine.disconnect()
        print("  ✅ Test complete")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Callback manipulation failed: {e}")
        logger.exception("Callback error")
        return False


async def test_memory_and_references():
    """Test for memory issues and circular references"""
    print("\n🧠 Testing Memory and References...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        import weakref
        
        # Create engine
        engine = VoiceEngine(api_key=api_key, mode="fast")
        engine_ref = weakref.ref(engine)
        
        print("  ✅ Engine created with weak reference")
        
        # Connect
        await engine.connect()
        print("  ✅ Connected")
        
        # Check references
        print(f"  📊 Engine refcount: {sys.getrefcount(engine)}")
        print(f"  📊 Base engine refcount: {sys.getrefcount(engine._base)}")
        
        # Disconnect
        await engine.disconnect()
        print("  ✅ Disconnected")
        
        # Delete engine
        del engine
        gc.collect()
        
        # Check if cleaned up
        if engine_ref() is None:
            print("  ✅ Engine properly garbage collected")
        else:
            print("  ⚠️ Engine still referenced")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        logger.exception("Memory error")
        return False


async def test_rapid_operations():
    """Test rapid connect/disconnect/text_2_audio_response operations"""
    print("\n⚡ Testing Rapid Operations...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        for i in range(3):
            print(f"\n  🔄 Iteration {i+1}/3")
            
            # Create engine
            engine = VoiceEngine(api_key=api_key, mode="fast")
            
            # Connect
            await engine.connect()
            print("    ✅ Connected")
            
            # Send text
            await engine.send_text(f"Say 'test {i+1}'")
            print("    ✅ Text sent")
            
            # Wait briefly
            await asyncio.sleep(1)
            
            # Disconnect
            await engine.disconnect()
            print("    ✅ Disconnected")
            
            # Small delay between iterations
            await asyncio.sleep(0.5)
        
        print("  ✅ All iterations completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Rapid operations failed: {e}")
        logger.exception("Rapid operations error")
        return False


async def test_speak_step_by_step():
    """Test text_2_audio_response() method step by step"""
    print("\n🔍 Testing text_2_audio_response() Step by Step...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    
    try:
        engine = VoiceEngine(api_key=api_key, mode="fast")
        await engine.connect()
        print("  ✅ Connected")
        
        # Step 1: Create future
        audio_future = asyncio.Future()
        audio_chunks = []
        print("  ✅ Step 1: Future created")
        
        # Step 2: Define collectors
        def collect_audio(audio):
            audio_chunks.append(audio)
            print(f"    📦 Audio chunk: {len(audio)} bytes")
        
        def on_done():
            if not audio_future.done():
                if audio_chunks:
                    result = b"".join(audio_chunks)
                    audio_future.set_result(result)
                    print(f"    ✅ Future resolved: {len(result)} bytes")
                else:
                    audio_future.set_exception(EngineError("No audio"))
                    print("    ❌ Future rejected: No audio")
        
        print("  ✅ Step 2: Collectors defined")
        
        # Step 3: Save old handlers
        old_audio = engine.on_audio_response
        old_done = engine.on_response_done
        print("  ✅ Step 3: Old handlers saved")
        
        # Step 4: Set new handlers
        engine.on_audio_response = collect_audio
        engine.on_response_done = on_done
        print("  ✅ Step 4: New handlers set")
        
        # Step 5: Setup event handlers
        print("  🔄 Step 5: Setting up event handlers...")
        engine._setup_event_handlers()
        print("  ✅ Step 5: Event handlers setup complete")
        
        # Step 6: Send text
        print("  📤 Step 6: Sending text...")
        await engine.send_text("Say 'test'")
        print("  ✅ Step 6: Text sent")
        
        # Step 7: Wait for response
        print("  ⏳ Step 7: Waiting for response...")
        try:
            result = await asyncio.wait_for(audio_future, timeout=5.0)
            print(f"  ✅ Step 7: Got response: {len(result)} bytes")
        except asyncio.TimeoutError:
            print("  ⏱️ Step 7: Timeout waiting for response")
        
        # Step 8: Restore handlers
        engine.on_audio_response = old_audio
        engine.on_response_done = old_done
        engine._setup_event_handlers()
        print("  ✅ Step 8: Handlers restored")
        
        # Disconnect
        await engine.disconnect()
        print("  ✅ Disconnected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Step-by-step test failed: {e}")
        logger.exception("Step-by-step error")
        traceback.print_exc()
        return False


def main():
    """Run all extreme voice engine tests"""
    print("🧪 RealtimeVoiceAPI - Voice Engine Extreme Debugging")
    print("=" * 60)
    print("Finding the exact cause of segmentation fault")
    print()
    
    tests = [
        ("Minimal VoiceEngine", test_minimal_voice_engine),
        ("Connect Only", test_connect_only),
        ("Event Handler Setup", test_event_handler_setup),
        ("Memory and References", test_memory_and_references),
        ("Callback Manipulation", test_callback_manipulation),
        ("Speak Step by Step", test_speak_step_by_step),
        ("Speak Isolation", test_speak_isolation),
        ("Rapid Operations", test_rapid_operations),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = asyncio.run(test_func())
            results.append((test_name, result))
            
            # Force garbage collection between tests
            gc.collect()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            logger.exception(f"{test_name} crash")
            traceback.print_exc()
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
    
    # Analyze results
    if passed < total:
        failed_tests = [name for name, result in results if not result]
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        
        # Specific analysis
        if "Speak Isolation" in failed_tests:
            print("\n🔍 The issue is specifically in the text_2_audio_response() method")
        elif "Event Handler Setup" in failed_tests:
            print("\n🔍 The issue is in event handler setup")
        elif "Callback Manipulation" in failed_tests:
            print("\n🔍 The issue is in callback handling")
    else:
        print("\n🎉 All tests passed! The issue might be more complex.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)