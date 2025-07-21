#!/usr/bin/env python3
"""
Simple VoxTerm Integration Test

Minimal test based on the working test_1_text_debug.py

python -m voxterm.smoke_tests.test_4_simple_integration
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
from voxterm import VoxTermCLI


async def test_simple():
    """Simple test that mirrors the working test_1"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key")
        return False
    
    print("\n🔥 Simple VoxTerm Integration Test\n")
    
    # Create engine exactly like test_1
    config = VoiceEngineConfig(
        api_key=api_key,
        mode="fast",
        voice="alloy",
        log_level="WARNING"  # Less verbose
    )
    
    engine = VoiceEngine(config=config)
    
    # Track if we got a response
    got_response = False
    
    # Set callbacks
    def on_text(text):
        nonlocal got_response
        got_response = True
        print(f"AI: {text}", end="", flush=True)
    
    def on_done():
        print("\n[Response complete]")
    
    def on_error(error):
        print(f"\nError: {error}")
    
    engine.on_text_response = on_text
    engine.on_response_done = on_done
    engine.on_error = on_error
    
    # Test 1: Direct engine test (like test_1_text_debug)
    print("TEST 1: Direct Engine Communication")
    print("-" * 40)
    
    # Connect
    print("Connecting...")
    await engine.connect()
    print("✓ Connected")
    
    # Send text
    print("\nSending: 'Hello, can you hear me?'")
    await engine.send_text("Hello, can you hear me?")
    
    # Wait for response
    await asyncio.sleep(5)
    
    if got_response:
        print("\n✅ Direct engine test PASSED")
    else:
        print("\n❌ Direct engine test FAILED - no response")
    
    # Disconnect
    await engine.disconnect()
    
    # Test 2: VoxTermCLI creation
    print("\n\nTEST 2: VoxTermCLI Integration")
    print("-" * 40)
    
    # Recreate engine for CLI test
    engine2 = VoiceEngine(config=config)
    
    # Test different modes
    modes = ["text", "push_to_talk", "always_on", "turn_based"]
    all_good = True
    
    for mode in modes:
        try:
            cli = VoxTermCLI(engine2, mode=mode)
            print(f"✓ Created CLI with {mode} mode")
        except Exception as e:
            print(f"❌ Failed to create CLI with {mode} mode: {e}")
            all_good = False
    
    if all_good:
        print("\n✅ CLI integration test PASSED")
    else:
        print("\n❌ CLI integration test FAILED")
    
    # Summary
    print("\n" + "="*60)
    if got_response and all_good:
        print("✅ ALL TESTS PASSED")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple())
    sys.exit(0 if success else 1)