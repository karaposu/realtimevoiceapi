#!/usr/bin/env python3
"""
VoxTerm Automated Integration Test

Non-interactive test with VoiceEngine that validates basic functionality.
Requires OPENAI_API_KEY to be set.

python -m voxterm.smoke_tests.test_3_automated_integration
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Import VoxTerm and VoiceEngine
from voxterm import VoxTermCLI
from realtimevoiceapi import VoiceEngine, VoiceEngineConfig


class TestCallbacks:
    """Track callbacks for testing"""
    
    def __init__(self):
        self.text_responses = []
        self.user_transcripts = []
        self.errors = []
        self.response_complete_count = 0
        
    def on_text(self, text):
        self.text_responses.append(text)
        print(f"AI: {text}", end="", flush=True)
        
    def on_transcript(self, text):
        self.user_transcripts.append(text)
        print(f"\nUser: {text}")
        
    def on_done(self):
        self.response_complete_count += 1
        print("\n[Response complete]")
        
    def on_error(self, error):
        self.errors.append(str(error))
        print(f"\nError: {error}")


async def test_text_mode():
    """Test text mode functionality"""
    print("\n" + "="*60)
    print("TEST: Text Mode with VoiceEngine")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in .env")
        return False
        
    try:
        # Create VoiceEngine
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            log_level="INFO"  # More verbose to debug
        )
        
        engine = VoiceEngine(config=config)
        callbacks = TestCallbacks()
        
        # Set up callbacks directly (not using class methods)
        def on_text(text):
            callbacks.text_responses.append(text)
            print(f"AI: {text}", end="", flush=True)
            
        def on_done():
            callbacks.response_complete_count += 1
            print("\n[Response complete]")
            
        def on_error(error):
            callbacks.errors.append(str(error))
            print(f"\nError: {error}")
            
        def on_transcript(text):
            callbacks.user_transcripts.append(text)
            print(f"\nTranscript: {text}")
            
        engine.on_text_response = on_text
        engine.on_response_done = on_done
        engine.on_error = on_error
        
        # Also check for audio transcript callback
        if hasattr(engine, 'on_audio_transcript'):
            engine.on_audio_transcript = on_text
        
        # Connect
        print("\nConnecting to VoiceEngine...")
        await engine.connect()
        print("✓ Connected successfully")
        
        # Test sending text
        print("\nSending test message...")
        test_message = "Say 'Test successful'"
        await engine.send_text(test_message)
        
        # Wait for response
        print("\nWaiting for response...")
        await asyncio.sleep(5)
        
        # Check results
        success = True
        
        if callbacks.errors:
            print(f"\n❌ Errors occurred: {callbacks.errors}")
            success = False
            
        if not callbacks.text_responses:
            print("\n❌ No text response received")
            success = False
        else:
            print(f"\n✓ Received {len(''.join(callbacks.text_responses))} characters of response")
            
        if callbacks.response_complete_count == 0:
            print("❌ Response complete callback not triggered")
            success = False
        else:
            print(f"✓ Response completed {callbacks.response_complete_count} time(s)")
            
        # Disconnect
        print("\nDisconnecting...")
        await engine.disconnect()
        print("✓ Disconnected successfully")
        
        if success:
            print("\n✅ Text mode test PASSED")
        else:
            print("\n❌ Text mode test FAILED")
            
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cli_initialization():
    """Test VoxTermCLI can be initialized with different modes"""
    print("\n" + "="*60)
    print("TEST: VoxTermCLI Initialization")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in .env")
        return False
        
    try:
        # Create a minimal engine
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            log_level="ERROR"  # Minimal logging
        )
        
        engine = VoiceEngine(config=config)
        
        # Test creating CLI with different modes
        modes_to_test = ["push_to_talk", "always_on", "text", "turn_based"]
        
        for mode in modes_to_test:
            print(f"\nTesting {mode} mode...")
            cli = VoxTermCLI(engine, mode=mode)
            
            # Check mode was set correctly
            if hasattr(cli, 'mode') and cli.mode is not None:
                print(f"✓ Created CLI with {mode} mode")
            else:
                print(f"❌ Failed to create CLI with {mode} mode")
                return False
                
        print("\n✅ CLI initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all automated tests"""
    print("\n🔥 VoxTerm Automated Integration Tests\n")
    
    tests = [
        ("CLI Initialization", test_cli_initialization),
        ("Text Mode", test_text_mode),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} test crashed: {e}")
    
    print("\n" + "="*60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)