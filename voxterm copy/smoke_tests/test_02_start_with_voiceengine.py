#!/usr/bin/env python3
"""
VoxTerm Smoke Test 02: Integration with Real VoiceEngine

Tests VoxTerm integration with the actual realtimevoiceapi.VoiceEngine.
Requires OPENAI_API_KEY to be set.
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

from voxterm.terminal.terminal import VoxTerminal
from voxterm.terminal.runner import VoxTermRunner, run_terminal, VoxTermContext
from voxterm.core.state import get_state_manager

# Import the real VoiceEngine
try:
    from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
    VOICEENGINE_AVAILABLE = True
except ImportError:
    print("❌ realtimevoiceapi not found. Please ensure it's installed.")
    VOICEENGINE_AVAILABLE = False
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_voiceengine_basic_integration():
    """Test basic VoxTerm + VoiceEngine integration"""
    print("=" * 60)
    print("TEST 1: Basic VoiceEngine Integration")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        return
    
    try:
        # Create VoiceEngine with config
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="qa",  # Q&A mode for testing
            voice="alloy",
            log_level="INFO"
        )
        
        engine = VoiceEngine(config=config)
        print("✓ VoiceEngine created")
        
        # Create terminal
        terminal = VoxTerminal(
            title="VoiceEngine Integration Test",
            mode="text"  # Start with text mode
        )
        
        # Bind engine
        terminal.bind_voice_engine(engine)
        print("✓ VoiceEngine bound to terminal")
        
        # Initialize and start terminal
        await terminal.initialize()
        await terminal.start()
        print("✓ Terminal started")
        
        # Connect VoiceEngine
        await engine.connect()
        print("✓ VoiceEngine connected to OpenAI")
        
        # Test sending a text message
        print("\n📤 Sending test message...")
        await engine.send_text("Hello from VoxTerm! Please respond with a short greeting.")
        
        # Wait for response
        print("⏳ Waiting for response...")
        await asyncio.sleep(3.0)
        
        # Check conversation state
        state = get_state_manager().get_state()
        messages = state.conversation.messages
        
        print(f"\n📊 Conversation has {len(messages)} messages")
        for msg in messages:
            print(f"  - {msg.role}: {msg.content[:50]}...")
        
        assert len(messages) >= 1, "No messages in conversation"
        print("✓ Text messaging works with real API")
        
        # Disconnect and shutdown
        await engine.disconnect()
        await terminal.shutdown()
        
        print("\n✅ Basic VoiceEngine integration test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Basic integration test FAILED: {e}\n")
        raise


async def test_voiceengine_with_runner():
    """Test VoiceEngine with terminal runner"""
    print("=" * 60)
    print("TEST 2: VoiceEngine with Runner")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        return
    
    try:
        # Create engine
        engine = VoiceEngine(config=VoiceEngineConfig(
            api_key=api_key,
            mode="chat",
            voice="nova"
        ))
        
        # Create runner with engine
        runner = VoxTermRunner.with_voice_engine(
            engine,
            title="Runner + VoiceEngine Test",
            mode="text"
        )
        
        print("✓ Runner created with VoiceEngine")
        
        # Initialize
        await runner.terminal.initialize()
        await runner.terminal.start()
        
        # Connect engine
        await engine.connect()
        print("✓ Connected via runner")
        
        # Quick test
        await engine.send_text("Say 'VoxTerm Runner Test Success!'")
        await asyncio.sleep(2.0)
        
        # Cleanup
        await engine.disconnect()
        await runner.terminal.shutdown()
        
        print("\n✅ Runner + VoiceEngine test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Runner test FAILED: {e}\n")
        raise


async def test_voiceengine_audio_mode():
    """Test VoiceEngine in audio mode (push-to-talk)"""
    print("=" * 60)
    print("TEST 3: VoiceEngine Audio Mode")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        return
    
    try:
        # Create engine with audio config
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="chat",
            voice="alloy",
            vad_enabled=False,  # Manual control for push-to-talk
            log_level="DEBUG"
        )
        
        engine = VoiceEngine(config=config)
        
        # Create terminal in push-to-talk mode
        terminal = VoxTerminal(
            title="Audio Mode Test",
            mode="push_to_talk"
        )
        
        terminal.bind_voice_engine(engine)
        await terminal.initialize()
        await terminal.start()
        
        print("✓ Terminal in push-to-talk mode")
        
        # Connect engine
        await engine.connect()
        print("✓ VoiceEngine connected")
        
        # Simulate push-to-talk
        audio_manager = terminal.audio_input_manager
        
        print("\n🎤 Simulating push-to-talk...")
        await audio_manager.start_recording()
        print("✓ Recording started")
        
        # Simulate some recording time
        await asyncio.sleep(0.5)
        
        await audio_manager.stop_recording()
        print("✓ Recording stopped")
        
        # Note: In a real scenario, actual audio would be captured
        # For this test, we're just verifying the flow works
        
        # Cleanup
        await engine.disconnect()
        await terminal.shutdown()
        
        print("\n✅ Audio mode test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Audio mode test FAILED: {e}\n")
        raise


async def test_voiceengine_context_manager():
    """Test VoiceEngine with context manager"""
    print("=" * 60)
    print("TEST 4: VoiceEngine Context Manager")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        return
    
    try:
        # Create and connect engine
        engine = VoiceEngine(config=VoiceEngineConfig(
            api_key=api_key,
            mode="qa"
        ))
        await engine.connect()
        
        # Use context manager
        async with VoxTermContext(
            title="Context Manager Test",
            mode="text",
            voice_engine=engine
        ) as terminal:
            print("✓ Terminal active in context")
            
            # Send a test message
            await engine.send_text("Context manager test - please acknowledge.")
            await asyncio.sleep(2.0)
            
            # Check terminal is running
            assert terminal.state.value == "running"
            print("✓ Operations work within context")
        
        # Terminal should be cleaned up
        print("✓ Context exited cleanly")
        
        # Disconnect engine
        await engine.disconnect()
        
        print("\n✅ Context manager test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Context manager test FAILED: {e}\n")
        raise


async def test_voiceengine_streaming():
    """Test streaming responses from VoiceEngine"""
    print("=" * 60)
    print("TEST 5: Streaming Response Test")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        return
    
    try:
        engine = VoiceEngine(config=VoiceEngineConfig(
            api_key=api_key,
            mode="chat",
            voice="alloy"
        ))
        
        terminal = VoxTerminal(title="Streaming Test", mode="text")
        terminal.bind_voice_engine(engine)
        
        await terminal.initialize()
        await terminal.start()
        await engine.connect()
        
        print("✓ Setup complete")
        
        # Track streaming
        chunks_received = []
        
        def on_text_chunk(text):
            chunks_received.append(text)
            print(".", end="", flush=True)
        
        # Override the text callback to track chunks
        original_callback = engine.on_text_response
        engine.on_text_response = on_text_chunk
        
        print("\n📤 Sending message for streaming response...")
        await engine.send_text("Count from 1 to 5 slowly.")
        
        print("\n⏳ Receiving streamed response", end="")
        await asyncio.sleep(3.0)
        print()
        
        # Restore callback
        engine.on_text_response = original_callback
        
        print(f"\n✓ Received {len(chunks_received)} chunks")
        assert len(chunks_received) > 1, "Should receive multiple chunks"
        
        # Cleanup
        await engine.disconnect()
        await terminal.shutdown()
        
        print("\n✅ Streaming test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Streaming test FAILED: {e}\n")
        raise


async def test_voiceengine_mode_switching():
    """Test switching modes with VoiceEngine"""
    print("=" * 60)
    print("TEST 6: Mode Switching with VoiceEngine")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
        return
    
    try:
        engine = VoiceEngine(config=VoiceEngineConfig(
            api_key=api_key,
            mode="chat"
        ))
        
        terminal = VoxTerminal(title="Mode Switch Test", mode="text")
        terminal.bind_voice_engine(engine)
        
        await terminal.initialize()
        await terminal.start()
        await engine.connect()
        
        # Start in text mode
        assert terminal.get_mode() == "text"
        print("✓ Started in text mode")
        
        # Switch to push-to-talk
        await terminal._switch_mode("push_to_talk")
        assert terminal.get_mode() == "push_to_talk"
        print("✓ Switched to push-to-talk mode")
        
        # Switch to always-on
        await terminal._switch_mode("always_on")
        assert terminal.get_mode() == "always_on"
        print("✓ Switched to always-on mode")
        
        # Switch back to text
        await terminal._switch_mode("text")
        assert terminal.get_mode() == "text"
        print("✓ Switched back to text mode")
        
        # Cleanup
        await engine.disconnect()
        await terminal.shutdown()
        
        print("\n✅ Mode switching test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Mode switching test FAILED: {e}\n")
        raise


async def run_all_tests():
    """Run all VoiceEngine integration tests"""
    print("\n🔥 VoxTerm Smoke Test Suite - Real VoiceEngine Integration\n")
    
    if not VOICEENGINE_AVAILABLE:
        print("❌ Cannot run tests - realtimevoiceapi not available")
        return False
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Cannot run tests - OPENAI_API_KEY not set")
        print("\n💡 To run these tests:")
        print("   1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("   2. Or create a .env file with: OPENAI_API_KEY=your-key")
        return False
    
    print(f"✓ Using API key: ...{api_key[-8:]}")
    print()
    
    tests = [
        test_voiceengine_basic_integration,
        test_voiceengine_with_runner,
        test_voiceengine_audio_mode,
        test_voiceengine_context_manager,
        test_voiceengine_streaming,
        test_voiceengine_mode_switching
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"Test failed: {e}", exc_info=True)
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def demo_quick_start():
    """Demo the quick start functionality"""
    print("\n" + "=" * 60)
    print("DEMO: Quick Start Example")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Skipping demo - no API key")
        return
    
    print("This is how you'd use VoxTerm in a real application:\n")
    
    print("```python")
    print("from voxterm import run_terminal")
    print("from realtimevoiceapi import VoiceEngine")
    print("")
    print('engine = VoiceEngine(api_key="...")')
    print("run_terminal(voice_engine=engine)")
    print("```")
    
    print("\n✅ That's it! VoxTerm handles the rest.\n")


def main():
    """Main entry point"""
    success = asyncio.run(run_all_tests())
    
    if success:
        demo_quick_start()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()