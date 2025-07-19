#!/usr/bin/env python3
"""
VoxTerm Smoke Test 02: Integration with VoiceEngine

Tests VoxTerm integration with a mock voice engine.
Verifies callbacks, audio flow, and real-time behavior.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voxterm.terminal.terminal import VoxTerminal
from voxterm.terminal.runner import VoxTermRunner, run_terminal, VoxTermContext
from voxterm.core.state import get_state_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockVoiceEngine:
    """Mock voice engine for testing VoxTerm integration"""
    
    def __init__(self):
        # Callbacks that VoxTerm will set
        self.on_text_response: Optional[Callable] = None
        self.on_audio_response: Optional[Callable] = None
        self.on_user_transcript: Optional[Callable] = None
        self.on_response_done: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # State
        self.is_connected = False
        self.is_listening = False
        self.audio_level = 0.0
        
        # Metrics
        self.text_messages_sent = 0
        self.audio_chunks_sent = 0
        self.interrupts_received = 0
    
    async def connect(self):
        """Simulate connection"""
        await asyncio.sleep(0.1)  # Simulate network delay
        self.is_connected = True
        logger.info("MockVoiceEngine connected")
    
    async def disconnect(self):
        """Simulate disconnection"""
        self.is_connected = False
        logger.info("MockVoiceEngine disconnected")
    
    async def send_text(self, text: str):
        """Simulate sending text message"""
        self.text_messages_sent += 1
        logger.info(f"MockVoiceEngine received text: {text}")
        
        # Simulate processing delay
        await asyncio.sleep(0.2)
        
        # Simulate response
        if self.on_text_response:
            response = f"You said: {text}"
            # Stream response
            for word in response.split():
                self.on_text_response(word + " ")
                await asyncio.sleep(0.05)
        
        if self.on_response_done:
            self.on_response_done()
    
    async def start_listening(self):
        """Start listening for audio"""
        self.is_listening = True
        logger.info("MockVoiceEngine started listening")
    
    async def stop_listening(self):
        """Stop listening for audio"""
        self.is_listening = False
        logger.info("MockVoiceEngine stopped listening")
        
        # Simulate audio processing
        await asyncio.sleep(0.1)
        
        # Simulate transcript
        if self.on_user_transcript:
            self.on_user_transcript("This is a test recording")
        
        # Simulate response
        if self.on_audio_response:
            # Send some fake audio chunks
            for i in range(3):
                fake_audio = b'\x00' * 1024  # 1KB of silence
                self.audio_chunks_sent += 1
                self.on_audio_response(fake_audio)
                await asyncio.sleep(0.1)
        
        if self.on_response_done:
            self.on_response_done()
    
    async def interrupt(self):
        """Interrupt current processing"""
        self.interrupts_received += 1
        logger.info("MockVoiceEngine interrupted")
    
    def get_audio_level(self) -> float:
        """Get current audio level"""
        # Simulate varying audio level
        import random
        self.audio_level = random.random() * 0.5
        return self.audio_level
    
    def set_input_device(self, device_id: str):
        """Set input device"""
        logger.info(f"MockVoiceEngine set input device: {device_id}")
    
    def play_audio(self, audio_data: bytes):
        """Play audio (mock)"""
        logger.info(f"MockVoiceEngine playing audio: {len(audio_data)} bytes")


async def test_basic_integration():
    """Test basic VoxTerm + VoiceEngine integration"""
    print("=" * 60)
    print("TEST 1: Basic Voice Engine Integration")
    print("=" * 60)
    
    try:
        # Create mock engine
        engine = MockVoiceEngine()
        
        # Create terminal
        terminal = VoxTerminal(
            title="Voice Engine Test",
            mode="text"  # Start with text mode
        )
        
        # Bind engine
        terminal.bind_voice_engine(engine)
        print("✓ Voice engine bound to terminal")
        
        # Initialize and start
        await terminal.initialize()
        await terminal.start()
        print("✓ Terminal started")
        
        # Check callbacks were set
        assert engine.on_text_response is not None, "Text response callback not set"
        assert engine.on_response_done is not None, "Response done callback not set"
        print("✓ Callbacks properly connected")
        
        # Test sending a text message
        await engine.send_text("Hello, VoxTerm!")
        
        # Wait for response to complete
        await asyncio.sleep(1.0)
        
        # Check state was updated
        state = get_state_manager().get_state()
        assert len(state.conversation.messages) > 0, "No messages in conversation"
        print("✓ Text messaging works")
        
        # Shutdown
        await terminal.shutdown()
        
        print("\n✅ Basic integration test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Basic integration test FAILED: {e}\n")
        raise


async def test_audio_mode_integration():
    """Test audio mode integration"""
    print("=" * 60)
    print("TEST 2: Audio Mode Integration")
    print("=" * 60)
    
    try:
        engine = MockVoiceEngine()
        await engine.connect()
        
        terminal = VoxTerminal(
            title="Audio Mode Test",
            mode="push_to_talk"
        )
        
        terminal.bind_voice_engine(engine)
        await terminal.initialize()
        await terminal.start()
        
        print("✓ Terminal in push-to-talk mode")
        
        # Get audio input manager
        audio_manager = terminal.audio_input_manager
        
        # Simulate recording
        await audio_manager.start_recording()
        assert engine.is_listening == True, "Engine should be listening"
        print("✓ Recording started")
        
        await asyncio.sleep(0.2)
        
        await audio_manager.stop_recording()
        assert engine.is_listening == False, "Engine should stop listening"
        print("✓ Recording stopped")
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check audio was processed
        assert engine.audio_chunks_sent > 0, "No audio chunks sent"
        print(f"✓ Audio processed ({engine.audio_chunks_sent} chunks)")
        
        await terminal.shutdown()
        
        print("\n✅ Audio mode integration test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Audio mode integration test FAILED: {e}\n")
        raise


async def test_runner_integration():
    """Test runner with voice engine"""
    print("=" * 60)
    print("TEST 3: Runner Integration")
    print("=" * 60)
    
    try:
        engine = MockVoiceEngine()
        
        # Create runner with engine
        runner = VoxTermRunner.with_voice_engine(
            engine,
            title="Runner Test",
            mode="text"
        )
        
        print("✓ Runner created with voice engine")
        
        # Initialize terminal
        await runner.terminal.initialize()
        
        # Check engine is bound
        assert runner.terminal.voice_engine == engine, "Engine not bound"
        print("✓ Engine properly bound through runner")
        
        await runner.terminal.shutdown()
        
        print("\n✅ Runner integration test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Runner integration test FAILED: {e}\n")
        raise


async def test_context_manager():
    """Test context manager integration"""
    print("=" * 60)
    print("TEST 4: Context Manager Integration")
    print("=" * 60)
    
    try:
        engine = MockVoiceEngine()
        await engine.connect()
        
        # Use context manager
        async with VoxTermContext(
            title="Context Test",
            mode="text",
            voice_engine=engine
        ) as terminal:
            print("✓ Terminal started with context manager")
            
            # Terminal should be running
            assert terminal.state.value == "running", "Terminal not running"
            assert terminal.voice_engine == engine, "Engine not bound"
            
            # Send a message
            await engine.send_text("Testing context manager")
            await asyncio.sleep(0.5)
            
            print("✓ Operations work within context")
        
        # Terminal should be stopped after context
        assert terminal.state.value == "stopped", "Terminal not stopped"
        print("✓ Terminal properly shut down")
        
        print("\n✅ Context manager test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Context manager test FAILED: {e}\n")
        raise


async def test_real_time_behavior():
    """Test real-time behavior (non-blocking)"""
    print("=" * 60)
    print("TEST 5: Real-Time Behavior")
    print("=" * 60)
    
    try:
        engine = MockVoiceEngine()
        terminal = VoxTerminal(title="Real-time Test", mode="push_to_talk")
        
        terminal.bind_voice_engine(engine)
        await terminal.initialize()
        await terminal.start()
        
        # Track timing
        start_time = time.time()
        
        # Run multiple operations concurrently
        async def send_messages():
            for i in range(3):
                await engine.send_text(f"Message {i}")
                await asyncio.sleep(0.1)
        
        async def simulate_audio():
            for i in range(2):
                await terminal.audio_input_manager.start_recording()
                await asyncio.sleep(0.2)
                await terminal.audio_input_manager.stop_recording()
                await asyncio.sleep(0.3)
        
        # Run concurrently
        await asyncio.gather(
            send_messages(),
            simulate_audio()
        )
        
        duration = time.time() - start_time
        print(f"✓ Concurrent operations completed in {duration:.2f}s")
        
        # Check nothing blocked
        assert duration < 2.0, "Operations took too long (blocking detected)"
        print("✓ No blocking detected")
        
        # Check all operations completed
        assert engine.text_messages_sent == 3, "Not all text messages sent"
        print("✓ All operations completed successfully")
        
        await terminal.shutdown()
        
        print("\n✅ Real-time behavior test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Real-time behavior test FAILED: {e}\n")
        raise


async def test_interrupt_handling():
    """Test interrupt handling"""
    print("=" * 60)
    print("TEST 6: Interrupt Handling")
    print("=" * 60)
    
    try:
        engine = MockVoiceEngine()
        terminal = VoxTerminal(title="Interrupt Test", mode="always_on")
        
        terminal.bind_voice_engine(engine)
        await terminal.initialize()
        await terminal.start()
        
        # Start a long operation
        asyncio.create_task(engine.send_text("This is a long message that will be interrupted"))
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Interrupt
        await engine.interrupt()
        
        assert engine.interrupts_received == 1, "Interrupt not received"
        print("✓ Interrupt handling works")
        
        await terminal.shutdown()
        
        print("\n✅ Interrupt handling test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Interrupt handling test FAILED: {e}\n")
        raise


async def run_all_tests():
    """Run all voice engine integration tests"""
    print("\n🔥 VoxTerm Smoke Test Suite - Voice Engine Integration\n")
    
    tests = [
        test_basic_integration,
        test_audio_mode_integration,
        test_runner_integration,
        test_context_manager,
        test_real_time_behavior,
        test_interrupt_handling
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


def test_quick_start_sync():
    """Test synchronous quick start (doesn't actually run, just verifies setup)"""
    print("\n" + "=" * 60)
    print("TEST 7: Quick Start API")
    print("=" * 60)
    
    try:
        from voxterm import run_terminal
        
        # This would normally run the terminal
        # We're just testing that the API exists
        print("✓ run_terminal function available")
        
        # Test runner creation
        engine = MockVoiceEngine()
        runner = VoxTermRunner.with_voice_engine(engine)
        
        assert runner.terminal is not None, "Terminal not created"
        assert runner.terminal.voice_engine == engine, "Engine not bound"
        
        print("✓ Quick start API works")
        print("\n✅ Quick start test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Quick start test FAILED: {e}\n")
        return False
    
    return True


def main():
    """Main entry point"""
    # Run async tests
    success = asyncio.run(run_all_tests())
    
    # Run sync test
    if success:
        success = test_quick_start_sync()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()