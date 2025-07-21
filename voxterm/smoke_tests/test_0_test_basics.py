#!/usr/bin/env python3
"""
VoxTerm Smoke Test 0: Test Basics

Tests basic VoxTerm functionality without VoiceEngine.
Just verifies that modes and keyboard handling work correctly.


python -m voxterm.smoke_tests.test_0_test_basics
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voxterm.modes import PushToTalkMode, AlwaysOnMode, TextMode, TurnBasedMode, create_mode
from voxterm.keyboard import KeyboardHandler, SimpleKeyboard


class MockEngine:
    """Mock engine for testing modes"""
    
    def __init__(self):
        self.listening = False
        self.messages = []
        self.actions = []
        
    async def start_listening(self):
        self.listening = True
        self.actions.append(("start_listening", time.time()))
        
    async def stop_listening(self):
        self.listening = False
        self.actions.append(("stop_listening", time.time()))
        
    async def send_text(self, text: str):
        self.messages.append(text)
        self.actions.append(("send_text", text))
        
    def clear_audio_buffer(self):
        self.actions.append(("clear_buffer", time.time()))


def test_push_to_talk_mode():
    """Test push-to-talk mode basics"""
    print("\n" + "="*50)
    print("TEST: Push-to-Talk Mode")
    print("="*50)
    
    engine = MockEngine()
    mode = PushToTalkMode(engine)
    
    async def run_test():
        # Test help text
        print(f"Help: {mode.get_help()}")
        assert "SPACE" in mode.get_help()
        
        # Test key press
        print("\nSimulating space press...")
        await mode.on_key_down("space")
        assert mode.is_recording == True
        print("✓ Recording flag set")
        
        # Wait a bit
        await asyncio.sleep(0.3)
        
        # Test key release (should return True for normal press)
        print("\nSimulating space release...")
        result = await mode.on_key_up("space")
        assert mode.is_recording == False
        assert result == True  # Should signal to send
        print("✓ Recording stopped, send signal returned")
        
        # Test short press (should return False to cancel)
        print("\nTesting short press...")
        await mode.on_key_down("space")
        result = await mode.on_key_up("space")  # Immediate release
        assert result == False  # Should signal to cancel
        print("✓ Short press cancelled")
        
        print("\n✅ Push-to-talk mode test PASSED")
    
    asyncio.run(run_test())


def test_always_on_mode():
    """Test always-on mode basics"""
    print("\n" + "="*50)
    print("TEST: Always-On Mode")
    print("="*50)
    
    engine = MockEngine()
    mode = AlwaysOnMode(engine)
    
    async def run_test():
        # Start listening (mode doesn't control engine directly anymore)
        await mode.start()
        print("✓ Mode started")
        
        # Test pause
        print("\nTesting pause...")
        result = await mode.on_key_down("p")
        assert mode.is_paused == True
        assert result == {"action": "pause"}
        print("✓ Pause action returned")
        
        # Test resume
        print("\nTesting resume...")
        result = await mode.on_key_down("p")
        assert mode.is_paused == False
        assert result == {"action": "resume"}
        print("✓ Resume action returned")
        
        # Stop
        await mode.stop()
        print("✓ Mode stopped")
        
        print("\n✅ Always-on mode test PASSED")
    
    asyncio.run(run_test())


def test_text_mode():
    """Test text mode basics"""
    print("\n" + "="*50)
    print("TEST: Text Mode")
    print("="*50)
    
    engine = MockEngine()
    mode = TextMode(engine)
    
    async def run_test():
        # Test text input
        print("Testing text input...")
        result = await mode.on_text_input("Hello, this is a test")
        assert result == "Hello, this is a test"
        print("✓ Text processed and returned")
        
        # Test empty input (should return None)
        result1 = await mode.on_text_input("")
        result2 = await mode.on_text_input("   ")
        assert result1 is None
        assert result2 is None
        print("✓ Empty input returns None")
        
        print("\n✅ Text mode test PASSED")
    
    asyncio.run(run_test())


def test_turn_based_mode():
    """Test turn-based mode basics"""
    print("\n" + "="*50)
    print("TEST: Turn-Based Mode")
    print("="*50)
    
    engine = MockEngine()
    mode = TurnBasedMode(engine)
    
    async def run_test():
        # Initial state
        assert mode.is_my_turn == True
        print("✓ Starts with user's turn")
        
        # Test turn management (simplified - no key handling in this mode)
        print("\nSimulating AI response...")
        mode.is_my_turn = False  # Simulate user sent message
        
        # Simulate AI response complete
        mode.on_response_complete()
        assert mode.is_my_turn == True
        print("✓ Turn returned to user after AI response")
        
        print("\n✅ Turn-based mode test PASSED")
    
    asyncio.run(run_test())


def test_mode_factory():
    """Test mode factory function"""
    print("\n" + "="*50)
    print("TEST: Mode Factory")
    print("="*50)
    
    engine = MockEngine()
    
    # Test valid modes
    modes_to_test = [
        ("push_to_talk", PushToTalkMode),
        ("ptt", PushToTalkMode),
        ("always_on", AlwaysOnMode),
        ("continuous", AlwaysOnMode),
        ("text", TextMode),
        ("type", TextMode),
        ("turn_based", TurnBasedMode),
        ("turns", TurnBasedMode),
    ]
    
    for mode_name, expected_class in modes_to_test:
        mode = create_mode(mode_name, engine)
        assert isinstance(mode, expected_class), f"Failed to create {mode_name}"
        print(f"✓ Created {mode_name} -> {expected_class.__name__}")
    
    # Test invalid mode
    invalid_mode = create_mode("invalid_mode", engine)
    assert invalid_mode is None
    print("✓ Invalid mode returns None")
    
    print("\n✅ Mode factory test PASSED")


def test_keyboard_handler():
    """Test keyboard handler basics"""
    print("\n" + "="*50)
    print("TEST: Keyboard Handler")
    print("="*50)
    
    handler = KeyboardHandler()
    
    # Track callbacks
    press_count = 0
    release_count = 0
    
    def on_press():
        nonlocal press_count
        press_count += 1
        
    def on_release():
        nonlocal release_count
        release_count += 1
    
    # Register callbacks
    handler.on_press('a', on_press)
    handler.on_release('a', on_release)
    
    # Simulate key events
    handler._handle_key_event('press', 'a')
    handler._handle_key_event('release', 'a')
    
    assert press_count == 1
    assert release_count == 1
    print("✓ Callbacks triggered correctly")
    
    # Test key normalization
    handler.on_press(' ', on_press)  # space as ' '
    handler._handle_key_event('press', 'space')  # but handle 'space'
    assert press_count == 2
    print("✓ Key normalization works")
    
    print("\n✅ Keyboard handler test PASSED")

def test_simple_keyboard():
    """Test SimpleKeyboard helper"""
    print("\n" + "="*50)
    print("TEST: SimpleKeyboard Helper")
    print("="*50)
    
    kb = SimpleKeyboard()
    
    space_pressed = False
    space_released = False
    key_pressed = False
    
    def on_space_press():
        nonlocal space_pressed
        space_pressed = True
        
    def on_space_release():
        nonlocal space_released
        space_released = True
        
    def on_m_press():
        nonlocal key_pressed
        key_pressed = True
    
    # Register handlers
    kb.on_space(on_space_press, on_space_release)
    kb.on_key('m', on_m_press)
    
    # Simulate events through the handler's callbacks dict directly
    # This ensures we test the actual registration
    assert 'space' in kb.handler.callbacks['press']
    assert 'space' in kb.handler.callbacks['release']
    assert 'm' in kb.handler.callbacks['press']
    print("✓ Handlers registered correctly")
    
    # Call the callbacks directly
    kb.handler.callbacks['press']['space']()
    kb.handler.callbacks['release']['space']()
    kb.handler.callbacks['press']['m']()
    
    assert space_pressed == True
    assert space_released == True
    assert key_pressed == True
    print("✓ SimpleKeyboard works correctly")
    
    print("\n✅ SimpleKeyboard test PASSED")



def run_all_tests():
    """Run all basic tests"""
    print("\n🔥 VoxTerm Basic Tests (No VoiceEngine)\n")
    
    tests = [
        test_push_to_talk_mode,
        test_always_on_mode,
        test_text_mode,
        test_turn_based_mode,
        test_mode_factory,
        test_keyboard_handler,
        test_simple_keyboard,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)