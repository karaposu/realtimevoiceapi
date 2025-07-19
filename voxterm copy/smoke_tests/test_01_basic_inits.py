# here is voxterm/smoke_tests/test_01_basic_inits.py
"""
VoxTerm Smoke Test 01: Basic Initialization

Tests basic VoxTerm initialization without a voice engine.
Verifies all components can be created and initialized properly.


python -m voxterm.smoke_tests.test_01_basic_inits
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voxterm.terminal.terminal import VoxTerminal
from voxterm.terminal.runner import VoxTermRunner
from voxterm.config.settings import TerminalSettings, PRESETS
from voxterm.core.state import get_state_manager
from voxterm.core.base import ComponentState  # Fixed import
from voxterm.core.events import get_event_bus, EventType

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_initialization():
    """Test basic terminal initialization"""
    print("=" * 60)
    print("TEST 1: Basic Terminal Initialization")
    print("=" * 60)
    
    try:
        # Create terminal with default settings
        terminal = VoxTerminal(
            title="Test Terminal",
            mode="text"  # Start with text mode (no audio needed)
        )
        
        print("✓ Terminal created")
        
        # Initialize
        await terminal.initialize()
        print("✓ Terminal initialized")
        
        # Check component states
        assert terminal.state == ComponentState.READY, f"Terminal state should be READY, got {terminal.state}"
        print("✓ Terminal state is READY")
        
        # Check components exist
        assert terminal.keyboard_manager is not None, "Keyboard manager not created"
        assert terminal.audio_input_manager is not None, "Audio input manager not created"
        assert terminal.text_input_handler is not None, "Text input handler not created"
        assert terminal.display is not None, "Display not created"
        print("✓ All components created")
        
        # Check modes
        assert len(terminal.modes) == 4, f"Should have 4 modes, got {len(terminal.modes)}"
        assert "text" in terminal.modes, "Text mode not found"
        assert "push_to_talk" in terminal.modes, "Push-to-talk mode not found"
        print("✓ All modes available")
        
        # Start terminal
        await terminal.start()
        print("✓ Terminal started")
        
        # Check running state
        assert terminal.state == ComponentState.RUNNING, "Terminal should be running"
        assert terminal._running == True, "Terminal _running flag should be True"
        print("✓ Terminal is running")
        
        # Test mode switching
        await terminal._switch_mode("push_to_talk")
        assert terminal.current_mode.mode_type.value == "push_to_talk", "Mode switch failed"
        print("✓ Mode switching works")
        
        # Shutdown
        await terminal.shutdown()
        print("✓ Terminal shutdown complete")
        
        assert terminal.state == ComponentState.STOPPED, "Terminal should be stopped"
        print("✓ Terminal state is STOPPED")
        
        print("\n✅ Basic initialization test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Basic initialization test FAILED: {e}\n")
        raise


async def test_configuration():
    """Test configuration system"""
    print("=" * 60)
    print("TEST 2: Configuration System")
    print("=" * 60)
    
    try:
        # Test with custom config
        config = TerminalSettings()
        config.display.theme = "light"
        config.key_bindings.push_to_talk = "ctrl"
        config.display.show_timestamps = False
        
        terminal = VoxTerminal(
            title="Config Test",
            mode="text",
            config=config
        )
        
        await terminal.initialize()
        print("✓ Terminal initialized with custom config")
        
        # Check config was applied
        assert terminal.get_config("display.theme") == "light", "Theme config not applied"
        assert terminal.get_config("key_bindings.push_to_talk") == "ctrl", "Key binding not applied"
        assert terminal.get_config("display.show_timestamps") == False, "Timestamp config not applied"
        print("✓ Custom configuration applied")
        
        # Test runtime config changes
        success = terminal.set_config("display.show_timestamps", True)
        assert success == True, "Config update failed"
        assert terminal.get_config("display.show_timestamps") == True, "Config not updated"
        print("✓ Runtime configuration update works")
        
        await terminal.shutdown()
        
        print("\n✅ Configuration test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Configuration test FAILED: {e}\n")
        raise


async def test_state_management():
    """Test state management"""
    print("=" * 60)
    print("TEST 3: State Management")
    print("=" * 60)
    
    try:
        terminal = VoxTerminal(title="State Test", mode="text")
        await terminal.initialize()
        await terminal.start()
        
        # Get state manager
        state_manager = get_state_manager()
        state = state_manager.get_state()
        
        print("✓ State manager accessible")
        
        # Check initial state
        assert state.input_mode.value == "text", "Initial mode should be text"
        assert state.connection_state.value == "disconnected", "Should start disconnected"
        print("✓ Initial state correct")
        
        # Test state updates
        state_manager.update_connection("connected", 45.5)
        state = state_manager.get_state()
        assert state.connection_state.value == "connected", "Connection state not updated"
        assert state.api_latency_ms == 45.5, "Latency not updated"
        print("✓ State updates work")
        
        # Test conversation state
        msg = state_manager.add_message("user", "Hello")
        assert len(state.conversation.messages) == 1, "Message not added"
        assert msg.role == "user", "Message role incorrect"
        assert msg.content == "Hello", "Message content incorrect"
        print("✓ Conversation state works")
        
        await terminal.shutdown()
        
        print("\n✅ State management test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ State management test FAILED: {e}\n")
        raise


async def test_event_system():
    """Test event system"""
    print("=" * 60)
    print("TEST 4: Event System")
    print("=" * 60)
    
    try:
        terminal = VoxTerminal(title="Event Test", mode="text")
        
        # Track events
        received_events = []
        
        def event_handler(event):
            received_events.append(event)
        
        # Subscribe to events
        event_bus = get_event_bus()
        event_bus.subscribe(EventType.INFO, event_handler)
        event_bus.subscribe(EventType.MODE_CHANGE, event_handler)
        
        print("✓ Event subscriptions set up")
        
        await terminal.initialize()
        await terminal.start()
        
        # Should have received initialization events
        assert len(received_events) > 0, "No events received"
        print(f"✓ Received {len(received_events)} events during startup")
        
        # Clear and test mode change event
        received_events.clear()
        await terminal._switch_mode("push_to_talk")
        
        # Wait for events to propagate
        await asyncio.sleep(0.1)
        
        mode_events = [e for e in received_events if e.type == EventType.MODE_CHANGE]
        assert len(mode_events) > 0, "No mode change events received"
        print("✓ Mode change events work")
        
        await terminal.shutdown()
        
        print("\n✅ Event system test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Event system test FAILED: {e}\n")
        raise


async def test_runner():
    """Test terminal runner"""
    print("=" * 60)
    print("TEST 5: Terminal Runner")
    print("=" * 60)
    
    try:
        # Test runner creation
        runner = VoxTermRunner.create_terminal(
            title="Runner Test",
            mode="text"
        )
        
        assert runner.terminal is not None, "Terminal not created"
        print("✓ Runner created terminal")
        
        # Test with preset
        runner_preset = VoxTermRunner.create_terminal(
            title="Preset Test",
            mode="text",
            preset="minimal"
        )
        
        await runner_preset.terminal.initialize()
        config = runner_preset.terminal.get_config("display.show_status_bar")
        assert config == False, "Minimal preset not applied"
        print("✓ Preset configuration works")
        
        await runner_preset.terminal.shutdown()
        
        print("\n✅ Runner test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Runner test FAILED: {e}\n")
        raise


async def test_display_message():
    """Test message display functionality"""
    print("=" * 60)
    print("TEST 6: Message Display")
    print("=" * 60)
    
    try:
        terminal = VoxTerminal(title="Display Test", mode="text")
        await terminal.initialize()
        await terminal.start()
        
        # Display a message
        terminal.display_message({
            "role": "system",
            "content": "Test message"
        })
        
        # Check message was added to state
        state = get_state_manager().get_state()
        assert len(state.conversation.messages) == 1, "Message not added"
        assert state.conversation.messages[0].content == "Test message", "Message content wrong"
        print("✓ Message display works")
        
        # Test user message
        terminal.display_message({
            "role": "user",
            "content": "Hello from user"
        })
        
        assert len(state.conversation.messages) == 2, "Second message not added"
        print("✓ Multiple messages work")
        
        await terminal.shutdown()
        
        print("\n✅ Message display test PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Message display test FAILED: {e}\n")
        raise


async def run_all_tests():
    """Run all smoke tests"""
    print("\n🔥 VoxTerm Smoke Test Suite - Basic Initialization\n")
    
    tests = [
        test_basic_initialization,
        test_configuration,
        test_state_management,
        test_event_system,
        test_runner,
        test_display_message
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


def main():
    """Main entry point"""
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()