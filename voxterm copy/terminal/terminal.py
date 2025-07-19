"""
VoxTerm Main Terminal Implementation

The main VoxTerminal class that coordinates all components.
"""

import asyncio
import threading
import time
import signal
import sys
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

from ..core.base import BaseTerminal, VoiceEngineAdapter, ComponentState
from ..core.events import get_event_bus, EventType, Event, emit_event
from ..core.state import get_state_manager, StateManager, InputMode
from ..config.manager import get_config_manager, ConfigManager
from ..config.settings import TerminalSettings

from ..input.keyboard import KeyboardManager, KeyboardBackend
from ..input.audio_input import AudioInputManager
from ..input.text_input import TextInputHandler

from ..output.display import TerminalDisplay
from ..output.audio_output import AudioOutputManager
from ..output.formatters import MessageFormatter

from ..modes.base_mode import BaseMode
from ..modes.push_to_talk import PushToTalkMode
from ..modes.always_on import AlwaysOnMode
from ..modes.text_mode import TextMode
from ..modes.turn_based import TurnBasedMode


class VoxTerminal(BaseTerminal):
    """
    Main VoxTerm terminal interface.
    
    Coordinates all components to provide a rich terminal UI
    for voice applications without blocking real-time audio.
    """
    
    def __init__(
        self,
        title: str = "VoxTerm Voice Chat",
        mode: str = "push_to_talk",
        config: Optional[TerminalSettings] = None,
        keyboard_backend: KeyboardBackend = KeyboardBackend.PYNPUT
    ):
        super().__init__(title)
        
        # Configuration
        self.config_manager = get_config_manager()
        if config:
            self.config_manager._config = config
        
        # State management
        self.state_manager = get_state_manager()
        
        # Event bus
        self.event_bus = get_event_bus()
        
        # Components
        self.keyboard_manager = KeyboardManager(backend=keyboard_backend)
        self.audio_input_manager = AudioInputManager()
        self.text_input_handler = TextInputHandler()
        self.audio_output_manager = AudioOutputManager()
        self.display = TerminalDisplay()
        
        # Modes
        self.modes: Dict[str, BaseMode] = {
            "push_to_talk": PushToTalkMode(),
            "always_on": AlwaysOnMode(),
            "text": TextMode(),
            "turn_based": TurnBasedMode()
        }
        self.current_mode: Optional[BaseMode] = None
        self.default_mode = mode
        
        # Voice engine adapter
        self.voice_engine_adapter: Optional[VoiceEngineAdapter] = None
        self.voice_engine = None
        
        # Runtime state
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self._on_ready_callback: Optional[Callable] = None
        self._on_shutdown_callback: Optional[Callable] = None
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    async def initialize(self) -> None:
        """Initialize all components"""
        self.set_state(ComponentState.INITIALIZING)
        
        try:
            # Initialize components in order
            await self.display.initialize()
            await self.keyboard_manager.initialize()
            await self.audio_input_manager.initialize()
            await self.text_input_handler.initialize()
            await self.audio_output_manager.initialize()
            
            # Initialize modes
            for mode in self.modes.values():
                await mode.initialize()
                # Bind components to mode
                mode.bind_components(
                    keyboard_manager=self.keyboard_manager,
                    audio_input_manager=self.audio_input_manager,
                    text_input_handler=self.text_input_handler,
                    voice_engine=self.voice_engine
                )
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Set initial mode
            await self._switch_mode(self.default_mode)
            
            self.set_state(ComponentState.READY)
            
            # Call ready callback
            if self._on_ready_callback:
                self._on_ready_callback()
            
            emit_event(Event(
                type=EventType.INFO,
                source="VoxTerminal",
                data={"message": f"VoxTerm initialized - {self.title}"}
            ))
            
        except Exception as e:
            self.set_state(ComponentState.ERROR)
            emit_event(Event(
                type=EventType.ERROR,
                source="VoxTerminal",
                data={"error": f"Initialization failed: {e}"}
            ))
            raise
    
    async def start(self) -> None:
        """Start all components"""
        if self._running:
            return
        
        self.set_state(ComponentState.STARTING)
        
        try:
            # Start components
            await self.display.start()
            await self.keyboard_manager.start()
            await self.audio_input_manager.start()
            await self.text_input_handler.start()
            await self.audio_output_manager.start()
            
            # Start current mode
            if self.current_mode:
                await self.current_mode.start()
            
            self._running = True
            self.set_state(ComponentState.RUNNING)
            
            # Initial display render
            self.display.render({"section": "full"})
            
            emit_event(Event(
                type=EventType.INFO,
                source="VoxTerminal",
                data={"message": "VoxTerm started"}
            ))
            
        except Exception as e:
            self.set_state(ComponentState.ERROR)
            emit_event(Event(
                type=EventType.ERROR,
                source="VoxTerminal",
                data={"error": f"Start failed: {e}"}
            ))
            raise
    
    async def stop(self) -> None:
        """Stop the terminal (implements abstract method)"""
        await self.shutdown()
    
    async def run(self) -> None:
        """Run the terminal UI (main entry point)"""
        try:
            # Initialize if not already done
            if self.state != ComponentState.READY:
                await self.initialize()
            
            # Start components
            await self.start()
            
            # Run main loop
            self._main_task = asyncio.create_task(self._main_loop())
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown all components"""
        if self.state == ComponentState.STOPPED:
            return
        
        self.set_state(ComponentState.STOPPING)
        self._running = False
        
        emit_event(Event(
            type=EventType.SHUTDOWN,
            source="VoxTerminal"
        ))
        
        # Stop current mode
        if self.current_mode:
            await self.current_mode.stop()
        
        # Stop components in reverse order
        await self.audio_output_manager.stop()
        await self.text_input_handler.stop()
        await self.audio_input_manager.stop()
        await self.keyboard_manager.stop()
        await self.display.stop()
        
        # Stop event bus
        self.event_bus.stop()
        
        # Cancel main task
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        self.set_state(ComponentState.STOPPED)
        
        # Call shutdown callback
        if self._on_shutdown_callback:
            self._on_shutdown_callback()
        
        print("\nGoodbye!")
    
    def bind_voice_engine(self, engine: Any) -> None:
        """
        Bind a voice engine to the terminal.
        
        The engine should have methods like:
        - connect() / disconnect()
        - send_text(text: str)
        - send_audio(audio: bytes)
        - start_listening() / stop_listening()
        - interrupt()
        """
        self.voice_engine = engine
        
        # Bind to components
        self.audio_input_manager.bind_voice_engine(engine)
        self.audio_output_manager.bind_voice_engine(engine)
        
        # Update modes with engine reference
        for mode in self.modes.values():
            mode.voice_engine = engine
        
        # Set up engine callbacks if it supports them
        self._setup_voice_engine_callbacks()
        
        emit_event(Event(
            type=EventType.INFO,
            source="VoxTerminal",
            data={"message": "Voice engine bound"}
        ))
    
    async def _main_loop(self):
        """Main event processing loop"""
        while self._running:
            try:
                # Process command queue
                if not self._command_queue.empty():
                    try:
                        command = self._command_queue.get_nowait()
                        await self._process_command(command)
                    except:
                        pass
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.05)
                
            except Exception as e:
                emit_event(Event(
                    type=EventType.ERROR,
                    source="VoxTerminal",
                    data={"error": f"Main loop error: {e}"}
                ))
    
    async def _process_command(self, command: Dict[str, Any]):
        """Process a command"""
        cmd_type = command.get("type")
        
        if cmd_type == "switch_mode":
            await self._switch_mode(command.get("mode"))
        elif cmd_type == "quit":
            self._shutdown_event.set()
        elif cmd_type == "config":
            self._handle_config_command(command)
        elif cmd_type == "clear":
            self.state_manager.get_state().conversation.clear()
            self.display.clear()
        else:
            # Forward to current mode
            if self.current_mode:
                await self.current_mode.handle_command(command)
    
    async def _switch_mode(self, mode_name: str):
        """Switch to a different interaction mode"""
        if mode_name not in self.modes:
            emit_event(Event(
                type=EventType.WARNING,
                source="VoxTerminal",
                data={"message": f"Unknown mode: {mode_name}"}
            ))
            return
        
        # Deactivate current mode
        if self.current_mode:
            await self.current_mode.deactivate()
        
        # Activate new mode
        self.current_mode = self.modes[mode_name]
        await self.current_mode.activate()
        
        # Update state
        input_mode = {
            "push_to_talk": InputMode.PUSH_TO_TALK,
            "always_on": InputMode.ALWAYS_ON,
            "text": InputMode.TEXT,
            "turn_based": InputMode.TURN_BASED
        }.get(mode_name, InputMode.TEXT)
        
        self.state_manager.update_state({"input_mode": input_mode})
        
        emit_event(Event(
            type=EventType.MODE_CHANGE,
            source="VoxTerminal",
            data={"mode": mode_name}
        ))
    
    def _setup_event_handlers(self):
        """Set up event handlers"""
        # Control events
        self.event_bus.subscribe(EventType.CONTROL, self._on_control_event)
        
        # Text events
        self.text_input_handler.on_message(self._on_text_message)
        self.text_input_handler.on_command(self._on_text_command)
        
        # Keyboard events
        self.keyboard_manager.on_key_action("quit", lambda _: self._shutdown_event.set())
        self.keyboard_manager.on_key_action("clear", lambda _: self.send_command({"type": "clear"}))
        self.keyboard_manager.on_key_action("mute_toggle", lambda _: self._toggle_mute())
        self.keyboard_manager.on_key_action("toggle_logs", lambda _: self._toggle_logs())
        
        # Push-to-talk
        self.keyboard_manager.on_push_to_talk_start(self._on_ptt_start)
        self.keyboard_manager.on_push_to_talk_end(self._on_ptt_end)
    
    def _setup_voice_engine_callbacks(self):
        """Set up voice engine callbacks if supported"""
        if not self.voice_engine:
            return
        
        # Text responses
        if hasattr(self.voice_engine, 'on_text_response'):
            self.voice_engine.on_text_response = self._on_ai_text
        
        # Audio responses
        if hasattr(self.voice_engine, 'on_audio_response'):
            self.voice_engine.on_audio_response = self._on_ai_audio
        
        # Transcripts
        if hasattr(self.voice_engine, 'on_user_transcript'):
            self.voice_engine.on_user_transcript = self._on_user_transcript
        
        # Response completion
        if hasattr(self.voice_engine, 'on_response_done'):
            self.voice_engine.on_response_done = self._on_response_done
        
        # Errors
        if hasattr(self.voice_engine, 'on_error'):
            self.voice_engine.on_error = self._on_engine_error
    
    def _on_control_event(self, event: Event):
        """Handle control events"""
        command = event.data.get("command")
        
        if command == "switch_mode":
            mode = event.data.get("args", {}).get("mode")
            if mode:
                self.send_command({"type": "switch_mode", "mode": mode})
    
    def _on_text_message(self, message: str):
        """Handle text message from user"""
        # Add to conversation
        self.state_manager.add_message("user", message)
        
        # Send to voice engine if available
        if self.voice_engine and hasattr(self.voice_engine, 'send_text'):
            asyncio.create_task(self.voice_engine.send_text(message))
    
    def _on_text_command(self, command):
        """Handle text command"""
        # Built-in commands are handled by text input handler
        # We just update display
        self.display.render({"section": "conversation"})
    
    def _on_ptt_start(self):
        """Handle push-to-talk start"""
        # Mode handles the actual recording
        pass
    
    def _on_ptt_end(self):
        """Handle push-to-talk end"""
        # Mode handles the actual recording
        pass
    
    def _on_ai_text(self, text: str):
        """Handle AI text response"""
        # Update partial response
        state = self.state_manager.get_state()
        state.conversation.partial_assistant_response += text
        
        # Trigger display update
        self.display.render({"section": "conversation"})
    
    def _on_ai_audio(self, audio: bytes):
        """Handle AI audio response"""
        # Play audio
        self.audio_output_manager.play_audio(audio)
    
    def _on_user_transcript(self, transcript: str):
        """Handle user transcript"""
        # Update partial transcript
        state = self.state_manager.get_state()
        state.conversation.partial_user_transcript = transcript
        
        # Trigger display update
        self.display.render({"section": "conversation"})
    
    def _on_response_done(self):
        """Handle response completion"""
        state = self.state_manager.get_state()
        
        # Move partial response to complete message
        if state.conversation.partial_assistant_response:
            self.state_manager.add_message(
                "assistant",
                state.conversation.partial_assistant_response
            )
            state.conversation.partial_assistant_response = ""
        
        # Clear partial user transcript
        if state.conversation.partial_user_transcript:
            self.state_manager.add_message(
                "user",
                state.conversation.partial_user_transcript
            )
            state.conversation.partial_user_transcript = ""
        
        # Notify current mode
        if self.current_mode and hasattr(self.current_mode, 'on_response_complete'):
            self.current_mode.on_response_complete()
        
        # Update display
        self.display.render({"section": "conversation"})
    
    def _on_engine_error(self, error: Exception):
        """Handle voice engine error"""
        emit_event(Event(
            type=EventType.ERROR,
            source="VoiceEngine",
            data={"error": str(error)}
        ))
    
    def _toggle_mute(self):
        """Toggle mute state"""
        current_mute = self.state_manager.get_state().audio.is_muted
        self.audio_input_manager.set_muted(not current_mute)
    
    def _toggle_logs(self):
        """Toggle log display"""
        current_show = self.config_manager.get("display.show_logs", False)
        self.config_manager.set("display.show_logs", not current_show)
        self.display.render({"section": "full"})
    
    def _handle_config_command(self, command: Dict[str, Any]):
        """Handle configuration command"""
        path = command.get("path")
        value = command.get("value")
        
        if path and value is not None:
            success = self.config_manager.set(path, value)
            if success:
                emit_event(Event(
                    type=EventType.INFO,
                    source="VoxTerminal",
                    data={"message": f"Config updated: {path} = {value}"}
                ))
            else:
                emit_event(Event(
                    type=EventType.WARNING,
                    source="VoxTerminal",
                    data={"message": f"Failed to update config: {path}"}
                ))
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Public API
    
    def on_ready(self, callback: Callable):
        """Set callback for when terminal is ready"""
        self._on_ready_callback = callback
    
    def on_shutdown(self, callback: Callable):
        """Set callback for shutdown"""
        self._on_shutdown_callback = callback
    
    def display_message(self, message: Dict[str, Any]) -> None:
        """Display a message (implements base class method)"""
        role = message.get("role", "system")
        content = message.get("content", "")
        
        self.state_manager.add_message(role, content)
        self.display.render({"section": "conversation"})
    
    def get_mode(self) -> str:
        """Get current mode name"""
        if self.current_mode:
            return self.current_mode.mode_type.value
        return "unknown"
    
    def set_mode(self, mode: str):
        """Set interaction mode"""
        self.send_command({"type": "switch_mode", "mode": mode})
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config_manager.get(path, default)
    
    def set_config(self, path: str, value: Any) -> bool:
        """Set configuration value"""
        return self.config_manager.set(path, value)