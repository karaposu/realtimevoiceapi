"""
VoxTerm Text Input Handling

Manages text-based input including commands and messages.
"""

from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from queue import Queue

from ..core.base import BaseComponent, ComponentState, InputHandler
from ..core.events import Event, EventType, emit_event, TextEvent, ControlEvent


class TextInputType(Enum):
    """Types of text input"""
    MESSAGE = "message"      # Regular conversation message
    COMMAND = "command"      # System command (starts with /)
    SEARCH = "search"        # Search in conversation
    CONFIG = "config"        # Configuration command


@dataclass
class TextCommand:
    """Parsed text command"""
    command: str
    args: List[str]
    raw: str


class TextInputHandler(InputHandler, BaseComponent):
    """
    Handles text input from the terminal.
    
    Features:
    - Command parsing (commands start with /)
    - Input history
    - Auto-completion
    - Multi-line support
    """
    
    def __init__(self):
        BaseComponent.__init__(self, "TextInputHandler")
        
        # Input state
        self._current_input = ""
        self._input_history: List[str] = []
        self._history_index = -1
        self._multiline_mode = False
        self._multiline_buffer: List[str] = []
        
        # Input queue for async processing
        self._input_queue: Queue = Queue()
        
        # Callbacks
        self._message_callback: Optional[Callable[[str], None]] = None
        self._command_callback: Optional[Callable[[TextCommand], None]] = None
        
        # Commands registry
        self._commands: Dict[str, Dict[str, Any]] = self._register_default_commands()
        
        # Auto-completion
        self._completion_candidates: List[str] = []
        self._completion_index = 0
    
    async def initialize(self) -> None:
        """Initialize text input handler"""
        self.set_state(ComponentState.READY)
    
    async def start(self) -> None:
        """Start text input processing"""
        self.set_state(ComponentState.RUNNING)
        # Start input processing task
        asyncio.create_task(self._process_input_queue())
    
    async def stop(self) -> None:
        """Stop text input processing"""
        self.set_state(ComponentState.STOPPED)
    
    def get_supported_types(self) -> List[str]:
        """Return supported input types"""
        return ["text", "command"]
    
    async def handle_input(self, input_data: Any) -> None:
        """Process input data"""
        if isinstance(input_data, str):
            self._input_queue.put(input_data)
    
    def handle_char(self, char: str) -> None:
        """Handle a single character input"""
        if char == '\n' or char == '\r':
            self._handle_enter()
        elif char == '\b' or char == '\x7f':  # Backspace
            self._handle_backspace()
        elif char == '\t':  # Tab
            self._handle_tab()
        elif char == '\x1b[A':  # Up arrow
            self._handle_history_up()
        elif char == '\x1b[B':  # Down arrow
            self._handle_history_down()
        else:
            # Regular character
            self._current_input += char
            self._update_display()
    
    def _handle_enter(self):
        """Handle enter key"""
        if self._multiline_mode:
            # Add to multiline buffer
            self._multiline_buffer.append(self._current_input)
            self._current_input = ""
            
            # Check for end of multiline (empty line)
            if not self._current_input and len(self._multiline_buffer) > 0:
                # Process multiline input
                full_input = '\n'.join(self._multiline_buffer)
                self._process_input(full_input)
                self._multiline_buffer.clear()
                self._multiline_mode = False
        else:
            # Single line input
            if self._current_input.strip():
                self._process_input(self._current_input)
                self._add_to_history(self._current_input)
            
            self._current_input = ""
            self._history_index = -1
        
        self._update_display()
    
    def _handle_backspace(self):
        """Handle backspace key"""
        if self._current_input:
            self._current_input = self._current_input[:-1]
            self._update_display()
    
    def _handle_tab(self):
        """Handle tab key for auto-completion"""
        if not self._current_input:
            return
        
        # Get completion candidates
        if not self._completion_candidates:
            self._completion_candidates = self._get_completions(self._current_input)
            self._completion_index = 0
        
        if self._completion_candidates:
            # Cycle through completions
            completion = self._completion_candidates[self._completion_index]
            self._current_input = completion
            self._completion_index = (self._completion_index + 1) % len(self._completion_candidates)
            self._update_display()
    
    def _handle_history_up(self):
        """Navigate up in history"""
        if self._input_history and self._history_index < len(self._input_history) - 1:
            self._history_index += 1
            self._current_input = self._input_history[-(self._history_index + 1)]
            self._update_display()
    
    def _handle_history_down(self):
        """Navigate down in history"""
        if self._history_index > 0:
            self._history_index -= 1
            self._current_input = self._input_history[-(self._history_index + 1)]
        elif self._history_index == 0:
            self._history_index = -1
            self._current_input = ""
        
        self._update_display()
    
    def _process_input(self, input_text: str):
        """Process complete input"""
        input_text = input_text.strip()
        
        if not input_text:
            return
        
        # Check if it's a command
        if input_text.startswith('/'):
            self._process_command(input_text)
        else:
            self._process_message(input_text)
    
    def _process_command(self, input_text: str):
        """Process a command"""
        # Parse command
        parts = input_text[1:].split()
        if not parts:
            return
        
        command = TextCommand(
            command=parts[0],
            args=parts[1:],
            raw=input_text
        )
        
        # Emit command event
        emit_event(ControlEvent(
            type=EventType.CONTROL,
            source=self.name,
            command=command.command,
            args={"args": command.args}
        ))
        
        # Call callback
        if self._command_callback:
            threading.Thread(
                target=self._command_callback,
                args=(command,),
                daemon=True
            ).start()
        
        # Handle built-in commands
        if command.command in self._commands:
            handler = self._commands[command.command]["handler"]
            handler(command)
    
    def _process_message(self, message: str):
        """Process a regular message"""
        # Emit text event
        emit_event(TextEvent(
            type=EventType.TEXT_INPUT,
            source=self.name,
            text=message
        ))
        
        # Call callback
        if self._message_callback:
            threading.Thread(
                target=self._message_callback,
                args=(message,),
                daemon=True
            ).start()
    
    async def _process_input_queue(self):
        """Process queued inputs asynchronously"""
        while self.state == ComponentState.RUNNING:
            try:
                # Non-blocking get with timeout
                input_text = await asyncio.get_event_loop().run_in_executor(
                    None, self._input_queue.get, True, 0.1
                )
                self._process_input(input_text)
            except:
                await asyncio.sleep(0.1)
    
    def _add_to_history(self, text: str):
        """Add input to history"""
        if text and (not self._input_history or self._input_history[-1] != text):
            self._input_history.append(text)
            # Limit history size
            if len(self._input_history) > 100:
                self._input_history.pop(0)
    
    def _get_completions(self, partial: str) -> List[str]:
        """Get auto-completion candidates"""
        completions = []
        
        # Command completions
        if partial.startswith('/'):
            prefix = partial[1:]
            for cmd in self._commands:
                if cmd.startswith(prefix):
                    completions.append('/' + cmd)
        
        # Add other completion sources (e.g., usernames, common phrases)
        
        return completions
    
    def _update_display(self):
        """Update the input display"""
        # This would update the terminal display
        # In VoxTerm, this emits an event that the display component handles
        emit_event(Event(
            type=EventType.UI_REFRESH,
            source=self.name,
            data={"input": self._current_input}
        ))
    
    def _register_default_commands(self) -> Dict[str, Dict[str, Any]]:
        """Register default commands"""
        return {
            "help": {
                "description": "Show help",
                "handler": self._cmd_help
            },
            "clear": {
                "description": "Clear conversation",
                "handler": self._cmd_clear
            },
            "voice": {
                "description": "Change voice",
                "handler": self._cmd_voice
            },
            "mode": {
                "description": "Change input mode",
                "handler": self._cmd_mode
            },
            "mute": {
                "description": "Toggle mute",
                "handler": self._cmd_mute
            },
            "history": {
                "description": "Show input history",
                "handler": self._cmd_history
            },
            "multiline": {
                "description": "Toggle multiline mode",
                "handler": self._cmd_multiline
            }
        }
    
    def _cmd_help(self, cmd: TextCommand):
        """Show help"""
        help_text = "Available commands:\n"
        for command, info in self._commands.items():
            help_text += f"  /{command} - {info['description']}\n"
        
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={"message": help_text}
        ))
    
    def _cmd_clear(self, cmd: TextCommand):
        """Clear conversation"""
        from ..core.state import get_state_manager
        get_state_manager().get_state().conversation.clear()
        
        emit_event(Event(
            type=EventType.UI_REFRESH,
            source=self.name,
            data={"action": "clear"}
        ))
    
    def _cmd_voice(self, cmd: TextCommand):
        """Change voice"""
        if cmd.args:
            voice = cmd.args[0]
            from ..config.manager import set_config
            set_config("voice.current_voice", voice)
            
            emit_event(Event(
                type=EventType.INFO,
                source=self.name,
                data={"message": f"Voice changed to: {voice}"}
            ))
    
    def _cmd_mode(self, cmd: TextCommand):
        """Change input mode"""
        if cmd.args:
            mode = cmd.args[0]
            emit_event(Event(
                type=EventType.MODE_CHANGE,
                source=self.name,
                data={"mode": mode}
            ))
    
    def _cmd_mute(self, cmd: TextCommand):
        """Toggle mute"""
        emit_event(Event(
            type=EventType.MUTE_TOGGLE,
            source=self.name
        ))
    
    def _cmd_history(self, cmd: TextCommand):
        """Show input history"""
        history = "\n".join(self._input_history[-10:])
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={"message": f"Recent history:\n{history}"}
        ))
    
    def _cmd_multiline(self, cmd: TextCommand):
        """Toggle multiline mode"""
        self._multiline_mode = not self._multiline_mode
        status = "enabled" if self._multiline_mode else "disabled"
        
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={"message": f"Multiline mode {status}"}
        ))
    
    # Callbacks
    def on_message(self, callback: Callable[[str], None]):
        """Set callback for messages"""
        self._message_callback = callback
    
    def on_command(self, callback: Callable[[TextCommand], None]):
        """Set callback for commands"""
        self._command_callback = callback
    
    def register_command(self, name: str, description: str, handler: Callable):
        """Register a custom command"""
        self._commands[name] = {
            "description": description,
            "handler": handler
        }
    
    def get_current_input(self) -> str:
        """Get current input buffer"""
        return self._current_input
    
    def set_current_input(self, text: str):
        """Set current input buffer (for programmatic input)"""
        self._current_input = text
        self._update_display()
    
    def clear_input(self):
        """Clear current input"""
        self._current_input = ""
        self._completion_candidates.clear()
        self._update_display()