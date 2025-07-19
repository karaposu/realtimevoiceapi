"""
VoxTerm Terminal Display Manager

Manages the terminal display with non-blocking updates.
"""

import asyncio
import threading
import queue
import time
import os
import sys
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from ..core.base import BaseComponent, ComponentState, OutputRenderer
from ..core.events import Event, EventType, subscribe, get_event_bus
from ..core.state import get_state_manager, TerminalState, Message
from ..config.manager import get_config


class DisplaySection(Enum):
    """Terminal display sections"""
    STATUS_BAR = "status_bar"
    CONVERSATION = "conversation"
    INPUT_LINE = "input_line"
    CONTROL_HINTS = "control_hints"
    LOGS = "logs"


@dataclass
class DisplayUpdate:
    """Display update request"""
    section: DisplaySection
    content: Any
    priority: int = 0
    timestamp: float = 0.0


class TerminalDisplay(OutputRenderer, BaseComponent):
    """
    Terminal display manager for VoxTerm.
    
    Features:
    - Non-blocking display updates
    - Section-based rendering
    - Smooth scrolling
    - Color support
    - Responsive layout
    """
    
    def __init__(self):
        BaseComponent.__init__(self, "TerminalDisplay")
        
        # Display state
        self._terminal_size = self._get_terminal_size()
        self._display_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._display_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Content buffers
        self._conversation_buffer: List[Message] = []
        self._log_buffer: List[str] = []
        self._current_input = ""
        
        # Display configuration
        self._show_timestamps = True
        self._show_status_bar = True
        self._show_control_hints = True
        self._show_logs = False
        
        # Update control
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / 30  # 30 FPS max
        
        # ANSI color codes
        self._colors = self._load_colors()
    
    async def initialize(self) -> None:
        """Initialize display manager"""
        # Subscribe to events
        self._subscribe_to_events()
        
        # Load display settings
        self._load_settings()
        
        self.set_state(ComponentState.READY)
    
    async def start(self) -> None:
        """Start display manager"""
        self._running = True
        
        # Start display thread
        self._display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="VoxTerm-Display"
        )
        self._display_thread.start()
        
        # Initial render
        self._queue_update(DisplaySection.CONVERSATION, None, priority=10)
        
        self.set_state(ComponentState.RUNNING)
    
    async def stop(self) -> None:
        """Stop display manager"""
        self._running = False
        
        if self._display_thread:
            self._display_thread.join(timeout=1.0)
        
        self.set_state(ComponentState.STOPPED)
    
    def render(self, data: Any) -> None:
        """Queue a render update (synchronous, non-blocking)"""
        if isinstance(data, dict):
            section_name = data.get("section", "conversation")
            
            # Map string to enum
            section_map = {
                "full": DisplaySection.CONVERSATION,
                "conversation": DisplaySection.CONVERSATION,
                "status": DisplaySection.STATUS_BAR,
                "input": DisplaySection.INPUT_LINE,
                "hints": DisplaySection.CONTROL_HINTS,
                "logs": DisplaySection.LOGS
            }
            
            section = section_map.get(section_name, DisplaySection.CONVERSATION)
            content = data.get("content")
            priority = data.get("priority", 0)
            
            self._queue_update(section, content, priority)
    
    async def render_async(self, data: Any) -> None:
        """Async version of render for compatibility"""
        self.render(data)
    
    def clear(self) -> None:
        """Clear the terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get display capabilities"""
        return {
            "color": True,
            "unicode": True,
            "mouse": False,
            "responsive": True
        }
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        event_bus = get_event_bus()
        
        # UI events
        event_bus.subscribe(EventType.UI_REFRESH, self._on_ui_refresh)
        event_bus.subscribe(EventType.UI_RESIZE, self._on_ui_resize)
        
        # Message events
        event_bus.subscribe(EventType.TEXT_OUTPUT, self._on_text_output)
        event_bus.subscribe(EventType.TEXT_INPUT, self._on_text_input)
        
        # State events
        event_bus.subscribe(EventType.CONNECTION_STATE, self._on_connection_state)
        event_bus.subscribe(EventType.MODE_CHANGE, self._on_mode_change)
        
        # Log events
        event_bus.subscribe(EventType.INFO, self._on_log_event)
        event_bus.subscribe(EventType.ERROR, self._on_log_event)
    
    def _load_settings(self):
        """Load display settings from config"""
        self._show_timestamps = get_config("display.show_timestamps", True)
        self._show_status_bar = get_config("display.show_status_bar", True)
        self._show_control_hints = get_config("display.show_control_hints", True)
    
    def _load_colors(self) -> Dict[str, str]:
        """Load color codes based on theme"""
        theme = get_config("display.theme", "dark")
        
        if theme == "dark":
            return {
                "reset": "\033[0m",
                "bold": "\033[1m",
                "dim": "\033[2m",
                "user": "\033[36m",      # Cyan
                "assistant": "\033[32m",  # Green
                "system": "\033[33m",     # Yellow
                "error": "\033[31m",      # Red
                "info": "\033[34m",       # Blue
                "status": "\033[35m",     # Magenta
                "input": "\033[37m",      # White
            }
        else:
            # Light theme colors
            return {
                "reset": "\033[0m",
                "bold": "\033[1m",
                "dim": "\033[2m",
                "user": "\033[34m",      # Blue
                "assistant": "\033[32m",  # Green
                "system": "\033[33m",     # Yellow
                "error": "\033[91m",      # Light Red
                "info": "\033[36m",       # Cyan
                "status": "\033[35m",     # Magenta
                "input": "\033[30m",      # Black
            }
    
    def _queue_update(self, section: DisplaySection, content: Any, priority: int = 0):
        """Queue a display update"""
        update = DisplayUpdate(
            section=section,
            content=content,
            priority=priority,
            timestamp=time.time()
        )
        
        try:
            # Use negative priority for queue (higher priority first)
            self._display_queue.put_nowait((-priority, update.timestamp, update))
        except queue.Full:
            pass  # Drop updates if queue is full
    
    def _display_loop(self):
        """Main display loop (runs in separate thread)"""
        while self._running:
            try:
                # Get update with timeout
                _, _, update = self._display_queue.get(timeout=0.1)
                
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - self._last_update_time
                
                if time_since_last < self._min_update_interval:
                    time.sleep(self._min_update_interval - time_since_last)
                
                # Process update
                self._process_update(update)
                self._last_update_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but keep running
                import logging
                logging.error(f"Display error: {e}")
    
    def _process_update(self, update: DisplayUpdate):
        """Process a display update"""
        if update.section == DisplaySection.CONVERSATION:
            self._render_full_display()
        elif update.section == DisplaySection.STATUS_BAR:
            self._render_status_bar()
        elif update.section == DisplaySection.INPUT_LINE:
            self._render_input_line()
        elif update.section == DisplaySection.CONTROL_HINTS:
            self._render_control_hints()
        elif update.section == DisplaySection.LOGS:
            self._render_logs()
    
    def _render_full_display(self):
        """Render the full terminal display"""
        # Clear and reset cursor
        self.clear()
        
        # Get current terminal size
        self._terminal_size = self._get_terminal_size()
        height, width = self._terminal_size
        
        # Calculate section heights
        status_height = 1 if self._show_status_bar else 0
        hints_height = 1 if self._show_control_hints else 0
        input_height = 2  # Input line + separator
        
        if self._show_logs:
            log_height = min(10, height // 3)
            conversation_height = height - status_height - hints_height - input_height - log_height - 1
        else:
            log_height = 0
            conversation_height = height - status_height - hints_height - input_height
        
        # Render sections
        if self._show_status_bar:
            self._render_status_bar()
        
        self._render_conversation(conversation_height)
        
        if self._show_logs:
            self._render_separator()
            self._render_logs(log_height)
        
        self._render_separator()
        self._render_input_line()
        
        if self._show_control_hints:
            self._render_control_hints()
    
    def _render_status_bar(self):
        """Render the status bar"""
        state = get_state_manager().get_state()
        width = self._terminal_size[1]
        
        # Build status components
        components = []
        
        # Connection status
        conn_icon = "🟢" if state.connection_state.value == "connected" else "🔴"
        components.append(f"{conn_icon} {state.connection_state.value.title()}")
        
        # Mode
        mode = state.input_mode.value.replace("_", " ").title()
        components.append(f"Mode: {mode}")
        
        # Mic status
        mic_status = "OFF" if state.audio.is_muted else "ON"
        components.append(f"Mic: {mic_status}")
        
        # Latency
        if state.api_latency_ms:
            components.append(f"⚡ {state.api_latency_ms:.0f}ms")
        
        # Join components
        status_text = " │ ".join(components)
        
        # Pad and truncate to width
        status_text = status_text[:width-2].ljust(width-2)
        
        # Print with styling
        print(f"{self._colors['status']}┌{'─' * (width-2)}┐{self._colors['reset']}")
        print(f"{self._colors['status']}│{self._colors['reset']} {status_text} {self._colors['status']}│{self._colors['reset']}")
        print(f"{self._colors['status']}└{'─' * (width-2)}┘{self._colors['reset']}")
    
    def _render_conversation(self, max_lines: int):
        """Render the conversation area"""
        state = get_state_manager().get_state()
        messages = state.conversation.get_recent_messages(50)
        
        # Convert messages to display lines
        display_lines = []
        for msg in messages:
            lines = self._format_message(msg)
            display_lines.extend(lines)
        
        # Add partial messages if any
        if state.conversation.partial_user_transcript:
            lines = self._format_partial("You", state.conversation.partial_user_transcript)
            display_lines.extend(lines)
        
        if state.conversation.partial_assistant_response:
            lines = self._format_partial("AI", state.conversation.partial_assistant_response)
            display_lines.extend(lines)
        
        # Show last N lines that fit
        if len(display_lines) > max_lines:
            display_lines = display_lines[-max_lines:]
        
        # Render lines
        for line in display_lines:
            print(line)
        
        # Fill remaining space
        for _ in range(max_lines - len(display_lines)):
            print()
    
    def _render_input_line(self):
        """Render the input line"""
        width = self._terminal_size[1]
        
        # Get current input
        input_text = self._current_input
        prompt = "> "
        
        # Truncate if too long
        max_input_width = width - len(prompt) - 2
        if len(input_text) > max_input_width:
            input_text = "..." + input_text[-(max_input_width-3):]
        
        # Print input line
        print(f"{self._colors['input']}{prompt}{input_text}{self._colors['reset']}")
    
    def _render_control_hints(self):
        """Render control hints"""
        state = get_state_manager().get_state()
        width = self._terminal_size[1]
        
        # Get mode-specific hints
        hints = []
        
        if state.input_mode.value == "push_to_talk":
            hints.append("[SPACE] Hold to talk")
        elif state.input_mode.value == "always_on":
            hints.append("[P] Pause/Resume")
        elif state.input_mode.value == "text":
            hints.append("[ENTER] Send message")
        
        # Common hints
        hints.extend([
            "[M] Mute",
            "[L] Toggle logs",
            "[Q] Quit"
        ])
        
        # Join and truncate
        hint_text = " │ ".join(hints)[:width-2]
        
        # Print with dim style
        print(f"{self._colors['dim']}{hint_text}{self._colors['reset']}")
    
    def _render_logs(self, max_lines: int = 5):
        """Render log area"""
        # Show recent log entries
        recent_logs = self._log_buffer[-max_lines:]
        
        for log in recent_logs:
            print(f"{self._colors['dim']}{log[:self._terminal_size[1]-2]}{self._colors['reset']}")
        
        # Fill remaining lines
        for _ in range(max_lines - len(recent_logs)):
            print()
    
    def _render_separator(self):
        """Render a separator line"""
        width = self._terminal_size[1]
        print(f"{self._colors['dim']}{'─' * width}{self._colors['reset']}")
    
    def _format_message(self, message: Message) -> List[str]:
        """Format a message for display"""
        lines = []
        width = self._terminal_size[1]
        
        # Timestamp
        timestamp = ""
        if self._show_timestamps:
            dt = datetime.fromtimestamp(message.timestamp)
            timestamp = dt.strftime(get_config("display.timestamp_format", "%H:%M:%S"))
            timestamp = f"[{timestamp}] "
        
        # Role
        role_text = "You" if message.role == "user" else "AI"
        role_color = self._colors["user"] if message.role == "user" else self._colors["assistant"]
        
        # First line with role
        prefix = f"{timestamp}{role_color}{role_text}:{self._colors['reset']} "
        
        # Wrap message content
        content_lines = self._wrap_text(message.content, width - len(timestamp) - 5)
        
        # Add first line with prefix
        if content_lines:
            lines.append(prefix + content_lines[0])
            
            # Add remaining lines with indent
            indent = " " * (len(timestamp) + 5)
            for line in content_lines[1:]:
                lines.append(indent + line)
        
        # Add blank line after message
        lines.append("")
        
        return lines
    
    def _format_partial(self, role: str, content: str) -> List[str]:
        """Format a partial message"""
        lines = []
        width = self._terminal_size[1]
        
        # Role with indicator
        role_color = self._colors["user"] if role == "You" else self._colors["assistant"]
        prefix = f"{role_color}{role}:{self._colors['reset']} "
        
        # Add typing indicator
        if role == "AI":
            content += " ▌"
        else:
            content += " [Recording...]"
        
        # Wrap content
        content_lines = self._wrap_text(content, width - 5)
        
        # Format lines
        if content_lines:
            lines.append(prefix + content_lines[0])
            indent = " " * 5
            for line in content_lines[1:]:
                lines.append(indent + line)
        
        return lines
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width"""
        if not text:
            return []
        
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line:
                    current_line += " "
                current_line += word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _get_terminal_size(self) -> tuple:
        """Get terminal size (height, width)"""
        try:
            import shutil
            width, height = shutil.get_terminal_size()
            return (height, width)
        except:
            return (24, 80)  # Default size
    
    # Event handlers
    def _on_ui_refresh(self, event: Event):
        """Handle UI refresh event"""
        self._queue_update(DisplaySection.CONVERSATION, None)
    
    def _on_ui_resize(self, event: Event):
        """Handle terminal resize"""
        self._terminal_size = self._get_terminal_size()
        self._queue_update(DisplaySection.CONVERSATION, None, priority=10)
    
    def _on_text_output(self, event: Event):
        """Handle text output event"""
        # Update conversation buffer
        state = get_state_manager().get_state()
        self._conversation_buffer = state.conversation.messages
        
        self._queue_update(DisplaySection.CONVERSATION, None)
    
    def _on_text_input(self, event: Event):
        """Handle text input event"""
        if "input" in event.data:
            self._current_input = event.data["input"]
            self._queue_update(DisplaySection.INPUT_LINE, None)
    
    def _on_connection_state(self, event: Event):
        """Handle connection state change"""
        self._queue_update(DisplaySection.STATUS_BAR, None)
    
    def _on_mode_change(self, event: Event):
        """Handle mode change"""
        self._queue_update(DisplaySection.STATUS_BAR, None)
        self._queue_update(DisplaySection.CONTROL_HINTS, None)
    
    def _on_log_event(self, event: Event):
        """Handle log events"""
        if self._show_logs:
            log_text = f"{event.type.name}: {event.data.get('message', '')}"
            self._log_buffer.append(log_text)
            
            # Limit buffer size
            if len(self._log_buffer) > 100:
                self._log_buffer.pop(0)
            
            self._queue_update(DisplaySection.LOGS, None)