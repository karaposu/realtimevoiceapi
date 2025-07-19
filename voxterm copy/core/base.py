"""
VoxTerm Core Base Classes

Provides abstract base classes and interfaces for the terminal UI framework.
All components inherit from these to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import asyncio
from enum import Enum
import threading
import queue


class ComponentState(Enum):
    """Component lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"  # Added this
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseComponent(ABC):
    """
    Base class for all VoxTerm components.
    
    Provides common functionality for lifecycle management,
    event handling, and thread-safe operation.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.UNINITIALIZED
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state_lock = threading.RLock()
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the component"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the component"""
        pass
    
    def set_state(self, state: ComponentState) -> None:
        """Thread-safe state setting"""
        with self._state_lock:
            old_state = self.state
            self.state = state
            self.emit("state_changed", old_state=old_state, new_state=state)
    
    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable) -> None:
        """Unregister an event handler"""
        if event in self._event_handlers:
            self._event_handlers[event].remove(handler)
    
    def emit(self, event: str, **kwargs) -> None:
        """Emit an event (non-blocking)"""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                # Run handlers in thread pool to avoid blocking
                threading.Thread(
                    target=handler,
                    kwargs=kwargs,
                    daemon=True
                ).start()


class InputHandler(ABC):
    """Abstract base for input handlers"""
    
    @abstractmethod
    async def handle_input(self, input_data: Any) -> None:
        """Process input data"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Return list of supported input types"""
        pass


class OutputRenderer(ABC):
    """Abstract base for output renderers"""
    
    @abstractmethod
    async def render(self, data: Any) -> None:
        """Render output to terminal"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the output"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Return renderer capabilities"""
        pass


class BaseTerminal(BaseComponent):
    """
    Abstract base class for terminal implementations.
    
    Provides the core structure for a VoxTerm terminal interface.
    """
    
    def __init__(self, title: str = "VoxTerm"):
        super().__init__("terminal")
        self.title = title
        self.input_handlers: List[InputHandler] = []
        self.output_renderer: Optional[OutputRenderer] = None
        self._message_queue: queue.Queue = queue.Queue()
        self._command_queue: queue.Queue = queue.Queue()
        
    @abstractmethod
    async def run(self) -> None:
        """Run the terminal UI"""
        pass
    
    @abstractmethod
    def bind_voice_engine(self, engine: Any) -> None:
        """Bind a voice engine to the terminal"""
        pass
    
    def add_input_handler(self, handler: InputHandler) -> None:
        """Add an input handler"""
        self.input_handlers.append(handler)
    
    def set_output_renderer(self, renderer: OutputRenderer) -> None:
        """Set the output renderer"""
        self.output_renderer = renderer
    
    def display_message(self, message: Dict[str, Any]) -> None:
        """Queue a message for display (non-blocking)"""
        try:
            self._message_queue.put_nowait(message)
        except queue.Full:
            # Drop old messages if queue is full
            try:
                self._message_queue.get_nowait()
                self._message_queue.put_nowait(message)
            except queue.Empty:
                pass
    
    def send_command(self, command: Dict[str, Any]) -> None:
        """Queue a command (non-blocking)"""
        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            pass  # Drop commands if queue is full


class VoiceEngineAdapter(ABC):
    """
    Abstract adapter for integrating voice engines with VoxTerm.
    
    Implementations handle the specifics of different voice APIs
    while providing a consistent interface to VoxTerm.
    """
    
    @abstractmethod
    def connect_to_terminal(self, terminal: BaseTerminal, engine: Any) -> None:
        """Connect a voice engine to the terminal"""
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the voice engine"""
        pass
    
    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """Check if the engine supports a feature"""
        pass