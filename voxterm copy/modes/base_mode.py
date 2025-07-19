"""
VoxTerm Base Interaction Mode

Abstract base class for all interaction modes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..core.base import BaseComponent, ComponentState
from ..core.events import Event, EventType, emit_event
from ..input.keyboard import KeyEvent


class ModeType(Enum):
    """Types of interaction modes"""
    TEXT = "text"
    PUSH_TO_TALK = "push_to_talk"
    ALWAYS_ON = "always_on"
    TURN_BASED = "turn_based"


@dataclass
class ModeConfig:
    """Configuration for an interaction mode"""
    name: str
    type: ModeType
    description: str
    keybindings: Dict[str, str]
    settings: Dict[str, Any]


class BaseMode(BaseComponent, ABC):
    """
    Abstract base class for interaction modes.
    
    Each mode defines how the user interacts with the voice system.
    """
    
    def __init__(self, name: str, mode_type: ModeType):
        super().__init__(f"Mode_{name}")
        self.mode_type = mode_type
        self.is_active = False
        
        # Components references
        self.keyboard_manager = None
        self.audio_input_manager = None
        self.text_input_handler = None
        self.voice_engine = None
        
        # Mode-specific state
        self._state: Dict[str, Any] = {}
        
        # Callbacks
        self._mode_changed_callback: Optional[Callable] = None
    
    @abstractmethod
    async def activate(self) -> None:
        """Activate this mode"""
        pass
    
    @abstractmethod
    async def deactivate(self) -> None:
        """Deactivate this mode"""
        pass
    
    @abstractmethod
    def handle_key_event(self, event: KeyEvent) -> bool:
        """
        Handle a keyboard event.
        
        Returns True if the event was handled.
        """
        pass
    
    @abstractmethod
    def get_status_text(self) -> str:
        """Get status text for this mode"""
        pass
    
    @abstractmethod
    def get_help_text(self) -> str:
        """Get help text for this mode"""
        pass
    
    async def initialize(self) -> None:
        """Initialize the mode"""
        self.set_state(ComponentState.READY)
    
    async def start(self) -> None:
        """Start the mode"""
        self.set_state(ComponentState.RUNNING)
    
    async def stop(self) -> None:
        """Stop the mode"""
        if self.is_active:
            await self.deactivate()
        self.set_state(ComponentState.STOPPED)
    
    async def handle_command(self, command: Dict[str, Any]) -> None:
        """
        Handle a command sent to this mode.
        
        Default implementation does nothing.
        Override in subclasses if needed.
        """
        pass
    
    def bind_components(
        self,
        keyboard_manager=None,
        audio_input_manager=None,
        text_input_handler=None,
        voice_engine=None
    ):
        """Bind required components"""
        if keyboard_manager:
            self.keyboard_manager = keyboard_manager
        if audio_input_manager:
            self.audio_input_manager = audio_input_manager
        if text_input_handler:
            self.text_input_handler = text_input_handler
        if voice_engine:
            self.voice_engine = voice_engine
    
    def set_active(self, active: bool):
        """Set mode active state"""
        self.is_active = active
        
        emit_event(Event(
            type=EventType.MODE_CHANGE,
            source=self.name,
            data={
                "mode": self.mode_type.value,
                "active": active
            }
        ))
    
    def update_state(self, key: str, value: Any):
        """Update mode-specific state"""
        self._state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get mode-specific state"""
        return self._state.get(key, default)
    
    def on_mode_changed(self, callback: Callable):
        """Set callback for mode changes"""
        self._mode_changed_callback = callback
    
    def emit_mode_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a mode-specific event"""
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={
                "mode_event": event_type,
                "mode": self.mode_type.value,
                **data
            }
        ))