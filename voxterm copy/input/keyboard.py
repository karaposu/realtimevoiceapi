"""
VoxTerm Keyboard Input Handling

Manages keyboard input with support for multiple backends and modes.
"""

from typing import Optional, Dict, Callable, Set, Any
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio
import time
from abc import ABC, abstractmethod

from ..core.base import BaseComponent, ComponentState
from ..core.events import Event, EventType, emit_event, ControlEvent
from ..config.manager import get_config


class KeyboardBackend(Enum):
    """Available keyboard backends"""
    PYNPUT = "pynput"
    KEYBOARD = "keyboard"
    CURSES = "curses"
    MANUAL = "manual"  # For testing


@dataclass
class KeyEvent:
    """Keyboard event"""
    key: str  # Key name/code
    action: str  # "press" or "release"
    modifiers: Set[str] = None  # {"ctrl", "alt", "shift", "meta"}
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = set()
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def matches(self, binding: str) -> bool:
        """Check if this key event matches a binding string"""
        # Parse binding like "ctrl+c" or "space"
        parts = binding.lower().split('+')
        
        if len(parts) == 1:
            # Simple key
            return self.key.lower() == parts[0] and not self.modifiers
        else:
            # Key with modifiers
            binding_mods = set(parts[:-1])
            binding_key = parts[-1]
            return (self.key.lower() == binding_key and 
                    self.modifiers == binding_mods)


class KeyboardHandler(ABC):
    """Abstract base for keyboard handlers"""
    
    @abstractmethod
    def start(self) -> None:
        """Start capturing keyboard input"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop capturing keyboard input"""
        pass
    
    @abstractmethod
    def set_callback(self, callback: Callable[[KeyEvent], None]) -> None:
        """Set the callback for key events"""
        pass


class PynputHandler(KeyboardHandler):
    """Keyboard handler using pynput library"""
    
    def __init__(self):
        self.callback: Optional[Callable] = None
        self.listener = None
        self._pressed_keys = set()
        
        try:
            from pynput import keyboard
            self.keyboard = keyboard
        except ImportError:
            raise ImportError("pynput not installed. Run: pip install pynput")
    
    def start(self) -> None:
        """Start keyboard listener"""
        self.listener = self.keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
    
    def stop(self) -> None:
        """Stop keyboard listener"""
        if self.listener:
            self.listener.stop()
    
    def set_callback(self, callback: Callable[[KeyEvent], None]) -> None:
        """Set the callback"""
        self.callback = callback
    
    def _on_press(self, key):
        """Handle key press"""
        if not self.callback:
            return
        
        # Get key name and modifiers
        key_name, modifiers = self._parse_key(key)
        
        # Avoid duplicate press events
        if key_name not in self._pressed_keys:
            self._pressed_keys.add(key_name)
            event = KeyEvent(
                key=key_name,
                action="press",
                modifiers=modifiers
            )
            self.callback(event)
    
    def _on_release(self, key):
        """Handle key release"""
        if not self.callback:
            return
        
        # Get key name and modifiers
        key_name, modifiers = self._parse_key(key)
        
        # Remove from pressed keys
        self._pressed_keys.discard(key_name)
        
        event = KeyEvent(
            key=key_name,
            action="release",
            modifiers=modifiers
        )
        self.callback(event)
    
    def _parse_key(self, key):
        """Parse pynput key object"""
        modifiers = set()
        
        # Check for modifiers
        try:
            if self.keyboard.Controller().ctrl_pressed:
                modifiers.add("ctrl")
            if self.keyboard.Controller().alt_pressed:
                modifiers.add("alt")
            if self.keyboard.Controller().shift_pressed:
                modifiers.add("shift")
        except:
            pass
        
        # Get key name
        if hasattr(key, 'char') and key.char:
            key_name = key.char
        elif hasattr(key, 'name'):
            key_name = key.name
        else:
            key_name = str(key)
        
        return key_name, modifiers


class KeyboardManager(BaseComponent):
    """
    Main keyboard input manager for VoxTerm.
    
    Features:
    - Multiple backend support
    - Key binding management
    - Mode-specific handlers
    - Push-to-talk support
    """
    
    def __init__(self, backend: KeyboardBackend = KeyboardBackend.PYNPUT):
        super().__init__("KeyboardManager")
        
        self.backend = backend
        self.handler: Optional[KeyboardHandler] = None
        
        # State
        self._current_mode = "normal"
        self._push_to_talk_active = False
        self._push_to_talk_start_time: Optional[float] = None
        
        # Bindings cache
        self._bindings: Dict[str, str] = {}
        self._mode_handlers: Dict[str, Callable] = {}
        
        # Callbacks
        self._key_callbacks: Dict[str, Callable] = {}
        self._push_to_talk_callbacks = {
            "start": None,
            "end": None
        }
    
    async def initialize(self) -> None:
        """Initialize keyboard manager"""
        # Create handler based on backend
        if self.backend == KeyboardBackend.PYNPUT:
            self.handler = PynputHandler()
        else:
            # Add other backends as needed
            raise NotImplementedError(f"Backend {self.backend} not implemented")
        
        # Set our callback
        self.handler.set_callback(self._handle_key_event)
        
        # Load key bindings from config
        self._load_bindings()
        
        self.set_state(ComponentState.READY)
    
    async def start(self) -> None:
        """Start keyboard capture"""
        if self.handler:
            self.handler.start()
        self.set_state(ComponentState.RUNNING)
        
        emit_event(Event(
            type=EventType.INFO,
            source=self.name,
            data={"message": "Keyboard input started"}
        ))
    
    async def stop(self) -> None:
        """Stop keyboard capture"""
        if self.handler:
            self.handler.stop()
        self.set_state(ComponentState.STOPPED)
    
    def _load_bindings(self):
        """Load key bindings from configuration"""
        from ..config.manager import get_config
        
        # Get all key bindings
        bindings = get_config("key_bindings", {})
        if hasattr(bindings, 'to_dict'):
            self._bindings = bindings.to_dict()
        else:
            self._bindings = bindings
    
    def _handle_key_event(self, event: KeyEvent):
        """Handle a keyboard event"""
        # Check for push-to-talk
        ptt_binding = self._bindings.get("push_to_talk", "space")
        if event.matches(ptt_binding):
            self._handle_push_to_talk(event)
            return
        
        # Only process key press events for other bindings
        if event.action != "press":
            return
        
        # Check mode-specific handler first
        if self._current_mode in self._mode_handlers:
            if self._mode_handlers[self._current_mode](event):
                return  # Handled by mode
        
        # Check global bindings
        for action, binding in self._bindings.items():
            if event.matches(binding):
                self._handle_action(action, event)
                break
    
    def _handle_push_to_talk(self, event: KeyEvent):
        """Handle push-to-talk key"""
        if event.action == "press" and not self._push_to_talk_active:
            # Start push-to-talk
            self._push_to_talk_active = True
            self._push_to_talk_start_time = time.time()
            
            emit_event(ControlEvent(
                type=EventType.AUDIO_INPUT_START,
                source=self.name,
                command="push_to_talk_start"
            ))
            
            if self._push_to_talk_callbacks["start"]:
                # Run callback in thread to avoid blocking
                threading.Thread(
                    target=self._push_to_talk_callbacks["start"],
                    daemon=True
                ).start()
        
        elif event.action == "release" and self._push_to_talk_active:
            # End push-to-talk
            self._push_to_talk_active = False
            
            duration = time.time() - self._push_to_talk_start_time
            
            emit_event(ControlEvent(
                type=EventType.AUDIO_INPUT_END,
                source=self.name,
                command="push_to_talk_end",
                args={"duration": duration}
            ))
            
            if self._push_to_talk_callbacks["end"]:
                # Run callback in thread
                threading.Thread(
                    target=self._push_to_talk_callbacks["end"],
                    daemon=True
                ).start()
    
    def _handle_action(self, action: str, event: KeyEvent):
        """Handle a bound action"""
        # Emit control event
        emit_event(ControlEvent(
            type=EventType.CONTROL,
            source=self.name,
            command=action,
            args={"key": event.key, "modifiers": list(event.modifiers)}
        ))
        
        # Call specific callback if registered
        if action in self._key_callbacks:
            callback = self._key_callbacks[action]
            # Run in thread to avoid blocking
            threading.Thread(
                target=callback,
                args=(event,),
                daemon=True
            ).start()
    
    def set_mode(self, mode: str):
        """Set the current input mode"""
        self._current_mode = mode
        
        emit_event(Event(
            type=EventType.MODE_CHANGE,
            source=self.name,
            data={"mode": mode}
        ))
    
    def register_mode_handler(self, mode: str, handler: Callable[[KeyEvent], bool]):
        """
        Register a mode-specific key handler.
        
        Handler should return True if it handled the event.
        """
        self._mode_handlers[mode] = handler
    
    def on_push_to_talk_start(self, callback: Callable):
        """Set callback for push-to-talk start"""
        self._push_to_talk_callbacks["start"] = callback
    
    def on_push_to_talk_end(self, callback: Callable):
        """Set callback for push-to-talk end"""
        self._push_to_talk_callbacks["end"] = callback
    
    def on_key_action(self, action: str, callback: Callable[[KeyEvent], None]):
        """Set callback for a specific key action"""
        self._key_callbacks[action] = callback
    
    def update_binding(self, action: str, binding: str):
        """Update a key binding at runtime"""
        self._bindings[action] = binding
        
        # Update config
        from ..config.manager import set_config
        set_config(f"key_bindings.{action}", binding)
    
    @property
    def is_push_to_talk_active(self) -> bool:
        """Check if push-to-talk is active"""
        return self._push_to_talk_active
    
    def get_bindings(self) -> Dict[str, str]:
        """Get current key bindings"""
        return self._bindings.copy()