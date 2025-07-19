"""
VoxTerm State Management

Thread-safe state management for terminal and conversation state.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto
import threading
import time
from collections import deque


class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class RecordingState(Enum):
    """Recording states"""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class PlaybackState(Enum):
    """Playback states"""
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"


class InputMode(Enum):
    """Input modes"""
    TEXT = "text"
    PUSH_TO_TALK = "push_to_talk"
    ALWAYS_ON = "always_on"
    TURN_BASED = "turn_based"


@dataclass
class AudioState:
    """Audio-related state"""
    is_muted: bool = False
    input_volume: float = 1.0
    output_volume: float = 1.0
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    recording_state: RecordingState = RecordingState.IDLE
    playback_state: PlaybackState = PlaybackState.IDLE
    last_audio_level: float = 0.0


@dataclass
class Message:
    """A message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = True


@dataclass
class ConversationState:
    """Conversation state"""
    messages: List[Message] = field(default_factory=list)
    current_turn: str = "user"  # "user" or "assistant"
    is_assistant_speaking: bool = False
    partial_user_transcript: str = ""
    partial_assistant_response: str = ""
    
    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to the conversation"""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        return message
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages"""
        return list(self.messages[-count:])
    
    def clear(self):
        """Clear conversation history"""
        self.messages.clear()
        self.partial_user_transcript = ""
        self.partial_assistant_response = ""


@dataclass
class TerminalState:
    """Complete terminal state"""
    # Connection
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    api_latency_ms: Optional[float] = None
    
    # Audio
    audio: AudioState = field(default_factory=AudioState)
    
    # Input
    input_mode: InputMode = InputMode.TEXT
    
    # Conversation
    conversation: ConversationState = field(default_factory=ConversationState)
    
    # UI
    show_logs: bool = False
    show_metrics: bool = True
    terminal_width: int = 80
    terminal_height: int = 24
    
    # Metrics
    session_start_time: float = field(default_factory=time.time)
    total_interactions: int = 0
    total_tokens_used: int = 0
    
    # Voice settings
    voice_name: str = "alloy"
    language: str = "en"
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.session_start_time


class StateManager:
    """
    Thread-safe state manager for VoxTerm.
    
    Provides atomic updates and change notifications.
    """
    
    def __init__(self):
        self._state = TerminalState()
        self._lock = threading.RLock()
        self._observers: List[callable] = []
        self._state_history: deque = deque(maxlen=100)  # Keep last 100 states
        
    def get_state(self) -> TerminalState:
        """Get a copy of the current state"""
        with self._lock:
            # Return a copy to prevent external modification
            import copy
            return copy.deepcopy(self._state)
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update state with a dictionary of changes.
        
        Example:
            state_manager.update_state({
                'connection_state': ConnectionState.CONNECTED,
                'audio.is_muted': True
            })
        """
        with self._lock:
            old_state = self.get_state()
            
            for key, value in updates.items():
                if '.' in key:
                    # Handle nested updates like 'audio.is_muted'
                    parts = key.split('.')
                    obj = self._state
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                else:
                    setattr(self._state, key, value)
            
            # Record state change
            self._state_history.append({
                'timestamp': time.time(),
                'changes': updates
            })
            
            # Notify observers
            self._notify_observers(old_state, self._state)
    
    def update_connection(self, state: ConnectionState, latency_ms: Optional[float] = None):
        """Update connection state"""
        updates = {'connection_state': state}
        if latency_ms is not None:
            updates['api_latency_ms'] = latency_ms
        self.update_state(updates)
    
    def update_audio(self, **kwargs):
        """Update audio state"""
        updates = {f'audio.{k}': v for k, v in kwargs.items()}
        self.update_state(updates)
    
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the conversation"""
        with self._lock:
            message = self._state.conversation.add_message(role, content, **metadata)
            self._state.total_interactions += 1
            self._notify_observers(None, self._state)
            return message
    
    def observe(self, callback: callable):
        """
        Add a state observer.
        
        Callback signature: callback(old_state, new_state)
        """
        self._observers.append(callback)
    
    def unobserve(self, callback: callable):
        """Remove a state observer"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, old_state: Optional[TerminalState], new_state: TerminalState):
        """Notify all observers of state change"""
        for observer in self._observers:
            try:
                # Run in thread to avoid blocking
                threading.Thread(
                    target=observer,
                    args=(old_state, new_state),
                    daemon=True
                ).start()
            except Exception:
                pass  # Ignore observer errors
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get state change history"""
        with self._lock:
            return list(self._state_history)
    
    def reset(self):
        """Reset to initial state"""
        with self._lock:
            old_state = self.get_state()
            self._state = TerminalState()
            self._state_history.clear()
            self._notify_observers(old_state, self._state)


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager