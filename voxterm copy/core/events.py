"""
VoxTerm Event System

Non-blocking event system for communication between components.
Designed to never block the audio pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Callable
from enum import Enum, auto
import time
import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import weakref


class EventType(Enum):
    """Standard event types in VoxTerm"""
    # Audio events
    AUDIO_INPUT_START = auto()
    AUDIO_INPUT_END = auto()
    AUDIO_INPUT_DATA = auto()
    AUDIO_OUTPUT_START = auto()
    AUDIO_OUTPUT_END = auto()
    AUDIO_OUTPUT_DATA = auto()
    
    # Text events
    TEXT_INPUT = auto()
    TEXT_OUTPUT = auto()
    
    # Control events
    CONTROL = auto()  # Added this
    INTERRUPT = auto()
    MUTE_TOGGLE = auto()
    MODE_CHANGE = auto()
    
    # State events
    CONNECTION_STATE = auto()
    RECORDING_STATE = auto()
    PLAYBACK_STATE = auto()
    
    # UI events
    UI_REFRESH = auto()
    UI_RESIZE = auto()
    LOG_TOGGLE = auto()
    
    # System events
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    SHUTDOWN = auto()


@dataclass
class Event:
    """Base event class"""
    type: EventType
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    
    def __post_init__(self):
        # Ensure data is always a dict
        if self.data is None:
            self.data = {}


@dataclass
class AudioInputEvent(Event):
    """Audio input event with audio data"""
    type: EventType = EventType.AUDIO_INPUT_DATA
    audio_data: Optional[bytes] = None
    duration_ms: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.audio_data:
            self.data["size_bytes"] = len(self.audio_data)


@dataclass
class TextEvent(Event):
    """Text input/output event"""
    text: str = ""
    is_final: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        self.data["text_length"] = len(self.text)


@dataclass 
class ControlEvent(Event):
    """Control command event"""
    type: EventType = EventType.CONTROL  # Default to CONTROL type
    command: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Central event dispatcher for VoxTerm.
    
    Features:
    - Non-blocking event dispatch
    - Weak references to prevent memory leaks
    - Priority queue for important events
    - Thread-safe operation
    """
    
    def __init__(self, max_workers: int = 4):
        self._subscribers: Dict[EventType, List[weakref.ref]] = {}
        self._event_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._dispatch_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Metrics
        self.events_dispatched = 0
        self.events_dropped = 0
        
    def start(self):
        """Start the event dispatcher"""
        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="VoxTerm-EventDispatcher"
        )
        self._dispatch_thread.start()
    
    def stop(self):
        """Stop the event dispatcher"""
        self._running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False)
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type.
        
        Uses weak references to prevent memory leaks.
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            # Store weak reference to callback
            self._subscribers[event_type].append(weakref.ref(callback))
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self._subscribers:
                # Remove dead references and the specific callback
                self._subscribers[event_type] = [
                    ref for ref in self._subscribers[event_type]
                    if ref() is not None and ref() != callback
                ]
    
    def emit(self, event: Event) -> None:
        """
        Emit an event (non-blocking).
        
        Events are queued and dispatched asynchronously.
        """
        if not self._running:
            return
        
        try:
            # Priority is negative so higher priority comes first
            self._event_queue.put_nowait((-event.priority, time.time(), event))
        except queue.Full:
            self.events_dropped += 1
    
    def emit_sync(self, event: Event) -> None:
        """
        Emit an event synchronously (use sparingly).
        
        Blocks until all handlers have been called.
        """
        self._dispatch_event(event)
    
    def _dispatch_loop(self):
        """Main event dispatch loop"""
        while self._running:
            try:
                # Get event with timeout
                priority, timestamp, event = self._event_queue.get(timeout=0.1)
                self._dispatch_event(event)
                self.events_dispatched += 1
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but keep running
                import logging
                logging.error(f"Event dispatch error: {e}")
    
    def _dispatch_event(self, event: Event):
        """Dispatch an event to all subscribers"""
        with self._lock:
            subscribers = self._subscribers.get(event.type, [])
            # Clean up dead references
            alive_subscribers = []
            
            for ref in subscribers:
                callback = ref()
                if callback is not None:
                    alive_subscribers.append(ref)
                    # Execute callback in thread pool
                    self._executor.submit(self._safe_call, callback, event)
            
            # Update subscribers list
            if alive_subscribers != subscribers:
                self._subscribers[event.type] = alive_subscribers
    
    def _safe_call(self, callback: Callable, event: Event):
        """Safely call a callback"""
        try:
            callback(event)
        except Exception as e:
            # Emit error event (but avoid infinite loop)
            if event.type != EventType.ERROR:
                error_event = Event(
                    type=EventType.ERROR,
                    source="EventBus",
                    data={"error": str(e), "original_event": event.type.name}
                )
                self.emit(error_event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            return {
                "events_dispatched": self.events_dispatched,
                "events_dropped": self.events_dropped,
                "queue_size": self._event_queue.qsize(),
                "subscribers": {
                    event_type.name: len([r for r in refs if r() is not None])
                    for event_type, refs in self._subscribers.items()
                }
            }


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
        _event_bus.start()
    return _event_bus


def emit_event(event: Event) -> None:
    """Convenience function to emit an event"""
    get_event_bus().emit(event)


def subscribe(event_type: EventType, callback: Callable[[Event], None]) -> None:
    """Convenience function to subscribe to events"""
    get_event_bus().subscribe(event_type, callback)