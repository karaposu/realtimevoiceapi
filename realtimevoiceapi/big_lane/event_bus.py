"""
Event Bus System - Big Lane Component

Flexible event-driven architecture for coordinating components.
Supports wildcards, priorities, async handlers, and event replay.

Key Features:
event_bus.py:

Pattern Matching: Supports wildcards for flexible subscriptions
Priority Delivery: Higher priority handlers get events first
Event History: Can replay events for debugging
Weak References: Auto-cleanup of dead handlers
Interceptors: Can modify or block events
Error Handling: Robust error handling with dedicated handlers
Metrics: Comprehensive metrics for monitoring


"""

import asyncio
import time
import re
import logging
from typing import Dict, List, Callable, Any, Optional, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import weakref
import inspect


@dataclass
class Event:
    """Base event structure"""
    type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    
    def matches(self, pattern: str) -> bool:
        """Check if event matches pattern (supports wildcards)"""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"[^.]*")
        return re.match(f"^{regex_pattern}$", self.type) is not None


class EventPriority(Enum):
    """Standard event priorities"""
    LOW = -1
    NORMAL = 0
    HIGH = 1
    CRITICAL = 2


@dataclass
class Subscription:
    """Subscription details"""
    id: str
    pattern: str
    handler: Callable
    priority: int = 0
    filter_func: Optional[Callable[[Event], bool]] = None
    once: bool = False
    weak: bool = False  # Use weak reference
    
    def __hash__(self):
        return hash(self.id)


class EventBus:
    """
    Flexible event bus for big lane coordination.
    
    Features:
    - Pattern matching with wildcards
    - Async and sync handlers
    - Priority-based delivery
    - Event filtering
    - Event replay
    - Metrics and debugging
    """
    
    def __init__(
        self,
        name: str = "main",
        history_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        
        # Subscriptions organized by pattern for efficiency
        self._subscriptions: Dict[str, Set[Subscription]] = defaultdict(set)
        self._pattern_cache: Dict[str, List[str]] = {}  # Cache pattern matches
        
        # Event history for replay and debugging
        self._history: deque = deque(maxlen=history_size)
        self._history_enabled = True
        
        # Metrics
        self._metrics = {
            "events_emitted": 0,
            "events_delivered": 0,
            "events_dropped": 0,
            "active_subscriptions": 0,
            "total_subscriptions": 0
        }
        
        # Event processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Error handlers
        self._error_handlers: List[Callable[[Exception, Event], None]] = []
        
        # Event interceptors (for debugging/logging)
        self._interceptors: List[Callable[[Event], Optional[Event]]] = []
    
    def start(self):
        """Start event processing"""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        self.logger.info(f"EventBus '{self.name}' started")
    
    async def stop(self):
        """Stop event processing"""
        if not self._running:
            return
        
        self._running = False
        
        # Process remaining events
        await self._event_queue.join()
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"EventBus '{self.name}' stopped")
    
    def subscribe(
        self,
        pattern: str,
        handler: Callable,
        priority: int = 0,
        filter_func: Optional[Callable[[Event], bool]] = None,
        once: bool = False,
        weak: bool = False
    ) -> str:
        """
        Subscribe to events matching pattern.
        
        Args:
            pattern: Event pattern (supports * wildcard)
            handler: Callback function (sync or async)
            priority: Delivery priority (higher = earlier)
            filter_func: Additional filter function
            once: Unsubscribe after first event
            weak: Use weak reference (auto-cleanup)
            
        Returns:
            Subscription ID
        """
        sub_id = f"sub_{int(time.time() * 1000000)}"
        
        # Wrap handler in weak reference if requested
        if weak:
            handler_ref = weakref.ref(handler)
            
            def weak_handler(*args, **kwargs):
                h = handler_ref()
                if h:
                    return h(*args, **kwargs)
                else:
                    # Handler was garbage collected
                    self.unsubscribe(sub_id)
            
            handler = weak_handler
        
        subscription = Subscription(
            id=sub_id,
            pattern=pattern,
            handler=handler,
            priority=priority,
            filter_func=filter_func,
            once=once,
            weak=weak
        )
        
        self._subscriptions[pattern].add(subscription)
        self._pattern_cache.clear()  # Clear cache when subscriptions change
        
        # Update metrics
        self._metrics["total_subscriptions"] += 1
        self._metrics["active_subscriptions"] = sum(
            len(subs) for subs in self._subscriptions.values()
        )
        
        self.logger.debug(f"Subscribed {sub_id} to pattern '{pattern}'")
        
        return sub_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe by ID"""
        removed = False
        
        for pattern, subs in self._subscriptions.items():
            to_remove = [s for s in subs if s.id == subscription_id]
            for sub in to_remove:
                subs.remove(sub)
                removed = True
        
        if removed:
            self._pattern_cache.clear()
            self._metrics["active_subscriptions"] = sum(
                len(subs) for subs in self._subscriptions.values()
            )
            self.logger.debug(f"Unsubscribed {subscription_id}")
        
        return removed
    
    def unsubscribe_pattern(self, pattern: str) -> int:
        """Unsubscribe all handlers for a pattern"""
        if pattern in self._subscriptions:
            count = len(self._subscriptions[pattern])
            del self._subscriptions[pattern]
            self._pattern_cache.clear()
            
            self._metrics["active_subscriptions"] = sum(
                len(subs) for subs in self._subscriptions.values()
            )
            
            return count
        return 0
    
    async def emit(
        self,
        event: Union[Event, str],
        data: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        **kwargs
    ):
        """
        Emit an event.
        
        Can pass Event object or build one from parameters.
        """
        # Build event if needed
        if isinstance(event, str):
            event = Event(
                type=event,
                data=data or {},
                priority=priority,
                source=self.name,
                **kwargs
            )
        
        # Apply interceptors
        for interceptor in self._interceptors:
            result = interceptor(event)
            if result is None:
                # Interceptor blocked the event
                self._metrics["events_dropped"] += 1
                return
            event = result
        
        # Add to history
        if self._history_enabled:
            self._history.append(event)
        
        # Queue for processing
        await self._event_queue.put(event)
        self._metrics["events_emitted"] += 1
        
        self.logger.debug(f"Emitted event: {event.type}")
    
    def emit_sync(self, event: Union[Event, str], **kwargs):
        """Synchronous emit (creates task)"""
        asyncio.create_task(self.emit(event, **kwargs))
    
    async def _process_events(self):
        """Main event processing loop"""
        while self._running:
            try:
                # Get event with timeout to allow checking _running
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Find matching subscriptions
                matching_subs = self._find_matching_subscriptions(event)
                
                # Sort by priority (highest first)
                matching_subs.sort(key=lambda s: s.priority, reverse=True)
                
                # Deliver to handlers
                for subscription in matching_subs:
                    try:
                        # Apply filter if present
                        if subscription.filter_func:
                            if not subscription.filter_func(event):
                                continue
                        
                        # Call handler
                        if inspect.iscoroutinefunction(subscription.handler):
                            await subscription.handler(event)
                        else:
                            # Run sync handler in executor to not block
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                subscription.handler,
                                event
                            )
                        
                        self._metrics["events_delivered"] += 1
                        
                        # Remove if one-time subscription
                        if subscription.once:
                            self.unsubscribe(subscription.id)
                            
                    except Exception as e:
                        self.logger.error(
                            f"Error in event handler for {event.type}: {e}"
                        )
                        
                        # Call error handlers
                        for error_handler in self._error_handlers:
                            try:
                                error_handler(e, event)
                            except Exception as eh_error:
                                self.logger.error(
                                    f"Error in error handler: {eh_error}"
                                )
                
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
    
    def _find_matching_subscriptions(self, event: Event) -> List[Subscription]:
        """Find all subscriptions matching the event"""
        matching = []
        
        # Check cache first
        if event.type in self._pattern_cache:
            patterns = self._pattern_cache[event.type]
        else:
            # Find all matching patterns
            patterns = []
            for pattern in self._subscriptions.keys():
                if event.matches(pattern):
                    patterns.append(pattern)
            
            # Cache result
            self._pattern_cache[event.type] = patterns
        
        # Collect subscriptions
        for pattern in patterns:
            matching.extend(self._subscriptions[pattern])
        
        return matching
    
    def add_interceptor(self, interceptor: Callable[[Event], Optional[Event]]):
        """
        Add event interceptor.
        
        Interceptor can modify or block events (return None to block).
        """
        self._interceptors.append(interceptor)
    
    def add_error_handler(self, handler: Callable[[Exception, Event], None]):
        """Add error handler for failed event handlers"""
        self._error_handlers.append(handler)
    
    def get_history(
        self,
        pattern: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get event history"""
        events = list(self._history)
        
        # Filter by pattern if provided
        if pattern:
            events = [e for e in events if e.matches(pattern)]
        
        # Limit results
        if limit:
            events = events[-limit:]
        
        return events
    
    def replay_events(
        self,
        pattern: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None
    ):
        """Replay historical events"""
        events = self.get_history(pattern)
        
        # Filter by time
        if since:
            events = [e for e in events if e.timestamp >= since]
        if until:
            events = [e for e in events if e.timestamp <= until]
        
        # Re-emit events
        for event in events:
            # Mark as replay
            event.metadata["replayed"] = True
            event.metadata["original_timestamp"] = event.timestamp
            event.timestamp = time.time()
            
            self.emit_sync(event)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return {
            **self._metrics,
            "queue_size": self._event_queue.qsize(),
            "history_size": len(self._history),
            "pattern_cache_size": len(self._pattern_cache),
            "unique_patterns": len(self._subscriptions)
        }
    
    def clear_history(self):
        """Clear event history"""
        self._history.clear()
    
    def set_history_enabled(self, enabled: bool):
        """Enable/disable history tracking"""
        self._history_enabled = enabled


    

    

    async def stop(self):
        """Stop the event bus"""
        if not self._running:
            return
        
        self._running = False
        
        # Only try to cancel worker task if it exists
        if hasattr(self, '_worker_task') and self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Clear the queue if it exists
        if hasattr(self, '_event_queue'):
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        self.logger.info(f"EventBus '{self.name}' stopped")


class EventBusNetwork:
    """
    Network of event buses for complex applications.
    
    Allows creating isolated buses with bridging.
    """
    
    def __init__(self):
        self.buses: Dict[str, EventBus] = {}
        self.bridges: List[Tuple[str, str, str]] = []  # (from_bus, to_bus, pattern)
    
    def create_bus(self, name: str, **kwargs) -> EventBus:
        """Create a new event bus"""
        if name in self.buses:
            raise ValueError(f"Bus '{name}' already exists")
        
        bus = EventBus(name=name, **kwargs)
        self.buses[name] = bus
        return bus
    
    def get_bus(self, name: str = "main") -> EventBus:
        """Get event bus by name"""
        if name not in self.buses:
            # Auto-create main bus
            if name == "main":
                return self.create_bus("main")
            raise ValueError(f"Bus '{name}' not found")
        
        return self.buses[name]
    
    def bridge(self, from_bus: str, to_bus: str, pattern: str = "*"):
        """Bridge events from one bus to another"""
        if from_bus not in self.buses:
            raise ValueError(f"Source bus '{from_bus}' not found")
        if to_bus not in self.buses:
            raise ValueError(f"Target bus '{to_bus}' not found")
        
        # Subscribe to source bus
        source = self.buses[from_bus]
        target = self.buses[to_bus]
        
        async def bridge_handler(event: Event):
            # Forward to target bus
            event.metadata["bridged_from"] = from_bus
            await target.emit(event)
        
        source.subscribe(pattern, bridge_handler)
        self.bridges.append((from_bus, to_bus, pattern))
    
    async def start_all(self):
        """Start all buses"""
        for bus in self.buses.values():
            bus.start()
    
    async def stop_all(self):
        """Stop all buses"""
        for bus in self.buses.values():
            await bus.stop()