
# here is websocket_connection.py
"""
WebSocket Connection Manager

Provider-agnostic WebSocket connection handling with support for both
fast lane (minimal overhead) and big lane (full features) implementations.

Provider Agnostic: Takes URL and headers as config, not hardcoded
Fast/Big Lane Support: Different features enabled based on config
Serialization Strategy: Pluggable serializers for different formats
Backpressure Handling: Queue with size limits for big lane
Reconnection Logic: Exponential backoff with configurable limits
Metrics Optional: Can be disabled for fast lane
Better Error Handling: Connection state machine
Type Safety: Uses protocols and enums

This can be shared by both lanes with different configurations, maintaining performance for fast lane while providing full features for big lane.


"""

import asyncio
import json
import time
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Dict, Any, Union, Protocol
from urllib.parse import urlparse
import logging

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode, WebSocketException

from .exceptions import ConnectionError, AuthenticationError
from .message_protocol import MessageValidator, MessageParser


# ============== Types and Protocols ==============

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    RAW = "raw"


@dataclass
class ConnectionConfig:
    """Connection configuration"""
    url: str
    headers: Dict[str, str]
    
    # Connection parameters
    ping_interval: float = 20.0
    ping_timeout: float = 10.0
    close_timeout: float = 10.0
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    compression: Optional[str] = None
    
    # Behavior flags
    auto_reconnect: bool = True
    reconnect_max_attempts: int = 5
    reconnect_backoff_base: float = 2.0
    reconnect_backoff_max: float = 60.0
    
    # Serialization
    serialization_format: SerializationFormat = SerializationFormat.JSON
    
    # Performance
    enable_message_queue: bool = True  # False for fast lane
    max_queue_size: int = 1000
    enable_metrics: bool = True  # False for fast lane


class MessageSerializer(Protocol):
    """Protocol for message serialization"""
    def serialize(self, data: Any) -> Union[str, bytes]: ...
    def deserialize(self, data: Union[str, bytes]) -> Any: ...


# ============== Serializers ==============

class JsonSerializer:
    """JSON message serialization"""
    def serialize(self, data: Any) -> str:
        return json.dumps(data)
    
    def deserialize(self, data: Union[str, bytes]) -> Any:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)


class RawSerializer:
    """No serialization - pass through"""
    def serialize(self, data: Any) -> Any:
        return data
    
    def deserialize(self, data: Any) -> Any:
        return data


# ============== Base Connection ==============

class WebSocketConnection:
    """
    Base WebSocket connection manager
    
    Supports both fast lane (minimal features) and big lane (full features)
    through configuration.
    """
    
    def __init__(
        self,
        config: ConnectionConfig,
        logger: Optional[logging.Logger] = None,
        message_handler: Optional[Callable] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.message_handler = message_handler
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connection_id: Optional[str] = None
        
        # Tasks
        self.listener_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None
        self.send_worker_task: Optional[asyncio.Task] = None  # Initialize to None
        
        # Metrics (disabled for fast lane)
        if config.enable_metrics:
            self.metrics = ConnectionMetrics()
        else:
            self.metrics = None
        
        # Message queue (disabled for fast lane)
        if config.enable_message_queue:
            self.send_queue = asyncio.Queue(maxsize=config.max_queue_size)
            self.send_worker_task: Optional[asyncio.Task] = None
        else:
            self.send_queue = None
        
        # Serializer
        self.serializer = self._create_serializer()
        
        # SSL context
        self.ssl_context = self._create_ssl_context()
        
        # Reconnection state
        self.reconnect_attempts = 0
        self.last_connect_time = 0
        
    def _create_serializer(self) -> MessageSerializer:
        """Create appropriate serializer"""
        if self.config.serialization_format == SerializationFormat.JSON:
            return JsonSerializer()
        elif self.config.serialization_format == SerializationFormat.RAW:
            return RawSerializer()
        else:
            raise ValueError(f"Unsupported serialization: {self.config.serialization_format}")
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        context = ssl.create_default_context()
        # Can be customized based on provider requirements
        return context
    
    # ============== Connection Lifecycle ==============
    
    async def connect(self) -> None:
        """Establish WebSocket connection"""
        if self.state != ConnectionState.DISCONNECTED:
            raise ConnectionError(f"Cannot connect in state: {self.state}")
        
        self.state = ConnectionState.CONNECTING
        self.reconnect_attempts = 0
        
        try:
            await self._do_connect()
        except Exception as e:
            self.state = ConnectionState.DISCONNECTED
            raise
    
    async def _do_connect(self) -> None:
        """Internal connection logic"""
        self.logger.info(f"Connecting to {self.config.url}")
        
        # Parse URL to check if SSL needed
        parsed = urlparse(self.config.url)
        use_ssl = parsed.scheme in ['wss', 'https']
        
        # Create connection
        self.websocket = await websockets.connect(
            self.config.url,
            additional_headers=self.config.headers,
            ssl=self.ssl_context if use_ssl else None,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
            close_timeout=self.config.close_timeout,
            max_size=self.config.max_message_size,
            compression=self.config.compression
        )
        
        self.state = ConnectionState.CONNECTED
        self.connection_id = f"conn_{int(time.time() * 1000)}"
        self.last_connect_time = time.time()
        
        if self.metrics:
            self.metrics.on_connect()
        
        self.logger.info(f"Connected: {self.connection_id}")
        
        # Start background tasks
        self.listener_task = asyncio.create_task(self._message_listener())
        
        if self.send_queue is not None:
            self.send_worker_task = asyncio.create_task(self._send_worker())
    
    async def disconnect(self) -> None:
        """Close connection gracefully"""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        self.state = ConnectionState.CLOSING
        self.logger.info("Disconnecting")
        
        # Cancel reconnection if in progress
        if self.reconnect_task:
            self.reconnect_task.cancel()
        
        # Cancel background tasks
        if self.listener_task:
            self.listener_task.cancel()
        
        # Cancel send worker if it exists (only for big lane)
        if hasattr(self, 'send_worker_task') and self.send_worker_task:
            self.send_worker_task.cancel()
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
        
        self.state = ConnectionState.CLOSED
        self.websocket = None
        
        if self.metrics:
            self.metrics.on_disconnect()
    # ============== Message Handling ==============
    
    async def send(self, data: Any) -> None:
        """
        Send message through WebSocket
        
        For fast lane: Direct send
        For big lane: Queued send with backpressure
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError(f"Cannot send in state: {self.state}")
        
        if self.send_queue is not None:
            # Big lane: Queue the message
            try:
                await asyncio.wait_for(
                    self.send_queue.put(data),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                raise ConnectionError("Send queue full - backpressure")
        else:
            # Fast lane: Direct send
            await self._send_now(data)
    
    async def _send_now(self, data: Any) -> None:
        """Direct send implementation"""
        if not self.websocket:
            raise ConnectionError("No active connection")
        
        # Serialize
        message = self.serializer.serialize(data)
        
        # Send
        await self.websocket.send(message)
        
        if self.metrics:
            self.metrics.on_message_sent(len(str(message)))
        
        # Log based on size
        if len(str(message)) > 1000:
            self.logger.debug(f"Sent large message ({len(str(message))} bytes)")
        else:
            self.logger.debug("Sent message")
    
    async def _send_worker(self) -> None:
        """Background worker for queued sends (big lane only)"""
        while self.state == ConnectionState.CONNECTED:
            try:
                # Get message from queue
                data = await self.send_queue.get()
                
                # Send it
                await self._send_now(data)
                
            except ConnectionClosed:
                self.logger.warning("Connection closed during send")
                await self._handle_connection_lost()
                break
            except Exception as e:
                self.logger.error(f"Send worker error: {e}")
    
    async def _message_listener(self) -> None:
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    # Deserialize
                    data = self.serializer.deserialize(message)
                    
                    if self.metrics:
                        self.metrics.on_message_received(len(str(message)))
                    
                    # Dispatch to handler
                    if self.message_handler:
                        if asyncio.iscoroutinefunction(self.message_handler):
                            await self.message_handler(data)
                        else:
                            # Run sync handler in thread pool to avoid blocking
                            await asyncio.get_event_loop().run_in_executor(
                                None, self.message_handler, data
                            )
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except ConnectionClosed:
            self.logger.info("Connection closed by server")
            await self._handle_connection_lost()
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
            await self._handle_connection_lost()
    
    # ============== Reconnection ==============
    
    async def _handle_connection_lost(self) -> None:
        """Handle unexpected disconnection"""
        self.state = ConnectionState.DISCONNECTED
        self.websocket = None
        
        if self.config.auto_reconnect and self.reconnect_attempts < self.config.reconnect_max_attempts:
            self.state = ConnectionState.RECONNECTING
            self.reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff"""
        while self.reconnect_attempts < self.config.reconnect_max_attempts:
            self.reconnect_attempts += 1
            
            # Calculate backoff
            backoff = min(
                self.config.reconnect_backoff_base ** self.reconnect_attempts,
                self.config.reconnect_backoff_max
            )
            
            self.logger.info(
                f"Reconnection attempt {self.reconnect_attempts}/{self.config.reconnect_max_attempts} "
                f"in {backoff:.1f}s"
            )
            
            await asyncio.sleep(backoff)
            
            try:
                await self._do_connect()
                self.logger.info("Reconnection successful")
                return
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")
        
        self.logger.error("Max reconnection attempts reached")
        self.state = ConnectionState.CLOSED
    
    # ============== Status and Metrics ==============
    
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self.state == ConnectionState.CONNECTED and self.websocket is not None
    
    async def ping(self) -> bool:
        """Send ping to test connection"""
        if not self.is_connected():
            return False
        
        try:
            pong = await self.websocket.ping()
            await asyncio.wait_for(pong, timeout=5.0)
            return True
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        stats = {
            "state": self.state.value,
            "connection_id": self.connection_id,
            "connected": self.is_connected(),
            "reconnect_attempts": self.reconnect_attempts
        }
        
        if self.metrics:
            stats.update(self.metrics.get_stats())
        
        if self.send_queue:
            stats["queue_size"] = self.send_queue.qsize()
        
        return stats


# ============== Connection Metrics ==============

class ConnectionMetrics:
    """Track connection metrics (big lane only)"""
    
    def __init__(self):
        self.connect_count = 0
        self.disconnect_count = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.last_activity_time = None
        self.connect_time = None
    
    def on_connect(self):
        self.connect_count += 1
        self.connect_time = time.time()
    
    def on_disconnect(self):
        self.disconnect_count += 1
        self.connect_time = None
    
    def on_message_sent(self, size: int):
        self.messages_sent += 1
        self.bytes_sent += size
        self.last_activity_time = time.time()
    
    def on_message_received(self, size: int):
        self.messages_received += 1
        self.bytes_received += size
        self.last_activity_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "connects": self.connect_count,
            "disconnects": self.disconnect_count,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received
        }
        
        if self.connect_time:
            stats["uptime_seconds"] = time.time() - self.connect_time
        
        if self.last_activity_time:
            stats["idle_seconds"] = time.time() - self.last_activity_time
        
        return stats


# ============== Specialized Connections ==============

class FastLaneConnection(WebSocketConnection):
    """Optimized connection for fast lane - minimal overhead"""
    
    def __init__(self, url: str, headers: Dict[str, str], **kwargs):
        config = ConnectionConfig(
            url=url,
            headers=headers,
            enable_message_queue=False,  # No queue
            enable_metrics=False,        # No metrics
            auto_reconnect=False,        # No auto-reconnect
            **kwargs
        )
        super().__init__(config)


class BigLaneConnection(WebSocketConnection):
    """Full-featured connection for big lane"""
    
    def __init__(self, url: str, headers: Dict[str, str], **kwargs):
        config = ConnectionConfig(
            url=url,
            headers=headers,
            enable_message_queue=True,   # Full queue support
            enable_metrics=True,         # Full metrics
            auto_reconnect=True,         # Auto-reconnect
            **kwargs
        )
        super().__init__(config)