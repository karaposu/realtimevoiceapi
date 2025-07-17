# realtimevoiceapi/connection.py

"""
WebSocket connection for OpenAI Realtime API

This module handles the low-level WebSocket communication with OpenAI's Realtime API,
including connection establishment, message sending/receiving, and error handling.
"""

import asyncio
import websockets
import json
import logging
import ssl
import time
from typing import Optional, Callable, Dict, Any
from websockets.exceptions import ConnectionClosed, InvalidStatusCode, WebSocketException

from .exceptions import ConnectionError, AuthenticationError, RealtimeError


class RealtimeConnection:
    """
    Manages WebSocket connection to OpenAI Realtime API
    
    Handles connection establishment, message transmission, and connection lifecycle.
    """
    
    REALTIME_URL = "wss://api.openai.com/v1/realtime"
    
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        """
        Initialize connection manager
        
        Args:
            api_key: OpenAI API key with Realtime API access
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.connection_id: Optional[str] = None
        
        # Message handling
        self.message_handler: Optional[Callable] = None
        self.message_listener_task: Optional[asyncio.Task] = None
        
        # Connection metrics
        self.connect_time: Optional[float] = None
        self.last_message_time: Optional[float] = None
        self.messages_sent = 0
        self.messages_received = 0
        
        # SSL context for secure connection
        self.ssl_context = ssl.create_default_context()
    
    async def connect(self, model: str = "gpt-4o-realtime-preview") -> bool:
        """
        Establish WebSocket connection to OpenAI Realtime API
        
        Args:
            model: Model to use for the session
            
        Returns:
            True if connection successful, raises exception otherwise
        """
        if self.is_connected:
            self.logger.warning("Already connected, disconnecting first")
            await self.disconnect()
        
        try:
            # Add model parameter to URL
            url = f"{self.REALTIME_URL}?model={model}"
            
            self.logger.info(f"Connecting to {url}")
            
            # With websockets 15.0.1, we should use additional_headers
            self.websocket = await websockets.connect(
                url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                },
                ssl=self.ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024,
                compression=None
            )
            
            self.is_connected = True
            self.connect_time = time.time()
            self.connection_id = f"conn_{int(self.connect_time * 1000)}"
            
            self.logger.info(f"Connected successfully: {self.connection_id}")
            
            # Start message listener
            self.message_listener_task = asyncio.create_task(self._message_listener())
            
            return True
            
        except InvalidStatusCode as e:
            if e.status_code == 401:
                raise AuthenticationError(f"Invalid API key or insufficient permissions: {e}")
            elif e.status_code == 429:
                raise ConnectionError(f"Rate limit exceeded: {e}")
            elif e.status_code == 403:
                raise AuthenticationError(f"Access forbidden - check API key permissions: {e}")
            else:
                raise ConnectionError(f"Connection failed with HTTP {e.status_code}: {e}")
                
        except WebSocketException as e:
            raise ConnectionError(f"WebSocket connection failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {e}")
            raise ConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self):
        """
        Close WebSocket connection and cleanup
        """
        if not self.is_connected:
            return
        
        try:
            self.logger.info("Disconnecting from Realtime API")
            
            # Cancel message listener
            if self.message_listener_task and not self.message_listener_task.done():
                self.message_listener_task.cancel()
                try:
                    await self.message_listener_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            # Reset state
            self.is_connected = False
            self.message_listener_task = None
            
            # Log connection statistics
            if self.connect_time:
                duration = time.time() - self.connect_time
                self.logger.info(
                    f"Disconnected after {duration:.1f}s "
                    f"(sent: {self.messages_sent}, received: {self.messages_received})"
                )
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            self.is_connected = False
    
    async def send_event(self, event: Dict[str, Any]):
        """
        Send event to WebSocket
        
        Args:
            event: Event dictionary to send
            
        Raises:
            ConnectionError: If not connected or send fails
        """
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to Realtime API")
        
        try:
            # Add event ID if not present
            if "event_id" not in event:
                event["event_id"] = f"evt_{int(time.time() * 1000000)}"
            
            # Serialize and send
            message = json.dumps(event)
            await self.websocket.send(message)
            
            # Update metrics
            self.messages_sent += 1
            self.last_message_time = time.time()
            
            # Log message (truncate large messages)
            event_type = event.get("type", "unknown")
            if len(message) > 1000:
                self.logger.debug(f"Sent large event: {event_type} ({len(message)} bytes)")
            else:
                self.logger.debug(f"Sent event: {event_type}")
                
        except ConnectionClosed:
            self.is_connected = False
            raise ConnectionError("Connection closed while sending event")
            
        except Exception as e:
            self.logger.error(f"Failed to send event: {e}")
            raise ConnectionError(f"Send failed: {e}")
    
    def set_message_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Set handler for incoming messages
        
        Args:
            handler: Async or sync callable that receives message dictionaries
        """
        self.message_handler = handler
        self.logger.debug("Message handler set")
    
    async def _message_listener(self):
        """
        Listen for incoming WebSocket messages
        
        This runs in a background task and dispatches messages to the handler.
        """
        try:
            self.logger.debug("Message listener started")
            
            async for message in self.websocket:
                try:
                    # Parse JSON message
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')
                    
                    event_data = json.loads(message)
                    
                    # Update metrics
                    self.messages_received += 1
                    self.last_message_time = time.time()
                    
                    # Log received message
                    event_type = event_data.get("type", "unknown")
                    event_id = event_data.get("event_id", "no-id")
                    
                    if len(message) > 1000:
                        self.logger.debug(f"Received large event: {event_type} ({len(message)} bytes)")
                    else:
                        self.logger.debug(f"Received event: {event_type} [{event_id}]")
                    
                    # Dispatch to handler
                    if self.message_handler:
                        try:
                            if asyncio.iscoroutinefunction(self.message_handler):
                                await self.message_handler(event_data)
                            else:
                                self.message_handler(event_data)
                        except Exception as e:
                            self.logger.error(f"Error in message handler for {event_type}: {e}")
                    else:
                        self.logger.warning(f"No message handler set, dropping event: {event_type}")
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse message as JSON: {e}")
                    self.logger.debug(f"Invalid message: {message[:200]}...")
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except ConnectionClosed:
            self.logger.info("WebSocket connection closed")
            self.is_connected = False
            
        except asyncio.CancelledError:
            self.logger.debug("Message listener cancelled")
            
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
            self.is_connected = False
        
        finally:
            self.logger.debug("Message listener stopped")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics
        
        Returns:
            Dictionary with connection metrics
        """
        stats = {
            "connected": self.is_connected,
            "connection_id": self.connection_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "last_message_time": self.last_message_time
        }
        
        if self.connect_time:
            stats["connection_duration"] = time.time() - self.connect_time
            stats["connect_time"] = self.connect_time
        
        return stats
    
    async def ping(self) -> bool:
        """
        Send a ping to test connection
        
        Returns:
            True if ping successful, False otherwise
        """
        if not self.is_connected or not self.websocket:
            return False
        
        try:
            pong_waiter = await self.websocket.ping()
            await asyncio.wait_for(pong_waiter, timeout=5.0)
            self.logger.debug("Ping successful")
            return True
            
        except asyncio.TimeoutError:
            self.logger.warning("Ping timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Ping failed: {e}")
            return False
    
    async def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for connection to be established
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected within timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected:
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    def is_alive(self) -> bool:
        """
        Check if connection is alive and responsive
        
        Returns:
            True if connection appears healthy
        """
        if not self.is_connected or not self.websocket:
            return False
        
        # Check if WebSocket is still open
        if self.websocket.closed:
            self.is_connected = False
            return False
        
        # Check if we've received messages recently (within last 60 seconds)
        if self.last_message_time:
            time_since_last_message = time.time() - self.last_message_time
            if time_since_last_message > 60:
                self.logger.warning(f"No messages received for {time_since_last_message:.1f}s")
                return False
        
        return True
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()