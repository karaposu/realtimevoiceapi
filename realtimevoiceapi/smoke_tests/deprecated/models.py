@dataclass
class EngineComponents:
    """Container for all engine components"""
    strategy: Optional[BaseStrategy] = None
    audio_manager: Optional[AudioManager] = None
    buffered_player: Optional[BufferedAudioPlayer] = None
    
    # Processing tasks
    audio_processing_task: Optional[asyncio.Task] = None
    
    def cleanup_tasks(self):
        """Cancel all tasks"""
        if self.audio_processing_task and not self.audio_processing_task.done():
            self.audio_processing_task.cancel()
    
    def has_audio(self) -> bool:
        """Check if audio components are available"""
        return self.audio_manager is not None and self.buffered_player is not None
    
    def clear(self):
        """Clear all component references"""
        self.strategy = None
        self.audio_manager = None
        self.buffered_player = None
        self.audio_processing_task = None


@dataclass
class EventHandlerRegistry:
    """Manages event handlers with wrapping and state tracking"""
    
    # Handler storage
    user_handlers: Dict[StreamEventType, Callable] = field(default_factory=dict)
    wrapped_handlers: Dict[StreamEventType, Callable] = field(default_factory=dict)
    
    # Handler call counts for metrics
    handler_calls: Dict[StreamEventType, int] = field(default_factory=dict)
    handler_errors: Dict[StreamEventType, int] = field(default_factory=dict)
    
    def register_handler(self, event_type: StreamEventType, handler: Callable):
        """Register a user handler"""
        self.user_handlers[event_type] = handler
        self.handler_calls[event_type] = 0
        self.handler_errors[event_type] = 0
    
    def wrap_handler(self, event_type: StreamEventType, wrapper: Callable) -> Callable:
        """Wrap a handler with additional functionality"""
        original = self.user_handlers.get(event_type)
        if not original:
            return wrapper
        
        def wrapped_handler(event: StreamEvent):
            # Track calls
            self.handler_calls[event_type] = self.handler_calls.get(event_type, 0) + 1
            
            try:
                # Call wrapper with original
                return wrapper(original, event)
            except Exception as e:
                self.handler_errors[event_type] = self.handler_errors.get(event_type, 0) + 1
                raise
        
        self.wrapped_handlers[event_type] = wrapped_handler
        return wrapped_handler
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get handler metrics"""
        return {
            "registered_handlers": len(self.user_handlers),
            "handler_calls": dict(self.handler_calls),
            "handler_errors": dict(self.handler_errors)
        }
    
    def clear(self):
        """Clear all handlers"""
        self.user_handlers.clear()
        self.wrapped_handlers.clear()
        self.handler_calls.clear()
        self.handler_errors.clear()



@dataclass
class SessionInfo:
    """Session-specific information"""
    session_id: str = field(default_factory=lambda: f"session_{int(time.time() * 1000)}")
    mode: Optional[str] = None
    config: Optional[EngineConfig] = None
    
    # Timing
    created_at: float = field(default_factory=time.time)
    connected_at: Optional[float] = None
    disconnected_at: Optional[float] = None
    
    # Activity tracking
    last_activity: float = field(default_factory=time.time)
    total_messages_sent: int = 0
    total_messages_received: int = 0
    
    def mark_activity(self):
        """Update last activity time"""
        self.last_activity = time.time()
    
    def mark_connected(self):
        """Mark session as connected"""
        self.connected_at = time.time()
    
    def mark_disconnected(self):
        """Mark session as disconnected"""
        self.disconnected_at = time.time()
    
    @property
    def duration(self) -> float:
        """Get session duration"""
        if self.connected_at:
            end_time = self.disconnected_at or time.time()
            return end_time - self.connected_at
        return 0.0
    
    @property
    def is_active(self) -> bool:
        """Check if session is active"""
        return self.connected_at is not None and self.disconnected_at is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "connected_at": self.connected_at,
            "disconnected_at": self.disconnected_at,
            "duration": self.duration,
            "is_active": self.is_active,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "last_activity": self.last_activity
        } 




@dataclass
class AudioConfiguration:
    """Consolidated audio configuration"""
    sample_rate: int = 24000
    channels: int = 1
    bit_depth: int = 16
    chunk_duration_ms: int = 100
    
    # Device settings
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    
    # VAD settings
    vad_enabled: bool = True
    vad_config: Optional[VADConfig] = None
    
    # Buffer settings
    min_buffer_chunks: int = 2
    max_buffer_chunks: int = 5
    
    def to_audio_config(self) -> AudioConfig:
        """Convert to AudioConfig"""
        return AudioConfig(
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            chunk_duration_ms=self.chunk_duration_ms
        )
    
    def to_audio_manager_config(self) -> AudioManagerConfig:
        """Convert to AudioManagerConfig"""
        return AudioManagerConfig(
            input_device=self.input_device,
            output_device=self.output_device,
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
            vad_enabled=self.vad_enabled,
            vad_config=self.vad_config
        )
    




# class BaseEngine:
#     """
#     Base implementation for voice engine with organized dataclasses.
#     """
    
#     def __init__(self, logger: Optional[logging.Logger] = None):
#         """Initialize base engine"""
#         self.logger = logger or logging.getLogger(__name__)
        
#         # Core structures
#         self.state = CoreEngineState()
#         self.components = EngineComponents()
#         self.event_registry = EventHandlerRegistry()
#         self.session = SessionInfo()
#         self.audio_config = AudioConfiguration()
#         self.metrics = EngineMetrics()
    