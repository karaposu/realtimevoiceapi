# here is  core/engine_state.py 

from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class CoreEngineState:
    """Core engine state tracking"""
    
    # Connection state
    is_connected: bool = False
    connection_time: Optional[float] = None
    
    # Audio input state
    is_listening: bool = False
    listening_start_time: Optional[float] = None
    
    # Audio output state
    response_audio_started: bool = False
    response_audio_complete: bool = False
    
    # Session info
    session_start_time: Optional[float] = None
    stream_state: StreamState = StreamState.IDLE
    
    # Metrics
    total_interactions: int = 0
    total_audio_chunks_sent: int = 0
    total_audio_chunks_received: int = 0
    
    @property
    def is_ai_speaking(self) -> bool:
        """Check if AI is currently speaking"""
        return self.response_audio_started and not self.response_audio_complete
    
    @property
    def uptime(self) -> float:
        """Get session uptime in seconds"""
        if self.session_start_time:
            return time.time() - self.session_start_time
        return 0.0
    
    @property
    def is_ready(self) -> bool:
        """Check if ready for new interactions"""
        return (
            self.is_connected and 
            self.stream_state == StreamState.ACTIVE and
            not self.is_ai_speaking
        )
    
    def reset_response_state(self):
        """Reset response-related state"""
        self.response_audio_started = False
        self.response_audio_complete = False
    
    def mark_interaction(self):
        """Mark a new interaction"""
        self.total_interactions += 1
        self.reset_response_state()
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for metrics"""
        return {
            "is_connected": self.is_connected,
            "is_listening": self.is_listening,
            "is_ai_speaking": self.is_ai_speaking,
            "is_ready": self.is_ready,
            "stream_state": self.stream_state.value,
            "uptime": self.uptime,
            "total_interactions": self.total_interactions,
            "audio_chunks_sent": self.total_audio_chunks_sent,
            "audio_chunks_received": self.total_audio_chunks_received
        }