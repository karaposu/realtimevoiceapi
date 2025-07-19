
# here is realtimevoiceapi/session_manager.py

"""
Session State Management

Manages runtime session state across different providers.
Different from session.py which handles configuration.
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from ..core.stream_protocol import StreamState, StreamID


class SessionState(Enum):
    """Runtime session states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class Session:
    """Runtime session information"""
    id: str
    provider: str
    stream_id: StreamID
    state: SessionState
    created_at: float
    config: Dict[str, Any]  # Provider-specific config
    
    # Runtime state
    last_activity: float = 0
    error_count: int = 0
    reconnect_count: int = 0
    
    # Usage tracking
    audio_seconds_used: float = 0
    text_tokens_used: int = 0
    function_calls_made: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    Manages active sessions across all providers.
    
    This is about runtime state, not configuration.
    Works with session.py's configs but tracks live sessions.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.provider_sessions: Dict[str, List[str]] = {}  # provider -> session_ids
        
    def create_session(
        self,
        provider: str,
        stream_id: StreamID,
        config: Dict[str, Any]
    ) -> Session:
        """Create new session"""
        session = Session(
            id=f"sess_{uuid.uuid4().hex[:12]}",
            provider=provider,
            stream_id=stream_id,
            state=SessionState.INITIALIZING,
            created_at=time.time(),
            config=config
        )
        
        self.sessions[session.id] = session
        
        # Track by provider
        if provider not in self.provider_sessions:
            self.provider_sessions[provider] = []
        self.provider_sessions[provider].append(session.id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def update_state(self, session_id: str, new_state: SessionState):
        """Update session state"""
        session = self.sessions.get(session_id)
        if session:
            session.state = new_state
            session.last_activity = time.time()
    
    def track_usage(
        self,
        session_id: str,
        audio_seconds: float = 0,
        text_tokens: int = 0,
        function_calls: int = 0
    ):
        """Track resource usage"""
        session = self.sessions.get(session_id)
        if session:
            session.audio_seconds_used += audio_seconds
            session.text_tokens_used += text_tokens
            session.function_calls_made += function_calls
            session.last_activity = time.time()
    
    def end_session(self, session_id: str):
        """End a session"""
        session = self.sessions.get(session_id)
        if session:
            session.state = SessionState.ENDED
            session.metadata["ended_at"] = time.time()
            session.metadata["duration"] = time.time() - session.created_at
    
    def get_active_sessions(self, provider: Optional[str] = None) -> List[Session]:
        """Get all active sessions, optionally filtered by provider"""
        active = [
            s for s in self.sessions.values()
            if s.state == SessionState.ACTIVE
        ]
        
        if provider:
            active = [s for s in active if s.provider == provider]
            
        return active
    
    def cleanup_old_sessions(self, max_age_seconds: float = 3600):
        """Remove old ended sessions"""
        now = time.time()
        to_remove = []
        
        for sid, session in self.sessions.items():
            if session.state == SessionState.ENDED:
                if now - session.created_at > max_age_seconds:
                    to_remove.append(sid)
        
        for sid in to_remove:
            session = self.sessions.pop(sid)
            # Remove from provider tracking
            if session.provider in self.provider_sessions:
                self.provider_sessions[session.provider].remove(sid)