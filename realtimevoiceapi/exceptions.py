# here is realtimevoiceapi/exceptions.py


"""Custom exceptions for RealtimeVoiceAPI"""

class RealtimeError(Exception):
    """Base exception for Realtime API errors"""
    pass

class ConnectionError(RealtimeError):
    """Raised when connection to the API fails"""
    pass

class AuthenticationError(RealtimeError):
    """Raised when authentication fails"""
    pass

class SessionError(RealtimeError):
    """Raised when session operations fail"""
    pass

class AudioError(RealtimeError):
    """Raised when audio processing fails"""
    pass

class StreamError(RealtimeError):
    """Raised when stream operations fail"""
    pass

class EngineError(RealtimeError):
    """Raised when engine operations fail"""
    pass

class ConfigurationError(RealtimeError):
    """Raised when configuration is invalid"""
    pass

class RateLimitError(RealtimeError):
    """Raised when rate limits are exceeded"""
    pass

class APIError(RealtimeError):
    """Raised when the API returns an error"""
    
    def __init__(self, message: str, error_code: str = None, error_type: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.error_type = error_type