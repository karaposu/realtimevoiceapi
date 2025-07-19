"""
VoxTerm - Terminal UI Framework for Voice Applications

A lightweight, non-blocking terminal interface for real-time voice applications.
"""

__version__ = "0.1.0"

# Core exports
from .terminal.terminal import VoxTerminal
from .terminal.runner import (
    VoxTermRunner,
    run_terminal,
    run_terminal_async,
    VoxTermContext
)

# Configuration
from .config.settings import TerminalSettings, PRESETS
from .config.manager import get_config, set_config

# State
from .core.state import get_state_manager

# Events
from .core.events import get_event_bus, Event, EventType

# Integration helpers
from .integration.base import BaseIntegration
from .integration.voice_engine import VoiceEngineAdapter

__all__ = [
    # Main classes
    'VoxTerminal',
    'VoxTermRunner',
    'VoxTermContext',
    
    # Functions
    'run_terminal',
    'run_terminal_async',
    'get_config',
    'set_config',
    'get_state_manager',
    'get_event_bus',
    
    # Types
    'TerminalSettings',
    'Event',
    'EventType',
    'BaseIntegration',
    'VoiceEngineAdapter',
    
    # Constants
    'PRESETS',
    '__version__'
]