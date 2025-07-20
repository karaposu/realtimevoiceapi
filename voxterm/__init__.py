"""
VoxTerm - Minimalist CLI for Voice Chat

A lightweight command-line interface for voice chat APIs.
"""

from .cli import VoxTermCLI
from .keyboard import SimpleKeyboard, KeyboardHandler
from .modes import (
    PushToTalkMode,
    AlwaysOnMode,
    TextMode,
    TurnBasedMode,
    create_mode
)
from .menu import VoxTermMenu, MenuState
from .launcher import launch_voxterm, main
from .settings import TerminalSettings, create_minimal_settings, create_debug_settings
from .session_manager import SessionManager, create_session
from .stream_protocol import StreamProtocol, StreamConfig, StreamSession, StreamType

__version__ = "1.0.0"

__all__ = [
    "VoxTermCLI",
    "VoxTermMenu",
    "launch_voxterm",
    "main",
    "SimpleKeyboard", 
    "KeyboardHandler",
    "PushToTalkMode",
    "AlwaysOnMode",
    "TextMode",
    "TurnBasedMode",
    "create_mode",
    "TerminalSettings",
    "create_minimal_settings",
    "create_debug_settings",
    "SessionManager",
    "create_session",
    "StreamProtocol",
    "StreamConfig",
    "StreamSession",
    "StreamType"
]