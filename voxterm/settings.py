"""
VoxTerm Settings - Configuration for the voice chat terminal

Simple dataclasses for configuring VoxTerm behavior.
No persistence - just runtime configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class KeyBindings:
    """Keyboard shortcut configuration"""
    # Mode controls
    push_to_talk: str = "space"
    text_input: str = "t"
    
    # Audio controls
    mute_toggle: str = "m"
    volume_up: str = "+"
    volume_down: str = "-"
    pause_resume: str = "p"  # For always-on mode
    
    # UI controls
    interrupt: str = "escape"
    clear_screen: str = "c"
    
    # System
    quit: str = "q"
    help: str = "h"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class DisplaySettings:
    """Display configuration"""
    # Status display
    show_status_bar: bool = True
    show_control_tips: bool = True
    show_latency: bool = True
    show_token_count: bool = False
    show_timestamp: bool = True
    
    # Format settings
    timestamp_format: str = "%H:%M:%S"
    
    # Visual indicators
    show_recording_indicator: bool = True
    show_processing_indicator: bool = True
    show_connection_status: bool = True


@dataclass
class AudioDisplaySettings:
    """Audio visualization settings"""
    # Audio level display
    show_input_level: bool = True
    show_output_level: bool = False
    
    # VAD indicators
    show_vad_status: bool = True
    vad_active_indicator: str = "🔴"
    vad_inactive_indicator: str = "⚪"


@dataclass
class VoiceSettings:
    """Voice configuration (for display/UI)"""
    # Available options
    available_voices: List[str] = field(default_factory=lambda: [
        "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    ])
    current_voice: str = "alloy"
    
    available_languages: List[str] = field(default_factory=lambda: ["en"])
    current_language: str = "en"


@dataclass
class BehaviorSettings:
    """Behavior configuration"""
    # Auto-actions
    auto_clear_on_disconnect: bool = False
    auto_scroll: bool = True
    
    # Interaction behavior
    confirm_before_quit: bool = False
    show_welcome_message: bool = True
    
    # Error handling
    show_error_details: bool = True
    retry_on_error: bool = True
    max_retries: int = 3


@dataclass
class AudioSettings:
    """Audio processing settings"""
    # Echo cancellation (if supported by engine)
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain_control: bool = True
    
    # VAD settings for always-on mode
    vad_threshold: float = 0.6  # 0.0-1.0, higher = less sensitive
    vad_silence_duration_ms: int = 800
    vad_prefix_padding_ms: int = 300


@dataclass
class TerminalSettings:
    """Complete terminal configuration"""
    # Components
    key_bindings: KeyBindings = field(default_factory=KeyBindings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    audio_display: AudioDisplaySettings = field(default_factory=AudioDisplaySettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    
    # Mode preferences
    default_mode: str = "turn_based"


# Default settings instance
DEFAULT_SETTINGS = TerminalSettings()


# Quick access functions
def create_settings(**kwargs) -> TerminalSettings:
    """Create settings with overrides"""
    settings = TerminalSettings()
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    return settings


def create_minimal_settings() -> TerminalSettings:
    """Create minimal settings (no visual fluff)"""
    settings = TerminalSettings()
    settings.display.show_status_bar = False
    settings.display.show_control_tips = False
    settings.display.show_timestamp = False
    settings.audio_display.show_input_level = False
    settings.behavior.show_welcome_message = False
    return settings


def create_debug_settings() -> TerminalSettings:
    """Create settings for debugging"""
    settings = TerminalSettings()
    settings.log_level = LogLevel.DEBUG
    settings.display.show_latency = True
    settings.display.show_token_count = True
    settings.display.show_connection_status = True
    settings.audio_display.show_input_level = True
    settings.audio_display.show_vad_status = True
    return settings