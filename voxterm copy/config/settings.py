"""
VoxTerm Configuration Settings

Defines all configuration options for VoxTerm.
Note: Audio/VAD settings are delegated to the voice engine.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class Theme(Enum):
    """Available color themes"""
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    CUSTOM = "custom"


class LogLevel(Enum):
    """Log verbosity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    NONE = "none"


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
    
    # UI controls
    interrupt: str = "escape"
    clear_screen: str = "ctrl+l"
    toggle_logs: str = "l"
    toggle_metrics: str = "s"
    
    # Navigation
    scroll_up: str = "up"
    scroll_down: str = "down"
    page_up: str = "pageup"
    page_down: str = "pagedown"
    
    # System
    quit: str = "q"
    help: str = "h"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class DisplaySettings:
    """Display configuration"""
    # Theme
    theme: Theme = Theme.DARK
    custom_theme: Optional[Dict[str, str]] = None
    
    # Layout
    show_status_bar: bool = True
    show_control_hints: bool = True
    show_timestamps: bool = True
    timestamp_format: str = "%H:%M:%S"
    
    # Message display
    max_message_length: int = 1000
    message_wrap_width: Optional[int] = None  # None = terminal width
    show_role_labels: bool = True
    
    # Metrics display
    show_latency: bool = True
    show_audio_levels: bool = True
    show_token_count: bool = False
    
    # Animation
    enable_animations: bool = True
    typing_indicator: bool = True
    
    # Window
    min_terminal_width: int = 80
    min_terminal_height: int = 24


@dataclass
class AudioDisplaySettings:
    """Audio visualization settings"""
    # Note: Actual audio processing is handled by voice engine
    show_volume_meter: bool = True
    volume_meter_width: int = 20
    volume_meter_chars: str = "▁▂▃▄▅▆▇█"
    
    show_waveform: bool = False
    waveform_height: int = 3


@dataclass
class VoiceSettings:
    """Voice configuration (passed to voice engine)"""
    # These are just for display/UI purposes
    # Actual voice configuration is in the voice engine
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
    
    # Confirmations
    confirm_quit: bool = True
    confirm_clear: bool = True
    
    # Limits
    max_history_size: int = 1000
    max_log_entries: int = 500
    
    # Timeouts (UI-related only)
    status_message_timeout: float = 3.0
    error_message_timeout: float = 5.0


@dataclass
class TerminalSettings:
    """Complete terminal configuration"""
    # Components
    key_bindings: KeyBindings = field(default_factory=KeyBindings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    audio_display: AudioDisplaySettings = field(default_factory=AudioDisplaySettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    
    # Performance
    ui_update_rate_hz: float = 30.0  # Max UI updates per second
    event_queue_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key_bindings": self.key_bindings.to_dict(),
            "display": {
                "theme": self.display.theme.value,
                "show_status_bar": self.display.show_status_bar,
                "show_timestamps": self.display.show_timestamps,
                # ... other display settings
            },
            "voice": {
                "current_voice": self.voice.current_voice,
                "current_language": self.voice.current_language,
            },
            "log_level": self.log_level.value,
            # ... other settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TerminalSettings':
        """Create from dictionary"""
        settings = cls()
        
        # Update key bindings
        if "key_bindings" in data:
            for key, value in data["key_bindings"].items():
                if hasattr(settings.key_bindings, key):
                    setattr(settings.key_bindings, key, value)
        
        # Update display settings
        if "display" in data:
            if "theme" in data["display"]:
                settings.display.theme = Theme(data["display"]["theme"])
            # ... update other display settings
        
        # ... update other components
        
        return settings


# Preset configurations
PRESETS = {
    "minimal": TerminalSettings(
        display=DisplaySettings(
            show_status_bar=False,
            show_control_hints=False,
            show_timestamps=False,
            show_latency=False,
            show_audio_levels=False,
        ),
        audio_display=AudioDisplaySettings(
            show_volume_meter=False,
        ),
    ),
    
    "developer": TerminalSettings(
        display=DisplaySettings(
            show_timestamps=True,
            show_latency=True,
            show_token_count=True,
        ),
        log_level=LogLevel.DEBUG,
        behavior=BehaviorSettings(
            max_log_entries=2000,
        ),
    ),
    
    "accessibility": TerminalSettings(
        display=DisplaySettings(
            theme=Theme.HIGH_CONTRAST,
            enable_animations=False,
        ),
        key_bindings=KeyBindings(
            # Larger keys for easier access
            push_to_talk="enter",
            mute_toggle="f1",
        ),
    ),
}