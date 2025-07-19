"""
VoxTerm Text Formatting Utilities

Provides text formatting helpers for terminal output.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import re


class FormatStyle(Enum):
    """Text formatting styles"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    RICH = "rich"


class ANSICode:
    """ANSI escape codes for terminal formatting"""
    # Reset
    RESET = "\033[0m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class TextFormatter:
    """Base text formatter"""
    
    def __init__(self, style: FormatStyle = FormatStyle.PLAIN):
        self.style = style
    
    def format(self, text: str, **options) -> str:
        """Format text based on style"""
        if self.style == FormatStyle.PLAIN:
            return self._format_plain(text, **options)
        elif self.style == FormatStyle.MARKDOWN:
            return self._format_markdown(text, **options)
        elif self.style == FormatStyle.RICH:
            return self._format_rich(text, **options)
        else:
            return text
    
    def _format_plain(self, text: str, **options) -> str:
        """Plain text formatting (no special formatting)"""
        return text
    
    def _format_markdown(self, text: str, **options) -> str:
        """Convert markdown to terminal formatting"""
        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', f'{ANSICode.BOLD}\\1{ANSICode.RESET}', text)
        text = re.sub(r'__(.*?)__', f'{ANSICode.BOLD}\\1{ANSICode.RESET}', text)
        
        # Italic
        text = re.sub(r'\*(.*?)\*', f'{ANSICode.ITALIC}\\1{ANSICode.RESET}', text)
        text = re.sub(r'_(.*?)_', f'{ANSICode.ITALIC}\\1{ANSICode.RESET}', text)
        
        # Code
        text = re.sub(r'`(.*?)`', f'{ANSICode.CYAN}\\1{ANSICode.RESET}', text)
        
        # Headers
        text = re.sub(r'^# (.*?)$', f'{ANSICode.BOLD}{ANSICode.BLUE}\\1{ANSICode.RESET}', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', f'{ANSICode.BOLD}{ANSICode.GREEN}\\1{ANSICode.RESET}', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.*?)$', f'{ANSICode.BOLD}{ANSICode.YELLOW}\\1{ANSICode.RESET}', text, flags=re.MULTILINE)
        
        return text
    
    def _format_rich(self, text: str, **options) -> str:
        """Rich text formatting with custom tags"""
        # Custom tags like [red]text[/red]
        color_pattern = r'\[(\w+)\](.*?)\[/\1\]'
        
        def replace_color(match):
            color = match.group(1).upper()
            content = match.group(2)
            
            if hasattr(ANSICode, color):
                return f"{getattr(ANSICode, color)}{content}{ANSICode.RESET}"
            else:
                return content
        
        text = re.sub(color_pattern, replace_color, text)
        
        return text


class MessageFormatter:
    """Format messages for terminal display"""
    
    def __init__(self, show_timestamps: bool = True, timestamp_format: str = "%H:%M:%S"):
        self.show_timestamps = show_timestamps
        self.timestamp_format = timestamp_format
        self.text_formatter = TextFormatter(FormatStyle.MARKDOWN)
    
    def format_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Format a message for display.
        
        Returns list of formatted lines.
        """
        lines = []
        
        # Timestamp
        prefix_parts = []
        if self.show_timestamps and timestamp:
            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime(self.timestamp_format)
            prefix_parts.append(f"{ANSICode.DIM}[{time_str}]{ANSICode.RESET}")
        
        # Role with color
        role_colors = {
            "user": ANSICode.CYAN,
            "assistant": ANSICode.GREEN,
            "system": ANSICode.YELLOW,
            "error": ANSICode.RED
        }
        
        role_color = role_colors.get(role.lower(), ANSICode.WHITE)
        role_text = role.title() if role != "assistant" else "AI"
        prefix_parts.append(f"{role_color}{role_text}:{ANSICode.RESET}")
        
        # Metadata indicators
        if metadata:
            if metadata.get("thinking"):
                prefix_parts.append(f"{ANSICode.DIM}[thinking]{ANSICode.RESET}")
            if metadata.get("tool"):
                prefix_parts.append(f"{ANSICode.MAGENTA}[{metadata['tool']}]{ANSICode.RESET}")
        
        # Build prefix
        prefix = " ".join(prefix_parts) + " "
        
        # Format content
        formatted_content = self.text_formatter.format(content)
        
        # Split into lines
        content_lines = formatted_content.split('\n')
        
        # Add first line with prefix
        if content_lines:
            lines.append(prefix + content_lines[0])
            
            # Add remaining lines with indent
            indent = " " * self._visible_length(prefix)
            for line in content_lines[1:]:
                lines.append(indent + line)
        
        return lines
    
    def format_partial_message(
        self,
        role: str,
        content: str,
        is_typing: bool = True
    ) -> List[str]:
        """Format a partial/streaming message"""
        lines = self.format_message(role, content)
        
        if is_typing and lines:
            # Add typing indicator to last line
            lines[-1] += f" {ANSICode.BLINK}▌{ANSICode.RESET}"
        
        return lines
    
    def _visible_length(self, text: str) -> int:
        """Calculate visible length of text (excluding ANSI codes)"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return len(ansi_escape.sub('', text))


class LogFormatter:
    """Format log entries for terminal display"""
    
    def __init__(self, show_timestamp: bool = True, show_level: bool = True):
        self.show_timestamp = show_timestamp
        self.show_level = show_level
    
    def format_log(
        self,
        level: str,
        message: str,
        source: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> str:
        """Format a log entry"""
        parts = []
        
        # Timestamp
        if self.show_timestamp:
            if timestamp is None:
                timestamp = datetime.now()
            else:
                timestamp = datetime.fromtimestamp(timestamp)
            
            time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
            parts.append(f"{ANSICode.DIM}{time_str}{ANSICode.RESET}")
        
        # Level with color
        if self.show_level:
            level_colors = {
                "DEBUG": ANSICode.DIM,
                "INFO": ANSICode.BLUE,
                "WARNING": ANSICode.YELLOW,
                "ERROR": ANSICode.RED,
                "CRITICAL": ANSICode.BG_RED + ANSICode.WHITE
            }
            
            level_color = level_colors.get(level.upper(), ANSICode.WHITE)
            level_text = level.upper()[:5].ljust(5)  # Fixed width
            parts.append(f"{level_color}{level_text}{ANSICode.RESET}")
        
        # Source
        if source:
            parts.append(f"{ANSICode.CYAN}[{source}]{ANSICode.RESET}")
        
        # Message
        parts.append(message)
        
        return " ".join(parts)


class ProgressFormatter:
    """Format progress indicators"""
    
    @staticmethod
    def format_bar(
        current: float,
        total: float,
        width: int = 20,
        filled_char: str = "█",
        empty_char: str = "░"
    ) -> str:
        """Format a progress bar"""
        if total == 0:
            percentage = 0
        else:
            percentage = current / total
        
        filled_width = int(width * percentage)
        empty_width = width - filled_width
        
        bar = filled_char * filled_width + empty_char * empty_width
        percentage_text = f"{percentage * 100:.1f}%"
        
        return f"[{bar}] {percentage_text}"
    
    @staticmethod
    def format_spinner(frame: int, style: str = "dots") -> str:
        """Format a spinner animation"""
        spinners = {
            "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            "line": ["-", "\\", "|", "/"],
            "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
            "pulse": ["·", "•", "●", "•"],
        }
        
        frames = spinners.get(style, spinners["dots"])
        return frames[frame % len(frames)]


class AudioLevelFormatter:
    """Format audio level indicators"""
    
    @staticmethod
    def format_meter(
        level: float,
        width: int = 20,
        chars: str = "▁▂▃▄▅▆▇█"
    ) -> str:
        """Format an audio level meter"""
        # Ensure level is between 0 and 1
        level = max(0.0, min(1.0, level))
        
        # Calculate how many segments to fill
        segments = int(level * width)
        
        # Build meter
        meter = ""
        for i in range(width):
            if i < segments:
                # Map position to character intensity
                char_index = min(int(i / width * len(chars)), len(chars) - 1)
                meter += chars[char_index]
            else:
                meter += " "
        
        # Add color based on level
        if level > 0.9:
            color = ANSICode.RED
        elif level > 0.7:
            color = ANSICode.YELLOW
        else:
            color = ANSICode.GREEN
        
        return f"{color}[{meter}]{ANSICode.RESET}"
    
    @staticmethod
    def format_waveform(
        samples: List[float],
        width: int = 40,
        height: int = 5
    ) -> List[str]:
        """Format a simple waveform visualization"""
        if not samples:
            return [" " * width] * height
        
        # Downsample to fit width
        step = max(1, len(samples) // width)
        downsampled = samples[::step][:width]
        
        # Normalize to height
        max_val = max(abs(s) for s in downsampled) if downsampled else 1
        if max_val > 0:
            normalized = [int(s / max_val * (height // 2)) for s in downsampled]
        else:
            normalized = [0] * len(downsampled)
        
        # Build waveform lines
        lines = []
        for y in range(height, 0, -1):
            line = ""
            for x, val in enumerate(normalized):
                if y == height // 2 + 1:  # Center line
                    line += "─"
                elif abs(val) >= y - height // 2 - 1:
                    line += "█"
                else:
                    line += " "
            lines.append(f"{ANSICode.CYAN}{line}{ANSICode.RESET}")
        
        return lines