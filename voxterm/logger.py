"""
Simple logging wrapper for VoxTerm

Logs all output to a file with timestamps when enabled.
"""

import sys
from datetime import datetime
from typing import Optional, TextIO


class SimpleLogger:
    """Simple logger that writes to both console and file"""
    
    def __init__(self, log_file_path: Optional[str] = None):
        self.log_file_path = log_file_path
        self.file_handle: Optional[TextIO] = None
        self.enabled = False
        
        # Store original print and stdout
        self._original_print = print
        self._original_stdout = sys.stdout
        
    def enable(self, log_file_path: str):
        """Enable logging to file"""
        try:
            self.log_file_path = log_file_path
            self.file_handle = open(log_file_path, 'a', encoding='utf-8')
            self.enabled = True
            
            # Write header
            self.file_handle.write(f"\n{'='*60}\n")
            self.file_handle.write(f"VoxTerm Session Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.write(f"{'='*60}\n\n")
            self.file_handle.flush()
            
        except Exception as e:
            print(f"Failed to enable logging: {e}")
            self.enabled = False
            
    def disable(self):
        """Disable logging and close file"""
        if self.file_handle:
            try:
                self.file_handle.write(f"\n{'='*60}\n")
                self.file_handle.write(f"Session ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.file_handle.write(f"{'='*60}\n")
                self.file_handle.close()
            except:
                pass
                
        self.file_handle = None
        self.enabled = False
        
    def write(self, text: str, include_timestamp: bool = True):
        """Write to both console and file"""
        # Always write to console
        print(text, end='')
        
        # Write to file if enabled
        if self.enabled and self.file_handle:
            try:
                if include_timestamp and text.strip():
                    # Add timestamp for non-empty lines
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.file_handle.write(f"[{timestamp}] {text}")
                else:
                    self.file_handle.write(text)
                    
                self.file_handle.flush()
            except:
                # Silently fail to not interrupt the session
                pass
                
    def log_event(self, event_type: str, message: str):
        """Log a specific event with clear formatting"""
        if self.enabled and self.file_handle:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
                self.file_handle.write(f"\n[{timestamp}] {event_type}: {message}\n")
                self.file_handle.flush()
            except:
                pass
                
    def log_audio_event(self, event: str, duration: Optional[float] = None, size: Optional[int] = None):
        """Log audio-specific events"""
        details = []
        if duration:
            details.append(f"duration={duration:.2f}s")
        if size:
            details.append(f"size={size} bytes")
            
        detail_str = f" ({', '.join(details)})" if details else ""
        self.log_event("AUDIO", f"{event}{detail_str}")
        
    def log_api_event(self, event: str, method: Optional[str] = None):
        """Log API interaction events"""
        if method:
            self.log_event("API", f"{event} - {method}")
        else:
            self.log_event("API", event)
            
    def log_user_action(self, action: str):
        """Log user actions"""
        self.log_event("USER", action)


# Global logger instance
logger = SimpleLogger()


def setup_logging(settings):
    """Setup logging based on settings"""
    global logger
    
    if settings.log_to_file and settings.log_file_path:
        logger.enable(settings.log_file_path)
    else:
        logger.disable()
        
        
def log_print(*args, **kwargs):
    """Replacement for print that also logs"""
    # Convert args to string like print does
    text = ' '.join(str(arg) for arg in args)
    end = kwargs.get('end', '\n')
    
    # Write through logger
    logger.write(text + end)
    
    
# Function to patch print for logging
def patch_print():
    """Replace built-in print with logging version"""
    import builtins
    builtins.print = log_print
    
    
def restore_print():
    """Restore original print"""
    import builtins
    builtins.print = logger._original_print