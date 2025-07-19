"""
VoxTerm Configuration Manager

Manages runtime configuration with live updates and validation.
"""

import threading
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import json
import os
from dataclasses import asdict

from .settings import TerminalSettings, PRESETS
from ..core.events import Event, EventType, emit_event


class ConfigManager:
    """
    Configuration manager for VoxTerm.
    
    Features:
    - Thread-safe configuration access
    - Live configuration updates
    - Validation
    - Change notifications
    - Preset support
    """
    
    def __init__(self, initial_config: Optional[TerminalSettings] = None):
        self._config = initial_config or TerminalSettings()
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
        self._validators: Dict[str, Callable] = {}
        
        # Register default validators
        self._register_default_validators()
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by path.
        
        Example: get('display.theme') or get('key_bindings.quit')
        """
        with self._lock:
            try:
                value = self._config
                for part in path.split('.'):
                    value = getattr(value, part)
                return value
            except AttributeError:
                return default
    
    def set(self, path: str, value: Any) -> bool:
        """
        Set a configuration value by path.
        
        Returns True if successful, False if validation failed.
        """
        with self._lock:
            # Validate first
            if not self._validate(path, value):
                return False
            
            old_config = self._copy_config()
            
            try:
                # Navigate to the parent object
                parts = path.split('.')
                obj = self._config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                
                # Set the value
                setattr(obj, parts[-1], value)
                
                # Notify observers
                self._notify_observers(path, value, old_config)
                
                # Emit configuration change event
                emit_event(Event(
                    type=EventType.INFO,
                    source="ConfigManager",
                    data={"path": path, "value": value}
                ))
                
                return True
                
            except Exception as e:
                # Restore old config on error
                self._config = old_config
                return False
    
    def update(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """
        Update multiple configuration values.
        
        Returns dict of path -> success for each update.
        """
        results = {}
        for path, value in updates.items():
            results[path] = self.set(path, value)
        return results
    
    def load_preset(self, preset_name: str) -> bool:
        """Load a configuration preset"""
        if preset_name not in PRESETS:
            return False
        
        with self._lock:
            self._config = PRESETS[preset_name]
            self._notify_observers("*", preset_name, None)
            return True
    
    def load_from_file(self, file_path: Path) -> bool:
        """Load configuration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                self._config = TerminalSettings.from_dict(data)
                self._notify_observers("*", "file_load", None)
                return True
                
        except Exception as e:
            emit_event(Event(
                type=EventType.ERROR,
                source="ConfigManager",
                data={"error": f"Failed to load config: {e}"}
            ))
            return False
    
    def save_to_file(self, file_path: Path) -> bool:
        """Save configuration to JSON file"""
        try:
            with self._lock:
                data = self._config.to_dict()
            
            os.makedirs(file_path.parent, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            emit_event(Event(
                type=EventType.ERROR,
                source="ConfigManager",
                data={"error": f"Failed to save config: {e}"}
            ))
            return False
    
    def get_all(self) -> TerminalSettings:
        """Get a copy of all settings"""
        with self._lock:
            return self._copy_config()
    
    def observe(self, callback: Callable[[str, Any, Optional[TerminalSettings]], None]):
        """
        Add a configuration observer.
        
        Callback signature: callback(path, new_value, old_config)
        """
        self._observers.append(callback)
    
    def unobserve(self, callback: Callable):
        """Remove a configuration observer"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def add_validator(self, path: str, validator: Callable[[Any], bool]):
        """
        Add a custom validator for a configuration path.
        
        Validator should return True if value is valid.
        """
        self._validators[path] = validator
    
    def _validate(self, path: str, value: Any) -> bool:
        """Validate a configuration value"""
        # Check custom validators
        if path in self._validators:
            try:
                return self._validators[path](value)
            except Exception:
                return False
        
        # Check parent path validators (e.g., 'display' for 'display.theme')
        parts = path.split('.')
        for i in range(len(parts)):
            parent_path = '.'.join(parts[:i+1])
            if parent_path in self._validators:
                try:
                    if not self._validators[parent_path](value):
                        return False
                except Exception:
                    return False
        
        return True
    
    def _register_default_validators(self):
        """Register default validators"""
        # UI update rate must be positive
        self.add_validator(
            'ui_update_rate_hz',
            lambda v: isinstance(v, (int, float)) and v > 0
        )
        
        # Queue size must be positive
        self.add_validator(
            'event_queue_size',
            lambda v: isinstance(v, int) and v > 0
        )
        
        # Key bindings must be non-empty strings
        def validate_key_binding(v):
            return isinstance(v, str) and len(v) > 0
        
        for attr in dir(self._config.key_bindings):
            if not attr.startswith('_'):
                self.add_validator(f'key_bindings.{attr}', validate_key_binding)
    
    def _copy_config(self) -> TerminalSettings:
        """Create a deep copy of the configuration"""
        import copy
        return copy.deepcopy(self._config)
    
    def _notify_observers(self, path: str, value: Any, old_config: Optional[TerminalSettings]):
        """Notify all observers of configuration change"""
        for observer in self._observers:
            try:
                # Run in thread to avoid blocking
                threading.Thread(
                    target=observer,
                    args=(path, value, old_config),
                    daemon=True
                ).start()
            except Exception:
                pass  # Ignore observer errors


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return get_config_manager().get(path, default)


def set_config(path: str, value: Any) -> bool:
    """Convenience function to set configuration value"""
    return get_config_manager().set(path, value)