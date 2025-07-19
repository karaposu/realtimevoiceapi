"""
Base Integration Class for VoxTerm

Provides the interface for integrating voice engines with VoxTerm.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.base import BaseTerminal


class BaseIntegration(ABC):
    """
    Abstract base class for voice engine integrations.
    
    Subclass this to create custom integrations for different voice APIs.
    """
    
    @abstractmethod
    def connect_to_terminal(self, terminal: BaseTerminal, engine: Any) -> None:
        """
        Connect a voice engine to the terminal.
        
        This method should:
        1. Bind the engine to terminal components
        2. Set up callbacks between engine and terminal
        3. Configure any engine-specific settings
        
        Args:
            terminal: The VoxTerm terminal instance
            engine: The voice engine instance
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the voice engine.
        
        Returns:
            Dict containing engine name, version, capabilities, etc.
        """
        pass
    
    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """
        Check if the engine supports a specific feature.
        
        Args:
            feature: Feature name (e.g., "streaming", "vad", "interruption")
            
        Returns:
            True if the feature is supported
        """
        pass
    
    def configure_for_terminal(self, engine: Any, terminal_settings: Dict[str, Any]) -> None:
        """
        Configure the engine based on terminal settings.
        
        Override this to apply terminal settings to the engine.
        
        Args:
            engine: The voice engine instance
            terminal_settings: Terminal configuration dict
        """
        pass
    
    def validate_engine(self, engine: Any) -> bool:
        """
        Validate that the engine has required methods/attributes.
        
        Override this to add engine-specific validation.
        
        Args:
            engine: The voice engine instance
            
        Returns:
            True if engine is valid
        """
        return True