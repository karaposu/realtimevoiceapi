"""
VoiceEngine Integration for VoxTerm

Provides integration between VoxTerm and realtimevoiceapi.VoiceEngine.
"""

from typing import Any, Dict, Optional
import logging

from .base import BaseIntegration
from ..core.base import BaseTerminal

logger = logging.getLogger(__name__)


class VoiceEngineAdapter(BaseIntegration):
    """
    Adapter for integrating realtimevoiceapi.VoiceEngine with VoxTerm.
    
    This adapter handles the specific integration between VoxTerm's
    terminal interface and the VoiceEngine API.
    """
    
    def __init__(self):
        self.engine_info = {
            "name": "realtimevoiceapi.VoiceEngine",
            "version": "1.0",
            "capabilities": [
                "streaming",
                "text_input",
                "audio_input",
                "audio_output",
                "vad",
                "interruption",
                "transcription"
            ]
        }
    
    def connect_to_terminal(self, terminal: BaseTerminal, engine: Any) -> None:
        """
        Connect VoiceEngine to VoxTerm terminal.
        
        Sets up all necessary callbacks and bindings between
        the engine and terminal components.
        """
        # Validate engine
        if not self.validate_engine(engine):
            raise ValueError("Invalid VoiceEngine instance")
        
        # Bind engine to terminal
        terminal.voice_engine = engine
        
        # Bind to components
        if hasattr(terminal, 'audio_input_manager'):
            terminal.audio_input_manager.bind_voice_engine(engine)
        
        if hasattr(terminal, 'audio_output_manager'):
            terminal.audio_output_manager.bind_voice_engine(engine)
        
        # Update modes with engine reference
        if hasattr(terminal, 'modes'):
            for mode in terminal.modes.values():
                mode.voice_engine = engine
        
        # Set up callbacks from engine to terminal
        self._setup_engine_callbacks(engine, terminal)
        
        logger.info("VoiceEngine connected to VoxTerm terminal")
    
    def _setup_engine_callbacks(self, engine: Any, terminal: BaseTerminal) -> None:
        """Set up VoiceEngine callbacks to terminal methods"""
        
        # Text response callback
        if hasattr(engine, 'on_text_response'):
            def on_text(text: str):
                # Update partial response in state
                from ..core.state import get_state_manager
                state = get_state_manager().get_state()
                state.conversation.partial_assistant_response += text
                
                # Trigger display update
                if hasattr(terminal, 'display'):
                    terminal.display.render({"section": "conversation"})
            
            engine.on_text_response = on_text
        
        # Audio response callback
        if hasattr(engine, 'on_audio_response'):
            def on_audio(audio: bytes):
                # Play audio through terminal's audio manager
                if hasattr(terminal, 'audio_output_manager'):
                    terminal.audio_output_manager.play_audio(audio)
            
            engine.on_audio_response = on_audio
        
        # User transcript callback
        if hasattr(engine, 'on_user_transcript'):
            def on_transcript(transcript: str):
                # Update user transcript in state
                from ..core.state import get_state_manager
                state = get_state_manager().get_state()
                state.conversation.partial_user_transcript = transcript
                
                # Trigger display update
                if hasattr(terminal, 'display'):
                    terminal.display.render({"section": "conversation"})
            
            engine.on_user_transcript = on_transcript
        
        # Response done callback
        if hasattr(engine, 'on_response_done'):
            def on_done():
                from ..core.state import get_state_manager
                state = get_state_manager().get_state()
                
                # Move partial responses to complete messages
                if state.conversation.partial_assistant_response:
                    get_state_manager().add_message(
                        "assistant",
                        state.conversation.partial_assistant_response
                    )
                    state.conversation.partial_assistant_response = ""
                
                if state.conversation.partial_user_transcript:
                    get_state_manager().add_message(
                        "user", 
                        state.conversation.partial_user_transcript
                    )
                    state.conversation.partial_user_transcript = ""
                
                # Notify current mode
                if hasattr(terminal, 'current_mode') and terminal.current_mode:
                    if hasattr(terminal.current_mode, 'on_response_complete'):
                        terminal.current_mode.on_response_complete()
                
                # Update display
                if hasattr(terminal, 'display'):
                    terminal.display.render({"section": "conversation"})
            
            engine.on_response_done = on_done
        
        # Error callback
        if hasattr(engine, 'on_error'):
            def on_error(error: Exception):
                from ..core.events import Event, EventType, emit_event
                emit_event(Event(
                    type=EventType.ERROR,
                    source="VoiceEngine",
                    data={"error": str(error)}
                ))
            
            engine.on_error = on_error
        
        # Connection state callback
        if hasattr(engine, 'on_connection_change'):
            def on_connection(connected: bool):
                from ..core.state import get_state_manager, ConnectionState
                state = connected and ConnectionState.CONNECTED or ConnectionState.DISCONNECTED
                get_state_manager().update_connection(state)
            
            engine.on_connection_change = on_connection
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get VoiceEngine information"""
        return self.engine_info
    
    def supports_feature(self, feature: str) -> bool:
        """Check if VoiceEngine supports a feature"""
        return feature in self.engine_info["capabilities"]
    
    def validate_engine(self, engine: Any) -> bool:
        """Validate VoiceEngine instance"""
        # Check for required methods
        required_methods = [
            'connect', 'disconnect', 'send_text'
        ]
        
        for method in required_methods:
            if not hasattr(engine, method):
                logger.error(f"VoiceEngine missing required method: {method}")
                return False
        
        # Check for callback attributes
        required_callbacks = [
            'on_text_response', 'on_audio_response', 
            'on_response_done', 'on_error'
        ]
        
        for callback in required_callbacks:
            if not hasattr(engine, callback):
                logger.warning(f"VoiceEngine missing callback: {callback}")
        
        return True
    
    def configure_for_terminal(self, engine: Any, terminal_settings: Dict[str, Any]) -> None:
        """Apply terminal settings to VoiceEngine"""
        # Apply voice setting if available
        if 'voice' in terminal_settings and hasattr(engine, 'config'):
            if hasattr(engine.config, 'voice'):
                engine.config.voice = terminal_settings['voice']['current_voice']
        
        # Apply other relevant settings as needed
        # This is where terminal preferences can influence engine behavior