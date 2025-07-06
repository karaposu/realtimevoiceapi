# realtimevoiceapi/session.py
"""Session configuration and management for OpenAI Realtime API"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from .models import (
    AudioFormatType, ModalityType, VoiceType, ToolChoiceType,
    TurnDetectionConfig, TranscriptionConfig, Tool
)


@dataclass
class SessionConfig:
    """Configuration for a Realtime API session"""
    
    # Model configuration
    model: str = "gpt-4o-realtime-preview"
    modalities: List[ModalityType] = field(default_factory=lambda: ["text", "audio"])
    
    # Instructions and behavior
    instructions: str = "You are a helpful assistant."
    voice: VoiceType = "alloy"
    
    # Audio settings
    input_audio_format: AudioFormatType = "pcm16"
    output_audio_format: AudioFormatType = "pcm16"
    input_audio_transcription: Optional[Union[TranscriptionConfig, Dict[str, Any]]] = None
    
    # Response configuration
    temperature: float = 0.8
    max_response_output_tokens: Union[int, str] = "inf"
    speed: float = 1.0
    
    # Turn detection - Updated to handle API requirements
    turn_detection: Optional[Union[TurnDetectionConfig, Dict[str, Any]]] = field(
        default_factory=lambda: TurnDetectionConfig()
    )
    
    # Tools
    tools: List[Tool] = field(default_factory=list)
    tool_choice: ToolChoiceType = "auto"
    
    # Logger
    logger: Optional[logging.Logger] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize logger after creation"""
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        result = {
            "model": self.model,
            "modalities": self.modalities,
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
            "speed": self.speed,
            "tool_choice": self.tool_choice
        }
        
        # Handle input audio transcription
        if self.input_audio_transcription:
            if isinstance(self.input_audio_transcription, dict):
                # Handle dictionary input
                transcription_config = {
                    "model": self.input_audio_transcription.get("model", "whisper-1"),
                    "prompt": self.input_audio_transcription.get("prompt", "")
                }
                # Only include language if it's specified and not None
                if self.input_audio_transcription.get("language"):
                    transcription_config["language"] = self.input_audio_transcription.get("language")
                result["input_audio_transcription"] = transcription_config
            else:
                # Handle TranscriptionConfig object
                transcription_config = {
                    "model": self.input_audio_transcription.model,
                    "prompt": self.input_audio_transcription.prompt
                }
                # Only include language if it's specified and not None
                if self.input_audio_transcription.language:
                    transcription_config["language"] = self.input_audio_transcription.language
                result["input_audio_transcription"] = transcription_config
        
        # FIXED: Handle turn_detection - API requires server_vad or semantic_vad
        if self.turn_detection is None:
            # Default to server_vad if None
            if self.logger:
                self.logger.info("No turn_detection specified, defaulting to server_vad")
            result["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True
            }
        elif isinstance(self.turn_detection, dict):
            # Handle dictionary input
            td_type = self.turn_detection.get("type", "server_vad")
            
            # Validate type - MUST be server_vad or semantic_vad
            if td_type not in ["server_vad", "semantic_vad"]:
                if self.logger:
                    self.logger.warning(
                        f"Invalid turn_detection type '{td_type}', defaulting to 'server_vad'. "
                        f"Valid options are: 'server_vad', 'semantic_vad'"
                    )
                td_type = "server_vad"
            
            # Build config based on type
            td_config = {
                "type": td_type,
                "prefix_padding_ms": self.turn_detection.get("prefix_padding_ms", 300),
                "silence_duration_ms": self.turn_detection.get("silence_duration_ms", 500),
                "create_response": self.turn_detection.get("create_response", True)
            }
            
            # Only include threshold for server_vad
            if td_type == "server_vad":
                td_config["threshold"] = self.turn_detection.get("threshold", 0.5)
            
            result["turn_detection"] = td_config
            
        else:
            # Handle TurnDetectionConfig object - USE ITS to_dict() METHOD
            td_type = self.turn_detection.type
            
            # Validate type - MUST be server_vad or semantic_vad
            if td_type not in ["server_vad", "semantic_vad"]:
                if self.logger:
                    self.logger.warning(
                        f"Invalid turn_detection type '{td_type}', defaulting to 'server_vad'. "
                        f"Valid options are: 'server_vad', 'semantic_vad'"
                    )
                # Create a new config with corrected type
                corrected_config = TurnDetectionConfig(
                    type="server_vad",
                    threshold=0.5,
                    prefix_padding_ms=self.turn_detection.prefix_padding_ms,
                    silence_duration_ms=self.turn_detection.silence_duration_ms,
                    create_response=self.turn_detection.create_response
                )
                result["turn_detection"] = corrected_config.to_dict()
            else:
                # Use the TurnDetectionConfig's to_dict() method which handles threshold exclusion
                result["turn_detection"] = self.turn_detection.to_dict()
        
        # Handle tools
        if self.tools:
            result["tools"] = [
                {
                    "type": tool.type,
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in self.tools
            ]
        
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionConfig':
        """Create SessionConfig from dictionary"""
        # Extract turn detection config
        turn_detection = None
        if "turn_detection" in data and data["turn_detection"]:
            td_data = data["turn_detection"]
            # Ensure valid type
            td_type = td_data.get("type", "server_vad")
            if td_type not in ["server_vad", "semantic_vad"]:
                td_type = "server_vad"
                
            turn_detection = TurnDetectionConfig(
                type=td_type,
                threshold=td_data.get("threshold", 0.5),
                prefix_padding_ms=td_data.get("prefix_padding_ms", 300),
                silence_duration_ms=td_data.get("silence_duration_ms", 500),
                create_response=td_data.get("create_response", True)
            )
        
        # Extract transcription config
        transcription = None
        if "input_audio_transcription" in data and data["input_audio_transcription"]:
            tc_data = data["input_audio_transcription"]
            transcription = TranscriptionConfig(
                model=tc_data.get("model", "whisper-1"),
                language=tc_data.get("language"),
                prompt=tc_data.get("prompt", "")
            )
        
        # Extract tools
        tools = []
        if "tools" in data and data["tools"]:
            for tool_data in data["tools"]:
                tools.append(Tool(
                    type=tool_data.get("type", "function"),
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    parameters=tool_data.get("parameters", {})
                ))
        
        return cls(
            model=data.get("model", "gpt-4o-realtime-preview"),
            modalities=data.get("modalities", ["text", "audio"]),
            instructions=data.get("instructions", "You are a helpful assistant."),
            voice=data.get("voice", "alloy"),
            input_audio_format=data.get("input_audio_format", "pcm16"),
            output_audio_format=data.get("output_audio_format", "pcm16"),
            input_audio_transcription=transcription,
            temperature=data.get("temperature", 0.8),
            max_response_output_tokens=data.get("max_response_output_tokens", "inf"),
            speed=data.get("speed", 1.0),
            turn_detection=turn_detection,
            tools=tools,
            tool_choice=data.get("tool_choice", "auto")
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Validate temperature
        if self.temperature < 0.6 or self.temperature > 1.2:
            issues.append(f"Temperature {self.temperature} out of range [0.6, 1.2]")
        
        # Validate speed
        if self.speed < 0.25 or self.speed > 1.5:
            issues.append(f"Speed {self.speed} out of range [0.25, 1.5]")
        
        # Validate modalities
        if not self.modalities:
            issues.append("At least one modality must be specified")
        
        # Validate audio format if audio modality is enabled
        if "audio" in self.modalities:
            if self.input_audio_format not in ["pcm16", "g711_ulaw", "g711_alaw"]:
                issues.append(f"Invalid input audio format: {self.input_audio_format}")
            if self.output_audio_format not in ["pcm16", "g711_ulaw", "g711_alaw"]:
                issues.append(f"Invalid output audio format: {self.output_audio_format}")
            if self.voice not in ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]:
                issues.append(f"Invalid voice: {self.voice}")
        
        # Validate turn detection
        if self.turn_detection:
            if isinstance(self.turn_detection, TurnDetectionConfig):
                if self.turn_detection.type not in ["server_vad", "semantic_vad"]:
                    issues.append(f"Invalid turn detection type: {self.turn_detection.type}")
                if self.turn_detection.threshold is not None and (self.turn_detection.threshold < 0 or self.turn_detection.threshold > 1):
                    issues.append(f"Turn detection threshold {self.turn_detection.threshold} out of range [0, 1]")
        
        return issues


# Preset configurations for common use cases
class SessionPresets:
    """Pre-configured session profiles for common use cases"""
    
    @staticmethod
    def voice_assistant() -> SessionConfig:
        """Standard voice assistant configuration"""
        return SessionConfig(
            instructions="You are a helpful voice assistant. Be concise and friendly.",
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.8,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                silence_duration_ms=500,
                create_response=True
            )
        )
    
    @staticmethod
    def transcription_service() -> SessionConfig:
        """Configuration optimized for transcription"""
        return SessionConfig(
            instructions="Transcribe audio accurately. Do not add any commentary.",
            modalities=["text"],
            input_audio_transcription=TranscriptionConfig(
                model="whisper-1",
                prompt="Transcribe exactly what is said."
            ),
            temperature=0.3,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.3,
                create_response=True
            )
        )
    
    @staticmethod
    def conversational_ai() -> SessionConfig:
        """Configuration for natural conversation"""
        return SessionConfig(
            instructions="You are a conversational AI. Engage naturally and ask follow-up questions when appropriate.",
            modalities=["text", "audio"],
            voice="alloy",
            temperature=0.9,
            speed=1.0,
            turn_detection=TurnDetectionConfig(
                type="semantic_vad",  # Use semantic VAD for more natural turn-taking
                threshold=None,       # No threshold for semantic VAD
                silence_duration_ms=700,
                create_response=True
            )
        )
    
    @staticmethod
    def customer_service() -> SessionConfig:
        """Configuration for customer service applications"""
        return SessionConfig(
            instructions="""You are a professional customer service representative. 
            Be polite, helpful, and solution-oriented. Ask clarifying questions when needed.""",
            modalities=["text", "audio"],
            voice="shimmer",  # Professional sounding voice
            temperature=0.7,
            speed=0.95,  # Slightly slower for clarity
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.4,
                silence_duration_ms=800,  # Longer pause for customer service
                create_response=True
            )
        )
    
    @staticmethod
    def audio_only() -> SessionConfig:
        """Configuration for audio-only interaction"""
        return SessionConfig(
            instructions="Communicate only through speech. Be clear and expressive.",
            modalities=["audio"],  # Audio only
            voice="nova",
            temperature=0.8,
            speed=1.0,
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                silence_duration_ms=600,
                create_response=True
            )
        )


# Configuration Management
class ConfigManager:
    """Manages configuration profiles and validation"""
    
    def __init__(self):
        self.profiles = {}
        self.load_default_profiles()
        
    def load_default_profiles(self):
        """Load default configuration profiles"""
        self.profiles = {
            "voice_assistant": {
                "config": SessionPresets.voice_assistant(),
                "description": "Standard voice assistant with balanced settings",
                "created_at": time.time()
            },
            "transcription": {
                "config": SessionPresets.transcription_service(),
                "description": "Optimized for accurate audio transcription",
                "created_at": time.time()
            },
            "conversational": {
                "config": SessionPresets.conversational_ai(),
                "description": "Natural conversation with semantic turn detection",
                "created_at": time.time()
            },
            "customer_service": {
                "config": SessionPresets.customer_service(),
                "description": "Professional customer service configuration",
                "created_at": time.time()
            },
            "audio_only": {
                "config": SessionPresets.audio_only(),
                "description": "Audio-only interaction mode",
                "created_at": time.time()
            }
        }
        
    def create_profile(self, name: str, config: SessionConfig, 
                      description: str = "") -> None:
        """Create and save a configuration profile"""
        # Validate configuration
        issues = config.validate()
        if issues:
            raise ValueError(f"Invalid config: {issues}")
            
        self.profiles[name] = {
            "config": config,
            "description": description,
            "created_at": time.time()
        }
        
    def get_profile(self, name: str) -> Optional[SessionConfig]:
        """Get a configuration profile by name"""
        profile = self.profiles.get(name)
        return profile["config"] if profile else None
        
    def list_profiles(self) -> Dict[str, str]:
        """List all available profiles with descriptions"""
        return {
            name: profile["description"] 
            for name, profile in self.profiles.items()
        }
        
    def get_optimized_config(self, use_case: str, 
                           hardware_profile: str = "standard") -> SessionConfig:
        """Get optimized config for specific use case and hardware"""
        base_config = self.profiles.get(use_case, {}).get("config")
        if not base_config:
            raise ValueError(f"Unknown use case: {use_case}")
            
        # For now, return base config (optimization can be added later)
        return base_config
        
    def validate_all_profiles(self) -> Dict[str, List[str]]:
        """Validate all profiles and return any issues"""
        results = {}
        for name, profile in self.profiles.items():
            issues = profile["config"].validate()
            if issues:
                results[name] = issues
        return results