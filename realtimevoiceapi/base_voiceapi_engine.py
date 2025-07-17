#!/usr/bin/env python3


# python -m realtimevoiceapi.base_voiceapi_engine
"""
Base VoiceAPI Engine - Comprehensive Framework for OpenAI Realtime API

This module provides a high-level, easy-to-use interface for all voice API features,
incorporating all tested functionality from the smoke tests.

Features:
- Simple and advanced audio input/output
- Text and voice conversations
- Multiple voice options
- Audio streaming
- Event handling
- Session management
- Audio format validation
- Buffer management
- Function calling support
"""
#!/usr/bin/env python3
"""
Base VoiceAPI Engine - Comprehensive Framework for OpenAI Realtime API

This module provides a high-level, easy-to-use interface for all voice API features,
incorporating all tested functionality from the smoke tests.

Features:
- Simple and advanced audio input/output
- Text and voice conversations
- Multiple voice options
- Audio streaming
- Event handling
- Session management
- Audio format validation
- Buffer management
- Function calling support
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Callable, Dict, Any, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import the core RealtimeVoiceAPI components
from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
from realtimevoiceapi.models import TurnDetectionConfig, Tool, FunctionCall
from realtimevoiceapi.audio import AudioConfig, AudioFormat
from realtimevoiceapi.session import SessionPresets


from dotenv import load_dotenv
load_dotenv()


import os
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)



class VoiceMode(Enum):
    """Voice interaction modes"""
    TEXT_ONLY = "text_only"
    VOICE_ONLY = "voice_only"
    TEXT_AND_VOICE = "text_and_voice"


class VADType(Enum):
    """Voice Activity Detection types"""
    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"


@dataclass
class VoiceResponse:
    """Structured response from voice API"""
    text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    audio_duration_ms: float = 0.0
    response_time_ms: float = 0.0
    events: List[str] = None
    success: bool = False
    error: Optional[str] = None


class BaseVoiceAPIEngine:
    """
    High-level voice API engine providing easy access to all features.
    
    This class encapsulates all the tested functionality from the smoke tests
    into a clean, flexible API for building voice applications.
    """
    
    def __init__(
        self,
        api_key: str,
        voice: str = "alloy",
        mode: VoiceMode = VoiceMode.TEXT_AND_VOICE,
        vad_type: VADType = VADType.SERVER_VAD,
        logger: Optional[logging.Logger] = None,
        auto_reconnect: bool = True
    ):
        """
        Initialize the Voice API Engine.
        
        Args:
            api_key: OpenAI API key with Realtime access
            voice: Voice to use (alloy, echo, shimmer, etc.)
            mode: Interaction mode
            vad_type: Voice Activity Detection type
            logger: Optional logger
            auto_reconnect: Enable auto-reconnection
        """
        self.api_key = api_key
        self.voice = voice
        self.mode = mode
        self.vad_type = vad_type
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize core components
        self.client = RealtimeClient(api_key, logger, auto_reconnect)
        self.audio_processor = AudioProcessor(logger)
        
        # State tracking
        self.is_connected = False
        self.current_session_config = None
        self.conversation_history = []
        
        # Response tracking
        self._current_response = VoiceResponse()
        self._response_handlers = {}
        self._function_handlers = {}
        
        # Audio recording for debugging
        self.save_audio_responses = False
        self.audio_output_dir = Path("voice_outputs")
    
    # === Connection Management ===
    
    async def connect(
        self,
        instructions: str = "You are a helpful voice assistant.",
        temperature: float = 0.8,
        max_tokens: Union[int, str] = "inf",
        **kwargs
    ) -> bool:
        """
        Connect to voice API with configuration.
        
        Args:
            instructions: System instructions
            temperature: Response randomness (0.6-1.2)
            max_tokens: Maximum response tokens
            **kwargs: Additional SessionConfig parameters
            
        Returns:
            True if connected successfully
        """
        print(f"🔌 Attempting to connect to Voice API...")
        print(f"   Voice: {self.voice}")
        print(f"   Mode: {self.mode.value}")
        print(f"   VAD Type: {self.vad_type.value}")
        
        try:
            # Build session configuration
            print("📋 Building session configuration...")
            config = self._build_session_config(
                instructions=instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            print(f"   Modalities: {config.modalities}")
            print(f"   Temperature: {config.temperature}")
            print(f"   Voice: {config.voice}")
            print(f"   Turn detection: {config.turn_detection.type if config.turn_detection else 'None'}")
            if config.turn_detection:
                print(f"   Auto response: {config.turn_detection.create_response}")
            
            # Log the full config dict
            config_dict = config.to_dict()
            print(f"   Full config: {config_dict}")
            
            # Set up internal event handlers
            print("🎯 Setting up event handlers...")
            self._setup_event_handlers()
            
            # Connect with configuration
            print(f"🌐 Connecting to OpenAI Realtime API...")
            self.logger.info(f"Connecting with voice '{self.voice}' and {self.vad_type.value}")
            await self.client.connect(config)
            
            self.is_connected = True
            self.current_session_config = config
            
            print("✅ Successfully connected to Voice API!")
            print(f"   Session active: {self.client.is_session_active}")
            print(f"   Session ID: {self.client.session_id}")
            
            # Give a moment for all events to settle
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            self.logger.error(f"Connection failed: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from voice API"""
        print("\n🔌 Disconnecting from Voice API...")
        print(f"   Current state: connected={self.is_connected}")
        print(f"   Conversation history: {len(self.conversation_history)} items")
        
        if self.client:
            await self.client.disconnect()
        
        self.is_connected = False
        print("✅ Disconnected successfully")
        self.logger.info("Disconnected from Voice API")
    
    async def reconnect(self) -> bool:
        """Reconnect with existing configuration"""
        if not self.current_session_config:
            raise ValueError("No previous session configuration")
        
        await self.disconnect()
        await asyncio.sleep(1)
        
        return await self.connect(
            instructions=self.current_session_config.instructions,
            temperature=self.current_session_config.temperature,
            max_tokens=self.current_session_config.max_response_output_tokens
        )
    
    # === Simple Conversation Methods ===
    
    async def send_text(self, text: str, wait_for_response: bool = True) -> VoiceResponse:
        """
        Send text message and optionally wait for response.
        
        Args:
            text: Text message to send
            wait_for_response: Whether to wait for complete response
            
        Returns:
            VoiceResponse with results
        """
        print(f"\n📤 Sending text: '{text}'")
        
        if not self.is_connected:
            print("❌ Not connected to Voice API!")
            raise RuntimeError("Not connected to Voice API")
        
        start_time = time.time()
        print("🔄 Resetting current response tracker...")
        self._reset_current_response()
        
        try:
            # Clear any previous audio in buffer
            if self.client.get_audio_output_duration() > 0:
                print("   🧹 Clearing previous audio from buffer")
                self.client.get_audio_output(clear_buffer=True)
            
            # Send text
            print("📨 Calling client.send_text()...")
            message_id = await self.client.send_text(text)
            print(f"✅ Text sent successfully! Message ID: {message_id}")
            
            # Check client state
            client_status = self.client.get_status()
            print(f"   Client status: connected={client_status['connected']}, session_active={client_status['session_active']}")
            print(f"   Pending responses: {client_status['pending_responses']}")
            
            self.logger.info(f"Sent text: {text[:50]}...")
            
            if wait_for_response:
                print("⏳ Waiting for response...")
                # Wait for response completion
                response = await self._wait_for_response(timeout=30)
                response.response_time_ms = (time.time() - start_time) * 1000
                
                print(f"📥 Response received:")
                print(f"   Success: {response.success}")
                print(f"   Text: {response.text[:100] if response.text else 'None'}")
                print(f"   Audio: {'Yes' if response.audio_bytes else 'No'} ({response.audio_duration_ms:.1f}ms)")
                print(f"   Response time: {response.response_time_ms:.1f}ms")
                if response.error:
                    print(f"   Error: {response.error}")
                
                return response
            else:
                print("✅ Text sent (not waiting for response)")
                return VoiceResponse(success=True)
                
        except Exception as e:
            print(f"❌ Failed to send text: {e}")
            print(f"   Error type: {type(e).__name__}")
            self.logger.error(f"Failed to send text: {e}")
            return VoiceResponse(success=False, error=str(e))
    
    async def send_audio(
        self,
        audio_bytes: bytes,
        wait_for_response: bool = True,
        timeout: float = 30.0
    ) -> VoiceResponse:
        """
        Send audio and optionally wait for response.
        
        Args:
            audio_bytes: PCM16 audio data at 24kHz
            wait_for_response: Whether to wait for response
            timeout: Maximum time to wait
            
        Returns:
            VoiceResponse with results
        """
        print(f"\n🎤 Sending audio: {len(audio_bytes)} bytes")
        
        if not self.is_connected:
            print("❌ Not connected to Voice API!")
            raise RuntimeError("Not connected to Voice API")
        
        start_time = time.time()
        print("🔄 Resetting current response tracker...")
        self._reset_current_response()
        
        try:
            # Validate audio
            print("🔍 Validating audio format...")
            is_valid, error_msg = self.audio_processor.validator.validate_audio_data(
                audio_bytes, AudioFormat.PCM16
            )
            if not is_valid:
                print(f"❌ Invalid audio format: {error_msg}")
                raise ValueError(f"Invalid audio: {error_msg}")
            
            duration_ms = self.audio_processor.get_audio_duration_ms(audio_bytes)
            print(f"✅ Audio validated: {duration_ms:.1f}ms duration")
            
            # Clear any previous audio in buffer
            if self.client.get_audio_output_duration() > 0:
                print("   🧹 Clearing previous audio from buffer")
                self.client.get_audio_output(clear_buffer=True)
            
            # Send audio using simple method
            print("📡 Using send_audio_simple()...")
            await self.client.send_audio_simple(audio_bytes)
            print("✅ Audio sent successfully")
            
            if wait_for_response:
                print("⏳ Waiting for response...")
                # Wait for response completion using our own handlers
                response = await self._wait_for_response(timeout=timeout)
                response.response_time_ms = (time.time() - start_time) * 1000
                
                print(f"📥 Audio response received:")
                print(f"   Success: {response.success}")
                print(f"   Text: {response.text[:100] if response.text else 'None'}")
                print(f"   Audio: {'Yes' if response.audio_bytes else 'No'} ({response.audio_duration_ms:.1f}ms)")
                print(f"   Response time: {response.response_time_ms:.1f}ms")
                if response.events:
                    print(f"   Events: {response.events}")
                if response.error:
                    print(f"   Error: {response.error}")
                
                return response
            else:
                print("✅ Audio sent (not waiting for response)")
                return VoiceResponse(success=True)
                
        except Exception as e:
            print(f"❌ Failed to send audio: {e}")
            print(f"   Error type: {type(e).__name__}")
            self.logger.error(f"Failed to send audio: {e}")
            return VoiceResponse(success=False, error=str(e))
    
    async def send_audio_file(
        self,
        file_path: Union[str, Path],
        wait_for_response: bool = True
    ) -> VoiceResponse:
        """
        Send audio file and optionally wait for response.
        
        Args:
            file_path: Path to audio file
            wait_for_response: Whether to wait for response
            
        Returns:
            VoiceResponse with results
        """
        try:
            # Load audio file
            audio_bytes = self.audio_processor.load_wav_file(file_path)
            return await self.send_audio(audio_bytes, wait_for_response)
            
        except Exception as e:
            self.logger.error(f"Failed to send audio file: {e}")
            return VoiceResponse(success=False, error=str(e))
    
    # === Advanced Audio Methods ===
    
    async def stream_audio(
        self,
        audio_bytes: bytes,
        chunk_size_ms: int = 100,
        real_time: bool = True
    ) -> VoiceResponse:
        """
        Stream audio in chunks.
        
        Args:
            audio_bytes: Audio data to stream
            chunk_size_ms: Size of each chunk in milliseconds
            real_time: Whether to simulate real-time streaming
            
        Returns:
            VoiceResponse with results
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Voice API")
        
        start_time = time.time()
        self._reset_current_response()
        
        try:
            # Stream audio chunks
            await self.client.send_audio_chunks(
                audio_bytes, chunk_size_ms, real_time
            )
            
            # Wait for response
            response = await self._wait_for_response(timeout=30)
            response.response_time_ms = (time.time() - start_time) * 1000
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to stream audio: {e}")
            return VoiceResponse(success=False, error=str(e))
    
    async def record_and_send(
        self,
        duration: float = 3.0,
        device: Optional[str] = None
    ) -> VoiceResponse:
        """
        Record audio from microphone and send.
        
        Args:
            duration: Recording duration in seconds
            device: Audio device to use (None for default)
            
        Returns:
            VoiceResponse with results
        """
        try:
            import sounddevice as sd
            import numpy as np
            
            self.logger.info(f"Recording {duration}s of audio...")
            
            # Record audio
            recording = sd.rec(
                int(duration * AudioConfig.SAMPLE_RATE),
                samplerate=AudioConfig.SAMPLE_RATE,
                channels=AudioConfig.CHANNELS,
                dtype='int16',
                device=device
            )
            sd.wait()
            
            # Convert to bytes
            audio_bytes = recording.tobytes()
            
            # Send recorded audio
            return await self.send_audio(audio_bytes)
            
        except ImportError:
            return VoiceResponse(
                success=False,
                error="sounddevice not installed. Install with: pip install sounddevice"
            )
        except Exception as e:
            return VoiceResponse(success=False, error=str(e))
    
    # === Voice Configuration ===
    
    async def change_voice(self, voice: str) -> bool:
        """
        Change the current voice.
        
        Args:
            voice: New voice (alloy, echo, shimmer, etc.)
            
        Returns:
            True if voice changed successfully
        """
        valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
        if voice not in valid_voices:
            raise ValueError(f"Invalid voice. Choose from: {valid_voices}")
        
        self.voice = voice
        
        if self.is_connected and self.current_session_config:
            # Update session with new voice
            self.current_session_config.voice = voice
            await self.client.configure_session(self.current_session_config)
            
        return True
    
    async def set_instructions(self, instructions: str) -> bool:
        """
        Update system instructions.
        
        Args:
            instructions: New system instructions
            
        Returns:
            True if updated successfully
        """
        if self.is_connected and self.current_session_config:
            self.current_session_config.instructions = instructions
            await self.client.configure_session(self.current_session_config)
            
        return True
    
    async def use_preset(self, preset: str) -> bool:
        """
        Use a preset configuration.
        
        Args:
            preset: Preset name (voice_assistant, transcription, conversational, etc.)
            
        Returns:
            True if preset applied successfully
        """
        preset_map = {
            "voice_assistant": SessionPresets.voice_assistant,
            "transcription": SessionPresets.transcription_service,
            "conversational": SessionPresets.conversational_ai,
            "customer_service": SessionPresets.customer_service,
            "audio_only": SessionPresets.audio_only
        }
        
        if preset not in preset_map:
            raise ValueError(f"Unknown preset. Choose from: {list(preset_map.keys())}")
        
        # Get preset configuration
        config = preset_map[preset]()
        
        # Apply voice override
        config.voice = self.voice
        
        # Reconnect with new config
        if self.is_connected:
            await self.client.disconnect()
            await self.client.connect(config)
            self.current_session_config = config
            
        return True
    
    # === Conversation Management ===
    
    async def start_conversation(
        self,
        opening_message: Optional[str] = None
    ) -> VoiceResponse:
        """
        Start a new conversation.
        
        Args:
            opening_message: Optional opening message
            
        Returns:
            Response to opening message if provided
        """
        self.conversation_history.clear()
        
        if opening_message:
            response = await self.send_text(opening_message)
            self.conversation_history.append({
                "role": "user",
                "content": opening_message,
                "timestamp": time.time()
            })
            if response.text:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.text,
                    "timestamp": time.time()
                })
            return response
        
        return VoiceResponse(success=True)
    
    async def continue_conversation(
        self,
        user_input: Union[str, bytes]
    ) -> VoiceResponse:
        """
        Continue conversation with text or audio input.
        
        Args:
            user_input: Text string or audio bytes
            
        Returns:
            VoiceResponse with assistant's reply
        """
        if isinstance(user_input, str):
            response = await self.send_text(user_input)
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
        else:
            response = await self.send_audio(user_input)
            self.conversation_history.append({
                "role": "user",
                "content": "[Audio Message]",
                "timestamp": time.time(),
                "audio_duration_ms": self.audio_processor.get_audio_duration_ms(user_input)
            })
        
        if response.text:
            self.conversation_history.append({
                "role": "assistant",
                "content": response.text,
                "timestamp": time.time()
            })
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    # === Audio Processing ===
    
    def validate_audio(self, audio_bytes: bytes) -> Tuple[bool, str]:
        """
        Validate audio format for API compatibility.
        
        Args:
            audio_bytes: Audio data to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        return self.audio_processor.validate_realtime_api_format(audio_bytes)
    
    def convert_audio_file(self, input_path: Union[str, Path]) -> bytes:
        """
        Convert any audio file to API-compatible format.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Converted audio bytes
        """
        return self.audio_processor.convert_from_any_format(input_path)
    
    def save_response_audio(
        self,
        response: VoiceResponse,
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Save response audio to file.
        
        Args:
            response: VoiceResponse containing audio
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to saved file or None
        """
        if not response.audio_bytes:
            print("⚠️  No audio in response to save")
            return None
        
        if output_path is None:
            # Auto-generate filename
            self.audio_output_dir.mkdir(exist_ok=True)
            timestamp = int(time.time() * 1000)
            output_path = self.audio_output_dir / f"response_{timestamp}.wav"
            print(f"💾 Auto-generating filename: {output_path}")
        
        print(f"💾 Saving {len(response.audio_bytes)} bytes of audio...")
        self.audio_processor.save_wav_file(response.audio_bytes, output_path)
        print(f"✅ Audio saved to: {output_path}")
        return Path(output_path)
    
    # === Event Handling ===
    
    def on_event(self, event_type: str, handler: Callable):
        """
        Register custom event handler.
        
        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        self.client.on_event(event_type, handler)
        self._response_handlers[event_type] = handler
    
    def off_event(self, event_type: str, handler: Callable):
        """Remove event handler"""
        self.client.off_event(event_type, handler)
        if event_type in self._response_handlers:
            del self._response_handlers[event_type]
    
    # === Function Calling ===
    
    async def add_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ):
        """
        Add a function that the assistant can call.
        
        Args:
            name: Function name
            description: Function description
            parameters: JSON Schema parameters
            handler: Function to execute
        """
        # Add tool to configuration
        tool = Tool(
            type="function",
            name=name,
            description=description,
            parameters=parameters
        )
        
        if self.current_session_config:
            self.current_session_config.tools.append(tool)
            await self.client.configure_session(self.current_session_config)
        
        # Register handler
        self._function_handlers[name] = handler
    
    # === Utility Methods ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        status = {
            "connected": self.is_connected,
            "voice": self.voice,
            "mode": self.mode.value,
            "vad_type": self.vad_type.value,
            "conversation_length": len(self.conversation_history)
        }
        
        if self.is_connected:
            status.update(self.client.get_status())
        
        return status
    
    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio to text using the API.
        
        Args:
            audio_bytes: Audio to transcribe
            
        Returns:
            Transcribed text
        """
        return await self.client.transcribe_audio(audio_bytes)
    
    # === Private Methods ===
    
    def _build_session_config(
        self,
        instructions: str,
        temperature: float,
        max_tokens: Union[int, str],
        **kwargs
    ) -> SessionConfig:
        """Build session configuration based on mode and settings"""
        # Determine modalities based on mode
        modalities = []
        if self.mode in [VoiceMode.TEXT_ONLY, VoiceMode.TEXT_AND_VOICE]:
            modalities.append("text")
        if self.mode in [VoiceMode.VOICE_ONLY, VoiceMode.TEXT_AND_VOICE]:
            modalities.append("audio")
        
        print(f"   🎯 Mode: {self.mode.value} -> Modalities: {modalities}")
        
        # Build turn detection config
        turn_detection = TurnDetectionConfig(
            type=self.vad_type.value,
            threshold=kwargs.get("vad_threshold", 0.5) if self.vad_type == VADType.SERVER_VAD else None,
            prefix_padding_ms=kwargs.get("prefix_padding_ms", 300),
            silence_duration_ms=kwargs.get("silence_duration_ms", 500),
            create_response=kwargs.get("auto_response", True)
        )
        
        # Create session config
        config = SessionConfig(
            instructions=instructions,
            modalities=modalities,
            voice=self.voice,
            input_audio_format=kwargs.get("input_audio_format", "pcm16"),
            output_audio_format=kwargs.get("output_audio_format", "pcm16"),
            temperature=temperature,
            max_response_output_tokens=max_tokens,
            speed=kwargs.get("speed", 1.0),
            turn_detection=turn_detection,
            tools=kwargs.get("tools", []),
            tool_choice=kwargs.get("tool_choice", "auto")
        )
        
        # Add transcription if requested
        if kwargs.get("enable_transcription", False):
            config.input_audio_transcription = {
                "model": "whisper-1",
                "language": kwargs.get("transcription_language"),
                "prompt": kwargs.get("transcription_prompt", "")
            }
        
        return config
    
    def _setup_event_handlers(self):
        """Set up internal event handlers"""
        print("📡 Registering event handlers...")
        
        # Session events
        @self.client.on_event("session.created")
        async def handle_session_created(data):
            print("   🎉 Session created event received!")
            session = data.get("session", {})
            print(f"      Session ID: {session.get('id', 'unknown')}")
        
        @self.client.on_event("session.updated")
        async def handle_session_updated(data):
            print("   🔄 Session updated event received!")
        
        @self.client.on_event("response.created")
        async def handle_response_created(data):
            print("   🚀 Response created event received!")
            response = data.get("response", {})
            print(f"      Response ID: {response.get('id', 'unknown')}")
        
        @self.client.on_event("conversation.item.created")
        async def handle_conversation_item_created(data):
            item = data.get("item", {})
            print(f"   💬 Conversation item created:")
            print(f"      Type: {item.get('type', 'unknown')}")
            print(f"      Role: {item.get('role', 'unknown')}")
            content = item.get("content", [])
            for c in content:
                content_type = c.get("type", "unknown")
                print(f"      Content type: {content_type}")
                if content_type == "text":
                    print(f"      Text: {c.get('text', '')}")
                elif content_type == "audio":
                    print(f"      Audio: {len(c.get('audio', ''))} bytes")
        
        # Track content part events
        @self.client.on_event("response.content_part.added")
        async def handle_content_part_added(data):
            part = data.get("part", {})
            part_type = part.get("type", "unknown")
            print(f"   📄 Content part added: {part_type}")
            if part_type == "text":
                print(f"      Text: {part.get('text', '')}")
            elif part_type == "audio":
                print(f"      Audio transcript: {part.get('transcript', '')}")
        
        @self.client.on_event("response.content_part.done")
        async def handle_content_part_done(data):
            part = data.get("part", {})
            part_type = part.get("type", "unknown")
            print(f"   ✓ Content part done: {part_type}")
            if part_type == "text":
                text = part.get("text", "")
                if text and not self._current_response.text:
                    # Sometimes text comes in content parts instead of deltas
                    self._current_response.text = text
        
        # Track response events
        @self.client.on_event("response.text.delta")
        async def handle_text_delta(data):
            text = data.get("delta", "")
            if text:  # Only print if there's actual text
                print(f"   📝 Text delta: '{text}'")
            if self._current_response.text is None:
                self._current_response.text = ""
            self._current_response.text += text
        
        print("   ✓ Registered response.text.delta handler")
        
        @self.client.on_event("response.audio.delta")
        async def handle_audio_delta(data):
            # Audio is accumulated in client's buffer
            print("   🔊 Audio delta received")
            pass
        
        @self.client.on_event("response.done")
        async def handle_response_done(data):
            print("   ✅ Response done event!")
            self._current_response.success = True
            # Get accumulated audio
            audio_duration = self.client.get_audio_output_duration()
            if audio_duration > 0:
                print(f"   🎵 Getting {audio_duration:.1f}ms of audio from buffer")
                self._current_response.audio_bytes = self.client.get_audio_output(clear_buffer=True)
                self._current_response.audio_duration_ms = audio_duration
        
        print("   ✓ Registered response.done handler")
        
        @self.client.on_event("error")
        async def handle_error(data):
            error = data.get("error", {})
            error_msg = error.get("message", "Unknown error")
            print(f"   ❌ Error event: {error_msg}")
            self._current_response.error = error_msg
            self._current_response.success = False
        
        # Track audio transcript events (for audio input transcription)
        @self.client.on_event("response.audio_transcript.delta")
        async def handle_audio_transcript_delta(data):
            text = data.get("delta", "")
            if text:
                print(f"   🎤 Audio transcript delta: '{text}'")
                # Add to text response as well
                if self._current_response.text is None:
                    self._current_response.text = ""
                self._current_response.text += text
        
        @self.client.on_event("response.audio_transcript.done")
        async def handle_audio_transcript_done(data):
            print("   🎤 Audio transcript done")
            transcript = data.get("transcript", "")
            if transcript:
                print(f"      Final transcript: '{transcript}'")
        
        # Track VAD events
        @self.client.on_event("input_audio_buffer.speech_started")
        async def handle_speech_start(data):
            print("   🎙️ Speech started")
            if self._current_response.events is None:
                self._current_response.events = []
            self._current_response.events.append("speech_started")
        
        @self.client.on_event("input_audio_buffer.speech_stopped")
        async def handle_speech_stop(data):
            print("   🔇 Speech stopped")
            if self._current_response.events is None:
                self._current_response.events = []
            self._current_response.events.append("speech_stopped")
        
        print("✅ Event handlers registered successfully")
    
    def _reset_current_response(self):
        """Reset current response tracking"""
        self._current_response = VoiceResponse()
        print("   🔄 Response tracker reset")
    
    async def _wait_for_response(self, timeout: float = 30.0) -> VoiceResponse:
        """Wait for response completion"""
        start_time = time.time()
        print(f"⏱️  Waiting up to {timeout}s for response...")
        
        dots = 0
        last_update = time.time()
        
        while (time.time() - start_time) < timeout:
            # Show progress dots
            if time.time() - last_update > 1.0:
                dots = (dots + 1) % 4
                print(f"\r⏳ Waiting{'.' * dots}    ", end='', flush=True)
                last_update = time.time()
            
            # Check response status
            if self._current_response.success:
                print(f"\r✅ Response completed successfully!          ")
                return self._current_response
            
            if self._current_response.error:
                print(f"\r❌ Response error: {self._current_response.error}")
                return self._current_response
            
            await asyncio.sleep(0.1)
        
        # Timeout
        print(f"\r⏰ Response timeout after {timeout}s!")
        self._current_response.error = "Response timeout"
        return self._current_response
    
    # === Context Manager Support ===
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.is_connected:
            await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# === Convenience Functions ===

async def quick_voice_chat(api_key: str, message: str) -> VoiceResponse:
    """
    Quick one-off voice chat.
    
    Args:
        api_key: OpenAI API key
        message: Message to send
        
    Returns:
        VoiceResponse with results
    """
    async with BaseVoiceAPIEngine(api_key) as engine:
        return await engine.send_text(message)


async def transcribe_audio_file(api_key: str, file_path: Union[str, Path]) -> str:
    """
    Transcribe an audio file to text.
    
    Args:
        api_key: OpenAI API key
        file_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    engine = BaseVoiceAPIEngine(api_key, mode=VoiceMode.TEXT_ONLY)
    await engine.connect(
        instructions="Transcribe the audio exactly as spoken.",
        temperature=0.3
    )
    
    try:
        audio_bytes = engine.convert_audio_file(file_path)
        result = await engine.transcribe_audio(audio_bytes)
        return result
    finally:
        await engine.disconnect()


# === Example Usage ===

async def example_usage():
    """Example of using the BaseVoiceAPIEngine"""
    
    # Initialize engine

   
    engine = BaseVoiceAPIEngine(
        api_key=api_key,
        voice="alloy",
        mode=VoiceMode.TEXT_AND_VOICE,
        vad_type=VADType.SERVER_VAD
    )
    
    # Connect with custom instructions
    await engine.connect(
        instructions="You are a friendly assistant. Be conversational and helpful.",
        temperature=0.8
    )
    
    # Simple text conversation
    print("\n📤 Testing simple text conversation...")
    response = await engine.send_text("Hello! How are you today?")
    
    if response.success:
        print(f"✅ Got response: {response.text}")
    else:
        print(f"❌ No response. Error: {response.error}")
    
    # Send audio file
    if Path("test_voice.wav").exists():
        print("\n🎤 Testing audio file input...")
        response = await engine.send_audio_file("test_voice.wav")
        
        if response.success:
            print(f"✅ Audio response duration: {response.audio_duration_ms}ms")
            if response.text:
                print(f"   Text: {response.text}")
        else:
            print(f"❌ Audio test failed: {response.error}")
    
    # Disconnect
    await engine.disconnect()


if __name__ == "__main__":
    # Run example usage
   
    print(api_key)
    if api_key:
        asyncio.run(example_usage())
    else:
        print("Please set OPENAI_API_KEY environment variable")