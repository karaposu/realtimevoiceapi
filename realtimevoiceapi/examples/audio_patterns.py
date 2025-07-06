# realtimevoiceapi/examples/audio_patterns.py
"""
Common audio input patterns for RealtimeVoiceAPI

This module provides pre-configured patterns and helper functions
for common audio interaction scenarios.
"""

import asyncio
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from ..client import RealtimeClient
from ..session import SessionConfig
from ..models import TurnDetectionConfig, TranscriptionConfig
from ..audio import AudioProcessor, AudioConfig
from ..exceptions import AudioError, SessionError


logger = logging.getLogger(__name__)


def get_audio_config(
    voice: str = "alloy",
    vad_sensitivity: float = 0.5,
    auto_respond: bool = True,
    vad_type: str = "server_vad"
) -> SessionConfig:
    """
    Get a standard audio configuration.
    
    Args:
        voice: Voice to use for responses
        vad_sensitivity: VAD threshold (0.0-1.0, lower = more sensitive)
        auto_respond: Whether to automatically respond after speech
        vad_type: Type of VAD to use ("server_vad" or "semantic_vad")
    
    Returns:
        Configured SessionConfig for audio interaction
    """
    return SessionConfig(
        modalities=["text", "audio"],
        voice=voice,
        input_audio_format="pcm16",
        output_audio_format="pcm16",
        turn_detection=TurnDetectionConfig(
            type=vad_type,  # Always required - no "none" option
            threshold=vad_sensitivity,
            silence_duration_ms=500,
            create_response=auto_respond
        )
    )


def get_transcription_config(
    language: Optional[str] = None,
    prompt: str = "Transcribe exactly what is said."
) -> SessionConfig:
    """
    Get configuration optimized for transcription.
    
    Args:
        language: Optional language code (e.g., "en", "es", "fr")
        prompt: Transcription prompt
    
    Returns:
        SessionConfig optimized for transcription
    """
    return SessionConfig(
        instructions="Transcribe the audio exactly as spoken. Do not add any commentary or corrections.",
        modalities=["text"],  # Text only for pure transcription
        temperature=0.3,  # Low temperature for accuracy
        input_audio_transcription=TranscriptionConfig(
            model="whisper-1",
            language=language,
            prompt=prompt
        ),
        turn_detection=TurnDetectionConfig(
            type="server_vad",
            threshold=0.3,  # More sensitive for transcription
            silence_duration_ms=800,  # Longer pause for natural speech
            create_response=True
        )
    )


async def transcribe_audio(client: RealtimeClient, audio_bytes: bytes) -> str:
    """
    Use the API to transcribe audio to text.
    
    Args:
        client: Connected RealtimeClient
        audio_bytes: Audio to transcribe
        
    Returns:
        Transcribed text
    """
    # Save current config
    original_config = client.session_config
    
    # Switch to transcription config
    await client.configure_session(get_transcription_config())
    
    try:
        # Send audio and get response
        text, _ = await client.send_audio_and_wait_for_response(audio_bytes)
        return text.strip()
        
    finally:
        # Restore original config
        if original_config:
            await client.configure_session(original_config)


async def audio_conversation_turn(
    client: RealtimeClient,
    audio_bytes: bytes,
    wait_for_audio: bool = True,
    timeout: float = 30.0
) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Single conversation turn with audio input.
    
    Args:
        client: Connected RealtimeClient
        audio_bytes: User's audio input
        wait_for_audio: Whether to wait for audio response
        timeout: Maximum time to wait for response
        
    Returns:
        Tuple of (response_text, response_audio_bytes)
    """
    # Clear any previous audio
    await client.clear_audio_input()
    await asyncio.sleep(0.2)
    
    # Send audio
    await client.send_audio_simple(audio_bytes)
    
    # Track responses
    response_text = ""
    response_done = False
    audio_start_size = len(client.audio_output_buffer)
    
    async def on_text(data):
        nonlocal response_text
        response_text += data.get("delta", "")
    
    async def on_done(data):
        nonlocal response_done
        response_done = True
    
    # Register handlers
    client.on_event("response.text.delta", on_text)
    client.on_event("response.done", on_done)
    
    try:
        # Wait for response
        start_time = asyncio.get_event_loop().time()
        while not response_done and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)
            
            # If we have text but not waiting for audio, return early
            if response_text and not wait_for_audio:
                break
        
        # Get audio response if available
        response_audio = None
        if len(client.audio_output_buffer) > audio_start_size:
            response_audio = bytes(client.audio_output_buffer[audio_start_size:])
        
        return response_text, response_audio
        
    finally:
        # Cleanup
        client.off_event("response.text.delta", on_text)
        client.off_event("response.done", on_done)


async def stream_microphone_to_api(
    client: RealtimeClient,
    microphone_callback,
    chunk_duration_ms: int = 100,
    stop_event: Optional[asyncio.Event] = None
):
    """
    Stream audio from microphone to API in real-time.
    
    Args:
        client: Connected RealtimeClient
        microphone_callback: Async function that returns audio chunks
        chunk_duration_ms: Duration of each audio chunk
        stop_event: Optional event to stop streaming
    
    Note: microphone_callback should return None or empty bytes to skip
    """
    if not client.is_session_active:
        raise SessionError("No active session")
    
    # Clear buffer before starting
    await client.clear_audio_input()
    
    try:
        while stop_event is None or not stop_event.is_set():
            # Get audio chunk from microphone
            audio_chunk = await microphone_callback(chunk_duration_ms)
            
            if audio_chunk and len(audio_chunk) > 0:
                # Send to API
                audio_b64 = client.audio_processor.bytes_to_base64(audio_chunk)
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await client.connection.send_event(event)
            
            # Small delay to prevent overwhelming the connection
            await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Microphone streaming error: {e}")
        raise


class AudioConversationManager:
    """
    Manages multi-turn audio conversations with state tracking.
    """
    
    def __init__(self, client: RealtimeClient):
        self.client = client
        self.turn_count = 0
        self.conversation_history: List[Dict[str, Any]] = []
        
    async def add_turn(
        self,
        audio_bytes: bytes,
        save_audio: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Add a conversation turn.
        
        Args:
            audio_bytes: User's audio input
            save_audio: Whether to save audio files
            output_dir: Directory to save audio files
            
        Returns:
            Dictionary with turn information
        """
        self.turn_count += 1
        turn_id = f"turn_{self.turn_count}"
        
        # Send audio and get response
        response_text, response_audio = await audio_conversation_turn(
            self.client, audio_bytes
        )
        
        turn_data = {
            "turn_id": turn_id,
            "turn_number": self.turn_count,
            "user_audio_size": len(audio_bytes),
            "response_text": response_text,
            "response_audio_size": len(response_audio) if response_audio else 0,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Save audio if requested
        if save_audio and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Save user audio
            user_file = output_dir / f"{turn_id}_user.wav"
            self.client.audio_processor.save_wav_file(audio_bytes, user_file)
            turn_data["user_audio_file"] = str(user_file)
            
            # Save response audio
            if response_audio:
                response_file = output_dir / f"{turn_id}_response.wav"
                self.client.audio_processor.save_wav_file(response_audio, response_file)
                turn_data["response_audio_file"] = str(response_file)
        
        self.conversation_history.append(turn_data)
        return turn_data
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.turn_count = 0


# Preset patterns for specific use cases

async def voice_assistant_pattern(api_key: str, audio_input: bytes) -> Tuple[str, bytes]:
    """
    Simple voice assistant pattern - send audio, get audio response.
    
    Args:
        api_key: OpenAI API key
        audio_input: User's audio input
        
    Returns:
        Tuple of (transcript, response_audio)
    """
    async with RealtimeClient(api_key) as client:
        # Configure for voice assistant
        config = SessionConfig(
            instructions="You are a helpful voice assistant. Keep responses concise and friendly.",
            modalities=["text", "audio"],
            voice="alloy",
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.5,
                create_response=True
            )
        )
        
        await client.connect(config)
        
        # Send audio and get response
        text, audio = await client.send_audio_and_wait_for_response(audio_input)
        
        return text, audio


async def customer_service_pattern(
    api_key: str,
    audio_input: bytes,
    customer_context: str = ""
) -> Dict[str, Any]:
    """
    Customer service pattern with context and professional tone.
    
    Args:
        api_key: OpenAI API key
        audio_input: Customer's audio input
        customer_context: Context about the customer/issue
        
    Returns:
        Dictionary with response details
    """
    async with RealtimeClient(api_key) as client:
        # Configure for customer service
        instructions = f"""You are a professional customer service representative.
        Be polite, helpful, and solution-oriented.
        {f'Customer context: {customer_context}' if customer_context else ''}
        Ask clarifying questions when needed."""
        
        config = SessionConfig(
            instructions=instructions,
            modalities=["text", "audio"],
            voice="shimmer",  # Professional voice
            temperature=0.7,  # More consistent responses
            speed=0.95,  # Slightly slower for clarity
            turn_detection=TurnDetectionConfig(
                type="server_vad",
                threshold=0.4,
                silence_duration_ms=800,  # Longer pause for customers
                create_response=True
            )
        )
        
        await client.connect(config)
        
        # Send audio and get response
        text, audio = await client.send_audio_and_wait_for_response(audio_input)
        
        return {
            "response_text": text,
            "response_audio": audio,
            "audio_duration_ms": client.audio_processor.get_audio_duration_ms(audio) if audio else 0,
            "session_id": client.session_id
        }


async def interactive_storytelling_pattern(
    api_key: str,
    audio_input: bytes,
    story_context: str = "",
    voice: str = "nova"
) -> Dict[str, Any]:
    """
    Interactive storytelling with dynamic voice.
    
    Args:
        api_key: OpenAI API key
        audio_input: User's audio input
        story_context: Current story context
        voice: Voice to use for narration
        
    Returns:
        Dictionary with story continuation
    """
    async with RealtimeClient(api_key) as client:
        # Configure for storytelling
        instructions = f"""You are an interactive storyteller.
        Create engaging, imaginative responses based on user input.
        {f'Story so far: {story_context}' if story_context else 'Start a new story based on user input.'}
        Use vivid descriptions and maintain narrative continuity."""
        
        config = SessionConfig(
            instructions=instructions,
            modalities=["text", "audio"],
            voice=voice,
            temperature=1.0,  # More creative
            speed=1.1,  # Slightly faster for engagement
            turn_detection=TurnDetectionConfig(
                type="semantic_vad",  # Better for understanding story inputs
                threshold=0.5,
                create_response=True
            )
        )
        
        await client.connect(config)
        
        # Send audio and get response
        text, audio = await client.send_audio_and_wait_for_response(audio_input, timeout=45)
        
        return {
            "story_text": text,
            "story_audio": audio,
            "voice_used": voice,
            "duration_ms": client.audio_processor.get_audio_duration_ms(audio) if audio else 0
        }


# Utility functions

def validate_audio_format(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate audio format for Realtime API.
    
    Args:
        audio_bytes: Audio data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    processor = AudioProcessor()
    return processor.validator.validate_realtime_api_format(audio_bytes)


def prepare_audio_for_api(
    audio_bytes: bytes,
    source_sample_rate: Optional[int] = None
) -> bytes:
    """
    Prepare audio for API by ensuring correct format.
    
    Args:
        audio_bytes: Raw audio data
        source_sample_rate: Source sample rate (if known)
        
    Returns:
        Audio data ready for API (24kHz, 16-bit PCM, mono)
    """
    processor = AudioProcessor()
    
    # If source rate is different, resample
    if source_sample_rate and source_sample_rate != AudioConfig.SAMPLE_RATE:
        audio_bytes = processor.converter.resample_audio(
            audio_bytes, source_sample_rate, AudioConfig.SAMPLE_RATE
        )
    
    # Ensure little-endian format
    audio_bytes = processor.ensure_little_endian_pcm16(audio_bytes)
    
    # Validate result
    is_valid, error = validate_audio_format(audio_bytes)
    if not is_valid:
        raise AudioError(f"Audio preparation failed: {error}")
    
    return audio_bytes