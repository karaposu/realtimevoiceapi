"""
Example: Using the new send_recorded_audio API

This example demonstrates the difference between:
1. Real-time streaming (send_audio) - for continuous audio input
2. Record-and-send (send_recorded_audio) - for complete audio recordings
"""

import asyncio
from realtimevoiceapi import VoiceEngine, VoiceEngineConfig


async def example_usage():
    # Configure engine
    config = VoiceEngineConfig(
        api_key="your-api-key",
        mode="fast",
        vad_enabled=False  # We'll handle recording boundaries manually
    )
    
    engine = VoiceEngine(config)
    await engine.connect()
    
    # Example 1: Real-time streaming (current behavior)
    # Used for continuous microphone input
    print("Example 1: Real-time streaming")
    async def stream_audio():
        # Simulate streaming audio chunks
        for chunk in audio_chunks:  # audio_chunks would be from microphone
            await engine.send_audio(chunk)
        # Note: No automatic response - need manual trigger or VAD
    
    # Example 2: Record-and-send (NEW API)
    # Used for push-to-talk, file upload, or pre-recorded audio
    print("\nExample 2: Record-and-send")
    
    # Record complete audio (e.g., while key is held)
    recorded_audio = b""  # Complete audio recording
    
    # Send and automatically get response
    await engine.send_recorded_audio(recorded_audio)
    # Response is automatically triggered!
    
    # Example 3: Record-and-send without auto-response
    print("\nExample 3: Manual response control")
    await engine.send_recorded_audio(recorded_audio, auto_respond=False)
    # ... do something else ...
    await engine._strategy.trigger_response()  # Manual trigger
    
    await engine.disconnect()


# Usage in VoxTerm with push-to-talk
async def voxterm_push_to_talk_example(engine):
    """
    Example of how VoxTerm should use the new API
    """
    # When space key is pressed
    audio_buffer = []
    
    # While space key is held
    while space_key_held:
        chunk = capture_audio_chunk()
        audio_buffer.append(chunk)
    
    # When space key is released
    complete_audio = b"".join(audio_buffer)
    
    # OLD WAY (workaround):
    # await engine.send_audio(complete_audio)
    # await engine._strategy.stream_manager.connection.send(
    #     MessageFactory.response_create()
    # )
    
    # NEW WAY (clean API):
    await engine.send_recorded_audio(complete_audio)


if __name__ == "__main__":
    asyncio.run(example_usage())