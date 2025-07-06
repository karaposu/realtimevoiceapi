"""Voice conversation example with audio input/output"""

import asyncio
import logging
from pathlib import Path
from realtimevoiceapi import RealtimeClient, SessionConfig

async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = RealtimeClient(api_key="your-openai-api-key")
    
    # Configure session for voice
    config = SessionConfig(
        instructions="You are a friendly voice assistant. Speak naturally and expressively.",
        modalities=["text", "audio"],
        voice="alloy",
        input_audio_format="pcm16",
        output_audio_format="pcm16"
    )
    
    # Track conversation state
    response_count = 0
    
    # Setup event handlers
    @client.on_event("response.text.delta")
    async def handle_text_delta(event_data):
        text = event_data.get("delta", "")
        print(text, end="", flush=True)
    
    @client.on_event("response.done")
    async def handle_response_done(event_data):
        nonlocal response_count
        response_count += 1
        
        # Save audio output
        audio_file = f"response_{response_count}.wav"
        if client.save_audio_output(audio_file):
            print(f"\n[Audio saved: {audio_file}]")
        print("="*50)
    
    try:
        # Connect
        print("Connecting to OpenAI Realtime API...")
        await client.connect(config)
        print("Connected!")
        
        # Send text message
        print("\nSending text message...")
        print("User: Hello! Can you introduce yourself?")
        print("Assistant: ", end="")
        await client.send_text("Hello! Can you introduce yourself?")
        await asyncio.sleep(5)
        
        # Send audio file (if exists)
        audio_input_path = "input_question.wav"
        if Path(audio_input_path).exists():
            print(f"\nSending audio file: {audio_input_path}")
            print("Assistant: ", end="")
            await client.send_audio_file(audio_input_path)
            await asyncio.sleep(5)
        else:
            print(f"\n[Audio file {audio_input_path} not found, skipping audio test]")
        
        # Send another text message
        print("\nUser: Can you tell me about the weather?")
        print("Assistant: ", end="")
        await client.send_text("Can you tell me about the weather?")
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()
        print("\nDisconnected.")

if __name__ == "__main__":
    asyncio.run(main())

##############################################################################
# README.md for the package
##############################################################################

"""
# RealtimeVoiceAPI

A Python client for OpenAI's Realtime API, enabling real-time voice conversations with GPT-4.

## Installation

```bash
pip install websockets aiohttp numpy pydub
```

## Quick Start

### Text Conversation

```python
import asyncio
from realtimevoiceapi import RealtimeClient, SessionConfig

async def main():
    client = RealtimeClient(api_key="your-openai-api-key")
    
    config = SessionConfig(
        instructions="You are a helpful assistant",
        modalities=["text"]
    )
    
    @client.on_event("response.text.delta")
    async def handle_text(event_data):
        print(event_data.get("delta", ""), end="")
    
    await client.connect(config)
    await client.send_text("Hello!")
    await asyncio.sleep(3)
    await client.disconnect()

asyncio.run(main())
```

### Voice Conversation

```python
import asyncio
from realtimevoiceapi import RealtimeClient, SessionConfig

async def main():
    client = RealtimeClient(api_key="your-openai-api-key")
    
    config = SessionConfig(
        instructions="You are a friendly voice assistant",
        modalities=["text", "audio"],
        voice="alloy"
    )
    
    @client.on_event("response.done")
    async def save_audio(event_data):
        client.save_audio_output("response.wav")
    
    await client.connect(config)
    await client.send_audio_file("question.wav")
    await asyncio.sleep(5)
    await client.disconnect()

asyncio.run(main())
```

## Features

- ✅ Real-time WebSocket connection to OpenAI
- ✅ Text and audio conversations
- ✅ Event-driven architecture
- ✅ Audio file processing (WAV format)
- ✅ Streaming audio input/output
- ✅ Session configuration
- ✅ Error handling
- ✅ Comprehensive examples

## Testing

Run the examples:

```bash
python -m realtimevoiceapi.examples.basic_conversation
python -m realtimevoiceapi.examples.voice_conversation
```

## Requirements

- Python 3.8+
- OpenAI API key with Realtime API access
- Audio files in
"""