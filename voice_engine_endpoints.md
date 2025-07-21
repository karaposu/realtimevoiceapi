# VoiceEngine API Endpoints

## Overview

VoiceEngine provides a unified interface for real-time voice interactions with AI. It supports two primary modes of operation:
- **Real-time streaming**: Continuous audio input with VAD-based turn detection
- **Record-and-send**: Complete audio recordings with explicit boundaries

## Connection Management

### `connect(retry_count=3)`
Establishes connection to the voice API.
```python
await engine.connect()
```

### `disconnect()`
Cleanly disconnects from the voice API.
```python
await engine.disconnect()
```

### `is_connected`
Property that returns current connection status.
```python
if engine.is_connected:
    # Ready to send/receive
```

## Audio Control

### `start_listening()`
Starts capturing audio from the microphone for real-time processing.
```python
await engine.start_listening()
```

### `stop_listening()`
Stops audio capture from the microphone.
```python
await engine.stop_listening()
```

## Communication Methods

### `send_text(text)`
Sends text message to the AI and automatically triggers a response.
```python
await engine.send_text("Hello, how are you?")
# Response is triggered automatically
```

### `send_audio(audio_data)` 
Sends audio data for **real-time streaming**. Does NOT automatically trigger response.
- Use for: Continuous microphone input, live streaming
- Behavior: Relies on VAD or manual trigger for response
```python
# Stream audio chunks as they arrive
while recording:
    chunk = capture_audio()
    await engine.send_audio(chunk)
# No automatic response - VAD or manual trigger needed
```

### `send_recorded_audio(audio_data, auto_respond=True)` 
Sends **complete audio recording** and optionally triggers response.
- Use for: Push-to-talk, audio files, pre-recorded content
- Behavior: Automatically triggers response when `auto_respond=True`
```python
# Record complete audio
audio_buffer = record_until_key_released()
await engine.send_recorded_audio(audio_buffer)
# Response triggered automatically!

# Or without auto-response
await engine.send_recorded_audio(audio_buffer, auto_respond=False)
# Manually trigger later if needed
```

### `interrupt()`
Interrupts the current AI response.
```python
await engine.interrupt()
```

## Convenience Methods

### `text_2_audio_response(text, timeout=30.0)`
Converts text to speech with real-time playback.
```python
audio_bytes = await engine.text_2_audio_response("Hello world")
```

### `prompt_and_wait_for_response(prompt, timeout=30.0)`
Sends a text prompt and waits for complete response.
```python
response = await engine.prompt_and_wait_for_response("Tell me a joke")
print(response.text)
```

## Event Handlers

VoiceEngine supports various event callbacks:

```python
engine.on_audio_chunk = lambda audio: process_audio(audio)
engine.on_transcript = lambda text: print(f"User: {text}")
engine.on_response_text = lambda text: print(f"AI: {text}")
engine.on_response_audio = lambda audio: play_audio(audio)
engine.on_error = lambda error: handle_error(error)
engine.on_state_change = lambda state: update_ui(state)
```

## State Management

### `get_state()`
Returns current engine state.
```python
state = engine.get_state()
# StreamState.ACTIVE, StreamState.IDLE, etc.
```

## Metrics and Usage

### `get_metrics()`
Returns performance metrics.
```python
metrics = engine.get_metrics()
print(f"Latency: {metrics['latency_ms']}ms")
```

### `get_usage()`
Returns usage statistics for the session.
```python
usage = engine.get_usage()
print(f"Audio minutes: {usage.audio_output_seconds / 60}")
```

### `estimate_cost()`
Estimates the cost of the current session.
```python
cost = await engine.estimate_cost()
print(f"Estimated cost: ${cost.total:.2f}")
```

## Usage Examples

### Example 1: Real-time Voice Chat
```python
# For continuous conversation with VAD
config = VoiceEngineConfig(
    api_key="...",
    vad_enabled=True,
    vad_type="client"
)
engine = VoiceEngine(config)
await engine.connect()
await engine.start_listening()
# Audio is processed continuously, VAD handles turn-taking
```

### Example 2: Push-to-Talk
```python
# For push-to-talk interface
config = VoiceEngineConfig(
    api_key="...",
    vad_enabled=False  # Manual control
)
engine = VoiceEngine(config)
await engine.connect()

# When button pressed
audio_buffer = []
while button_held:
    chunk = capture_audio()
    audio_buffer.append(chunk)

# When button released
complete_audio = b"".join(audio_buffer)
await engine.send_recorded_audio(complete_audio)
```

### Example 3: Audio File Processing
```python
# For pre-recorded audio files
with open("recording.wav", "rb") as f:
    audio_data = f.read()

await engine.send_recorded_audio(audio_data)
```

## Key Differences: send_audio vs send_recorded_audio

| Aspect | `send_audio()` | `send_recorded_audio()` |
|--------|---------------|------------------------|
| Use Case | Real-time streaming | Complete recordings |
| Audio Boundaries | Continuous, no clear end | Explicit start/end |
| Response Trigger | VAD or manual | Automatic (by default) |
| Typical Sources | Live microphone | Files, push-to-talk |
| Network Behavior | Many small chunks | One complete buffer |
| Turn Detection | Server/client VAD | Explicit completion |

## Best Practices

1. **Choose the right method**:
   - Use `send_audio()` for live microphone streaming with VAD
   - Use `send_recorded_audio()` for push-to-talk or files

2. **Handle connection errors**:
   ```python
   try:
       await engine.connect()
   except EngineError as e:
       # Handle connection failure
   ```

3. **Set up event handlers before connecting**:
   ```python
   engine.on_response_text = handle_response
   await engine.connect()
   ```

4. **Clean up properly**:
   ```python
   try:
       # Use engine
   finally:
       await engine.disconnect()
   ```

5. **Monitor usage for cost control**:
   ```python
   usage = engine.get_usage()
   if usage.audio_output_seconds > 3600:  # 1 hour
       # Consider warning user about costs
   ```