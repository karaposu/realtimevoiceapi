# VoxTerm 🎙️

**A lightweight, non-blocking terminal UI framework for real-time voice applications**

VoxTerm provides a rich terminal interface for voice-enabled applications without compromising real-time performance. It's designed as a thin UI layer that works seamlessly with voice engines like `realtimevoiceapi.VoiceEngine`, handling display and user interaction while your voice engine handles the actual audio processing.

## 🚀 Key Features

- **Zero-latency UI**: Event-driven architecture that never blocks audio streams
- **Voice Engine Agnostic**: Works with any voice API (OpenAI, Google, Azure, custom)
- **Flexible Input Modes**: Push-to-talk, always-on, turn-based, or text-only
- **Real-time Display**: Live transcriptions, status updates, and audio levels
- **Non-intrusive**: Runs in separate thread, preserving voice engine's real-time capabilities
- **Minimal Dependencies**: Pure Python with optional enhancements

## 🎯 What VoxTerm Is (and Isn't)

### ✅ VoxTerm IS:
- A **terminal UI framework** for voice applications
- A **display layer** for conversations, status, and controls  
- An **input coordinator** for keyboard, commands, and mode switching
- A **non-blocking event system** that preserves real-time performance

### ❌ VoxTerm is NOT:
- A voice API or speech recognition system
- An audio processing library
- A VAD implementation (uses your voice engine's VAD)
- A WebSocket/networking layer

## 🏗️ Architecture

VoxTerm is designed as a thin, event-driven layer that sits **alongside** your voice engine, not in front of it:

```
┌─────────────────┐         ┌──────────────────┐
│   Your Voice    │ <-----> │   Voice API      │
│   Engine        │         │ (OpenAI, etc)    │
│ (Audio Process) │         └──────────────────┘
└────────┬────────┘                
         │ Events (text, audio, state)
         ↓
┌─────────────────┐
│    VoxTerm      │ ← Terminal UI (separate thread)
│  (UI Display)   │ ← Never blocks audio pipeline
└─────────────────┘
```

## 📁 Project Structure

```
voxterm/
├── README.md
├── pyproject.toml
├── src/
│   └── voxterm/
│       ├── __init__.py
│       ├── core/                 # Core abstractions
│       │   ├── __init__.py
│       │   ├── terminal.py       # Main VoxTerminal class
│       │   ├── events.py         # Event system (non-blocking)
│       │   └── interfaces.py     # API contracts
│       │
│       ├── display/              # Terminal rendering (separate thread)
│       │   ├── __init__.py
│       │   ├── renderer.py       # Terminal display engine
│       │   ├── components.py     # UI components (status bar, etc)
│       │   ├── formatters.py     # Message formatting
│       │   └── themes.py         # Color schemes
│       │
│       ├── input/                # Input handling
│       │   ├── __init__.py
│       │   ├── keyboard.py       # Keyboard event handling
│       │   ├── commands.py       # Command parsing
│       │   └── modes.py          # Input mode controllers
│       │
│       ├── integration/          # Voice engine integrations
│       │   ├── __init__.py
│       │   ├── base.py           # Base integration class
│       │   └── voice_engine.py   # VoiceEngine integration
│       │
│       └── utils/
│           ├── __init__.py
│           ├── async_helpers.py  # Async utilities
│           └── terminal.py       # Terminal utilities
│
├── examples/
│   ├── simple_chat.py            # Basic usage
│   ├── voice_engine_chat.py      # With VoiceEngine
│   └── custom_integration.py     # Custom voice API
│
└── tests/
    ├── test_display.py
    ├── test_events.py
    └── test_integration.py
```

## 🎯 Implementation Goals

### 1. **Zero-Blocking Architecture**
- UI runs in separate thread/process
- Event-driven communication via queues
- No synchronous calls in audio path
- Display updates are fire-and-forget

### 2. **Voice Engine Integration First**
- Built specifically for `realtimevoiceapi.VoiceEngine`
- Uses VoiceEngine's VAD, audio handling, and networking
- VoxTerm only handles display and keyboard input
- Clean integration via callbacks and events

### 3. **Minimal Overhead**
- < 5ms latency for event propagation
- < 1% CPU usage during idle
- Memory-efficient message buffering
- No audio processing or copying

### 4. **Developer Experience**
- Simple 5-line integration
- Sensible defaults
- Progressive enhancement
- Clear extension points

## 🚀 Quick Start

### Basic Usage with VoiceEngine

```python
from voxterm import VoxTerminal
from realtimevoiceapi import VoiceEngine

# Create your voice engine
engine = VoiceEngine(api_key="...")

# Create terminal UI
terminal = VoxTerminal(
    title="AI Voice Chat",
    mode="push_to_talk"  # or "always_on", "turn_based", "text"
)

# Connect VoiceEngine to VoxTerm (non-blocking)
terminal.bind_voice_engine(engine)

# Run the UI (starts in separate thread)
terminal.run()

# Your voice engine continues to run at full speed!
await engine.connect()
```

### Manual Integration

```python
# VoxTerm exposes simple callbacks
terminal.on_user_input = lambda text: engine.send_text(text)
terminal.on_push_to_talk_start = lambda: engine.start_listening()
terminal.on_push_to_talk_end = lambda: engine.stop_listening()
terminal.on_interrupt = lambda: engine.interrupt()

# Feed display updates from your engine
engine.on_text_response = terminal.display_ai_text
engine.on_user_transcript = terminal.display_user_text
engine.on_state_change = terminal.update_status
```

## 🔧 Configuration

VoxTerm delegates all audio/VAD configuration to your voice engine:

```python
# VoxTerm configuration (UI only)
terminal_config = {
    "theme": "dark",
    "show_timestamps": True,
    "show_audio_levels": True,
    "key_bindings": {
        "push_to_talk": "space",
        "interrupt": "escape",
        "mute": "m",
        "quit": "q"
    }
}

# Voice configuration (handled by your engine)
engine_config = {
    "vad_threshold": 0.5,      # VoxTerm doesn't touch this
    "sample_rate": 24000,      # VoxTerm doesn't care
    "voice": "alloy"           # VoxTerm just displays it
}
```

## 🎨 UI Components

### Status Bar
```
┌─────────────────────────────────────────────────┐
│ 🟢 Connected │ Mode: Push-to-Talk │ Mic: ON │ ⚡ 45ms │
└─────────────────────────────────────────────────┘
```

### Conversation Display
```
[09:34:12] You: What's the weather like today?
           
[09:34:13] AI: I don't have access to real-time weather data,
               but I'd be happy to help you find weather 
               information for your area.

[09:34:18] You: [Listening...]
```

### Control Hints
```
[SPACE] Hold to talk │ [M] Mute │ [L] Toggle logs │ [Q] Quit
```

## 🔌 Integration Examples

### OpenAI Realtime
```python
from voxterm import VoxTerminal
from realtimevoiceapi import VoiceEngine

terminal = VoxTerminal()
engine = VoiceEngine(api_key="...")
terminal.bind_voice_engine(engine)
terminal.run()
```

### Google Speech
```python
from voxterm import VoxTerminal, BaseIntegration

class GoogleSpeechIntegration(BaseIntegration):
    def connect_to_terminal(self, terminal, speech_client):
        # Wire up callbacks
        pass

terminal = VoxTerminal()
integration = GoogleSpeechIntegration()
integration.connect_to_terminal(terminal, google_client)
terminal.run()
```

## 🏃 Performance

VoxTerm is designed to have **zero impact** on real-time voice processing:

- **Display updates**: Async, non-blocking queue (< 1ms)
- **Keyboard events**: Separate thread with event queue
- **Memory usage**: < 10MB for typical conversation
- **CPU usage**: < 1% during conversation
- **Latency added**: 0ms to voice pipeline

## 🛠️ Advanced Usage

### Custom Display Components
```python
from voxterm.display import Component

class TokenCounter(Component):
    def render(self, state):
        return f"Tokens: {state.token_count}"

terminal.add_component(TokenCounter(), position="status_right")
```

### Custom Input Modes
```python
from voxterm.input import InputMode

class DoubleTapMode(InputMode):
    """Double-tap space to toggle recording"""
    def handle_key(self, key):
        # Implementation
        pass

terminal.add_input_mode("double_tap", DoubleTapMode())
```

## 📝 Design Principles

1. **Never Block Audio**: UI updates are always async and queued
2. **Delegate to Voice Engine**: VoxTerm doesn't implement VAD, audio processing, or networking
3. **Event-Driven**: Loose coupling via events, not direct calls
4. **Thread-Safe**: UI thread is isolated from audio processing
5. **Extensible**: Easy to add new modes, themes, and integrations

## 🚧 Roadmap

- [ ] Rich TUI mode with `textual`
- [ ] Web terminal via `xterm.js`
- [ ] Recording/playback controls
- [ ] Multi-language UI support
- [ ] Plugin system for custom widgets

## 📄 License

MIT License - Use freely in your voice applications!

---

**Remember**: VoxTerm is just the UI layer. It's designed to make your voice application look good in the terminal without slowing down your real-time audio processing. Your voice engine does the heavy lifting - VoxTerm just makes it pretty! 🎨