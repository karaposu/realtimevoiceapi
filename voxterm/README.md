# VoxTerm - Minimalist CLI for Voice Chat

VoxTerm is a lightweight command-line interface that makes it easy to use voice chat APIs (like OpenAI's Realtime API) from your terminal. No fancy UI, no complex frameworks - just simple keyboard controls for voice conversations.

## 📋 What is VoxTerm?

VoxTerm is a **thin CLI wrapper** that adds keyboard controls to voice engines. Think of it as the minimal glue between your keyboard and a voice API:

```
Your Keyboard → VoxTerm → VoiceEngine → AI Voice API
```

## 🎯 Philosophy

- **Minimalist**: ~500 lines of code total
- **Simple**: Just keyboard input and print statements
- **Focused**: Does one thing - CLI controls for voice chat
- **Non-blocking**: Never interferes with real-time audio
- **Flexible**: Works with any voice engine that has the right methods

## 🚀 Quick Start

```python
from voxterm import VoxTermCLI
from realtimevoiceapi import VoiceEngine

# Create your voice engine
engine = VoiceEngine(api_key="your-key")

# Wrap it with VoxTerm
cli = VoxTermCLI(engine, mode="push_to_talk")

# Run it
import asyncio
asyncio.run(cli.run())
```

That's it! Now you have:
- Hold SPACE to talk
- Press M to mute
- Press Q to quit

## 📁 Project Structure

VoxTerm is intentionally tiny:

```
voxterm/
├── __init__.py      # Package exports
├── cli.py           # Main CLI class (100 lines)
├── modes.py         # Input modes (200 lines)
└── keyboard.py      # Keyboard handling (200 lines)
```

### `cli.py` - The Main CLI
```python
class VoxTermCLI:
    def __init__(self, voice_engine, mode="push_to_talk"):
        self.engine = voice_engine
        self.mode = mode
        
    async def run(self):
        # Connect engine
        # Setup keyboard
        # Print messages
        # That's all!
```

### `modes.py` - Input Modes
Simple classes that handle different interaction patterns:

- **PushToTalkMode**: Hold key → record → release → send
- **AlwaysOnMode**: Continuous listening with VAD
- **TextMode**: Type messages instead of speaking
- **TurnBasedMode**: Explicit turn-taking

Each mode is just a class with `on_key_down()` and `on_key_up()` methods.

### `keyboard.py` - Keyboard Input
Basic keyboard handling that works across platforms:

```python
keyboard = SimpleKeyboard()
keyboard.on_space(on_press_func, on_release_func)
keyboard.on_key('m', mute_func)
keyboard.start()
```

## 🎮 Usage Modes

### Push-to-Talk (Default)
```bash
$ python -m voxterm --mode ptt

🎤 Voice Chat (push_to_talk mode)
Commands: [space] talk, [m] mute, [q] quit

[Hold SPACE to talk...]
🔴 Recording... (2.3s) Sending...
You: How's the weather today?
AI: I don't have access to real-time weather data...
```

### Always-On (VAD)
```bash
$ python -m voxterm --mode always_on

🎤 Always listening (VAD active)
[Just speak naturally, AI will respond when you pause]
```

### Text Mode
```bash
$ python -m voxterm --mode text

💬 Type your messages:
You: Hello!
AI: Hi there! How can I help you today?
```

## 🔧 Integration

VoxTerm expects a voice engine with these methods:

```python
# Required methods
async engine.connect()
async engine.disconnect()
async engine.start_listening()
async engine.stop_listening()
async engine.send_text(text: str)

# Required callbacks
engine.on_text_response = func(text: str)
engine.on_user_transcript = func(text: str)
```

Works out of the box with:
- `realtimevoiceapi.VoiceEngine`
- Any engine with a similar interface

## 🎨 Customization

### Custom Modes
```python
class MyCustomMode:
    def __init__(self, engine):
        self.engine = engine
        
    async def on_key_down(self, key: str):
        if key == "r":  # Custom recording key
            await self.engine.start_listening()
```

### Custom Key Bindings
```python
cli = VoxTermCLI(engine)
cli.keyboard.on_key('t', lambda: print("Custom action!"))
```

## 🚫 What VoxTerm Doesn't Do

- ❌ No UI rendering or colors
- ❌ No audio processing
- ❌ No network/WebSocket handling  
- ❌ No state management
- ❌ No configuration files
- ❌ No fancy terminal graphics

VoxTerm just connects your keyboard to a voice engine. The voice engine handles everything else.

## 📊 Why So Simple?

Real-world usage showed that for CLI voice chat, you need:
1. A way to trigger recording (keyboard)
2. A way to see what was said (print)
3. Different modes for different use cases

That's exactly what VoxTerm provides - nothing more, nothing less.

## 🏃 Example: Complete Voice Chat in 10 Lines

```python
import asyncio
from voxterm import VoxTermCLI
from realtimevoiceapi import VoiceEngine

async def main():
    engine = VoiceEngine(api_key="your-key")
    cli = VoxTermCLI(engine, mode="push_to_talk")
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📝 License

MIT - Use it however you want!

---

**Remember**: VoxTerm is just the keyboard controls. Your voice engine does the actual work. We just make it easy to use from the command line! 🎤