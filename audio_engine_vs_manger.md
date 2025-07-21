## AudioManager vs AudioEngine Relationship

### Current Architecture Options

There are **two valid architectural approaches**:

### Option 1: AudioEngine REPLACES AudioManager
```
┌─────────────────────────────────────────────┐
│                AudioEngine                  │
│         (Complete audio solution)           │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   Directly manages all components    │   │
│  ├─────────────────────────────────────┤   │
│  │ - DirectAudioCapture                 │   │
│  │ - DirectAudioPlayer                  │   │
│  │ - FastVADDetector                    │   │
│  │ - BufferedAudioPlayer                │   │
│  │ - AudioProcessor                     │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**In this approach:**
- AudioManager is **deprecated/removed**
- AudioEngine directly manages all audio components
- BaseEngine only interacts with AudioEngine
- Simpler architecture, fewer layers

### Option 2: AudioEngine USES AudioManager
```
┌─────────────────────────────────────────────┐
│                AudioEngine                  │
│         (High-level orchestration)          │
│                                             │
│  - Advanced features (interrupts, duplex)   │
│  - Network adaptation                       │
│  - Conversation state                       │
└────────────────┬────────────────────────────┘
                 │ Uses
                 ▼
┌─────────────────────────────────────────────┐
│              AudioManager                   │
│         (Low-level management)              │
│                                             │
│  - Device management                        │
│  - Basic capture/playback                   │
│  - Component lifecycle                      │
└─────────────────────────────────────────────┘
```

**In this approach:**
- AudioManager handles **low-level component management**
- AudioEngine adds **high-level features** on top
- Separation of concerns
- More flexible but more complex

## Recommended Approach: Option 1 (Replace)

I recommend **AudioEngine REPLACES AudioManager** for these reasons:

### 1. **Eliminates Redundancy**
AudioManager and AudioEngine have overlapping responsibilities:
- Both manage audio components
- Both handle capture/playback
- Both process audio through pipelines

Having both creates confusion about which to use when.

### 2. **Cleaner Architecture**
```python
# Instead of:
BaseEngine -> AudioEngine -> AudioManager -> Components

# We have:
BaseEngine -> AudioEngine -> Components
```

### 3. **Single Source of Truth**
All audio logic lives in AudioEngine:
- Configuration management
- Resource lifecycle  
- Processing pipelines
- Event handling

### 4. **Easier Testing**
- Mock one system instead of two
- Clear boundaries
- Less integration complexity

## Migration Path

### What AudioEngine Absorbs from AudioManager:

```python
class AudioEngine:
    def __init__(self, config: AudioEngineConfig):
        # Direct component management (from AudioManager)
        self._capture: Optional[DirectAudioCapture] = None
        self._player: Optional[DirectAudioPlayer] = None
        self._vad: Optional[FastVADDetector] = None
        
        # Enhanced features (new in AudioEngine)
        self._processor: Optional[AudioProcessor] = None
        self._buffer_pool: Optional[BufferPool] = None
        self._jitter_buffer: Optional[JitterBuffer] = None
        
        # State management (enhanced from AudioManager)
        self._conversation_state = ConversationState()
        self._network_adapter = NetworkAdapter()
```

### What Gets Enhanced:

| AudioManager Feature | AudioEngine Enhancement |
|---------------------|------------------------|
| Basic capture/playback | + Bidirectional streaming support |
| Simple VAD | + Interruption detection, endpointing |
| Basic state tracking | + Full conversation state machine |
| Device management | + Hot-swapping, fallback devices |
| Simple metrics | + Latency budgets, performance tracking |

### Code Example - Before and After:

**Before (with AudioManager):**
```python
# In BaseEngine
self._audio_manager = AudioManager(config)
await self._audio_manager.initialize()
queue = await self._audio_manager.start_capture()

# Process manually
while True:
    chunk = await queue.get()
    if self._audio_manager.process_vad(chunk):
        processed = self._audio_engine.process_audio(chunk)  # Wait, two systems?
        await self._strategy.send_audio(processed)
```

**After (AudioEngine only):**
```python
# In BaseEngine
self._audio_engine = AudioEngine(config)
await self._audio_engine.initialize()

# Single unified interface
async for audio_event in self._audio_engine.capture_stream():
    # Already processed, VAD-filtered, and enhanced
    await self._strategy.send_audio(audio_event.audio)
```

## Implementation Strategy

### Phase 1: Create AudioEngine with Core Features
```python
class AudioEngine:
    """Replaces AudioManager with enhanced functionality"""
    
    async def initialize(self):
        # Initialize components directly
        self._init_capture_component()
        self._init_playback_component()
        self._init_vad_component()
        self._init_processor()
```

### Phase 2: Migrate BaseEngine
```python
# Change:
# self._audio_manager = AudioManager(...)
# To:
self._audio_engine = AudioEngine.create_for_mode(self._mode)
```

### Phase 3: Remove AudioManager
- Delete `audio_manager.py`
- Update imports
- Update tests

## Conclusion

**AudioEngine should REPLACE AudioManager entirely** because:

1. **Simpler Architecture**: One audio system instead of two
2. **Better Abstraction**: Higher-level interface for BaseEngine
3. **No Confusion**: Clear where audio logic lives
4. **Future-Proof**: Easier to add features to one system
5. **Performance**: Fewer layers = less overhead

The goal is to have AudioEngine be the **single, complete audio subsystem** that BaseEngine uses for all audio operations, making the codebase cleaner and more maintainable.