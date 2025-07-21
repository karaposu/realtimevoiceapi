# Audio Engine Module - Comprehensive Documentation

## Overview

The Audio Engine module is a high-performance, real-time audio processing framework designed for voice communication applications. It provides low-latency audio capture, processing, playback, and voice activity detection (VAD) with adaptive performance optimization.

## Core Components

### 1. AudioEngine
**Location**: `audioengine/audio_engine.py`

The central audio processing component that handles real-time audio transformation with multiple processing modes.

#### Key Features:
- **Multi-mode Processing**: Supports REALTIME, QUALITY, and BALANCED modes
- **Adaptive Performance**: Automatically adjusts processing based on system load
- **Low Latency**: Sub-millisecond processing times in REALTIME mode
- **Metric Tracking**: Comprehensive performance monitoring

#### Usage:
```python
from audioengine import AudioEngine, AudioConfig, ProcessingMode

# Create engine with specific mode
config = AudioConfig(sample_rate=24000)
engine = AudioEngine(mode=ProcessingMode.REALTIME, config=config)

# Process audio
processed = engine.process_audio(audio_bytes)

# Switch modes dynamically
engine.optimize_for_latency()  # Switch to REALTIME
engine.optimize_for_quality()  # Switch to QUALITY

# Get performance metrics
metrics = engine.get_metrics()
```

#### Processing Modes:
- **REALTIME**: Minimal latency (<1ms), suitable for live conversations
- **QUALITY**: Enhanced processing with noise reduction (~5ms latency)
- **BALANCED**: Adaptive mode that balances quality and latency

### 2. AudioProcessor
**Location**: `audioengine/audio_processor.py`

Low-level audio processing with optimized algorithms for each mode.

#### Features:
- **Noise Reduction**: Available in QUALITY mode
- **Echo Suppression**: Built-in echo cancellation
- **Gain Control**: Automatic gain adjustment
- **Buffer Pooling**: Zero-copy buffer management for efficiency

#### Usage:
```python
from audioengine import AudioProcessor, ProcessingMode

processor = AudioProcessor(mode=ProcessingMode.QUALITY, config=config)
processed = processor.process_audio(audio_bytes)

# Access buffer pool for zero-copy operations
buffer = processor.buffer_pool.acquire()
# ... use buffer ...
processor.buffer_pool.release(buffer)
```

### 3. AudioManager
**Location**: `audioengine/audio_manager.py`

High-level audio system coordinator that manages capture, playback, and VAD.

#### Features:
- **Unified Interface**: Single point of control for audio operations
- **VAD Integration**: Built-in voice activity detection
- **State Management**: Tracks audio system state
- **Event Handling**: Callbacks for audio events

#### Usage:
```python
from audioengine import AudioManager, AudioManagerConfig, VADConfig

config = AudioManagerConfig(
    sample_rate=24000,
    vad_enabled=True,
    vad_config=VADConfig(
        type=VADType.ENERGY_BASED,
        energy_threshold=0.02
    )
)

manager = AudioManager(config)

# Process audio with VAD
vad_state = manager.process_vad(audio_bytes)  # Returns: 'speech', 'silence', etc.

# Handle capture/playback (if interfaces available)
manager.start_capture()
manager.stop_capture()
```

### 4. BufferedAudioPlayer
**Location**: `audioengine/buffered_audio_player.py`

Smart audio playback with buffering for streaming scenarios.

#### Features:
- **Intelligent Buffering**: Accumulates chunks before playback to prevent gaps
- **Completion Tracking**: Knows when all audio has been played
- **Batch Processing**: Efficiently plays multiple chunks together
- **Metrics Collection**: Tracks playback statistics
- **Callback Support**: Notifies on completion and chunk playback

#### Usage:
```python
from audioengine import BufferedAudioPlayer, AudioConfig

config = AudioConfig(sample_rate=24000)
player = BufferedAudioPlayer(config)

# Set callbacks
def on_complete():
    print("Playback finished!")

def on_chunk_played(num_chunks):
    print(f"Played {num_chunks} chunks")

player.set_completion_callback(on_complete)
player.set_chunk_played_callback(on_chunk_played)

# Stream audio chunks
for chunk in audio_stream:
    player.play(chunk)

# Mark when no more audio will come
player.mark_complete()

# Get metrics
metrics = player.get_metrics()
# {'chunks_received': 50, 'chunks_played': 50, 'is_playing': False, ...}

# Stop playback
player.stop(force=True)  # force=True immediately stops
```

#### Key Properties:
- `is_playing`: Currently playing audio
- `is_actively_playing`: Has audio in buffer and playing
- `buffer_duration_ms`: Total duration of buffered audio
- `chunks_played`: Number of chunks played
- `is_complete`: All audio has been played

### 5. FastVADDetector
**Location**: `audioengine/fast_vad_detector.py`

High-performance voice activity detection with minimal latency.

#### Features:
- **Multiple Algorithms**: Energy-based and advanced ML-based detection
- **Pre-buffering**: Captures audio before speech starts
- **State Machine**: Smooth transitions between speech/silence
- **Event Callbacks**: Notifies on speech start/end
- **Configurable Sensitivity**: Adjustable thresholds and timing

#### Usage:
```python
from audioengine import FastVADDetector, VADConfig, VADType

vad_config = VADConfig(
    type=VADType.ENERGY_BASED,
    energy_threshold=0.02,
    speech_start_ms=50,      # Time to confirm speech start
    speech_end_ms=200,       # Time to confirm speech end
    pre_buffer_ms=200        # Audio to capture before speech
)

vad = FastVADDetector(config=vad_config, audio_config=audio_config)

# Set event handlers
vad.on_speech_start = lambda: print("Speech started!")
vad.on_speech_end = lambda: print("Speech ended!")

# Process audio chunks
for chunk in audio_stream:
    state = vad.process_chunk(chunk)
    # Returns: 'silence', 'speech_starting', 'speech', 'speech_ending'
    
    if state == 'speech_starting':
        # Get audio from before speech started
        prebuffer = vad.get_pre_buffer()
```

#### VAD States:
- `silence`: No speech detected
- `speech_starting`: Potential speech detected, confirming
- `speech`: Active speech
- `speech_ending`: Speech may have ended, confirming
- `speech_continuing`: Speech ongoing (for long utterances)

### 6. Audio Types and Configuration
**Location**: `audioengine/audio_types.py`

#### AudioConfig
```python
@dataclass
class AudioConfig:
    sample_rate: int = 24000      # Hz
    channels: int = 1             # Mono/Stereo
    bits_per_sample: int = 16     # Bit depth
    frame_duration_ms: int = 20   # Chunk size
```

#### ProcessingMode
```python
class ProcessingMode(Enum):
    REALTIME = "realtime"    # Lowest latency
    QUALITY = "quality"      # Best quality
    BALANCED = "balanced"    # Adaptive
```

#### VADConfig
```python
@dataclass
class VADConfig:
    type: VADType = VADType.ENERGY_BASED
    energy_threshold: float = 0.01
    speech_start_ms: int = 100
    speech_end_ms: int = 300
    pre_buffer_ms: int = 200
    min_speech_duration_ms: int = 50
```

## Factory Functions

### create_fast_lane_engine
Creates an optimized engine for minimal latency:
```python
engine = create_fast_lane_engine(sample_rate=24000, chunk_duration_ms=20)
```

### create_adaptive_engine
Creates an engine that automatically adjusts processing mode:
```python
engine = create_adaptive_engine(
    sample_rate=24000,
    chunk_duration_ms=20,
    latency_target_ms=30.0,
    quality_threshold_ms=50.0
)
```

## Performance Characteristics

### Processing Latencies
- **REALTIME mode**: < 1ms per 20ms chunk
- **QUALITY mode**: < 5ms per 20ms chunk
- **BALANCED mode**: 1-5ms (adaptive)

### Throughput
- Processes > 1000 chunks/second
- Maintains 2x realtime factor (processes faster than realtime)

### Memory Usage
- Efficient buffer pooling reduces allocations
- Pre-allocated buffers for zero-copy operations
- Configurable buffer sizes

## Common Usage Patterns

### 1. Real-time Voice Conversation
```python
# Setup
config = AudioConfig(sample_rate=24000)
engine = create_fast_lane_engine(sample_rate=24000)
vad = FastVADDetector(config=VADConfig(type=VADType.ENERGY_BASED))
player = BufferedAudioPlayer(config)

# Process incoming audio
def process_user_audio(chunk):
    processed = engine.process_audio(chunk)
    vad_state = vad.process_chunk(processed)
    
    if vad_state in ['speech', 'speech_starting']:
        send_to_network(processed)

# Play received audio
def play_received_audio(chunk):
    player.play(chunk)
```

### 2. Adaptive Quality Processing
```python
engine = create_adaptive_engine(latency_target_ms=30.0)

# Engine automatically switches modes based on performance
for chunk in audio_stream:
    processed = engine.process_audio(chunk)
    # Mode switches happen automatically
```

### 3. Speech Detection with Pre-buffer
```python
vad = FastVADDetector(
    config=VADConfig(pre_buffer_ms=200),
    audio_config=config
)

chunks_buffer = []

for chunk in audio_stream:
    state = vad.process_chunk(chunk)
    
    if state == 'speech_starting':
        # Get audio from before speech started
        prebuffer = vad.get_pre_buffer()
        send_audio(prebuffer)  # Send pre-speech audio
        chunks_buffer = [chunk]
    elif state == 'speech':
        chunks_buffer.append(chunk)
    elif state == 'speech_ending':
        # Send complete utterance
        send_utterance(chunks_buffer)
        chunks_buffer = []
```

### 4. Streaming Audio Playback
```python
player = BufferedAudioPlayer(config)

# Stream chunks as they arrive
async def audio_stream_handler():
    async for chunk in audio_stream:
        player.play(chunk)
    
    # Signal end of stream
    player.mark_complete()
    
    # Wait for completion
    while player.is_playing:
        await asyncio.sleep(0.1)
```

## Error Handling

The audio engine includes comprehensive error handling:

```python
from audioengine.exceptions import AudioError, AudioConfigError

try:
    engine.process_audio(invalid_audio)
except AudioError as e:
    print(f"Audio processing error: {e}")
```

## Thread Safety

- **AudioEngine**: Thread-safe for processing operations
- **BufferedAudioPlayer**: Thread-safe, uses internal locking
- **FastVADDetector**: Not thread-safe, use one instance per thread
- **AudioManager**: Thread-safe for all operations

## Best Practices

1. **Choose the Right Mode**:
   - Use REALTIME for live conversations
   - Use QUALITY for recordings or when latency isn't critical
   - Use BALANCED for adaptive scenarios

2. **Buffer Management**:
   - Reuse buffers when possible using BufferPool
   - Process audio in consistent chunk sizes (typically 20ms)

3. **VAD Configuration**:
   - Adjust thresholds based on environment noise
   - Use pre-buffering for complete utterance capture
   - Set appropriate timing for your use case

4. **Performance Monitoring**:
   - Regularly check metrics to detect issues
   - Monitor latency to ensure real-time performance
   - Track buffer depths to prevent overflow

5. **Resource Cleanup**:
   - Always call stop() on players when done
   - Release buffers back to pool
   - Properly shutdown components

## Integration Example

```python
from audioengine import (
    AudioEngine, BufferedAudioPlayer, FastVADDetector,
    AudioConfig, ProcessingMode, VADConfig, VADType,
    create_fast_lane_engine
)

class VoiceProcessor:
    def __init__(self):
        self.config = AudioConfig(sample_rate=24000)
        self.engine = create_fast_lane_engine(sample_rate=24000)
        self.vad = FastVADDetector(
            config=VADConfig(
                type=VADType.ENERGY_BASED,
                pre_buffer_ms=200
            ),
            audio_config=self.config
        )
        self.player = BufferedAudioPlayer(self.config)
        
    def process_microphone_input(self, audio_chunk):
        # Process audio
        processed = self.engine.process_audio(audio_chunk)
        
        # Detect speech
        vad_state = self.vad.process_chunk(processed)
        
        if vad_state == 'speech_starting':
            # Include pre-buffer for complete utterance
            prebuffer = self.vad.get_pre_buffer()
            return prebuffer + processed, True
        elif vad_state in ['speech', 'speech_continuing']:
            return processed, True
        else:
            return processed, False
    
    def play_response(self, audio_chunk):
        self.player.play(audio_chunk)
    
    def get_metrics(self):
        return {
            'engine': self.engine.get_metrics(),
            'playback': self.player.get_metrics()
        }
```

## Limitations and Considerations

1. **Platform Dependencies**: Some features may require platform-specific audio interfaces
2. **Sample Rate**: Optimized for 24kHz, other rates may have different performance
3. **Chunk Size**: Best performance with 20ms chunks (480 samples at 24kHz)
4. **Memory**: Pre-allocates buffers, consider memory usage for embedded systems

This comprehensive audio engine provides all the building blocks needed for real-time voice applications with a focus on low latency, high quality, and reliable performance.