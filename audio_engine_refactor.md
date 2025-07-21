

## How AudioEngine Integrates with BaseEngine

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        BaseEngine                           │
│  (Orchestrates voice interaction flow)                      │
│                                                             │
│  - Connection management                                    │
│  - Event routing                                           │
│  - Strategy coordination                                    │
│  - State management                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │ Delegates all audio operations
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                       AudioEngine                           │
│  (Handles all audio operations)                            │
│                                                             │
│  - Audio capture/playback                                   │
│  - Processing & enhancement                                 │
│  - VAD & interruption                                      │
│  - Buffering & streaming                                    │
└─────────────────────────────────────────────────────────────┘
```

### Primary Purpose

AudioEngine serves as the **complete audio subsystem** for BaseEngine, providing:

1. **Abstraction**: BaseEngine doesn't need to know about audio implementation details
2. **Modularity**: Audio logic can be developed and tested independently
3. **Flexibility**: Easy to swap audio implementations or add new features
4. **Performance**: Centralized optimization and resource management

### How BaseEngine Uses AudioEngine

#### 1. **Initialization Phase**
```python
# In BaseEngine
async def setup_fast_lane_audio(self, sample_rate, chunk_duration_ms, ...):
    # Create AudioEngine with appropriate configuration
    self._audio_engine = AudioEngine.create_for_fast_lane(
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        vad_config=vad_config
    )
    
    # Initialize the engine
    await self._audio_engine.initialize()
    
    # BaseEngine doesn't manage individual components anymore
    # Just uses AudioEngine's unified interface
```

#### 2. **Audio Capture Flow**
```python
# Old way (BaseEngine managing components):
# self._audio_manager.start_capture()
# audio_queue = self._audio_manager.get_queue()
# vad_state = self._audio_manager.process_vad(chunk)

# New way (through AudioEngine):
async def start_audio_processing(self):
    # Start capture with built-in processing
    audio_stream = await self._audio_engine.start_capture()
    
    async for audio_event in audio_stream:
        if audio_event.type == AudioEventType.AUDIO_CHUNK:
            # Audio is already processed and VAD-filtered
            await self._strategy.send_audio(audio_event.data)
        elif audio_event.type == AudioEventType.SPEECH_END:
            # Handle end of speech
            await self._handle_turn_change()
```

#### 3. **Audio Playback Flow**
```python
# BaseEngine receives audio from API
async def _on_audio_chunk(self, event: StreamEvent):
    audio_data = event.data["audio"]
    
    # Queue for playback through AudioEngine
    await self._audio_engine.queue_playback(audio_data)
    
    # AudioEngine handles buffering, enhancement, and playback
    # BaseEngine just needs to know when playback is complete
```

#### 4. **Interruption Handling**
```python
# User interrupts while AI is speaking
async def interrupt(self):
    # Single call to AudioEngine handles everything
    interrupt_result = await self._audio_engine.interrupt_playback()
    
    # AudioEngine returns what was interrupted and cleans up
    if interrupt_result.audio_flushed_ms > 0:
        self.logger.info(f"Interrupted {interrupt_result.audio_flushed_ms}ms of audio")
    
    # Notify strategy about interruption
    await self._strategy.interrupt()
```

### Key Integration Points

#### 1. **Event System Integration**
```python
# AudioEngine provides events that BaseEngine subscribes to
self._audio_engine.subscribe(AudioEventType.VAD_SPEECH_START, self._on_speech_start)
self._audio_engine.subscribe(AudioEventType.VAD_SPEECH_END, self._on_speech_end)
self._audio_engine.subscribe(AudioEventType.PLAYBACK_COMPLETE, self._on_playback_done)
```

#### 2. **State Synchronization**
```python
# BaseEngine queries AudioEngine for state
@property
def is_ai_speaking(self) -> bool:
    return self._audio_engine.is_playing_audio()

@property
def is_user_speaking(self) -> bool:
    return self._audio_engine.get_vad_state() == VADState.SPEAKING
```

#### 3. **Mode Management**
```python
# BaseEngine can switch audio modes based on strategy
if self._mode == "fast":
    self._audio_engine.optimize_for_latency()
else:
    self._audio_engine.optimize_for_quality()
```

### Specific Use Cases

#### 1. **Fast Lane Voice Interaction**
```python
# BaseEngine sets up fast lane
await self._audio_engine.configure_fast_lane(
    max_latency_ms=50,
    vad_threshold=0.02,
    interrupt_threshold_ms=30
)

# Audio flows through with minimal latency
# AudioEngine handles all optimizations internally
```

#### 2. **Network Adaptation**
```python
# BaseEngine monitors network conditions
if network_latency > 100:
    # Tell AudioEngine to adapt
    self._audio_engine.configure_jitter_buffer(
        min_delay_ms=50,
        max_delay_ms=200
    )
```

#### 3. **Testing and Debugging**
```python
# BaseEngine can use AudioEngine's test features
if self._test_mode:
    # Inject test audio instead of real capture
    self._audio_engine.inject_test_audio(test_audio_file)
    
    # Record session for analysis
    self._audio_engine.start_session_recording()
```

### Benefits of This Architecture

1. **Simplified BaseEngine Code**
   - No direct audio device management
   - No complex audio processing logic
   - Cleaner event handling

2. **Better Testing**
   ```python
   # Can mock entire AudioEngine for BaseEngine tests
   mock_engine = MockAudioEngine()
   mock_engine.simulate_user_speech("Hello AI")
   mock_engine.simulate_network_delay(100)
   ```

3. **Performance Optimization**
   - AudioEngine can optimize internally without BaseEngine changes
   - Resource pooling and caching managed in one place
   - Mode switching handled transparently

4. **Feature Addition**
   - New audio features added to AudioEngine
   - BaseEngine automatically benefits
   - No need to modify BaseEngine for audio improvements

### Example: Complete Voice Interaction Flow

```python
# 1. User starts speaking
# AudioEngine detects speech via VAD
await audio_engine.event: VADSpeechStart

# 2. BaseEngine receives event
async def on_speech_start(event):
    self._state = State.USER_SPEAKING
    await self._strategy.begin_user_turn()

# 3. Audio chunks flow through
# AudioEngine -> processes -> filters -> sends to BaseEngine
# BaseEngine -> forwards to strategy/API

# 4. User stops speaking  
await audio_engine.event: VADSpeechEnd

# 5. BaseEngine triggers response
async def on_speech_end(event):
    await self._strategy.commit_user_input()
    await self._strategy.request_response()

# 6. AI response arrives
# BaseEngine receives audio chunks from API
# Forwards to AudioEngine for playback

# 7. AudioEngine handles playback
# - Buffers for smooth playback
# - Applies enhancements
# - Monitors for completion

# 8. Playback completes
await audio_engine.event: PlaybackComplete
# BaseEngine knows AI finished speaking
```

This architecture creates a clean separation where BaseEngine handles the high-level voice interaction flow while AudioEngine manages all audio-specific complexity.


## Benefits of Using AudioEngine Instead of Direct Submodule Access

### 1. **Centralized Control & Consistency**
- Single interface for all audio operations instead of managing multiple components
- Consistent error handling and state management across all audio operations
- Unified configuration management - change once, applies everywhere

### 2. **Abstraction & Flexibility**
- BaseEngine doesn't need to know about implementation details (DirectAudioCapture, VAD, etc.)
- Easy to swap implementations (e.g., switch from sounddevice to PyAudio)
- Can add new audio sources/sinks without changing BaseEngine

### 3. **Performance Optimization**
- Centralized resource pooling and buffer management
- Intelligent mode switching based on load
- Zero-copy optimizations managed in one place
- Pipeline optimization for specific use cases

### 4. **Simplified Testing**
- Mock one interface instead of multiple components
- Test audio logic independently from BaseEngine
- Easier to create test scenarios (fake audio, controlled timing)

### 5. **Better Resource Management**
- Coordinated lifecycle management
- Prevents resource leaks through centralized cleanup
- Manages thread safety in one place

## Comprehensive AudioEngine Capabilities

### 1. **Audio Capture Management**
**Description**: Manage audio input from any source (microphone, file, stream)
**External Interface**:
```python
async def start_capture() -> AsyncIterator[AudioBytes]
async def stop_capture() -> None
def set_capture_device(device_id: Optional[int]) -> None
```
**Requirements**:
- Support multiple audio sources (hardware, file, network)
- Configurable chunk sizes and sample rates
- Thread-safe queue management
**Limitations**:
- Single capture source at a time
- Fixed format during capture session

### 2. **Audio Playback Management**
**Description**: Handle audio output to any sink (speaker, file, buffer)
**External Interface**:
```python
async def play_audio(audio: AudioBytes) -> None
def play_audio_sync(audio: AudioBytes) -> bool
async def stop_playback() -> None
def set_playback_device(device_id: Optional[int]) -> None
```
**Requirements**:
- Non-blocking playback
- Queue management for continuous playback
- Support for different output sinks
**Limitations**:
- Single playback stream per engine
- Format must match configuration

### 3. **Audio Processing Pipeline**
**Description**: Process audio through configurable pipeline (enhance, filter, convert)
**External Interface**:
```python
def process_audio(audio: AudioBytes) -> AudioBytes
async def process_audio_async(audio: AudioBytes) -> AudioBytes
def add_processor(processor: AudioProcessor, stage: ProcessingStage) -> None
def remove_processor(processor_id: str) -> None
```
**Requirements**:
- Support for multiple processing stages
- Zero-copy processing for fast lane
- Configurable processing based on mode
**Limitations**:
- Processing adds latency
- Some processors require specific formats

### 4. **Voice Activity Detection (VAD)**
**Description**: Detect speech/silence in audio streams
**External Interface**:
```python
def enable_vad(config: VADConfig) -> None
def disable_vad() -> None
def process_vad(audio: AudioBytes) -> VADState
async def get_vad_events() -> AsyncIterator[VADEvent]
```
**Requirements**:
- Multiple VAD algorithms (energy, ML-based)
- Configurable thresholds and timings
- Event-based notifications
**Limitations**:
- Adds processing overhead
- May have false positives/negatives

### 5. **Buffering and Stream Management**
**Description**: Manage audio buffers for smooth playback and recording
**External Interface**:
```python
def create_stream_buffer(config: BufferConfig) -> StreamBuffer
def get_buffer_status() -> BufferStatus
def flush_buffers() -> None
async def wait_for_buffer_ready() -> None
```
**Requirements**:
- Configurable buffer sizes
- Overflow/underflow handling
- Metrics for buffer health
**Limitations**:
- Memory usage scales with buffer size
- Adds latency for larger buffers

### 6. **Mode Management**
**Description**: Switch between fast lane (low latency) and quality modes
**External Interface**:
```python
def set_mode(mode: ProcessingMode) -> None
def get_current_mode() -> ProcessingMode
def enable_adaptive_mode(config: AdaptiveConfig) -> None
def force_fast_lane() -> None
```
**Requirements**:
- Seamless mode switching
- Preserve audio continuity during switch
- Adaptive switching based on metrics
**Limitations**:
- Mode switches may cause brief audio artifacts
- Some features unavailable in fast lane

### 7. **Resource Lifecycle Management**
**Description**: Initialize, manage, and cleanup all audio resources
**External Interface**:
```python
async def initialize() -> None
async def shutdown() -> None
async def reset() -> None
def get_resource_status() -> ResourceStatus
```
**Requirements**:
- Graceful initialization with fallbacks
- Clean shutdown without hangs
- Resource leak prevention
**Limitations**:
- Initialization may take time
- Some resources may not be available

### 8. **Metrics and Monitoring**
**Description**: Provide detailed performance and health metrics
**External Interface**:
```python
def get_metrics() -> AudioMetrics
def get_performance_stats() -> PerformanceStats
def enable_detailed_metrics() -> None
def export_metrics() -> Dict[str, Any]
```
**Requirements**:
- Real-time performance metrics
- Historical data for analysis
- Low overhead collection
**Limitations**:
- Detailed metrics add slight overhead
- Memory usage for history

### 9. **Error Handling and Recovery**
**Description**: Handle errors gracefully and attempt recovery
**External Interface**:
```python
def set_error_handler(handler: ErrorHandler) -> None
async def recover_from_error() -> bool
def get_error_history() -> List[AudioError]
def set_recovery_policy(policy: RecoveryPolicy) -> None
```
**Requirements**:
- Categorized error types
- Automatic recovery attempts
- Error notification system
**Limitations**:
- Not all errors are recoverable
- Recovery may interrupt audio

### 10. **Configuration Management**
**Description**: Manage all audio configuration in one place
**External Interface**:
```python
def update_config(config: AudioConfig) -> None
def get_config() -> AudioConfig
def validate_config(config: AudioConfig) -> ValidationResult
def apply_preset(preset: AudioPreset) -> None
```
**Requirements**:
- Runtime configuration updates
- Configuration validation
- Preset management
**Limitations**:
- Some settings require restart
- Hardware limitations apply

### 11. **Event System**
**Description**: Provide event-based notifications for audio events
**External Interface**:
```python
def subscribe(event_type: AudioEventType, callback: Callable) -> str
def unsubscribe(subscription_id: str) -> None
async def wait_for_event(event_type: AudioEventType) -> AudioEvent
def emit_event(event: AudioEvent) -> None
```
**Requirements**:
- Async and sync event handlers
- Event filtering and routing
- Thread-safe event dispatch
**Limitations**:
- Event handlers shouldn't block
- Memory usage with many subscribers

### 12. **Audio Format Management**
**Description**: Handle format conversion and validation
**External Interface**:
```python
def convert_format(audio: AudioBytes, target: AudioFormat) -> AudioBytes
def validate_format(audio: AudioBytes) -> bool
def get_supported_formats() -> List[AudioFormat]
def negotiate_format(requested: AudioFormat) -> AudioFormat
```
**Requirements**:
- Support common audio formats
- Efficient format conversion
- Format negotiation
**Limitations**:
- Some conversions lose quality
- Conversion adds latency

### 13. **Queue and Flow Control**
**Description**: Manage audio data flow between components
**External Interface**:
```python
def get_capture_queue() -> AudioQueue
def get_playback_queue() -> AudioQueue
def set_flow_control(policy: FlowControlPolicy) -> None
def get_queue_metrics() -> QueueMetrics
```
**Requirements**:
- Backpressure handling
- Priority queue support
- Queue monitoring
**Limitations**:
- Queue size limits memory
- May drop audio under pressure

### 14. **Testing and Debugging Support**
**Description**: Built-in testing and debugging capabilities
**External Interface**:
```python
def enable_debug_mode() -> None
def inject_test_audio(audio: AudioBytes) -> None
def record_session(filepath: str) -> None
def simulate_error(error_type: ErrorType) -> None
```
**Requirements**:
- Audio injection for testing
- Session recording/playback
- Debug logging
**Limitations**:
- Debug mode affects performance
- Recording uses disk space

### 15. **State Persistence**
**Description**: Save and restore audio engine state
**External Interface**:
```python
def save_state() -> AudioEngineState
def restore_state(state: AudioEngineState) -> None
def export_session() -> SessionData
def import_session(data: SessionData) -> None
```
**Requirements**:
- Serialize configuration and metrics
- Session continuity
- State validation
**Limitations**:
- Hardware state not preserved
- Large states affect performance

This comprehensive capability list ensures AudioEngine can serve as a complete audio subsystem for BaseEngine while remaining independently testable and reusable.

### 16. **Realtime Throughput Management**
**Description**: Ensure constant low-latency processing for continuous audio streams without accumulating delay
**External Interface**:
```python
def enable_realtime_mode(constraints: RealtimeConstraints) -> None
def get_latency_metrics() -> LatencyMetrics
def set_max_acceptable_latency(ms: float) -> None
async def monitor_throughput() -> AsyncIterator[ThroughputEvent]
```
**Requirements**:
- Process audio chunks faster than real-time (< chunk duration)
- Maintain constant latency over extended periods
- Automatic degradation if unable to maintain throughput
- Continuous latency monitoring and reporting
**Limitations**:
- Requires dedicated CPU resources
- May sacrifice quality for latency
- System load affects performance

### 17. **Bidirectional (Full Duplex) Streaming**
**Description**: Handle simultaneous audio capture and playback for natural conversations
**External Interface**:
```python
async def start_full_duplex() -> Tuple[AudioStream, AudioStream]
def set_duplex_mode(mode: DuplexMode) -> None
def enable_echo_cancellation(config: EchoCancelConfig) -> None
def get_duplex_metrics() -> DuplexMetrics
```
**Requirements**:
- Synchronize capture and playback clocks
- Prevent feedback loops
- Maintain separate processing paths
- Handle acoustic echo cancellation (AEC)
**Limitations**:
- Increased CPU usage
- AEC may affect audio quality
- Hardware must support full duplex

### 18. **Interrupt and Preemption Handling**
**Description**: Instantly stop AI playback when user starts speaking
**External Interface**:
```python
async def interrupt_playback(reason: InterruptReason) -> None
def set_interrupt_threshold(threshold: InterruptThreshold) -> None
def enable_barge_in_detection() -> None
async def wait_for_interrupt() -> InterruptEvent
def get_interrupt_latency() -> float
```
**Requirements**:
- Sub-50ms interrupt response time
- Clean audio stopping without artifacts
- Preserve user audio during interruption
- Queue flushing and state cleanup
**Limitations**:
- May cut off AI mid-word
- Requires fast VAD for detection
- Network delays affect response

### 19. **Audio Queue Management**
**Description**: Manage audio queues for smooth streaming with rapid flushing
**External Interface**:
```python
def flush_playback_queue() -> int  # Returns flushed byte count
def flush_all_queues() -> None
def set_queue_priority(priority: QueuePriority) -> None
def get_queue_depths() -> QueueDepths
async def drain_queues(timeout_ms: float) -> bool
```
**Requirements**:
- Instant queue flushing (< 10ms)
- Priority-based queue management
- Atomic queue operations
- Queue depth monitoring
**Limitations**:
- Flushing loses buffered audio
- May cause brief silence
- Memory usage with deep queues

### 20. **Network Resilience and Jitter Buffering**
**Description**: Handle network variations without affecting audio quality
**External Interface**:
```python
def configure_jitter_buffer(config: JitterBufferConfig) -> None
def set_network_adaptation(policy: NetworkAdaptationPolicy) -> None
def get_network_stats() -> NetworkStats
def simulate_network_conditions(conditions: NetworkConditions) -> None
async def handle_packet_loss() -> PacketLossEvent
```
**Requirements**:
- Adaptive jitter buffer (20-200ms)
- Packet loss concealment
- Dynamic bitrate adjustment
- Network quality monitoring
**Limitations**:
- Jitter buffer adds latency
- Quality degradation with high packet loss
- Requires network statistics

### 21. **Conversation State Management**
**Description**: Track and manage conversation flow states
**External Interface**:
```python
def get_conversation_state() -> ConversationState
def set_turn_taking_mode(mode: TurnTakingMode) -> None
async def wait_for_turn_change() -> TurnChangeEvent
def force_turn_change(to: Speaker) -> None
def get_conversation_metrics() -> ConversationMetrics
```
**Requirements**:
- Track who is speaking
- Manage turn-taking logic
- Handle overlapping speech
- Conversation flow metrics
**Limitations**:
- Complex with multiple speakers
- Cultural differences in turn-taking
- Requires careful VAD tuning

### 22. **Audio Continuity and Gap Handling**
**Description**: Ensure continuous audio even with processing delays or network issues
**External Interface**:
```python
def enable_gap_filling(strategy: GapFillingStrategy) -> None
def set_continuity_threshold(ms: float) -> None
def get_gap_statistics() -> GapStats
async def monitor_continuity() -> AsyncIterator[ContinuityEvent]
```
**Requirements**:
- Detect audio gaps > 20ms
- Fill gaps with comfort noise or interpolation
- Maintain natural sound
- Gap occurrence tracking
**Limitations**:
- Gap filling may sound artificial
- Adds complexity to pipeline
- Memory for gap detection

### 23. **Latency Budget Management**
**Description**: Allocate and track latency budgets across processing stages
**External Interface**:
```python
def set_latency_budget(budget: LatencyBudget) -> None
def allocate_stage_budget(stage: ProcessingStage, ms: float) -> None
def get_latency_breakdown() -> LatencyBreakdown
def enable_auto_optimization() -> None
async def monitor_budget_violations() -> AsyncIterator[BudgetViolation]
```
**Requirements**:
- Track per-stage latency
- Automatic optimization when over budget
- Detailed latency reporting
- Budget violation alerts
**Limitations**:
- Optimization may reduce quality
- Minimum latencies exist
- Hardware dependent

### 24. **Voice Endpoint Detection**
**Description**: Detect when user has finished speaking for response generation
**External Interface**:
```python
def configure_endpointing(config: EndpointConfig) -> None
def get_endpoint_state() -> EndpointState
async def wait_for_endpoint() -> EndpointEvent
def set_endpoint_timeout(ms: float) -> None
def force_endpoint() -> None
```
**Requirements**:
- Accurate end-of-speech detection
- Configurable silence thresholds
- Cultural speech pattern adaptation
- Fast decision making
**Limitations**:
- May cut off slow speakers
- Pauses may trigger false endpoints
- Language-dependent accuracy

### 25. **Audio Session Recording and Replay**
**Description**: Record entire conversation sessions for analysis and debugging
**External Interface**:
```python
def start_session_recording(config: RecordingConfig) -> str
def stop_session_recording() -> SessionRecording
def replay_session(recording: SessionRecording) -> None
def export_session_audio(format: ExportFormat) -> bytes
def get_session_timeline() -> SessionTimeline
```
**Requirements**:
- Record all audio streams
- Synchronize multiple streams
- Include metadata and events
- Efficient storage format
**Limitations**:
- High disk/memory usage
- Privacy considerations
- Performance impact

### 26. **Adaptive Quality Control**
**Description**: Dynamically adjust audio quality based on system conditions
**External Interface**:
```python
def enable_adaptive_quality() -> None
def set_quality_constraints(constraints: QualityConstraints) -> None
def get_quality_metrics() -> QualityMetrics
def force_quality_level(level: QualityLevel) -> None
async def monitor_quality_changes() -> AsyncIterator[QualityChange]
```
**Requirements**:
- Monitor system resources
- Gradual quality adjustments
- Maintain minimum quality threshold
- User-configurable preferences
**Limitations**:
- Quality changes may be noticeable
- Requires continuous monitoring
- Trade-offs with latency

### 27. **Multi-Stream Synchronization**
**Description**: Synchronize multiple audio streams for complex scenarios
**External Interface**:
```python
def create_sync_group(streams: List[AudioStream]) -> SyncGroup
def set_sync_tolerance(ms: float) -> None
def get_sync_status() -> SyncStatus
async def wait_for_sync() -> None
def force_resync() -> None
```
**Requirements**:
- Clock synchronization
- Drift compensation
- Stream alignment
- Sync status monitoring
**Limitations**:
- Complexity with many streams
- Network streams harder to sync
- CPU overhead









Yes, this refactor makes excellent sense and can absolutely be done without hurting real-time capabilities! In fact, it's a very professional architecture pattern used in high-performance audio systems. Here's why:

## Why This Architecture Works for Real-time

### 1. **It's a Coordination Layer, Not a Processing Layer**
AudioEngine acts as a thin orchestrator that:
- Manages component lifecycle
- Routes data efficiently
- Coordinates state
- **Does NOT** add processing steps to the audio path

### 2. **Zero-Copy Audio Path Preservation**
```python
class AudioEngine:
    def process_audio(self, audio_bytes: AudioBytes) -> AudioBytes:
        # Direct pass-through to processors
        # No copying, just pointer passing
        return self._active_processor.process(audio_bytes)
```

### 3. **Direct Memory Access Patterns**
```python
# Bad: Creates copies
def route_audio(self, data):
    copy = data[:]  # DON'T DO THIS
    return self.processor.process(copy)

# Good: Passes references
def route_audio(self, data):
    return self.processor.process(data)  # Direct reference
```

## Implementation Strategy for Real-time Performance

### 1. **Use Composition, Not Inheritance**
```python
class AudioEngine:
    def __init__(self):
        # Direct access to components when needed
        self._capture = DirectAudioCapture()
        self._player = BufferedAudioPlayer()
        self._vad = VADProcessor()
        
    @property
    def capture_queue(self):
        # Direct access for hot paths
        return self._capture._queue
```

### 2. **Optimize Hot Paths**
```python
class AudioEngine:
    async def stream_audio(self):
        # Fast path: minimal overhead
        if self._mode == AudioMode.FAST_LANE:
            # Direct streaming, no extra processing
            async for chunk in self._capture.stream_direct():
                yield chunk
        else:
            # Quality mode: can add processing
            async for chunk in self._capture.stream():
                processed = self._enhance_audio(chunk)
                yield processed
```

### 3. **Lock-Free Event Routing**
```python
class AudioEngine:
    def __init__(self):
        # Pre-allocated ring buffers for events
        self._event_ring = RingBuffer(size=1024)
        self._use_direct_callbacks = True
        
    def emit_event(self, event):
        if self._use_direct_callbacks:
            # Direct callback - no queuing
            self._invoke_handlers(event)
        else:
            # Async mode - queue for later
            self._event_ring.put_nowait(event)
```

### 4. **Thread-Aware Design**
```python
class AudioEngine:
    def __init__(self):
        # Components share thread when possible
        self._audio_thread = threading.Thread(target=self._audio_loop)
        self._capture.use_thread(self._audio_thread)
        self._player.use_thread(self._audio_thread)
```

## Real-time Preserving Patterns

### 1. **Inline Critical Operations**
```python
class AudioEngine:
    # Use inline methods for critical paths
    @inline
    def should_process_audio(self) -> bool:
        return self._active and not self._muted
    
    async def capture_loop(self):
        while self.should_process_audio():  # Inlined check
            chunk = await self._capture.get_chunk()
            # Direct processing...
```

### 2. **Pre-allocate Resources**
```python
class AudioEngine:
    def __init__(self):
        # Pre-allocate buffers
        self._buffer_pool = BufferPool(
            count=32,
            size=self._chunk_size
        )
        
        # Pre-compile processing chains
        self._fast_chain = self._compile_fast_chain()
```

### 3. **Conditional Feature Loading**
```python
class AudioEngine:
    def enable_feature(self, feature: AudioFeature):
        if feature == AudioFeature.ECHO_CANCELLATION:
            # Only load if needed
            if not self._echo_canceller:
                self._echo_canceller = EchoCanceller()
            self._processing_chain.add(self._echo_canceller)
```

## Performance Benchmarks to Maintain

With proper implementation, AudioEngine should maintain:

1. **Latency overhead**: < 0.1ms per call
2. **Memory overhead**: < 1MB for coordination
3. **CPU overhead**: < 1% for routing/coordination
4. **Audio path latency**: Identical to direct access

## Example: Fast Lane Audio Path

```python
# Before (BaseEngine managing directly)
async def audio_loop(self):
    while True:
        chunk = await self._audio_capture.get_chunk()  # 10μs
        vad_result = self._vad.process(chunk)         # 100μs  
        if vad_result.is_speech:
            await self._ws.send(chunk)                 # 50μs
        # Total: 160μs

# After (Through AudioEngine)
async def audio_loop(self):
    async for audio_event in self._audio_engine.stream():
        if audio_event.is_speech:
            await self._ws.send(audio_event.data)
        # Total: 160μs (same!)
```

## Key Success Factors

1. **Keep AudioEngine Thin**: It coordinates, doesn't process
2. **Direct Access When Needed**: Expose internal queues for hot paths
3. **Smart Buffering**: Reuse buffers, don't allocate
4. **Profile Everything**: Measure actual overhead
5. **Mode-Aware**: Fast lane bypasses unnecessary features

## Testing for Real-time Guarantees

```python
def test_audio_engine_latency():
    engine = AudioEngine(mode=AudioMode.FAST_LANE)
    
    # Measure overhead
    start = time.perf_counter_ns()
    for _ in range(10000):
        engine.process_audio(test_chunk)
    overhead_ns = (time.perf_counter_ns() - start) / 10000
    
    assert overhead_ns < 100  # Less than 100ns overhead
```

This architecture is used successfully in:
- Game engines (Unity, Unreal)
- Professional DAWs (Pro Tools, Logic)
- Video conferencing (Zoom, Discord)
- Voice assistants (Alexa, Siri)

The key is implementation quality, not the architecture itself!