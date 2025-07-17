## Core Modules Needed

### 1. **Transport Layer Modules**

**websocket_connection.py**
- Manages WebSocket lifecycle
- Handles reconnection logic
- Message serialization/deserialization
- Shared by both lanes (network overhead dominates)

**message_protocol.py**
- Defines message types and formats
- Protocol validation
- Encoding/decoding utilities
- Thin layer, safe to share

### 2. **Audio Foundation Modules**

**audio_types.py**
- Audio format definitions
- Validation rules
- Conversion tables
- Pure data, no runtime cost

**audio_processor.py**
- Low-level audio operations
- Format conversion
- Validation functions
- Shared utilities, but used differently by each lane

### 3. **Contract/Interface Modules**

**stream_protocol.py**
- Defines IStreamManager interface
- Defines IAudioHandler interface
- Common event types
- Zero runtime cost, pure contracts

**provider_protocol.py**
- Provider capability definitions
- Configuration contracts
- Cost models
- Shared understanding, no implementation

### 4. **Fast Lane Specific Modules**

**direct_audio_capture.py**
- Hardware-level audio access
- Pre-allocated buffers
- Zero-copy operations
- Runs in audio thread

**fast_vad_detector.py**
- Lightweight energy detection
- Minimal state machine
- No allocations in hot path
- Direct callbacks

**fast_stream_manager.py**
- Implements IStreamManager minimally
- Direct WebSocket writes
- No event system
- Hardcoded for specific provider

### 5. **Big Lane Specific Modules**

**audio_pipeline.py**
- Composable audio processors
- Plugin architecture
- Quality enhancement
- Full abstraction

**event_bus.py**
- Pub/sub system
- Async event dispatch
- Event replay
- Debugging support

**stream_orchestrator.py**
- Coordinates multiple streams
- State management
- Error recovery
- Provider abstraction

**response_aggregator.py**
- Assembles streaming responses
- Handles partial data
- Timeout management
- Clean abstractions

### 6. **Shared Service Modules**

**session_manager.py**
- Session state tracking
- Configuration management
- Shared by both, but lightweight

**cost_tracker.py**
- Usage monitoring
- Billing calculations
- Runs outside hot path

**metrics_collector.py**
- Performance metrics
- Quality tracking
- Async reporting

## How Fast Lane is Assembled

**Composition: Direct Pipeline**

1. **Entry Point**: `direct_audio_capture` → `fast_vad_detector`
   - Audio callback triggers VAD directly
   - No queues, no events

2. **Processing**: VAD detection → Immediate action
   - If speech: `fast_stream_manager.send_audio()`
   - Direct memory pass-through

3. **Network**: `fast_stream_manager` → `websocket_connection`
   - Direct method calls
   - Pre-formatted messages
   - No serialization overhead

4. **Response Path**: WebSocket → Audio output
   - Minimal buffering
   - Direct playback initiation

**Key Characteristics**:
- Linear flow, no branching
- ~5-6 function calls total
- No object creation in hot path
- Provider-specific optimization

## How Big Lane is Assembled

**Composition: Flexible Orchestra**

1. **Entry Point**: Multiple options
   - Text input → `stream_orchestrator`
   - Audio file → `audio_pipeline` → `stream_orchestrator`
   - Microphone → `audio_pipeline` → `stream_orchestrator`

2. **Processing**: Full pipeline
   - Input → Event bus notification
   - Audio enhancement plugins
   - Format detection/conversion
   - Quality analysis

3. **Orchestration**: `stream_orchestrator` coordinates
   - Route to appropriate provider
   - Manage multiple streams
   - Handle complex interactions

4. **Response**: Full aggregation
   - `response_aggregator` assembles chunks
   - Event notifications at each stage
   - Flexible output routing

**Key Characteristics**:
- Event-driven architecture
- Unlimited processing stages
- Full debugging/logging
- Provider-agnostic

## The Merge Points

### Shared Foundation
Both lanes share:
- WebSocket connection (different usage patterns)
- Message protocol definitions
- Basic audio types
- Configuration objects

### Divergence Points
Where they split:
- **Audio input**: Direct capture vs Pipeline
- **VAD approach**: Local-only vs Server-coordinated
- **State management**: Minimal vs Full
- **Error handling**: Fail-fast vs Recovery

### Integration Layer
The `conversation_engine.py` acts as the smart router:
- Detects requested mode
- Instantiates appropriate lane
- Provides unified API to applications
- Handles lane switching if needed

## Module Interaction Example

**Fast Lane Flow**:
```
Mic → direct_audio → fast_vad → fast_stream → websocket → Speaker
     (2ms)          (5ms)      (1ms)         (network)   (5ms)
```

**Big Lane Flow**:
```
Mic → pipeline → enhance → orchestrator → provider_adapter → websocket → 
aggregator → event_bus → response_handler → pipeline → Speaker
```

The beauty is that both lanes feel native to their use case - the fast lane feels like systems programming, while the big lane feels like modern async Python.

# Decision:
Unified Engine with Strategy Pattern
Create a single voice_engine.py (not base_voiceapi_engine.py) that uses the Strategy Pattern to switch between fast and big lane implementations:

Why Unified is Better:

Single API Surface: Users don't need to know about lanes
Automatic Optimization: Can auto-select the best lane
Seamless Upgrades: Can switch lanes without changing code
Shared Code: Common functionality isn't duplicated
Better Testing: One interface to test


Implementation Structure:
voice_engine.py                    # Main unified engine
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py          # Common interface
│   ├── fast_lane_strategy.py    # Fast implementation
│   └── big_lane_strategy.py     # Full-featured implementation
├── fast_lane/
│   ├── direct_audio_capture.py
│   ├── fast_vad_detector.py
│   └── fast_stream_manager.py
└── big_lane/
    ├── audio_pipeline.py
    ├── stream_orchestrator.py
    └── response_aggregator.py



engine = VoiceEngine( mode="fast")






Test Suite Structure
smoke_tests/
├── test_01_core_modules.py      # Test basic modules in isolation
├── test_02_audio_modules.py     # Test audio-related modules
├── test_03_messaging.py         # Test message protocol & websocket
├── test_04_fast_lane_units.py   # Test fast lane components
├── test_05_big_lane_units.py    # Test big lane components
├── test_06_integration.py       # Integration tests
├── test_07_voice_engine.py      # Full voice engine tests
└── run_all_tests.py            # Test runner


# we dont want to deal with complex vad logic like adaptive vad for now. so you can dismiss them during implementation