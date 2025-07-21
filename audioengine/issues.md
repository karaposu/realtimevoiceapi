# AudioEngine Known Issues

## 1. Stream Buffer Hang Issue

### Summary
The `add_to_stream_buffer()` method in AudioEngine causes tests to hang indefinitely when used with AudioStreamBuffer.

### Affected Components
- `AudioEngine.add_to_stream_buffer()` method
- `AudioStreamBuffer` class
- Tests: test_01_audio_engine_basics.py (Test 6), test_02_audio_engine_processing.py (Test 6)

### Problem Description

When calling `engine.add_to_stream_buffer()` in a loop, the program hangs. This occurs in the following test pattern:

```python
for i in range(10):
    test_audio = generate_test_audio(20)  # 20ms chunks
    result = engine.add_to_stream_buffer(test_audio)  # <-- Hangs here
```

### Root Cause Analysis

#### 1. Buffer Size Mismatch
- Test sends 20ms chunks (960 bytes at 24kHz)
- AudioEngine waits for 200ms chunks (9,600 bytes)
- After 10 iterations, only 9,600 bytes are added (exactly one 200ms chunk)

The method implementation:
```python
def add_to_stream_buffer(self, audio_bytes: AudioBytes) -> Optional[AudioBytes]:
    # ...
    self._stream_buffer.add_audio(audio_bytes)  # Adds audio to buffer
    
    chunk_size = int(self.config.sample_rate * 200 / 1000 * self.config.channels * 2)
    
    if self._stream_buffer.get_available_bytes() >= chunk_size:
        chunk = self._stream_buffer.get_chunk(chunk_size)
        if chunk:
            return self.process_audio(chunk)
```

#### 2. API Mismatch
The AudioEngine expects certain methods from AudioStreamBuffer that either don't exist or work differently:
- `has_data()` method doesn't exist (causes AttributeError)
- `get_chunk()` requires a `chunk_size` parameter
- Different internal method signatures than expected

#### 3. Potential Async/Sync Conflict
AudioStreamBuffer might be designed for async operations:
```python
async def stream_chunks(self):
    while not self.is_complete:
        if self.has_data():
            yield self.get_chunk()
        else:
            await asyncio.sleep(0.01)  # <-- Could hang here forever
```

But AudioEngine uses it synchronously, causing a pattern mismatch.

#### 4. Missing Complete Signal
The buffer might be waiting for a "stream complete" signal that never comes in the test scenario.

### Evidence from Debug Tests

1. **Method Not Found Errors**:
   ```
   AttributeError: 'AudioStreamBuffer' object has no attribute 'has_data'
   ```

2. **Parameter Mismatch**:
   ```
   TypeError: AudioStreamBuffer.get_chunk() missing 1 required positional argument: 'chunk_size'
   ```

3. **Configuration Issues**:
   ```
   AttributeError: 'AudioConfig' object has no attribute 'chunk_size_bytes'
   ```

### Workaround
Currently, tests skip the stream buffer functionality:
```python
# Skip this test for now - it has issues
print("⚠ Skipping stream buffer test due to implementation issues")
return True
```

### Impact
- Stream buffering functionality is non-functional
- Quality mode processing that relies on buffering may not work as expected
- Tests must skip stream buffer related functionality

### Proposed Solutions

1. **Fix API Alignment**: Ensure AudioEngine and AudioStreamBuffer have matching expectations
2. **Add Missing Methods**: Implement `has_data()` and other expected methods in AudioStreamBuffer
3. **Clarify Sync/Async Pattern**: Decide if stream buffer should be sync or async and align usage
4. **Add Proper Complete Handling**: Ensure buffer can be properly completed without hanging
5. **Fix Configuration**: Ensure AudioConfig has the expected methods or change how chunk size is calculated

### Related Code Locations
- `/audioengine/audioengine/audio_engine.py`: Line 374-395 (add_to_stream_buffer method)
- `/audioengine/audioengine/audio_processor.py`: AudioStreamBuffer class
- `/audioengine/audioengine/audio_types.py`: AudioConfig class

### Status
- **Priority**: Medium (core functionality works without stream buffering)
- **Workaround**: Available (skip stream buffer tests)
- **Fix Required**: Yes, for full quality mode support