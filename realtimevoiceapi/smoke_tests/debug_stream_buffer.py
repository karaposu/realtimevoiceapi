"""
Debug Test for Stream Buffer
Isolated test to understand why stream buffer is hanging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
import numpy as np
from audioengine.audioengine.audio_processor import AudioStreamBuffer
from audioengine.audioengine.audio_types import AudioBytes, AudioConfig, BufferConfig

def generate_test_audio(duration_ms: int = 100, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio (sine wave)"""
    print(f"[DEBUG] Generating {duration_ms}ms of audio at {sample_rate}Hz")
    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767 * 0.5).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    print(f"[DEBUG] Generated {len(audio_bytes)} bytes of audio")
    return audio_bytes

async def test_stream_buffer_detailed():
    """Test stream buffer with detailed debugging"""
    print("\n=== DETAILED STREAM BUFFER TEST ===")
    
    try:
        # Create audio config
        print("[DEBUG] Creating AudioConfig...")
        audio_config = AudioConfig()
        audio_config.sample_rate = 24000
        audio_config.chunk_duration_ms = 20
        audio_config.channels = 1
        print(f"[DEBUG] AudioConfig: sample_rate={audio_config.sample_rate}, chunk_duration_ms={audio_config.chunk_duration_ms}")
        
        # Create buffer config
        buffer_config = BufferConfig()
        print(f"[DEBUG] BufferConfig: max_size_bytes={buffer_config.max_size_bytes}")
        
        # Create stream buffer
        print("[DEBUG] Creating AudioStreamBuffer...")
        buffer = AudioStreamBuffer(config=buffer_config, audio_config=audio_config)
        print(f"[DEBUG] Buffer created: {buffer}")
        
        # Generate test audio
        print("\n[DEBUG] Generating test audio...")
        test_audio = generate_test_audio(200)  # 200ms
        print(f"[DEBUG] Test audio generated: {len(test_audio)} bytes")
        
        # Add to buffer
        print("\n[DEBUG] Adding audio to buffer...")
        buffer.add_audio(test_audio)
        print("[DEBUG] Audio added successfully")
        
        # Get buffer info
        print(f"[DEBUG] Buffer has data: {buffer.has_data()}")
        print(f"[DEBUG] Buffer size: {buffer.get_buffer_size()} bytes")
        
        # Try to get a chunk
        print("\n[DEBUG] Attempting to get chunk...")
        chunk_size = int(audio_config.sample_rate * audio_config.chunk_duration_ms / 1000 * audio_config.channels * 2)
        print(f"[DEBUG] Expected chunk size: {chunk_size} bytes")
        
        chunk_count = 0
        max_chunks = 20  # Safety limit
        
        while buffer.has_data() and chunk_count < max_chunks:
            print(f"\n[DEBUG] Getting chunk {chunk_count + 1}...")
            print(f"[DEBUG] Buffer size before: {buffer.get_buffer_size()}")
            
            chunk = buffer.get_chunk()
            
            if chunk:
                print(f"[DEBUG] Got chunk: {len(chunk)} bytes")
                chunk_count += 1
            else:
                print("[DEBUG] No chunk available (buffer empty or insufficient data)")
                break
                
            print(f"[DEBUG] Buffer size after: {buffer.get_buffer_size()}")
            
        print(f"\n[DEBUG] Total chunks retrieved: {chunk_count}")
        
        # Test async iteration
        print("\n[DEBUG] Testing async iteration...")
        buffer.add_audio(generate_test_audio(100))  # Add more audio
        
        print("[DEBUG] Starting async iteration with timeout...")
        chunks_from_iter = []
        
        async def iterate_with_timeout():
            try:
                async with asyncio.timeout(2):  # 2 second timeout
                    print("[DEBUG] Entering async for loop...")
                    async for chunk in buffer.stream_chunks():
                        print(f"[DEBUG] Got chunk from iterator: {len(chunk)} bytes")
                        chunks_from_iter.append(chunk)
                        if len(chunks_from_iter) >= 3:  # Stop after 3 chunks
                            print("[DEBUG] Got enough chunks, breaking")
                            break
            except asyncio.TimeoutError:
                print("[DEBUG] Iteration timed out (expected)")
        
        await iterate_with_timeout()
        print(f"[DEBUG] Got {len(chunks_from_iter)} chunks from async iteration")
        
        # Test mark complete
        print("\n[DEBUG] Testing mark_complete...")
        buffer.mark_complete()
        print("[DEBUG] Buffer marked complete")
        
        # Try iteration after complete
        print("[DEBUG] Testing iteration after marking complete...")
        final_chunks = []
        async for chunk in buffer.stream_chunks():
            print(f"[DEBUG] Got final chunk: {len(chunk)} bytes")
            final_chunks.append(chunk)
        
        print(f"[DEBUG] Got {len(final_chunks)} final chunks")
        
        print("\n✓ Stream buffer test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Stream buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_problematic_scenario():
    """Test the specific scenario that might be causing the hang"""
    print("\n=== TESTING PROBLEMATIC SCENARIO ===")
    
    try:
        audio_config = AudioConfig()
        audio_config.sample_rate = 24000
        audio_config.chunk_duration_ms = 20
        buffer_config = BufferConfig()
        buffer = AudioStreamBuffer(config=buffer_config, audio_config=audio_config)
        
        # Add exactly enough for 2 chunks
        chunk_size = int(audio_config.sample_rate * audio_config.chunk_duration_ms / 1000 * audio_config.channels * 2)
        print(f"[DEBUG] Chunk size: {chunk_size}")
        
        # Create audio that's exactly 2 chunks
        audio_data = b'\x00' * (chunk_size * 2)
        print(f"[DEBUG] Adding {len(audio_data)} bytes (exactly 2 chunks)")
        
        buffer.add_audio(audio_data)
        
        # Get chunks one by one
        print("\n[DEBUG] Getting chunks manually...")
        chunk1 = buffer.get_chunk()
        print(f"[DEBUG] Chunk 1: {len(chunk1) if chunk1 else 'None'} bytes")
        
        chunk2 = buffer.get_chunk()
        print(f"[DEBUG] Chunk 2: {len(chunk2) if chunk2 else 'None'} bytes")
        
        chunk3 = buffer.get_chunk()
        print(f"[DEBUG] Chunk 3: {len(chunk3) if chunk3 else 'None'} bytes (should be None)")
        
        # Now test async iteration on empty buffer
        print("\n[DEBUG] Testing async iteration on empty buffer...")
        print("[DEBUG] This might hang if not handled properly")
        
        iteration_count = 0
        async def test_empty_iteration():
            nonlocal iteration_count
            try:
                async with asyncio.timeout(1):
                    async for chunk in buffer.stream_chunks():
                        iteration_count += 1
                        print(f"[DEBUG] Unexpected chunk in empty buffer: {len(chunk)}")
            except asyncio.TimeoutError:
                print("[DEBUG] Empty buffer iteration timed out (this might be the issue)")
        
        await test_empty_iteration()
        print(f"[DEBUG] Iteration count: {iteration_count}")
        
        print("\n✓ Problematic scenario test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Problematic scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run debug tests"""
    print("=" * 60)
    print("Stream Buffer Debug Tests")
    print("=" * 60)
    
    # Run detailed test
    result1 = await test_stream_buffer_detailed()
    
    # Run problematic scenario test
    result2 = await test_problematic_scenario()
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    print(f"Detailed test: {'PASS' if result1 else 'FAIL'}")
    print(f"Problematic scenario: {'PASS' if result2 else 'FAIL'}")
    
    return result1 and result2

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)