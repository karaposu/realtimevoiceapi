#!/usr/bin/env python3
"""
Test 02: Audio Modules - Test audio processing in isolation

Tests:
- AudioProcessor: Core audio processing
- AudioStreamBuffer: Buffer management
- Audio format conversions
- Audio quality analysis
"""

import sys
import logging
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_audio_processor_basics():
    """Test basic audio processor functionality"""
    print("\n🎵 Testing AudioProcessor Basics...")
    
    try:
        from realtimevoiceapi.audio_processor import AudioProcessor
        from realtimevoiceapi.audio_types import AudioConfig, ProcessingMode
        
        # Create processor
        processor = AudioProcessor(
            config=AudioConfig(),
            mode=ProcessingMode.BALANCED
        )
        assert processor.config.sample_rate == 24000
        print("  ✅ AudioProcessor creation works")
        
        # Test encoding/decoding
        test_data = b"Hello Audio World!" * 100
        encoded = processor.bytes_to_base64(test_data)
        decoded = processor.base64_to_bytes(encoded)
        assert decoded == test_data
        print("  ✅ Base64 encoding/decoding works")
        
        # Test duration calculation
        # Create 1 second of audio (24000 samples * 2 bytes per sample)
        audio_bytes = b'\x00\x00' * 24000
        duration = processor.calculate_duration(audio_bytes)
        assert abs(duration - 1000.0) < 1  # ~1000ms
        print("  ✅ Duration calculation works")
        
        # Test bytes needed calculation
        bytes_needed = processor.calculate_bytes_needed(500)  # 500ms
        assert bytes_needed == 24000  # 500ms at 24kHz mono 16-bit
        print("  ✅ Bytes calculation works")
        
        # Test validation
        valid, error = processor.validate_format(audio_bytes)
        assert valid == True
        assert error is None
        print("  ✅ Format validation works")
        
        # Test invalid format
        invalid_audio = b'\x00'  # Odd number of bytes for PCM16
        valid, error = processor.validate_format(invalid_audio)
        assert valid == False
        assert "even number of bytes" in error
        print("  ✅ Invalid format detection works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AudioProcessor basics failed: {e}")
        logger.exception("AudioProcessor error")
        return False


def test_audio_chunking():
    """Test audio chunking functionality"""
    print("\n🔪 Testing Audio Chunking...")
    
    try:
        from realtimevoiceapi.audio_processor import AudioProcessor
        from realtimevoiceapi.audio_types import AudioConfig
        
        processor = AudioProcessor()
        
        # Create 1 second of audio
        audio_bytes = b'\x00\x00' * 24000
        
        # Test default chunking (100ms chunks)
        chunks = processor.chunk_audio(audio_bytes)
        assert len(chunks) == 10  # 1000ms / 100ms = 10
        assert all(len(chunk) == 4800 for chunk in chunks)  # 100ms each
        print("  ✅ Default chunking works")
        
        # Test custom chunk size
        chunks = processor.chunk_audio(audio_bytes, chunk_duration_ms=250)
        assert len(chunks) == 4  # 1000ms / 250ms = 4
        assert all(len(chunk) == 12000 for chunk in chunks)  # 250ms each
        print("  ✅ Custom chunk size works")
        
        # Test edge cases
        small_audio = b'\x00\x00' * 100  # Very small audio
        chunks = processor.chunk_audio(small_audio, chunk_duration_ms=100)
        assert len(chunks) == 1
        assert len(chunks[0]) == 200
        print("  ✅ Small audio chunking works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio chunking failed: {e}")
        logger.exception("Chunking error")
        return False


def test_audio_quality_analysis():
    """Test audio quality analysis"""
    print("\n📊 Testing Audio Quality Analysis...")
    
    try:
        import numpy as np
        numpy_available = True
    except ImportError:
        numpy_available = False
        print("  ⚠️ NumPy not available, skipping quality tests")
        return True
    
    try:
        from realtimevoiceapi.audio_processor import AudioProcessor
        from realtimevoiceapi.audio_types import AudioMetadata, AudioFormat
        
        processor = AudioProcessor()
        
        # Generate test audio with known properties
        sample_rate = 24000
        duration = 0.5  # 500ms
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate sine wave with 0.5 amplitude
        signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        audio_data = (signal * 32767).astype(np.int16)
        audio_bytes = audio_data.tobytes()
        
        # Analyze quality
        metadata = processor.analyze_quality(audio_bytes)
        
        assert isinstance(metadata, AudioMetadata)
        assert metadata.format == AudioFormat.PCM16
        assert abs(metadata.duration_ms - 500.0) < 1
        assert metadata.size_bytes == len(audio_bytes)
        print("  ✅ Basic metadata correct")
        
        # Check amplitude measurements
        assert metadata.peak_amplitude is not None
        assert 0.4 < metadata.peak_amplitude < 0.6  # ~0.5
        assert metadata.rms_amplitude is not None
        assert metadata.rms_amplitude < metadata.peak_amplitude
        print("  ✅ Amplitude analysis works")
        
        # Check speech detection
        assert metadata.is_speech is not None
        print(f"  ✅ Speech detection works (detected: {metadata.is_speech})")
        
        # Test silence detection
        silence = np.zeros(int(sample_rate * 0.1), dtype=np.int16)
        silence_bytes = silence.tobytes()
        silence_metadata = processor.analyze_quality(silence_bytes)
        
        assert silence_metadata.peak_amplitude < 0.01
        assert silence_metadata.is_speech == False
        print("  ✅ Silence detection works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quality analysis failed: {e}")
        logger.exception("Quality analysis error")
        return False


def test_audio_stream_buffer():
    """Test audio stream buffer"""
    print("\n💾 Testing Audio Stream Buffer...")
    
    try:
        from realtimevoiceapi.audio_processor import AudioStreamBuffer
        from realtimevoiceapi.audio_types import BufferConfig, AudioConfig
        
        # Test dynamic buffer (big lane)
        buffer_config = BufferConfig(
            max_size_bytes=10000,
            overflow_strategy="drop_oldest",
            use_circular=False
        )
        
        buffer = AudioStreamBuffer(config=buffer_config)
        
        # Add audio
        chunk1 = b'\x00\x01' * 1000  # 2000 bytes
        chunk2 = b'\x02\x03' * 1000  # 2000 bytes
        
        success = buffer.add_audio(chunk1)
        assert success == True
        assert buffer.get_available_bytes() == 2000
        print("  ✅ Dynamic buffer add works")
        
        success = buffer.add_audio(chunk2)
        assert success == True
        assert buffer.get_available_bytes() == 4000
        print("  ✅ Multiple adds work")
        
        # Get chunk
        retrieved = buffer.get_chunk(2000)
        assert retrieved == chunk1
        assert buffer.get_available_bytes() == 2000
        print("  ✅ Chunk retrieval works")
        
        # Test overflow
        large_chunk = b'\xFF' * 8000
        success = buffer.add_audio(large_chunk)
        assert success == True
        # Should have dropped oldest data
        assert buffer.get_available_bytes() <= buffer_config.max_size_bytes
        print("  ✅ Overflow handling works")
        
        # Test circular buffer (fast lane)
        circular_config = BufferConfig(
            max_size_bytes=10000,
            use_circular=True,
            pre_allocate=True
        )
        
        circular_buffer = AudioStreamBuffer(config=circular_config)
        
        # Add and retrieve
        success = circular_buffer.add_audio(chunk1)
        assert success == True
        retrieved = circular_buffer.get_chunk(2000)
        assert retrieved == chunk1
        print("  ✅ Circular buffer works")
        
        # Test stats
        stats = circular_buffer.get_stats()
        assert stats["total_added"] == 2000
        assert stats["total_consumed"] == 2000
        assert stats["available_bytes"] == 0
        print("  ✅ Buffer statistics work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stream buffer test failed: {e}")
        logger.exception("Stream buffer error")
        return False


def test_audio_io():
    """Test audio file I/O"""
    print("\n📁 Testing Audio File I/O...")
    
    try:
        from realtimevoiceapi.audio_processor import AudioProcessor
        import tempfile
        
        processor = AudioProcessor()
        
        # Generate test audio
        try:
            import numpy as np
            # Create 1 second of 440Hz sine wave
            sample_rate = 24000
            t = np.linspace(0, 1, sample_rate)
            signal = 0.3 * np.sin(2 * np.pi * 440 * t)
            audio_data = (signal * 32767).astype(np.int16)
            test_audio = audio_data.tobytes()
        except ImportError:
            # Fallback to simple test audio
            test_audio = b'\x00\x00\xFF\x7F' * 6000  # Simple pattern
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        try:
            # Save
            processor.save_wav_file(test_audio, temp_path)
            assert temp_path.exists()
            assert temp_path.stat().st_size > len(test_audio)  # WAV has headers
            print("  ✅ WAV file saving works")
            
            # Load
            loaded_audio = processor.load_wav_file(temp_path)
            assert loaded_audio == test_audio
            print("  ✅ WAV file loading works")
            
            # Test metadata preservation
            duration_original = processor.calculate_duration(test_audio)
            duration_loaded = processor.calculate_duration(loaded_audio)
            assert abs(duration_original - duration_loaded) < 1
            print("  ✅ Audio properties preserved")
            
        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio I/O test failed: {e}")
        logger.exception("Audio I/O error")
        return False


def test_audio_format_conversion():
    """Test audio format conversions"""
    print("\n🔄 Testing Audio Format Conversion...")
    
    try:
        import numpy as np
        numpy_available = True
    except ImportError:
        print("  ⚠️ NumPy not available, skipping conversion tests")
        return True
    
    try:
        from realtimevoiceapi.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        
        # Test mono conversion
        # Create stereo audio (interleaved L/R channels)
        left_channel = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 2400))
        right_channel = np.sin(2 * np.pi * 880 * np.linspace(0, 0.1, 2400))
        stereo = np.zeros(4800, dtype=np.int16)
        stereo[0::2] = (left_channel * 16000).astype(np.int16)
        stereo[1::2] = (right_channel * 16000).astype(np.int16)
        stereo_bytes = stereo.tobytes()
        
        # Convert to mono
        mono_bytes = processor.ensure_mono(stereo_bytes, channels=2)
        assert len(mono_bytes) == len(stereo_bytes) // 2
        print("  ✅ Stereo to mono conversion works")
        
        # Test resampling
        # Create audio at 16kHz
        samples_16k = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 16000).astype(np.int16)
        audio_16k = samples_16k.tobytes()
        
        # Resample to 24kHz
        audio_24k = processor.resample(audio_16k, from_rate=16000, to_rate=24000)
        assert len(audio_24k) == int(len(audio_16k) * 1.5)  # 24/16 = 1.5
        print("  ✅ Resampling works")
        
        # Test no-op resampling
        same = processor.resample(audio_24k, from_rate=24000, to_rate=24000)
        assert same == audio_24k
        print("  ✅ No-op resampling optimization works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Format conversion failed: {e}")
        logger.exception("Format conversion error")
        return False


def main():
    """Run all audio module tests"""
    print("🧪 RealtimeVoiceAPI - Test 02: Audio Modules")
    print("=" * 60)
    print("Testing audio processing modules in isolation")
    print()
    
    tests = [
        ("AudioProcessor Basics", test_audio_processor_basics),
        ("Audio Chunking", test_audio_chunking),
        ("Audio Quality Analysis", test_audio_quality_analysis),
        ("Audio Stream Buffer", test_audio_stream_buffer),
        ("Audio File I/O", test_audio_io),
        ("Audio Format Conversion", test_audio_format_conversion),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All audio modules working correctly!")
        print("Next: Run test_03_messaging.py")
    else:
        print(f"\n❌ {total - passed} audio module(s) need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)