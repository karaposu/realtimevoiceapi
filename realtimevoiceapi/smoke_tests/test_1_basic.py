
#!/usr/bin/env python3
"""
Test 1: Basic Package and Audio Processing (Fixed)

This test verifies:
- Package imports work correctly
- Audio processing functions work
- No API connection required

Run: python -m realtimevoiceapi.smoke_tests.test_1_basic
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path so we can import realtimevoiceapi
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required imports work"""
    print("🔗 Testing Package Imports...")
    
    try:
        # Core imports
        from realtimevoiceapi import RealtimeClient, SessionConfig, AudioProcessor
        print("  ✅ Core classes imported")
        
        # Type imports (updated to use models.py)
        from realtimevoiceapi.models import Tool, TurnDetectionConfig, TranscriptionConfig
        print("  ✅ Type classes imported")
        
        # Exception imports
        from realtimevoiceapi.exceptions import RealtimeError, AudioError
        print("  ✅ Exception classes imported")
        
        # Test creating instances
        processor = AudioProcessor()
        client = RealtimeClient("test-key")
        config = SessionConfig()
        
        print("  ✅ All objects can be instantiated")
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_audio_processing():
    """Test audio processing functionality"""
    print("\n🎵 Testing Audio Processing...")
    
    try:
        from realtimevoiceapi import AudioProcessor
        
        processor = AudioProcessor()
        
        # Test 1: Base64 encoding/decoding
        test_data = b"RealtimeVoiceAPI test data!"
        encoded = processor.bytes_to_base64(test_data)
        decoded = processor.base64_to_bytes(encoded)
        
        if decoded == test_data:
            print("  ✅ Base64 encoding/decoding works")
        else:
            print("  ❌ Base64 test failed")
            return False
        
        # Test 2: Audio info calculation (with synthetic data)
        try:
            import numpy as np
            
            # Generate 1 second of 440Hz sine wave
            sample_rate = 24000
            duration = 1.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_array = np.sin(2 * np.pi * frequency * t) * 0.5
            audio_array = (audio_array * 32767).astype(np.int16)
            audio_bytes = audio_array.tobytes()
            
            # Test audio info
            info = processor.get_audio_info(audio_bytes)
            expected_duration = 1000.0  # 1 second in ms
            
            if abs(info['duration_ms'] - expected_duration) < 10:
                print(f"  ✅ Audio duration calculation: {info['duration_ms']:.1f}ms")
            else:
                print(f"  ❌ Audio duration wrong: {info['duration_ms']} vs {expected_duration}")
                return False
            
            # Test 3: Audio chunking
            chunks = processor.chunk_audio(audio_bytes, 250)  # 250ms chunks
            expected_chunks = 4  # 1000ms / 250ms = 4 chunks
            
            if len(chunks) == expected_chunks:
                print(f"  ✅ Audio chunking: {len(chunks)} chunks created")
            else:
                print(f"  ❌ Audio chunking wrong: {len(chunks)} vs {expected_chunks}")
                return False
            
            # Test 4: WAV file save/load
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                processor.save_wav_file(audio_bytes, temp_path)
                loaded_audio = processor.load_wav_file(temp_path)
                
                if loaded_audio == audio_bytes:
                    print("  ✅ WAV file save/load works")
                else:
                    print("  ❌ WAV file save/load failed")
                    return False
                    
            finally:
                if Path(temp_path).exists():
                    os.unlink(temp_path)
            
            # Test 5: Audio quality analysis
            analysis = processor.analyze_audio_quality(audio_bytes)
            
            if analysis['quality_score'] > 0.8:
                print(f"  ✅ Audio quality analysis: score {analysis['quality_score']:.2f}")
            else:
                print(f"  ❌ Audio quality analysis failed: {analysis}")
                return False
                
        except ImportError:
            print("  ℹ️ NumPy not available, skipping advanced audio tests")
            print("  ✅ Basic audio processing works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio processing failed: {e}")
        logger.exception("Audio processing error")
        return False


def test_session_config():
    """Test session configuration"""
    print("\n⚙️ Testing Session Configuration...")
    
    try:
        from realtimevoiceapi import SessionConfig
        from realtimevoiceapi.models import Tool, TurnDetectionConfig  # Updated import
        
        # Test 1: Basic config
        config = SessionConfig(
            instructions="Test instructions",
            modalities=["text"],
            temperature=0.5
        )
        
        config_dict = config.to_dict()
        if "instructions" in config_dict and config_dict["instructions"] == "Test instructions":
            print("  ✅ Basic config creation works")
        else:
            print("  ❌ Basic config failed")
            return False
        
        # Test 2: Config with tools
        tool = Tool(
            name="test_function",
            description="A test function",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }
        )
        
        config_with_tools = SessionConfig(
            tools=[tool],
            tool_choice="auto"
        )
        
        tools_dict = config_with_tools.to_dict()
        if "tools" in tools_dict and len(tools_dict["tools"]) == 1:
            print("  ✅ Config with tools works")
        else:
            print("  ❌ Config with tools failed")
            return False
        
        # Test 3: Turn detection config
        turn_detection = TurnDetectionConfig(
            type="server_vad",
            threshold=0.5
        )
        
        config_with_vad = SessionConfig(turn_detection=turn_detection)
        vad_dict = config_with_vad.to_dict()
        
        if "turn_detection" in vad_dict:
            print("  ✅ Turn detection config works")
        else:
            print("  ❌ Turn detection config failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Session config failed: {e}")
        logger.exception("Session config error")
        return False


def main():
    """Run all basic tests"""
    print("🧪 RealtimeVoiceAPI - Test 1: Basic Package and Audio")
    print("=" * 60)
    print("This test checks imports and audio processing (no API required)")
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Audio Processing", test_audio_processing),
        ("Session Configuration", test_session_config)
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
    print("📊 Test 1 Results")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Test 1 PASSED! Basic functionality works correctly.")
        print("Next: Run 'python -m realtimevoiceapi.smoke_tests.test_2_text_api' to test API connection")
    else:
        print(f"\n❌ Test 1 FAILED! {total - passed} test(s) need attention.")
        print("\n🔧 Common issues:")
        print("  - Missing dependencies: pip install numpy pydub websockets")
        print("  - Import path issues: make sure you're in the project root")
        print("  - File permissions for temporary files")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)