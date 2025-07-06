#!/usr/bin/env python3
"""
Simple working example for RealtimeVoiceAPI

This is a minimal example that demonstrates basic functionality
without complex imports or features that might cause issues.

Usage: python simple_example.py
"""

import asyncio
import os
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️ python-dotenv not installed. Using system environment variables only.")
    print("   Install with: pip install python-dotenv")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_text_test():
    """Basic text conversation test"""
    print("🤖 Basic Text Test")
    print("-" * 30)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        # Import here to catch any import issues
        from realtimevoiceapi import RealtimeClient, SessionConfig
        
        print("✅ Imports successful")
        
        # Create client
        client = RealtimeClient(api_key)
        print("✅ Client created")
        
        # Simple session config
        config = SessionConfig(
            instructions="You are a helpful assistant. Keep responses very brief.",
            modalities=["text"],  # Text only for simplicity
            temperature=0.5
        )
        print("✅ Session config created")
        
        # Track response
        full_response = ""
        response_done = False
        
        # Event handlers
        @client.on_event("response.text.delta")
        async def handle_text(event_data):
            nonlocal full_response
            text = event_data.get("delta", "")
            full_response += text
            print(text, end="", flush=True)
        
        @client.on_event("response.done")
        async def handle_done(event_data):
            nonlocal response_done
            response_done = True
            print()  # New line after response
        
        @client.on_event("error") 
        async def handle_error(event_data):
            error = event_data.get("error", {})
            print(f"\n❌ API Error: {error}")
        
        print("✅ Event handlers set")
        
        # Connect
        print("🔌 Connecting to OpenAI Realtime API...")
        await client.connect(config)
        print("✅ Connected!")
        
        # Send test message
        test_message = "Say hello in exactly 3 words."
        print(f"\n👤 User: {test_message}")
        print("🤖 Assistant: ", end="")
        
        await client.send_text(test_message)
        
        # Wait for response
        timeout = 10
        elapsed = 0
        while not response_done and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1
        
        if response_done:
            print(f"\n✅ Success! Response: '{full_response.strip()}'")
            result = True
        else:
            print("\n⏰ Timeout - no response received")
            result = False
        
        # Disconnect
        await client.disconnect()
        print("✅ Disconnected")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in place:")
        print("  - realtimevoiceapi/__init__.py")
        print("  - realtimevoiceapi/client.py")
        print("  - realtimevoiceapi/connection.py")
        print("  - realtimevoiceapi/session.py")
        print("  - realtimevoiceapi/events.py")
        print("  - realtimevoiceapi/audio.py")
        print("  - realtimevoiceapi/types.py")
        print("  - realtimevoiceapi/exceptions.py")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test error")
        return False


async def audio_processing_test():
    """Test audio processing without API"""
    print("\n🎵 Audio Processing Test")
    print("-" * 30)
    
    try:
        from realtimevoiceapi import AudioProcessor
        
        processor = AudioProcessor()
        print("✅ AudioProcessor created")
        
        # Test base64 encoding
        test_data = b"Hello audio world!"
        encoded = processor.bytes_to_base64(test_data)
        decoded = processor.base64_to_bytes(encoded)
        
        if decoded == test_data:
            print("✅ Base64 encoding/decoding works")
        else:
            print("❌ Base64 test failed")
            return False
        
        # Test with NumPy if available
        try:
            import numpy as np
            
            # Generate 1 second of test audio
            sample_rate = 24000
            duration = 1.0
            samples = int(sample_rate * duration)
            audio_array = np.random.randint(-1000, 1000, samples, dtype=np.int16)
            audio_bytes = audio_array.tobytes()
            
            # Test audio info
            info = processor.get_audio_info(audio_bytes)
            print(f"✅ Audio info: {info['duration_ms']:.1f}ms, {info['size_bytes']} bytes")
            
            # Test chunking
            chunks = processor.chunk_audio(audio_bytes, 100)  # 100ms chunks
            print(f"✅ Audio chunking: {len(chunks)} chunks")
            
        except ImportError:
            print("ℹ️ NumPy not available, skipping advanced audio tests")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False


async def package_structure_test():
    """Test package structure"""
    print("\n📦 Package Structure Test")
    print("-" * 30)
    
    try:
        # Test individual imports
        from realtimevoiceapi.exceptions import RealtimeError
        print("✅ Exceptions module")
        
        from realtimevoiceapi.types import Tool
        print("✅ Types module")
        
        from realtimevoiceapi.audio import AudioProcessor
        print("✅ Audio module")
        
        from realtimevoiceapi.events import EventDispatcher
        print("✅ Events module")
        
        from realtimevoiceapi.connection import RealtimeConnection
        print("✅ Connection module")
        
        from realtimevoiceapi.session import SessionConfig
        print("✅ Session module")
        
        from realtimevoiceapi.client import RealtimeClient
        print("✅ Client module")
        
        # Test main package import
        import realtimevoiceapi
        print("✅ Main package import")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 RealtimeVoiceAPI Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Structure", package_structure_test),
        ("Audio Processing", audio_processing_test),
        ("Basic API Test", basic_text_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*10} {test_name} {'='*10}")
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ {test_name} interrupted by user")
            break
            
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is working correctly.")
        print("\n💡 Next steps:")
        print("  1. Try running more complex examples")
        print("  2. Build your own voice application")
        print("  3. Check the documentation for advanced features")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("\n🔧 Common solutions:")
        print("  - Make sure all files are in the realtimevoiceapi/ directory")
        print("  - Check for syntax errors in the Python files")
        print("  - Install missing dependencies: pip install websockets numpy pydub")
        print("  - Verify your OpenAI API key has Realtime API access")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)