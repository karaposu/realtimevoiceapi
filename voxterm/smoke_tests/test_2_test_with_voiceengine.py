#!/usr/bin/env python3
"""
VoxTerm Smoke Test 1: Test with Real VoiceEngine

Interactive test with actual VoiceEngine connection.
Requires OPENAI_API_KEY to be set.

python -m voxterm.smoke_tests.test_2_test_with_voiceengine
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Import VoxTerm and VoiceEngine
from voxterm import VoxTermCLI
from realtimevoiceapi import VoiceEngine, VoiceEngineConfig


class TextModeWrapper:
    """Wrapper to handle text mode properly with VoiceEngine"""
    
    def __init__(self, engine):
        self.engine = engine
        self._transcript_buffer = ""
        self._original_callbacks = {}
        
    def setup_text_mode(self):
        """Configure engine for text-only interaction"""
        # Save original callbacks
        self._original_callbacks = {
            'on_audio': getattr(self.engine, 'on_audio_response', None),
            'on_text': getattr(self.engine, 'on_text_response', None),
        }
        
        # Override audio callback to capture transcript instead
        def on_audio(audio):
            # In text mode, we don't play audio
            pass
        
        # Set up to capture transcripts
        self.engine.on_audio_response = on_audio
        
    def restore_callbacks(self):
        """Restore original callbacks"""
        for name, callback in self._original_callbacks.items():
            if callback:
                setattr(self.engine, name.replace('_', ''), callback)


async def run_interactive():
    """Run full interactive test"""
    print("\n🎙️ VoxTerm Interactive Test with VoiceEngine")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set!")
        print("\n💡 To run this test:")
        print("   export OPENAI_API_KEY='your-api-key'")
        print("   # or create a .env file")
        return
    
    # Check audio devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"\n✅ Audio ready ({len(devices)} devices found)")
    except Exception as e:
        print(f"\n⚠️  Audio check failed: {e}")
        
    # Choose mode
    print("\n📋 Available modes:")
    print("1. push_to_talk - Hold space to record")
    print("2. always_on - Continuous listening")
    print("3. text - Type messages")
    print("4. turn_based - Take turns speaking")
    
    mode_choice = input("\nSelect mode (1-4) or name [default: 1]: ").strip()
    
    mode_map = {
        "1": "push_to_talk",
        "2": "always_on", 
        "3": "text",
        "4": "turn_based",
    }
    
    mode = mode_map.get(mode_choice, mode_choice or "push_to_talk")
    
    # Create VoiceEngine with appropriate config
    if mode in ["text", "type"]:
        # For text mode, we might want different settings
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            latency_mode="standard",  # Don't need ultra-low for text
            vad_enabled=False,
            log_level="WARNING"  # Less verbose
        )
    else:
        # Voice modes
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            latency_mode="ultra_low",
            vad_enabled=(mode == "always_on"),
            chunk_duration_ms=100,
            log_level="INFO"
        )
    
    engine = VoiceEngine(config=config)
    
    # For text mode, set up special handling
    text_wrapper = None
    if mode in ["text", "type"]:
        # For text mode, we just rely on VoxTerm's callback handling
        # No need for special wrappers that might cause duplicates
        pass
    
    # Create and run VoxTermCLI
    cli = VoxTermCLI(engine, mode=mode)
    
    try:
        await cli.run()
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_mode(mode_name: str, duration: int = 30):
    """Test a specific mode (for automated testing)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
        
    try:
        # Create VoiceEngine
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="fast",
            voice="alloy",
            latency_mode="ultra_low",
            vad_enabled=(mode_name == "always_on"),
            log_level="WARNING"  # Less verbose for automated tests
        )
        
        engine = VoiceEngine(config=config)
        cli = VoxTermCLI(engine, mode=mode_name)
        
        # Run with timeout
        print(f"\n⏱️  Running {mode_name} mode for {duration} seconds...")
        await asyncio.wait_for(cli.run(), timeout=duration)
        
        return True
        
    except asyncio.TimeoutError:
        print(f"\n⏰ Test completed after {duration} seconds")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    """Main entry point"""
    # Check if running automated test or interactive
    if len(sys.argv) > 1:
        # Automated test with specific mode
        mode = sys.argv[1]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        print(f"\n🤖 Running automated test: {mode} mode for {duration}s")
        success = asyncio.run(test_mode(mode, duration))
        sys.exit(0 if success else 1)
    else:
        # Interactive test
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()