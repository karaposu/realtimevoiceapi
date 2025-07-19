"""
VoxTerm Launcher - Main entry point with menu flow
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from .menu import VoxTermMenu
from .settings import TerminalSettings
from typing import Optional


async def launch_voxterm(api_key: Optional[str] = None):
    """Launch VoxTerm with menu interface"""
    
    # Get API key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No API key found!")
        print("\n💡 Set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   or create a .env file")
        return
    
    # Import VoiceEngine
    try:
        from realtimevoiceapi import VoiceEngine, VoiceEngineConfig
    except ImportError:
        print("❌ realtimevoiceapi not found")
        print("   pip install realtimevoiceapi")
        return
    
    # Create engine with basic config
    config = VoiceEngineConfig(
        api_key=api_key,
        mode="fast",
        voice="alloy",
        latency_mode="ultra_low",
        log_level="WARNING"  # Quiet by default
    )
    
    engine = VoiceEngine(config=config)
    
    # Create settings
    settings = TerminalSettings()
    
    # Create and run menu
    menu = VoxTermMenu(engine, settings)
    
    try:
        await menu.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        if menu.connected:
            try:
                await engine.disconnect()
            except:
                pass


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Check for audio support
    try:
        import sounddevice as sd
        # Silent check, no output
    except ImportError:
        print("⚠️  Audio support not found")
        print("   pip install sounddevice")
        print("   (Text mode will still work)")
        print()
    
    # Run the launcher
    asyncio.run(launch_voxterm())


if __name__ == "__main__":
    main()