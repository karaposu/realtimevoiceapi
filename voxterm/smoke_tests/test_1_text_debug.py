#!/usr/bin/env python3
"""
Debug test for text mode with VoiceEngine

python -m voxterm.smoke_tests.test_1_text_debug
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from realtimevoiceapi import VoiceEngine, VoiceEngineConfig


async def test_text_mode():
    """Test text mode directly"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key")
        return
    
    # Create engine
    config = VoiceEngineConfig(
        api_key=api_key,
        mode="fast",
        voice="alloy",
        log_level="DEBUG"  # More logging
    )
    
    engine = VoiceEngine(config=config)
    
    # Set callbacks
    def on_text(text):
        print(f"AI: {text}", end="", flush=True)
    
    def on_done():
        print("\n[Response complete]")
    
    def on_error(error):
        print(f"\nError: {error}")
    
    engine.on_text_response = on_text
    engine.on_response_done = on_done
    engine.on_error = on_error
    
    # Connect
    print("Connecting...")
    await engine.connect()
    print("Connected!")
    
    # Check available methods
    print("\nEngine methods:")
    for attr in dir(engine):
        if not attr.startswith('_') and callable(getattr(engine, attr)):
            print(f"  - {attr}")
    
    # Try to send text
    print("\nTrying to send text...")
    
    # Check if send_text exists
    if hasattr(engine, 'send_text'):
        print("  ✓ send_text method exists")
        try:
            await engine.send_text("Hello, can you hear me?")
            print("  ✓ Text sent successfully")
            
            # Wait for response
            await asyncio.sleep(5)
        except Exception as e:
            print(f"  ✗ Error sending text: {e}")
    else:
        print("  ✗ No send_text method found")
        
        # Look for alternative methods
        if hasattr(engine, 'send_message'):
            print("  Found send_message method instead")
        elif hasattr(engine, 'send'):
            print("  Found send method instead")
    
    # Disconnect
    print("\nDisconnecting...")
    await engine.disconnect()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(test_text_mode())