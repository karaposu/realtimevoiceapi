#!/usr/bin/env python3
"""Minimal connection test"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env first
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtimevoiceapi import VoiceEngine, VoiceEngineConfig


async def test_connection():
    """Test minimal connection"""
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key: {api_key[:10]}... (length: {len(api_key)})")
    
    # Try exact same config as launcher.py
    config = VoiceEngineConfig(
        api_key=api_key,
        mode="fast",
        voice="alloy",
        latency_mode="ultra_low",
        log_level="INFO"
    )
    
    engine = VoiceEngine(config)
    
    try:
        print("Connecting...")
        await engine.connect()
        print("✅ Connected successfully!")
        await engine.disconnect()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_connection())