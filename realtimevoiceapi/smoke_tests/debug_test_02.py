"""
Debug version of test_02 to find which test is hanging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

print("Starting debug test 02...")

# Try importing
print("Importing AudioEngine...")
from audioengine.audioengine.audio_engine import AudioEngine
print("✓ AudioEngine imported")

print("Importing AudioProcessor...")
from audioengine.audioengine.audio_processor import AudioProcessor
print("✓ AudioProcessor imported")

print("Creating AudioEngine instance...")
engine = AudioEngine()
print("✓ AudioEngine created")

print("Accessing processor...")
processor = engine.processor
print(f"✓ Processor accessed: {processor}")

print("\nDebug test completed successfully!")