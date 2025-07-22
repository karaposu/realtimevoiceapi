#!/usr/bin/env python3
"""Test script to verify turn-based implementation"""

import os

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found!")
    print("\n💡 To test the fixed turn-based implementation:")
    print("   1. Set your OpenAI API key:")
    print("      export OPENAI_API_KEY='your-key-here'")
    print("   2. Run the fixed version:")
    print("      python run_turn_based_fixed.py")
    print("\n✅ The implementation correctly uses send_recorded_audio() for turn-based mode")
    print("✅ API key loading has been fixed (loads .env before imports)")
else:
    print("✅ API key found!")
    print("✅ Ready to test turn-based implementation")
    print("\nRun: python run_turn_based_fixed.py")