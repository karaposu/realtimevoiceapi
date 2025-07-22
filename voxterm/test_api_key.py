#!/usr/bin/env python3
"""Test API key configuration"""

import os
from dotenv import load_dotenv

# Load env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print(f"API Key found: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key length: {len(api_key)}")
    print(f"API Key prefix: {api_key[:7]}...")
    
    # Check format
    if api_key.startswith("sk-proj-"):
        print("✅ Project API key format detected")
    elif api_key.startswith("sk-"):
        print("✅ Standard API key format detected")
    else:
        print("⚠️  Unusual API key format")
    
    # Check for common issues
    if " " in api_key:
        print("❌ API key contains spaces!")
    if "\n" in api_key or "\r" in api_key:
        print("❌ API key contains newlines!")
    if api_key.startswith('"') or api_key.endswith('"'):
        print("❌ API key has quotes!")
    if api_key.startswith("'") or api_key.endswith("'"):
        print("❌ API key has single quotes!")
        
print("\nTo test if run.py works with the same key:")
print("python run.py")