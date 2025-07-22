#!/usr/bin/env python3
"""Debug API key issues"""

import os
from dotenv import load_dotenv

# Load env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"Raw API key length: {len(api_key)}")
    print(f"First 10 chars: {repr(api_key[:10])}")
    print(f"Last 5 chars: {repr(api_key[-5:])}")
    
    # Check for hidden characters
    import string
    printable = set(string.printable)
    
    non_printable = [c for c in api_key if c not in printable or c in '\r\n\t']
    if non_printable:
        print(f"❌ Found non-printable characters: {non_printable}")
    else:
        print("✅ No hidden characters found")
        
    # Try stripping whitespace
    stripped = api_key.strip()
    if len(stripped) != len(api_key):
        print(f"⚠️  API key has whitespace! Original: {len(api_key)}, Stripped: {len(stripped)}")
    
    # Check if it's the actual key or a placeholder
    if "your-key" in api_key or "YOUR_KEY" in api_key or len(api_key) < 40:
        print("❌ This looks like a placeholder, not a real API key!")