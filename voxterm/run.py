#!/usr/bin/env python3
"""
Simple runner script for VoxTerm
Run with: python run.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import and run
from voxterm.launcher import main

if __name__ == "__main__":
    main()