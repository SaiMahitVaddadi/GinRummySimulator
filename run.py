#!/usr/bin/env python3
"""
Main entry point for Gin Rummy Simulator
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from main.simulator import main

if __name__ == "__main__":
    main()
