#!/usr/bin/env python3
"""FAST v4 Universal Environment Setup Wrapper (delegates to core/setup_env.py)."""
import os
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent / "core"
sys.path.insert(0, str(CORE_DIR))
os.chdir(str(CORE_DIR))

from setup_env import main

if __name__ == "__main__":
    main()
