#!/usr/bin/env python3
"""Backwards-compatible entry point.

Prefer ``uv run gin-rummy ...`` or ``python -m gin_rummy ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gin_rummy.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
