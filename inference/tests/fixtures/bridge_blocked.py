"""Simulate a Torch call that must be terminated during server shutdown."""
from pathlib import Path
import sys
import time

for line in sys.stdin:
    Path("started").write_text("yes")
    time.sleep(60)
