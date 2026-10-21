"""Emit an overlong frame without a newline, then behave after restart."""
import json
from pathlib import Path
import sys
import time

marker = Path("oversized-started")
if not marker.exists():
    marker.write_text("yes")
    sys.stdout.write("x" * 65537)
    sys.stdout.flush()
    time.sleep(60)
else:
    for line in sys.stdin:
        request = json.loads(line)
        print(json.dumps(dict(id=request["id"], event="result", result=dict(ready=True))), flush=True)
