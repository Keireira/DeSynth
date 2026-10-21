"""Return an invalid command ID until the supervisor restarts the process."""
import json
from pathlib import Path
import sys

marker = Path("restarted")
WAS_STARTED = marker.exists()
marker.write_text("yes", encoding="utf-8")
for line in sys.stdin:
    request = json.loads(line)
    identifier = request["id"] if WAS_STARTED else request["id"] + 1
    print(json.dumps({"id": identifier, "event": "result", "result": {"ready": True}}), flush=True)
