"""Return an invalid command ID until the supervisor restarts the process."""
import json
from pathlib import Path
import sys

marker = Path("restarted")
was_started = marker.exists()
marker.write_text("yes")
for line in sys.stdin:
    request = json.loads(line)
    identifier = request["id"] if was_started else request["id"] + 1
    print(json.dumps(dict(id=identifier, event="result", result=dict(ready=True))), flush=True)
