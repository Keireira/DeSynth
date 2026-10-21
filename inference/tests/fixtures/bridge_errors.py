"""Fail one command without breaking the next protocol exchange."""
import json
import sys

for index, line in enumerate(sys.stdin):
    request = json.loads(line)
    if index == 0:
        print(json.dumps({"id": request["id"], "event": "error", "error": "out of memory", "oom": True}), flush=True)
    else:
        print(json.dumps({"id": request["id"], "event": "result", "result": {"ready": True}}), flush=True)
