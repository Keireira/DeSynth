"""Deterministic Pillow-only sampler for Rust HTTP integration tests."""
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageFilter


def emit(identifier, event, **values):
    print(json.dumps(dict(id=identifier, event=event, **values)), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    identifier = request["id"]
    operation = request["op"]
    if operation == "probe":
        emit(identifier, "result", result={
            "ready": True, "missing": [], "device": "CPU",
            "runtime": {"ready": True, "default_device": "cpu", "devices": [{
                "id": "cpu", "name": "CPU", "type": "cpu",
                "precisions": ["float32"], "offload_modes": ["none"],
            }]},
        })
    elif operation == "sample":
        with Path("sample-requests.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(request) + "\n")
        emit(identifier, "progress", progress=37.5, stage="Sampling test image")
        time.sleep(0.05)
        if Path("fail-sample").is_file():
            emit(identifier, "error", error="Intentional sampler failure", oom=False)
            continue
        if Path("oom-preferred").is_file() and "Q4_K_M" in request["transformer"]:
            emit(identifier, "error", error="Intentional out of memory", oom=True)
            continue
        with Image.open(request["source"]) as source:
            clean = source.convert("RGB").filter(ImageFilter.GaussianBlur(0.7))
            red, green, blue = clean.split()
            red = red.point(lambda value: max(0, value - 4))
            clean = Image.merge("RGB", (red, green, blue))
            clean.save(request["output"])
        emit(identifier, "result", result={"output": request["output"]})
    else:
        emit(identifier, "result", result={"released": True})
