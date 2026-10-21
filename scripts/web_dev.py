"""Start the frontend dev server with the selected local API port."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-port", type=int, default=8758)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--pnpm", default="pnpm")
    args = parser.parse_args()
    environment = {**os.environ, "DESYNTH_BACKEND_PORT": str(args.backend_port)}
    executable = shutil.which(args.pnpm)
    if not executable:
        parser.exit(1, "pnpm is required. Install pnpm and run make web-install.\n")
    result = subprocess.run(
        [executable, "dev", "--port", str(args.port)],
        cwd=Path(__file__).resolve().parent.parent,
        env=environment,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
