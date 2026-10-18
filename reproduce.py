"""128-dimensional detection features, 8-dimensional cues, five location hypotheses, 40 epochs seed0; detection-conditioned identity/existence evaluation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["objectledger.py", args.data, "ledger.example.json", "--variant", "full", "--epochs", "40", "--slots", "5", "--seed", "0", "--output", str(output / "full")],
        ["objectledger.py", args.data, "ledger.example.json", "--variant", "continuous", "--epochs", "40", "--slots", "5", "--seed", "0", "--output", str(output / "continuous")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--data', required=True, type=absolute_path)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
