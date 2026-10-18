"""Finite textured-object scenes with hidden-state truth separated from noisy appearance and scene observations."""

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
    source = args.data or str(output / 'objects.jsonl')
    steps = []
    if not args.data:
        steps.append(['finite_data.py', '--output', str(output / 'objects.jsonl'), '--seed', '7'])
    return steps + [
        ["objectledger.py", source, "ledger.example.json", "--variant", "full", "--epochs", "40", "--slots", "5", "--seed", "0", "--output", str(output / "full")],
        ["objectledger.py", source, "ledger.example.json", "--variant", "continuous", "--epochs", "40", "--slots", "5", "--seed", "0", "--output", str(output / "continuous")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/finite-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--data', type=absolute_path, help='Optional prepared input; otherwise generate the finite setting')
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
