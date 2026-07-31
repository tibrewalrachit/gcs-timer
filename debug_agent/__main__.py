"""CLI: python -m debug_agent --benchmark mul --gcs-output run.log --question "..."

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile) and the benchmark
reports under bm/<name>. --gcs-output is the captured stdout of a GCS_Timer
run; omit it to debug from the PrimeTime/HSPICE reports alone.
"""

import argparse
from pathlib import Path

from .report_db import ReportDB
from . import debug_question


def main():
    ap = argparse.ArgumentParser(description="GCS-Timer timing debug agent")
    ap.add_argument("--question", required=True, help="debug task in natural language")
    ap.add_argument("--benchmark", action="append", default=None,
                    help="benchmark name(s); repeatable (default: mul)")
    ap.add_argument("--bm-root", default="bm", help="benchmark root directory")
    ap.add_argument("--gcs-output", default=None,
                    help="file with captured GCS_Timer stdout for the benchmark(s)")
    args = ap.parse_args()

    stdout = Path(args.gcs_output).read_text() if args.gcs_output else ""
    dbs = {
        name: ReportDB(Path(args.bm_root) / name, stdout)
        for name in (args.benchmark or ["mul"])
    }
    print(debug_question(args.question, dbs))


if __name__ == "__main__":
    main()
