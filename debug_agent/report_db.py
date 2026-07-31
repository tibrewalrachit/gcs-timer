"""Structural Report Database for GCS-Timer benchmarks.

Parses the timing reports available for a benchmark into nested Python
structures that the Expert Report Agent can query with generated code
(the "Structural Report Database" of arXiv:2504.11502, Fig. 4c).

Reports per benchmark directory ``bm/<name>``:

* ``gcs_at``   -- GCS-Timer's own Output AT Report, parsed from captured stdout
* ``pt_at``    -- ``test.pt``: PrimeTime GBA arrival times per output port
* ``pt_cell``  -- ``pt_cell.results``: PrimeTime per-cell-arc slews/delays
* ``pt_net``   -- ``pt_net.results``: PrimeTime per-net-arc delays
* ``hspice``   -- ``spice_deck_all.txt``: HSPICE golden stage results (raw rows)
* ``gcs_log``  -- full GCS-Timer stdout (timing/GPU-memory/runtime lines)
"""

import re
from pathlib import Path


def parse_gcs_output(text):
    """Parse GCS-Timer stdout into {'at': {pin: {'fall': x, 'rise': y}}, 'log': [...]}."""
    at = {}
    in_report = False
    log_lines = []
    for line in text.splitlines():
        if "Output AT Report" in line:
            in_report = True
            continue
        if in_report:
            fields = line.split()
            if len(fields) == 3 and fields[0] != "Pin":
                try:
                    at[fields[0]] = {"fall": float(fields[1]), "rise": float(fields[2])}
                    continue
                except ValueError:
                    pass
            if at and fields:
                in_report = False
        stripped = line.strip()
        if stripped:
            log_lines.append(stripped)
    return {"at": at, "log": log_lines}


def parse_pt_at(path):
    """Parse test.pt: one line per output port -> {pin: {'fall': x, 'rise': y}}."""
    at = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) == 3:
            at[fields[0]] = {"fall": float(fields[1]), "rise": float(fields[2])}
    return at


def parse_float_rows(path):
    """Parse a whitespace-separated numeric file into a list of float rows."""
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            rows.append([float(f) for f in fields])
        except ValueError:
            continue
    return rows


class ReportDB:
    """All reports for one benchmark, keyed by report name."""

    def __init__(self, bench_dir, gcs_stdout=""):
        bench_dir = Path(bench_dir)
        self.benchmark = bench_dir.name
        gcs = parse_gcs_output(gcs_stdout)
        self.reports = {
            "gcs_at": gcs["at"],
            "gcs_log": gcs["log"],
            "pt_at": parse_pt_at(bench_dir / "test.pt"),
            "pt_cell": parse_float_rows(bench_dir / "pt_cell.results"),
            "pt_net": parse_float_rows(bench_dir / "pt_net.results"),
            "hspice": parse_float_rows(bench_dir / "spice_deck_all.txt"),
        }

    def __getitem__(self, name):
        return self.reports[name]

    def summary(self):
        """Small structural overview shown to the agents (never the full data)."""
        r = self.reports
        return {
            "benchmark": self.benchmark,
            "gcs_at": f"{len(r['gcs_at'])} output ports (dict pin -> {{fall, rise}})",
            "pt_at": f"{len(r['pt_at'])} output ports (dict pin -> {{fall, rise}})",
            "pt_cell": f"{len(r['pt_cell'])} cell-arc rows "
                       "(list of [slew_fall, slew_rise, delay_fall, delay_rise])",
            "pt_net": f"{len(r['pt_net'])} net-arc rows (list of [delay_fall, delay_rise])",
            "hspice": f"{len(r['hspice'])} raw HSPICE rows (variable-length float lists)",
            "gcs_log": f"{len(r['gcs_log'])} log lines from the GCS-Timer run",
        }
