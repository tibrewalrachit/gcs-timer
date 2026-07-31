"""Timing Debug Relation Graph (TDRG) for GCS-Timer.

Nodes are reports; each node describes the report's attributes, structure and
debugging usage. Edges encode the debug relations an experienced timing
engineer would traverse between reports (arXiv:2504.11502, Fig. 4b).
"""

NODES = {
    "gcs_at": (
        "GCS-Timer graph-based analysis (GBA) result: arrival time of each "
        "primary output port. Dict pin -> {fall: AT_ps, rise: AT_ps}. This is "
        "the tool-under-debug's answer; values follow the same sign convention "
        "as pt_at, so they can be compared per pin and per edge directly."
    ),
    "pt_at": (
        "PrimeTime GBA reference (test.pt): arrival time of each primary "
        "output port. Dict pin -> {fall: AT_ps, rise: AT_ps}. There is no "
        "golden GBA result; PrimeTime is a reference, not ground truth."
    ),
    "pt_cell": (
        "PrimeTime per-cell-arc stage results (pt_cell.results): list of rows "
        "[input_slew_fall, input_slew_rise, cell_delay_fall, cell_delay_rise] "
        "in ps, one row per (gate, input pin) arc in netlist order. Rows carry "
        "no names; use for distribution-level analysis (max/mean/percentiles)."
    ),
    "pt_net": (
        "PrimeTime per-net-arc stage results (pt_net.results): list of rows "
        "[net_delay_fall, net_delay_rise] in ps, one row per (net, sink pin) "
        "arc in netlist order. Unnamed rows; use for distribution-level "
        "analysis of interconnect delay."
    ),
    "hspice": (
        "HSPICE transistor-level golden stage results (spice_deck_all.txt): "
        "raw variable-length float rows grouped per driver stage. HSPICE is "
        "inherently more accurate than any gate-level calculator, so it acts "
        "as the reference for judging whether GCS-Timer or PrimeTime computes "
        "a stage delay more accurately (only meaningful in EVALUATE=1 runs "
        "where all tools share the same input slews)."
    ),
    "gcs_log": (
        "GCS-Timer run log: banner, lib/verilog/spef parse timings, GPU memory "
        "usage, CUDA error codes ('[END] init graph. error = N', 0 means OK), "
        "kernel timings and total runtime. First place to look for crashes, "
        "GPU out-of-memory, or abnormal runtimes."
    ),
}

EDGES = [
    ("gcs_at", "pt_at",
     "Compare arrival times per output pin and edge (fall/rise) to find the "
     "ports where GCS-Timer and PrimeTime disagree most; rank by |delta|."),
    ("gcs_at", "gcs_log",
     "If arrival times are missing or absurd, check the run log for CUDA "
     "errors, GPU memory exhaustion, or parse failures first."),
    ("pt_at", "pt_cell",
     "An endpoint AT is the accumulation of cell and net stage delays; a "
     "large endpoint mismatch usually traces back to stage-delay differences. "
     "Inspect the cell-delay distribution for outliers."),
    ("pt_at", "pt_net",
     "Same accumulation argument for interconnect: inspect the net-delay "
     "distribution for outlier stages."),
    ("pt_cell", "hspice",
     "Judge cell-delay accuracy against the HSPICE golden values; a tool "
     "whose cell delays sit closer to HSPICE is more accurate at that stage."),
    ("pt_net", "hspice",
     "Judge net-delay accuracy against the HSPICE golden values."),
]


class TDRG:
    """Text rendering of the graph for agent prompts."""

    def __init__(self, nodes=NODES, edges=EDGES):
        self.nodes = nodes
        self.edges = edges

    def to_prompt(self):
        lines = ["Timing Debug Relation Graph (reports and debug relations):", "", "NODES:"]
        for name, desc in self.nodes.items():
            lines.append(f"- {name}: {desc}")
        lines.append("")
        lines.append("EDGES (debug traces between reports):")
        for a, b, rel in self.edges:
            lines.append(f"- {a} <-> {b}: {rel}")
        return "\n".join(lines)
