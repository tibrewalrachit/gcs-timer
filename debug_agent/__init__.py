"""LLM-based timing debug agent for GCS-Timer.

An adaptation of "Timing Analysis Agent: Autonomous Multi-Corner Multi-Mode
(MCMM) Timing Debugging with Timing Debug Relation Graph"
(Nainani et al., NVIDIA, arXiv:2504.11502) to GCS-Timer's reports:

* Structural Report Database  -> report_db.ReportDB
* Timing Debug Relation Graph -> tdrg.TDRG
* MCMM Planner Agent          -> agents.PlannerAgent   (plans across benchmarks)
* TDRG Traversal Agent        -> agents.TraversalAgent (plans across reports)
* Expert Report Agent         -> agents.ExpertReportAgent (coding agentic retrieval)
"""

from .report_db import ReportDB
from .tdrg import TDRG
from .agents import PlannerAgent, TraversalAgent, ExpertReportAgent, debug_question

__all__ = [
    "ReportDB",
    "TDRG",
    "PlannerAgent",
    "TraversalAgent",
    "ExpertReportAgent",
    "debug_question",
]
