"""Hierarchical timing-debug agents (arXiv:2504.11502, Section IV).

Level 1  PlannerAgent      -- decomposes the user's debug task into
                              per-benchmark subtasks and synthesizes the
                              final answer (the paper's MCMM Planner; our
                              "corners/modes" are the EPFL benchmarks).
Level 2  TraversalAgent    -- given one subtask, traverses the TDRG and
                              decides which reports to query.
Level 3  ExpertReportAgent -- "Agentic Coding" retrieval: writes Python that
                              runs against the ReportDB and returns only the
                              data needed, keeping raw reports out of the
                              LLM context.
"""

import contextlib
import io
import json
import traceback

import anthropic

from .tdrg import TDRG

MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
THINKING = {"type": "adaptive"}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "benchmark": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["benchmark", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}


class ExpertReportAgent:
    """Level 3: retrieves data from one benchmark's ReportDB by writing code."""

    RUN_PYTHON_TOOL = {
        "name": "run_python",
        "description": (
            "Execute Python code against the structural report database. The "
            "variable `db` is a dict of report name -> parsed report data. "
            "Print whatever you need to see; stdout is returned to you. Keep "
            "printed output small (aggregate, slice, or rank instead of "
            "dumping whole reports)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    }

    def __init__(self, client, db, max_rounds=6):
        self.client = client
        self.db = db
        self.max_rounds = max_rounds

    def _run_code(self, code):
        namespace = {"db": self.db.reports}
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace)  # trusted session: our own reports + LLM-written analysis code
            out = stdout.getvalue()
            return (out if out.strip() else "(code ran but printed nothing)"), False
        except Exception:
            return stdout.getvalue() + traceback.format_exc(limit=3), True

    def retrieve(self, request):
        system = (
            "You are an Expert Report Agent for static timing analysis "
            f"reports of benchmark '{self.db.benchmark}'. Retrieve exactly the "
            "information requested by writing Python with the run_python tool. "
            "Report database structure:\n"
            + json.dumps(self.db.summary(), indent=2)
            + "\nWhen you have the data, reply with a compact plain-text "
            "answer containing the retrieved values (no code, no commentary)."
        )
        messages = [{"role": "user", "content": request}]
        for _ in range(self.max_rounds):
            response = self.client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, thinking=THINKING,
                system=system, tools=[self.RUN_PYTHON_TOOL], messages=messages,
            )
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return "".join(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                out, is_err = self._run_code(tu.input["code"])
                results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": out[:20000], "is_error": is_err,
                })
            messages.append({"role": "user", "content": results})
        return "Retrieval did not converge within the round limit."


class TraversalAgent:
    """Level 2: plans multi-report retrieval on the TDRG for one subtask."""

    def __init__(self, client, db, tdrg=None, max_rounds=8):
        self.client = client
        self.db = db
        self.tdrg = tdrg or TDRG()
        self.expert = ExpertReportAgent(client, db)
        self.max_rounds = max_rounds
        self.tool = {
            "name": "retrieve_from_report",
            "description": (
                "Ask the Expert Report Agent to retrieve information from the "
                "report database (it writes and runs code for you). Phrase a "
                "precise, self-contained retrieval request, naming the "
                "report(s) and the computation you want."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"request": {"type": "string"}},
                "required": ["request"],
                "additionalProperties": False,
            },
        }

    def solve(self, subtask):
        system = (
            "You are a TDRG Traversal Agent debugging GCS-Timer, a "
            "GPU-accelerated CCS-model static timing analyzer, on benchmark "
            f"'{self.db.benchmark}'. Plan which reports to consult using the "
            "debug relation graph below, retrieve data step by step with the "
            "retrieve_from_report tool, then reason to a conclusion.\n\n"
            + self.tdrg.to_prompt()
            + "\n\nWhen finished, reply with your analysis result: concrete "
            "numbers, the worst offenders, and the most likely explanation."
        )
        messages = [{"role": "user", "content": subtask}]
        for _ in range(self.max_rounds):
            response = self.client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, thinking=THINKING,
                system=system, tools=[self.tool], messages=messages,
            )
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return "".join(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                answer = self.expert.retrieve(tu.input["request"])
                results.append({
                    "type": "tool_result", "tool_use_id": tu.id, "content": answer,
                })
            messages.append({"role": "user", "content": results})
        return "Traversal did not converge within the round limit."


class PlannerAgent:
    """Level 1: plans across benchmarks and synthesizes the final answer."""

    def __init__(self, client):
        self.client = client

    def plan(self, question, benchmarks):
        response = self.client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, thinking=THINKING,
            output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
            messages=[{
                "role": "user",
                "content": (
                    "You are the Planner Agent of a timing-debug system for "
                    "GCS-Timer. Break the user's task into one self-contained "
                    "subtask per relevant benchmark (available benchmarks with "
                    f"loaded reports: {benchmarks}). Keep the plan minimal — "
                    "only benchmarks the task actually concerns.\n\nTask: "
                    + question
                ),
            }],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)["subtasks"]

    def answer(self, question, results):
        report = "\n\n".join(
            f"[{r['benchmark']}] {r['description']}\nResult: {r['result']}"
            for r in results
        )
        response = self.client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, thinking=THINKING,
            messages=[{
                "role": "user",
                "content": (
                    "You are the Planner Agent of a timing-debug system for "
                    "GCS-Timer. Compile the subtask results below into the "
                    "final answer for the user. Lead with the direct answer, "
                    "then the supporting numbers.\n\nUser task: " + question
                    + "\n\nSubtask results:\n" + report
                ),
            }],
        )
        return "".join(b.text for b in response.content if b.type == "text")


def debug_question(question, dbs, log=print):
    """Run the full hierarchy: plan -> traverse per benchmark -> final answer.

    dbs: dict benchmark name -> ReportDB
    """
    client = anthropic.Anthropic()
    planner = PlannerAgent(client)
    subtasks = planner.plan(question, sorted(dbs))
    log(f"[planner] {len(subtasks)} subtask(s)")
    results = []
    for st in subtasks:
        bench = st["benchmark"]
        if bench not in dbs:
            log(f"[planner] skipping unknown benchmark {bench!r}")
            continue
        log(f"[traversal:{bench}] {st['description']}")
        result = TraversalAgent(client, dbs[bench]).solve(st["description"])
        results.append({**st, "result": result})
    return planner.answer(question, results)
