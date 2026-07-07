# Run GCS-Timer on a Modal Labs H100 GPU.
#
# Usage:
#   modal run modal_app.py                          # run the 'mul' benchmark on GPU
#   modal run modal_app.py --benchmark log2         # run another benchmark (mul, log2, div, hyp)
#   modal run modal_app.py --benchmark mul --cpu    # run the CPU version for comparison
#   modal run modal_app.py --benchmark all          # run all four benchmarks
#
# The image build (done once on Modal's infrastructure, then cached):
#   1. Starts from an NVIDIA CUDA 12.4 devel image (provides nvcc).
#   2. Downloads the four ASAP7 CCS liberty files from the OpenROAD
#      asap7sc7p5t_28 repository into /app/lib (handles plain, gzipped,
#      and git-lfs stored files).
#   3. Copies this repo's src/ and bm/ directories into /app.
#   4. Unzips the div/hyp spef files and compiles with
#      nvcc -arch=sm_90 (H100 is compute capability 9.0).

import os

import modal

app = modal.App("gcs-timer")

# Select GPU at run time: GCS_GPU=B200 modal run modal_app.py ...
GPU_TYPE = os.environ.get("GCS_GPU", "B200")

# sm_100 (B200/Blackwell) plus sm_90 (H100/Hopper) fallback for comparison runs
NVCC_FLAGS = (
    "-x cu -std=c++14 -O3 -lineinfo "
    "-gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100"
)

# The ASAP7 CCS liberty files ship as small .lib.7z archives; download just
# the four GCS-Timer needs (raw URL, falling back to the media URL in case a
# file is stored in Git LFS and raw returns a pointer stub).
ASAP7_RAW = "https://raw.githubusercontent.com/The-OpenROAD-Project/asap7sc7p5t_28/master/LIB/CCS"
ASAP7_MEDIA = "https://media.githubusercontent.com/media/The-OpenROAD-Project/asap7sc7p5t_28/master/LIB/CCS"
LIB_FILES = [
    "asap7sc7p5t_INVBUF_RVT_TT_ccs_220122",
    "asap7sc7p5t_SIMPLE_RVT_TT_ccs_211120",
    "asap7sc7p5t_AO_RVT_TT_ccs_211120",
    "asap7sc7p5t_OA_RVT_TT_ccs_211120",
]

_fetch_libs = " && ".join(
    f"curl -fSL --retry 3 -o /tmp/{f}.lib.7z {ASAP7_RAW}/{f}.lib.7z && "
    f"if [ $(stat -c%s /tmp/{f}.lib.7z) -lt 10000 ]; then "
    f"curl -fSL --retry 3 -o /tmp/{f}.lib.7z {ASAP7_MEDIA}/{f}.lib.7z; fi && "
    f"7z x -o/app/lib /tmp/{f}.lib.7z && test -f /app/lib/{f}.lib && rm /tmp/{f}.lib.7z"
    for f in LIB_FILES
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "unzip", "p7zip-full")
    .run_commands(
        f"mkdir -p /app/lib && {_fetch_libs}",
        "ls -lh /app/lib",
    )
    .pip_install("anthropic")
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("bm", "/app/bm", copy=True)
    .add_local_dir("debug_agent", "/app/debug_agent", copy=True)
    .run_commands(
        "cd /app/bm/div && unzip -o test.spef.zip && rm test.spef.zip",
        "cd /app/bm/hyp && unzip -o test.spef.zip && rm test.spef.zip",
        f"cd /app && nvcc src/main.cpp {NVCC_FLAGS} -o GCS_Timer",
        f"cd /app && nvcc src/main.cpp {NVCC_FLAGS} -DEVALUATE=1 -o GCS_Timer_eval",
    )
)


@app.function(image=image, gpu=GPU_TYPE, cpu=8, memory=32768, timeout=3600)
def run_gcs_timer(benchmark: str = "mul", cpu: bool = False,
                  evaluate: bool = False, quiet: bool = False) -> str:
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], cwd="/app")
    binary = "./GCS_Timer_eval" if evaluate else "./GCS_Timer"
    cmd = [binary, benchmark] + (["-CPU"] if cpu else []) + (["-quiet"] if quiet else [])
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} exited with code {result.returncode}")
    return output


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=3600,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
)
def debug_gcs_timer(question: str, benchmark: str = "mul") -> str:
    """Run GCS-Timer on the H100, then debug its results with the timing
    analysis agent (arXiv:2504.11502-style TDRG + hierarchical agents)."""
    import sys

    sys.path.insert(0, "/app")
    from debug_agent import ReportDB, debug_question

    benchmarks = ["mul", "log2", "div", "hyp"] if benchmark == "all" else [benchmark]
    dbs = {}
    for bm in benchmarks:
        stdout = run_gcs_timer.local(benchmark=bm)
        dbs[bm] = ReportDB(f"/app/bm/{bm}", stdout)
    answer = debug_question(question, dbs)
    print("\n===== TIMING DEBUG AGENT ANSWER =====\n" + answer, flush=True)
    return answer


@app.local_entrypoint()
def main(benchmark: str = "mul", cpu: bool = False, question: str = "",
         evaluate: bool = False, quiet: bool = False):
    """Run benchmarks on the GPU selected by GCS_GPU (default B200); pass
    --evaluate to run the EVALUATE=1 accuracy harness, --question '...' to
    run the timing debug agent (needs Modal secret 'anthropic-api-key')."""
    if question:
        debug_gcs_timer.remote(question=question, benchmark=benchmark)
        return
    benchmarks = ["mul", "log2", "div", "hyp"] if benchmark == "all" else [benchmark]
    for bm in benchmarks:
        print(f"\n===== {bm} ({'CPU' if cpu else 'GPU'}"
              f"{', EVALUATE' if evaluate else ''}) =====")
        run_gcs_timer.remote(benchmark=bm, cpu=cpu, evaluate=evaluate, quiet=quiet)
