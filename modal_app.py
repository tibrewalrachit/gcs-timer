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

import modal

app = modal.App("gcs-timer")

ASAP7_REPO = "https://github.com/The-OpenROAD-Project/asap7sc7p5t_28.git"
LIB_FILES = [
    "asap7sc7p5t_INVBUF_RVT_TT_ccs_220122",
    "asap7sc7p5t_SIMPLE_RVT_TT_ccs_211120",
    "asap7sc7p5t_AO_RVT_TT_ccs_211120",
    "asap7sc7p5t_OA_RVT_TT_ccs_211120",
]

_copy_libs = " && ".join(
    f"if [ -f /tmp/asap7/LIB/CCS/{f}.lib ]; then cp /tmp/asap7/LIB/CCS/{f}.lib /app/lib/; "
    f"elif [ -f /tmp/asap7/LIB/CCS/{f}.lib.gz ]; then gunzip -c /tmp/asap7/LIB/CCS/{f}.lib.gz > /app/lib/{f}.lib; "
    f"else echo 'MISSING {f}' && ls /tmp/asap7/LIB/CCS && exit 1; fi"
    for f in LIB_FILES
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "git-lfs", "unzip")
    .run_commands(
        # Sparse clone keeps the download to just LIB/CCS; lfs pull covers
        # the case where the .lib files are stored in git-lfs.
        f"git clone --depth 1 --filter=blob:none --sparse {ASAP7_REPO} /tmp/asap7",
        "cd /tmp/asap7 && git sparse-checkout set LIB/CCS && (git lfs pull --include 'LIB/CCS/*' || true)",
        f"mkdir -p /app/lib && {_copy_libs}",
        "rm -rf /tmp/asap7 && ls -lh /app/lib",
    )
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("bm", "/app/bm", copy=True)
    .run_commands(
        "cd /app/bm/div && unzip -o test.spef.zip && rm test.spef.zip",
        "cd /app/bm/hyp && unzip -o test.spef.zip && rm test.spef.zip",
        "cd /app && nvcc src/main.cpp -x cu -arch=sm_90 -o GCS_Timer -std=c++14 -O3",
    )
)


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=3600)
def run_gcs_timer(benchmark: str = "mul", cpu: bool = False) -> str:
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], cwd="/app")
    cmd = ["./GCS_Timer", benchmark] + (["-CPU"] if cpu else [])
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"GCS_Timer exited with code {result.returncode}")
    return output


@app.local_entrypoint()
def main(benchmark: str = "mul", cpu: bool = False):
    benchmarks = ["mul", "log2", "div", "hyp"] if benchmark == "all" else [benchmark]
    for bm in benchmarks:
        print(f"\n===== {bm} ({'CPU' if cpu else 'GPU'}) =====")
        run_gcs_timer.remote(benchmark=bm, cpu=cpu)
