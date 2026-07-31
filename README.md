# GCS-Timer

GCS-Timer is a GPU-accelerated static timing analyzer based on the advanced Composite Current Source (CCS) timing model. 

Check out the following paper for more details.

* Shiju Lin, Guannan Guo, Tsung-Wei Huang, Weihua Sheng, Evangeline F.Y. Young and Martin D.F. Wong, ["GCS-Timer: GPU-Accelerated Current Source Model Based Static Timing Analysis"](https://shijulin.github.io/files/430_Camera_Ready_Paper.pdf), ACM/IEEE Design Automation Conference (DAC), San Francisco, CA, USA, June 23–27, 2024.

## Download
```bash
git clone https://github.com/cuhk-eda/GCS-Timer.git
mkdir GCS-Timer/lib
```
Download the following to the `GCS-Timer/lib` directory from the `LIB/CCS` folder of [ASAP7](https://github.com/The-OpenROAD-Project/asap7sc7p5t_28).
* asap7sc7p5t_INVBUF_RVT_TT_ccs_220122.lib
* asap7sc7p5t_SIMPLE_RVT_TT_ccs_211120.lib
* asap7sc7p5t_AO_RVT_TT_ccs_211120.lib
* asap7sc7p5t_OA_RVT_TT_ccs_211120.lib

## Compile
```bash
nvcc src/main.cpp -x cu -arch=sm_86 -o GCS_Timer -std=c++14 -O3
```
You may want to change `-arch=sm_86` according to your GPU.

## Run
```bash
./GCS_Timer <benchmark_name> [-CPU]
```
`benchmark_name` can be `mul`, `log2`, `div` and `hyp`, which are the designs used for evaluation in the paper.

`-CPU` enables the CPU version.

Examples:

Running `mul` on GPU

```bash
./GCS_Timer mul
```

Running `mul` on CPU

```bash
./GCS_Timer mul -CPU
```

The spef files of `div` and `hyp` are zipped and need to be unzipped first.

## Running on Modal Labs (H100)

`modal_app.py` packages the whole flow (downloading the ASAP7 libs, unzipping the spef files, compiling with `-arch=sm_90` and running) into a [Modal](https://modal.com) app that executes on an H100 GPU.

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>   # or `modal setup`

modal run modal_app.py                        # run 'mul' on the H100
modal run modal_app.py --benchmark log2       # any of: mul, log2, div, hyp, all
modal run modal_app.py --benchmark mul --cpu  # CPU version for comparison
```

The first run builds the container image (fetches the ASAP7 CCS liberty files and compiles GCS-Timer); subsequent runs reuse the cached image and start immediately.

## Timing Debug Agent

`debug_agent/` implements an LLM-based timing debugging assistant modeled on ["Timing Analysis Agent: Autonomous Multi-Corner Multi-Mode (MCMM) Timing Debugging with Timing Debug Relation Graph"](https://arxiv.org/abs/2504.11502) (Nainani et al., NVIDIA), adapted to GCS-Timer's reports:

* a **Structural Report Database** parses GCS-Timer's output AT report plus the PrimeTime (`test.pt`, `pt_cell.results`, `pt_net.results`) and HSPICE (`spice_deck_all.txt`) references shipped with each benchmark;
* a **Timing Debug Relation Graph (TDRG)** connects those reports with expert debug relations (e.g. endpoint AT mismatch → stage-delay distributions → HSPICE golden comparison);
* a three-level agent hierarchy — **Planner** (across benchmarks) → **TDRG Traversal** (across reports) → **Expert Report Agent** (agentic *coding* retrieval that writes Python against the report database) — answers debug questions with Claude.

Run it on Modal right after the H100 timing run (create a Modal secret named `anthropic-api-key` holding your `ANTHROPIC_API_KEY` first):

```bash
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...
modal run modal_app.py --benchmark mul --question \
  "Which output ports disagree most between GCS-Timer and PrimeTime, and what is the likely cause?"
```

Or locally against the shipped reports:

```bash
pip install anthropic
python -m debug_agent --benchmark mul --gcs-output run.log --question "..."
```

## Benchmarks and Evaluation

The benchmarks are the four largest arithmetic designs from the [EPFL combinational benchmark suite](https://www.epfl.ch/labs/lsi/page-102566-en-html/benchmarks/). They are synthesized, placed and routed using commercial tools. Each design consists of a verilog file (`test.v`) and a spef file (`test.spef`) for timing analysis. Delay results of each cell and net are provided in `pt_cell.results/pt_net.results` (by PrimeTime) and `spice_deck_all.txt` (by HSPICE). `test.pt` contains the graph-based analysis results by PrimeTime.

### Stage Delay Calculation
To evaluate the net/cell delay accuracy of GCS-Timer, variable `EVALUATE` in file `gpu_timer.hpp` must be set to `1`. This sets the input slew of each driver gate to the same value as that is used to generate the PrimeTime and HSPICE delay results (this value is extracted from PrimeTime) to ensure fair comparison. When `EVALUATE` is set to `1`, GCS-Timer reads `pt_cell.results/pt_net.results/spice_deck_all.txt` and output the comparison results.

HSPICE is a transistor-level simulation tool that is inherently more accurate than gate-level delay calculators such as PrimeTime and GCS-Timer. Using HSPICE as a reference tool, we show that GCS-Timer is more accurate than PrimeTime in terms of net/cell delay calculation.

### Graph-Based Analysis (GBA)
By default (`EVALUATE=0`), GCS-Timer generates the arrival time of each output port, which can be compared with PrimeTime's timing results in `test.pt`. This comparison is for reference only, as there are no "golden" results (GBA is beyond the capabilities of HSPICE). Nevertheless, we believe that GCS-Timer is more accurate than PrimeTime, because GBA essentially accumulates cell/net delays in a graph and the individual cell/net delays calculated by GCS-Timer are more accurate.

## Contact
[Shiju Lin]([https://shijulin.github.io/](https://personal.hkust-gz.edu.cn/shijulin/)) (email: shijulin@hkust-gz.edu.cn)

## License
BSD 3-Clause License
