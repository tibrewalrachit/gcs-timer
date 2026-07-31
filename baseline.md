# GCS-Timer B200 Optimization — Phase 0 Baseline

Recorded after Phase 0 instrumentation (commit `281db38`) and before any
optimization. All numbers from Modal `gpu="B200"` containers (NVIDIA B200,
183 GB, CUDA 12.8, driver-reported), binaries built with
`-std=c++14 -O3 -lineinfo -gencode sm_90 -gencode sm_100`, fixed
`deltaT = 1`, FP64 everywhere, level-barrier scheduling (as-shipped).

Timing source: `cudaEvent` pairs per kernel group, `std::chrono` for host
sections (Phase 0.1). "GPU STA total" = setup chains + pass0 + pass1 +
propagate (excludes parsing and database init).

## Per-section wall time (ms) — B200

| Section                         | mul     | log2    | div     | hyp      |
|---------------------------------|---------|---------|---------|----------|
| read_lib (liberty parse, host)  | 5753.4  | 3808.7  | 3846.8  | 3994.6   |
| design load (verilog+spef, host)| 1075.8  | 1166.7  | 2768.8  | 4461.8   |
| init_database_GPU + init_graph  | 1570.9  | 1518.7  | 1693.5  | 2448.4   |
| input nets (mat+inv+delay)      | 56.7    | 22.7    | 68.9    | 208.7    |
| pass0 setup (matA+invA/BC/D)    | 11.2    | 12.0    | 20.8    | 64.4     |
| pass0 simulate (level loop)     | 65.6    | 106.1   | 251.7   | 1933.2   |
| pass1 setup (refactorization)   | 13.9    | 15.2    | 24.7    | 81.3     |
| pass1 simulate (size groups)    | 28.7    | 34.0    | 73.1    | 158.2    |
| propagate + AT readback         | 1.3     | 2.1     | 7.5     | 65.3     |
| **GPU STA total**               | **120.9** | **169.4** | **377.8** | **2302.5** |
| update_timing (wall, s)         | 1.75    | 1.72    | 2.18    | 5.02     |
| device memory in use (MB)       | 4236    | 4808    | 7576    | 19724    |

Observations driving the phase plan:

* `pass0 simulate` dominates GPU STA (54% on mul → 84% on hyp): FP64 dense
  mat-vec per timestep with fixed `deltaT = 1` and a device-wide sync per
  topological level. Targets: Phase 2 (FP32/mixed), Phase 3 (tree solvers,
  adaptive dt), Phase 4 (scheduling).
* The factorization chains (pass0+pass1 setup) are comparatively small in
  wall time on B200 (~25–146 ms) but issue O(max_m) launches with global
  traffic; Phase 1 collapses them into per-block shared-memory Cholesky.
* Device memory scales with n² stage/RC inverse storage (19.7 GB on hyp);
  Phase 3.1 tree solvers eliminate O(n²) storage for tree nets.
* Host-side parsing (liberty + spef) dwarfs GPU time; out of scope for the
  GPU phases but noted for completeness.

## Accuracy (EVALUATE=1 harness, B200, FP64 baseline)

Stage-delay comparison against HSPICE golden (`MY-SP` = GCS-Timer vs
HSPICE; `PT-SP` = PrimeTime vs HSPICE; mean abs error over arcs with
error ≥ 0.5 ps). **Regression gate for all later phases: MY-SP must stay
within 0.1% relative / 0.1 ps absolute of these values.**

| bench | category  | MY-SP (ps / %) | PT-SP (ps / %) | SP failed |
|-------|-----------|----------------|----------------|-----------|
| mul   | cell_fall | 0.59 / 2.95    | 0.58 / 2.44    | 1/65021   |
| mul   | cell_rise | 0.59 / 2.64    | 0.60 / 2.24    | 3/65021   |
| mul   | net_fall  | 0.10 / 0.65    | 0.06 / 1.41    | 254/59794 |
| mul   | net_rise  | 0.21 / 1.56    | 0.18 / 2.30    | 326/59794 |
| mul   | **all**   | **0.38 / 1.98**| 0.36 / 2.11    | 584/249630|
| log2  | cell_fall | 0.60 / 2.93    | 0.65 / 2.60    | 14/75085  |
| log2  | cell_rise | 0.55 / 2.40    | 0.60 / 2.11    | 9/75085   |
| log2  | net_fall  | 0.10 / 0.58    | 0.07 / 3.54    | 101/74290 |
| log2  | net_rise  | 0.12 / 0.79    | 0.10 / 3.73    | 183/74290 |
| log2  | **all**   | **0.35 / 1.68**| 0.36 / 2.99    | 307/298750|
| div   | cell_fall | 0.59 / 3.32    | 0.57 / 2.66    | 5/212904  |
| div   | cell_rise | 0.51 / 2.56    | 0.54 / 2.07    | 10/212904 |
| div   | net_fall  | 0.04 / 0.87    | 0.04 / 4.90    | 110/208661|
| div   | net_rise  | 0.04 / 0.85    | 0.04 / 4.60    | 162/208661|
| div   | **all**   | **0.30 / 1.91**| 0.30 / 3.54    | 287/843130|
| hyp   | cell_fall | 0.67 / 3.23    | 0.66 / 2.77    | 14/418011 |
| hyp   | cell_rise | 0.61 / 2.49    | 0.63 / 2.11    | 72/418011 |
| hyp   | net_fall  | 0.11 / 0.45    | 0.11 / 2.01    | 97/403726 |
| hyp   | net_rise  | 0.16 / 0.71    | 0.14 / 2.24    | 146/403726|
| hyp   | **all**   | **0.39 / 1.74**| 0.39 / 2.28    | 329/1643474|

EVALUATE-mode update_timing wall (s): mul 2.25, log2 2.40, div 7.09,
hyp 12.86 (includes reading PT/SPICE golden files).

## H100 (sm_90) comparison — same binaries, GCS_GPU=H100

| bench | GPU STA total (ms) | pass0 sim | pass1 sim | mem (MB) |
|-------|--------------------|-----------|-----------|----------|
| mul   | 141.7              | 67.8      | 37.2      | 4135     |
| log2  | 195.2              | 112.5     | 41.9      | 4707     |
| div   | 431.9              | 254.3     | 101.1     | 7475     |
| hyp   | 2439.1             | 1971.2    | 214.1     | 19623    |

B200 is only **1.06–1.17× faster than H100** on this FP64,
level-barrier-bound workload — far below the hardware's potential, which
is precisely the thesis of Phases 1–5 (B200's FP64 axis is weak; the wins
must come from FP32/mixed precision, shared-memory factorization, tree
solvers, and killing the level barriers).
