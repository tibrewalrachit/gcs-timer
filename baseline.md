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
HSPICE; `PT-SP` = PrimeTime vs HSPICE). Regression gate for all later
phases: MY-SP must stay within 0.1% relative / 0.1 ps absolute of these
values.

<!-- PENDING: filled from the --evaluate run -->

## H100 (sm_90) comparison

<!-- PENDING: filled from the GCS_GPU=H100 run -->
