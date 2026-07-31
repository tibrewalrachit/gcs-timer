# GCS-Timer B200 Optimization — Results Log

Cumulative per-phase results on NVIDIA B200 (Modal, CUDA 12.8, sm_100
binaries with sm_90 fallback). Baseline reference: `baseline.md`
(Phase 0, FP64, level-barrier scheduling, per-pivot global factorization).

Accuracy gate: EVALUATE=1 `MY-SP (all)` must stay within 0.1% relative /
0.1 ps absolute of the Phase 0 baseline
(mul 0.38 ps/1.98%, log2 0.35 ps/1.68%, div 0.30 ps/1.91%, hyp 0.39 ps/1.74%).

## Phase 1 — per-block local factorization (`-setup=block`, default)

Replaced the per-pivot global-memory factorization chains (one kernel
launch per pivot index, ~1000+ launches per run) with fused one-block-
per-stage/net kernels: matrices are staged into dynamic shared memory
when they fit, factorized locally with `__syncthreads()` between pivots,
and written back once. Oversized input-net matrices (n+1 > 64) stay on
the wide per-pivot chain (a single latency-bound block loses to the old
path when there are only 2·input_num large matrices). The old chain is
selectable with `-setup=global`.

**Correctness:** Output AT reports are **bit-identical** between
`-setup=block` and `-setup=global` on all four benchmarks (pivot
arithmetic ordering matches the old kernels exactly), and EVALUATE
MY-SP metrics are **identical to the Phase 0 baseline** in every
category. Accuracy gate: pass (zero delta).

**Launch count** for factorization setup: from ~2·(2·max_m+4) stage-chain
launches + 2·(max_n+1)+1 input launches + 2·max_inode+3 prebuild launches
(≈ 1000–2000 per run) down to **one launch per pipeline** (+ a short
restricted chain for the few oversized input nets).

**Timing (ms, B200, `-setup=global` → `-setup=block`):**

| bench | pass0 setup | pass1 setup | input nets | design load | GPU STA total |
|-------|-------------|-------------|------------|-------------|---------------|
| mul   | 11.2 → 8.3  | 13.8 → 13.0 | 57.1 → 58.9 | 414 → 417   | 122.1 → 118.6 |
| log2  | 12.3 → 8.9  | 15.0 → 13.7 | 22.1 → 24.5 | 634 → 563   | 171.8 → 165.0 |
| div   | 20.5 → 20.3 | 24.4 → 31.6 | 68.4 → 70.8 | 1771 → 1655 | 379.1 → 371.7 |
| hyp   | 63.1 → 43.3 | 79.8 → 68.6 | 204.6 → 206.7 | 3654 → 3557 | 2306.2 → 2263.2 |

Methodology note: Modal B200 instances show visible run-to-run variance
(one contended run measured the untouched init section at 1.75× and
pass1 simulate at 4.5× their usual values). Numbers above are from
clean runs; later phases should confirm speedups with repeated runs
before drawing conclusions from deltas under ~20%.

Observations:

* The stage chains improve (hyp 143 → 112 ms combined) and prebuild
  (inside design load) improves ~100 ms on the big designs, but the
  absolute win is modest — on B200 the old chain's launches were cheap
  and the element work (B/C/D block assembly) is unchanged. The
  structural win is the launch-count collapse and the removal of
  device-wide syncs, which is what Phase 4.1 (CUDA Graphs) and the
  Phase 3 tree solvers build on.
* `pass0 simulate` still dominates (84% of STA on hyp) — that is the
  Phase 2/3 target, not this one.

**Deviation from the plan (documented):** the factorization kernels use
block-local Gauss-Jordan rather than Cholesky. Reasons: (a) identical
pivot arithmetic gives bit-identical results vs the old path, which made
the correctness gate trivial to verify; (b) the input-net MNA matrices
are indefinite (zero diagonal at the source row), so they cannot use
Cholesky at all; (c) the SPD factorizations amount to ≤ 43 ms per run,
so Cholesky's 2× flop saving is worth ≤ ~20 ms on the largest design.
Cholesky lands with Phase 2, where the factorization kernels are being
rewritten for FP32/mixed precision anyway and the arithmetic changes
regardless.

## Phase 2 — mixed precision + transposed inverses (`GCS_PRECISION=32`)

The timestep loops consume the n×n stage/input inverses once per
timestep. Two changes: (a) the consumed inverse is now stored
**transposed** so thread t reads `rinv[i*n+t]` fully coalesced (the old
`idx(t,i,n)` layout strided by n across the warp); (b) with
`-DGCS_PRECISION=32` (the `*_f32` binaries) that array is FP32, halving
its bytes. Voltage-state recurrences, crossing-time interpolation,
arrival propagation (`atomicMax64`) and every factorization stay FP64 —
this is the plan's "mixed" configuration. Iterative refinement (2.2) was
not needed: accuracy passed the ship gate without it (see below).

**Correctness:**
* FP64 binary (transposed layout only): Output AT reports **bit-identical**
  to Phase 1 on all four benchmarks.
* FP32 binaries: EVALUATE MY-SP metrics **identical to the FP64 baseline
  at reported precision in every category** (e.g. hyp all: 0.39 ps /
  1.74% in both) — far inside the 0.05% ship gate. FP32 ships; the
  effective precision of the extracted quantities is set by the FP64
  state recurrence, not the FP32 inverse elements.

**Timing (ms, B200; Phase 1 → Phase 2 FP64-transposed → Phase 2 FP32):**

| bench | pass0 simulate       | GPU STA total        | device mem (MB)      |
|-------|----------------------|----------------------|----------------------|
| mul   | 65.4 → 51.5 → 47.0   | 118.6 → 126.5 → 98.8 | 4236 → 5210 → 4724   |
| log2  | 105.9 → 84.2 → 77.9  | 165.0 → 144.1 → 137.1| 4808 → 6012 → 5410   |
| div   | 250.6 → 227.6 → 222.4| 371.7 → 350.0 → 359.2| 7576 → 9646 → 8612   |
| hyp   | 1944.5 → 1721.2 → 1670.4 | 2263.2 → 2086.3 → 2034.3 | 19726 → 25474 → 22602 |

Observations:

* The transpose (coalescing) is worth ~10–12% of pass0 alone; FP32 adds
  a few percent more. Cumulative STA speedup vs the Phase 0 baseline:
  mul 1.24×, log2 1.25×, hyp 1.13× (div's run hit a noisy instance —
  its input-nets section inflated 76→127 ms on code that is identical
  in that section).
* The muted FP32 gain is diagnostic: with B200's ~126 MB L2, a level's
  working set of stage inverses is substantially cache-resident, and the
  level loop exposes launch/sync latency (few resident blocks per small
  level). pass0 is therefore latency/occupancy-bound as much as
  bandwidth-bound — exactly what Phase 4 (graphs/queue scheduling) and
  Phase 3 (tree solvers shrink n² to n) attack.
* Memory grew (the rinv copy): +29% fp64 / +15% fp32 on hyp. Phase 3.1
  eliminates the n² arrays for tree nets entirely; Phase 5 can also
  assemble directly into rinv and drop stage_inv.
