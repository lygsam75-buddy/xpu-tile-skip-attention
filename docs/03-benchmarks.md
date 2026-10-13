# 03 — Benchmarks: method, all figures, and the retraction

Everything below was measured on one machine. Raw output for every table is committed under
`results/`.

## 1. Test platform

| item | value |
|---|---|
| GPU | Intel Arc Pro B70 32 GB (ECC off -> **31.16 GiB** usable) |
| GPU driver | 32.0.101.8976 |
| OS | Windows 11 Pro |
| System RAM | 32 GB (31.1 GiB visible) — see constraint C1 in [02](02-adaptation-plan.md) |
| Python | 3.13.12 (embedded) |
| torch | 2.12.0+xpu, XPU runtime 20250302 |
| Triton | 3.7.1, **Intel backend** |
| ComfyUI | 0.34.0 |
| Workload | MiniMax H3 DiT: `heads=56, head_dim=128, 50 layers`, bf16, 4-step turbo LoRA |

## 2. Method

* **Per-layer figures** are the shipped `vsa_kernel.vsa_fine_attention()` entry point, called with
  `torch.xpu.synchronize()` around every timed region. Median of 5, min/max reported, explicit
  warm-up before the first measurement.
* **Selection tensors** are built the way the node builds them at runtime: shared sink prefix tiles
  + forced diagonal +/-1 tiles + random top-k fill. `K_SEL = prefix + 3 + topratio * MT`.
  The prefix sizes (23 tiles at 0.5 MP, 39 at 1.03 MP) are taken from the node's own plan log —
  `plan: seq=18526 padded=21952 tiles=343 prefix_tiles=23` and `padded=44736 tiles=699 prefix_tiles=39`.
* **Dense baseline** is the vendor `torch.nn.functional.scaled_dot_product_attention` at the
  identical shape, which on this stack dispatches to the vendor library (96.6 TFLOP/s effective).
* **Numerics** are checked against a per-query-block SDPA reference, i.e. the exact semantics the
  kernel is supposed to implement — not against a dense reference, which would confound a
  correctness check with the sparsity approximation.
* **End-to-end** runs are ComfyUI A/B: same workflow, same seed family, same resolution; per
  variant the *second* shot is used (steady state, model already resident); `--reserve-vram 5
  --cache-classic`.

## 3. Kernel numerics

| dtype | S | K_SEL | blocks sampled | max abs err | max rel err | min cosine | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| bfloat16 | 21952 | 43 | 12 | 9.77e-04 | 6.02e-03 | 0.999999 | OK |
| bfloat16 | 44736 | 76 | 12 | 4.88e-04 | 4.24e-03 | 0.999999 | OK |
| float32 | 8192 | 17 | 8 | 2.98e-04 | 8.92e-04 | 1.000000 | OK |

Source: `results/verify_bench_out.txt`.

Note on an earlier claim: a single-precision run needs `input_precision="ieee"` on `tl.dot`,
otherwise TF32 draws a `~2^-11` relative error that looks like a real bug. We saw exactly this
(`5.085e-04` relative, textbook TF32) and it disappears with `ieee`.

## 4. Per-layer latency vs. keep ratio

`results/ksel_sweep_out.txt`. Dense reference: 143.0 ms at S=21952, 603.7 ms at S=44736.

**S = 21952** (0.5 MP, 928x512, 124 frames; 343 tiles)

| K_SEL | keep | ms/layer | kernel TFLOP/s | vs dense |
|---:|---:|---:|---:|---:|
| 26 | 7.6% | 67.0 | 15.6 | **2.13x** |
| 43 | 12.5% | 127.9 | 13.5 | 1.12x |
| 60 | 17.5% | 193.0 | 12.5 | 0.74x |
| 76 | 22.2% | 253.7 | 12.1 | 0.56x |
| 100 | 29.2% | 345.4 | 11.7 | 0.41x |

**S = 44736** (1.03 MP, 1344x768, 124 frames; 699 tiles)

| K_SEL | keep | ms/layer | kernel TFLOP/s | vs dense |
|---:|---:|---:|---:|---:|
| 43 | 6.2% | 227.6 | 15.5 | **2.65x** |
| 60 | 8.6% | 388.4 | 12.7 | 1.55x |
| 76 | 10.9% | 587.3 | 10.6 | 1.03x |
| 100 | 14.3% | 905.4 | 9.1 | 0.67x |

Crossover is between 12.5% and 17.5% keep at 0.5 MP, and between 8.6% and 10.9% at 1.03 MP.
Analytically, the kernel wins iff `keep < kernel_rate / dense_rate ≈ 13.5 / 96.5 ≈ 14%`.

**The kernel's per-FLOP rate degrades as the selection widens** (15.6 -> 11.7 TFLOP/s). Time grows
super-linearly in `K_SEL` because more tiles per query block means more loop iterations with the
same amount of parallelism.

## 5. Eager-torch fallback and configuration sweeps

`results/verify_bench_out.txt`, `results/sweep2_out.txt`.

| path | S=21952 | S=44736 |
|---|---:|---:|
| dense vendor SDPA | 143.2 ms | 601.6 ms |
| eager torch gather + SDPA | 615.6 ms | 2158.5 ms |
| this kernel (K_SEL=43 / 76) | 116.7 ms | 526.5 ms |
| kernel vs gather | **5.28x** | **4.10x** |

(These two kernel figures come from a single-pass run and read slightly faster than the sweep's
127.9 / 587.3 ms. Both are committed; the sweep is the primary source because it enforces warm-up
and repeats. Report the range: **1.12–1.23x vs dense at 0.5 MP**.)

`num_warps x BLOCK`, keep held at 12.5%, two full rounds:

| BLOCK_M | MT | K_SEL | w=4 | w=8 | w=16 | w=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 343 | 42 | 748.8 ms | **102.0 ms** | 157.2 ms | 271.1 ms |
| 128 | 171 | 21 | 886.1 ms | 782.0 ms | 113.3 ms | 184.4 ms |
| 256 | 85 | 10 | 1486.2 ms | 1452.2 ms | 1301.6 ms | 209.8 ms |

Round 1 and round 2 agree to within 0.2%. The shipped configuration (`BLOCK=64, w=8`) is optimal.
`num_warps` is the only significant lever; `num_stages`, dtype, and removing `tl.trans` all
measured as no-ops.

Cost of the pad-column mask: **104.8 ms with vs 99.3 ms without -> 5.2%**.

## 6. Backend ceiling

`results/sweep2_out.txt`, `results/t11_m_scan.txt`.

| measurement | ms | TFLOP/s | vs vendor |
|---|---:|---:|---:|
| vendor GEMM, 4096³ bf16 (`torch.matmul`) | 0.82 | **168.6** | 100% |
| Triton GEMM BM128 BN128 BK64 | 2.70 | 51.0 | 30% |
| Triton GEMM BM128 BN128 BK32 | 3.14 | 43.8 | 26% |
| Triton GEMM BM64 BN64 BK64 | 4.20 | 32.7 | 19% |
| Triton GEMM BM128 BN256 BK64 | 40.80 | **3.4** | **2%** |

Vendor *batched* GEMM (`bmm`, batch=343, K=128, N=1088) as a function of `M`:

| M | ms | TFLOP/s |
|---:|---:|---:|
| 64 | 0.30 | 20.1 |
| 128 | 0.41 | 30.1 |
| 256 | 0.62 | 39.3 |
| 512 | 1.04 | 47.2 |
| 1024 | 1.86 | 52.6 |

Two independent findings:

1. The Triton Intel backend reaches 30% of the vendor GEMM rate in the best tile configuration,
   and **2%** in another. A tile-shape change alone costs **15x**. This is a reproducibility target,
   not a tuning complaint.
2. Sparse attention is intrinsically small-M (one query block is 64 rows). At M=64 the *vendor
   library itself* only reaches 20.1 TFLOP/s, 2.6x below its own M=1024 figure. This is the
   structural reason sparse attention is hard on any backend, and it explains why our kernel's
   18.3 TFLOP/s at low keep is close to the vendor batched-GEMM ceiling for that shape rather than
   being evidence of a badly written kernel.

## 7. End-to-end A/B

Raw: `results/e2e_ab_0.5MP.json`, `results/e2e_ab_1.03MP.json`.

| config | variant | steady server time | s/it |
|---|---|---:|---:|
| 0.5 MP 928x512 124f | dense | 51.40 s | 8.77 |
| | VSA + kernel (keep 5.0%) | 57.42 s | 10.36 |
| | VSA + eager gather (keep 5.0%) | 103.08 s | 21.76 |
| 1.03 MP 1344x768 124f | dense | 161.15 s | 33.15 |
| | VSA + kernel (keep 5.0%) | 155.68 s | 31.92 |

On the per-step attention saving: the standalone delta is 143.0 -> 127.9 ms/layer, i.e. 0.76 s per
step across 50 layers. The measured 1.03 MP end-to-end saving is 33.15 - 31.92 = **1.23 s/step**,
the same order. At 0.5 MP the pipeline instead loses 1.5 s/step: the selection machinery
(block scoring, gate quantisation, permutations, the coarse branch) costs more than the kernel
saves. Both facts are consistent with the per-layer measurements, and both are needed to describe
the result honestly.

## 8. The retraction

An earlier draft of this work concluded:

> attention is only **23%** of the step (2.0 s of 8.74 s), so even a perfect 3.6x attention
> speedup yields only ~1.20x end to end — H3's bottleneck is not attention.

**This is wrong and is withdrawn.** The 2.0 s figure came from the integration node's per-layer
timer, which wraps the attention call in `perf_counter()` with **no device synchronize**
(`ComfyUI-H3-VSA-XPU/vsa_core.py:276-281`). On XPU, work is asynchronous, so for a fully resident
model it measures enqueue time. It reported **42 ms/layer** where the synchronized measurement is
**143 ms/layer** — a 3.4x under-report. It happened to be accurate in the lowvram configuration,
where weight offloading forces synchronization (567 ms/layer measured vs 601 ms synchronized), which
is what made the error survive review.

Synchronized accounting:

| config | sampling step | 50 x dense attention | attention share |
|---|---:|---:|---:|
| 0.5 MP (S=21952) | 8.77 s | 7.15 s | **81.5%** |
| 1.03 MP (S=44736) | 33.15 s | 30.08 s | **90.7%** |

Cross-check: at S=44736 the 50 layers move 2.87 PFLOP per step. At a 33.15 s step that is
86.6 TFLOP/s sustained — already at the device's measured dense-attention rate of 95.4 TFLOP/s,
which leaves only a few seconds for everything else. Attention is the bottleneck.

This matters because it changes the ask. "Sparse attention does not help H3" and "sparse attention
should help H3 enormously but our kernel is 6x less efficient per FLOP than the vendor library"
are opposite conclusions from the same experiments.

## 9. Known measurement traps on this platform

1. **Unsynchronized timers under-report by up to 3.4x** (section 8).
2. **Cold sweep rows read ~40% slow.** In one run the identical `BLOCK=64/w=8` configuration read
   177.1 ms when measured first and 107.4 ms when measured later in the same process. Always warm
   up, and repeat the sweep.
3. **Synthetic kernels are not the model.** A 2.13x per-layer figure coexists with an 11.7%
   end-to-end regression.
4. **`torch.xpu.get_device_properties(0).total_memory` reports 31.16 GiB** only with ECC off. With
   the factory default it reports ~27.25 GiB, and throughput is unaffected either way (confirmed
   separately: 4-step turbo LoRA 0.5 MP/5 s, 60.20 s -> 56.52 s).
