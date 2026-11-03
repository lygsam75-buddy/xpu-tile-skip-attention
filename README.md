# XPU tile-skipping attention — how much of Veda's speedup survives on Intel Arc

**A measured answer to a specific question: Veda's tile-skipping sparse-attention kernel is
NVIDIA-Hopper-only. If we rebuild it in Triton for Intel XPU, does the win survive?**

Short answer:

* **Yes, a correct XPU kernel can be built** — the toolchain works, the numbers match the dense
  reference, and it is **5.3x faster than the eager-torch fallback** it replaces.
* **No, the headline speedup does not survive as-is.** The kernel only beats the vendor dense
  SDPA when fewer than **~11–14%** of key tiles are kept. MiniMax H3's real VSA configuration
  keeps **10.9–12.5%** — i.e. it sits exactly on the break-even line, giving **1.03–1.12x** at
  attention level and roughly break-even end to end.
* **The reason is the Triton Intel backend, not the algorithm.** On this hardware, `tl.dot`
  reaches **~13–17 TFLOP/s** on attention shapes while the vendor SDPA reaches **96 TFLOP/s**
  and the vendor GEMM **168 TFLOP/s**. A kernel running at 1/6 of the vendor's per-FLOP rate
  needs sparsity below 1/6 to win. That is the whole story, and it is a fixable gap.

A secondary result matters as much: our first draft of this report concluded "attention is only
23% of the step, so Amdahl caps the gain at 1.2x". **That conclusion was wrong** and is retracted
below. With device-synchronized timing, attention is **81.5% (0.5MP) to 90.7% (1.03MP)** of the
sampling step. H3 at these resolutions is attention-bound, which makes the backend gap the single
highest-leverage thing to close. See [docs/03-benchmarks.md](docs/03-benchmarks.md#the-retraction).

---

## 1. What Veda does, and why it cannot run here

[Veda](https://arxiv.org/abs/2605.30325) (ICML 2026, *Scalable Video Diffusion via Distilled
Sparse Attention*) turns theoretical sparsity into wall-clock speed with a **tile-skipping
hardware kernel**, implemented with **ThunderKittens DSL and Hopper TMA** — producer warps fetch
selected key/value tiles non-contiguously into shared memory while consumer warps run Tensor Core
math. That mechanism is Hopper-specific. On an Intel Arc GPU there is no port of it, and no
drop-in substitute.

So the question we measured is not "is Veda right" but **"is the missing Intel support a physical
limit or an ecosystem choice?"** Our result: **an ecosystem/backend choice.** A Triton kernel on
the Intel backend can express the same algorithm and is numerically correct. It is simply not fast
enough per FLOP yet for the sparsity levels a real model actually uses.

## 2. What we built

* `kernel/vsa_kernel.py` — a single Triton kernel (online-softmax, per query block loads only its
  own selected key tiles, with a pad-column mask). No tile union, no materialized score matrix.
  This is the file that ships inside a ComfyUI node and that every number below was measured on
  (`sha256[:16] = 8b85c9c6b1c62f0e`).
* `bench/` — the reproducible measurement scripts, including a Windows/MSVC launcher.
* `docs/04-windows-toolchain.md` — three concrete workarounds needed to compile *any* Triton
  kernel on Windows Intel XPU today. These are upstream-fixable.

## 3. Per-layer latency at H3's real geometry

Geometry: `heads=56, head_dim=128, BLOCK=64, 50 DiT layers`, bf16.
Baseline is the vendor dense `scaled_dot_product_attention` at the same shape.
All figures device-synchronized, median of 5, warm-up enforced (see the note on method).

| tokens S | frame config | dense vendor SDPA | kept tiles | this kernel | speedup | kernel TFLOP/s |
|---:|---|---:|---:|---:|---:|---:|
| 21952 | 0.5 MP, 928x512, 124f | 143.0 ms | 7.6% | **67.0 ms** | **2.13x** | 15.6 |
| 21952 | | | 12.5% | 127.9 ms | 1.12x | 13.5 |
| 21952 | | | 17.5% | 193.0 ms | 0.74x | 12.5 |
| 21952 | | | 22.2% | 253.7 ms | 0.56x | 12.1 |
| 44736 | 1.03 MP, 1344x768, 124f | 603.7 ms | 6.2% | **227.6 ms** | **2.65x** | 15.5 |
| 44736 | | | 8.6% | 388.4 ms | 1.55x | 12.7 |
| 44736 | | | 10.9% | 587.3 ms | 1.03x | 10.6 |
| 44736 | | | 14.3% | 905.4 ms | 0.67x | 9.1 |

The eager-torch fallback that this kernel replaces (gather selected tiles, then SDPA) costs
**615.6 ms** at S=21952 and **2158.5 ms** at S=44736 — the kernel is **5.3x / 4.1x** faster than it.
So within the sparse design space the kernel is a large, unambiguous win. The problem is that the
sparse design space itself has to beat *dense*, and at realistic keep ratios it barely does.

**Numerics** (kernel vs. per-query-block SDPA, the exact target semantics): max relative error
`6.0e-03` (bf16), `8.9e-04` (fp32), minimum cosine similarity `0.999999`. The kernel is correct.

## 4. Where the win goes: the backend's per-FLOP rate

| measurement | result |
|---|---|
| vendor dense SDPA, attention shape | **96.6 TFLOP/s** |
| this kernel, attention shape | **13–17 TFLOP/s** (≈ 1/6) |
| vendor GEMM 4096³ bf16 | **168.6 TFLOP/s** |
| best Triton GEMM (BM128/BN128/BK64) | **51.0 TFLOP/s** (30%) |
| Triton GEMM BM128/BN256/BK64 | **3.4 TFLOP/s** (2%) — 15x cliff from tile shape alone |
| vendor *batched* GEMM, M=64 | 20.1 TFLOP/s (vs 52.6 at M=1024) |

Two consequences:

1. **Break-even is analytic.** The kernel wins iff `keep < 13.5/96.5 ≈ 14%` (matches the measured
   crossover between 12.5% and 17.5%). H3's real selection keeps 10.9–12.5%. No amount of kernel
   tuning changes this — only a better backend per-FLOP rate does.
2. **Sparse attention is intrinsically small-M.** One query block is 64 rows. The vendor library's
   *own* batched GEMM only reaches 20 TFLOP/s at M=64. This is the structural reason sparse
   attention is hard, and it is not unique to Triton.

## 5. End-to-end, in ComfyUI

MiniMax H3, 4-step turbo LoRA, per variant the second shot (steady state), ComfyUI 0.34.0:

| config | dense | VSA + this kernel | VSA + eager-torch path |
|---|---:|---:|---:|
| 0.5 MP, 928x512, 124f | 51.40 s | 57.42 s (**+11.7%**) | 103.08 s (+101%) |
| 1.03 MP, 1344x768, 124f | 161.15 s | 155.68 s (**−3.4%**) | not run |

Read this as: at 0.5 MP the sparse pipeline's own selection machinery costs more than the kernel
saves; at 1.03 MP the kernel's saving is finally visible. Neither is a product-level win. This is
the correct place to be honest rather than to quote the 2.1x per-layer number.

## 6. The correction we had to make

The first version of this report claimed attention was **23%** of the step and therefore Amdahl
capped any gain at ~1.2x. That came from the integration node's own per-layer timer, which wraps
the attention call in `perf_counter()` **without a device synchronize**
(`vsa_core.py:276-281`). XPU work is asynchronous, so on a fully-resident model it measures
launch time, not execution: it reported **42 ms/layer** where the synchronized measurement is
**143 ms/layer** — a 3.4x under-report. (It was accidentally accurate only in the lowvram
configuration, where weight offloading forces synchronization: 567 vs 601 ms/layer.)

With synchronized timing, attention is **81.5%** of the step at 0.5 MP and **90.7%** at 1.03 MP.
This flips the conclusion: sparse attention is the right lever for H3, and the reason it is not
paying off is the backend's per-FLOP rate, not a small attention share.

## 7. Reproduce

```
# any Triton-compiling path needs MSVC on Windows; see docs/04
set OMNIXPU_ENABLE=1
set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

python_embeded\python.exe bench\verify_bench.py    # numerics + per-layer latency
python_embeded\python.exe bench\ksel_sweep.py      # speedup vs keep ratio  (headline table)
python_embeded\python.exe bench\sweep2.py          # warps x BLOCK, mask cost, GEMM ceiling
```

Raw output for every table above is in `results/`.

## 8. What we are asking Intel

See [docs/05-request-to-intel.md](docs/05-request-to-intel.md). In one line: **the Triton Intel
backend runs `tl.dot` at ~1/6 (attention) to ~1/3 (GEMM) of the vendor library's rate on Arc Pro
B70, and a tiling change alone can drop it 15x.** Closing that gap would make sparse attention a
real end-to-end win on this class of hardware — which matters precisely because these cards have
32 GB of VRAM and a modest compute budget, i.e. exactly the trade-off where sparsity pays.

## Questions

Questions, repro mismatches, or tests you'd like to see -> open an [issue](https://github.com/lygsam75-buddy/xpu-tile-skip-attention/issues/new).

## Provenance and license

Kernel and benchmark code here were written for this investigation and are licensed
**GPL-3.0**. Algorithmic semantics (sink prefix, diagonal tiles, per-block top-k, coarse gate
branch) follow the upstream ComfyUI VSA sparse-attention nodes; Veda is cited as the design being
adapted. See [PROVENANCE.md](PROVENANCE.md).
