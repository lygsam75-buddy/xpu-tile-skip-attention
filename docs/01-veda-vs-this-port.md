# 01 — Veda vs. this port: item-by-item

Veda is a *training + kernel* framework. This repository is a *kernel-only* port of the inference
half, onto a backend Veda does not target. That framing explains almost every deviation below.

Reference: *Veda: Scalable Video Diffusion via Distilled Sparse Attention*, ICML 2026,
[arXiv:2605.30325](https://arxiv.org/abs/2605.30325).

## 1. Side-by-side

| axis | Veda (paper) | This port (Intel Arc Pro B70) | why it differs |
|---|---|---|---|
| **Tile scoring** | Learned estimator distilled from a full-attention backbone; "statistics-aware" (extrema-aware) pooling instead of plain mean pooling | Uses the model's **existing** learned gate (`to_gate_compress`) for the coarse branch, plus 64-token-tile top-k computed at runtime | We cannot retrain or distill. Nothing in the pipeline may change model weights. The mask-quality axis Veda optimises is simply unavailable to us |
| **Tiling** | Per-layer, per-head factorisation `(p_t,p_h,p_w)` searched offline over the hardware tile budget; space-biased heads get spatial tiles, time-biased heads temporal tiles (+7.2% motion, +9.6% overall vs static) | Fixed `4x4x4` cube reordering for the video segment, 64-row tiles, prefix (text/ref/audio) rows zero-padded and moved to the front; selection shared across heads | The upstream ComfyUI VSA node fixes this layout, and the model's real geometry is 56 heads x 128 dim — retiling per head is a substantial rewrite with no measured endpoint |
| **Sparsity** | 87.5–95% on Waver-1B; 90% on Wan2.1-1.3B | keep 10.9–12.5%, i.e. **87.5–89.1% sparse** | Comparable. H3 is a 56-head audio+video DiT; we kept the upstream heuristic and did not push keep lower, because output quality at higher sparsity was not validated on this box |
| **Kernel** | ThunderKittens DSL + Hopper TMA. Producer warps pull selected KV tiles non-contiguously into SMEM; consumer warps run Tensor Cores. Reaches ~80% of FlashAttention-3 MFU | Triton kernel on the Intel backend. Per query block, loads only its own selected tiles and runs online softmax. No materialised score matrix, no tile union. Reaches **13–17 TFLOP/s** vs the vendor SDPA's **96.6 TFLOP/s** — i.e. ~**1/6** | ThunderKittens and TMA do not exist on this hardware. Triton is the only portable DSL available, and it is the ceiling we hit |
| **Self-attention speedup** | **10.5x** (Waver-T2V-12B, 720P, 10 s); 7.08x (Wan2.1-14B, 720P, 5 s) | **1.03–1.12x** at the model's real keep ratio; 2.13–2.65x only if keep is pushed to ~6–8% | See the arithmetic in section 2 |
| **End-to-end speedup** | **5.1x** (Waver-12B); 2.63x (Wan2.1-14B) | **−3.4% to +11.7%** | Same cause |
| **Attention share of step** | 92% before, 50% after | 81.5% (0.5 MP) to 90.7% (1.03 MP) — measured, synchronized | The two models live in the same regime: large-video attention really is the bottleneck |

## 2. The gap is entirely kernel efficiency, not sparsity

Both implementations reach roughly 88–90% sparsity. Take the ratio of achieved speedup to
theoretically available speedup in each case:

| | kept | kernel rate vs dense baseline | implied attention time | reported speedup |
|---|---:|---:|---:|---:|
| Veda (Waver-12B) | ~10% | **~0.80** of FlashAttention-3 MFU | 0.10 / 0.80 = 0.125 | **10.5x** |
| This port (H3, 0.5 MP) | 12.5% | **~0.14** of vendor dense SDPA | 0.125 / 0.14 = 0.89 | **1.12x** |

The sparsity terms are nearly identical. The entire remaining factor — roughly **5.7x** — is the
per-FLOP rate of the kernel backend. That is the finding worth acting on, and it is why
[docs/05](05-request-to-intel.md) asks about the backend rather than about the algorithm.

Corollary: this is **not** an argument that Veda's method fails on Intel hardware. It is an
argument that a 0.8-MFU kernel on a 1/6-rate backend cannot pay for itself, and that the algorithm
is currently being measured through the backend rather than on its own merits.

## 3. What transferred cleanly

* **The tile-skipping formulation itself.** Per query block, gather only selected KV tiles; online
  softmax over just those; no materialisation. Expressible in Triton, numerically exact against a
  per-block SDPA reference (max rel. err `6.0e-3` bf16, `8.9e-4` fp32, min cosine `0.999999`).
* **Per-block selection kills the tile-union problem.** The eager path shares one union of selected
  tiles across a chunk of query blocks, which at `chunk=8` consumes 41.9% of key tiles where the
  design intends 12.5%. Because the kernel gives each query block exactly its own top-k, the union
  issue disappears by construction — no attention mask needed.
* **The break-even framing.** Veda reports self-attention overhead dropping 92% -> 50%. Our
  measurement of 81.5–90.7% attention share independently confirms that this is the right thing to
  attack on a video DiT.

## 4. What did not transfer

* **ThunderKittens / TMA producer-consumer pipelining.** No counterpart on XPU. This is the
  single largest unexplained factor in the efficiency gap and is exactly what we are asking Intel
  about.
* **Distilled tile scoring.** Requires training. Not available to a deployed ComfyUI node.
* **Head-aware tiling.** Requires an offline per-layer, per-head search over a calibration set —
  and a way to express 56 different tilings in one kernel launch. Not attempted.
* **Aggressive sparsity (90–95%).** We validated correctness, not quality, at high sparsity. We did
  not have a quality metric for H3 audio+video outputs that we trusted, so we did not push keep
  below the upstream default. This is a stated gap, not a negative result.

## 5. Honest statement of scope

Anything in this repository is about **inference-side kernel behaviour on Intel Arc**. It does not
evaluate Veda's distillation quality, its head-aware tiling search, or its published end-to-end
numbers. Those are a different experiment on different hardware.
