# 02 — The adaptation plan: Veda's method under our actual constraints

Veda assumes a training cluster and an NVIDIA Hopper GPU. We have neither. This document records
what we changed, what it cost, and — equally important — the dead ends, so that nobody (including
Intel) has to re-walk them.

## 1. The hard constraints

These are not preferences. They are the conditions under which the result has to hold.

| # | constraint | consequence |
|---|---|---|
| C1 | **32 GB system RAM.** 90% of the target user base runs 8 GB VRAM + 32 GB RAM. Memory cannot be "just increased" — many machines have only two DIMM slots. | The page file is the only expandable memory. **No solution may require more RAM.** Any plan of the form "add memory" is invalid and is not proposed |
| C2 | **31.16 GiB VRAM visible** (Arc Pro B70 with ECC off; 27.25 GiB with the factory-default ECC on). | A 22 GB DiT plus the gate tensors does not fit. The pipeline is already in a lowvram/offload regime at 1.03 MP |
| C3 | **Windows 11 + ComfyUI 0.34.0 + torch 2.12.0+xpu.** The artefact must be a droppable custom node. | No build step, no training step, no changes to model weights, no changes to the sampling schedule |
| C4 | **No ThunderKittens, no TMA, no CUDA.** Triton via the Intel backend is the only portable DSL. | The single largest efficiency factor in Veda is structurally unavailable |
| C5 | **The card's price band.** It is deliberately positioned far below the NVIDIA equivalents, so a solution that requires different hardware is not a solution. | The plan has to extract value from the backend that exists |
| C6 | **Output semantics must not change.** | Every path must be numerically verifiable against a reference, and must fall back safely |

## 2. What we changed, and what it cost

| # | Veda does | we did instead | measured cost |
|---|---|---|---|
| A1 | ThunderKittens + Hopper TMA producer/consumer kernel, ~80% of FA-3 MFU | Triton kernel, per query block loads its own tiles, online softmax | **13–17 TFLOP/s vs the vendor dense SDPA's 96.6.** This is *the* cost, and the reason the headline speedup collapses from 10.5x to 1.1x |
| A2 | Distilled, statistics-aware tile scoring | Reuse the model's own `to_gate_compress` gate for the coarse branch; runtime 64-token-tile top-k for the fine branch | The mask-quality axis Veda optimises is unavailable. We did not measure what quality we left on the table — stated as a gap, not claimed as a loss |
| A3 | Per-layer, per-head tiling search | Fixed `4x4x4` cube reordering, 64-row tiles, prefix rows padded to the front, selection shared across all 56 heads | Not measured end to end. Selecting per head would need `K_SEL` to become a per-head tensor and 56 distinct index streams |
| A4 | — | Pad-column mask inside the kernel (`-inf` for rows beyond `tile_klen`) | **5.2%** of kernel time (104.8 ms vs 99.3 ms). Kept: it is what makes prefix-tile padding correct |
| A5 | — | `BLOCK=64, num_warps=8` as the shipped default | Measured optimal. With keep fixed at 12.5%: `w=4` is **7.3x slower** (748.8 ms), `w=16` 1.5x slower, `w=32` 2.7x slower. Larger `BLOCK` at the same keep% performs 2–4x the redundant FLOPs and is still slower in wall clock |
| A6 | — | Kernel path with an eager-torch gather fallback | Fallback is **5.3x slower** but correct — the right trade for a user-facing node |
| A7 | — | `_ensure_triton_env()` inside the kernel module: it inherits `vcvars64` and patches the Windows SDK itself | Zero configuration for the user; without it, the node cannot compile on any Windows machine in a normal (non-developer) process. See [04](04-windows-toolchain.md) |
| A8 | — | Gate weights INT8 tensorwise-quantised, streamed CPU -> GPU layer by layer | Required by C1/C2. Not separately benchmarked |

## 3. Priority plan

**P0 — done.** Correct kernel, verified numerics, integrated into the node behind a `use_kernel`
switch, end-to-end A/B in both resolutions, old path retained as fallback.

**P1 — highest expected value, and it is not ours to do.** Close the backend per-FLOP gap.
Receipt: a kernel that runs at 1/6 of the vendor's rate cannot win at realistic keep ratios.
Everything downstream is blocked on this. See [05-request-to-intel.md](05-request-to-intel.md).

**P2 — deferred, gated on P1.** Push sparsity below the upstream default keep (10.9–12.5%).
At the current rate, more sparsity is the *only* lever that wins (2.13x already at 7.6% keep), so
this is attractive — but we have no quality metric for H3 audio+video output that we trust, and we
will not trade unmeasured quality for speed in a user-facing node.

**P3 — explicitly rejected, measured.** Coarser tiles to raise the GEMM `M` dimension. At equal
keep%, `BLOCK=128` performs 2x and `BLOCK=256` 4x the redundant FLOPs; both are slower in wall
clock than `BLOCK=64`. Raises the sparsity ratio for the same quality, but the arithmetic does not
pay.

## 4. Dead ends (measured — do not re-walk)

| path | result | evidence |
|---|---|---|
| Materialise the sparse score matrix, then use vendor batched GEMM (`einsum`/`bmm`) to avoid hand-written Triton entirely | **8.5 TFLOP/s**, slower than the kernel's 13.5 | `results/t11_m_scan.txt` |
| Add an attention mask to make the eager path's top-k per-block exact (fix the tile union) | Slower at **every** chunk size — SDPA drops off the flash path. chunk=8 union 406.5 ms -> chunk=8 masked 534.0 ms | `REPORT_xpu_tile_skip.md` §4 |
| Tune `chunk_qblocks` | At most 17% (406 -> 348 ms) and it adds launch overhead | same |
| `num_stages` 1–4, dtype fp16<->bf16, removing `tl.trans` | **Zero effect.** The platform is hard; `num_warps` is the only lever | `REPORT_xpu_tile_skip.md` §3.2 |
| `torch.compile` + `flex_attention` + `BlockMask` on XPU | **Crashes** (segfault at L=2048 and L=8192) | 2026-09-22 session |
| `omni_xpu_kernel.cute.sol_attn_v2` (a prebuilt XPU sparse-attention sidecar) | `is_available()` returns True and it reports **3.96x** — but the output is **all zeros** (cosine 0.0000). Broken build; unusable | 2026-09-22 session |

The last two are worth flagging to Intel directly: a shipped XPU attention sidecar that silently
returns zeros while reporting a speedup is worse than no sidecar.

## 5. Measurement discipline we had to impose

Three traps we hit in this investigation. All are documented because they changed conclusions.

1. **A per-layer timer without a device synchronize under-reports by 3.4x on XPU.** This flipped
   our own bottleneck conclusion. Every figure in this repository is device-synchronized.
2. **Cold first sweep rows read ~40% slow.** The first `(BLOCK, warps)` combination measured
   ~40% slower than the same combination measured later in the same process. Warm-up is now
   explicit and every sweep is run twice; see `bench/sweep2.py` round `r1`/`r2`.
3. **A per-FLOP rate on synthetic data is not the model's behaviour.** A standalone kernel
   benchmark at 2.13x became +11.7% wall-clock *slower* end to end, because the sparse pipeline's
   own selection machinery is not free. Both numbers are reported; neither substitutes for the
   other.
