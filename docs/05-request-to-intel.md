# 05 — Request to Intel: the Triton Intel backend is the blocker

## The ask, in one line

On Intel Arc Pro B70, the Triton Intel backend executes `tl.dot` at **~1/6 of the vendor library's
rate on attention shapes** and **~1/3 on GEMM**, and a tile-shape change alone drops it **15x**.
Because that is the only DSL available for building a custom sparse-attention kernel on this
hardware, an algorithm whose entire premise is "skip 90% of the work" cannot win: it needs a
per-FLOP rate better than 1/6 just to break even, and at realistic sparsity it lands on the
break-even line. **Please help close this gap.**

Everything below is reproducible in a few minutes on a single Arc card, with no model weights
required.

## System in question

| item | value |
|---|---|
| GPU | Intel Arc Pro B70 32 GB, driver 32.0.101.8976 |
| OS | Windows 11 Pro |
| torch | 2.12.0+xpu (XPU runtime 20250302) |
| Triton | 3.7.1, Intel backend |
| Workload | MiniMax H3 DiT attention: `heads=56, head_dim=128, BLOCK=64`, bf16 |

## Evidence 1 — attention per-FLOP rate

Same shape, same dtype, per layer:

| implementation | ms/layer (S=21952) | TFLOP/s |
|---|---:|---:|
| vendor `scaled_dot_product_attention` (dense) | 143.0 | **96.6** |
| our Triton tile-skipping kernel (keep 12.5%) | 127.9 | **13.5** |
| our Triton kernel (keep 7.6%) | 67.0 | 15.6 |

Consequence: the kernel wins only when fewer than **~14%** of key tiles are kept. The model we are
serving keeps 10.9–12.5%. Measured speedup vs dense: **1.03x–2.13x, i.e. essentially
break-even at the production setting**.

**Question 1.** Is `tl.dot` emitting XMX/DPAS for these shapes (`M=64, N=64, K=128`, bf16, plus the
second `tl.dot` with `M=64, N=128, K=64`)? Our data suggest no: 13–17 TFLOP/s is consistent with
fp16/bf16 SIMD `fma` rather than matrix hardware, and the kernel's throughput is invariant to
`num_stages`, dtype changes, and `tl.trans` removal — all of which would matter for a memory-bound
kernel but not for one that is simply issuing the wrong instructions. If XMX is not being selected,
what is the selection rule, and what shape or annotation would trigger it?

## Evidence 2 — GEMM tile-shape cliff

4096³ bf16:

| kernel | ms | TFLOP/s | vs vendor |
|---|---:|---:|---:|
| `torch.matmul` (vendor) | 0.82 | 168.6 | 100% |
| Triton, `BM128 / BN128 / BK64` | 2.70 | 51.0 | 30% |
| Triton, `BM128 / BN128 / BK32` | 3.14 | 43.8 | 26% |
| Triton, `BM64 / BN64 / BK64` | 4.20 | 32.7 | 19% |
| Triton, `BM128 / BN256 / BK64` | 40.80 | **3.4** | **2%** |

**Question 2.** A **15x** swing from block shape alone, with no warning and no performance model,
is a reproducibility hazard. Is `BM128/BN256/BK64` a known-bad configuration, and is there a
documented set of supported tile shapes for the Xe backend? A tuning table would be enough.

## Evidence 3 — small-M is the structural problem

The vendor *batched* GEMM (`bmm`, batch=343, K=128, N=1088), swept over `M`:

| M | TFLOP/s |
|---:|---:|
| 64 | 20.1 |
| 128 | 30.1 |
| 256 | 39.3 |
| 512 | 47.2 |
| 1024 | 52.6 |

Sparse attention is intrinsically small-M — one query block is 64 rows. At `M=64` even the vendor
library reaches only 20.1 TFLOP/s, 2.6x below its own `M=1024` figure.

**Question 3.** What is the recommended strategy for small-M batched GEMM on Xe? Specifically:
should a tile-skipping kernel pack multiple query blocks into one GEMM (trading register pressure
for `M`), or is there a preferred gather-and-attend primitive? Context: our kernel's 18.3 TFLOP/s
at low keep is close to the vendor's own `M=64` batched-GEMM ceiling, so we do not believe our
kernel is badly written — we believe the shape is hostile to every path available to us.

## Evidence 4 — we cannot ship the kernel to users on Windows

Three independent blockers prevent compiling *any* Triton kernel on a normal Windows + Intel XPU
install. Details in [04-windows-toolchain.md](04-windows-toolchain.md). The one that is a clean
upstream bug:

**`triton/runtime/build.py:96-101`**

```python
else:
    if os.name != "nt":
        ccflags += ["--std=c++17"]                       # correctly skipped on Windows
    if os.environ.get("TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS", "1") == "1":
        ccflags += ["-Wno-deprecated-declarations"]      # GCC flag, injected anyway
    if os.environ.get("TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING", "1") == "1":
        ccflags += ["-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING"]   # and again
```

`cl.exe` treats both flags as input files and fails with exit status 2. The environment-variable
defaults are `"1"`, so the flags are injected by default.

**Question 4.** Can the two injections be guarded by `if os.name != "nt"`, matching the
`--std=c++17` guard three lines above? Workaround for anyone reading this today:
`TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS=0` and
`TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=0`.

**Question 5.** Separately: `vcvars64.bat` detects the Windows SDK by calling `reg.exe`. When
`reg.exe` is unavailable, `WindowsSdkDir` is left empty and `cl` cannot find `<stdio.h>`. Could the
detection fall back to scanning `%ProgramFiles(x86)%\Windows Kits\10\Include`?

## Two correctness hazards worth flagging

**`omni_xpu_kernel.cute.sol_attn_v2`.** `is_available()` returns `True`, and it reports a **3.96x**
speedup — while returning **all zeros** (cosine similarity `0.0000` against the dense reference).
A fast path that is silently wrong is more dangerous than no fast path: it will be selected by
anyone who checks availability and speed, and it will corrupt output silently. We only caught it
because we sanity-check throughput against a cosine check. Worth an assertion upstream.

**`torch.compile` + `flex_attention` + `BlockMask` on XPU.** Segfaults reproducibly at sequence
length 2048 and 8192 with our H3 geometry. We could not narrow it further because it takes the
process down without a Python traceback. A minimal failing case is in the repository history if
useful.

## How we would verify a fix

Any change in the above should move these three numbers, which we will re-measure and publish:

1. Triton GEMM 4096³ bf16 at `BM128/BN128/BK64`: currently **51.0 TFLOP/s** (target: closer to
   168.6).
2. `BM128/BN256/BK64`: currently **3.4 TFLOP/s** (target: same ballpark as its neighbours).
3. Our attention kernel at keep 12.5%, S=21952: currently **127.9 ms/layer** against a dense
   baseline of **143.0 ms**. If the backend reaches even half the vendor SDPA rate, this becomes a
   ~2.5x win at the production setting instead of 1.12x — and H3 is **81.5%–90.7% attention-bound**
   per step, so that translates directly into end-to-end time.

## Reproduce

```
set OMNIXPU_ENABLE=1
set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

python_embeded\python.exe bench\verify_bench.py    # attention numerics + per-layer latency
python_embeded\python.exe bench\ksel_sweep.py      # speedup vs keep ratio
python_embeded\python.exe bench\sweep2.py          # warps x BLOCK, mask cost, GEMM ceiling
python_embeded\python.exe bench\run_with_msvc.py bench\diag_msvc.py   # toolchain diagnostics
```

No model weights, no ComfyUI and no GPU-specific setup beyond torch-xpu are needed for the
measurements in this document.
