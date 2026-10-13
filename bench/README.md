# Benchmarks

All scripts import the kernel from `../kernel/` by default, so a clone is self-contained. Set
`VSA_NODE_DIR` to point at a different folder containing `vsa_kernel.py` (e.g. a real ComfyUI node
install) to measure that copy instead.

```
set OMNIXPU_ENABLE=1
set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

| script | what it measures | runtime |
|---|---|---|
| `verify_bench.py` | kernel numerics vs a per-query-block SDPA reference (bf16 and fp32), and per-layer latency for dense SDPA / eager gather / this kernel at both real H3 shapes | ~3 min |
| `ksel_sweep.py` | speedup vs. keep ratio, through the shipped entry point, with enforced warm-up. **This is the source of the headline table** | ~4 min |
| `sweep2.py` | `num_warps` x `BLOCK` with keep held constant (two rounds), cost of the pad mask, and the GEMM ceiling vs the vendor library | ~6 min |
| `run_with_msvc.py` | wrapper that executes a target script inside an inherited `vcvars64` environment | — |
| `diag_msvc.py` | compiler diagnostics; decodes `cl.exe` output as GBK so the real error is visible | — |

## Two things to know before reading any number

**Device synchronization is mandatory.** `torch.xpu` work is asynchronous. A timer placed around an
attention call without `torch.xpu.synchronize()` measures enqueue time and under-reported by
**3.4x** in our integration path (42 ms/layer reported vs 143 ms/layer actual). Every measurement
here synchronizes.

**Warm up before the first row.** In an early version of the sweep, the first `(BLOCK, num_warps)`
combination read ~40% slower than the identical combination measured later in the same process. All
sweeps now warm up explicitly and are run twice (`r1`/`r2` in the output), and the two rounds agree
to within 0.2%.

## Usage

```bash
# from the repository root
python bench/verify_bench.py
python bench/ksel_sweep.py
python bench/sweep2.py

# compiler diagnostics (wrap the target so it inherits vcvars64)
python bench/run_with_msvc.py bench/diag_msvc.py
```

A `python_embeded` interpreter from a ComfyUI install works; any Python 3.10+ with `torch` (XPU)
and `triton` 3.7+ is fine.
