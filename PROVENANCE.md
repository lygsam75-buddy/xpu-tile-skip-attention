# Provenance

## What is original here

| path | origin |
|---|---|
| `kernel/vsa_kernel.py` | Written for this investigation. The Triton kernel, the toolchain bootstrap (`_ensure_triton_env`, `_patch_sdk`) and the host-side wrapper are all ours |
| `bench/*.py` | Written for this investigation, except `run_with_msvc.py` and `diag_msvc.py`, which were also written here and are included because they are the reference implementation of the Windows workarounds |
| `docs/*` | Written for this investigation |
| `results/*` | Raw captured output from this machine, as committed |

No third-party source code is vendored in this repository.

## What this work is derived from (cited, not vendored)

* **Veda** — *Scalable Video Diffusion via Distilled Sparse Attention*, ICML 2026,
  [arXiv:2605.30325](https://arxiv.org/abs/2605.30325). The design being adapted: tile-skipping
  sparse attention with a hardware kernel. Its kernel is ThunderKittens + Hopper TMA and is not
  included or ported here line by line; what is ported is the *shape* of the idea (per query block,
  load only selected KV tiles, online softmax over just those).
* **Upstream ComfyUI VSA sparse attention** and **`sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA`**
  (GPL-3.0). The algorithmic semantics our kernel implements are aligned with these node
  implementations: sink/prefix tiles forced exact, diagonal +/-1 tiles forced exact, per-block
  top-k, softmax (tail=False) over the exact tiles only, and a coarse gate branch added back to the
  output. This alignment is a semantic interface, not copied code.
* **`hao-ai-lab/FastVideo`, `NVlabs/Sana`, `ModelTC/lightx2v`, `OpenVDN/vdn-minimax-h3`** — read for
  context on how H3 acceleration is approached elsewhere. Not used.
* **MiniMax H3** and **ComfyUI** are not included. No model weights are distributed here. The
  `0.5MP` / `1.03MP` labels in the benchmarks refer to H3's real attention geometry (`heads=56`,
  `head_dim=128`, `50` DiT layers) as reported by the model at runtime.

## License

**GPL-3.0.** The semantics of this kernel are aligned with upstream GPL-3.0 ComfyUI node code, and
the work is intended to be droppable into that ecosystem, so GPL-3.0 is the consistent choice.

## Hardware and software

All figures were produced on a single machine:

```
Intel Arc Pro B70 32 GB (ECC off -> 31.16 GiB), driver 32.0.101.8976
Windows 11 Pro, 32 GB system RAM
torch 2.12.0+xpu (XPU runtime 20250302), triton 3.7.1 (Intel backend), Python 3.13.12
ComfyUI 0.34.0
```

Kernel source measured: `kernel/vsa_kernel.py`,
`sha256 = 8b85c9c6b1c62f0e6bf57fb9265635cd73cdb70f33080fe17aacf2a20030efe2`.

Measurement scope: **inference-side kernel behaviour on Intel Arc only**. Nothing here evaluates
Veda's distillation quality, its head-aware tiling search, or its published end-to-end numbers.
