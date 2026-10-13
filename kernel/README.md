# The kernel

`vsa_kernel.py` — single-file, no build step. It is self-contained: import it, call
`vsa_fine_attention()`, and it bootstraps its own compile environment on first call.

## Interface

```python
out = vsa_fine_attention(q, k, v, sel_idx, klen, scale)
```

| arg | shape | meaning |
|---|---|---|
| `q`, `k`, `v` | `[S, H, D]` | model layout (`[seq, heads, head_dim]`). `S = MT * BLOCK` |
| `sel_idx` | `[MT, K_SEL]` int64 | for each query block, the key tile indices it keeps. **Shared across heads** |
| `klen` | `[MT]` int32 | live rows in each key tile (prefix tiles are zero-padded) |
| `scale` | float | softmax scale, normally `D ** -0.5` |
| returns | `[H, MT, BLOCK, D]` | same layout as the eager-torch path it replaces |

## What it does per query block

1. Load the query block (`BLOCK` rows) once.
2. Loop over that block's `K_SEL` selected key tiles only. Never touches an unselected tile, and
   never materialises the full key or the full score matrix.
3. Per tile: `tl.dot(q, tl.trans(k))`, apply the pad-column mask (`-inf` past `klen`), update the
   online-softmax running max/sum, and accumulate `p @ v`.
4. Normalise and store.

Because each query block consumes exactly its own top-k, the **tile-union problem does not exist
here**. The eager path shares one union of tiles across a chunk of query blocks, which at
`chunk=8` consumes 41.9% of key tiles where the design intends 12.5%. No attention mask is needed
to fix that; it is correct by construction.

## Environment requirements

* torch with XPU support, and triton with the Intel backend
* first call on Windows additionally needs MSVC (`cl.exe`). `_ensure_triton_env()` inherits
  `vcvars64` and patches the Windows SDK paths itself. Override the `vcvars64.bat` location with
  `VSA_VCVARS64` if Visual Studio lives somewhere unusual. See
  [../docs/04-windows-toolchain.md](../docs/04-windows-toolchain.md)
* recommended: `OMNIXPU_ENABLE=1`, `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`

## Measured behaviour

Correctness: max relative error `6.0e-3` (bf16), `8.9e-4` (fp32), min cosine `0.999999`, against a
per-query-block SDPA reference.

Throughput, and the honest limits, are in [../docs/03-benchmarks.md](../docs/03-benchmarks.md).
Headline: **5.3x faster than the eager-torch path it replaces, but only 1.03–2.13x faster than
vendor dense SDPA depending on how many tiles are kept**, because the Triton Intel backend runs
`tl.dot` at roughly 1/6 of the vendor library's rate on this shape.

Launch configuration that ships: `BLOCK=64`, `num_warps=8`. Measured optimal — `num_warps=4` is
7.3x slower and `num_warps=16` 1.5x slower, with `num_stages`, dtype and `tl.trans` having no
measurable effect.
