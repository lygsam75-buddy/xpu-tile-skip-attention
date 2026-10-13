# Raw results

Captured output, committed verbatim. Each file maps to tables in
[../docs/03-benchmarks.md](../docs/03-benchmarks.md).

| file | produced by | feeds |
|---|---|---|
| `verify_bench_out.txt` | `bench/verify_bench.py` | numerics table; dense / gather / kernel per-layer at both shapes |
| `ksel_sweep_out.txt` | `bench/ksel_sweep.py` | **the headline speedup-vs-keep table** |
| `sweep2_out.txt` | `bench/sweep2.py` | `num_warps` x `BLOCK` (two rounds); pad-mask cost; GEMM ceiling vs vendor |
| `t11_m_scan.txt` | vendor batched-GEMM `M` sweep | small-M ceiling; equivalent rates for the candidate paths |
| `e2e_ab_0.5MP.json` | ComfyUI A/B harness | 0.5 MP dense / VSA+kernel / VSA+eager |
| `e2e_ab_1.03MP.json` | ComfyUI A/B harness | 1.03 MP dense / VSA+kernel |

## Notes on interpretation

* Per-layer latency is **device-synchronized median**, not mean. `min`/`max` are printed alongside;
  when a spread is wide, the case is called out in the docs rather than averaged away.
* Two kernel figures exist for the same `K_SEL` at 0.5 MP: **116.5 ms** (single-pass run,
  `verify_bench_out.txt`) and **127.9 ms** (`ksel_sweep_out.txt`, which enforces warm-up and runs
  the full sweep). The documented range is **1.12–1.22x vs dense**. The sweep is the primary source
  because it repeats.
* `TFLOP/s` in the per-layer tables is computed on **each path's own FLOP count**, so the dense and
  sparse columns are deliberately not comparable with each other. The comparable quantity is
  milliseconds of wall clock, and the ratio printed below each table.
* The ComfyUI A/B JSONs use `steady_*` fields, i.e. the second shot with the model already
  resident. First-shot figures include model load and are not used.
