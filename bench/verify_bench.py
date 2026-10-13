# -*- coding: utf-8 -*-
"""Verify + benchmark the shipped XPU tile-skipping attention kernel.

Target of measurement: the kernel that actually ships inside
    ComfyUI/custom_nodes/ComfyUI-H3-VSA-XPU/vsa_kernel.py
(imported, not re-typed), at MiniMax-H3's real attention geometry.

Produces:
  1. Environment dump (device, driver, torch/triton versions)
  2. SHA256 of the measured kernel module
  3. Numerics: kernel vs. per-query-block SDPA reference (semantically exact,
     no tile-union), sampled over query blocks -- bf16 and fp32
  4. Per-layer latency at real shapes: dense SDPA / VSA torch gather / XPU kernel
  5. Sequential-metric cross-check: the numbers must be reproducible, so every
     figure is a median of N timed reps with min/max reported.

Run (bare, no MSVC wrapper needed -- the module bootstraps its own toolchain):
    set OMNIXPU_ENABLE=1
    set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    python_embeded\\python.exe verify_bench.py

Author: 老K (WorkBuddy) for 刚仔哥哥. SPDX-License-Identifier: GPL-3.0
"""
import hashlib
import os
import sys
import time

import torch
import torch.nn.functional as F

NODE_DIR = os.environ.get("VSA_NODE_DIR")
if not NODE_DIR:                      # self-contained: use the kernel shipped in this repo
    NODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernel")
if not os.path.isfile(os.path.join(NODE_DIR, "vsa_kernel.py")):
    raise SystemExit("vsa_kernel.py not found in %s\nset VSA_NODE_DIR to the "
                     "folder that contains it." % NODE_DIR)
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

import vsa_kernel  # noqa: E402  the shipped kernel

H, D, BLOCK = 56, 128, 64          # H3 DiT attention geometry
SCALE = D ** -0.5

# (label, S_padded, n_prefix_tiles) -- taken from the node's own plan logs
SHAPES = [
    ("0.5MP  928x512x124f", 21952, 23),
    ("1.03MP 1344x768x124f", 44736, 39),
]
KEEP = 0.05


def env_dump():
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    props = torch.xpu.get_device_properties(0)
    n_sp = len(props.sub_group_sizes) if hasattr(props, "sub_group_sizes") else "?"
    print("  device            : %s" % torch.xpu.get_device_name(0))
    print("  total memory      : %.2f GiB" % (props.total_memory / 1024 ** 3))
    print("  torch             : %s" % torch.__version__)
    print("  triton            : %s" % __import__("triton").__version__)
    print("  xpu backend       : %s" % torch.version.xpu)
    print("  sub-group sizes   : %s" % n_sp)
    p = os.path.join(NODE_DIR, "vsa_kernel.py")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print("  measured module   : vsa_kernel.py  sha256[:16] = %s" % h)
    print()


def make_sel(MT, n_prefix, K_SEL, device):
    """Realistic per-query-block tile selection, same construction the node's
    integration test uses: shared sink prefix + forced diagonal +-1 + topk fill."""
    g = torch.Generator().manual_seed(0)
    sel = torch.zeros(MT, K_SEL, dtype=torch.int64)
    for m in range(MT):
        s = set(range(n_prefix))                       # shared sink prefix tiles
        for d in (-1, 0, 1):                           # forced diagonal tiles
            if 0 <= m + d < MT:
                s.add(m + d)
        while len(s) < K_SEL:                          # top-k fill
            s.add(int(torch.randint(0, MT, (1,), generator=g)))
        sel[m] = torch.tensor(sorted(s), dtype=torch.int64)
    return sel.to(device)


def klen_full(MT, device, dtype=torch.int32):
    """All tiles full -- synthetic data has no padding rows."""
    return torch.full((MT,), BLOCK, dtype=dtype, device=device)


def ref_per_block(q, k, v, sel, sample):
    """Exact reference: for each sampled query block, SDPA against ONLY the tiles
    it selected. This is the semantics the kernel is supposed to implement
    (per-block top-k, no chunk union)."""
    MT = sel.shape[0]
    kt = k.view(H, MT, BLOCK, D)
    vt = v.view(H, MT, BLOCK, D)
    hrow = torch.arange(H, device=q.device).view(H, 1)
    out = {}
    for m in sample:
        kb = kt[hrow, sel[m].view(1, -1)].reshape(H, sel.shape[1] * BLOCK, D)
        vb = vt[hrow, sel[m].view(1, -1)].reshape(H, sel.shape[1] * BLOCK, D)
        qm = q[:, m * BLOCK:(m + 1) * BLOCK]
        out[m] = F.scaled_dot_product_attention(
            qm.unsqueeze(0), kb.unsqueeze(0), vb.unsqueeze(0), scale=SCALE)[0]
    return out


def numerics(S, n_prefix, dtype, sample_n=12):
    MT = S // BLOCK
    K_SEL = n_prefix + 3 + int(KEEP * MT)
    dev = "xpu"
    torch.manual_seed(0)
    # kernel wants [S, H, D]
    q = torch.randn(S, H, D, device=dev, dtype=dtype)
    k = torch.randn(S, H, D, device=dev, dtype=dtype)
    v = torch.randn(S, H, D, device=dev, dtype=dtype)
    sel = make_sel(MT, n_prefix, K_SEL, dev)

    o = vsa_kernel.vsa_fine_attention(q, k, v, sel, klen_full(MT, dev), SCALE)
    torch.xpu.synchronize()
    o = o.view(H, S, D)

    # reference works on [H, S, D]
    qh, kh, vh = q.permute(1, 0, 2).contiguous(), k.permute(1, 0, 2).contiguous(), \
        v.permute(1, 0, 2).contiguous()
    step = max(1, MT // sample_n)
    sample = list(range(0, MT, step))[:sample_n]
    ref = ref_per_block(qh, kh, vh, sel, sample)

    worst_abs = worst_rel = worst_cos = 0.0
    for m in sample:
        a = o[:, m * BLOCK:(m + 1) * BLOCK].float()
        b = ref[m].float()
        worst_abs = max(worst_abs, (a - b).abs().max().item())
        refmx = b.abs().max().item()
        if refmx > 0:
            worst_rel = max(worst_rel, (a - b).abs().max().item() / refmx)
        cos = F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
        worst_cos = min(worst_cos, cos) if worst_cos else cos
    ver = "OK" if worst_cos > 0.999 else ("SUSPECT" if worst_cos > 0.99 else "BROKEN")
    print("  [%s] S=%-6d K_SEL=%-3d  sampled %d blocks"
          % (dtype, S, K_SEL, len(sample)))
    print("       max_abs=%.3e  max_rel=%.3e  min_cosine=%.6f  %s"
          % (worst_abs, worst_rel, worst_cos, ver))
    return ver


def bench_median(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def gather_path(q, k, v, sel):
    """Exact mirror of the node's torch fallback fine branch at chunk=1:
    per query block, gather its own tiles then run SDPA. Semantically correct,
    but implemented in eager torch (this is the baseline the kernel replaces).
    q,k,v are [S,H,D]; returns [H,S,D]."""
    MT = sel.shape[0]
    kt = k.view(MT, BLOCK, H, D).permute(2, 0, 1, 3).contiguous()   # [H,MT,B,D]
    vt = v.view(MT, BLOCK, H, D).permute(2, 0, 1, 3).contiguous()
    qh = q.permute(1, 0, 2).contiguous()
    hrow = torch.arange(H, device=q.device).view(H, 1)
    out = torch.empty(H, MT * BLOCK, D, device=q.device, dtype=q.dtype)
    for m in range(MT):
        kb = kt[hrow, sel[m].view(1, -1)].reshape(H, sel.shape[1] * BLOCK, D)
        vb = vt[hrow, sel[m].view(1, -1)].reshape(H, sel.shape[1] * BLOCK, D)
        qm = qh[:, m * BLOCK:(m + 1) * BLOCK]
        out[:, m * BLOCK:(m + 1) * BLOCK] = F.scaled_dot_product_attention(
            qm.unsqueeze(0), kb.unsqueeze(0), vb.unsqueeze(0), scale=SCALE)[0]
    return out


def perf(label, S, n_prefix, iters=5):
    MT = S // BLOCK
    K_SEL = n_prefix + 3 + int(KEEP * MT)
    dev = "xpu"
    dt = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(S, H, D, device=dev, dtype=dt)
    k = torch.randn(S, H, D, device=dev, dtype=dt)
    v = torch.randn(S, H, D, device=dev, dtype=dt)
    sel = make_sel(MT, n_prefix, K_SEL, dev)
    kl = klen_full(MT, dev)
    qh, kh, vh = q.permute(1, 0, 2).contiguous(), k.permute(1, 0, 2).contiguous(), \
        v.permute(1, 0, 2).contiguous()

    def dense():
        F.scaled_dot_product_attention(qh.unsqueeze(0), kh.unsqueeze(0),
                                       vh.unsqueeze(0), scale=SCALE)

    def kern():
        vsa_kernel.vsa_fine_attention(q, k, v, sel, kl, SCALE)

    print("\n%s   S=%d  MT=%d  K_SEL=%d  (keep=%.1f%%)" %
          (label, S, MT, K_SEL, 100.0 * K_SEL / MT))
    print("  %-22s %10s %10s %10s %12s" % ("path", "ms/layer", "min", "max", "TFLOP/s*"))
    flops_sparse = 4.0 * H * MT * K_SEL * BLOCK * BLOCK * D
    res = {}
    for name, fn, fl in (("dense SDPA", dense, 4.0 * H * S * S * D),
                         ("torch gather (fallback)", lambda: gather_path(q, k, v, sel),
                          flops_sparse),
                         ("XPU kernel (shipped)", kern, flops_sparse)):
        md, mn, mx = bench_median(fn, iters=iters)
        res[name] = md
        print("  %-22s %10.2f %10.2f %10.2f %12.1f"
              % (name, md * 1e3, mn * 1e3, mx * 1e3, fl / md / 1e12))
    print("  * TFLOP/s computed on each path's OWN FLOP count, so 'dense' and the")
    print("    sparse paths are not directly comparable in that column.")
    print("  >> kernel vs dense      : %.2fx" % (res["dense SDPA"] / res["XPU kernel (shipped)"]))
    print("  >> kernel vs gather     : %.2fx" % (res["torch gather (fallback)"] /
                                                res["XPU kernel (shipped)"]))
    return res


def main():
    env_dump()
    print("=" * 78)
    print("1. NUMERICS  (kernel vs per-query-block SDPA, the exact target semantics)")
    print("=" * 78)
    for S, npf in ((21952, 23), (44736, 39)):
        numerics(S, npf, torch.bfloat16)
    numerics(8192, 8, torch.float32, sample_n=8)

    print()
    print("=" * 78)
    print("2. PER-LAYER LATENCY AT REAL H3 SHAPES")
    print("=" * 78)
    for label, S, npf in SHAPES:
        perf(label, S, npf)

    print("\ndone.")


if __name__ == "__main__":
    main()
