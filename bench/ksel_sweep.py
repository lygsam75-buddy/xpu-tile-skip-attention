# -*- coding: utf-8 -*-
"""K_SEL sweep against the SHIPPED entry point, with proper warm-up.

Why this script exists: the on-disk report states a single headline number
("3.6x vs dense") measured at K_SEL=17. But the node's own plan log shows the
real runtime selection is K_SEL = sink_prefix(23) + diagonal(3) + top-k(17) = 43
at 0.5MP, and 39+3+35 = 76 at 1.03MP -- i.e. 10.9-12.5% keep, not 5%.

This measures the shipped function `vsa_kernel.vsa_fine_attention` across the
whole K_SEL range, so the report can state where the kernel actually wins and
where it loses. Warm-up is explicit because on this platform the first sweep
row reads ~40% slow if measured cold (measured; see sweep2.py).

    set OMNIXPU_ENABLE=1 && set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    python_embeded\\python.exe ksel_sweep.py

SPDX-License-Identifier: GPL-3.0
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bench import H, D, BLOCK, SCALE, make_sel, klen_full  # noqa: E402

NODE_DIR = os.environ.get("VSA_NODE_DIR")
if not NODE_DIR:                      # self-contained: use the kernel shipped in this repo
    NODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernel")
if not os.path.isfile(os.path.join(NODE_DIR, "vsa_kernel.py")):
    raise SystemExit("vsa_kernel.py not found in %s\nset VSA_NODE_DIR to the "
                     "folder that contains it." % NODE_DIR)
sys.path.insert(0, NODE_DIR)
import vsa_kernel  # noqa: E402  the shipped module

SHAPES = [(21952, 23, "0.5MP  928x512x124f"),
          (44736, 39, "1.03MP 1344x768x124f")]
K_SELS = [17, 26, 43, 60, 76, 100]


def bench(fn, iters=5, warmup=2):
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


def main():
    dev, dt = "xpu", torch.bfloat16
    vsa_kernel._ensure_triton_env()
    print("shipped entry point : vsa_kernel.vsa_fine_attention()")
    print("device              : %s" % torch.xpu.get_device_name(0))
    print()

    for S, n_prefix, label in SHAPES:
        MT = S // BLOCK
        torch.manual_seed(0)
        q = torch.randn(S, H, D, device=dev, dtype=dt)
        k = torch.randn(S, H, D, device=dev, dtype=dt)
        v = torch.randn(S, H, D, device=dev, dtype=dt)
        qh = q.permute(1, 0, 2).contiguous()
        kh = k.permute(1, 0, 2).contiguous()
        vh = v.permute(1, 0, 2).contiguous()

        def dense():
            F.scaled_dot_product_attention(qh.unsqueeze(0), kh.unsqueeze(0),
                                           vh.unsqueeze(0), scale=SCALE)

        for _ in range(2):
            dense()
        torch.xpu.synchronize()
        md, dmin, dmax = bench(dense, iters=7)
        print("=" * 84)
        print("%s   S=%d  MT=%d  BLOCK=%d" % (label, S, MT, BLOCK))
        print("=" * 84)
        print("  dense SDPA (vendor)  %8.2f ms  [%.2f .. %.2f]  %6.1f TFLOP/s (dense FLOPs)"
              % (md * 1e3, dmin * 1e3, dmax * 1e3, 4.0 * H * S * S * D / md / 1e12))
        print()
        print("  %-7s %-7s %10s %10s %11s %11s"
              % ("K_SEL", "keep%", "ms/layer", "TFLOP/s", "vs dense", "vs gather"))
        print("  " + "-" * 66)

        # warm every K_SEL in this shape before measuring any of them
        sels = {}
        for ks in K_SELS:
            if ks - 3 < n_prefix:
                continue
            sels[ks] = make_sel(MT, n_prefix, ks, dev)
        kl = klen_full(MT, dev)
        for _ in range(2):
            for ks, sel in sels.items():
                vsa_kernel.vsa_fine_attention(q, k, v, sel, kl, SCALE)
        torch.xpu.synchronize()

        for ks, sel in sels.items():
            m, lo, hi = bench(lambda: vsa_kernel.vsa_fine_attention(q, k, v, sel, kl, SCALE))
            fl = 4.0 * H * MT * ks * BLOCK * BLOCK * D
            print("  %-7d %-7.1f %10.2f %10.1f %10.2fx %11s"
                  % (ks, 100.0 * ks / MT, m * 1e3, fl / m / 1e12, md / m, "--"))
        print("  -> break-even vs dense is near keep = %.0f%% "
              % (100.0 * (1.0 / (96.5 / 16.5))))
        print()


if __name__ == "__main__":
    main()
