# -*- coding: utf-8 -*-
"""Fair-config sweep + backend ceiling, part 2.

Three measurements the first pass could not answer cleanly:

  D. num_warps x BLOCK_M  WITH keep% HELD CONSTANT (the first pass let keep
     drift because K_SEL was fixed while MT changed -- not apples-to-apples).
  E. cost of the pad-column mask that the shipped kernel adds over the
     bare-bones ablation kernel (one extra load + one tl.where per tile).
  F. Triton GEMM vs vendor GEMM at several block shapes -- the backend ceiling.

    set OMNIXPU_ENABLE=1 && set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    python_embeded\\python.exe sweep2.py

SPDX-License-Identifier: GPL-3.0
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NODE_DIR = os.environ.get("VSA_NODE_DIR")
if not NODE_DIR:                      # self-contained: use the kernel shipped in this repo
    NODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernel")
if not os.path.isfile(os.path.join(NODE_DIR, "vsa_kernel.py")):
    raise SystemExit("vsa_kernel.py not found in %s\nset VSA_NODE_DIR to the "
                     "folder that contains it." % NODE_DIR)
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bench import H, D, SCALE, make_sel, klen_full  # noqa: E402

S_MAIN = 21952
KEEP = 0.125


@triton.jit
def _k_pad(q_ptr, k_ptr, v_ptr, idx_ptr, klen_ptr, o_ptr, S, MT, scale,
           K_SEL: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
    """shipped kernel, pad mask ON"""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + pid_h * S * D + offs_m[:, None] * D + offs_d[None, :])
    m_i = tl.full([BLOCK], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK], tl.float32)
    acc = tl.zeros([BLOCK, D], tl.float32)
    for t in range(K_SEL):
        tile = tl.load(idx_ptr + (pid_h * MT + pid_m) * K_SEL + t)
        tile_klen = tl.load(klen_ptr + tile)
        offs_n = tile * BLOCK + tl.arange(0, BLOCK)
        k = tl.load(k_ptr + pid_h * S * D + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        n_local = tl.arange(0, BLOCK)
        s = tl.where(n_local[None, :] >= tile_klen, float("-inf"), s)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + pid_h * S * D + offs_n[:, None] * D + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + pid_h * S * D + offs_m[:, None] * D + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty))


@triton.jit
def _k_nopad(q_ptr, k_ptr, v_ptr, idx_ptr, klen_ptr, o_ptr, S, MT, scale,
             K_SEL: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
    """same kernel, pad mask OFF (the ablation variant)"""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + pid_h * S * D + offs_m[:, None] * D + offs_d[None, :])
    m_i = tl.full([BLOCK], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK], tl.float32)
    acc = tl.zeros([BLOCK, D], tl.float32)
    for t in range(K_SEL):
        tile = tl.load(idx_ptr + (pid_h * MT + pid_m) * K_SEL + t)
        offs_n = tile * BLOCK + tl.arange(0, BLOCK)
        k = tl.load(k_ptr + pid_h * S * D + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + pid_h * S * D + offs_n[:, None] * D + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + pid_h * S * D + offs_m[:, None] * D + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty))


def launch(kern, q, k, v, sel, klen, num_warps, block):
    Hh, S, Dd = q.shape[1], q.shape[0], q.shape[2]
    MT, K_SEL = sel.shape[0], sel.shape[1]
    qh = q.permute(1, 0, 2).contiguous()
    kh = k.permute(1, 0, 2).contiguous()
    vh = v.permute(1, 0, 2).contiguous()
    idx = sel.unsqueeze(0).expand(Hh, MT, K_SEL).contiguous().to(torch.int32)
    kl = klen.to(torch.int32).contiguous()
    o = torch.empty(Hh, S, Dd, device=q.device, dtype=q.dtype)
    kern[(MT, Hh)](qh, kh, vh, idx, kl, o, S, MT, SCALE,
                   K_SEL=K_SEL, BLOCK=block, D=Dd, num_warps=num_warps)
    return o


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
    return ts[len(ts) // 2]


def main():
    dev, dt = "xpu", torch.bfloat16
    import vsa_kernel                                              # noqa: E402
    vsa_kernel._ensure_triton_env()      # inherit vcvars64 so Triton can compile
    torch.manual_seed(0)
    q = torch.randn(S_MAIN, H, D, device=dev, dtype=dt)
    k = torch.randn(S_MAIN, H, D, device=dev, dtype=dt)
    v = torch.randn(S_MAIN, H, D, device=dev, dtype=dt)
    qh, kh, vh = q.permute(1, 0, 2).contiguous(), k.permute(1, 0, 2).contiguous(), \
        v.permute(1, 0, 2).contiguous()
    md = bench(lambda: F.scaled_dot_product_attention(qh.unsqueeze(0), kh.unsqueeze(0),
                                                      vh.unsqueeze(0), scale=SCALE))
    print("reference dense SDPA (vendor) = %.2f ms\n" % (md * 1e3))

    # ---- build every config up front, then warm the device once ----
    configs = []
    for block in (64, 128, 256):
        mt = S_MAIN // block
        ksel = max(8, int(KEEP * mt))
        n_prefix = max(2, min(23, ksel - 3))
        sel = make_sel(mt, n_prefix, ksel, dev)
        kln = klen_full(mt, dev)
        for w in (4, 8, 16, 32):
            configs.append((block, mt, ksel, w, sel, kln))

    print("   warming up %d configs (2 passes) ..." % len(configs), flush=True)
    for _ in range(2):
        for block, mt, ksel, w, sel, kln in configs:
            try:
                launch(_k_pad, q, k, v, sel, kln, w, block)
            except Exception:                                        # noqa: BLE001
                pass
    torch.xpu.synchronize()

    print("\nD. num_warps x BLOCK_M, keep held at %.1f%%  (S=%d)"
          % (100 * KEEP, S_MAIN))
    print("   %-9s %-6s %-7s %-10s %s"
          % ("BLOCK_M", "MT", "K_SEL", "round", "".join("%12s" % ("w=%d" % w)
                                                        for w in (4, 8, 16, 32))))
    print("   " + "-" * 82)
    times = {(b, w): [] for b, w in {(c[0], c[3]) for c in configs}}
    for rnd in (1, 2):
        for block in (64, 128, 256):
            row = []
            for w in (4, 8, 16, 32):
                cfg = [c for c in configs if c[0] == block and c[3] == w][0]
                sel, kln = cfg[4], cfg[5]
                try:
                    t = bench(lambda: launch(_k_pad, q, k, v, sel, kln, w, block),
                              iters=3, warmup=1) * 1e3
                    row.append("%12s" % ("%.1f ms" % t))
                    times[(block, w)].append(t)
                except Exception as e:                               # noqa: BLE001
                    row.append("%12s" % "ERR")
                    print("      BLOCK=%d w=%d -> %s" % (block, w, str(e)[:70]))
            print("   %-9d %-6d %-7d %-10s %s"
                  % (block, S_MAIN // block, max(8, int(KEEP * (S_MAIN // block))),
                     "r%d" % rnd, "".join(row)))
    best = min(times.items(), key=lambda kv: sorted(kv[1])[0] if kv[1] else 1e9)
    print("   best (fastest round): BLOCK=%d w=%d -> %.1f ms  (rounds: %s)"
          % (best[0][0], best[0][1], min(best[1]), ["%.1f" % x for x in best[1]]))

    print("\nE. cost of the pad-column mask (K_SEL=43, BLOCK=64)")
    mt, ksel = S_MAIN // 64, 43
    sel = make_sel(mt, 23, ksel, dev)
    kln = klen_full(mt, dev)
    tp = bench(lambda: launch(_k_pad, q, k, v, sel, kln, 8, 64))
    tn = bench(lambda: launch(_k_nopad, q, k, v, sel, kln, 8, 64))
    print("   with pad mask    %8.2f ms" % (tp * 1e3))
    print("   without          %8.2f ms" % (tn * 1e3))
    print("   -> mask costs    %8.2f ms  (%.1f%%)" % ((tp - tn) * 1e3,
                                                      100 * (tp - tn) / tp))

    print("\nF. backend ceiling: GEMM 4096x4096x4096, bf16")
    a = torch.randn(4096, 4096, device=dev, dtype=dt)
    b = torch.randn(4096, 4096, device=dev, dtype=dt)
    t = bench(lambda: torch.matmul(a, b))
    print("   torch.matmul / vendor library   %7.2f ms  %7.1f TFLOP/s"
          % (t * 1e3, 2 * 4096 ** 3 / t / 1e12))

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, M, N, K,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], tl.float32)
        for kk in range(0, K, BK):
            av = tl.load(a_ptr + rm[:, None] * K + (kk + rk)[None, :])
            bv = tl.load(b_ptr + (kk + rk)[:, None] * N + rn[None, :])
            acc = tl.dot(av, bv, acc)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty))

    M = N = K = 4096
    c = torch.empty(M, N, device=dev, dtype=dt)
    for bm, bn, bk in ((128, 128, 32), (128, 256, 64), (128, 128, 64), (64, 64, 64)):
        try:
            g = (M // bm, N // bn)
            tt = bench(lambda: gemm[g](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, num_warps=8),
                       iters=3, warmup=1)
            print("   Triton GEMM BM=%-4d BN=%-4d BK=%-3d %7.2f ms  %7.1f TFLOP/s"
                  % (bm, bn, bk, tt * 1e3, 2 * M * N * K / tt / 1e12))
        except Exception as e:                                       # noqa: BLE001
            print("   Triton GEMM BM=%-4d BN=%-4d BK=%-3d  ERR %s"
                  % (bm, bn, bk, str(e)[:60]))


if __name__ == "__main__":
    main()
