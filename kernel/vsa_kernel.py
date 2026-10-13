# -*- coding: utf-8 -*-
"""VSA fine stage 的 XPU tile-skipping attention 内核（Triton intel 后端）。

替代 vsa_core 里纯 torch 的 index_select + SDPA 路径：
- 每个 (head, query-block) 只 load 自己被选中的 K_SEL 个 key tile，做 online-softmax，
  不物化整段 K/V，也不做 chunk 并集（逐 query block 只取自己的 top-k，天然修正 union 语义）。
- 带 pad 列 mask：被选 tile 内 block 偏移 >= live 行数的列 -> -inf，与 torch 路径的
  padmask 语义一致（prefix 段 tile 末行是零填充）。

实测（B70 / H=56 D=128 BLOCK=64）：
  K_SEL=17(纯 topk 5%) ~37ms/层；真实 VSA K_SEL=43(sel_max=12.5%) ~93ms/层；
  稠密 SDPA ~133ms/层；torch gather 路径(union) ~406ms/层。

环境要求：首次编译需 MSVC（cl.exe）。ComfyUI 正常进程里没有 vcvars 环境，
故首次调用前用 _ensure_triton_env() 继承 vcvars64 + 关掉 triton 注入的 GCC 参数。
（本机 VS Build Tools 18 已装；reg.exe 在正常进程里可用，SDK 探测正常。）
License: GPL-3.0。
"""
import os
import subprocess

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    _TRITON_ERR = None
except Exception as _e:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_ERR = _e


_ENV_READY = False
_VCVARS = (r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
           r"\VC\Auxiliary\Build\vcvars64.bat")
_WIN_KITS = r"C:\Program Files (x86)\Windows Kits\10"
_SDK_VER = "10.0.26100.0"


def is_available():
    return _HAS_TRITON


def triton_err():
    return _TRITON_ERR


def _patch_sdk():
    """防御性补 SDK 路径（正常进程 vcvars 已能探测，此处兜底）。"""
    inc = [p for p in os.environ.get("INCLUDE", "").split(os.pathsep) if p.strip()]
    lib = [p for p in os.environ.get("LIB", "").split(os.pathsep) if p.strip()]
    added = 0
    for sub in ("ucrt", "shared", "um", "winrt"):
        p = os.path.join(_WIN_KITS, "Include", _SDK_VER, sub)
        if os.path.isdir(p) and p not in inc:
            inc.append(p)
            added += 1
    for sub in (("ucrt", "x64"), ("um", "x64")):
        p = os.path.join(_WIN_KITS, "Lib", _SDK_VER, *sub)
        if os.path.isdir(p) and p not in lib:
            lib.append(p)
            added += 1
    if added:
        os.environ["INCLUDE"] = os.pathsep.join(inc)
        os.environ["LIB"] = os.pathsep.join(lib)
        os.environ.setdefault("WindowsSDKVersion", _SDK_VER + "\\")
        os.environ.setdefault("WindowsSdkDir", _WIN_KITS + "\\")
        os.environ.setdefault("UniversalCRTSdkDir", _WIN_KITS + "\\")
    return added


def _ensure_triton_env():
    global _ENV_READY
    if _ENV_READY or not _HAS_TRITON:
        return
    _ENV_READY = True
    # triton 在非 icpx 路径会注入两个 GCC 参数，cl.exe 不认 -> 关掉
    os.environ.setdefault("TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS", "0")
    os.environ.setdefault("TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING", "0")
    vcvars = os.environ.get("VSA_VCVARS64", _VCVARS)
    if not os.path.exists(vcvars):
        print("[H3-VSA-XPU] vcvars64 不存在: %s（内核编译可能失败）" % vcvars, flush=True)
        return
    try:
        out = subprocess.run('"%s" && set' % vcvars, shell=True,
                             capture_output=True, text=True, encoding="gbk",
                             errors="replace")
        for line in (out.stdout or "").splitlines():
            if "=" in line and not line.startswith("="):
                k, v = line.split("=", 1)
                if k and not k.startswith("="):
                    os.environ[k] = v
    except Exception as e:
        print("[H3-VSA-XPU] vcvars64 继承失败: %r" % e, flush=True)
    _patch_sdk()


if _HAS_TRITON:

    @triton.jit
    def _vsa_fine_fwd(q_ptr, k_ptr, v_ptr, idx_ptr, klen_ptr, o_ptr,
                      S, MT, scale,
                      K_SEL: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
        pid_m = tl.program_id(0)   # query block (0..MT-1)
        pid_h = tl.program_id(1)   # head (0..H-1)

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
            # pad 列 mask：被选 tile 内 block 偏移 >= live 行数的列 -> -inf
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


def vsa_fine_attention(q, k, v, sel_idx, klen, scale):
    """VSA fine stage：tile-skipping attention（XPU 内核）。

    参数
    ----
    q, k, v : [S, H, D]（模型布局 [n, heads, hd]，S = MT * 64）
    sel_idx : [MT, K_SEL] int64，每个 query block 被选中的 key tile 索引（跨 head 共享）
    klen    : [MT] int32，每个 tile 的 live 行数（block_len）
    scale   : float

    返回
    ----
    out : [H, MT, BLOCK, D]，与 vsa_core 里纯 torch 路径的 out 同形。
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用: %r" % (_TRITON_ERR,))
    H = q.shape[1]
    S = q.shape[0]
    MT = sel_idx.shape[0]
    K_SEL = sel_idx.shape[1]
    BLOCK = S // MT
    D = q.shape[2]
    device = q.device

    _ensure_triton_env()

    q_h = q.permute(1, 0, 2).contiguous()
    k_h = k.permute(1, 0, 2).contiguous()
    v_h = v.permute(1, 0, 2).contiguous()
    idx = sel_idx.unsqueeze(0).expand(H, MT, K_SEL).contiguous().to(torch.int32)
    klen32 = klen.to(torch.int32).contiguous()
    o = torch.empty(H, S, D, device=device, dtype=q.dtype)

    _vsa_fine_fwd[(MT, H)](q_h, k_h, v_h, idx, klen32, o, S, MT, scale,
                           K_SEL=K_SEL, BLOCK=BLOCK, D=D, num_warps=8)
    return o.view(H, MT, BLOCK, D)
