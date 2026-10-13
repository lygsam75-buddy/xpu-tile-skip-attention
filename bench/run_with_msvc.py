# -*- coding: utf-8 -*-
"""在 MSVC (vcvars64) 环境下执行目标脚本。

背景：本机装了 VS Build Tools 18，但 cl.exe 不在 PATH 里；
triton 在 XPU 路径下编译 launcher 时找不到 C 编译器会直接抛
"Failed to find C compiler. Please specify via CC environment variable"。

做法：跑一次 vcvars64.bat 并把它的环境变量继承过来，再启动目标脚本。
"""
import os
import subprocess
import sys

VCVARS = (r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
          r"\VC\Auxiliary\Build\vcvars64.bat")

WIN_KITS = r"C:\Program Files (x86)\Windows Kits\10"
SDK_VER = "10.0.26100.0"


def patch_sdk(env):
    """补全 Windows SDK 的 INCLUDE / LIB。

    vcvars64.bat 靠 reg.exe 探测 SDK 版本，而 reg.exe 被安全策略拦截，
    导致 WindowsSdkDir 为空、INCLUDE/LIB 里只有 MSVC 自己的目录
    （cl 连 <stdio.h> 都找不到，必然 exit 2）。SDK 本体是装好的，这里手动补。
    """
    inc_parts = [p for p in env.get("INCLUDE", "").split(os.pathsep) if p.strip()]
    lib_parts = [p for p in env.get("LIB", "").split(os.pathsep) if p.strip()]
    added = []

    for sub in ("ucrt", "shared", "um", "winrt"):
        p = os.path.join(WIN_KITS, "Include", SDK_VER, sub)
        if os.path.isdir(p) and p not in inc_parts:
            inc_parts.append(p)
            added.append(p)
    for sub in (("ucrt", "x64"), ("um", "x64")):
        p = os.path.join(WIN_KITS, "Lib", SDK_VER, *sub)
        if os.path.isdir(p) and p not in lib_parts:
            lib_parts.append(p)
            added.append(p)

    env["INCLUDE"] = os.pathsep.join(inc_parts)
    env["LIB"] = os.pathsep.join(lib_parts)
    env.setdefault("WindowsSDKVersion", SDK_VER + "\\")
    env.setdefault("WindowsSdkDir", WIN_KITS + "\\")
    env.setdefault("UniversalCRTSdkDir", WIN_KITS + "\\")
    return added


def msvc_env():
    # 必须走 shell=True：list 形式在 Windows 上会把 vcvars64 路径的引号吃掉
    cmdline = '"%s" && set' % VCVARS
    out = subprocess.run(cmdline, shell=True, capture_output=True,
                         text=True, encoding='gbk', errors='replace')
    txt = out.stdout or ''
    if 'VCToolsInstallDir' not in txt and 'INCLUDE=' not in txt:
        raise RuntimeError(
            "vcvars64 没吐出环境变量。rc=%s\n--- stdout 尾 ---\n%s\n--- stderr ---\n%s"
            % (out.returncode, txt[-600:], (out.stderr or '')[-600:]))
    env = dict(os.environ)
    n = 0
    for line in txt.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            # 排除 cmd 内部的伪变量（如 =C:）
            if not k or k.startswith('='):
                continue
            env[k] = v
            n += 1
    return env, n


def main():
    if len(sys.argv) < 2:
        print("用法: python run_with_msvc.py <目标脚本> [参数...]")
        return 2
    env, n = msvc_env()
    # triton 在「非 icpx」路径下会默认追加两个 GCC 风格参数
    #   -Wno-deprecated-declarations / -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING
    # （见 triton/runtime/build.py:98-101）。cl.exe 不认，直接把它们当输入文件，
    # 编译以 exit 2 失败。作者只在 Windows 分支跳过了 --std=c++17，漏了这两个。
    # 用环境变量关掉，无需改 triton 源码。
    env["TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS"] = "0"
    env["TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING"] = "0"
    if not (env.get("PATH") and "VC" in env.get("PATH", "")):
        print("[msvc] 警告：PATH 里似乎没有 VC 目录，可能没抓到环境")
    added = patch_sdk(env)
    print("[msvc] 继承 %d 个环境变量；VCToolsInstallDir=%s"
          % (n, env.get("VCToolsInstallDir", "?")[:70]))
    print("[msvc] 补入 Windows SDK 路径 %d 条 → INCLUDE %d 项 / LIB %d 项"
          % (len(added), len(env["INCLUDE"].split(os.pathsep)),
             len(env["LIB"].split(os.pathsep))))
    target = sys.argv[1]
    cwd = os.path.dirname(os.path.abspath(target)) or None
    r = subprocess.run([sys.executable, os.path.abspath(target)] + sys.argv[2:],
                       env=env, cwd=cwd)
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
