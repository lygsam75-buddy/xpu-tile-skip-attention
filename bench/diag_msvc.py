# -*- coding: utf-8 -*-
"""诊断：MSVC 环境下 cl.exe 到底报什么错。

trition 在编译 launcher 失败后会用 utf-8 解码 cl 的输出（中文 Windows 是 GBK），
于是真实错误被 UnicodeDecodeError 掩盖。这里手动跑一遍最小 C++ 编译，把原文打出来。
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_with_msvc import msvc_env, patch_sdk  # noqa: E402


def main():
    env, n = msvc_env()
    env["TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS"] = "0"
    env["TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING"] = "0"
    added = patch_sdk(env)
    print("补入 SDK 路径 %d 条" % len(added))

    print("=== 关键环境变量 ===")
    for k in ("VCToolsInstallDir", "WindowsSdkDir", "UniversalCRTSdkDir",
              "WindowsSDKVersion", "INCLUDE", "LIB"):
        v = env.get(k, "<未设置>")
        print("%s = %s" % (k, v if len(v) < 260 else v[:260] + " ..."))
        if k in ("INCLUDE", "LIB") and v not in ("<未设置>", ""):
            for part in v.split(os.pathsep):
                if part.strip():
                    print("      %s  [%s]" % (part, "存在" if os.path.isdir(part) else "缺失"))

    where = subprocess.run("where cl", shell=True, capture_output=True,
                           text=True, errors="replace", env=env)
    print("\n=== where cl ===\n%s" % (where.stdout or where.stderr))

    tmp = tempfile.mkdtemp(prefix="cltest_")
    src = os.path.join(tmp, "hello.cpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write("#include <cstdio>\nint main(){ std::printf(\"hi\\n\"); return 0; }\n")

    print("\n=== 最小 C++ 编译（cl /nologo /EHsc /MD）===")
    cl = shutil.which("cl", path=env.get("PATH", "")) or "cl"
    print("cl = %s" % cl)
    exe = os.path.join(tmp, "hello.exe")
    r = subprocess.run([cl, "/nologo", "/EHsc", "/MD", src, "/Fe:" + exe],
                       capture_output=True, env=env)
    print("rc = %s" % r.returncode)
    print("--- stdout ---")
    print(r.stdout.decode("gbk", errors="replace")[:2000])
    print("--- stderr ---")
    print(r.stderr.decode("gbk", errors="replace")[:2000])
    print("产物存在: %s" % os.path.exists(exe))


if __name__ == "__main__":
    main()
