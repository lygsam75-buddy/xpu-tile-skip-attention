# 04 — Compiling Triton kernels on Windows + Intel XPU: three blockers

Out of the box, **no Triton kernel can be compiled** on a normal Windows + Intel XPU install. Three
independent defects must be worked around. All three are upstream-fixable, and one of them is a
two-line bug in `triton/runtime/build.py`.

This is not a complaint about a development environment. It is the reason a `triton.jit` kernel
cannot be shipped to end users on Windows today, which in turn is why sparse-attention kernels do
not reach the hands that would use them.

Environment where these were found: Windows 11 Pro, VS Build Tools 18 (installed, `cl.exe` not on
`PATH`), Windows SDK 10.0.26100.0, triton 3.7.1 with the Intel backend, torch 2.12.0+xpu.

---

## Blocker 1 — `Failed to find C compiler. Please specify via CC environment variable`

`vcvars64.bat` has to be executed for `cl.exe` to be on `PATH`. Two traps:

* **Trap 1a.** `subprocess.run(["cmd", "/c", '"<path>\\vcvars64.bat" && set'])` silently swallows the
  quotes on Windows, so the batch file never runs and the environment comes back unchanged. It must
  be `subprocess.run('"%s" && set' % vcvars, shell=True)` — a single string, `shell=True`.
* **Trap 1b.** A "successful" compile in a developer shell can be an illusion: the run may hit a
  stale cached launcher in `~/.triton/cache` instead of actually invoking the compiler. Our first
  smoke test passed for exactly this reason and told us nothing. Delete the cache entry, or check
  for a `[msvc]` line / `VCToolsInstallDir`, before believing a pass.

Reference implementation: `bench/run_with_msvc.py`.

## Blocker 2 — `cl.EXE ... exit status 2`, with GCC flags on the command line

This is a genuine upstream bug, and the smallest of the three to fix.

`triton/runtime/build.py:96-101`:

```python
if cxx is icpx:
    ccflags += ["-fsycl"]
else:
    if os.name != "nt":
        ccflags += ["--std=c++17"]
    if os.environ.get("TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS", "1") == "1":
        ccflags += ["-Wno-deprecated-declarations"]
    if os.environ.get("TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING", "1") == "1":
        ccflags += ["-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING"]
```

The Windows branch correctly skips `--std=c++17` (clang-style, rejected by `cl.exe`) but then
unconditionally injects **two more GCC-style flags**, and the environment-variable defaults are
`"1"`, i.e. injected by default. `cl.exe` does not recognise them and treats them as input files,
failing with exit status 2 and an error that points at nothing useful.

**Workaround, no source patch:**

```
set TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS=0
set TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=0
```

**Suggested upstream fix:** guard both injections with `if os.name != "nt"`, matching the
`--std=c++17` guard directly above them.

## Blocker 3 — `cl` cannot find `<stdio.h>`

`vcvars64.bat` determines the Windows SDK version by calling `reg.exe`. When `reg.exe` is blocked
(it was, in our sandboxed execution context, and it is blocked in plenty of managed Windows setups),
`WindowsSdkDir` ends up empty and `INCLUDE`/`LIB` contain only MSVC's own directories. The SDK is
installed; the environment simply does not know where it is. `cl` then fails on the first standard
header.

**Workaround:** add the SDK paths explicitly.

```
Include: <SDK>\Include\10.0.26100.0\{ucrt,shared,um,winrt}
Lib:     <SDK>\Lib\10.0.26100.0\{ucrt\x64,um\x64}
+ set WindowsSDKVersion, WindowsSdkDir, UniversalCRTSdkDir
```

Implemented in `bench/run_with_msvc.py::patch_sdk()` and mirrored inside the shipped kernel module
(`kernel/vsa_kernel.py::_patch_sdk()`) as a defensive fallback. Verified: SDK path count `6`,
`INCLUDE` 11 entries, `LIB` 5 entries, `rc = 0`.

## Diagnostic note — a fourth problem that hides the first three

Triton decodes `cl.exe`'s console output as UTF-8. On a Chinese-locale Windows, `cl` emits GBK, so
any non-ASCII byte raises `UnicodeDecodeError: 0xce` and **the real compiler error is replaced by a
decoding error**. Always decode `cl` output as GBK when diagnosing.

`bench/diag_msvc.py` does this and prints the raw `cl` invocation.

---

## What good looks like

The shipped module in `kernel/vsa_kernel.py` bootstraps all of the above itself
(`_ensure_triton_env()` + `_patch_sdk()`), so an end user's ComfyUI process compiles the kernel with
zero configuration. It also tolerates `reg.exe` being blocked, because the SDK paths are patched
rather than probed.

That is the right shape for a workaround, but it is a workaround: every Triton-on-Windows-XPU
adopter re-implements it. Fixing blockers 2 and 3 upstream removes the entire class.

## Runtime environment variables

Two more are needed for correct device selection on multi-adapter systems:

```
set OMNIXPU_ENABLE=1
set ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
```

Without device ordering, the runtime can pick the wrong adapter (integrated graphics) and any
measurement becomes noise. Worth documenting next to the compiler notes, since the failure mode
looks like a performance problem rather than a device-selection problem.
