# MIOpen gfx1150 Research

Research and patches for [ROCm/MIOpen](https://github.com/ROCm/MIOpen) gfx1150 (RDNA 3.5 / Strix Point) convolution solver support.

## Three Bugs, Three Owners

On gfx1150 (AMD Ryzen AI 9 HX 370, Radeon 890M), MIOpen training hits three independent issues:

### Bug 1: MIOpen CK Whitelists (this repo's patch)

MIOpen gates CK solver selection on device whitelists. gfx1150 is missing from both:

| Whitelist | File | RDNA 3 | RDNA 3.5 | RDNA 4 |
|-----------|------|--------|----------|--------|
| Modern CK | `ck_utility_common.hpp` | Not listed | Not listed | Not listed |
| Legacy CK | `legacy_ck_common.hpp` | gfx1100-1102 | **Missing** | gfx1200-1201 |

**Fix**: Add gfx1150 to the legacy CK whitelist. The modern CK whitelist is MI-series only by design (a known limitation with a TODO to fix).

### Bug 2: CK Device Code Crash (upstream CK issue)

When gfx1150 is added to the modern CK whitelist, CK-compiled device kernels crash at runtime:
```
hip_code_object.cpp:400: StatCO::getStatFunc: Assertion `err == hipSuccess' failed.
```
This is the same class of ISA error as Winograd Fury (`gfx115*` illegal opcode). CK's device codegen has instructions incompatible with RDNA 3.5, even though CK headers claim gfx1150 support. Requires upstream CK fixes.

### Bug 3: PyTorch Workspace Allocation (upstream PyTorch issue)

PyTorch's ATen layer passes `workspace_size=0` to MIOpen's Gemm convolution solvers. MIOpen's own Find2 API allocates workspace correctly (`problem.cpp:506`), but PyTorch uses the legacy Find1 API. Affects all targets but masked on gfx1100 by Winograd/CK availability.

**Workaround**: `MIOPEN_DEBUG_CONV_GEMM=0`

## Solver Availability Matrix

| Solver | gfx1100 | gfx1150 (before) | gfx1150 (after patch) |
|--------|---------|-------------------|-----------------------|
| Modern CK (14 solvers) | Blocked | Blocked | Blocked (CK crash) |
| Legacy CK (dlops fwd) | OK | Blocked | **OK** |
| Winograd Fury | OK | Blocked (ISA) | Blocked (ISA) |
| Gemm (rocBLAS) | OK | workspace=0 | workspace=0 |
| Direct (various) | OK | OK (slow) | OK (slow) |

## The Patch

`patches/0001-add-gfx1150-to-ck-whitelists.patch` adds gfx1150 to the legacy CK whitelist:

```diff
 // legacy_ck_common.hpp
            StartsWith(handle.GetDeviceName(), "gfx1102") ||
+           StartsWith(handle.GetDeviceName(), "gfx1150") ||
            StartsWith(handle.GetDeviceName(), "gfx1200") ||
```

## Build

```bash
git clone --depth 1 https://github.com/ROCm/MIOpen.git miopen-gfx1150
cd miopen-gfx1150
git apply patches/0001-add-gfx1150-to-ck-whitelists.patch

mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_INSTALL_PREFIX=/opt/miopen-gfx1150 \
  -DMIOPEN_BACKEND=HIP \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx1150" \
  -DMIOPEN_USE_COMPOSABLEKERNEL=ON \
  -DMIOPEN_USE_MLIR=OFF \
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF \
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DHALF_INCLUDE_DIR=/usr/include

ninja -j6 libMIOpen.so  # ~377 targets, builds in ~30min on Ryzen AI 9
```

Build verified: 377/377 targets, 0 failures, libMIOpen.so = 787MB.

## Key Findings

1. **CK 7.2.0 headers claim gfx1150 support** (`__gfx1150__` in ck.hpp, device_prop.hpp) but compiled device code crashes at runtime. The whitelist is the wrong place to gate — CK's own runtime checks should handle this.

2. **The modern CK whitelist is MI-series only** — not just missing gfx1150, but ALL RDNA GPUs. Even gfx1100 doesn't get modern CK solvers. The code has a TODO: "This function should probably always return true."

3. **The workspace bug is PyTorch-side** — MIOpen's Find2 API (`problem.cpp:506`) correctly allocates workspace. PyTorch's legacy convolution API doesn't.

## Hardware

- AMD Ryzen AI 9 HX 370 (Strix Point)
- Radeon 890M iGPU (gfx1150, RDNA 3.5)
- ROCm 7.2.0, CK 7.2.0, CachyOS Linux 6.19

## In-the-Wild Reproducers

Real-world production workloads that hit the bugs documented above, with measurements:

- [**2026-04-15 — OmniVoice voice cloning (bug #3)**](reproducers/2026-04-15-omnivoice-gfx1150.md) — k2-fsa/OmniVoice diffusion TTS generating a 20-second voice clone on a 6-second reference. The `GemmFwdRest` workspace=0 fallback fires 40+ times per generation with workspace requests up to 424 MB. `MIOPEN_FIND_MODE=FAST` workaround produces a sustained **3.5× speedup** (2m 52s → 49s) with zero quality loss. Second production-shaped reproducer on gfx1150/gfx1151-class hardware after TimLawrenz's NanoDiT training observations on gfx1151 Strix Halo.

If you're running into these bugs on gfx1150 and your workload looks different from the ones above, contributions welcome — open a PR with a new `reproducers/YYYY-MM-DD-<workload>.md` file.
