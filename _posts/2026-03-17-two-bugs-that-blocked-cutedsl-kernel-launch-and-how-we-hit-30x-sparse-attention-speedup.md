---
layout: post
title: "Two Bugs That Blocked CuTeDSL Kernel Launch (And How We Hit 30x Sparse Attention Speedup)"
date: 2026-03-17 18:33:53 +0000
categories: [performance]
tags: [cuda, cutedsl, mlir, gpu-kernels]
excerpt: "We hit two undocumented CuTeDSL integration bugs — a missing MLIR context and a TVM-FFI type error — then reached 30x sparse attention speedup by extracting raw CUfunction handles and parallelizing the reduce kernel across head groups."
---


We were porting a high-performance sparse attention kernel from Triton (~84x speedup, ~29us latency) to CuTeDSL, NVIDIA's Python DSL for writing CUTLASS-style GPU kernels. The goal was to see if we could match or exceed Triton by getting closer to the metal. Two undocumented bugs blocked us before the kernel ever ran — and the fixes revealed a lot about how CuTeDSL works internally.

## Bug 1: MLIR Context Not Active at Launch

The first crash was cryptic:

```
ir.Attribute.parse() called with no active MLIR context
```

CuTeDSL's `KernelLauncher.launch()` calls an internal `_generate_kernel_attrs()` method, which calls `ir.Attribute.parse()` directly — before any MLIR context has been set up. This happens even on a warm, already-compiled kernel. The fix was a one-time monkey-patch applied at import time:

```python
from cutlass._mlir import ir as _mlir_ir

_KernelLauncher = cutlass.cutlass_dsl.cutlass.KernelLauncher
_original_launch = _KernelLauncher.launch

def _patched_launch(self, *args, **kwargs):
    if _mlir_ir.Context.current is None:
        with _mlir_ir.Context():
            return _original_launch(self, *args, **kwargs)
    return _original_launch(self, *args, **kwargs)

_KernelLauncher.launch = _patched_launch
```

The guard on `Context.current is None` is important — if a context is already active (e.g. during initial JIT compilation), wrapping in a second one would create an orphaned inner context and break IR operations that assume a single active context.

## Bug 2: TVM-FFI Rejects Unannotated Tensors

The CuTeDSL docs mention setting `CUTE_DSL_ENABLE_TVM_FFI=1` to enable TVM-FFI compilation for faster dispatch. We set it as a global environment variable and hit a second crash:

```
DSLRuntimeError: Unsupported argument type: <class 'torch.Tensor'>
for annotated type: None
```

The TVM-FFI argument converter requires all `@cute.jit` parameters to have explicit type annotations. Any unannotated `torch.Tensor` parameter — including internal helpers — causes the entire dispatch path to fail at call time, not at compile time.

The fix: remove the env var entirely. Use CuTeDSL's normal JIT path for compilation, then extract raw `CUfunction` handles from the compiled kernels and launch them via `cuLaunchKernel` through a minimal C shim. This gives us the same low-overhead dispatch without the annotation constraint.

A related gotcha: `@cute.kernel` functions cannot be launched directly from Python. They must be called from a `@cute.jit` wrapper. This is because operand extraction calls `__extract_mlir_values__()`, which only exists on MLIR-level tensor objects, not on the Python-side `runtime._Tensor` wrappers that Python code holds.

## Kernel Architecture

With the integration bugs resolved, we built a split-K dual-kernel design:

- **Compute kernel**: Warp-0 runs 16×8×16 MMA (tensor cores) to compute partial attention scores; all warps cooperate on element-wise output accumulation via scatter-gather from paged KV cache. Grid = `(T, NUM_SPLITS)`.
- **Reduce kernel**: Merges partial log-sum-exp results across splits. Grid = `(T, 1)` initially.

## The Optimization That Mattered Most

We iterated through several changes and measured their impact on average latency across 23 workloads:

| Change | Avg Latency | Notes |
|--------|-------------|-------|
| Initial port, 32 splits | 118us | Baseline |
| NUM_SPLITS=8 | 176us | Too little parallelism |
| NUM_SPLITS=32 | 114us | Sweet spot |
| Merged CKV+KPE gather loops | ~110us | Minor |
| **Parallel reduce: grid=(T, 4)** | **73us** | **-36%** |

The biggest single gain came from parallelizing the reduce kernel. The original design had one block per token, processing all 16 attention heads serially. Splitting into 4 blocks of 4 heads each — grid `(T, N_RBLOCKS)` with `N_RBLOCKS=4` — gave a 36% latency reduction on its own. Each head group is independent in the log-sum-exp merge, so there's no synchronization cost.

The C dispatch shim that launches both kernels back-to-back on the same stream:

```c
void dual_launch(
    void* func_compute, void* func_reduce,
    unsigned T, unsigned num_splits, unsigned threads,
    unsigned smem_compute, unsigned smem_reduce,
    unsigned n_rblocks,
    void* stream,
    void** args_compute, void** args_reduce)
{
    g_launch(func_compute, T, num_splits, 1, threads, 1, 1,
             smem_compute, stream, args_compute, 0);
    g_launch(func_reduce, T, n_rblocks, 1, threads, 1, 1,
             smem_reduce, stream, args_reduce, 0);
}
```

## Results and Remaining Gap

All 23 correctness checks passed (max abs error 1.56e-02, within the 0.02 threshold). Final average latency: **73us (~30x speedup)** vs Triton's 29us (~84x).

The remaining gap is structural: CuTeDSL's scatter-gather pattern currently issues scalar GMEM loads for each key index. Triton's compiler generates async pipelined vectorized loads for the same pattern. Closing that gap would require either inline PTX for vectorized loads or TMA-based gather — neither trivially available in CuTeDSL's Python API today.

## Takeaways

The MLIR context bug is a library-level issue that affects any CuTeDSL user running kernels outside an explicitly managed context — the monkey-patch is a necessary workaround until it's fixed upstream. The TVM-FFI pitfall is subtler: global env vars that change argument dispatch behavior should be scoped, not set at process startup. And for reduction kernels over independent groups, always check whether the reduction dimension can be parallelized across blocks — it rarely gets profiled explicitly but often dominates at small batch sizes.

