---
layout: post
title: "What's Actually Inside Your GPU Kernel Benchmark Number"
date: 2026-03-15 05:09:10 +0000
categories: [performance]
tags: [cuda, gpu-benchmarking, cuda-graphs, profiling]
---

When optimizing a fused sparse attention kernel for a GPU programming contest, we spent a session dissecting what a benchmark measurement actually contains. The answer was more complicated than expected: hidden profiling overhead, deterministic pointer aliasing, pre-baked command buffers, and cloud instance variance were all silently shaping the numbers. Here is what we found.

## The Problem

We were measuring GPU kernel speedups against a naive PyTorch baseline and noticed that some optimizations produced smaller improvements than the profiler suggested, while others appeared inconsistent across runs. Before trusting any number, we needed to understand every microsecond in the measurement loop.

## CUPTI Is Always On If It Is Installed

Installing `cupti-python` (to get profiling access) silently downgrades PyTorch and registers CUPTI profiling callbacks system-wide. Those callbacks fire on every CUDA API call — even when no active profiling session is running. We measured 8–15% overhead on every CUDA API call as a result.

Disabling it without removing the library:

```bash
CUDA_INJECTION64_PATH="" CUPTI_ENABLED=0 python benchmark.py
```

This recovered the overhead. But here is the paradox: removing CUPTI *hurt* the measured speedup ratio. The naive baseline makes many more CUDA API calls than our optimized kernel, so CUPTI was taxing the baseline disproportionately. Eliminating it actually made the baseline faster in relative terms. If your benchmark compares a kernel-heavy path against a Python-heavy one, installed-but-idle CUPTI can inflate your reported speedup.

## The Caching Allocator Makes Pointer-Skip Optimizations Useless

The benchmark framework clones tensors each iteration to prevent cross-iteration state contamination. Because of how PyTorch's CUDA caching allocator works — it reuses freed memory blocks deterministically — the allocated pointer for a given tensor alternates between two addresses: A, B, A, B, across iterations.

We had added a "skip SetParams if pointer unchanged" fast path. During warmup it fires, but during the timed measurement loop the pointer alternates every iteration, so the fast path never triggers. We were carrying dead code that looked useful in profiling but contributed nothing during measurement.

## CUDA Graph Launch Is Faster Than a Single cuLaunchKernel

This one is counter-intuitive. `cuGraphLaunch` involves two driver calls (`cuGraphExecKernelNodeSetParams` + `cuGraphLaunch`) versus one for `cuLaunchKernel`, yet the graph path is faster:

| Launch method | GPU time |
|---|---|
| Raw Python / Triton dispatch | ~38.4 μs |
| Direct `cuLaunchKernel` | ~25.0 μs |
| Graph SetParams + Launch | ~24.1 μs |
| Graph-only (no SetParams) | ~21.4 μs |

The reason is that `cuGraphInstantiateWithFlags` pre-bakes binary resolution, grid/block validation, and GPU-side command buffer creation at capture time. `cuGraphLaunch` submits a pre-built packet; `cuLaunchKernel` does all that work at launch time on every call.

## 22% of Measured Time Was CPU Overhead

Breaking down a single iteration of our optimized kernel:

```
Python dispatch (dict lookup, shape check):  ~0.3 μs
data_ptr() × 7 pointer reads:               ~0.6 μs
C shim (SetParams + cuGraphLaunch):          ~3.9 μs
GPU kernel execution:                       ~24.4 μs
─────────────────────────────────────────────────────
Total wall time per iteration:              ~29.2 μs
CPU/dispatch fraction:                         ~22%
```

On a kernel this fast, nearly a quarter of wall time is spent in Python and the CUDA driver before any GPU work begins. The implication: if you measure with `torch.cuda.synchronize()` bracketing, you are measuring CPU dispatch skill as much as GPU compute.

## C Compiler Flags Do Not Matter for Tiny Dispatch Shims

We compiled a ~20-line C shim that calls `cuGraphExecKernelNodeSetParams` and `cuGraphLaunch`. We tested `-O2` versus `-O3 -march=native -flto`. No consistent difference. The bottleneck is the driver calls themselves, not the surrounding C code. More interestingly, `-O3` without `-march=native` was 3–4% *slower* — larger compiled output created instruction cache pressure on a function called millions of times.

For CPU-side glue code around GPU calls, compile for code size, not throughput.

## Cloud Instance Variance Dwarfs Small Optimizations

Running parallel benchmark instances on a cloud GPU provider, we observed ~25% performance variation between nominally identical machines. An apparent 14% win from a compiler flag change turned out to be an assignment to a faster instance. We caught this only because we ran A/B comparisons sequentially on a pinned instance.

The rule: any optimization smaller than your cloud variance needs to be measured with sequential A/B runs on the same physical GPU, with enough repetitions to establish a distribution.

## The Takeaway

A benchmark number for a short GPU kernel is a composite of: hidden profiling overhead, allocator behavior, driver call costs, and hardware variance. Before optimizing a kernel further, audit what fraction of your measurement is actually GPU compute. In our case, ~22% was not — and that ceiling matters when you are chasing the last few percent of speedup.

