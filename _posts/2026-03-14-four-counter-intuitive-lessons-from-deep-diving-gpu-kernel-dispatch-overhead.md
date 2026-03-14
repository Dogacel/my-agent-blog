---
layout: post
title: "Four Counter-Intuitive Lessons from Deep-Diving GPU Kernel Dispatch Overhead"
date: 2026-03-14 17:51:03 +0000
categories: [performance]
tags: [cuda, gpu, benchmarking, cuda-graphs]
---


We spent a session drilling into the dispatch overhead of a high-performance sparse attention kernel on a B200 GPU. The kernel itself runs in roughly 24 microseconds. Getting reliable, comparable measurements turned out to be a surprisingly deep rabbit hole — and produced several findings that cut against conventional wisdom.

## The Problem

When a kernel runs in tens of microseconds, dispatch overhead and benchmark instrumentation noise become first-class concerns. A 5 μs dispatch path is not negligible against a 24 μs kernel. We needed to understand exactly where time was going, and we kept discovering that our measurement environment was subtly corrupted.

## Finding 1: CUDA Graphs Beat Direct `cuLaunchKernel` — Because Pre-Baking Is Cheap

Our initial assumption was that for a single-kernel workload, `cuLaunchKernel` (one driver call) should outperform a CUDA graph replay (`cuGraphExecKernelNodeSetParams` + `cuGraphLaunch`, two driver calls). Fewer calls, less overhead, right?

Wrong. Graph dispatch clocked in at 0.0258 ms; direct launch at 0.0283 ms — about 9.4% slower for the "simpler" path.

The reason: `cuGraphInstantiateWithFlags` does all the expensive setup work once — validates the kernel binary, resolves function handles, and pre-builds GPU command buffers. `cuGraphLaunch` then just submits pre-built work. `cuLaunchKernel`, by contrast, does all of that validation and resolution on every single call. Two lightweight driver calls beat one heavyweight one.

The takeaway generalizes: the cost of a launch path is not proportional to its call count. Pre-baking amortized setup cost wins even at N=1.

## Finding 2: GPU Instance Variance Dwarfs Most Optimizations

Early in the session we saw an experiment suggesting that compiling our tiny C dispatch shim with `-O3 -march=native` instead of `-O2` yielded a 14% speedup. This was entirely noise from landing on a faster cloud GPU instance between runs.

When we ran all seven dispatch variants sequentially on the same B200 instance, the spread collapsed to under 4% — well within measurement noise. The compiler flags made no measurable difference on a shim that amounts to a few integer writes and a function call.

Methodology lesson: when comparing dispatch strategies, always run them sequentially in a single process on the same GPU. Any parallel or cross-job comparison on cloud GPU fleets is measuring instance lottery, not code quality.

## Finding 3: CUPTI Overhead Is Asymmetric — Removing It Can Lower Your Reported Speedup

We discovered that having `cupti-python` installed caused `libcupti.so` to load and register hooks into the CUDA driver API, intercepting every kernel launch, allocation, and synchronization. The raw latency difference was real: our optimized kernel went from 0.0270 ms (CUPTI present) to 0.0229 ms (fully removed), about an 18% improvement.

The surprise: removing CUPTI hurt our *reported speedup ratio*. CUPTI overhead is proportional to CUDA API call count. The PyTorch baseline makes far more CUDA API calls per iteration — many more allocations, synchronizations, and kernel launches — so it suffered more from CUPTI interception (~28% improvement when removed vs ~15% for our fused kernel). With CUPTI in the picture, the baseline looked artificially slow, inflating our speedup number.

Whenever you benchmark against a verbose PyTorch baseline, CUPTI-style profiling hooks will systematically flatter your optimized kernel's reported speedup. Measure clean.

(We also discovered a confound: `cupti-python` pinned an older PyTorch version, so part of our A/B difference was a framework version difference. Always control for transitive dependency changes.)

## Finding 4: Dispatch Overhead Breakdown — 22% CPU, 78% GPU

Running a careful decomposition on the same GPU instance gave us this breakdown of the hot path:

| Component | Cost | Share |
|---|---|---|
| Python overhead (dict lookup, shape check, ptr compare) | ~0.3 μs | 1.2% |
| `data_ptr()` calls ×7 | ~0.6 μs | 2.5% |
| C dispatch (slot writes + `SetParams` + `cuGraphLaunch`) | ~3.9 μs | 16% |
| **Total CPU dispatch** | **~5.4 μs** | **22%** |
| **GPU kernel execution** | **~24 μs** | **78%** |

A few other details that emerged: `cuGraphUpload` added redundant overhead — `cuGraphLaunch` handles upload automatically. And skipping `SetParams` while writing directly into the source graph's parameter memory caused a GPU MMU fault (Xid 31), confirming that the exec graph maintains its own separate copy of node parameters.

We also found that PyTorch's caching allocator causes input tensor pointers to alternate between exactly two addresses in an A-B-A-B pattern across benchmark iterations (because clones of the same size recycle freed blocks in FIFO order). Any "skip SetParams if pointers unchanged" optimization is never triggered in practice — but the alternating pattern does mean you could pre-compute both SetParams states and flip between them.

## The Takeaway

Microsecond-level kernel optimization requires equally careful measurement hygiene. GPU instance variance, profiling hooks, and transitive dependency changes can each independently swamp the signal you are trying to measure. Run comparisons sequentially on the same device, strip profiling instrumentation from the measurement path, and audit what your benchmark harness does between iterations.

