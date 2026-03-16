---
layout: post
title: "Why Sorting Sparse Indices for Memory Coalescing Made Our Kernel 2.4–3x Slower"
date: 2026-03-16 20:17:58 +0000
categories: [performance]
tags: [gpu, triton, sparse-attention, memory-coalescing]
excerpt: "Sorting sparse attention indices to improve DRAM coalescing backfired badly — the fix destroyed split-K load balance and eliminated cross-SM L2 cache sharing, making the kernel 2.4–3x slower despite reducing uncoalesced loads."
---


GPU profiling taught us a painful lesson: a fix that looks obviously correct in isolation can be disastrously wrong in context.

## The Setup

We're running a fused split-K sparse attention kernel in Triton on an NVIDIA B200. The kernel takes `sparse_indices` — a `[num_tokens, 2048]` int32 tensor of random page indices into the KV cache — and scatter-gathers 512-dimensional key vectors for each token. Nsight Compute flagged ~40% wasted bytes from uncoalesced global memory loads, exactly what you'd expect from random scatter-gather access.

The proposed fix was obvious: sort `sparse_indices` before launching the kernel so that adjacent indices map to adjacent cache lines, coalescing those scattered loads into sequential bursts.

The sort itself happens once on the cold path and is free during benchmark timing (the CUDA graph replays with already-sorted data). The dot products `q @ K^T` and `p @ K` are order-independent, so reordering K rows is mathematically valid. This should be a free win.

## The Experiment

We pre-sorted a copy of the index tensor before CUDA graph capture:

```python
sorted_idx = torch.empty_like(sparse_indices)
sort_idx_tmp = torch.empty_like(sparse_indices, dtype=torch.int64)
torch.sort(sparse_indices, dim=-1, out=(sorted_idx, sort_idx_tmp))

def launch_fn():
    _splitk_fused[(T, NUM_SPLITS)](
        q_nope, q_pe, Kc_flat, Kp_flat, sorted_idx,  # <-- sorted
        ...
    )
```

Then we benchmarked against the original unsorted version across multiple workloads.

| Workload | Original | Sorted Indices | Slowdown |
|----------|----------|----------------|----------|
| Small T  | 0.027 ms (41x) | 0.081 ms (14x) | **3.0x slower** |
| Medium T | 0.029 ms (47x) | 0.078 ms (19x) | **2.7x slower** |
| Large T  | 0.030 ms (96x) | 0.073 ms (41x) | **2.4x slower** |

The sorted version was between 2.4x and 3x slower. The coalescing optimization made things dramatically worse.

## Why It Backfired

Three mechanisms combined to destroy performance.

**1. Split-K load imbalance.** The kernel splits the 2048 topk indices across `NUM_SPLITS` parallel programs (2–8, depending on token count). With unsorted indices, each split gets a random subset of pages — all splits finish at roughly the same time. With sorted indices, split 0 gets the lowest page indices (plus all the `-1` padding tokens), while the last split gets the highest. The padding tokens are trivial no-ops, so split 0 races ahead and idles while the other splits are still processing. The semaphore-based last-finisher reduction can't proceed until every split completes. What looked like better memory access turned into a severe load imbalance stall.

**2. Eliminated cross-SM L2 sharing.** With random indices, multiple tokens often reference overlapping sets of pages. Different SMs processing different tokens hit the same L2 cache lines — accidental but real reuse. With sorted indices, each token's split accesses a disjoint sorted range. The access patterns are now perfectly non-overlapping, eliminating any cross-SM cache reuse that was happening for free.

**3. Memory controller hot-spots.** With all tokens sorted independently, every SM simultaneously requests low-address pages first, then mid-range, then high. This creates temporal hot-spots on specific HBM memory controllers — all 148 SMs hammering the same addresses at the same moment — while the rest of the memory subsystem sits idle. Random access naturally spreads load across all memory controllers evenly.

## The Takeaway

The 40% uncoalesced-load penalty flagged by Nsight Compute was real, but it was the *cost* of a random access pattern that provided other benefits: natural load balancing across splits, cross-SM L2 reuse from overlapping page sets, and even DRAM controller utilization. Sorting removed the symptom while destroying the properties that made the kernel efficient.

When optimizing GPU kernels, global memory access patterns don't exist in isolation — they interact with parallelism structure (split-K balance), cache hierarchy (L2 sharing between SMs), and memory subsystem architecture (controller distribution). A change that improves one metric in a microbenchmark can cascade into failures across all three.

The random scatter-gather pattern, despite looking wasteful to a profiler, was doing useful work.

