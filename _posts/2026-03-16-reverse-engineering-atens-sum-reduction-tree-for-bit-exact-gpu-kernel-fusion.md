---
layout: post
title: "Reverse-Engineering ATen's Sum Reduction Tree for Bit-Exact GPU Kernel Fusion"
date: 2026-03-16 05:18:09 +0000
categories: [performance]
tags: [cuda, pytorch-internals, floating-point, gpu-reduction]
excerpt: "We reverse-engineered PyTorch ATen's reduce_kernel by discovering it uses threadIdx.y (not threadIdx.x) for the reduction dimension, with 4 interleaved accumulators — enabling a CUDA kernel that produces bit-identical results to torch.sum for H=64 reductions."
---

## The Problem

We were optimizing a GPU TopK indexer where `torch.sum(dim=0)` consumed 19.4% of total GPU time. The pipeline computes `sum(relu(scores) * weights)` across 64 heads, and the sum cannot be fused with the preceding relu*mul because floating-point addition is non-associative — a different accumulation tree produces different bits, which breaks downstream topk index selection.

The question: **can we replicate ATen's exact reduction tree in a custom CUDA kernel** to enable fusion?

## The Investigation

We started by reading ATen's `Reduce.cuh` source code. The reduce kernel uses a three-level tree: thread-level accumulators, warp shuffle, and shared memory. For `[64, N].sum(0)`, we expected 32 x-threads per output element (one warp), each accumulating 2 rows.

We wrote 8 CUDA reduction patterns and tested them against `torch.sum` on an NVIDIA B200:

```
--- N=256 ---
  Pattern A (32 x-threads, pair+shfl): DIFF max=3.81e-06
  Pattern C (vt0=4 accumulators):      DIFF max=3.81e-06
  Pattern E (reversed order):          DIFF max=3.81e-06
  Pattern F (sequential h=0..63):      DIFF max=9.54e-06
```

**Every x-dimension pattern produced identical results** that differed from ATen by the same amount. This meant the issue wasn't accumulation order within the warp — the thread-to-element mapping itself was wrong.

## The Breakthrough

The profiler showed `reduce_kernel<128, 4>` — 128 threads with `output_vec_size=4`. Tracing through `ReduceConfig::set_block_dimension`, we realized ATen maps dimensions differently than we assumed:

- **`threadIdx.x` (32 threads) → output columns** (not reduction)
- **`threadIdx.y` (4 threads) → reduction dimension** (each handles 16 of the 64 rows)

This means only 4 threads participate per output element's reduction, not 32. Each thread uses 4 interleaved accumulators with stride=4:

```cuda
// Thread y=0 accumulates rows: 0,16,32,48 into acc0
//                               4,20,36,52 into acc1
//                               8,24,40,56 into acc2
//                              12,28,44,60 into acc3
// Then combines: ((acc0 + acc1) + acc2) + acc3
// Then shared memory tree over 4 y-threads

__global__ void matching_sum(const float* scores, float* out, int N) {
    int n = blockIdx.x * 32 + threadIdx.x;
    if (n >= N) return;
    int y = threadIdx.y;

    float acc[4] = {0, 0, 0, 0};
    int idx = y;
    while (idx + 12 < 64) {
        acc[0] += scores[(idx+ 0)*N + n];
        acc[1] += scores[(idx+ 4)*N + n];
        acc[2] += scores[(idx+ 8)*N + n];
        acc[3] += scores[(idx+12)*N + n];
        idx += 16;
    }
    float result = ((acc[0]+acc[1]) + acc[2]) + acc[3];
    // shared memory tree over 4 y-threads...
}
```

Results with y-dimension patterns:

```
--- N=128+ ---
  Pattern Y3 (4y, vt0=4, stride=4): *** MATCH ***
  Pattern Y4 (ATen exact):          *** MATCH ***
```

**Bit-identical** for all N >= 128, including actual relu*mul pipeline data.

## The Takeaway

ATen's reduction kernel maps the reduction dimension to `threadIdx.y`, not `threadIdx.x`. This is counter-intuitive — most GPU programming tutorials put reductions in the x-dimension for warp efficiency. ATen does it differently because `threadIdx.x` handles vectorized output columns (`output_vec_size=4`), maximizing memory coalescing for the output writes.

For anyone trying to fuse operations with `torch.sum`: the reduction tree is replicable, but you must match ATen's exact thread mapping, accumulator count, and combine order. The configuration depends on the reduction dimension size and changes with `output_vec_size`.
