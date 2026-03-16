---
layout: post
title: "What NCU Taught Us About Triton's Scatter-Gather Codegen (And Why Our Optimizations Backfired)"
date: 2026-03-16 21:50:41 +0000
categories: [performance]
tags: [triton, gpu, performance, cuda]
excerpt: "We ran Nsight Compute on a Triton sparse attention kernel and discovered it already generates async pipelined scatter-gather loads — but three targeted optimizations (removing masks, adding pipeline stages, combining both) all made things worse."
---


We had a Triton sparse attention kernel running at ~88x speedup over a PyTorch baseline. The inner loop does scatter-gather loads from a paged KV cache using random indices — exactly the kind of pattern everyone warns you about. We suspected there was more performance on the table. We were wrong, and NCU showed us why.

## The Setup

The kernel processes DeepSeek-style sparse attention: for each token, it loads 2048 KV entries from random locations in a paged cache, computes attention scores with online softmax, and accumulates an output. The inner loop looks roughly like this:

```python
for k_start in range(0, CHUNK_SIZE, BLOCK_K):
    k_offs = k_start + tl.arange(0, BLOCK_K)
    idx = tl.load(idx_base + k_offs)          # load sparse indices
    valid = (idx != -1)
    safe_idx = tl.where(valid, idx, tl.zeros_like(idx))

    kc_ptrs = CKV + safe_idx[:, None].to(tl.int64) * D_CKV + d_ckv[None, :]
    k_nope = tl.load(kc_ptrs, mask=valid[:, None], other=0.0)  # scatter-gather

    scores = tl.dot(q_nope, tl.trans(k_nope))
    # ... softmax, output accumulation ...
```

We profiled this with NCU on a Blackwell B200 (148 SMs) for two configurations: T=4 tokens (small) and T=64 tokens (large).

## What NCU Actually Found

**Speed-of-light summary:**

| Metric | T=4 | T=64 |
|--------|-----|------|
| SM (compute) throughput | 3.3% | 17.7% |
| Memory throughput | 7.1% | 38.1% |
| L1/TEX throughput | 53% | 48% |
| Occupancy | 12.5% | 12.5% |
| Waves per SM | 0.22 | 0.86 |

The numbers look alarming. But then we checked the PTX.

**The surprise: Triton was already pipelining the scatter-gather.**

We wrote a standalone benchmark to check what load instructions the compiler generated:

```python
# Simple gather (no surrounding loop):    ld.global.v4.b32         — blocking
# Full kernel pattern (loop + tl.dot):    cp.async.ca.shared.global — async!
```

When the gather loop surrounds a `tl.dot` with `num_stages=3`, Triton generates `cp.async.ca.shared.global` — asynchronous copies from HBM to shared memory. The next iteration's KV data is being fetched while the current iteration runs tensor core MMA. This is the same pattern a hand-written CUDA pipeline would use.

**The catch: 4-byte copy width.** Each async copy moves only 4 bytes:

```
cp.async.ca.shared.global [ %r5 + 0 ], [ %rd16 + 0 ], 0x4, %r174
                                                         ^^^
```

The hardware supports 4, 8, and 16-byte `cp.async`. The compiler chose 4-byte because the `mask=` parameter forces per-element predicates, preventing wider vectorization. We hypothesized that removing the mask might let the compiler emit 16-byte copies — a potential 4x reduction in instruction count for the same data.

## Three Experiments, Three Failures

We ran three experiments, each in an isolated worktree, benchmarked against 3 representative workloads.

**Experiment A — Remove load masks:**

Instead of `tl.load(kc_ptrs, mask=valid[:, None], other=0.0)`, clamp the index to 0 and rely on the existing `tl.where` on scores to zero out invalid entries.

Result: **8–16% slower** across all workloads. Passes correctness (the score masking covers it), but Triton still generates 4-byte copies because scatter addresses are non-contiguous regardless of masking. Worse, it now loads garbage data for padding rows, wasting bandwidth on real (but useless) HBM traffic.

**Experiment B — Increase `num_stages` from 3 to 5:**

More pipeline stages → more requests in flight → better latency hiding.

Result: **RUNTIME_ERROR on all workloads.** Shared memory exhaustion. Each pipeline stage buffers `[BLOCK_K=128, D_CKV=512]` of bfloat16 = 128KB. At `num_stages=5` we blow past the 228KB shared memory limit. There's no room to breathe.

**Experiment C — Both combined:**

Same runtime error — the shared memory wall doesn't care that we removed the masks.

## The Stage Timing Reveal

We also ran a stage-level dissection: variants that stop after loading, after score dots, after softmax, after output accumulation. The result was clarifying:

| Variant | T=1 | T=4 | T=16 | T=64 |
|---------|-----|-----|------|------|
| Load only | 30 µs | 31 µs | 31 µs | 30 µs |
| + score dots | 27 µs | 33 µs | 30 µs | 25 µs |
| + softmax | 21 µs | 33 µs | 32 µs | 44 µs |
| + output dot | 26 µs | 33 µs | 34 µs | 51 µs |
| Full kernel | 34 µs | 31 µs | 30 µs | **54 µs** |

At T=1 through T=16, adding more compute *does not increase wall time* — sometimes it even decreases it, because more surrounding work gives the pipeliner more to overlap. Everything hides behind the 30 µs HBM latency floor.

At T=64, the kernel breaks through: 54 µs total vs. 30 µs memory floor, meaning ~24 µs (44%) of compute is *not* hidden by pipelining. That's where the compute utilization in NCU comes from.

## What the 12.5% Occupancy Actually Means

NCU flags low occupancy (12.5% — 1 block per SM) due to high register count (~158/thread) and 171 KB of shared memory per block. The tool suggests improving occupancy.

This is a red herring. Adding more blocks per SM would just create more warps all stalled on the same random HBM requests. Random scatter-gather through a 75 MB working set (T=64) thrashes L2 (64 MB on B200) regardless of how many warps are waiting. Higher occupancy doesn't increase effective random-access bandwidth.

The 38% memory throughput at T=64 isn't 38% of peak — it's the realistic ceiling for a workload touching ~151 MB of random HBM per invocation in 86 µs.

## The Takeaway

When profiling a kernel that does scatter-gather over a large working set:

1. **Check the PTX before assuming the compiler is naive.** Triton's `num_stages` pipeline generates `cp.async` for gather loops too, not just sequential tile loads.
2. **Low occupancy is not always the problem.** For random-access memory-bound kernels, more warps share the same bottleneck — they don't hide each other's latency.
3. **Mask removal can backfire.** Even if masks aren't the reason for narrow copies, removing them increases real bandwidth consumption on invalid entries.
4. **The memory floor is physics.** A 30 µs floor for 2048-entry random gather on HBM is set by DRAM latency and access pattern, not by instruction scheduling choices. Software can't eliminate it — only hide it with enough surrounding compute.

The 88x speedup was already capturing nearly everything available on this hardware for this problem shape. The lesson: NCU's recommendations are starting points, not prescriptions. Always check whether the constraint it's flagging is actually addressable before spending engineering time on it.

