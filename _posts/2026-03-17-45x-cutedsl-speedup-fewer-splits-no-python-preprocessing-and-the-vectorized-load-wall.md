---
layout: post
title: "4.5x CuTeDSL Speedup: Fewer Splits, No Python Preprocessing, and the Vectorized Load Wall"
date: 2026-03-17 19:06:20 +0000
categories: [performance]
tags: [cuda, cutedsl, gpu-kernels, attention]
excerpt: "We systematically improved a CuTeDSL sparse attention kernel from 9.6x to 43.6x speedup by tuning split count from 32 to 64 and eliminating Python preprocessing — then hit a hard wall trying to vectorize scatter-gather loads through swizzled SMEM."
---


## The Problem

After getting our CuTeDSL sparse attention kernel to compile and produce correct outputs (all 23 workloads passing), the initial numbers were disappointing: 9.6x speedup at 118us latency, versus the Triton baseline at 84x and ~29us. The kernel architecture was sound — dual-kernel split-K with cuLaunchKernel dispatch via a C shim — but we were leaving a lot of performance on the table. We needed to understand exactly where the time was going.

## The Investigation

We started with profiling instinct: the kernel was launching `NUM_SPLITS * T` compute blocks, with `NUM_SPLITS=32` meaning 32 blocks per token, each doing 2 iterations over `BLOCK_K=32` keys. That is a lot of small blocks. Two costs compound here: first, each block has fixed overhead (SMEM allocation, register file setup, barrier synchronization), and second, more blocks means more scheduling pressure on the SM dispatcher.

The second bottleneck was visible in the Python-side hot path. Before every kernel call, we were doing:

```python
st['mask_buf'].copy_((sparse_indices.view(-1) >= 0).float())
st['idx_buf'].copy_(sparse_indices.view(-1).clamp(min=0))
```

Two `.copy_()` calls, a `.view()`, a boolean comparison, a `.float()` cast, and a `.clamp()` — all synchronizing on the Python side before the C dispatch shim could fire. For small T workloads this preprocessing was costing roughly 20-30us on its own.

## The Solution

**Step 1: Move index handling into the kernel.**

Instead of preprocessing -1 padding indices in Python, we added a guard directly in the gather loop:

```python
# Inside @cute.kernel
for k_idx in cutlass.range(BLOCK_K):
    raw_idx = s_idx[(k_idx,)]
    if raw_idx >= 0:
        ckv_base = raw_idx * D_CKV
        for i_dim in cutlass.range_constexpr(D_CKV // THREADS):
            d_off = tidx + i_dim * THREADS
            sKn[(k_idx, d_off)] = ckv[(ckv_base + d_off,)]
    # else: sKn row stays zeroed from SMEM init
```

This eliminated both `.copy_()` calls and all the Python-side tensor operations from the hot path. The cost moved onto GPU threads that are already running, and the branch is near-perfectly predicted since padding indices cluster at the end of the topk list.

**Step 2: Tune `NUM_SPLITS`.**

We swept from 32 down to 8 and up to 64. The relationship was non-linear:

| NUM_SPLITS | CHUNK_SIZE | Iters/block | Avg Latency | Speedup |
|-----------|-----------|-------------|-------------|---------|
| 32 | 64 | 2 | 81us | 13.9x |
| 8 | 256 | 8 | 64us | 34.5x |
| 64 | 32 | 1 | **53us** | **43.6x** |

The sweet spot was `NUM_SPLITS=64` — exactly one `BLOCK_K=32` iteration per compute block. This minimizes per-block overhead while maximizing the number of independent blocks the SM scheduler can overlap. At this setting, small T workloads hit **26us — faster than the Triton baseline's 29us**.

The reduce kernel runs as 16 independent blocks (one per head), which is cheap compared to the compute phase.

## The Vectorized Load Wall

With 43.6x on the board, we investigated the remaining 2x gap versus Triton. NCU told us the story: Triton generates `ld.global.v4.b32` (16-byte vectorized loads), while our CuTeDSL kernel generates `ld.global.b16` (2-byte scalar loads). That is an 8x difference in load throughput per instruction.

We tried three approaches to close this gap:

**Adjacent-element loads.** Each thread loads two consecutive bf16 values hoping the compiler fuses them:

```python
d_base = tidx * 2
sKn[(k_idx, d_base)]     = ckv[(ckv_base + d_base,)]
sKn[(k_idx, d_base + 1)] = ckv[(ckv_base + d_base + 1,)]
```

Result: no change. The CuTeDSL compiler does not auto-vectorize adjacent bf16 stores to swizzled SMEM.

**Non-swizzled K layout.** Replace the swizzled `tile_to_shape` layout with a plain row-major layout for the K matrix:

```python
sKn_ly = cute.make_layout((BLOCK_K, D_CKV), stride=(D_CKV, 1))
```

Result: runtime error. The MMA copy atoms (`s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16)`) require the SMEM source to have a swizzled layout that matches the MMA tile's expected bank access pattern. A plain row-major layout is incompatible.

**i32 recast to swizzled SMEM.** Cast the bf16 pointer to i32 and write 32-bit values:

```python
sKn_i32 = cute.recast_ptr(sKn_ptr, BFloat16, cutlass.Int32)
sKn_i32[(k_idx, d_off // 2)] = ...  # pack two bf16
```

Result: incorrect outputs. The swizzle function scrambles the physical address mapping in a way that is consistent for same-type element access but breaks when the element width changes. Writing i32 to a bf16-swizzled layout writes to wrong locations.

The root cause is architectural: CuTeDSL's composed swizzle layouts are defined relative to the element type. A `make_swizzle(3, 3, 3)` layout over bf16 addresses different banks than the same swizzle over i32. There is no clean way to load wider types into a swizzled bf16 SMEM layout without either breaking the layout contract or staging through a flat intermediate — and the staging overhead exceeds the vectorization gain.

Triton avoids this because `tl.load` with a 2D pointer block handles the full scatter-gather-vectorize operation in one compiler pass, before SMEM layout decisions are made.

## The Takeaway

For split-K attention kernels in CuTeDSL, the highest-leverage optimizations are (in order): eliminate Python preprocessing by moving index guards into the kernel, then tune `NUM_SPLITS` so each block does exactly one tile's worth of work. The vectorized load gap versus Triton is real but currently inaccessible — it requires either a CuTeDSL compiler improvement for swizzled-SMEM stores or a layout redesign that accepts the MMA bank-conflict penalty. For large-T workloads the scalar load bottleneck remains; for small-T, the lower block-launch overhead of the CuTeDSL path already wins.

