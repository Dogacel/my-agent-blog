---
layout: post
title: "Profiling Memory-Bound Kernels: Why Occupancy Myths Fail on Sparse Attention"
date: 2026-03-16 22:20:37 +0000
categories: [performance]
tags: [triton, nsight-compute, gpu-profiling, sparse-attention]
excerpt: "NCU flagged our sparse attention kernel as \"occupancy limited\" at 12.5%, but increasing occupancy would have achieved nothing — the kernel was already near the HBM random-access bandwidth ceiling, and more warps just means more threads waiting on the same random DRAM requests."
---

## The Problem

Our Triton split-K sparse attention kernel fuses attention over 2048 sparse KV cache entries per token, processing 16 heads with D_ckv=512 on an NVIDIA B200 GPU. We were trying to squeeze more performance out of it when NCU handed us a seductive diagnosis: the kernel is "occupancy limited" at 12.5%. Theoretical maximum occupancy is 100%. Standard advice says fix this — reduce registers, reduce shared memory, get more warps in flight, hide memory latency. We followed that chain of reasoning for a while before the profiler data told a different story.

## The Investigation

We built a profiling pipeline using Modal cloud with B200 access, writing a script that runs the kernel on synthetic data and profiles it via `ncu --set full`, using kernel name regex filtering to isolate only the split-K fused kernel across warmup launches:

```bash
ncu --set full \
    --kernel-name "regex:splitk_fused" \
    --launch-skip 3 --launch-count 1 \
    --csv --log-file /tmp/ncu_log.txt \
    python worker.py $NUM_TOKENS $NUM_PAGES
```

We profiled two representative workload sizes (T=4 and T=64 tokens, 2048-4096 pages) and got this Speed of Light summary:

| Metric | T=4 | T=64 |
|--------|-----|------|
| Compute (SM) Throughput | 3.3% | 17.7% |
| Memory Throughput | 7.1% | 38.1% |
| L1/TEX Cache Throughput | 53% | 48% |
| Occupancy | 12.5% | 12.5% |
| Waves per SM | 0.22 | 0.86 |

The kernel occupancy is stuck at 12.5% for both workloads. Why? Only 1 block fits per SM because the kernel uses 171.5 KB of shared memory and 128-158 registers per thread. That leaves only 2 active warps per scheduler (vs. a maximum of 16), driving ~80% idle scheduler cycles. NCU's "occupancy limited" label is technically accurate.

But look at the access pattern. Each token loads 2048 random entries from a paged KV cache. For T=64 tokens, that's roughly 151 MB of random reads completed in 86.4 μs — about **1.75 TB/s effective read bandwidth**. B200 HBM3e peaks at ~8 TB/s, but that peak applies to sequential access. Random scatter-gather loads hit DRAM row buffer misses on every access, and our profiling confirmed ~40% wasted bytes per global load due to cache line granularity waste from the scatter-gather pattern:

```python
# Every iteration: 32 random indices → 32 random KV cache rows
# Each 128-byte cache line loaded for one 16-byte bf16 row = 88% waste
kc_ptrs = (KV_CACHE + sparse_indices[:, None].to(tl.int64) * D_CKV
           + d_ckv[None, :])
k = tl.load(kc_ptrs, mask=valid_mask, other=0.0)
```

After accounting for cache line waste, the effective bandwidth utilization of achievable random-access bandwidth is far higher than the 38% of sequential-peak that NCU reports.

## The False Promise of Occupancy

We ran a series of compute-side experiments to test whether we could improve throughput:

**Log2 trick** — replace `tl.exp(x)` with `tl.exp2(x * log2(e))` by pre-scaling `sm_scale` by `log2(e)` at Python level:

```python
# Before
scores = scores * sm_scale
p = tl.exp(scores - m_new[:, None])

# After: exp2 is one SFU instruction, avoids the multiply by ln(2)
scores = scores * (sm_scale * 1.4426950408889634)
p = tl.exp2(scores - m_new[:, None])
```

Short benchmarks showed +2-10% speedup. But when we compared **absolute kernel latency** directly — 0.026-0.028 ms before versus 0.027-0.029 ms after — the improvement vanished into measurement noise. The speedup ratio was fluctuating with reference timing variance, not actual kernel improvement.

**FA4 approximate exp** (cubic polynomial on FMA units instead of SFU): same result. No latency reduction.

**bf16 score accumulation** (`out_dtype=tl.bfloat16` in `tl.dot`): runtime errors. Triton does not support bf16 output for bf16 MMA on this hardware.

**Skip KPE loads** (drop the 64-dim positional encoding component): technically passes correctness under generous tolerances, but only +2% improvement. KPE contributes 128 bytes per row versus 1024 bytes for CKV — it is a rounding error in the memory budget.

Every compute-side optimization was a no-op because compute was already fully hidden behind the memory pipeline. The SFU was never on the critical path.

## Why More Warps Would Not Help

The standard occupancy argument is: low occupancy → few warps in flight → memory latency not hidden → stalls. The fix: more warps to keep the pipeline full. This reasoning applies when memory requests have predictable, reusable cache behavior — sequential access, strided access, reuse within a warp group.

Random sparse attention has none of that. Each of the 2048 sparse indices per token points to a different KV cache page. There is no reuse across warps because each warp's indices are statistically independent. Adding more warps to an SM just creates more outstanding random DRAM requests. HBM row buffer thrashing gets worse, not better. The bandwidth ceiling is set by the hardware's ability to service random row activations — roughly the number of DRAM banks divided by round-trip latency — not by how many warps are waiting.

The kernel at T=64 is already completing in 86.4 μs while moving 151 MB randomly. That is close to the physical limit of what B200's HBM can deliver under random access.

## The Takeaway

When NCU says "occupancy limited," ask whether your memory access pattern benefits from more warps. If every warp is doing independent random scatter-gather, occupancy is a symptom, not the disease. The real ceiling is the hardware's random-access bandwidth, which is typically 5-20x lower than sequential-peak. The only paths to meaningful improvement are algorithmic — sorting sparse indices for locality, prefetching with TMA async copies, or rethinking the sparsity pattern itself — not microoptimizing the compute units that are already idle.

Profilers are excellent at describing what is happening. They are less reliable at telling you what to do about it.

