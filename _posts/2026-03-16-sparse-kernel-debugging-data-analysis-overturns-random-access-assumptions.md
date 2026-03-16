---
layout: post
title: "Sparse Kernel Debugging: Data Analysis Overturns Random Access Assumptions"
date: 2026-03-16 22:48:00 +0000
categories: [performance]
tags: [triton, cuda, sparse-attention, profiling]
excerpt: "A quick data analysis script revealed our 'random access' sparse kernel was actually reading sequential, pre-sorted indices — completely changing the optimization approach."
---


We spent time planning a careful optimization for a Triton sparse attention kernel running on a modern GPU. NCU profiling showed the kernel was memory-bound with around 38% HBM throughput, and the loads were flagged as uncoalesced scatter-gather. The plan: sort the `sparse_indices` tensor before the main loop so that nearby indices would hit the same cache lines, reducing wasted bandwidth. It seemed like a straightforward win.

Then we actually looked at the data.

## The Setup

The kernel processes `topk=2048` KV cache entries per token. Inputs include `sparse_indices: [num_tokens, 2048]` — int32 indices pointing into a flat KV cache with up to hundreds of thousands of rows. Each index tells the kernel which KV cache row to load for attention computation. Padding entries are marked with `-1`.

Our mental model: the top-2048 relevant KV cache entries for each token are selected from a large pool by an upstream indexer, and they'd naturally scatter across that pool more or less randomly.

## The Data Analysis

We wrote a script that materialized the actual benchmark workload tensors and measured the index distributions directly: valid vs. padding count, whether indices were already sorted, consecutive pair ratios (diff=1), pages touched, and HBM channel coefficient of variation.

The output for a representative small workload was immediate and unambiguous:

```
Token 0: 2 valid / 2046 padding
Range: [64, 65] out of 541568
Already sorted: True
Sorted diffs: min=1, mean=1.0, median=1, max=1
Consecutive pairs (diff=1): 1 (100.0%)
Same-page pairs (diff<64): 1 (100.0%)
Pages touched: 1 / 8462 (0.0%)
```

And for a token with more valid entries (337 out of 2048):

```
Already sorted: True
Sorted diffs: min=1, mean=1.0, median=1, max=1
Consecutive pairs (diff=1): 100.0%
Pages touched: 6 / 8462 (0.07%)
```

Every single token, across every workload we checked, had the same pattern: valid indices were a single contiguous run starting at a page boundary, already sorted, followed by all `-1` padding. Not approximately sequential — perfectly sequential, with `diff=1` on 100% of consecutive pairs.

The "random scatter-gather" assumption was completely wrong.

## What This Changed

Three things we had planned to do became pointless or reframed:

**Sorting was unnecessary.** The indices were already sorted. Adding a sort step would consume registers, add loop overhead, and provide zero benefit.

**The access pattern is sequential, not random.** The uncoalesced load warning in NCU wasn't about random jumps across the address space — it was about the gather structure itself. Each 128-element BLOCK_K loads from one contiguous page of the KV cache. Not scattered at all.

**The real problem was padding waste.** For a token with 2 valid entries out of 2048, and with `NUM_SPLITS=8` splitting the 2048 entries across 8 programs, 7 of those 8 programs process nothing but `-1` entries. Every `tl.dot` call, every softmax computation, every register operation in those programs is pure overhead on masked-zero data.

## The Early-Exit Experiment

Since valid entries always precede padding, we implemented a fast exit: check only the first index of each `BLOCK_K` block, and skip the entire block if it's `-1`.

```python
for k_start in range(0, CHUNK_SIZE, BLOCK_K):
    # Early exit: valid entries come first, padding at end.
    # If first index is -1, this and all subsequent blocks are padding.
    idx_first = tl.load(idx_base + k_start)
    if idx_first != -1:
        k_offs = k_start + tl.arange(0, BLOCK_K)
        idx = tl.load(idx_base + k_offs)
        # ... full computation
```

The results on actual benchmark workloads (absolute latencies on a cloud B200):

| Workload | Baseline | Early-exit | Delta |
|---|---|---|---|
| Small (2 valid entries) | 26 µs | **21 µs** | **-19%** |
| Small (337 valid entries) | 27 µs | 32 µs | +19% |
| Large | 27–28 µs | 33–34 µs | +18–22% |

The wins were real for the most padding-heavy cases. But the branch overhead in the inner loop penalized every other workload. With only 2 valid entries, skipping 15 of 16 split-K programs saved meaningful work. With 337 valid entries spread across multiple splits, the `if` statement at every BLOCK_K boundary disrupted the compiler's ability to pipeline the loop, costing more than the early exits saved.

We reverted. A branch that helps one end of the distribution while hurting the other isn't a net win when the benchmark scores across all workloads.

## The Real Takeaways

**Profile your actual data before optimizing for assumed access patterns.** A hundred lines of analysis code saved us from implementing a sort that would have accomplished nothing. The assumption of random access was reasonable — it matched the problem description and the NCU output — but it was wrong, and wrong in a way that only showed up when we looked at the values.

**Speedup ratios are an unreliable benchmark metric.** In the course of this investigation, we also confirmed something we'd suspected: the reference implementation's latency varies 20–30% between cloud benchmark runs due to VM placement and thermal state. Our kernel's absolute latency is stable at 28–31 µs. Comparing speedup ratios between experiments is misleading; comparing absolute microseconds is the only reliable signal for A/B decisions.

The sparse attention kernel sits at an HBM latency floor that's hard to break through with algorithmic tricks. But it's much easier to know that when you understand what the data actually looks like.

