---
layout: post
title: "Bit-Exact GPU TopK: When relu, sum, and Padding All Bite You Differently"
date: 2026-03-15 14:51:02 +0000
categories: [debugging]
tags: [CUDA, PyTorch, precision, GPU]
excerpt: "We discovered three independent precision traps in a GPU FP8 TopK indexer — PyTorch's unreplicable reduction tree, zero-padding enabling batched bmm, and ATen relu vs. custom CUDA relu — and fixed them to achieve 2.5x more speedup with 128/128 bit-exact results."
---

## The Problem

We were building an FP8 TopK indexer kernel: dequant a paged FP8 K cache, run `Q @ K.T`, apply relu + weighted sum, then pick the top-2048 indices. The benchmark requires **bit-exact** `topk_indices` — even a single floating-point difference that shuffles which index lands in the top-K counts as a failure. Our custom CUDA kernel kept failing roughly 40–60 workloads out of 128, even when the numeric differences were tiny (max abs diff ~1e-5). Three separate root causes turned out to be responsible, each requiring a different fix.

---

## The Investigation

### Trap 1: PyTorch's reduction tree is opaque

Our first instinct was to fuse the `relu + multiply by weights + sum(dim=0)` step into a single CUDA kernel, avoiding a round-trip to global memory. We wrote a warp-reduction kernel and tested several accumulation strategies against `torch.sum(dim=0)` on a `[64, N]` float32 tensor:

- Sequential accumulation (`h = 0..63`)
- Reverse sequential (`h = 63..0`)
- Pairwise tree reduction

All three failed. Roughly 70–80% of output elements differed, with max absolute differences around 1e-5. PyTorch's ATen `sum` kernel uses an internal GPU reduction tree with a specific thread/block tiling that we could not replicate from outside. This is a hard constraint: **you cannot match ATen `sum` bit-exactly from a custom CUDA kernel without literally duplicating its implementation.**

Fix: keep `torch.sum` untouched.

### Trap 2: Custom CUDA relu differs from ATen relu

With the sum fixed, we still had failures. We isolated the remaining culprit systematically:

1. Identity kernel (copy input to output) — **pass**
2. relu-only kernel — **fail**
3. multiply-only kernel — **pass**

Step 2 pinpointed relu. We had tried both common patterns:

```cuda
// Variant A
float v = ...; output[i] = (v > 0.0f) ? v : 0.0f;

// Variant B
float v = ...; output[i] = fmaxf(v, 0.0f);
```

Both produced different bit patterns than ATen's `relu_()`. The differences were subtle — only values very close to zero shifted — but enough to change which indices land in the top-K.

Fix: keep `torch.relu_()` (ATen) for the relu step; a custom CUDA kernel for the elementwise multiply alone passed bit-exactly.

### Trap 3: Zero-padding K enables batched bmm — but stale buffer data was the real culprit

Our sequence lengths varied across batch items, which previously forced a Python loop of per-item `torch.mm` calls. We tried batching with `torch.bmm` after zero-padding shorter K matrices:

```python
# Pad each item's K to [max_seq, D] with zeros, stack into [B, max_seq, D]
scores = torch.bmm(Q, K_padded.transpose(1, 2))  # [B, 64, max_seq]
scores = scores[:, :, :sl]  # trim to actual seq_len per item
```

Initial attempts failed. After investigation, the failure was **not** caused by the padding zeros affecting matmul precision — zero-padded rows contribute exactly zero to the dot product and match `torch.mm(Q, K_orig.T)` bit-exactly on the hardware we tested. The actual culprit was **stale garbage data** beyond each item's `seq_len` in a shared buffer that was being reused across items without clearing. Once we zeroed the padding region explicitly, `bmm` passed 128/128.

This distinction matters: zero-padding is safe for precision, but you must ensure the padded region is actually zero and not leftover data.

---

## The Solution

The final pipeline:

```python
# 1. Dequant FP8 K cache → float32 (vectorized CUDA kernel, warp-level uint32 loads)
dequant_fp8_kernel(k_index_cache_fp8, k_float32_padded)  # zero-pads beyond seq_len

# 2. Batched matmul — single bmm call, no Python loop
scores = torch.bmm(q_float32, k_float32_padded.transpose(1, 2))  # [B, 64, max_seq]

# 3. ATen relu — cannot be replaced
torch.relu_(scores)

# 4. Custom CUDA kernel: elementwise multiply by weights (this is safe to fuse)
weighted_mul_kernel(scores, weights)  # [B, 64, max_seq] *= weights[B, 64]

# 5. ATen sum — cannot be replaced
agg = scores.sum(dim=1)  # [B, max_seq]

# 6. topk
topk_indices = agg.topk(2048).indices
```

The result: **128/128 passing with abs_err=0**, and a jump from 2.36x to 5.93x average speedup over the PyTorch baseline, primarily from eliminating the per-item Python matmul loop.

---

## The Takeaway

When a benchmark requires bit-exact output from a GPU pipeline, the debugging methodology matters as much as the fix: isolate each operation individually (identity → relu-only → mul-only), never assume two mathematically equivalent implementations produce the same bits, and remember that ATen's internal reduction kernels are implementation details you cannot safely replicate from custom CUDA code.

