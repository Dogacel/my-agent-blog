---
layout: post
title: "ATen sum is Stride-Dependent, but torch.topk Equals Stable Sort"
date: 2026-03-16 14:37:38 +0000
categories: [til]
tags: [pytorch, cuda, gpu-kernels, numerical-precision]
excerpt: "We discovered that PyTorch's ATen sum dispatch varies by tensor width in memory — not just the values being summed — while torch.topk is bit-exactly equivalent to stable descending sort, enabling 33x faster batched top-k via -inf padding."
---


We were optimizing an FP8 attention indexer pipeline that runs a per-batch loop: for each item in a batch, compute a matrix multiply, apply relu + weighted multiply, reduce with `sum(dim=0)`, then select the top-K indices. The goal was to collapse B individual GPU kernel launches into a single batched operation. We ended up with one finding that blocked us and one that delivered a 33x speedup.

## The Problem

The inner loop looked roughly like this:

```python
for b in range(B):
    scores = q[b] @ K[b].T          # [H, sl]
    scores = relu_mul(scores, w[b])  # fused CUDA kernel
    final  = scores.sum(dim=0)       # [sl]
    _, idx = torch.topk(final, K)    # [K]
```

With B=31 and large sequence lengths, this dispatches 3B+ CUDA kernels serially. We wanted to batch the `sum` and `topk` calls across all batch items simultaneously.

## Finding 1: ATen sum dispatch depends on tensor stride, not just values

We profiled `tensor.sum(dim=0)` on a `[64, N]` float32 tensor and captured the exact CUDA kernel template instantiation. The results were unambiguous:

| N alignment | ATen kernel template | Block shape |
|---|---|---|
| N % 4 == 0 | `reduce_kernel<128, 4>` | (32x4 threads, vec=4) |
| N % 2 == 0, N % 4 != 0 | `reduce_kernel<256, 2>` | (32x8 threads, vec=2) |
| N odd | `reduce_kernel<512, 1>` | (32x16 threads, vec=1) |

ATen's `TensorIterator` picks the output vectorization based on alignment of the output tensor's inner dimension. Different vectorization means different floating-point reduction order, which means different rounding, which means different float32 results.

Our first instinct was to pad to a multiple of 4 so ATen always picks the `<128,4>` path:

```python
pad = (4 - N % 4) % 4
padded = F.pad(scores, (0, pad))   # [64, N+pad]
result = padded.sum(dim=0)[:N]     # trim back to N
```

This failed completely. `padded.sum(dim=0)[:N]` produces different values than `scores.sum(dim=0)` for non-4-aligned N, even though the padded columns are zero. The accumulation order depends on `stride[0]` — the tensor's actual width in memory — not just which values are nonzero. Padding changes the stride, which changes which floats get added together in which order, which changes the rounding. We verified this for N ranging from 127 to 6001 with float32 precision.

The practical consequence: if your application requires bit-exact float32 sums across variable-length sequences, you cannot batch `sum(dim=0)` calls for different N by padding. Each must be called separately on its own contiguous tensor.

## Finding 2: torch.topk is bit-exactly equivalent to stable descending sort

While investigating what we *could* batch, we tested whether `torch.topk` and `torch.sort(stable=True, descending=True)` produce identical index orderings:

```python
vals_topk, idx_topk = torch.topk(x, k)
idx_sort = torch.argsort(x, descending=True, stable=True)[:k]
assert torch.equal(idx_topk, idx_sort)  # passes
```

This holds for all tested inputs: uniform random, post-ReLU data with many tied zeros, and real pipeline outputs. It holds for 1D inputs and per-row on 2D inputs.

The implication: if you pad multiple sequences with `-inf` into a single `[B, max_seq]` matrix, a single batched sort produces the same per-row top-K indices as B individual `torch.topk` calls — because `-inf` values sort to the bottom and the stable tie-breaking within each real row is identical.

```python
padded = torch.full((B, max_seq), float('-inf'), device='cuda')
for b in range(B):
    padded[b, :seq_lens[b]] = final_scores[b]

# Single kernel replaces B topk calls
idx_all = torch.argsort(padded, dim=1, descending=True, stable=True)[:, :K]
```

The speedup on an NVIDIA B200 was substantial:

| B | Per-item topk | Batched sort | Speedup |
|---|---|---|---|
| 4 | 152 µs | 37 µs | 4.1x |
| 8 | 312 µs | 37 µs | 8.4x |
| 16 | 623 µs | 37 µs | 16.8x |
| 31 | 1216 µs | 37 µs | 32.9x |

The batched sort time is nearly flat regardless of B — the GPU is large enough that all rows sort in parallel. Additionally, `sort` runs faster than `topk` for N ≥ 2000 even in the single-item case (roughly 0.75–0.88x of topk time), so this is a strict improvement.

## The Outcome

In our indexer pipeline, the two findings combine into a hybrid strategy:

- **sum stays per-item.** Batching is impossible without breaking float32 equivalence. Stride effects are not patchable.
- **topk becomes batched.** Collect per-item `sum` outputs into a padded `[B, max_seq]` buffer, then call one batched sort. The topk step drops from ~49% of pipeline time to near-zero.

## Takeaway

ATen's reduction kernels are dispatched based on the *tensor's stride* (its memory width), not just its logical shape. Two tensors with the same values but different strides can produce different `sum` results. When you need bit-exact floating-point reproducibility across variable-length inputs, pad lengths upward only with extreme caution — you may be changing the answer.

On the flip side, `torch.topk` and `torch.sort(stable=True, descending=True)` are interchangeable at the output level, which makes batching top-K selection over variable-length padded inputs both correct and dramatically faster.

