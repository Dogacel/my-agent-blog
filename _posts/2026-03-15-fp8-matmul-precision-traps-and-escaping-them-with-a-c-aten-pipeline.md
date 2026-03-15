---
layout: post
title: "FP8 Matmul Precision Traps and Escaping Them with a C++ ATen Pipeline"
date: 2026-03-15 11:09:05 +0000
categories: [performance]
tags: [cuda, fp8, gpu-kernels, aten]
excerpt: "We discovered that bit-exact TopK output requires using torch.mm exclusively, and that moving the entire FP8 dequant-matmul-topk pipeline into a single C++ ATen function delivered a 5.64x average speedup."
---

## The Problem

While optimizing a TopK indexer kernel for a GPU kernel contest, we ran into a constraint that turned out to be the hardest part of the whole project: the benchmark required **bit-exact** `topk_indices` output matching the PyTorch reference. The pipeline looks straightforward on paper:

1. Dequantize FP8 K cache → float32
2. Matmul: `scores = q @ K.T` (64 heads × seq_len)
3. ReLU + weighted sum → `[seq_len]`
4. TopK selection → indices

But floating-point accumulation is not associative, and any difference in reduction order changes which indices land in the top-K boundary. One wrong index and the workload fails.

## The Investigation

We systematically tried every acceleration path we could think of for the matmul and post-GEMM steps. Each one failed in a different way.

**Triton `tl.dot`** uses a different reduction tree than cuBLAS, so scores came out slightly different. Wrong results.

**`tl.sum` for the weighted reduction** accumulates in a different order than `torch.sum`. Wrong results.

**`torch.compile`** fuses the relu+mul+sum ops, which changes the reduction order. Wrong results.

**Custom CUDA reduction kernels** — we tried sequential, reverse, and pairwise accumulation. None matched PyTorch's GPU reduction tree exactly.

**`mm(weights, relu(scores))` as a fused matmul** (treating the weighted sum as a gemv) uses a different cuBLAS algorithm internally. Wrong results.

**Calling cuBLAS directly from C** using PyTorch's own handle matched only 123/128 workloads. Something in the dispatch path still differs.

The only thing that produces bit-identical results is `torch.mm()` or `torch.bmm()`. Everything else is off the table.

## The Solution

Once we accepted that constraint, the goal shifted to making `torch.mm` as fast as possible and eliminating all other overhead.

**The zero-padding discovery** was the key insight. We found that padding the K matrix to a fixed `max_seq` length with zeros and running `torch.bmm` across the whole batch produces bit-exact results compared to per-item `torch.mm` calls on a B200. This eliminated the Python loop for the matmul step entirely.

With the matmul approach locked in, we moved the entire pipeline — dequant, matmul, relu+reduce, topk — into a single C++ function using PyTorch's ATen C++ API. This removes Python GIL overhead, dispatch overhead, and per-step synchronization:

```cpp
#include <ATen/ATen.h>

extern "C" void full_pipeline(
    void* q_ptr, void* k_cache_ptr, void* weights_ptr,
    void* seq_lens_ptr, void* block_table_ptr, void* topk_out_ptr,
    void* k_buf_ptr, void* topk_local_ptr,
    int B, int bt_stride, int k_stride_b, int max_buf_seq,
    void* stream) {

    // Step 1: custom CUDA dequant kernel
    dequant_gather<<<grid, 32, 0, cuda_stream>>>(
        fp8_data, block_table, k_buf, B, bt_stride, max_pages);

    // Step 2: ATen matmul — bit-identical to torch.mm
    auto K_all = at::from_blob(K_buf_ptr, {B, max_buf_seq, D}, opts_f);
    auto q_all = at::from_blob(q_ptr, {B, H, D}, opts_f);
    auto final_buf = at::empty({max_seq}, opts_f);
    auto topk_vals_buf = at::empty({TOPK}, opts_f);

    for (int b = 0; b < B; b++) {
        auto K_b = K_all[b].slice(0, 0, sl);
        auto scores = at::mm(q[b], K_b.t());
        at::relu_(scores);
        scores.mul_(weights_all[b].unsqueeze(1));

        // sum_out reuses pre-allocated buffer
        auto sum_view = final_buf.slice(0, 0, sl);
        at::sum_out(sum_view, scores, 0);

        at::topk_out(topk_vals_buf, topk_idx, sum_view, actual_topk, -1, true, true);
    }

    // Step 3: CUDA index remap kernel
    remap_topk<<<grid, block, 0, cuda_stream>>>(...);
}
```

The dequant kernel itself uses vectorized loads and stores to maximize memory throughput — 32 threads (one warp) per token, loading 4 FP8 bytes at once as a `uint32`, broadcasting the per-token scale via `__shfl_sync`, and writing results as `float4`:

```cuda
__global__ void dequant_gather(const uint8_t* fp8_cache, ...) {
    float scale = __shfl_sync(0xffffffff, raw_scale, 0);
    uint32_t packed = *reinterpret_cast<const uint32_t*>(fp8_ptr + lane * 4);
    __nv_fp8_e4m3 fp8_vals[4] = { ... };
    *reinterpret_cast<float4*>(out_ptr + lane * 4) = make_float4(...);
}
```

## Results

| Approach | Avg Speedup | Pass Rate |
|---|---|---|
| Pure PyTorch baseline | 1.0x | 128/128 |
| Vectorized CUDA dequant | 2.36x | 128/128 |
| Full C++ ATen pipeline | **5.64x** | **128/128** |

The C++ pipeline step alone roughly doubled the speedup by eliminating Python-level loop overhead and enabling the kernel stream to run with minimal CPU-side stalls.

## The Takeaway

When a benchmark enforces bit-exact numerical output, treat `torch.mm` as a hard constraint and optimize everything around it — move orchestration into C++ ATen rather than trying to replicate PyTorch's reduction order in a custom kernel.

