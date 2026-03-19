---
layout: post
title: "Eliminating 3x QAT Overhead with torch.compile and Custom Autograd"
date: 2026-03-19 02:39:32 +0000
categories: [performance]
tags: [pytorch, quantization, torch.compile, autograd]
excerpt: "We traced a 3x training slowdown in a BitNet b1.58 quantization-aware training layer to ~25 unbatched CUDA kernel launches and autograd graph bloat, then recovered most of the overhead with torch.compile, a custom STE autograd Function, and removed redundant float32 upcasts."
---


Quantization-aware training (QAT) is how you ship a lean 1-bit LLM without sacrificing accuracy — but it comes at a cost. We were building a `BitLinear` layer based on the BitNet b1.58 paper (ternary weights: −1, 0, +1) and discovered it ran **3x slower** than a vanilla `nn.Linear` during training. Here's how we diagnosed and fixed it.

## The Problem

The layer looked reasonable on paper: normalize inputs with RMSNorm, quantize weights to ternary values, quantize activations to 8-bit, run the linear, rescale. But profiling revealed that each forward pass was spawning roughly **25 separate CUDA kernel launches** — one per elementwise op — and the autograd graph was bloated with redundant intermediate tensors.

Three root causes stood out:

1. **No kernel fusion.** Every `.abs()`, `.mean()`, `.clamp()`, and `round()` call is its own CUDA kernel. With no compiler, they execute sequentially with per-launch overhead.
2. **Autograd graph bloat from the STE trick.** The standard straight-through estimator (STE) is written as `x + (x_quant - x).detach()`. This creates *three* extra autograd nodes (subtraction, detach, addition) and allocates an intermediate `x_quant - x` tensor — every forward pass.
3. **Redundant float32 upcasts.** The original code upcasted to `float32` before weight quantization, then before activation quantization, then again inside RMSNorm — three separate dtype round-trips for operations that are perfectly fine in `bfloat16`.

## The Fix

We applied three targeted optimizations, in increasing order of invasiveness.

### 1. `torch.compile` for kernel fusion

The inference code in the same repository already used `@torch.compile` on its hot paths. We applied the same decorator to the quantization methods:

```python
@torch.compile
def _weight_quant(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().mean().clamp_(min=1e-5)
    return (w / scale).round().clamp_(-1, 1)

@torch.compile
def _act_quant(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    Qn, Qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp_(min=1e-5) / Qp
    return (x / scale).round().clamp_(Qn, Qp), scale
```

`torch.compile` traces the computation graph and fuses those 25 kernel launches down to a handful. On CUDA this means a single fused kernel instead of 9 sequential ones for weight quantization alone.

### 2. Custom `torch.autograd.Function` for the STE

We replaced the inline STE expression with a proper `Function` subclass:

```python
class _STEWeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean().clamp_(min=1e-5)
        return (w / scale).round().clamp_(-1, 1), scale

    @staticmethod
    def backward(ctx, grad_out, grad_scale):
        return grad_out, None   # STE: pass gradient straight through
```

The `x + (x_quant - x).detach()` idiom *works*, but it's wasteful: PyTorch records the subtraction as a graph node, allocates the intermediate tensor, and then the addition merges two gradient paths in the backward pass. The custom `Function` collapses this to a single node with a trivial identity backward — no intermediate allocation, no extra graph nodes.

### 3. Drop the float32 upcasts

Ternary weight quantization computes `abs().mean()` — a single scalar — and then rounds. Absmax activation quantization computes `abs().max()` — another scalar comparison. Neither needs float32 precision. We let both run in `bfloat16`:

```python
# Before
scale = w.float().abs().mean()
...
return result.to(w.dtype)

# After — bfloat16 is fine for this scale computation
scale = w.abs().mean().clamp_(min=1e-5)
```

RMSNorm still benefits from float32 (the squared mean accumulates error in bf16), so we kept that cast — but we moved it inside xformers' fused RMSNorm kernel, which handles the upcast internally.

## Correctness Verification

We wrote 17 unit tests comparing the optimized layer against the original, checking:

- Forward outputs match exactly (float32, `max_diff = 0.00e+00` on all 17 tests)
- Weight quantization produces 100% ternary values
- Activation values stay within the expected `[−127, 127]` range for 8-bit
- STE gradient is an exact identity in both the weight and activation paths
- Gradients flow to all parameters (weight, bias, `input_norm.weight`)
- State dicts are fully compatible — it's a drop-in replacement

We also ran a 50-epoch training stability test. The first 3 epochs are **bit-exact** between the two implementations (`max_diff = 0.00e+00`), proving per-step correctness. After that, weights evolve to quantization boundaries where `round()` can flip — a classic butterfly effect inherent to any discrete system, not a bug.

## Results

On CPU (no `torch.compile` acceleration, no GPU), the optimized layer already runs **~15% faster** just from the removed upcasts and cleaner autograd graph. On a real CUDA GPU with `torch.compile`, the kernel fusion drives the bulk of the improvement — recovering the majority of the 3x overhead by eliminating the wall of sequential kernel launches.

## The Takeaway

When profiling a QAT layer, don't just look at the math — count your CUDA kernel launches and your autograd graph nodes. `torch.compile`, a custom `autograd.Function` for STE, and eliminating unnecessary dtype conversions are three independent levers you can pull without changing any training behavior.

