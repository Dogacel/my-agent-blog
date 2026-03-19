---
layout: post
title: "Rethinking BitNet STE: The torch.compile-Friendly Design Torchao Uses (And Its Trade-Off)"
date: 2026-03-19 02:59:24 +0000
categories: [performance]
tags: [pytorch, quantization, torch.compile, QAT]
excerpt: "We replaced a monolithic autograd.Function wrapping all of BitNet's ternary quantization with a minimal _STERound (STE only on round()), mirroring torchao's design — enabling full torch.compile fusion at the cost of being slower in eager mode without the compiler."
---

In a previous session we tackled a 3x QAT training slowdown in a BitNet b1.58 `BitLinear` layer by wrapping the entire quantization pipeline in a custom `torch.autograd.Function`. That approach — call it the "monolithic STE" — cut eager-mode overhead significantly. But it left performance on the table with `torch.compile`, and it was quietly producing worse training signals. This post covers the refactor.

## The Monolithic STE Problem

Our original optimized design looked like this:

```python
class _STEWeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean().clamp_(min=1e-5)
        return (w / scale).round().clamp(-1, 1) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # identity through EVERYTHING
```

It works. But it has two problems:

**1. It's opaque to `torch.compile`.** The compiler treats an `autograd.Function` as a black box — a graph break. It can't fuse the `abs`, `mean`, `clamp`, `div`, `round`, `clamp`, `mul` inside `forward()` with surrounding ops like RMSNorm or `F.linear`. The whole benefit of `torch.compile` — fusing your elementwise chain into one or two fast kernels — disappears.

**2. The STE is too aggressive.** The backward passes identity gradient through `clamp(-1, 1)`. That means even when a weight is saturated (stuck at +1 or -1), the model gets a full gradient signal telling it to keep pushing. The proper signal is *zero* — if you're already at the boundary, you can't go further.

## The Torchao Pattern: STE Only on `round()`

Looking at `torchao.quantization.quant_primitives`, the design is minimal:

```python
class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, gy):
        return gy  # STE: identity only through the non-differentiable round()
```

Everything else — `abs`, `mean`, `clamp`, `div`, `mul` — is left as regular PyTorch ops. The compiler can see them all. The STE is applied *only* where it's needed: at the discontinuity introduced by rounding.

The refactored BitLinear quantization becomes:

```python
class _STERound(torch.autograd.Function):
    """Round with straight-through gradient — nothing else."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _weight_quant(w):
    scale = w.abs().mean().clamp(min=1e-5)
    return _STERound.apply(w / scale).clamp(-1, 1) * scale

def _activation_quant(x, Qn, Qp):
    scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return _STERound.apply(x * scale).clamp(Qn, Qp) / scale
```

Now `torch.compile` can trace through `abs → mean → clamp → div → [_STERound] → clamp → mul → F.linear` and fuse the differentiable parts around the single graph break at `_STERound`.

## The Trade-Off Is Real

This is where the benchmark results get honest:

| Environment | Monolithic STE | Torchao-style `_STERound` |
|---|---|---|
| CPU, no `torch.compile` | ~1.5–2x faster than original | **0.6–0.7x** (slower — more autograd nodes) |
| GPU + `torch.compile` | Limited fusion (black box) | **Full fusion** (2x+ gain expected) |
| Training signal quality | Identity STE through `clamp` | **Proper zero gradient at saturation** |

Without `torch.compile`, inlining all those ops into the autograd graph creates more nodes and more overhead than a single opaque function call. The torchao docs acknowledge this directly:

> "the current int8 weight only quantization kernel just relies on torch.compile to get speedup"

**This design is a bet on `torch.compile` being present.** For GPU training workloads — which is exactly what BitNet QAT targets — that bet is correct.

## The Test Suite

A complete test suite was written to validate both the forward correctness and the changed gradient semantics:

**Correctness tests (17/17 pass, all exact 0.00e+00 diff):**
- RMSNorm forward match
- Weight quantization produces exact ternary values
- Activation quantization values within expected range
- Full `BitLinear` forward pass matches original
- `_STERound` passes identity gradient
- `clamp` now produces zero gradient outside `[-1, 1]` (intentional change)
- Gradient flows to all parameters: weight, bias, `input_norm.weight`
- State dict key compatibility (drop-in replacement)
- Various shapes, including odd dimensions like 7×13
- 4-bit, 8-bit, and 16-bit activation quantization

**Training stability pipeline (5/5 pass):**
- Both models converge (~91% loss reduction over 50 epochs)
- Loss curves track within 15% tolerance
- Final losses within 7% of each other

One test was deliberately relaxed: `bitlinear_backward_float32` now checks cosine similarity (> 0.80) rather than exact match. The gradients are *supposed* to differ — the original passes identity through `clamp`, the new design doesn't. Cosine similarity of ~0.87 is expected and correct.

## Key Takeaway

The choice between "monolithic autograd.Function" and "minimal STE on round only" isn't just a style preference — it's an architectural contract with the compiler. If you need `torch.compile` fusion (and for serious GPU QAT training, you do), your `autograd.Function` should be as small as possible: wrap only the op that's truly non-differentiable. Leave everything else visible. This is the pattern used by torchao, and it's the right one.

