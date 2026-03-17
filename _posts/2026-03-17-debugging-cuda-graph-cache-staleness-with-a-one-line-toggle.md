---
layout: post
title: "Debugging CUDA Graph Cache Staleness with a One-Line Toggle"
date: 2026-03-17 19:16:38 +0000
categories: [debugging]
tags: [CUDA, Triton, GPU, debugging]
excerpt: "When a CUDA graph is captured once and replayed across different benchmark workloads, stale kernel parameters can silently corrupt results — we added a minimal no-cache toggle to isolate whether pointer updates were actually taking effect."
---

CUDA graphs are a powerful tool for reducing dispatch overhead: capture a kernel launch once, then replay it cheaply by updating only the pointer arguments that change. But that caching logic introduces a subtle failure mode — **what if the cache is serving a stale graph when the inputs have fundamentally changed?**

## The Problem

Our sparse attention kernel caches captured CUDA graphs in a dict keyed by the number of tokens `T`:

```python
_state: dict[int, GraphDispatch] = {}

def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
           sm_scale, output, lse):
    T = q_nope.shape[0]
    st = _state.get(T)
    if st is None:
        st = GraphDispatch(launch_fn=lambda: _launch(...), changing_arg_indices=[0,1,2,3,4,6,7])
        _state[T] = st
    st.replay([q_nope.data_ptr(), ...])
```

On replay, the dispatcher calls `cuGraphExecKernelNodeSetParams` to patch in the new pointers before launching. In theory, every call with the same `T` gets a fresh set of pointers and correct results. In practice, we wanted to **verify** that the pointer patching was actually working across benchmark test cases — especially when the same `T` value appears in multiple workloads with different cache contents.

## The Investigation

The concern was straightforward: `_state` persists across the lifetime of the process, so a graph captured for `T=4` during one test case gets replayed for the next `T=4` test case. The replay path updates 7 pointer arguments (`q_nope`, `q_pe`, `ckv_cache`, `kpe_cache`, `sparse_indices`, `output`, `lse`), which should be enough to fully redirect all memory accesses. But if the pointer update was missing an argument, or the graph node extraction had an off-by-one, results would silently be wrong.

The first instinct was to test with an environment variable:

```python
_NO_GRAPH_CACHE = os.environ.get("DSA_NO_GRAPH_CACHE", "0") == "1"
```

But the kernel runs on a remote GPU worker — environment variables set locally don't propagate to the remote process. The fix needed to live entirely in the Python source.

## The Solution

We added a single boolean flag at module level with a comment explaining its purpose:

```python
# Toggle to True to disable graph caching (fresh capture every call, for debugging)
_NO_GRAPH_CACHE = True

def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
           sm_scale, output, lse):
    T = q_nope.shape[0]
    if _NO_GRAPH_CACHE:
        _state.pop(T, None)   # evict before lookup → forces fresh capture
    st = _state.get(T)
    ...
```

Setting `_NO_GRAPH_CACHE = True` forces a fresh CUDA graph capture on every single call. This is deliberately slow — graph capture involves a full CUDA stream synchronization and kernel trace — but it eliminates the cache as a variable entirely. If results are correct with the flag on and wrong with it off, the cache is the culprit. If both modes produce identical results, the pointer update logic is sound.

The `_state.pop(T, None)` before `_state.get(T)` pattern is intentional: pop the entry if it exists, then fall through to the `None` branch that rebuilds it. It avoids duplicating the build logic and keeps the hot path unchanged when the flag is `False`.

## The Takeaway

When debugging a caching layer that involves raw memory pointers, the fastest path to a definitive answer is a **one-line cache bypass** that forces ground-truth behavior. Trying to reason about whether seven pointer updates are all correct is far harder than just measuring whether disabling the cache changes the output.

The same pattern applies broadly: any time you have a "fast path" optimization (cache, precomputed state, lazy initialization), add a debug toggle that forces the slow but obviously-correct path. Keep it in the source, not as a command-line flag, when the code runs in an environment you don't fully control.

