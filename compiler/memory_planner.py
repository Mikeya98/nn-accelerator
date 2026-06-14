"""
Memory planner — assigns byte offsets to every tensor in the IR graph.

Two separate pools are managed:

1. **Workspace buffer** (scratchpad)
   Runtime tensors that flow between PL compute instructions.  A greedy
   interval (liveness-based) allocator reuses memory whose lifetimes do
   not overlap, minimising the total workspace footprint.

2. **Weight buffer**
   Constant tensors (weights, biases) are packed contiguously.  Their
   order follows the order they are first referenced in the instruction
   stream, so the PS loader can simply copy a single blob into DDR.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .ir import Graph, Tensor, Node

logger = logging.getLogger(__name__)


def plan_memory(graph: Graph) -> None:
    """Assign ``offset`` to every tensor and set
    ``graph.workspace_size`` / ``graph.weight_size``.

    This mutates the graph in-place.
    """
    # ── Separate constant vs runtime tensors ─────────────────────
    runtime_tensors = _collect_runtime_tensors(graph)
    weight_tensors = _collect_weight_tensors(graph)

    # ── Liveness analysis on runtime tensors ─────────────────────
    intervals = _liveness_analysis(graph, runtime_tensors)

    # ── Greedy allocation for workspace ──────────────────────────
    _greedy_allocate(intervals)

    # ── Pack weights contiguously ────────────────────────────────
    _pack_weights(weight_tensors)

    # ── Write totals back to graph ───────────────────────────────
    max_ws = 0
    for t in runtime_tensors:
        if t.offset is not None:
            end = t.offset + t.size_bytes
            if end > max_ws:
                max_ws = end

    max_w = 0
    for t in weight_tensors:
        if t.offset is not None:
            end = t.offset + t.size_bytes
            if end > max_w:
                max_w = end

    graph.workspace_size = max_ws
    graph.weight_size = max_w

    logger.info(
        "Memory plan: workspace=%d KiB, weight=%d KiB",
        max_ws // 1024, max_w // 1024,
    )


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _collect_runtime_tensors(graph: Graph) -> list[Tensor]:
    """Return all runtime (non-constant) tensors in the graph."""
    seen: set[str] = set()
    result: list[Tensor] = []

    for t in graph.inputs:
        if not t.is_constant:
            if t.name not in seen:
                seen.add(t.name)
                result.append(t)

    for node in graph.nodes:
        for t in node.outputs:
            if not t.is_constant and t.name not in seen:
                seen.add(t.name)
                result.append(t)

    for t in graph.outputs:
        if not t.is_constant and t.name not in seen:
            seen.add(t.name)
            result.append(t)

    return result


def _collect_weight_tensors(graph: Graph) -> list[Tensor]:
    """Return all FP32 constant tensors (weights, biases) in first-reference order.

    Non-FP32 initializers (e.g. int64 shape tensors) are excluded — they
    are compile-time metadata, not runtime weight data.
    """
    weights: list[Tensor] = []
    seen: set[str] = set()

    # Walk nodes in order; record FP32 constants as they are first encountered
    for node in graph.nodes:
        for t in node.inputs:
            if t.is_constant and t.dtype == "float32" and t.name not in seen:
                seen.add(t.name)
                weights.append(t)

    # Also include any unreferenced FP32 initializers
    for name, t in graph.initializers.items():
        if name not in seen and t.dtype == "float32":
            seen.add(name)
            weights.append(t)

    return weights


def _liveness_analysis(
    graph: Graph,
    runtime_tensors: list[Tensor],
) -> list[dict]:
    """Compute [first_def_or_use, last_use] interval for each runtime tensor.

    Returns a list of dicts sorted by start index, each containing
    ``tensor``, ``start``, ``end``.
    """
    # Map tensor name → index
    name_to_tensor = {t.name: t for t in runtime_tensors}

    first: dict[str, int] = {}
    last: dict[str, int] = {}

    n_nodes = len(graph.nodes)

    # Input tensors are "defined" at index 0
    for t in graph.inputs:
        if t.name in name_to_tensor:
            first[t.name] = 0

    for i, node in enumerate(graph.nodes):
        # Outputs are defined at node i
        for t in node.outputs:
            if t.name in name_to_tensor:
                if t.name not in first:
                    first[t.name] = i
                last[t.name] = i  # at least i

        # Inputs are used at node i
        for t in node.inputs:
            if t.name in name_to_tensor:
                if t.name not in first:
                    first[t.name] = i
                last[t.name] = i

    # Output tensors are used up to the end
    for t in graph.outputs:
        if t.name in name_to_tensor:
            last[t.name] = n_nodes  # pseudo "last" edge

    # Build interval list
    intervals = []
    for name, t in name_to_tensor.items():
        if name in first:
            intervals.append({
                "tensor": t,
                "start": first[name],
                "end": last.get(name, first[name]),
            })

    intervals.sort(key=lambda x: (x["start"], x["end"]))
    return intervals


def _greedy_allocate(intervals: list[dict]) -> None:
    """Allocate workspace offsets using a greedy interval allocator.

    Each interval gets the smallest offset that does not overlap with
    any previously allocated live range.

    We maintain a list of ``(free_after_node_index, offset, size)``
    representing freed slots.
    """
    # Align to 4-byte boundaries
    def align4(n: int) -> int:
        return (n + 3) & ~3

    # Active allocations: list of (end_node, offset, size)
    active: list[tuple[int, int, int]] = []
    # Freed slots: list of (end_node, offset, size)
    freed: list[tuple[int, int, int]] = []

    next_free = 0  # next unused offset at the end of the heap

    for item in intervals:
        tensor = item["tensor"]
        start = item["start"]
        size = align4(tensor.size_bytes)

        if size == 0:
            tensor.offset = 0
            continue

        # Expire any active allocs that finished before this interval starts
        expired = [a for a in active if a[0] <= start]
        for a in expired:
            active.remove(a)
            freed.append(a)
        # Also remove stale freed slots
        freed[:] = [f for f in freed if f[0] > start]

        # Try to reuse a freed slot big enough
        placed = False
        freed.sort(key=lambda x: x[1])  # by offset
        for i, (f_end, f_offset, f_size) in enumerate(freed):
            if f_size >= size:
                tensor.offset = f_offset
                active.append((item["end"], f_offset, size))
                # If the freed slot was bigger, split it
                if f_size > size:
                    freed[i] = (f_end, f_offset + size, f_size - size)
                else:
                    freed.pop(i)
                placed = True
                break

        if not placed:
            tensor.offset = next_free
            active.append((item["end"], next_free, size))
            next_free += size


def _pack_weights(weights: list[Tensor]) -> None:
    """Assign contiguous offsets to weight tensors."""
    offset = 0
    for t in weights:
        t.offset = offset
        offset += t.size_bytes
