"""
Unit tests for the memory planner.
"""

import numpy as np
import pytest

from compiler.ir import Graph, Tensor, Node
from compiler.memory_planner import plan_memory


def _make_tensor(name, shape, data=None):
    return Tensor(name=name, shape=shape, dtype="float32", data=data)


class TestMemoryPlanner:
    def test_simple_linear_chain(self):
        """A → B → C: each tensor has non-overlapping lifetime."""
        x = _make_tensor("x", (1, 64))
        h = _make_tensor("h", (1, 32))
        y = _make_tensor("y", (1, 16))

        w1 = _make_tensor("W1", (64, 32), data=np.ones((64, 32), dtype=np.float32))
        w2 = _make_tensor("W2", (32, 16), data=np.ones((32, 16), dtype=np.float32))

        n1 = Node(name="fc1", op_type="Gemm", inputs=[x, w1], outputs=[h])
        n2 = Node(name="fc2", op_type="Gemm", inputs=[h, w2], outputs=[y])

        graph = Graph(
            inputs=[x],
            outputs=[y],
            nodes=[n1, n2],
            initializers={"W1": w1, "W2": w2},
        )

        plan_memory(graph)

        # All runtime tensors should have offsets
        assert x.offset is not None
        assert h.offset is not None
        assert y.offset is not None

        # h and x can share memory (their lifetimes don't overlap)
        # x is used at node 0, h is used at nodes 0 and 1
        # Since both are live at node 0, they should NOT overlap
        # But h's first write is at node 0, so they could share if
        # the allocator detects that x is dead after node 0 ends.
        # (Basic test: all offsets are valid, no out-of-bounds)
        assert x.offset + x.size_bytes <= graph.workspace_size
        assert h.offset + h.size_bytes <= graph.workspace_size
        assert y.offset + y.size_bytes <= graph.workspace_size

    def test_weights_packed_contiguously(self):
        """Weights are laid out back-to-back."""
        x = _make_tensor("x", (1, 4))
        y = _make_tensor("y", (1, 3))

        w = _make_tensor("W", (4, 3), data=np.ones((4, 3), dtype=np.float32))
        b = _make_tensor("B", (3,), data=np.zeros(3, dtype=np.float32))

        n = Node(name="fc", op_type="Gemm", inputs=[x, w, b], outputs=[y])

        graph = Graph(
            inputs=[x],
            outputs=[y],
            nodes=[n],
            initializers={"W": w, "B": b},
        )

        plan_memory(graph)

        assert w.offset is not None
        assert b.offset is not None
        assert b.offset == w.offset + w.size_bytes
        assert graph.weight_size == w.size_bytes + b.size_bytes

    def test_workspace_size_is_positive(self):
        x = _make_tensor("x", (1000,))
        y = _make_tensor("y", (1000,))
        n = Node(name="relu", op_type="Relu", inputs=[x], outputs=[y])
        graph = Graph(inputs=[x], outputs=[y], nodes=[n])

        plan_memory(graph)
        assert graph.workspace_size >= 4000  # at least 1000 × 4 bytes
