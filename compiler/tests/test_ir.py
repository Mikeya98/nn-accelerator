"""
Unit tests for compiler IR data structures.
"""

import pytest
import numpy as np
from compiler.ir import Tensor, Node, Graph


class TestTensor:
    def test_runtime_tensor(self):
        t = Tensor(name="x", shape=(1, 3, 224, 224))
        assert t.name == "x"
        assert t.shape == (1, 3, 224, 224)
        assert t.is_runtime
        assert not t.is_constant
        assert t.num_elements == 1 * 3 * 224 * 224
        assert t.size_bytes == 1 * 3 * 224 * 224 * 4
        assert t.offset is None

    def test_constant_tensor(self):
        data = np.random.randn(64, 128).astype(np.float32)
        t = Tensor(name="W", shape=(64, 128), data=data)
        assert t.is_constant
        assert not t.is_runtime
        assert t.num_elements == 64 * 128
        assert t.size_bytes == 64 * 128 * 4
        np.testing.assert_array_equal(t.data, data)

    def test_scalar_tensor(self):
        t = Tensor(name="scalar", shape=())
        assert t.num_elements == 1
        assert t.size_bytes == 4

    def test_dynamic_dim(self):
        t = Tensor(name="dynamic", shape=(-1, 256))
        assert t.num_elements == 0  # unknown dim → 0

    def test_offset_assignable(self):
        t = Tensor(name="x", shape=(10,))
        t.offset = 1024
        assert t.offset == 1024


class TestNode:
    def test_simple_node(self):
        x = Tensor(name="x", shape=(10,))
        w = Tensor(name="w", shape=(10, 5), data=np.ones((10, 5), dtype=np.float32))
        y = Tensor(name="y", shape=(5,))
        node = Node(
            name="fc1",
            op_type="Gemm",
            inputs=[x, w],
            outputs=[y],
            attrs={"transB": 0},
        )
        assert node.name == "fc1"
        assert node.op_type == "Gemm"
        assert len(node.inputs) == 2
        assert len(node.outputs) == 1
        assert node.attrs["transB"] == 0

    def test_shape_op_flag(self):
        node = Node(name="reshape", op_type="Reshape", is_shape_op=True)
        assert node.is_shape_op


class TestGraph:
    def test_empty_graph(self):
        g = Graph()
        assert len(g.nodes) == 0
        assert len(g.inputs) == 0
        assert g.workspace_size == 0

    def test_graph_with_nodes(self):
        x = Tensor(name="x", shape=(1, 10))
        w = Tensor(name="w", shape=(10, 5), data=np.ones((10, 5), dtype=np.float32))
        y = Tensor(name="y", shape=(1, 5))
        node = Node(name="fc", op_type="Gemm", inputs=[x, w], outputs=[y])
        g = Graph(
            inputs=[x],
            outputs=[y],
            nodes=[node],
            initializers={"w": w},
            workspace_size=1024,
            weight_size=200,
        )
        assert g.inputs[0] is x
        assert g.outputs[0] is y
        assert len(g.nodes) == 1
        assert g.initializers["w"] is w
        assert g.workspace_size == 1024
        assert g.weight_size == 200
