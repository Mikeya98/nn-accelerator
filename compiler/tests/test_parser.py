"""
Unit tests for ONNX parser — uses onnx.helper to construct synthetic models.
"""

import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from compiler.parser import parse_onnx
from compiler.ir import Graph, Node, Tensor


def _save_and_parse(graph_def) -> Graph:
    """Helper: save a GraphProto to a temp file and parse it."""
    model = helper.make_model(
        graph_def,
        producer_name="test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        path = f.name
    try:
        return parse_onnx(path)
    finally:
        os.unlink(path)


def _make_tensor_value_info(name, shape, elem_type=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, elem_type, shape)


# ════════════════════════════════════════════════════════════════════

class TestParseRelu:
    def test_simple_relu(self):
        x = _make_tensor_value_info("x", [1, 10])
        y = _make_tensor_value_info("y", [1, 10])
        node = helper.make_node("Relu", inputs=["x"], outputs=["y"])
        graph_def = helper.make_graph([node], "relu_test", [x], [y])

        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 1
        n = graph.nodes[0]
        assert n.op_type == "Relu"
        assert n.inputs[0].name == "x"
        assert n.outputs[0].name == "y"
        assert n.outputs[0].shape == (1, 10)


class TestParseGemm:
    def test_simple_gemm(self):
        x = _make_tensor_value_info("x", [1, 64])
        y = _make_tensor_value_info("y", [1, 32])

        w_data = np.random.randn(64, 32).astype(np.float32)
        b_data = np.random.randn(32).astype(np.float32)

        w_init = helper.make_tensor("W", TensorProto.FLOAT, [64, 32], w_data.tobytes(), raw=True)
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [32], b_data.tobytes(), raw=True)

        node = helper.make_node(
            "Gemm",
            inputs=["x", "W", "B"],
            outputs=["y"],
            alpha=1.0,
            beta=1.0,
            transB=0,
        )

        graph_def = helper.make_graph(
            [node], "gemm_test", [x], [y],
            initializer=[w_init, b_init],
        )

        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op_type == "Gemm"
        assert graph.nodes[0].outputs[0].shape == (1, 32)

        # Check initializers loaded
        assert "W" in graph.initializers
        assert "B" in graph.initializers
        np.testing.assert_array_almost_equal(graph.initializers["W"].data, w_data)
        np.testing.assert_array_almost_equal(graph.initializers["B"].data, b_data)


class TestParsePool:
    def test_maxpool(self):
        x = _make_tensor_value_info("x", [1, 3, 32, 32])
        y = _make_tensor_value_info("y", [1, 3, 16, 16])

        node = helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        )

        graph_def = helper.make_graph([node], "pool_test", [x], [y])
        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 1
        n = graph.nodes[0]
        assert n.op_type == "MaxPool"
        assert n.outputs[0].shape == (1, 3, 16, 16)
        assert n.attrs["kernel_shape"] == [2, 2]


class TestParseShapeOps:
    def test_reshape(self):
        x = _make_tensor_value_info("x", [1, 100])
        y = _make_tensor_value_info("y", [10, 10])

        shape_data = np.array([10, 10], dtype=np.int64)
        shape_init = helper.make_tensor(
            "shape", TensorProto.INT64, [2], shape_data.tobytes(), raw=True
        )

        node = helper.make_node("Reshape", inputs=["x", "shape"], outputs=["y"])

        graph_def = helper.make_graph(
            [node], "reshape_test", [x], [y],
            initializer=[shape_init],
        )
        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 1
        n = graph.nodes[0]
        assert n.is_shape_op
        assert n.outputs[0].shape == (10, 10)

    def test_transpose(self):
        x = _make_tensor_value_info("x", [1, 3, 32, 32])
        y = _make_tensor_value_info("y", [32, 32, 3, 1])

        node = helper.make_node(
            "Transpose", inputs=["x"], outputs=["y"],
            perm=[2, 3, 1, 0],
        )

        graph_def = helper.make_graph([node], "transpose_test", [x], [y])
        graph = _save_and_parse(graph_def)

        assert graph.nodes[0].is_shape_op
        assert graph.nodes[0].outputs[0].shape == (32, 32, 3, 1)


class TestParseChain:
    """FC → ReLU → FC chain."""

    def test_fc_relu_chain(self):
        x = _make_tensor_value_info("x", [1, 64])
        h = _make_tensor_value_info("h", [1, 32])
        y = _make_tensor_value_info("y", [1, 16])

        w1 = np.random.randn(64, 32).astype(np.float32)
        b1 = np.random.randn(32).astype(np.float32)
        w2 = np.random.randn(32, 16).astype(np.float32)
        b2 = np.random.randn(16).astype(np.float32)

        w1_init = helper.make_tensor("W1", TensorProto.FLOAT, [64, 32], w1.tobytes(), raw=True)
        b1_init = helper.make_tensor("B1", TensorProto.FLOAT, [32], b1.tobytes(), raw=True)
        w2_init = helper.make_tensor("W2", TensorProto.FLOAT, [32, 16], w2.tobytes(), raw=True)
        b2_init = helper.make_tensor("B2", TensorProto.FLOAT, [16], b2.tobytes(), raw=True)

        fc1 = helper.make_node("Gemm", inputs=["x", "W1", "B1"], outputs=["fc1_out"])
        relu = helper.make_node("Relu", inputs=["fc1_out"], outputs=["h"])
        fc2 = helper.make_node("Gemm", inputs=["h", "W2", "B2"], outputs=["y"])

        graph_def = helper.make_graph(
            [fc1, relu, fc2], "chain_test", [x], [y],
            initializer=[w1_init, b1_init, w2_init, b2_init],
        )
        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 3
        assert graph.nodes[0].op_type == "Gemm"
        assert graph.nodes[1].op_type == "Relu"
        assert graph.nodes[2].op_type == "Gemm"

        # Shape propagation through chain
        assert graph.nodes[0].outputs[0].shape == (1, 32)  # FC1 output
        assert graph.nodes[1].outputs[0].shape == (1, 32)  # ReLU output
        assert graph.nodes[2].outputs[0].shape == (1, 16)  # FC2 output


class TestParseConstant:
    """Constant nodes should be folded at compile time."""

    def test_constant_node(self):
        """Constant node creates a tensor with the right shape and data."""
        x = _make_tensor_value_info("x", [1, 3, 32, 32])
        y = _make_tensor_value_info("y", [1, 512])

        # Create a Constant node that produces the target shape for Reshape
        shape_data = np.array([1, 512], dtype=np.int64)
        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_shape"],
            value=helper.make_tensor(
                name="const_val",
                data_type=TensorProto.INT64,
                dims=[2],
                vals=shape_data.tobytes(),
                raw=True,
            ),
        )
        reshape_node = helper.make_node(
            "Reshape",
            inputs=["x", "const_shape"],
            outputs=["y"],
        )

        graph_def = helper.make_graph(
            [const_node, reshape_node], "const_test", [x], [y],
        )
        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 2

        # Constant node should be a shape op
        const_n = graph.nodes[0]
        assert const_n.op_type == "Constant"
        assert const_n.is_shape_op
        assert const_n.outputs[0].shape == (2,)

        # Reshape node should see the folded shape
        reshape_n = graph.nodes[1]
        assert reshape_n.op_type == "Reshape"
        assert reshape_n.is_shape_op
        assert reshape_n.outputs[0].shape == (1, 512)

    def test_scalar_constant(self):
        """Constant with a scalar value."""
        x = _make_tensor_value_info("x", [1, 10])
        y = _make_tensor_value_info("y", [1, 10])

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scale"],
            value=helper.make_tensor(
                name="scale_val",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=np.array([2.0], dtype=np.float32).tobytes(),
                raw=True,
            ),
        )
        mul_node = helper.make_node("Mul", inputs=["x", "scale"], outputs=["y"])

        graph_def = helper.make_graph(
            [const_node, mul_node], "scalar_const_test", [x], [y],
        )
        graph = _save_and_parse(graph_def)

        const_n = graph.nodes[0]
        assert const_n.op_type == "Constant"
        assert const_n.is_shape_op
        assert const_n.outputs[0].shape == ()


class TestParseGRU:
    def test_simple_gru(self):
        seq_len, batch, input_size = 5, 1, 64
        hidden_size = 32

        x = _make_tensor_value_info("x", [seq_len, batch, input_size])
        y = _make_tensor_value_info("y", [seq_len, 1, batch, hidden_size])

        W_data = np.random.randn(1, 3 * hidden_size, input_size).astype(np.float32)
        R_data = np.random.randn(1, 3 * hidden_size, hidden_size).astype(np.float32)
        B_data = np.random.randn(1, 6 * hidden_size).astype(np.float32)

        W_init = helper.make_tensor(
            "W", TensorProto.FLOAT, [1, 3 * hidden_size, input_size],
            W_data.tobytes(), raw=True,
        )
        R_init = helper.make_tensor(
            "R", TensorProto.FLOAT, [1, 3 * hidden_size, hidden_size],
            R_data.tobytes(), raw=True,
        )
        B_init = helper.make_tensor(
            "B", TensorProto.FLOAT, [1, 6 * hidden_size],
            B_data.tobytes(), raw=True,
        )

        gru = helper.make_node(
            "GRU",
            inputs=["x", "W", "R", "B"],
            outputs=["y"],
            hidden_size=hidden_size,
        )

        graph_def = helper.make_graph(
            [gru], "gru_test", [x], [y],
            initializer=[W_init, R_init, B_init],
        )
        graph = _save_and_parse(graph_def)

        assert len(graph.nodes) == 1
        n = graph.nodes[0]
        assert n.op_type == "GRU"
        assert n.outputs[0].shape == (seq_len, 1, batch, hidden_size)
        assert n.attrs["hidden_size"] == hidden_size


class TestUnsupportedOp:
    def test_unsupported_op_raises(self):
        x = _make_tensor_value_info("x", [1, 10])
        y = _make_tensor_value_info("y", [1, 10])
        node = helper.make_node("FancyOp", inputs=["x"], outputs=["y"])
        graph_def = helper.make_graph([node], "bad_test", [x], [y])

        with pytest.raises(NotImplementedError, match="FancyOp"):
            _save_and_parse(graph_def)
