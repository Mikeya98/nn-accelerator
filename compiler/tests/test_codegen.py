"""
Unit tests for instruction code generation.
"""

import numpy as np
import pytest

from compiler.ir import Graph, Tensor, Node
from compiler.codegen import generate, Instruction
from compiler.isa import (
    OP_FC, OP_RELU, OP_POOL, OP_END, OP_GRU, OP_ELEM_MUL, OP_ELEM_ADD,
    FLAG_HAS_BIAS, FLAG_POOL_MAX,
    INSTRUCTION_SIZE,
)


def _make_tensor(name, shape, data=None, offset=None):
    t = Tensor(name=name, shape=shape, dtype="float32", data=data)
    t.offset = offset
    return t


class TestCodegenFC:
    def test_simple_fc(self):
        x = _make_tensor("x", (1, 64), offset=0)
        w = _make_tensor("W", (64, 32), data=np.ones((64, 32), dtype=np.float32), offset=0)
        b = _make_tensor("B", (32,), data=np.zeros(32, dtype=np.float32), offset=w.size_bytes)
        y = _make_tensor("y", (1, 32), offset=256)

        node = Node(name="fc1", op_type="Gemm", inputs=[x, w, b], outputs=[y])
        graph = Graph(inputs=[x], outputs=[y], nodes=[node], workspace_size=1024)

        result = generate(graph)

        assert len(result.instructions) >= 2  # FC + END
        fc_instr = result.instructions[0]
        assert fc_instr.opcode == OP_FC
        assert fc_instr.flags & FLAG_HAS_BIAS
        assert fc_instr.input0_addr == 0
        assert fc_instr.weight_addr == 0
        assert fc_instr.bias_addr == w.size_bytes
        assert fc_instr.output_addr == 256
        assert fc_instr.dim0 == 64
        assert fc_instr.dim1 == 32

        # Last instruction must be END
        assert result.instructions[-1].opcode == OP_END


class TestCodegenRelu:
    def test_simple_relu(self):
        x = _make_tensor("x", (1, 100), offset=0)
        y = _make_tensor("y", (1, 100), offset=400)

        node = Node(name="relu1", op_type="Relu", inputs=[x], outputs=[y])
        graph = Graph(inputs=[x], outputs=[y], nodes=[node], workspace_size=1024)

        result = generate(graph)

        relu_instr = result.instructions[0]
        assert relu_instr.opcode == OP_RELU
        assert relu_instr.input0_addr == 0
        assert relu_instr.output_addr == 400
        assert relu_instr.dim0 == 100


class TestCodegenPool:
    def test_maxpool(self):
        x = _make_tensor("x", (1, 3, 32, 32), offset=0)
        y = _make_tensor("y", (1, 3, 16, 16), offset=4096)

        node = Node(
            name="pool1",
            op_type="MaxPool",
            inputs=[x],
            outputs=[y],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
        graph = Graph(inputs=[x], outputs=[y], nodes=[node], workspace_size=8192)

        result = generate(graph)

        pool_instr = result.instructions[0]
        assert pool_instr.opcode == OP_POOL
        assert pool_instr.flags & FLAG_POOL_MAX
        assert pool_instr.dim0 == 32  # H
        assert pool_instr.dim1 == 32  # W
        assert pool_instr.dim2 == 3   # C
        assert pool_instr.dim3 == 2   # KH
        assert pool_instr.dim4 == 2   # KW
        assert pool_instr.dim5 == 2   # stride

    def test_avgpool(self):
        x = _make_tensor("x", (1, 3, 32, 32), offset=0)
        y = _make_tensor("y", (1, 3, 16, 16), offset=4096)

        node = Node(
            name="pool1",
            op_type="AveragePool",
            inputs=[x],
            outputs=[y],
            attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
        graph = Graph(inputs=[x], outputs=[y], nodes=[node], workspace_size=8192)

        result = generate(graph)

        pool_instr = result.instructions[0]
        assert pool_instr.opcode == OP_POOL
        assert not (pool_instr.flags & FLAG_POOL_MAX)  # avg pool → flag clear


class TestCodegenElementWise:
    def test_mul(self):
        a = _make_tensor("a", (100,), offset=0)
        b = _make_tensor("b", (100,), offset=400)
        c = _make_tensor("c", (100,), offset=800)

        node = Node(name="mul1", op_type="Mul", inputs=[a, b], outputs=[c])
        graph = Graph(inputs=[a], outputs=[c], nodes=[node], workspace_size=1200)

        result = generate(graph)
        instr = result.instructions[0]
        assert instr.opcode == OP_ELEM_MUL
        assert instr.input0_addr == 0
        assert instr.input1_addr == 400
        assert instr.output_addr == 800
        assert instr.dim0 == 100

    def test_add(self):
        a = _make_tensor("a", (100,), offset=0)
        b = _make_tensor("b", (100,), offset=400)
        c = _make_tensor("c", (100,), offset=800)

        node = Node(name="add1", op_type="Add", inputs=[a, b], outputs=[c])
        graph = Graph(inputs=[a], outputs=[c], nodes=[node], workspace_size=1200)

        result = generate(graph)
        instr = result.instructions[0]
        assert instr.opcode == OP_ELEM_ADD


class TestCodegenShapeOpSkip:
    def test_reshape_skipped(self):
        x = _make_tensor("x", (1, 100), offset=0)
        y = _make_tensor("y", (10, 10), offset=0)

        reshape_node = Node(name="reshape", op_type="Reshape", inputs=[x], outputs=[y], is_shape_op=True)
        relu_node = Node(name="relu", op_type="Relu", inputs=[y], outputs=[_make_tensor("z", (10, 10), offset=0)])

        graph = Graph(
            inputs=[x],
            outputs=[relu_node.outputs[0]],
            nodes=[reshape_node, relu_node],
            workspace_size=1024,
        )

        result = generate(graph)

        # Only ReLU + END, Reshape skipped
        assert len(result.instructions) == 2
        assert result.instructions[0].opcode == OP_RELU


class TestCodegenGRU:
    def test_gru_generates_packed_weights(self):
        seq_len, batch, input_size, hidden_size = 3, 1, 64, 32
        H = hidden_size

        x = _make_tensor("x", (seq_len, batch, input_size), offset=0)
        y = _make_tensor("y", (seq_len, 1, batch, hidden_size), offset=768)

        W_data = np.random.randn(1, 3 * H, input_size).astype(np.float32)
        R_data = np.random.randn(1, 3 * H, H).astype(np.float32)
        B_data = np.random.randn(1, 6 * H).astype(np.float32)

        W = _make_tensor("W", (1, 3 * H, input_size), data=W_data, offset=0)
        R = _make_tensor("R", (1, 3 * H, H), data=R_data, offset=W.size_bytes)
        B = _make_tensor("B", (1, 6 * H), data=B_data, offset=W.size_bytes + R.size_bytes)

        node = Node(
            name="gru1",
            op_type="GRU",
            inputs=[x, W, R, B],
            outputs=[y],
            attrs={"hidden_size": hidden_size},
        )
        graph = Graph(inputs=[x], outputs=[y], nodes=[node], workspace_size=1024)

        result = generate(graph)

        assert len(result.instructions) == 2  # GRU + END
        gru_instr = result.instructions[0]
        assert gru_instr.opcode == OP_GRU
        assert gru_instr.dim0 == input_size
        assert gru_instr.dim1 == hidden_size
        assert gru_instr.dim2 == batch

        # Extra weights created
        assert len(result.extra_weights) == 2  # packed weight + packed bias

        # Packed weight: W_ih[3H*I] ++ W_hh[3H*H] = 3*32*64 + 3*32*32
        packed_w = result.extra_weights[0]
        assert packed_w.num_elements == 3 * H * input_size + 3 * H * H

        # Packed bias: B_ih[3H] ++ B_hh[3H] = 6*H
        packed_b = result.extra_weights[1]
        assert packed_b.num_elements == 6 * H

        # Verify gate reordering
        # Original ONNX W: shape [1, 3*H, I], gate order z,r,n
        # After reorder: r,z,n
        # So the first H rows of packed_w should be the r_gate from ONNX W
        # (which was at rows H:2H in the original)
        onnx_W_z = W_data[0, 0:H, :]      # z gate
        onnx_W_r = W_data[0, H:2*H, :]    # r gate
        onnx_W_n = W_data[0, 2*H:3*H, :]  # n gate

        packed_w_data = packed_w.data.reshape(-1, input_size + H)
        # Actually the packed weight is 1-D. Let me reshape properly.
        # W_ih section: [3*H, I], then W_hh section: [3*H, H]
        W_ih_section = packed_w.data[:3*H*input_size].reshape(3*H, input_size)
        W_hh_section = packed_w.data[3*H*input_size:].reshape(3*H, H)

        # First gate in W_ih should be r_gate
        np.testing.assert_array_equal(W_ih_section[0:H, :], onnx_W_r)
        # Second gate should be z_gate
        np.testing.assert_array_equal(W_ih_section[H:2*H, :], onnx_W_z)
        # Third gate should be n_gate
        np.testing.assert_array_equal(W_ih_section[2*H:3*H, :], onnx_W_n)
