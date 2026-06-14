"""
End-to-end tests: synthetic ONNX model → BIN file → verification.

Tests the full compiler pipeline:
  ONNX → parse → memory plan → codegen → BIN write
"""

import os
import struct
import tempfile

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from compiler.parser import parse_onnx
from compiler.memory_planner import plan_memory
from compiler.codegen import generate
from compiler.bin_writer import write_bin, HEADER_SIZE, NN_BIN_MAGIC
from compiler.isa import (
    OP_FC, OP_RELU, OP_END, OP_POOL, OP_GRU,
    FLAG_HAS_BIAS, FLAG_POOL_MAX,
    INSTRUCTION_SIZE,
)


def _make_tensor_value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _save_onnx(graph_def) -> str:
    """Save a GraphProto to a temp file, return path."""
    model = helper.make_model(
        graph_def,
        producer_name="test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        return f.name


def _run_pipeline(onnx_path: str, model_name: str = "test") -> tuple[str, bytes]:
    """Run the full compile pipeline, return (bin_path, bin_data)."""
    graph = parse_onnx(onnx_path)
    plan_memory(graph)
    codegen_result = generate(graph)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    write_bin(graph, codegen_result, bin_path, model_name)

    with open(bin_path, "rb") as f:
        data = f.read()

    return bin_path, data


def _parse_header(data: bytes) -> dict:
    fields = struct.unpack_from("<I H H I I I I I I 64s 160s", data, 0)
    return {
        "magic": fields[0],
        "ver_major": fields[1],
        "ver_minor": fields[2],
        "num_instructions": fields[3],
        "weight_size": fields[4],
        "workspace_size": fields[5],
        "input_size": fields[6],
        "output_size": fields[7],
        "checksum": fields[8],
        "model_name": fields[9].rstrip(b"\x00").decode("utf-8"),
    }


def _parse_instruction(data: bytes, idx: int) -> dict:
    offset = HEADER_SIZE + idx * INSTRUCTION_SIZE
    fields = struct.unpack_from("<B B H I I I I I I I I I I I I I I I", data, offset)
    return {
        "opcode": fields[0],
        "flags": fields[1],
        "seq_len": fields[2],
        "input0_addr": fields[3],
        "input1_addr": fields[4],
        "output_addr": fields[5],
        "weight_addr": fields[6],
        "bias_addr": fields[7],
        "workspace_addr": fields[8],
        "scale_addr": fields[9],
        "dim0": fields[10],
        "dim1": fields[11],
        "dim2": fields[12],
        "dim3": fields[13],
        "dim4": fields[14],
        "dim5": fields[15],
        "dim6": fields[16],
        "dim7": fields[17],
    }


def _weights_offset(num_instr: int) -> int:
    return HEADER_SIZE + num_instr * INSTRUCTION_SIZE


# ════════════════════════════════════════════════════════════════════

class TestE2EFC:
    """End-to-end: single FC layer."""

    def test_fc_only(self):
        x = _make_tensor_value_info("x", [1, 4])
        y = _make_tensor_value_info("y", [1, 3])

        w_data = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        b_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        w_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 3], w_data.tobytes(), raw=True)
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [3], b_data.tobytes(), raw=True)

        node = helper.make_node("Gemm", inputs=["x", "W", "B"], outputs=["y"])
        graph_def = helper.make_graph(
            [node], "fc_test", [x], [y],
            initializer=[w_init, b_init],
        )

        onnx_path = _save_onnx(graph_def)
        try:
            bin_path, data = _run_pipeline(onnx_path, "fc_model")

            hdr = _parse_header(data)
            assert hdr["magic"] == NN_BIN_MAGIC
            assert hdr["model_name"] == "fc_model"
            assert hdr["num_instructions"] == 2  # FC + END
            assert hdr["input_size"] == 16  # 1*4*4
            assert hdr["output_size"] == 12  # 1*3*4

            # First instruction: FC
            instr = _parse_instruction(data, 0)
            assert instr["opcode"] == OP_FC
            assert instr["flags"] & FLAG_HAS_BIAS
            assert instr["dim0"] == 4  # M = input features
            assert instr["dim1"] == 3  # N = output features

            # Last instruction: END
            end_instr = _parse_instruction(data, hdr["num_instructions"] - 1)
            assert end_instr["opcode"] == OP_END

            # Verify weights in BIN
            w_offset = _weights_offset(hdr["num_instructions"])
            recovered_w = np.frombuffer(
                data[w_offset:w_offset + 48], dtype=np.float32
            ).reshape(4, 3)
            np.testing.assert_array_almost_equal(recovered_w, w_data)

        finally:
            os.unlink(onnx_path)
            os.unlink(bin_path)


class TestE2EFCReluChain:
    """End-to-end: FC → ReLU → FC chain."""

    def test_fc_relu_fc(self):
        x = _make_tensor_value_info("x", [1, 4])
        y = _make_tensor_value_info("y", [1, 2])

        w1 = np.ones((4, 3), dtype=np.float32) * 0.5
        b1 = np.zeros(3, dtype=np.float32)
        w2 = np.ones((3, 2), dtype=np.float32) * 0.3
        b2 = np.array([0.1, -0.1], dtype=np.float32)

        w1_init = helper.make_tensor("W1", TensorProto.FLOAT, [4, 3], w1.tobytes(), raw=True)
        b1_init = helper.make_tensor("B1", TensorProto.FLOAT, [3], b1.tobytes(), raw=True)
        w2_init = helper.make_tensor("W2", TensorProto.FLOAT, [3, 2], w2.tobytes(), raw=True)
        b2_init = helper.make_tensor("B2", TensorProto.FLOAT, [2], b2.tobytes(), raw=True)

        fc1 = helper.make_node("Gemm", inputs=["x", "W1", "B1"], outputs=["h_pre"])
        relu = helper.make_node("Relu", inputs=["h_pre"], outputs=["h"])
        fc2 = helper.make_node("Gemm", inputs=["h", "W2", "B2"], outputs=["y"])

        graph_def = helper.make_graph(
            [fc1, relu, fc2], "chain_test", [x], [y],
            initializer=[w1_init, b1_init, w2_init, b2_init],
        )

        onnx_path = _save_onnx(graph_def)
        try:
            bin_path, data = _run_pipeline(onnx_path, "chain_model")

            hdr = _parse_header(data)
            # 3 compute ops + END = 4 instructions
            assert hdr["num_instructions"] == 4

            instrs = [_parse_instruction(data, i) for i in range(4)]
            assert instrs[0]["opcode"] == OP_FC    # Gemm 1
            assert instrs[1]["opcode"] == OP_RELU  # ReLU
            assert instrs[2]["opcode"] == OP_FC    # Gemm 2
            assert instrs[3]["opcode"] == OP_END

            # Check address chaining: FC1 output → ReLU input → FC2 input
            fc1_out = instrs[0]["output_addr"]
            relu_in = instrs[1]["input0_addr"]
            relu_out = instrs[1]["output_addr"]
            fc2_in = instrs[2]["input0_addr"]

            assert fc1_out == relu_in, "FC1 output must feed into ReLU input"
            assert relu_out == fc2_in, "ReLU output must feed into FC2 input"

        finally:
            os.unlink(onnx_path)
            os.unlink(bin_path)


class TestE2EPool:
    """End-to-end: MaxPool."""

    def test_maxpool(self):
        x = _make_tensor_value_info("x", [1, 3, 8, 8])
        y = _make_tensor_value_info("y", [1, 3, 4, 4])

        node = helper.make_node(
            "MaxPool", inputs=["x"], outputs=["y"],
            kernel_shape=[2, 2], strides=[2, 2],
        )
        graph_def = helper.make_graph([node], "pool_test", [x], [y])

        onnx_path = _save_onnx(graph_def)
        try:
            bin_path, data = _run_pipeline(onnx_path, "pool_model")

            hdr = _parse_header(data)
            assert hdr["num_instructions"] == 2  # POOL + END

            instr = _parse_instruction(data, 0)
            assert instr["opcode"] == OP_POOL
            assert instr["flags"] & FLAG_POOL_MAX
            assert instr["dim0"] == 8   # H
            assert instr["dim1"] == 8   # W
            assert instr["dim2"] == 3   # C
            assert instr["dim3"] == 2   # KH
            assert instr["dim4"] == 2   # KW
            assert instr["dim5"] == 2   # stride

        finally:
            os.unlink(onnx_path)
            os.unlink(bin_path)


class TestE2EGRU:
    """End-to-end: GRU model."""

    def test_gru_e2e(self):
        seq_len, batch, input_size = 3, 1, 64
        hidden_size = 32
        H = hidden_size

        x = _make_tensor_value_info("x", [seq_len, batch, input_size])
        y = _make_tensor_value_info("y", [seq_len, 1, batch, H])

        W_data = np.random.randn(1, 3 * H, input_size).astype(np.float32)
        R_data = np.random.randn(1, 3 * H, H).astype(np.float32)
        B_data = np.random.randn(1, 6 * H).astype(np.float32)

        W_init = helper.make_tensor(
            "W", TensorProto.FLOAT, [1, 3 * H, input_size],
            W_data.tobytes(), raw=True,
        )
        R_init = helper.make_tensor(
            "R", TensorProto.FLOAT, [1, 3 * H, H],
            R_data.tobytes(), raw=True,
        )
        B_init = helper.make_tensor(
            "B", TensorProto.FLOAT, [1, 6 * H],
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

        onnx_path = _save_onnx(graph_def)
        try:
            bin_path, data = _run_pipeline(onnx_path, "gru_model")

            hdr = _parse_header(data)
            assert hdr["num_instructions"] == 2  # GRU + END

            instr = _parse_instruction(data, 0)
            assert instr["opcode"] == OP_GRU
            assert instr["dim0"] == input_size
            assert instr["dim1"] == hidden_size
            assert instr["dim2"] == batch

            # GRU instruction should have weight and bias addresses set
            assert instr["weight_addr"] > 0, "GRU must have weight_addr"
            assert instr["bias_addr"] > 0, "GRU must have bias_addr"

            # Workspace should be allocated for GRU scratch
            assert hdr["workspace_size"] >= 9 * batch * H * 4

        finally:
            os.unlink(onnx_path)
            os.unlink(bin_path)


class TestE2EShapeOps:
    """End-to-end: model with Reshape ops that should be folded."""

    def test_reshape_folded(self):
        x = _make_tensor_value_info("x", [1, 100])
        y = _make_tensor_value_info("y", [1, 100])

        shape_data = np.array([1, 100], dtype=np.int64)
        shape_init = helper.make_tensor(
            "shape", TensorProto.INT64, [2], shape_data.tobytes(), raw=True,
        )

        reshape = helper.make_node("Reshape", inputs=["x", "shape"], outputs=["flattened"])
        relu = helper.make_node("Relu", inputs=["flattened"], outputs=["y"])

        graph_def = helper.make_graph(
            [reshape, relu], "reshape_test", [x], [y],
            initializer=[shape_init],
        )

        onnx_path = _save_onnx(graph_def)
        try:
            bin_path, data = _run_pipeline(onnx_path, "reshape_model")

            hdr = _parse_header(data)
            # Only ReLU + END (Reshape folded away)
            assert hdr["num_instructions"] == 2

            instr = _parse_instruction(data, 0)
            assert instr["opcode"] == OP_RELU

        finally:
            os.unlink(onnx_path)
            os.unlink(bin_path)
