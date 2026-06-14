"""
Unit tests for BIN file serialization / deserialization.
"""

import os
import struct
import tempfile

import numpy as np
import pytest

from compiler.ir import Graph, Tensor, Node
from compiler.codegen import generate, Instruction, CodegenResult
from compiler.bin_writer import write_bin, HEADER_SIZE, NN_BIN_MAGIC
from compiler.isa import OP_FC, OP_RELU, OP_END, FLAG_HAS_BIAS, INSTRUCTION_SIZE


def _make_tensor(name, shape, data=None, offset=None):
    t = Tensor(name=name, shape=shape, dtype="float32", data=data)
    t.offset = offset
    return t


class TestBinWriter:
    def test_write_and_verify_header(self):
        """Write a minimal BIN and verify the header fields."""
        x = _make_tensor("x", (1, 4), offset=0)
        y = _make_tensor("y", (1, 3), offset=16)
        w = _make_tensor("W", (4, 3), data=np.ones((4, 3), dtype=np.float32), offset=0)

        node = Node(name="fc", op_type="Gemm", inputs=[x, w], outputs=[y])
        graph = Graph(
            inputs=[x],
            outputs=[y],
            nodes=[node],
            initializers={"W": w},
            workspace_size=128,
            weight_size=w.size_bytes,
        )

        codegen_result = generate(graph)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            out_path = f.name

        try:
            write_bin(graph, codegen_result, out_path, model_name="test_model")

            with open(out_path, "rb") as f:
                data = f.read()

            # Parse header (little-endian)
            magic, ver_maj, ver_min, num_instr, weight_sz, ws_sz, \
                in_sz, out_sz, checksum, name_bytes, reserved = \
                struct.unpack_from("<I H H I I I I I I 64s 160s", data, 0)

            assert magic == NN_BIN_MAGIC
            assert ver_maj == 1
            assert ver_min == 0
            assert num_instr == len(codegen_result.instructions)
            assert ws_sz == codegen_result.workspace_size
            assert in_sz == x.size_bytes
            assert out_sz == y.size_bytes

            name = name_bytes.rstrip(b"\x00").decode("utf-8")
            assert name == "test_model"

            assert checksum != 0  # CRC32 should be non-zero for non-empty data

        finally:
            os.unlink(out_path)

    def test_instructions_serialized_correctly(self):
        """Verify that instructions round-trip through BIN format."""
        x = _make_tensor("x", (1, 10), offset=0)
        y = _make_tensor("y", (1, 10), offset=40)

        node = Node(name="relu", op_type="Relu", inputs=[x], outputs=[y])
        graph = Graph(inputs=[x], outputs=[y], nodes=[node], workspace_size=128)

        codegen_result = generate(graph)

        assert codegen_result.instructions[0].opcode == OP_RELU
        assert codegen_result.instructions[-1].opcode == OP_END

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            out_path = f.name

        try:
            write_bin(graph, codegen_result, out_path, "relu_test")

            with open(out_path, "rb") as f:
                data = f.read()

            # First instruction starts after header (256 bytes)
            instr_offset = HEADER_SIZE
            # Parse first instruction
            fields = struct.unpack_from(
                "<B B H I I I I I I I I I I I I I I I",
                data, instr_offset,
            )
            opcode = fields[0]
            assert opcode == OP_RELU

            # Second instruction (END) at offset 256+64 = 320
            fields2 = struct.unpack_from(
                "<B B H I I I I I I I I I I I I I I I",
                data, HEADER_SIZE + INSTRUCTION_SIZE,
            )
            assert fields2[0] == OP_END

        finally:
            os.unlink(out_path)

    def test_weights_serialized(self):
        """Weight data is preserved through serialization."""
        x = _make_tensor("x", (1, 2), offset=0)
        y = _make_tensor("y", (1, 3), offset=8)

        w_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        w = _make_tensor("W", (2, 3), data=w_data, offset=0)

        node = Node(name="fc", op_type="Gemm", inputs=[x, w], outputs=[y])
        graph = Graph(
            inputs=[x],
            outputs=[y],
            nodes=[node],
            initializers={"W": w},
            workspace_size=128,
            weight_size=w.size_bytes,
        )

        codegen_result = generate(graph)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            out_path = f.name

        try:
            write_bin(graph, codegen_result, out_path, "weight_test")

            with open(out_path, "rb") as f:
                data = f.read()

            # Weights start after header + instructions
            weights_offset = HEADER_SIZE + len(codegen_result.instructions) * INSTRUCTION_SIZE
            weight_section = data[weights_offset:]

            recovered = np.frombuffer(weight_section[:w.size_bytes], dtype=np.float32)
            np.testing.assert_array_almost_equal(recovered.reshape(2, 3), w_data)

        finally:
            os.unlink(out_path)
