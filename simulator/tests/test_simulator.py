"""
Verification tests: Compiler → Simulator ≟ ONNX Runtime.

Each test:
1. Builds a synthetic ONNX model
2. Compiles it to BIN with the compiler
3. Runs the BIN through the simulator
4. Runs the same model through ONNX Runtime
5. Asserts the results match within FP32 tolerance
"""

import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

from compiler.parser import parse_onnx
from compiler.memory_planner import plan_memory
from compiler.codegen import generate
from compiler.bin_writer import write_bin
from simulator.simulator import Simulator


def _compile_onnx_graph(graph_def, model_name="test") -> str:
    """Compile a GraphProto → BIN file, returns BIN path."""
    model = helper.make_model(
        graph_def,
        producer_name="sim_test",
        opset_imports=[helper.make_opsetid("", 10)],
        ir_version=7,  # ONNX IR v7 for compatibility with onnxruntime
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        onnx_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    try:
        graph = parse_onnx(onnx_path)
        plan_memory(graph)
        cg = generate(graph)
        write_bin(graph, cg, bin_path, model_name)
    finally:
        os.unlink(onnx_path)

    return bin_path


def _run_onnx_runtime(graph_def, inputs: dict) -> dict:
    """Run a GraphProto through ONNX Runtime, return output dict."""
    import onnxruntime as ort

    model = helper.make_model(
        graph_def,
        producer_name="ort_test",
        opset_imports=[helper.make_opsetid("", 10)],
        ir_version=7,  # ONNX IR v7 for compatibility with onnxruntime
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        onnx_path = f.name

    try:
        session = ort.InferenceSession(onnx_path)
        output_names = [o.name for o in session.get_outputs()]
        ort_inputs = {k: v.astype(np.float32) for k, v in inputs.items()}
        result = session.run(output_names, ort_inputs)
        return {name: result[i] for i, name in enumerate(output_names)}
    finally:
        os.unlink(onnx_path)


def _make_tensor_value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


# ════════════════════════════════════════════════════════════════════

class TestSimulatorFC:
    def test_fc_only(self):
        M, N = 4, 3
        x = _make_tensor_value_info("x", [1, M])
        y = _make_tensor_value_info("y", [1, N])

        w_data = np.random.randn(M, N).astype(np.float32) * 0.5
        b_data = np.random.randn(N).astype(np.float32) * 0.1

        w_init = helper.make_tensor("W", TensorProto.FLOAT, [M, N], w_data.tobytes(), raw=True)
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [N], b_data.tobytes(), raw=True)

        node = helper.make_node("Gemm", inputs=["x", "W", "B"], outputs=["y"])
        graph_def = helper.make_graph(
            [node], "fc_test", [x], [y], initializer=[w_init, b_init],
        )

        bin_path = _compile_onnx_graph(graph_def, "fc_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)

            input_data = np.random.randn(1, M).astype(np.float32)
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            ort_output = _run_onnx_runtime(graph_def, {"x": input_data})

            np.testing.assert_allclose(
                sim_output, ort_output["y"].ravel(),
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)


class TestSimulatorFCRelu:
    def test_fc_relu_chain(self):
        M, H, N = 8, 5, 3
        x = _make_tensor_value_info("x", [1, M])
        y = _make_tensor_value_info("y", [1, N])

        w1 = np.random.randn(M, H).astype(np.float32) * 0.3
        b1 = np.random.randn(H).astype(np.float32) * 0.1
        w2 = np.random.randn(H, N).astype(np.float32) * 0.3
        b2 = np.random.randn(N).astype(np.float32) * 0.1

        w1_i = helper.make_tensor("W1", TensorProto.FLOAT, [M, H], w1.tobytes(), raw=True)
        b1_i = helper.make_tensor("B1", TensorProto.FLOAT, [H], b1.tobytes(), raw=True)
        w2_i = helper.make_tensor("W2", TensorProto.FLOAT, [H, N], w2.tobytes(), raw=True)
        b2_i = helper.make_tensor("B2", TensorProto.FLOAT, [N], b2.tobytes(), raw=True)

        fc1 = helper.make_node("Gemm", inputs=["x", "W1", "B1"], outputs=["h_pre"])
        relu = helper.make_node("Relu", inputs=["h_pre"], outputs=["h"])
        fc2 = helper.make_node("Gemm", inputs=["h", "W2", "B2"], outputs=["y"])

        graph_def = helper.make_graph(
            [fc1, relu, fc2], "fc_relu", [x], [y],
            initializer=[w1_i, b1_i, w2_i, b2_i],
        )

        bin_path = _compile_onnx_graph(graph_def, "fc_relu_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)

            input_data = np.random.randn(1, M).astype(np.float32)
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            ort_output = _run_onnx_runtime(graph_def, {"x": input_data})

            np.testing.assert_allclose(
                sim_output, ort_output["y"].ravel(),
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)


class TestSimulatorPool:
    def test_maxpool(self):
        H, W, C = 8, 8, 3
        x = _make_tensor_value_info("x", [1, C, H, W])
        y = _make_tensor_value_info("y", [1, C, 4, 4])

        node = helper.make_node(
            "MaxPool", inputs=["x"], outputs=["y"],
            kernel_shape=[2, 2], strides=[2, 2],
        )
        graph_def = helper.make_graph([node], "pool", [x], [y])

        bin_path = _compile_onnx_graph(graph_def, "pool_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)

            input_data = np.random.randn(1, C, H, W).astype(np.float32)
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            ort_output = _run_onnx_runtime(graph_def, {"x": input_data})

            np.testing.assert_allclose(
                sim_output, ort_output["y"].ravel(),
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)


class TestSimulatorGRU:
    def test_gru(self):
        seq_len, batch, I, H = 3, 1, 8, 5
        rng = np.random.RandomState(42)

        x = _make_tensor_value_info("x", [seq_len, batch, I])
        y = _make_tensor_value_info("y", [seq_len, 1, batch, H])

        W_data = rng.randn(1, 3 * H, I).astype(np.float32) * 0.3
        R_data = rng.randn(1, 3 * H, H).astype(np.float32) * 0.3
        B_data = rng.randn(1, 6 * H).astype(np.float32) * 0.1

        W_i = helper.make_tensor("W", TensorProto.FLOAT, [1, 3 * H, I], W_data.tobytes(), raw=True)
        R_i = helper.make_tensor("R", TensorProto.FLOAT, [1, 3 * H, H], R_data.tobytes(), raw=True)
        B_i = helper.make_tensor("B", TensorProto.FLOAT, [1, 6 * H], B_data.tobytes(), raw=True)

        gru = helper.make_node(
            "GRU", inputs=["x", "W", "R", "B"], outputs=["y"],
            hidden_size=H,
        )
        graph_def = helper.make_graph(
            [gru], "gru", [x], [y], initializer=[W_i, R_i, B_i],
        )

        bin_path = _compile_onnx_graph(graph_def, "gru_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)

            input_data = rng.randn(seq_len, batch, I).astype(np.float32)
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            # Reference: manually compute the ONNX GRU formula
            W = W_data[0]; R = R_data[0]; B = B_data[0]
            W_z, W_r, W_h = W[0:H], W[H:2*H], W[2*H:3*H]
            R_z, R_r, R_h = R[0:H], R[H:2*H], R[2*H:3*H]
            W_bz, W_br, W_bh = B[0:H], B[H:2*H], B[2*H:3*H]
            R_bz, R_br, R_bh = B[3*H:4*H], B[4*H:5*H], B[5*H:6*H]
            x_data = input_data
            h = np.zeros((batch, H), dtype=np.float32)
            for t in range(seq_len):
                xt = x_data[t]
                z = 1.0/(1.0+np.exp(-(xt@W_z.T+W_bz+h@R_z.T+R_bz)))
                r = 1.0/(1.0+np.exp(-(xt@W_r.T+W_br+h@R_r.T+R_br)))
                n = np.tanh(xt@W_h.T+W_bh+r*(h@R_h.T)+R_bh)
                h = (1-z)*n+z*h
            ref_h = h.ravel()  # [batch*H]

            np.testing.assert_allclose(
                sim_output[:len(ref_h)], ref_h,
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)


class TestSimulatorSigmoidTanh:
    def test_sigmoid(self):
        N = 100
        x = _make_tensor_value_info("x", [N])
        y = _make_tensor_value_info("y", [N])
        node = helper.make_node("Sigmoid", inputs=["x"], outputs=["y"])
        graph_def = helper.make_graph([node], "sigm", [x], [y])

        bin_path = _compile_onnx_graph(graph_def, "sigm_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)
            input_data = np.random.randn(N).astype(np.float32) * 3
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            ort_output = _run_onnx_runtime(graph_def, {"x": input_data})
            np.testing.assert_allclose(
                sim_output, ort_output["y"].ravel(),
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)

    def test_tanh(self):
        N = 100
        x = _make_tensor_value_info("x", [N])
        y = _make_tensor_value_info("y", [N])
        node = helper.make_node("Tanh", inputs=["x"], outputs=["y"])
        graph_def = helper.make_graph([node], "tanh", [x], [y])

        bin_path = _compile_onnx_graph(graph_def, "tanh_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)
            input_data = np.random.randn(N).astype(np.float32) * 3
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            ort_output = _run_onnx_runtime(graph_def, {"x": input_data})
            np.testing.assert_allclose(
                sim_output, ort_output["y"].ravel(),
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)


class TestSimulatorElemWise:
    def test_mul_add(self):
        N = 50
        a = _make_tensor_value_info("a", [N])
        b = _make_tensor_value_info("b", [N])
        c = _make_tensor_value_info("c", [N])
        d = _make_tensor_value_info("d", [N])

        mul_node = helper.make_node("Mul", inputs=["a", "b"], outputs=["c"])
        add_node = helper.make_node("Add", inputs=["c", "b"], outputs=["d"])

        graph_def = helper.make_graph([mul_node, add_node], "elem", [a, b], [d])

        bin_path = _compile_onnx_graph(graph_def, "elem_test")
        try:
            sim = Simulator()
            sim.load_bin(bin_path)
            a_data = np.random.randn(N).astype(np.float32)
            b_data = np.random.randn(N).astype(np.float32)

            # Two inputs: concatenate them sequentially in the input buffer
            input_data = np.concatenate([a_data, b_data])
            # Override input_base to use the correct layout
            # Actually, the memory planner should handle this. The BIN's
            # input_size should be (N+N)*4, so both fit.
            sim.set_input(input_data)
            sim.run()
            sim_output = sim.get_output()

            ort_output = _run_onnx_runtime(graph_def, {"a": a_data, "b": b_data})

            np.testing.assert_allclose(
                sim_output, ort_output["d"].ravel(),
                rtol=1e-5, atol=1e-6,
            )
        finally:
            os.unlink(bin_path)
