"""
ONNX model parser — loads an ONNX proto, builds the compiler IR.

Responsibilities
----------------
1. Load & validate the ONNX model (opset version, element types).
2. Extract initializers (weights, biases) as constant Tensors.
3. Walk the graph in topological order, building Nodes.
4. Infer output shapes for every node (shape propagation).
5. Fold shape-only ops (Reshape, Transpose, …) so downstream
   nodes consume the resolved tensor shapes directly.
6. Reject unsupported ops with clear error messages.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np
import onnx
from onnx import numpy_helper

from .ir import Graph, Node, Tensor
from .isa import ONNX_OP_MAP, SHAPE_OPS, OP_FC, FLAG_HAS_BIAS

logger = logging.getLogger(__name__)

# ONNX element type → dtype string  (we only accept FLOAT)
_ONNX_DTYPE_MAP: dict[int, str] = {
    1: "float32",
    10: "float16",
    11: "float64",
    6: "int32",
    7: "int64",
}

_SUPPORTED_DTYPES = {"float32"}


# ════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════

def parse_onnx(model_path: str, input_shape: Optional[dict[str, tuple[int, ...]]] = None) -> Graph:
    """Load an ONNX file and return a compiler IR Graph.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` file.
    input_shape:
        Optional per-input shape overrides, e.g. ``{"x": (1, 3, 224, 224)}``.
        Useful when the model has dynamic input dims.
    """
    model = onnx.load(model_path)
    _check_model(model)

    graph = Graph()

    # ── Extract initializers (constants) ─────────────────────────
    # Only FP32 initializers are compute weights/biases.  Non-FP32
    # tensors (e.g. int64 shape constants) are stored as-is for
    # shape-ops to use, but won't go into the weight buffer.
    for init in model.graph.initializer:
        tensor = _tensor_from_onnx(init)
        graph.initializers[tensor.name] = tensor

    # ── Build a value-info lookup for shapes ─────────────────────
    value_shapes: dict[str, tuple[int, ...]] = {}

    for vi in model.graph.input:
        shape_tuple = _shape_from_proto(vi)
        if input_shape and vi.name in input_shape:
            shape_tuple = input_shape[vi.name]
        value_shapes[vi.name] = shape_tuple

    for vi in model.graph.output:
        value_shapes[vi.name] = _shape_from_proto(vi)

    for vi in model.graph.value_info:
        value_shapes[vi.name] = _shape_from_proto(vi)

    # Initializers also carry shape info
    for init in model.graph.initializer:
        value_shapes[init.name] = tuple(init.dims)

    # ── Mark graph inputs / outputs ──────────────────────────────
    graph_input_names = {vi.name for vi in model.graph.input}
    graph_output_names = {vi.name for vi in model.graph.output}

    # ── Create a name→Tensor registry so we can look up by edge ─
    tensor_registry: dict[str, Tensor] = {}

    def _get_or_create_tensor(name: str) -> Tensor:
        if name not in tensor_registry:
            shape = value_shapes.get(name, ())
            data = None
            if name in graph.initializers:
                data = graph.initializers[name].data
            tensor_registry[name] = Tensor(
                name=name,
                shape=shape,
                dtype="float32",
                data=data,
            )
        return tensor_registry[name]

    # ── Build graph inputs ───────────────────────────────────────
    for vi in model.graph.input:
        t = _get_or_create_tensor(vi.name)
        graph.inputs.append(t)

    # ── Build set of names produced by nodes ────────────────────
    # This is needed for correct in-degree calculation: inputs that
    # are initializers or graph inputs have no producing node and
    # should NOT count toward a consumer's in-degree.
    produced_by_nodes: set[str] = set()
    for onnx_node in model.graph.node:
        for out in onnx_node.output:
            if out:  # skip empty output names
                produced_by_nodes.add(out)

    # ── Build adjacency for topological sort ────────────────────
    consumers: dict[str, list[int]] = {}
    in_degree = [0] * len(model.graph.node)
    for i, onnx_node in enumerate(model.graph.node):
        for inp in onnx_node.input:
            consumers.setdefault(inp, []).append(i)
            if inp in produced_by_nodes:
                in_degree[i] += 1

    queue: deque[int] = deque(i for i, d in enumerate(in_degree) if d == 0)
    topo_order: list[int] = []

    while queue:
        idx = queue.popleft()
        topo_order.append(idx)
        onnx_node = model.graph.node[idx]
        for out in onnx_node.output:
            for consumer_idx in consumers.get(out, []):
                in_degree[consumer_idx] -= 1
                if in_degree[consumer_idx] == 0:
                    queue.append(consumer_idx)

    if len(topo_order) != len(model.graph.node):
        # Cyclic graph — ONNX shouldn't have these, but be safe
        missing = len(model.graph.node) - len(topo_order)
        raise ValueError(f"Graph appears to be cyclic: {missing} node(s) not reachable.")

    # ── Walk nodes in topological order ──────────────────────────
    for idx in topo_order:
        onnx_node = model.graph.node[idx]
        op_type = onnx_node.op_type

        # Fetch inputs
        input_tensors = [_get_or_create_tensor(name) for name in onnx_node.input]

        # Resolve attributes
        attrs = _extract_attrs(onnx_node)

        # Shape inference → output shapes
        output_shapes = _infer_shape(op_type, input_tensors, attrs)

        # Create output tensors
        output_tensors = []
        for out_name, out_shape in zip(onnx_node.output, output_shapes):
            if out_name:
                # Constant nodes: propagate the value data to the output tensor
                # so downstream shape-ops (e.g. Reshape) can read it.
                data = None
                if op_type == "Constant":
                    data = attrs.get("value")
                t = Tensor(name=out_name, shape=out_shape, dtype="float32", data=data)
                tensor_registry[out_name] = t
                value_shapes[out_name] = out_shape
                output_tensors.append(t)

        is_shape = op_type in SHAPE_OPS

        if not is_shape and op_type not in ONNX_OP_MAP:
            raise NotImplementedError(
                f"Unsupported ONNX op '{op_type}' in node '{onnx_node.name or '(unnamed)'}'. "
                f"Supported ops: {sorted(ONNX_OP_MAP.keys())} + shape ops: {sorted(SHAPE_OPS)}"
            )

        node = Node(
            name=onnx_node.name or f"{op_type}_{idx}",
            op_type=op_type,
            inputs=input_tensors,
            outputs=output_tensors,
            attrs=attrs,
            is_shape_op=is_shape,
        )
        graph.nodes.append(node)

    # ── Mark graph outputs ───────────────────────────────────────
    for vi in model.graph.output:
        if vi.name in tensor_registry:
            graph.outputs.append(tensor_registry[vi.name])
        else:
            t = Tensor(name=vi.name, shape=_shape_from_proto(vi), dtype="float32")
            graph.outputs.append(t)

    logger.info(
        "Parsed ONNX model: %d nodes, %d initializers, "
        "%d inputs, %d outputs",
        len(graph.nodes), len(graph.initializers),
        len(graph.inputs), len(graph.outputs),
    )
    return graph


# ════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════

def _check_model(model: onnx.ModelProto) -> None:
    """Validate model-level properties."""
    ir_version = model.ir_version
    if ir_version < 4:
        raise ValueError(f"ONNX IR version {ir_version} is too old; need >= 4.")
    # Check opset
    opset = _get_opset(model)
    if opset < 7:
        raise ValueError(f"ONNX opset {opset} is too old; need >= 7.")
    logger.info("ONNX IR v%d, opset %d — OK", ir_version, opset)


def _get_opset(model: onnx.ModelProto) -> int:
    for opset_import in model.opset_import:
        if opset_import.domain in ("", "ai.onnx"):
            return opset_import.version
    return 1


def _tensor_from_onnx(tp: onnx.TensorProto) -> Tensor:
    """Convert an ONNX TensorProto (initializer) to our IR Tensor."""
    np_arr = numpy_helper.to_array(tp)
    dtype_str = _ONNX_DTYPE_MAP.get(tp.data_type, "unknown")
    return Tensor(
        name=tp.name,
        shape=tuple(tp.dims),
        dtype=dtype_str,
        data=np_arr,
    )


def _shape_from_proto(vip) -> tuple[int, ...]:
    """Extract shape from a ValueInfoProto, or return () if unknown."""
    shape = ()
    tp = vip.type
    if tp.HasField("tensor_type"):
        tt = tp.tensor_type
        if tt.HasField("shape"):
            shape = tuple(
                d.dim_value if d.HasField("dim_value") else -1
                for d in tt.shape.dim
            )
    return shape


def _extract_attrs(onnx_node) -> dict[str, Any]:
    """Extract node attributes into a plain dict."""
    attrs = {}
    for attr in onnx_node.attribute:
        name = attr.name
        if attr.type == onnx.AttributeProto.FLOAT:
            attrs[name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attrs[name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[name] = attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.TENSOR:
            attrs[name] = numpy_helper.to_array(attr.t)
        else:
            attrs[name] = onnx.helper.get_attribute_value(attr)
    return attrs


# ════════════════════════════════════════════════════════════════════
# Shape inference
# ════════════════════════════════════════════════════════════════════

def _infer_shape(
    op_type: str,
    inputs: list[Tensor],
    attrs: dict[str, Any],
) -> list[tuple[int, ...]]:
    """Infer output shape(s) for an ONNX op.

    Returns one shape tuple per output.
    """
    if op_type in ("Relu", "Sigmoid", "Tanh", "LeakyRelu"):
        return [inputs[0].shape]

    if op_type in ("Mul", "Add", "Sub", "Div"):
        # Element-wise: output shape = broadcast(input0, input1)
        s0, s1 = inputs[0].shape, inputs[1].shape
        return [_broadcast_shape(s0, s1)]

    if op_type == "Gemm":
        # A[M,K] × B[K,N] + C[N]  →  [M, N]
        transA = attrs.get("transA", 0)
        transB = attrs.get("transB", 0)
        a_shape = inputs[0].shape
        b_shape = inputs[1].shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError(f"Gemm needs 2-D inputs, got {a_shape}, {b_shape}")
        M = a_shape[1] if transA else a_shape[0]
        N = b_shape[0] if transB else b_shape[1]
        return [(M, N)]

    if op_type == "MatMul":
        a_shape, b_shape = inputs[0].shape, inputs[1].shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError(f"MatMul needs >=2-D inputs, got {a_shape}, {b_shape}")
        *batch_a, M, K = a_shape
        *batch_b, K2, N = b_shape
        if K != K2:
            raise ValueError(f"MatMul inner dim mismatch: {K} vs {K2}")
        batch = _broadcast_shape(tuple(batch_a), tuple(batch_b))
        return [(*batch, M, N)]

    if op_type in ("MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool"):
        return [_pool_output_shape(inputs[0].shape, attrs, op_type)]

    if op_type == "GRU":
        return [_gru_output_shape(inputs, attrs)]

    if op_type == "Reshape":
        shape_attr = attrs.get("shape")
        if shape_attr is not None:
            shape_list = list(shape_attr)
            # Replace 0 with the original dim, -1 with inferred
            in_shape = inputs[0].shape
            for i, v in enumerate(shape_list):
                if v == 0 and i < len(in_shape):
                    shape_list[i] = in_shape[i]
            return [tuple(shape_list)]
        # If 'shape' is an input (dynamic reshape), try to resolve from constant
        if len(inputs) >= 2 and inputs[1].data is not None:
            shape_data = inputs[1].data.astype(np.int64).tolist()
            return [tuple(shape_data)]
        return [inputs[0].shape]  # best-effort

    if op_type == "Transpose":
        perm = attrs.get("perm", None)
        in_shape = inputs[0].shape
        if perm is None:
            perm = list(reversed(range(len(in_shape))))
        out_shape = tuple(in_shape[p] for p in perm)
        return [out_shape]

    if op_type == "Concat":
        axis = attrs.get("axis", 0)
        # Sum along axis; all other dims must match
        total_axis = sum(inp.shape[axis] for inp in inputs if len(inp.shape) > axis)
        ref_shape = list(inputs[0].shape)
        ref_shape[axis] = total_axis
        return [tuple(ref_shape)]

    if op_type == "Squeeze":
        axes = attrs.get("axes", None)
        in_shape = list(inputs[0].shape)
        if axes is None:
            out_shape = tuple(d for d in in_shape if d != 1)
        else:
            for a in sorted(axes, reverse=True):
                if a < len(in_shape):
                    in_shape.pop(a)
            out_shape = tuple(in_shape)
        return [out_shape]

    if op_type == "Unsqueeze":
        axes = attrs.get("axes", [])
        in_shape = list(inputs[0].shape)
        for a in sorted(axes):
            in_shape.insert(a, 1)
        return [tuple(in_shape)]

    if op_type == "Flatten":
        axis = attrs.get("axis", 1)
        in_shape = inputs[0].shape
        if len(in_shape) == 0:
            return [(1,)]
        outer = 1
        for d in in_shape[:axis]:
            outer *= d
        inner = 1
        for d in in_shape[axis:]:
            inner *= d
        return [(outer, inner)]

    if op_type == "Constant":
        # Constant node carries a 'value' attribute (already converted to np.ndarray).
        # Its output shape is simply the shape of that array.
        value = attrs.get("value")
        if value is not None and hasattr(value, "shape"):
            return [tuple(value.shape)]
        return [()]  # scalar constant

    if op_type == "BatchNormalization":
        return [inputs[0].shape]

    if op_type == "Dropout":
        return [inputs[0].shape]

    raise NotImplementedError(f"Shape inference not implemented for '{op_type}'")


def _broadcast_shape(s0: tuple[int, ...], s1: tuple[int, ...]) -> tuple[int, ...]:
    """NumPy-style broadcast shape."""
    if not s0:
        return s1
    if not s1:
        return s0
    # Pad to same length
    ndim = max(len(s0), len(s1))
    p0 = (1,) * (ndim - len(s0)) + s0
    p1 = (1,) * (ndim - len(s1)) + s1
    out = []
    for d0, d1 in zip(p0, p1):
        if d0 == d1:
            out.append(d0)
        elif d0 == 1:
            out.append(d1)
        elif d1 == 1:
            out.append(d0)
        elif d0 == -1 or d1 == -1:
            out.append(max(d0, d1))
        else:
            raise ValueError(f"Cannot broadcast {s0} with {s1}")
    return tuple(out)


def _pool_output_shape(
    in_shape: tuple[int, ...],
    attrs: dict[str, Any],
    op_type: str,
) -> tuple[int, ...]:
    """Infer output shape for pooling ops."""
    if len(in_shape) < 3:
        raise ValueError(f"Pooling needs >=3-D input (H,W,C), got {in_shape}")

    if op_type.startswith("Global"):
        # Global pooling → output spatial dims = 1
        H, W, C = in_shape[-3], in_shape[-2], in_shape[-1]
        return in_shape[:-3] + (1, 1, C)

    H_in, W_in = in_shape[-2] if len(in_shape) >= 3 else in_shape[-1], in_shape[-3] if len(in_shape) >= 3 else in_shape[-2]
    # Simplified: assume NHWC or NCHW? ONNX uses NCHW.
    # Our ISA uses HWC. Let's handle NCHW → HWC conversion later in codegen.
    # For shape purposes, assume the last 3 dims are (C, H, W) or (H, W, C).
    # Actually ONNX standard is NCHW. Let's go with that for shape inference.
    if len(in_shape) == 4:
        # NCHW
        N, C, H, W = in_shape
    elif len(in_shape) == 3:
        # CHW
        C, H, W = in_shape[0], in_shape[1], in_shape[2]
    else:
        raise ValueError(f"Unexpected pooling input shape: {in_shape}")

    kernel_shape = attrs.get("kernel_shape", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    strides = attrs.get("strides", [1, 1])
    dilations = attrs.get("dilations", [1, 1])

    KH, KW = kernel_shape[0], kernel_shape[1]
    # Effective kernel size with dilation
    KH_eff = (KH - 1) * dilations[0] + 1
    KW_eff = (KW - 1) * dilations[1] + 1

    H_out = (H + pads[0] + pads[2] - KH_eff) // strides[0] + 1
    W_out = (W + pads[1] + pads[3] - KW_eff) // strides[1] + 1

    if len(in_shape) == 4:
        return (N, C, H_out, W_out)
    else:
        return (C, H_out, W_out)


def _gru_output_shape(
    inputs: list[Tensor],
    attrs: dict[str, Any],
) -> tuple[int, ...]:
    """Infer output shape for ONNX GRU.

    ONNX GRU inputs: X[seq_len, batch, input_size], W[dir, 3H, I], R[dir, 3H, H]
    Optional: B[dir, 6H], sequence_lens, initial_h[batch, dir, H]

    Returns shape of Y (the output).
    """
    x_shape = inputs[0].shape
    if len(x_shape) != 3:
        raise ValueError(f"GRU X must be 3-D [seq_len, batch, input_size], got {x_shape}")

    seq_len, batch, input_size = x_shape

    hidden_size = attrs.get("hidden_size")
    if hidden_size is None:
        # Infer from W shape: W[dir, 3*H, I]
        if len(inputs) >= 2 and len(inputs[1].shape) == 3:
            hidden_size = inputs[1].shape[1] // 3
        else:
            raise ValueError("Cannot infer GRU hidden_size; specify it as an attribute.")

    direction = attrs.get("direction", "forward")
    num_directions = 2 if direction == "bidirectional" else 1

    # Y shape: [seq_len, num_directions, batch, hidden_size]
    return (seq_len, num_directions, batch, hidden_size)
