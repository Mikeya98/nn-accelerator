"""
Code generator — walks the IR graph and emits a list of ``nn_instruction_t``
entries ready for serialisation.

Shape-only ops (Reshape, Transpose, …) are skipped — they only affect
tensor shapes at compile time and produce no PL instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .ir import Graph, Node, Tensor
from .isa import (
    OP_FC, OP_POOL, OP_RELU, OP_SIGMOID, OP_TANH, OP_GRU,
    OP_ELEM_MUL, OP_ELEM_ADD, OP_END, OP_NOP,
    FLAG_HAS_BIAS, FLAG_POOL_MAX, FLAG_LINEAR_BEFORE_RESET,
    FLAG_STORE_INTERMEDIATE,
    ACT_NONE, ACT_RELU, ACT_SIGMOID, ACT_TANH, ACT_MASK,
    INSTRUCTION_SIZE, OPCODE_NAMES, SHAPE_OPS,
)

logger = logging.getLogger(__name__)

# ── IR op → canonical op type (used in instruction dispatch) ───────
# After parser, ONNX op names are still in op_type.  We normalise here.
_OP_TABLE: dict[str, int] = {
    "Gemm":         OP_FC,
    "MatMul":       OP_FC,
    "Relu":         OP_RELU,
    "MaxPool":      OP_POOL,
    "AveragePool":  OP_POOL,
    "GlobalMaxPool":      OP_POOL,
    "GlobalAveragePool":  OP_POOL,
    "Sigmoid":      OP_SIGMOID,
    "Tanh":         OP_TANH,
    "GRU":          OP_GRU,
    "Mul":          OP_ELEM_MUL,
    "Add":          OP_ELEM_ADD,
}


@dataclass
class Instruction:
    """Python-side representation of one 64-byte instruction word."""
    opcode: int = OP_NOP
    flags: int = 0
    seq_len: int = 0
    input0_addr: int = 0
    input1_addr: int = 0
    output_addr: int = 0
    weight_addr: int = 0
    bias_addr: int = 0
    workspace_addr: int = 0
    scale_addr: int = 0
    dim0: int = 0
    dim1: int = 0
    dim2: int = 0
    dim3: int = 0
    dim4: int = 0
    dim5: int = 0
    dim6: int = 0
    dim7: int = 0

    def as_tuple(self) -> tuple:
        """Return fields in the order expected by the binary format."""
        return (
            self.opcode, self.flags, self.seq_len,
            self.input0_addr, self.input1_addr, self.output_addr,
            self.weight_addr, self.bias_addr,
            self.workspace_addr, self.scale_addr,
            self.dim0, self.dim1, self.dim2, self.dim3,
            self.dim4, self.dim5, self.dim6, self.dim7,
        )


@dataclass
class CodegenResult:
    """Output of code generation."""
    instructions: list[Instruction] = field(default_factory=list)
    extra_weights: list[Tensor] = field(default_factory=list)
    workspace_size: int = 0


# ════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════

def generate(graph: Graph) -> CodegenResult:
    """Generate instructions from the IR graph.

    Returns a ``CodegenResult`` with the instruction list and any
    extra weight tensors (e.g. GRU packed weights) that need to be
    serialised.
    """
    result = CodegenResult()

    # GRU scratch is allocated after the regular workspace.  Since
    # instructions execute sequentially, all GRU nodes share a single
    # scratch buffer sized for the worst-case GRU in the model.
    scratch_base = graph.workspace_size
    max_gru_scratch = 0

    for node in graph.nodes:
        if node.is_shape_op:
            logger.debug("Skip shape op: %s (%s)", node.name, node.op_type)
            continue

        opcode = _OP_TABLE.get(node.op_type)
        if opcode is None:
            raise ValueError(f"No opcode mapping for '{node.op_type}'")

        handler = _HANDLERS.get(opcode)
        if handler is None:
            raise NotImplementedError(
                f"No codegen handler for opcode {OPCODE_NAMES.get(opcode, hex(opcode))}"
            )

        if opcode == OP_GRU:
            instrs, scratch_needed = handler(node, opcode, scratch_base)
            if scratch_needed > max_gru_scratch:
                max_gru_scratch = scratch_needed
        else:
            instrs, _ = handler(node, opcode)

        result.instructions.extend(instrs)

        # Collect extra weight tensors from GRU handler
        for instr in instrs:
            if hasattr(instr, '_gru_weight') and instr._gru_weight is not None:
                result.extra_weights.append(instr._gru_weight)
            if hasattr(instr, '_gru_bias') and instr._gru_bias is not None:
                result.extra_weights.append(instr._gru_bias)

    # Terminate
    result.instructions.append(Instruction(opcode=OP_END))
    result.workspace_size = scratch_base + max_gru_scratch

    logger.info(
        "Generated %d instructions (%d compute + 1 END), "
        "workspace=%d KiB (%.1f KiB scratch)",
        len(result.instructions), len(result.instructions) - 1,
        result.workspace_size // 1024, max_gru_scratch // 1024,
    )
    return result


# ════════════════════════════════════════════════════════════════════
# Per-op handlers
# ════════════════════════════════════════════════════════════════════

def _handle_fc(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    """Gemm / MatMul → FC instruction(s).

    ONNX Gemm: Y = alpha * A·B + beta * C
    With alpha=1, beta=1 this matches ISA FC: y = x·W + b.
    """
    instr = Instruction(opcode=OP_FC)

    x, W = node.inputs[0], node.inputs[1]
    has_bias = len(node.inputs) >= 3
    bias = node.inputs[2] if has_bias else None

    # ── transA / transB ─────────────────────────────────────────
    # The ISA always does y[M] = x[M] · W[M×N].  If the ONNX Gemm
    # has transposed inputs we simply use the shape and trust that
    # the weight data is already in the right layout (ONNX stores
    # weights in the layout the Gemm expects).
    transB = node.attrs.get("transB", 0)
    M = x.shape[-1]  # input features (last dim of x)
    N = W.shape[1] if not transB else W.shape[0]

    instr.input0_addr = _offset(x)
    instr.weight_addr = _offset(W)
    instr.output_addr = _offset(node.outputs[0])

    if has_bias and bias is not None and bias.is_constant:
        instr.flags |= FLAG_HAS_BIAS
        instr.bias_addr = _offset(bias)

    # ── Fused activation ────────────────────────────────────────
    # Check if the next node is an activation op that can be fused.
    # (This requires the next node to be a single-input element-wise
    #  op whose only consumer is the FC output.)
    _fuse_activation(instr, node)

    instr.dim0 = M
    instr.dim1 = N

    return [instr], 0


def _handle_relu(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    instr = Instruction(opcode=opcode)
    instr.input0_addr = _offset(node.inputs[0])
    instr.output_addr = _offset(node.outputs[0])
    instr.dim0 = node.inputs[0].num_elements
    return [instr], 0


def _handle_sigmoid(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    return _handle_relu(node, opcode)


def _handle_tanh(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    return _handle_relu(node, opcode)


def _handle_pool(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    instr = Instruction(opcode=OP_POOL)
    x = node.inputs[0]
    attrs = node.attrs

    instr.input0_addr = _offset(x)
    instr.output_addr = _offset(node.outputs[0])

    # Determine max vs avg
    if node.op_type in ("MaxPool", "GlobalMaxPool"):
        instr.flags |= FLAG_POOL_MAX

    # Shape: ONNX uses NCHW, ISA uses HWC.
    # We handle the layout in the PL engine or driver.
    # For now, pass dimensions as-is; the PL interprets them.
    shape = x.shape
    if len(shape) == 4:   # NCHW
        N, C, H, W = shape
    elif len(shape) == 3:  # CHW
        N, C, H, W = 1, shape[0], shape[1], shape[2]
    else:
        raise ValueError(f"Pooling input shape {shape} not supported")

    if node.op_type.startswith("Global"):
        KH, KW = H, W
        stride = 1
    else:
        kernel = attrs.get("kernel_shape", [1, 1])
        KH, KW = kernel[0], kernel[1]
        strides = attrs.get("strides", [1, 1])
        stride = strides[0]  # assume symmetric

    instr.dim0 = H
    instr.dim1 = W
    instr.dim2 = C
    instr.dim3 = KH
    instr.dim4 = KW
    instr.dim5 = stride

    return [instr], 0


def _handle_elem_binary(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    instr = Instruction(opcode=opcode)
    instr.input0_addr = _offset(node.inputs[0])
    instr.input1_addr = _offset(node.inputs[1])
    instr.output_addr = _offset(node.outputs[0])
    instr.dim0 = node.inputs[0].num_elements
    return [instr], 0


def _handle_gru(node: Node, opcode: int, scratch_base: int = 0) -> tuple[list[Instruction], int]:
    """Generate GRU instruction with packed weights.

    The ONNX GRU op has 3-6 inputs:
      0: X   [seq_len, batch, input_size]
      1: W   [num_directions, 3*H, I]   — input weights
      2: R   [num_directions, 3*H, H]   — hidden/recurrence weights
      3: B   [num_directions, 6*H]       — bias (optional)
      4: sequence_lens (optional)
      5: initial_h [batch, num_directions, H] (optional)

    ISA weight layout: W_ih[r|z|n] ++ W_hh[r|z|n]
    ISA bias layout:   B_ih[r|z|n] ++ B_hh[r|z|n]

    ONNX gate order:   z, r, n  (update, reset, new)
    ISA gate order:    r, z, n  (reset, update, new)
    → Gate reordering needed!

    Parameters
    ----------
    scratch_base:
        Byte offset in workspace where GRU scratch can be placed.
    """
    x = node.inputs[0]
    W_onnx = node.inputs[1]   # W_ih in ONNX
    R_onnx = node.inputs[2]   # W_hh (recurrence) in ONNX
    has_bias = len(node.inputs) >= 4
    has_initial_h = len(node.inputs) >= 6

    seq_len, batch, input_size = x.shape
    hidden_size = node.attrs.get("hidden_size", W_onnx.shape[1] // 3)
    direction = node.attrs.get("direction", "forward")

    if direction != "forward":
        raise NotImplementedError("Bidirectional GRU not yet supported")

    # ── Gate reorder: ONNX [z,r,n] → ISA [r,z,n] ────────────────
    W_data = W_onnx.data  # shape: [1, 3*H, I] for single-direction
    R_data = R_onnx.data  # shape: [1, 3*H, H]

    # Remove direction dim
    W_2d = W_data.reshape(-1, W_data.shape[-1])  # [3*H, I]
    R_2d = R_data.reshape(-1, R_data.shape[-1])  # [3*H, H]

    H = hidden_size
    # ONNX order: rows 0:H = z_gate, H:2H = r_gate, 2H:3H = n_gate
    # ISA order:  rows 0:H = r_gate, H:2H = z_gate, 2H:3H = n_gate
    W_reorder = np.concatenate([
        W_2d[H:2*H],    # r_gate
        W_2d[0:H],      # z_gate
        W_2d[2*H:3*H],  # n_gate
    ], axis=0)  # shape: [3*H, I]

    R_reorder = np.concatenate([
        R_2d[H:2*H],
        R_2d[0:H],
        R_2d[2*H:3*H],
    ], axis=0)  # shape: [3*H, H]

    # Concatenate W_ih ++ W_hh into a single contiguous weight blob.
    # The PL engine expects: first 3*H*I bytes = W_ih, next 3*H*H bytes = W_hh.
    W_bytes = W_reorder.astype(np.float32).tobytes()
    R_bytes = R_reorder.astype(np.float32).tobytes()
    packed_weight_data = np.frombuffer(W_bytes + R_bytes, dtype=np.float32).copy()

    weight_tensor = Tensor(
        name=f"{node.name}_packed_weight",
        shape=(len(packed_weight_data),),
        dtype="float32",
        data=packed_weight_data,
    )
    # We need to assign this an offset.  It'll go into extra_weights
    # which bin_writer will serialize after existing weights.

    # ── Bias reorder ─────────────────────────────────────────────
    bias_tensor = None
    if has_bias:
        B_data = node.inputs[3].data  # [1, 6*H]
        B_flat = B_data.reshape(-1)   # [6*H]
        # ONNX bias: [W_bz | W_br | W_bn | R_bz | R_br | R_bn]
        #            each chunk is H elements
        # ISA B_ih:  [W_br | W_bz | W_bn]
        # ISA B_hh:  [R_br | R_bz | R_bn]
        W_bz, W_br, W_bn = B_flat[0:H], B_flat[H:2*H], B_flat[2*H:3*H]
        R_bz, R_br, R_bn = B_flat[3*H:4*H], B_flat[4*H:5*H], B_flat[5*H:6*H]

        B_ih = np.concatenate([W_br, W_bz, W_bn])  # [3*H]
        B_hh = np.concatenate([R_br, R_bz, R_bn])  # [3*H]
        packed_bias = np.concatenate([B_ih, B_hh])  # [6*H]

        bias_tensor = Tensor(
            name=f"{node.name}_packed_bias",
            shape=(len(packed_bias),),
            dtype="float32",
            data=packed_bias.astype(np.float32),
        )

    # ── Scratch workspace for GRU internals ──────────────────────
    # PL needs ≥ 9·batch·H·4 bytes.  All GRU nodes share scratch at
    # scratch_base; the worst-case size extends total workspace.
    scratch_size = 9 * batch * H * 4
    scratch_offset = scratch_base

    # ── Fill instruction ─────────────────────────────────────────
    instr = Instruction(opcode=OP_GRU)
    instr.input0_addr = _offset(x)
    instr.output_addr = _offset(node.outputs[0])

    if has_initial_h:
        instr.input1_addr = _offset(node.inputs[5])
    else:
        instr.input1_addr = 0  # h0 all zeros — PL handles this

    # weight_addr and bias_addr point into the weight buffer.
    # These packed tensors will be serialised as extra weights.
    instr.weight_addr = -1  # placeholder, resolved by bin_writer
    instr.bias_addr = -1 if has_bias else 0
    instr.workspace_addr = scratch_offset

    instr.dim0 = input_size
    instr.dim1 = hidden_size
    instr.dim2 = batch

    if has_bias and bias_tensor is not None:
        instr.flags |= FLAG_HAS_BIAS
    if node.attrs.get("linear_before_reset", 0):
        instr.flags |= FLAG_LINEAR_BEFORE_RESET

    # seq_len for internal PL loop
    instr.seq_len = seq_len

    # ── Extra metadata for bin_writer ────────────────────────────
    # Attach extra tensors that need weight-buffer offsets.
    instr._gru_weight = weight_tensor
    instr._gru_bias = bias_tensor

    return [instr], scratch_size


def _handle_nop(node: Node, opcode: int) -> tuple[list[Instruction], int]:
    return [Instruction(opcode=OP_NOP)], 0


# ── Handler dispatch table ──────────────────────────────────────────
_HANDLERS: dict[int, Callable] = {
    OP_FC:        _handle_fc,
    OP_POOL:      _handle_pool,
    OP_RELU:      _handle_relu,
    OP_SIGMOID:   _handle_sigmoid,
    OP_TANH:      _handle_tanh,
    OP_GRU:       _handle_gru,
    OP_ELEM_MUL:  _handle_elem_binary,
    OP_ELEM_ADD:  _handle_elem_binary,
    OP_NOP:       _handle_nop,
}


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _offset(t: Tensor) -> int:
    """Return the byte offset of a tensor, or 0 if not yet assigned."""
    if t.offset is not None:
        return t.offset
    return 0


def _fuse_activation(instr: Instruction, node: Node) -> None:
    """Check if the FC output is immediately consumed by a single
    activation op that can be fused into the FC instruction."""
    out = node.outputs[0]
    # We'd need the graph-level consumer info for this.
    # For now this is a stub — activation fusion will be implemented
    # as a separate IR optimisation pass.
    pass
