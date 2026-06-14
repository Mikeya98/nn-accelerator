"""
FP32 operator implementations — the arithmetic heart of the simulator.

Each function takes a `Memory` + decoded `Instruction` and performs
the computation exactly as the PL hardware would.

All addresses in the instruction are **byte offsets** relative to a
base address.  The executor resolves them to absolute DDR addresses
before calling these functions.

Gate order for GRU (ISA convention):
    gate 0 = r (reset), gate 1 = z (update), gate 2 = n (new)
"""

from __future__ import annotations

import numpy as np

from .memory import Memory
from .loader import Instruction
from compiler.isa import (
    FLAG_HAS_BIAS, FLAG_POOL_MAX, FLAG_LINEAR_BEFORE_RESET,
    ACT_MASK, ACT_RELU, ACT_SIGMOID, ACT_TANH,
)


def op_fc(mem: Memory, instr: Instruction, base_weight: int, base_workspace: int) -> None:
    """Fully Connected: y[M] = x[M] · W[M×N] + b[N]."""
    M, N = instr.dim0, instr.dim1

    x = mem.read_f32(base_workspace + instr.input0_addr, M)
    W = mem.read_f32(base_weight + instr.weight_addr, M * N).reshape(M, N)

    y = x @ W  # [M] @ [M×N] → [N]

    if instr.flags & FLAG_HAS_BIAS:
        b = mem.read_f32(base_weight + instr.bias_addr, N)
        y = y + b

    # Fused activation (bits [5:4] of flags)
    act = instr.flags & ACT_MASK
    if act == ACT_RELU:
        y = np.maximum(y, 0)
    elif act == ACT_SIGMOID:
        y = 1.0 / (1.0 + np.exp(-y))
    elif act == ACT_TANH:
        y = np.tanh(y)

    mem.write_f32(base_workspace + instr.output_addr, y)


def op_relu(mem: Memory, instr: Instruction, base_workspace: int) -> None:
    """Element-wise ReLU."""
    n = instr.dim0
    x = mem.read_f32(base_workspace + instr.input0_addr, n)
    y = np.maximum(x, 0)
    mem.write_f32(base_workspace + instr.output_addr, y)


def op_sigmoid(mem: Memory, instr: Instruction, base_workspace: int) -> None:
    """Element-wise Sigmoid."""
    n = instr.dim0
    x = mem.read_f32(base_workspace + instr.input0_addr, n)
    # Clip to avoid overflow
    x_clipped = np.clip(x, -20.0, 20.0)
    y = 1.0 / (1.0 + np.exp(-x_clipped))
    mem.write_f32(base_workspace + instr.output_addr, y)


def op_tanh(mem: Memory, instr: Instruction, base_workspace: int) -> None:
    """Element-wise Tanh."""
    n = instr.dim0
    x = mem.read_f32(base_workspace + instr.input0_addr, n)
    y = np.tanh(x)
    mem.write_f32(base_workspace + instr.output_addr, y)


def op_pool(mem: Memory, instr: Instruction, base_workspace: int) -> None:
    """2-D Pooling over a spatial feature map.

    Data layout: the ISA specifies HWC, but ONNX models use NCHW.
    We handle both by interpreting dims correctly:
        dim0=H, dim1=W, dim2=C, dim3=KH, dim4=KW, dim5=stride.

    The raw bytes in memory are NCHW (C planes of H×W each).
    We reshape as [C, H, W] and pool each channel independently.
    """
    H, W, C = instr.dim0, instr.dim1, instr.dim2
    KH, KW = instr.dim3, instr.dim4
    stride = instr.dim5 if instr.dim5 > 0 else KH
    is_max = bool(instr.flags & FLAG_POOL_MAX)

    H_out = (H - KH) // stride + 1
    W_out = (W - KW) // stride + 1

    # Data is NCHW in memory: this is the same as C planes of H×W
    feat = mem.read_f32(base_workspace + instr.input0_addr, C * H * W)
    feat = feat.reshape(C, H, W)  # [C, H, W] — channel-first

    pooled = np.zeros((C, H_out, W_out), dtype=np.float32)

    for c in range(C):
        ch = feat[c]  # [H, W]
        for ho in range(H_out):
            for wo in range(W_out):
                hs, ws = ho * stride, wo * stride
                patch = ch[hs:hs + KH, ws:ws + KW]  # [KH, KW]
                if is_max:
                    pooled[c, ho, wo] = np.max(patch)
                else:
                    pooled[c, ho, wo] = np.mean(patch)

    # Write in NCHW order (C first)
    mem.write_f32(base_workspace + instr.output_addr, pooled.ravel())


def op_elem_mul(mem: Memory, instr: Instruction, base_workspace: int) -> None:
    """Element-wise multiply: c[i] = a[i] * b[i]."""
    n = instr.dim0
    a = mem.read_f32(base_workspace + instr.input0_addr, n)
    b = mem.read_f32(base_workspace + instr.input1_addr, n)
    mem.write_f32(base_workspace + instr.output_addr, a * b)


def op_elem_add(mem: Memory, instr: Instruction, base_workspace: int) -> None:
    """Element-wise add: c[i] = a[i] + b[i]."""
    n = instr.dim0
    a = mem.read_f32(base_workspace + instr.input0_addr, n)
    b = mem.read_f32(base_workspace + instr.input1_addr, n)
    mem.write_f32(base_workspace + instr.output_addr, a + b)


def op_gru(
    mem: Memory,
    instr: Instruction,
    base_weight: int,
    base_workspace: int,
) -> None:
    """GRU cell — the most complex instruction.

    Weight layout (gate-reordered by compiler: r, z, n):
        weight = W_ih[3·H·I] ++ W_hh[3·H·H]
    Bias layout:
        bias   = B_ih[3·H] ++ B_hh[3·H]

    dim0 = I (input_size), dim1 = H (hidden_size), dim2 = batch_size
    seq_len from instruction field.
    """
    I = instr.dim0   # input_size
    H = instr.dim1   # hidden_size
    batch = instr.dim2
    seq_len = instr.seq_len

    # ── Load weights ──────────────────────────────────────────────
    # W_ih: 3 gates × [H, I], flattened row-major → 3*H*I floats
    w_ih_bytes = 3 * H * I * 4
    W_ih_flat = mem.read_f32(base_weight + instr.weight_addr, 3 * H * I)

    # W_hh: 3 gates × [H, H]
    W_hh_flat = mem.read_f32(base_weight + instr.weight_addr + w_ih_bytes, 3 * H * H)

    # Extract per-gate matrices: gate 0=r, 1=z, 2=n
    def _gate_W(W_flat, gate_idx):
        start = gate_idx * H
        if W_flat is W_ih_flat:
            return W_flat.reshape(3 * H, I)[start:start + H, :]  # [H, I]
        else:
            return W_flat.reshape(3 * H, H)[start:start + H, :]  # [H, H]

    W_ir = _gate_W(W_ih_flat, 0)  # reset:  [H, I]
    W_iz = _gate_W(W_ih_flat, 1)  # update: [H, I]
    W_in = _gate_W(W_ih_flat, 2)  # new:    [H, I]

    W_hr = _gate_W(W_hh_flat, 0)  # reset:  [H, H]
    W_hz = _gate_W(W_hh_flat, 1)  # update: [H, H]
    W_hn = _gate_W(W_hh_flat, 2)  # new:    [H, H]

    # ── Load biases (optional) ────────────────────────────────────
    has_bias = bool(instr.flags & FLAG_HAS_BIAS)
    b_ir = b_iz = b_in = np.zeros(H, dtype=np.float32)
    b_hr = b_hz = b_hn = np.zeros(H, dtype=np.float32)

    if has_bias:
        bias_flat = mem.read_f32(base_weight + instr.bias_addr, 6 * H)
        # B_ih: [3*H] with gate order r, z, n
        b_ir = bias_flat[0*H : 1*H]
        b_iz = bias_flat[1*H : 2*H]
        b_in = bias_flat[2*H : 3*H]
        # B_hh: [3*H]
        b_hr = bias_flat[3*H : 4*H]
        b_hz = bias_flat[4*H : 5*H]
        b_hn = bias_flat[5*H : 6*H]

    # ── Load input sequence ───────────────────────────────────────
    x = mem.read_f32(base_workspace + instr.input0_addr, seq_len * batch * I)
    x = x.reshape(seq_len, batch, I)  # time-major

    # Initial hidden state
    if instr.input1_addr != 0:
        h = mem.read_f32(base_workspace + instr.input1_addr, batch * H)
        h = h.reshape(batch, H)
    else:
        h = np.zeros((batch, H), dtype=np.float32)

    linear_before_reset = bool(instr.flags & FLAG_LINEAR_BEFORE_RESET)

    # ── Time-step loop ────────────────────────────────────────────
    for t in range(seq_len):
        x_t = x[t]  # [batch, I]

        # r = σ( W_ir·x + b_ir + W_hr·h + b_hr )
        r_linear = x_t @ W_ir.T + b_ir + h @ W_hr.T + b_hr
        r = 1.0 / (1.0 + np.exp(-np.clip(r_linear, -20.0, 20.0)))

        # z = σ( W_iz·x + b_iz + W_hz·h + b_hz )
        z_linear = x_t @ W_iz.T + b_iz + h @ W_hz.T + b_hz
        z = 1.0 / (1.0 + np.exp(-np.clip(z_linear, -20.0, 20.0)))

        # n = tanh( W_in·x + b_in + r ⊙ W_hn·h + b_hn )
        # In non-LBR mode, ONNX: h_t = tanh(W_n·X + r * (R_n·H) + W_bn + R_bn)
        #            i.e. hidden bias R_bn is NOT multiplied by r.
        # In LBR mode,    ONNX: h_t = tanh(W_n·X + R_n·(r * H) + W_bn + R_bn)
        if linear_before_reset:
            n_linear = x_t @ W_in.T + b_in + (r * h) @ W_hn.T + b_hn
        else:
            n_linear = x_t @ W_in.T + b_in + r * (h @ W_hn.T) + b_hn
        n = np.tanh(n_linear)

        # h' = (1-z) ⊙ n + z ⊙ h
        h = (1.0 - z) * n + z * h

    # ── Write final hidden state ──────────────────────────────────
    mem.write_f32(base_workspace + instr.output_addr, h.ravel())
