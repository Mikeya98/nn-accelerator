"""
Python equivalent of nn_isa.h constants.

These mirror the C header exactly so the compiler can generate
correct instruction words without cross-language drift.
"""

# ── Opcodes ──────────────────────────────────────────────────────────
OP_NOP        = 0x00
OP_FC         = 0x01   # Fully Connected:  y = x·W + b
OP_CONV2D     = 0x02   # 2D Convolution (reserved)
OP_POOL       = 0x03   # Max / Average Pooling
OP_RELU       = 0x04   # f(x) = max(0, x)
OP_SIGMOID    = 0x05   # f(x) = 1 / (1+exp(-x))
OP_TANH       = 0x06   # f(x) = tanh(x)
OP_GRU        = 0x07   # Gated Recurrent Unit cell
OP_ELEM_MUL   = 0x08   # c[i] = a[i] * b[i]
OP_ELEM_ADD   = 0x09   # c[i] = a[i] + b[i]
OP_END        = 0xFF   # Terminate execution

# ── Flags ────────────────────────────────────────────────────────────
FLAG_HAS_BIAS              = 0x01
FLAG_POOL_MAX              = 0x02   # 1=max  0=avg  (POOL)
FLAG_LINEAR_BEFORE_RESET   = 0x02   # ONNX attr (GRU)
FLAG_STORE_INTERMEDIATE    = 0x04   # Store all h_t (GRU)

# Activation fused into compute ops — bits [5:4]
ACT_NONE     = 0x00
ACT_RELU     = 0x10
ACT_SIGMOID  = 0x20
ACT_TANH     = 0x30
ACT_MASK     = 0x30

# ── Instruction size ─────────────────────────────────────────────────
INSTRUCTION_SIZE = 64  # bytes

# ── Human-readable tables ────────────────────────────────────────────
OPCODE_NAMES: dict[int, str] = {
    OP_NOP:       "NOP",
    OP_FC:        "FC",
    OP_CONV2D:    "CONV2D",
    OP_POOL:      "POOL",
    OP_RELU:      "RELU",
    OP_SIGMOID:   "SIGMOID",
    OP_TANH:      "TANH",
    OP_GRU:       "GRU",
    OP_ELEM_MUL:  "ELEM_MUL",
    OP_ELEM_ADD:  "ELEM_ADD",
    OP_END:       "END",
}

# ONNX op → (opcode, default_flags) mapping
ONNX_OP_MAP: dict[str, tuple[int, int]] = {
    "Gemm":         (OP_FC,       FLAG_HAS_BIAS),
    "MatMul":       (OP_FC,       0),             # no bias unless Add follows
    "Relu":         (OP_RELU,     0),
    "MaxPool":      (OP_POOL,     FLAG_POOL_MAX),
    "AveragePool":  (OP_POOL,     0),
    "GlobalMaxPool":      (OP_POOL,     FLAG_POOL_MAX),
    "GlobalAveragePool":  (OP_POOL,     0),
    "Sigmoid":      (OP_SIGMOID,  0),
    "Tanh":         (OP_TANH,     0),
    "GRU":          (OP_GRU,      0),
    "Mul":          (OP_ELEM_MUL, 0),
    "Add":          (OP_ELEM_ADD, 0),
}

# Shape-only ops — folded at compile time, no PL instruction emitted
SHAPE_OPS = {"Reshape", "Transpose", "Concat", "Squeeze", "Unsqueeze", "Flatten", "Constant"}
