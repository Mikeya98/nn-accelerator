#ifndef NN_ISA_H
#define NN_ISA_H
/**
 * Neural Network Instruction Set Architecture — FP32
 *
 * This is the contract between compiler (PC), PS driver (ARM), and PL
 * compute engine (FPGA). Every instruction is exactly 64 bytes so the PL
 * can use a fixed-size BRAM instruction buffer.
 *
 * All addresses are BYTE OFFSETS relative to a base the PS loader fills
 * in at runtime — the compiler never bakes absolute DDR addresses.
 */

#include <stdint.h>

/* ================================================================
 * Opcodes
 * ================================================================ */

#define NN_OP_NOP        0x00
#define NN_OP_FC         0x01   /* Fully Connected:  y = x·W + b      */
#define NN_OP_CONV2D     0x02   /* 2D Convolution    (reserved)        */
#define NN_OP_POOL       0x03   /* Max / Average Pooling               */
#define NN_OP_RELU       0x04   /* f(x) = max(0, x)                    */
#define NN_OP_SIGMOID    0x05   /* f(x) = 1 / (1+exp(-x))             */
#define NN_OP_TANH       0x06   /* f(x) = tanh(x)                      */
#define NN_OP_GRU        0x07   /* Gated Recurrent Unit cell           */
#define NN_OP_ELEM_MUL   0x08   /* c[i] = a[i] * b[i]                  */
#define NN_OP_ELEM_ADD   0x09   /* c[i] = a[i] + b[i]                  */
#define NN_OP_END        0xFF   /* Terminate execution                  */

/* ================================================================
 * Flags byte (nn_instruction_t.flags)
 * ================================================================ */

#define NN_FLAG_HAS_BIAS              0x01   /* MatMul + bias          */
#define NN_FLAG_POOL_MAX              0x02   /* 1=max  0=avg  (POOL)   */
#define NN_FLAG_LINEAR_BEFORE_RESET   0x02   /* ONNX attr    (GRU)     */
#define NN_FLAG_STORE_INTERMEDIATE    0x04   /* Store all h_t  (GRU)   */

/* Activation fused into compute ops — bits [5:4] */
#define NN_ACT_NONE     0x00
#define NN_ACT_RELU     0x10
#define NN_ACT_SIGMOID  0x20
#define NN_ACT_TANH     0x30
#define NN_ACT_MASK     0x30

/* ================================================================
 * Instruction word  (64 bytes, 8-byte aligned)
 * ================================================================ */

typedef struct {
    uint8_t  opcode;          /* [0x00]                              */
    uint8_t  flags;           /* [0x01] bias / activation / variant  */
    uint16_t seq_len;         /* [0x02] time-steps for GRU           */

    uint32_t input0_addr;     /* [0x04] primary data                 */
    uint32_t input1_addr;     /* [0x08] secondary (h_prev, B, ...)   */
    uint32_t output_addr;     /* [0x0C] result                       */
    uint32_t weight_addr;     /* [0x10] weight matrix                */
    uint32_t bias_addr;       /* [0x14] bias vector   (if HAS_BIAS)  */
    uint32_t workspace_addr;  /* [0x18] scratchpad for intermediates */
    uint32_t scale_addr;      /* [0x1C] reserved (quantized future)  */

    uint32_t dim0;            /* [0x20]                              */
    uint32_t dim1;            /* [0x24]                              */
    uint32_t dim2;            /* [0x28]                              */
    uint32_t dim3;            /* [0x2C]                              */
    uint32_t dim4;            /* [0x30]                              */
    uint32_t dim5;            /* [0x34]                              */
    uint32_t dim6;            /* [0x38]                              */
    uint32_t dim7;            /* [0x3C]                              */
} nn_instruction_t;

/*
 * Compile-time size check.
 * _Static_assert is C11; Vivado HLS synthesis handles it, but the
 * C-simulation GCC may compile in C++98 mode where it doesn't exist.
 * We use a portable trick that works everywhere.
 */
#ifdef __cplusplus
  /* C++: use static_assert if available (C++11+), else skip. */
  #if __cplusplus >= 201103L
    static_assert(sizeof(nn_instruction_t) == 64,
                  "Instruction must be exactly 64 bytes");
  #endif
#else
  /* C: use _Static_assert if C11+, else skip. */
  #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Static_assert(sizeof(nn_instruction_t) == 64,
                   "Instruction must be exactly 64 bytes");
  #endif
#endif

/* ================================================================
 * Per-opcode field mapping
 * ================================================================
 *
 * ── FC ───────────────────────────────────────────────────────────
 *   y = x[M] · W[M×N] + b[N]   (+ optional activation)
 *
 *   input0   →  x[M]              dim0 = M  (input features)
 *   weight   →  W[M][N] row-major dim1 = N  (output features)
 *   bias     →  b[N]
 *   output   →  y[N]
 *
 * ── POOL ─────────────────────────────────────────────────────────
 *   2-D pooling over every channel independently.
 *
 *   input0   →  feature map [H][W][C]
 *   output   →  pooled map
 *   dim0 = H_in     dim1 = W_in     dim2 = C
 *   dim3 = KH       dim4 = KW       dim5 = stride  (0 → ==KH,KW)
 *   flags:  NN_FLAG_POOL_MAX set = max-pool;  clear = avg-pool
 *
 * ── RELU / SIGMOID / TANH ────────────────────────────────────────
 *   Element-wise on dim0 elements.
 *   input0 → x[dim0]     output → y[dim0]
 *
 * ── ELEM_MUL / ELEM_ADD ──────────────────────────────────────────
 *   Binary element-wise on dim0 elements.
 *   input0 → a[dim0]     input1 → b[dim0]     output → c[dim0]
 *
 * ── GRU ──────────────────────────────────────────────────────────
 *   ONNX-compatible GRU cell (single-layer, forward only).
 *
 *   input0     → x  [seq_len][batch][input_size]  (time-major)
 *   input1     → h0 [batch][hidden_size]           (init state)
 *   output     → hT [batch][hidden_size]            (final state)
 *   weight     → W_ih [3·H][I] ++ W_hh [3·H][H]   (concatenated)
 *   bias       → B_ih [3·H]   ++ B_hh [3·H]        (if HAS_BIAS)
 *   workspace  → scratch, ≥ 9·batch·H·sizeof(float)
 *
 *   dim0 = input_size    (I)
 *   dim1 = hidden_size   (H)
 *   dim2 = batch_size
 *
 *   Weight layout  (row-major, gate-stacked):
 *     W_ih:  [reset_gate | update_gate | new_gate]  each H×I
 *     W_hh:  [reset_gate | update_gate | new_gate]  each H×H
 *     B_ih:  [R_bias | Z_bias | N_bias]             each H
 *     B_hh:  [R_bias | Z_bias | N_bias]             each H
 *
 *   Gate computation  (standard GRU):
 *     r = σ( W_ir·x + b_ir + W_hr·h + b_hr )
 *     z = σ( W_iz·x + b_iz + W_hz·h + b_hz )
 *     n = tanh( W_in·x + b_in + r ⊙ (W_hn·h + b_hn) )
 *     h' = (1-z) ⊙ n + z ⊙ h
 *
 *   FLAG_LINEAR_BEFORE_RESET changes n-gate to:
 *     n = tanh( W_in·x + b_in + W_hn·(r ⊙ h) + b_hn )
 *
 *   If seq_len > 1 the PL loops internally; the PS only sees the
 *   final h.  Set FLAG_STORE_INTERMEDIATE to write every h_t
 *   contiguously at output_addr  (needs seq_len·batch·H floats).
 */

#endif /* NN_ISA_H */
