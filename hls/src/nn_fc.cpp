/**
 * Fully Connected — tiled matrix-vector multiply with FP32 accumulation.
 *
 * y[N] = x[M] · W[M×N] + b[N]
 *
 * The outer product is tiled over the output dimension N to fit weights
 * into on-chip BRAM.  Each tile processes VEC_WIDTH output features
 * in parallel using VEC_WIDTH DSP48 MAC lanes.
 */

#include "nn_fc.h"
#include "nn_act.h"

// ── Tiling parameters ────────────────────────────────────────────────
// NOTE: Vivado HLS 2018.3 pragma processor may not resolve #define
// macros inside pragma arguments.  These values are duplicated literally
// in the pragmas below; keep them in sync.
//   FC_TILE_N  = 8    (output features per tile)
//   FC_BRAM_W  = 1024 (max weight elements cached in BRAM per tile)

// DDR byte-offset → float-index helper (used throughout).
#define BYTE2FLOAT(off)  ((off) >> 2)

void fc_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                weight_base,
    uint32_t                workspace_base
) {
    const uint32_t M = instr->dim0;   // input features
    const uint32_t N = instr->dim1;   // output features

    data_t *ws_ptr = ddr + BYTE2FLOAT(workspace_base);
    data_t *wt_ptr = ddr + BYTE2FLOAT(weight_base);

    // ── Read input vector x[M] from workspace ──────────────────────
    data_t  x_buf[MAX_INPUT_FEATURES];
    // x_buf: HLS auto-infers BRAM from size

    uint32_t x_byte_off = instr->input0_addr;
    for (uint32_t i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        x_buf[i] = ws_ptr[BYTE2FLOAT(x_byte_off) + i];
    }

    // ── Output buffer ──────────────────────────────────────────────
    data_t  y_buf[MAX_OUTPUT_FEATURES];
    // y_buf: HLS auto-infers BRAM from size
    for (uint32_t j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        y_buf[j] = 0.0f;
    }

    // ── Bias load (if HAS_BIAS) ────────────────────────────────────
    if (instr->flags & NN_FLAG_HAS_BIAS) {
        uint32_t b_byte_off = instr->bias_addr;
        for (uint32_t j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            y_buf[j] = wt_ptr[BYTE2FLOAT(b_byte_off) + j];
        }
    }

    // ── Tiled matrix-vector multiply ───────────────────────────────
    // W is [M×N] row-major.  Element (i,j) is at W[i*N + j].
    //
    // We read weight tiles into local BRAM and accumulate:
    //   y[j] += Σ_i x[i] × W[i][j]

    uint32_t w_byte_off = instr->weight_addr;

    for (uint32_t i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        // Read one row of W into BRAM
        data_t  w_row[MAX_OUTPUT_FEATURES];
        // w_row: HLS auto-handles array partitioning

        for (uint32_t j = 0; j < N; j++) {
            // UNROLL removed for Vivado 2018.3 compat
            w_row[j] = wt_ptr[BYTE2FLOAT(w_byte_off) + i * N + j];
        }

        // Accumulate: y[j] += x[i] * W[i][j]
        data_t xi = x_buf[i];
        for (uint32_t j = 0; j < N; j++) {
            // UNROLL removed for Vivado 2018.3 compat
            y_buf[j] += xi * w_row[j];
        }
    }

    // ── Fused activation ───────────────────────────────────────────
    uint8_t act = instr->flags & NN_ACT_MASK;
    if (act == NN_ACT_RELU) {
        for (uint32_t j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            if (y_buf[j] < 0.0f) y_buf[j] = 0.0f;
        }
    } else if (act == NN_ACT_SIGMOID) {
        for (uint32_t j = 0; j < N; j++) {
            #pragma HLS PIPELINE
            y_buf[j] = sigmoid_f32(y_buf[j]);
        }
    } else if (act == NN_ACT_TANH) {
        for (uint32_t j = 0; j < N; j++) {
            #pragma HLS PIPELINE
            y_buf[j] = tanh_f32(y_buf[j]);
        }
    }

    // ── Write output y[N] to workspace ─────────────────────────────
    uint32_t y_byte_off = instr->output_addr;
    for (uint32_t j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        ws_ptr[BYTE2FLOAT(y_byte_off) + j] = y_buf[j];
    }
}
