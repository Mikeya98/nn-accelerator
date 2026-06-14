/**
 * NN Accelerator — Top-level PL Engine  (single-file, synthesis-safe)
 *
 * All compute functions are defined in this same translation unit
 * to avoid Vivado 2018.3's Pointer Array Geometry crash when the
 * m_axi ddr pointer crosses module boundaries.
 *
 * AXI Interfaces:
 *   s_axi_control  (AXI4-Lite Slave)  → register file
 *   m_axi_ddr      (AXI4 Master)      → DDR memory access
 */

#include "nn_engine.h"
#include "nn_isa.h"
#include "nn_params.h"

#include <string.h>

// ── Sigmoid LUT ──────────────────────────────────────────────────────
#define SIG_LUT_SIZE 256
#define SIG_X_MIN   (-8.0f)
#define SIG_X_MAX   ( 8.0f)
#define SIG_STEP    ((SIG_X_MAX - SIG_X_MIN) / (float)SIG_LUT_SIZE)
#define SIG_INV_STEP ((float)SIG_LUT_SIZE / (SIG_X_MAX - SIG_X_MIN))

static const data_t sig_slope[SIG_LUT_SIZE] = {
    #include "sigmoid_lut.inc"
};

static const data_t sig_intercept[SIG_LUT_SIZE] = {
    0.000335f, 0.000365f, 0.000397f, 0.000433f, 0.000471f, 0.000513f, 0.000559f, 0.000608f,
    0.000662f, 0.000721f, 0.000785f, 0.000855f, 0.000931f, 0.001014f, 0.001104f, 0.001202f,
    0.001309f, 0.001425f, 0.001552f, 0.001689f, 0.001839f, 0.002002f, 0.002179f, 0.002372f,
    0.002582f, 0.002810f, 0.003058f, 0.003328f, 0.003621f, 0.003940f, 0.004287f, 0.004664f,
    0.005074f, 0.005520f, 0.006004f, 0.006531f, 0.007103f, 0.007725f, 0.008401f, 0.009136f,
    0.009934f, 0.010801f, 0.011743f, 0.012765f, 0.013874f, 0.015077f, 0.016382f, 0.017796f,
    0.019329f, 0.020990f, 0.022789f, 0.024737f, 0.026846f, 0.029127f, 0.031594f, 0.034261f,
    0.037142f, 0.040253f, 0.043610f, 0.047231f, 0.051133f, 0.055336f, 0.059859f, 0.064723f,
    0.069949f, 0.075559f, 0.081574f, 0.088015f, 0.094902f, 0.102254f, 0.110087f, 0.118415f,
    0.127250f, 0.136598f, 0.146462f, 0.156839f, 0.167721f, 0.179092f, 0.190933f, 0.203216f,
    0.215908f, 0.228969f, 0.242355f, 0.256019f, 0.269909f, 0.283971f, 0.298151f, 0.312395f,
    0.326651f, 0.340869f, 0.355004f, 0.369015f, 0.382866f, 0.396527f, 0.409974f, 0.423188f,
    0.436154f, 0.448862f, 0.461305f, 0.473479f, 0.485383f, 0.497018f, 0.508386f, 0.519492f,
    0.530340f, 0.540937f, 0.551287f, 0.561399f, 0.571276f, 0.580927f, 0.590356f, 0.599570f,
    0.608574f, 0.617375f, 0.625978f, 0.634387f, 0.642610f, 0.650649f, 0.658510f, 0.666197f,
    0.673714f, 0.681065f, 0.688252f, 0.695280f, 0.702151f, 0.708868f, 0.715434f, 0.721852f,
    0.728124f, 0.734252f, 0.740240f, 0.746089f, 0.751802f, 0.757381f, 0.762828f, 0.768146f,
    0.773337f, 0.778403f, 0.783347f, 0.788171f, 0.792877f, 0.797467f, 0.801945f, 0.806311f,
    0.810569f, 0.814720f, 0.818767f, 0.822711f, 0.826556f, 0.830302f, 0.833953f, 0.837510f,
    0.840975f, 0.844351f, 0.847638f, 0.850840f, 0.853958f, 0.856994f, 0.859950f, 0.862828f,
    0.865629f, 0.868356f, 0.871010f, 0.873594f, 0.876108f, 0.878555f, 0.880937f, 0.883255f,
    0.885510f, 0.887705f, 0.889841f, 0.891919f, 0.893941f, 0.895908f, 0.897822f, 0.899684f,
    0.901496f, 0.903259f, 0.904974f, 0.906642f, 0.908266f, 0.909846f, 0.911383f, 0.912879f,
    0.914335f, 0.915752f, 0.917131f, 0.918473f, 0.919779f, 0.921051f, 0.922289f, 0.923495f,
    0.924669f, 0.925813f, 0.926927f, 0.928013f, 0.929071f, 0.930102f, 0.931107f, 0.932087f,
    0.933043f, 0.933975f, 0.934884f, 0.935771f, 0.936637f, 0.937482f, 0.938307f, 0.939113f,
    0.939900f, 0.940669f, 0.941420f, 0.942155f, 0.942874f, 0.943577f, 0.944265f, 0.944938f,
    0.945597f, 0.946243f, 0.946876f, 0.947496f, 0.948104f, 0.948700f, 0.949285f, 0.949859f,
    0.950422f, 0.950975f, 0.951518f, 0.952052f, 0.952576f, 0.953092f, 0.953599f, 0.954098f,
    0.954589f, 0.955072f, 0.955548f, 0.956017f, 0.956479f, 0.956934f, 0.957383f, 0.957826f,
    0.958263f, 0.958694f, 0.959119f, 0.959539f, 0.959954f, 0.960364f, 0.960769f, 0.961169f,
    0.961565f, 0.961956f, 0.962344f, 0.962727f, 0.963106f, 0.963482f, 0.963854f, 0.964223f
};

// ── Activation helpers ────────────────────────────────────────────

static data_t sigmoid_f32(data_t x) {
    if (x <= SIG_X_MIN) return 0.0f;
    if (x >= SIG_X_MAX) return 1.0f;
    int idx = (int)((x - SIG_X_MIN) * SIG_INV_STEP);
    if (idx >= SIG_LUT_SIZE) idx = SIG_LUT_SIZE - 1;
    if (idx < 0) idx = 0;
    data_t x0 = SIG_X_MIN + (data_t)idx * SIG_STEP;
    return sig_intercept[idx] + sig_slope[idx] * (x - x0);
}

static data_t tanh_f32(data_t x) {
    return 2.0f * sigmoid_f32(2.0f * x) - 1.0f;
}

// ── Byte → float index ────────────────────────────────────────────
#define BYTE2FLOAT(off)  ((off) >> 2)

// ══════════════════════════════════════════════════════════════════
//  Sub-functions: all DDR access through caller-provided pointers
//  BUT these are all in the SAME translation unit as nn_engine.
// ══════════════════════════════════════════════════════════════════

// ── Instruction fetch ─────────────────────────────────────────────

static void fetch_instruction(
    data_t            *ddr,
    uint32_t           instr_base,
    uint32_t           pc,
    nn_instruction_t  *instr_out
) {
    uint32_t byte_offset = instr_base + pc * sizeof(nn_instruction_t);
    uint32_t word_addr   = BYTE2FLOAT(byte_offset);

    // Read 16 uint32 words via memcpy (synthesizable, no pointer cast)
    uint32_t raw[16];
    for (int i = 0; i < 16; i++) {
        data_t tmp = ddr[word_addr + i];
        // Use union to avoid pointer reinterpretation
        union { data_t f; uint32_t u; } conv;
        conv.f = tmp;
        raw[i] = conv.u;
    }

    instr_out->opcode  = (uint8_t)( raw[0]        & 0xFF);
    instr_out->flags   = (uint8_t)((raw[0] >>  8) & 0xFF);
    instr_out->seq_len = (uint16_t)((raw[0] >> 16) & 0xFFFF);

    instr_out->input0_addr    = raw[1];
    instr_out->input1_addr    = raw[2];
    instr_out->output_addr    = raw[3];
    instr_out->weight_addr    = raw[4];
    instr_out->bias_addr      = raw[5];
    instr_out->workspace_addr = raw[6];
    instr_out->scale_addr     = raw[7];

    instr_out->dim0 = raw[8];
    instr_out->dim1 = raw[9];
    instr_out->dim2 = raw[10];
    instr_out->dim3 = raw[11];
    instr_out->dim4 = raw[12];
    instr_out->dim5 = raw[13];
    instr_out->dim6 = raw[14];
    instr_out->dim7 = raw[15];
}

// ── FC compute ────────────────────────────────────────────────────

static void fc_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                weight_base,
    uint32_t                workspace_base
) {
    const uint32_t M = instr->dim0;
    const uint32_t N = instr->dim1;

    data_t *ws_ptr = ddr + BYTE2FLOAT(workspace_base);
    data_t *wt_ptr = ddr + BYTE2FLOAT(weight_base);

    // Read x
    data_t x_buf[MAX_INPUT_FEATURES];
    uint32_t x_off = BYTE2FLOAT(instr->input0_addr);
    for (uint32_t i = 0; i < M; i++) {
        x_buf[i] = ws_ptr[x_off + i];
    }

    // Init y (with bias if present)
    data_t y_buf[MAX_OUTPUT_FEATURES];
    if (instr->flags & NN_FLAG_HAS_BIAS) {
        uint32_t b_off = BYTE2FLOAT(instr->bias_addr);
        for (uint32_t j = 0; j < N; j++) {
            y_buf[j] = wt_ptr[b_off + j];
        }
    } else {
        for (uint32_t j = 0; j < N; j++) {
            y_buf[j] = 0.0f;
        }
    }

    // MatMul: y[j] += sum_i x[i] * W[i][j]
    uint32_t w_off = BYTE2FLOAT(instr->weight_addr);
    for (uint32_t i = 0; i < M; i++) {
        data_t xi = x_buf[i];
        for (uint32_t j = 0; j < N; j++) {
            y_buf[j] += xi * wt_ptr[w_off + i * N + j];
        }
    }

    // Fused activation
    uint8_t act = instr->flags & NN_ACT_MASK;
    if (act == NN_ACT_RELU) {
        for (uint32_t j = 0; j < N; j++) {
            if (y_buf[j] < 0.0f) y_buf[j] = 0.0f;
        }
    } else if (act == NN_ACT_SIGMOID) {
        for (uint32_t j = 0; j < N; j++) {
            y_buf[j] = sigmoid_f32(y_buf[j]);
        }
    } else if (act == NN_ACT_TANH) {
        for (uint32_t j = 0; j < N; j++) {
            y_buf[j] = tanh_f32(y_buf[j]);
        }
    }

    // Write output
    uint32_t y_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t j = 0; j < N; j++) {
        ws_ptr[y_off + j] = y_buf[j];
    }
}

// ── Pooling compute ───────────────────────────────────────────────

static void pool_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    const uint32_t H      = instr->dim0;
    const uint32_t W      = instr->dim1;
    const uint32_t C      = instr->dim2;
    const uint32_t KH     = instr->dim3;
    const uint32_t KW     = instr->dim4;
    const uint32_t stride = (instr->dim5 > 0) ? instr->dim5 : KH;
    const bool     is_max = (instr->flags & NN_FLAG_POOL_MAX) != 0;

    const uint32_t H_out = (H - KH) / stride + 1;
    const uint32_t W_out = (W - KW) / stride + 1;

    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    uint32_t in_off  = BYTE2FLOAT(instr->input0_addr);
    uint32_t out_off = BYTE2FLOAT(instr->output_addr);

    for (uint32_t c = 0; c < C; c++) {
        data_t line_buf[MAX_POOL_HEIGHT][MAX_POOL_WIDTH];
        uint32_t ch_base = in_off + c * H * W;

        for (uint32_t ho = 0; ho < H_out; ho++) {
            for (uint32_t kh = 0; kh < KH; kh++) {
                uint32_t row = ho * stride + kh;
                if (row < H) {
                    for (uint32_t w = 0; w < W; w++) {
                        line_buf[kh][w] = ws[ch_base + row * W + w];
                    }
                }
            }
            for (uint32_t wo = 0; wo < W_out; wo++) {
                data_t result = is_max ? -1e30f : 0.0f;
                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        uint32_t col = wo * stride + kw;
                        if (col < W) {
                            data_t v = line_buf[kh][col];
                            if (is_max) {
                                if (v > result) result = v;
                            } else {
                                result += v;
                            }
                        }
                    }
                }
                if (!is_max) result /= (data_t)(KH * KW);
                ws[out_off + c * H_out * W_out + ho * W_out + wo] = result;
            }
        }
    }
}

// ── Element-wise activation ───────────────────────────────────────

static void relu_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    uint32_t in_off  = BYTE2FLOAT(instr->input0_addr);
    uint32_t out_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t i = 0; i < N; i++) {
        data_t v = ws[in_off + i];
        ws[out_off + i] = (v > 0.0f) ? v : 0.0f;
    }
}

static void sigmoid_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    uint32_t in_off  = BYTE2FLOAT(instr->input0_addr);
    uint32_t out_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t i = 0; i < N; i++) {
        ws[out_off + i] = sigmoid_f32(ws[in_off + i]);
    }
}

static void tanh_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    uint32_t in_off  = BYTE2FLOAT(instr->input0_addr);
    uint32_t out_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t i = 0; i < N; i++) {
        ws[out_off + i] = tanh_f32(ws[in_off + i]);
    }
}

// ── Element-wise binary ───────────────────────────────────────────

static void elem_mul_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    uint32_t a_off = BYTE2FLOAT(instr->input0_addr);
    uint32_t b_off = BYTE2FLOAT(instr->input1_addr);
    uint32_t o_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t i = 0; i < N; i++) {
        ws[o_off + i] = ws[a_off + i] * ws[b_off + i];
    }
}

static void elem_add_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    uint32_t a_off = BYTE2FLOAT(instr->input0_addr);
    uint32_t b_off = BYTE2FLOAT(instr->input1_addr);
    uint32_t o_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t i = 0; i < N; i++) {
        ws[o_off + i] = ws[a_off + i] + ws[b_off + i];
    }
}

// ── GRU compute ───────────────────────────────────────────────────

static void gru_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                weight_base,
    uint32_t                workspace_base,
    data_t                  scratch[GRU_SCRATCH_DEPTH]
) {
    const uint32_t I      = instr->dim0;
    const uint32_t H      = instr->dim1;
    const uint32_t B      = instr->dim2;
    const uint32_t seq_len = instr->seq_len;
    const bool has_bias   = (instr->flags & NN_FLAG_HAS_BIAS) != 0;
    const bool linear_before_reset = (instr->flags & NN_FLAG_LINEAR_BEFORE_RESET) != 0;

    data_t *ws = ddr + BYTE2FLOAT(workspace_base);
    data_t *wt = ddr + BYTE2FLOAT(weight_base);

    uint32_t w_byte_off = instr->weight_addr;
    const data_t *W_ih = wt + BYTE2FLOAT(w_byte_off);
    const data_t *W_hh = W_ih + 3 * H * I;
    const data_t *W_ir = W_ih + 0 * H * I;
    const data_t *W_iz = W_ih + 1 * H * I;
    const data_t *W_in = W_ih + 2 * H * I;
    const data_t *W_hr = W_hh + 0 * H * H;
    const data_t *W_hz = W_hh + 1 * H * H;
    const data_t *W_hn = W_hh + 2 * H * H;

    // Bias offsets (index into wt as float[], not pointer)
    uint32_t b_ih_off = 0;  // [3·H] — index relative to weight_base
    uint32_t b_hh_off = 0;  // [3·H]
    if (has_bias) {
        b_ih_off = BYTE2FLOAT(instr->bias_addr);
        b_hh_off = b_ih_off + 3 * H;
    }

    // Load initial hidden state
    data_t h_buf[GRU_MAX_HIDDEN];
    if (instr->input1_addr != 0) {
        uint32_t h_off = BYTE2FLOAT(instr->input1_addr);
        for (uint32_t j = 0; j < H; j++) h_buf[j] = ws[h_off + j];
    } else {
        for (uint32_t j = 0; j < H; j++) h_buf[j] = 0.0f;
    }

    uint32_t x_off = BYTE2FLOAT(instr->input0_addr);

    for (uint32_t t = 0; t < seq_len; t++) {
        // Load x_t
        data_t x_buf[MAX_INPUT_FEATURES];
        for (uint32_t i = 0; i < I; i++) {
            x_buf[i] = ws[x_off + t * I + i];
        }

        // Clear scratch
        for (uint32_t j = 0; j < 6 * H; j++) scratch[j] = 0.0f;

        // 6 partial MatMuls: vec × mat (row-major W via DDR pointer)
        // r_gate_input:  W_ir · x
        for (uint32_t i = 0; i < I; i++) {
            data_t xi = x_buf[i];
            for (uint32_t j = 0; j < H; j++) scratch[0*H + j] += xi * W_ir[i * H + j];
        }
        // r_gate_hidden: W_hr · h
        for (uint32_t i = 0; i < H; i++) {
            data_t hi = h_buf[i];
            for (uint32_t j = 0; j < H; j++) scratch[1*H + j] += hi * W_hr[i * H + j];
        }
        // z_gate_input:  W_iz · x
        for (uint32_t i = 0; i < I; i++) {
            data_t xi = x_buf[i];
            for (uint32_t j = 0; j < H; j++) scratch[2*H + j] += xi * W_iz[i * H + j];
        }
        // z_gate_hidden: W_hz · h
        for (uint32_t i = 0; i < H; i++) {
            data_t hi = h_buf[i];
            for (uint32_t j = 0; j < H; j++) scratch[3*H + j] += hi * W_hz[i * H + j];
        }
        // n_gate_input:  W_in · x
        for (uint32_t i = 0; i < I; i++) {
            data_t xi = x_buf[i];
            for (uint32_t j = 0; j < H; j++) scratch[4*H + j] += xi * W_in[i * H + j];
        }
        // n_gate_hidden: W_hn · h
        for (uint32_t i = 0; i < H; i++) {
            data_t hi = h_buf[i];
            for (uint32_t j = 0; j < H; j++) scratch[5*H + j] += hi * W_hn[i * H + j];
        }

        // Add biases (use integer offset index, not pointer)
        if (has_bias) {
            for (uint32_t j = 0; j < H; j++) {
                scratch[0*H + j] += wt[b_ih_off + 0*H + j];
                scratch[1*H + j] += wt[b_hh_off + 0*H + j];
                scratch[2*H + j] += wt[b_ih_off + 1*H + j];
                scratch[3*H + j] += wt[b_hh_off + 1*H + j];
                scratch[4*H + j] += wt[b_ih_off + 2*H + j];
                scratch[5*H + j] += wt[b_hh_off + 2*H + j];
            }
        }

        // Gates
        data_t r_buf[GRU_MAX_HIDDEN];
        data_t z_buf[GRU_MAX_HIDDEN];
        for (uint32_t j = 0; j < H; j++) {
            r_buf[j] = sigmoid_f32(scratch[0*H + j] + scratch[1*H + j]);
            z_buf[j] = sigmoid_f32(scratch[2*H + j] + scratch[3*H + j]);
        }

        if (linear_before_reset) {
            data_t rh_buf[GRU_MAX_HIDDEN];
            for (uint32_t j = 0; j < H; j++) {
                rh_buf[j] = r_buf[j] * h_buf[j];
            }
            for (uint32_t j = 0; j < H; j++) scratch[5*H + j] = 0.0f;
            for (uint32_t i = 0; i < H; i++) {
                data_t rhi = rh_buf[i];
                for (uint32_t j = 0; j < H; j++) scratch[5*H + j] += rhi * W_hn[i * H + j];
            }
            if (has_bias) {
                for (uint32_t j = 0; j < H; j++) scratch[5*H + j] += wt[b_hh_off + 2*H + j];
            }
        }

        // n = tanh(n_input + r ⊙ n_hidden),  h' = (1-z)*n + z*h
        for (uint32_t j = 0; j < H; j++) {
            data_t n_val = tanh_f32(scratch[4*H + j] + r_buf[j] * scratch[5*H + j]);
            h_buf[j] = (1.0f - z_buf[j]) * n_val + z_buf[j] * h_buf[j];
        }
    }

    // Write final hidden state
    uint32_t out_off = BYTE2FLOAT(instr->output_addr);
    for (uint32_t j = 0; j < H; j++) ws[out_off + j] = h_buf[j];
}

// ══════════════════════════════════════════════════════════════════
//  Top-level function
// ══════════════════════════════════════════════════════════════════

void nn_engine(
    volatile uint32_t  reg_file[32],
    data_t             *ddr
) {
    #pragma HLS INTERFACE s_axilite    port=reg_file   bundle=control
    #pragma HLS INTERFACE m_axi        port=ddr        bundle=ddr_port \
                        depth=2097152  offset=slave    latency=64
    #pragma HLS INTERFACE ap_ctrl_none port=return

    uint32_t instr_base     = reg_file[REG_INSTR_ADDR     / 4];
    uint32_t weight_base    = reg_file[REG_WEIGHT_ADDR    / 4];
    uint32_t input_base     = reg_file[REG_INPUT_ADDR     / 4];
    uint32_t output_base    = reg_file[REG_OUTPUT_ADDR    / 4];
    uint32_t workspace_base = reg_file[REG_WORKSPACE_ADDR / 4];

    volatile uint32_t ctrl_val = reg_file[REG_CTRL / 4];
    if (!(ctrl_val & CTRL_START)) return;

    data_t gru_scratch[GRU_SCRATCH_DEPTH];

    nn_instruction_t instr;
    uint32_t pc = 0;
    bool done = false;

    while (!done) {
        fetch_instruction(ddr, instr_base, pc, &instr);

        switch (instr.opcode) {
        case NN_OP_FC:
            fc_compute(ddr, &instr, weight_base, workspace_base);
            break;
        case NN_OP_RELU:
            relu_compute(ddr, &instr, workspace_base);
            break;
        case NN_OP_SIGMOID:
            sigmoid_compute(ddr, &instr, workspace_base);
            break;
        case NN_OP_TANH:
            tanh_compute(ddr, &instr, workspace_base);
            break;
        case NN_OP_POOL:
            pool_compute(ddr, &instr, workspace_base);
            break;
        case NN_OP_ELEM_MUL:
            elem_mul_compute(ddr, &instr, workspace_base);
            break;
        case NN_OP_ELEM_ADD:
            elem_add_compute(ddr, &instr, workspace_base);
            break;
        case NN_OP_GRU:
            gru_compute(ddr, &instr, weight_base, workspace_base, gru_scratch);
            break;
        case NN_OP_NOP:
            break;
        case NN_OP_END:
            done = true;
            break;
        default:
            done = true;
            break;
        }
        pc++;
    }

    reg_file[REG_STATUS / 4] = STATUS_DONE;
}
