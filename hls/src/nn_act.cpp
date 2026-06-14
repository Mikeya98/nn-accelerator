/**
 * Activation functions — optimised for FPGA (no hardware division).
 *
 * Sigmoid:   f(x) = 1 / (1 + exp(-x))
 * Tanh:      f(x) = tanh(x) = 2·σ(2x) − 1
 *
 * Both use piecewise linear approximation with 256-segment LUT
 * stored in BRAM (initialised as ROM).
 */

#include "nn_act.h"

// ── Sigmoid piecewise linear approximation ───────────────────────────
// We approximate σ(x) with 256 segments in the range [-8, 8].
// Outside this range, σ(x) ≈ 0 (x < -8) or ≈ 1 (x > 8).
//
// Each segment i stores:
//   slope[i]     — derivative at segment start
//   intercept[i] — function value at segment start
//
// σ(x) ≈ slope[idx] · (x - x_min) + intercept[idx]
// where idx = floor((x - x_min) * inv_step)

#define SIG_LUT_SIZE 256
#define SIG_X_MIN   (-8.0f)
#define SIG_X_MAX   ( 8.0f)
#define SIG_STEP    ((SIG_X_MAX - SIG_X_MIN) / (float)SIG_LUT_SIZE)
#define SIG_INV_STEP ((float)SIG_LUT_SIZE / (SIG_X_MAX - SIG_X_MIN))

// Pre-computed tables (initialised as ROM in BRAM)
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

data_t sigmoid_f32(data_t x) {
    #pragma HLS INLINE
    if (x <= SIG_X_MIN) return 0.0f;
    if (x >= SIG_X_MAX) return 1.0f;

    int idx = (int)((x - SIG_X_MIN) * SIG_INV_STEP);
    if (idx >= SIG_LUT_SIZE) idx = SIG_LUT_SIZE - 1;
    if (idx < 0) idx = 0;

    data_t x0 = SIG_X_MIN + (data_t)idx * SIG_STEP;
    return sig_intercept[idx] + sig_slope[idx] * (x - x0);
}

data_t tanh_f32(data_t x) {
    #pragma HLS INLINE
    // tanh(x) = 2·σ(2x) - 1
    return 2.0f * sigmoid_f32(2.0f * x) - 1.0f;
}

// ═══════════════════════════════════════════════════════════════════════

static void elem_act_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base,
    data_t (*act_fn)(data_t)
) {
    uint32_t N = instr->dim0;
    #define BYTE2FLOAT(o) ((o) >> 2)
    data_t *ws = ddr + BYTE2FLOAT(workspace_base);

    uint32_t in_off  = BYTE2FLOAT(instr->input0_addr);
    uint32_t out_off = BYTE2FLOAT(instr->output_addr);

    for (uint32_t i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        ws[out_off + i] = act_fn(ws[in_off + i]);
    }
}

void relu_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    #define R_BYTE2FLOAT(o) ((o) >> 2)
    data_t *ws = ddr + R_BYTE2FLOAT(workspace_base);
    uint32_t in_off  = R_BYTE2FLOAT(instr->input0_addr);
    uint32_t out_off = R_BYTE2FLOAT(instr->output_addr);

    for (uint32_t i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        data_t v = ws[in_off + i];
        ws[out_off + i] = (v > 0.0f) ? v : 0.0f;
    }
}

void sigmoid_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    elem_act_compute(ddr, instr, workspace_base, sigmoid_f32);
}

void tanh_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    elem_act_compute(ddr, instr, workspace_base, tanh_f32);
}
