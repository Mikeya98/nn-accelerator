/**
 * GRU compute engine — FSM-driven sequential MatMul + gate logic.
 *
 * ==================== Gate equations (ISA convention) =================
 *
 *   r = σ( W_ir·x + b_ir + W_hr·h + b_hr )
 *   z = σ( W_iz·x + b_iz + W_hz·h + b_hz )
 *   n = tanh( W_in·x + b_in + r * (W_hn·h) + b_hn )   (non-LBR)
 *   h' = (1-z) * n + z * h
 *
 * Gate order in packed weight:  0=r, 1=z, 2=n
 *
 * ==================== Scratch layout =================================
 *
 *   gru_scratch usage:
 *     [0     ..  H-1]   : r_gate_input  (W_ir·x + b_ir)
 *     [H     .. 2H-1]   : r_gate_hidden (W_hr·h + b_hr)
 *     [2H    .. 3H-1]   : z_gate_input
 *     [3H    .. 4H-1]   : z_gate_hidden
 *     [4H    .. 5H-1]   : n_gate_input
 *     [5H    .. 6H-1]   : n_gate_hidden
 */

#include "nn_gru.h"
#include "nn_act.h"

// ── Internal helpers ─────────────────────────────────────────────────

/**
 * Vector × Matrix multiply:  y[N] = x[M] · W[M×N] (row-major W).
 * Both x and y are in BRAM (local).  W is in DDR.
 */
static void vec_mat_mul(
    const data_t  x[],        // [M]
    uint32_t      M,
    const data_t *W_ddr,      // [M×N] row-major in DDR
    uint32_t      N,
    data_t        y[]         // [N] output (must be pre-initialised)
) {
    for (uint32_t i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        data_t xi = x[i];
        for (uint32_t j = 0; j < N; j++) {
            // UNROLL removed for Vivado 2018.3 compat
            y[j] += xi * W_ddr[i * N + j];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════

void gru_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                weight_base,
    uint32_t                workspace_base,
    data_t                  scratch[GRU_SCRATCH_DEPTH]
) {
    #pragma HLS INLINE off

    const uint32_t I      = instr->dim0;  // input_size
    const uint32_t H      = instr->dim1;  // hidden_size
    const uint32_t B      = instr->dim2;  // batch_size
    const uint32_t seq_len = instr->seq_len;
    const bool has_bias   = (instr->flags & NN_FLAG_HAS_BIAS) != 0;
    const bool linear_before_reset = (instr->flags & NN_FLAG_LINEAR_BEFORE_RESET) != 0;

    #define GBYTE2F(o) ((o) >> 2)

    data_t *ws = ddr + GBYTE2F(workspace_base);
    data_t *wt = ddr + GBYTE2F(weight_base);

    // ── Weight pointers ────────────────────────────────────────────
    uint32_t w_byte_off = instr->weight_addr;
    const data_t *W_ih = wt + GBYTE2F(w_byte_off);   // [3·H·I]
    const data_t *W_hh = W_ih + 3 * H * I;            // [3·H·H]

    // Per-gate weight pointers (gate order: 0=r, 1=z, 2=n)
    const data_t *W_ir = W_ih + 0 * H * I;   // r: [H×I]
    const data_t *W_iz = W_ih + 1 * H * I;   // z: [H×I]
    const data_t *W_in = W_ih + 2 * H * I;   // n: [H×I]
    const data_t *W_hr = W_hh + 0 * H * H;   // r: [H×H]
    const data_t *W_hz = W_hh + 1 * H * H;   // z: [H×H]
    const data_t *W_hn = W_hh + 2 * H * H;   // n: [H×H]

    // ── Bias pointers ──────────────────────────────────────────────
    const data_t *B_ih = NULL;
    const data_t *B_hh = NULL;
    if (has_bias) {
        uint32_t b_byte_off = instr->bias_addr;
        B_ih = wt + GBYTE2F(b_byte_off);       // [3·H]
        B_hh = B_ih + 3 * H;                    // [3·H]
    }

    // ── Load initial hidden state ──────────────────────────────────
    data_t  h_buf[GRU_MAX_HIDDEN];
    // h_buf: HLS auto-infers BRAM
    if (instr->input1_addr != 0) {
        uint32_t h_off = GBYTE2F(instr->input1_addr);
        for (uint32_t j = 0; j < H; j++) {
            #pragma HLS PIPELINE II=1
            h_buf[j] = ws[h_off + j];
        }
    } else {
        for (uint32_t j = 0; j < H; j++) {
            #pragma HLS PIPELINE II=1
            h_buf[j] = 0.0f;
        }
    }

    // ── Input sequence pointer ─────────────────────────────────────
    uint32_t x_off = GBYTE2F(instr->input0_addr);

    // ── Time-step loop ─────────────────────────────────────────────
    for (uint32_t t = 0; t < seq_len; t++) {

        // Load x_t [I]
        data_t  x_buf[MAX_INPUT_FEATURES];
        // x_buf: HLS auto-infers BRAM
        for (uint32_t i = 0; i < I; i++) {
            #pragma HLS PIPELINE II=1
            x_buf[i] = ws[x_off + t * I + i];  // batch=1 simplified
        }

        // ── CLEAR scratch for this time step ──────────────────────
        for (uint32_t j = 0; j < 6 * H; j++) {
            #pragma HLS PIPELINE II=1
            scratch[j] = 0.0f;
        }

        // ── Compute 6 partial MatMul results into scratch ─────────
        // r_gate_input:  W_ir · x
        vec_mat_mul(x_buf, I, W_ir, H, scratch + 0 * H);
        // r_gate_hidden: W_hr · h
        vec_mat_mul(h_buf, H, W_hr, H, scratch + 1 * H);
        // z_gate_input:  W_iz · x
        vec_mat_mul(x_buf, I, W_iz, H, scratch + 2 * H);
        // z_gate_hidden: W_hz · h
        vec_mat_mul(h_buf, H, W_hz, H, scratch + 3 * H);
        // n_gate_input:  W_in · x
        vec_mat_mul(x_buf, I, W_in, H, scratch + 4 * H);
        // n_gate_hidden: W_hn · h
        vec_mat_mul(h_buf, H, W_hn, H, scratch + 5 * H);

        // ── Add biases ────────────────────────────────────────────
        if (has_bias) {
            for (uint32_t j = 0; j < H; j++) {
                #pragma HLS PIPELINE II=1
                scratch[0*H + j] += B_ih[0*H + j];  // b_ir
                scratch[1*H + j] += B_hh[0*H + j];  // b_hr
                scratch[2*H + j] += B_ih[1*H + j];  // b_iz
                scratch[3*H + j] += B_hh[1*H + j];  // b_hz
                scratch[4*H + j] += B_ih[2*H + j];  // b_in
                scratch[5*H + j] += B_hh[2*H + j];  // b_hn
            }
        }

        // ── Gate computations ─────────────────────────────────────
        // r = σ(r_input + r_hidden)
        data_t r_buf[GRU_MAX_HIDDEN];
        data_t z_buf[GRU_MAX_HIDDEN];
        // r_buf, z_buf: HLS auto-infers BRAM
        // (original: RESOURCE variable=r_buf core=RAM_2P_BRAM)
        // (original: RESOURCE variable=z_buf core=RAM_2P_BRAM)

        for (uint32_t j = 0; j < H; j++) {
            #pragma HLS PIPELINE II=1
            data_t r_sum = scratch[0*H + j] + scratch[1*H + j];
            r_buf[j] = sigmoid_f32(r_sum);

            data_t z_sum = scratch[2*H + j] + scratch[3*H + j];
            z_buf[j] = sigmoid_f32(z_sum);
        }

        // ── n = tanh(n_input + r * n_hidden + b_hn) ───────────────
        //   (non-LBR): r * n_hidden
        //   (LBR):     W_hn · (r * h)  — this replaces n_hidden entirely
        //
        // For LBR, we need to recompute the hidden part.
        // Here we handle both cases:
        if (linear_before_reset) {
            // Recompute n_gate_hidden: W_hn · (r ⊙ h)
            data_t rh_buf[GRU_MAX_HIDDEN];
            // rh_buf: HLS auto-infers BRAM
            for (uint32_t j = 0; j < H; j++) {
                #pragma HLS PIPELINE II=1
                rh_buf[j] = r_buf[j] * h_buf[j];
            }
            for (uint32_t j = 0; j < H; j++) {
                #pragma HLS PIPELINE II=1
                scratch[5*H + j] = 0.0f;
            }
            vec_mat_mul(rh_buf, H, W_hn, H, scratch + 5 * H);
            if (has_bias) {
                for (uint32_t j = 0; j < H; j++) {
                    #pragma HLS PIPELINE II=1
                    scratch[5*H + j] += B_hh[2*H + j];
                }
            }
        }

        // n = tanh( n_input + r ⊙ n_hidden )
        for (uint32_t j = 0; j < H; j++) {
            #pragma HLS PIPELINE II=1
            data_t n_sum = scratch[4*H + j] + r_buf[j] * scratch[5*H + j];
            data_t n_val = tanh_f32(n_sum);

            // h' = (1 - z) * n + z * h
            h_buf[j] = (1.0f - z_buf[j]) * n_val + z_buf[j] * h_buf[j];
        }

    } // end time-step loop

    // ── Write final hidden state ───────────────────────────────────
    uint32_t out_off = GBYTE2F(instr->output_addr);
    for (uint32_t j = 0; j < H; j++) {
        #pragma HLS PIPELINE II=1
        ws[out_off + j] = h_buf[j];
    }
}
