/**
 * 2-D Pooling — line-buffer architecture with sliding window.
 *
 * Processes the feature map channel-by-channel, using BRAM line buffers
 * to cache KH rows of width W for efficient sliding-window access.
 */

#include "nn_pool.h"

void pool_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    const uint32_t H      = instr->dim0;      // input height
    const uint32_t W      = instr->dim1;      // input width
    const uint32_t C      = instr->dim2;      // channels
    const uint32_t KH     = instr->dim3;      // kernel height
    const uint32_t KW     = instr->dim4;      // kernel width
    const uint32_t stride = (instr->dim5 > 0) ? instr->dim5 : KH;
    const bool     is_max = (instr->flags & NN_FLAG_POOL_MAX) != 0;

    const uint32_t H_out  = (H - KH) / stride + 1;
    const uint32_t W_out  = (W - KW) / stride + 1;

    #define BYTE2F(o) ((o) >> 2)
    data_t *ws = ddr + BYTE2F(workspace_base);
    uint32_t in_off  = BYTE2F(instr->input0_addr);
    uint32_t out_off = BYTE2F(instr->output_addr);

    // ── Line buffer: (KH-1) rows cached in BRAM ────────────────────
    // We read data in NCHW order from DDR: C planes of H×W each.
    // For pooling, it's most efficient to process one channel at a time.

    for (uint32_t c = 0; c < C; c++) {
        // Buffer a sliding window of KH rows for this channel
        data_t  line_buf[KH][MAX_POOL_WIDTH];
        // line_buf: HLS auto-handles partitioning

        // Initialise first KH rows
        uint32_t ch_base = in_off + c * H * W;

        for (uint32_t ho = 0; ho < H_out; ho++) {
            // Load new rows into line buffer as we slide down
            for (uint32_t kh = 0; kh < KH; kh++) {
                uint32_t row = ho * stride + kh;
                if (row < H) {
                    for (uint32_t w = 0; w < W; w++) {
                        #pragma HLS PIPELINE II=1
                        line_buf[kh][w] = ws[ch_base + row * W + w];
                    }
                }
            }

            // Slide horizontally
            for (uint32_t wo = 0; wo < W_out; wo++) {
                data_t result;
                if (is_max) {
                    result = -1e30f;  // very negative
                } else {
                    result = 0.0f;
                }

                // Apply kernel window
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

                if (!is_max) {
                    result /= (data_t)(KH * KW);
                }

                // Write output
                ws[out_off + c * H_out * W_out + ho * W_out + wo] = result;
            }
        }
    }
}
