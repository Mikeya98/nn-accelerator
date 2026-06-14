/**
 * Element-wise binary operations — fully pipelined, II=1.
 */

#include "nn_elemwise.h"

void elem_mul_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    #define EM_BYTE2F(o) ((o) >> 2)
    data_t *ws = ddr + EM_BYTE2F(workspace_base);
    uint32_t a_off = EM_BYTE2F(instr->input0_addr);
    uint32_t b_off = EM_BYTE2F(instr->input1_addr);
    uint32_t o_off = EM_BYTE2F(instr->output_addr);

    for (uint32_t i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        ws[o_off + i] = ws[a_off + i] * ws[b_off + i];
    }
}

void elem_add_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
) {
    uint32_t N = instr->dim0;
    #define EA_BYTE2F(o) ((o) >> 2)
    data_t *ws = ddr + EA_BYTE2F(workspace_base);
    uint32_t a_off = EA_BYTE2F(instr->input0_addr);
    uint32_t b_off = EA_BYTE2F(instr->input1_addr);
    uint32_t o_off = EA_BYTE2F(instr->output_addr);

    for (uint32_t i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        ws[o_off + i] = ws[a_off + i] + ws[b_off + i];
    }
}
