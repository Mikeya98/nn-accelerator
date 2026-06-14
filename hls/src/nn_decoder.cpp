/**
 * Instruction fetch & decode.
 */

#include "nn_decoder.h"
#include <string.h>

void fetch_instruction(
    data_t            *ddr,
    uint32_t           instr_base,
    uint32_t           pc,
    nn_instruction_t  *instr_out
) {
    // Instructions occupy 64 bytes each in DDR.
    // We read them as a raw buffer of 64 bytes (16 × uint32_t).
    //
    // DDR is byte-addressable and typed as data_t (float) here.
    // Instruction table is at (ddr + instr_base/4) in float-index space.
    // (instr_base is a byte offset; divide by 4 for 32-bit indexing.)
    //
    // We use uint32_t* alias for instruction access.

    const uint32_t INSTR_BYTES = sizeof(nn_instruction_t); // 64

    // Byte offset of instruction 'pc' within the instruction table
    uint32_t byte_offset = instr_base + pc * INSTR_BYTES;

    // DDR is accessed as float* ; we alias it as uint32_t* for raw reads.
    // HLS: using memcpy generates efficient AXI burst transactions.
    uint32_t raw[16];  // 64 B = 16 × uint32_t

    // raw: HLS auto-handles — keep as-is for decoding

    // Read 16 uint32 words from DDR
    for (int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        // Convert byte offset to uint32 word index
        uint32_t word_addr = (byte_offset >> 2) + i;  // >> 2 = /4
        raw[i] = ((volatile uint32_t*)ddr)[word_addr];
    }

    // ── Decode fields ───────────────────────────────────────────────
    // Word 0: [7:0]=opcode, [15:8]=flags, [31:16]=seq_len
    instr_out->opcode   = (uint8_t)( raw[0]        & 0xFF);
    instr_out->flags    = (uint8_t)((raw[0] >>  8) & 0xFF);
    instr_out->seq_len  = (uint16_t)((raw[0] >> 16) & 0xFFFF);

    // Words 1–7: address fields
    instr_out->input0_addr     = raw[1];
    instr_out->input1_addr     = raw[2];
    instr_out->output_addr     = raw[3];
    instr_out->weight_addr     = raw[4];
    instr_out->bias_addr       = raw[5];
    instr_out->workspace_addr  = raw[6];
    instr_out->scale_addr      = raw[7];

    // Words 8–15: dimension fields
    instr_out->dim0 = raw[8];
    instr_out->dim1 = raw[9];
    instr_out->dim2 = raw[10];
    instr_out->dim3 = raw[11];
    instr_out->dim4 = raw[12];
    instr_out->dim5 = raw[13];
    instr_out->dim6 = raw[14];
    instr_out->dim7 = raw[15];
}
