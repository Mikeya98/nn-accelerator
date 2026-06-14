#ifndef NN_DECODER_H
#define NN_DECODER_H

#include "nn_params.h"
#include "nn_isa.h"

/**
 * Fetch a single 64-byte instruction from DDR.
 *
 * Instructions are stored as flat byte arrays in DDR.  Each
 * instruction is exactly INSTRUCTION_SIZE (64) bytes.
 *
 * @param ddr           Base pointer to DDR memory
 * @param instr_base    Byte offset of the instruction table in DDR
 * @param pc            Program counter (instruction index)
 * @param instr_out     Pointer to decoded instruction struct
 */
void fetch_instruction(
    data_t            *ddr,
    uint32_t           instr_base,
    uint32_t           pc,
    nn_instruction_t  *instr_out
);

#endif /* NN_DECODER_H */
