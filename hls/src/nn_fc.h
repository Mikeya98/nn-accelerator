#ifndef NN_FC_H
#define NN_FC_H

#include "nn_params.h"
#include "nn_isa.h"

/**
 * Fully Connected layer:  y[N] = x[M] · W[M·N] + b[N]
 *
 * Weight matrix W is stored row-major in DDR at weight_base + weight_addr.
 * Input x and output y are in the workspace buffer.
 *
 * Supports optional fused activation (ReLU / Sigmoid / Tanh) via flags.
 */
void fc_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                weight_base,
    uint32_t                workspace_base
);

#endif /* NN_FC_H */
