#ifndef NN_GRU_H
#define NN_GRU_H

#include "nn_params.h"
#include "nn_isa.h"

/**
 * GRU cell — multi-time-step gated recurrent unit.
 *
 * This is the most complex instruction.  The GRU FSM sequences through
 * 6 MatMuls per time step (3 gates × 2 weight matrices each) and then
 * applies the gate equations.
 *
 * Weight layout in DDR (gate-reordered by compiler to ISA order: r, z, n):
 *   W_ih [3·H·I]  — input-to-hidden weights for gates r, z, n
 *   W_hh [3·H·H]  — hidden-to-hidden weights for gates r, z, n
 * Bias layout:
 *   B_ih [3·H]    — input biases for gates r, z, n
 *   B_hh [3·H]    — hidden biases for gates r, z, n
 *
 * @param gru_scratch  Local BRAM scratchpad (GRU_SCRATCH_DEPTH elements).
 */
void gru_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                weight_base,
    uint32_t                workspace_base,
    data_t                  gru_scratch[GRU_SCRATCH_DEPTH]
);

#endif /* NN_GRU_H */
