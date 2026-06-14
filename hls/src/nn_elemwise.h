#ifndef NN_ELEMWISE_H
#define NN_ELEMWISE_H

#include "nn_params.h"
#include "nn_isa.h"

void elem_mul_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
);

void elem_add_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
);

#endif /* NN_ELEMWISE_H */
