#ifndef NN_POOL_H
#define NN_POOL_H

#include "nn_params.h"
#include "nn_isa.h"

/**
 * 2-D Pooling over a H×W×C feature map.
 *
 * Uses a BRAM line-buffer architecture.  dim0=H, dim1=W, dim2=C.
 * dim3=KH, dim4=KW, dim5=stride.
 */
void pool_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
);

#endif /* NN_POOL_H */
