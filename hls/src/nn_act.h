#ifndef NN_ACT_H
#define NN_ACT_H

#include "nn_params.h"
#include "nn_isa.h"

// ── Activation function declarations ────────────────────────────────

/** Fast FP32 sigmoid using piecewise linear approximation. */
data_t sigmoid_f32(data_t x);

/** Fast FP32 tanh using piecewise linear approximation. */
data_t tanh_f32(data_t x);

// ── Element-wise activation ops ─────────────────────────────────────

void relu_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
);

void sigmoid_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
);

void tanh_compute(
    data_t                 *ddr,
    const nn_instruction_t *instr,
    uint32_t                workspace_base
);

#endif /* NN_ACT_H */
