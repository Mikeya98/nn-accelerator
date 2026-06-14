#ifndef NN_ENGINE_H
#define NN_ENGINE_H
/**
 * NN Accelerator — Top-level PL Engine IP
 *
 * Interface:
 *   s_axi_control   AXI4-Lite Slave   (register access from PS)
 *   m_axi_ddr       AXI4 Master       (DDR read/write)
 *   interrupt       1-bit output      (asserted on STATUS_DONE)
 *
 * Usage in Vivado Block Design:
 *   1. Drop nn_engine IP into the design.
 *   2. Connect s_axi_control to PS GP0 (M_AXI_GP0).
 *   3. Connect m_axi_ddr to PS HP0 (S_AXI_HP0).
 *   4. Connect interrupt to PS IRQ_F2P[0].
 *   5. Assign register base address in Address Editor.
 */

#include "nn_params.h"
#include "nn_isa.h"

// ── Top-level function ────────────────────────────────────────────────

/**
 * Main PL engine entry point.
 *
 * Sleeps until CTRL.START is written, then:
 *   1. Reads instructions from DDR (via m_axi_ddr).
 *   2. Decodes and executes each instruction.
 *   3. Writes results back to DDR.
 *   4. Sets STATUS = DONE, asserts interrupt.
 *
 * @param[in,out]  reg_file   Internal register file (mapped to s_axi)
 * @param[in]      ddr_base   Base address of the DDR memory space
 */
void nn_engine(
    volatile uint32_t  reg_file[32],   // s_axi_control → register file
    data_t            *ddr             // m_axi_ddr → DDR pointer
);

#endif /* NN_ENGINE_H */
