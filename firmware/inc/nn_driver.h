#ifndef NN_DRIVER_H
#define NN_DRIVER_H
/**
 * NN Accelerator — AXI4-Lite Register Driver
 *
 * Minimal MMIO driver for the PL nn_engine IP's s_axi_control interface.
 * All register accesses go through the two primitives nn_reg_read() and
 * nn_reg_write(); higher-level functions build on top of them.
 *
 * The register map is defined in nn_platform.h and matches
 * nn_accelerator/hls/src/nn_params.h exactly.
 */

#include "nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Initialization ──────────────────────────────────────────────────── */

/**
 * Initialize the register driver.  Maps the AXI-Lite base address and
 * performs a sanity read to verify the register file is accessible.
 *
 * Returns 0 on success, -1 if the AXI base address appears unreachable.
 */
int nn_driver_init(void);

/* ── Low-level MMIO ──────────────────────────────────────────────────── */

/**
 * Write a 32-bit value to a PL register at `byte_offset` from AXI_BASE.
 * Includes a DMB memory barrier to ensure the store reaches the PL.
 */
void nn_reg_write(uint32_t byte_offset, uint32_t value);

/**
 * Read a 32-bit value from a PL register at `byte_offset` from AXI_BASE.
 */
uint32_t nn_reg_read(uint32_t byte_offset);

/* ── PL control ──────────────────────────────────────────────────────── */

/**
 * Configure all five address registers for a loaded BIN.
 *
 * @param instr_base     DDR physical address of the instruction buffer
 *                       (typically NN_BIN_BASE + 256).
 * @param weight_base    DDR physical address of the weight buffer
 *                       (typically NN_BIN_BASE + 256 + N*64).
 * @param workspace_base DDR physical address of the workspace
 *                       (NN_WORKSPACE_BASE).
 *
 * REG_INPUT_ADDR and REG_OUTPUT_ADDR are set to `workspace_base` — the
 * compiler assigns non-overlapping offsets within that single region.
 */
void nn_configure_registers(uint32_t instr_base,
                            uint32_t weight_base,
                            uint32_t workspace_base);

/**
 * Kick off PL inference by writing START=1 to the CTRL register.
 * The PL begins fetching instructions on the next clock cycle.
 * A DMB barrier guarantees the write is visible before returning.
 */
void nn_pl_start(void);

/**
 * Reset the PL engine by pulsing the RESET bit, then clearing it.
 * A small delay allows the reset to propagate through the pipeline.
 */
void nn_pl_reset(void);

/**
 * Busy-poll the STATUS register until it reads STATUS_DONE.
 *
 * @param timeout_ms  Maximum time to wait in milliseconds.
 *                    0 means wait forever.
 *
 * Returns 0 when DONE is seen, -1 on timeout.
 */
int nn_pl_wait_done(uint32_t timeout_ms);

/**
 * Return the current STATUS register value (STATUS_IDLE or STATUS_DONE).
 */
uint32_t nn_pl_get_status(void);

#if NN_USE_INTERRUPTS
/**
 * Block until the PL asserts its interrupt line (F2P IRQ).
 *
 * @param timeout_ms  Maximum time to wait in milliseconds.
 *                    0 means wait forever.
 *
 * Returns 0 when the interrupt fires, -1 on timeout.
 *
 * Requires the GIC and IRQ_F2P to be configured in the BSP.
 * The ISR simply sets a volatile flag that this function waits on.
 */
int nn_pl_wait_interrupt(uint32_t timeout_ms);
#endif /* NN_USE_INTERRUPTS */

#ifdef __cplusplus
}
#endif

#endif /* NN_DRIVER_H */
