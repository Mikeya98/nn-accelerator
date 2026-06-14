/**
 * NN Accelerator — AXI4-Lite Register Driver Implementation
 *
 * Bare-metal MMIO for the PL nn_engine IP.  On bare-metal ZYNQ the ARM
 * physical address space == virtual address space (no MMU), so a direct
 * pointer cast from the AXI base address is correct.
 *
 * Every register write is followed by a DMB (Data Memory Barrier) to
 * ensure the store reaches the PL before subsequent ARM instructions
 * execute.  This is required because the AXI GP0 port may buffer writes
 * and the Cortex-A9's store buffer can reorder them.
 */

#include "nn_driver.h"
#include <stddef.h>   /* NULL */

/* ── Module state ──────────────────────────────────────────────────────── */

/* Pointer to the nn_engine register file in the PS AXI address space.
 * Set once by nn_driver_init(). */
static volatile uint32_t *g_reg_file = NULL;

/* Flag set by the interrupt service routine (if NN_USE_INTERRUPTS). */
#if NN_USE_INTERRUPTS
volatile int g_pl_done_flag = 0;
#endif

/* ── Helper: coarse delay loop (~1 ms on Cortex-A9 @ 666 MHz) ─────────── */

static void delay_approx_ms(uint32_t ms)
{
    uint32_t i, j;
    for (i = 0; i < ms; i++) {
        /* Tuned for ~666 MHz; one iteration ~15 cycles → ~1 ms per 44000 iters.
         * We over-approximate with NN_DELAY_1MS_LOOPS (100,000) for margin. */
        for (j = 0; j < NN_DELAY_1MS_LOOPS; j++) {
            __asm__ volatile ("" ::: "memory");  /* prevent loop elimination */
        }
    }
}

/* ── Memory barrier ────────────────────────────────────────────────────── */

/**
 * Data Memory Barrier — ensures all previous explicit memory accesses
 * have completed before any subsequent ones.  Essential for PL register
 * writes: without it the ARM store buffer may delay the write long after
 * the CPU continues executing.
 */
static inline void dmb(void)
{
    __asm__ volatile ("dmb" ::: "memory");
}

/* ── Public API ────────────────────────────────────────────────────────── */

int nn_driver_init(void)
{
    g_reg_file = (volatile uint32_t *)NN_AXI_BASE_ADDR;

    /* Sanity check: read the STATUS register.  If the PL bitstream is
     * loaded and the AXI GP0 interconnect is properly configured this
     * returns 0 (IDLE) or 1 (DONE) — either is fine.  If the address
     * space is unmapped or the interconnect is misconfigured a data
     * abort will fire, which the data-abort handler catches.
     *
     * On a host-side / QEMU build where the register file is stubbed,
     * this read simply returns whatever the stub provides. */
    volatile uint32_t status = g_reg_file[NN_REG_STATUS / 4];
    (void)status;

    return 0;
}

void nn_reg_write(uint32_t byte_offset, uint32_t value)
{
    g_reg_file[byte_offset / 4] = value;
    dmb();
}

uint32_t nn_reg_read(uint32_t byte_offset)
{
    return g_reg_file[byte_offset / 4];
}

/* ── Convenience functions ─────────────────────────────────────────────── */

void nn_configure_registers(uint32_t instr_base,
                            uint32_t weight_base,
                            uint32_t workspace_base)
{
    /* Translate ARM-view DDR addresses to PL AXI addresses.
     * See nn_platform.h § "HP0 vs ARM address space" for details. */
    uint32_t pl_workspace = NN_ARM_TO_PL_ADDR(workspace_base);
    uint32_t pl_instr     = NN_ARM_TO_PL_ADDR(instr_base);
    uint32_t pl_weight    = NN_ARM_TO_PL_ADDR(weight_base);

    /* Order does not matter (the PL only acts on CTRL.START), but we
     * follow the simulator convention: workspace first, then data
     * buffers.  No dmb() needed between writes — nn_reg_write() does it. */
    nn_reg_write(NN_REG_WORKSPACE_ADDR, pl_workspace);
    nn_reg_write(NN_REG_INPUT_ADDR,     pl_workspace);
    nn_reg_write(NN_REG_OUTPUT_ADDR,    pl_workspace);
    nn_reg_write(NN_REG_INSTR_ADDR,     pl_instr);
    nn_reg_write(NN_REG_WEIGHT_ADDR,    pl_weight);
}

void nn_pl_start(void)
{
    /* Write START=1.  The PL sees this on the next clock cycle and
     * begins fetching instructions.  nn_reg_write() already does DMB. */
    nn_reg_write(NN_REG_CTRL, NN_CTRL_START);
}

void nn_pl_reset(void)
{
    /* Assert RESET, wait for it to propagate, then de-assert.
     * The PL CTRL_RESET clears internal state (pipeline, FSM, counters)
     * but does NOT clear the address registers — those are preserved. */
    nn_reg_write(NN_REG_CTRL, NN_CTRL_RESET);
    delay_approx_ms(1);
    nn_reg_write(NN_REG_CTRL, 0);
    delay_approx_ms(1);
}

int nn_pl_wait_done(uint32_t timeout_ms)
{
    uint32_t elapsed = 0;
    const uint32_t poll_interval_ms = 1;

    while (nn_reg_read(NN_REG_STATUS) != NN_STATUS_DONE) {
        if (timeout_ms > 0 && elapsed >= timeout_ms) {
            return -1;   /* timeout */
        }
        delay_approx_ms(poll_interval_ms);
        elapsed += poll_interval_ms;
    }
    return 0;
}

uint32_t nn_pl_get_status(void)
{
    return nn_reg_read(NN_REG_STATUS);
}

#if NN_USE_INTERRUPTS
int nn_pl_wait_interrupt(uint32_t timeout_ms)
{
    uint32_t elapsed = 0;
    const uint32_t poll_interval_ms = 1;

    g_pl_done_flag = 0;

    while (!g_pl_done_flag) {
        if (timeout_ms > 0 && elapsed >= timeout_ms) {
            return -1;   /* timeout */
        }
        delay_approx_ms(poll_interval_ms);
        elapsed += poll_interval_ms;
    }
    return 0;
}
#endif /* NN_USE_INTERRUPTS */
