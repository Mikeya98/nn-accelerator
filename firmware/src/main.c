/**
 * NN Accelerator — PS Firmware Entry Point
 *
 * Bare-metal C application for ZYNQ 7045 (ARM Cortex-A9).
 *
 * Orchestrates the inference flow:
 *   1. Platform init  (UART, caches, timer)
 *   2. Driver init    (AXI-Lite MMIO mapping)
 *   3. BIN loading    (from DDR pre-load or SD card)
 *   4. Validation     (magic, version, CRC32)
 *   5. PL reset       (clear stale pipeline state)
 *   6. Input data     (write to workspace)
 *   7. Register config (5 address registers)
 *   8. PL start       (CTRL = START)
 *   9. Wait DONE      (poll STATUS or interrupt)
 *  10. Output data    (read from workspace + output_offset)
 *  11. Print results  (summary over UART)
 *
 * The firmware is intentionally minimal — no model parsing, no ONNX
 * awareness.  All complexity lives in the PC-side compiler.
 */

#include "nn_platform.h"
#include "nn_driver.h"
#include "nn_loader.h"
#include "nn_isa.h"       /* for NN_OP_NOP, NN_OP_END */
#include "nn_bin.h"       /* for NN_BIN_MAGIC */

#include <string.h>       /* memcpy, memset */
#include <stdlib.h>       /* malloc */

/* ── Constants ────────────────────────────────────────────────────────── */

/* Maximum number of output floats to print in the summary. */
#define MAX_PRINT_OUTPUTS   8

/* ── Forward declarations ─────────────────────────────────────────────── */

static int  platform_init(void);
static void data_abort_handler(void) __attribute__((noreturn));
static void print_banner(void);
static void print_instruction_summary(const nn_bin_handle_t *bin);
static void print_output_sample(const float *data, uint32_t count,
                                uint32_t max_print);
static void halt(void);

/* ── Global state ─────────────────────────────────────────────────────── */

/* Track whether PL was started (for cleanup on error). */
static volatile int g_pl_started = 0;

/* ═══════════════════════════════════════════════════════════════════════
 *  Entry point
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    nn_bin_handle_t *bin = NULL;
    int ret = 0;

    /* ── 1. Platform init ──────────────────────────────────── */
    if (platform_init() != 0) {
        nn_printf("FATAL: Platform init failed\r\n");
        halt();
    }
    print_banner();

    /* ── 2. Driver init ────────────────────────────────────── */
    nn_printf("[1/8] Initialising PL driver ...\r\n");
    if (nn_driver_init() != 0) {
        nn_printf("FATAL: PL driver init failed — check AXI_BASE_ADDR\r\n");
        halt();
    }
    nn_printf("      AXI base: 0x%08lX  STATUS=%lu\r\n",
              (unsigned long)NN_AXI_BASE_ADDR,
              (unsigned long)nn_pl_get_status());

    /* ── 3. Reset PL to known state ────────────────────────── */
    nn_printf("[2/8] Resetting PL engine ...\r\n");
    nn_pl_reset();

    /* ── 4. Load BIN ───────────────────────────────────────── */
    nn_printf("[3/8] Loading BIN ...\r\n");

#if NN_USE_SD_CARD
    /* SD-card path: read model.bin from the FAT filesystem. */
    extern int nn_sd_init(void);  /* from nn_sd.c */
    if (nn_sd_init() == 0) {
        bin = nn_load_from_sd("model.bin");
    } else {
        nn_printf("      SD card not available; trying pre-loaded BIN at 0x%08lX\r\n",
                  (unsigned long)NN_BIN_BASE);
        bin = nn_load_from_memory(NN_BIN_BASE);
    }
#else
    /* Pre-loaded path: BIN already at NN_BIN_BASE (via JTAG / SDK). */
    bin = nn_load_from_memory(NN_BIN_BASE);
#endif

    if (!bin) {
        nn_printf("FATAL: Failed to load BIN\r\n");
        halt();
    }

    /* ── 5. Validate BIN ───────────────────────────────────── */
    nn_printf("[4/8] Validating BIN ...\r\n");
    if (nn_validate(bin) != 0) {
        nn_printf("FATAL: BIN validation failed\r\n");
        nn_free_bin(bin);
        halt();
    }

    /* ── 6. Instruction summary ────────────────────────────── */
    print_instruction_summary(bin);

    /* ── 7. Prepare input data ─────────────────────────────── */
    nn_printf("[5/8] Writing input data ...\r\n");
    {
        const nn_bin_header_t *hdr = nn_get_header(bin);
        uint32_t input_size = hdr->input_size;

        if (input_size > 0) {
            /* The input region starts at workspace offset 0.  For a
             * bare-metal test, we either:
             *   a) Use pre-loaded data at WORKSPACE_BASE (set by SDK/JTAG).
             *   b) Zero-fill it (useful for testing without real data).
             *   c) Load from SD card (input.bin).
             *
             * Here we zero-fill by default as a safe fallback.  The user
             * can pre-load real data via Xilinx SDK's "DDR memory write"
             * before launching the firmware. */
            float *input_ptr = (float *)NN_WORKSPACE_BASE;
#if NN_USE_SD_CARD
            /* Attempt to load input.bin from SD card.  If it fails,
             * fall back to zero-fill. */
            extern int nn_sd_read_file(const char *, uint8_t *, uint32_t);
            int sz = nn_sd_read_file("input.bin", (uint8_t *)input_ptr, input_size);
            if (sz == (int)input_size) {
                nn_printf("      Loaded %d B input data from SD\r\n", sz);
            } else {
                nn_printf("      input.bin not found (%d), zero-filling %lu B\r\n",
                          sz, (unsigned long)input_size);
                memset(input_ptr, 0, input_size);
            }
#else
            /* On bare-metal without SD card, check if data is already
             * present (non-zero first byte implies pre-loaded data). */
            if (((uint8_t *)input_ptr)[0] == 0
                && ((uint8_t *)input_ptr)[1] == 0
                && ((uint8_t *)input_ptr)[2] == 0
                && ((uint8_t *)input_ptr)[3] == 0) {
                /* All-zero → probably no data loaded.  Zero-fill
                 * explicitly for deterministic behavior. */
                memset(input_ptr, 0, input_size);
                nn_printf("      Input zero-filled (%lu B) — "
                          "pre-load real data for inference\r\n",
                          (unsigned long)input_size);
            } else {
                nn_printf("      Using pre-loaded input data (%lu B)\r\n",
                          (unsigned long)input_size);
            }
#endif
        } else {
            nn_printf("      Model has no inputs\r\n");
        }
    }

    /* ── 8. Configure registers and start PL ────────────────── */
    nn_printf("[6/8] Configuring PL registers ...\r\n");
    {
        uint32_t instr_base    = nn_get_instr_base(bin);
        uint32_t weight_base   = nn_get_weight_base(bin);

        nn_configure_registers(instr_base, weight_base,
                               NN_WORKSPACE_BASE);

        nn_printf("      INSTR_ADDR    = 0x%08lX\r\n", (unsigned long)instr_base);
        nn_printf("      WEIGHT_ADDR   = 0x%08lX\r\n", (unsigned long)weight_base);
        nn_printf("      WORKSPACE_ADDR= 0x%08lX\r\n", (unsigned long)NN_WORKSPACE_BASE);
    }

    nn_printf("[7/8] Starting PL inference ...\r\n");
    nn_pl_start();
    g_pl_started = 1;

    /* ── 9. Wait for completion ─────────────────────────────── */
    nn_printf("      Waiting for PL DONE (timeout=%lu ms) ...\r\n",
              (unsigned long)NN_PL_TIMEOUT_MS);

#if NN_USE_INTERRUPTS
    ret = nn_pl_wait_interrupt(NN_PL_TIMEOUT_MS);
#else
    ret = nn_pl_wait_done(NN_PL_TIMEOUT_MS);
#endif

    if (ret != 0) {
        nn_printf("ERROR: PL timeout after %lu ms\r\n",
                  (unsigned long)NN_PL_TIMEOUT_MS);
        nn_printf("       STATUS = %lu (expected %lu)\r\n",
                  (unsigned long)nn_pl_get_status(),
                  (unsigned long)NN_STATUS_DONE);
        /* Attempt PL reset to recover. */
        nn_pl_reset();
        nn_free_bin(bin);
        halt();
    }
    nn_printf("      PL DONE received.\r\n");

    /* ── 10. Read output ────────────────────────────────────── */
    nn_printf("[8/8] Reading output ...\r\n");
    {
        const nn_bin_header_t *hdr = nn_get_header(bin);
        uint32_t output_size   = hdr->output_size;
        uint32_t output_offset = nn_find_output_offset(bin);
        uint32_t output_count  = output_size / sizeof(float);

        if (output_count > 0) {
            /* Allocate a buffer for the output.  For large outputs,
             * read directly from DDR without copying. */
            float *out_buf = (float *)malloc(output_size);
            if (out_buf) {
                const float *src = (const float *)(NN_WORKSPACE_BASE + output_offset);
                memcpy(out_buf, src, output_size);
                nn_printf("      Output: %lu floats (%lu B) from "
                          "workspace+0x%04lX\r\n",
                          (unsigned long)output_count,
                          (unsigned long)output_size,
                          (unsigned long)output_offset);
                print_output_sample(out_buf, output_count, MAX_PRINT_OUTPUTS);
                free(out_buf);
            } else {
                /* Can't allocate — print directly from DDR. */
                nn_printf("      Output: %lu floats from workspace+0x%04lX\r\n",
                          (unsigned long)output_count,
                          (unsigned long)output_offset);
                print_output_sample(
                    (const float *)(NN_WORKSPACE_BASE + output_offset),
                    output_count, MAX_PRINT_OUTPUTS);
            }
        } else {
            nn_printf("      Model has no outputs\r\n");
        }
    }

    /* ── 11. Done ───────────────────────────────────────────── */
    nn_printf("\r\n==== Inference complete ====\r\n");
    nn_free_bin(bin);

    /* Idle loop — in a real system you might trigger the next inference
     * or enter a low-power state. */
    while (1) {
        __asm__ volatile ("wfi");  /* Wait For Interrupt — low power */
    }

    return 0;  /* unreachable */
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Platform initialisation
 * ═══════════════════════════════════════════════════════════════════════ */

static int platform_init(void)
{
    /* ── Enable L1 caches ────────────────────────────────────── */
    /* On Cortex-A9: SCTLR.I (bit 12) = instruction cache,
     *              SCTLR.C (bit  2) = data cache.
     * These are typically already enabled by the FSBL; we just verify. */
    __asm__ volatile (
        "mrc p15, 0, r0, c1, c0, 0  \n"   /* read SCTLR               */
        "orr r0, r0, #0x1000        \n"   /* set I-cache enable (I)   */
        "orr r0, r0, #0x4           \n"   /* set D-cache enable (C)   */
        "mcr p15, 0, r0, c1, c0, 0  \n"   /* write SCTLR              */
        ::: "r0", "memory"
    );

    /* ── Data abort handler ──────────────────────────────────── */
    /* Install a minimal handler that prints fault info and halts.
     * The vector table at 0x00000000 (or VBAR) must already be set
     * up by the FSBL — this replaces the data abort entry at offset
     * 0x10 in the vector table.
     *
     * On bare-metal ZYNQ the vector table is at 0x00000000 (low
     * vectors) unless the SCTLR.V bit is set (high vectors at
     * 0xFFFF0000).  The FSBL typically configures low vectors.
     *
     * We write the handler address directly into the vector table.
     * This is safe because the MMU is disabled and the vector table
     * region is writable. */
    {
        uint32_t *vector_table = (uint32_t *)0x00000000UL;
        /* Data abort is exception vector #4 (offset 0x10, word index 4).
         * We store a branch instruction: LDR PC, [PC, #-0x0]
         * Actually simpler: store the absolute handler address in a jump
         * table, or use an LDR PC,[PC,#imm] pattern.
         *
         * For bare-metal simplicity, just use a direct branch:
         *   B <handler>  =  0xEA000000 | ((offset/4 - 2) & 0x00FFFFFF)
         * This requires the handler to be within ±32 MiB of the vector
         * table — always true on ZYNQ.
         */
        uint32_t handler_addr = (uint32_t)&data_abort_handler;
        uint32_t vector_addr  = (uint32_t)&vector_table[4]; /* offset 0x10 */
        int32_t  offset       = (int32_t)(handler_addr - vector_addr - 8);
        uint32_t branch_instr = 0xEA000000U | ((uint32_t)(offset >> 2) & 0x00FFFFFFU);
        vector_table[4] = branch_instr;

        /* Also install for prefetch abort (offset 0x0C, word index 3)
         * reusing the same handler. */
        vector_addr  = (uint32_t)&vector_table[3]; /* offset 0x0C */
        offset       = (int32_t)(handler_addr - vector_addr - 8);
        branch_instr = 0xEA000000U | ((uint32_t)(offset >> 2) & 0x00FFFFFFU);
        vector_table[3] = branch_instr;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Data abort handler
 * ═══════════════════════════════════════════════════════════════════════ */

static void data_abort_handler(void)
{
    uint32_t dfsr, dfar;

    /* Read CP15 fault registers. */
    __asm__ volatile (
        "mrc p15, 0, %0, c5, c0, 0  \n"   /* DFSR  — Data Fault Status Register  */
        "mrc p15, 0, %1, c6, c0, 0  \n"   /* DFAR  — Data Fault Address Register  */
        : "=r"(dfsr), "=r"(dfar)
    );

    nn_printf("\r\n"
              "╔══════════════════════════════════╗\r\n"
              "║       DATA ABORT — FAULT          ║\r\n"
              "╠══════════════════════════════════╣\r\n"
              "║ DFAR (fault addr): 0x%08lX      ║\r\n"
              "║ DFSR (status)    : 0x%08lX      ║\r\n"
              "╚══════════════════════════════════╝\r\n",
              (unsigned long)dfar,
              (unsigned long)dfsr);

    /* Common failure diagnosis. */
    if ((dfsr & 0x40F) == 0x005) {
        nn_printf("  → Translation fault (page), stage 1.\r\n");
    } else if ((dfsr & 0x40F) == 0x007) {
        nn_printf("  → Translation fault (section), stage 1.\r\n");
    } else if ((dfsr & 0x40F) == 0x009) {
        nn_printf("  → Domain fault, stage 1.\r\n");
    } else if ((dfsr & 0x40F) == 0x00D) {
        nn_printf("  → Permission fault, stage 1.\r\n");
    } else if (dfsr & (1U << 12)) {
        nn_printf("  → External abort (AXI DECERR/SLVERR).\r\n");
        nn_printf("     Check AXI interconnect and PL IP state.\r\n");
        if (g_pl_started) {
            nn_printf("     PL was running — reset may be needed.\r\n");
        }
    }

    if (dfar >= NN_AXI_BASE_ADDR && dfar < NN_AXI_BASE_ADDR + NN_REG_FILE_SIZE) {
        nn_printf("  → Fault in PL register space!\r\n");
        nn_printf("     Check: is nn_engine IP loaded and connected to GP0?\r\n");
    }

    halt();
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static void print_banner(void)
{
    nn_printf("\r\n"
              "╔══════════════════════════════════════════╗\r\n"
              "║      NN Accelerator — PS Firmware        ║\r\n"
              "║      ZYNQ 7045  /  Bare-metal C          ║\r\n"
              "║      Version 1.0                         ║\r\n"
              "╚══════════════════════════════════════════╝\r\n"
              "\r\n");
}

static void print_instruction_summary(const nn_bin_handle_t *bin)
{
    uint32_t n = nn_get_num_instructions(bin);
    if (n == 0) return;

    const nn_instruction_t *instrs = nn_get_instructions(bin);

    /* Count by opcode */
    uint32_t count_fc = 0, count_pool = 0, count_relu = 0;
    uint32_t count_sigmoid = 0, count_tanh = 0;
    uint32_t count_gru = 0, count_emul = 0, count_eadd = 0;
    uint32_t count_nop = 0, count_end = 0, count_other = 0;

    uint32_t i;
    nn_printf("      Instructions (%lu):\r\n", (unsigned long)n);
    for (i = 0; i < n && i < 16; i++) {
        const nn_instruction_t *p = &instrs[i];
        const char *opname;
        switch (p->opcode) {
        case NN_OP_FC:         opname = "FC";         break;
        case NN_OP_POOL:       opname = "POOL";       break;
        case NN_OP_RELU:       opname = "ReLU";       break;
        case NN_OP_SIGMOID:    opname = "Sigmoid";    break;
        case NN_OP_TANH:       opname = "Tanh";       break;
        case NN_OP_GRU:        opname = "GRU";        break;
        case NN_OP_ELEM_MUL:   opname = "ElemMul";    break;
        case NN_OP_ELEM_ADD:   opname = "ElemAdd";    break;
        case NN_OP_NOP:        opname = "NOP";        break;
        case NN_OP_END:        opname = "END";        break;
        default:               opname = "???";        break;
        }
        nn_printf("        [%2lu] %-8s  out=0x%04lX\r\n",
                  (unsigned long)i, opname,
                  (unsigned long)p->output_addr);
    }
    if (n > 16) {
        nn_printf("        ...  (%lu total, see BIN header)\r\n", (unsigned long)n);
    }

    /* Full count */
    for (i = 0; i < n; i++) {
        switch (instrs[i].opcode) {
        case NN_OP_FC:        count_fc++;       break;
        case NN_OP_POOL:      count_pool++;      break;
        case NN_OP_RELU:      count_relu++;      break;
        case NN_OP_SIGMOID:   count_sigmoid++;   break;
        case NN_OP_TANH:      count_tanh++;      break;
        case NN_OP_GRU:       count_gru++;       break;
        case NN_OP_ELEM_MUL:  count_emul++;      break;
        case NN_OP_ELEM_ADD:  count_eadd++;      break;
        case NN_OP_NOP:       count_nop++;       break;
        case NN_OP_END:       count_end++;       break;
        default:              count_other++;      break;
        }
    }
    nn_printf("      Breakdown:");
    if (count_fc)      nn_printf(" FC=%lu",       (unsigned long)count_fc);
    if (count_gru)     nn_printf(" GRU=%lu",      (unsigned long)count_gru);
    if (count_pool)    nn_printf(" POOL=%lu",     (unsigned long)count_pool);
    if (count_relu)    nn_printf(" ReLU=%lu",     (unsigned long)count_relu);
    if (count_sigmoid) nn_printf(" Sig=%lu",      (unsigned long)count_sigmoid);
    if (count_tanh)    nn_printf(" Tanh=%lu",     (unsigned long)count_tanh);
    if (count_emul)    nn_printf(" EMul=%lu",     (unsigned long)count_emul);
    if (count_eadd)    nn_printf(" EAdd=%lu",     (unsigned long)count_eadd);
    if (count_nop)     nn_printf(" NOP=%lu",      (unsigned long)count_nop);
    if (count_other)   nn_printf(" ???=%lu",      (unsigned long)count_other);
    nn_printf(" END=%lu\r\n", (unsigned long)count_end);
}

static void print_output_sample(const float *data, uint32_t count,
                                uint32_t max_print)
{
    uint32_t i;
    uint32_t print_n = (count < max_print) ? count : max_print;

    nn_printf("      [");
    for (i = 0; i < print_n; i++) {
        nn_printf("%+.6f", (double)data[i]);
        if (i + 1 < print_n) nn_printf(", ");
    }
    if (count > max_print) {
        nn_printf(", ... (+%lu more)", (unsigned long)(count - max_print));
    }
    nn_printf("]\r\n");
}

static void halt(void)
{
    nn_printf("\r\n==== HALTED ====\r\n");
    /* Disable interrupts and spin forever. */
    __asm__ volatile ("cpsid if");
    while (1) {
        __asm__ volatile ("wfi");
    }
}
