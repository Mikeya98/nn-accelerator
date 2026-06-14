#ifndef NN_PLATFORM_H
#define NN_PLATFORM_H
/**
 * NN Accelerator — ZYNQ 7045 Platform Configuration
 *
 * SINGLE CONFIGURATION POINT for adapting the firmware to different
 * Vivado Block Designs.  All addresses, sizes, and feature flags are
 * #define'd here so nothing else in the firmware needs to change.
 *
 * The register offsets and bit definitions match nn_accelerator/hls/src/nn_params.h
 * exactly — do not edit them independently.
 */

#include <stdint.h>

/* =========================================================================
 * 1.  AXI Base Address  (set in Vivado Address Editor → s_axi_control)
 * =========================================================================
 *
 * This is where the PS sees the nn_engine IP's AXI4-Lite slave in its
 * memory map.  Typical GP0 range: 0x4000_0000 – 0x7FFF_FFFF.
 * Default 0x43C00000 is a common starting assignment; update to match
 * the value shown in Address Editor → Offset Address.
 */
#ifndef NN_AXI_BASE_ADDR
#define NN_AXI_BASE_ADDR       0x43C00000UL
#endif

/* =========================================================================
 * 2.  DDR Memory Layout  (1 GB DDR for ZYNQ 7045)
 * ========================================================================= */

/* Physical base of DDR as seen by the ARM cores.  On ZYNQ this is
 * typically 0x00100000 (the first 1 MB is reserved for OCM / boot ROM). */
#ifndef NN_DDR_BASE
#define NN_DDR_BASE            0x00100000UL
#endif

/* Where the BIN file is loaded in DDR.  The PL reads instructions and
 * weights directly from this region (no memcpy to separate buffers).
 *
 *   NN_BIN_BASE +   0  : Header         (256 B)
 *   NN_BIN_BASE + 256  : Instructions   (N × 64 B)
 *   NN_BIN_BASE + 256 + N*64 : Weights  (header.weight_size B)
 */
#ifndef NN_BIN_BASE
#define NN_BIN_BASE            0x08000000UL
#endif
#define NN_BIN_MAX_SIZE        (16UL * 1024 * 1024)   /* 16 MiB */

/* Workspace buffer: input, output, and scratch all live here at
 * compiler-assigned byte offsets.  REG_INPUT_ADDR, REG_OUTPUT_ADDR,
 * and REG_WORKSPACE_ADDR all point to this base. */
#ifndef NN_WORKSPACE_BASE
#define NN_WORKSPACE_BASE      0x03000000UL
#endif
#define NN_WORKSPACE_MAX_SIZE  (16UL * 1024 * 1024)   /* 16 MiB */

/*
 * IMPORTANT — HP0 vs ARM address space:
 *
 * The ARM cores see DDR starting at 0x00100000, but the PL's HP0 port
 * may see DDR starting at 0x00000000 (depending on the AXI interconnect
 * configuration in the Vivado Block Design).
 *
 * The firmware writes register values that the PL uses as AXI addresses
 * on its m_axi_ddr bus.  If the PL's DDR view differs from ARM's, the
 * ARM-view addresses must be translated before writing to registers.
 *
 * Two common configurations:
 *
 *   (A) HP0 DDR offset = 0x00000000  (PL sees DDR from 0x00000000)
 *       → ARM address 0x00100000  =  PL AXI address 0x00000000
 *       → Subtract 0x00100000 from ARM addresses when writing PL regs.
 *       Set NN_HP0_ADDR_OFFSET = -0x00100000UL (as a signed adjustment).
 *
 *   (B) HP0 DDR offset = 0x00100000  (PL sees DDR same as ARM)
 *       → ARM address = PL AXI address = 0x00100000
 *       → No translation needed.
 *       Set NN_HP0_ADDR_OFFSET = 0 (default).
 *
 * The adjustment is applied automatically in nn_reg_write() when
 * writing address registers (INSTR_ADDR, WEIGHT_ADDR, INPUT_ADDR,
 * OUTPUT_ADDR, WORKSPACE_ADDR).  Set this to 0 if your Block Design
 * uses configuration (B), or to -0x00100000 for configuration (A).
 *
 * To determine your configuration: check the Vivado Address Editor
 * for the nn_engine m_axi_ddr interface's base address.
 */
#ifndef NN_HP0_ADDR_OFFSET
#define NN_HP0_ADDR_OFFSET     0   /* change to -0x00100000UL for HP0 @ 0 */
#endif

/* Macro to convert an ARM-view DDR address to a PL-view AXI address. */
#define NN_ARM_TO_PL_ADDR(arm_addr)  ((uint32_t)((arm_addr) + NN_HP0_ADDR_OFFSET))

/* Firmware code / data / stack live above this address (see lscript.ld). */
#ifndef NN_FW_BASE
#define NN_FW_BASE             0x3F000000UL
#endif

/* =========================================================================
 * 3.  Register offsets  (byte offsets from NN_AXI_BASE_ADDR)
 * =========================================================================
 *
 * Must match nn_params.h exactly:
 *   #define REG_CTRL           0x00
 *   #define REG_STATUS         0x04
 *   #define REG_INSTR_ADDR     0x08
 *   #define REG_WEIGHT_ADDR    0x0C
 *   #define REG_INPUT_ADDR     0x10
 *   #define REG_OUTPUT_ADDR    0x14
 *   #define REG_WORKSPACE_ADDR 0x18
 */

#define NN_REG_CTRL            0x00U
#define NN_REG_STATUS          0x04U
#define NN_REG_INSTR_ADDR      0x08U
#define NN_REG_WEIGHT_ADDR     0x0CU
#define NN_REG_INPUT_ADDR      0x10U
#define NN_REG_OUTPUT_ADDR     0x14U
#define NN_REG_WORKSPACE_ADDR  0x18U

/* End of register file (32 × 32-bit words = 128 bytes = 0x80). */
#define NN_REG_FILE_SIZE       0x80U

/* =========================================================================
 * 4.  Register bit definitions
 * ========================================================================= */

/* CTRL register (NN_REG_CTRL) */
#define NN_CTRL_START          0x01U
#define NN_CTRL_RESET          0x02U

/* STATUS register (NN_REG_STATUS) */
#define NN_STATUS_IDLE         0U
#define NN_STATUS_DONE         1U

/* =========================================================================
 * 5.  Timing
 * ========================================================================= */

/* PL timeout in milliseconds.  The Cortex-A9 at 666 MHz with a ~1 ms
 * coarse delay loop gives roughly 10,000 iterations before timeout. */
#ifndef NN_PL_TIMEOUT_MS
#define NN_PL_TIMEOUT_MS       10000U
#endif

/* Coarse busy-wait loop count for ~1 ms on Cortex-A9 @ 666 MHz.
 * Tune this for your actual FCLK frequency if accurate timing matters. */
#define NN_DELAY_1MS_LOOPS     100000U

/* =========================================================================
 * 6.  Feature flags  (compile-time selection)
 * ========================================================================= */

/* Set to 1 to use GIC interrupt (F2P IRQ) instead of polling STATUS. */
#ifndef NN_USE_INTERRUPTS
#define NN_USE_INTERRUPTS      0
#endif

/* Set to 1 to enable SD card loading (requires xilffs / FAT library). */
#ifndef NN_USE_SD_CARD
#define NN_USE_SD_CARD         0
#endif

/* =========================================================================
 * 7.  UART
 * ========================================================================= */

/* UART baud rate for debug output.  ZYNQ PS UART1 is typically used
 * in Xilinx BSPs. */
#define NN_UART_BAUDRATE       115200U

/* Helper macro — use xil_printf when available, fall back to nothing. */
#ifdef __XILINX_BSP__
#include "xil_printf.h"
#define nn_printf   xil_printf
#else
/* Stub for host-side / QEMU compilation — define your own print if needed. */
#define nn_printf(...)   ((void)0)
#endif

/* =========================================================================
 * 8.  Compile-time assertions
 * ========================================================================= */

/* Verify the platform header is included by a C99-compatible compiler. */
#if __STDC_VERSION__ < 199901L
#error "C99 or later required"
#endif

#endif /* NN_PLATFORM_H */
