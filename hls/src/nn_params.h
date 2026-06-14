#ifndef NN_PARAMS_H
#define NN_PARAMS_H
/**
 * HLS-specific parameters and compile-time constants.
 *
 * These are tuned for ZYNQ 7045 (XC7Z045):
 *   - 900 DSP48E slices
 *   - 19.2 Mb BRAM (545 × 36 Kb blocks)
 *   - Target clock: 150 MHz (6.67 ns period)
 */

#include <stdint.h>

// ── Data types ───────────────────────────────────────────────────────
// FP32 = C float; HLS synthesises float ops via DSP48 + LUT
typedef float  data_t;

// ── AXI interface widths ────────────────────────────────────────────
#define AXI_DATA_WIDTH   64    // 64-bit AXI Master for DDR bandwidth
#define AXI_ADDR_WIDTH   32

// ── BRAM buffer sizes (in elements, not bytes) ──────────────────────
// NOTE: For smaller models, reduce these to save BRAM.
// The Vivado 2018.3 crash was fixed by single-file architecture,
// NOT by reducing these sizes.
#define MAX_INPUT_FEATURES   4096
#define MAX_OUTPUT_FEATURES  4096
#define MAX_CHANNELS          512
#define MAX_POOL_HEIGHT       256
#define MAX_POOL_WIDTH        256

#define INSTR_BUFFER_DEPTH    256   // max instructions to cache in BRAM

// ── GRU scratch sizing ───────────────────────────────────────────────
#define GRU_MAX_HIDDEN        512
#define GRU_SCRATCH_DEPTH     (6 * GRU_MAX_HIDDEN)  // 6×512 = 3072 floats

// ── Compute parallelism ──────────────────────────────────────────────
#define MAC_UNITS      8       // parallel multiply-accumulate lanes
#define VEC_WIDTH      MAC_UNITS

// ── Pipeline depths (for II=1 at target frequency) ──────────────────
#define FLOAT_ADD_LAT    4
#define FLOAT_MUL_LAT    4
#define FLOAT_FMA_LAT    6

// ── Register offsets ─────────────────────────────────────────────────
#define REG_CTRL           0x00
#define REG_STATUS         0x04
#define REG_INSTR_ADDR     0x08
#define REG_WEIGHT_ADDR    0x0C
#define REG_INPUT_ADDR     0x10
#define REG_OUTPUT_ADDR    0x14
#define REG_WORKSPACE_ADDR 0x18

#define STATUS_IDLE    0
#define STATUS_DONE    1

// CTRL register bits
#define CTRL_START      0x01
#define CTRL_RESET      0x02

#endif /* NN_PARAMS_H */
