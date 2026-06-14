#ifndef NN_LOADER_H
#define NN_LOADER_H
/**
 * NN Accelerator — BIN File Loader
 *
 * Parses compiler-generated BIN files, validates them (magic, version,
 * CRC32), and provides access to instructions and weights.
 *
 * Two loading paths are supported:
 *   1. nn_load_from_memory()  — BIN already in DDR (JTAG / SDK pre-load).
 *   2. nn_load_from_sd()      — BIN read from an SD card (requires xilffs).
 *
 * The BIN layout is defined in nn_accelerator/common/nn_bin.h:
 *
 *   offset 0:       Header        (256 B)
 *   offset 256:     Instructions  (N × 64 B)
 *   offset 256+N*64: Weights      (header.weight_size B)
 */

#include <stdint.h>

/* Pull ISA and BIN type definitions from the single source of truth.
 * Include path must be set to: -I../../common */
#include "nn_isa.h"
#include "nn_bin.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ────────────────────────────────────────────────────── */
typedef struct nn_bin_handle nn_bin_handle_t;

/* ── Loading ──────────────────────────────────────────────────────────── */

/**
 * Parse a BIN file already loaded at `addr` in DDR.
 *
 * This is the primary loading path for development: use Xilinx SDK's
 * "DDR memory write" or JTAG to load the BIN, then call this function
 * with the load address.
 *
 * Returns NULL on parse error (bad magic, invalid size, etc.).
 * The handle must be freed with nn_free_bin().
 */
nn_bin_handle_t *nn_load_from_memory(uint32_t addr);

/**
 * Load a BIN file from an SD card via the xilffs FAT library.
 *
 * Requires NN_USE_SD_CARD = 1 and nn_sd_init() to have been called.
 * The BIN is read into NN_BIN_BASE, then loaded as if by
 * nn_load_from_memory(NN_BIN_BASE).
 *
 * Returns NULL on error.
 */
nn_bin_handle_t *nn_load_from_sd(const char *path);

/**
 * Free a BIN handle.  Does NOT free the underlying DDR buffers
 * (instructions and weights remain available for the PL to use).
 */
void nn_free_bin(nn_bin_handle_t *handle);

/* ── Validation ──────────────────────────────────────────────────────── */

/**
 * Validate a loaded BIN:
 *   - Magic number check
 *   - Version compatibility (major == 1)
 *   - CRC32 of instructions + weights
 *   - Workspace size sanity check
 *
 * Returns 0 on success, -1 on any failure.
 * Prints a diagnostic to UART on each failure.
 */
int nn_validate(const nn_bin_handle_t *handle);

/* ── Accessors ───────────────────────────────────────────────────────── */

const nn_bin_header_t *nn_get_header(const nn_bin_handle_t *handle);
uint32_t              nn_get_num_instructions(const nn_bin_handle_t *handle);
const nn_instruction_t *nn_get_instructions(const nn_bin_handle_t *handle);
const uint8_t         *nn_get_weight_blob(const nn_bin_handle_t *handle);
uint32_t              nn_get_instr_base(const nn_bin_handle_t *handle);
uint32_t              nn_get_weight_base(const nn_bin_handle_t *handle);
uint32_t              nn_get_total_size(const nn_bin_handle_t *handle);

/* ── Utilities ───────────────────────────────────────────────────────── */

/**
 * Find the output byte offset for reading inference results.
 *
 * Walks instructions in reverse (from last to first), skipping NOP
 * (0x00) and END (0xFF), and returns the output_addr of the first
 * compute instruction found.
 *
 * This offset is relative to REG_WORKSPACE_ADDR.
 */
uint32_t nn_find_output_offset(const nn_bin_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif /* NN_LOADER_H */
