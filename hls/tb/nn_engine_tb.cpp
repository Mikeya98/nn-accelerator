/**
 * HLS C Simulation Testbench for nn_engine IP.
 *
 * Usage in Vivado HLS:
 *   1. Open Vivado HLS 2018.3
 *   2. Create new project → Add source files from hls/src/
 *   3. Add testbench file hls/tb/nn_engine_tb.cpp
 *   4. Run C Simulation
 *
 * The testbench:
 *   1. Allocates a simulated DDR buffer
 *   2. Loads a BIN file (compiled by the PC compiler)
 *   3. Writes register configuration
 *   4. Calls nn_engine()
 *   5. Verifies STATUS == DONE
 *
 * For full numerical verification, pair with the Python simulator
 * (output comparison is done at the Python level).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../src/nn_engine.h"
#include "../src/nn_isa.h"
#include "../src/nn_bin.h"
#include "../src/nn_params.h"

// ── Simulated DDR size (16 MiB) ──────────────────────────────────────
#define DDR_SIZE  (16 * 1024 * 1024 / sizeof(data_t))

// ── Test infrastructure ──────────────────────────────────────────────

static data_t    ddr_mem[DDR_SIZE];
static uint32_t  reg_file[32];

/**
 * Load a BIN file into the simulated DDR.
 * Returns 0 on success, -1 on error.
 */
static int load_bin_to_ddr(const char *bin_path) {
    FILE *f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open BIN file '%s'\n", bin_path);
        return -1;
    }

    // Read entire BIN into DDR at offset 0
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize > (long)(DDR_SIZE * sizeof(data_t))) {
        fprintf(stderr, "ERROR: BIN file too large for simulated DDR\n");
        fclose(f);
        return -1;
    }

    uint8_t *ddr_bytes = (uint8_t*)ddr_mem;
    size_t nread = fread(ddr_bytes, 1, fsize, f);
    fclose(f);

    if (nread != (size_t)fsize) {
        fprintf(stderr, "ERROR: Short read on BIN file\n");
        return -1;
    }

    printf("INFO: Loaded BIN '%s' (%ld bytes)\n", bin_path, fsize);
    return 0;
}

// ── Main testbench ───────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    const char *bin_path = "model.bin";
    if (argc > 1) {
        bin_path = argv[1];
    }

    printf("══════════════════════════════════════════════\n");
    printf("  NN Engine HLS C Simulation\n");
    printf("  BIN: %s\n", bin_path);
    printf("══════════════════════════════════════════════\n\n");

    // ── Load BIN ─────────────────────────────────────────────────
    memset(ddr_mem, 0, sizeof(ddr_mem));
    if (load_bin_to_ddr(bin_path) != 0) {
        return 1;
    }

    // ── Parse BIN header to get sizes ────────────────────────────
    const nn_bin_header_t *hdr = (const nn_bin_header_t *)ddr_mem;
    if (hdr->magic != NN_BIN_MAGIC) {
        fprintf(stderr, "ERROR: Bad magic 0x%08X, expected 0x%08X\n",
                hdr->magic, NN_BIN_MAGIC);
        return 1;
    }

    uint32_t num_instructions = hdr->num_instructions;
    uint32_t weight_size      = hdr->weight_size;
    uint32_t workspace_size   = hdr->workspace_size;
    uint32_t input_size       = hdr->input_size;
    uint32_t output_size      = hdr->output_size;

    printf("Model info:\n");
    printf("  Instructions: %u\n", num_instructions);
    printf("  Weight size:  %u B\n", weight_size);
    printf("  Workspace:    %u B\n", workspace_size);
    printf("  Input size:   %u B\n", input_size);
    printf("  Output size:  %u B\n\n", output_size);

    // ── DDR address layout (must match PS driver) ────────────────
    uint32_t instr_base    = 256;  // right after the 256 B header
    uint32_t weight_base   = 256 + num_instructions * 64;
    uint32_t workspace_base = 0x00100000;  // 1 MiB into DDR

    // ── Configure registers ──────────────────────────────────────
    memset(reg_file, 0, sizeof(reg_file));

    reg_file[REG_INSTR_ADDR     / 4] = instr_base;
    reg_file[REG_WEIGHT_ADDR    / 4] = weight_base;
    reg_file[REG_INPUT_ADDR     / 4] = workspace_base;
    reg_file[REG_OUTPUT_ADDR    / 4] = workspace_base;
    reg_file[REG_WORKSPACE_ADDR / 4] = workspace_base;

    printf("Register configuration:\n");
    printf("  INSTR_ADDR     = 0x%08X\n", instr_base);
    printf("  WEIGHT_ADDR    = 0x%08X\n", weight_base);
    printf("  WORKSPACE_ADDR = 0x%08X\n\n", workspace_base);

    // ── Write test input data (random) ────────────────────────────
    // Place input at offset 0 of workspace
    uint32_t input_floats = input_size / sizeof(data_t);
    data_t *ws = ddr_mem + workspace_base / sizeof(data_t);
    printf("Generating %u random input floats ...\n", input_floats);
    for (uint32_t i = 0; i < input_floats; i++) {
        ws[i] = (data_t)((rand() % 2000 - 1000) / 1000.0);
    }

    // ── Kick off PL ──────────────────────────────────────────────
    reg_file[REG_CTRL / 4] = CTRL_START;
    printf("Starting PL engine (CTRL=0x%X) ...\n", reg_file[REG_CTRL / 4]);

    nn_engine(reg_file, ddr_mem);

    // ── Check status ─────────────────────────────────────────────
    uint32_t status = reg_file[REG_STATUS / 4];
    printf("\nPL engine returned. STATUS = 0x%X\n", status);

    if (status == STATUS_DONE) {
        printf("✅ Inference completed successfully.\n");

        // Print first few output values
        uint32_t output_floats = output_size / sizeof(data_t);
        printf("Output (%u floats):\n", output_floats);
        for (uint32_t i = 0; i < output_floats && i < 10; i++) {
            printf("  [%2u] = %f\n", i, (float)ws[i + input_floats]);
        }
        return 0;
    } else {
        printf("❌ Inference FAILED (status=%u)\n", status);
        return 1;
    }
}
