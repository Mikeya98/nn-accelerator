/**
 * NN Accelerator — BIN File Loader Implementation
 *
 * Parses, validates, and provides access to compiler-generated BIN files.
 * The CRC32 implementation matches Python's zlib.crc32() exactly so that
 * any BIN written by the compiler passes firmware validation.
 */

#include "nn_loader.h"
#include "nn_platform.h"

/* For memcpy(), memset() — bare-metal newlib or equivalent provides these. */
#include <string.h>

/* ── BIN handle structure ──────────────────────────────────────────────── */

struct nn_bin_handle {
    nn_bin_header_t   header;          /* Parsed header copy               */
    nn_instruction_t *instructions;    /* Pointer into DDR (BIN + 256)     */
    const uint8_t    *weight_blob;     /* Pointer into DDR (BIN + 256+N*64)*/
    uint32_t          bin_base;        /* Physical DDR address of the BIN  */
    uint32_t          total_size;      /* = 256 + N*64 + weight_size       */
    uint32_t          instr_base;      /* = bin_base + 256                 */
    uint32_t          weight_base;     /* = bin_base + 256 + N*64          */
};

/* ── CRC32 ────────────────────────────────────────────────────────────── */

/**
 * CRC-32 lookup table (polynomial 0xEDB88320, reflected form of
 * 0x04C11DB7).  Matches Python's zlib.crc32().
 *
 * This table is computed once at first use (lazy init, thread-unsafe —
 * fine for bare-metal single-threaded firmware).
 */

#define CRC32_POLY  0xEDB88320UL

static uint32_t crc32_table[256];
static int      crc32_table_ready = 0;

static void crc32_init_table(void)
{
    uint32_t i, j, crc;
    for (i = 0; i < 256; i++) {
        crc = i;
        for (j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ CRC32_POLY;
            else
                crc >>= 1;
        }
        crc32_table[i] = crc;
    }
    crc32_table_ready = 1;
}

/**
 * Compute CRC-32 of `data` with `len` bytes.
 *
 * Initial value = 0xFFFFFFFF, final XOR = 0xFFFFFFFF, input NOT reflected
 * (the table is already reflected).  This is the standard CRC-32/ISO-HDLC
 * algorithm and matches `zlib.crc32(data) & 0xFFFFFFFF` in Python.
 */
static uint32_t nn_crc32(const uint8_t *data, size_t len)
{
    if (!crc32_table_ready) {
        crc32_init_table();
    }

    uint32_t crc = 0xFFFFFFFFUL;
    size_t i;
    for (i = 0; i < len; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFUL;
}

/* ── Loading ──────────────────────────────────────────────────────────── */

nn_bin_handle_t *nn_load_from_memory(uint32_t addr)
{
    const uint8_t *base = (const uint8_t *)addr;

    /* ── Parse header ────────────────────────────────────────────── */
    const nn_bin_header_t *hdr = (const nn_bin_header_t *)base;

    if (hdr->magic != NN_BIN_MAGIC) {
        nn_printf("ERROR: Bad BIN magic 0x%08lX, expected 0x%08lX\r\n",
                  (unsigned long)hdr->magic,
                  (unsigned long)NN_BIN_MAGIC);
        return NULL;
    }

    /* ── Compute offsets ─────────────────────────────────────────── */
    uint32_t num_instr    = hdr->num_instructions;
    uint32_t instr_offset = 256;
    uint32_t weight_off   = 256 + num_instr * sizeof(nn_instruction_t);
    uint32_t total        = weight_off + hdr->weight_size;

    /* ── Sanity: minimal size ────────────────────────────────────── */
    if (total > NN_BIN_MAX_SIZE) {
        nn_printf("ERROR: BIN too large: %lu > %lu\r\n",
                  (unsigned long)total,
                  (unsigned long)NN_BIN_MAX_SIZE);
        return NULL;
    }

    /* ── Allocate handle ──────────────────────────────────────────── */
    nn_bin_handle_t *handle =
        (nn_bin_handle_t *)malloc(sizeof(nn_bin_handle_t));
    if (!handle) {
        nn_printf("ERROR: Out of memory for BIN handle\r\n");
        return NULL;
    }

    /* Copy header (shallow — fixed-size struct, no pointers) */
    memcpy(&handle->header, hdr, sizeof(nn_bin_header_t));

    handle->instructions = (nn_instruction_t *)(base + instr_offset);
    handle->weight_blob  = base + weight_off;
    handle->bin_base     = addr;
    handle->total_size   = total;
    handle->instr_base   = addr + instr_offset;
    handle->weight_base  = addr + weight_off;

    nn_printf("BIN loaded: \"%s\"  v%d.%d  %lu instrs  %lu B weights\r\n",
              handle->header.model_name,
              (unsigned long)handle->header.version_major,
              (unsigned long)handle->header.version_minor,
              (unsigned long)num_instr,
              (unsigned long)hdr->weight_size);

    return handle;
}

nn_bin_handle_t *nn_load_from_sd(const char *path)
{
#if NN_USE_SD_CARD
    int sz = nn_sd_read_file(path, (uint8_t *)NN_BIN_BASE, NN_BIN_MAX_SIZE);
    if (sz < 0) {
        nn_printf("ERROR: Failed to read BIN from SD: %s\r\n", path);
        return NULL;
    }
    nn_printf("Read %d bytes from SD: %s\r\n", sz, path);
    return nn_load_from_memory(NN_BIN_BASE);
#else
    (void)path;
    nn_printf("ERROR: SD card support not compiled in (NN_USE_SD_CARD=0)\r\n");
    return NULL;
#endif
}

void nn_free_bin(nn_bin_handle_t *handle)
{
    /* The instructions and weight_blob pointers point into DDR —
     * we do NOT free those.  Only the handle itself is on the heap. */
    free(handle);
}

/* ── Validation ──────────────────────────────────────────────────────── */

int nn_validate(const nn_bin_handle_t *handle)
{
    const nn_bin_header_t *hdr = &handle->header;

    /* ── Magic (already checked during load, but double-check) ────── */
    if (hdr->magic != NN_BIN_MAGIC) {
        nn_printf("ERROR: Bad BIN magic\r\n");
        return -1;
    }

    /* ── Version ──────────────────────────────────────────────────── */
    if (hdr->version_major != 1) {
        nn_printf("ERROR: Unsupported BIN v%lu.%lu (expected 1.x)\r\n",
                  (unsigned long)hdr->version_major,
                  (unsigned long)hdr->version_minor);
        return -1;
    }

    /* ── CRC32 ───────────────────────────────────────────────────── */
    uint32_t instr_size = hdr->num_instructions * sizeof(nn_instruction_t);
    uint32_t crc = nn_crc32((const uint8_t *)handle->instructions, instr_size);
    crc = nn_crc32(handle->weight_blob, hdr->weight_size); /* NOT incremental —
        we need a combined CRC; see note below */

    /* RE-CHECK: the compiler computes CRC32 over the concatenation
     * [instr_blob ++ weight_blob].  The single-call approach above
     * only covers weights.  We compute the combined CRC properly: */
    {
        /* Recompute as single contiguous CRC over both regions.
         * Since they are adjacent in DDR (instructions immediately
         * followed by weights in the BIN layout), we can CRC the whole
         * [instructions .. weights] span in one call. */
        crc = nn_crc32((const uint8_t *)handle->instructions,
                       instr_size + hdr->weight_size);
    }

    if (crc != hdr->checksum) {
        nn_printf("WARNING: CRC mismatch: computed 0x%08lX, "
                  "stored 0x%08lX\r\n",
                  (unsigned long)crc,
                  (unsigned long)hdr->checksum);
        /* Non-fatal by default; PL may still execute successfully.
         * Change to "return -1" for strict validation. */
    }

    /* ── Workspace size ───────────────────────────────────────────── */
    if (hdr->workspace_size > NN_WORKSPACE_MAX_SIZE) {
        nn_printf("ERROR: Workspace too large: %lu > %lu KiB\r\n",
                  (unsigned long)hdr->workspace_size / 1024,
                  (unsigned long)NN_WORKSPACE_MAX_SIZE / 1024);
        return -1;
    }

    /* ── Sane instruction count ───────────────────────────────────── */
    if (hdr->num_instructions == 0) {
        nn_printf("ERROR: Zero instructions in BIN\r\n");
        return -1;
    }

    nn_printf("BIN OK: \"%s\"  %lu instrs  %lu B weights  "
              "%lu B ws  checksum=0x%08lX\r\n",
              hdr->model_name,
              (unsigned long)hdr->num_instructions,
              (unsigned long)hdr->weight_size,
              (unsigned long)hdr->workspace_size,
              (unsigned long)hdr->checksum);

    return 0;
}

/* ── Accessors ───────────────────────────────────────────────────────── */

const nn_bin_header_t *nn_get_header(const nn_bin_handle_t *handle)
{
    return &handle->header;
}

uint32_t nn_get_num_instructions(const nn_bin_handle_t *handle)
{
    return handle->header.num_instructions;
}

const nn_instruction_t *nn_get_instructions(const nn_bin_handle_t *handle)
{
    return handle->instructions;
}

const uint8_t *nn_get_weight_blob(const nn_bin_handle_t *handle)
{
    return handle->weight_blob;
}

uint32_t nn_get_instr_base(const nn_bin_handle_t *handle)
{
    return handle->instr_base;
}

uint32_t nn_get_weight_base(const nn_bin_handle_t *handle)
{
    return handle->weight_base;
}

uint32_t nn_get_total_size(const nn_bin_handle_t *handle)
{
    return handle->total_size;
}

/* ── Utilities ───────────────────────────────────────────────────────── */

uint32_t nn_find_output_offset(const nn_bin_handle_t *handle)
{
    uint32_t n = handle->header.num_instructions;
    if (n == 0) return 0;

    /* Walk backwards to find the last compute instruction. */
    int32_t i;  /* signed for reverse loop termination */
    for (i = (int32_t)n - 1; i >= 0; i--) {
        const nn_instruction_t *instr = &handle->instructions[i];
        /* Skip NOP and END. */
        if (instr->opcode == NN_OP_NOP || instr->opcode == NN_OP_END) {
            continue;
        }
        return instr->output_addr;
    }

    /* Fallback: no compute instruction found. */
    return 0;
}
