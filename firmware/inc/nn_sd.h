#ifndef NN_SD_H
#define NN_SD_H
/**
 * NN Accelerator — SD Card (xilffs) Interface
 *
 * Optional module for loading BIN files and input data from an SD card
 * formatted with a FAT32 filesystem.
 *
 * Requires the Xilinx xilffs library in the BSP:
 *   - In Vitis / SDK:  Board Support Package → xilffs → enable
 *   - Header:  #include "ff.h"
 *   - Library: -lxilffs
 *
 * Compile with NN_USE_SD_CARD=1 to enable.
 */

#include "nn_platform.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if NN_USE_SD_CARD

/**
 * Initialize the SD card controller and mount the FAT filesystem.
 * Must be called once before any nn_sd_read_file() calls.
 *
 * Returns 0 on success, -1 if no card is present or the filesystem
 * cannot be mounted.
 */
int nn_sd_init(void);

/**
 * Read an entire file from the SD card into a caller-provided buffer.
 *
 * @param filename  Path on the FAT filesystem (e.g., "model.bin").
 * @param buffer    Destination buffer in DDR.
 * @param max_size  Maximum bytes to read (buffer must be at least this large).
 *
 * Returns the number of bytes read on success, or -1 on any error
 * (file not found, read error, etc.).  A short read (return < max_size
 * but > 0) indicates the file is smaller than max_size — this is fine.
 */
int nn_sd_read_file(const char *filename, uint8_t *buffer, uint32_t max_size);

#else  /* !NN_USE_SD_CARD */

/* Stubs — always return failure. */

static inline int nn_sd_init(void)
{
    return -1;
}

static inline int nn_sd_read_file(const char *filename,
                                  uint8_t *buffer, uint32_t max_size)
{
    (void)filename;
    (void)buffer;
    (void)max_size;
    return -1;
}

#endif /* NN_USE_SD_CARD */

#ifdef __cplusplus
}
#endif

#endif /* NN_SD_H */
