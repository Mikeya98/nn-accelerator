/**
 * NN Accelerator — SD Card Implementation
 *
 * Thin wrapper around the Xilinx xilffs (FAT) library.
 *
 * Compile with:  NN_USE_SD_CARD=1
 * BSP required:  xilffs
 */

#include "nn_sd.h"

#if NN_USE_SD_CARD

/* Xilinx FAT File System */
#include "ff.h"

/* Xilinx SD controller driver (ZYNQ7 PS). */
#include "xsdps.h"        /* XSdPs — ZYNQ SD/SDIO peripheral driver */

/* ── Module state ──────────────────────────────────────────────────────── */

/* FAT filesystem object.  Must persist across calls. */
static FATFS g_fatfs;

/* Mount status flag. */
static int g_sd_mounted = 0;

/* ── Platform-specific SD instance ─────────────────────────────────────── */

/* On ZYNQ 7045 the SD controller base address and device ID depend on
 * which SD interface is used (SD0 or SD1).  SD0 is typically the
 * boot SD card; SD1 is available as a secondary slot. */
#define NN_SD_DEVICE_ID   XPAR_XSDPS_0_DEVICE_ID
/* Note: XPAR_XSDPS_0_DEVICE_ID is defined by the Xilinx BSP (xparameters.h).
 * If SD1 is used instead, change to XPAR_XSDPS_1_DEVICE_ID. */

/* ── Public API ────────────────────────────────────────────────────────── */

int nn_sd_init(void)
{
    FRESULT res;
    static XSdPs sd_card;  /* SD controller instance */

    /* ── Initialise SD host controller ──────────────────────────── */
    {
        XSdPs_Config *cfg = XSdPs_LookupConfig(NN_SD_DEVICE_ID);
        if (!cfg) {
            nn_printf("ERROR: SD controller config not found (ID=%d)\r\n",
                      NN_SD_DEVICE_ID);
            return -1;
        }

        int status = XSdPs_CfgInitialize(&sd_card, cfg, cfg->BaseAddress);
        if (status != XST_SUCCESS) {
            nn_printf("ERROR: SD controller init failed (status=%d)\r\n",
                      status);
            return -1;
        }
    }

    /* ── Mount FAT filesystem ───────────────────────────────────── */
    /* f_mount with NULL as second arg forces re-mount.
     * On ZYNQ the SD card appears as logical drive "0:". */
    res = f_mount(&g_fatfs, "0:", 1);
    if (res != FR_OK) {
        nn_printf("ERROR: f_mount failed (FRESULT=%d).  "
                  "Is an SD card inserted?\r\n", (int)res);
        return -1;
    }

    g_sd_mounted = 1;
    nn_printf("SD card mounted OK\r\n");
    return 0;
}

int nn_sd_read_file(const char *filename, uint8_t *buffer, uint32_t max_size)
{
    if (!g_sd_mounted) {
        nn_printf("ERROR: SD card not mounted\r\n");
        return -1;
    }

    FIL fil;
    FRESULT res;

    /* The xilffs library expects paths relative to the mounted volume.
     * "0:model.bin" format, or just "model.bin" if the default drive
     * was set.  We prepend "0:" for explicitness. */
    char full_path[128];
    {
        /* If the caller already includes a drive prefix, use as-is;
         * otherwise prepend "0:". */
        if (filename[0] != '\0' && filename[1] == ':') {
            /* Already has drive prefix: "0:model.bin" */
            strncpy(full_path, filename, sizeof(full_path) - 1);
            full_path[sizeof(full_path) - 1] = '\0';
        } else {
            /* No drive prefix — add "0:". */
            full_path[0] = '0';
            full_path[1] = ':';
            strncpy(full_path + 2, filename, sizeof(full_path) - 3);
            full_path[sizeof(full_path) - 1] = '\0';
        }
    }

    res = f_open(&fil, full_path, FA_READ);
    if (res != FR_OK) {
        nn_printf("ERROR: f_open(\"%s\") failed (FRESULT=%d)\r\n",
                  full_path, (int)res);
        return -1;
    }

    UINT bytes_read = 0;
    res = f_read(&fil, buffer, (UINT)max_size, &bytes_read);
    f_close(&fil);

    if (res != FR_OK) {
        nn_printf("ERROR: f_read failed (FRESULT=%d)\r\n", (int)res);
        return -1;
    }

    return (int)bytes_read;
}

#endif /* NN_USE_SD_CARD */
