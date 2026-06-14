#ifndef NN_BIN_H
#define NN_BIN_H
/**
 * BIN file format — compiler output, consumed by the PS loader at boot.
 *
 * Layout (single contiguous file, loaded from Flash to DDR):
 *
 *   ┌──────────────────────────────┐  offset 0
 *   │  Header          (256 B)     │
 *   ├──────────────────────────────┤
 *   │  Instruction[0]  (64 B)      │
 *   │  Instruction[1]  (64 B)      │
 *   │  ...                         │
 *   │  Instruction[N-1] (64 B)     │
 *   ├──────────────────────────────┤
 *   │  Weight Buffer   (W bytes)   │
 *   │  (all weight / bias tensors  │
 *   │   contiguously, 4B aligned)  │
 *   └──────────────────────────────┘
 *
 * All addresses inside instructions are BYTE OFFSETS into the weight
 * buffer or the runtime workspace (see nn_isa.h).  The PS loader
 * resolves them to absolute DDR addresses after copying the weight
 * buffer into place.
 */

#include <stdint.h>

#define NN_BIN_MAGIC  0x31424E4E   /* "NNB1" — Neural Network Binary v1 */

typedef struct {
    uint32_t magic;                 /* 0x00  NN_BIN_MAGIC                  */
    uint16_t version_major;         /* 0x04  currently 1                   */
    uint16_t version_minor;         /* 0x06  currently 0                   */
    uint32_t num_instructions;      /* 0x08                                */
    uint32_t weight_size;           /* 0x0C  bytes                         */
    uint32_t workspace_size;        /* 0x10  max scratch bytes needed      */
    uint32_t input_size;            /* 0x14  model input buffer  (bytes)   */
    uint32_t output_size;           /* 0x18  model output buffer (bytes)   */
    uint32_t checksum;              /* 0x1C  CRC32 of [instrs + weights]   */
    char     model_name[64];        /* 0x20  null-terminated               */
    uint8_t  reserved[160];         /* 0x60  → pad to 256 B                */
} nn_bin_header_t;

#ifdef __cplusplus
  #if __cplusplus >= 201103L
    static_assert(sizeof(nn_bin_header_t) == 256, "Header must be 256 bytes");
  #endif
#else
  #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Static_assert(sizeof(nn_bin_header_t) == 256, "Header must be 256 bytes");
  #endif
#endif

/**
 * High-level BIN layout helpers:
 *
 *   header_offset        = 0
 *   instructions_offset  = 256
 *   instructions_size    = num_instructions * 64
 *   weights_offset       = 256 + instructions_size
 *   total_file_size      = weights_offset + weight_size
 */

#endif /* NN_BIN_H */
