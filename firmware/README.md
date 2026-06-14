# NN Accelerator — PS Firmware

Bare-metal C firmware for the ZYNQ 7045 ARM Cortex-A9 processor.
Minimal driver that loads a compiler-generated BIN file, configures the
PL `nn_engine` IP, starts inference, and reads back results.

## Architecture

```
┌─────────────────────────────────────────┐
│  PS (ARM Cortex-A9)                     │
│  ┌───────────┐   ┌──────────┐           │
│  │ main.c    │──▶│ loader   │  BIN parse│
│  │  入口编排 │   │ CRC32    │  & valid  │
│  └─────┬─────┘   └──────────┘           │
│        │                                 │
│  ┌─────▼─────┐                          │
│  │ driver    │  AXI-Lite MMIO           │
│  │ reg r/w   │  GP0 → s_axi_control     │
│  └─────┬─────┘                          │
└────────┼────────────────────────────────┘
         │ AXI4-Lite
┌────────▼────────────────────────────────┐
│  PL (FPGA)                              │
│  nn_engine IP                           │
│  ┌──────┐ ┌──┐ ┌────┐ ┌───┐ ┌──────┐  │
│  │decoder│ │FC│ │Pool│ │Act│ │GRU   │  │
│  └──────┘ └──┘ └────┘ └───┘ └──────┘  │
└─────────────────────────────────────────┘
```

## DDR Memory Layout

```
0x08000000 ┌─ BIN file (loaded by FSBL/JTAG or SD) ─┐
           │ Header (256 B)                          │
           │ Instructions (N × 64 B)  ← INSTR_ADDR   │
           │ Weights                  ← WEIGHT_ADDR  │
           └─────────────────────────────────────────┘

0x03000000 ┌─ Workspace (16 MiB) ────────────────────┐
           │ Input  @ offset 0                        │
           │ Output @ compiler-assigned offset        │
           │ Scratch                                   │
           └──────────────────────────────────────────┘

0x3F000000 ┌─ Firmware code / data / stack ──────────┐
```

## File Map

```
firmware/
├── inc/
│   ├── nn_platform.h   ← ALL configuration (#define)
│   ├── nn_driver.h     ← Register driver API
│   ├── nn_loader.h     ← BIN loader API
│   └── nn_sd.h         ← SD card API (optional)
├── src/
│   ├── main.c          ← Entry point & orchestration
│   ├── nn_driver.c     ← MMIO read/write
│   ├── nn_loader.c     ← BIN parse + CRC32 validation
│   └── nn_sd.c         ← xilffs wrapper (optional)
├── lscript.ld          ← Linker script
└── README.md
```

## Configuration

All hardware-specific settings are in **`inc/nn_platform.h`**:

| Macro | Default | Description |
|---|---|---|
| `NN_AXI_BASE_ADDR` | `0x43C00000` | AXI-Lite base of `nn_engine` IP |
| `NN_BIN_BASE` | `0x08000000` | Where BIN is loaded in DDR |
| `NN_WORKSPACE_BASE` | `0x03000000` | Workspace buffer base |
| `NN_PL_TIMEOUT_MS` | `10000` | PL poll timeout in ms |
| `NN_USE_INTERRUPTS` | `0` | Set to 1 for IRQ-driven completion |
| `NN_USE_SD_CARD` | `0` | Set to 1 to enable SD card loading |

To adapt to a different Vivado Block Design, update ONLY `NN_AXI_BASE_ADDR`
to match the value in Vivado Address Editor → `nn_engine` → Offset Address.

## Building

### Prerequisites
- Xilinx SDK 2018.3 / Vitis (for ZYNQ7 BSP)
- `arm-none-eabi-gcc` (for standalone cross-compilation)

### With Xilinx SDK / Vitis
1. Create a new Application Project targeting your hardware platform (.xsa/.hdf).
2. Set the BSP to **standalone** (bare-metal).
3. Add `common/` to the include path (`-I../../common`).
4. If using SD card: enable **xilffs** in BSP settings.
5. Copy `firmware/src/*.c` and `firmware/inc/*.h` into the project.
6. Build → generates `.elf` file.

### Standalone (arm-none-eabi-gcc)
```bash
arm-none-eabi-gcc \
  -mcpu=cortex-a9 -mthumb -mfpu=neon -mfloat-abi=hard \
  -I inc -I ../common \
  -Wall -Wextra -O2 \
  -T lscript.ld \
  -o firmware.elf \
  src/main.c src/nn_driver.c src/nn_loader.c

# Generate binary for bare-metal boot
arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
```

## Running

### Development (JTAG / SDK)
1. Download bitstream + firmware ELF via Xilinx SDK.
2. Use SDK's "DDR Memory Write" to load `model.bin` at `0x08000000`.
3. (Optional) Write input data to `0x03000000`.
4. Launch firmware.
5. Observe UART output at 115200 baud.

### Deployment (SD Card)
1. Format SD card as FAT32.
2. Copy `model.bin` and `input.bin` to the SD card.
3. Copy firmware ELF (or boot.bin with FSBL) to the SD card.
4. Boot ZYNQ from SD card.
5. Firmware loads BIN and input from SD automatically.

## UART Output Example

```
╔══════════════════════════════════════════╗
║      NN Accelerator — PS Firmware        ║
║      ZYNQ 7045  /  Bare-metal C          ║
║      Version 1.0                         ║
╚══════════════════════════════════════════╝

[1/8] Initialising PL driver ...
      AXI base: 0x43C00000  STATUS=0
[2/8] Resetting PL engine ...
[3/8] Loading BIN ...
BIN loaded: "gru_test"  v1.0  42 instrs  12345 B weights
[4/8] Validating BIN ...
BIN OK: "gru_test"  42 instrs  12345 B weights  4096 B ws  checksum=0xABCD1234
      Instructions (42):
        [ 0] FC        out=0x0010
        [ 1] Sigmoid   out=0x00E0
        ...
      Breakdown: FC=12 GRU=1 POOL=2 Sigmoid=4 Tanh=3 EMul=4 EAdd=6 END=1
[5/8] Writing input data ...
      Input zero-filled (128 B) — pre-load real data for inference
[6/8] Configuring PL registers ...
      INSTR_ADDR    = 0x08000100
      WEIGHT_ADDR   = 0x08000B80
      WORKSPACE_ADDR= 0x03000000
[7/8] Starting PL inference ...
      Waiting for PL DONE (timeout=10000 ms) ...
      PL DONE received.
[8/8] Reading output ...
      Output: 32 floats (128 B) from workspace+0x0010
      [+0.123456, -0.654321, ...]

==== Inference complete ====
```

## Verification

Before deploying to hardware, verify the firmware logic against the
Python simulator (golden reference):

```bash
# Run the simulator on a test BIN (produces expected output values)
cd ../simulator
python -m pytest tests/test_simulator.py -v

# Compare firmware register sequence:
#   The simulator's Simulator.run() method (lines 116-151 of simulator.py)
#   performs the exact register writes that the firmware implements.
```

## License

Internal project — no license.
