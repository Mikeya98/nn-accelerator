# NN Accelerator PL Engine — HLS IP Core

ZYNQ 7045 (XC7Z045) FPGA compute engine.  HLS C++ → Vivado IP.

## Architecture (single-file, Vivado 2018.3 compatible)

```
nn_engine.cpp  ←  ALL compute functions in ONE translation unit
  ├── nn_engine()          Top-level AXI wrapper
  ├── fetch_instruction()  DDR → 64B instruction decode
  ├── fc_compute()         FC (MatMul + bias + optional activation)
  ├── pool_compute()       Max/Avg Pooling (line buffer)
  ├── relu_compute()       ReLU
  ├── sigmoid_compute()    Sigmoid (256-segment LUT)
  ├── tanh_compute()       Tanh (= 2·σ(2x)−1)
  ├── elem_mul_compute()   Element-wise multiply
  ├── elem_add_compute()   Element-wise add
  └── gru_compute()        GRU (FSM: 6×MatMul + gates + multi-step)
```

**Why single-file?**  Vivado 2018.3 has a bug (`Pointer Array Geometry` crash)
when the `m_axi` DDR pointer passes across module boundaries.  Grouping all
`static` functions in one translation unit avoids this.  The archived
multi-file modules in `src/` are functionally identical but kept for reference.

## Register Map

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x00 | CTRL | W | bit0=START, bit1=RESET |
| 0x04 | STATUS | R | 0=IDLE, 1=DONE |
| 0x08 | INSTR_ADDR | W | DDR byte address of instructions |
| 0x0C | WEIGHT_ADDR | W | DDR byte address of weight buffer |
| 0x10 | INPUT_ADDR | W | DDR byte address of input |
| 0x14 | OUTPUT_ADDR | W | DDR byte address of output |
| 0x18 | WORKSPACE_ADDR | W | DDR byte address of workspace |

## Quick Start

```bash
# 1. Generate test BIN
cd nn_accelerator
python -m compiler model.onnx -o hls/model.bin

# 2. C Simulation
cd hls
set DO_CSIM=1 && vivado_hls -f run_hls.tcl

# 3. Synthesis
vivado_hls -f run_hls.tcl

# 4. RTL output
#    nn_accelerator/solution1/syn/verilog/nn_engine.v
#    nn_accelerator/solution1/syn/vhdl/nn_engine.vhd
```

## Synthesis Results (2026-06-06, Vivado 2018.3)

```
✅ C Simulation     — passed (FC model, 2 instructions)
✅ Synthesizability — passed  
✅ Architecture Syn — passed (10 modules implemented)
✅ RTL Generation   — passed (Verilog + VHDL + SystemC)
```

### Implemented Modules

| Module | Function | DSP cores |
|--------|----------|-----------|
| fetch_instruction | DDR→64B decode | mul |
| fc_compute | MatMul + bias + act | fadd, fcmp, fmul, sitofp |
| gru_compute | 6×MatMul FSM + gates | fadd, fcmp, fmul, fsub, mul, sitofp, udiv |
| pool_compute | Line-buffer sliding window | fadd, fcmp, fdiv, mul, udiv, uitofp |
| relu_compute | max(0,x) | fcmp |
| sigmoid_compute | LUT piecewise | fadd, fcmp, fmul, sitofp |
| tanh_compute | 2σ(2x)−1 | fadd, fcmp, fmul, sitofp |
| elem_mul/add | vector mul/add | fmul (mul) / fadd (add) |

### Memory Inference

| Buffer | Type |
|--------|------|
| gru_compute h_buf, x_buf | Block RAM |
| pool_compute line_buf | Block RAM |
| nn_engine gru_scratch | Block RAM |
| sigmoid LUT tables | Auto ROM |
| fetch_instruction raw | Distributed RAM |

## Block Design Integration

1. Synthesis → RTL in `nn_accelerator/solution1/syn/verilog/`
2. Vivado Block Design → Add RTL module `nn_engine`
3. Connect:
   - `s_axi_control` → PS `M_AXI_GP0`
   - `m_axi_ddr` → PS `S_AXI_HP0`  
   - `interrupt` → PS `IRQ_F2P[0]`
   - `ap_clk` → PS `FCLK_CLK0` (150 MHz)
   - `ap_rst_n` → PS `FCLK_RESET0_N`
4. Address Editor → assign base to `s_axi_control` (≥128 B)

## Configuration

Key parameters in `src/nn_params.h`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAC_UNITS | 8 | Parallel MAC lanes |
| MAX_INPUT_FEATURES | 4096 | Max FC input dimension |
| MAX_OUTPUT_FEATURES | 4096 | Max FC output dimension |
| GRU_MAX_HIDDEN | 512 | Max GRU hidden size |
| INSTR_BUFFER_DEPTH | 256 | Max instructions cached |
| AXI_DATA_WIDTH | 64 | DDR bus width (bits) |

## Files

```
hls/
├── src/
│   ├── nn_engine.cpp    ← ACTIVE synthesis file (all compute)
│   ├── nn_engine.h      ← Top-level declaration
│   ├── nn_isa.h         ← ISA types (64B instruction)
│   ├── nn_bin.h         ← BIN format header
│   ├── nn_params.h      ← Parameters & register offsets
│   ├── sigmoid_lut.inc  ← Sigmoid LUT (256 entries)
│   ├── nn_act.h/cpp     ← (archived) original multi-file modules
│   ├── nn_fc.h/cpp      ← (archived)
│   ├── nn_gru.h/cpp     ← (archived)
│   ├── nn_pool.h/cpp    ← (archived)
│   ├── nn_elemwise.h/cpp← (archived)
│   └── nn_decoder.h/cpp ← (archived)
├── tb/
│   └── nn_engine_tb.cpp ← C simulation testbench
├── run_hls.tcl          ← Synthesis script
├── tcl/pack_ip.tcl      ← IP packaging helper
└── README.md
```
