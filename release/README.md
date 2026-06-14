# NN Accelerator — Release Package v1.1

## What's new in v1.1
- Fixed: ONNX Constant op now parsed correctly (no more "Shape inference not implemented" error)
- Fixed: IP component.xml now includes full <spirit:model> with 81 port definitions
  Vivado IP catalog can now scan and import the IP without "ip flow 19-1977" errors

## Contents
```
nn_compiler.exe          - ONNX to BIN compiler (standalone, no Python needed)
nn_engine_ip_v1.0.zip    - PL FPGA IP core with valid component.xml
integrate.tcl            - Vivado integration script
```

## 1. Compiler: nn_compiler.exe

### Usage
```cmd
:: Basic usage
nn_compiler.exe model.onnx -o model.bin

:: With name and verbose output
nn_compiler.exe gru_model.onnx -o gru_model.bin --name my_gru -v

:: Override input shape
nn_compiler.exe model.onnx -o model.bin --input-shape x:1,128
```

### Supported Ops (v1.1)
FC/Gemm, ReLU, Sigmoid, Tanh, MaxPool, AvgPool, GlobalMaxPool, GlobalAvgPool,
GRU, ElemMul, ElemAdd, Reshape, Transpose, Concat, Squeeze, Unsqueeze, Flatten,
Constant, BatchNormalization, Dropout

## 2. PL IP Core: nn_engine_ip_v1.0.zip

### Integration (simple method)
1. Extract zip contents
2. In Vivado Tcl console:
   ```tcl
   source integrate.tcl
   ```
3. IP "nn_engine" now appears in your project's IP catalog
4. Add to Block Design, run Connection Automation

### Block Design Connections
| IP Port        | PS Port        | Notes            |
|----------------|----------------|------------------|
| s_axi_control  | M_AXI_GP0      | AXI-Lite slave   |
| m_axi_ddr      | S_AXI_HP0      | AXI4 master, 64b |
| ap_clk         | FCLK_CLK0      | 150 MHz (6.67ns) |
| ap_rst_n       | FCLK_RESET0_N  | Active-low       |

### Resources (XC7Z045, 150 MHz)
| Resource  | Used    | %  |
|-----------|---------|----|
| BRAM_18K  | 53      | 5% |
| DSP48E    | 59      | 7% |
| FF        | 23,363  | 5% |
| LUT       | 24,688  | 11%|

## Complete Flow
```
ONNX model -> nn_compiler.exe -> model.bin
                                    |
                          +---------+---------+
                          |                   |
                    PS Firmware (ARM)   PL IP (FPGA)
                    - Load BIN          - Fetch & decode
                    - Write registers   - Compute (FC/GRU/...)
                    - Start PL          - Signal DONE
                    - Read output
```

For firmware, see nn_accelerator/firmware/ in the project repo.
