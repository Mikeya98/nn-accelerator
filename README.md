# NN Accelerator — ONNX Neural Network Inference on FPGA

[![Platform](https://img.shields.io/badge/platform-ZYNQ--7045-blue)](https://github.com/Mikeya98/nn-accelerator)
[![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Python%20%7C%20C-orange)](https://github.com/Mikeya98/nn-accelerator)

**A complete ONNX-to-FPGA neural network inference chain.**  
Takes an ONNX model → compiles to custom ISA → runs on a hand-written HLS accelerator IP on Xilinx ZYNQ-7045 platform.

---

## Architecture

```
ONNX Model (.onnx)
     │
     ▼
┌─────────────┐
│  Compiler    │  Python: parse → IR → codegen → binary instruction stream
└──────┬──────┘
       │  .bin (custom ISA)
       ▼
┌─────────────┐
│  Simulator   │  Python: cycle-accurate instruction-level simulator
└──────┬──────┘
       │  verified
       ▼
┌─────────────┐
│  HLS IP      │  Vivado HLS C++: conv, fc, pool, activation, GRU, elemwise
└──────┬──────┘
       │  AXI
       ▼
┌─────────────┐
│  Firmware    │  Bare-metal C driver (ARM Cortex-A9 PS)
└─────────────┘
```

## Features

- **Full ONNX pipeline**: parser → intermediate representation → codegen → binary `.bin` files
- **Custom ISA**: 16-bit instruction set optimized for FPGA execution
- **HLS accelerator**: hand-written in C++ (Vivado HLS), supports Conv2D, FullyConnected, MaxPool, ReLU/Sigmoid, GRU, Element-wise ops
- **Instruction simulator**: verify correctness before synthesis
- **Bare-metal firmware**: lightweight ARM driver with interrupt-driven execution
- **IP packaging**: ready-to-use Vivado IP with deployment guide

## Project Structure

```
nn_accelerator/
├── compiler/         # Python compiler (ONNX → IR → .bin)
│   ├── parser.py         # ONNX model parser
│   ├── ir.py             # Intermediate representation
│   ├── codegen.py        # ISA code generator
│   └── tests/            # Unit & end-to-end tests
├── hls/              # Vivado HLS accelerator
│   ├── src/              # C++ HLS sources (engine + layers)
│   ├── tb/               # HLS testbench
│   └── tcl/              # Synthesis & packaging scripts
├── simulator/        # Python instruction-level simulator
├── firmware/         # Bare-metal PS firmware (ARM Cortex-A9)
├── common/           # Shared ISA definitions (C + Python)
├── ip_release/       # IP deployment guide & integration scripts
└── docs/             # Architecture & design documents
```

## Quick Start

### 1. Compile an ONNX model

```bash
cd compiler
python -m compiler model.onnx -o output.bin
```

### 2. Verify with simulator

```bash
cd simulator
python simulator.py output.bin
```

### 3. Build HLS IP

```bash
cd hls
vivado_hls -f run_hls.tcl
```

### 4. Deploy on hardware

See [ip_release/deployment_guide.md](ip_release/deployment_guide.md) for integration steps.

## Supported Ops

| Op | Status |
|----|--------|
| Conv2D | ✅ |
| FullyConnected | ✅ |
| MaxPool | ✅ |
| ReLU | ✅ |
| Sigmoid | ✅ |
| GRU | ✅ |
| Element-wise (Add/Mul) | ✅ |

## Platforms

- Xilinx ZYNQ-7045 (XC7Z045), verified with Vivado 2018.3

## Author

**Mikeya98** — Embedded AI Engineer  
M.S. Artificial Intelligence & Automation, HUST (2023)  
Focus: FPGA acceleration, real-time systems, neural network deployment on edge devices.

---

*This project is a personal open-source effort to explore FPGA-based AI acceleration techniques.*
