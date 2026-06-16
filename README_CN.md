# NN Accelerator — 从 ONNX 到 FPGA，我一个人写了一条推理加速全链路



[![Platform](https://img.shields.io/badge/platform-ZYNQ--7045-blue)](https://github.com/Mikeya98/nn-accelerator)
[![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Python%20%7C%20C-orange)](https://github.com/Mikeya98/nn-accelerator)

**一个完整的 ONNX → FPGA 神经网络推理加速器。**

给一个 `.onnx` 模型 → 编译器吐出指令流 → HLS 手写的计算引擎在 ZYNQ 7045 上跑起来。

这不是课程作业，是从编译器、指令集、模拟器、HLS IP 到底层固件，全部独立手写的一条全链路。

---

## 为什么做这个

训练模型的框架一抓一大把，但把模型塞进 FPGA 跑推理，中间横着一条谁都不想趟的沟：ONNX 的图你怎么拆成指令？HLS 的流水线怎么对齐 DDR 的带宽？PS 和 PL 之间除了 AXI 还要什么？

市面上不缺某一个环节的实现，缺的是**从头到尾的完整链路**。这个项目填的就是这个坑 — 每一行代码都是手写的，每一层的设计理由都写在文档里。

---

## 全链路架构

```
                    你的 ONNX 模型 (.onnx)
                           │
                           ▼
            ┌──────────────────────────┐
            │     Compiler (Python)     │
            │  ONNX 解析 → IR → 代码生成  │
            │  输出: .bin 指令流          │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │    Simulator (Python)     │
            │  周期级指令模拟器            │
            │  芯片没跑之前先验证指令对不对    │
            └────────────┬─────────────┘
                         │  验证通过
                         ▼
            ┌──────────────────────────┐
            │    HLS IP (C++, Vivado)   │
            │  Conv | FC | Pool | GRU   │
            │  ReLU | Sigmoid | Elemwise │
            │  AXI 总线接 PS 侧 DDR       │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │   Firmware (C, Bare-metal)│
            │  ARM Cortex-A9 裸机驱动     │
            │  中断驱动执行 + 结果回读      │
            └──────────────────────────┘
```

一句话概括：**PC 端写编译器把模型翻译成指令，PL 端跑通用计算引擎逐条执行，PS 端跑极简驱动只管搬运和发中断。**

设计哲学：把复杂度前置到编译器和工具链，让跑在嵌入式上的那部分尽可能"傻快"。

---

## 项目结构

```
nn_accelerator/
├── compiler/         # Python 编译器（ONNX → IR → .bin）
│   ├── parser.py         # ONNX 模型解析器
│   ├── ir.py             # 中间表示（IR）设计
│   ├── codegen.py        # 指令生成器（自定义 ISA）
│   └── tests/            # 单元测试 + 端到端测试
│
├── hls/              # Vivado HLS 计算引擎（C++）
│   ├── src/              # 引擎核心 + 各层加速实现
│   ├── tb/               # HLS 测试平台
│   └── tcl/              # 综合 + IP 打包脚本
│
├── simulator/        # Python 指令级周期模拟器
│
├── firmware/         # PS 侧裸机固件（ARM Cortex-A9）
│
├── common/           # 共享 ISA 定义（C + Python 双版本）
│
├── ip_release/       # IP 发布包 + 上板部署指南
│   ├── nn_engine_ip_v1.0.zip   # 打包好的 IP，直接导入 Vivado
│   └── 上板部署指南.md           # 从 IP 导入到上板跑通的每一步
│
├── release/          # 编译器 + 模拟器打包发布（EXE + 源码）
│
├── scripts/          # 辅助脚本
│
└── docs/             # 架构设计文档
    ├── ARCHITECTURE.md       # 架构详解
    └── 上板测试计划.md         # 硬件验证方案
```

---

## 快速开始

### 1. 用编译器把 ONNX 模型编译成指令

```bash
cd compiler
python -m compiler model.onnx -o output.bin
```

编译器会打印每一层的解析结果和生成的指令统计。你可以在 PC 上反复跑，直到满意为止 — 不需要连硬件。

### 2. 用模拟器验证指令正确性

```bash
cd simulator
python simulator.py output.bin
```

周期级的指令模拟器，会输出每一层的计算结果。这个阶段把 bug 杀干净，比上板之后用 ILA 抓信号高效一万倍。

### 3. 用 Vivado HLS 综合 + 打包 IP

```bash
cd hls
vivado_hls -f run_hls.tcl
```

HLS 综合通过后会生成 RTL，打包成 Vivado IP。或者直接用 `ip_release/` 下已经打包好的 `nn_engine_ip_v1.0.zip`。

### 4. 上板部署

参考 [ip_release/上板部署指南.md](ip_release/上板部署指南.md)，从 Block Design 搭硬件平台 → 生成 Bitstream → PS 固件编译 → 整体验证。

---

## 支持的操作

| 算子 | 状态 | 说明 |
|------|------|------|
| Conv2D | ✅ | 支持任意 kernel size / stride / padding |
| FullyConnected | ✅ | 矩阵向量乘法，权重驻留 DDR |
| MaxPool | ✅ | 行缓冲流水线实现 |
| ReLU | ✅ | — |
| Sigmoid | ✅ | 查找表实现 |
| GRU | ✅ | 门控循环单元，支持隐藏状态驻留 |
| Element-wise Add/Mul | ✅ | 逐元素运算 |

---

## 自定义 ISA

编译器不生成 x86 或 ARM 的指令，而是生成一套 **16 位自定义指令集**，专门为 FPGA 计算引擎设计：

| 特征 | 设计选择 |
|------|----------|
| 指令宽度 | 16 bit，便于 PL 侧单周期译码 |
| 寻址模式 | 寄存器间接 + DDR 线性地址 |
| 层间同步 | 指令级依赖编码，无需 PS 介入 |
| 扩展性 | 新增算子只需加 opcode，不动引擎架构 |

指令格式和编码细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 配套文章

这个项目的设计和实现细节，写了系列文章进行拆解：

| 篇序 | 主题 | 链接 |
|------|------|------|
| 1 | 从 ONNX 到 FPGA：推理加速器全链路设计 | [知乎](https://zhuanlan.zhihu.com/p/2049970004704204147) \| [掘金](https://juejin.cn/post/7651833115542978623) |
| 2 | 手写神经网络指令集：编译器 IR 与代码生成 | 撰写中 |
| 3 | HLS 写 FPGA 计算引擎：模块拆解与踩坑合集 | 撰写中 |
| 4 | 上板验证全流程 + 性能分析与总结 | 撰写中 |

---

## 平台

- **芯片**：Xilinx ZYNQ-7045 (XC7Z045)
- **开发环境**：Vivado 2018.3 + HLS
- **编译器/模拟器**：Python 3.7+，无第三方深度学习框架依赖

---

## 关于作者

**Mikeya98** — 嵌入式 AI 工程师

华科人工智能与自动化学院硕士（2023）。关注 FPGA 加速、实时系统、神经网络端侧部署。

这个项目是个人独立完成的开放源码项目，旨在探索 FPGA 上 AI 推理加速的工程实践。所有代码和文档均为个人学习与研究成果。

---

*如果这个项目对你有帮助，欢迎 Star ⭐ 和分享。有问题可以在 Issues 里提，或者去知乎/掘金文章下面留言讨论。*
