# NN 加速器上板部署指南

> 适用：Vivado 2018.3, ZYNQ 7045 (xc7z045ffg900-2)
> 日期：2026-06-09

---

## 一、文件清单

```
ip_release/
├── nn_engine_ip_v1.0.zip   ← IP 核发布包（已含 Vivado 原生 component.xml）
├── integrate.tcl            ← Tcl 脚本（快速添加 IP 到工程）
├── pack_ip.tcl              ← IP 重打包脚本（仅需重生成 IP 时使用）
└── 上板部署指南.md           ← 本文件
```

---

## 二、将 IP 核加入 Vivado 工程

### 方法 1：GUI 手动添加（推荐）

1. 解压 `nn_engine_ip_v1.0.zip` 到 `E:/work/nn_ip/nn_engine_1.0/`
2. 打开 Vivado 工程
3. 菜单 `Project Settings` → `IP` → `Repository Manager`
4. 点 `+` 添加路径：`E:/work/nn_ip`
5. 点 `Refresh All`
6. 确认 `nn_accel : nn : nn_engine : 1.0` 出现在 IP Catalog

### 方法 2：Tcl Console（快捷）

```tcl
# 在 Vivado Tcl Console 中执行
set_property ip_repo_paths [list E:/work/nn_ip] [current_project]
update_ip_catalog
```

---

## 三、Block Design 集成

### 3.1 添加 IP

在 Block Design 中：`Add IP` → 搜索 `nn_engine` → 双击添加

### 3.2 接口连接表

| IP 端口 | 类型 | 方向 | 连接到 | 说明 |
|---------|------|------|--------|------|
| `ap_clk` | Clock | input | **PS FCLK_CLK0** | 150MHz 时钟（6.67ns 周期） |
| `ap_rst_n` | Reset (低有效) | input | **PS FCLK_RESET0_N** | 复位信号 |
| `m_axi_ddr_port` | AXI4 Master | output | **PS S_AXI_HP0** | DMA 直读直写 DDR |
| `s_axi_control` | AXI4-Lite Slave | input | **PS M_AXI_GP0** | 控制寄存器 |
| `s_axi_AXILiteS` | AXI4-Lite Slave | input | **PS M_AXI_GP0** | 数据通道 |

### 3.3 连接步骤

```
ZYNQ7 Processing System
├── FCLK_CLK0       ──►  ap_clk (nn_engine)
├── FCLK_RESET0_N   ──►  ap_rst_n (nn_engine)
├── M_AXI_GP0       ──►  AXI SmartConnect ──► s_axi_control (nn_engine)
│                                    └──────► s_axi_AXILiteS (nn_engine)
└── S_AXI_HP0       ◄──  AXI SmartConnect ◄── m_axi_ddr_port (nn_engine)
```

**操作**：添加 IP 后，点击 Block Design 上方 **Run Connection Automation**，Vivado 会自动连接大部分接口。确认：

1. `m_axi_ddr_port` → PS **S_AXI_HP0**（不是 GP 口）
2. 两个 AXI-Lite → PS **M_AXI_GP0**（经 SmartConnect）
3. Clock → FCLK_CLK0，Reset → FCLK_RESET0_N

### 3.4 地址分配

菜单 `Address Editor` 中，给以下接口分配地址空间：

| 接口 | 地址空间 | 说明 |
|------|---------|------|
| `s_axi_control` | ≥ 128 Bytes | 控制寄存器（CTRL/STATUS/地址指针） |
| `s_axi_AXILiteS` | ≥ 128 Bytes | 数据通道 |

---

## 四、寄存器定义（s_axi_control 基址偏移）

| 偏移 | 名称 | 位宽 | 读写 | 说明 |
|------|------|------|------|------|
| 0x00 | CTRL | 32 | W | bit[0]=START, bit[1]=RESET |
| 0x04 | STATUS | 32 | R | 0=IDLE, 1=DONE |
| 0x08 | INSTR_ADDR | 32 | R/W | 指令缓冲区物理地址 |
| 0x0C | WEIGHT_ADDR | 32 | R/W | 权重数据物理地址 |
| 0x10 | INPUT_ADDR | 32 | R/W | 输入数据物理地址 |
| 0x14 | OUTPUT_ADDR | 32 | R/W | 输出数据物理地址 |
| 0x18 | WORKSPACE_ADDR | 32 | R/W | 工作空间物理地址 |

### 操作流程

```
1. PS 用 ONNX→BIN 编译器生成指令二进制 (.bin)
2. PS 将 .bin 写入 DDR 某处
3. PS 将权重数据写入 DDR
4. PS 将输入数据写入 DDR
5. PS 写寄存器：INSTR_ADDR, WEIGHT_ADDR, INPUT_ADDR, OUTPUT_ADDR
6. PS 写 CTRL bit[0]=1 (START)
7. PS 轮询 STATUS，等 DONE=1
8. PS 从 OUTPUT_ADDR 读取结果
```

---

## 五、如需重新生成 IP（仅 Vivado 2018.3）

如果修改了 HLS 源码需要重新打包 IP：

1. 拷贝整个 `nn_accelerator` 项目到有 Vivado 的电脑
2. 打开 Vivado，在 Tcl Console 中执行：
   ```tcl
   source E:/work/nn_ip/ip_release/pack_ip.tcl
   ```
3. 生成完毕后，`ip_release/nn_engine_1.0/` 下即为新 IP
4. 运行 `python scripts/gen_component_xml.py` 不再需要 — Vivado 原生打包已替代

---

## 六、工程检查清单

- [ ] IP Catalog 中能看到 `nn_engine`（nn_accel:nn:nn_engine:1.0）
- [ ] Block Design 中添加 nn_engine 不报错
- [ ] Run Connection Automation 自动连好所有接口
- [ ] m_axi_ddr_port → S_AXI_HP0（DDR 高性能口）
- [ ] s_axi_control、s_axi_AXILiteS → M_AXI_GP0（控制口）
- [ ] Address Editor 中分配了地址
- [ ] Validate Design 通过（无错误）
- [ ] Generate Bitstream 成功
