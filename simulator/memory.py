"""
DDR memory model + PL register file for the instruction simulator.

Simulates the hardware memory layout:

    ┌──────────────────────┐  offset 0 (base address)
    │  Weight buffer        │  ← BIN weights loaded here
    ├──────────────────────┤
    │  Input buffer         │  ← PS writes input data
    ├──────────────────────┤
    │  Output buffer        │  ← PL writes results
    ├──────────────────────┤
    │  Workspace            │  ← intermediate tensors + GRU scratch
    └──────────────────────┘

PL control registers (AXI-Lite):

    Offset  Register       R/W   Description
    ────────────────────────────────────────────
    0x00    CTRL           W     PS writes 1 → PL starts inference
    0x04    STATUS         R     PL writes 1 when done; PS polls this
    0x08    INSTR_ADDR     R/W   BIN instructions base address in DDR
    0x0C    WEIGHT_ADDR    R/W   Weight buffer base address in DDR
    0x10    INPUT_ADDR     R/W   Input data address in DDR
    0x14    OUTPUT_ADDR    R/W   Output data address in DDR
    0x18    WORKSPACE_ADDR R/W   Workspace base address in DDR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Register offsets ─────────────────────────────────────────────────
REG_CTRL           = 0x00
REG_STATUS         = 0x04
REG_INSTR_ADDR     = 0x08
REG_WEIGHT_ADDR    = 0x0C
REG_INPUT_ADDR     = 0x10
REG_OUTPUT_ADDR    = 0x14
REG_WORKSPACE_ADDR = 0x18
REG_COUNT          = 0x1C  # 7 registers × 4 bytes = 28 bytes

# ── Status values ────────────────────────────────────────────────────
STATUS_IDLE = 0
STATUS_BUSY = 2   # PL is running (optional, if you want busy indication)
STATUS_DONE = 1


@dataclass
class Memory:
    """Simulated DDR memory.

    The PS driver loads the entire BIN into this space, then places
    input / output / workspace buffers at configured offsets.

    Attributes
    ----------
    size:
        Total simulated DDR size in bytes (default 16 MiB).
    buf:
        Bytearray backing store.
    """

    size: int = 16 * 1024 * 1024  # 16 MiB
    buf: bytearray = field(default_factory=lambda: bytearray(16 * 1024 * 1024))

    def read(self, addr: int, size: int) -> bytes:
        """Read raw bytes from DDR."""
        return bytes(self.buf[addr : addr + size])

    def write(self, addr: int, data: bytes) -> None:
        """Write raw bytes to DDR."""
        end = addr + len(data)
        if end > self.size:
            raise MemoryError(f"DDR write at {addr:#x} exceeds {self.size} bytes")
        self.buf[addr:end] = data

    def read_f32(self, addr: int, count: int) -> np.ndarray:
        """Read FP32 values from DDR."""
        raw = self.read(addr, count * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def write_f32(self, addr: int, data: np.ndarray) -> None:
        """Write FP32 values to DDR."""
        raw = data.astype(np.float32).tobytes()
        self.write(addr, raw)


@dataclass
class RegisterFile:
    """PL control registers (simulated AXI-Lite peripheral).

    Attributes
    ----------
    regs:
        32-entry uint32 register file, though only 7 are used.
    """

    regs: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        # Initialise all registers to 0
        for i in range(32):
            self.regs.setdefault(i, 0)

    def read(self, offset: int) -> int:
        """Read a 32-bit register."""
        return self.regs.get(offset, 0)

    def write(self, offset: int, value: int) -> None:
        """Write a 32-bit register."""
        self.regs[offset] = value & 0xFFFFFFFF

    # ── Convenience accessors ──────────────────────────────────────
    @property
    def ctrl(self) -> int:
        return self.read(REG_CTRL)

    @ctrl.setter
    def ctrl(self, v: int) -> None:
        self.write(REG_CTRL, v)

    @property
    def status(self) -> int:
        return self.read(REG_STATUS)

    @status.setter
    def status(self, v: int) -> None:
        self.write(REG_STATUS, v)

    @property
    def instr_addr(self) -> int:
        return self.read(REG_INSTR_ADDR)

    @instr_addr.setter
    def instr_addr(self, v: int) -> None:
        self.write(REG_INSTR_ADDR, v)

    @property
    def weight_addr(self) -> int:
        return self.read(REG_WEIGHT_ADDR)

    @weight_addr.setter
    def weight_addr(self, v: int) -> None:
        self.write(REG_WEIGHT_ADDR, v)

    @property
    def input_addr(self) -> int:
        return self.read(REG_INPUT_ADDR)

    @input_addr.setter
    def input_addr(self, v: int) -> None:
        self.write(REG_INPUT_ADDR, v)

    @property
    def output_addr(self) -> int:
        return self.read(REG_OUTPUT_ADDR)

    @output_addr.setter
    def output_addr(self, v: int) -> None:
        self.write(REG_OUTPUT_ADDR, v)

    @property
    def workspace_addr(self) -> int:
        return self.read(REG_WORKSPACE_ADDR)

    @workspace_addr.setter
    def workspace_addr(self, v: int) -> None:
        self.write(REG_WORKSPACE_ADDR, v)
