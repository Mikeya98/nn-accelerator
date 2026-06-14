"""
BIN file loader — reads the compiler-generated binary into simulator data
structures.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from compiler.isa import INSTRUCTION_SIZE, OPCODE_NAMES

# ── BIN format constants (must match nn_bin.h) ────────────────────────
NN_BIN_MAGIC  = 0x31424E4E
HEADER_SIZE   = 256
HEADER_FMT    = "<I H H I I I I I I 64s 160s"
INSTR_FMT     = "<B B H I I I I I I I I I I I I I I I"

# C struct field-order assertion: above format must be 256 B.
assert struct.calcsize(HEADER_FMT) == HEADER_SIZE


@dataclass
class BinHeader:
    """Parsed BIN header fields."""
    magic: int = 0
    version_major: int = 0
    version_minor: int = 0
    num_instructions: int = 0
    weight_size: int = 0
    workspace_size: int = 0
    input_size: int = 0
    output_size: int = 0
    checksum: int = 0
    model_name: str = ""


@dataclass
class Instruction:
    """A single 64-byte instruction decoded from BIN."""
    opcode: int = 0
    flags: int = 0
    seq_len: int = 0
    input0_addr: int = 0
    input1_addr: int = 0
    output_addr: int = 0
    weight_addr: int = 0
    bias_addr: int = 0
    workspace_addr: int = 0
    scale_addr: int = 0
    dim0: int = 0
    dim1: int = 0
    dim2: int = 0
    dim3: int = 0
    dim4: int = 0
    dim5: int = 0
    dim6: int = 0
    dim7: int = 0

    def __repr__(self) -> str:
        name = OPCODE_NAMES.get(self.opcode, f"OP_{self.opcode:#04x}")
        return (
            f"Instr({name}, flags={self.flags:#04x}, "
            f"in0={self.input0_addr:#x}, in1={self.input1_addr:#x}, "
            f"out={self.output_addr:#x}, w={self.weight_addr:#x}, "
            f"b={self.bias_addr:#x}, ws={self.workspace_addr:#x}, "
            f"dims=[{self.dim0},{self.dim1},{self.dim2},{self.dim3},"
            f"{self.dim4},{self.dim5},{self.dim6},{self.dim7}])"
        )


@dataclass
class LoadedBinary:
    """A parsed BIN file ready for execution.

    Attributes
    ----------
    header:
        Parsed header.
    instructions:
        List of instructions (including the terminating END).
    weight_blob:
        Raw weight buffer bytes (to be loaded into DDR).
    """

    header: BinHeader = field(default_factory=BinHeader)
    instructions: list[Instruction] = field(default_factory=list)
    weight_blob: bytes = b""


def load_bin(path: str) -> LoadedBinary:
    """Load a BIN file from disk and return a ``LoadedBinary``.

    Raises
    ------
    ValueError:
        If magic is wrong or the file is malformed.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        raise ValueError(f"BIN file too small: {len(data)} bytes (need ≥ {HEADER_SIZE})")

    lb = LoadedBinary()

    # ── Parse header ───────────────────────────────────────────────
    fields = struct.unpack_from(HEADER_FMT, data, 0)
    hdr = BinHeader(
        magic=fields[0],
        version_major=fields[1],
        version_minor=fields[2],
        num_instructions=fields[3],
        weight_size=fields[4],
        workspace_size=fields[5],
        input_size=fields[6],
        output_size=fields[7],
        checksum=fields[8],
        model_name=fields[9].rstrip(b"\x00").decode("utf-8", errors="replace"),
    )
    lb.header = hdr

    if hdr.magic != NN_BIN_MAGIC:
        raise ValueError(
            f"Bad magic: {hdr.magic:#010x}, expected {NN_BIN_MAGIC:#010x}. "
            f"Is this a valid NNB1 file?"
        )

    # ── Parse instructions ─────────────────────────────────────────
    instr_offset = HEADER_SIZE
    for i in range(hdr.num_instructions):
        off = instr_offset + i * INSTRUCTION_SIZE
        if off + INSTRUCTION_SIZE > len(data):
            raise ValueError(f"Instruction {i} truncated at offset {off}")
        fields = struct.unpack_from(INSTR_FMT, data, off)
        instr = Instruction(
            opcode=fields[0],
            flags=fields[1],
            seq_len=fields[2],
            input0_addr=fields[3],
            input1_addr=fields[4],
            output_addr=fields[5],
            weight_addr=fields[6],
            bias_addr=fields[7],
            workspace_addr=fields[8],
            scale_addr=fields[9],
            dim0=fields[10],
            dim1=fields[11],
            dim2=fields[12],
            dim3=fields[13],
            dim4=fields[14],
            dim5=fields[15],
            dim6=fields[16],
            dim7=fields[17],
        )
        lb.instructions.append(instr)

    # ── Extract weight blob ────────────────────────────────────────
    weights_start = instr_offset + hdr.num_instructions * INSTRUCTION_SIZE
    weights_end = weights_start + hdr.weight_size
    if weights_end > len(data):
        raise ValueError(
            f"Weight section truncated: need {weights_end} bytes, have {len(data)}"
        )
    lb.weight_blob = data[weights_start:weights_end]

    return lb
