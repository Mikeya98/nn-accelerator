"""
Instruction executor — fetch / decode / dispatch loop.

Mimics the PL hardware: reads instructions from DDR, decodes opcodes,
and dispatches to the corresponding FP32 compute functions.
"""

from __future__ import annotations

import logging

from .memory import Memory, RegisterFile, STATUS_IDLE, STATUS_DONE
from .loader import LoadedBinary, Instruction
from . import ops
from compiler.isa import (
    OP_NOP, OP_FC, OP_POOL, OP_RELU, OP_SIGMOID, OP_TANH,
    OP_GRU, OP_ELEM_MUL, OP_ELEM_ADD, OP_END,
    OPCODE_NAMES, INSTRUCTION_SIZE,
)

logger = logging.getLogger(__name__)


def execute(
    mem: Memory,
    regs: RegisterFile,
    binary: LoadedBinary,
) -> None:
    """Run the full instruction sequence.

    This is the hardware-equivalent entry point:

    1. Load instructions + weights into DDR at the configured addresses.
    2. Set STATUS = BUSY (if a busy flag is used).
    3. Fetch & execute instructions until OP_END.
    4. Set STATUS = DONE.

    Parameters
    ----------
    mem:
        Pre-configured DDR model (weights + input already placed).
    regs:
        Pre-configured register file (address registers set by PS).
    binary:
        The loaded BIN file.
    """
    base_instr = regs.instr_addr
    base_weight = regs.weight_addr
    base_workspace = regs.workspace_addr

    # ── Load weight blob into DDR ─────────────────────────────────
    if binary.weight_blob:
        mem.write(base_weight, binary.weight_blob)

    logger.info(
        "Executor start: %d instructions, weight_base=%#x, ws_base=%#x",
        len(binary.instructions), base_weight, base_workspace,
    )

    # ── Instruction loop ──────────────────────────────────────────
    pc = 0  # program counter (instruction index)
    total = len(binary.instructions)

    while pc < total:
        instr = binary.instructions[pc]

        if instr.opcode == OP_END:
            logger.debug("[%3d] END → stop", pc)
            break

        elif instr.opcode == OP_NOP:
            logger.debug("[%3d] NOP", pc)
            pc += 1
            continue

        elif instr.opcode == OP_FC:
            logger.debug(
                "[%3d] FC   M=%d N=%d  in0=%#x w=%#x b=%#x out=%#x",
                pc, instr.dim0, instr.dim1,
                instr.input0_addr, instr.weight_addr,
                instr.bias_addr, instr.output_addr,
            )
            ops.op_fc(mem, instr, base_weight, base_workspace)
            pc += 1

        elif instr.opcode == OP_RELU:
            logger.debug("[%3d] RELU n=%d", pc, instr.dim0)
            ops.op_relu(mem, instr, base_workspace)
            pc += 1

        elif instr.opcode == OP_SIGMOID:
            logger.debug("[%3d] SIGM n=%d", pc, instr.dim0)
            ops.op_sigmoid(mem, instr, base_workspace)
            pc += 1

        elif instr.opcode == OP_TANH:
            logger.debug("[%3d] TANH n=%d", pc, instr.dim0)
            ops.op_tanh(mem, instr, base_workspace)
            pc += 1

        elif instr.opcode == OP_POOL:
            logger.debug(
                "[%3d] POOL %s H=%d W=%d C=%d KH=%d KW=%d S=%d",
                pc,
                "MAX" if instr.flags & 0x02 else "AVG",
                instr.dim0, instr.dim1, instr.dim2,
                instr.dim3, instr.dim4, instr.dim5,
            )
            ops.op_pool(mem, instr, base_workspace)
            pc += 1

        elif instr.opcode == OP_ELEM_MUL:
            logger.debug("[%3d] MUL  n=%d", pc, instr.dim0)
            ops.op_elem_mul(mem, instr, base_workspace)
            pc += 1

        elif instr.opcode == OP_ELEM_ADD:
            logger.debug("[%3d] ADD  n=%d", pc, instr.dim0)
            ops.op_elem_add(mem, instr, base_workspace)
            pc += 1

        elif instr.opcode == OP_GRU:
            logger.debug(
                "[%3d] GRU  I=%d H=%d batch=%d seq=%d",
                pc, instr.dim0, instr.dim1, instr.dim2, instr.seq_len,
            )
            ops.op_gru(mem, instr, base_weight, base_workspace)
            pc += 1

        else:
            name = OPCODE_NAMES.get(instr.opcode, f"0x{instr.opcode:02X}")
            raise NotImplementedError(
                f"Unsupported opcode {name} at pc={pc}"
            )

    # ── Done ──────────────────────────────────────────────────────
    regs.status = STATUS_DONE
    logger.info("Executor done at pc=%d", pc)
