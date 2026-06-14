"""
Top-level simulator API — the PS driver's view.

Usage::

    from simulator import Simulator

    sim = Simulator()
    sim.load_bin("model.bin")
    sim.set_input(input_data)       # numpy array, FP32
    sim.run()
    result = sim.get_output()       # numpy array, FP32

This mirrors exactly what the PS firmware does:
    1. Load BIN from Flash to DDR
    2. Write input data to DDR
    3. Configure address registers
    4. Write CTRL = 1 to start PL
    5. Poll STATUS until DONE
    6. Read output from DDR
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .memory import Memory, RegisterFile, STATUS_DONE
from .loader import LoadedBinary, load_bin
from .executor import execute

logger = logging.getLogger(__name__)

# Default DDR layout (byte offsets within the simulated DDR)
# In real hardware the PS sets these registers to match the physical DDR map.
DEFAULT_DDR_BASE      = 0x0000_0000
DEFAULT_INSTR_BASE    = 0x0000_0000   # BIN instructions
DEFAULT_WEIGHT_BASE   = 0x0001_0000   # weight buffer (64 KiB after instrs)
DEFAULT_WORKSPACE_BASE = 0x0010_0000  # workspace (input / output / scratch)


class Simulator:
    """Instruction-level BIN simulator.

    All data buffers (weights, input, output, workspace) sit inside a
    single contiguous DDR region.  The compiler assigns byte offsets
    within that region; the simulator resolves them against the
    configured base addresses.

    Parameters
    ----------
    ddr_base:
        Base address of the simulated DDR region.
    instr_base:
        Offset within DDR where BIN instructions are placed.
    weight_base:
        Offset within DDR where the weight buffer starts.
    workspace_base:
        Offset within DDR for workspace (input / output / scratch all
        live here, at compiler-assigned offsets).
    """

    def __init__(
        self,
        ddr_base: int = 0x0000_0000,
        instr_base: int = 0x0000_0000,
        weight_base: int = 0x0001_0000,
        workspace_base: int = 0x0010_0000,
    ):
        self.mem = Memory()
        self.regs = RegisterFile()

        self.ddr_base = ddr_base
        self.instr_base = instr_base
        self.weight_base = weight_base
        self.workspace_base = workspace_base

        self._binary: Optional[LoadedBinary] = None
        self._ran: bool = False

    # ── PS-side API ──────────────────────────────────────────────

    def load_bin(self, path: str) -> None:
        """Load a compiler-generated BIN file.

        This corresponds to the PS firmware copying the BIN from Flash
        to DDR at boot time.
        """
        logger.info("Loading BIN: %s", path)
        self._binary = load_bin(path)
        self._ran = False

        hdr = self._binary.header
        logger.info(
            "  Model: %s  v%d.%d  Instrs: %d  Weights: %d B  "
            "Workspace: %d B  Input: %d B  Output: %d B",
            hdr.model_name, hdr.version_major, hdr.version_minor,
            hdr.num_instructions, hdr.weight_size,
            hdr.workspace_size, hdr.input_size, hdr.output_size,
        )

    def set_input(self, data: np.ndarray) -> None:
        """Write model input data to DDR.

        The PS driver copies input data into the workspace at the offset
        assigned by the compiler (always offset 0 for the first graph input).
        """
        data = data.astype(np.float32)
        dst = self.workspace_base  # compiler assigns input at workspace offset 0
        logger.info("Writing input: shape=%s (%d bytes) to %#x", data.shape, data.nbytes, dst)
        self.mem.write_f32(dst, data.ravel())
        self._ran = False

    def run(self, verbose: bool = False) -> None:
        """Start PL inference and wait for completion.

        PS-side sequence:
            regs.input_addr     = workspace_base
            regs.output_addr    = workspace_base
            regs.instr_addr     = instr_base
            regs.weight_addr    = weight_base
            regs.workspace_addr = workspace_base
            regs.ctrl           = 1          → kick PL
            while regs.status != DONE: pass  → poll
        """
        if self._binary is None:
            raise RuntimeError("No BIN loaded. Call load_bin() first.")

        if verbose:
            logging.getLogger("simulator.executor").setLevel(logging.DEBUG)

        # ── Configure registers ──────────────────────────────────
        # Input, output, and workspace all sit in the same region;
        # the instructions use compiler-assigned offsets within it.
        self.regs.input_addr = self.workspace_base
        self.regs.output_addr = self.workspace_base
        self.regs.instr_addr = self.instr_base
        self.regs.weight_addr = self.weight_base
        self.regs.workspace_addr = self.workspace_base

        logger.info("Kicking PL (ctrl=1) …")
        self.regs.ctrl = 1

        # ── Execute ──────────────────────────────────────────────
        execute(self.mem, self.regs, self._binary)

        assert self.regs.status == STATUS_DONE, "PL did not signal DONE!"
        self._ran = True
        logger.info("Inference complete.")

    def get_output(self) -> np.ndarray:
        """Read model output from DDR.

        Returns the output as a flat FP32 numpy array.  The caller
        is responsible for reshaping according to the model spec.
        """
        if not self._ran:
            raise RuntimeError("No inference run yet. Call run() first.")
        if self._binary is None:
            raise RuntimeError("No BIN loaded.")

        output_size = self._binary.header.output_size
        output_floats = output_size // 4
        # Output is at the compiler-assigned workspace offset.
        # We need to find the output_addr from the last compute instruction.
        # Walk backwards through instructions to find the output address.
        output_addr = self.workspace_base  # fallback
        for instr in reversed(self._binary.instructions):
            if instr.opcode not in (0x00, 0xFF):  # not NOP or END
                output_addr = self.workspace_base + instr.output_addr
                break

        data = self.mem.read_f32(output_addr, output_floats)
        logger.info("Read output: %d floats (%d bytes) from %#x", len(data), output_size, output_addr)
        return data
