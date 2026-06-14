"""
BIN file writer — serialises the compiler output to the binary format
defined in ``nn_bin.h``.

Layout
------
+------------------+ offset 0
| Header   (256 B) |
+------------------+
| Instr[0] (64 B)  |
| ...               |
| Instr[N-1]        |
+------------------+
| Weight buffer     |
| (all weights +    |
|  extra packed     |
|  weights)         |
+------------------+
"""

from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path
from typing import Optional

from .codegen import Instruction, CodegenResult
from .ir import Graph, Tensor
from .isa import INSTRUCTION_SIZE

logger = logging.getLogger(__name__)

# ── BIN format constants ────────────────────────────────────────────
NN_BIN_MAGIC = 0x31424E4E   # "NNB1"
HEADER_SIZE = 256
VERSION_MAJOR = 1
VERSION_MINOR = 0

# struct format for the header (little-endian)
#   I = uint32, H = uint16, 64s = char[64], 156s = uint8[156]
_HEADER_FMT = "<I H H I I I I I I 64s 160s"
assert struct.calcsize(_HEADER_FMT) == HEADER_SIZE, \
    f"Header struct size mismatch: {struct.calcsize(_HEADER_FMT)} != {HEADER_SIZE}"

# Instruction format (little-endian):
#   2B opcode:flags:seq_len (we pack them into 2x uint8 + uint16 then struct)
# Actually easier: pack each field individually then combine
#   B=uint8, H=uint16, I=uint32
_INSTR_FMT = "<B B H I I I I I I I I I I I I I I I"


def write_bin(
    graph: Graph,
    codegen_result: CodegenResult,
    output_path: str,
    model_name: str = "",
) -> None:
    """Write the complete BIN file.

    Parameters
    ----------
    graph:
        The IR graph (memory planner must have already run).
    codegen_result:
        Output from ``codegen.generate()``.
    output_path:
        Destination file path.
    model_name:
        Human-readable model name (max 63 chars, null-terminated).
    """
    instructions = codegen_result.instructions
    extra_weights = codegen_result.extra_weights

    # ── Assign offsets to extra weights (GRU packed weights, etc.) ─
    _assign_extra_weight_offsets(graph, codegen_result)

    # ── Compute sizes ────────────────────────────────────────────
    num_instr = len(instructions)
    instr_bytes = num_instr * INSTRUCTION_SIZE

    # Collect all weight tensors in serialisation order
    all_weights = _collect_all_weights(graph, extra_weights)
    weight_bytes = _total_weight_size(all_weights)

    # Build the weight blob
    weight_blob = _serialize_weights(all_weights)

    # Build the instruction blob
    instr_blob = _serialize_instructions(instructions)

    # ── Checksum (CRC32 of instructions + weights) ────────────────
    checksum = zlib.crc32(instr_blob + weight_blob) & 0xFFFFFFFF

    # ── Compute model input / output sizes ───────────────────────
    input_size = sum(t.size_bytes for t in graph.inputs)
    output_size = sum(t.size_bytes for t in graph.outputs)

    # ── Workspace size (may have been extended by GRU scratch) ───
    workspace_size = max(graph.workspace_size, codegen_result.workspace_size)

    # ── Pack header ──────────────────────────────────────────────
    name_bytes = model_name.encode("utf-8", errors="replace")[:63]
    name_field = name_bytes.ljust(64, b"\x00")

    header = struct.pack(
        _HEADER_FMT,
        NN_BIN_MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        num_instr,
        weight_bytes,
        workspace_size,
        input_size,
        output_size,
        checksum,
        name_field,
        b"\x00" * 160,  # reserved
    )

    # ── Write ────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(instr_blob)
        f.write(weight_blob)

    file_size = output_path.stat().st_size
    logger.info(
        "Wrote BIN: %s (%d B: %d header + %d instrs + %d weights, "
        "workspace=%d KiB, checksum=0x%08X)",
        output_path, file_size, HEADER_SIZE, num_instr,
        weight_bytes, workspace_size // 1024, checksum,
    )


# ════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════

def _assign_extra_weight_offsets(graph: Graph, result: CodegenResult) -> None:
    """Assign byte offsets to any extra weight tensors (e.g. GRU packed
    weights).  These are appended after the existing weight buffer.

    Also updates the corresponding instruction address fields.
    """
    # Calculate where existing weights end
    existing_end = graph.weight_size

    offset = existing_end
    for instr in result.instructions:
        if hasattr(instr, '_gru_weight') and instr._gru_weight is not None:
            wt: Tensor = instr._gru_weight
            wt.offset = offset
            instr.weight_addr = offset
            offset += wt.size_bytes

        if hasattr(instr, '_gru_bias') and instr._gru_bias is not None:
            bt: Tensor = instr._gru_bias
            bt.offset = offset
            instr.bias_addr = offset
            offset += bt.size_bytes

    # Update graph weight_size to include extras
    graph.weight_size = offset


def _collect_all_weights(
    graph: Graph,
    extra_weights: list[Tensor],
) -> list[Tensor]:
    """Collect all weight tensors in the order they should be serialised."""
    result: list[Tensor] = []
    seen: set[str] = set()

    # Existing initializers in first-reference order
    for node in graph.nodes:
        for t in node.inputs:
            if t.is_constant and t.name not in seen:
                seen.add(t.name)
                result.append(t)

    # Extra weights (GRU packed, etc.)
    for t in extra_weights:
        if t.name not in seen:
            seen.add(t.name)
            result.append(t)

    return result


def _total_weight_size(weights: list[Tensor]) -> int:
    return sum(t.size_bytes for t in weights)


def _serialize_weights(weights: list[Tensor]) -> bytes:
    """Concatenate all weight tensor data into a single byte string.

    Each weight tensor is written at its assigned offset (relative to
    the start of the weight buffer).  Unused gaps are zero-filled.
    """
    if not weights:
        return b""

    total = _total_weight_size(weights)
    buf = bytearray(total)

    for t in weights:
        if t.data is None:
            continue
        if t.offset is None:
            raise ValueError(f"Weight tensor '{t.name}' has no offset assigned")
        data = t.data.astype("float32").tobytes()
        buf[t.offset : t.offset + len(data)] = data

    return bytes(buf)


def _serialize_instructions(instructions: list[Instruction]) -> bytes:
    """Pack all instructions into a contiguous byte buffer."""
    buf = bytearray()

    for instr in instructions:
        # Remove private attributes before serialisation
        fields = instr.as_tuple()
        packed = struct.pack(_INSTR_FMT, *fields)
        # Pad to 64 bytes if needed
        packed = packed.ljust(INSTRUCTION_SIZE, b"\x00")
        buf.extend(packed)

    return bytes(buf)
