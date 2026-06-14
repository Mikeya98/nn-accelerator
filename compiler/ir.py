"""
Internal Representation for the NN Compiler.

Defines the data structures that sit between ONNX parsing and code generation:
- Tensor: a named, shaped piece of data (runtime or constant)
- Node:   a compute operation with inputs, outputs, and attributes
- Graph:  the full model as a topologically-sorted DAG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Tensor:
    """A tensor in the compiler IR.

    Attributes
    ----------
    name:
        Unique name within the graph (matches ONNX value_info name).
    shape:
        Tuple of dimension sizes.  Scalar is ().  Dynamic dims are -1.
    dtype:
        ONNX element type as a string; we only accept "float32" (FP32).
    data:
        For constant tensors (weights, biases) the numpy array payload.
        Runtime tensors have ``data = None``.
    offset:
        Byte offset assigned by the memory planner.
        - Runtime tensors → offset in the *workspace* buffer.
        - Weight tensors   → offset in the *weight* buffer.
        ``None`` means not yet allocated.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    data: Optional[np.ndarray] = None
    offset: Optional[int] = None

    @property
    def num_elements(self) -> int:
        """Total number of scalar elements (product of shape dims)."""
        n = 1
        for d in self.shape:
            if d <= 0:
                return 0  # dynamic / unknown
            n *= d
        return n

    @property
    def size_bytes(self) -> int:
        """Size in bytes (FP32 = 4 bytes per element)."""
        return self.num_elements * 4

    @property
    def is_constant(self) -> bool:
        return self.data is not None

    @property
    def is_runtime(self) -> bool:
        return self.data is None


@dataclass
class Node:
    """A single operation node in the compute graph.

    Attributes
    ----------
    name:
        Original ONNX node name (or auto-generated).
    op_type:
        Normalised op type string, e.g. "FC", "Relu", "GRU", "Reshape".
    inputs:
        Input tensors (may be constant or runtime).
    outputs:
        Output tensors (at least one).
    attrs:
        String-keyed dictionary of attributes gleaned from ONNX.
    is_shape_op:
        True for Reshape / Transpose / Concat / Squeeze / Unsqueeze / Flatten.
        These are folded during code generation — they produce no PL instructions.
    """

    name: str
    op_type: str
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    is_shape_op: bool = False


@dataclass
class Graph:
    """Top-level container for the compiled model's IR.

    Attributes
    ----------
    inputs:
        Model input tensors (graph inputs in ONNX).
    outputs:
        Model output tensors (graph outputs in ONNX).
    nodes:
        All compute nodes in topological order.  Shape-only nodes are
        included here but flagged with ``is_shape_op = True``.
    initializers:
        All constant tensors keyed by name (weights, biases, …).
    workspace_size:
        Total scratchpad bytes required (set by memory planner).
    weight_size:
        Total weight buffer bytes (set by memory planner).
    """

    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    initializers: dict[str, Tensor] = field(default_factory=dict)
    workspace_size: int = 0
    weight_size: int = 0
