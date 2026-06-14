#!/usr/bin/env python3
"""
NN Accelerator Compiler — ONNX → BIN

Usage:
    python -m compiler model.onnx -o model.bin
    python -m compiler model.onnx -o model.bin --name my_model --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .parser import parse_onnx
from .memory_planner import plan_memory
from .codegen import generate, CodegenResult
from .bin_writer import write_bin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="NN Accelerator Compiler — compile ONNX to BIN",
    )
    ap.add_argument(
        "model", type=str,
        help="Path to the ONNX model file",
    )
    ap.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output BIN file path (default: <model_name>.bin)",
    )
    ap.add_argument(
        "--name", type=str, default=None,
        help="Model name stored in the BIN header (default: derived from filename)",
    )
    ap.add_argument(
        "--input-shape", type=str, nargs="*", default=None,
        help="Override input shapes, e.g. --input-shape x:1,3,224,224",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = ap.parse_args(argv)

    # ── Logging ──────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)s  %(message)s",
    )
    logger = logging.getLogger("compiler")

    # ── Resolve paths ────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("ONNX file not found: %s", model_path)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = model_path.with_suffix(".bin").name

    model_name = args.name or model_path.stem

    # ── Parse input shape overrides ──────────────────────────────
    input_shapes = None
    if args.input_shape:
        input_shapes = {}
        for spec in args.input_shape:
            name, shape_str = spec.split(":", 1)
            shape = tuple(int(d) for d in shape_str.split(","))
            input_shapes[name] = shape

    # ── Compile pipeline ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("NN Accelerator Compiler")
    logger.info("  Model:  %s", model_path)
    logger.info("  Output: %s", output_path)
    logger.info("=" * 60)

    # 1. Parse ONNX → IR
    logger.info("Step 1/4: Parsing ONNX model ...")
    try:
        graph = parse_onnx(str(model_path), input_shapes)
    except Exception as e:
        logger.error("Failed to parse ONNX: %s", e)
        if args.verbose:
            raise
        return 1

    logger.info("  Nodes: %d  Initializers: %d  Inputs: %d  Outputs: %d",
                len(graph.nodes), len(graph.initializers),
                len(graph.inputs), len(graph.outputs))

    # 2. Memory planning
    logger.info("Step 2/4: Planning memory layout ...")
    try:
        plan_memory(graph)
    except Exception as e:
        logger.error("Memory planning failed: %s", e)
        if args.verbose:
            raise
        return 1

    logger.info("  Workspace: %d KiB  Weights: %d KiB",
                graph.workspace_size // 1024, graph.weight_size // 1024)

    # 3. Code generation
    logger.info("Step 3/4: Generating instructions ...")
    try:
        codegen_result = generate(graph)
    except Exception as e:
        logger.error("Code generation failed: %s", e)
        if args.verbose:
            raise
        return 1

    logger.info("  Instructions: %d  Extra weights: %d",
                len(codegen_result.instructions), len(codegen_result.extra_weights))

    # 4. Write BIN
    logger.info("Step 4/4: Writing BIN file ...")
    try:
        write_bin(graph, codegen_result, output_path, model_name)
    except Exception as e:
        logger.error("Failed to write BIN: %s", e)
        if args.verbose:
            raise
        return 1

    logger.info("=" * 60)
    logger.info("Done! Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
