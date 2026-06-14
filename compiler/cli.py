#!/usr/bin/env python3
"""
NN Accelerator Compiler — standalone entry point for PyInstaller.

Usage:
    nn_compiler.exe model.onnx -o model.bin
    nn_compiler.exe model.onnx -o model.bin --name my_model -v
"""

import sys
from compiler.main import main

if __name__ == "__main__":
    sys.exit(main())
