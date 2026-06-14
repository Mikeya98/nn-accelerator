"""Allow running the compiler via ``python -m compiler``."""

from .main import main
import sys

sys.exit(main())
