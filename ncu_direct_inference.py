#!/usr/bin/env python3
"""Legacy stub for the retired direct Hugging Face Nsight Compute path."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "ncu_direct_inference.py is no longer used by this repository. "
        "The workflow now profiles TensorRT Edge-LLM runtime execution only.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
