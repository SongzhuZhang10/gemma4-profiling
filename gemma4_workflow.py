#!/usr/bin/env python3
"""Compatibility shim for the renamed profiling workflow module."""

from __future__ import annotations

import sys

from profiling_workflow import main


if __name__ == "__main__":
    print(
        "gemma4_workflow.py is deprecated; use profiling_workflow.py instead.",
        file=sys.stderr,
    )
    sys.exit(main())
