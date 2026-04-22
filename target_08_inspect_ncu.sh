#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/workflow_env.sh"

require_python_env
ensure_workflow_dirs

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  bash target_08_inspect_ncu.sh --phase decode
  bash target_08_inspect_ncu.sh --phase decode --view kernels --family gemm_like --limit 15
  bash target_08_inspect_ncu.sh --report artifacts/reports/decode.ncu-rep --view launches --sort dram_pct --limit 20

Notes:
  - The first query for a report builds a JSON cache beside the .ncu-rep file.
  - Later queries reuse that cache, so they are much faster than reopening the GUI.
EOF
  exit 0
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/profiling_workflow.py" inspect-ncu "$@"
