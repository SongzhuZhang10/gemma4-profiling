#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/edge_llm_env.sh"

require_python_env
ensure_workflow_dirs

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  bash 13_target_inspect_ncu.sh --phase decode
  bash 13_target_inspect_ncu.sh --phase decode --iteration 64 --view kernels --family gemm_like --limit 15
  bash 13_target_inspect_ncu.sh --report artifacts/reports/decode_iter064.ncu-rep --view launches --sort mem_pct --limit 20

Notes:
  - The first query for a report builds a JSON cache beside the .ncu-rep file.
  - Later queries reuse that cache, so they are much faster than reopening the GUI.
EOF
  exit 0
fi

"$PYTHON_BIN" "$WORKFLOW_ROOT/edge_llm_workflow.py" inspect-ncu "$@"
