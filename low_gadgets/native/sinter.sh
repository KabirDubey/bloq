#!/usr/bin/env bash
# Sinter (pymatching, check.py's noise model) at d = 5 for the Track A surgery gadgets: G5, G6 (k = 1), G7.
# The retry rules cap each circuit at 10 min of CPU; `time` prints it. Output: native/out/<name>.sinter.json.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=${CHECK_PY:-$HOME/Projects/dev/tqec_dev/tqec/.venv/bin/python}
check=${CHECK:-$HOME/Projects/dev/tqec_dev/docs/research/Nov0426-low-gadgets-dx/check.py}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}
for name in g5n_surgery_zx_h_x_d5 g5n_surgery_zx_h_z_d5 g6n_bridge_zx_h_x_k1_d5 g6n_bridge_zx_h_z_k1_d5 \
            g7n_cx_zx_d5-fill0 g7n_cx_zx_d5-fill1; do
  stim=$here/out/$name.stim
  rounds=$("$py" "$here/stats.py" "$check" "$stim" | "$py" -c "import json,sys; print(json.load(sys.stdin)['rounds'])")
  echo "== $name (rounds $rounds)"
  time "$py" "$check" "$stim" --d 5 --rounds "$rounds" --sinter --json "$here/out/$name.sinter.json" | tail -3
done
