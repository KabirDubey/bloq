#!/usr/bin/env bash
# Track B (L2): the compact X-top memory of arXiv:2605.30455 (Sec. 1, Fig. 5(c)) through bloq's own
# compile pipeline, using the experimental cube template behind BLOQ_EXPERIMENTAL_LOW_COMPACT=1
# (bloq_compile/src/block/fixed_bulk/low_compact.rs). Checks each circuit with check.py at d=3 and d=5.
set -euo pipefail
cd "$(dirname "$0")"
BLOQ=${BLOQ:-../../target/release/bloq}
CHECK_PY=${CHECK_PY:-$HOME/Projects/dev/tqec_dev/tqec/.venv/bin/python}
CHECK=${CHECK:-$HOME/Projects/dev/tqec_dev/docs/research/Nov0426-low-gadgets-dx/check.py}
mkdir -p out/l2
for b in z x; do
  for d in 3 5; do
    name=l2_compact_memory_${b}_d$d
    BLOQ_EXPERIMENTAL_LOW_COMPACT=1 "$BLOQ" compile ../inputs/g0_memory_$b.blog -d "$d" -o out/l2/$name.stim -q --align-moments
    "$CHECK_PY" stats.py "$CHECK" out/l2/$name.stim
    "$CHECK_PY" "$CHECK" out/l2/$name.stim --d "$d" --rounds $((2 * d)) --json out/l2/$name.check.json | grep '"pass"'
  done
done
