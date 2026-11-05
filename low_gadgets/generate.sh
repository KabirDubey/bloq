#!/usr/bin/env bash
# Compile the Low et al. gadgets (arXiv:2605.30455) that bloq builds from BLOG inputs, then run the shared check.
# Usage: low_gadgets/generate.sh [gadget-prefix ...]   (default: every inputs/*.blog)
# Needs `cargo build --release -p bloq-cli` first. Output: out/<gadget>_<variant>_d<d>.{stim,check.json}.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bloq=${BLOQ:-$here/../target/release/bloq}
py=${CHECK_PY:-$HOME/Projects/dev/tqec_dev/tqec/.venv/bin/python}
check=${CHECK:-$HOME/Projects/dev/tqec_dev/docs/research/Nov0426-low-gadgets-dx/check.py}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}
prefixes=("${@:-}")
for blog in "$here"/inputs/*.blog; do
  name=$(basename "$blog" .blog)
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  for d in 3 5; do
    out="$here/out/${name}_d$d"
    "$bloq" compile "$blog" -d "$d" -o "$out.stim" -q
    # Rounds: every input here is one cube of height=2d.
    "$py" "$check" "$out.stim" --d "$d" --rounds $((2 * d)) --json "$out.check.json" > /dev/null
    "$py" -c "import json,sys; j=json.load(open(sys.argv[1])); print(sys.argv[1], 'pass' if j['pass'] else 'FAIL', j['distance'])" "$out.check.json"
  done
done

# G1 onward: circuits built from bloq primitives (L3) by the low-gadgets package in rust/.
# Build: (cd rust && CARGO_TARGET_DIR=../../target cargo build --release --offline). 2d rounds each.
# This rewrites the .check.json without sinter results; rerun check.py with --sinter for those.
gen=${GEN:-$here/../target/release/low-gadgets}
for name in g1_memory_xtop_z g1_memory_xtop_x g2_memory_ztop_z g2_memory_ztop_x; do
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  basis=${name##*_}
  top=x-top; [[ $name == g2_* ]] && top=z-top
  for d in 3 5; do
    out="$here/out/${name}_d$d"
    "$gen" compact-memory "$top" "$basis" "$d" $((2 * d)) "$out.stim"
    "$py" "$check" "$out.stim" --d "$d" --rounds $((2 * d)) --json "$out.check.json" > /dev/null
    "$py" -c "import json,sys; j=json.load(open(sys.argv[1])); print(sys.argv[1], 'pass' if j['pass'] else 'FAIL', j['distance'])" "$out.check.json"
  done
done

# G3: transversal Hadamard (Fig. 7): d Z-top rounds (the last with the square end cycle), H, d X-top rounds.
for basis in z x; do
  name=g3_hadamard_$basis
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  for d in 3 5; do
    out="$here/out/${name}_d$d"
    "$gen" hadamard "$basis" "$d" "$d" "$d" "$out.stim"
    "$py" "$check" "$out.stim" --d "$d" --rounds $((2 * d)) --json "$out.check.json" > /dev/null
    "$py" -c "import json,sys; j=json.load(open(sys.argv[1])); print(sys.argv[1], 'pass' if j['pass'] else 'FAIL', j['distance'])" "$out.check.json"
  done
done

# G4: patch walking (Fig. 48), one lattice step right per 2 rounds; 2d rounds, so d steps.
for name in g4_walking_xtop_z g4_walking_xtop_x g4_walking_ztop_z g4_walking_ztop_x; do
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  basis=${name##*_}
  top=x-top; [[ $name == *ztop* ]] && top=z-top
  for d in 3 5; do
    out="$here/out/${name}_d$d"
    "$gen" walking "$top" "$basis" "$d" $((2 * d)) "$out.stim"
    "$py" "$check" "$out.stim" --d "$d" --rounds $((2 * d)) --json "$out.check.json" > /dev/null
    "$py" -c "import json,sys; j=json.load(open(sys.argv[1])); print(sys.argv[1], 'pass' if j['pass'] else 'FAIL', j['distance'])" "$out.check.json"
  done
done

# G5: lattice-surgery merge and split (Figs. 50, 51): two patches, merged along a horizontal (h) or
# vertical (v) seam, then split. Rounds: 2 separate, 2 + m merged, 2 split, 2 separate; m = d - 2,
# so d merged rounds (arXiv:2605.30455 section 1.2; A11, A15).
for top in x-top z-top; do for o in h v; do for basis in z x; do
  name=g5_surgery_${top/-/}_${o}_${basis}
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  for d in 3 5; do
    out="$here/out/${name}_d$d"
    m=$(( d - 2 ))
    "$gen" surgery "$top" "$o" "$basis" "$d" "$out.stim"
    "$py" "$check" "$out.stim" --d "$d" --rounds $((8 + m)) --json "$out.check.json" > /dev/null
    "$py" -c "import json,sys; j=json.load(open(sys.argv[1])); print(sys.argv[1], 'pass' if j['pass'] else 'FAIL', j['distance'])" "$out.check.json"
  done
done; done; done

# G6: H-bridge benchmark (Fig. 9): the G5 merge with k bridge patches between the two patches.
# k = 0 is G5. At odd k d the bridge is k d + 1 lattice units wide (A17). Same d + 6 rounds.
for k in 1 2 3; do for top in x-top z-top; do for o in h v; do for basis in z x; do
  name=g6_hbridge_${top/-/}_${o}_${basis}_k$k
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  for d in 3 5; do
    out="$here/out/${name}_d$d"
    "$gen" h-bridge "$top" "$o" "$basis" "$d" "$k" "$out.stim"
    "$py" "$check" "$out.stim" --d "$d" --rounds $((d + 6)) --json "$out.check.json" > /dev/null
    "$py" -c "import json,sys; j=json.load(open(sys.argv[1])); print(sys.argv[1], 'pass' if j['pass'] else 'FAIL', j['distance'])" "$out.check.json"
  done
done; done; done; done
