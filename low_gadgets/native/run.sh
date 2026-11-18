#!/usr/bin/env bash
# Track A (L1): compile the native BLOG inputs with bloq, check them, count their structure.
# Usage: low_gadgets/native/run.sh [prefix ...]   Needs `cargo build --release -p bloq-cli`.
# Output: native/out/<name>_d<d>[-fill<n>].{stim,check.json}, and one line per circuit in native/out/summary.jsonl.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bloq=${BLOQ:-$here/../../target/release/bloq}
py=${CHECK_PY:-$HOME/Projects/dev/tqec_dev/tqec/.venv/bin/python}
check=${CHECK:-$HOME/Projects/dev/tqec_dev/docs/research/Nov0426-low-gadgets-dx/check.py}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}
mkdir -p "$here/out"
prefixes=("${@:-}")
for blog in "$here"/inputs/*.blog; do
  name=$(basename "$blog" .blog)
  if [[ -n "${prefixes[0]}" ]]; then
    match=0; for p in "${prefixes[@]}"; do [[ $name == "$p"* ]] && match=1; done
    (( match )) || continue
  fi
  fill=(); [[ $name == g7n_* ]] && fill=(--auto-fill)
  for d in 3 5; do
    rm -f "$here/out/${name}_d$d"*.stim
    "$bloq" compile "$blog" -d "$d" -o "$here/out/${name}_d$d.stim" -q --align-moments ${fill[@]+"${fill[@]}"}
    for stim in "$here/out/${name}_d$d"*.stim; do
      base=${stim%.stim}
      stats=$("$py" "$here/stats.py" "$check" "$stim")
      rounds=$("$py" -c "import json,sys; print(json.loads(sys.argv[1])['rounds'])" "$stats")
      "$py" "$check" "$stim" --d "$d" --rounds "$rounds" --json "$base.check.json" > /dev/null || true
      "$py" - "$base.check.json" "$stats" "$d" <<'PY' | tee -a "$here/out/summary.jsonl"
import json, sys
j, s = json.load(open(sys.argv[1])), json.loads(sys.argv[2])
s.update(d=int(sys.argv[3]), check="pass" if j["pass"] else "FAIL", distance=j["distance"])
print(json.dumps(s))
PY
    done
  done
done
