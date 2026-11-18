"""Structure of a stim circuit after check.py's drop_empty_moments: qubits used, detectors, observables, moments,
two-qubit gates, and rounds (moments that hold a measurement).

    python stats.py CHECK_PY CIRCUIT.stim ...    # one JSON line per circuit
"""

import importlib.util
import json
import sys

import stim

spec = importlib.util.spec_from_file_location("check", sys.argv[1])
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
for path in sys.argv[2:]:
    c = check.drop_empty_moments(stim.Circuit.from_file(path))
    used, two, moments, rounds, measured = set(), 0, 1, 0, False
    for inst in c:
        if inst.name == "TICK":
            moments, rounds, measured = moments + 1, rounds + measured, False
            continue
        gd = stim.gate_data(inst.name)
        if gd.is_unitary or gd.produces_measurements or gd.is_reset:
            used.update(t.value for t in inst.targets_copy() if t.is_qubit_target)
        measured |= gd.produces_measurements
        if gd.is_two_qubit_gate:
            two += len(inst.targets_copy()) // 2
    rounds += measured
    print(json.dumps(dict(circuit=path.rsplit("/", 1)[-1], qubits=len(used), detectors=c.num_detectors,
                          observables=c.num_observables, moments=moments, two_qubit=two, rounds=rounds)))
