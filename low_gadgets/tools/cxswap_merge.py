"""Merge each CX(t,c) at tick k-1 followed by CX(c,t) at tick k into one CXSWAP c t (the paper's native gate,
arXiv:2605.30455 Fig. 48), then estimate the per-round logical rate under the shared check's noise model plus
DEPOLARIZE2(p) after CXSWAP (stim + pymatching, at least 300 errors or 2M shots, p = 1e-3).
The merge is an exact identity (each qubit is used once per tick), so the circuit's detectors are unchanged.
Usage: cxswap_merge.py <circuit.stim> <rounds> <path/to/check.py> [--asis]"""
import sys, importlib.util, stim, sinter
spec = importlib.util.spec_from_file_location("check", sys.argv[3])
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
c = stim.Circuit.from_file(sys.argv[1]).flattened()
moments, cur = [], []
for ins in c:
    if ins.name == "TICK": moments.append(cur); cur = []
    else: cur.append(ins)
moments.append(cur)
def pairs(ins): t = [q.value for q in ins.targets_copy()]; return list(zip(t[::2], t[1::2]))
merged = 0
if "--asis" not in sys.argv:
    for k in range(1, len(moments)):
        prev = {(a, b) for ins in moments[k-1] if ins.name == "CX" for a, b in pairs(ins)}
        new = []
        for ins in moments[k]:
            if ins.name != "CX": new.append(ins); continue
            keep, sw = [], []
            for a, b in pairs(ins):
                (sw if (b, a) in prev else keep).append((a, b))
            for a, b in sw: prev.discard((b, a))
            merged += len(sw)
            if keep: new.append(stim.CircuitInstruction("CX", [x for p in keep for x in p]))
            if sw: new.append(stim.CircuitInstruction("CXSWAP", [x for p in sw for x in p]))
        # rebuild k-1 without the merged CX
        removed = {(b, a) for ins in new if ins.name == "CXSWAP" for a, b in pairs(ins)}
        np = []
        for ins in moments[k-1]:
            if ins.name == "CX":
                ps = [p for p in pairs(ins) if p not in removed]
                if ps: np.append(stim.CircuitInstruction("CX", [x for p in ps for x in p]))
            else: np.append(ins)
        moments[k-1], moments[k] = np, new
out = stim.Circuit()
for i, m in enumerate(moments):
    for ins in m: out.append(ins)
    if i < len(moments) - 1: out.append("TICK")
orig_noisy = check.noisy
def noisy(circ, p):
    from tqec.utils.noise_model import NoiseModel, NoiseRule
    import tqec.utils.noise_model as nm
    nm.OP_TYPES.setdefault("CXSWAP", nm.OP_TYPES["CX"])
    real = nm.NoiseModel.uniform_depolarizing
    def ud(pp):
        m = real(pp); m.gate_rules["CXSWAP"] = NoiseRule(after={"DEPOLARIZE2": pp}); return m
    nm.NoiseModel.uniform_depolarizing = staticmethod(ud)
    try: return orig_noisy(circ, p)
    finally: nm.NoiseModel.uniform_depolarizing = staticmethod(real)
check.noisy = noisy
n = check.noisy(out, 1e-3)
n.detector_error_model(decompose_errors=True)  # determinism
ticks = check.drop_empty_moments(out).num_ticks
import pymatching, numpy as np
dem = n.detector_error_model(decompose_errors=True); mt = pymatching.Matching.from_detector_error_model(dem)
smp = n.compile_detector_sampler(); shots = errs = 0
while errs < 300 and shots < 2_000_000:
    det, obs = smp.sample(100_000, separate_observables=True)
    errs += int(np.any(mt.decode_batch(det) != obs, axis=1).sum()); shots += 100_000
R = int(sys.argv[2]); ps = errs / shots; pr = 1 - (1 - ps) ** (1 / R)
print(sys.argv[1].split('/')[-1], "merged", merged, "ticks", ticks, f"per_round {pr:.3g} shots {shots} errors {errs}")
