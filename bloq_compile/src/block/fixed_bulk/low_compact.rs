//! EXPERIMENT (local, not for upstream): the compact X-top memory of Low et al., "A Denser Planar
//! Surface Code" (arXiv:2605.30455, Sec. 1, Fig. 5(c)), as the template of an isolated cube.
//!
//! Set `BLOQ_EXPERIMENTAL_LOW_COMPACT=1` and an isolated regular cube (no pipes) compiles to rounds
//! of G_X and G_X^-1 in turn instead of the fixed-bulk rotated patch. Everything around the template
//! (placement, flow composition into detectors, the observable gateway, readouts, emission) is
//! bloq's own. The round schedules and the Pauli propagation that derives each round's flows are
//! ported from the L3 generator (`low_gadgets/rust/src/main.rs`: `g_x_round`, `RoundSpec`,
//! `build_round`).
//!
//! Geometry: the compact patch has its d^2 data qubits where the fixed-bulk patch has them,
//! (2i+1, 2j+1), and one measure qubit per data qubit at (2a+2, 2b), so it fits in bloq's cell.
//! On it the middle data row (Z) and column (X) are deterministic observables after G_X and G_X^-1,
//! with no extra records, so the fixed-bulk gateway (`line_qubits`) applies unchanged.

use std::collections::{BTreeMap, HashMap, HashSet};

use bloq_circuit::{Chunk, ChunkOrLoop, CoordCircuit, Flow, GateType, Pauli, PauliBasis, PauliMap};
use bloq_graph::{Basis, CubeKind};
use glam::IVec2;

/// Whether the experiment switch is set.
pub(super) fn enabled() -> bool {
    std::env::var_os("BLOQ_EXPERIMENTAL_LOW_COMPACT").is_some_and(|v| v != "0")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum B {
    X,
    Z,
}

impl B {
    fn of(basis: Basis) -> Self {
        match basis {
            Basis::X => B::X,
            Basis::Z => B::Z,
        }
    }
    fn pauli(self) -> Pauli {
        match self {
            B::X => Pauli::X,
            B::Z => Pauli::Z,
        }
    }
    fn measure_basis(self) -> PauliBasis {
        match self {
            B::X => PauliBasis::X,
            B::Z => PauliBasis::Z,
        }
    }
    fn reset_gate(self) -> GateType {
        match self {
            B::X => GateType::RX,
            B::Z => GateType::RZ,
        }
    }
}

/// One CX step of a measure qubit: the data neighbour offset and whether the measure qubit controls.
type Step = Option<(IVec2, bool)>;

struct MeasureQubit {
    q: IVec2,
    reset: B,
    meas: B,
    steps: [Step; 4],
}

/// The G_X round of Fig. 5(c), in the paper's doubled coordinates (y down): data at (2i, 2j+1),
/// measure qubits at (2a+1, 2b) except the unused top-right slot (2d-1, 0).
fn g_x_round(d: i32) -> (Vec<IVec2>, Vec<MeasureQubit>) {
    const SW: IVec2 = IVec2::new(-1, 1);
    const SE: IVec2 = IVec2::new(1, 1);
    const NW: IVec2 = IVec2::new(-1, -1);
    let c = |o: IVec2| Some((o, true));
    let t = |o: IVec2| Some((o, false));
    let data = (0..d)
        .flat_map(|j| (0..d).map(move |i| IVec2::new(2 * i, 2 * j + 1)))
        .collect();
    let mut meas = Vec::new();
    for b in 0..d {
        for a in 0..d {
            let q = IVec2::new(2 * a + 1, 2 * b);
            let (reset, measure, steps) = if b == 0 {
                if a == d - 1 {
                    continue;
                }
                let steps = if a % 2 == 0 {
                    [c(SW), c(SE), None, None]
                } else {
                    [None, None, c(SE), c(SW)]
                };
                (B::X, B::X, steps)
            } else if a == d - 1 {
                let steps = if (d + b) % 2 == 0 {
                    [t(SW), t(NW), None, None]
                } else {
                    [None, None, t(NW), t(SW)]
                };
                (B::Z, B::Z, steps)
            } else if (a + b) % 2 == 0 {
                (B::X, B::Z, [c(SW), c(SE), t(NW), t(SW)])
            } else {
                (B::Z, B::X, [t(SW), t(NW), c(SE), c(SW)])
            };
            meas.push(MeasureQubit {
                q,
                reset,
                meas: measure,
                steps,
            });
        }
    }
    (data, meas)
}

/// One round: resets, four CX layers, measurements. Qubits not reset hold data at the start.
#[derive(Clone)]
struct RoundSpec {
    qubits: Vec<IVec2>,
    resets: Vec<(IVec2, B)>,
    layers: Vec<Vec<(IVec2, IVec2)>>,
    measures: Vec<(IVec2, B)>,
}

impl RoundSpec {
    fn g_x(d: i32) -> Self {
        let (mut qubits, meas) = g_x_round(d);
        qubits.extend(meas.iter().map(|m| m.q));
        let mut layers = vec![Vec::new(); 4];
        for (k, layer) in layers.iter_mut().enumerate() {
            for m in &meas {
                if let Some((offset, controls)) = m.steps[k] {
                    let data = m.q + offset;
                    layer.push(if controls { (m.q, data) } else { (data, m.q) });
                }
            }
        }
        Self {
            qubits,
            resets: meas.iter().map(|m| (m.q, m.reset)).collect(),
            layers,
            measures: meas.iter().map(|m| (m.q, m.meas)).collect(),
        }
    }
    fn data(&self) -> Vec<IVec2> {
        let r: HashSet<IVec2> = self.resets.iter().map(|&(q, _)| q).collect();
        self.qubits
            .iter()
            .copied()
            .filter(|q| !r.contains(q))
            .collect()
    }
    /// The time reverse: CX layers reversed, resets where this round measures and the converse.
    fn inverse(&self) -> Self {
        Self {
            qubits: self.qubits.clone(),
            resets: self.measures.clone(),
            layers: self.layers.iter().rev().cloned().collect(),
            measures: self.resets.clone(),
        }
    }
    fn mapped(&self, f: impl Fn(IVec2) -> IVec2) -> Self {
        Self {
            qubits: self.qubits.iter().map(|&q| f(q)).collect(),
            resets: self.resets.iter().map(|&(q, b)| (f(q), b)).collect(),
            layers: (self.layers.iter())
                .map(|l| l.iter().map(|&(c, t)| (f(c), f(t))).collect())
                .collect(),
            measures: self.measures.iter().map(|&(q, b)| (f(q), b)).collect(),
        }
    }
}

/// A CSS Pauli product as X and Z bits per qubit.
#[derive(Clone, Default)]
struct Pauli2(BTreeMap<(i32, i32), (bool, bool)>);

impl Pauli2 {
    fn single(q: IVec2, b: B) -> Self {
        let mut p = Self::default();
        p.0.insert((q.x, q.y), (b == B::X, b == B::Z));
        p
    }
    fn get(&self, q: IVec2) -> (bool, bool) {
        self.0.get(&(q.x, q.y)).copied().unwrap_or_default()
    }
    fn set(&mut self, q: IVec2, v: (bool, bool)) {
        if v == (false, false) {
            self.0.remove(&(q.x, q.y));
        } else {
            self.0.insert((q.x, q.y), v);
        }
    }
    fn cx(&mut self, c: IVec2, t: IVec2) {
        let (xc, zc) = self.get(c);
        let (xt, zt) = self.get(t);
        self.set(t, (xt ^ xc, zt));
        self.set(c, (xc, zc ^ zt));
    }
    fn times(mut self, other: &Self) -> Self {
        for (&(x, y), &(ox, oz)) in &other.0 {
            let q = IVec2::new(x, y);
            let (sx, sz) = self.get(q);
            self.set(q, (sx ^ ox, sz ^ oz));
        }
        self
    }
}

type Bits = std::collections::BTreeSet<(i32, i32, bool)>;

/// The bits of `p` that the qubits in `allowed` would not accept (and any Y elsewhere).
fn wrong_bits(p: &Pauli2, allowed: &HashMap<IVec2, B>) -> Bits {
    p.0.iter()
        .flat_map(|(&(x, y), &(px, pz))| {
            let bad = match allowed.get(&IVec2::new(x, y)) {
                Some(B::Z) => [px, false],
                Some(B::X) => [false, pz],
                None => [px && pz, false],
            };
            [(x, y, false), (x, y, true)]
                .into_iter()
                .zip(bad)
                .filter(|&(_, b)| b)
                .map(|(k, _)| k)
        })
        .collect()
}

/// Products of `items` with only allowed Paulis on the `allowed` qubits (GF(2) elimination).
fn deterministic(
    items: Vec<(Vec<usize>, Pauli2)>,
    allowed: &HashMap<IVec2, B>,
) -> Vec<(Vec<usize>, Pauli2)> {
    let mut out = Vec::new();
    let mut pivots: Vec<(Bits, Vec<usize>, Pauli2)> = Vec::new();
    for (idx, p) in items {
        let mut cur = (wrong_bits(&p, allowed), idx, p);
        loop {
            let Some(&lead) = cur.0.iter().next() else {
                cur.1.sort_unstable();
                out.push((cur.1, cur.2));
                break;
            };
            match pivots.iter().find(|pv| pv.0.iter().next() == Some(&lead)) {
                Some(pv) => {
                    cur.0 = cur.0.symmetric_difference(&pv.0).copied().collect();
                    for &x in &pv.1 {
                        match cur.1.iter().position(|&y| y == x) {
                            Some(i) => {
                                cur.1.swap_remove(i);
                            }
                            None => cur.1.push(x),
                        }
                    }
                    cur.2 = cur.2.times(&pv.2);
                }
                None => {
                    pivots.push(cur);
                    break;
                }
            }
        }
    }
    out
}

/// The data part of a propagated product, and the measure qubits it touches.
fn split(p: &Pauli2, allowed: &HashMap<IVec2, B>) -> (BTreeMap<(i32, i32), B>, Vec<IVec2>) {
    let mut data = BTreeMap::new();
    let mut touched = Vec::new();
    for (&(x, y), &v) in &p.0 {
        let q = IVec2::new(x, y);
        let b = match v {
            (true, false) => B::X,
            (false, true) => B::Z,
            _ => panic!("low_compact: Y at {q}"),
        };
        match allowed.get(&q) {
            Some(&want) if want == b => touched.push(q),
            Some(_) => panic!("low_compact: measure qubit {q} carries the wrong Pauli"),
            None => {
                data.insert((x, y), b);
            }
        }
    }
    (data, touched)
}

fn propagate(mut p: Pauli2, spec: &RoundSpec, backward: bool) -> Pauli2 {
    let layers: Vec<&Vec<(IVec2, IVec2)>> = if backward {
        spec.layers.iter().rev().collect()
    } else {
        spec.layers.iter().collect()
    };
    for layer in layers {
        for &(c, t) in layer {
            p.cx(c, t);
        }
    }
    p
}

fn pauli_map(p: &BTreeMap<(i32, i32), B>) -> PauliMap {
    PauliMap::from_unique_entries(p.iter().map(|(&(x, y), b)| (IVec2::new(x, y), b.pauli())))
}

/// One round as a chunk with its detector flows (no logical flow: the gateway reads the observable).
fn build_round(
    spec: &RoundSpec,
    init: Option<B>,
    readout: Option<B>,
) -> Result<Chunk, crate::CompileError> {
    let mut circuit = CoordCircuit::new();
    for b in [B::X, B::Z] {
        let mut targets: Vec<IVec2> = (spec.resets.iter())
            .filter(|&&(_, rb)| rb == b)
            .map(|&(q, _)| q)
            .collect();
        if init == Some(b) {
            targets.extend(spec.data());
        }
        circuit.do_gate(b.reset_gate(), targets)?;
    }
    circuit.tick();
    for layer in &spec.layers {
        circuit.do_gate(GateType::CX, layer.iter().flat_map(|&(c, t)| [c, t]))?;
        circuit.tick();
    }
    let mut ids = vec![0; spec.measures.len()];
    for b in [B::X, B::Z] {
        let (idx, qs): (Vec<usize>, Vec<IVec2>) = (spec.measures.iter().enumerate())
            .filter(|(_, (_, mb))| *mb == b)
            .map(|(i, &(q, _))| (i, q))
            .unzip();
        for (i, id) in idx.into_iter().zip(circuit.measure(b.measure_basis(), qs)) {
            ids[i] = id;
        }
    }
    let data_id: HashMap<(i32, i32), u32> = match readout {
        // The data readout shares the round's measurement moment: bloq counts one
        // measurement moment per round when it aligns moments.
        Some(b) => {
            let data = spec.data();
            let got = circuit.measure(b.measure_basis(), data.clone());
            data.iter()
                .zip(got)
                .map(|(q, id)| ((q.x, q.y), id))
                .collect()
        }
        None => HashMap::new(),
    };
    let reset_basis: HashMap<IVec2, B> = spec.resets.iter().copied().collect();
    let meas_basis: HashMap<IVec2, B> = spec.measures.iter().copied().collect();
    let id_of: HashMap<IVec2, u32> = (spec.measures.iter())
        .zip(&ids)
        .map(|(&(q, _), &id)| (q, id))
        .collect();
    let all_in =
        |p: &BTreeMap<(i32, i32), B>, b: Option<B>| b.is_some_and(|b| p.values().all(|&v| v == b));

    let mut flows = Vec::new();
    // Backward: what (products of) this round's measurements read off the data at its start.
    let back = (spec.measures.iter().enumerate())
        .map(|(i, &(q, b))| (vec![i], propagate(Pauli2::single(q, b), spec, true)))
        .collect();
    for (idx, p) in deterministic(back, &reset_basis) {
        let (s_in, _) = split(&p, &reset_basis);
        let flow = if !s_in.is_empty() && init.is_none() {
            Flow::new(pauli_map(&s_in), PauliMap::empty())
        } else if all_in(&s_in, init) {
            Flow::new(PauliMap::empty(), PauliMap::empty())
        } else {
            continue;
        };
        let ms: Vec<u32> = idx.iter().map(|&i| ids[i]).collect();
        flows.push(
            flow.with_measurements(ms)
                .with_center(spec.measures[idx[0]].0),
        );
    }
    // Forward: what (products of) this round's resets prepare on the data at its end.
    let fwd = (spec.resets.iter().enumerate())
        .map(|(i, &(q, b))| (vec![i], propagate(Pauli2::single(q, b), spec, false)))
        .collect();
    for (idx, p) in deterministic(fwd, &meas_basis) {
        let (s_out, via) = split(&p, &meas_basis);
        let mut ms: Vec<u32> = via.iter().map(|v| id_of[v]).collect();
        let end = if readout.is_none() {
            pauli_map(&s_out)
        } else if all_in(&s_out, readout) {
            ms.extend(s_out.keys().map(|k| data_id[k]));
            PauliMap::empty()
        } else {
            continue;
        };
        flows.push(
            Flow::new(PauliMap::empty(), end)
                .with_measurements(ms)
                .with_center(spec.resets[idx[0]].0),
        );
    }
    Ok(Chunk { circuit, flows })
}

/// The chunks of an isolated compact memory cube of `kind` for `rounds` rounds (at least 2):
/// G_X (with data preparation), then G_X^-1 and G_X in turn, the last with data readout.
pub(super) fn compact_cube_chunks(
    kind: CubeKind,
    distance: u32,
    rounds: u32,
    prepare: bool,
    read: bool,
) -> Result<Vec<ChunkOrLoop>, crate::CompileError> {
    let d = distance as i32;
    let basis = B::of(kind.z());
    // Paper coordinates to bloq's: data (2i, 2j+1) -> (2i+1, 2j+1). Kinds with X boundaries on
    // the x faces are the mirror in x = y.
    let transposed = kind.x() == Basis::X;
    let to_bloq = move |q: IVec2| {
        let q = q + IVec2::new(1, 0);
        if transposed { IVec2::new(q.y, q.x) } else { q }
    };
    // A cube above a temporal pipe starts with G_X^-1, so a cube with an odd round count below
    // (which ends with G_X) is followed by its inverse. The template does not know the global round
    // parity, so a chain of three or more cubes, or an even count below, still mismatches.
    let g = RoundSpec::g_x(d).mapped(to_bloq);
    let (first, second) = if prepare {
        (g.clone(), g.inverse())
    } else {
        (g.inverse(), g)
    };
    let spec = |r: u32| if r.is_multiple_of(2) { &first } else { &second };
    let rounds = rounds.max(2);
    let last = rounds - 1;
    let mut chunks = vec![ChunkOrLoop::Single(Box::new(build_round(
        &first,
        prepare.then_some(basis),
        None,
    )?))];
    let middle = last - 1; // rounds 1..last
    if middle >= 2 {
        chunks.push(ChunkOrLoop::Loop {
            body: vec![
                build_round(&second, None, None)?,
                build_round(&first, None, None)?,
            ],
            repetitions: middle / 2,
        });
    }
    if middle % 2 == 1 {
        chunks.push(ChunkOrLoop::Single(Box::new(build_round(
            &second, None, None,
        )?)));
    }
    chunks.push(ChunkOrLoop::Single(Box::new(build_round(
        spec(last),
        None,
        read.then_some(basis),
    )?)));
    Ok(chunks)
}
