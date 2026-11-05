//! Compact-patch memory of Low et al., "A Denser Planar Surface Code" (arXiv:2605.30455, Sec. 1,
//! Fig. 5), built from bloq primitives outside the compile pipeline (level L3).
//!
//! bloq parts used: `CoordCircuit` for the gates, one `Chunk` per round with its declared `Flow`s,
//! `StimFlowVerifier` to check every chunk's flows with stim, `FlowEngine` to fuse the flows of
//! consecutive rounds into detectors, and `emit_annotated_stim` for the final text.
//! Not from bloq: the gate schedule (read from Fig. 5(c)) and the Pauli propagation that derives each
//! round's flows (bloq has no public flow solver).
//!
//! Usage: `low-gadgets compact-memory <x-top|z-top> <x|z> <d> <rounds> <out.stim>`

use std::collections::{BTreeMap, HashMap};

use bloq_circuit::{
    Chunk, CoordCircuit, Flow, FlowEngine, GateType, OffsetFlows, Pauli, PauliBasis, PauliMap,
};
use bloq_stim::{StimFlowVerifier, emit_annotated_stim};
use glam::IVec2;

type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum B {
    X,
    Z,
}

impl B {
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

const SW: IVec2 = IVec2::new(-1, 1);
const SE: IVec2 = IVec2::new(1, 1);
const NW: IVec2 = IVec2::new(-1, -1);

#[derive(Clone, Debug)]
struct MeasureQubit {
    q: IVec2,
    reset: B,
    meas: B,
    steps: [Step; 4],
}

/// The G_X round of Fig. 5(c) for any odd d, in doubled coordinates with y pointing down.
///
/// Data qubits sit at (2i, 2j+1), measure qubits at (2a+1, 2b) for i, j, a, b in 0..d, except the
/// unused top-right slot (2d-1, 0): 2d^2 slots, 2d^2-1 qubits. The per-class schedules were read off
/// the d = 5 figure's vector data; at d = 5 this reproduces its 80 CX gates exactly.
fn g_x_round(d: i32) -> (Vec<IVec2>, Vec<MeasureQubit>) {
    let c = |o: IVec2| Some((o, true));
    let t = |o: IVec2| Some((o, false));
    let data = (0..d)
        .flat_map(|j| (0..d).map(move |i| IVec2::new(2 * i, 2 * j + 1)))
        .collect();
    let mut meas = Vec::new();
    for b in 0..d {
        for a in 0..d {
            let q = IVec2::new(2 * a + 1, 2 * b);
            let m = if b == 0 {
                if a == d - 1 {
                    continue;
                }
                // Top boundary: X checks, active in the first or second half of the round.
                let steps = if a % 2 == 0 {
                    [c(SW), c(SE), None, None]
                } else {
                    [None, None, c(SE), c(SW)]
                };
                MeasureQubit {
                    q,
                    reset: B::X,
                    meas: B::X,
                    steps,
                }
            } else if a == d - 1 {
                // Right boundary: Z checks, in phase with the bulk qubit to their left.
                let steps = if (d + b) % 2 == 0 {
                    [t(SW), t(NW), None, None]
                } else {
                    [None, None, t(NW), t(SW)]
                };
                MeasureQubit {
                    q,
                    reset: B::Z,
                    meas: B::Z,
                    steps,
                }
            } else if (a + b) % 2 == 0 {
                MeasureQubit {
                    q,
                    reset: B::X,
                    meas: B::Z,
                    steps: [c(SW), c(SE), t(NW), t(SW)],
                }
            } else {
                MeasureQubit {
                    q,
                    reset: B::Z,
                    meas: B::X,
                    steps: [t(SW), t(NW), c(SE), c(SW)],
                }
            };
            meas.push(m);
        }
    }
    (data, meas)
}

/// G_X^-1: the CX layers in reverse order, resetting in G_X's measure basis and measuring in its
/// reset basis (assumption A3 in the log: the time reverse of G_X).
fn inverse(meas: &[MeasureQubit]) -> Vec<MeasureQubit> {
    meas.iter()
        .map(|m| {
            let mut steps = m.steps;
            steps.reverse();
            MeasureQubit {
                q: m.q,
                reset: m.meas,
                meas: m.reset,
                steps,
            }
        })
        .collect()
}

/// CX layers of one round as (control, target) pairs per tick.
fn cx_layers(meas: &[MeasureQubit]) -> Result<Vec<Vec<(IVec2, IVec2)>>, Error> {
    let mut layers = vec![Vec::new(); 4];
    for (k, layer) in layers.iter_mut().enumerate() {
        let mut used = std::collections::HashSet::new();
        for m in meas {
            if let Some((offset, measure_controls)) = m.steps[k] {
                let data = m.q + offset;
                if !used.insert(data) || !used.insert(m.q) {
                    return Err(format!("qubit used twice in CX layer {k} at {data}").into());
                }
                layer.push(if measure_controls {
                    (m.q, data)
                } else {
                    (data, m.q)
                });
            }
        }
    }
    Ok(layers)
}

/// A CSS-only Pauli product as X and Z bit sets.
#[derive(Clone, Default, Debug)]
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
    /// Conjugate by CX (self-inverse, so the same rule propagates forward and backward).
    fn cx(&mut self, c: IVec2, t: IVec2) {
        let (xc, zc) = self.get(c);
        let (xt, zt) = self.get(t);
        self.set(t, (xt ^ xc, zt));
        self.set(c, (xc, zc ^ zt));
    }
    fn iter(&self) -> impl Iterator<Item = (IVec2, Option<B>)> + '_ {
        self.0.iter().map(|(&(x, y), &v)| {
            let b = match v {
                (true, false) => Some(B::X),
                (false, true) => Some(B::Z),
                _ => None,
            };
            (IVec2::new(x, y), b)
        })
    }
}

/// The data and measure parts of a propagated product. Errors if a measure qubit carries a Pauli
/// that is not `allowed` there (it would not be deterministic) or a data qubit carries Y.
fn split(
    p: &Pauli2,
    allowed: &HashMap<IVec2, B>,
) -> Result<(BTreeMap<(i32, i32), B>, Vec<IVec2>), Error> {
    let mut data = BTreeMap::new();
    let mut touched = Vec::new();
    for (q, b) in p.iter() {
        let b = b.ok_or_else(|| format!("Y at {q}"))?;
        match allowed.get(&q) {
            Some(&want) if want == b => touched.push(q),
            Some(_) => return Err(format!("measure qubit {q} carries the wrong Pauli").into()),
            None => {
                data.insert((q.x, q.y), b);
            }
        }
    }
    Ok((data, touched))
}

fn pauli_map(p: &BTreeMap<(i32, i32), B>) -> PauliMap {
    PauliMap::from_unique_entries(p.iter().map(|(&(x, y), b)| (IVec2::new(x, y), b.pauli())))
}

/// What the experiment does at the time boundaries.
#[derive(Clone, Copy)]
struct Ends {
    init: Option<B>,
    readout: Option<B>,
}

/// Emit one round into `circuit`; returns the measure-qubit and data measurement ids.
fn emit_round(
    circuit: &mut CoordCircuit,
    data: &[IVec2],
    meas: &[MeasureQubit],
    layers: &[Vec<(IVec2, IVec2)>],
    ends: Ends,
) -> Result<(Vec<u32>, Vec<u32>), Error> {
    for b in [B::X, B::Z] {
        let mut targets: Vec<IVec2> = meas.iter().filter(|m| m.reset == b).map(|m| m.q).collect();
        if ends.init == Some(b) {
            targets.extend(data);
        }
        circuit.do_gate(b.reset_gate(), targets)?;
    }
    circuit.tick();
    for layer in layers {
        circuit.do_gate(GateType::CX, layer.iter().flat_map(|&(c, t)| [c, t]))?;
        circuit.tick();
    }
    let mut ids = vec![0; meas.len()];
    for b in [B::X, B::Z] {
        let (idx, qs): (Vec<usize>, Vec<IVec2>) = meas
            .iter()
            .enumerate()
            .filter(|(_, m)| m.meas == b)
            .map(|(i, m)| (i, m.q))
            .unzip();
        for (i, id) in idx.into_iter().zip(circuit.measure(b.measure_basis(), qs)) {
            ids[i] = id;
        }
    }
    let data_ids = match ends.readout {
        Some(b) => {
            circuit.tick();
            circuit.measure(b.measure_basis(), data.iter().copied())
        }
        None => Vec::new(),
    };
    Ok((ids, data_ids))
}

/// One round as a verified chunk plus the logical flow kept apart from the detector flows.
struct Round {
    chunk: Chunk,
    detector_flows: usize,
}

fn build_round(
    data: &[IVec2],
    meas: &[MeasureQubit],
    logical: &BTreeMap<(i32, i32), B>,
    ends: Ends,
) -> Result<(Round, Vec<u32>, Vec<u32>), Error> {
    let layers = cx_layers(meas)?;
    let mut circuit = CoordCircuit::new();
    let (ids, data_ids) = emit_round(&mut circuit, data, meas, &layers, ends)?;
    let reset_basis: HashMap<IVec2, B> = meas.iter().map(|m| (m.q, m.reset)).collect();
    let meas_basis: HashMap<IVec2, B> = meas.iter().map(|m| (m.q, m.meas)).collect();
    let id_of: HashMap<IVec2, u32> = meas.iter().zip(&ids).map(|(m, &id)| (m.q, id)).collect();
    let data_id: HashMap<(i32, i32), u32> = data
        .iter()
        .zip(&data_ids)
        .map(|(q, &id)| ((q.x, q.y), id))
        .collect();
    let all_in =
        |p: &BTreeMap<(i32, i32), B>, b: Option<B>| b.is_some_and(|b| p.values().all(|&v| v == b));

    let mut flows = Vec::new();
    for m in meas {
        // Backward: what the measurement of m reads off the data at the start of the round.
        let mut back = Pauli2::single(m.q, m.meas);
        for layer in layers.iter().rev() {
            for &(c, t) in layer {
                back.cx(c, t);
            }
        }
        let (s_in, _) = split(&back, &reset_basis)?;
        if !s_in.is_empty() && ends.init.is_none() {
            flows.push(
                Flow::new(pauli_map(&s_in), PauliMap::empty())
                    .with_measurements([id_of[&m.q]])
                    .with_center(m.q),
            );
        } else if all_in(&s_in, ends.init) {
            flows.push(
                Flow::new(PauliMap::empty(), PauliMap::empty())
                    .with_measurements([id_of[&m.q]])
                    .with_center(m.q),
            );
        }
        // Forward: what the reset of m prepares on the data at the end of the round.
        let mut fwd = Pauli2::single(m.q, m.reset);
        for layer in &layers {
            for &(c, t) in layer {
                fwd.cx(c, t);
            }
        }
        let (s_out, via) = split(&fwd, &meas_basis)?;
        let mut ms: Vec<u32> = via.iter().map(|q| id_of[q]).collect();
        if ends.readout.is_none() {
            flows.push(
                Flow::new(PauliMap::empty(), pauli_map(&s_out))
                    .with_measurements(ms)
                    .with_center(m.q),
            );
        } else if all_in(&s_out, ends.readout) {
            ms.extend(s_out.keys().map(|k| data_id[k]));
            flows.push(
                Flow::new(PauliMap::empty(), PauliMap::empty())
                    .with_measurements(ms)
                    .with_center(m.q),
            );
        }
    }
    let detector_flows = flows.len();
    // The logical crosses every round unchanged (checked by the verifier below).
    let start = if ends.init.is_some() {
        PauliMap::empty()
    } else {
        pauli_map(logical)
    };
    let (end, ms) = if ends.readout.is_some() {
        (
            PauliMap::empty(),
            logical.keys().map(|k| data_id[k]).collect::<Vec<_>>(),
        )
    } else {
        (pauli_map(logical), Vec::new())
    };
    flows.push(Flow::new(start, end).with_measurements(ms));
    let chunk = Chunk { circuit, flows };
    chunk.verify_flows(None, None)?;
    Ok((
        Round {
            chunk,
            detector_flows,
        },
        ids,
        data_ids,
    ))
}

fn compact_memory(top: &str, basis: B, d: i32, rounds: usize) -> Result<String, Error> {
    if top != "x-top" {
        return Err(format!("{top}: only x-top (Fig. 5 left) is implemented").into());
    }
    if d < 3 || d % 2 == 0 || rounds < 2 {
        return Err("need odd d >= 3 and at least 2 rounds".into());
    }
    let (data, g_x) = g_x_round(d);
    let g_x_inv = inverse(&g_x);
    // Logical X runs down the left column, logical Z along the top row; both are left unchanged by
    // G_X and G_X^-1, so a memory in either basis reads it from the final data measurement.
    let logical: BTreeMap<(i32, i32), B> = match basis {
        B::X => (0..d).map(|j| ((0, 2 * j + 1), B::X)).collect(),
        B::Z => (0..d).map(|i| ((2 * i, 1), B::Z)).collect(),
    };
    let mut full = CoordCircuit::new();
    let mut built = Vec::new();
    let mut observable = Vec::new();
    for r in 0..rounds {
        let meas = if r % 2 == 0 { &g_x } else { &g_x_inv };
        let ends = Ends {
            init: (r == 0).then_some(basis),
            readout: (r + 1 == rounds).then_some(basis),
        };
        let (round, local_ids, local_data) = build_round(&data, meas, &logical, ends)?;
        if r > 0 {
            full.tick();
        }
        let layers = cx_layers(meas)?;
        let (global_ids, global_data) = emit_round(&mut full, &data, meas, &layers, ends)?;
        let map: HashMap<u32, u32> = local_ids
            .iter()
            .chain(&local_data)
            .copied()
            .zip(global_ids.iter().chain(&global_data).copied())
            .collect();
        if let Some(obs) = round.chunk.flows.last().filter(|_| ends.readout.is_some()) {
            observable = obs.measurements.iter().map(|m| map[m]).collect();
        }
        built.push((round, map));
    }
    let mut engine: FlowEngine<'_, u32> = FlowEngine::new();
    for (round, map) in &built {
        let flows = &round.chunk.flows[..round.detector_flows];
        engine
            .append_group_allowing_dangling(&[OffsetFlows::new(flows, IVec2::ZERO)], |_, m| {
                map[m]
            })?;
    }
    let detectors: Vec<Vec<u32>> = engine
        .finish()?
        .into_iter()
        .map(|f| {
            assert!(!f.sign, "every flow here is positive");
            f.measurements.into_iter().collect()
        })
        .collect();
    Ok(emit_annotated_stim(&full, &detectors, &[(0, observable)])?)
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, top, basis, d, rounds, out] if cmd == "compact-memory" => {
            let basis = match basis.as_str() {
                "x" => B::X,
                "z" => B::Z,
                other => return Err(format!("basis {other}: want x or z").into()),
            };
            let text = compact_memory(top, basis, d.parse()?, rounds.parse()?)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        _ => Err(
            "usage: low-gadgets compact-memory <x-top|z-top> <x|z> <d> <rounds> <out.stim>".into(),
        ),
    }
}
