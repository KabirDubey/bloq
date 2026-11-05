//! Compact-patch memory of Low et al., "A Denser Planar Surface Code" (arXiv:2605.30455, Sec. 1,
//! Fig. 5), built from bloq primitives outside the compile pipeline (level L3).
//!
//! bloq parts used: `CoordCircuit` for the gates, one `Chunk` per round with its declared `Flow`s,
//! `StimFlowVerifier` to check every chunk's flows with stim, `FlowEngine` to fuse the flows of
//! consecutive rounds into detectors, and `emit_annotated_stim` for the final text.
//! Not from bloq: the gate schedules (read from Fig. 5(c) and (d)) and the Pauli propagation that derives each
//! round's flows (bloq has no public flow solver).
//!
//! Usage: `low-gadgets compact-memory <x-top|z-top> <x|z> <d> <rounds> <out.stim>`
//! (x-top: Fig. 5 left and (c); z-top: Fig. 5 right and (d)).

use std::collections::{BTreeMap, HashMap, HashSet};

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

/// One round as resets, four CX layers and measurements. Qubits that are not reset hold data at the
/// start of the round; qubits that are not measured hold data at its end. In G_X both sets are the
/// same; in G_Z some boundary qubits swap roles (Fig. 5(d)).
#[derive(Clone, Debug)]
struct RoundSpec {
    qubits: Vec<IVec2>,
    resets: Vec<(IVec2, B)>,
    layers: Vec<Vec<(IVec2, IVec2)>>,
    measures: Vec<(IVec2, B)>,
}

impl RoundSpec {
    fn data_in(&self) -> Vec<IVec2> {
        let r: HashSet<IVec2> = self.resets.iter().map(|&(q, _)| q).collect();
        self.qubits
            .iter()
            .copied()
            .filter(|q| !r.contains(q))
            .collect()
    }
    fn data_out(&self) -> Vec<IVec2> {
        let m: HashSet<IVec2> = self.measures.iter().map(|&(q, _)| q).collect();
        self.qubits
            .iter()
            .copied()
            .filter(|q| !m.contains(q))
            .collect()
    }
    /// The time reverse: CX layers in reverse order, resets where this round measures (in the
    /// measure basis) and measurements where it resets (assumption A3 in the log).
    fn inverse(&self) -> Self {
        Self {
            qubits: self.qubits.clone(),
            resets: self.measures.clone(),
            layers: self.layers.iter().rev().cloned().collect(),
            measures: self.resets.clone(),
        }
    }
    fn from_measure_qubits(data: Vec<IVec2>, meas: &[MeasureQubit]) -> Result<Self, Error> {
        let mut qubits = data;
        qubits.extend(meas.iter().map(|m| m.q));
        Ok(Self {
            qubits,
            resets: meas.iter().map(|m| (m.q, m.reset)).collect(),
            layers: cx_layers(meas)?,
            measures: meas.iter().map(|m| (m.q, m.meas)).collect(),
        })
    }
}

/// CX layers of one round as (control, target) pairs per tick.
fn cx_layers(meas: &[MeasureQubit]) -> Result<Vec<Vec<(IVec2, IVec2)>>, Error> {
    let mut layers = vec![Vec::new(); 4];
    for (k, layer) in layers.iter_mut().enumerate() {
        let mut used = HashSet::new();
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

/// The G_Z round of Fig. 5(d) for any odd d, on the same slots as G_X but with the empty slot at
/// the bottom-left (0, 2d-1) instead of the top-right.
///
/// Rules read off the d = 5 figure's vector data (at d = 5 they reproduce its 88 CX gates, 24
/// resets and 24 measurements exactly). Every slot (2a+1, 2b) pairs with its SW neighbour in
/// ticks 1 and 4 and follows the G_X classes in ticks 2 and 3. Bulk, top row and right column
/// reset and measure in place with the G_X bulk bases. On the left column (X checks) and the bottom
/// row (Z checks) a reset or a measurement moves to the SW neighbour on alternate checks, so those
/// qubits change role between data and measure each round.
fn g_z_round(d: i32) -> Result<RoundSpec, Error> {
    let qubits: Vec<IVec2> = (0..2 * d)
        .flat_map(|y| (0..2 * d).map(move |x| IVec2::new(x, y)))
        .filter(|q| (q.x + q.y) % 2 == 1 && *q != IVec2::new(0, 2 * d - 1))
        .collect();
    let exists: HashSet<IVec2> = qubits.iter().copied().collect();
    let mut layers = vec![Vec::new(); 4];
    let mut resets = Vec::new();
    let mut measures = Vec::new();
    let mut add = |tick: usize, m: IVec2, offset: IVec2, measure_controls: bool| {
        let n = m + offset;
        if exists.contains(&n) {
            layers[tick].push(if measure_controls { (m, n) } else { (n, m) });
        }
    };
    for b in 0..d {
        for a in 0..d {
            let m = IVec2::new(2 * a + 1, 2 * b);
            let even = (a + b) % 2 == 0;
            // Ticks 1 and 4: the SW neighbour.
            if b == d - 1 {
                if a > 0 {
                    add(0, m, SW, false);
                    add(3, m, SW, false);
                }
            } else if a == 0 {
                add(0, m, SW, true);
                add(3, m, SW, true);
            } else {
                add(0, m, SW, even);
                add(3, m, SW, !even);
            }
            // Ticks 2 and 3: as in G_X.
            if b == 0 {
                if a < d - 1 {
                    add(if a % 2 == 0 { 1 } else { 2 }, m, SE, true);
                }
            } else if a == d - 1 {
                add(if (d + b) % 2 == 0 { 1 } else { 2 }, m, NW, false);
            } else if even {
                add(1, m, SE, true);
                add(2, m, NW, false);
            } else {
                add(1, m, NW, false);
                add(2, m, SE, true);
            }
            // Resets and measurements.
            if a == 0 && b < d - 1 {
                let moved = IVec2::new(0, 2 * b + 1);
                let (r, me) = if b % 2 == 0 { (m, moved) } else { (moved, m) };
                resets.push((r, B::X));
                measures.push((me, B::X));
            } else if b == d - 1 && a > 0 {
                let moved = IVec2::new(2 * a, 2 * d - 1);
                let (r, me) = if a % 2 == 1 { (m, moved) } else { (moved, m) };
                resets.push((r, B::Z));
                measures.push((me, B::Z));
            } else if a > 0 || b < d - 1 {
                let (r, me) = if even { (B::X, B::Z) } else { (B::Z, B::X) };
                resets.push((m, r));
                measures.push((m, me));
            }
        }
    }
    for (k, layer) in layers.iter().enumerate() {
        let mut used = HashSet::new();
        if !layer.iter().all(|&(c, t)| used.insert(c) && used.insert(t)) {
            return Err(format!("G_Z: a qubit is used twice in CX layer {k}").into());
        }
    }
    Ok(RoundSpec {
        qubits,
        resets,
        layers,
        measures,
    })
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
    fn times(mut self, other: &Self) -> Self {
        for (&(x, y), &(ox, oz)) in &other.0 {
            let q = IVec2::new(x, y);
            let (sx, sz) = self.get(q);
            self.set(q, (sx ^ ox, sz ^ oz));
        }
        self
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

/// Emit one round into `circuit`; returns the measurement ids (in `spec.measures` order) and the
/// final data measurement ids (in `spec.data_out()` order).
fn emit_round(
    circuit: &mut CoordCircuit,
    spec: &RoundSpec,
    ends: Ends,
) -> Result<(Vec<u32>, Vec<u32>), Error> {
    for b in [B::X, B::Z] {
        let mut targets: Vec<IVec2> = spec
            .resets
            .iter()
            .filter(|&&(_, rb)| rb == b)
            .map(|&(q, _)| q)
            .collect();
        if ends.init == Some(b) {
            targets.extend(spec.data_in());
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
        let (idx, qs): (Vec<usize>, Vec<IVec2>) = spec
            .measures
            .iter()
            .enumerate()
            .filter(|(_, (_, mb))| *mb == b)
            .map(|(i, &(q, _))| (i, q))
            .unzip();
        for (i, id) in idx.into_iter().zip(circuit.measure(b.measure_basis(), qs)) {
            ids[i] = id;
        }
    }
    let data_ids = match ends.readout {
        Some(b) => {
            circuit.tick();
            circuit.measure(b.measure_basis(), spec.data_out())
        }
        None => Vec::new(),
    };
    Ok((ids, data_ids))
}

fn propagate(mut p: Pauli2, layers: &[Vec<(IVec2, IVec2)>], backward: bool) -> Pauli2 {
    let order: Vec<&Vec<(IVec2, IVec2)>> = if backward {
        layers.iter().rev().collect()
    } else {
        layers.iter().collect()
    };
    for layer in order {
        for &(c, t) in layer {
            p.cx(c, t);
        }
    }
    p
}

type Logical = BTreeMap<(i32, i32), B>;

/// One round as a verified chunk. The last flow is the logical; the others are detector flows.
struct Round {
    chunk: Chunk,
    ids: Vec<u32>,
    data_ids: Vec<u32>,
    logical_out: Logical,
}

fn build_round(spec: &RoundSpec, logical_in: &Logical, ends: Ends) -> Result<Round, Error> {
    let mut circuit = CoordCircuit::new();
    let (ids, data_ids) = emit_round(&mut circuit, spec, ends)?;
    let reset_basis: HashMap<IVec2, B> = spec.resets.iter().copied().collect();
    let meas_basis: HashMap<IVec2, B> = spec.measures.iter().copied().collect();
    let id_of: HashMap<IVec2, u32> = spec
        .measures
        .iter()
        .zip(&ids)
        .map(|(&(q, _), &id)| (q, id))
        .collect();
    let data_id: HashMap<(i32, i32), u32> = spec
        .data_out()
        .iter()
        .zip(&data_ids)
        .map(|(q, &id)| ((q.x, q.y), id))
        .collect();
    let all_in = |p: &Logical, b: Option<B>| b.is_some_and(|b| p.values().all(|&v| v == b));

    let mut flows = Vec::new();
    for &(q, b) in &spec.measures {
        // Backward: what this measurement reads off the data at the start of the round.
        let (s_in, _) = split(
            &propagate(Pauli2::single(q, b), &spec.layers, true),
            &reset_basis,
        )?;
        let flow = if !s_in.is_empty() && ends.init.is_none() {
            Flow::new(pauli_map(&s_in), PauliMap::empty())
        } else if all_in(&s_in, ends.init) {
            Flow::new(PauliMap::empty(), PauliMap::empty())
        } else {
            continue;
        };
        flows.push(flow.with_measurements([id_of[&q]]).with_center(q));
    }
    for &(q, b) in &spec.resets {
        // Forward: what this reset prepares on the data at the end of the round.
        let (s_out, via) = split(
            &propagate(Pauli2::single(q, b), &spec.layers, false),
            &meas_basis,
        )?;
        let mut ms: Vec<u32> = via.iter().map(|v| id_of[v]).collect();
        let end = if ends.readout.is_none() {
            pauli_map(&s_out)
        } else if all_in(&s_out, ends.readout) {
            ms.extend(s_out.keys().map(|k| data_id[k]));
            PauliMap::empty()
        } else {
            continue;
        };
        flows.push(
            Flow::new(PauliMap::empty(), end)
                .with_measurements(ms)
                .with_center(q),
        );
    }
    // The logical: propagate forward; measured qubits it crosses join its measurement set.
    let mut p = Pauli2::default();
    for (&(x, y), &b) in logical_in {
        p = p.times(&Pauli2::single(IVec2::new(x, y), b));
    }
    let (logical_out, via) = split(&propagate(p, &spec.layers, false), &meas_basis)?;
    let mut ms: Vec<u32> = via.iter().map(|v| id_of[v]).collect();
    let start = match ends.init {
        Some(b) if all_in(logical_in, Some(b)) => PauliMap::empty(),
        Some(_) => return Err("the logical is not in the initial basis".into()),
        None => pauli_map(logical_in),
    };
    let end = match ends.readout {
        Some(b) if all_in(&logical_out, Some(b)) => {
            ms.extend(logical_out.keys().map(|k| data_id[k]));
            PauliMap::empty()
        }
        Some(_) => return Err("the logical is not in the readout basis".into()),
        None => pauli_map(&logical_out),
    };
    flows.push(Flow::new(start, end).with_measurements(ms));
    let chunk = Chunk { circuit, flows };
    chunk.verify_flows(None, None)?;
    Ok(Round {
        chunk,
        ids,
        data_ids,
        logical_out,
    })
}

fn compact_memory(top: &str, basis: B, d: i32, rounds: usize) -> Result<String, Error> {
    if d < 3 || d % 2 == 0 || rounds < 2 {
        return Err("need odd d >= 3 and at least 2 rounds".into());
    }
    let first = match top {
        "x-top" => {
            let (data, meas) = g_x_round(d);
            RoundSpec::from_measure_qubits(data, &meas)?
        }
        "z-top" => g_z_round(d)?,
        other => return Err(format!("{other}: want x-top or z-top").into()),
    };
    let second = first.inverse();
    // X-top: logical X down the left column, logical Z along the top row; G_X and G_X^-1 leave both
    // unchanged. Z-top: logical X along the top row, logical Z down the left column (one data qubit
    // per row, at x = 0 or 1). Under G_Z the logical X moves down a row per round until it reaches
    // the bottom boundary; the measurements it crosses join the observable (assumption A5).
    let data_in = first.data_in();
    let logical: Logical = data_in
        .iter()
        .filter(|q| match (top, basis) {
            ("x-top", B::X) | ("z-top", B::Z) => q.x <= 1,
            _ => q.y == 1,
        })
        .map(|q| ((q.x, q.y), basis))
        .collect();
    if logical.len() != d as usize {
        return Err(format!("logical has weight {} at d = {d}", logical.len()).into());
    }
    let mut full = CoordCircuit::new();
    let mut built = Vec::new();
    let mut observable = Vec::new();
    let mut current = logical;
    for r in 0..rounds {
        let spec = if r % 2 == 0 { &first } else { &second };
        let ends = Ends {
            init: (r == 0).then_some(basis),
            readout: (r + 1 == rounds).then_some(basis),
        };
        let round = build_round(spec, &current, ends)?;
        if r > 0 {
            full.tick();
        }
        let (global_ids, global_data) = emit_round(&mut full, spec, ends)?;
        let map: HashMap<u32, u32> = round
            .ids
            .iter()
            .chain(&round.data_ids)
            .copied()
            .zip(global_ids.iter().chain(&global_data).copied())
            .collect();
        let logical_flow = round.chunk.flows.last().expect("the logical flow");
        observable.extend(logical_flow.measurements.iter().map(|m| map[m]));
        current = round.logical_out.clone();
        built.push((round, map));
    }
    let mut engine: FlowEngine<'_, u32> = FlowEngine::new();
    for (round, map) in &built {
        let flows = &round.chunk.flows[..round.chunk.flows.len() - 1];
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
