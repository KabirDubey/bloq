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
//! (x-top: Fig. 5 left and (c); z-top: Fig. 5 right and (d)), and
//! `low-gadgets hadamard <x|z> <d> <z-top rounds> <x-top rounds> <out.stim>` (Fig. 7), and
//! `low-gadgets walking <x-top|z-top> <x|z> <d> <rounds> <out.stim>` (Fig. 48; one step per 2 rounds).

mod walk_template;

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
    /// Hadamards on data qubits, in the reset moment (before the CX layers).
    h_first: Vec<IVec2>,
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
            h_first: Vec::new(),
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
            h_first: Vec::new(),
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
        h_first: Vec::new(),
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
    fn h(&mut self, q: IVec2) {
        let (x, z) = self.get(q);
        self.set(q, (z, x));
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

/// The Pauli bits of `p` that a qubit in `allowed` would not accept (Y on any other qubit).
fn wrong_bits(p: &Pauli2, allowed: &HashMap<IVec2, B>) -> std::collections::BTreeSet<(i32, i32, bool)> {
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

/// `target` times the product of `items` that cancels its disallowed part, if one exists.
fn fix_by(target: Pauli2, items: &[Pauli2], allowed: &HashMap<IVec2, B>) -> Option<Pauli2> {
    let mut pivots: Vec<(std::collections::BTreeSet<(i32, i32, bool)>, Pauli2)> = Vec::new();
    let reduce = |mut w: std::collections::BTreeSet<(i32, i32, bool)>,
                  mut p: Pauli2,
                  pivots: &[(std::collections::BTreeSet<(i32, i32, bool)>, Pauli2)]| {
        while let Some(&lead) = w.iter().next() {
            let Some(pv) = pivots.iter().find(|pv| pv.0.iter().next() == Some(&lead)) else {
                break;
            };
            w = w.symmetric_difference(&pv.0).copied().collect();
            p = p.times(&pv.1);
        }
        (w, p)
    };
    for it in items {
        let (w, p) = reduce(wrong_bits(it, allowed), it.clone(), &pivots);
        if !w.is_empty() {
            pivots.push((w, p));
        }
    }
    let (w, p) = reduce(wrong_bits(&target, allowed), target, &pivots);
    w.is_empty().then_some(p)
}

/// The products of `items` that carry only `allowed` Paulis on the qubits `allowed` names: every
/// item that does so alone, plus a basis of the products of the others found by GF(2) elimination
/// on their disallowed parts. Walking corners (Fig. 48) measure some checks only as such products.
fn deterministic(
    items: Vec<(Vec<usize>, Pauli2)>,
    allowed: &HashMap<IVec2, B>,
) -> Vec<(Vec<usize>, Pauli2)> {
    let wrong = |p: &Pauli2| wrong_bits(p, allowed);
    let mut out = Vec::new();
    let mut pivots: Vec<(std::collections::BTreeSet<(i32, i32, bool)>, Vec<usize>, Pauli2)> = Vec::new();
    for (idx, p) in items {
        let mut cur = (wrong(&p), idx, p);
        loop {
            let Some(&lead) = cur.0.iter().next() else {
                cur.1.sort_unstable();
                out.push((cur.1, cur.2));
                break;
            };
            match pivots.iter().find(|pv| pv.0.iter().next() == Some(&lead)) {
                Some(pv) => {
                    cur.0 = cur.0.symmetric_difference(&pv.0).copied().collect();
                    let mut ids = cur.1.clone();
                    xor_extend_usize(&mut ids, &pv.1);
                    cur.1 = ids;
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

fn xor_extend_usize(a: &mut Vec<usize>, b: &[usize]) {
    for &x in b {
        if let Some(i) = a.iter().position(|&y| y == x) {
            a.swap_remove(i);
        } else {
            a.push(x);
        }
    }
}

/// The data and measure parts of a propagated product. Errors if a measure qubit carries a Pauli
/// that is not `allowed` there (it would not be deterministic) or a data qubit carries Y.
fn split(p: &Pauli2, allowed: &HashMap<IVec2, B>) -> Result<(Logical, Vec<IVec2>), Error> {
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
    if !spec.h_first.is_empty() {
        circuit.do_gate(GateType::H, spec.h_first.iter().copied())?;
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

/// Conjugate `p` through the round's unitary part (Hadamards, then CX layers), or back through it.
fn propagate(mut p: Pauli2, spec: &RoundSpec, backward: bool) -> Pauli2 {
    if !backward {
        spec.h_first.iter().for_each(|&q| p.h(q));
    }
    let order: Vec<&Vec<(IVec2, IVec2)>> = if backward {
        spec.layers.iter().rev().collect()
    } else {
        spec.layers.iter().collect()
    };
    for layer in order {
        for &(c, t) in layer {
            p.cx(c, t);
        }
    }
    if backward {
        spec.h_first.iter().for_each(|&q| p.h(q));
    }
    p
}

type Logical = BTreeMap<(i32, i32), B>;

/// Add `more` to the measurement set `ms` modulo 2.
fn xor_extend(ms: &mut Vec<u32>, more: &[u32]) {
    for m in more {
        match ms.iter().position(|x| x == m) {
            Some(i) => {
                ms.remove(i);
            }
            None => ms.push(*m),
        }
    }
}

/// One round as a verified chunk. The last flow is the logical; the others are detector flows.
struct Round {
    chunk: Chunk,
    ids: Vec<u32>,
    data_ids: Vec<u32>,
    logical_out: Logical,
    /// Single-qubit Paulis this round leaves fixed at its end, with the measurements that fix them.
    fixed: Vec<((i32, i32), B, Vec<u32>)>,
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
    let mut fixed = Vec::new();
    let back: Vec<(Vec<usize>, Pauli2)> = spec
        .measures
        .iter()
        .enumerate()
        .map(|(i, &(q, b))| (vec![i], propagate(Pauli2::single(q, b), spec, true)))
        .collect();
    for (idx, p) in deterministic(back, &reset_basis) {
        // Backward: what these measurements read off the data at the start of the round.
        let (s_in, _) = split(&p, &reset_basis)?;
        let flow = if !s_in.is_empty() && ends.init.is_none() {
            Flow::new(pauli_map(&s_in), PauliMap::empty())
        } else if all_in(&s_in, ends.init) {
            Flow::new(PauliMap::empty(), PauliMap::empty())
        } else {
            continue;
        };
        let ms: Vec<u32> = idx.iter().map(|&i| ids[i]).collect();
        flows.push(flow.with_measurements(ms).with_center(spec.measures[idx[0]].0));
    }
    // Forward: what each reset (or product of resets) prepares on the data at the end of the round.
    let fwd: Vec<(Vec<usize>, Pauli2)> = spec
        .resets
        .iter()
        .enumerate()
        .map(|(i, &(q, b))| (vec![i], propagate(Pauli2::single(q, b), spec, false)))
        .collect();
    let mut outs = Vec::new();
    for (idx, p) in deterministic(fwd, &meas_basis) {
        let q = spec.resets[idx[0]].0;
        let (s_out, via) = split(&p, &meas_basis)?;
        let ms: Vec<u32> = via.iter().map(|v| id_of[v]).collect();
        if let [(&k, &pb)] = s_out.iter().collect::<Vec<_>>()[..] {
            fixed.push((k, pb, ms.clone()));
        }
        outs.push((q, s_out, ms));
    }
    // A product with a qubit the round fixes (a reset that ends on one data qubit, such as the
    // corner of the square end cycle) drops that qubit, so it matches the next round's flows.
    let reduce = |s_out: &mut Logical, ms: &mut Vec<u32>| {
        if s_out.len() < 2 {
            return;
        }
        for (k, pb, fms) in &fixed {
            if s_out.get(k) == Some(pb) {
                s_out.remove(k);
                xor_extend(ms, fms);
            }
        }
    };
    for (q, mut s_out, mut ms) in outs {
        reduce(&mut s_out, &mut ms);
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
    // At a walking corner (Fig. 48) the logical may need a product of resets to stay deterministic.
    let resets: Vec<Pauli2> = spec
        .resets
        .iter()
        .map(|&(q, b)| propagate(Pauli2::single(q, b), spec, false))
        .collect();
    let p = propagate(p, spec, false);
    let p = fix_by(p.clone(), &resets, &meas_basis).unwrap_or(p);
    let (mut logical_out, via) = split(&p, &meas_basis)?;
    let mut ms: Vec<u32> = via.iter().map(|v| id_of[v]).collect();
    if ends.readout.is_none() {
        reduce(&mut logical_out, &mut ms);
    }
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
        fixed,
    })
}

/// The first round of a memory in the given orientation, and the logical of `basis` on its input
/// data. X-top: logical X down the left column, logical Z along the top row; G_X and G_X^-1 leave
/// both unchanged. Z-top: logical X along the top row, logical Z down the left column (one data
/// qubit per row, at x = 0 or 1). Under G_Z the logical X moves down a row per round until it
/// reaches the bottom boundary; the measurements it crosses join the observable (assumption A5).
fn orientation(top: &str, basis: B, d: i32) -> Result<(RoundSpec, Logical), Error> {
    if d < 3 || d % 2 == 0 {
        return Err("need odd d >= 3".into());
    }
    let first = match top {
        "x-top" => {
            let (data, meas) = g_x_round(d);
            RoundSpec::from_measure_qubits(data, &meas)?
        }
        "z-top" => g_z_round(d)?,
        other => return Err(format!("{other}: want x-top or z-top").into()),
    };
    let logical: Logical = first
        .data_in()
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
    Ok((first, logical))
}

/// Run `specs` in order: noiseless-basis preparation of the first round's data in `init`, readout
/// of the last round's data in `readout`. Every round is a verified chunk; `FlowEngine` fuses the
/// detector flows of consecutive rounds and the logical flows give the observable.
fn run_rounds(specs: &[RoundSpec], logical: Logical, init: B, readout: B) -> Result<String, Error> {
    let mut full = CoordCircuit::new();
    let mut built = Vec::new();
    let mut observable = Vec::new();
    let mut current = logical;
    for (r, spec) in specs.iter().enumerate() {
        let ends = Ends {
            init: (r == 0).then_some(init),
            readout: (r + 1 == specs.len()).then_some(readout),
        };
        if r > 0 {
            // The logical may not sit on a qubit this round resets; multiply it by a single-qubit
            // Pauli the previous round fixed there.
            let (prev, map): &(Round, HashMap<u32, u32>) = built.last().expect("a previous round");
            for &(q, _) in &spec.resets {
                let key = (q.x, q.y);
                let Some(&b) = current.get(&key) else {
                    continue;
                };
                let (_, _, ms) = prev
                    .fixed
                    .iter()
                    .find(|(k, fb, _)| *k == key && *fb == b)
                    .ok_or_else(|| format!("round {r} resets {q}, which holds the logical"))?;
                current.remove(&key);
                observable.extend(ms.iter().map(|m| map[m]));
            }
            full.tick();
        }
        let round = build_round(spec, &current, ends)?;
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
    // A measurement can join the observable twice (logical flow and a fixed Pauli): keep odd counts.
    let mut parity: BTreeMap<u32, bool> = BTreeMap::new();
    for m in observable {
        *parity.entry(m).or_default() ^= true;
    }
    let observable: Vec<u32> = parity
        .into_iter()
        .filter(|&(_, odd)| odd)
        .map(|(m, _)| m)
        .collect();
    // Drop flows that end only on qubits the next round resets: nothing reads them.
    let live: Vec<Vec<Flow>> = built
        .iter()
        .enumerate()
        .map(|(r, (round, _))| {
            let reset_next: HashSet<IVec2> = specs
                .get(r + 1)
                .map(|s| s.resets.iter().map(|&(q, _)| q).collect())
                .unwrap_or_default();
            round.chunk.flows[..round.chunk.flows.len() - 1]
                .iter()
                .filter(|f| f.end.is_empty() || !f.end.iter().all(|(q, _)| reset_next.contains(q)))
                .cloned()
                .collect()
        })
        .collect();
    let mut engine: FlowEngine<'_, u32> = FlowEngine::new();
    for (flows, (_, map)) in live.iter().zip(&built) {
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

fn compact_memory(top: &str, basis: B, d: i32, rounds: usize) -> Result<String, Error> {
    if rounds < 2 {
        return Err("need at least 2 rounds".into());
    }
    let (first, logical) = orientation(top, basis, d)?;
    let second = first.inverse();
    let specs: Vec<RoundSpec> = (0..rounds)
        .map(|r| {
            if r % 2 == 0 {
                first.clone()
            } else {
                second.clone()
            }
        })
        .collect();
    run_rounds(&specs, logical, basis, basis)
}

/// G_Z with its moved measurements put back in place: each one becomes the tick-4 CX followed by a
/// SWAP (a CXSwap), built as one extra boundary CX layer before tick 4 (SWAP . CX(c,t) =
/// CX(c,t) . CX(t,c)). The bottom-left data qubit (1, 2d-2) also moves to the unused corner
/// (0, 2d-1), reset in Z, so the end cycle is the square of Fig. 7 left (arXiv:2605.30455).
fn g_z_square_end(d: i32) -> Result<RoundSpec, Error> {
    let mut s = g_z_round(d)?;
    let mut extra = Vec::new();
    for (q, _) in s.measures.iter_mut() {
        if q.x % 2 == 1 {
            continue; // already on its slot
        }
        let m = *q + IVec2::new(1, -1);
        let &(c, t) = s.layers[3]
            .iter()
            .find(|&&(c, t)| (c, t) == (m, *q) || (c, t) == (*q, m))
            .ok_or_else(|| format!("no tick-4 CX between {m} and {q}"))?;
        extra.push((t, c));
        *q = m;
    }
    let slot = IVec2::new(1, 2 * d - 2);
    let corner = IVec2::new(0, 2 * d - 1);
    s.qubits.push(corner);
    s.resets.push((corner, B::Z));
    extra.push((slot, corner));
    s.layers[3].push((corner, slot));
    s.layers.insert(3, extra);
    for (k, layer) in s.layers.iter().enumerate() {
        let mut used = HashSet::new();
        if !layer.iter().all(|&(c, t)| used.insert(c) && used.insert(t)) {
            return Err(format!("square end: a qubit is used twice in CX layer {k}").into());
        }
    }
    Ok(s)
}

/// Timelike logical Hadamard (arXiv:2605.30455, Sec. 1, Fig. 7): `before` Z-top rounds, the last
/// with the square end cycle, then transversal H on the data in the next reset moment and `after`
/// X-top rounds (G_X first: starting with G_X^-1 leaves a stabilizer unmatched). Prepares the
/// logical in `basis` and reads it out in the other basis.
fn hadamard(basis: B, d: i32, before: usize, after: usize) -> Result<String, Error> {
    if before.is_multiple_of(2) || after < 1 {
        return Err("need an odd number of Z-top rounds and at least one X-top round".into());
    }
    let (gz, logical) = orientation("z-top", basis, d)?;
    let gz_inv = gz.inverse();
    let (xa, _) = orientation("x-top", basis, d)?;
    let xb = xa.inverse();
    let mut specs: Vec<RoundSpec> = (0..before - 1)
        .map(|r| {
            if r % 2 == 0 {
                gz.clone()
            } else {
                gz_inv.clone()
            }
        })
        .collect();
    let square = g_z_square_end(d)?;
    let mut first_x = xa.clone();
    let data: HashSet<IVec2> = square.data_out().into_iter().collect();
    first_x.h_first = first_x
        .data_in()
        .into_iter()
        .filter(|q| data.contains(q))
        .collect();
    specs.push(square);
    specs.push(first_x);
    specs.extend((1..after).map(|r| if r % 2 == 0 { xa.clone() } else { xb.clone() }));
    let flip = match basis {
        B::X => B::Z,
        B::Z => B::X,
    };
    run_rounds(&specs, logical, basis, flip)
}

/// Fold one doubled coordinate (a qubit or a gate centre, times 2) of a distance-d walk onto the
/// d = 5 template of Fig. 48. The half nearer the low edge keeps its distance to that edge, the
/// other half its distance to the high edge (2d); the bulk repeats every 4 (the X/Z checkerboard),
/// so the shift is always a multiple of 4 and keeps every qubit's role.
fn fold(c2: i32, d: i32) -> i32 {
    if c2 <= 2 * d {
        let mut c = c2;
        while c > 12 {
            c -= 8;
        }
        c
    } else {
        let mut c = c2 - 2 * (2 * d - 10);
        while c < 8 {
            c += 8;
        }
        c
    }
}

/// The two rounds of one walking step (Fig. 48, arXiv:2605.30455) at distance d, shifted right by
/// `step` lattice steps. Each round is read from the d = 5 template (ticks 0-5 and 6-11) by
/// folding every qubit and gate onto it. A CXSwap(c, t) becomes CX(t, c) then CX(c, t)
/// (stim's decomposition); the CX(t, c) goes into the previous CX layer when its qubits are free
/// there, otherwise into one extra layer (assumption A8 in the log).
fn walk_rounds(top: &str, d: i32, step: i32) -> Result<[RoundSpec; 2], Error> {
    let template = match top {
        "x-top" => walk_template::X_WALK,
        "z-top" => walk_template::Z_WALK,
        other => return Err(format!("{other}: want x-top or z-top").into()),
    };
    let basis = |c: char| if c == 'X' { B::X } else { B::Z };
    let shift = IVec2::new(2 * step, 0);
    let pos = |x: i32, y: i32| (fold(2 * x, d) / 2, fold(2 * y, d) / 2);
    let build = |ticks: &[walk_template::Tick]| -> Result<RoundSpec, Error> {
        let singles = |pick: &dyn Fn(&walk_template::Tick) -> &[((i32, i32), char)]| {
            let table: HashMap<(i32, i32), char> = pick(&ticks[0])
                .iter()
                .chain(pick(&ticks[5]))
                .copied()
                .collect();
            let mut out = Vec::new();
            for y in 0..=2 * d {
                for x in 0..=2 * d + 1 {
                    if let Some(&b) = table.get(&pos(x, y)) {
                        out.push((IVec2::new(x, y) + shift, basis(b)));
                    }
                }
            }
            out
        };
        let resets = singles(&|t| t.resets);
        let measures = singles(&|t| t.measures);
        let mut layers: Vec<Vec<(IVec2, IVec2)>> = Vec::new();
        let mut swaps: Vec<Vec<(IVec2, IVec2)>> = Vec::new();
        for tick in &ticks[1..5] {
            // Key: doubled centre and direction control -> target.
            let table: HashMap<((i32, i32), (i32, i32)), bool> = tick
                .gates
                .iter()
                .map(|&(s, c, t)| (((c.0 + t.0, c.1 + t.1), (t.0 - c.0, t.1 - c.1)), s))
                .collect();
            let (mut layer, mut swap) = (Vec::new(), Vec::new());
            for y in 0..=2 * d {
                for x in 0..=2 * d + 1 {
                    for dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                        let centre = (fold(2 * x + dir.0, d), fold(2 * y + dir.1, d));
                        if let Some(&s) = table.get(&(centre, dir)) {
                            let c = IVec2::new(x, y) + shift;
                            let t = c + IVec2::new(dir.0, dir.1);
                            layer.push((c, t));
                            if s {
                                swap.push((t, c));
                            }
                        }
                    }
                }
            }
            layers.push(layer);
            swaps.push(swap);
        }
        // Expand CXSwaps, last layer first so the indices stay valid.
        for k in (0..layers.len()).rev() {
            let extra = std::mem::take(&mut swaps[k]);
            if extra.is_empty() {
                continue;
            }
            let free = k > 0
                && extra.iter().all(|&(a, b)| {
                    layers[k - 1]
                        .iter()
                        .all(|&(c, t)| ![c, t].contains(&a) && ![c, t].contains(&b))
                });
            if free {
                layers[k - 1].extend(extra);
            } else {
                layers.insert(k, extra);
            }
        }
        let mut qubits: Vec<IVec2> = resets.iter().chain(&measures).map(|&(q, _)| q).collect();
        qubits.extend(layers.iter().flatten().flat_map(|&(c, t)| [c, t]));
        qubits.sort_by_key(|q| (q.y, q.x));
        qubits.dedup();
        for (k, layer) in layers.iter().enumerate() {
            let mut used = HashSet::new();
            if !layer.iter().all(|&(c, t)| used.insert(c) && used.insert(t)) {
                return Err(format!("walk: a qubit is used twice in CX layer {k}").into());
            }
        }
        Ok(RoundSpec {
            qubits,
            resets,
            h_first: Vec::new(),
            layers,
            measures,
        })
    };
    let a = build(&template[0..6])?;
    let b = build(&template[6..12])?;
    let n = (d * d) as usize;
    for (name, s) in [("A", &a), ("B", &b)] {
        if s.data_in().len() != n || s.data_out().len() != n {
            return Err(format!("walk round {name}: data in/out is not d^2 at d = {d}").into());
        }
    }
    Ok([a, b])
}

/// Patch walking (arXiv:2605.30455, Sec. 1.1.3, Fig. 48): `rounds` rounds, one lattice step to the
/// right every 2 rounds, preparing and reading out the logical in `basis` on the moving patch.
fn walking(top: &str, basis: B, d: i32, rounds: usize) -> Result<String, Error> {
    if d < 3 || d % 2 == 0 || rounds < 2 {
        return Err("need odd d >= 3 and at least 2 rounds".into());
    }
    let mut specs = Vec::new();
    for r in 0..rounds {
        let [a, b] = walk_rounds(top, d, (r / 2) as i32)?;
        specs.push(if r % 2 == 0 { a } else { b });
    }
    for (r, w) in specs.windows(2).enumerate() {
        let out: HashSet<IVec2> = w[0].data_out().into_iter().collect();
        let inn: HashSet<IVec2> = w[1].data_in().into_iter().collect();
        if out != inn {
            return Err(format!("walk: round {r} data out differs from round {} data in", r + 1).into());
        }
    }
    let logical: Logical = specs[0]
        .data_in()
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
    run_rounds(&specs, logical, basis, basis)
}

fn parse_basis(s: &str) -> Result<B, Error> {
    match s {
        "x" => Ok(B::X),
        "z" => Ok(B::Z),
        other => Err(format!("basis {other}: want x or z").into()),
    }
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, top, basis, d, rounds, out] if cmd == "compact-memory" => {
            let basis = parse_basis(basis)?;
            let text = compact_memory(top, basis, d.parse()?, rounds.parse()?)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        [cmd, basis, d, before, after, out] if cmd == "hadamard" => {
            let basis = parse_basis(basis)?;
            let text = hadamard(basis, d.parse()?, before.parse()?, after.parse()?)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        [cmd, top, basis, d, rounds, out] if cmd == "walking" => {
            let basis = parse_basis(basis)?;
            let text = walking(top, basis, d.parse()?, rounds.parse()?)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        _ => Err(concat!(
            "usage: low-gadgets compact-memory <x-top|z-top> <x|z> <d> <rounds> <out.stim>\n",
            "       low-gadgets hadamard <x|z> <d> <z-top rounds> <x-top rounds> <out.stim>\n",
            "       low-gadgets walking <x-top|z-top> <x|z> <d> <rounds> <out.stim>"
        )
        .into()),
    }
}
