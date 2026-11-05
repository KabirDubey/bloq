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
//! `low-gadgets walking <x-top|z-top> <x|z> <d> <rounds> <out.stim>` (Fig. 48; one step per 2 rounds), and
//! `low-gadgets surgery <x-top|z-top> <h|v> <x|z> <d> <out.stim>` (Figs. 50, 51; H-bridge of width 0).
//! `low-gadgets h-bridge <x-top|z-top> <h|v> <x|z> <d> <k> <out.stim>` (Fig. 9: k bridge patches; G6).

#[rustfmt::skip]
mod surgery_template;
#[rustfmt::skip]
mod walk_template;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use bloq_circuit::{
    Chunk, CoordCircuit, Flow, FlowEngine, FlowMeasurements, GateType, OffsetFlows, Pauli,
    PauliBasis, PauliMap,
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
    /// The round mirrored along the `\` diagonal: (x, y) -> (y, x), bases and gate roles kept.
    fn reflected(&self) -> Self {
        let r = |q: IVec2| IVec2::new(q.y, q.x);
        let mut qubits: Vec<IVec2> = self.qubits.iter().map(|&q| r(q)).collect();
        qubits.sort_by_key(|q| (q.y, q.x));
        Self {
            qubits,
            resets: self.resets.iter().map(|&(q, b)| (r(q), b)).collect(),
            h_first: self.h_first.iter().map(|&q| r(q)).collect(),
            layers: (self.layers.iter())
                .map(|l| l.iter().map(|&(c, t)| (r(c), r(t))).collect())
                .collect(),
            measures: self.measures.iter().map(|&(q, b)| (r(q), b)).collect(),
        }
    }
    /// The first two CX layers of `self` (with its resets) followed by the last two of `other`
    /// (with its measurements): a round that changes gate sequence at the mid cycle. Both rounds
    /// must use the same qubits.
    fn then_at_mid(&self, other: &Self) -> Result<Self, Error> {
        let a: HashSet<IVec2> = self.qubits.iter().copied().collect();
        let b: HashSet<IVec2> = other.qubits.iter().copied().collect();
        if a != b || self.layers.len() != 4 || other.layers.len() != 4 {
            return Err("mid-cycle switch: the rounds differ in qubits or depth".into());
        }
        Ok(Self {
            qubits: self.qubits.clone(),
            resets: self.resets.clone(),
            h_first: self.h_first.clone(),
            layers: [&self.layers[..2], &other.layers[2..]].concat(),
            measures: other.measures.clone(),
        })
    }
    /// The round moved by `o`.
    fn shifted(&self, o: IVec2) -> Self {
        Self {
            qubits: self.qubits.iter().map(|&q| q + o).collect(),
            resets: self.resets.iter().map(|&(q, b)| (q + o, b)).collect(),
            h_first: self.h_first.iter().map(|&q| q + o).collect(),
            layers: (self.layers.iter())
                .map(|l| l.iter().map(|&(c, t)| (c + o, t + o)).collect())
                .collect(),
            measures: self.measures.iter().map(|&(q, b)| (q + o, b)).collect(),
        }
    }
    /// The part of the round on the qubits `keep` accepts; an error if a gate leaves that set.
    fn restricted(&self, keep: &dyn Fn(IVec2) -> bool) -> Result<Self, Error> {
        if let Some(&(c, t)) = (self.layers.iter().flatten()).find(|&&(c, t)| keep(c) != keep(t)) {
            return Err(format!("restrict: gate {c} -> {t} crosses the cut").into());
        }
        Ok(Self {
            qubits: self.qubits.iter().copied().filter(|&q| keep(q)).collect(),
            resets: self
                .resets
                .iter()
                .copied()
                .filter(|&(q, _)| keep(q))
                .collect(),
            h_first: self.h_first.iter().copied().filter(|&q| keep(q)).collect(),
            layers: (self.layers.iter())
                .map(|l| l.iter().copied().filter(|&(c, _)| keep(c)).collect())
                .collect(),
            measures: self
                .measures
                .iter()
                .copied()
                .filter(|&(q, _)| keep(q))
                .collect(),
        })
    }
    /// The part of the round on one patch: the qubits `slot` accepts, and the qubits outside it
    /// (seam measure qubits) whose gates all reach into it.
    fn patch(&self, slot: &dyn Fn(IVec2) -> bool) -> Result<Self, Error> {
        let mut partners: HashMap<IVec2, Vec<IVec2>> = HashMap::new();
        for &(c, t) in self.layers.iter().flatten() {
            partners.entry(c).or_default().push(t);
            partners.entry(t).or_default().push(c);
        }
        let keep: HashSet<IVec2> = (self.qubits.iter().copied())
            .filter(|&q| slot(q) || partners.get(&q).is_some_and(|p| p.iter().all(|&r| slot(r))))
            .collect();
        self.restricted(&|q| keep.contains(&q))
    }
    /// Two rounds on disjoint qubits run side by side.
    fn union(&self, other: &Self) -> Result<Self, Error> {
        let mine: HashSet<IVec2> = self.qubits.iter().copied().collect();
        if let Some(q) = other.qubits.iter().find(|q| mine.contains(q)) {
            return Err(format!("union: both rounds use {q}").into());
        }
        let mut qubits = self.qubits.clone();
        qubits.extend(&other.qubits);
        qubits.sort_by_key(|q| (q.y, q.x));
        let cat = |a: &[(IVec2, B)], b: &[(IVec2, B)]| [a, b].concat();
        Ok(Self {
            qubits,
            resets: cat(&self.resets, &other.resets),
            h_first: [&self.h_first[..], &other.h_first[..]].concat(),
            layers: (self.layers.iter().zip(&other.layers))
                .map(|(a, b)| [&a[..], &b[..]].concat())
                .collect(),
            measures: cat(&self.measures, &other.measures),
        })
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
    g_z_rect(d, d)
}

/// `g_z_round` on a patch of w x h data qubits (the merged patch of a lattice surgery). The rules
/// are those of the square patch with every phase taken from the checkerboard parity a + b, which
/// at w = h = d (odd) gives back the square rules (assumption A12 in the log).
fn g_z_rect(w: i32, h: i32) -> Result<RoundSpec, Error> {
    let qubits: Vec<IVec2> = (0..2 * h)
        .flat_map(|y| (0..2 * w).map(move |x| IVec2::new(x, y)))
        .filter(|q| (q.x + q.y) % 2 == 1 && *q != IVec2::new(0, 2 * h - 1))
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
    for b in 0..h {
        for a in 0..w {
            let m = IVec2::new(2 * a + 1, 2 * b);
            let even = (a + b) % 2 == 0;
            // Ticks 1 and 4: the SW neighbour.
            if b == h - 1 {
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
                if a < w - 1 {
                    add(if a % 2 == 0 { 1 } else { 2 }, m, SE, true);
                }
            } else if a == w - 1 {
                add(if (w + b) % 2 == 0 { 1 } else { 2 }, m, NW, false);
            } else if even {
                add(1, m, SE, true);
                add(2, m, NW, false);
            } else {
                add(1, m, NW, false);
                add(2, m, SE, true);
            }
            // Resets and measurements.
            if a == 0 && b < h - 1 {
                let moved = IVec2::new(0, 2 * b + 1);
                let (r, me) = if b % 2 == 0 { (m, moved) } else { (moved, m) };
                resets.push((r, B::X));
                measures.push((me, B::X));
            } else if b == h - 1 && a > 0 {
                let moved = IVec2::new(2 * a, 2 * h - 1);
                let (r, me) = if (a + b) % 2 == 1 {
                    (m, moved)
                } else {
                    (moved, m)
                };
                resets.push((r, B::Z));
                measures.push((me, B::Z));
            } else if a > 0 || b < h - 1 {
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

/// The prior-art Z-top round of Fig. 54 (arXiv:2605.30455, after Yuga2026NoHook) for any odd d,
/// in a (d+1) x (d+1) measure grid: data qubits at (odd, odd) in 1..2d, measure qubits at
/// (even, even) in 0..=2d except the bottom-left corner (0, 2d). Every measure qubit resets and
/// measures in place, and runs the same four steps, cut to the data qubits that exist; the step
/// directions and the reset and measure bases depend only on the checkerboard parity
/// ((x + y) / 2) % 2, with the left column (X) and bottom row (Z) as exceptions. Read off the
/// d = 5 figure's vector data: at d = 5 this gives its 100 CX gates, 35 resets and 35 measurements.
fn g_z_big(d: i32) -> Result<RoundSpec, Error> {
    let n = 2 * d;
    let data: Vec<IVec2> = (0..d)
        .flat_map(|j| (0..d).map(move |i| IVec2::new(2 * i + 1, 2 * j + 1)))
        .collect();
    let exists: HashSet<IVec2> = data.iter().copied().collect();
    let mut qubits = data.clone();
    let mut layers = vec![Vec::new(); 4];
    let mut resets = Vec::new();
    let mut measures = Vec::new();
    for y in (0..=n).step_by(2) {
        for x in (0..=n).step_by(2) {
            let m = IVec2::new(x, y);
            if x == 0 && y == n {
                continue;
            }
            qubits.push(m);
            let odd = (x + y) / 2 % 2 == 1;
            let (r, me) = if x == 0 {
                (B::X, B::X)
            } else if y == n {
                (B::Z, B::Z)
            } else if odd {
                (B::X, B::Z)
            } else {
                (B::Z, B::X)
            };
            resets.push((m, r));
            measures.push((m, me));
            // (tick, offset, measure qubit controls)
            let se = IVec2::new(1, 1);
            let steps = [
                (0, SW, odd),
                (if odd { 1 } else { 2 }, se, true),
                (if odd { 2 } else { 1 }, NW, false),
                (3, SW, !odd),
            ];
            for (tick, o, ctrl) in steps {
                let q = m + o;
                if exists.contains(&q) {
                    layers[tick].push(if ctrl { (m, q) } else { (q, m) });
                }
            }
        }
    }
    qubits.sort_by_key(|q| (q.y, q.x));
    for (k, layer) in layers.iter().enumerate() {
        let mut used = HashSet::new();
        if !layer.iter().all(|&(c, t)| used.insert(c) && used.insert(t)) {
            return Err(format!("big Z-top: a qubit is used twice in CX layer {k}").into());
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
fn wrong_bits(
    p: &Pauli2,
    allowed: &HashMap<IVec2, B>,
) -> std::collections::BTreeSet<(i32, i32, bool)> {
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
    let reduce =
        |mut w: std::collections::BTreeSet<(i32, i32, bool)>,
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

/// As `fix_by`, returning the tagged items the fix uses (an item tagged `None` may be used freely).
fn fix_by_index(
    target: &Pauli2,
    items: &[(Option<usize>, Pauli2)],
    allowed: &HashMap<IVec2, B>,
) -> Option<Vec<usize>> {
    type Bits = std::collections::BTreeSet<(i32, i32, bool)>;
    let mut pivots: Vec<(Bits, BTreeSet<usize>)> = Vec::new();
    let reduce = |mut w: Bits, mut used: BTreeSet<usize>, pivots: &[(Bits, BTreeSet<usize>)]| {
        while let Some(&lead) = w.iter().next() {
            let Some(pv) = pivots.iter().find(|pv| pv.0.iter().next() == Some(&lead)) else {
                break;
            };
            w = w.symmetric_difference(&pv.0).copied().collect();
            used = used.symmetric_difference(&pv.1).copied().collect();
        }
        (w, used)
    };
    for (tag, it) in items {
        let (w, used) = reduce(
            wrong_bits(it, allowed),
            tag.into_iter().copied().collect(),
            &pivots,
        );
        if !w.is_empty() {
            pivots.push((w, used));
        }
    }
    let (w, used) = reduce(wrong_bits(target, allowed), BTreeSet::new(), &pivots);
    w.is_empty().then(|| used.into_iter().collect())
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
    let mut pivots: Vec<(
        std::collections::BTreeSet<(i32, i32, bool)>,
        Vec<usize>,
        Pauli2,
    )> = Vec::new();
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

/// `l` times a product of `born` (Pauli, measurements) with no support on the qubits `region`
/// accepts, and the measurements of that product (GF(2) elimination on the region's bits).
fn clear_region(
    l: &Logical,
    born: &[(Logical, Vec<u32>)],
    region: &dyn Fn(IVec2) -> bool,
) -> Option<(Logical, Vec<u32>)> {
    type Bits = BTreeSet<(i32, i32, bool)>;
    let to_pauli = |l: &Logical| {
        (l.iter()).fold(Pauli2::default(), |p, (&(x, y), &b)| {
            p.times(&Pauli2::single(IVec2::new(x, y), b))
        })
    };
    let bits = |p: &Pauli2| -> Bits {
        p.0.iter()
            .filter(|(k, _)| region(IVec2::new(k.0, k.1)))
            .flat_map(|(&(x, y), &(px, pz))| [((x, y, false), px), ((x, y, true), pz)])
            .filter(|&(_, on)| on)
            .map(|(k, _)| k)
            .collect()
    };
    type Pivot = (Bits, Pauli2, Vec<u32>);
    let reduce = |mut w: Bits, mut p: Pauli2, mut ms: Vec<u32>, pivots: &[Pivot]| {
        while let Some(&lead) = w.iter().next() {
            let Some(pv) = pivots.iter().find(|pv| pv.0.iter().next() == Some(&lead)) else {
                break;
            };
            w = w.symmetric_difference(&pv.0).copied().collect();
            p = p.times(&pv.1);
            xor_extend(&mut ms, &pv.2);
        }
        (w, p, ms)
    };
    let mut pivots: Vec<Pivot> = Vec::new();
    for (s, ms) in born {
        let p = to_pauli(s);
        let (w, p, ms) = reduce(bits(&p), p, ms.clone(), &pivots);
        if !w.is_empty() {
            pivots.push((w, p, ms));
        }
    }
    let p = to_pauli(l);
    let (w, p, ms) = reduce(bits(&p), p, Vec::new(), &pivots);
    if !w.is_empty() {
        return None;
    }
    let mut out = Logical::new();
    for (q, b) in p.iter() {
        out.insert((q.x, q.y), b?);
    }
    Some((out, ms))
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
    /// Every product this round's resets prepare on the output data, with the measurements that
    /// fix its value (the stabilizers measured this round).
    born: Vec<(Logical, Vec<u32>)>,
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
        flows.push(
            flow.with_measurements(ms)
                .with_center(spec.measures[idx[0]].0),
        );
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
    let mut born = Vec::new();
    for (q, mut s_out, mut ms) in outs {
        reduce(&mut s_out, &mut ms);
        born.push((s_out.clone(), ms.clone()));
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
        born,
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
        "z-top-big" => g_z_big(d)?,
        other => return Err(format!("{other}: want x-top, z-top or z-top-big").into()),
    };
    let logical: Logical = first
        .data_in()
        .iter()
        .filter(|q| match (top, basis) {
            ("x-top", B::X) | ("z-top" | "z-top-big", B::Z) => q.x <= 1,
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
    run_rounds_multi(specs, &[logical], init, readout, &|_, _| false)
}

/// As `run_rounds`, with one observable per entry of `logicals`.
fn run_rounds_multi(
    specs: &[RoundSpec],
    logicals: &[Logical],
    init: B,
    readout: B,
    may_start: &dyn Fn(usize, &Flow) -> bool,
) -> Result<String, Error> {
    run_rounds_deform(specs, logicals, init, readout, may_start, &[])
}

/// A logical moved at the end of round `round`: observable `observable` is multiplied by a
/// product of the stabilizers that round measures (and single Paulis its resets fix) that turns
/// it into the target. Of the entries with one `group`, the first that works applies; an error
/// if none does.
struct Deform<'a> {
    group: usize,
    round: usize,
    observable: usize,
    target: Target<'a>,
}

enum Target<'a> {
    /// The logical times this product (which must be one of the round's stabilizers).
    Times(&'a Logical),
    /// Any logical with no support on the qubits the region accepts.
    Off(&'a dyn Fn(IVec2) -> bool),
}

/// As `run_rounds_multi`, moving logicals as `deform` says (the CX of G7 moves a logical across
/// a merged patch, so it can pass the next merge; A20 in the log).
fn run_rounds_deform(
    specs: &[RoundSpec],
    logicals: &[Logical],
    init: B,
    readout: B,
    may_start: &dyn Fn(usize, &Flow) -> bool,
    deform: &[Deform<'_>],
) -> Result<String, Error> {
    let mut full = CoordCircuit::new();
    let mut built: Vec<(Round, HashMap<u32, u32>)> = Vec::new();
    let mut observables: Vec<Vec<u32>> = vec![Vec::new(); logicals.len()];
    let mut currents: Vec<Logical> = logicals.to_vec();
    let mut applied: HashSet<usize> = HashSet::new();
    for (r, spec) in specs.iter().enumerate() {
        let ends = Ends {
            init: (r == 0).then_some(init),
            readout: (r + 1 == specs.len()).then_some(readout),
        };
        if r > 0 {
            // A logical may not sit on a qubit this round resets; multiply it by a single-qubit
            // Pauli the previous round fixed there.
            let (prev, map): &(Round, HashMap<u32, u32>) = built.last().expect("a previous round");
            for (current, observable) in currents.iter_mut().zip(&mut observables) {
                for &(q, _) in &spec.resets {
                    let key = (q.x, q.y);
                    let Some(&b) = current.get(&key) else {
                        continue;
                    };
                    let (_, _, ms) = prev
                        .fixed
                        .iter()
                        .find(|(k, fb, _)| *k == key && *fb == b)
                        .ok_or_else(|| format!("round {r} resets {q}, which holds a logical"))?;
                    current.remove(&key);
                    observable.extend(ms.iter().map(|m| map[m]));
                }
            }
            // A merge across a bridge (G6) brings in data the previous round reset: a logical that
            // crosses it is deterministic only with the bridge's fixed Paulis multiplied in.
            let meas_basis: HashMap<IVec2, B> = spec.measures.iter().copied().collect();
            let data_in: HashSet<(i32, i32)> = spec.data_in().iter().map(|q| (q.x, q.y)).collect();
            let resets: Vec<Pauli2> = spec
                .resets
                .iter()
                .map(|&(q, b)| propagate(Pauli2::single(q, b), spec, false))
                .collect();
            for (current, observable) in currents.iter_mut().zip(&mut observables) {
                let mut p = Pauli2::default();
                for (&(x, y), &b) in current.iter() {
                    p = p.times(&Pauli2::single(IVec2::new(x, y), b));
                }
                let p = propagate(p, spec, false);
                if fix_by(p.clone(), &resets, &meas_basis).is_some() {
                    continue;
                }
                let cands: Vec<_> = prev
                    .fixed
                    .iter()
                    .filter(|(k, _, _)| !current.contains_key(k) && data_in.contains(k))
                    .collect();
                let mut items: Vec<(Option<usize>, Pauli2)> =
                    resets.iter().map(|p| (None, p.clone())).collect();
                items.extend(cands.iter().enumerate().map(|(i, (k, b, _))| {
                    let q = IVec2::new(k.0, k.1);
                    (Some(i), propagate(Pauli2::single(q, *b), spec, false))
                }));
                let Some(used) = fix_by_index(&p, &items, &meas_basis) else {
                    continue;
                };
                for i in used {
                    let (k, b, ms) = cands[i];
                    current.insert(*k, *b);
                    observable.extend(ms.iter().map(|m| map[m]));
                }
            }
            full.tick();
        }
        let (global_ids, global_data) = emit_round(&mut full, spec, ends)?;
        let mut first = None;
        for (k, (current, observable)) in currents.iter_mut().zip(&mut observables).enumerate() {
            let round = build_round(spec, current, ends).map_err(|e| format!("round {r}: {e}"))?;
            let map: HashMap<u32, u32> = round
                .ids
                .iter()
                .chain(&round.data_ids)
                .copied()
                .zip(global_ids.iter().chain(&global_data).copied())
                .collect();
            let logical_flow = round.chunk.flows.last().expect("the logical flow");
            observable.extend(logical_flow.measurements.iter().map(|m| map[m]));
            *current = round.logical_out.clone();
            if k == 1 && std::env::var_os("LG_TRACE").is_some() {
                let mut v: Vec<_> = current.iter().map(|(q, b)| (*q, *b == B::X)).collect();
                v.sort();
                eprintln!("trace: round {r} out {v:?}");
            }
            for m in deform.iter().filter(|m| m.round == r && m.observable == k) {
                if applied.contains(&m.group) {
                    continue;
                }
                // The checks this round measures and the single Paulis its resets fix.
                let mut items = round.born.clone();
                items.extend(
                    round
                        .fixed
                        .iter()
                        .map(|(q, b, ms)| (Logical::from([(*q, *b)]), ms.clone())),
                );
                let target = match m.target {
                    Target::Times(t) => t,
                    Target::Off(region) => {
                        let Some((moved, ms)) = clear_region(current, &items, region) else {
                            if std::env::var_os("LG_DEBUG").is_some() {
                                let mut v: Vec<_> = current.iter().collect();
                                v.sort_by_key(|(k, _)| **k);
                                eprintln!("debug: round {r} off fails on {v:?}");
                            }
                            continue;
                        };
                        applied.insert(m.group);
                        if std::env::var_os("LG_DEBUG").is_some() {
                            let mut v: Vec<_> = moved.iter().collect();
                            v.sort_by_key(|(k, _)| **k);
                            eprintln!("debug: round {r} group {} moved to {v:?}", m.group);
                        }
                        *current = moved;
                        observable.extend(ms.iter().map(|m| map[m]));
                        continue;
                    }
                };
                let Some((rest, ms)) = clear_region(target, &items, &|_| true) else {
                    continue;
                };
                debug_assert!(rest.is_empty());
                for (&q, &b) in target {
                    match current.get(&q) {
                        Some(&c) if c == b => {
                            current.remove(&q);
                        }
                        Some(_) => return Err(format!("deform: mixed Pauli at {q:?}").into()),
                        None => {
                            current.insert(q, b);
                        }
                    }
                }
                applied.insert(m.group);
                observable.extend(ms.iter().map(|m| map[m]));
            }
            first.get_or_insert((round, map));
        }
        built.push(first.expect("at least one logical"));
        for m in deform
            .iter()
            .filter(|m| m.round == r && !applied.contains(&m.group))
        {
            if !deform.iter().any(|n| n.group == m.group && n.round > r) {
                return Err(
                    format!("round {r}: no target of deformation {} works", m.group).into(),
                );
            }
        }
    }
    // A measurement can join an observable twice (logical flow and a fixed Pauli): keep odd counts.
    let observables: Vec<(u32, Vec<u32>)> = observables
        .into_iter()
        .enumerate()
        .map(|(k, observable)| {
            let mut parity: BTreeMap<u32, bool> = BTreeMap::new();
            for m in observable {
                *parity.entry(m).or_default() ^= true;
            }
            let ms = parity
                .into_iter()
                .filter(|&(_, odd)| odd)
                .map(|(m, _)| m)
                .collect();
            (k as u32, ms)
        })
        .collect();
    // Drop flows that end only on qubits the next round resets: nothing reads them.
    let mut live: Vec<Vec<Flow>> = built
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
    fuse_products(&mut live);
    let mut engine: FlowEngine<'_, u32> = FlowEngine::new();
    for (r, (flows, (_, map))) in live.iter().zip(&built).enumerate() {
        // A check measured for the first time (a merge's seam) has an input no earlier round
        // outputs: it is random, so it starts no detector. Only `may_start` checks may do so.
        let ends: HashSet<String> = match r {
            0 => HashSet::new(),
            _ => live[r - 1].iter().map(|f| f.end.to_string()).collect(),
        };
        let mut skip = false;
        for f in flows.iter().filter(|f| !f.start.is_empty()) {
            if r > 0 && !ends.contains(&f.start.to_string()) {
                if !may_start(r, f) {
                    return Err(format!("round {r}: unmatched flow input {}", f.start).into());
                }
                skip = true;
            }
        }
        if std::env::var_os("LOW_DEBUG").is_some() {
            for f in flows {
                eprintln!(
                    "r{r} start {} end {} m{}",
                    f.start,
                    f.end,
                    f.measurements.len()
                );
            }
        }
        let parts = [OffsetFlows::new(flows, IVec2::ZERO)];
        let appended = if skip {
            engine.append_group_skipping_unmatched_inputs(&parts, |_, m| map[m])
        } else {
            engine.append_group_allowing_dangling(&parts, |_, m| map[m])
        };
        appended.map_err(|e| format!("round {r}: {e}"))?;
    }
    let detectors: Vec<Vec<u32>> = engine
        .finish()?
        .into_iter()
        .map(|f| {
            assert!(!f.sign, "every flow here is positive");
            f.measurements.into_iter().collect()
        })
        .collect();
    Ok(emit_annotated_stim(&full, &detectors, &observables)?)
}

/// Lattice surgery (Figs. 50, 51 of arXiv:2605.30455) measures a seam check for the first time as
/// the product of two boundary checks of the round before. The flow engine matches boundaries
/// exactly, so add that product to the earlier round as one more flow (GF(2) elimination over the
/// earlier round's freshly measured checks inside the seam check's support). Then drop each flow
/// whose end no flow of the next round reads, so a boundary that comes back later (after the split)
/// does not meet a stale copy of itself.
fn fuse_products(live: &mut [Vec<Flow>]) {
    for r in 1..live.len() {
        // Merge: a check of round r reads the product of checks round r - 1 measured.
        let ends: HashSet<String> = live[r - 1].iter().map(|f| f.end.to_string()).collect();
        let mut extra = Vec::new();
        for f in live[r]
            .iter()
            .filter(|f| !f.start.is_empty() && !ends.contains(&f.start.to_string()))
        {
            let parts = live[r - 1]
                .iter()
                .filter(|g| g.start.is_empty())
                .map(|g| (&g.end, &g.measurements));
            if let Some(m) = product_of(&f.start, parts) {
                extra.push(Flow::new(PauliMap::empty(), f.start.clone()).with_measurements(m));
            }
        }
        live[r - 1].extend(extra);
        // Split: the product of checks round r measures reads a check of round r - 1.
        let starts: HashSet<String> = live[r].iter().map(|f| f.start.to_string()).collect();
        let mut extra = Vec::new();
        for f in live[r - 1]
            .iter()
            .filter(|f| !f.end.is_empty() && !starts.contains(&f.end.to_string()))
        {
            let parts = live[r]
                .iter()
                .filter(|g| g.end.is_empty() && !g.start.is_empty())
                .map(|g| (&g.start, &g.measurements));
            if let Some(m) = product_of(&f.end, parts) {
                extra.push(Flow::new(f.end.clone(), PauliMap::empty()).with_measurements(m));
            }
        }
        live[r].extend(extra);
    }
    for r in 0..live.len().saturating_sub(1) {
        let read: HashSet<String> = live[r + 1].iter().map(|f| f.start.to_string()).collect();
        live[r].retain(|f| f.end.is_empty() || read.contains(&f.end.to_string()));
    }
}

/// The measurement parity of a product of `parts` (each inside `target`'s support) equal to
/// `target`, by Gaussian elimination; `None` if there is none.
fn product_of<'a>(
    target: &PauliMap,
    parts: impl Iterator<Item = (&'a PauliMap, &'a FlowMeasurements)>,
) -> Option<BTreeSet<u32>> {
    let support: HashSet<IVec2> = target.iter().map(|(q, _)| *q).collect();
    let mut pivots: Vec<(PauliMap, BTreeSet<u32>)> = Vec::new();
    let reduce = |mut p: PauliMap, mut m: BTreeSet<u32>, pivots: &[(PauliMap, BTreeSet<u32>)]| {
        for (pp, pm) in pivots {
            let lead = pp.iter().next().expect("pivots are non-empty");
            if p.get(lead.0) == Some(lead.1) {
                p = &p ^ pp;
                m = m.symmetric_difference(pm).copied().collect();
            }
        }
        (p, m)
    };
    let mut used = 0;
    for (p, ms) in parts.filter(|(p, _)| p.iter().all(|(q, _)| support.contains(q))) {
        let (p, m) = reduce(p.clone(), ms.iter().copied().collect(), &pivots);
        if !p.is_empty() {
            pivots.push((p, m));
            used += 1;
        }
    }
    let (rest, m) = reduce(target.clone(), BTreeSet::new(), &pivots);
    (rest.is_empty() && used > 1).then_some(m)
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

/// Z-top memory that idles as a reflected X-top (arXiv:2605.30455, Sec. 1.1.1, "Memory"): one
/// G_Z round, then at the next mid cycle the X-top sequence mirrored along the `\` diagonal, which
/// alternates for `rounds - 4` rounds, then back to G_Z at a mid cycle and one last Z-top round.
/// The four ways to pair the halves (G or G^-1 on each side of each switch) are tried in turn and
/// the first that gives a valid circuit is kept (assumption A24 in the log).
fn reflect_memory(basis: B, d: i32, rounds: usize) -> Result<String, Error> {
    if rounds < 5 {
        return Err("need at least 5 rounds".into());
    }
    let (gz, logical) = orientation("z-top", basis, d)?;
    let gz_inv = gz.inverse();
    let (gx, _) = orientation("x-top", basis, d)?;
    let gx = gx.reflected();
    let gx_inv = gx.inverse();
    let mut last = None;
    for (enter, leave) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        // Round 0 is G_Z, so round 1 starts as G_Z^-1.
        let (x_in, x_alt) = if enter == 0 {
            (&gx, &gx_inv)
        } else {
            (&gx_inv, &gx)
        };
        let mut specs = vec![gz.clone(), gz_inv.then_at_mid(x_in)?];
        for k in 0..rounds - 4 {
            specs.push(if k % 2 == 0 { x_alt } else { x_in }.clone());
        }
        // The X-top round that would come next, and the Z-top pair after the switch.
        let x_next = if (rounds - 4) % 2 == 0 { x_alt } else { x_in };
        let (z_out, z_last) = if leave == 0 {
            (&gz, &gz_inv)
        } else {
            (&gz_inv, &gz)
        };
        specs.push(x_next.then_at_mid(z_out)?);
        specs.push(z_last.clone());
        match run_rounds(&specs, logical.clone(), basis, basis) {
            Ok(text) => {
                eprintln!("reflect: enter {enter} leave {leave}");
                return Ok(text);
            }
            Err(e) => {
                eprintln!("reflect: enter {enter} leave {leave}: {e}");
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| "no pairing works".into()))
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
            return Err(format!(
                "walk: round {r} data out differs from round {} data in",
                r + 1
            )
            .into());
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

/// Fold one doubled coordinate of a distance-d lattice surgery onto the d = 5 template along an
/// axis with `edges` patch edges at 0, 2d, 4d, ... (doubled 0, 4d, ...; the seam is the middle
/// one). Each point keeps its offset to the nearest edge; the bulk repeats every 4 (doubled 8),
/// so every shift is a multiple of 8 for odd d and keeps each qubit's role.
fn fold_axis(c2: i32, d: i32, edges: i32) -> i32 {
    let k = ((c2 + 2 * d) / (4 * d)).clamp(0, edges - 1);
    let mut t = c2 - 4 * d * k + 20 * k;
    while t > 20 * k + 10 {
        t -= 8;
    }
    while t < 20 * k - 10 {
        t += 8;
    }
    t
}

/// One round (6 ticks: resets, 4 CX layers, measurements) of a d = 5 merge or split figure, folded
/// to distance d. `horizontal`: the two patches sit side by side (seam along y). `bridge`: the
/// width (in coordinate units, a multiple of 4) of the bridge between the patches (G6). A
/// `merged` round fills the bridge with copies of the bulk just left of (or above) the seam, so the
/// figure's seam sits at the second patch's edge; a separate round leaves it empty.
fn surgery_round(
    ticks: &[walk_template::Tick],
    d: i32,
    horizontal: bool,
    bridge: i32,
    merged: bool,
) -> Result<RoundSpec, Error> {
    if bridge % 4 != 0 {
        return Err("surgery: the bridge width must be a multiple of 4 coordinate units".into());
    }
    let (ex, ey) = if horizontal { (3, 2) } else { (2, 3) };
    let (w, h) = if horizontal {
        (4 * d + bridge, 2 * d)
    } else {
        (2 * d, 4 * d + bridge)
    };
    // Doubled coordinate along the long axis -> doubled coordinate of the unbridged layout.
    let long = |c2: i32| -> Option<i32> {
        let (lo, hi) = (4 * d, 4 * d + 2 * bridge);
        if c2 < lo {
            Some(c2)
        } else if c2 >= hi {
            Some(c2 - 2 * bridge)
        } else if merged {
            Some(lo - 8 + (c2 - lo).rem_euclid(8))
        } else {
            None
        }
    };
    let fx = |c2: i32| {
        if horizontal {
            long(c2).map(|c| fold_axis(c, d, ex))
        } else {
            Some(fold_axis(c2, d, ex))
        }
    };
    let fy = |c2: i32| {
        if horizontal {
            Some(fold_axis(c2, d, ey))
        } else {
            long(c2).map(|c| fold_axis(c, d, ey))
        }
    };
    let basis = |c: char| if c == 'X' { B::X } else { B::Z };
    let singles = |pick: &dyn Fn(&walk_template::Tick) -> &[((i32, i32), char)]| {
        let table: HashMap<(i32, i32), char> = pick(&ticks[0])
            .iter()
            .chain(pick(&ticks[5]))
            .copied()
            .collect();
        let mut out = Vec::new();
        for y in 0..=h {
            for x in 0..=w {
                let (Some(tx), Some(ty)) = (fx(2 * x), fy(2 * y)) else {
                    continue;
                };
                if let Some(&b) = table.get(&(tx / 2, ty / 2)) {
                    out.push((IVec2::new(x, y), basis(b)));
                }
            }
        }
        out
    };
    let resets = singles(&|t| t.resets);
    let measures = singles(&|t| t.measures);
    let mut layers = Vec::new();
    for tick in &ticks[1..5] {
        if tick.gates.iter().any(|g| g.0) {
            return Err("surgery: the merge and split figures have no CXSwap".into());
        }
        let table: HashSet<((i32, i32), (i32, i32))> = tick
            .gates
            .iter()
            .map(|&(_, c, t)| ((c.0 + t.0, c.1 + t.1), (t.0 - c.0, t.1 - c.1)))
            .collect();
        let mut layer = Vec::new();
        for y in 0..=h {
            for x in 0..=w {
                for dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    let (Some(tx), Some(ty)) = (fx(2 * x + dir.0), fy(2 * y + dir.1)) else {
                        continue;
                    };
                    if table.contains(&((tx, ty), dir)) {
                        let c = IVec2::new(x, y);
                        let t = c + IVec2::new(dir.0, dir.1);
                        let inside = |q: IVec2| fx(2 * q.x).is_some() && fy(2 * q.y).is_some();
                        if !inside(c) || !inside(t) {
                            return Err(format!(
                                "surgery: a separate round has a gate {c} -> {t} into the bridge"
                            )
                            .into());
                        }
                        layer.push((c, t));
                    }
                }
            }
        }
        let mut used = HashSet::new();
        if !layer.iter().all(|&(c, t)| used.insert(c) && used.insert(t)) {
            return Err(format!("surgery: a qubit is used twice in a CX layer at d = {d}").into());
        }
        layers.push(layer);
    }
    let mut qubits: Vec<IVec2> = resets.iter().chain(&measures).map(|&(q, _)| q).collect();
    qubits.extend(layers.iter().flatten().flat_map(|&(c, t)| [c, t]));
    qubits.sort_by_key(|q| (q.y, q.x));
    qubits.dedup();
    Ok(RoundSpec {
        qubits,
        resets,
        h_first: Vec::new(),
        layers,
        measures,
    })
}

/// Make consecutive rounds agree on their data. A data qubit of one round that the next round does
/// not touch idles through it; a data qubit the next round needs and the previous round does not
/// touch is reset there in `fresh` (the bridge qubits of Sec. 1.2 step 2, and the spare corner of
/// the Z-top figures; assumption A13 in the log). Any other disagreement is an error.
fn glue(specs: &mut [RoundSpec], fresh: B, bridge: &dyn Fn(IVec2) -> bool) -> Result<(), Error> {
    glue_with(specs, &|_, _| fresh, &|_, q| bridge(q).then_some(fresh))
}

/// As `glue`, with the reset basis of each fresh qubit given by `fresh(r, q)`, and a data qubit
/// that round r drops measured in `drop(r, q)` when that is `Some` (left idle otherwise). `r` is
/// the index of the round that resets, measures or idles the qubit.
fn glue_with(
    specs: &mut [RoundSpec],
    fresh: &dyn Fn(usize, IVec2) -> B,
    drop: &dyn Fn(usize, IVec2) -> Option<B>,
) -> Result<(), Error> {
    for r in 0..specs.len() - 1 {
        let (head, tail) = specs.split_at_mut(r + 1);
        let (prev, next) = (&mut head[r], &mut tail[0]);
        let out: HashSet<IVec2> = prev.data_out().into_iter().collect();
        let inn: HashSet<IVec2> = next.data_in().into_iter().collect();
        // Sorted, so the emitted reset and measurement order is the same on every run.
        let sorted = |set: &HashSet<IVec2>| {
            let mut v: Vec<IVec2> = set.iter().copied().collect();
            v.sort_by_key(|q| (q.y, q.x));
            v
        };
        for q in sorted(&out) {
            if !inn.contains(&q) {
                if next.qubits.contains(&q) {
                    return Err(format!("round {} drops data qubit {q}", r + 1).into());
                }
                next.qubits.push(q);
                if let Some(b) = drop(r + 1, q) {
                    // A bridge qubit the split leaves: measure it in its reset basis (G6).
                    next.measures.push((q, b));
                }
            }
        }
        for q in sorted(&inn) {
            if !out.contains(&q) {
                let gated = prev.layers.iter().flatten().any(|&(c, t)| c == q || t == q);
                if !gated && prev.measures.iter().any(|&(m, _)| m == q) {
                    // A corner the round only measures (no gate): keep it as data instead.
                    prev.measures.retain(|&(m, _)| m != q);
                    continue;
                }
                if prev.measures.iter().any(|&(m, _)| m == q) {
                    // The Z-top merge (Fig. 51) starts with the corner the split round B resets in X
                    // already in use; read that as a reset the figure does not draw (assumption A13).
                    next.resets.push((q, fresh(r + 1, q)));
                    continue;
                }
                if prev.qubits.contains(&q) {
                    return Err(format!(
                        "round {} needs data qubit {q} that round {r} measures",
                        r + 1
                    )
                    .into());
                }
                prev.qubits.push(q);
                prev.resets.push((q, fresh(r, q)));
            }
        }
    }
    Ok(())
}

/// The logical of one patch at offset `o`: the data column (home x - o.x == 2 line) or row
/// (home y - o.y == 2 line + 1) of the round's input data inside the patch.
fn patch_logical(
    top: &str,
    basis: B,
    d: i32,
    spec: &RoundSpec,
    o: IVec2,
    line: i32,
) -> Result<Logical, Error> {
    let column = matches!((top, basis), ("x-top", B::X) | ("z-top", B::Z));
    let data: HashSet<IVec2> = spec.data_in().into_iter().collect();
    // Z-top data moves onto slots: one on (x, y) with x odd came from (x - 1, y + 1). Pick the
    // line by these home sites; where a moved qubit and an unused corner share a home (the corner
    // carries a weight-3 Z check), the moved one is on the line.
    let home = |q: IVec2| {
        if q.x % 2 == 0 {
            q
        } else {
            IVec2::new(q.x - 1, q.y + 1)
        }
    };
    let l: Logical = data
        .iter()
        .filter(|&&q| q.x % 2 == 1 || !data.contains(&IVec2::new(q.x + 1, q.y - 1)))
        .filter(|&&q| {
            let h = home(q) - o;
            h.x >= 0
                && h.y >= 0
                && h.x < 2 * d
                && h.y < 2 * d
                && if column {
                    h.x == 2 * line
                } else {
                    h.y == 2 * line + 1
                }
        })
        .map(|q| ((q.x, q.y), basis))
        .collect();
    if l.len() != d as usize {
        return Err(format!("patch logical has weight {} at d = {d}", l.len()).into());
    }
    Ok(l)
}

/// The rounds of one merge and split (Figs. 50, 51): merge rounds A and B, the d - 2 merged middle
/// rounds, split rounds A and B. `units`: the bridge width in lattice units (G6; 0 for G5).
struct SurgeryRounds {
    ma: RoundSpec,
    mb: RoundSpec,
    middle: Vec<RoundSpec>,
    sa: RoundSpec,
    sb: RoundSpec,
}

fn surgery_rounds(top: &str, horizontal: bool, d: i32, units: i32) -> Result<SurgeryRounds, Error> {
    let bridge = 2 * units;
    use surgery_template as t;
    let (merge, split) = match (top, horizontal) {
        ("x-top", true) => (t::X_H_MERGE, t::X_H_SPLIT),
        ("x-top", false) => (t::X_V_MERGE, t::X_V_SPLIT),
        ("z-top", true) => (t::Z_H_MERGE, t::Z_H_SPLIT),
        ("z-top", false) => (t::Z_V_MERGE, t::Z_V_SPLIT),
        (other, _) => return Err(format!("{other}: want x-top or z-top").into()),
    };
    let ma = surgery_round(&merge[0..6], d, horizontal, bridge, false)?;
    let mb = surgery_round(&merge[6..12], d, horizontal, bridge, true)?;
    let sa = surgery_round(&split[0..6], d, horizontal, bridge, true)?;
    let sb = surgery_round(&split[6..12], d, horizontal, bridge, false)?;
    let mut middle = Vec::new();
    if top == "x-top" {
        let mbi = mb.inverse();
        // d - 2 middle rounds, so d merged rounds in all (arXiv:2605.30455 section 1.2); an odd
        // count, since mB's output does not chain into split round A and mB^-1's does (A11).
        for k in 0..d - 2 {
            middle.push(if k % 2 == 0 { mbi.clone() } else { mb.clone() });
        }
    } else {
        let mut gm = if horizontal {
            g_z_rect(2 * d + units, d)?
        } else {
            g_z_rect(d, 2 * d + units)?
        };
        // The merged patch of the figures keeps each patch's spare corner qubit as data (it sits
        // in a weight-3 check); g_z_rect leaves it out. Graft in the corner's gates from mB^-1,
        // whose corner data agree with mB's output (A14).
        let mbi = mb.inverse();
        let have: HashSet<IVec2> = gm.data_in().into_iter().collect();
        for q in mb.data_out().into_iter().filter(|q| !have.contains(q)) {
            gm.qubits.push(q);
            for (k, layer) in mbi.layers.iter().enumerate() {
                for &(c, t) in layer.iter().filter(|&&(c, t)| c == q || t == q) {
                    let busy = gm.layers[k]
                        .iter()
                        .any(|&(a, b)| [a, b].contains(&c) || [a, b].contains(&t));
                    if !busy {
                        gm.layers[k].push((c, t));
                    }
                }
            }
        }
        let gmi = gm.inverse();
        // d - 2 middle rounds, as for the X-top. With fewer, an individual observable that takes in
        // the merge outcome has timelike distance below d (3 at d = 5 with 1 middle round; A15).
        for k in 0..d - 2 {
            middle.push(if k % 2 == 0 { gm.clone() } else { gmi.clone() });
        }
    }
    Ok(SurgeryRounds {
        ma,
        mb,
        middle,
        sa,
        sb,
    })
}

/// H-bridge lattice surgery with bridge width 0 (arXiv:2605.30455, Sec. 1.2; Figs. 50 and 51):
/// two patches, 2 separate rounds, merge (rounds A and B of the merge figure), merged rounds, split
/// (rounds A and B of the split figure), 2 separate rounds; perfect preparation and readout in
/// `basis`. With d - 2 middle rounds the patches are merged for d rounds; d + 6 rounds in all (A11).
/// Observables: each patch's logical when `basis` is the one the merge measures (XX or ZZ), else
/// the product of the two.
fn surgery(top: &str, orient: &str, basis: B, d: i32, k: i32) -> Result<String, Error> {
    if d < 3 || d % 2 == 0 {
        return Err("need odd d >= 3".into());
    }
    // The bridge: k ancilla patches of d lattice units (2d coordinate units) each (G6). At odd k d
    // that shifts the second patch by half a checkerboard period; the folded figures need whole
    // periods, so the bridge gets one more lattice column (k d + 1 units; A17).
    if k < 0 {
        return Err("h-bridge: need k >= 0".into());
    }
    let units = k * d + (k * d) % 2;
    let bridge = 2 * units;
    let horizontal = match orient {
        "h" => true,
        "v" => false,
        other => return Err(format!("{other}: want h or v").into()),
    };
    let SurgeryRounds {
        ma,
        mb,
        middle,
        sa,
        sb,
    } = surgery_rounds(top, horizontal, d, units)?;
    let o = if horizontal {
        IVec2::new(2 * d + bridge, 0)
    } else {
        IVec2::new(0, 2 * d + bridge)
    };
    // Separate rounds: split round B, which measures no inter-boundary stabilizer, and its inverse.
    // (A shifted copy of the single-patch round does not fit: the second patch sits on the merged
    // checkerboard, which at odd d is the mirror of its own.)
    let g = sb.clone();
    let gi = sb.inverse();
    // Separate rounds before and after: the phase (G, G^-1 or G^-1, G) whose data agrees with
    // the figure rounds next to them.
    // Bridge data qubits: the split measures them out in the basis they were reset in.
    let in_bridge = |q: IVec2| {
        let c = if horizontal { q.x } else { q.y };
        (2 * d..2 * d + bridge).contains(&c)
    };
    let mut errors = Vec::new();
    for (pre, post) in [
        ((&g, &gi), (&g, &gi)),
        ((&gi, &g), (&g, &gi)),
        ((&g, &gi), (&gi, &g)),
        ((&gi, &g), (&gi, &g)),
    ] {
        for fresh in [B::X, B::Z] {
            let mut specs = vec![pre.0.clone(), pre.1.clone(), ma.clone(), mb.clone()];
            specs.extend(middle.iter().cloned());
            specs.extend([sa.clone(), sb.clone(), post.0.clone(), post.1.clone()]);
            if let Err(e) = glue(&mut specs, fresh, &in_bridge) {
                errors.push(format!("glue: {e}"));
                continue;
            }
            // The merge measures the product of the logicals parallel to the seam; each of those
            // survives alone, the perpendicular ones only as a product.
            let column = matches!((top, basis), ("x-top", B::X) | ("z-top", B::Z));
            let cross = column == horizontal;
            // The second data line of each patch: the first one passes the Z-top spare corner,
            // whose extra degree of freedom a weight-1 error flips (A14).
            let line = 1;
            let l1 = patch_logical(top, basis, d, &specs[0], IVec2::ZERO, line)?;
            let l2 = patch_logical(top, basis, d, &specs[0], o, line)?;
            let logicals = if cross {
                vec![l1, l2]
            } else {
                vec![l1.into_iter().chain(l2).collect()]
            };
            // Seam checks: within one data column (or row) of the boundary between the patches.
            let seam = |_: usize, f: &Flow| {
                f.start.iter().all(|(q, _)| {
                    let c = if horizontal { q.x } else { q.y };
                    (2 * d - 3..=2 * d + bridge + 1).contains(&c)
                })
            };
            match run_rounds_multi(&specs, &logicals, basis, basis, &seam) {
                Ok(text) => {
                    eprintln!("surgery: {} rounds, fresh qubits in {fresh:?}", specs.len());
                    return Ok(text);
                }
                Err(e) => errors.push(format!("fresh {fresh:?}: {e}")),
            }
        }
    }
    errors.dedup();
    Err(format!(
        "surgery: no phase and fresh basis works:\n  {}",
        errors.join("\n  ")
    )
    .into())
}

/// Logical CX by lattice surgery with an ancilla patch (Fig. 53, `fig:CNOT_X_Z`), measurement
/// based: ancilla A prepared in |0>, M_XX(A, T) by a vertical merge, M_ZZ(C, A) by a horizontal
/// merge, A measured in X. The figure runs both merges on one ancilla in d + 2 rounds with the
/// first split under the second merge; here the two merge-and-split blocks of G5 run one after
/// the other, with one round between them in which A idles alone (assumptions A19, A21).
/// Layout: target T at (2d, 0), control C at (0, 2d), A at (2d, 2d) (Fig. 53 mirrored top to
/// bottom), so that A is the second patch of both merges and keeps one checkerboard; under G_Z the
/// X row of A drifts away from T, so the XX merge comes first (A20). Observables, with perfect
/// preparation and readout in `basis`: Z: Z_C and Z_T (ends as Z_C Z_T); X: X_T and X_C (ends
/// as X_C X_T).
fn cx(top: &str, basis: B, d: i32) -> Result<String, Error> {
    if d < 3 || d % 2 == 0 {
        return Err("need odd d >= 3".into());
    }
    let h = surgery_rounds(top, true, d, 0)?;
    let v = surgery_rounds(top, false, d, 0)?;
    let (oc, ot, oa) = (
        IVec2::new(0, 2 * d),
        IVec2::new(2 * d, 0),
        IVec2::new(2 * d, 2 * d),
    );
    let in_c = |q: IVec2| q.x < 2 * d && q.y >= 2 * d;
    let in_t = |q: IVec2| q.x >= 2 * d && q.y < 2 * d;
    let in_a = |q: IVec2| q.x >= 2 * d && q.y >= 2 * d;
    // Idle rounds of C and T: their part of the split round B of their merge, and its inverse.
    let c_idle = h.sb.shifted(oc).patch(&in_c)?;
    let t_idle = v.sb.shifted(ot).patch(&in_t)?;
    let alt = |g: &RoundSpec, n: usize, phase: usize| -> Vec<RoundSpec> {
        let gi = g.inverse();
        (0..n)
            .map(|i| {
                if (i + phase) % 2 == 0 {
                    g.clone()
                } else {
                    gi.clone()
                }
            })
            .collect()
    };
    let block = |s: &SurgeryRounds, o: IVec2| -> Vec<RoundSpec> {
        let mut r = vec![s.ma.shifted(o), s.mb.shifted(o)];
        r.extend(s.middle.iter().map(|m| m.shifted(o)));
        r.extend([s.sa.shifted(o), s.sb.shifted(o)]);
        r
    };
    let step1 = block(&v, ot);
    let step2 = block(&h, oc);
    let n = step1.len();
    // The round between the steps in which A idles alone (A21): a part of a split round B, or
    // its inverse, whichever joins the two steps.
    let a_h = h.sb.shifted(oc).patch(&in_a)?;
    let a_v = v.sb.shifted(ot).patch(&in_a)?;
    // Two rounds end on A's part of the split round B of the C-A pair, which resets A's spare
    // corner in X as G5 does before its merge (A13).
    let gaps: Vec<Vec<RoundSpec>> = vec![
        vec![],
        vec![a_v.inverse()],
        vec![a_h.inverse()],
        vec![a_h.inverse(), a_h.clone()],
        vec![a_v.inverse(), a_h.clone()],
        vec![a_v.inverse(), a_v.clone(), a_h.inverse(), a_h.clone()],
    ];
    let mut errors = Vec::new();
    for (gi, gap) in gaps.iter().enumerate() {
        let g = gap.len();
        for phases in 0..16usize {
            let p = |i: usize| (phases >> i) & 1;
            // T: 2 idle rounds, step 1, the gap and n + 2 idle rounds. C: 2 + n idle rounds, the
            // gap, step 2, 2 idle rounds.
            let t_pre = alt(&t_idle, 2, p(0));
            let t_post = alt(&t_idle, g + n + 2, p(1));
            let c_pre = alt(&c_idle, n + 2 + g, p(2));
            let c_post = alt(&c_idle, 2, p(3));
            let mut specs = Vec::new();
            for i in 0..2 {
                specs.push(t_pre[i].union(&c_pre[i])?);
            }
            for i in 0..n {
                specs.push(step1[i].union(&c_pre[i + 2])?);
            }
            for (i, a) in gap.iter().enumerate() {
                specs.push(a.union(&t_post[i])?.union(&c_pre[n + 2 + i])?);
            }
            for i in 0..n {
                specs.push(step2[i].union(&t_post[g + i])?);
            }
            for i in 0..2 {
                specs.push(t_post[g + n + i].union(&c_post[i])?);
            }
            let first2 = 2 + n + g;
            // X basis: each (merged round, row of A, row of T) for the X_A X_T pick-up is its own
            // attempt; the row fixes how the drifted row meets A's spare corner (A14) at the ZZ seam.
            let picks: Vec<(usize, i32, i32)> = if basis == B::X {
                (2..2 + n - 2)
                    .flat_map(|r| (0..d).flat_map(move |la| (0..d).map(move |l| (r, la, l))))
                    .collect()
            } else {
                vec![(0, 0, 0)]
            };
            // A's spare Z-top corner (A14), bottom left, on one of its two sites.
            let corners = [
                None,
                Some(IVec2::new(2 * d, 4 * d - 1)),
                Some(IVec2::new(2 * d + 1, 4 * d - 2)),
            ];
            for fresh in [B::X, B::Z] {
                for corner in corners {
                    // X basis: X on A's corner joins the row at one round near the ZZ merge.
                    let mut csites: Vec<Option<(IVec2, usize)>> = vec![None];
                    if basis == B::X {
                        for c in [
                            IVec2::new(2 * d, 4 * d - 1),
                            IVec2::new(2 * d + 1, 4 * d - 2),
                        ] {
                            csites.extend((first2 - 2..first2 + 2).map(|r| Some((c, r))));
                        }
                    }
                    for csite in csites {
                        for &(pr, pla, plt) in &picks {
                            let mut specs = specs.clone();
                            // A's data: reset in Z before step 1, measured in X when step 2 ends. Its spare
                            // corner is reset in X, as the X init of a patch does: under G_Z the X row of A
                            // picks it up on its way to the bottom boundary (A22).
                            let fresh_of = |_: usize, q: IVec2| match () {
                                _ if Some(q) == corner => B::X,
                                _ if in_a(q) => B::Z,
                                _ => fresh,
                            };
                            let drop =
                                |r: usize, q: IVec2| (in_a(q) && r >= first2).then_some(B::X);
                            if let Err(e) = glue_with(&mut specs, &fresh_of, &drop) {
                                errors.push(format!("gap {gi} glue: {e}"));
                                continue;
                            }
                            let line = 1;
                            let at = |r: usize, o: IVec2, l: i32| {
                                patch_logical(top, basis, d, &specs[r + 1], o, l)
                            };
                            let lc = patch_logical(top, basis, d, &specs[0], oc, line)?;
                            let lt = patch_logical(top, basis, d, &specs[0], ot, line)?;
                            let both = |a: &Logical, b: &Logical| -> Logical {
                                a.iter().chain(b).map(|(&k, &v)| (k, v)).collect()
                            };
                            // Deformations of observable 1 (A20), each at the first candidate that is a product
                            // of that round's checks and fixed single Paulis.
                            // Z: Z_T picks up Z_A (A's |0>, the round before step 1) on the same column, so it
                            // passes the XX seam; then the ZZ merge's checks move Z_A into C.
                            // X: X_C picks up X_A X_T (the XX merge's checks), so it passes the ZZ seam with
                            // its row in A; A's X readout then takes that row.
                            let mut times: Vec<(usize, usize, Logical)> = Vec::new();
                            let mut offs: Vec<(usize, usize)> = Vec::new();
                            let logicals = if basis == B::Z {
                                times.push((0, 1, at(1, oa, line)?));
                                offs.extend((first2..first2 + n - 2).map(|r| (1, r)));
                                vec![lc.clone(), lt.clone()]
                            } else {
                                times.push((0, pr, both(&at(pr, oa, pla)?, &at(pr, ot, plt)?)));
                                if let Some((c, cr)) = csite {
                                    times.push((1, cr, Logical::from([((c.x, c.y), B::X)])));
                                }
                                vec![lt.clone(), lc.clone()]
                            };
                            let mut deform: Vec<Deform> = (times.iter())
                                .map(|(g, r, t)| Deform {
                                    group: *g,
                                    round: *r,
                                    observable: 1,
                                    target: Target::Times(t),
                                })
                                .collect();
                            deform.extend(offs.iter().map(|&(g, r)| Deform {
                                group: g,
                                round: r,
                                observable: 1,
                                target: Target::Off(&in_a),
                            }));
                            // Checks that may start unmatched: those inside A or on its two seams.
                            let seam = |_: usize, f: &Flow| {
                                (f.start.iter()).all(|(q, _)| q.x >= 2 * d - 3 && q.y >= 2 * d - 3)
                            };
                            match run_rounds_deform(&specs, &logicals, basis, basis, &seam, &deform) {
                    Ok(text) => {
                        eprintln!(
                            "cx: {} rounds, gap {gi}, phases {phases:04b}, fresh {fresh:?}, \
                             corner {corner:?}, csite {csite:?}, pick {:?}",
                            specs.len(),
                            (pr, pla, plt)
                        );
                        return Ok(text);
                    }
                    Err(e) => errors.push(format!(
                        "gap {gi} phases {phases:04b} fresh {fresh:?} corner {corner:?} csite {csite:?} pick {:?}: {e}",
                        (pr, pla, plt)
                    )),
                }
                        }
                    }
                }
            }
        }
    }
    errors.dedup();
    Err(format!("cx: nothing works:\n  {}", errors.join("\n  ")).into())
}

/// The logical CX in the order of Fig. 53 (arXiv:2605.30455) as low-lead reads it (A26): A is
/// reset in X, ZZ(C, A) merge and split, a gap, XX(A, T) merge and split, A read out in Z. The two
/// merges run one after the other here, not overlapped as in the figure (deviation, A26).
fn cx_zz_first(top: &str, basis: B, d: i32, verbose: bool) -> Result<String, Error> {
    if d < 3 || d % 2 == 0 {
        return Err("need odd d >= 3".into());
    }
    let h = surgery_rounds(top, true, d, 0)?;
    let v = surgery_rounds(top, false, d, 0)?;
    let (oc, ot, oa) = (
        IVec2::new(0, 2 * d),
        IVec2::new(2 * d, 0),
        IVec2::new(2 * d, 2 * d),
    );
    let in_c = |q: IVec2| q.x < 2 * d && q.y >= 2 * d;
    let in_t = |q: IVec2| q.x >= 2 * d && q.y < 2 * d;
    let in_a = |q: IVec2| q.x >= 2 * d && q.y >= 2 * d;
    let c_idle = h.sb.shifted(oc).patch(&in_c)?;
    let t_idle = v.sb.shifted(ot).patch(&in_t)?;
    let alt = |g: &RoundSpec, n: usize, phase: usize| -> Vec<RoundSpec> {
        let gi = g.inverse();
        (0..n)
            .map(|i| {
                if (i + phase) % 2 == 0 {
                    g.clone()
                } else {
                    gi.clone()
                }
            })
            .collect()
    };
    let block = |s: &SurgeryRounds, o: IVec2| -> Vec<RoundSpec> {
        let mut r = vec![s.ma.shifted(o), s.mb.shifted(o)];
        r.extend(s.middle.iter().map(|m| m.shifted(o)));
        r.extend([s.sa.shifted(o), s.sb.shifted(o)]);
        r
    };
    let step1 = block(&h, oc);
    let step2 = block(&v, ot);
    let n = step1.len();
    let a_h = h.sb.shifted(oc).patch(&in_a)?;
    let a_v = v.sb.shifted(ot).patch(&in_a)?;
    let gaps: Vec<Vec<RoundSpec>> = vec![
        vec![],
        vec![a_h.inverse()],
        vec![a_v.inverse()],
        vec![a_v.inverse(), a_v.clone()],
        vec![a_h.inverse(), a_v.clone()],
        vec![a_h.inverse(), a_h.clone(), a_v.inverse(), a_v.clone()],
    ];
    let mut errors = Vec::new();
    for (gi, gap) in gaps.iter().enumerate() {
        let g = gap.len();
        for phases in 0..16usize {
            let p = |i: usize| (phases >> i) & 1;
            // C: 2 idle rounds, step 1, the gap and n + 2 idle rounds. T: 2 + n idle rounds, the
            // gap, step 2, 2 idle rounds.
            let c_pre = alt(&c_idle, 2, p(0));
            let c_post = alt(&c_idle, g + n + 2, p(1));
            let t_pre = alt(&t_idle, n + 2 + g, p(2));
            let t_post = alt(&t_idle, 2, p(3));
            let mut specs = Vec::new();
            for i in 0..2 {
                specs.push(t_pre[i].union(&c_pre[i])?);
            }
            for i in 0..n {
                specs.push(step1[i].union(&t_pre[i + 2])?);
            }
            for (i, a) in gap.iter().enumerate() {
                specs.push(a.union(&c_post[i])?.union(&t_pre[n + 2 + i])?);
            }
            for i in 0..n {
                specs.push(step2[i].union(&c_post[g + i])?);
            }
            for i in 0..2 {
                specs.push(t_post[i].union(&c_post[g + n + i])?);
            }
            let first2 = 2 + n + g;
            // A's spare Z-top corner (A14): reset with the rest of A, or in the other basis.
            let corners = [
                None,
                Some(IVec2::new(2 * d, 4 * d - 1)),
                Some(IVec2::new(2 * d + 1, 4 * d - 2)),
            ];
            // Fresh basis of the strips: one basis for both, or "by merge" (None): X for the ZZ
            // strip, Z for the XX strip, as Fig. 53 resets them (low-lead's reading).
            for fresh_opt in [None, Some(B::X), Some(B::Z)] {
                for corner in corners {
                    let lines: Vec<(i32, i32)> = if basis == B::X {
                        (0..d).flat_map(|a| (0..d).map(move |b| (a, b))).collect()
                    } else {
                        vec![(1, 1)]
                    };
                    for &(lcl, lal) in &lines {
                        // One choice for both bases, so the X and Z circuits differ only in their
                        // preparation and readout basis: the first that builds both at d = 3 (A26).
                        // LG_CX_PICK="<gap> <phases> <fresh>" (e.g. "1 1010 Some(Z)") tries another;
                        // LG_CX_PICK=any searches all.
                        let pick = std::env::var("LG_CX_PICK").unwrap_or("2 1010 None".into());
                        if pick != "any" && pick != format!("{gi} {phases:04b} {fresh_opt:?}") {
                            continue;
                        }
                        let mut specs = specs.clone();
                        // A's data: reset in X before step 1 (|+>), measured in Z when step 2 ends.
                        let fresh_of = |r_: usize, q: IVec2| match () {
                            _ if Some(q) == corner => B::Z,
                            _ if in_a(q) => B::X,
                            _ => fresh_opt.unwrap_or(if r_ < first2 { B::X } else { B::Z }),
                        };
                        let drop = |r: usize, q: IVec2| (in_a(q) && r >= first2).then_some(B::Z);
                        if let Err(e) = glue_with(&mut specs, &fresh_of, &drop) {
                            errors.push(format!("gap {gi} glue: {e}"));
                            continue;
                        }
                        let line = 1;
                        let at = |r: usize, o: IVec2, l: i32| {
                            patch_logical(top, basis, d, &specs[r + 1], o, l)
                        };
                        let lc = patch_logical(top, basis, d, &specs[0], oc, lcl)?;
                        let lt = patch_logical(top, basis, d, &specs[0], ot, line)?;
                        let both = |a: &Logical, b: &Logical| -> Logical {
                            a.iter().chain(b).map(|(&k, &v)| (k, v)).collect()
                        };
                        // Z: observable 1 is Z_C Z_T; the ZZ merge's checks move Z_C into A, so it
                        // passes the XX seam as one column with Z_T; A's Z readout then takes it.
                        // X: observable 1 is X_C; it picks up X_A (A's |+>) so it passes the ZZ seam,
                        // then the XX merge's checks move X_A into T.
                        let mut times: Vec<(usize, usize, Logical)> = Vec::new();
                        let mut offs: Vec<(usize, usize)> = Vec::new();
                        let mut edge_offs: Vec<(usize, usize)> = Vec::new();
                        let a_edge = |q: IVec2| in_a(q) && q.y >= 4 * d - 2;
                        // A, and T's two rows next to the XX seam (its spare corner sits there).
                        let a_seam = |q: IVec2| q.x >= 2 * d && q.y >= 2 * d - 2;
                        let region: &dyn Fn(IVec2) -> bool;
                        let logicals = if basis == B::Z {
                            offs.extend((4..2 + n - 2).map(|r| (1, r)));
                            // The Z column drifts back into C during the merge (the round's flows):
                            // move it into A again in each later round of step 1.
                            offs.extend((5..2 + n).map(|r| (10 + r, r)));
                            region = &in_c;
                            vec![lc.clone(), both(&lt, &lc)]
                        } else {
                            times.push((0, 1, at(1, oa, lal)?));
                            offs.extend((first2..first2 + n - 2).map(|r| (1, r)));
                            // The X row drifts up 2 rows per merged round (the round's flows): move it
                            // back down into T in each later round of step 2, its own group per round.
                            offs.extend((first2 + 1..first2 + n).map(|r| (10 + r, r)));
                            region = &a_seam;
                            // The row drifts onto A's far boundary row, through A's spare corner,
                            // where no X check covers it: move it off that row before the XX merge.
                            for r in 2 + n - 2..first2 {
                                edge_offs.push((2, r));
                            }
                            vec![lt.clone(), lc.clone()]
                        };
                        let mut deform: Vec<Deform> = (times.iter())
                            .map(|(g, r, t)| Deform {
                                group: *g,
                                round: *r,
                                observable: 1,
                                target: Target::Times(t),
                            })
                            .collect();
                        deform.extend(offs.iter().map(|&(g, r)| Deform {
                            group: g,
                            round: r,
                            observable: 1,
                            target: Target::Off(region),
                        }));
                        // X_T (observable 0) drifts the same way during step 2.
                        if basis == B::X {
                            deform.extend((first2..first2 + n).map(|r| Deform {
                                group: 40 + r,
                                round: r,
                                observable: 0,
                                target: Target::Off(&a_seam),
                            }));
                        }
                        deform.extend(edge_offs.iter().map(|&(g, r)| Deform {
                            group: g,
                            round: r,
                            observable: 1,
                            target: Target::Off(&a_edge),
                        }));
                        let seam = |_: usize, f: &Flow| {
                            (f.start.iter()).all(|(q, _)| q.x >= 2 * d - 3 && q.y >= 2 * d - 3)
                        };
                        match run_rounds_deform(&specs, &logicals, basis, basis, &seam, &deform) {
                        Ok(text) => {
                            eprintln!(
                                "cx-zz-first: {} rounds, gap {gi}, phases {phases:04b}, fresh {fresh_opt:?}, corner {corner:?}, lines {lcl} {lal}",
                                specs.len()
                            );
                            return Ok(text);
                        }
                        Err(e) => errors.push(format!(
                            "gap {gi} phases {phases:04b} fresh {fresh_opt:?} corner {corner:?} lines {lcl} {lal}: {e}"
                        )),
                    }
                    }
                }
            }
        }
    }
    errors.dedup();
    if !verbose {
        errors.truncate(40);
    }
    Err(format!("cx-zz-first: nothing works:\n  {}", errors.join("\n  ")).into())
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
        [cmd, basis, d, rounds, out] if cmd == "reflect-memory" => {
            let basis = parse_basis(basis)?;
            let text = reflect_memory(basis, d.parse()?, rounds.parse()?)?;
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
        [cmd, top, orient, basis, d, out] if cmd == "surgery" => {
            let basis = parse_basis(basis)?;
            let text = surgery(top, orient, basis, d.parse()?, 0)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        [cmd, top, basis, d, out] if cmd == "cx" => {
            let basis = parse_basis(basis)?;
            let text = cx(top, basis, d.parse()?)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        [cmd, top, basis, d, out] if cmd == "cx-zz-first" => {
            let basis = parse_basis(basis)?;
            let text = cx_zz_first(top, basis, d.parse()?, true)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        [cmd, top, orient, basis, d, k, out] if cmd == "h-bridge" => {
            let basis = parse_basis(basis)?;
            let text = surgery(top, orient, basis, d.parse()?, k.parse()?)?;
            std::fs::write(out, text)?;
            Ok(())
        }
        _ => Err(concat!(
            "usage: low-gadgets compact-memory <x-top|z-top|z-top-big> <x|z> <d> <rounds> <out.stim>\n",
            "       low-gadgets reflect-memory <x|z> <d> <rounds> <out.stim>\n",
            "       low-gadgets hadamard <x|z> <d> <z-top rounds> <x-top rounds> <out.stim>\n",
            "       low-gadgets walking <x-top|z-top> <x|z> <d> <rounds> <out.stim>\n",
            "       low-gadgets surgery <x-top|z-top> <h|v> <x|z> <d> <out.stim>\n",
            "       low-gadgets h-bridge <x-top|z-top> <h|v> <x|z> <d> <bridge patches k> <out.stim>\n",
            "       low-gadgets cx <x-top|z-top> <x|z> <d> <out.stim>\n",
            "       low-gadgets cx-zz-first <x-top|z-top> <x|z> <d> <out.stim>"
        )
        .into()),
    }
}
