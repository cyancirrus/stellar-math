use rand_distr::{Distribution, Normal};
use std::cmp::{Eq, Ordering, PartialEq};
use std::collections::BinaryHeap;

// Max heap structure for sync, for async generate the values used for prio and take max store aggs as atomics
const EPSILON: f32 = 1e-6;

pub struct Aggregate {
    pub sum: f32,
    pub square: f32,
    pub count: f32,
}

pub struct Bandit {
    pub mean: f32,
    pub std_dev: f32,
    distr: Normal<f32>,
}

struct HeapNode {
    cat: usize,
    prio: f32,
}

impl Eq for HeapNode {}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        let abs_diff = (self.prio - other.prio).abs();
        self.cat == other.cat && abs_diff < EPSILON
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prio
            .partial_cmp(&other.prio)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Bandit {
    pub fn new(mean: f32, std_dev: f32) -> Self {
        Self {
            mean,
            std_dev,
            distr: Normal::new(mean, std_dev).unwrap(),
        }
    }
    fn observe(&self) -> f32 {
        self.distr.sample(&mut rand::rng())
    }
}

/// # exploration_priority
///
/// * agg -> contains raw aggregates
/// * t -> number of iterations so far
pub fn exploration_priority(agg: &Aggregate, t: usize) -> f32 {
    let mean = agg.sum / agg.count;
    let var = agg.square / agg.count - mean * mean;
    let explore = (2.0 * t as f32).ln() / agg.count.sqrt();
    mean + var.sqrt() * explore
    // mean +  explore
}

/// # max_multi_bandit
///
/// * bandits -> levers to generate observations
/// * aggs -> aggregates for const derivations
/// * t -> time horizon
pub fn max_multi_bandit(bandits: &[Bandit], aggs: &mut [Aggregate], t: usize) -> f32 {
    debug_assert_eq!(bandits.len(), aggs.len());
    debug_assert!(bandits.len() != 0);
    let n = bandits.len();
    let mut reward = 0f32;
    let mut pqueue: BinaryHeap<HeapNode> = BinaryHeap::new();
    for cat in 0..n {
        pqueue.push(HeapNode {
            cat,
            prio: f32::MAX,
        });
    }

    for k in 0..t {
        let p = pqueue.pop().unwrap();
        let obs = bandits[p.cat].observe();
        let a = &mut aggs[p.cat];
        a.sum += obs;
        a.square += obs * obs;
        a.count += 1f32;
        let prio = exploration_priority(&a, k);
        pqueue.push(HeapNode { cat: p.cat, prio });
        reward += obs;
    }
    reward
}

// fn sim() {
//     // interesting that as t increased ratio -> 1;
//     let n = 8;
//     let t = 124;
//     let sims = 4;
//     let mut rng = rand::rng();
//     for _ in 0..sims {
//         let mut bandits = Vec::with_capacity(n);
//         let mut aggs = Vec::with_capacity(n);
//         println!("---------------------------");
//         let mut best = f32::NEG_INFINITY;

//         for _ in 0..n {
//             let mean: f32 = rng.sample(StandardNormal);
//             // let mut std_dev: f32 = rng.sample(StandardNormal);
//             // std_dev = std_dev.abs();
//             let std_dev = 0.5;
//             best = best.max(mean);
//             bandits.push(Bandit::new(mean, std_dev));
//             aggs.push(Aggregate {
//                 sum: 0f32,
//                 square: 0f32,
//                 count: 0f32,
//             });
//         }
//         let max = max_multi_bandit(&mut bandits, &mut aggs, t);
//         let poss = best * t as f32;
//         println!("achieved max {max:?}");
//         println!("possible mean {poss:?}");
//         println!("ratio {:?}", max / poss);
//     }
// }

// fn main() {
//     sim();
// }
