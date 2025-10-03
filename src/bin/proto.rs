#![allow(dead_code)]
// #[cfg(target_arch = "x86_64")]

// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR


// move code into examples directory
// cargo run --example demo
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};

struct DecisionTree {
    dims:usize, // number of dims
    card:usize, // data size
    nodes:usize, // number of nodes
    measure:Vec<Measure>, // measures for running values
    assignment:Vec<Assignment>, // idx -> node
    features:Vec<Vec<Feature>>, // sorted features (idx, f[idx;j], y[idx])
}

struct Graph {
}

// Need partition crtieria ie feature and value and map
#[derive(Clone, Copy)]
struct Measure {
    card:usize,
    offset:usize,
    feature:usize,
    sum_linear:f32, // Sum y
    sum_squared:f32, // Sum y * y;
    partition:Option<f32>,
    left:Option<usize>,
    right:Option<usize>,
}

struct Assignment {
    idx:usize,
    node:usize,
}

#[derive(Clone, Copy)]
struct Feature {
    idx:usize,
    value:f32,
    label:f32,
}

impl Feature {
    fn new() -> Self {
        Self { idx: 0, value:0_f32, label:0_f32 }
    }
}

impl DecisionTree {
    fn new(data:Vec<Vec<f32>>) -> Self {
        if data.is_empty() || data[0].is_empty() { panic!("data is empty"); }
        let card = data.len();
        let label = data[0].len();
        let dims = label-1;
        let assignment = (0..card).map(|idx| Assignment {idx, node: 0 } ).collect();
        let mut buffer = vec![Feature { idx:0, value:0_f32, label:0_f32 }; card];
        let mut features = Vec::with_capacity(dims);
        let measure = vec![Measure::derive(&data)];
        for d in 0..dims {
            for idx in 0..card {
                buffer[idx] = Feature { idx, value: data[idx][d], label: data[idx][label-1] } ;
            }
            // sort by feature
            buffer.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
            features.push(buffer.clone());
        }
        Self {
            dims,
            card,
            nodes:1,
            measure,
            assignment,
            features,
        }
    }
    fn split(&mut self) {
        // running candidate partitions
        let mut runnings:Vec<Measure> = (0..self.nodes).map( |idx| {
            let m = &self.measure[idx];
            Measure::new(m.feature, m.offset)
        }).collect();
        let (mut measure_delta, mut measure_partition) = (0_f32, 0_f32);
        let (mut current_idx, left_nidx, right_nidx) = (0, self.nodes, self.nodes+1);
        let mut left_node= Measure::new(usize::MAX,usize::MAX);
        // find best new partition
        for d in 0..self.dims {
            let feature = &self.features[d];
            for idx in 0..self.card {
                let node = &self.assignment[idx];
                runnings[node.idx].increment(&feature[idx]);
                let delta = self.measure[node.idx].delta(&runnings[node.idx]);
                if delta < measure_delta { continue; }
                current_idx = node.idx;
                measure_delta = delta;
                measure_partition = self.features[d][idx].value;
                left_node = runnings[node.idx].clone();
            }
        }
        let current_node = self.measure[current_idx];
        // update assignments for nodes
        for i in 0..current_node.card {
            let fidx = i + current_node.offset;
            let feature = self.features[left_node.feature][fidx];
            if i < left_node.card {
                self.assignment[feature.idx].node = left_nidx;
            } else {
                self.assignment[feature.idx].node = right_nidx;
            }
        }
        let mut buffer = vec![Feature::new(); current_node.card];
        let current = &self.measure[current_idx];
        let (mut lidx, mut ridx) = (current_node.offset, current_node.offset + left_node.card);
        // sort the sub partitions in the data
        for d in 0..self.dims {
            if d == current_idx { continue; }
            let dimension = &self.features[d];
            for fidx in current.offset..current.card {
                let feature = self.features[left_node.feature][fidx];
                if self.assignment[feature.idx].node == left_nidx {
                    buffer[ridx] = dimension[fidx];
                    ridx += 1;
                } else {
                    buffer[lidx] = dimension[fidx];
                    lidx += 1;
                }
            }
            self.features[d][current.offset..current.offset+current.card].copy_from_slice(&buffer);
        }
        let right_node = self.measure[current_idx].derive_right(&left_node);
        let current = &mut self.measure[current_idx];
        current.partition = Some(measure_partition);
        current.left = Some(left_nidx);
        current.right = Some(right_nidx);
        self.measure.push(left_node);
        self.measure.push(right_node);
        self.nodes += 2;
    }
}

impl Measure {
    fn new(feature:usize, offset:usize) -> Self {
        Self {
            feature,
            offset,
            card:0,
            sum_linear:0_f32,
            sum_squared:0_f32,
            partition:None,
            left:None,
            right:None,
        }
    }
    fn increment(&mut self, feature:&Feature) {
        self.card += 1;
        self.sum_linear += feature.value;
        self.sum_squared += feature.value * feature.value;
    }
    fn derive(data:&Vec<Vec<f32>>)  -> Self {
        if data.is_empty() || data[0].is_empty() { panic!("data is empty"); }
        let card = data.len();
        let label = data[0].len()-1;
        let (mut sum_linear, mut sum_squared) = (0_f32, 0_f32);
        for idx in 0..card {
            let val = data[idx][label];
            sum_linear += val;
            sum_squared += val * val;
        }
        Measure {feature:label, offset:0, card, sum_linear, sum_squared, partition:None, left:None, right:None}
    }
    fn delta(&self, running:&Self) -> f32 {
        let current_variance = self.sum_squared - self.sum_linear * self.sum_linear;
        let variance_node_left = running.sum_squared - running.sum_linear * running.sum_linear;
        let variance_node_right = (self.sum_squared - running.sum_squared) - (self.sum_linear - running.sum_linear) * (self.sum_linear - running.sum_linear); 
        current_variance - (variance_node_left + variance_node_right)
    }
    fn derive_right(&mut self, left:&Self) -> Self {
        Self {
            feature:self.feature,
            offset:left.offset + left.card,
            card:self.card - left.card,
            sum_linear:self.sum_linear - left.sum_linear,
            sum_squared:self.sum_squared - left.sum_squared,
            partition:None,
            left:None,
            right:None,
        }
    }
}
fn main() {
}
