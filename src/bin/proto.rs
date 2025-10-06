#![allow(dead_code)]
// #[cfg(target_arch = "x86_64")]

// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR


// move code into examples directory
// cargo run --example demo

//TODO: Refactor with column major form after this implement a GMM
struct DecisionTree {
    dims:usize, // number of dims
    card:usize,
    // node:usize, // number of nodes
    leafs:Vec<usize>,
    nodes:Vec<Node>,
    metadata:Vec<Metadata>,
    // measure:Vec<Measure>, // measures for running values
    assignment:Vec<Assignment>, // idx -> node
    // features:Vec<Vec<Feature>>, // sorted features (idx, f[idx;j], y[idx])
    features:Vec<Vec<Feature>>, // sorted features (idx, f[idx;j], y[idx])
}

// split into Measure | Split Info

#[derive(Clone, Copy)]
struct Node {
    id:usize,
    prediction:f32,
    partition:Option<Partition>,
}
#[derive(Clone, Copy)]
struct Partition {
    dim:usize,
    value:f32, // value on which to split
    left:usize, // split left 
    right:usize, // split right
}

#[derive(Clone, Copy)]
struct Metadata {
    // minimal
    id:usize,
    dim:usize,
    // descriptives
    offset:usize,
    card:usize,
    sum_linear:f32, // Sum y
    sum_squares:f32, // Sum y * y;
}

struct Assignment {
    idx:usize,
    node:usize,
}

#[derive(Clone, Copy)]
// TODO: we only need indices
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
        let metadata = Metadata::derive(&data);
        let node = Node { id: 0, prediction: metadata.predict(), partition: None };
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
            leafs: vec![0],
            nodes: vec![node],
            metadata:vec![metadata],
            assignment,
            features,
        }
    }
    fn train(&mut self, nodes:usize) {
        for _ in 0..nodes {
            self.split();
        }
    }
    fn split(&mut self) {
        // running candidate partitions
        // TODO: Some nodes are no longer active this makes all nodes
        let leafs = self.leafs.len();
        let nodes = self.nodes.len();
        let mut runnings:Vec<Metadata> = (0..leafs).map( |i| {
            let idx = self.leafs[i];
            let parent = &self.metadata[idx];
            Metadata::default(parent.dim, parent.dim, parent.offset)
        }).collect();
        let (mut measure_delta, mut measure_partition) = (0_f32, 0_f32);
        let (mut current_idx, left_nidx, right_nidx) = (0, nodes, nodes+1);
        let mut left_meta= Metadata { 
            id:usize::MAX,
            card:usize::MAX,
            dim:usize::MAX,
            offset:usize::MAX,
            sum_linear:f32::MAX,
            sum_squares:f32::MAX
        };
        // find best new partition
        for d in 0..self.dims {
            let feature = &self.features[d];
            for idx in 0..self.card {
                let node = &self.assignment[idx];
                runnings[node.idx].increment(&feature[idx]);
                let delta = self.metadata[node.idx].delta(&runnings[node.idx]);
                if delta < measure_delta { continue; }
                current_idx = node.idx;
                measure_delta = delta;
                measure_partition = self.features[d][idx].value;
                left_meta = runnings[node.idx].clone();
            }
        }
        let current_node = self.metadata[current_idx];
        // update assignments for nodes
        for i in 0..current_node.card {
            let fidx = i + current_node.offset;
            let feature = self.features[left_meta.dim][fidx];
            if i < left_meta.card {
                self.assignment[feature.idx].node = left_nidx;
            } else {
                self.assignment[feature.idx].node = right_nidx;
            }
        }
        let mut buffer = vec![Feature::new(); current_node.card];
        let current = &self.metadata[current_idx];
        let (mut lidx, mut ridx) = (current_node.offset, current_node.offset + left_meta.card);
        // sort the sub partitions in the data
        // TODO: this needs to change to sorting the indices
        for d in 0..self.dims {
            if d == current_idx { continue; }
            let dim = &self.features[d];
            for fidx in current.offset..current.card + current.offset{
                let feature = self.features[left_meta.dim][fidx];
                if self.assignment[feature.idx].node == left_nidx {
                    buffer[ridx] = dim[fidx];
                    ridx += 1;
                } else {
                    buffer[lidx] = dim[fidx];
                    lidx += 1;
                }
            }
            self.features[d][current.offset..current.offset+current.card].copy_from_slice(&buffer);
        }
        let right_meta = self.metadata[current_idx].derive_right(&left_meta);
        let current_node = &mut self.nodes[current_idx];
        current_node.partition = Some(Partition {
            dim:left_meta.dim,
            value:measure_partition,
            left: left_nidx,
            right: right_nidx

        });
        self.metadata.push(left_meta);
        self.metadata.push(right_meta);
        let left_node = Node { id: left_nidx, prediction: left_meta.predict(), partition: None };
        let right_node = Node { id: right_nidx, prediction: right_meta.predict(), partition: None };
        self.nodes.push(left_node);
        self.nodes.push(right_node);
    }

    fn predict(&self, data:Vec<f32>) -> f32 {
        let mut node = self.nodes[0];
        while let Some(partition) = node.partition {
            if data[partition.dim] < partition.value {
                node = self.nodes[partition.left];
            } else {
                node = self.nodes[partition.right];
            }
        }
        node.prediction
    }
}

impl Node {
    // Contains information for decision tree structure
    fn new(id:usize, prediction:f32, partition:Option<Partition>) -> Self {
        Self {
            id,
            prediction,
            partition,
        }
    }
}


impl Metadata {
    // Contains information for splitting criterions
    fn default(id:usize, dim:usize, offset:usize) -> Self {
        Self {
            id: id,
            dim:dim,
            offset: offset,
            card: 0,
            sum_linear:0_f32, // Sum y
            sum_squares:0_f32, // Sum y * y;
        }
    }
    fn delta(&self, running:&Self) -> f32 {
        let (card, l_card, r_card) = (self.card as f32, running.card as f32, (self.card - running.card) as f32);
        
        let sse_curr= self.sum_squares - self.sum_linear * self.sum_linear / card;
        let sse_left = running.sum_squares - running.sum_linear * running.sum_linear / l_card;
        let sse_right = (self.sum_squares - running.sum_squares) - (self.sum_linear - running.sum_linear) * (self.sum_linear - running.sum_linear) / r_card;
        // weighted variance
        (sse_curr - sse_left - sse_right) / card
    }
    fn increment(&mut self, feature:&Feature) {
        self.card += 1;
        self.sum_linear += feature.label;
        self.sum_squares += feature.label * feature.label;
    }
    fn derive(data:&Vec<Vec<f32>>)  -> Self {
        if data.is_empty() || data[0].is_empty() { panic!("data is empty"); }
        let card = data.len();
        let label = data[0].len()-1;
        let (mut sum_linear, mut sum_squares) = (0_f32, 0_f32);
        for idx in 0..card {
            let val = data[idx][label];
            sum_linear += val;
            sum_squares += val * val;
        }
        Self {
            id:0,
            dim:label,
            offset:0,
            card,
            sum_linear,
            sum_squares
        }
    }
    fn derive_right(&mut self, left:&Self) -> Self {
        Self {
            id:left.id + 1,
            dim:self.dim,
            offset:left.offset + left.card,
            card:self.card - left.card,
            sum_linear:self.sum_linear - left.sum_linear,
            sum_squares:self.sum_squares - left.sum_squares,
        }
    }
    fn predict(&self) -> f32 {
        self.sum_linear / (self.card as f32)
    }
}

fn main() {
}
