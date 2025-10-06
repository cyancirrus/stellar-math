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

    fn find_partitions(&mut self) -> (usize, usize, f32, usize) {
        //  returns (ancestor, dimension, value)
        // TODO: Some nodes are no longer active this makes all nodes
        let leafs = self.leafs.len();
        let mut runnings:Vec<Metadata> = (0..leafs).map( |i| {
            let idx = self.leafs[i];
            let parent = &self.metadata[idx];
            Metadata::default(parent.dim, parent.dim, parent.offset)
        }).collect();
        let (mut ancestor, mut dimension) = (usize::MAX, usize::MAX);
        let (mut delta, mut partition) = (0_f32, 0_f32);
        let mut target = Metadata { 
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
                let del = self.metadata[node.idx].delta(&runnings[node.idx]);
                if del< delta { continue; }
                ancestor = node.idx;
                dimension = d;
                delta = delta;
                partition = self.features[d][idx].value;
                target = runnings[node.idx].clone();
            }
        }
        let compliment = self.metadata[ancestor].derive_compliment(&target);
        self.metadata.push(target);
        self.metadata.push(compliment);
        (ancestor, dimension, partition, target.card)
    }
    fn update_assignment(&mut self, dim:usize, children:(usize, usize), range:(usize, usize, usize)) {
        let (start, split, end) = range;
        // update assignments for nodes
        for idx in start..end {
            let feature = self.features[dim][idx];
            if idx < split {
                self.assignment[feature.idx].node = children.0;
            } else {
                self.assignment[feature.idx].node = children.1;
            }
        }
    }
    fn sort_dimensions(&mut self, dim:usize, childs:(usize, usize), range:(usize, usize, usize)) {
        let (start, split, end) = range;
        let (mut lidx, mut ridx) = (0, split - start);
        let mut buffer = vec![Feature::new();end - start];
        // TODO: this needs to change to sorting the indices
        for d in 0..self.dims {
            if d == dim { continue; }
            let dimension = &self.features[d];
            for idx in start..end {
                let feature = dimension[idx];
                if self.assignment[feature.idx].node == childs.0 {
                    buffer[lidx] = dimension[idx];
                    ridx += 1;
                } else {
                    buffer[ridx] = dimension[idx];
                    lidx += 1;
                }
            }
            self.features[d][start..end].copy_from_slice(&buffer);
        }
    }
    fn update_ancestors(&mut self, ancestor:usize) {
        // remove the ancestor from the item
        for idx in 0..self.leafs.len() {
            if self.leafs[idx] != ancestor { continue; }
            self.leafs.swap_remove(idx);
        }
    }
    fn update_metadata(&mut self, ancestor:usize, split:(usize, f32), childs:(usize, usize)) {
        let mut node = self.nodes[ancestor];
        node.partition = Some(Partition {
            dim:split.0,
            value:split.1,
            left: childs.0,
            right: childs.1
        });
    }
    fn update_nodes(&mut self, childs:(usize, usize), predictions:(f32, f32)) {
        let left_node = Node { id: childs.0, prediction: predictions.0, partition: None };
        let right_node = Node { id: childs.1, prediction: predictions.1, partition: None };
        self.nodes.push(left_node);
        self.nodes.push(right_node);
    }
    fn split(&mut self) {
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
    fn derive_compliment(&mut self, left:&Self) -> Self {
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
