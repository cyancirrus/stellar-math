use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

// TODO: Implement Kahan Summation
// reference : https://en.wikipedia.org/wiki/Kahan_summation_algorithm

pub struct DecisionTree<'a> {
    data: &'a Vec<Vec<f32>>, // feature major form, individual observations are columns
    sample: (usize, usize),  // observation sample, dimension sample
    dims: usize,             // number of dims
    card: usize,
    assign: Vec<usize>, // idx -> node
    nodes: Vec<Node>,
    metadata: Vec<Metadata>,
    dimensions: Vec<Vec<usize>>, // idx sorted by dimension
}
pub struct DecisionTreeModel {
    pub nodes: Vec<Node>,
    pub metadata: Vec<Metadata>,
}
#[derive(Clone, Copy)]
pub struct Node {
    prediction: f32,
    partition: Option<Partition>,
}
#[derive(Clone, Copy)]
struct Partition {
    dim: usize,   // dimension to consider
    value: f32,   // value of dimension
    left: usize,  // split left x <= value
    right: usize, // split right x > value
}
#[derive(Clone, Copy)]
pub struct Metadata {
    // minimal
    pub dim: usize,
    // descriptives
    offset: usize,
    card: usize,
    sum_linear: f32,  // Sum y
    sum_squares: f32, // Sum y * y;
}

impl DecisionTreeModel {
    pub fn predict(&self, data: &[f32]) -> f32 {
        let mut node = &self.nodes[0];
        while let Some(partition) = &node.partition {
            if data[partition.dim] < partition.value {
                node = &self.nodes[partition.left];
            } else {
                node = &self.nodes[partition.right];
            }
        }
        node.prediction
    }
    pub fn analyze_gains(&self) -> Vec<f32> {
        let nodes = self.metadata.len();
        let mut reductions = Vec::with_capacity(nodes);
        let mut queue = VecDeque::from([0]);

        while let Some(idx) = queue.pop_front() {
            if let Some(p) = self.nodes[idx].partition {
                let delta = self.metadata[idx].sse()
                    - self.metadata[p.left].sse()
                    - self.metadata[p.right].sse();
                reductions.push(delta);
                queue.push_back(p.left);
                queue.push_back(p.right);
            }
        }
        reductions
    }
    pub fn analyze_variance(&self) -> Vec<f32> {
        let total_sse = self.metadata[0].sse();
        let gains = self.analyze_gains();
        let mut cumulative = 0_f32;

        for g in &gains {
            cumulative += g;
            println!("Variance Explained: {:3}%", 100f32 * cumulative / total_sse);
        }
        gains
    }
}

impl<'a> DecisionTree<'a> {
    pub fn new(data: &'a Vec<Vec<f32>>, obs_sample: f32, dim_sample: f32) -> Self {
        if data.is_empty() || data[0].is_empty() {
            panic!("data is empty");
        }
        if obs_sample > 1_f32 || obs_sample < 0_f32 || dim_sample > 1_f32 || dim_sample < 0_f32 {
            panic!("cannot subsample range");
        }
        let label = data.len();
        let dims = label - 1;
        let n_rows = data[0].len();
        let mut rng = rand::rng();
        let mut buffer: Vec<usize> = (0..n_rows).collect();
        let sample = (
            (obs_sample * n_rows as f32 + 1_f32) as usize,
            (dim_sample * dims as f32 + 1f32) as usize,
        );
        buffer.shuffle(&mut rng);
        buffer.truncate(sample.0);
        let card = buffer.len();
        let assign: Vec<usize> = vec![0; n_rows];
        let mut dimensions = Vec::with_capacity(dims);
        let metadata = Metadata::derive(&buffer, &data);
        let node = Node {
            prediction: metadata.predict(),
            partition: None,
        };
        for d in 0..dims {
            // sort indices by dimension
            buffer.sort_by(|a, b| data[d][*a].partial_cmp(&data[d][*b]).unwrap());
            dimensions.push(buffer.clone());
        }
        Self {
            data,
            sample,
            assign,
            dims,
            card,
            nodes: vec![node],
            metadata: vec![metadata],
            dimensions,
        }
    }
    pub fn train(&mut self, nodes: usize) -> DecisionTreeModel {
        for _ in 0..nodes {
            self.split();
        }
        DecisionTreeModel {
            nodes: self.nodes.clone(),
            metadata: self.metadata.clone(),
        }
    }
    fn split(&mut self) {
        let childs = (self.nodes.len(), self.nodes.len() + 1);
        let (split, range) = self.find_partition();
        self.update_assignment(split.1, childs, range);
        self.sort_dimensions(split.1, childs, range);
        self.update_metadata(split, childs);
    }
    fn find_partition(&mut self) -> ((usize, usize, f32), (usize, usize, usize)) {
        let nodes = self.nodes.len();
        let yindex = self.dims;
        let (mut ancestor, mut dimension) = (usize::MAX, usize::MAX);
        let (mut delta, mut partition) = (f32::NEG_INFINITY, f32::NEG_INFINITY);

        let mut rng = rand::rng();
        // considered dims
        let mut dims: Vec<usize> = (0..self.dims).collect();
        dims.shuffle(&mut rng);
        dims.truncate(self.sample.1);

        let mut runnings: Vec<Metadata> = (0..nodes)
            .map(|idx| {
                let parent = &self.metadata[idx];
                Metadata::empty_from(parent.dim, parent.offset)
            })
            .collect();
        let mut target = Metadata {
            card: usize::MAX,
            dim: usize::MAX,
            offset: usize::MAX,
            sum_linear: f32::MAX,
            sum_squares: f32::MAX,
        };
        let output = &self.data[yindex];
        for d in dims {
            for node in &mut runnings {
                node.reset();
            }
            let dval = &self.data[d];
            for &idx in &self.dimensions[d] {
                let node = self.assign[idx];
                let (dval, yval) = (dval[idx], output[idx]);
                runnings[node].increment(yval);
                let del = self.metadata[node].delta(&runnings[node]);
                if del < delta {
                    continue;
                }
                ancestor = node;
                dimension = d;
                delta = del;
                partition = dval;
                target = runnings[node].clone();
            }
        }
        let parent = self.metadata[ancestor];
        let complement = parent.derive_complement(&target);
        self.metadata.push(target);
        self.metadata.push(complement);
        let left_node = Node {
            prediction: target.predict(),
            partition: None,
        };
        let right_node = Node {
            prediction: complement.predict(),
            partition: None,
        };
        self.nodes.push(left_node);
        self.nodes.push(right_node);
        let split = (ancestor, dimension, partition);
        let range = (
            parent.offset,
            parent.offset + target.card,
            parent.offset + parent.card,
        );
        (split, range)
    }
    fn update_assignment(
        &mut self,
        dim: usize,
        childs: (usize, usize),
        range: (usize, usize, usize),
    ) {
        let (start, split, end) = range;
        // update assignments for nodes
        for idx in start..end {
            let nidx = self.dimensions[dim][idx];
            if idx < split {
                self.assign[nidx] = childs.0;
            } else {
                self.assign[nidx] = childs.1;
            }
        }
    }
    fn sort_dimensions(
        &mut self,
        dim: usize,
        childs: (usize, usize),
        range: (usize, usize, usize),
    ) {
        let (start, split, end) = range;
        let mut buffer = vec![usize::MAX; end - start];
        for d in 0..self.dims {
            if d == dim {
                continue;
            }
            let (mut lidx, mut ridx) = (0, split - start);
            let dimension = &self.dimensions[d];
            for idx in start..end {
                let feature = dimension[idx];
                if self.assign[feature] == childs.0 {
                    buffer[lidx] = dimension[idx];
                    lidx += 1;
                } else if self.assign[feature] == childs.1 {
                    buffer[ridx] = dimension[idx];
                    ridx += 1;
                } else {
                    panic!("corruption in the split assignment");
                }
            }
            self.dimensions[d][start..end].copy_from_slice(&buffer);
        }
    }
    fn update_metadata(&mut self, split: (usize, usize, f32), childs: (usize, usize)) {
        let node = &mut self.nodes[split.0];
        node.partition = Some(Partition {
            dim: split.1,
            value: split.2,
            left: childs.0,
            right: childs.1,
        });
    }
}

impl Metadata {
    pub fn sse(&self) -> f32 {
        self.sum_squares - self.sum_linear * self.sum_linear / self.card as f32
    }
    // Contains information for splitting criterions
    fn empty_from(dim: usize, offset: usize) -> Self {
        Self {
            dim: dim,
            offset: offset,
            card: 0,
            sum_linear: 0_f32,  // Sum y
            sum_squares: 0_f32, // Sum y * y;
        }
    }
    fn reset(&mut self) {
        self.card = 0;
        self.sum_linear = 0_f32;
        self.sum_squares = 0_f32;
    }
    fn delta(&self, running: &Self) -> f32 {
        if self.card == 0 || running.card == 0 || self.card == running.card {
            return 0_f32;
        };
        let (card, l_card, r_card) = (
            self.card as f32,
            running.card as f32,
            (self.card - running.card) as f32,
        );
        let sse_curr = self.sum_squares - self.sum_linear * self.sum_linear / card;
        let sse_left = running.sum_squares - running.sum_linear * running.sum_linear / l_card;
        let sse_right = (self.sum_squares - running.sum_squares)
            - (self.sum_linear - running.sum_linear) * (self.sum_linear - running.sum_linear)
                / r_card;
        // weighted variance
        (sse_curr - sse_left - sse_right) / card
    }
    fn increment(&mut self, output: f32) {
        self.card += 1;
        self.sum_linear += output;
        self.sum_squares += output * output;
    }
    fn derive(include: &[usize], data: &Vec<Vec<f32>>) -> Self {
        if data.is_empty() || data[0].is_empty() {
            panic!("data is empty");
        }
        let label = data.len() - 1;
        let card = include.len();
        let (mut sum_linear, mut sum_squares) = (0_f32, 0_f32);
        for &idx in include {
            let val = data[label][idx];
            sum_linear += val;
            sum_squares += val * val;
        }
        Self {
            dim: label,
            offset: 0,
            card,
            sum_linear,
            sum_squares,
        }
    }
    fn derive_complement(&self, target: &Self) -> Self {
        Self {
            dim: self.dim,
            offset: target.offset + target.card,
            card: self.card - target.card,
            sum_linear: self.sum_linear - target.sum_linear,
            sum_squares: self.sum_squares - target.sum_squares,
        }
    }
    fn predict(&self) -> f32 {
        self.sum_linear / (self.card as f32)
    }
}
