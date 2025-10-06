// current work in progressshould not be public yet

struct DecisionTree {
    data: Vec<Vec<f32>>, // feature major form, individual observations are columns
    dims: usize,         // number of dims
    card: usize,
    assign: Vec<usize>, // idx -> node
    nodes: Vec<Node>,
    metadata: Vec<Metadata>,
    dimensions: Vec<Vec<usize>>, // idx sorted by dimension
}
struct Node {
    prediction: f32,
    partition: Option<Partition>,
}
struct Partition {
    dim: usize,   // dimension to consider
    value: f32,   // value of dimension
    left: usize,  // split left x <= value
    right: usize, // split right x > value
}

#[derive(Clone, Copy)]
struct Metadata {
    // minimal
    dim: usize,
    // descriptives
    offset: usize,
    card: usize,
    sum_linear: f32,  // Sum y
    sum_squares: f32, // Sum y * y;
}

impl DecisionTree {
    fn new(data: Vec<Vec<f32>>) -> Self {
        if data.is_empty() || data[0].is_empty() {
            panic!("data is empty");
        }
        let label = data.len();
        let dims = label - 1;
        let card = data[0].len();
        let assign: Vec<usize> = (0..card).collect();
        let mut buffer: Vec<usize> = (0..card).collect();
        let mut dimensions = Vec::with_capacity(dims);
        let metadata = Metadata::derive(&data);
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
            assign,
            dims,
            card,
            nodes: vec![node],
            metadata: vec![metadata],
            dimensions,
        }
    }
    fn train(&mut self, nodes: usize) {
        for _ in 0..nodes {
            self.split();
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
        let (mut delta, mut partition) = (0_f32, 0_f32);
        // if iterate over leaves becomes worse could use hashmap but doesn't appear great
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
        for d in 0..self.dims {
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
        let (mut lidx, mut ridx) = (0, split - start);
        let mut buffer = vec![usize::MAX; end - start];
        for d in 0..self.dims {
            if d == dim {
                continue;
            }
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
    fn predict(&self, data: Vec<f32>) -> f32 {
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
}

impl Metadata {
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
    fn delta(&self, running: &Self) -> f32 {
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
    fn derive(data: &Vec<Vec<f32>>) -> Self {
        if data.is_empty() || data[0].is_empty() {
            panic!("data is empty");
        }
        let label = data.len() - 1;
        let card = data[0].len();
        let (mut sum_linear, mut sum_squares) = (0_f32, 0_f32);
        for val in &data[label] {
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
