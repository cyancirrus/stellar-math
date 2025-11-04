#![allow(dead_code)]
use rand::Rng;
use stellar::structure::ndarray::NdArray;
use std::collections::HashMap;
use rand_distr::StandardNormal;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::cmp::Ordering;
use std::rc::Rc;

struct MinNode {
    id:usize,
    cost:f32,
}


#[derive(Debug)]
struct EdgeNode {
    edge_id:usize,
    node_id:usize,
    cost:f32,
    speed:f32,
    distance: f32,
}
impl PartialEq for MinNode {
    fn eq(&self, other:&Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for MinNode {}
impl Ord for MinNode {
    fn cmp(&self, other:&Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}
impl PartialOrd for MinNode {
    fn partial_cmp(&self, other:&Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
struct Network {
    web: HashMap<usize, Vec<usize>>,
    // contains cost for edges
    edges: Vec<Vec<f32>>,
    // contains node_id -> (x,y)
    nodes: Vec<(f32, f32)>,
    // contains (src, target) -> speed
    speed: Vec<Vec<f32>>,
    // contains number of nodes
    size: usize,
}

impl Network {
    fn find_path(&self, start:usize, end:usize) -> Vec<usize> {
        let mut heap: BinaryHeap<MinNode>= BinaryHeap::new();
        let mut seen:Vec<bool> =  vec![false; self.size];
        let mut prev:Vec<usize> = vec![usize::MAX; self.size];
        heap.push(MinNode { id: start, cost: 0f32});

        while let Some(node) = heap.pop() {
            seen[node.id] = true;
            if node.id == end { break; }
            for &target in self.web.get(&node.id).unwrap() {
                if !seen[target] {
                    prev[target] = node.id;
                    heap.push( 
                        MinNode {
                            id: target,
                            cost: node.cost + self.edges[node.id][target],
                        }
                    );
                }
            }
        }
        let mut path = vec![];
        if prev[end] == usize::MAX { return path };
        let mut curr = end;
        while curr != start {
            path.push(curr);
            curr = prev[curr];
        }
        path.push(start);
        path.reverse();
        path
    }
}


struct Vehicle {
    theta: f32,
    velocity: f32,
}
struct Metadata {
    prediction: (f32, f32),
    measurement: (f32, f32),
    var_prediction: NdArray,
    var_meaurement: NdArray,
}

impl Vehicle {
    fn derivative(&self) -> (f32, f32) {
        (self.theta.cos(), self.theta.sin()) 
    }
    fn prediction_position(&self) -> (f32, f32) {
        (self.theta.cos(), self.theta.sin()) 
    }
    fn prediction(&self, meta:&mut Metadata) -> (f32, f32) {
        meta.prediction.0 += self.velocity * self.theta.cos(); 
        meta.prediction.1 += self.velocity * self.theta.sin(); 
        meta.prediction
    }
    fn measurement(&self, path:Path) -> (f32, f32) {
        let mut rng = rand::rng();
        let rx:f32 = rng.sample(StandardNormal);
        let ry:f32 = rng.sample(StandardNormal);
        (path.position.0 + rx, path.position.1 + ry)
        
    }
}

// x[k ; k] = x[k] + P[k] * y_k;
// y_k = z[k] - h[k];
// x[k ; k - 1] = f(x_k, u_k) + wk, wk ~ N(0, Q_k)
// z[k ; k - 1] = h(x_k, v_k) + vk, vk ~ N(0, R_k)

// F_k ~ df/dx | x_k | k - 1
// f(x + dh) = f(x) + dh * f'(x);
// => variance proportional to the delta term
// => F_k = df/dx => Var
// P_{k|k-1} = F_k * P_{k|k-1} * F_k' + Q_k
// S_k = H_k P_{k|k-1} H'k + R_k;
// K_k = P_{k|k-1} H'kS_k^-1


fn main() {
    println!("hello world");
    // // test_gmm_3d_kmeans_gmm();
    // // test_gmm_3d();
    // if let Err(e) = test_kmeans_gmm_visual() {
    //     eprintln!("Error: {}", e);
    // }
}
