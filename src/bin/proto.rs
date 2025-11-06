#![allow(dead_code)]
use rand::Rng;
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::create_identity_matrix;
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
    // contains cost for edges could flatten this
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

struct State {
    theta: f32,
    velocity: f32,
    x: f32,
    y: f32, 
    dx: f32,
    dy: f32,
}

struct GpsSignal {}
struct VehicleSignal {}

struct Reference {
    x:f32,
    y:f32,
}
struct GpsData {
    x:f32,
    y:f32,
}
struct VehicleData {
    theta: f32,
    velocity:f32,
}
impl GpsSignal {
    fn derive(basis:Reference, new:GpsData) -> State {
        let dt = 0.01f32; // to eventually become a clock but static for the moment
        let w = (new.x - basis.x, new.y - basis.y); // (dx, dy)
        let theta = w.1.atan2(w.0); // theta based upon the changes
        let velocity = (w.0 * w.0 + w.1 * w.1).sqrt(); //magnitude of root(dx^2 + dy^2)
        // update the previous state
        let dx = dt * velocity * theta.cos();
        let dy = dt * velocity * theta.sin();
        State {
            theta,
            velocity,
            x: basis.x + dx,
            y: basis.y + dy,
            dx,
            dy,
        }
    }
    fn jacobian(&self, state:State) -> NdArray {
        // theta, velocity, x, y
        let dt = 0.01f32; // to eventually become a clock but static for the moment
        let n = 4;
        let mut matrix = create_identity_matrix(n);
        matrix.data[2 * n] = -dt * state.velocity * state.theta.sin();
        matrix.data[2 * n + 1] = dt * state.theta.cos();
        matrix.data[3 * n] = dt * state.velocity * state.theta.cos();
        matrix.data[3 * n + 1] = dt * state.theta.sin();
        matrix
    }
}
impl VehicleSignal {
    fn derive(basis:Reference, new:VehicleData) -> State {
        let dt = 0.01f32; // to eventually become a clock but static for the moment
        let dx = dt * new.velocity * new.theta.cos();
        let dy = dt * new.velocity * new.theta.sin();
        State {
            theta: new.theta,
            velocity: new.velocity,
            x: basis.x + dx,
            y: basis.y + dy,
            dx,
            dy,
        }
    }
    fn jacobian(&self, state:State) -> NdArray {
        // theta, velocity, x, y
        let dt = 0.01f32; // to eventually become a clock but static for the moment
        let n = 4;
        let mut matrix = create_identity_matrix(n);
        let velocity_squared = (state.velocity * state.velocity).max(1e-6);
        matrix.data[2] = - state.dy/velocity_squared;
        matrix.data[3] = state.dx /velocity_squared;
        matrix.data[n + 2] = state.dx/state.velocity;
        matrix.data[n + 3] = state.dy/state.velocity;
        matrix.data[2 * n + 2] = dt * (state.dx / state.velocity * state.theta.cos() + state.velocity/velocity_squared * state.dy * state.theta.sin());
        matrix.data[2 * n + 3] = dt * (state.dy / state.velocity * state.theta.cos() - state.velocity/velocity_squared * state.dx * state.theta.sin());
        matrix.data[3 * n + 2] = dt * (state.dx / state.velocity * state.theta.sin() - state.velocity/velocity_squared * state.dy * state.theta.cos());
        matrix.data[3 * n + 3] = dt * (state.dy / state.velocity * state.theta.sin() + state.velocity/velocity_squared * state.dx * state.theta.cos());
        matrix
    } 
}

trait Sensor {
    fn measure(&self, truth: &State) -> State;
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
