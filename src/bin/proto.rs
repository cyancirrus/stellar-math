#![allow(dead_code)]
use rand::Rng;
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::mult_mat_vec;
use stellar::algebra::ndmethods::in_place_add;
use stellar::algebra::ndmethods::in_place_sub;
use stellar::algebra::vector::vec_in_place_add;
use std::collections::HashMap;
use std::collections::BinaryHeap;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::lu::lu_decompose;
use std::cmp::Ordering;
use std::marker::PhantomData;

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

impl State {
    fn computational(&self) -> Vec<f32> {
        vec![self.theta, self.velocity, self.x, self.y]
    }
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

impl Reference {
    fn new(prediction:Vec<f32>) -> Self {
        Self {
            x: prediction[2],
            y: prediction[3],
        }
    }
}

impl Signal for GpsSignal {
    type State = State;
    type Reference = Reference;
    type Data = GpsData;
    fn derive(basis:&Reference, new:&GpsData) -> State {
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
    fn jacobian(state:&State) -> NdArray {
        // theta, velocity, x, y
        let dt = 0.01f32; // to eventually become a clock but static for the moment
        let n = 4;
        let mut matrix = create_identity_matrix(n);
        // dx
        matrix.data[2 * n] = -dt * state.velocity * state.theta.sin();
        matrix.data[2 * n + 1] = dt * state.theta.cos();
        // dy
        matrix.data[3 * n] = dt * state.velocity * state.theta.cos();
        matrix.data[3 * n + 1] = dt * state.theta.sin();
        matrix
    }
    fn representation(state:&State) -> Vec<f32> {
        vec![
            state.theta,
            state.velocity,
            state.x,
            state.y,
        ]
    }
    fn insight(state:&State, data:&GpsData) -> Vec<f32> {
        vec![
            state.theta,
            state.velocity,
            data.x - state.x,
            data.y - state.y,
        ]
    }
}
impl Signal for VehicleSignal {
    type State = State;
    type Reference = Reference;
    type Data = VehicleData;
    fn derive(basis:&Reference, new:&VehicleData) -> State {
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
    fn jacobian(state:&State) -> NdArray {
        // theta, velocity, x, y
        let dt = 0.01f32; // to eventually become a clock but static for the moment
        let n = 4;
        let mut matrix = create_identity_matrix(n);
        let velocity_squared = (state.velocity * state.velocity).max(1e-6);
        // dtheta
        matrix.data[2] = - state.dy/velocity_squared;
        matrix.data[3] = state.dx /velocity_squared;
        // dvelocity
        matrix.data[n + 2] = state.dx/state.velocity;
        matrix.data[n + 3] = state.dy/state.velocity;
        // dx
        matrix.data[2 * n + 2] = dt * (state.dx / state.velocity * state.theta.cos() + state.velocity/velocity_squared * state.dy * state.theta.sin());
        matrix.data[2 * n + 3] = dt * (state.dy / state.velocity * state.theta.cos() - state.velocity/velocity_squared * state.dx * state.theta.sin());
        // dy
        matrix.data[3 * n + 2] = dt * (state.dx / state.velocity * state.theta.sin() - state.velocity/velocity_squared * state.dy * state.theta.cos());
        matrix.data[3 * n + 3] = dt * (state.dy / state.velocity * state.theta.sin() + state.velocity/velocity_squared * state.dx * state.theta.cos());
        matrix
    } 
    fn representation(state:&State) -> Vec<f32> {
        vec![
            state.theta,
            state.velocity,
            state.x,
            state.y,
        ]
    }
    fn insight(state:&State, data:&VehicleData) -> Vec<f32> {
        vec![
            data.theta - state.theta,
            data.velocity - state.velocity,
            state.x,
            state.y,
        ]
    }
}

trait Signal  {
    type State;
    type Reference;
    type Data;
    // S: State, R: Reference, D: Data;
    fn derive(basis:&Self::Reference, data:&Self::Data) -> Self::State;
    fn jacobian(state:&Self::State) -> NdArray;
    fn representation(state:&Self::State) -> Vec<f32>;
    fn insight(state:&Self::State, data:&Self::Data) -> Vec<f32>;
}


struct Ekf<R, S, F, H>
where
    F: Signal<State = S, Reference = R>, 
    H: Signal<State = S, Reference = R>,
{
    basis:R,
    // Signal Variances
    q_variance: NdArray,
    r_variance:NdArray,
    // Computational Matrices
    p:NdArray,
    h:NdArray,
    k:NdArray,
    // prediction and measurement
    _boo_f: PhantomData<F>,
    _boo_h: PhantomData<H>,
    _boo_s: PhantomData<S>,
}

impl <R, S, F, H> Ekf <R, S, F, H>
where
    F: Signal<State = S, Reference = R> ,
    H: Signal<State = S, Reference = R>,
{
    fn update_p(&mut self, state:F::State) {
        // takes in VehicleState
        // df/dx | x_{k|k-1};
        let f = F::jacobian(&state);
        let result = tensor_mult(4, &f, &self.p);
        self.p = tensor_mult(4, &result, &f.transpose());
        in_place_add(&mut self.p, &self.q_variance);
    }
    fn derive_k(&mut self, state:H::State) {
        // takes in GpsState
        // requires p to be updated prior
        self.h =  H::jacobian(&state);
        let mut s_k = tensor_mult(4, &self.h, &self.p);
        s_k = tensor_mult(4, &s_k, &self.h.transpose());
        in_place_add(&mut s_k, &self.r_variance);
        let mut k = tensor_mult(4, &self.p, &self.h);
        let lu = lu_decompose(s_k);
        lu.solve_inplace(&mut k);
        self.k = k;
    }
    fn finalize_p(&mut self) {
        let mut update = tensor_mult(4, &self.k, &self.h);
        update = tensor_mult(4, &update, &self.p);
        in_place_sub(&mut self.p, &update);
    }
    fn output(&mut self, prediction:&mut Vec<f32>, measurement:&Vec<f32>) {
        debug_assert_eq!(prediction.len(), measurement.len());
        let y_star = mult_mat_vec(&self.k, measurement);
        vec_in_place_add(prediction, &y_star);
    }
    fn predict_x(&mut self, basis:R, prediction:F::Data, measurement:H::Data) -> Vec<f32> {
        let vstate = F::derive(&basis, &prediction);
        let gstate = H::derive(&basis, &measurement);
        let mut prediction = F::representation(&vstate);
        let measurement = H::insight(&gstate, &measurement);
        self.update_p(vstate);
        self.derive_k(gstate);
        self.finalize_p();
        self.output(&mut prediction, &measurement);
        // TODO: update this to inegrate well
        // self.basis = R::new(prediction);
        prediction
    }
}


// struct EkfVehicleGps {
//     basis:Reference,
//     // variance for the prediction
//     q_variance:NdArray,
//     // variance for the measurement
//     r_variance:NdArray,
//     p:NdArray,
//     h:NdArray,
//     k:NdArray,
// }

// impl EkfVehicleGps {
//     fn update_p(&mut self, state:State) {
//         // takes in VehicleState
//         // df/dx | x_{k|k-1};
//         let f = VehicleSignal::jacobian(&state);
//         let result = tensor_mult(4, &f, &self.p);
//         self.p = tensor_mult(4, &result, &f.transpose());
//         in_place_add(&mut self.p, &self.q_variance);
//     }
//     fn derive_k(&mut self, state:State) {
//         // takes in GpsState
//         // requires p to be updated prior
//         self.h =  GpsSignal::jacobian(&state);
//         let mut s_k = tensor_mult(4, &self.h, &self.p);
//         s_k = tensor_mult(4, &s_k, &self.h.transpose());
//         in_place_add(&mut s_k, &self.r_variance);
//         let mut k = tensor_mult(4, &self.p, &self.h);
//         let lu = lu_decompose(s_k);
//         lu.solve_inplace(&mut k);
//         self.k = k;
//     }
//     fn finalize_p(&mut self) {
//         let mut update = tensor_mult(4, &self.k, &self.h);
//         update = tensor_mult(4, &update, &self.p);
//         in_place_sub(&mut self.p, &update);
//     }
//     fn output(&mut self, prediction:&mut Vec<f32>, measurement:&Vec<f32>) {
//         debug_assert_eq!(prediction.len(), measurement.len());
//         let y_star = mult_mat_vec(&self.k, measurement);
//         vec_in_place_add(prediction, &y_star);
//     }
//     fn predict_x(&mut self, basis:Reference, vehicle:VehicleData, gps:GpsData) -> Vec<f32> {
//         let vstate = VehicleSignal::derive(&basis, &vehicle);
//         let gstate = GpsSignal::derive(&basis, &gps);
//         let mut prediction = VehicleSignal::representation(&vstate);
//         let measurement = GpsSignal::insight(&gstate, &gps);
//         self.update_p(vstate);
//         self.derive_k(gstate);
//         self.finalize_p();
//         self.output(&mut prediction, &measurement);
//         self.basis = Reference{ x:prediction[2], y: prediction[3]};
//         prediction
//     }
// }

fn main() {
    println!("hello world");
    // // test_gmm_3d_kmeans_gmm();
    // // test_gmm_3d();
    // if let Err(e) = test_kmeans_gmm_visual() {
    //     eprintln!("Error: {}", e);
    // }
}
