use crate::decomposition::lower_upper::LuPivotDecompose;
use crate::structure::ndarray::NdArray;

const EPSILON: f32 = 1e-6;

pub struct DualLinear {}
impl DualLinear {
    pub fn transform(a: NdArray) -> NdArray {
        debug_assert!(a.dims[0] >= a.dims[1], "m < n should not take dual");
        let (m, n) = (a.dims[0], a.dims[1]);
        let n_cols = 2 * m + 2 * n;
        let mut data = Vec::with_capacity(n * n_cols);
        for row in 0..n {
            // A'
            for col in 0..m {
                data.push(a.data[col * n + row]);
            }
            // -A'
            for col in 0..m {
                data.push(-a.data[col * n + row]);
            }
            // I_slack
            for col in 0..n {
                data.push(if row == col { 1.0 } else { 0.0 });
            }
            // I_artificial
            for col in 0..n {
                data.push(if row == col { 1.0 } else { 0.0 });
            }
        }
        debug_assert!(2 * n * (m + n) == data.len(), "data length not correct");
        NdArray {
            dims: vec![n, n_cols],
            data,
        }
    }
}

// Ax = b; min sum(x); x[i] >= 0;
pub struct LinearProgram {
    pub class: ProgramClass,
    // base matrix dimension
    m: usize,
    n: usize,
    // cost vector, target
    c: Vec<f32>,
    b: Vec<f32>,
    pub x: Vec<f32>,
    basis: Vec<usize>,
    // [A', -A', I_slack, I_artificial];
    // [m, m, n, n];
    constraint: NdArray,
}

pub enum ProgramClass {
    Primal,
    Dual,
}

impl LinearProgram {
    pub fn new(c: Vec<f32>, b: Vec<f32>, matrix: NdArray) -> Self {
        let (m, n) = (matrix.dims[0], matrix.dims[1]);
        debug_assert!(m >= n, "Primal initializer not yet created");
        debug_assert_eq!(c.len(), n);
        debug_assert_eq!(b.len(), m);
        let x = vec![1.0; n];
        let basis = (2 * m + n..2 * m + 2 * n).collect();
        LinearProgram {
            class: ProgramClass::Dual,
            m,
            n,
            c,
            b,
            x,
            basis,
            constraint: DualLinear::transform(matrix),
        }
    }
    fn setup_phase_one(&mut self) {
        self.basis = (2 * self.m + self.n..2 * self.m + 2 * self.n).collect();
    }
    fn get_basis_matrix(&self) -> NdArray {
        let mut data = Vec::with_capacity(self.m * self.m);
        for &col in &self.basis {
            for row in 0..self.m {
                data.push(self.constraint.data[row * self.n + col]);
            }
        }
        NdArray {
            dims: vec![self.m, self.m],
            data,
        }
    }
    fn pivot(&mut self, entering_idx: usize, leaving_idx: usize) {
        self.basis[leaving_idx] = entering_idx;
    }
    fn compute_direction(&self, entering_idx: usize) -> Vec<f32> {
        // Travels in the inverse of this vector
        let b = self.get_basis_matrix();
        let lu = LuPivotDecompose::new(b.transpose());
        let n_cols = 2 * self.n + 2 * self.m;

        let mut a_j = vec![0.0; self.n];
        for i in 0..self.n {
            // column basis vector
            a_j[i] = self.constraint.data[i * n_cols + entering_idx];
        }
        let mut direction = a_j;
        lu.solve_inplace_vec(&mut direction);
        direction
    }
    fn select_entering_variable(&self, delta_costs: &[f32]) -> Option<usize> {
        // Option returned in order to get terminating condition
        println!("delta costs {delta_costs:?}");
        let mut best_idx = None;
        let mut best_val = 0.0;

        for i in 0..delta_costs.len() {
            // More negative is better cost reduction
            if delta_costs[i] < best_val {
                best_val = delta_costs[i];
                best_idx = Some(i);
            }
        }
        best_idx
    }
    fn compute_phase_one_delta_cost(&self) -> Vec<f32> {
        let b = self.get_basis_matrix();
        let lu = LuPivotDecompose::new(b.transpose());
        println!("lu {:?}", lu.matrix);
        let n_cols = 2 * self.m + 2 * self.n;

        let mut cost_b = vec![0.0; self.n];
        for (i, &basis_idx) in self.basis.iter().enumerate() {
            if basis_idx >= 2 * self.m + self.n {
                cost_b[i] = 1.0;
            }
        }
        let mut y = cost_b.clone();
        lu.solve_inplace_vec(&mut y);
        let mut delta = vec![0.0; n_cols];
        for j in 0..n_cols {
            if self.basis.contains(&j) {
                continue;
            }
            // penalize if in artificial
            let c_j = if j > 2 * self.m + self.n { 1.0 } else { 0.0 };
            let mut y_dot_aj = 0.0;
            for i in 0..self.n {
                y_dot_aj += y[i] * self.constraint.data[i * n_cols + j];
            }
            delta[j] = c_j - y_dot_aj;
        }
        delta
    }
    fn compute_phase_two_delta_cost(&self) -> Vec<f32> {
        let b = self.get_basis_matrix();
        let lu = LuPivotDecompose::new(b.transpose());
        let n_cols = 2 * self.m + 2 * self.n;

        let mut cost_b = vec![0.0; self.n];
        for (i, &basis_idx) in self.basis.iter().enumerate() {
            cost_b[i] = self.c[basis_idx];
        }
        let mut y = cost_b.clone();
        lu.solve_inplace_vec(&mut y);
        let mut delta = vec![0.0; n_cols];
        for j in 0..n_cols {
            if self.basis.contains(&j) {
                continue;
            }
            let mut y_dot_ai = 0.0;
            for i in 0..self.n {
                y_dot_ai += y[i] * self.constraint.data[i * n_cols + j];
            }
            // for maximization want positive delta
            delta[j] = self.c[j] - y_dot_ai;
        }
        delta
    }
    fn ratio_test(&self, direction: &[f32]) -> Option<usize> {
        let x_b = self.get_basic_solution();
        let mut min_ratio = f32::INFINITY;
        let mut idx_leaving = None;
        // for (i, &basis_idx) in self.basis.iter().enumerate() {
        for (i, _) in self.basis.iter().enumerate() {
            if direction[i] > EPSILON {
                let ratio = x_b[i] / direction[i];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    idx_leaving = Some(i);
                }
            }
        }
        idx_leaving
    }
    fn get_basic_solution(&self) -> Vec<f32> {
        // let basis = self.get_basis_matrix();
        // let lu = LuPivotDecompose::new(basis.transpose());
        let x_basic = self.b.clone();
        let mut x = vec![0.0; self.n];
        for (i, &basis_idx) in self.basis.iter().enumerate() {
            x[basis_idx] = x_basic[i];
        }
        x
    }
    pub fn run_phase_one(&mut self) -> Result<(), String> {
        self.setup_phase_one();
        loop {
            let delta = self.compute_phase_one_delta_cost();
            println!("hello");
            let entering = match self.select_entering_variable(&delta) {
                Some(idx) => idx,
                None => break,
            };
            println!("world");
            let direction = self.compute_direction(entering);
            let leaving = match self.ratio_test(&direction) {
                Some(idx) => idx,
                None => return Err(format!("Unbounded should not happen in phase 1").into()),
            };
            self.pivot(entering, leaving)
        }
        let x = self.get_basic_solution();
        for &basic_idx in &self.basis {
            if basic_idx >= 2 * self.m + self.n {
                if x[basic_idx].abs() > EPSILON {
                    return Err(format!("Infeasable Problem").into());
                }
            }
        }
        Ok(())
    }
    pub fn run_phase_two(&mut self) -> Result<Vec<f32>, String> {
        loop {
            let delta = self.compute_phase_two_delta_cost();
            let entering = match self.select_entering_variable(&delta) {
                Some(idx) => idx,
                None => break,
            };
            let direction = self.compute_direction(entering);
            let leaving = match self.ratio_test(&direction) {
                Some(idx) => idx,
                None => return Err(format!("Unbounded").into()),
            };
            self.pivot(entering, leaving);
        }
        Ok(self.get_basic_solution())
    }
}

pub enum Phase {
    FindVertex,
    FindOptimum,
}

// fn main() {
//     // let thing = vec![vec![3], vec![1,3], vec![2], vec![2,3], vec![0,2], vec![0,1]];
//     let c = vec![1.0; 4];
//     let b = vec![3.0, 5.0, 4.0, 7.0];
//     let matrix = NdArray::new(
//         vec![4, 4],
//         vec![
//             0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
//             0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
//         ],
//     );
//     let mut lp = LinearProgram::new(c,  b, matrix);
//     println!("phase one");
//     let res_one = lp.run_phase_one();
//     println!("phase two");
//     let res_two = lp.run_phase_two();
//     match (res_one, res_two) {
//         (Ok(r1), Ok(r2)) => {
//             println!("Successful run of both!");
//             println!("Result {r2:?}");
//         },
//         (Err(e1), Err(e2)) => {
//             println!("Unsucessful\ne1: {e1:?}\ne2: {e2:?}");
//         },
//         (Err(e1), Ok(_)) => {
//             println!("Unsucessful\ne1: {e1:?}");
//         }
//         (Ok(_), Err(e2)) => {
//             println!("Unsucessful\ne2: {e2:?}");
//         }
//     }
//     // // test_gmm_3d_kmeans_gmm();
//     // // test_gmm_3d();
//     // if let Err(e) = test_kmeans_gmm_visual() {
//     //     eprintln!("Error: {}", e);
//     // }
// }
