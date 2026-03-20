#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

struct HhMatrixRowMajor {}

///  AutumnDecomp
/// QrDecomp
///
/// * h: HouseholderMatrix - row major form
/// * t: tau vector with cardinality
/// * rows: number of rows in the original matrix A
/// * cols: number of cols in the original matrix A
/// * card: TODO: need this for rank-k
pub struct AutumnDecomp {
    pub h: NdArray,
    pub t: Vec<f32>,
    cols: usize,
    rows: usize,
}

fn params(v: &mut [f32], rows: usize, p: usize) -> f32 {
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element == 0f32 {
        return max_element;
    }
    let mut magnitude_squared = 0f32;
    let inv_max_element = 1f32 / max_element;
    for val in v.iter_mut() {
        *val *= inv_max_element;
        magnitude_squared += *val * *val;
    }
    // let g = v[0].signum() * magnitude_squared.sqrt();
    let g = -v[0].signum() * magnitude_squared.sqrt();
    let scale = v[0] + g;
    let inv_scale = 1f32 / scale;
    for val in v[1..].iter_mut() {
        *val *= inv_scale;
    }
    v[0] = -g * max_element;
    scale / g
}

impl AutumnDecomp {
    fn new(mut h: NdArray, workspace: &mut [f32]) -> Self {
        debug_assert!(h.dims[0] <= h.dims[1]);
        let (rows, cols) = (h.dims[0], h.dims[1]);
        debug_assert!(workspace.len() >= rows);
        let mut t = vec![0f32; rows];
        let mut active_range = rows;
        for p in 0..rows {
            active_range -= 1;
            let tau = &mut t[p];
            let offset = p * cols;
            let (projection, target) = h.data.split_at_mut(offset + cols);
            let projection = &mut projection[offset + p..offset + cols];
            *tau = params(projection, rows, p);
            let proj_suffix = &projection[1..];
            let mut split_range = proj_suffix.len();
            for i in 0..active_range {
                let roffset = i * cols;
                let mut wi = target[roffset + p];
                {
                    let mut targ_suffix = &mut target[roffset + p + 1..roffset + cols];
                    targ_suffix = &mut targ_suffix[..split_range];
                    for j in 0..split_range {
                        wi += targ_suffix[j] * proj_suffix[j];
                    }
                    wi *= *tau;
                    for j in 0..split_range {
                        targ_suffix[j] -= wi * proj_suffix[j];
                    }
                }
                target[roffset + p] -= wi;
            }
        }
        Self { h, t, rows, cols }
    }
}

impl AutumnDecomp {
    pub fn left_apply_q(&self, target: &mut NdArray, workspace: &mut [f32]) {
        // Q * A
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        // implied dimension of q ~ cols x cols
        debug_assert_eq!(cols, trows);
        debug_assert!(workspace.len() >= tcols);
        let h = &self.h.data;
        let t = &mut target.data;
        let n = &self.t;
        let mut offset = 0;
        let mut workspace = &mut workspace[0..tcols];
        for p in 0..rows {
            let tau = n[p];
            let h_suffix = &h[offset + p + 1..offset + cols];
            {
                let toffset = p * tcols;
                workspace.copy_from_slice(&t[toffset..toffset + tcols]);
            }
            for i in p + 1..trows {
                let toffset = i * tcols;
                let t_suffix = &t[toffset..toffset + tcols];
                let scalar = h_suffix[i - p - 1];
                for j in 0..tcols {
                    workspace[j] += scalar * t_suffix[j];
                }
            }
            {
                let toffset = p * tcols;
                let t_suffix = &mut t[toffset..toffset + tcols];
                for j in 0..tcols {
                    let temp = tau * workspace[j];
                    workspace[j] = temp;
                    t_suffix[j] -= temp;
                }
            }
            for i in p + 1..trows {
                let toffset = i * tcols;
                let t_suffix = &mut t[toffset..toffset + tcols];
                let scalar = h_suffix[i - p - 1];
                for j in 0..tcols {
                    t_suffix[j] -= scalar * workspace[j];
                }
            }
            offset += cols;
        }
    }
    pub fn left_apply_qt(&self, target: &mut NdArray, workspace: &mut [f32]) {
        // Q * A
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        // implied dimension of q ~ cols x cols
        debug_assert_eq!(cols, trows);
        debug_assert!(workspace.len() >= tcols);
        let h = &self.h.data;
        let t = &mut target.data;
        let n = &self.t;
        let mut workspace = &mut workspace[0..tcols];
        let mut offset = rows * cols;
        let mut toffset = trows * tcols;
        let mut roffset;
        for p in (0..rows).rev() {
            let tau = n[p];
            offset -= cols;
            toffset -= tcols;
            let h_suffix = &h[offset + p + 1..offset + cols];
            {
                workspace.copy_from_slice(&t[toffset..toffset + tcols]);
            }
            roffset = p * tcols;
            for i in p + 1..trows {
                roffset += tcols;
                let t_suffix = &t[roffset..roffset + tcols];
                let scalar = h_suffix[i - p - 1];
                for j in 0..tcols {
                    workspace[j] += scalar * t_suffix[j];
                }
            }
            {
                let t_suffix = &mut t[toffset..toffset + tcols];
                for j in 0..tcols {
                    // scale w by tao for reuse below
                    let temp = workspace[j] * tau;
                    workspace[j] = temp;
                    t_suffix[j] -= temp;
                }
            }
            roffset = p * tcols;
            for i in p + 1..trows {
                roffset += tcols;
                let t_suffix = &mut t[roffset..roffset + tcols];
                let scalar = h_suffix[i - p - 1];
                for j in 0..tcols {
                    t_suffix[j] -= scalar * workspace[j];
                }
            }
        }
    }
    pub fn right_apply_q(&self, target: &mut NdArray) {
        // A * Q
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        // implied dimension of q ~ cols x cols
        debug_assert_eq!(tcols, cols);
        let h = &self.h.data;
        let t = &mut target.data;
        let n = &self.t;
        let mut offset = rows * cols;
        for p in (0..rows).rev() {
            offset -= cols;
            let tau = n[p];
            let h_suffix = &h[offset + p + 1..offset + cols];
            let split_range = h_suffix.len();
            for i in 0..trows {
                let roffset = i * tcols;
                let mut wi = t[roffset + p];
                {
                    let mut targ_suffix = &mut t[roffset + p + 1..roffset + tcols];
                    for j in 0..split_range {
                        wi += h_suffix[j] * targ_suffix[j];
                    }
                    wi *= tau;
                    for j in 0..split_range {
                        targ_suffix[j] -= wi * h_suffix[j];
                    }
                }
                t[roffset + p] -= wi;
            }
        }
    }
    pub fn right_apply_qt(&self, target: &mut NdArray) {
        // A * Q'
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        // implied dimension of q' ~ cols x cols
        debug_assert_eq!(tcols, cols);
        let h = &self.h.data;
        let t = &mut target.data;
        let n = &self.t;
        let mut offset = 0;
        for p in 0..rows {
            let tau = n[p];
            let h_suffix = &h[offset + p + 1..offset + cols];
            let split_range = h_suffix.len();
            for i in 0..trows {
                let roffset = i * tcols;
                let mut wi = t[roffset + p];
                {
                    let mut targ_suffix = &mut t[roffset + p + 1..roffset + tcols];
                    targ_suffix = &mut targ_suffix[..split_range];
                    for j in 0..split_range {
                        wi += h_suffix[j] * targ_suffix[j];
                    }
                    wi *= tau;
                    for j in 0..split_range {
                        targ_suffix[j] -= wi * h_suffix[j];
                    }
                }
                t[roffset + p] -= wi;
            }
            offset += cols;
        }
    }
    pub fn left_apply_l(&self, target: &mut NdArray, workspace: &mut [f32]) {
        debug_assert_eq!(self.h.dims[1], target.dims[1]);
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let h = &self.h.data;
        let t = &mut target.data;
        debug_assert!(workspace.len() >= tcols);
        for p in (0..rows).rev() {
            let offset = p * cols;
            let h_suffix = &h[offset..=offset + p];
            {
                let outer_suffix = &t[..tcols];
                let scalar = h_suffix[0];
                for j in 0..tcols {
                    workspace[j] = scalar * outer_suffix[j];
                }
            }
            for i in (1..=p) {
                let roffset = i * tcols;
                let outer_suffix = &t[roffset..roffset + tcols];
                let scalar = h_suffix[i];
                for j in 0..tcols {
                    workspace[j] += scalar * outer_suffix[j];
                }
            }
            let toffset = p * tcols;
            t[toffset..toffset + tcols].copy_from_slice(&workspace);
        }
    }
    pub fn left_apply_lt(&self, target: &mut NdArray, workspace: &mut [f32]) {
        debug_assert_eq!(target.dims[0], self.h.dims[0]);
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let h = &self.h.data;
        if cols > trows {
            target.resize_rows(cols);
        }
        let t = &mut target.data;
        debug_assert!(workspace.len() >= cols);
        for i in 0..rows {
            let offset = i * cols;
            let toffset = i * tcols;
            workspace.copy_from_slice(&t[toffset..toffset + tcols]);
            let h_suffix = &h[offset..offset + cols];
            for k in (0..i) {
                let scalar = h_suffix[k];
                let woffset = k * tcols;
                let w_suffix = &mut t[woffset..woffset + tcols];
                for j in 0..tcols {
                    w_suffix[j] += scalar * workspace[j];
                }
            }
            {
                let scalar = h_suffix[i];
                let w_suffix = &mut t[toffset..toffset + tcols];
                for j in 0..tcols {
                    w_suffix[j] *= scalar;
                }
            }
        }
        if cols < trows {
            target.resize_rows(cols);
        }
    }
    pub fn right_apply_l(&self, target: &mut NdArray) {
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        debug_assert_eq!(target.dims[1], self.h.dims[0]);
        if cols > tcols {
            target.resize_cols(cols);
        }
        let t = &mut target.data;
        let h = &self.h.data;
        let mut offset = 0;
        let mut roffset;
        for i in 0..trows {
            roffset = 0;
            let t_suffix = &mut t[offset..offset + tcols];
            for k in 0..rows {
                let outer_suffix = &h[roffset..=roffset + k];
                let scalar = t_suffix[k];
                for j in 0..k {
                    t_suffix[j] += scalar * outer_suffix[j];
                }
                t_suffix[k] = scalar * outer_suffix[k];
                roffset += cols;
            }
            offset += tcols;
        }
        if cols < tcols {
            target.resize_cols(cols);
        }
    }
    pub fn right_apply_lt(&self, target: &mut NdArray) {
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        debug_assert_eq!(tcols, rows);
        let mut t = &mut target.data;
        let mut h = &self.h.data;
        let mut dij;
        let mut toffset = 0;
        for i in 0..rows {
            let t_suffix = &mut t[toffset..toffset + tcols];
            let tlen = t_suffix.len();
            for j in (0..trows).rev() {
                let hoffset = j * cols;
                let mut h_suffix = &h[hoffset..hoffset + cols];
                h_suffix = &h_suffix[..tlen];
                dij = t_suffix[0] * h_suffix[0];
                for k in 1..=j {
                    dij += t_suffix[k] * h_suffix[k];
                }
                t_suffix[j] = dij;
            }
            toffset += tcols;
        }
    }
}

fn test_retrieve_l(a: &AutumnDecomp) -> NdArray {
    let mut h = a.h.clone();
    let (rows, cols) = (h.dims[0], h.dims[1]);
    for i in 0..rows {
        for j in i + 1..cols {
            h.data[i * cols + j] = 0f32;
        }
    }
    h
}

fn test_lower_applys() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let b = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let l = test_retrieve_l(&autumn);
    let expected = matrix_mult(&l, &b);
    let mut result = b.clone();
    autumn.left_apply_l(&mut result, &mut workspace);
    assert!(approx_vector_eq(&expected.data, &result.data));

    let expected = matrix_mult(&b, &l);
    let mut result = b.clone();
    autumn.right_apply_l(&mut result);
    assert!(approx_vector_eq(&expected.data, &result.data));

    let lt = l.transpose();
    let expected = matrix_mult(&lt, &b);
    let mut result = b.clone();
    autumn.left_apply_lt(&mut result, &mut workspace);
    assert!(approx_vector_eq(&expected.data, &result.data));

    let expected = matrix_mult(&b, &lt);
    let mut result = b.clone();
    autumn.right_apply_lt(&mut result);
    assert!(approx_vector_eq(&expected.data, &result.data));
}

fn test_autumn_reconstruct() {
    let dims = [1, 2, 3, 8, 16];
    for n in dims {
        test_n_autumn_reconstruct(n);
    }
}
fn test_autumn_orthogonal() {
    let dims = [1, 4];
    for n in dims {
        test_n_autumn_orthogonal_left(n);
        test_n_autumn_orthogonal_right(n);
    }
}
fn test_n_autumn_reconstruct(n: usize) {
    let a = generate_random_matrix(n, n);
    let expected = a.clone();
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    autumn.right_apply_l(&mut i);
    autumn.right_apply_q(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
    let mut i = create_identity_matrix(n);
    autumn.left_apply_q(&mut i, &mut workspace);
    autumn.left_apply_l(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
    let mut i = create_identity_matrix(n);
    autumn.left_apply_l(&mut i, &mut workspace);
    autumn.right_apply_q(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
    let mut i = create_identity_matrix(n);
    autumn.left_apply_lt(&mut i, &mut workspace);
    autumn.left_apply_qt(&mut i, &mut workspace);
    i.transpose_square();
    assert!(approx_vector_eq(&i.data, &expected.data));
    let mut i = create_identity_matrix(n);
    autumn.right_apply_qt(&mut i);
    autumn.right_apply_lt(&mut i);
    i.transpose_square();
    assert!(approx_vector_eq(&i.data, &expected.data));
}
fn test_n_autumn_orthogonal_right(n: usize) {
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.right_apply_q(&mut i);
    autumn.right_apply_qt(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
    autumn.right_apply_qt(&mut i);
    autumn.right_apply_q(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
}
fn test_n_autumn_orthogonal_left(n: usize) {
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.left_apply_qt(&mut i, &mut workspace);
    autumn.left_apply_q(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
    autumn.left_apply_q(&mut i, &mut workspace);
    autumn.left_apply_qt(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
}
fn test_decomp_rectangle() {
    // A * Q
    let (m, n) = (4, 8);
    let a = generate_random_matrix(m, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut workspace = vec![0f32; n];
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.right_apply_q(&mut i);
    autumn.right_apply_qt(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
    autumn.right_apply_qt(&mut i);
    autumn.right_apply_q(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
}
fn test_autumn_q_transpose_consistency() {
    let (m, n) = (4, 4);
    let a = generate_random_matrix(m, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);

    // Test: (Q * I)^T == I^T * Q^T
    let mut left_q = create_identity_matrix(n);
    autumn.left_apply_q(&mut left_q, &mut workspace);
    let left_q_t = left_q.transpose();

    let mut right_qt = create_identity_matrix(n);
    autumn.right_apply_qt(&mut right_qt);

    assert!(
        approx_vector_eq(&left_q_t.data, &right_qt.data),
        "Q left vs QT right failed"
    );
}
fn test_autumn_qt_transpose_consistency() {
    let (m, n) = (4, 4);
    let a = generate_random_matrix(m, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);

    // Test: (Q^T * I)^T == I^T * Q
    let mut left_qt = create_identity_matrix(n);
    autumn.left_apply_qt(&mut left_qt, &mut workspace);
    let left_qt_t = left_qt.transpose();

    let mut right_q = create_identity_matrix(n);
    autumn.right_apply_q(&mut right_q);

    assert!(
        approx_vector_eq(&left_qt_t.data, &right_q.data),
        "QT left vs Q right failed"
    );
}
fn test_autumn_l_transpose_consistency() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);

    // Test: (L * I)^T == I^T * L^T
    // Since Autumn stores L in the lower triangle of H
    let mut left_l = create_identity_matrix(n);
    autumn.left_apply_l(&mut left_l, &mut workspace);
    let left_l_t = left_l.transpose();

    let mut right_lt = create_identity_matrix(n);
    autumn.right_apply_lt(&mut right_lt);

    assert!(
        approx_vector_eq(&left_l_t.data, &right_lt.data),
        "L left vs LT right failed"
    );
}
fn test_autumn_lt_transpose_consistency() {
    let n = 3;
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut left_lt = create_identity_matrix(n);
    autumn.left_apply_lt(&mut left_lt, &mut workspace);
    let left_lt_t = left_lt.transpose();
    let mut right_l = create_identity_matrix(n);
    autumn.right_apply_l(&mut right_l);
    assert!(
        approx_vector_eq(&left_lt_t.data, &right_l.data),
        "LT left vs L right failed"
    );
}

fn main() {
    test_lower_applys();
    test_autumn_reconstruct();
    test_autumn_orthogonal();
    test_decomp_rectangle();
    test_autumn_qt_transpose_consistency();
    test_autumn_lt_transpose_consistency();
    test_autumn_l_transpose_consistency();
    test_autumn_q_transpose_consistency();
}
