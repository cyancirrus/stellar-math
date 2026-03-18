#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_matrix;
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
        Self {
            h,
            t,
            rows,
            cols,
        }
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
                    t_suffix[j] -= tau * workspace[j];
                }
            }
            for i in p + 1..trows {
                let toffset = i * tcols;
                let t_suffix = &mut t[toffset..toffset + tcols];
                let scalar = tau * h_suffix[i - p - 1];
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
        for p in (0..rows).rev() {
            let tau = n[p];
            offset -= cols;
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
                    t_suffix[j] -= tau * workspace[j];
                }
            }
            for i in p + 1..trows {
                let toffset = i * tcols;
                let t_suffix = &mut t[toffset..toffset + tcols];
                let scalar = tau * h_suffix[i - p - 1];
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
    pub fn right_apply_qt(&self, target: &mut NdArray, workspace: &mut [f32]) {
        // A * Q'
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        // implied dimension of q' ~ cols x cols
        debug_assert_eq!(tcols, cols);
        debug_assert!(workspace.len() >= trows);
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
    pub fn right_apply_l(&self, target: &mut NdArray, workspace: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        debug_assert_eq!(target.dims[1], self.h.dims[0]);
        debug_assert!(workspace.len() >= cols);
        let h = &self.h.data;
        if cols > tcols {
            target.resize_cols(cols);
        }
        let t = &mut target.data;
        for i in 0..rows {
            let offset = i * tcols;
            let t_suffix = &mut t[offset..offset + tcols];
            workspace.fill(0f32);
            for k in (0..tcols) {
                let roffset = k * cols;
                let outer_suffix = &h[roffset..=roffset + i];
                let scalar = t_suffix[k];
                for j in 0..=i {
                    workspace[j] += scalar * outer_suffix[j];
                }
            }
            t_suffix.copy_from_slice(&workspace);
        }
        if cols < tcols {
            target.resize_cols(cols);
        }
    }
    pub fn left_apply_lt(&self, target: &mut NdArray) {
        debug_assert_eq!(target.dims[0], self.h.dims[0]);
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        let h = &self.h.data;
        if cols > trows {
            target.resize_rows(cols);
        }
        let t = &mut target.data;
        let mut outer_row = vec![0f32; cols];
        for i in 0..rows {
            let offset = i * cols;
            let toffset = i * tcols;
            outer_row.copy_from_slice(&t[toffset..toffset + tcols]);
            let h_suffix = &h[offset..offset + cols];
            for k in (0..=i) {
                let scalar = h_suffix[k];
                let woffset = k * tcols;
                let w_suffix = &mut t[woffset..woffset + tcols];
                if k != i {
                    for j in 0..tcols {
                        w_suffix[j] += scalar * outer_row[j];
                    }
                } else {
                    for j in 0..tcols {
                        w_suffix[j] *= scalar;
                    }
                }
            }
        }
        if cols < trows {
            target.resize_rows(cols);
        }
    }
    // TODO: Test this fn
    pub fn right_apply_lt(&self, target: &mut NdArray, workspace: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        debug_assert_eq!(tcols, rows);
        debug_assert!(workspace.len() >= rows);
        let mut t = &mut target.data;
        let mut h = &self.h.data;
        let mut dij;
        let workspace = &mut workspace[0..rows];
        for i in (0..rows) {
            let toffset = i * tcols;
            let t_suffix = &mut t[toffset..toffset + tcols];
            let tlen = t_suffix.len();
            for j in 0..trows {
                dij = 0f32;
                let hoffset = j * cols;
                let mut h_suffix = &h[hoffset..hoffset + cols];
                h_suffix = &h_suffix[..tlen];
                for k in 0..=j {
                    dij += t_suffix[k] * h_suffix[k];
                }
                workspace[j] = dij;
            }
            t_suffix.copy_from_slice(workspace);
        }
    }
}

fn test_autumn_reconstruct() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let expected = a.clone();
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    autumn.right_apply_l(&mut i, &mut workspace);
    autumn.right_apply_q(&mut i);
    assert!(approx_vector_eq(&i.data, &expected.data));
}

fn test_autumn_orthogonal_qqt() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.right_apply_q(&mut i);
    autumn.right_apply_qt(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
}

fn test_autumn_orthogonal_qtq() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.right_apply_qt(&mut i, &mut workspace);
    autumn.right_apply_q(&mut i);
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
    autumn.right_apply_qt(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
}

fn test_autumn_orthogonal_left_qtq() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.left_apply_qt(&mut i, &mut workspace);
    autumn.left_apply_q(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
}

fn test_autumn_orthogonal_left_qqt() {
    let n = 4;
    let a = generate_random_matrix(n, n);
    let mut workspace = vec![0f32; n];
    let autumn = AutumnDecomp::new(a.clone(), &mut workspace);
    let mut i = create_identity_matrix(n);
    let expected = i.clone();
    autumn.left_apply_q(&mut i, &mut workspace);
    autumn.left_apply_qt(&mut i, &mut workspace);
    assert!(approx_vector_eq(&i.data, &expected.data));
}

fn main() {
    test_autumn_reconstruct();
    test_autumn_orthogonal_qqt();
    test_autumn_orthogonal_qtq();
    test_decomp_rectangle();
    test_autumn_orthogonal_left_qtq();
    test_autumn_orthogonal_left_qqt();
    // // let mut a = NdArray::new(vec![2, 2], vec![4f32,1f32,2f32,3f32]);
    // let n = 3;
    // let a = generate_random_matrix(n, n);
    // let mut workspace = vec![0f32;n];
    // let autumn = AutumnDecomp::new(a.clone());
    // println!("lq_decomp {:?}", autumn.h);
    // let mut i = create_identity_matrix(n);
    // autumn.left_apply_l(&mut i);
    // println!("left apply l {i:?}");
    // let mut i = create_identity_matrix(n);
    // autumn.left_apply_lt(&mut i);
    // println!("left apply lt {i:?}");

    // let mut a = NdArray::new(vec![2, 2], vec![1f32, 0f32, 0f32, 1f32]);
    // let mut workspace = vec![0f32;2];
    // let autumn = AutumnDecomp::new(a.clone());
    // println!("lq_decomp {:?}", autumn.h);
    // println!("lq_tau {:?}", autumn.t);
    // let mut i = create_identity_matrix(2);
    // println!("input {a:?}");
    // autumn.right_apply_l(&mut i);
    // println!("midstep {i:?}");
    // autumn.right_apply_q(&mut i);
    // println!("output {i:?}");

    // let mut a = NdArray::new(vec![1, 1], vec![4f32]);
    // let mut workspace = vec![0f32;1];
    // let autumn = AutumnDecomp::new(a.clone());
    // println!("lq_decomp {:?}", autumn.h);
    // println!("lq_tau {:?}", autumn.t);
    // let mut i = create_identity_matrix(1);
    // println!("input {a:?}");
    // autumn.right_apply_l(&mut i);
    // println!("midstep {i:?}");
    // autumn.right_apply_q(&mut i);
    // println!("output {i:?}");

    // let mut a = NdArray::new(vec![2, 2], vec![4f32,1f32,2f32,3f32]);
    // let mut workspace = vec![0f32;2];
    // let autumn = AutumnDecomp::new(a.clone());
    // println!("lq_decomp {:?}", autumn.h);
    // let mut i = create_identity_matrix(2);
    // println!("input {a:?}");
    // autumn.right_apply_l(&mut i);
    // autumn.right_apply_q(&mut i);
    // println!("output {i:?}");
}
