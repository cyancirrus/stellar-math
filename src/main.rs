#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

pub struct LqBlockDecomp {
    pub h: NdArray,
    pub t: NdArray,
}

fn params(v: &mut [f32]) -> f32 {
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

impl LqBlockDecomp {
    pub fn new(mut h: NdArray) -> Self {
        debug_assert!(h.dims[0] <= h.dims[1]);
        let (rows, cols) = (h.dims[0], h.dims[1]);
        let mut tri = NdArray {
            data: vec![0f32; rows * rows],
            dims: vec![rows, rows],
        };
        let t = &mut tri.data;
        let mut active_range = rows;
        for p in 0..rows {
            active_range -= 1;
            let tau = &mut t[p];
            let offset = p * cols;
            let (projection, target) = h.data.split_at_mut(offset + cols);
            let projection = &mut projection[offset + p..offset + cols];
            *tau = params(projection);
            let proj_suffix = &projection[1..];
            let split_range = proj_suffix.len();
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
        Self { h, t: tri }
    }
    fn t_iteration(&self, tri: &mut NdArray, workspace: &mut [f32], tau: f32, k: usize) {
        // T[k] = ((T, 0), (-tau[k]* h[k]' Y[k-1]T[k-1], tau));
        let (rows, cols) = (self.h.dims[0], self.h.dims[1]);
        let h = &self.h.data;
        let t = &mut tri.data;
        // diagonal element stores the L[ii] element not householder
        let w_k = &h[k * cols + k + 1..k * cols + cols];
        debug_assert!(k < workspace.len());
        let mut hoffset = 0;
        // h'Y :: Y
        let koffset = k * cols;
        let h_k_tail = &h[koffset + k + 1..koffset + cols];
        for l in 0..k {
            // initial element of householder vector is 1
            let mut dot = h[hoffset + k];
            let h_i_tail = &h[hoffset + k + 1..hoffset + cols];
            for j in 0..h_k_tail.len() {
                dot += h_i_tail[j] * h_k_tail[j];
            }
            workspace[l] = dot;
            hoffset += cols;
        }
        let mut toffset = 0;
        let (t_upper, t_target) = t.split_at_mut(koffset);

        // h'T :: T ~ bottom-left triangular
        for l in 0..k {
            // outer product iteration style
            let outer = - workspace[l] * tau;
            let t_tail = &t_upper[toffset..=toffset + l];
            for j in 0..=l { 
                t_target[j] += outer * t_tail[j];
            }
            toffset += cols;
        }
        t[koffset + k] = tau;
    }
}

// 0 0 0 0


fn main() {}
