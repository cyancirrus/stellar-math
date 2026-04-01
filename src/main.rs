#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;


/// LqBlockDecomp
///
/// takes in a basis and mutates the original matrix
/// to contain L on the lower triangle part and
/// Lower ~ L 
/// Householder ~ Y' 
/// Triangular ~ T;
/// * h: ~ [ L \ Y' ] 
/// * t:  T
/// 
/// Matrix decomp has form
/// A = LQ;
/// A = L * (I - YTY'); 
pub struct LqBlockDecomp {
    pub h: NdArray,
    pub t: NdArray,
}

// params
//
// takes in a slice, where we find the rotation vector
// in order when multiplied by the original matrix returns
// a zero'd matrix
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

/// triangle iteration
///
/// triangle iteration for WY decomposition of Q
/// LQ implementation which makes T ~ lower triangle
///
/// * h: householder data stored in upper right matrix
/// * t: lower block traingular matrix growing row by row
/// * h_dim: col x col in original matrix space
/// * t_dim: row x row in original matrix space
/// * tau: scalar of similarity of the household reflection
/// * k: iteration index
fn triangle_iteration(
    h: &mut [f32],
    t: &mut [f32],
    workspace: &mut [f32],
    h_dim: usize,
    t_dim: usize,
    k: usize,
    tau: f32,
) {
    // T[k] = ((T, 0), (-tau[k]* h[k]' Y[k-1]T[k-1], tau));
    // diagonal element stores the L[ii] element not householder
    debug_assert!(k < workspace.len());
    let mut hoffset = 0;
    // h'Y :: Y
    let koffset = k * h_dim;
    let h_k_tail = &h[koffset + k + 1..koffset + h_dim];
    for l in 0..k {
        // initial element of householder vector is 1
        let mut dot = h[hoffset + k];
        let h_i_tail = &h[hoffset + k + 1..hoffset + h_dim];
        for j in 0..h_k_tail.len() {
            dot += h_i_tail[j] * h_k_tail[j];
        }
        workspace[l] = dot;
        hoffset += h_dim;
    }
    let mut toffset = 0;
    let (t_upper, t_target) = t.split_at_mut(koffset);

    // h'T :: T ~ bottom-left triangular
    for l in 0..k {
        // outer product iteration style
        let outer = -workspace[l] * tau;
        let t_tail = &t_upper[toffset..=toffset + l];
        for j in 0..=l {
            t_target[j] += outer * t_tail[j];
        }
        toffset += h_dim;
    }
    t[koffset + k] = tau;
}
impl LqBlockDecomp {
    pub fn new(mut basis: NdArray, mut triangle: NdArray, workspace: &mut [f32]) -> Self {
        let (rows, cols) = (basis.dims[0], basis.dims[1]);
        debug_assert!(rows <= cols);
        let h = &mut basis.data;
        let t = &mut triangle.data;
        t.fill(0f32);
        let mut active_range = rows;
        for k in 0..rows {
            active_range -= 1;
            let offset = k * cols;
            let (projection, target) = basis.data.split_at_mut(offset + cols);
            let projection = &mut projection[offset + k..offset + cols];

            let tau = params(projection);
            triangle_iteration(projection, t, workspace, cols, rows, k, tau);

            let proj_suffix = &projection[1..];
            let split_range = proj_suffix.len();
            let mut roffset = 0;
            for i in 0..active_range {
                let mut wi = target[roffset + k];
                {
                    let mut targ_suffix = &mut target[roffset + k + 1..roffset + cols];
                    targ_suffix = &mut targ_suffix[..split_range];
                    for j in 0..split_range {
                        wi += targ_suffix[j] * proj_suffix[j];
                    }
                    wi *= tau;
                    for j in 0..split_range {
                        targ_suffix[j] -= wi * proj_suffix[j];
                    }
                }
                target[roffset + k] -= wi;
                roffset += cols;
            }
        }
        Self {
            h: basis,
            t: triangle,
        }
    }
}
fn main() {}
