// ============================================================================
// LQ Block Decomposition — Compact WY Representation
// ============================================================================

// **`LqBlockDecomp`** — 0-alloc, in-place LQ factorization via blocked
// Householder reflections (compact WY form).
//
// - **Mutation contract:** Takes in a basis and mutates the original matrix
//   in place. On return, the lower-triangular part of the matrix holds `L`.
//
// - **Naming convention:**
//   - `Lower` ~ `L`
//   - `Householder` ~ `Y'`
//   - `Triangular` ~ `T`
//
// - **Storage layout:**
//   - `h`: packed storage `~ [ L \ Y' ]` — `L` and the Householder vectors
//     `Y'` share the same buffer, split across the diagonal.
//   - `t`: `T`, the block triangular factor for the compact WY update.
//
// - **Factorization form:** `A = LQ`, where `Q` is expressed implicitly via
//   the compact WY representation: `A = L * (I - Y T Y')`.
//
// *(Struct form below is kept only as a thin, allocating wrapper around the
// 0-alloc routine, for callers who want ownership instead of borrowing.)*
// pub struct LqBlockDecomp {
//     pub h: NdArray,
//     pub t: NdArray,
// }
const EPSILON: f32 = 1e-21;
/// params
///
/// takes in a slice, where we find the rotation vector
/// in order when multiplied by the original matrix returns
/// a zero'd matrix
fn params(v: &mut [f32]) -> f32 {
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element.abs() < EPSILON {
        return max_element;
    }
    let mut magnitude_squared = 0f32;
    let inv_max_element = 1f32 / max_element;
    for val in v.iter_mut() {
        *val *= inv_max_element;
        magnitude_squared += *val * *val;
    }
    // let g = -v[0].signum() * magnitude_squared.sqrt();
    let g = v[0].signum() * magnitude_squared.sqrt();
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
/// builds the kth row of the triangle matrix
///
/// * h: householder data stored in upper right matrix
/// * r: householder rotation for iteration k
/// * t: lower block traingular matrix growing row by row
/// * h_dim: col x col in original matrix space
/// * t_dim: row x row in original matrix space
/// * tau: scalar of similarity of the household reflection
/// * k: iteration index
fn triangle_iteration(
    h: &mut [f32],
    t: &mut [f32],
    r: &[f32],
    w: &mut [f32],
    h_dim: usize,
    t_dim: usize,
    k: usize,
    tau: f32,
) {
    // T[k] = ((T, 0), (-tau[k]* h[k]' Y[k-1]T[k-1], tau));
    // diagonal element stores the L[ii] element not householder
    let mut hoffset = 0;
    // h'Y :: Y
    let koffset = k * t_dim;
    let h_k_tail = r;
    for l in 0..k {
        // initial element of householder vector is 1
        let mut dot = h[hoffset + k];
        let h_i_tail = &h[hoffset + k + 1..hoffset + h_dim];
        for j in 0..h_k_tail.len() {
            dot += h_i_tail[j] * h_k_tail[j];
        }
        w[l] = dot;
        hoffset += h_dim;
    }
    let mut toffset = 0;
    let (t_upper, t_target) = t.split_at_mut(koffset);

    // h'T :: T ~ bottom-left triangular
    for l in 0..k {
        // outer product iteration style
        let outer = -w[l] * tau;
        let t_tail = &t_upper[toffset..=toffset + l];
        for j in 0..=l {
            t_target[j] += outer * t_tail[j];
        }
        toffset += t_dim;
    }
    t[koffset + k] = tau;
}
pub fn wy_decomposition(
    l_yt: &mut [f32],
    t: &mut [f32],
    w: &mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
) {
    // pub fn new(mut l_yt: NdArray, mut t_mat: NdArray, w: &mut [f32]) -> Self {
    // let (rows, cols) = (l_yt.dims[0], l_yt.dims[1]);
    debug_assert!(rows <= cols);
    debug_assert!(rows <= w.len());
    // let t = &mut t_mat.data;
    t.fill(0f32);
    let mut active_range = rows;
    let mut offset = 0;
    for k in 0..rows {
        active_range -= 1;
        let (done_rows, todo_rows) = l_yt.split_at_mut(offset);
        let (curr_row, trail_rows) = todo_rows.split_at_mut(cols);

        let v_active = &mut curr_row[k..];
        let tau = params(v_active);
        // implicit 1f32 on the diagonal -> increment index and handle explicitly
        let v_tail = &v_active[1..];
        triangle_iteration(done_rows, t, v_tail, w, cols, rows, k, tau);

        let split_range = v_tail.len();
        let mut roffset = 0;
        for _ in 0..active_range {
            let mut wi = trail_rows[roffset + k];
            {
                let mut targ_suffix = &mut trail_rows[roffset + k + 1..roffset + cols];
                targ_suffix = &mut targ_suffix[..split_range];
                for j in 0..split_range {
                    wi += targ_suffix[j] * v_tail[j];
                }
                wi *= tau;
                for j in 0..split_range {
                    targ_suffix[j] -= wi * v_tail[j];
                }
            }
            trail_rows[roffset + k] -= wi;
            roffset += stride;
        }
        offset += stride;
    }
}

// #![allow(unused)]
// use stellar::algebra::bmethods::contractions::{
//     tensor_lt_contraction, tensor_rut_contraction, tensor_tut_contraction, tensor_ut_contraction,
// };
// use stellar::algebra::ndmethods::create_identity_matrix;
// use stellar::algebra::ndmethods::{create_identity_vector, matrix_mult};
// use stellar::decomposition::lq::AutumnDecomp;
// use stellar::decomposition::wy::wy_decomposition;
// use stellar::random::generation::generate_random_vector;
// use stellar::structure::ndarray::NdArray;

// /// Dumb dense Q = I - Y*T*Y', built with plain matmul, no kernels/contractions.
// /// l_yt: packed [L \ Y'] storage after wy_decomposition, cols x cols.
// /// t_data: T after wy_decomposition, cols x cols.
// fn dumb_dense_q(l_yt: &[f32], t_data: &[f32], rows: usize, cols: usize) -> NdArray {
//     // Y' is rows x cols: implicit unit diagonal, strictly-right entries from l_yt
//     let mut y_prime_data = vec![0f32; rows * cols];
//     for i in 0..rows {
//         y_prime_data[i * cols + i] = 1.0;
//         for j in (i + 1)..cols {
//             y_prime_data[i * cols + j] = l_yt[i * cols + j];
//         }
//     }
//     let y_prime = NdArray {
//         dims: vec![rows, cols],
//         data: y_prime_data,
//     };
//     let t_mat = NdArray {
//         dims: vec![rows, rows],
//         data: t_data.to_vec(),
//     };
//     let y_mat = y_prime.transpose(); // cols x rows

//     let yt = matrix_mult(&y_mat, &t_mat); // cols x rows
//     let ytyt = matrix_mult(&yt, &y_prime); // cols x cols

//     let mut q_data = create_identity_vector(cols, cols);
//     for idx in 0..q_data.len() {
//         q_data[idx] -= ytyt.data[idx];
//     }
//     NdArray {
//         dims: vec![cols, cols],
//         data: q_data,
//     }
// }

// fn check_q(input_data: &[f32], l_yt: &[f32], t_data: &[f32], rows: usize, cols: usize) {
//     let dense_q = dumb_dense_q(l_yt, t_data, rows, cols);

//     let input_matrix = NdArray {
//         dims: vec![rows, cols],
//         data: input_data.to_vec(),
//     };
//     let autumn_ref = AutumnDecomp::new(input_matrix);
//     let mut workspace = vec![f32::NAN; cols];
//     let mut autumn_q = create_identity_matrix(cols);
//     autumn_ref.mat_left_apply_q(&mut autumn_q, &mut workspace);

//     println!("dumb dense Q : {dense_q:?}");
//     println!("autumn Q     : {autumn_q:?}");
// }

// fn test_thing() {
//     let (rows, cols, stride) = (4, 8, 8);
//     let mut l_yt = generate_random_vector(rows * cols);
//     // l_yt[2] = 0f32;
//     let mut t = generate_random_vector(rows * rows);
//     let mut w = vec![0f32; cols];

//     let mut o_buffer = create_identity_vector(cols, cols);
//     let mut t_buffer = o_buffer.clone();
//     let mut s_buffer = vec![0f32; cols * cols];

//     let input = l_yt.clone();

//     wy_decomposition(&mut l_yt, &mut t, &mut w, rows, cols, stride);
//     check_q(&input, &l_yt, &t, rows, cols);
// }
