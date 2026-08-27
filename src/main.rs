#![allow(unused)]
use stellar::algebra::bmethods::contractions::{
    tensor_lt_contraction, tensor_rut_contraction, tensor_tut_contraction, tensor_ut_contraction,
};
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{create_identity_vector, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::wy::wy_decomposition;
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

/// Dumb dense Q = I - Y*T*Y', built with plain matmul, no kernels/contractions.
/// l_yt: packed [L \ Y'] storage after wy_decomposition, cols x cols.
/// t_data: T after wy_decomposition, cols x cols.
fn dumb_dense_q(l_yt: &[f32], t_data: &[f32], rows: usize, cols: usize) -> NdArray {
    // Y' is rows x cols: implicit unit diagonal, strictly-right entries from l_yt
    let mut y_prime_data = vec![0f32; rows * cols];
    for i in 0..rows {
        y_prime_data[i * cols + i] = 1.0;
        for j in (i + 1)..cols {
            y_prime_data[i * cols + j] = l_yt[i * cols + j];
        }
    }
    let y_prime = NdArray {
        dims: vec![rows, cols],
        data: y_prime_data,
    };
    let t_mat = NdArray {
        dims: vec![rows, rows],
        data: t_data.to_vec(),
    };
    let y_mat = y_prime.transpose(); // cols x rows

    let yt = matrix_mult(&y_mat, &t_mat); // cols x rows
    let ytyt = matrix_mult(&yt, &y_prime); // cols x cols

    let mut q_data = create_identity_vector(cols, cols);
    for idx in 0..q_data.len() {
        q_data[idx] -= ytyt.data[idx];
    }
    NdArray {
        dims: vec![cols, cols],
        data: q_data,
    }
}

fn check_q(input_data: &[f32], l_yt: &[f32], t_data: &[f32], rows: usize, cols: usize) {
    let dense_q = dumb_dense_q(l_yt, t_data, rows, cols);

    let input_matrix = NdArray {
        dims: vec![rows, cols],
        data: input_data.to_vec(),
    };
    let autumn_ref = AutumnDecomp::new(input_matrix);
    let mut workspace = vec![f32::NAN; cols];
    let mut autumn_q = create_identity_matrix(cols);
    autumn_ref.mat_left_apply_q(&mut autumn_q, &mut workspace);

    println!("dumb dense Q : {dense_q:?}");
    println!("autumn Q     : {autumn_q:?}");
}

fn test_reconstruct() {
    let (rows, cols, stride) = (2, 2, 2);
    let mut l_yt = generate_random_vector(rows * cols);
    l_yt[2] = 0f32;
    let mut t = generate_random_vector(cols * cols);
    let mut w = vec![0f32; cols];

    let mut o_buffer = create_identity_vector(cols, cols);
    let mut t_buffer = o_buffer.clone();
    let mut s_buffer = vec![0f32; cols * cols];

    let input = l_yt.clone();

    wy_decomposition(&mut l_yt, &mut t, &mut w, rows, cols, stride);

    let l_yt_matrix = NdArray {
        dims: vec![rows, cols],
        data: l_yt.clone(),
    };
    let tri_matrix = NdArray {
        dims: vec![rows, cols],
        data: t.clone(),
    };
    println!("l_yt : {l_yt_matrix:?}");
    println!("tri : {tri_matrix:?}");

    // the compact WY representation: `A = L * (I - Y T Y')`.
    // XA = XL * (I - YTY');
    tensor_ut_contraction(
        &l_yt[1..],
        &o_buffer[stride..],
        &mut t_buffer,
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    println!("current {t_buffer:?}");
    tensor_lt_contraction(
        &t,
        &t_buffer,
        &mut s_buffer,
        1,
        0,
        rows,
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    for idx in 0..t_buffer.len() {
        t_buffer[idx] = s_buffer[idx];
    }
    println!("current {t_buffer:?}");
    tensor_tut_contraction(
        &l_yt[1..],
        &s_buffer[stride..],
        &mut t_buffer,
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    for idx in 0..o_buffer.len() {
        o_buffer[idx] -= t_buffer[idx];
    }
    println!("current {t_buffer:?}");
    t_buffer.fill(0f32);
    tensor_lt_contraction(
        &l_yt,
        &o_buffer,
        &mut t_buffer,
        1,
        0,
        rows,
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    println!("current {t_buffer:?}");

    let input = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    let reconstruct = NdArray {
        dims: vec![rows, cols],
        data: t_buffer.clone(),
    };
    println!("input : {input:?}");
    println!("reconstruct : {reconstruct:?}");
    let reference = AutumnDecomp::new(input);
    println!("reference LQ {:?}", reference.h);
    println!("reference LQ {:?}", reference.t);
}

fn main() {
    let (rows, cols, stride) = (4, 8, 8);
    let mut l_yt = generate_random_vector(rows * cols);
    // l_yt[2] = 0f32;
    let mut t = generate_random_vector(rows * rows);
    let mut w = vec![0f32; cols];

    let mut o_buffer = create_identity_vector(cols, cols);
    let mut t_buffer = o_buffer.clone();
    let mut s_buffer = vec![0f32; cols * cols];

    let input = l_yt.clone();

    wy_decomposition(&mut l_yt, &mut t, &mut w, rows, cols, stride);
    check_q(&input, &l_yt, &t, rows, cols);
}
