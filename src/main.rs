#![allow(unused)]
use std::time::Instant;
use stellar::algebra::bmethods::contractions::{
    tensor_lt_contraction, tensor_rut_contraction, tensor_tlt_contraction, tensor_tut_contraction,
    tensor_ut_contraction,
};
use stellar::algebra::bmethods::interface::{tensor_kernel, tensor_tlt_kernel, tensor_tut_kernel};
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{create_identity_vector, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::wy::{lhs_apply_q, wy_decomposition};
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

/// copies memory from b into a
fn import_slice(target: &mut [f32], data: &[f32]) {
    target[..data.len()].copy_from_slice(data);
    target[data.len()..].fill(0f32);
}

fn test_left_apply_q() {
    let (rows, cols, acols) = (2, 4, 2);
    debug_assert!(cols >= rows);
    let (s_x, s_y, s_t, s_tri) = (cols, cols, acols, rows);
    let mut l_yt = generate_random_vector(rows * cols);
    let mut tri = create_identity_vector(rows, rows);

    let mut w = vec![0f32; cols];
    let mut x_argument = create_identity_vector(cols, acols);
    let x_mat = NdArray {
        dims: vec![rows, acols],
        data: x_argument.clone(),
    };
    println!("argumnet rhs {x_mat:?}");
    let mut o_buffer = x_argument.clone();
    let mut t_buffer = vec![0f32; rows * acols];
    let mut big_buffer = vec![0f32; cols * acols];
    let mut q_argument = x_argument.clone();
    let mut s_buffer = vec![0f32; cols * cols];

    let input = l_yt.clone();
    let input_matrix = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    println!("input_matrix {input_matrix:?}");

    wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, cols);
    lhs_apply_q(&l_yt, &tri, &x_argument, &mut t_buffer, &mut q_argument, rows, cols, acols);
    t_buffer.fill(0f32);
    tensor_lt_contraction(
        &l_yt,
        &q_argument,
        &mut t_buffer,
        1,
        0,
        rows,
        cols,
        acols,
        s_x,
        s_t,
        s_t,
    );
    let result = t_buffer.clone();
    let result_matrix = NdArray {
        dims: vec![rows, acols],
        data: result.clone(),
    };
    let input = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    let arg_matrix = NdArray {
        dims: vec![cols, acols],
        data: x_argument.clone(),
    };
    let expected = matrix_mult(&input, &arg_matrix);
    println!("expected : {expected:?}");
    println!("reconstruct : {result_matrix:?}");
}

fn test_reconstruct() {
    let (rows, cols, stride) = (4, 8, 8);
    // let (rows, cols, stride) = (2, 4, 4);
    debug_assert!(cols >= rows);
    let (s_x, s_y, s_z, s_t, s_tri) = (cols, cols, cols, cols, rows);
    let mut l_yt = generate_random_vector(rows * cols);
    let mut tri = create_identity_vector(rows, rows);

    let mut w = vec![0f32; cols];
    // these are x's ie this will be added at the end
    let mut x_argument = create_identity_vector(cols, cols);
    let mut o_buffer = x_argument.clone();
    let mut t_buffer = vec![0f32; rows * cols];
    let mut big_buffer = vec![0f32; cols * cols];
    import_slice(&mut t_buffer, &o_buffer[..rows * cols]);
    let t_buffer_mat = NdArray {
        dims: vec![rows, cols],
        data: t_buffer.clone(),
    };
    println!("t_buffer {t_buffer_mat:?}");
    let mut s_buffer = vec![0f32; cols * cols];

    let input = l_yt.clone();

    wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, stride);

    let l_yt_matrix = NdArray {
        dims: vec![rows, cols],
        data: l_yt.clone(),
    };
    let tri_matrix = NdArray {
        dims: vec![rows, rows],
        data: tri.clone(),
    };
    println!("l_yt : {l_yt_matrix:?}");
    println!("tri : {tri_matrix:?}");

    // the compact WY representation: `A = L * (I - Y T Y')`.
    // A = LX - YTY'X;

    // y'x
    tensor_ut_contraction(
        &l_yt[1..],
        &o_buffer[s_y..],
        &mut t_buffer,
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        s_x,
        s_y,
        s_t,
    );
    let current = NdArray {
        dims: vec![rows, cols],
        data: t_buffer.clone(),
    };
    println!("check Y' created {current:?}");
    // t * [y'x];
    tensor_lt_contraction(
        &tri,
        &t_buffer,
        &mut s_buffer,
        1,
        0,
        // rows,
        rows,
        cols,
        cols,
        s_tri,
        s_t,
        s_t,
    );
    let current = NdArray {
        dims: vec![rows, cols],
        data: s_buffer.clone(),
    };
    println!("check TY' created {current:?}");
    // VALIDATED AS CORRECT
    //
    //
    //
    import_slice(&mut t_buffer, &s_buffer[..rows * s_t]);
    import_slice(&mut big_buffer, &s_buffer);
    println!("big_buffer {big_buffer:?}");
    // works and validated
    // [y']' * [ty'x ]
    // tensor_tlt_contraction(
    //     &l_yt[1..],
    //     &s_buffer[..],
    //     &mut t_buffer[stride..],
    //     cols - cols.min(rows) + 1,
    //     0,
    //     rows.saturating_sub(1),
    //     cols,
    //     cols,
    //     s_x,
    //     s_t, s_t,
    // );
    tensor_tlt_contraction(
        &l_yt[1..],
        &s_buffer[..],
        &mut big_buffer[stride..],
        // cols - cols.min(rows) + 1,
        rows - rows.min(cols) + 1,
        0,
        cols.saturating_sub(1),
        // cols.saturating_sub(1),
        rows,
        cols,
        s_x,
        s_t,
        s_t,
    );
    // THIS IS WHAT FAILS IE THE RHS TERM
    // let mut q_argument = create_identity_vector(rows, cols);
    // for k in 0..t_buffer.len() {
    //     q_argument[k] -= t_buffer[k];
    // }
    let mut q_argument = create_identity_vector(cols, cols);
    for k in 0..big_buffer.len() {
        q_argument[k] -= big_buffer[k];
    }
    println!("here length {:?}", t_buffer.len());
    let right_term = q_argument.clone();
    let right_term_matrix = NdArray {
        dims: vec![cols, cols],
        data: big_buffer.clone(),
    };
    println!("right_term {right_term_matrix:?}");
    // let mut t = create_identity_vector(cols, cols);
    t_buffer.fill(0f32);
    tensor_lt_contraction(
        &l_yt,
        &q_argument,
        &mut t_buffer,
        1,
        0,
        rows,
        cols,
        cols,
        s_x,
        s_y,
        s_t,
    );
    let result = t_buffer.clone();
    let result_matrix = NdArray {
        dims: vec![rows, cols],
        data: result.clone(),
    };
    let input = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    println!("input : {input:?}");
    println!("reconstruct : {result_matrix:?}");
    // let reference = AutumnDecomp::new(input);
    // println!("reference LQ {:?}", reference.h);
    // println!("reference LQ {:?}", reference.t);
}

fn validate_upper_upper_fma() {
    let (rows, cols, stride) = (4, 7, 7);
    let mut d = generate_random_vector(rows * cols);
    let mut t = generate_random_vector(cols * cols);
    let mut w = vec![0f32; cols];

    let mut o_buffer = generate_random_vector(cols * cols);
    // let mut t_buffer = o_buffer.clone();
    let mut t_buffer = vec![0f32; rows * cols];
    import_slice(&mut t_buffer, &o_buffer[0..rows * cols]);
    let mut t_clean = vec![0f32; cols * cols];
    // for testing
    let mut s_buffer = o_buffer.clone();

    let input = NdArray {
        dims: vec![cols, cols],
        data: o_buffer.clone(),
    };
    println!("input {input:?}");

    tensor_ut_contraction(
        &d[1..],
        &o_buffer[stride..],
        &mut t_clean[..],
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    tensor_ut_contraction(
        &d[1..],
        &o_buffer[stride..],
        &mut t_buffer[..],
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    for k in 0..t_clean.len() {
        t_clean[k] += s_buffer[k];
    }
    println!("t_clean {t_clean:?}");
    println!("t_buffer {t_buffer:?}");

    for i in 0..rows {
        for j in 0..i.min(cols) {
            d[i * stride + j] = 0f32;
        }
        d[i * stride + i] = 1f32;
    }
    let basis_matrix = NdArray {
        dims: vec![rows, cols],
        data: d,
    };
    println!("basis_matrix {basis_matrix:?}");
    let s_vector = NdArray {
        dims: vec![cols, cols],
        data: s_buffer,
    };
    let reconst = NdArray {
        dims: vec![rows, cols],
        data: t_buffer,
    };
    let reference = matrix_mult(&basis_matrix, &s_vector);
    println!("reconst {reconst:?}");
    println!("reference {reference:?}");
}

fn validate_transpose_upper_upper_fma() {
    // in terms of output space
    let (rows, shared, cols) = (4, 2, 4);
    let (s_x, s_y, s_t) = (rows, cols, cols);
    let mut d = generate_random_vector(shared * rows);
    let d_matrix = NdArray {
        dims: vec![shared, rows],
        data: d.clone(),
    };
    println!("raw x_matrix {d_matrix:?}");
    let mut o_buffer = generate_random_vector(shared * cols);
    let mut t_buffer = vec![0f32; rows * cols];
    import_slice(&mut t_buffer, &o_buffer[..shared * cols]);
    let mut t_clean = vec![0f32; rows * cols];
    // for testing
    let mut s_buffer = o_buffer.clone();

    let input = NdArray {
        dims: vec![shared, cols],
        data: o_buffer.clone(),
    };
    println!("input {input:?}");
    tensor_tlt_contraction(
        &d[1..],
        &o_buffer[..],
        &mut t_buffer[s_t..],
        cols - cols.min(rows) + 1,
        0,
        rows.saturating_sub(1),
        shared,
        cols,
        s_x,
        s_y,
        s_t,
    );
    println!("t_buffer {t_buffer:?}");
    let mut basis_matrix = NdArray {
        dims: vec![shared, rows],
        data: d,
    };
    basis_matrix = basis_matrix.transpose();
    filter_lower_trapezoid(&mut basis_matrix);
    set_diagonal_value(&mut basis_matrix, 1f32);
    println!("basis_matrix {basis_matrix:?}");
    println!("----------------------");
    let s_vector = NdArray {
        dims: vec![shared, cols],
        data: s_buffer,
    };
    println!("s_vector {s_vector:?}");
    let reconst = NdArray {
        dims: vec![rows, cols],
        data: t_buffer,
    };
    println!("reconst {reconst:?}");
    let reference = matrix_mult(&basis_matrix, &s_vector);
    // println!("t_clean_mat {t_clean_mat:?}");
    println!("----------------------");
    println!("reconst {reconst:?}");
    println!("reference {reference:?}");
}

pub fn filter_lower_trapezoid(a: &mut NdArray) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let t = cols.min(rows);
    let s = rows.saturating_sub(cols);
    // don't remove from last row
    for i in 1..t {
        for j in 0..i {
            d[(rows - i - s) * cols - j - 1] = 0f32;
        }
    }
}
pub fn set_diagonal_value(a: &mut NdArray, c: f32) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let mn = cols.min(rows);
    let dx = cols.saturating_sub(rows);
    for k in 0..mn {
        d[k * cols + dx + k] = c;
    }
}

fn debug_set_diagonal(rows: usize, cols: usize) {
    let mut d = generate_random_vector(rows * cols);
    let mut matrix = NdArray {
        dims: vec![rows, cols],
        data: d,
    };
    filter_lower_trapezoid(&mut matrix);
    set_diagonal_value(&mut matrix, 1000f32);
    println!("set diagonal {matrix:?}");
}

fn test_debug_set_diagonal() {
    let vals = vec![(1, 2), (2, 1), (4, 4), (3, 6), (6, 3)];
    for (r, c) in vals {
        debug_set_diagonal(r, c);
        println!("----------------------");
    }
}

fn main() {
    test_left_apply_q();
    // test_reconstruct();
    // validate_upper_upper_fma();
    // validate_transpose_upper_upper_fma();
}
