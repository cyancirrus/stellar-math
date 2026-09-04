#![allow(unused)]
use stellar::algebra::bmethods::contractions::{
    tensor_lt_contraction, tensor_rut_contraction, tensor_tlt_contraction, tensor_tut_contraction,
    tensor_ut_contraction,
};
use stellar::algebra::bmethods::interface::{tensor_tlt_kernel, tensor_tut_kernel};
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{create_identity_vector, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::wy::wy_decomposition;
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

/// copies memory from b into a
fn import_slice(target: &mut [f32], data:&[f32]) {
    target[..data.len()].copy_from_slice(data);
    target[data.len()..].fill(0f32);
}

fn test_reconstruct() {
    let (rows, cols, stride) = (2, 2, 2);
    debug_assert!(rows >= cols);
    let s_x = rows;
    let s_y = stride;
    let s_z = stride;
    let mut l_yt = generate_random_vector(rows * cols);
    // let mut t = generate_random_vector(cols * cols);
    let mut tri = create_identity_vector(cols, cols);

    let mut w = vec![0f32; cols];
    // these are x''s
    let mut x_argument = create_identity_vector(cols, cols);
    let mut o_buffer = x_argument.clone();
    let mut t_buffer = vec![0f32; rows * cols];
    import_slice(&mut t_buffer, &o_buffer);
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
        rows,
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    let current = NdArray {
        dims: vec![rows, cols],
        data: s_buffer.clone(),
    };
    println!("check TY' created {current:?}");
    // VALIDATED AS CORRECT
    import_slice(&mut t_buffer, &s_buffer);
    // works and validated
    // [y']' * [ty'x ]
    tensor_tlt_contraction(
        &l_yt[1..],
        &s_buffer[..],
        &mut t_buffer[stride..],
        cols - cols.min(rows) + 1,
        0,
        rows.saturating_sub(1),
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    // THIS IS WHAT FAILS IE THE RHS TERM
    
    let right_term = t_buffer.clone();
    let right_term_matrix = NdArray {
        dims: vec![rows, cols],
        data: right_term.clone(),
    };
    println!("right_term {right_term_matrix:?}");
    // let mut t = create_identity_vector(cols, cols);
    t_buffer.fill(0f32);
    tensor_lt_contraction(
        &l_yt,
        &x_argument,
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
    let left_term = t_buffer.clone();
    let left_term_matrix = NdArray {
        dims: vec![rows, cols],
        data: left_term.clone(),
    };
    println!("left_term {left_term_matrix:?}");
    let mut result = vec![0f32; rows * cols];
    for k in 0.. rows * cols {
        result[k] = left_term[k] - right_term[k];
    }

    let input = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    let reconstruct = NdArray {
        dims: vec![rows, cols],
        data: result.clone(),
    };
    println!("input : {input:?}");
    println!("reconstruct : {reconstruct:?}");
    let reference = AutumnDecomp::new(input);
    // println!("reference LQ {:?}", reference.h);
    // println!("reference LQ {:?}", reference.t);
}

fn validate_upper_upper_fma() {
    let (rows, cols, stride) = (4, 7, 7);
    let mut d = generate_random_vector(rows * cols);
    let mut t = generate_random_vector(cols * cols);
    let mut w = vec![0f32; cols];

    let mut o_buffer = generate_random_vector(cols * cols);
    let mut t_buffer = o_buffer.clone();
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
        data: t_clean,
    };
    let reference = matrix_mult(&basis_matrix, &s_vector);
    println!("reconst {reconst:?}");
    println!("reference {reference:?}");
}

fn validate_transpose_upper_upper_fma() {
    // let (rows, cols, stride) = (4, 7, 7);
    let (rows, cols, stride) = (6, 3, 3);
    let mut d = generate_random_vector(cols * rows);
    let d_matrix = NdArray {
        dims: vec![cols, rows],
        data: d.clone(),
    };
    println!("raw x_matrix {d_matrix:?}");
    let mut w = vec![0f32; cols];

    // let mut o_buffer = vec![1f32; cols * cols];
    let mut o_buffer = generate_random_vector(cols * cols);
    let mut t_buffer = vec![0f32; rows * cols];
    import_slice(&mut t_buffer, &o_buffer);
    let mut t_clean = vec![0f32; rows * cols];
    // for testing
    let mut s_buffer = o_buffer.clone();

    let input = NdArray {
        dims: vec![cols, cols],
        data: o_buffer.clone(),
    };
    println!("input {input:?}");
    tensor_tlt_contraction(
        &d[1..],
        &o_buffer[..],
        &mut t_clean[stride..],
        cols - cols.min(rows) + 1,
        0,
        rows.saturating_sub(1),
        cols,
        cols,
        rows,
        stride,
        stride,
    );
    tensor_tlt_contraction(
        &d[1..],
        &o_buffer[..],
        &mut t_buffer[stride..],
        cols - cols.min(rows) + 1,
        0,
        rows.saturating_sub(1),
        cols,
        cols,
        rows,
        stride,
        stride,
    );
    for k in 0..s_buffer.len() {
        t_clean[k] += s_buffer[k];
    }
    println!("t_clean {t_clean:?}");
    println!("t_buffer {t_buffer:?}");


    let t_clean_mat = NdArray {
        dims: vec![rows, cols],
        data: t_clean.clone(),
    };
    // println!("t_clean_mat {t_clean_mat:?}");
    // println!("----------------------");

    let mut basis_matrix = NdArray {
        dims: vec![cols, rows],
        data: d,
    };
    basis_matrix = basis_matrix.transpose();
    filter_lower_trapezoid(&mut basis_matrix);
    set_diagonal_value(&mut basis_matrix, 1f32);
    println!("basis_matrix {basis_matrix:?}");
    println!("----------------------");
    let s_vector = NdArray {
        dims: vec![cols, cols],
        data: s_buffer,
    };
    let reconst = NdArray {
        dims: vec![rows, cols],
        data: t_buffer,
    };
    let reference = matrix_mult(&basis_matrix, &s_vector);
    println!("t_clean_mat {t_clean_mat:?}");
    println!("----------------------");
    // println!("reconst {reconst:?}");
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

fn debug_set_diagonal(rows:usize, cols:usize) {
    let mut d = generate_random_vector(rows * cols);
    let mut matrix = NdArray {
        dims: vec![rows, cols],
        data: d
    };
    filter_lower_trapezoid(&mut matrix);
    set_diagonal_value(&mut matrix, 1000f32);
    println!("set diagonal {matrix:?}");
}

fn test_debug_set_diagonal() {
    let vals = vec![(1,2), (2,1), (4,4), (3,6), (6,3)];
    for (r,c) in vals {
        debug_set_diagonal(r, c);
        println!("----------------------");
    }
}


fn main() {
    test_reconstruct();
    // validate_upper_upper_fma();
    // validate_transpose_upper_upper_fma();
}
