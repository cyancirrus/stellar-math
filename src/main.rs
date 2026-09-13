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
use stellar::decomposition::wy::{lhs_apply_l, lhs_apply_q, lhs_apply_qt, solve, wy_decomposition};
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

fn test_solves() {
    // let (rows, cols, tcols) = (2, 4, 8);
    // let (rows, cols, tcols) = (4, 8, 12);
    // let (rows, cols, tcols) = (4, 4, 8);
    let (rows, cols, tcols) = (16, 24, 24);
    debug_assert!(cols >= rows);
    let mut l_yt = generate_random_vector(rows * cols);
    let original = l_yt.clone();
    let mut tri = create_identity_vector(rows, rows);
    let mut w = vec![0f32; cols];
    let mut x_argument = vec![0f32; cols * tcols];
    let y_argument = generate_random_vector(rows * tcols);
    let mut t_buffer = vec![0f32; rows * tcols];
    let mut s_buffer = vec![0f32; cols * tcols];

    wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, cols);
    solve(
        &l_yt,
        &tri,
        &mut x_argument,
        &y_argument,
        &mut t_buffer,
        &mut s_buffer,
        cols,
        rows,
        cols,
        tcols,
    );
    let expected = NdArray {
        dims: vec![rows, tcols],
        data: y_argument,
    };
    let x_inferred = NdArray {
        dims: vec![cols, tcols],
        data: s_buffer,
    };
    println!("x_inferred {x_inferred:?}");
    let original_mat = NdArray {
        dims: vec![rows, cols],
        data: original,
    };
    let reconstruct = matrix_mult(&original_mat, &x_inferred);
    println!("expected {expected:?}");
    println!("reconstruct {reconstruct:?}");
}

/// copies memory from b into a
fn import_slice(target: &mut [f32], data: &[f32]) {
    target[..data.len()].copy_from_slice(data);
    target[data.len()..].fill(0f32);
}

fn test_left_apply_qt() {
    let (rows, cols, acols) = (2, 8, 4);
    // let (rows, cols, acols) = (24, 24, 24);
    debug_assert!(cols >= rows);

    let mut l_yt = generate_random_vector(rows * cols);
    let mut tri = create_identity_vector(rows, rows);
    let mut w = vec![0f32; cols];

    let mut x_argument = generate_random_vector(rows * acols);

    let mut t_buffer = vec![0f32; cols * acols];
    let mut s_buffer = vec![0f32; cols * acols];
    let mut w_buffer = vec![0f32; rows * acols];
    println!("a");
    wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, cols);
    println!("b");
    lhs_apply_qt(
        &l_yt,
        &tri,
        &x_argument,
        &mut t_buffer,
        &mut s_buffer,
        rows,
        cols,
        acols,
    );
    println!("c");

    let after_qt = s_buffer.clone();
    t_buffer.fill(0f32);
    lhs_apply_q(
        &l_yt,
        &tri,
        &s_buffer,
        &mut w_buffer,
        &mut t_buffer,
        rows,
        cols,
        acols,
    );
    println!("d");
    let roundtrip = NdArray {
        dims: vec![rows, acols],
        data: t_buffer.clone(),
    };
    let original = NdArray {
        dims: vec![rows, acols],
        data: x_argument.clone(),
    };
    let mid = NdArray {
        dims: vec![cols, acols],
        data: after_qt,
    };
    println!("after Q'   : {mid:?}");
    println!("---------------------");
    println!("original  : {original:?}");
    println!("QQ'x      : {roundtrip:?}");
}

fn test_reconstruct() {
    let (rows, cols, acols) = (4, 6, 8);
    debug_assert!(cols >= rows);
    let (s_x, s_y, s_t, s_tri) = (cols, cols, acols, rows);
    let mut l_yt = generate_random_vector(rows * cols);
    let mut tri = create_identity_vector(rows, rows);
    let input_matrix = NdArray {
        dims: vec![rows, cols],
        data: l_yt.clone(),
    };
    println!("input_matrix {input_matrix:?}");
    let mut w = vec![0f32; cols];
    let mut x_argument = generate_random_vector(cols * acols);
    let mut x_original = x_argument.clone();
    let x_mat = NdArray {
        dims: vec![rows, acols],
        data: x_argument.clone(),
    };
    let mut o_buffer = x_argument.clone();
    let mut t_buffer = vec![0f32; cols * acols];
    let mut q_argument = x_argument.clone();
    let mut s_buffer = vec![0f32; cols * acols];

    let input = l_yt.clone();
    let input_matrix = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, cols);
    lhs_apply_q(
        &l_yt,
        &tri,
        &x_argument,
        &mut t_buffer,
        &mut s_buffer,
        rows,
        cols,
        acols,
    );
    t_buffer.fill(0f32);
    lhs_apply_l(&l_yt, &s_buffer, &mut t_buffer, rows, cols, acols, s_x, s_t);
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
        data: x_original.clone(),
    };
    let expected = matrix_mult(&input, &arg_matrix);
    println!("expected : {expected:?}");
    println!("reconstruct : {result_matrix:?}");
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



pub fn benchmark_wy() {
    // let rows = 64;
    // let cols = 128;
    // let rows = 256;
    let rows = 256;
    let cols = 1024;
    let stride = cols;

    // Generate pseudo-random test data (simple deterministic pattern to avoid external crates)
    let mut a = vec![0.0; rows * cols];
    for (i, val) in a.iter_mut().enumerate() {
        let idx = i as f32;
        *val = (idx.sin() * 43758.54).fract();
    }

    let mut t = vec![0.0; rows * rows];
    let mut w = vec![0.0; rows];
    let mut t_buffer = vec![0.0; cols * cols];
    let mut s_buffer = vec![0.0; cols * cols];

    // Warmup run
    let mut a_warmup = a.clone();
    wy_decomposition(&mut a_warmup, &mut t, &mut w, rows, cols, stride);

    // Benchmark decomposition
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        a_warmup.copy_from_slice(&a);
        wy_decomposition(&mut a_warmup, &mut t, &mut w, rows, cols, stride);
    }
    let duration = start.elapsed();
    println!(
        "WY Decomposition ({}x{}) avg time over {} runs: {:.2?}",
        rows,
        cols,
        iterations,
        duration / iterations
    );

    // Benchmark apply Q
    let mut x_argument = vec![1.0; cols * rows];
    let y_argument = vec![1.0; cols * rows];
    let start = Instant::now();
    for _ in 0..iterations {
        // solve(
        //     &a_warmup,
        //     &t,
        //     &mut x_argument,
        //     &y_argument,
        //     &mut t_buffer,
        //     &mut s_buffer,
        //     rows,
        //     cols,
        //     rows,
        //     1
        // )
        lhs_apply_q(
            &a_warmup,
            &t,
            &x_argument,
            &mut t_buffer,
            &mut s_buffer,
            rows,
            cols,
            rows,
        );
    }
    let duration = start.elapsed();
    println!(
        "LHS Apply Q' avg time over {} runs: {:.2?}",
        iterations,
        duration / iterations
    );
    // Benchmark apply Q
    let mut x_argument = vec![1.0; cols * rows];
    let y_argument = vec![1.0; cols * rows];
    let start = Instant::now();
    for _ in 0..iterations {
        lhs_apply_q(
            &a_warmup,
            &t,
            &x_argument,
            &mut t_buffer,
            &mut s_buffer,
            rows,
            cols,
            rows,
        );
    }
    let duration = start.elapsed();
    println!(
        "LHS Apply Q avg time over {} runs: {:.2?}",
        iterations,
        duration / iterations
    );
}
fn main() {
    test_solves();
    // println!("-----------------------------");
    // println!("-----------------------------");
    // println!("-----------------------------");
    // println!("-----------------------------");
    // println!("-----------------------------");
    // benchmark_wy();
    // test_left_apply_qt();
    // // test_left_apply_q();
    // test_reconstruct();
    // validate_upper_upper_fma();
    // validate_transpose_upper_upper_fma();
}
