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

fn test_reconstruct() {
    let (rows, cols, stride) = (1, 2, 2);
    let mut l_yt = generate_random_vector(rows * cols);
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
        dims: vec![rows, rows],
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
        rows.saturating_sub(1),
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    println!("current0 {t_buffer:?}");
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
    println!("current1 {t_buffer:?}");
    tensor_tut_contraction(
        &l_yt[1..],
        &s_buffer[stride..],
        &mut t_buffer[..],
        0,
        0,
        rows.saturating_sub(1),
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
        data: o_buffer.clone()
    };
    println!("input {input:?}");

    tensor_ut_contraction(
        &d[1..],
        &o_buffer[stride..],
        &mut t_clean[..],
        0,
        0,
        rows.saturating_sub(1),
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
        rows.saturating_sub(1),
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
    let (rows, cols, stride) = (2, 2, 2);
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
        data: o_buffer.clone()
    };
    println!("input {input:?}");

    tensor_tut_contraction(
        &d[1..],
        &o_buffer[..],
        &mut t_clean[stride..],
        0,
        1,
        cols.saturating_sub(1),
        rows.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    // tensor_tut_contraction(
    //     &d[1..],
    //     &o_buffer[stride..],
    //     &mut t_buffer[stride..],
    //     0,
    //     0,
    //     cols,
    //     rows,
    //     cols,
    //     stride,
    //     stride,
    //     stride,
    // );
    // for k in 0..t_clean.len() {
    //     // t_clean[k] += s_buffer[k];
    // }
    // println!("t_clean {t_clean:?}");
    // println!("t_buffer {t_buffer:?}");
    for i in 0..rows {
        for j in 0..=i.min(cols) {
            d[i * stride + j] = 0f32;
        }
        // d[i * stride + i] = 1f32;
    }

    let t_clean_mat = NdArray {
        dims: vec![rows, cols],
        data: t_clean.clone(),
    };

    // for i in 0..rows {
    //     for j in 0..i.min(cols) {
    //         d[i * stride + j] = 0f32;
    //     }
    //     d[i * stride + i] = 1f32;
    // }
    let mut basis_matrix = NdArray {
        dims: vec![rows, cols],
        data: d,
    };
    basis_matrix = basis_matrix.transpose();
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

fn main() {
    // test_reconstruct();
    validate_transpose_upper_upper_fma();
}
