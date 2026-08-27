#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_vector;
use stellar::random::generation::{generate_random_vector};
use stellar::decomposition::wy::wy_decomposition;
use stellar::decomposition::lq::AutumnDecomp;
use stellar::structure::ndarray::NdArray;
use stellar::algebra::bmethods::contractions::{
    tensor_lt_contraction,
    tensor_ut_contraction,
    tensor_tut_contraction,
    tensor_rut_contraction,
};

fn main() {
    let (rows, cols, stride) = (2, 2, 2);
    let mut l_yt = generate_random_vector(rows * cols);
    l_yt[1] = 0f32;
    let mut t = generate_random_vector(cols * cols);
    let mut w = vec![0f32; cols];
    

    let mut o_buffer = create_identity_vector(cols, cols);
    let mut t_buffer = o_buffer.clone();
    let mut s_buffer = vec![0f32; cols * cols];

    let input = l_yt.clone();
    
    wy_decomposition(
        &mut l_yt,
        &mut t,
        &mut w,
        rows,
        cols,
        stride
    );

    let l_yt_matrix = NdArray {
        dims:vec![rows, cols],
        data: l_yt.clone(),
    };
    let tri_matrix = NdArray {
        dims:vec![rows, cols],
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
        stride
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
        stride
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
        stride
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
        stride
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
