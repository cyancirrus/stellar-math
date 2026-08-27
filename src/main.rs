#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_vector;
use stellar::random::generation::{generate_random_vector};
use stellar::decomposition::wy::wy_decomposition;
use stellar::structure::ndarray::NdArray;
use stellar::algebra::bmethods::contractions::{
    tensor_lt_contraction,
    tensor_ut_contraction,
    tensor_tut_contraction,
    tensor_rut_contraction,
};

fn main() {
    let (rows, cols, stride) = (4, 4, 4);
    let mut l_yt = generate_random_vector(rows * cols);
    let mut t = generate_random_vector(rows * cols);
    let mut w = vec![0f32; cols];
    

    let mut o_buffer = create_identity_vector(rows, cols);
    let mut s_buffer = vec![0f32; rows * cols];
    let mut t_buffer = vec![0f32; rows * cols];

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

    //   the compact WY representation: `A = L * (I - Y T Y')`.
    // XA = XL * (I - YTY');
    tensor_ut_contraction(
        &l_yt,
        &o_buffer,
        &mut t_buffer,
        1,
        0,
        rows,
        rows,
        cols,
        stride,
        stride,
        stride
    );
    tensor_lt_contraction(
        &t,
        &t_buffer,
        &mut s_buffer,
        1,
        0,
        rows,
        rows,
        cols,
        stride,
        stride,
        stride
    );
    tensor_tut_contraction(
        &l_yt,
        &s_buffer,
        &mut t_buffer,
        1,
        0,
        rows,
        rows,
        cols,
        stride,
        stride,
        stride
    );
    for idx in 0..o_buffer.len() {
        o_buffer[idx] -= t_buffer[idx];
    }
    tensor_lt_contraction(
        &l_yt,
        &o_buffer,
        &mut t_buffer,
        0,
        0,
        rows,
        rows,
        cols,
        stride,
        stride,
        stride
    );
    
    let input = NdArray {
        dims: vec![rows, cols],
        data: input,
    };
    let reconstruct = NdArray {
        dims: vec![rows, cols],
        data: t_buffer.clone(),
    };
    println!("input : {input:?}");
    println!("reconstruct : {reconstruct:?}");
}
