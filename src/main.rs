#![allow(unused)]
use stellar::random::generation::{generate_random_vector};
use stellar::algebra::bmethods::contractions::{tensor_rut_contraction};
use stellar::decomposition::wy::wy_decomposition;
use stellar::structure::ndarray::NdArray;

fn main() {
    let (rows, cols, stride) = (4, 4, 4);
    let mut l_yt = generate_random_vector(rows * cols);
    let mut t = generate_random_vector(rows * cols);
    let mut w = vec![0f32; cols];

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
    let mut t_output = vec![0f32; rows * cols];
    println!("l_yt : {l_yt_matrix:?}");
    println!("tri : {tri_matrix:?}");

    tensor_rut_contraction(
        &l_yt,
        &t,
        &mut t_output,
        1,
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
        data: t_output.clone(),
    };
    println!("input : {input:?}");
    println!("reconstruct : {reconstruct:?}");
}
