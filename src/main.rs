#![allow(unused)]
use stellar::algebra::ndmethods::create_identity_vector;
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

use stellar::decomposition::svd::bidiagonalization::{lbidiagonal, ubidiagonal};
use stellar::decomposition::svd::bulge_chasing::{decomp_usym};
#[rustfmt::skip]
use stellar::decomposition::svd::verify::{
    full_ubidiagonal,
    full_lbidiagonal,
    full_decomp_ugivens,
    full_decomp_lgivens
};

fn diagonal(b: &[f32], card: usize, stride: usize) -> Vec<f32> {
    (0..card).map(|i| b[i * stride + i]).collect()
}

fn main() {
    let mut max_iters = 40;
    let mut tolerance = 1e-6;
    let mut absolute = 1e-5;


    let rows = 6;
    let cols = 6;
    let card = rows.min(cols);
    let stride = cols;
    let maximum = rows.max(cols);
    let mut b = generate_random_vector(rows * cols);
    let mut p = vec![0f32; maximum];
    let mut w = vec![0f32; maximum];

    ubidiagonal(&mut b, &mut p, &mut w, rows, cols, card, stride) ;
    let bmat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("bidiagonal {bmat:?}");

    decomp_usym(&mut b, card, stride, max_iters, tolerance, absolute);
    let smat = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("singular {smat:?}");

}
