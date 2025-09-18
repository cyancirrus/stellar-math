use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::givens::givens_iteration;
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::householder::householder_factor;

use rand::Rng;
use rand_distr::StandardNormal;
// use rand::prelude::*;


// #[cfg(target_arch = "x86_64")]




fn main() {
    let mut rng = rand::rng();
    let normal = StandardNormal;
    let x:f32 = rng.sample(StandardNormal); 
    println!("result {x:?}");
}

