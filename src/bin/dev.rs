#![allow(warnings)]
use StellarMath::algebra::{math, simd};
use StellarMath::decomposition::{qr, schur, householder};
use StellarMath::decomposition::householder::{householder_params, HouseholderReflection};
use StellarMath::decomposition::svd::{golub_kahan, golub_kahan_explicit, golub_kahan_lanczos};
use StellarMath::decomposition::bidiagonal::{bidiagonal_qr, fast_bidiagonal_qr};
use StellarMath::structure::ndarray::NdArray;
use StellarMath::algebra::ndmethods::{
    transpose,
    tensor_mult,
    create_identity_matrix,
};
use rand::Rng;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;





fn main() {
    let mut data:Vec<f32>;
    let mut dims:Vec<usize>;
    // data[0] = 1_f32;
    // data[1] = -1_f32;
    // data[2] = 4_f32;
    // data[3] = 1_f32;
    // {
    //     // Eigen values 2, -1
    //     let mut data = vec![0_f32; 4];
    //     let mut dims = vec![2; 2];
    //     data[0] = -1_f32;
    //     data[1] = 0_f32;
    //     data[2] = 5_f32;
    //     data[3] = 2_f32;
    // }
    {
        data = vec![0_f32; 9];
        dims = vec![3; 2];
        data[0] = 1_f32;
        data[1] = 2_f32;
        data[2] = 3_f32;
        data[3] = 3_f32;
        data[4] = 4_f32;
        data[5] = 5_f32;
        data[6] = 6_f32;
        data[7] = 7_f32;
        data[8] = 8_f32;
    }
    let x = NdArray::new(dims, data.clone());
    println!("x: {:?}", x);
    //
    let reference = golub_kahan_explicit(x.clone());
    println!("Reference {:?}", reference);

    let dev = golub_kahan(x.clone());
    println!("Implicit {:?}", dev);


    // let sym = symmetricize(x);
    // println!("Did it make symmetric? {:?}", sym);
    // let test = golub_kahan_lanczos(x.clone());
    // // println!("Test:\nU {:?}\nS {:?}\nV {:?}", test.0, test.1, test.2);
    // println!("Bidiagonal \nS {:?}", test.1);


    // let mut check = tensor_mult(2, &transpose(test.0), &test.1.clone());
    // check = tensor_mult(2, &check, &test.2.clone());
    // println!("Checking reconstruction {:?}", check);

    
    // let real_schur = real_schur_decomp(x.clone());
    // println!("real schur kernel {:?}", real_schur.kernel);
    
    // let sigma = bidiagonal_qr(test.1.clone());
    // println!("Bidiagonal QR {:?}", sigma);
    // println!("Expected eigen: 2, 1");
    
    // // From lapack white paper
    // let sigma = fast_bidiagonal_qr(fast_bidiagonal_qr(test.1.clone()));
    // println!("Fast Bidiagonal QR {:?}", sigma);
    // println!("Expected eigen: 2, 1");
    // // // println!("real schur rotation {:?}", real_schur.rotation);
    // //
    // let y = qr_decompose(x.clone());
    // println!("triangle {:?}", y.triangle);


    // let q = real_schur.rotation;
    // let q_star = transpose(q.clone());
    // println!("Schur rotation {:?}", q);
    // let q_orthogonality_check = tensor_mult(2, &q, &q_star);
    // println!("U orthogonality check {:?}", q_orthogonality_check);

    // let symmetric = make_symmetric(&real_schur.kernel);
    // println!("Symmetric values {:?}", symmetric);
    
    // // let eigen = givens_decomp(symmetric);
    // let eigen = eigen_square(y.triangle);
    // let eigen = jacobi_decomp(symmetric);
    // let eigen = givens_decomp(y.triangle);
    // let eigen = givens_decomp(x.clone());
    // println!("eigen values {:?}", eigen);
    
    // let eigen = givens_decomp(real_schur.kernel);
    // let eigen = jacobi_decomp(symmetric);
    // println!("eigen values {:?}", eigen.kernel);

}

