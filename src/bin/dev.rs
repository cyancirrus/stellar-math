#![allow(warnings)]
use std::f32::consts::PI;
use StellarMath::algebra::{math, simd};
use StellarMath::decomposition::{qr, schur, householder};
use StellarMath::decomposition::householder::{householder_params, HouseholderReflection};
use StellarMath::decomposition::svd::{golub_kahan, golub_kahan_explicit, golub_kahan_lanczos};
use StellarMath::decomposition::bidiagonal::{bidiagonal_qr, fast_bidiagonal_qr};
use StellarMath::structure::ndarray::NdArray;
use StellarMath::structure::ndsignal::{Complex, NdSignal};
use StellarMath::algebra::ndmethods::{
    transpose,
    tensor_mult,
    create_identity_matrix,
};
use StellarMath::algebra::ndsmethods::{
    complex_tensor_mult,
    create_dct_array,
};
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt;

fn generate_dummy_signal(n:usize) -> NdSignal {
    let mut dims = vec![0;2];
    let mut signal = Vec::with_capacity(n);
    dims[0] = n;
    dims[1] = 1;
    for t in 0..n {
        let measurement = {
            (PI * (t as f32) / 4_f32).cos()
            + (PI * (t as f32) / 2_f32).cos()
            + (PI * (t as f32) / 2_f32).sin()
        };

        signal.push(Complex::new(measurement, 0_f32));

    }
    println!("Signal {:?}", signal);
    NdSignal::new(dims, signal)
}

fn generate_dummy_series(n:usize) -> Vec<Complex> {
    let mut signal = Vec::with_capacity(n);
    for t in 0..n {
        let measurement = {
            (PI * (t as f32) / 4_f32).cos()
            + (PI * (t as f32) / 2_f32).cos()
            + (PI * (t as f32) / 2_f32).sin()
        };

        signal.push(Complex::new(measurement, 0_f32));

    }
    signal
}




fn butterfly(x:Complex, y:Complex) -> (Complex, Complex) {
    (x + y, x - y)
}







fn main() {
    // let mut dims:Vec<usize>;
    // let mut data:Vec<Complex>;
    // {
    //     dims = vec![2; 2];
    //     data = vec![Complex::new(0_f32, 0_f32); 4];
    //     data[0] = Complex::new(1_f32, 0.5_f32);
    //     data[1] = Complex::new(1_f32, 0_f32);
    //     data[2] = Complex::new(0.3_f32, 0.25_f32);
    //     data[3] = Complex::new(0.7_f32, 0.3_f32);
    // }

    // println!("Complex {}", data[0]);
    // let dev = NdSignal::new(dims, data);
    // println!("Check debug is working {:?}", dev);
    // let k = 8;
    // let data = generate_dummy_signal(k);
    // let dct_matrix = create_dct_array(k);

    // let dct_frequency = complex_tensor_mult(dct_matrix, data);
    // println!("Fourier transform:\n{:?}", dct_frequency);
    let k = 2;
    let data = generate_dummy_series(k);
    let butter = butterfly(data[0], data[1]);
    println!("Result of base case {:?}", butter);

}

