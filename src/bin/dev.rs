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



// fn butterfly(x:Complex, y:Complex) -> (Complex, Complex) {
//     let mut data = vec![Complex::new(0_f32, 0_f32);2];
//     data[0] = x + y;
//     data[1] = x - y;
//     data
// }


fn butterfly(x:Complex, y:Complex) -> (Complex, Complex) {
    (x + y, x - y)
}

fn twiddle(k:usize,n:usize) -> Complex {
    // exp( (-2 * pi * i * k / n)
    // = cos(*) - isin(*)
    let phase= 2_f32 * PI * k as f32 / n as f32;
    let a = Complex::new(phase.cos(), -phase.sin());
    println!("Phase pi normalized {}, {:?}", phase/PI, a); a
}


fn cooley_tukey(o:usize, n:usize, s:usize, x:&mut [Complex]) {
    if n == 1 {
        return;
    }
    let mut p:Complex;
    let mut q:Complex;
    let half= n>>1;
    let doub= s<<1;
    cooley_tukey(o, half, doub, x);
    cooley_tukey(o + s,  half, doub, x);
    println!("this is what x looks like {:?}", x);

    for k in o..half {
        // println!("mutating! x[{}] and x[{}]", ei, oi);
        let ei = o + k * 2 * s;
        let oi = ei + s;
        p = x[ei];
        q = twiddle(k, n) * x[oi];
        
        x[ei] = p + q;
        x[oi] = p - q;
    }
}

// fn cooley_tukey(start: usize, n: usize, stride: usize, x: &mut [Complex]) {
//     if n == 1 {
//         // Base case: one element is already transformed
//         return;
//     }
//     let mut p:Complex;
//     let mut q:Complex;
//     let half= n>>1;
//     let doub= stride<<1;

//     cooley_tukey(start, half, doub, x);
//     cooley_tukey(start + stride,  half, doub, x);

//     for k in 0..half {
//         let ei= start + k * doub;
//         let oi = ei + stride;

//         let q = twiddle(k, n) * x[oi];
//         let p = x[ei];
//         x[ei] = p + q;
//         x[oi] = p - q;
//     }
// }



fn fft_algorithm(mut x:Vec<Complex>) -> Vec<Complex> {
    let n = x.len();
    cooley_tukey(0, n, 1, &mut x);
    x
}

fn pretty_format(data:Vec<Complex>) -> NdSignal {
    let mut dims = vec![1;2];
    dims[0] = data.len();
    NdSignal { dims, data }
}

fn main() {
    let k = 4;
    let data = generate_dummy_signal(k);
    println!("Data {:?}", data);
    let dct_matrix = create_dct_array(k);

    let dct_frequency = complex_tensor_mult(dct_matrix, data);
    println!("Fourier transform:\n{:?}", dct_frequency);
    let data = generate_dummy_series(k);
    let butter = fft_algorithm(data);
    println!("Development Version {:?}", pretty_format(butter));

}

