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

#[allow(Non_Snake_Case)]
fn twiddle(k:f32,n:f32) -> Complex {
    // exp( (-2 * pi * i * k / n)
    // = cos(*) - isin(*)
    let phase= 2_f32 * PI * k / n;

    Complex::new(phase.cos(), -phase.sin())
}

fn cooley_tukey(x:&mut [Complex], n:usize, s:usize) {
    let mut p:Complex;
    let mut q:Complex;
    println!("This is the value of n: {}", n);
    if n > 1 {
        cooley_tukey(&mut x[0..], n / 2, 2 * s);
        println!("left inner n: {}", n);
        cooley_tukey(&mut x[s..], n / 2, 2 * s);
        println!("right inner n: {}", n);
        println!("this is what x looks like {:?}", x);
        for k in 0..n/2 {
            p = x[k];
            q = twiddle(k as f32, n as f32) * x[k + n /2];
            println!("mutating!");
            x[k] = p + q;
            x[k + n/2] = p - q;
            println!("k: {}, k-o: {}, p:{}, q:{}", k, k + n/2, p, q);
        }

    }
}

fn fft_algorithm(mut x:Vec<Complex>) -> Vec<Complex> {
    let n = x.len();
    cooley_tukey(&mut x, n, 2);
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

