#![allow(warnings)]
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::f32::consts::PI;
use std::fmt;
use StellarMath::algebra::ndmethods::{create_identity_matrix, tensor_mult, transpose};
use StellarMath::algebra::ndsmethods::{complex_tensor_mult, create_dct_array};
use StellarMath::algebra::{math, simd};
use StellarMath::decomposition::bidiagonal::{bidiagonal_qr, fast_bidiagonal_qr};
use StellarMath::decomposition::householder::{householder_params, HouseholderReflection};
use StellarMath::decomposition::svd::{golub_kahan, golub_kahan_explicit, golub_kahan_lanczos};
use StellarMath::decomposition::{householder, qr, schur};
use StellarMath::structure::ndarray::NdArray;
use StellarMath::structure::ndsignal::{Complex, NdSignal};

fn generate_dummy_signal(n: usize) -> NdSignal {
    let mut dims = vec![0; 2];
    let mut signal = Vec::with_capacity(n);
    dims[0] = n;
    dims[1] = 1;
    for t in 0..n {
        let measurement = {
            (PI * (t as f32) / 4_f32).cos()
                + (PI * (t as f32) / 5_f32).cos()
                + (PI * (t as f32) / 2_f32).sin()
        };

        signal.push(Complex::new(measurement, 0_f32));
    }
    println!("Signal {:?}", signal);
    NdSignal::new(dims, signal)
}

fn generate_dummy_series(n: usize) -> Vec<Complex> {
    let mut signal = Vec::with_capacity(n);
    for t in 0..n {
        let measurement = {
            (PI * (t as f32) / 4_f32).cos()
                + (PI * (t as f32) / 5_f32).cos()
                + (PI * (t as f32) / 2_f32).sin()
        };

        signal.push(Complex::new(measurement, 0_f32));
    }
    signal
}

fn twiddle(k: usize, n: usize) -> Complex {
    // exp(-i *) = cos(*) - isin(*)
    let phase = -2_f32 * PI * k as f32 / n as f32;
    let a = Complex::new(phase.cos(), -phase.sin());
    a
}

fn fft_recursive(x: &mut [Complex]) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    let mut even: Vec<Complex> = x.iter().step_by(2).cloned().collect();
    let mut odd: Vec<Complex> = x.iter().skip(1).step_by(2).cloned().collect();

    fft_recursive(&mut even);
    fft_recursive(&mut odd);

    for k in 0..n / 2 {
        let p = twiddle(k, n) * odd[k];
        let q = even[k];
        x[k] = p + q;
        x[k + n / 2] = p - q;
    }
}

fn twiddle_first(n: usize) -> Complex {
    // exp(-i pi / m) = cos(1/m) - isin(1/m)
    let phase = -2_f32 * PI / n as f32;
    let a = Complex::new(phase.cos(), -phase.sin());
    a
}

fn fft_iterative(x: &mut [Complex]) {
    let n = x.len();
    let bits = n.trailing_zeros();
    // bit reversal
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            x.swap(i, j);
        }
    }
    // cooley-tuckey
    for s in 1..=bits {
        let m = 1 << s;
        let half = m >> 1;
        let wm = twiddle_first(m);
        for k in (0..n).step_by(m) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let p = x[k + j];
                let q = w * x[k + j + half];
                x[k + j] = p + q;
                x[k + j + half] = p - q;
                w *= wm;
            }
        }
    }
}

fn fft_algorithm(mut x: Vec<Complex>) -> Vec<Complex> {
    let n = x.len();
    fft_iterative(&mut x);
    x
}

fn shuffle(x: &mut [Complex]) {
    let n = x.len();
    let bits = n.trailing_zeros();
    for i in 0..n / 2 {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        x.swap(i, j);
    }
}

fn pretty_format(data: Vec<Complex>) -> NdSignal {
    let mut dims = vec![1; 2];
    dims[0] = data.len();
    NdSignal { dims, data }
}

fn main() {
    let k = 16;
    let data = generate_dummy_signal(k);
    println!("Data {:?}", data);
    let dct_matrix = create_dct_array(k);

    let dct_frequency = complex_tensor_mult(dct_matrix, data);
    println!("Fourier transform:\n{:?}", dct_frequency);
    let data = generate_dummy_series(k);
    let butter = fft_algorithm(data);
    println!("Development Version {:?}", pretty_format(butter));
}
