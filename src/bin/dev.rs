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
    // println!("Phase pi normalized {}, {:?}", phase/PI, a);
    a
}

// ```
// modified (0, 2) target (0,1) state {s:2, o:0, n:2, ei:0, oi:2, k:0}
// modified (1, 3) target (2,3) state {s:2, o:1, n:2, ei:1, oi:3, k:0}

// modified (0, 1) target (0, 2) state {s:1, o:0, n:4, ei:0, oi:1, k:0}
// modified (2, 3) target (1, 3) state {s:1, o:0, n:4, ei:2, oi:3, k:1}
// ```
//
//
// e ~ o*2 + n*2 - s
// o ~ o*2 + k*2 + n/2

// fn cooley_tukey(o:usize, n:usize, s:usize, x:&mut [Complex]) {
//     if n == 1 {
//         return;
//     }
//     let mut p:Complex;
//     let mut q:Complex;
//     let half= n>>1;
//     let doub= s<<1;
//     cooley_tukey(o, half, doub, x);
//     cooley_tukey(o + s,  half, doub, x);
//     println!("this is what x looks like {:?}", x);

//     for k in 0..half {
//         let ei = o + k * 2 * s;
//         let oi = ei + s;
//         println!("mutating! x[{}] and x[{}]", ei, oi);
//         // println!("state {{s:{s}, o:{o}, n:{n}, ei:{ei}, oi:{oi}, k:{k}}}");
//         p = x[ei];
//         q = twiddle(k, n) * x[oi];
        
//         // x[ei] = p + q;
//         // x[oi] = p - q;
//         x[ei] = p + q;
//         x[oi] = p - q;
//     }
// }

// fn fft(x: &mut [Complex]) {
//     let n = x.len();
//     if n <= 1 {
//         return;
//     }

//     let mut even: Vec<Complex> = x.iter().step_by(2).cloned().collect();
//     let mut odd: Vec<Complex> = x.iter().skip(1).step_by(2).cloned().collect();

//     fft(&mut even);
//     fft(&mut odd);

//     for k in 0..n / 2 {
//         let p = twiddle(k, n) * odd[k];
//         let q = even[k];
//         x[k] =  p + q;
//         x[k + n / 2] = p - q;
//     }
// }

fn cooley_tukey(x:&mut [Complex]) {
    let mut p:Complex;
    let mut q:Complex;
    let n = x.len();
    
    for s in  1..(usize::BITS - n.leading_zeros()) {
        let m = 1<<s;
        let half = m>>1;
        let mut wm = twiddle(1, m);
        for k in  (0..n).step_by(m) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..m/2 {
                p = x[k + j];
                q = w * x[k + j + half];
                x[k + j] = p + q;
                x[k + j + half] = p - q;
                w *= wm;
            }
        }
    }
}



// fn cooley_tukey(o:usize, n:usize, s:usize, x:&mut [Complex]) {
//     if n == 1 {
//         return;
//     }
//     let mut p:Complex;
//     let mut q:Complex;
//     let half= n>>1;
//     let doub= s<<1;
//     cooley_tukey(o, half, doub, x);
//     cooley_tukey(o + s,  half, doub, x);
//     println!("this is what x looks like {:?}", x);

//     for k in 0..half {
//         let ei = o + k * 2 * s;
//         let oi = ei + s;
//         println!("mutating! x[{}] and x[{}]", ei, oi);
//         // println!("state {{s:{s}, o:{o}, n:{n}, ei:{ei}, oi:{oi}, k:{k}}}");
//         p = x[ei];
//         q = twiddle(k, n) * x[oi];
        
//         // x[ei] = p + q;
//         // x[oi] = p - q;
//         x[ei] = p + q;
//         x[oi] = p - q;
//     }
// }


fn fft_algorithm(mut x:Vec<Complex>) -> Vec<Complex> {
    let n = x.len();
    // shuffle(&mut x);
    // fft(&mut x);
    shuffle(&mut x);
    println!("Data {:?}", x);
    // cooley_tukey(0, n, 1, &mut x);
    cooley_tukey(&mut x);
    x
}

fn shuffle(x:&mut [Complex]) {
    let n = x.len();
    let bits = n.trailing_zeros();
    for i in 0..n/2 {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        x.swap(i, j);
    }
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

