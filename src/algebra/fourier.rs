use crate::structure::ndsignal::{Complex, NdSignal};
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::f32::consts::PI;

fn twiddle(k: usize, n: usize) -> Complex {
    // exp(-i *) = cos(*) - isin(*)
    let phase = -2_f32 * PI * k as f32 / n as f32;
    let a = Complex::new(phase.cos(), -phase.sin());
    a
}

fn twiddle_first(n: f32) -> Complex {
    // exp(-i pi / m) = cos(1/m) - isin(1/m)
    let phase = 2_f32 * PI / n as f32;
    let a = Complex::new(phase.cos(), -phase.sin());
    a
}

pub fn fft(x: &mut [Complex]) {
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
        let wm = twiddle_first(-(m as f32));
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

pub fn ifft(x: &mut [Complex]) {
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
        let wm = twiddle_first(m as f32);
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
    for i in 0..n {
        let alpha = 1_f32 / n as f32;
        x[i].scale(alpha);
    }
}

// fn fft_recursive(x: &mut [Complex]) {
//     let n = x.len();
//     if n <= 1 {
//         return;
//     }

//     let mut even: Vec<Complex> = x.iter().step_by(2).cloned().collect();
//     let mut odd: Vec<Complex> = x.iter().skip(1).step_by(2).cloned().collect();

//     fft_recursive(&mut even);
//     fft_recursive(&mut odd);

//     for k in 0..n / 2 {
//         let p = twiddle(k, n) * odd[k];
//         let q = even[k];
//         x[k] = p + q;
//         x[k + n / 2] = p - q;
//     }
// }

// fn main() {
//     // let k = 16;
//     // let mut data = generate_dummy_series(k);
//     // fft(&mut data);
//     // println!("Development Version {:?}", pretty_format(&data));
//     // ifft(&mut data);
//     // println!("Inverse Version {:?}", pretty_format(&data));
// }



// x,y,z -> y,z,x -> z,x,y
//
//
// y11 y12, ...
// a[y * (rows  * cols) + z * cols + x] = a[x * (rows * cols) y * cols + z]
