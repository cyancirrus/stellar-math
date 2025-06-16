#![allow(warnings)]
#[cfg(target_arch = "x86_64")]
use std::f32::consts::PI;
use StellarMath::algebra::fourier::{fft, ifft};
use StellarMath::structure::ndsignal::{Complex, NdSignal};

// fn generate_dummy_series(n: usize) -> Vec<Complex> {
//     let mut signal = Vec::with_capacity(n);
//     for t in 0..n {
//         let measurement = {
//             (PI * (t as f32) / 4_f32).cos()
//                 + (PI * (t as f32) / 5_f32).cos()
//                 + (PI * (t as f32) / 2_f32).sin()
//         };

//         signal.push(Complex::new(measurement, 0_f32));
//     }
//     signal
// }

// fn pretty_format(data: &[Complex]) -> NdSignal {
//     let mut dims = vec![1; 2];
//     dims[0] = data.len();
//     NdSignal {
//         dims,
//         data: data.to_vec(),
//     }
// }

fn debug_data() -> Vec<usize> {
    // dimensions indexed by (x,y,z)
    vec![
        0, 2,
        4, 6,
        //---
        1, 3,
        5, 7 
    ]
}

fn transpose(data:&mut [usize]) {
    // fastest to slowest
    let mut dims = vec![2,2,2];
    // static for prototyping
    let (x,y,z) = (2,2,2);
    for i in 0..x {
        for j in i..y {
            for k in j..z {
                let a = i*(x*y) + j*y + k;
                let b = k*(x*y) + i*y + j;
                let c = j*(x*y) + k*y + i;
                // a, b, c :: 
                // c, b, a :: swap(a, c)
                // b, c, a :: swap(b, c)
                println!(" a: {a:03b}, b: {b:03b}, c: {c:03b}");
                data.swap( a, b,);
                data.swap( b, c ,);
                // data.swap( a, c,);
            }
        }
    }
}

fn transpose_test(data:&mut[usize]) {
    for i in 1..data.len() {
        data.swap(i-1, i)
    }
}


fn pretty_print(data:&[usize]) {
    println!("----------");
    for i in (0..8).step_by(2) {
        if i == 4 {
            println!(" - - - - - -");
        }
        println!("[ {:03b}, {:03b} ]", data[i], data[i+1]);
    }
    println!("----------");
}

fn debug_transpose(data:&mut [usize]) {
    transpose(data);
    pretty_print(data);
}

fn main() {
    let mut data = debug_data();
    pretty_print(&data);
    println!("");
    let mut test = data.clone();
    debug_transpose(&mut test);
    debug_transpose(&mut test);
    debug_transpose(&mut test);
    assert_eq!(test, data);
}

